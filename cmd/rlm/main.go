// Package main provides a CLI for running RLM (Recursive Language Model) queries.
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	"github.com/XiaoConstantine/rlm-go/pkg/core"
	"github.com/XiaoConstantine/rlm-go/pkg/logger"
	"github.com/XiaoConstantine/rlm-go/pkg/repl"
	"github.com/XiaoConstantine/rlm-go/pkg/rlm"
)

var (
	contextFile   = flag.String("context", "", "Path to context file (or use stdin)")
	contextStr    = flag.String("context-string", "", "Context string directly")
	query         = flag.String("query", "", "Query to run against the context")
	model         = flag.String("model", "claude-sonnet-4-20250514", "Model to use")
	maxIterations = flag.Int("max-iterations", 30, "Maximum iterations")
	verbose       = flag.Bool("verbose", false, "Enable verbose output")
	logDir        = flag.String("log-dir", "", "Directory for JSONL logs (optional)")
	jsonOutput    = flag.Bool("json", false, "Output result as JSON")
)

// Supported Gemini models.
var geminiModels = map[string]bool{
	"gemini-3-flash-preview": true,
	"gemini-3-pro-preview":   true,
}

// AnthropicClient implements both rlm.LLMClient and repl.LLMClient interfaces.
type AnthropicClient struct {
	apiKey     string
	model      string
	maxTokens  int
	httpClient *http.Client
	verbose    bool
}

// NewAnthropicClient creates a new Anthropic client with connection pooling.
func NewAnthropicClient(apiKey, model string, verbose bool) *AnthropicClient {
	return &AnthropicClient{
		apiKey:    apiKey,
		model:     model,
		maxTokens: 4096,
		verbose:   verbose,
		httpClient: &http.Client{
			Timeout: 180 * time.Second,
			Transport: &http.Transport{
				MaxIdleConns:        100,
				MaxIdleConnsPerHost: 10,
				MaxConnsPerHost:     10,
				IdleConnTimeout:     90 * time.Second,
			},
		},
	}
}

type anthropicRequest struct {
	Model     string             `json:"model"`
	MaxTokens int                `json:"max_tokens"`
	Messages  []anthropicMessage `json:"messages"`
	System    string             `json:"system,omitempty"`
}

type anthropicMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type anthropicResponse struct {
	Content []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	} `json:"content"`
	Usage struct {
		InputTokens  int `json:"input_tokens"`
		OutputTokens int `json:"output_tokens"`
	} `json:"usage"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error,omitempty"`
}

// Complete implements rlm.LLMClient for root LLM orchestration.
func (c *AnthropicClient) Complete(ctx context.Context, messages []core.Message) (core.LLMResponse, error) {
	var systemPrompt string
	var apiMessages []anthropicMessage

	for _, msg := range messages {
		if msg.Role == "system" {
			systemPrompt = msg.Content
		} else {
			apiMessages = append(apiMessages, anthropicMessage{
				Role:    msg.Role,
				Content: msg.Content,
			})
		}
	}

	reqBody := anthropicRequest{
		Model:     c.model,
		MaxTokens: c.maxTokens,
		Messages:  apiMessages,
		System:    systemPrompt,
	}

	text, inputTokens, outputTokens, err := c.doRequest(ctx, reqBody)
	if err != nil {
		return core.LLMResponse{}, err
	}
	return core.LLMResponse{
		Content:          text,
		PromptTokens:     inputTokens,
		CompletionTokens: outputTokens,
	}, nil
}

// Query implements repl.LLMClient for sub-LLM calls from REPL.
func (c *AnthropicClient) Query(ctx context.Context, prompt string) (repl.QueryResponse, error) {
	reqBody := anthropicRequest{
		Model:     c.model,
		MaxTokens: c.maxTokens,
		Messages: []anthropicMessage{
			{Role: "user", Content: prompt},
		},
	}

	text, inputTokens, outputTokens, err := c.doRequest(ctx, reqBody)
	return repl.QueryResponse{
		Response:         text,
		PromptTokens:     inputTokens,
		CompletionTokens: outputTokens,
	}, err
}

// QueryBatched implements repl.LLMClient for concurrent sub-LLM calls.
func (c *AnthropicClient) QueryBatched(ctx context.Context, prompts []string) ([]repl.QueryResponse, error) {
	results := make([]repl.QueryResponse, len(prompts))
	var wg sync.WaitGroup

	for i, prompt := range prompts {
		wg.Add(1)
		go func(idx int, p string) {
			defer wg.Done()
			result, err := c.Query(ctx, p)
			if err != nil {
				results[idx] = repl.QueryResponse{Response: fmt.Sprintf("Error: %v", err)}
			} else {
				results[idx] = result
			}
		}(i, prompt)
	}

	wg.Wait()
	return results, nil
}

func (c *AnthropicClient) doRequest(ctx context.Context, reqBody anthropicRequest) (string, int, int, error) {
	start := time.Now()

	jsonBody, err := json.Marshal(reqBody)
	if err != nil {
		return "", 0, 0, fmt.Errorf("marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", "https://api.anthropic.com/v1/messages", bytes.NewReader(jsonBody))
	if err != nil {
		return "", 0, 0, fmt.Errorf("create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", c.apiKey)
	req.Header.Set("anthropic-version", "2023-06-01")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return "", 0, 0, fmt.Errorf("http request: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", 0, 0, fmt.Errorf("read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return "", 0, 0, fmt.Errorf("api error (status %d): %s", resp.StatusCode, string(body))
	}

	var apiResp anthropicResponse
	if err := json.Unmarshal(body, &apiResp); err != nil {
		return "", 0, 0, fmt.Errorf("unmarshal response: %w", err)
	}

	if c.verbose {
		fmt.Fprintf(os.Stderr, "  [API] %v, tokens: %d→%d\n",
			time.Since(start), apiResp.Usage.InputTokens, apiResp.Usage.OutputTokens)
	}

	if apiResp.Error != nil {
		return "", 0, 0, fmt.Errorf("api error: %s", apiResp.Error.Message)
	}

	var texts []string
	for _, block := range apiResp.Content {
		if block.Type == "text" {
			texts = append(texts, block.Text)
		}
	}

	return strings.Join(texts, ""), apiResp.Usage.InputTokens, apiResp.Usage.OutputTokens, nil
}

// GeminiClient implements both rlm.LLMClient and repl.LLMClient interfaces for Google's Gemini API.
type GeminiClient struct {
	apiKey     string
	model      string
	httpClient *http.Client
	verbose    bool
}

// NewGeminiClient creates a new Gemini client.
func NewGeminiClient(apiKey, model string, verbose bool) *GeminiClient {
	return &GeminiClient{
		apiKey:  apiKey,
		model:   model,
		verbose: verbose,
		httpClient: &http.Client{
			Timeout: 180 * time.Second,
			Transport: &http.Transport{
				MaxIdleConns:        100,
				MaxIdleConnsPerHost: 10,
				MaxConnsPerHost:     10,
				IdleConnTimeout:     90 * time.Second,
			},
		},
	}
}

type geminiRequest struct {
	Contents         []geminiContent       `json:"contents"`
	SystemInstruction *geminiContent       `json:"systemInstruction,omitempty"`
	GenerationConfig  *geminiGenConfig     `json:"generationConfig,omitempty"`
}

type geminiContent struct {
	Role  string       `json:"role,omitempty"`
	Parts []geminiPart `json:"parts"`
}

type geminiPart struct {
	Text string `json:"text"`
}

type geminiGenConfig struct {
	MaxOutputTokens int `json:"maxOutputTokens,omitempty"`
}

type geminiResponse struct {
	Candidates []struct {
		Content struct {
			Parts []struct {
				Text string `json:"text"`
			} `json:"parts"`
		} `json:"content"`
	} `json:"candidates"`
	UsageMetadata struct {
		PromptTokenCount     int `json:"promptTokenCount"`
		CandidatesTokenCount int `json:"candidatesTokenCount"`
		TotalTokenCount      int `json:"totalTokenCount"`
	} `json:"usageMetadata"`
	Error *struct {
		Message string `json:"message"`
		Code    int    `json:"code"`
	} `json:"error,omitempty"`
}

// Complete implements rlm.LLMClient for root LLM orchestration.
func (c *GeminiClient) Complete(ctx context.Context, messages []core.Message) (core.LLMResponse, error) {
	var systemContent *geminiContent
	var contents []geminiContent

	for _, msg := range messages {
		if msg.Role == "system" {
			systemContent = &geminiContent{
				Parts: []geminiPart{{Text: msg.Content}},
			}
		} else {
			role := msg.Role
			if role == "assistant" {
				role = "model"
			}
			contents = append(contents, geminiContent{
				Role:  role,
				Parts: []geminiPart{{Text: msg.Content}},
			})
		}
	}

	reqBody := geminiRequest{
		Contents:          contents,
		SystemInstruction: systemContent,
		GenerationConfig:  &geminiGenConfig{MaxOutputTokens: 8192},
	}

	text, promptTokens, completionTokens, err := c.doRequest(ctx, reqBody)
	if err != nil {
		return core.LLMResponse{}, err
	}
	return core.LLMResponse{
		Content:          text,
		PromptTokens:     promptTokens,
		CompletionTokens: completionTokens,
	}, nil
}

// Query implements repl.LLMClient for sub-LLM calls from REPL.
func (c *GeminiClient) Query(ctx context.Context, prompt string) (repl.QueryResponse, error) {
	reqBody := geminiRequest{
		Contents: []geminiContent{
			{
				Role:  "user",
				Parts: []geminiPart{{Text: prompt}},
			},
		},
		GenerationConfig: &geminiGenConfig{MaxOutputTokens: 8192},
	}

	text, promptTokens, completionTokens, err := c.doRequest(ctx, reqBody)
	return repl.QueryResponse{
		Response:         text,
		PromptTokens:     promptTokens,
		CompletionTokens: completionTokens,
	}, err
}

// QueryBatched implements repl.LLMClient for concurrent sub-LLM calls.
func (c *GeminiClient) QueryBatched(ctx context.Context, prompts []string) ([]repl.QueryResponse, error) {
	results := make([]repl.QueryResponse, len(prompts))
	var wg sync.WaitGroup

	for i, prompt := range prompts {
		wg.Add(1)
		go func(idx int, p string) {
			defer wg.Done()
			result, err := c.Query(ctx, p)
			if err != nil {
				results[idx] = repl.QueryResponse{Response: fmt.Sprintf("Error: %v", err)}
			} else {
				results[idx] = result
			}
		}(i, prompt)
	}

	wg.Wait()
	return results, nil
}

func (c *GeminiClient) doRequest(ctx context.Context, reqBody geminiRequest) (string, int, int, error) {
	start := time.Now()

	jsonBody, err := json.Marshal(reqBody)
	if err != nil {
		return "", 0, 0, fmt.Errorf("marshal request: %w", err)
	}

	url := fmt.Sprintf("https://generativelanguage.googleapis.com/v1beta/models/%s:generateContent?key=%s", c.model, c.apiKey)
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(jsonBody))
	if err != nil {
		return "", 0, 0, fmt.Errorf("create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return "", 0, 0, fmt.Errorf("http request: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", 0, 0, fmt.Errorf("read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return "", 0, 0, fmt.Errorf("api error (status %d): %s", resp.StatusCode, string(body))
	}

	var apiResp geminiResponse
	if err := json.Unmarshal(body, &apiResp); err != nil {
		return "", 0, 0, fmt.Errorf("unmarshal response: %w", err)
	}

	if apiResp.Error != nil {
		return "", 0, 0, fmt.Errorf("api error: %s", apiResp.Error.Message)
	}

	if c.verbose {
		fmt.Fprintf(os.Stderr, "  [API] %v, tokens: %d→%d\n",
			time.Since(start), apiResp.UsageMetadata.PromptTokenCount, apiResp.UsageMetadata.CandidatesTokenCount)
	}

	var texts []string
	for _, candidate := range apiResp.Candidates {
		for _, part := range candidate.Content.Parts {
			texts = append(texts, part.Text)
		}
	}

	return strings.Join(texts, ""), apiResp.UsageMetadata.PromptTokenCount, apiResp.UsageMetadata.CandidatesTokenCount, nil
}

// Result represents the JSON output format.
type Result struct {
	Response   string `json:"response"`
	Iterations int    `json:"iterations"`
	Duration   string `json:"duration"`
	Tokens     struct {
		Prompt     int `json:"prompt"`
		Completion int `json:"completion"`
		Total      int `json:"total"`
	} `json:"tokens"`
}

func main() {
	// Check for subcommands before parsing flags
	if len(os.Args) > 1 && os.Args[1] == "install-claude-code" {
		if err := installClaudeCodePlugin(); err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}
		return
	}

	flag.Usage = func() {
		fmt.Fprintf(os.Stderr, `rlm - Recursive Language Model CLI

Usage:
  rlm -context <file> -query "your question"
  rlm -context-string "data" -query "your question"
  cat file.txt | rlm -query "your question"

Subcommands:
  install-claude-code    Install RLM skill for Claude Code

Examples:
  rlm -context server.log -query "Find all error patterns"
  rlm -context data.json -query "Summarize the key findings" -verbose
  echo "long text..." | rlm -query "Analyze this"
  rlm install-claude-code

Options:
`)
		flag.PrintDefaults()
	}
	flag.Parse()

	// Determine provider and get API key
	isGemini := geminiModels[*model]
	var apiKey string
	var backend string

	if isGemini {
		apiKey = os.Getenv("GEMINI_API_KEY")
		if apiKey == "" {
			fmt.Fprintln(os.Stderr, "Error: GEMINI_API_KEY environment variable not set")
			os.Exit(1)
		}
		backend = "gemini"
	} else {
		apiKey = os.Getenv("ANTHROPIC_API_KEY")
		if apiKey == "" {
			fmt.Fprintln(os.Stderr, "Error: ANTHROPIC_API_KEY environment variable not set")
			os.Exit(1)
		}
		backend = "anthropic"
	}

	// Get context
	var contextData string
	switch {
	case *contextStr != "":
		contextData = *contextStr
	case *contextFile != "":
		data, err := os.ReadFile(*contextFile)
		if err != nil {
			fmt.Fprintf(os.Stderr, "Error reading context file: %v\n", err)
			os.Exit(1)
		}
		contextData = string(data)
	default:
		// Try reading from stdin
		stat, _ := os.Stdin.Stat()
		if (stat.Mode() & os.ModeCharDevice) == 0 {
			data, err := io.ReadAll(os.Stdin)
			if err != nil {
				fmt.Fprintf(os.Stderr, "Error reading stdin: %v\n", err)
				os.Exit(1)
			}
			contextData = string(data)
		}
	}

	if contextData == "" {
		fmt.Fprintln(os.Stderr, "Error: No context provided. Use -context, -context-string, or pipe to stdin")
		flag.Usage()
		os.Exit(1)
	}

	if *query == "" {
		fmt.Fprintln(os.Stderr, "Error: -query is required")
		flag.Usage()
		os.Exit(1)
	}

	// Create client based on provider
	var llmClient rlm.LLMClient
	var replClient repl.LLMClient

	if isGemini {
		client := NewGeminiClient(apiKey, *model, *verbose)
		llmClient = client
		replClient = client
	} else {
		client := NewAnthropicClient(apiKey, *model, *verbose)
		llmClient = client
		replClient = client
	}

	// Setup logger if requested
	var log *logger.Logger
	if *logDir != "" {
		var err error
		log, err = logger.New(*logDir, logger.Config{
			RootModel:     *model,
			MaxIterations: *maxIterations,
			Backend:       backend,
			BackendKwargs: map[string]any{"model_name": *model},
			Context:       contextData,
			Query:         *query,
		})
		if err != nil {
			fmt.Fprintf(os.Stderr, "Warning: could not create logger: %v\n", err)
		} else {
			defer func() { _ = log.Close() }()
			if *verbose {
				fmt.Fprintf(os.Stderr, "Logging to: %s\n", log.Path())
			}
		}
	}

	// Create RLM instance
	opts := []rlm.Option{
		rlm.WithMaxIterations(*maxIterations),
		rlm.WithVerbose(*verbose),
	}
	if log != nil {
		opts = append(opts, rlm.WithLogger(log))
	}
	r := rlm.New(llmClient, replClient, opts...)

	// Run
	if *verbose {
		fmt.Fprintf(os.Stderr, "Starting RLM completion...\n")
		fmt.Fprintf(os.Stderr, "Context size: %d characters\n", len(contextData))
		fmt.Fprintf(os.Stderr, "Model: %s\n", *model)
		fmt.Fprintf(os.Stderr, "Max iterations: %d\n\n", *maxIterations)
	}

	ctx := context.Background()
	result, err := r.Complete(ctx, contextData, *query)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}

	// Output
	if *jsonOutput {
		out := Result{
			Response:   result.Response,
			Iterations: result.Iterations,
			Duration:   result.Duration.String(),
		}
		out.Tokens.Prompt = result.Usage.PromptTokens
		out.Tokens.Completion = result.Usage.CompletionTokens
		out.Tokens.Total = result.Usage.TotalTokens
		enc := json.NewEncoder(os.Stdout)
		enc.SetIndent("", "  ")
		_ = enc.Encode(out)
	} else {
		if *verbose {
			fmt.Fprintln(os.Stderr, "\n"+strings.Repeat("=", 50))
			fmt.Fprintf(os.Stderr, "Iterations: %d\n", result.Iterations)
			fmt.Fprintf(os.Stderr, "Duration: %v\n", result.Duration)
			fmt.Fprintf(os.Stderr, "Tokens: %d prompt + %d completion = %d total\n",
				result.Usage.PromptTokens,
				result.Usage.CompletionTokens,
				result.Usage.TotalTokens)
			fmt.Fprintln(os.Stderr, strings.Repeat("=", 50))
		}
		fmt.Println(result.Response)
	}
}

// installClaudeCodePlugin installs the RLM skill for Claude Code.
func installClaudeCodePlugin() error {
	homeDir, err := os.UserHomeDir()
	if err != nil {
		return fmt.Errorf("failed to get home directory: %w", err)
	}

	skillsDir := filepath.Join(homeDir, ".claude", "skills", "rlm")

	// Create skills directory
	if err := os.MkdirAll(skillsDir, 0755); err != nil {
		return fmt.Errorf("failed to create directory %s: %w", skillsDir, err)
	}

	// Write SKILL.md
	skillMD := `---
name: rlm
description: Recursive Language Model for processing large contexts (>50KB). Use for complex analysis tasks where token efficiency matters. Achieves 40% token savings by letting the LLM programmatically explore context via Query() and FINAL() patterns.
allowed-tools:
  - Bash
---

# RLM - Recursive Language Model

**RLM** is an inference-time scaling strategy that enables LLMs to handle arbitrarily long contexts by treating prompts as external objects that can be programmatically examined and recursively processed.

- **License:** MIT
- **Repository:** https://github.com/XiaoConstantine/rlm-go

## When to Use

Use ` + "`rlm`" + ` instead of direct LLM calls when:
- Processing **large contexts** (>50KB of text)
- Token efficiency is important (40% savings on large contexts)
- The task requires **iterative exploration** of data
- Complex analysis that benefits from sub-queries

## Do NOT Use When

- Context is small (<10KB) - overhead not worth it
- Simple single-turn questions
- Tasks that don't require data exploration

## Command Usage

` + "```bash" + `
# Basic usage with context file
~/.local/bin/rlm -context <file> -query "<query>" -verbose

# With inline context
~/.local/bin/rlm -context-string "data" -query "<query>"

# Pipe context from stdin
cat largefile.txt | ~/.local/bin/rlm -query "<query>"

# JSON output for programmatic use
~/.local/bin/rlm -context <file> -query "<query>" -json
` + "```" + `

## Options

| Flag | Description | Default |
|------|-------------|---------|
| ` + "`-context`" + ` | Path to context file | - |
| ` + "`-context-string`" + ` | Context string directly | - |
| ` + "`-query`" + ` | Query to run against context | Required |
| ` + "`-model`" + ` | LLM model to use | claude-sonnet-4-20250514 |
| ` + "`-max-iterations`" + ` | Maximum iterations | 30 |
| ` + "`-verbose`" + ` | Enable verbose output | false |
| ` + "`-json`" + ` | Output result as JSON | false |
| ` + "`-log-dir`" + ` | Directory for JSONL logs | - |

## How It Works

RLM uses a Go REPL environment where LLM-generated code can:

1. **Access context** as a string variable
2. **Make recursive sub-LLM calls** via ` + "`Query()`" + ` for focused analysis
3. **Use standard Go operations** for text processing
4. **Signal completion** with ` + "`FINAL()`" + ` when done

### The Query() Pattern

` + "```go" + `
// LLM generates code like this inside the REPL:
chunk := context[0:10000]
summary := Query("Summarize the key findings in this text: " + chunk)
// ... iterate through more chunks
FINAL(combinedResult)
` + "```" + `

### The FINAL() Pattern

The LLM signals completion by calling:
- ` + "`FINAL(\"answer\")`" + ` - Return a string answer
- ` + "`FINAL_VAR(variableName)`" + ` - Return value of a variable

## Token Efficiency Benefits

For large contexts (>50KB), RLM typically achieves **40% token savings** by:
- Only sending relevant context chunks to sub-queries
- Avoiding repeated full-context processing
- Using programmatic iteration instead of full-context reasoning

## Examples

### Analyze Log Files
` + "```bash" + `
rlm -context server.log -query "Find all unique error patterns and their frequencies"
` + "```" + `

### Process JSON Data
` + "```bash" + `
rlm -context data.json -query "Extract all user IDs with failed transactions" -verbose
` + "```" + `

### Code Analysis
` + "```bash" + `
cat src/*.go | rlm -query "Identify all exported functions and their purposes"
` + "```" + `

## Requirements

- ` + "`ANTHROPIC_API_KEY`" + ` environment variable must be set
- Binary installed at ` + "`~/.local/bin/rlm`" + `

## Installation

` + "```bash" + `
# Quick install
curl -fsSL https://raw.githubusercontent.com/XiaoConstantine/rlm-go/main/install.sh | bash

# Or with Go
go install github.com/XiaoConstantine/rlm-go/cmd/rlm@latest
` + "```" + `
`
	if err := os.WriteFile(filepath.Join(skillsDir, "SKILL.md"), []byte(skillMD), 0644); err != nil {
		return fmt.Errorf("failed to write SKILL.md: %w", err)
	}

	fmt.Println("RLM skill installed for Claude Code")
	fmt.Printf("  Skill: %s\n", skillsDir)
	fmt.Println()
	fmt.Println("Restart Claude Code to activate the skill.")
	fmt.Println()
	fmt.Println("The skill provides:")
	fmt.Println("  - Documentation on when to use RLM for large context processing")
	fmt.Println("  - Command usage and examples")
	fmt.Println("  - Token efficiency guidelines")

	return nil
}
