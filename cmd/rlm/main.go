// Package main provides a CLI for running RLM (Recursive Language Model) queries.
package main

import (
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"

	"github.com/XiaoConstantine/rlm-go/pkg/logger"
	"github.com/XiaoConstantine/rlm-go/pkg/providers"
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
  rlm -model gemini-3-flash-preview -context file.txt -query "Analyze this"
  rlm -model gpt-5 -context file.txt -query "Summarize"
  echo "long text..." | rlm -query "Analyze this"
  rlm install-claude-code

Options:
`)
		flag.PrintDefaults()
	}
	flag.Parse()

	// Determine provider and get API key
	provider := providers.GetProvider(*model)
	envKey := provider.EnvKey()
	apiKey := os.Getenv(envKey)
	if apiKey == "" {
		fmt.Fprintf(os.Stderr, "Error: %s environment variable not set\n", envKey)
		os.Exit(1)
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
	var client providers.Client
	switch provider {
	case providers.Gemini:
		client = providers.NewGeminiClient(apiKey, *model, *verbose)
	case providers.OpenAI:
		client = providers.NewOpenAIClient(apiKey, *model, *verbose)
	default:
		client = providers.NewAnthropicClient(apiKey, *model, *verbose)
	}

	// Setup logger if requested
	var log *logger.Logger
	if *logDir != "" {
		var err error
		log, err = logger.New(*logDir, logger.Config{
			RootModel:     *model,
			MaxIterations: *maxIterations,
			Backend:       string(provider),
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
	r := rlm.New(client, client, opts...)

	// Run
	if *verbose {
		fmt.Fprintf(os.Stderr, "Starting RLM completion...\n")
		fmt.Fprintf(os.Stderr, "Context size: %d characters\n", len(contextData))
		fmt.Fprintf(os.Stderr, "Model: %s (%s)\n", *model, provider)
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
# Basic usage with context file (uses Anthropic by default)
~/.local/bin/rlm -context <file> -query "<query>" -verbose

# Use Gemini
~/.local/bin/rlm -model gemini-3-flash-preview -context <file> -query "<query>"

# Use OpenAI
~/.local/bin/rlm -model gpt-5-mini -context <file> -query "<query>"

# With inline context
~/.local/bin/rlm -context-string "data" -query "<query>"

# Pipe context from stdin
cat largefile.txt | ~/.local/bin/rlm -query "<query>"

# JSON output for programmatic use
~/.local/bin/rlm -context <file> -query "<query>" -json
` + "```" + `

## Supported Models

| Provider | Models | Env Variable |
|----------|--------|--------------|
| Anthropic | claude-* (default) | ANTHROPIC_API_KEY |
| Google | gemini-3-flash-preview, gemini-3-pro-preview | GEMINI_API_KEY |
| OpenAI | gpt-5, gpt-5-mini | OPENAI_API_KEY |

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
