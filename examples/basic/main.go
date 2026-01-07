// Package main demonstrates basic usage of rlm-go with Anthropic.
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"

	"github.com/XiaoConstantine/rlm-go/pkg/core"
	"github.com/XiaoConstantine/rlm-go/pkg/logger"
	"github.com/XiaoConstantine/rlm-go/pkg/repl"
	"github.com/XiaoConstantine/rlm-go/pkg/rlm"
)

// AnthropicClient implements both rlm.LLMClient and repl.LLMClient interfaces.
// Note: Streaming is optional - if not implemented, RLM falls back to non-streaming.
type AnthropicClient struct {
	apiKey     string
	model      string
	maxTokens  int
	httpClient *http.Client
}

// NewAnthropicClient creates a new Anthropic client with connection pooling.
func NewAnthropicClient(apiKey, model string) *AnthropicClient {
	return &AnthropicClient{
		apiKey:    apiKey,
		model:     model,
		maxTokens: 4096,
		httpClient: &http.Client{
			Timeout: 120 * time.Second,
			Transport: &http.Transport{
				MaxIdleConns:        100,
				MaxIdleConnsPerHost: 10,
				MaxConnsPerHost:     10,
				IdleConnTimeout:     90 * time.Second,
			},
		},
	}
}

// anthropicRequest represents the request body for Anthropic API.
type anthropicRequest struct {
	Model     string              `json:"model"`
	MaxTokens int                 `json:"max_tokens"`
	Messages  []anthropicMessage  `json:"messages"`
	System    string              `json:"system,omitempty"`
}

type anthropicMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// anthropicResponse represents the response from Anthropic API.
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
	// Extract system message if present
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
// Each goroutine writes to its own unique slice index, so no mutex is needed.
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

// doRequest makes the actual HTTP request to Anthropic API.
// Returns (text, inputTokens, outputTokens, error).
func (c *AnthropicClient) doRequest(ctx context.Context, reqBody anthropicRequest) (string, int, int, error) {
	start := time.Now()

	jsonBody, err := json.Marshal(reqBody)
	if err != nil {
		return "", 0, 0, fmt.Errorf("marshal request: %w", err)
	}
	marshalTime := time.Since(start)

	req, err := http.NewRequestWithContext(ctx, "POST", "https://api.anthropic.com/v1/messages", bytes.NewReader(jsonBody))
	if err != nil {
		return "", 0, 0, fmt.Errorf("create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", c.apiKey)
	req.Header.Set("anthropic-version", "2023-06-01")

	httpStart := time.Now()
	resp, err := c.httpClient.Do(req)
	if err != nil {
		return "", 0, 0, fmt.Errorf("http request: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()
	httpTime := time.Since(httpStart)

	readStart := time.Now()
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", 0, 0, fmt.Errorf("read response: %w", err)
	}
	readTime := time.Since(readStart)

	if resp.StatusCode != http.StatusOK {
		return "", 0, 0, fmt.Errorf("api error (status %d): %s", resp.StatusCode, string(body))
	}

	unmarshalStart := time.Now()
	var apiResp anthropicResponse
	if err := json.Unmarshal(body, &apiResp); err != nil {
		return "", 0, 0, fmt.Errorf("unmarshal response: %w", err)
	}
	unmarshalTime := time.Since(unmarshalStart)

	fmt.Printf("  [HTTP] marshal=%v, http=%v, read=%v, unmarshal=%v, total=%v\n",
		marshalTime, httpTime, readTime, unmarshalTime, time.Since(start))

	if apiResp.Error != nil {
		return "", 0, 0, fmt.Errorf("api error: %s", apiResp.Error.Message)
	}

	// Extract text from response
	var texts []string
	for _, block := range apiResp.Content {
		if block.Type == "text" {
			texts = append(texts, block.Text)
		}
	}

	return strings.Join(texts, ""), apiResp.Usage.InputTokens, apiResp.Usage.OutputTokens, nil
}

func main() {
	apiKey := os.Getenv("ANTHROPIC_API_KEY")
	if apiKey == "" {
		fmt.Println("Error: ANTHROPIC_API_KEY environment variable not set")
		os.Exit(1)
	}

	model := "claude-haiku-4-5"

	// Create Anthropic client
	client := NewAnthropicClient(apiKey, model)

	document := `
  Review 1: This product is amazing, best purchase ever!
  Review 2: Terrible quality, broke after one day.
  Review 3: It's okay, nothing special.
  Review 4: Absolutely love it, highly recommend!
  Review 5: Waste of money, very disappointed.
`
	// Repeat to make it longer
	document = strings.Repeat(document, 200)

	query := "What percentage of reviews are positive (4-5 stars) vs negative (1-2 stars)?"

	// Create logger with context and query
	log, err := logger.New("logs", logger.Config{
		RootModel:     model,
		MaxIterations: 10,
		Backend:       "anthropic",
		BackendKwargs: map[string]any{"model_name": model},
		Context:       document,
		Query:         query,
	})
	if err != nil {
		fmt.Printf("Warning: could not create logger: %v\n", err)
	} else {
		defer func() { _ = log.Close() }()
		fmt.Printf("Logging to: %s\n", log.Path())
	}

	// Create REPL pool for performance
	pool := repl.NewREPLPool(client, 3, true) // 3 REPLs, pre-warmed

	// Create RLM instance with optimizations
	r := rlm.New(client, client,
		rlm.WithMaxIterations(10),
		rlm.WithVerbose(true),
		rlm.WithLogger(log),
		rlm.WithREPLPool(pool),
		rlm.WithHistoryCompression(3, 500),
	)

	ctx := context.Background()

	fmt.Println("Starting RLM completion...")
	fmt.Printf("Context size: %d characters\n\n", len(document))

	result, err := r.Complete(ctx, document, query)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		os.Exit(1)
	}

	fmt.Println("\n" + strings.Repeat("=", 50))
	fmt.Printf("Final Answer: %s\n", result.Response)
	fmt.Printf("Iterations: %d\n", result.Iterations)
	fmt.Printf("Duration: %v\n", result.Duration)
}
