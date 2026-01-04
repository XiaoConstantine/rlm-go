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
	"github.com/XiaoConstantine/rlm-go/pkg/rlm"
)

// AnthropicClient implements both rlm.LLMClient and repl.LLMClient interfaces.
type AnthropicClient struct {
	apiKey     string
	model      string
	maxTokens  int
	httpClient *http.Client
}

// NewAnthropicClient creates a new Anthropic client.
func NewAnthropicClient(apiKey, model string) *AnthropicClient {
	return &AnthropicClient{
		apiKey:    apiKey,
		model:     model,
		maxTokens: 4096,
		httpClient: &http.Client{
			Timeout: 120 * time.Second,
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
	Error *struct {
		Message string `json:"message"`
	} `json:"error,omitempty"`
}

// Complete implements rlm.LLMClient for root LLM orchestration.
func (c *AnthropicClient) Complete(ctx context.Context, messages []core.Message) (string, error) {
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

	return c.doRequest(ctx, reqBody)
}

// Query implements repl.LLMClient for sub-LLM calls from REPL.
func (c *AnthropicClient) Query(ctx context.Context, prompt string) (string, error) {
	reqBody := anthropicRequest{
		Model:     c.model,
		MaxTokens: c.maxTokens,
		Messages: []anthropicMessage{
			{Role: "user", Content: prompt},
		},
	}

	return c.doRequest(ctx, reqBody)
}

// QueryBatched implements repl.LLMClient for concurrent sub-LLM calls.
func (c *AnthropicClient) QueryBatched(ctx context.Context, prompts []string) ([]string, error) {
	results := make([]string, len(prompts))
	var wg sync.WaitGroup
	var mu sync.Mutex

	for i, prompt := range prompts {
		wg.Add(1)
		go func(idx int, p string) {
			defer wg.Done()
			result, err := c.Query(ctx, p)
			mu.Lock()
			if err != nil {
				results[idx] = fmt.Sprintf("Error: %v", err)
			} else {
				results[idx] = result
			}
			mu.Unlock()
		}(i, prompt)
	}

	wg.Wait()
	return results, nil
}

// doRequest makes the actual HTTP request to Anthropic API.
func (c *AnthropicClient) doRequest(ctx context.Context, reqBody anthropicRequest) (string, error) {
	jsonBody, err := json.Marshal(reqBody)
	if err != nil {
		return "", fmt.Errorf("marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", "https://api.anthropic.com/v1/messages", bytes.NewReader(jsonBody))
	if err != nil {
		return "", fmt.Errorf("create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", c.apiKey)
	req.Header.Set("anthropic-version", "2023-06-01")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return "", fmt.Errorf("http request: %w", err)
	}
	defer resp.Body.Close()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", fmt.Errorf("read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return "", fmt.Errorf("api error (status %d): %s", resp.StatusCode, string(body))
	}

	var apiResp anthropicResponse
	if err := json.Unmarshal(body, &apiResp); err != nil {
		return "", fmt.Errorf("unmarshal response: %w", err)
	}

	if apiResp.Error != nil {
		return "", fmt.Errorf("api error: %s", apiResp.Error.Message)
	}

	// Extract text from response
	var texts []string
	for _, block := range apiResp.Content {
		if block.Type == "text" {
			texts = append(texts, block.Text)
		}
	}

	return strings.Join(texts, ""), nil
}

func main() {
	apiKey := os.Getenv("ANTHROPIC_API_KEY")
	if apiKey == "" {
		fmt.Println("Error: ANTHROPIC_API_KEY environment variable not set")
		os.Exit(1)
	}

	// Create Anthropic client
	client := NewAnthropicClient(apiKey, "claude-haiku-4-5")

	// Create RLM instance
	r := rlm.New(client, client,
		rlm.WithMaxIterations(10),
		rlm.WithVerbose(true),
	)

	document := `
  Review 1: This product is amazing, best purchase ever!
  Review 2: Terrible quality, broke after one day.
  Review 3: It's okay, nothing special.
  Review 4: Absolutely love it, highly recommend!
  Review 5: Waste of money, very disappointed.
`
	// Repeat to make it longer
	document = strings.Repeat(document, 200)

	ctx := context.Background()

	fmt.Println("Starting RLM completion...")
	fmt.Printf("Context size: %d characters\n\n", len(document))

	result, err := r.Complete(ctx, document, "What percentage of reviews are positive (4-5 stars) vs negative (1-2 stars)?")
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		os.Exit(1)
	}

	fmt.Println("\n" + strings.Repeat("=", 50))
	fmt.Printf("Final Answer: %s\n", result.Response)
	fmt.Printf("Iterations: %d\n", result.Iterations)
	fmt.Printf("Duration: %v\n", result.Duration)
}
