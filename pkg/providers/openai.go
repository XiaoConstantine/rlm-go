package providers

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"sync"
	"time"

	"github.com/XiaoConstantine/rlm-go/pkg/core"
	"github.com/XiaoConstantine/rlm-go/pkg/repl"
)

// OpenAIClient implements Client for OpenAI's API.
type OpenAIClient struct {
	apiKey     string
	model      string
	httpClient *http.Client
	verbose    bool
	baseURL    string // For testing; defaults to OpenAI API
}

// NewOpenAIClient creates a new OpenAI client.
func NewOpenAIClient(apiKey, model string, verbose bool) *OpenAIClient {
	return &OpenAIClient{
		apiKey:  apiKey,
		model:   model,
		verbose: verbose,
		baseURL: "https://api.openai.com",
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

type openaiRequest struct {
	Model    string          `json:"model"`
	Messages []openaiMessage `json:"messages"`
}

type openaiMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type openaiResponse struct {
	Choices []struct {
		Message struct {
			Content string `json:"content"`
		} `json:"message"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		TotalTokens      int `json:"total_tokens"`
	} `json:"usage"`
	Error *struct {
		Message string `json:"message"`
		Type    string `json:"type"`
	} `json:"error,omitempty"`
}

// Complete implements rlm.LLMClient for root LLM orchestration.
func (c *OpenAIClient) Complete(ctx context.Context, messages []core.Message) (core.LLMResponse, error) {
	var apiMessages []openaiMessage

	for _, msg := range messages {
		apiMessages = append(apiMessages, openaiMessage{
			Role:    msg.Role,
			Content: msg.Content,
		})
	}

	reqBody := openaiRequest{
		Model:    c.model,
		Messages: apiMessages,
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
func (c *OpenAIClient) Query(ctx context.Context, prompt string) (repl.QueryResponse, error) {
	reqBody := openaiRequest{
		Model: c.model,
		Messages: []openaiMessage{
			{Role: "user", Content: prompt},
		},
	}

	text, promptTokens, completionTokens, err := c.doRequest(ctx, reqBody)
	return repl.QueryResponse{
		Response:         text,
		PromptTokens:     promptTokens,
		CompletionTokens: completionTokens,
	}, err
}

// QueryBatched implements repl.LLMClient for concurrent sub-LLM calls.
func (c *OpenAIClient) QueryBatched(ctx context.Context, prompts []string) ([]repl.QueryResponse, error) {
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

func (c *OpenAIClient) doRequest(ctx context.Context, reqBody openaiRequest) (string, int, int, error) {
	start := time.Now()

	jsonBody, err := json.Marshal(reqBody)
	if err != nil {
		return "", 0, 0, fmt.Errorf("marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", c.baseURL+"/v1/chat/completions", bytes.NewReader(jsonBody))
	if err != nil {
		return "", 0, 0, fmt.Errorf("create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.apiKey)

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

	var apiResp openaiResponse
	if err := json.Unmarshal(body, &apiResp); err != nil {
		return "", 0, 0, fmt.Errorf("unmarshal response: %w", err)
	}

	if apiResp.Error != nil {
		return "", 0, 0, fmt.Errorf("api error: %s", apiResp.Error.Message)
	}

	if c.verbose {
		fmt.Printf("  [API] %v, tokens: %d→%d\n",
			time.Since(start), apiResp.Usage.PromptTokens, apiResp.Usage.CompletionTokens)
	}

	if len(apiResp.Choices) == 0 {
		return "", 0, 0, fmt.Errorf("no choices in response")
	}

	return apiResp.Choices[0].Message.Content, apiResp.Usage.PromptTokens, apiResp.Usage.CompletionTokens, nil
}
