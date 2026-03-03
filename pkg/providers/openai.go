package providers

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"

	"github.com/XiaoConstantine/rlm-go/pkg/core"
	"github.com/XiaoConstantine/rlm-go/pkg/rlm"
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
	Model         string               `json:"model"`
	Messages      []openaiMessage      `json:"messages"`
	Stream        bool                 `json:"stream,omitempty"`
	StreamOptions *openaiStreamOptions `json:"stream_options,omitempty"`
}

type openaiStreamOptions struct {
	IncludeUsage bool `json:"include_usage"`
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

// openaiStreamEvent represents a streaming response chunk from OpenAI.
type openaiStreamEvent struct {
	Choices []struct {
		Delta struct {
			Content string `json:"content,omitempty"`
		} `json:"delta"`
		FinishReason string `json:"finish_reason,omitempty"`
	} `json:"choices"`
	Usage *struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
	} `json:"usage,omitempty"`
}

// Complete implements rlm.LLMClient for root LLM orchestration.
func (c *OpenAIClient) Complete(ctx context.Context, messages []core.Message) (core.LLMResponse, error) {
	apiMessages := make([]openaiMessage, 0, len(messages))
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

// Query implements LLMClient for sub-LLM calls from REPL.
func (c *OpenAIClient) Query(ctx context.Context, prompt string) (core.QueryResponse, error) {
	reqBody := openaiRequest{
		Model: c.model,
		Messages: []openaiMessage{
			{Role: "user", Content: prompt},
		},
	}

	text, promptTokens, completionTokens, err := c.doRequest(ctx, reqBody)
	return core.QueryResponse{
		Response:         text,
		PromptTokens:     promptTokens,
		CompletionTokens: completionTokens,
	}, err
}

// QueryBatched implements LLMClient for concurrent sub-LLM calls.
func (c *OpenAIClient) QueryBatched(ctx context.Context, prompts []string) ([]core.QueryResponse, error) {
	return QueryBatchedConcurrent(ctx, prompts, c.Query)
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

// CompleteStream performs a streaming completion request.
// The handler is called for each chunk of content as it arrives.
// Returns the complete response with token usage after stream completes.
func (c *OpenAIClient) CompleteStream(ctx context.Context, messages []core.Message, handler rlm.StreamHandler) (core.LLMResponse, error) {
	apiMessages := make([]openaiMessage, 0, len(messages))
	for _, msg := range messages {
		apiMessages = append(apiMessages, openaiMessage{
			Role:    msg.Role,
			Content: msg.Content,
		})
	}

	reqBody := openaiRequest{
		Model:    c.model,
		Messages: apiMessages,
		Stream:   true,
		StreamOptions: &openaiStreamOptions{
			IncludeUsage: true,
		},
	}

	return c.doStreamRequest(ctx, reqBody, handler)
}

func (c *OpenAIClient) doStreamRequest(ctx context.Context, reqBody openaiRequest, handler rlm.StreamHandler) (core.LLMResponse, error) {
	start := time.Now()

	jsonBody, err := json.Marshal(reqBody)
	if err != nil {
		return core.LLMResponse{}, fmt.Errorf("marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", c.baseURL+"/v1/chat/completions", bytes.NewReader(jsonBody))
	if err != nil {
		return core.LLMResponse{}, fmt.Errorf("create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.apiKey)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return core.LLMResponse{}, fmt.Errorf("http request: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return core.LLMResponse{}, fmt.Errorf("api error (status %d): %s", resp.StatusCode, string(body))
	}

	// Parse SSE stream
	var fullContent strings.Builder
	var promptTokens, completionTokens int

	scanner := bufio.NewScanner(resp.Body)
	buf := make([]byte, 0, 4*1024)
	scanner.Buffer(buf, 1024*1024)

	for scanner.Scan() {
		// Check for cancellation before processing buffered lines.
		if ctx.Err() != nil {
			return core.LLMResponse{}, ctx.Err()
		}

		line := scanner.Text()

		// SSE format: "data: <json>" or "data: [DONE]"
		if !strings.HasPrefix(line, "data: ") {
			continue
		}

		data := strings.TrimPrefix(line, "data: ")
		if data == "" || data == "[DONE]" {
			continue
		}

		var event openaiStreamEvent
		if err := json.Unmarshal([]byte(data), &event); err != nil {
			continue
		}

		// Extract content from delta
		if len(event.Choices) > 0 {
			delta := event.Choices[0].Delta
			if delta.Content != "" {
				fullContent.WriteString(delta.Content)
				if handler != nil {
					if err := handler(delta.Content, false); err != nil {
						return core.LLMResponse{}, fmt.Errorf("handler error: %w", err)
					}
				}
			}

			// Check for finish
			if event.Choices[0].FinishReason != "" && handler != nil {
				if err := handler("", true); err != nil {
					return core.LLMResponse{}, fmt.Errorf("handler error: %w", err)
				}
			}
		}

		// Extract usage from final chunk (with stream_options.include_usage)
		if event.Usage != nil {
			promptTokens = event.Usage.PromptTokens
			completionTokens = event.Usage.CompletionTokens
		}
	}

	if err := scanner.Err(); err != nil {
		// Context cancellation closes the connection, surfacing as a read error.
		if ctx.Err() != nil {
			return core.LLMResponse{}, ctx.Err()
		}
		return core.LLMResponse{}, fmt.Errorf("scanner error: %w", err)
	}

	if c.verbose {
		fmt.Printf("  [API Stream] %v, tokens: %d→%d\n",
			time.Since(start), promptTokens, completionTokens)
	}

	return core.LLMResponse{
		Content:          fullContent.String(),
		PromptTokens:     promptTokens,
		CompletionTokens: completionTokens,
	}, nil
}
