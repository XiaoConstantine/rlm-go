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
	"sync"
	"time"

	"github.com/XiaoConstantine/rlm-go/pkg/core"
	"github.com/XiaoConstantine/rlm-go/pkg/repl"
	"github.com/XiaoConstantine/rlm-go/pkg/rlm"
)

// AnthropicClient implements Client for Anthropic's Claude API.
type AnthropicClient struct {
	apiKey              string
	model               string
	maxTokens           int
	httpClient          *http.Client
	verbose             bool
	baseURL             string // For testing; defaults to Anthropic API
	enablePrefixCaching bool   // Enable Anthropic's prompt caching (default: true)
}

// AnthropicClientOption is a functional option for configuring AnthropicClient.
type AnthropicClientOption func(*AnthropicClient)

// WithPrefixCaching enables or disables Anthropic's prefix caching feature.
func WithPrefixCaching(enabled bool) AnthropicClientOption {
	return func(c *AnthropicClient) {
		c.enablePrefixCaching = enabled
	}
}

// SystemContentBlock represents a content block in the system prompt with optional cache control.
type SystemContentBlock struct {
	Type         string        `json:"type"`
	Text         string        `json:"text"`
	CacheControl *CacheControl `json:"cache_control,omitempty"`
}

// CacheControl specifies caching behavior for a content block.
type CacheControl struct {
	Type string `json:"type"` // "ephemeral"
}

// NewAnthropicClient creates a new Anthropic client.
func NewAnthropicClient(apiKey, model string, verbose bool, opts ...AnthropicClientOption) *AnthropicClient {
	c := &AnthropicClient{
		apiKey:              apiKey,
		model:               model,
		maxTokens:           4096,
		verbose:             verbose,
		baseURL:             "https://api.anthropic.com",
		enablePrefixCaching: true, // Enabled by default
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
	for _, opt := range opts {
		opt(c)
	}
	return c
}

type anthropicRequest struct {
	Model     string               `json:"model"`
	MaxTokens int                  `json:"max_tokens"`
	Messages  []anthropicMessage   `json:"messages"`
	System    any                  `json:"system,omitempty"` // string or []SystemContentBlock
	Stream    bool                 `json:"stream,omitempty"`
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
		InputTokens              int `json:"input_tokens"`
		OutputTokens             int `json:"output_tokens"`
		CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
		CacheReadInputTokens     int `json:"cache_read_input_tokens"`
	} `json:"usage"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error,omitempty"`
}

// StreamHandler is an alias for rlm.StreamHandler to ensure interface compatibility.
type StreamHandler = rlm.StreamHandler

// streamEvent represents a Server-Sent Event from Anthropic's streaming API.
type streamEvent struct {
	Type  string `json:"type"`
	Index int    `json:"index,omitempty"`
	Delta struct {
		Type string `json:"type,omitempty"`
		Text string `json:"text,omitempty"`
	} `json:"delta,omitempty"`
	ContentBlock struct {
		Type string `json:"type,omitempty"`
		Text string `json:"text,omitempty"`
	} `json:"content_block,omitempty"`
	Message struct {
		Usage struct {
			InputTokens              int `json:"input_tokens"`
			OutputTokens             int `json:"output_tokens"`
			CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
			CacheReadInputTokens     int `json:"cache_read_input_tokens"`
		} `json:"usage,omitempty"`
	} `json:"message,omitempty"`
	Usage struct {
		OutputTokens int `json:"output_tokens"`
	} `json:"usage,omitempty"`
}

// prepareMessages extracts system prompt and converts messages to API format.
func (c *AnthropicClient) prepareMessages(messages []core.Message) (string, []anthropicMessage) {
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
	return systemPrompt, apiMessages
}

// Complete implements rlm.LLMClient for root LLM orchestration.
func (c *AnthropicClient) Complete(ctx context.Context, messages []core.Message) (core.LLMResponse, error) {
	systemPrompt, apiMessages := c.prepareMessages(messages)

	reqBody := anthropicRequest{
		Model:     c.model,
		MaxTokens: c.maxTokens,
		Messages:  apiMessages,
		System:    c.buildSystemPrompt(systemPrompt),
	}

	text, stats, err := c.doRequest(ctx, reqBody)
	if err != nil {
		return core.LLMResponse{}, err
	}
	return core.LLMResponse{
		Content:             text,
		PromptTokens:        stats.inputTokens,
		CompletionTokens:    stats.outputTokens,
		CacheCreationTokens: stats.cacheCreationTokens,
		CacheReadTokens:     stats.cacheReadTokens,
	}, nil
}

// buildSystemPrompt builds the system prompt with optional cache control.
func (c *AnthropicClient) buildSystemPrompt(prompt string) any {
	if prompt == "" {
		return nil
	}
	if !c.enablePrefixCaching {
		return prompt
	}
	return []SystemContentBlock{
		{
			Type: "text",
			Text: prompt,
			CacheControl: &CacheControl{
				Type: "ephemeral",
			},
		},
	}
}

// Model returns the model name used by this client.
func (c *AnthropicClient) Model() string {
	return c.model
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

	text, stats, err := c.doRequest(ctx, reqBody)
	return repl.QueryResponse{
		Response:         text,
		PromptTokens:     stats.inputTokens,
		CompletionTokens: stats.outputTokens,
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

type requestStats struct {
	inputTokens         int
	outputTokens        int
	cacheCreationTokens int
	cacheReadTokens     int
}

func (c *AnthropicClient) doRequest(ctx context.Context, reqBody anthropicRequest) (string, requestStats, error) {
	start := time.Now()
	empty := requestStats{}

	jsonBody, err := json.Marshal(reqBody)
	if err != nil {
		return "", empty, fmt.Errorf("marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", c.baseURL+"/v1/messages", bytes.NewReader(jsonBody))
	if err != nil {
		return "", empty, fmt.Errorf("create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", c.apiKey)
	req.Header.Set("anthropic-version", "2023-06-01")
	if c.enablePrefixCaching {
		req.Header.Set("anthropic-beta", "prompt-caching-2024-07-31")
	}

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return "", empty, fmt.Errorf("http request: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", empty, fmt.Errorf("read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return "", empty, fmt.Errorf("api error (status %d): %s", resp.StatusCode, string(body))
	}

	var apiResp anthropicResponse
	if err := json.Unmarshal(body, &apiResp); err != nil {
		return "", empty, fmt.Errorf("unmarshal response: %w", err)
	}

	if c.verbose {
		if c.enablePrefixCaching && (apiResp.Usage.CacheCreationInputTokens > 0 || apiResp.Usage.CacheReadInputTokens > 0) {
			fmt.Printf("  [API] %v, tokens: %d→%d (cache: +%d created, %d read)\n",
				time.Since(start), apiResp.Usage.InputTokens, apiResp.Usage.OutputTokens,
				apiResp.Usage.CacheCreationInputTokens, apiResp.Usage.CacheReadInputTokens)
		} else {
			fmt.Printf("  [API] %v, tokens: %d→%d\n",
				time.Since(start), apiResp.Usage.InputTokens, apiResp.Usage.OutputTokens)
		}
	}

	if apiResp.Error != nil {
		return "", empty, fmt.Errorf("api error: %s", apiResp.Error.Message)
	}

	var texts []string
	for _, block := range apiResp.Content {
		if block.Type == "text" {
			texts = append(texts, block.Text)
		}
	}

	stats := requestStats{
		inputTokens:         apiResp.Usage.InputTokens,
		outputTokens:        apiResp.Usage.OutputTokens,
		cacheCreationTokens: apiResp.Usage.CacheCreationInputTokens,
		cacheReadTokens:     apiResp.Usage.CacheReadInputTokens,
	}
	return strings.Join(texts, ""), stats, nil
}

// CompleteStream performs a streaming completion request.
// The handler is called for each chunk of content as it arrives.
// Returns the complete response with token usage after stream completes.
func (c *AnthropicClient) CompleteStream(ctx context.Context, messages []core.Message, handler StreamHandler) (core.LLMResponse, error) {
	systemPrompt, apiMessages := c.prepareMessages(messages)

	reqBody := anthropicRequest{
		Model:     c.model,
		MaxTokens: c.maxTokens,
		Messages:  apiMessages,
		System:    c.buildSystemPrompt(systemPrompt),
		Stream:    true,
	}

	return c.doStreamRequest(ctx, reqBody, handler)
}

func (c *AnthropicClient) doStreamRequest(ctx context.Context, reqBody anthropicRequest, handler StreamHandler) (core.LLMResponse, error) {
	start := time.Now()

	jsonBody, err := json.Marshal(reqBody)
	if err != nil {
		return core.LLMResponse{}, fmt.Errorf("marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", c.baseURL+"/v1/messages", bytes.NewReader(jsonBody))
	if err != nil {
		return core.LLMResponse{}, fmt.Errorf("create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", c.apiKey)
	req.Header.Set("anthropic-version", "2023-06-01")
	if c.enablePrefixCaching {
		req.Header.Set("anthropic-beta", "prompt-caching-2024-07-31")
	}

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
	var inputTokens, outputTokens int
	var cacheCreationTokens, cacheReadTokens int

	scanner := bufio.NewScanner(resp.Body)
	// Increase buffer size for potentially large events
	buf := make([]byte, 0, 64*1024)
	scanner.Buffer(buf, 1024*1024)

	for scanner.Scan() {
		select {
		case <-ctx.Done():
			return core.LLMResponse{}, ctx.Err()
		default:
		}

		line := scanner.Text()

		// SSE format: "event: <type>\ndata: <json>"
		if !strings.HasPrefix(line, "data: ") {
			continue
		}

		data := strings.TrimPrefix(line, "data: ")
		if data == "" {
			continue
		}

		var event streamEvent
		if err := json.Unmarshal([]byte(data), &event); err != nil {
			// Skip malformed events
			continue
		}

		switch event.Type {
		case "message_start":
			// Initial message with input token count and cache stats
			inputTokens = event.Message.Usage.InputTokens
			cacheCreationTokens = event.Message.Usage.CacheCreationInputTokens
			cacheReadTokens = event.Message.Usage.CacheReadInputTokens

		case "content_block_delta":
			// Content delta - text chunk
			if event.Delta.Type == "text_delta" && event.Delta.Text != "" {
				fullContent.WriteString(event.Delta.Text)
				if handler != nil {
					if err := handler(event.Delta.Text, false); err != nil {
						return core.LLMResponse{}, fmt.Errorf("handler error: %w", err)
					}
				}
			}

		case "message_delta":
			// Final message delta with output token count
			outputTokens = event.Usage.OutputTokens

		case "message_stop":
			// Stream complete
			if handler != nil {
				if err := handler("", true); err != nil {
					return core.LLMResponse{}, fmt.Errorf("handler error: %w", err)
				}
			}

		case "error":
			return core.LLMResponse{}, fmt.Errorf("stream error from API")
		}
	}

	if err := scanner.Err(); err != nil {
		return core.LLMResponse{}, fmt.Errorf("scanner error: %w", err)
	}

	if c.verbose {
		if c.enablePrefixCaching && (cacheCreationTokens > 0 || cacheReadTokens > 0) {
			fmt.Printf("  [API Stream] %v, tokens: %d→%d (cache: +%d created, %d read)\n",
				time.Since(start), inputTokens, outputTokens, cacheCreationTokens, cacheReadTokens)
		} else {
			fmt.Printf("  [API Stream] %v, tokens: %d→%d\n",
				time.Since(start), inputTokens, outputTokens)
		}
	}

	return core.LLMResponse{
		Content:          fullContent.String(),
		PromptTokens:     inputTokens,
		CompletionTokens: outputTokens,
	}, nil
}
