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
)

// GeminiClient implements Client for Google's Gemini API.
type GeminiClient struct {
	apiKey        string
	model         string
	httpClient    *http.Client
	verbose       bool
	baseURL       string // For testing; defaults to Gemini API
	trackCaching  bool   // Track and report implicit caching metrics (default: true)
}

// GeminiClientOption is a functional option for configuring GeminiClient.
type GeminiClientOption func(*GeminiClient)

// WithGeminiCaching enables or disables tracking of Gemini's implicit caching metrics.
// Note: Gemini 2.5+ models use implicit caching automatically; this controls whether
// the client tracks and reports cached token metrics in responses.
func WithGeminiCaching(enabled bool) GeminiClientOption {
	return func(c *GeminiClient) {
		c.trackCaching = enabled
	}
}

// NewGeminiClient creates a new Gemini client.
func NewGeminiClient(apiKey, model string, verbose bool, opts ...GeminiClientOption) *GeminiClient {
	c := &GeminiClient{
		apiKey:       apiKey,
		model:        model,
		verbose:      verbose,
		baseURL:      "https://generativelanguage.googleapis.com",
		trackCaching: true, // Enabled by default for Gemini 2.5+ implicit caching
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

type geminiRequest struct {
	Contents          []geminiContent  `json:"contents"`
	SystemInstruction *geminiContent   `json:"systemInstruction,omitempty"`
	GenerationConfig  *geminiGenConfig `json:"generationConfig,omitempty"`
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
		PromptTokenCount          int `json:"promptTokenCount"`
		CandidatesTokenCount      int `json:"candidatesTokenCount"`
		TotalTokenCount           int `json:"totalTokenCount"`
		CachedContentTokenCount   int `json:"cachedContentTokenCount,omitempty"`
	} `json:"usageMetadata"`
	Error *struct {
		Message string `json:"message"`
		Code    int    `json:"code"`
	} `json:"error,omitempty"`
}

// prepareMessages extracts system instruction and converts messages to API format.
func (c *GeminiClient) prepareMessages(messages []core.Message) (*geminiContent, []geminiContent) {
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
	return systemContent, contents
}

// Complete implements rlm.LLMClient for root LLM orchestration.
func (c *GeminiClient) Complete(ctx context.Context, messages []core.Message) (core.LLMResponse, error) {
	systemContent, contents := c.prepareMessages(messages)

	reqBody := geminiRequest{
		Contents:          contents,
		SystemInstruction: systemContent,
		GenerationConfig:  &geminiGenConfig{MaxOutputTokens: 8192},
	}

	text, stats, err := c.doRequest(ctx, reqBody)
	if err != nil {
		return core.LLMResponse{}, err
	}
	resp := core.LLMResponse{
		Content:          text,
		PromptTokens:     stats.promptTokens,
		CompletionTokens: stats.completionTokens,
	}
	// For Gemini implicit caching, cachedContentTokenCount represents tokens read from cache.
	// Map this to CacheReadTokens for consistency with Anthropic's interface.
	if c.trackCaching && stats.cachedContentTokens > 0 {
		resp.CacheReadTokens = stats.cachedContentTokens
	}
	return resp, nil
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

	text, stats, err := c.doRequest(ctx, reqBody)
	return repl.QueryResponse{
		Response:         text,
		PromptTokens:     stats.promptTokens,
		CompletionTokens: stats.completionTokens,
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

// Model returns the model name used by this client.
func (c *GeminiClient) Model() string {
	return c.model
}

// geminiRequestStats holds token usage statistics from a Gemini request.
type geminiRequestStats struct {
	promptTokens       int
	completionTokens   int
	cachedContentTokens int
}

func (c *GeminiClient) doRequest(ctx context.Context, reqBody geminiRequest) (string, geminiRequestStats, error) {
	start := time.Now()
	empty := geminiRequestStats{}

	jsonBody, err := json.Marshal(reqBody)
	if err != nil {
		return "", empty, fmt.Errorf("marshal request: %w", err)
	}

	url := fmt.Sprintf("%s/v1beta/models/%s:generateContent", c.baseURL, c.model)
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(jsonBody))
	if err != nil {
		return "", empty, fmt.Errorf("create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-goog-api-key", c.apiKey)

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

	var apiResp geminiResponse
	if err := json.Unmarshal(body, &apiResp); err != nil {
		return "", empty, fmt.Errorf("unmarshal response: %w", err)
	}

	if apiResp.Error != nil {
		return "", empty, fmt.Errorf("api error: %s", apiResp.Error.Message)
	}

	if c.verbose {
		if c.trackCaching && apiResp.UsageMetadata.CachedContentTokenCount > 0 {
			fmt.Printf("  [API] %v, tokens: %d→%d (cached: %d)\n",
				time.Since(start), apiResp.UsageMetadata.PromptTokenCount, apiResp.UsageMetadata.CandidatesTokenCount,
				apiResp.UsageMetadata.CachedContentTokenCount)
		} else {
			fmt.Printf("  [API] %v, tokens: %d→%d\n",
				time.Since(start), apiResp.UsageMetadata.PromptTokenCount, apiResp.UsageMetadata.CandidatesTokenCount)
		}
	}

	var texts []string
	for _, candidate := range apiResp.Candidates {
		for _, part := range candidate.Content.Parts {
			texts = append(texts, part.Text)
		}
	}

	stats := geminiRequestStats{
		promptTokens:        apiResp.UsageMetadata.PromptTokenCount,
		completionTokens:    apiResp.UsageMetadata.CandidatesTokenCount,
		cachedContentTokens: apiResp.UsageMetadata.CachedContentTokenCount,
	}
	return strings.Join(texts, ""), stats, nil
}

// geminiStreamChunk represents a chunk from Gemini's streaming API.
type geminiStreamChunk struct {
	Candidates []struct {
		Content struct {
			Parts []struct {
				Text string `json:"text"`
			} `json:"parts"`
			Role string `json:"role"`
		} `json:"content"`
		FinishReason  string `json:"finishReason,omitempty"`
		SafetyRatings []struct {
			Category    string `json:"category"`
			Probability string `json:"probability"`
		} `json:"safetyRatings,omitempty"`
	} `json:"candidates"`
	UsageMetadata struct {
		PromptTokenCount        int `json:"promptTokenCount"`
		CandidatesTokenCount    int `json:"candidatesTokenCount"`
		TotalTokenCount         int `json:"totalTokenCount"`
		CachedContentTokenCount int `json:"cachedContentTokenCount,omitempty"`
	} `json:"usageMetadata,omitempty"`
	Error *struct {
		Message string `json:"message"`
		Code    int    `json:"code"`
	} `json:"error,omitempty"`
}

// CompleteStream performs a streaming completion request.
// The handler is called for each chunk of content as it arrives.
// Returns the complete response with token usage after stream completes.
func (c *GeminiClient) CompleteStream(ctx context.Context, messages []core.Message, handler StreamHandler) (core.LLMResponse, error) {
	systemContent, contents := c.prepareMessages(messages)

	reqBody := geminiRequest{
		Contents:          contents,
		SystemInstruction: systemContent,
		GenerationConfig:  &geminiGenConfig{MaxOutputTokens: 8192},
	}

	return c.doStreamRequest(ctx, reqBody, handler)
}

func (c *GeminiClient) doStreamRequest(ctx context.Context, reqBody geminiRequest, handler StreamHandler) (core.LLMResponse, error) {
	start := time.Now()

	jsonBody, err := json.Marshal(reqBody)
	if err != nil {
		return core.LLMResponse{}, fmt.Errorf("marshal request: %w", err)
	}

	url := fmt.Sprintf("%s/v1beta/models/%s:streamGenerateContent?alt=sse", c.baseURL, c.model)
	req, err := http.NewRequestWithContext(ctx, "POST", url, bytes.NewReader(jsonBody))
	if err != nil {
		return core.LLMResponse{}, fmt.Errorf("create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-goog-api-key", c.apiKey)

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return core.LLMResponse{}, fmt.Errorf("http request: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		return core.LLMResponse{}, fmt.Errorf("api error (status %d): %s", resp.StatusCode, string(body))
	}

	var fullContent strings.Builder
	var promptTokens, completionTokens, cachedContentTokens int
	var streamCompleted bool

	scanner := bufio.NewScanner(resp.Body)
	buf := make([]byte, 0, 64*1024)
	scanner.Buffer(buf, 1024*1024)

	for scanner.Scan() {
		select {
		case <-ctx.Done():
			return core.LLMResponse{}, ctx.Err()
		default:
		}

		line := scanner.Text()

		if !strings.HasPrefix(line, "data: ") {
			continue
		}

		data := strings.TrimPrefix(line, "data: ")
		if data == "" {
			continue
		}

		var chunk geminiStreamChunk
		if err := json.Unmarshal([]byte(data), &chunk); err != nil {
			continue
		}

		if chunk.Error != nil {
			return core.LLMResponse{}, fmt.Errorf("stream error: %s", chunk.Error.Message)
		}

		for _, candidate := range chunk.Candidates {
			for _, part := range candidate.Content.Parts {
				if part.Text != "" {
					fullContent.WriteString(part.Text)
					if handler != nil {
						if err := handler(part.Text, false); err != nil {
							return core.LLMResponse{}, fmt.Errorf("handler error: %w", err)
						}
					}
				}
			}
		}

		if chunk.UsageMetadata.PromptTokenCount > 0 {
			promptTokens = chunk.UsageMetadata.PromptTokenCount
		}
		if chunk.UsageMetadata.CandidatesTokenCount > 0 {
			completionTokens = chunk.UsageMetadata.CandidatesTokenCount
		}
		if chunk.UsageMetadata.CachedContentTokenCount > 0 {
			cachedContentTokens = chunk.UsageMetadata.CachedContentTokenCount
		}

		// Check for stream completion signal from Gemini
		for _, candidate := range chunk.Candidates {
			if candidate.FinishReason == "STOP" || candidate.FinishReason == "MAX_TOKENS" ||
				candidate.FinishReason == "SAFETY" || candidate.FinishReason == "RECITATION" {
				// Stream complete - signal handler and exit loop
				streamCompleted = true
				if handler != nil {
					if err := handler("", true); err != nil {
						return core.LLMResponse{}, fmt.Errorf("handler error: %w", err)
					}
				}
				goto streamDone
			}
		}
	}

streamDone:
	if err := scanner.Err(); err != nil {
		return core.LLMResponse{}, fmt.Errorf("scanner error: %w", err)
	}

	// Only call handler if we didn't already signal completion via finishReason
	if handler != nil && !streamCompleted {
		if err := handler("", true); err != nil {
			return core.LLMResponse{}, fmt.Errorf("handler error: %w", err)
		}
	}

	if c.verbose {
		if c.trackCaching && cachedContentTokens > 0 {
			fmt.Printf("  [API Stream] %v, tokens: %d→%d (cached: %d)\n",
				time.Since(start), promptTokens, completionTokens, cachedContentTokens)
		} else {
			fmt.Printf("  [API Stream] %v, tokens: %d→%d\n",
				time.Since(start), promptTokens, completionTokens)
		}
	}

	result := core.LLMResponse{
		Content:          fullContent.String(),
		PromptTokens:     promptTokens,
		CompletionTokens: completionTokens,
	}
	// For Gemini implicit caching, cachedContentTokenCount represents tokens read from cache.
	// Map this to CacheReadTokens for consistency with Anthropic's interface.
	if c.trackCaching && cachedContentTokens > 0 {
		result.CacheReadTokens = cachedContentTokens
	}
	return result, nil
}
