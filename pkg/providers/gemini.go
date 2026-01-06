package providers

import (
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
	apiKey     string
	model      string
	httpClient *http.Client
	verbose    bool
	baseURL    string // For testing; defaults to Gemini API
}

// NewGeminiClient creates a new Gemini client.
func NewGeminiClient(apiKey, model string, verbose bool) *GeminiClient {
	return &GeminiClient{
		apiKey:  apiKey,
		model:   model,
		verbose: verbose,
		baseURL: "https://generativelanguage.googleapis.com",
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

	url := fmt.Sprintf("%s/v1beta/models/%s:generateContent?key=%s", c.baseURL, c.model, c.apiKey)
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
		fmt.Printf("  [API] %v, tokens: %d→%d\n",
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
