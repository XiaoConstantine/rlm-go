package providers

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/XiaoConstantine/rlm-go/pkg/core"
)

func TestGeminiClient_Complete_Success(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" {
			t.Errorf("Expected POST, got %s", r.Method)
		}
		if !strings.Contains(r.URL.Path, "/v1beta/models/") {
			t.Errorf("Expected /v1beta/models/ in path, got %s", r.URL.Path)
		}
		if !strings.Contains(r.URL.RawQuery, "key=test-key") {
			t.Errorf("Expected key=test-key in query, got %s", r.URL.RawQuery)
		}

		var req geminiRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("Failed to decode request: %v", err)
		}
		if req.SystemInstruction == nil {
			t.Error("Expected system instruction")
		}

		resp := geminiResponse{
			Candidates: []struct {
				Content struct {
					Parts []struct {
						Text string `json:"text"`
					} `json:"parts"`
				} `json:"content"`
			}{
				{
					Content: struct {
						Parts []struct {
							Text string `json:"text"`
						} `json:"parts"`
					}{
						Parts: []struct {
							Text string `json:"text"`
						}{
							{Text: "Hello from Gemini!"},
						},
					},
				},
			},
			UsageMetadata: struct {
				PromptTokenCount        int `json:"promptTokenCount"`
				CandidatesTokenCount    int `json:"candidatesTokenCount"`
				TotalTokenCount         int `json:"totalTokenCount"`
				CachedContentTokenCount int `json:"cachedContentTokenCount,omitempty"`
			}{
				PromptTokenCount:     15,
				CandidatesTokenCount: 8,
				TotalTokenCount:      23,
			},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	client := NewGeminiClient("test-key", "gemini-2.5-flash", false)
	client.baseURL = server.URL

	messages := []core.Message{
		{Role: "system", Content: "You are helpful"},
		{Role: "user", Content: "Hello"},
	}

	resp, err := client.Complete(context.Background(), messages)
	if err != nil {
		t.Fatalf("Complete failed: %v", err)
	}
	if resp.Content != "Hello from Gemini!" {
		t.Errorf("Expected 'Hello from Gemini!', got %s", resp.Content)
	}
	if resp.PromptTokens != 15 {
		t.Errorf("Expected 15 prompt tokens, got %d", resp.PromptTokens)
	}
	if resp.CompletionTokens != 8 {
		t.Errorf("Expected 8 completion tokens, got %d", resp.CompletionTokens)
	}
}

func TestGeminiClient_Complete_RoleMapping(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req geminiRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("Failed to decode request: %v", err)
		}
		// Verify assistant role is mapped to model
		for _, content := range req.Contents {
			if content.Role == "assistant" {
				t.Error("assistant role should be mapped to model")
			}
		}

		resp := geminiResponse{
			Candidates: []struct {
				Content struct {
					Parts []struct {
						Text string `json:"text"`
					} `json:"parts"`
				} `json:"content"`
			}{
				{
					Content: struct {
						Parts []struct {
							Text string `json:"text"`
						} `json:"parts"`
					}{
						Parts: []struct {
							Text string `json:"text"`
						}{
							{Text: "Response"},
						},
					},
				},
			},
			UsageMetadata: struct {
				PromptTokenCount        int `json:"promptTokenCount"`
				CandidatesTokenCount    int `json:"candidatesTokenCount"`
				TotalTokenCount         int `json:"totalTokenCount"`
				CachedContentTokenCount int `json:"cachedContentTokenCount,omitempty"`
			}{},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	client := NewGeminiClient("test-key", "gemini-2.5-flash", false)
	client.baseURL = server.URL

	messages := []core.Message{
		{Role: "user", Content: "Hello"},
		{Role: "assistant", Content: "Hi there"},
		{Role: "user", Content: "How are you?"},
	}

	_, err := client.Complete(context.Background(), messages)
	if err != nil {
		t.Fatalf("Complete failed: %v", err)
	}
}

func TestGeminiClient_Complete_Error(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"error": {"message": "Invalid request", "code": 400}}`))
	}))
	defer server.Close()

	client := NewGeminiClient("test-key", "gemini-2.5-flash", false)
	client.baseURL = server.URL

	_, err := client.Complete(context.Background(), []core.Message{{Role: "user", Content: "Hi"}})
	if err == nil {
		t.Error("Expected error, got nil")
	}
}

func TestGeminiClient_Query_Success(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := geminiResponse{
			Candidates: []struct {
				Content struct {
					Parts []struct {
						Text string `json:"text"`
					} `json:"parts"`
				} `json:"content"`
			}{
				{
					Content: struct {
						Parts []struct {
							Text string `json:"text"`
						} `json:"parts"`
					}{
						Parts: []struct {
							Text string `json:"text"`
						}{
							{Text: "Query response"},
						},
					},
				},
			},
			UsageMetadata: struct {
				PromptTokenCount        int `json:"promptTokenCount"`
				CandidatesTokenCount    int `json:"candidatesTokenCount"`
				TotalTokenCount         int `json:"totalTokenCount"`
				CachedContentTokenCount int `json:"cachedContentTokenCount,omitempty"`
			}{
				PromptTokenCount:     5,
				CandidatesTokenCount: 3,
			},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	client := NewGeminiClient("test-key", "gemini-2.5-flash", false)
	client.baseURL = server.URL

	resp, err := client.Query(context.Background(), "Test prompt")
	if err != nil {
		t.Fatalf("Query failed: %v", err)
	}
	if resp.Response != "Query response" {
		t.Errorf("Expected 'Query response', got %s", resp.Response)
	}
}

func TestGeminiClient_QueryBatched_Success(t *testing.T) {
	var callCount int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&callCount, 1)
		resp := geminiResponse{
			Candidates: []struct {
				Content struct {
					Parts []struct {
						Text string `json:"text"`
					} `json:"parts"`
				} `json:"content"`
			}{
				{
					Content: struct {
						Parts []struct {
							Text string `json:"text"`
						} `json:"parts"`
					}{
						Parts: []struct {
							Text string `json:"text"`
						}{
							{Text: "Response"},
						},
					},
				},
			},
			UsageMetadata: struct {
				PromptTokenCount        int `json:"promptTokenCount"`
				CandidatesTokenCount    int `json:"candidatesTokenCount"`
				TotalTokenCount         int `json:"totalTokenCount"`
				CachedContentTokenCount int `json:"cachedContentTokenCount,omitempty"`
			}{},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	client := NewGeminiClient("test-key", "gemini-2.5-flash", false)
	client.baseURL = server.URL

	prompts := []string{"Prompt 1", "Prompt 2"}
	results, err := client.QueryBatched(context.Background(), prompts)
	if err != nil {
		t.Fatalf("QueryBatched failed: %v", err)
	}
	if len(results) != 2 {
		t.Errorf("Expected 2 results, got %d", len(results))
	}
}

func TestGeminiClient_QueryBatched_Empty(t *testing.T) {
	client := NewGeminiClient("test-key", "gemini-2.5-flash", false)

	results, err := client.QueryBatched(context.Background(), []string{})
	if err != nil {
		t.Errorf("QueryBatched with empty input returned error: %v", err)
	}
	if len(results) != 0 {
		t.Errorf("Expected 0 results, got %d", len(results))
	}
}

func TestGeminiClient_Verbose(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := geminiResponse{
			Candidates: []struct {
				Content struct {
					Parts []struct {
						Text string `json:"text"`
					} `json:"parts"`
				} `json:"content"`
			}{
				{
					Content: struct {
						Parts []struct {
							Text string `json:"text"`
						} `json:"parts"`
					}{
						Parts: []struct {
							Text string `json:"text"`
						}{
							{Text: "Response"},
						},
					},
				},
			},
			UsageMetadata: struct {
				PromptTokenCount        int `json:"promptTokenCount"`
				CandidatesTokenCount    int `json:"candidatesTokenCount"`
				TotalTokenCount         int `json:"totalTokenCount"`
				CachedContentTokenCount int `json:"cachedContentTokenCount,omitempty"`
			}{
				PromptTokenCount:     10,
				CandidatesTokenCount: 5,
			},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	client := NewGeminiClient("test-key", "gemini-2.5-flash", true)
	client.baseURL = server.URL

	_, err := client.Query(context.Background(), "Test")
	if err != nil {
		t.Fatalf("Query failed: %v", err)
	}
}

func TestGeminiClient_APIError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := geminiResponse{
			Error: &struct {
				Message string `json:"message"`
				Code    int    `json:"code"`
			}{
				Message: "Quota exceeded",
				Code:    429,
			},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	client := NewGeminiClient("test-key", "gemini-2.5-flash", false)
	client.baseURL = server.URL

	_, err := client.Query(context.Background(), "Test")
	if err == nil {
		t.Error("Expected error for API error response")
	}
}

func TestGeminiClient_Model(t *testing.T) {
	client := NewGeminiClient("test-key", "gemini-2.5-flash", false)
	if client.Model() != "gemini-2.5-flash" {
		t.Errorf("Expected model 'gemini-2.5-flash', got %s", client.Model())
	}
}

func TestGeminiClient_WithGeminiCaching(t *testing.T) {
	// Test default caching enabled
	client := NewGeminiClient("test-key", "gemini-2.5-flash", false)
	if !client.trackCaching {
		t.Error("Expected trackCaching to be enabled by default")
	}

	// Test disabling caching
	clientDisabled := NewGeminiClient("test-key", "gemini-2.5-flash", false, WithGeminiCaching(false))
	if clientDisabled.trackCaching {
		t.Error("Expected trackCaching to be disabled")
	}

	// Test explicit enabling
	clientEnabled := NewGeminiClient("test-key", "gemini-2.5-flash", false, WithGeminiCaching(true))
	if !clientEnabled.trackCaching {
		t.Error("Expected trackCaching to be enabled")
	}
}

func TestGeminiClient_Complete_CacheHit(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Response with cached content token count (simulating a cache hit)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{
			"candidates": [{"content": {"parts": [{"text": "Hello from cache!"}]}}],
			"usageMetadata": {
				"promptTokenCount": 100,
				"candidatesTokenCount": 10,
				"totalTokenCount": 110,
				"cachedContentTokenCount": 80
			}
		}`))
	}))
	defer server.Close()

	client := NewGeminiClient("test-key", "gemini-2.5-flash", false)
	client.baseURL = server.URL

	messages := []core.Message{
		{Role: "user", Content: "Hello"},
	}

	resp, err := client.Complete(context.Background(), messages)
	if err != nil {
		t.Fatalf("Complete failed: %v", err)
	}
	if resp.Content != "Hello from cache!" {
		t.Errorf("Expected 'Hello from cache!', got %s", resp.Content)
	}
	if resp.PromptTokens != 100 {
		t.Errorf("Expected 100 prompt tokens, got %d", resp.PromptTokens)
	}
	if resp.CompletionTokens != 10 {
		t.Errorf("Expected 10 completion tokens, got %d", resp.CompletionTokens)
	}
	if resp.CacheReadTokens != 80 {
		t.Errorf("Expected 80 cache read tokens, got %d", resp.CacheReadTokens)
	}
}

func TestGeminiClient_Complete_CacheHitDisabled(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{
			"candidates": [{"content": {"parts": [{"text": "Response"}]}}],
			"usageMetadata": {
				"promptTokenCount": 100,
				"candidatesTokenCount": 10,
				"cachedContentTokenCount": 80
			}
		}`))
	}))
	defer server.Close()

	// Create client with caching tracking disabled
	client := NewGeminiClient("test-key", "gemini-2.5-flash", false, WithGeminiCaching(false))
	client.baseURL = server.URL

	messages := []core.Message{
		{Role: "user", Content: "Hello"},
	}

	resp, err := client.Complete(context.Background(), messages)
	if err != nil {
		t.Fatalf("Complete failed: %v", err)
	}
	// When caching is disabled, CacheReadTokens should be 0
	if resp.CacheReadTokens != 0 {
		t.Errorf("Expected 0 cache read tokens when tracking disabled, got %d", resp.CacheReadTokens)
	}
}

func TestGeminiClient_Complete_VerboseCacheHit(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{
			"candidates": [{"content": {"parts": [{"text": "Response"}]}}],
			"usageMetadata": {
				"promptTokenCount": 100,
				"candidatesTokenCount": 10,
				"cachedContentTokenCount": 80
			}
		}`))
	}))
	defer server.Close()

	// Verbose mode with caching enabled - just verify no panic
	client := NewGeminiClient("test-key", "gemini-2.5-flash", true)
	client.baseURL = server.URL

	messages := []core.Message{
		{Role: "user", Content: "Hello"},
	}

	_, err := client.Complete(context.Background(), messages)
	if err != nil {
		t.Fatalf("Complete failed: %v", err)
	}
}

func TestGeminiClient_CompleteStream_Success(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if !strings.Contains(r.URL.Path, "streamGenerateContent") {
			t.Errorf("Expected streamGenerateContent in path, got %s", r.URL.Path)
		}
		if !strings.Contains(r.URL.RawQuery, "alt=sse") {
			t.Errorf("Expected alt=sse in query, got %s", r.URL.RawQuery)
		}

		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.WriteHeader(http.StatusOK)

		events := []string{
			`data: {"candidates":[{"content":{"parts":[{"text":"Hello"}],"role":"model"}}]}`,
			`data: {"candidates":[{"content":{"parts":[{"text":", "}],"role":"model"}}]}`,
			`data: {"candidates":[{"content":{"parts":[{"text":"world!"}],"role":"model"}}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":5,"totalTokenCount":15}}`,
		}

		for _, event := range events {
			_, _ = fmt.Fprintln(w, event)
			_, _ = fmt.Fprintln(w, "")
			if f, ok := w.(http.Flusher); ok {
				f.Flush()
			}
		}
	}))
	defer server.Close()

	client := NewGeminiClient("test-key", "gemini-2.5-flash", false)
	client.baseURL = server.URL

	messages := []core.Message{
		{Role: "user", Content: "Hello"},
	}

	var chunks []string
	var gotDone bool
	handler := func(chunk string, done bool) error {
		if done {
			gotDone = true
		} else if chunk != "" {
			chunks = append(chunks, chunk)
		}
		return nil
	}

	resp, err := client.CompleteStream(context.Background(), messages, handler)
	if err != nil {
		t.Fatalf("CompleteStream failed: %v", err)
	}

	if resp.Content != "Hello, world!" {
		t.Errorf("Expected 'Hello, world!', got %s", resp.Content)
	}
	if resp.PromptTokens != 10 {
		t.Errorf("Expected 10 prompt tokens, got %d", resp.PromptTokens)
	}
	if resp.CompletionTokens != 5 {
		t.Errorf("Expected 5 completion tokens, got %d", resp.CompletionTokens)
	}
	if !gotDone {
		t.Error("Handler should have received done=true")
	}
	if len(chunks) != 3 {
		t.Errorf("Expected 3 chunks, got %d: %v", len(chunks), chunks)
	}
}

func TestGeminiClient_CompleteStream_WithSystemPrompt(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req geminiRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("Failed to decode request: %v", err)
		}
		if req.SystemInstruction == nil {
			t.Error("Expected system instruction")
		}

		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)

		_, _ = fmt.Fprintln(w, `data: {"candidates":[{"content":{"parts":[{"text":"Response"}],"role":"model"}}],"usageMetadata":{"promptTokenCount":15,"candidatesTokenCount":3}}`)
		_, _ = fmt.Fprintln(w, "")
	}))
	defer server.Close()

	client := NewGeminiClient("test-key", "gemini-2.5-flash", false)
	client.baseURL = server.URL

	messages := []core.Message{
		{Role: "system", Content: "You are helpful"},
		{Role: "user", Content: "Hello"},
	}

	resp, err := client.CompleteStream(context.Background(), messages, nil)
	if err != nil {
		t.Fatalf("CompleteStream failed: %v", err)
	}
	if resp.Content != "Response" {
		t.Errorf("Expected 'Response', got %s", resp.Content)
	}
}

func TestGeminiClient_CompleteStream_RoleMapping(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req geminiRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("Failed to decode request: %v", err)
		}
		for _, content := range req.Contents {
			if content.Role == "assistant" {
				t.Error("assistant role should be mapped to model")
			}
		}

		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)

		_, _ = fmt.Fprintln(w, `data: {"candidates":[{"content":{"parts":[{"text":"OK"}],"role":"model"}}]}`)
		_, _ = fmt.Fprintln(w, "")
	}))
	defer server.Close()

	client := NewGeminiClient("test-key", "gemini-2.5-flash", false)
	client.baseURL = server.URL

	messages := []core.Message{
		{Role: "user", Content: "Hello"},
		{Role: "assistant", Content: "Hi there"},
		{Role: "user", Content: "How are you?"},
	}

	_, err := client.CompleteStream(context.Background(), messages, nil)
	if err != nil {
		t.Fatalf("CompleteStream failed: %v", err)
	}
}

func TestGeminiClient_CompleteStream_Error(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"error": {"message": "Invalid request", "code": 400}}`))
	}))
	defer server.Close()

	client := NewGeminiClient("test-key", "gemini-2.5-flash", false)
	client.baseURL = server.URL

	messages := []core.Message{
		{Role: "user", Content: "Hello"},
	}

	_, err := client.CompleteStream(context.Background(), messages, nil)
	if err == nil {
		t.Error("Expected error, got nil")
	}
}

func TestGeminiClient_CompleteStream_StreamError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)

		_, _ = fmt.Fprintln(w, `data: {"error":{"message":"Rate limit exceeded","code":429}}`)
		_, _ = fmt.Fprintln(w, "")
	}))
	defer server.Close()

	client := NewGeminiClient("test-key", "gemini-2.5-flash", false)
	client.baseURL = server.URL

	messages := []core.Message{
		{Role: "user", Content: "Hello"},
	}

	_, err := client.CompleteStream(context.Background(), messages, nil)
	if err == nil {
		t.Error("Expected stream error")
	}
	if !strings.Contains(err.Error(), "Rate limit exceeded") {
		t.Errorf("Expected rate limit error, got: %v", err)
	}
}

func TestGeminiClient_CompleteStream_HandlerError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)

		_, _ = fmt.Fprintln(w, `data: {"candidates":[{"content":{"parts":[{"text":"Hello"}],"role":"model"}}]}`)
		_, _ = fmt.Fprintln(w, "")
	}))
	defer server.Close()

	client := NewGeminiClient("test-key", "gemini-2.5-flash", false)
	client.baseURL = server.URL

	messages := []core.Message{
		{Role: "user", Content: "Hello"},
	}

	handlerErr := fmt.Errorf("handler error")
	handler := func(chunk string, done bool) error {
		return handlerErr
	}

	_, err := client.CompleteStream(context.Background(), messages, handler)
	if err == nil {
		t.Error("Expected error from handler")
	}
	if !strings.Contains(err.Error(), "handler error") {
		t.Errorf("Expected handler error, got: %v", err)
	}
}

func TestGeminiClient_CompleteStream_NilHandler(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)

		_, _ = fmt.Fprintln(w, `data: {"candidates":[{"content":{"parts":[{"text":"Hello"}],"role":"model"}}],"usageMetadata":{"promptTokenCount":5,"candidatesTokenCount":1}}`)
		_, _ = fmt.Fprintln(w, "")
	}))
	defer server.Close()

	client := NewGeminiClient("test-key", "gemini-2.5-flash", false)
	client.baseURL = server.URL

	messages := []core.Message{
		{Role: "user", Content: "Hello"},
	}

	resp, err := client.CompleteStream(context.Background(), messages, nil)
	if err != nil {
		t.Fatalf("CompleteStream failed: %v", err)
	}
	if resp.Content != "Hello" {
		t.Errorf("Expected 'Hello', got %s", resp.Content)
	}
}

func TestGeminiClient_CompleteStream_ContextCancellation(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)

		_, _ = fmt.Fprintln(w, `data: {"candidates":[{"content":{"parts":[{"text":"Hello"}],"role":"model"}}]}`)
		_, _ = fmt.Fprintln(w, "")
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
		select {}
	}))
	defer server.Close()

	client := NewGeminiClient("test-key", "gemini-2.5-flash", false)
	client.baseURL = server.URL

	messages := []core.Message{
		{Role: "user", Content: "Hello"},
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()

	_, err := client.CompleteStream(ctx, messages, nil)
	if err == nil {
		t.Error("Expected context cancellation error")
	}
}

func TestGeminiClient_CompleteStream_Verbose(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)

		_, _ = fmt.Fprintln(w, `data: {"candidates":[{"content":{"parts":[{"text":"Response"}],"role":"model"}}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":5}}`)
		_, _ = fmt.Fprintln(w, "")
	}))
	defer server.Close()

	client := NewGeminiClient("test-key", "gemini-2.5-flash", true)
	client.baseURL = server.URL

	messages := []core.Message{
		{Role: "user", Content: "Test"},
	}

	_, err := client.CompleteStream(context.Background(), messages, nil)
	if err != nil {
		t.Fatalf("CompleteStream failed: %v", err)
	}
}

func TestGeminiClient_CompleteStream_MultipleChunksWithUsage(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)

		events := []string{
			`data: {"candidates":[{"content":{"parts":[{"text":"First"}],"role":"model"}}],"usageMetadata":{"promptTokenCount":10}}`,
			`data: {"candidates":[{"content":{"parts":[{"text":" Second"}],"role":"model"}}]}`,
			`data: {"candidates":[{"content":{"parts":[{"text":" Third"}],"role":"model"}}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":8}}`,
		}

		for _, event := range events {
			_, _ = fmt.Fprintln(w, event)
			_, _ = fmt.Fprintln(w, "")
		}
	}))
	defer server.Close()

	client := NewGeminiClient("test-key", "gemini-2.5-flash", false)
	client.baseURL = server.URL

	messages := []core.Message{
		{Role: "user", Content: "Hello"},
	}

	resp, err := client.CompleteStream(context.Background(), messages, nil)
	if err != nil {
		t.Fatalf("CompleteStream failed: %v", err)
	}

	if resp.Content != "First Second Third" {
		t.Errorf("Expected 'First Second Third', got %s", resp.Content)
	}
	if resp.PromptTokens != 10 {
		t.Errorf("Expected 10 prompt tokens, got %d", resp.PromptTokens)
	}
	if resp.CompletionTokens != 8 {
		t.Errorf("Expected 8 completion tokens, got %d", resp.CompletionTokens)
	}
}

func TestGeminiClient_CompleteStream_CacheHit(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)

		events := []string{
			`data: {"candidates":[{"content":{"parts":[{"text":"Hello"}],"role":"model"}}]}`,
			`data: {"candidates":[{"content":{"parts":[{"text":" from"}],"role":"model"}}]}`,
			`data: {"candidates":[{"content":{"parts":[{"text":" cache!"}],"role":"model"}}],"usageMetadata":{"promptTokenCount":100,"candidatesTokenCount":5,"cachedContentTokenCount":80}}`,
		}

		for _, event := range events {
			_, _ = fmt.Fprintln(w, event)
			_, _ = fmt.Fprintln(w, "")
		}
	}))
	defer server.Close()

	client := NewGeminiClient("test-key", "gemini-2.5-flash", false)
	client.baseURL = server.URL

	messages := []core.Message{
		{Role: "user", Content: "Hello"},
	}

	resp, err := client.CompleteStream(context.Background(), messages, nil)
	if err != nil {
		t.Fatalf("CompleteStream failed: %v", err)
	}

	if resp.Content != "Hello from cache!" {
		t.Errorf("Expected 'Hello from cache!', got %s", resp.Content)
	}
	if resp.PromptTokens != 100 {
		t.Errorf("Expected 100 prompt tokens, got %d", resp.PromptTokens)
	}
	if resp.CompletionTokens != 5 {
		t.Errorf("Expected 5 completion tokens, got %d", resp.CompletionTokens)
	}
	if resp.CacheReadTokens != 80 {
		t.Errorf("Expected 80 cache read tokens, got %d", resp.CacheReadTokens)
	}
}

func TestGeminiClient_CompleteStream_CacheHitDisabled(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)

		events := []string{
			`data: {"candidates":[{"content":{"parts":[{"text":"Response"}],"role":"model"}}],"usageMetadata":{"promptTokenCount":100,"candidatesTokenCount":5,"cachedContentTokenCount":80}}`,
		}

		for _, event := range events {
			_, _ = fmt.Fprintln(w, event)
			_, _ = fmt.Fprintln(w, "")
		}
	}))
	defer server.Close()

	// Create client with caching tracking disabled
	client := NewGeminiClient("test-key", "gemini-2.5-flash", false, WithGeminiCaching(false))
	client.baseURL = server.URL

	messages := []core.Message{
		{Role: "user", Content: "Hello"},
	}

	resp, err := client.CompleteStream(context.Background(), messages, nil)
	if err != nil {
		t.Fatalf("CompleteStream failed: %v", err)
	}

	// When caching is disabled, CacheReadTokens should be 0
	if resp.CacheReadTokens != 0 {
		t.Errorf("Expected 0 cache read tokens when tracking disabled, got %d", resp.CacheReadTokens)
	}
}

func TestGeminiClient_CompleteStream_VerboseCacheHit(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)

		events := []string{
			`data: {"candidates":[{"content":{"parts":[{"text":"Response"}],"role":"model"}}],"usageMetadata":{"promptTokenCount":100,"candidatesTokenCount":5,"cachedContentTokenCount":80}}`,
		}

		for _, event := range events {
			_, _ = fmt.Fprintln(w, event)
			_, _ = fmt.Fprintln(w, "")
		}
	}))
	defer server.Close()

	// Verbose mode with caching enabled - just verify no panic
	client := NewGeminiClient("test-key", "gemini-2.5-flash", true)
	client.baseURL = server.URL

	messages := []core.Message{
		{Role: "user", Content: "Hello"},
	}

	_, err := client.CompleteStream(context.Background(), messages, nil)
	if err != nil {
		t.Fatalf("CompleteStream failed: %v", err)
	}
}
