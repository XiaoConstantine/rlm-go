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

func TestAnthropicClient_Complete_Success(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" {
			t.Errorf("Expected POST, got %s", r.Method)
		}
		if r.URL.Path != "/v1/messages" {
			t.Errorf("Expected /v1/messages, got %s", r.URL.Path)
		}
		if r.Header.Get("x-api-key") != "test-key" {
			t.Errorf("Expected x-api-key test-key, got %s", r.Header.Get("x-api-key"))
		}
		if r.Header.Get("anthropic-version") != "2023-06-01" {
			t.Errorf("Expected anthropic-version 2023-06-01")
		}
		if r.Header.Get("anthropic-beta") != "prompt-caching-2024-07-31" {
			t.Errorf("Expected anthropic-beta header for prefix caching, got %s", r.Header.Get("anthropic-beta"))
		}

		var req map[string]any
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("Failed to decode request: %v", err)
		}
		system, ok := req["system"].([]any)
		if !ok || len(system) != 1 {
			t.Errorf("Expected system to be array with 1 element, got %v", req["system"])
		} else {
			block := system[0].(map[string]any)
			if block["text"] != "You are helpful" {
				t.Errorf("Expected system text 'You are helpful', got %v", block["text"])
			}
			cacheControl := block["cache_control"].(map[string]any)
			if cacheControl["type"] != "ephemeral" {
				t.Errorf("Expected cache_control type 'ephemeral', got %v", cacheControl["type"])
			}
		}

		resp := anthropicResponse{
			Content: []struct {
				Type string `json:"type"`
				Text string `json:"text"`
			}{
				{Type: "text", Text: "Hello, world!"},
			},
			Usage: struct {
				InputTokens              int `json:"input_tokens"`
				OutputTokens             int `json:"output_tokens"`
				CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
				CacheReadInputTokens     int `json:"cache_read_input_tokens"`
			}{
				InputTokens:  10,
				OutputTokens: 5,
			},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	client := NewAnthropicClient("test-key", "claude-test", false)
	client.baseURL = server.URL

	messages := []core.Message{
		{Role: "system", Content: "You are helpful"},
		{Role: "user", Content: "Hello"},
	}

	resp, err := client.Complete(context.Background(), messages)
	if err != nil {
		t.Fatalf("Complete failed: %v", err)
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
}

func TestAnthropicClient_Complete_Error(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"error": {"message": "Invalid request"}}`))
	}))
	defer server.Close()

	client := NewAnthropicClient("test-key", "claude-test", false)
	client.baseURL = server.URL

	_, err := client.Complete(context.Background(), []core.Message{{Role: "user", Content: "Hi"}})
	if err == nil {
		t.Error("Expected error, got nil")
	}
}

func TestAnthropicClient_Query_Success(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := anthropicResponse{
			Content: []struct {
				Type string `json:"type"`
				Text string `json:"text"`
			}{
				{Type: "text", Text: "Query response"},
			},
			Usage: struct {
				InputTokens              int `json:"input_tokens"`
				OutputTokens             int `json:"output_tokens"`
				CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
				CacheReadInputTokens     int `json:"cache_read_input_tokens"`
			}{
				InputTokens:  5,
				OutputTokens: 3,
			},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	client := NewAnthropicClient("test-key", "claude-test", false)
	client.baseURL = server.URL

	resp, err := client.Query(context.Background(), "Test prompt")
	if err != nil {
		t.Fatalf("Query failed: %v", err)
	}
	if resp.Response != "Query response" {
		t.Errorf("Expected 'Query response', got %s", resp.Response)
	}
	if resp.PromptTokens != 5 {
		t.Errorf("Expected 5 prompt tokens, got %d", resp.PromptTokens)
	}
}

func TestAnthropicClient_QueryBatched_Success(t *testing.T) {
	var callCount int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&callCount, 1)
		resp := anthropicResponse{
			Content: []struct {
				Type string `json:"type"`
				Text string `json:"text"`
			}{
				{Type: "text", Text: "Response"},
			},
			Usage: struct {
				InputTokens              int `json:"input_tokens"`
				OutputTokens             int `json:"output_tokens"`
				CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
				CacheReadInputTokens     int `json:"cache_read_input_tokens"`
			}{
				InputTokens:  1,
				OutputTokens: 1,
			},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	client := NewAnthropicClient("test-key", "claude-test", false)
	client.baseURL = server.URL

	prompts := []string{"Prompt 1", "Prompt 2", "Prompt 3"}
	results, err := client.QueryBatched(context.Background(), prompts)
	if err != nil {
		t.Fatalf("QueryBatched failed: %v", err)
	}
	if len(results) != 3 {
		t.Errorf("Expected 3 results, got %d", len(results))
	}
	if atomic.LoadInt32(&callCount) != 3 {
		t.Errorf("Expected 3 API calls, got %d", atomic.LoadInt32(&callCount))
	}
}

func TestAnthropicClient_QueryBatched_Empty(t *testing.T) {
	client := NewAnthropicClient("test-key", "claude-test", false)

	results, err := client.QueryBatched(context.Background(), []string{})
	if err != nil {
		t.Errorf("QueryBatched with empty input returned error: %v", err)
	}
	if len(results) != 0 {
		t.Errorf("Expected 0 results, got %d", len(results))
	}
}

func TestAnthropicClient_Verbose(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := anthropicResponse{
			Content: []struct {
				Type string `json:"type"`
				Text string `json:"text"`
			}{
				{Type: "text", Text: "Response"},
			},
			Usage: struct {
				InputTokens              int `json:"input_tokens"`
				OutputTokens             int `json:"output_tokens"`
				CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
				CacheReadInputTokens     int `json:"cache_read_input_tokens"`
			}{
				InputTokens:  10,
				OutputTokens: 5,
			},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	client := NewAnthropicClient("test-key", "claude-test", true)
	client.baseURL = server.URL

	_, err := client.Query(context.Background(), "Test")
	if err != nil {
		t.Fatalf("Query failed: %v", err)
	}
}

func TestAnthropicClient_PrefixCachingDisabled(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Header.Get("anthropic-beta") != "" {
			t.Errorf("Expected no anthropic-beta header, got %s", r.Header.Get("anthropic-beta"))
		}

		var req map[string]any
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("Failed to decode request: %v", err)
		}
		if _, ok := req["system"].(string); !ok {
			t.Errorf("Expected system to be a string when prefix caching is disabled, got %T", req["system"])
		}

		resp := anthropicResponse{
			Content: []struct {
				Type string `json:"type"`
				Text string `json:"text"`
			}{
				{Type: "text", Text: "Response"},
			},
			Usage: struct {
				InputTokens              int `json:"input_tokens"`
				OutputTokens             int `json:"output_tokens"`
				CacheCreationInputTokens int `json:"cache_creation_input_tokens"`
				CacheReadInputTokens     int `json:"cache_read_input_tokens"`
			}{
				InputTokens:  10,
				OutputTokens: 5,
			},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	client := NewAnthropicClient("test-key", "claude-test", false, WithPrefixCaching(false))
	client.baseURL = server.URL

	messages := []core.Message{
		{Role: "system", Content: "You are helpful"},
		{Role: "user", Content: "Hello"},
	}

	_, err := client.Complete(context.Background(), messages)
	if err != nil {
		t.Fatalf("Complete failed: %v", err)
	}
}

func TestAnthropicClient_PrefixCachingCacheHit(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte(`{
			"content": [{"type": "text", "text": "Hello"}],
			"usage": {
				"input_tokens": 100,
				"output_tokens": 10,
				"cache_creation_input_tokens": 0,
				"cache_read_input_tokens": 90
			}
		}`))
	}))
	defer server.Close()

	client := NewAnthropicClient("test-key", "claude-test", true)
	client.baseURL = server.URL

	messages := []core.Message{
		{Role: "system", Content: "You are helpful"},
		{Role: "user", Content: "Hello"},
	}

	_, err := client.Complete(context.Background(), messages)
	if err != nil {
		t.Fatalf("Complete failed: %v", err)
	}
}

func TestAnthropicClient_APIError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := anthropicResponse{
			Error: &struct {
				Message string `json:"message"`
			}{
				Message: "Rate limit exceeded",
			},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	client := NewAnthropicClient("test-key", "claude-test", false)
	client.baseURL = server.URL

	_, err := client.Query(context.Background(), "Test")
	if err == nil {
		t.Error("Expected error for API error response")
	}
}

func TestAnthropicClient_CompleteStream_Success(t *testing.T) {
	// Simulate SSE streaming response
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		// Verify request has stream: true
		var req anthropicRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("Failed to decode request: %v", err)
		}
		if !req.Stream {
			t.Error("Expected stream: true in request")
		}

		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.WriteHeader(http.StatusOK)

		// Write SSE events
		events := []string{
			`data: {"type":"message_start","message":{"usage":{"input_tokens":10,"output_tokens":0}}}`,
			`data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}`,
			`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}`,
			`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":", "}}`,
			`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"world!"}}`,
			`data: {"type":"content_block_stop","index":0}`,
			`data: {"type":"message_delta","usage":{"output_tokens":5}}`,
			`data: {"type":"message_stop"}`,
		}

		for _, event := range events {
			_, _ = fmt.Fprintln(w, event)
			_, _ = fmt.Fprintln(w, "") // Empty line between events
			if f, ok := w.(http.Flusher); ok {
				f.Flush()
			}
		}
	}))
	defer server.Close()

	client := NewAnthropicClient("test-key", "claude-test", false)
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

func TestAnthropicClient_CompleteStream_Error(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
		_, _ = w.Write([]byte(`{"error": {"message": "Invalid request"}}`))
	}))
	defer server.Close()

	client := NewAnthropicClient("test-key", "claude-test", false)
	client.baseURL = server.URL

	messages := []core.Message{
		{Role: "user", Content: "Hello"},
	}

	_, err := client.CompleteStream(context.Background(), messages, nil)
	if err == nil {
		t.Error("Expected error, got nil")
	}
}

func TestAnthropicClient_CompleteStream_HandlerError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)

		events := []string{
			`data: {"type":"message_start","message":{"usage":{"input_tokens":10,"output_tokens":0}}}`,
			`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}`,
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

	client := NewAnthropicClient("test-key", "claude-test", false)
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

func TestAnthropicClient_CompleteStream_NilHandler(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)

		events := []string{
			`data: {"type":"message_start","message":{"usage":{"input_tokens":10,"output_tokens":0}}}`,
			`data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"Hello"}}`,
			`data: {"type":"message_delta","usage":{"output_tokens":1}}`,
			`data: {"type":"message_stop"}`,
		}

		for _, event := range events {
			_, _ = fmt.Fprintln(w, event)
			_, _ = fmt.Fprintln(w, "")
		}
	}))
	defer server.Close()

	client := NewAnthropicClient("test-key", "claude-test", false)
	client.baseURL = server.URL

	messages := []core.Message{
		{Role: "user", Content: "Hello"},
	}

	// nil handler should work fine
	resp, err := client.CompleteStream(context.Background(), messages, nil)
	if err != nil {
		t.Fatalf("CompleteStream failed: %v", err)
	}
	if resp.Content != "Hello" {
		t.Errorf("Expected 'Hello', got %s", resp.Content)
	}
}

func TestAnthropicClient_CompleteStream_ContextCancellation(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)

		// Write initial event
		_, _ = fmt.Fprintln(w, `data: {"type":"message_start","message":{"usage":{"input_tokens":10,"output_tokens":0}}}`)
		_, _ = fmt.Fprintln(w, "")
		if f, ok := w.(http.Flusher); ok {
			f.Flush()
		}
		// Keep connection open (simulating slow stream)
		select {}
	}))
	defer server.Close()

	client := NewAnthropicClient("test-key", "claude-test", false)
	client.baseURL = server.URL

	messages := []core.Message{
		{Role: "user", Content: "Hello"},
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately

	_, err := client.CompleteStream(ctx, messages, nil)
	if err == nil {
		t.Error("Expected context cancellation error")
	}
}
