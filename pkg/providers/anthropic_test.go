package providers

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
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

		var req anthropicRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("Failed to decode request: %v", err)
		}
		if req.System != "You are helpful" {
			t.Errorf("Expected system prompt, got %s", req.System)
		}

		resp := anthropicResponse{
			Content: []struct {
				Type string `json:"type"`
				Text string `json:"text"`
			}{
				{Type: "text", Text: "Hello, world!"},
			},
			Usage: struct {
				InputTokens  int `json:"input_tokens"`
				OutputTokens int `json:"output_tokens"`
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
				InputTokens  int `json:"input_tokens"`
				OutputTokens int `json:"output_tokens"`
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
	callCount := 0
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		callCount++
		resp := anthropicResponse{
			Content: []struct {
				Type string `json:"type"`
				Text string `json:"text"`
			}{
				{Type: "text", Text: "Response"},
			},
			Usage: struct {
				InputTokens  int `json:"input_tokens"`
				OutputTokens int `json:"output_tokens"`
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
	if callCount != 3 {
		t.Errorf("Expected 3 API calls, got %d", callCount)
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
				InputTokens  int `json:"input_tokens"`
				OutputTokens int `json:"output_tokens"`
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
