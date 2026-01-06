package providers

import (
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"

	"github.com/XiaoConstantine/rlm-go/pkg/core"
)

func TestOpenAIClient_Complete_Success(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.Method != "POST" {
			t.Errorf("Expected POST, got %s", r.Method)
		}
		if r.URL.Path != "/v1/chat/completions" {
			t.Errorf("Expected /v1/chat/completions, got %s", r.URL.Path)
		}
		auth := r.Header.Get("Authorization")
		if !strings.HasPrefix(auth, "Bearer ") {
			t.Errorf("Expected Bearer token, got %s", auth)
		}

		var req openaiRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			t.Errorf("Failed to decode request: %v", err)
		}
		if req.Model != "gpt-5" {
			t.Errorf("Expected model gpt-5, got %s", req.Model)
		}
		if len(req.Messages) != 2 {
			t.Errorf("Expected 2 messages, got %d", len(req.Messages))
		}

		resp := openaiResponse{
			Choices: []struct {
				Message struct {
					Content string `json:"content"`
				} `json:"message"`
			}{
				{
					Message: struct {
						Content string `json:"content"`
					}{
						Content: "Hello from OpenAI!",
					},
				},
			},
			Usage: struct {
				PromptTokens     int `json:"prompt_tokens"`
				CompletionTokens int `json:"completion_tokens"`
				TotalTokens      int `json:"total_tokens"`
			}{
				PromptTokens:     12,
				CompletionTokens: 6,
				TotalTokens:      18,
			},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	client := NewOpenAIClient("test-key", "gpt-5", false)
	client.baseURL = server.URL

	messages := []core.Message{
		{Role: "system", Content: "You are helpful"},
		{Role: "user", Content: "Hello"},
	}

	resp, err := client.Complete(context.Background(), messages)
	if err != nil {
		t.Fatalf("Complete failed: %v", err)
	}
	if resp.Content != "Hello from OpenAI!" {
		t.Errorf("Expected 'Hello from OpenAI!', got %s", resp.Content)
	}
	if resp.PromptTokens != 12 {
		t.Errorf("Expected 12 prompt tokens, got %d", resp.PromptTokens)
	}
	if resp.CompletionTokens != 6 {
		t.Errorf("Expected 6 completion tokens, got %d", resp.CompletionTokens)
	}
}

func TestOpenAIClient_Complete_Error(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusUnauthorized)
		_, _ = w.Write([]byte(`{"error": {"message": "Invalid API key", "type": "invalid_request_error"}}`))
	}))
	defer server.Close()

	client := NewOpenAIClient("invalid-key", "gpt-5", false)
	client.baseURL = server.URL

	_, err := client.Complete(context.Background(), []core.Message{{Role: "user", Content: "Hi"}})
	if err == nil {
		t.Error("Expected error, got nil")
	}
}

func TestOpenAIClient_Query_Success(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := openaiResponse{
			Choices: []struct {
				Message struct {
					Content string `json:"content"`
				} `json:"message"`
			}{
				{
					Message: struct {
						Content string `json:"content"`
					}{
						Content: "Query response",
					},
				},
			},
			Usage: struct {
				PromptTokens     int `json:"prompt_tokens"`
				CompletionTokens int `json:"completion_tokens"`
				TotalTokens      int `json:"total_tokens"`
			}{
				PromptTokens:     5,
				CompletionTokens: 3,
				TotalTokens:      8,
			},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	client := NewOpenAIClient("test-key", "gpt-5", false)
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

func TestOpenAIClient_QueryBatched_Success(t *testing.T) {
	var callCount int32
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&callCount, 1)
		resp := openaiResponse{
			Choices: []struct {
				Message struct {
					Content string `json:"content"`
				} `json:"message"`
			}{
				{
					Message: struct {
						Content string `json:"content"`
					}{
						Content: "Response",
					},
				},
			},
			Usage: struct {
				PromptTokens     int `json:"prompt_tokens"`
				CompletionTokens int `json:"completion_tokens"`
				TotalTokens      int `json:"total_tokens"`
			}{
				PromptTokens:     1,
				CompletionTokens: 1,
				TotalTokens:      2,
			},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	client := NewOpenAIClient("test-key", "gpt-5", false)
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

func TestOpenAIClient_QueryBatched_Empty(t *testing.T) {
	client := NewOpenAIClient("test-key", "gpt-5", false)

	results, err := client.QueryBatched(context.Background(), []string{})
	if err != nil {
		t.Errorf("QueryBatched with empty input returned error: %v", err)
	}
	if len(results) != 0 {
		t.Errorf("Expected 0 results, got %d", len(results))
	}
}

func TestOpenAIClient_Verbose(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := openaiResponse{
			Choices: []struct {
				Message struct {
					Content string `json:"content"`
				} `json:"message"`
			}{
				{
					Message: struct {
						Content string `json:"content"`
					}{
						Content: "Response",
					},
				},
			},
			Usage: struct {
				PromptTokens     int `json:"prompt_tokens"`
				CompletionTokens int `json:"completion_tokens"`
				TotalTokens      int `json:"total_tokens"`
			}{
				PromptTokens:     10,
				CompletionTokens: 5,
				TotalTokens:      15,
			},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	client := NewOpenAIClient("test-key", "gpt-5", true)
	client.baseURL = server.URL

	_, err := client.Query(context.Background(), "Test")
	if err != nil {
		t.Fatalf("Query failed: %v", err)
	}
}

func TestOpenAIClient_APIError(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := openaiResponse{
			Error: &struct {
				Message string `json:"message"`
				Type    string `json:"type"`
			}{
				Message: "Rate limit exceeded",
				Type:    "rate_limit_error",
			},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	client := NewOpenAIClient("test-key", "gpt-5", false)
	client.baseURL = server.URL

	_, err := client.Query(context.Background(), "Test")
	if err == nil {
		t.Error("Expected error for API error response")
	}
}

func TestOpenAIClient_NoChoices(t *testing.T) {
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		resp := openaiResponse{
			Choices: []struct {
				Message struct {
					Content string `json:"content"`
				} `json:"message"`
			}{},
			Usage: struct {
				PromptTokens     int `json:"prompt_tokens"`
				CompletionTokens int `json:"completion_tokens"`
				TotalTokens      int `json:"total_tokens"`
			}{},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	client := NewOpenAIClient("test-key", "gpt-5", false)
	client.baseURL = server.URL

	_, err := client.Query(context.Background(), "Test")
	if err == nil {
		t.Error("Expected error for empty choices")
	}
	if !strings.Contains(err.Error(), "no choices") {
		t.Errorf("Expected 'no choices' error, got %v", err)
	}
}
