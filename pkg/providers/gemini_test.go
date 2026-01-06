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
				PromptTokenCount     int `json:"promptTokenCount"`
				CandidatesTokenCount int `json:"candidatesTokenCount"`
				TotalTokenCount      int `json:"totalTokenCount"`
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

	client := NewGeminiClient("test-key", "gemini-3-flash-preview", false)
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
				PromptTokenCount     int `json:"promptTokenCount"`
				CandidatesTokenCount int `json:"candidatesTokenCount"`
				TotalTokenCount      int `json:"totalTokenCount"`
			}{},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	client := NewGeminiClient("test-key", "gemini-3-flash-preview", false)
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

	client := NewGeminiClient("test-key", "gemini-3-flash-preview", false)
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
				PromptTokenCount     int `json:"promptTokenCount"`
				CandidatesTokenCount int `json:"candidatesTokenCount"`
				TotalTokenCount      int `json:"totalTokenCount"`
			}{
				PromptTokenCount:     5,
				CandidatesTokenCount: 3,
			},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	client := NewGeminiClient("test-key", "gemini-3-flash-preview", false)
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
				PromptTokenCount     int `json:"promptTokenCount"`
				CandidatesTokenCount int `json:"candidatesTokenCount"`
				TotalTokenCount      int `json:"totalTokenCount"`
			}{},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	client := NewGeminiClient("test-key", "gemini-3-flash-preview", false)
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
	client := NewGeminiClient("test-key", "gemini-3-flash-preview", false)

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
				PromptTokenCount     int `json:"promptTokenCount"`
				CandidatesTokenCount int `json:"candidatesTokenCount"`
				TotalTokenCount      int `json:"totalTokenCount"`
			}{
				PromptTokenCount:     10,
				CandidatesTokenCount: 5,
			},
		}
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(resp)
	}))
	defer server.Close()

	client := NewGeminiClient("test-key", "gemini-3-flash-preview", true)
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

	client := NewGeminiClient("test-key", "gemini-3-flash-preview", false)
	client.baseURL = server.URL

	_, err := client.Query(context.Background(), "Test")
	if err == nil {
		t.Error("Expected error for API error response")
	}
}
