package providers

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/XiaoConstantine/rlm-go/pkg/core"
)

func TestOpenAIClient_Complete(t *testing.T) {
	client := NewOpenAIClient("test-key", "gpt-5", false)

	messages := []core.Message{
		{Role: "system", Content: "You are helpful"},
		{Role: "user", Content: "Hello"},
	}

	// Verify client is properly configured
	if client.apiKey != "test-key" {
		t.Errorf("apiKey not set correctly")
	}
	if client.model != "gpt-5" {
		t.Errorf("model not set correctly")
	}
	if len(messages) != 2 {
		t.Errorf("messages not created correctly")
	}
}

func TestOpenAIClient_QueryBatched(t *testing.T) {
	client := NewOpenAIClient("test-key", "gpt-5", false)

	// Test that QueryBatched handles empty input
	ctx := context.Background()
	results, err := client.QueryBatched(ctx, []string{})
	if err != nil {
		t.Errorf("QueryBatched with empty input returned error: %v", err)
	}
	if len(results) != 0 {
		t.Errorf("QueryBatched with empty input returned %d results, want 0", len(results))
	}
}

func TestOpenAIRequest_Marshal(t *testing.T) {
	req := openaiRequest{
		Model: "gpt-5",
		Messages: []openaiMessage{
			{Role: "system", Content: "Be helpful"},
			{Role: "user", Content: "Hello"},
		},
	}

	data, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("Failed to marshal request: %v", err)
	}

	var unmarshaled openaiRequest
	if err := json.Unmarshal(data, &unmarshaled); err != nil {
		t.Fatalf("Failed to unmarshal request: %v", err)
	}

	if unmarshaled.Model != "gpt-5" {
		t.Errorf("Model mismatch: got %s, want gpt-5", unmarshaled.Model)
	}
	if len(unmarshaled.Messages) != 2 {
		t.Errorf("Messages length mismatch: got %d, want 2", len(unmarshaled.Messages))
	}
	if unmarshaled.Messages[0].Role != "system" {
		t.Errorf("First message role mismatch: got %s, want system", unmarshaled.Messages[0].Role)
	}
}

func TestOpenAIResponse_Unmarshal(t *testing.T) {
	jsonResp := `{
		"choices": [
			{
				"message": {
					"content": "Hello, how can I help you?"
				}
			}
		],
		"usage": {
			"prompt_tokens": 10,
			"completion_tokens": 8,
			"total_tokens": 18
		}
	}`

	var resp openaiResponse
	if err := json.Unmarshal([]byte(jsonResp), &resp); err != nil {
		t.Fatalf("Failed to unmarshal response: %v", err)
	}

	if len(resp.Choices) != 1 {
		t.Errorf("Choices length mismatch: got %d, want 1", len(resp.Choices))
	}
	if resp.Choices[0].Message.Content != "Hello, how can I help you?" {
		t.Errorf("Content mismatch")
	}
	if resp.Usage.PromptTokens != 10 {
		t.Errorf("PromptTokens mismatch: got %d, want 10", resp.Usage.PromptTokens)
	}
	if resp.Usage.CompletionTokens != 8 {
		t.Errorf("CompletionTokens mismatch: got %d, want 8", resp.Usage.CompletionTokens)
	}
}

func TestOpenAIResponse_Error(t *testing.T) {
	jsonResp := `{
		"error": {
			"message": "Invalid API key",
			"type": "invalid_request_error"
		}
	}`

	var resp openaiResponse
	if err := json.Unmarshal([]byte(jsonResp), &resp); err != nil {
		t.Fatalf("Failed to unmarshal error response: %v", err)
	}

	if resp.Error == nil {
		t.Fatal("Expected error to be non-nil")
	}
	if resp.Error.Message != "Invalid API key" {
		t.Errorf("Error message mismatch: got %s", resp.Error.Message)
	}
	if resp.Error.Type != "invalid_request_error" {
		t.Errorf("Error type mismatch: got %s", resp.Error.Type)
	}
}
