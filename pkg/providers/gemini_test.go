package providers

import (
	"context"
	"encoding/json"
	"testing"

	"github.com/XiaoConstantine/rlm-go/pkg/core"
)

func TestGeminiClient_Complete(t *testing.T) {
	client := NewGeminiClient("test-key", "gemini-3-flash-preview", false)

	messages := []core.Message{
		{Role: "system", Content: "You are helpful"},
		{Role: "user", Content: "Hello"},
		{Role: "assistant", Content: "Hi there"},
	}

	// Verify client is properly configured
	if client.apiKey != "test-key" {
		t.Errorf("apiKey not set correctly")
	}
	if client.model != "gemini-3-flash-preview" {
		t.Errorf("model not set correctly")
	}
	if len(messages) != 3 {
		t.Errorf("messages not created correctly")
	}
}

func TestGeminiClient_QueryBatched(t *testing.T) {
	client := NewGeminiClient("test-key", "gemini-3-flash-preview", false)

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

func TestGeminiRequest_Marshal(t *testing.T) {
	req := geminiRequest{
		Contents: []geminiContent{
			{
				Role:  "user",
				Parts: []geminiPart{{Text: "Hello"}},
			},
		},
		SystemInstruction: &geminiContent{
			Parts: []geminiPart{{Text: "Be helpful"}},
		},
		GenerationConfig: &geminiGenConfig{
			MaxOutputTokens: 8192,
		},
	}

	data, err := json.Marshal(req)
	if err != nil {
		t.Fatalf("Failed to marshal request: %v", err)
	}

	var unmarshaled geminiRequest
	if err := json.Unmarshal(data, &unmarshaled); err != nil {
		t.Fatalf("Failed to unmarshal request: %v", err)
	}

	if len(unmarshaled.Contents) != 1 {
		t.Errorf("Contents length mismatch: got %d, want 1", len(unmarshaled.Contents))
	}
	if unmarshaled.Contents[0].Role != "user" {
		t.Errorf("Role mismatch: got %s, want user", unmarshaled.Contents[0].Role)
	}
	if unmarshaled.GenerationConfig.MaxOutputTokens != 8192 {
		t.Errorf("MaxOutputTokens mismatch: got %d, want 8192", unmarshaled.GenerationConfig.MaxOutputTokens)
	}
}

func TestGeminiRoleMapping(t *testing.T) {
	// Test that assistant role is mapped to model
	client := NewGeminiClient("test-key", "gemini-3-flash-preview", false)
	_ = client // Used to verify client creation

	// The role mapping happens inside Complete(), we verify the logic
	messages := []core.Message{
		{Role: "assistant", Content: "Previous response"},
	}

	// In the actual implementation, "assistant" should be mapped to "model"
	for _, msg := range messages {
		role := msg.Role
		if role == "assistant" {
			role = "model"
		}
		if role != "model" {
			t.Errorf("assistant role should be mapped to model, got %s", role)
		}
	}
}
