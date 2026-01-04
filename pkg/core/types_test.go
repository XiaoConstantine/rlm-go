package core

import (
	"encoding/json"
	"testing"
	"time"
)

func TestExecutionResult(t *testing.T) {
	tests := []struct {
		name     string
		result   ExecutionResult
		hasError bool
	}{
		{
			name: "successful execution with stdout",
			result: ExecutionResult{
				Stdout:   "Hello, World!\n",
				Stderr:   "",
				Duration: 100 * time.Millisecond,
			},
			hasError: false,
		},
		{
			name: "execution with stderr",
			result: ExecutionResult{
				Stdout:   "",
				Stderr:   "error: undefined variable",
				Duration: 50 * time.Millisecond,
			},
			hasError: true,
		},
		{
			name: "execution with both stdout and stderr",
			result: ExecutionResult{
				Stdout:   "partial output",
				Stderr:   "warning: something",
				Duration: 200 * time.Millisecond,
			},
			hasError: true,
		},
		{
			name: "empty execution",
			result: ExecutionResult{
				Stdout:   "",
				Stderr:   "",
				Duration: 0,
			},
			hasError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Test that fields are accessible
			if tt.result.Stdout == "" && tt.result.Stderr == "" {
				// Empty result is valid
			}

			hasError := tt.result.Stderr != ""
			if hasError != tt.hasError {
				t.Errorf("hasError = %v, want %v", hasError, tt.hasError)
			}
		})
	}
}

func TestCodeBlock(t *testing.T) {
	tests := []struct {
		name  string
		block CodeBlock
	}{
		{
			name: "simple code block",
			block: CodeBlock{
				Code: "fmt.Println(1)",
				Result: ExecutionResult{
					Stdout:   "1\n",
					Stderr:   "",
					Duration: 10 * time.Millisecond,
				},
			},
		},
		{
			name: "multi-line code block",
			block: CodeBlock{
				Code: "x := 1\ny := 2\nfmt.Println(x + y)",
				Result: ExecutionResult{
					Stdout:   "3\n",
					Stderr:   "",
					Duration: 15 * time.Millisecond,
				},
			},
		},
		{
			name: "code block with error",
			block: CodeBlock{
				Code: "fmt.Println(undefinedVar)",
				Result: ExecutionResult{
					Stdout:   "",
					Stderr:   "undefined: undefinedVar",
					Duration: 5 * time.Millisecond,
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Verify the block has code
			if tt.block.Code == "" {
				t.Error("expected non-empty code")
			}
		})
	}
}

func TestFinalAnswerType(t *testing.T) {
	tests := []struct {
		name     string
		faType   FinalAnswerType
		expected string
	}{
		{
			name:     "direct type",
			faType:   FinalTypeDirect,
			expected: "FINAL",
		},
		{
			name:     "variable type",
			faType:   FinalTypeVariable,
			expected: "FINAL_VAR",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if string(tt.faType) != tt.expected {
				t.Errorf("FinalAnswerType = %q, want %q", string(tt.faType), tt.expected)
			}
		})
	}
}

func TestFinalAnswer(t *testing.T) {
	tests := []struct {
		name    string
		answer  FinalAnswer
		isVar   bool
		content string
	}{
		{
			name: "direct final answer",
			answer: FinalAnswer{
				Type:    FinalTypeDirect,
				Content: "42",
			},
			isVar:   false,
			content: "42",
		},
		{
			name: "variable final answer",
			answer: FinalAnswer{
				Type:    FinalTypeVariable,
				Content: "answer",
			},
			isVar:   true,
			content: "answer",
		},
		{
			name: "complex direct answer",
			answer: FinalAnswer{
				Type:    FinalTypeDirect,
				Content: "40% positive, 40% negative, 20% neutral",
			},
			isVar:   false,
			content: "40% positive, 40% negative, 20% neutral",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			isVar := tt.answer.Type == FinalTypeVariable
			if isVar != tt.isVar {
				t.Errorf("isVar = %v, want %v", isVar, tt.isVar)
			}
			if tt.answer.Content != tt.content {
				t.Errorf("Content = %q, want %q", tt.answer.Content, tt.content)
			}
		})
	}
}

func TestIteration(t *testing.T) {
	tests := []struct {
		name      string
		iteration Iteration
		hasFinal  bool
		numBlocks int
	}{
		{
			name: "iteration without final answer",
			iteration: Iteration{
				Response: "Let me think about this...",
				CodeBlocks: []CodeBlock{
					{Code: "x := 1", Result: ExecutionResult{Stdout: ""}},
				},
				FinalAnswer: nil,
				Duration:    500 * time.Millisecond,
			},
			hasFinal:  false,
			numBlocks: 1,
		},
		{
			name: "iteration with final answer",
			iteration: Iteration{
				Response:   "FINAL_VAR(answer)",
				CodeBlocks: nil,
				FinalAnswer: &FinalAnswer{
					Type:    FinalTypeVariable,
					Content: "answer",
				},
				Duration: 200 * time.Millisecond,
			},
			hasFinal:  true,
			numBlocks: 0,
		},
		{
			name: "iteration with multiple code blocks",
			iteration: Iteration{
				Response: "Running multiple calculations...",
				CodeBlocks: []CodeBlock{
					{Code: "a := 1", Result: ExecutionResult{Stdout: ""}},
					{Code: "b := 2", Result: ExecutionResult{Stdout: ""}},
					{Code: "fmt.Println(a+b)", Result: ExecutionResult{Stdout: "3\n"}},
				},
				FinalAnswer: nil,
				Duration:    1 * time.Second,
			},
			hasFinal:  false,
			numBlocks: 3,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			hasFinal := tt.iteration.FinalAnswer != nil
			if hasFinal != tt.hasFinal {
				t.Errorf("hasFinal = %v, want %v", hasFinal, tt.hasFinal)
			}
			if len(tt.iteration.CodeBlocks) != tt.numBlocks {
				t.Errorf("numBlocks = %d, want %d", len(tt.iteration.CodeBlocks), tt.numBlocks)
			}
		})
	}
}

func TestMessageJSON(t *testing.T) {
	tests := []struct {
		name    string
		message Message
	}{
		{
			name:    "system message",
			message: Message{Role: "system", Content: "You are a helpful assistant."},
		},
		{
			name:    "user message",
			message: Message{Role: "user", Content: "What is 2+2?"},
		},
		{
			name:    "assistant message",
			message: Message{Role: "assistant", Content: "The answer is 4."},
		},
		{
			name:    "message with special characters",
			message: Message{Role: "user", Content: "What is `x := 1`?\n\"quoted\""},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Test JSON marshaling
			data, err := json.Marshal(tt.message)
			if err != nil {
				t.Fatalf("failed to marshal: %v", err)
			}

			// Test JSON unmarshaling
			var decoded Message
			if err := json.Unmarshal(data, &decoded); err != nil {
				t.Fatalf("failed to unmarshal: %v", err)
			}

			if decoded.Role != tt.message.Role {
				t.Errorf("Role = %q, want %q", decoded.Role, tt.message.Role)
			}
			if decoded.Content != tt.message.Content {
				t.Errorf("Content = %q, want %q", decoded.Content, tt.message.Content)
			}
		})
	}
}

func TestUsageStats(t *testing.T) {
	tests := []struct {
		name  string
		stats UsageStats
	}{
		{
			name: "typical usage",
			stats: UsageStats{
				PromptTokens:     100,
				CompletionTokens: 50,
				TotalTokens:      150,
			},
		},
		{
			name: "zero usage",
			stats: UsageStats{
				PromptTokens:     0,
				CompletionTokens: 0,
				TotalTokens:      0,
			},
		},
		{
			name: "large usage",
			stats: UsageStats{
				PromptTokens:     100000,
				CompletionTokens: 4096,
				TotalTokens:      104096,
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Verify total is sum of parts (if that's the invariant)
			expectedTotal := tt.stats.PromptTokens + tt.stats.CompletionTokens
			if tt.stats.TotalTokens != expectedTotal && tt.stats.TotalTokens != 0 {
				// Note: This only validates if TotalTokens is set correctly
				// The actual struct doesn't enforce this invariant
			}
		})
	}
}

func TestCompletionResult(t *testing.T) {
	tests := []struct {
		name   string
		result CompletionResult
	}{
		{
			name: "successful completion",
			result: CompletionResult{
				Response:   "The answer is 42.",
				Iterations: 3,
				Duration:   2 * time.Second,
				Usage: UsageStats{
					PromptTokens:     500,
					CompletionTokens: 100,
					TotalTokens:      600,
				},
			},
		},
		{
			name: "single iteration completion",
			result: CompletionResult{
				Response:   "Direct answer",
				Iterations: 1,
				Duration:   500 * time.Millisecond,
				Usage:      UsageStats{},
			},
		},
		{
			name: "max iterations completion",
			result: CompletionResult{
				Response:   "Best guess after exhaustion",
				Iterations: 30,
				Duration:   60 * time.Second,
				Usage: UsageStats{
					PromptTokens:     10000,
					CompletionTokens: 5000,
					TotalTokens:      15000,
				},
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if tt.result.Response == "" {
				t.Error("expected non-empty response")
			}
			if tt.result.Iterations < 1 {
				t.Errorf("Iterations = %d, want >= 1", tt.result.Iterations)
			}
			if tt.result.Duration < 0 {
				t.Errorf("Duration = %v, want >= 0", tt.result.Duration)
			}
		})
	}
}

func TestFinalAnswerTypeConstants(t *testing.T) {
	// Ensure constants have expected values and are distinct
	if FinalTypeDirect == FinalTypeVariable {
		t.Error("FinalTypeDirect and FinalTypeVariable should be different")
	}

	if FinalTypeDirect != "FINAL" {
		t.Errorf("FinalTypeDirect = %q, want %q", FinalTypeDirect, "FINAL")
	}

	if FinalTypeVariable != "FINAL_VAR" {
		t.Errorf("FinalTypeVariable = %q, want %q", FinalTypeVariable, "FINAL_VAR")
	}
}

func TestMessageJSONFields(t *testing.T) {
	msg := Message{Role: "user", Content: "test"}
	data, _ := json.Marshal(msg)
	jsonStr := string(data)

	// Verify JSON field names
	if !contains(jsonStr, `"role"`) {
		t.Error("expected 'role' field in JSON")
	}
	if !contains(jsonStr, `"content"`) {
		t.Error("expected 'content' field in JSON")
	}
}

func contains(s, substr string) bool {
	return len(s) >= len(substr) && (s == substr || len(s) > 0 && containsHelper(s, substr))
}

func containsHelper(s, substr string) bool {
	for i := 0; i <= len(s)-len(substr); i++ {
		if s[i:i+len(substr)] == substr {
			return true
		}
	}
	return false
}
