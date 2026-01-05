package rlm

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/XiaoConstantine/rlm-go/pkg/core"
	"github.com/XiaoConstantine/rlm-go/pkg/repl"
)

// mockLLMClient implements LLMClient for testing the root LLM
type mockLLMClient struct {
	completeFunc func(ctx context.Context, messages []core.Message) (core.LLMResponse, error)
	calls        [][]core.Message
}

func (m *mockLLMClient) Complete(ctx context.Context, messages []core.Message) (core.LLMResponse, error) {
	m.calls = append(m.calls, messages)
	return m.completeFunc(ctx, messages)
}

// mockREPLClient implements repl.LLMClient for sub-LLM calls
type mockREPLClient struct {
	queryFunc func(ctx context.Context, prompt string) (repl.QueryResponse, error)
	batchFunc func(ctx context.Context, prompts []string) ([]repl.QueryResponse, error)
}

func (m *mockREPLClient) Query(ctx context.Context, prompt string) (repl.QueryResponse, error) {
	if m.queryFunc != nil {
		return m.queryFunc(ctx, prompt)
	}
	return repl.QueryResponse{Response: "mock response"}, nil
}

func (m *mockREPLClient) QueryBatched(ctx context.Context, prompts []string) ([]repl.QueryResponse, error) {
	if m.batchFunc != nil {
		return m.batchFunc(ctx, prompts)
	}
	results := make([]repl.QueryResponse, len(prompts))
	for i := range prompts {
		results[i] = repl.QueryResponse{Response: "batch response"}
	}
	return results, nil
}

func TestNew(t *testing.T) {
	client := &mockLLMClient{}
	replClient := &mockREPLClient{}

	rlm := New(client, replClient)

	if rlm == nil {
		t.Fatal("expected non-nil RLM")
	}
	if rlm.client != client {
		t.Error("client not set correctly")
	}
	if rlm.replClient != replClient {
		t.Error("replClient not set correctly")
	}
}

func TestNewWithOptions(t *testing.T) {
	client := &mockLLMClient{}
	replClient := &mockREPLClient{}

	rlm := New(client, replClient,
		WithMaxIterations(10),
		WithSystemPrompt("custom prompt"),
		WithVerbose(true),
	)

	if rlm.config.MaxIterations != 10 {
		t.Errorf("MaxIterations = %d, want 10", rlm.config.MaxIterations)
	}
	if rlm.config.SystemPrompt != "custom prompt" {
		t.Errorf("SystemPrompt = %q, want %q", rlm.config.SystemPrompt, "custom prompt")
	}
	if !rlm.config.Verbose {
		t.Error("Verbose should be true")
	}
}

func TestDefaultConfig(t *testing.T) {
	cfg := DefaultConfig()

	if cfg.MaxIterations != 30 {
		t.Errorf("MaxIterations = %d, want 30", cfg.MaxIterations)
	}
	if cfg.SystemPrompt != SystemPrompt {
		t.Error("SystemPrompt should equal SystemPrompt constant")
	}
	if cfg.Verbose {
		t.Error("Verbose should be false by default")
	}
	if cfg.Logger != nil {
		t.Error("Logger should be nil by default")
	}
}

func TestCompleteWithDirectFinal(t *testing.T) {
	callCount := 0
	client := &mockLLMClient{
		completeFunc: func(ctx context.Context, messages []core.Message) (core.LLMResponse, error) {
			callCount++
			return core.LLMResponse{Content: "FINAL(42)", PromptTokens: 10, CompletionTokens: 5}, nil
		},
	}
	replClient := &mockREPLClient{}

	rlm := New(client, replClient, WithMaxIterations(5))

	result, err := rlm.Complete(context.Background(), "test context", "What is the answer?")
	if err != nil {
		t.Fatalf("Complete() error: %v", err)
	}

	if result.Response != "42" {
		t.Errorf("Response = %q, want %q", result.Response, "42")
	}
	if result.Iterations != 1 {
		t.Errorf("Iterations = %d, want 1", result.Iterations)
	}
	if callCount != 1 {
		t.Errorf("LLM called %d times, want 1", callCount)
	}
}

func TestCompleteWithFinalVar(t *testing.T) {
	// Test that FINAL_VAR is only processed when there are NO code blocks in the response.
	// This ensures the model waits for execution results before providing a final answer.
	callCount := 0
	client := &mockLLMClient{
		completeFunc: func(ctx context.Context, messages []core.Message) (core.LLMResponse, error) {
			callCount++
			if callCount == 1 {
				// First call: code block to set up the variable (FINAL_VAR ignored because code blocks present)
				return core.LLMResponse{Content: "```go\nanswer := \"the answer is 42\"\n```\nFINAL_VAR(answer)", PromptTokens: 10, CompletionTokens: 20}, nil
			}
			// Second call: just FINAL_VAR without code blocks - now it's processed
			return core.LLMResponse{Content: "FINAL_VAR(answer)", PromptTokens: 10, CompletionTokens: 5}, nil
		},
	}
	replClient := &mockREPLClient{}

	rlm := New(client, replClient, WithMaxIterations(5))

	result, err := rlm.Complete(context.Background(), "test context", "What is the answer?")
	if err != nil {
		t.Fatalf("Complete() error: %v", err)
	}

	if result.Response != "the answer is 42" {
		t.Errorf("Response = %q, want %q", result.Response, "the answer is 42")
	}
	// Now takes 2 iterations: first sets up variable, second returns it
	if result.Iterations != 2 {
		t.Errorf("Iterations = %d, want 2", result.Iterations)
	}
}

func TestCompleteMultipleIterations(t *testing.T) {
	callCount := 0
	client := &mockLLMClient{
		completeFunc: func(ctx context.Context, messages []core.Message) (core.LLMResponse, error) {
			callCount++
			if callCount < 3 {
				return core.LLMResponse{Content: "```go\nfmt.Println(\"thinking...\")\n```", PromptTokens: 10, CompletionTokens: 15}, nil
			}
			return core.LLMResponse{Content: "FINAL(done after 3 iterations)", PromptTokens: 10, CompletionTokens: 10}, nil
		},
	}
	replClient := &mockREPLClient{}

	rlm := New(client, replClient, WithMaxIterations(10))

	result, err := rlm.Complete(context.Background(), "test context", "Think hard")
	if err != nil {
		t.Fatalf("Complete() error: %v", err)
	}

	if result.Response != "done after 3 iterations" {
		t.Errorf("Response = %q, want %q", result.Response, "done after 3 iterations")
	}
	if result.Iterations != 3 {
		t.Errorf("Iterations = %d, want 3", result.Iterations)
	}
}

func TestCompleteMaxIterationsExhausted(t *testing.T) {
	callCount := 0
	client := &mockLLMClient{
		completeFunc: func(ctx context.Context, messages []core.Message) (core.LLMResponse, error) {
			callCount++
			if callCount <= 3 {
				return core.LLMResponse{Content: "```go\nfmt.Println(\"still thinking\")\n```", PromptTokens: 10, CompletionTokens: 15}, nil
			}
			// Default answer prompt should extract FINAL
			return core.LLMResponse{Content: "FINAL(forced answer)", PromptTokens: 10, CompletionTokens: 10}, nil
		},
	}
	replClient := &mockREPLClient{}

	rlm := New(client, replClient, WithMaxIterations(3))

	result, err := rlm.Complete(context.Background(), "test context", "Think forever")
	if err != nil {
		t.Fatalf("Complete() error: %v", err)
	}

	if result.Response != "forced answer" {
		t.Errorf("Response = %q, want %q", result.Response, "forced answer")
	}
	if result.Iterations != 3 {
		t.Errorf("Iterations = %d, want 3", result.Iterations)
	}
}

func TestCompleteLLMError(t *testing.T) {
	client := &mockLLMClient{
		completeFunc: func(ctx context.Context, messages []core.Message) (core.LLMResponse, error) {
			return core.LLMResponse{}, errors.New("LLM unavailable")
		},
	}
	replClient := &mockREPLClient{}

	rlm := New(client, replClient)

	_, err := rlm.Complete(context.Background(), "test", "query")
	if err == nil {
		t.Error("expected error from Complete()")
	}
	if !strings.Contains(err.Error(), "LLM unavailable") {
		t.Errorf("error should contain 'LLM unavailable', got: %v", err)
	}
}

func TestCompleteContextCancellation(t *testing.T) {
	client := &mockLLMClient{
		completeFunc: func(ctx context.Context, messages []core.Message) (core.LLMResponse, error) {
			// Simulate long operation
			select {
			case <-ctx.Done():
				return core.LLMResponse{}, ctx.Err()
			case <-time.After(100 * time.Millisecond):
				return core.LLMResponse{Content: "```go\nfmt.Println(1)\n```"}, nil
			}
		},
	}
	replClient := &mockREPLClient{}

	rlm := New(client, replClient, WithMaxIterations(100))

	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately

	_, err := rlm.Complete(ctx, "test", "query")
	if err == nil {
		t.Error("expected context cancellation error")
	}
}

func TestBuildInitialMessages(t *testing.T) {
	client := &mockLLMClient{}
	replClient := &mockREPLClient{}

	rlm := New(client, replClient, WithSystemPrompt("test system prompt"))

	replEnv := repl.New(replClient)
	_ = replEnv.LoadContext("test context")

	messages := rlm.buildInitialMessages(replEnv, "test query")

	if len(messages) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(messages))
	}

	if messages[0].Role != "system" {
		t.Errorf("first message role = %q, want 'system'", messages[0].Role)
	}
	if messages[0].Content != "test system prompt" {
		t.Errorf("system message content = %q, want 'test system prompt'", messages[0].Content)
	}

	if messages[1].Role != "user" {
		t.Errorf("second message role = %q, want 'user'", messages[1].Role)
	}
	if !strings.Contains(messages[1].Content, "test query") {
		t.Error("user message should contain the query")
	}
}

func TestAppendIterationPrompt(t *testing.T) {
	client := &mockLLMClient{}
	replClient := &mockREPLClient{}
	rlm := New(client, replClient)

	initialMessages := []core.Message{
		{Role: "system", Content: "system"},
		{Role: "user", Content: "initial"},
	}

	testQuery := "What is the answer?"

	// First iteration - should not add anything
	messages := rlm.appendIterationPrompt(initialMessages, 0, testQuery)
	if len(messages) != 2 {
		t.Errorf("iteration 0: expected 2 messages, got %d", len(messages))
	}

	// Second iteration - should add continuation prompt
	messages = rlm.appendIterationPrompt(initialMessages, 1, testQuery)
	if len(messages) != 3 {
		t.Errorf("iteration 1: expected 3 messages, got %d", len(messages))
	}
	if messages[2].Role != "user" {
		t.Errorf("added message role = %q, want 'user'", messages[2].Role)
	}
	// The prompt should contain the query as a reminder
	if !strings.Contains(messages[2].Content, testQuery) {
		t.Errorf("added message should contain query reminder")
	}
}

func TestAppendIterationToHistory(t *testing.T) {
	client := &mockLLMClient{}
	replClient := &mockREPLClient{}
	rlm := New(client, replClient)

	messages := []core.Message{
		{Role: "system", Content: "system"},
		{Role: "user", Content: "query"},
	}

	response := "Let me help you"
	blocks := []core.CodeBlock{
		{
			Code: "fmt.Println(1)",
			Result: core.ExecutionResult{
				Stdout: "1\n",
			},
		},
	}

	newMessages := rlm.appendIterationToHistory(messages, response, blocks)

	// Should have original 2 + 1 assistant + 1 user (for code block)
	if len(newMessages) != 4 {
		t.Errorf("expected 4 messages, got %d", len(newMessages))
	}

	// Check assistant message
	if newMessages[2].Role != "assistant" {
		t.Errorf("message[2] role = %q, want 'assistant'", newMessages[2].Role)
	}
	if newMessages[2].Content != response {
		t.Error("assistant message content should match response")
	}

	// Check code execution result message
	if newMessages[3].Role != "user" {
		t.Errorf("message[3] role = %q, want 'user'", newMessages[3].Role)
	}
	if !strings.Contains(newMessages[3].Content, "fmt.Println(1)") {
		t.Error("user message should contain the code")
	}
	if !strings.Contains(newMessages[3].Content, "1\n") {
		t.Error("user message should contain the output")
	}
}

func TestContextMetadata(t *testing.T) {
	tests := []struct {
		name     string
		payload  any
		expected string
	}{
		{
			name:     "string context",
			payload:  "hello world",
			expected: "string, 11 chars",
		},
		{
			name:     "empty string",
			payload:  "",
			expected: "string, 0 chars",
		},
		{
			name:     "array context",
			payload:  []any{1, 2, 3},
			expected: "array, 3 items",
		},
		{
			name:     "empty array",
			payload:  []any{},
			expected: "array, 0 items",
		},
		{
			name:     "map context",
			payload:  map[string]any{"key": "value"},
			expected: "object, 1 keys",
		},
		{
			name:     "empty map",
			payload:  map[string]any{},
			expected: "object, 0 keys",
		},
		{
			name:     "int type",
			payload:  42,
			expected: "int",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := ContextMetadata(tt.payload)
			if result != tt.expected {
				t.Errorf("ContextMetadata() = %q, want %q", result, tt.expected)
			}
		})
	}
}

func TestTruncate(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		maxLen   int
		expected string
	}{
		{
			name:     "short string",
			input:    "hello",
			maxLen:   10,
			expected: "hello",
		},
		{
			name:     "exact length",
			input:    "hello",
			maxLen:   5,
			expected: "hello",
		},
		{
			name:     "truncated",
			input:    "hello world",
			maxLen:   5,
			expected: "hello...",
		},
		{
			name:     "empty string",
			input:    "",
			maxLen:   5,
			expected: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := truncate(tt.input, tt.maxLen)
			if result != tt.expected {
				t.Errorf("truncate() = %q, want %q", result, tt.expected)
			}
		})
	}
}

func TestTruncateString(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		maxLen   int
		expected string
	}{
		{
			name:     "short string",
			input:    "hello",
			maxLen:   10,
			expected: "hello",
		},
		{
			name:     "truncated with suffix",
			input:    "hello world this is long",
			maxLen:   5,
			expected: "hello\n... (truncated)",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := truncateString(tt.input, tt.maxLen)
			if result != tt.expected {
				t.Errorf("truncateString() = %q, want %q", result, tt.expected)
			}
		})
	}
}

func TestCompleteWithCodeExecution(t *testing.T) {
	callCount := 0
	client := &mockLLMClient{
		completeFunc: func(ctx context.Context, messages []core.Message) (core.LLMResponse, error) {
			callCount++
			if callCount == 1 {
				return core.LLMResponse{Content: "```go\nresult := 2 + 2\nfmt.Println(result)\n```", PromptTokens: 10, CompletionTokens: 15}, nil
			}
			return core.LLMResponse{Content: "FINAL(4)", PromptTokens: 10, CompletionTokens: 5}, nil
		},
	}
	replClient := &mockREPLClient{}

	rlm := New(client, replClient, WithMaxIterations(5))

	result, err := rlm.Complete(context.Background(), "test", "Calculate")
	if err != nil {
		t.Fatalf("Complete() error: %v", err)
	}

	// Should have run 2 iterations
	if result.Iterations != 2 {
		t.Errorf("Iterations = %d, want 2", result.Iterations)
	}
}

func TestCompleteWithMapContext(t *testing.T) {
	client := &mockLLMClient{
		completeFunc: func(ctx context.Context, messages []core.Message) (core.LLMResponse, error) {
			return core.LLMResponse{Content: "FINAL(done)", PromptTokens: 10, CompletionTokens: 5}, nil
		},
	}
	replClient := &mockREPLClient{}

	rlm := New(client, replClient)

	ctx := map[string]any{
		"data": []int{1, 2, 3},
		"name": "test",
	}

	result, err := rlm.Complete(context.Background(), ctx, "Process data")
	if err != nil {
		t.Fatalf("Complete() error: %v", err)
	}

	if result.Response != "done" {
		t.Errorf("Response = %q, want %q", result.Response, "done")
	}
}

func TestCompleteWithSliceContext(t *testing.T) {
	client := &mockLLMClient{
		completeFunc: func(ctx context.Context, messages []core.Message) (core.LLMResponse, error) {
			return core.LLMResponse{Content: "FINAL(processed)", PromptTokens: 10, CompletionTokens: 5}, nil
		},
	}
	replClient := &mockREPLClient{}

	rlm := New(client, replClient)

	ctx := []any{"item1", "item2", "item3"}

	result, err := rlm.Complete(context.Background(), ctx, "Process items")
	if err != nil {
		t.Fatalf("Complete() error: %v", err)
	}

	if result.Response != "processed" {
		t.Errorf("Response = %q, want %q", result.Response, "processed")
	}
}

func TestPromptsConstants(t *testing.T) {
	// Verify prompts contain expected content
	if !strings.Contains(SystemPrompt, "FINAL") {
		t.Error("SystemPrompt should mention FINAL")
	}
	if !strings.Contains(SystemPrompt, "FINAL_VAR") {
		t.Error("SystemPrompt should mention FINAL_VAR")
	}
	if !strings.Contains(SystemPrompt, "Query") {
		t.Error("SystemPrompt should mention Query function")
	}
	if !strings.Contains(SystemPrompt, "QueryBatched") {
		t.Error("SystemPrompt should mention QueryBatched function")
	}

	if !strings.Contains(UserPromptTemplate, "%s") {
		t.Error("UserPromptTemplate should have format placeholders")
	}

	if FirstIterationSuffix == "" {
		t.Error("FirstIterationSuffix should not be empty")
	}

	if IterationPromptTemplate == "" {
		t.Error("IterationPromptTemplate should not be empty")
	}

	if !strings.Contains(DefaultAnswerPrompt, "FINAL") {
		t.Error("DefaultAnswerPrompt should mention FINAL")
	}
}

func TestCompleteResultDuration(t *testing.T) {
	client := &mockLLMClient{
		completeFunc: func(ctx context.Context, messages []core.Message) (core.LLMResponse, error) {
			time.Sleep(10 * time.Millisecond)
			return core.LLMResponse{Content: "FINAL(done)", PromptTokens: 10, CompletionTokens: 5}, nil
		},
	}
	replClient := &mockREPLClient{}

	rlm := New(client, replClient)

	result, err := rlm.Complete(context.Background(), "test", "query")
	if err != nil {
		t.Fatalf("Complete() error: %v", err)
	}

	if result.Duration < 10*time.Millisecond {
		t.Errorf("Duration = %v, expected >= 10ms", result.Duration)
	}
}

func TestWithLogger(t *testing.T) {
	// Test that WithLogger option works (without creating actual logger)
	client := &mockLLMClient{}
	replClient := &mockREPLClient{}

	rlm := New(client, replClient, WithLogger(nil))

	if rlm.config.Logger != nil {
		t.Error("Logger should be nil when set to nil")
	}
}
