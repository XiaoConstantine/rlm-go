package rlm

import (
	"context"
	"errors"
	"strings"
	"testing"
	"time"

	"github.com/XiaoConstantine/rlm-go/pkg/core"
	"github.com/XiaoConstantine/rlm-go/pkg/repl"
	"github.com/XiaoConstantine/rlm-go/pkg/sandbox"
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

// mockStreamingLLMClient implements StreamingLLMClient for testing
type mockStreamingLLMClient struct {
	completeFunc       func(ctx context.Context, messages []core.Message) (core.LLMResponse, error)
	completeStreamFunc func(ctx context.Context, messages []core.Message, handler StreamHandler) (core.LLMResponse, error)
	calls              [][]core.Message
	streamCalls        [][]core.Message
}

func (m *mockStreamingLLMClient) Complete(ctx context.Context, messages []core.Message) (core.LLMResponse, error) {
	m.calls = append(m.calls, messages)
	return m.completeFunc(ctx, messages)
}

func (m *mockStreamingLLMClient) CompleteStream(ctx context.Context, messages []core.Message, handler StreamHandler) (core.LLMResponse, error) {
	m.streamCalls = append(m.streamCalls, messages)
	return m.completeStreamFunc(ctx, messages, handler)
}

func TestWithStreaming(t *testing.T) {
	client := &mockLLMClient{}
	replClient := &mockREPLClient{}

	rlm := New(client, replClient, WithStreaming(true))

	if !rlm.config.EnableStreaming {
		t.Error("EnableStreaming should be true")
	}
}

func TestWithStreamHandler(t *testing.T) {
	client := &mockLLMClient{}
	replClient := &mockREPLClient{}

	var called bool
	handler := func(chunk string, done bool) error {
		called = true
		return nil
	}

	rlm := New(client, replClient, WithStreamHandler(handler))

	if rlm.config.OnStreamChunk == nil {
		t.Error("OnStreamChunk should be set")
	}

	// Call the handler to verify it's the right one
	_ = rlm.config.OnStreamChunk("test", false)
	if !called {
		t.Error("Handler should have been called")
	}
}

func TestCompleteWithStreaming(t *testing.T) {
	var chunks []string
	handler := func(chunk string, done bool) error {
		if chunk != "" {
			chunks = append(chunks, chunk)
		}
		return nil
	}

	client := &mockStreamingLLMClient{
		completeFunc: func(ctx context.Context, messages []core.Message) (core.LLMResponse, error) {
			return core.LLMResponse{Content: "FINAL(42)", PromptTokens: 10, CompletionTokens: 5}, nil
		},
		completeStreamFunc: func(ctx context.Context, messages []core.Message, h StreamHandler) (core.LLMResponse, error) {
			// Simulate streaming chunks
			if h != nil {
				_ = h("FI", false)
				_ = h("NAL", false)
				_ = h("(42)", false)
				_ = h("", true)
			}
			return core.LLMResponse{Content: "FINAL(42)", PromptTokens: 10, CompletionTokens: 5}, nil
		},
	}
	replClient := &mockREPLClient{}

	rlm := New(client, replClient,
		WithStreaming(true),
		WithStreamHandler(handler),
		WithMaxIterations(5),
	)

	result, err := rlm.Complete(context.Background(), "test context", "What is the answer?")
	if err != nil {
		t.Fatalf("Complete() error: %v", err)
	}

	// Verify streaming was used
	if len(client.streamCalls) != 1 {
		t.Errorf("Expected 1 stream call, got %d", len(client.streamCalls))
	}
	if len(client.calls) != 0 {
		t.Errorf("Expected 0 regular calls, got %d", len(client.calls))
	}

	// Verify result
	if result.Response != "42" {
		t.Errorf("Response = %q, want %q", result.Response, "42")
	}

	// Verify chunks were received
	if len(chunks) != 3 {
		t.Errorf("Expected 3 chunks, got %d: %v", len(chunks), chunks)
	}
}

func TestCompleteWithStreamingFallback(t *testing.T) {
	// Test that non-streaming client falls back to regular Complete
	client := &mockLLMClient{
		completeFunc: func(ctx context.Context, messages []core.Message) (core.LLMResponse, error) {
			return core.LLMResponse{Content: "FINAL(42)", PromptTokens: 10, CompletionTokens: 5}, nil
		},
	}
	replClient := &mockREPLClient{}

	rlm := New(client, replClient,
		WithStreaming(true), // Enabled but client doesn't support it
		WithMaxIterations(5),
	)

	result, err := rlm.Complete(context.Background(), "test context", "What is the answer?")
	if err != nil {
		t.Fatalf("Complete() error: %v", err)
	}

	// Verify regular Complete was used
	if len(client.calls) != 1 {
		t.Errorf("Expected 1 regular call, got %d", len(client.calls))
	}

	if result.Response != "42" {
		t.Errorf("Response = %q, want %q", result.Response, "42")
	}
}

func TestCompleteWithStreamingDisabled(t *testing.T) {
	client := &mockStreamingLLMClient{
		completeFunc: func(ctx context.Context, messages []core.Message) (core.LLMResponse, error) {
			return core.LLMResponse{Content: "FINAL(42)", PromptTokens: 10, CompletionTokens: 5}, nil
		},
		completeStreamFunc: func(ctx context.Context, messages []core.Message, h StreamHandler) (core.LLMResponse, error) {
			t.Error("Stream should not be called when streaming is disabled")
			return core.LLMResponse{}, nil
		},
	}
	replClient := &mockREPLClient{}

	rlm := New(client, replClient,
		WithStreaming(false), // Explicitly disabled
		WithMaxIterations(5),
	)

	result, err := rlm.Complete(context.Background(), "test context", "What is the answer?")
	if err != nil {
		t.Fatalf("Complete() error: %v", err)
	}

	// Verify regular Complete was used
	if len(client.calls) != 1 {
		t.Errorf("Expected 1 regular call, got %d", len(client.calls))
	}

	if result.Response != "42" {
		t.Errorf("Response = %q, want %q", result.Response, "42")
	}
}

func TestWithREPLPool(t *testing.T) {
	replClient := &mockREPLClient{}
	pool := repl.NewREPLPool(replClient, 2, true)

	client := &mockLLMClient{
		completeFunc: func(ctx context.Context, messages []core.Message) (core.LLMResponse, error) {
			return core.LLMResponse{Content: "FINAL(done)", PromptTokens: 10, CompletionTokens: 5}, nil
		},
	}

	rlm := New(client, replClient,
		WithREPLPool(pool),
		WithMaxIterations(5),
	)

	if rlm.config.REPLPool != pool {
		t.Error("REPLPool should be set")
	}

	// Complete should work with pool
	result, err := rlm.Complete(context.Background(), "test", "query")
	if err != nil {
		t.Fatalf("Complete() error: %v", err)
	}

	if result.Response != "done" {
		t.Errorf("Response = %q, want %q", result.Response, "done")
	}
}

func TestCompleteWithREPLPoolMultipleCalls(t *testing.T) {
	replClient := &mockREPLClient{}
	pool := repl.NewREPLPool(replClient, 2, true)

	client := &mockLLMClient{
		completeFunc: func(ctx context.Context, messages []core.Message) (core.LLMResponse, error) {
			return core.LLMResponse{Content: "FINAL(done)", PromptTokens: 10, CompletionTokens: 5}, nil
		},
	}

	rlm := New(client, replClient,
		WithREPLPool(pool),
		WithMaxIterations(5),
	)

	// Run multiple completions to verify pool works correctly
	for i := 0; i < 5; i++ {
		result, err := rlm.Complete(context.Background(), "test", "query")
		if err != nil {
			t.Fatalf("Complete() #%d error: %v", i, err)
		}
		if result.Response != "done" {
			t.Errorf("Complete() #%d Response = %q, want %q", i, result.Response, "done")
		}
	}
}

func TestWithHistoryCompression(t *testing.T) {
	client := &mockLLMClient{}
	replClient := &mockREPLClient{}

	rlm := New(client, replClient, WithHistoryCompression(3, 500))

	if rlm.config.HistoryCompression == nil {
		t.Fatal("HistoryCompression should be set")
	}
	if !rlm.config.HistoryCompression.Enabled {
		t.Error("HistoryCompression should be enabled")
	}
	if rlm.config.HistoryCompression.VerbatimIterations != 3 {
		t.Errorf("VerbatimIterations = %d, want 3", rlm.config.HistoryCompression.VerbatimIterations)
	}
	if rlm.config.HistoryCompression.MaxSummaryTokens != 500 {
		t.Errorf("MaxSummaryTokens = %d, want 500", rlm.config.HistoryCompression.MaxSummaryTokens)
	}
}

func TestWithHistoryCompressionDefaults(t *testing.T) {
	client := &mockLLMClient{}
	replClient := &mockREPLClient{}

	// Test with invalid values - should use defaults
	rlm := New(client, replClient, WithHistoryCompression(0, 0))

	if rlm.config.HistoryCompression == nil {
		t.Fatal("HistoryCompression should be set")
	}
	if rlm.config.HistoryCompression.VerbatimIterations != 3 {
		t.Errorf("VerbatimIterations = %d, want 3 (default)", rlm.config.HistoryCompression.VerbatimIterations)
	}
	if rlm.config.HistoryCompression.MaxSummaryTokens != 500 {
		t.Errorf("MaxSummaryTokens = %d, want 500 (default)", rlm.config.HistoryCompression.MaxSummaryTokens)
	}
}

func TestCompressHistory(t *testing.T) {
	client := &mockLLMClient{}
	replClient := &mockREPLClient{}

	rlm := New(client, replClient, WithHistoryCompression(2, 500))

	// Build a message history with multiple iterations
	messages := []core.Message{
		{Role: "system", Content: "System prompt"},
		{Role: "user", Content: "Initial query with context"},
		// Iteration 1
		{Role: "assistant", Content: "```go\nfmt.Println(1)\n```"},
		{Role: "user", Content: "Code executed:\n```go\nfmt.Println(1)\n```\n\nREPL output:\n1"},
		// Iteration 2
		{Role: "assistant", Content: "```go\nfmt.Println(2)\n```"},
		{Role: "user", Content: "Code executed:\n```go\nfmt.Println(2)\n```\n\nREPL output:\n2"},
		// Iteration 3
		{Role: "assistant", Content: "```go\nfmt.Println(3)\n```"},
		{Role: "user", Content: "Code executed:\n```go\nfmt.Println(3)\n```\n\nREPL output:\n3"},
		// Iteration 4
		{Role: "assistant", Content: "```go\nfmt.Println(4)\n```"},
		{Role: "user", Content: "Code executed:\n```go\nfmt.Println(4)\n```\n\nREPL output:\n4"},
		// Iteration 5
		{Role: "assistant", Content: "FINAL(5)"},
		{Role: "user", Content: "Continue..."},
	}

	// With VerbatimIterations=2, we should keep only last 4 messages (2 iterations * 2 msgs)
	compressed := rlm.compressHistory(messages, 5)

	// Should have: system(1) + initial user(1) + summary(1) + last 4 verbatim = 7
	// But with our messagesPerIteration=2 estimate for VerbatimIterations=2, we keep 4 messages
	if len(compressed) > len(messages) {
		t.Errorf("Compressed length %d should not exceed original %d", len(compressed), len(messages))
	}

	// First two should be preserved
	if compressed[0].Role != "system" {
		t.Error("First message should be system")
	}
	if compressed[1].Role != "user" {
		t.Error("Second message should be user (initial query)")
	}

	// Check that summary was created
	if len(compressed) > 2 && !strings.Contains(compressed[2].Content, "[Previous iterations summary]") {
		// If there's a third message and history was compressed, it should be the summary
		t.Logf("Third message: %s", compressed[2].Content[:min(100, len(compressed[2].Content))])
	}
}

func TestCompressHistoryNotEnoughMessages(t *testing.T) {
	client := &mockLLMClient{}
	replClient := &mockREPLClient{}

	rlm := New(client, replClient, WithHistoryCompression(3, 500))

	// Build a short message history
	messages := []core.Message{
		{Role: "system", Content: "System prompt"},
		{Role: "user", Content: "Initial query"},
		{Role: "assistant", Content: "Response"},
		{Role: "user", Content: "Result"},
	}

	compressed := rlm.compressHistory(messages, 1)

	// Should not compress - not enough messages
	if len(compressed) != len(messages) {
		t.Errorf("Should not compress short history: got %d, want %d", len(compressed), len(messages))
	}
}

func TestCompressHistoryDisabled(t *testing.T) {
	client := &mockLLMClient{}
	replClient := &mockREPLClient{}

	rlm := New(client, replClient) // No compression

	messages := []core.Message{
		{Role: "system", Content: "System prompt"},
		{Role: "user", Content: "Initial query"},
		{Role: "assistant", Content: "Response 1"},
		{Role: "user", Content: "Result 1"},
		{Role: "assistant", Content: "Response 2"},
		{Role: "user", Content: "Result 2"},
		{Role: "assistant", Content: "Response 3"},
		{Role: "user", Content: "Result 3"},
		{Role: "assistant", Content: "Response 4"},
		{Role: "user", Content: "Result 4"},
	}

	compressed := rlm.compressHistory(messages, 4)

	// Should not compress when disabled
	if len(compressed) != len(messages) {
		t.Errorf("Should not compress when disabled: got %d, want %d", len(compressed), len(messages))
	}
}

func TestSummarizeIterations(t *testing.T) {
	client := &mockLLMClient{}
	replClient := &mockREPLClient{}

	rlm := New(client, replClient, WithHistoryCompression(2, 500))

	messages := []core.Message{
		{Role: "assistant", Content: "```go\nfmt.Println(\"hello\")\n```"},
		{Role: "user", Content: "Code executed:\n```go\nfmt.Println(\"hello\")\n```\n\nREPL output:\nhello"},
		{Role: "assistant", Content: "Let me try something else\n```go\nx := 1\n```"},
		{Role: "user", Content: "Code executed:\n```go\nx := 1\n```\n\nREPL output:\nNo output"},
		{Role: "assistant", Content: "FINAL(done)"},
	}

	summary := rlm.summarizeIterations(messages, 500)

	if !strings.Contains(summary, "[Previous iterations summary]") {
		t.Error("Summary should contain header")
	}
	if !strings.Contains(summary, "Iteration 1") {
		t.Error("Summary should mention Iteration 1")
	}
	if !strings.Contains(summary, "executed code") {
		t.Error("Summary should mention code execution")
	}
}

func TestCompleteWithHistoryCompression(t *testing.T) {
	// Test that compression doesn't break completion
	callCount := 0
	client := &mockLLMClient{
		completeFunc: func(ctx context.Context, messages []core.Message) (core.LLMResponse, error) {
			callCount++
			if callCount < 6 {
				return core.LLMResponse{Content: "```go\nfmt.Println(" + string(rune('0'+callCount)) + ")\n```", PromptTokens: 10, CompletionTokens: 15}, nil
			}
			return core.LLMResponse{Content: "FINAL(done)", PromptTokens: 10, CompletionTokens: 5}, nil
		},
	}
	replClient := &mockREPLClient{}

	rlm := New(client, replClient,
		WithHistoryCompression(2, 500),
		WithMaxIterations(10),
	)

	result, err := rlm.Complete(context.Background(), "test context", "Process this")
	if err != nil {
		t.Fatalf("Complete() error: %v", err)
	}

	if result.Response != "done" {
		t.Errorf("Response = %q, want %q", result.Response, "done")
	}
	if result.Iterations != 6 {
		t.Errorf("Iterations = %d, want 6", result.Iterations)
	}
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// Tests for Adaptive Iteration Strategy

func TestWithAdaptiveIteration(t *testing.T) {
	client := &mockLLMClient{}
	replClient := &mockREPLClient{}

	rlm := New(client, replClient, WithAdaptiveIteration())

	if rlm.config.AdaptiveIteration == nil {
		t.Fatal("AdaptiveIteration should be set")
	}
	if !rlm.config.AdaptiveIteration.Enabled {
		t.Error("AdaptiveIteration should be enabled")
	}
	if rlm.config.AdaptiveIteration.BaseIterations != 10 {
		t.Errorf("BaseIterations = %d, want 10", rlm.config.AdaptiveIteration.BaseIterations)
	}
	if rlm.config.AdaptiveIteration.MaxIterations != 50 {
		t.Errorf("MaxIterations = %d, want 50", rlm.config.AdaptiveIteration.MaxIterations)
	}
	if rlm.config.AdaptiveIteration.ContextScaleFactor != 100000 {
		t.Errorf("ContextScaleFactor = %d, want 100000", rlm.config.AdaptiveIteration.ContextScaleFactor)
	}
	if !rlm.config.AdaptiveIteration.EnableEarlyTermination {
		t.Error("EnableEarlyTermination should be true")
	}
	if rlm.config.AdaptiveIteration.ConfidenceThreshold != 1 {
		t.Errorf("ConfidenceThreshold = %d, want 1", rlm.config.AdaptiveIteration.ConfidenceThreshold)
	}
}

func TestWithAdaptiveIterationConfig(t *testing.T) {
	client := &mockLLMClient{}
	replClient := &mockREPLClient{}

	cfg := AdaptiveIterationConfig{
		BaseIterations:         5,
		MaxIterations:          25,
		ContextScaleFactor:     50000,
		EnableEarlyTermination: false,
		ConfidenceThreshold:    2,
	}

	rlm := New(client, replClient, WithAdaptiveIterationConfig(cfg))

	if rlm.config.AdaptiveIteration == nil {
		t.Fatal("AdaptiveIteration should be set")
	}
	if rlm.config.AdaptiveIteration.BaseIterations != 5 {
		t.Errorf("BaseIterations = %d, want 5", rlm.config.AdaptiveIteration.BaseIterations)
	}
	if rlm.config.AdaptiveIteration.MaxIterations != 25 {
		t.Errorf("MaxIterations = %d, want 25", rlm.config.AdaptiveIteration.MaxIterations)
	}
	if rlm.config.AdaptiveIteration.ContextScaleFactor != 50000 {
		t.Errorf("ContextScaleFactor = %d, want 50000", rlm.config.AdaptiveIteration.ContextScaleFactor)
	}
	if rlm.config.AdaptiveIteration.EnableEarlyTermination {
		t.Error("EnableEarlyTermination should be false")
	}
	if rlm.config.AdaptiveIteration.ConfidenceThreshold != 2 {
		t.Errorf("ConfidenceThreshold = %d, want 2", rlm.config.AdaptiveIteration.ConfidenceThreshold)
	}
}

func TestWithAdaptiveIterationConfigDefaults(t *testing.T) {
	client := &mockLLMClient{}
	replClient := &mockREPLClient{}

	// Test with zero values - should use defaults
	cfg := AdaptiveIterationConfig{}
	rlm := New(client, replClient, WithAdaptiveIterationConfig(cfg))

	if rlm.config.AdaptiveIteration.BaseIterations != 10 {
		t.Errorf("BaseIterations = %d, want 10 (default)", rlm.config.AdaptiveIteration.BaseIterations)
	}
	if rlm.config.AdaptiveIteration.MaxIterations != 50 {
		t.Errorf("MaxIterations = %d, want 50 (default)", rlm.config.AdaptiveIteration.MaxIterations)
	}
	if rlm.config.AdaptiveIteration.ContextScaleFactor != 100000 {
		t.Errorf("ContextScaleFactor = %d, want 100000 (default)", rlm.config.AdaptiveIteration.ContextScaleFactor)
	}
	if rlm.config.AdaptiveIteration.ConfidenceThreshold != 1 {
		t.Errorf("ConfidenceThreshold = %d, want 1 (default)", rlm.config.AdaptiveIteration.ConfidenceThreshold)
	}
}

func TestWithProgressHandler(t *testing.T) {
	client := &mockLLMClient{}
	replClient := &mockREPLClient{}

	var progressUpdates []IterationProgress
	handler := func(progress IterationProgress) {
		progressUpdates = append(progressUpdates, progress)
	}

	rlm := New(client, replClient, WithProgressHandler(handler))

	if rlm.config.OnProgress == nil {
		t.Error("OnProgress should be set")
	}

	// Call the handler to verify it works
	rlm.config.OnProgress(IterationProgress{CurrentIteration: 1, MaxIterations: 10})
	if len(progressUpdates) != 1 {
		t.Errorf("expected 1 progress update, got %d", len(progressUpdates))
	}
}

func TestComputeMaxIterations(t *testing.T) {
	replClient := &mockREPLClient{}
	client := &mockLLMClient{}

	tests := []struct {
		name        string
		contextSize int
		config      *AdaptiveIterationConfig
		maxDefault  int
		expected    int
	}{
		{
			name:        "no adaptive config uses default",
			contextSize: 500000,
			config:      nil,
			maxDefault:  30,
			expected:    30,
		},
		{
			name:        "small context uses base iterations",
			contextSize: 10000,
			config: &AdaptiveIterationConfig{
				Enabled:            true,
				BaseIterations:     10,
				MaxIterations:      50,
				ContextScaleFactor: 100000,
			},
			maxDefault: 30,
			expected:   10,
		},
		{
			name:        "medium context adds iterations",
			contextSize: 300000,
			config: &AdaptiveIterationConfig{
				Enabled:            true,
				BaseIterations:     10,
				MaxIterations:      50,
				ContextScaleFactor: 100000,
			},
			maxDefault: 30,
			expected:   13, // 10 + 300000/100000 = 13
		},
		{
			name:        "large context capped at max",
			contextSize: 10000000, // 10MB
			config: &AdaptiveIterationConfig{
				Enabled:            true,
				BaseIterations:     10,
				MaxIterations:      50,
				ContextScaleFactor: 100000,
			},
			maxDefault: 30,
			expected:   50,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rlm := New(client, replClient, WithMaxIterations(tt.maxDefault))
			rlm.config.AdaptiveIteration = tt.config

			result := rlm.computeMaxIterations(tt.contextSize)
			if result != tt.expected {
				t.Errorf("computeMaxIterations(%d) = %d, want %d", tt.contextSize, result, tt.expected)
			}
		})
	}
}

func TestDetectConfidence(t *testing.T) {
	tests := []struct {
		name     string
		response string
		expected bool
	}{
		{
			name:     "confident lowercase",
			response: "I'm confident the answer is 42",
			expected: true,
		},
		{
			name:     "confident uppercase",
			response: "I AM CONFIDENT in this result",
			expected: true,
		},
		{
			name:     "certain",
			response: "I'm certain this is correct",
			expected: true,
		},
		{
			name:     "definitive answer",
			response: "The final answer is 42",
			expected: true,
		},
		{
			name:     "found the answer",
			response: "I have found the answer we were looking for",
			expected: true,
		},
		{
			name:     "with certainty",
			response: "With certainty, the result is ALPHA-7892",
			expected: true,
		},
		{
			name:     "conclusively",
			response: "Conclusively, the secret code is XYZ",
			expected: true,
		},
		{
			name:     "no confidence signal",
			response: "Let me explore the context more",
			expected: false,
		},
		{
			name:     "thinking out loud",
			response: "I need to run more code to verify",
			expected: false,
		},
		{
			name:     "empty response",
			response: "",
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := detectConfidence(tt.response)
			if result != tt.expected {
				t.Errorf("detectConfidence() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestShouldTerminateEarly(t *testing.T) {
	replClient := &mockREPLClient{}
	client := &mockLLMClient{}

	tests := []struct {
		name              string
		config            *AdaptiveIterationConfig
		confidenceSignals int
		pendingCodeBlocks int
		expected          bool
	}{
		{
			name:              "no adaptive config",
			config:            nil,
			confidenceSignals: 5,
			pendingCodeBlocks: 0,
			expected:          false,
		},
		{
			name: "early termination disabled",
			config: &AdaptiveIterationConfig{
				Enabled:                true,
				EnableEarlyTermination: false,
				ConfidenceThreshold:    1,
			},
			confidenceSignals: 5,
			pendingCodeBlocks: 0,
			expected:          false,
		},
		{
			name: "threshold not met",
			config: &AdaptiveIterationConfig{
				Enabled:                true,
				EnableEarlyTermination: true,
				ConfidenceThreshold:    3,
			},
			confidenceSignals: 2,
			pendingCodeBlocks: 0,
			expected:          false,
		},
		{
			name: "pending code blocks",
			config: &AdaptiveIterationConfig{
				Enabled:                true,
				EnableEarlyTermination: true,
				ConfidenceThreshold:    1,
			},
			confidenceSignals: 5,
			pendingCodeBlocks: 1,
			expected:          false,
		},
		{
			name: "should terminate",
			config: &AdaptiveIterationConfig{
				Enabled:                true,
				EnableEarlyTermination: true,
				ConfidenceThreshold:    1,
			},
			confidenceSignals: 1,
			pendingCodeBlocks: 0,
			expected:          true,
		},
		{
			name: "threshold exactly met",
			config: &AdaptiveIterationConfig{
				Enabled:                true,
				EnableEarlyTermination: true,
				ConfidenceThreshold:    3,
			},
			confidenceSignals: 3,
			pendingCodeBlocks: 0,
			expected:          true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			rlm := New(client, replClient)
			rlm.config.AdaptiveIteration = tt.config

			result := rlm.shouldTerminateEarly(tt.confidenceSignals, tt.pendingCodeBlocks)
			if result != tt.expected {
				t.Errorf("shouldTerminateEarly() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestGetContextSize(t *testing.T) {
	tests := []struct {
		name     string
		payload  any
		expected int
	}{
		{
			name:     "string payload",
			payload:  "hello world",
			expected: 11,
		},
		{
			name:     "empty string",
			payload:  "",
			expected: 0,
		},
		{
			name:     "byte slice",
			payload:  []byte("hello"),
			expected: 5,
		},
		{
			name:    "map payload",
			payload: map[string]any{"key": "value"},
			// Size is approximate for non-string types
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := getContextSize(tt.payload)
			if tt.expected > 0 && result != tt.expected {
				t.Errorf("getContextSize() = %d, want %d", result, tt.expected)
			}
			if tt.expected == 0 && tt.name == "empty string" && result != 0 {
				t.Errorf("getContextSize() = %d, want 0", result)
			}
		})
	}
}

func TestCompleteWithProgressHandler(t *testing.T) {
	var progressUpdates []IterationProgress

	callCount := 0
	client := &mockLLMClient{
		completeFunc: func(ctx context.Context, messages []core.Message) (core.LLMResponse, error) {
			callCount++
			if callCount == 1 {
				return core.LLMResponse{Content: "```go\nfmt.Println(1)\n```", PromptTokens: 10, CompletionTokens: 15}, nil
			}
			return core.LLMResponse{Content: "FINAL(done)", PromptTokens: 10, CompletionTokens: 5}, nil
		},
	}
	replClient := &mockREPLClient{}

	handler := func(progress IterationProgress) {
		progressUpdates = append(progressUpdates, progress)
	}

	rlm := New(client, replClient,
		WithProgressHandler(handler),
		WithMaxIterations(10),
	)

	result, err := rlm.Complete(context.Background(), "test context", "query")
	if err != nil {
		t.Fatalf("Complete() error: %v", err)
	}

	if result.Response != "done" {
		t.Errorf("Response = %q, want %q", result.Response, "done")
	}

	// Should have received 2 progress updates (one per iteration)
	if len(progressUpdates) != 2 {
		t.Errorf("received %d progress updates, want 2", len(progressUpdates))
	}

	// Verify first progress update
	if len(progressUpdates) > 0 {
		if progressUpdates[0].CurrentIteration != 1 {
			t.Errorf("first update CurrentIteration = %d, want 1", progressUpdates[0].CurrentIteration)
		}
		if progressUpdates[0].MaxIterations != 10 {
			t.Errorf("first update MaxIterations = %d, want 10", progressUpdates[0].MaxIterations)
		}
	}
}

func TestCompleteWithAdaptiveIterationAndLargeContext(t *testing.T) {
	// Create a large context to test adaptive iteration scaling
	largeContext := strings.Repeat("data ", 60000) // ~300KB

	client := &mockLLMClient{
		completeFunc: func(ctx context.Context, messages []core.Message) (core.LLMResponse, error) {
			return core.LLMResponse{Content: "FINAL(done)", PromptTokens: 10, CompletionTokens: 5}, nil
		},
	}
	replClient := &mockREPLClient{}

	var maxIterationsUsed int
	handler := func(progress IterationProgress) {
		maxIterationsUsed = progress.MaxIterations
	}

	rlm := New(client, replClient,
		WithAdaptiveIteration(),
		WithProgressHandler(handler),
	)

	result, err := rlm.Complete(context.Background(), largeContext, "query")
	if err != nil {
		t.Fatalf("Complete() error: %v", err)
	}

	if result.Response != "done" {
		t.Errorf("Response = %q, want %q", result.Response, "done")
	}

	// With 300KB context, should have more than base 10 iterations
	// 10 + 300000/100000 = 13
	if maxIterationsUsed < 10 {
		t.Errorf("maxIterationsUsed = %d, expected >= 10", maxIterationsUsed)
	}
}

// Sandbox integration tests

func TestWithSandbox(t *testing.T) {
	client := &mockLLMClient{}
	replClient := &mockREPLClient{}

	rlm := New(client, replClient, WithSandbox())

	if rlm.config.Sandbox == nil {
		t.Fatal("expected Sandbox config to be set")
	}
	if !rlm.config.Sandbox.Enabled {
		t.Error("expected Sandbox.Enabled to be true")
	}
	if rlm.config.Sandbox.Config == nil {
		t.Error("expected Sandbox.Config to be set")
	}
	if rlm.config.Sandbox.Config.Backend != sandbox.BackendAuto {
		t.Errorf("expected BackendAuto, got %s", rlm.config.Sandbox.Config.Backend)
	}
}

func TestWithSandboxConfig(t *testing.T) {
	client := &mockLLMClient{}
	replClient := &mockREPLClient{}

	cfg := sandbox.Config{
		Backend: sandbox.BackendLocal,
		Memory:  "256m",
		CPUs:    0.5,
	}

	rlm := New(client, replClient, WithSandboxConfig(cfg))

	if rlm.config.Sandbox == nil {
		t.Fatal("expected Sandbox config to be set")
	}
	if rlm.config.Sandbox.Config.Backend != sandbox.BackendLocal {
		t.Errorf("expected BackendLocal, got %s", rlm.config.Sandbox.Config.Backend)
	}
	if rlm.config.Sandbox.Config.Memory != "256m" {
		t.Errorf("expected 256m, got %s", rlm.config.Sandbox.Config.Memory)
	}
	if rlm.config.Sandbox.Config.CPUs != 0.5 {
		t.Errorf("expected 0.5, got %f", rlm.config.Sandbox.Config.CPUs)
	}
}

func TestWithSandboxBackend(t *testing.T) {
	client := &mockLLMClient{}
	replClient := &mockREPLClient{}

	rlm := New(client, replClient, WithSandboxBackend(sandbox.BackendLocal))

	if rlm.config.Sandbox == nil {
		t.Fatal("expected Sandbox config to be set")
	}
	if rlm.config.Sandbox.Config.Backend != sandbox.BackendLocal {
		t.Errorf("expected BackendLocal, got %s", rlm.config.Sandbox.Config.Backend)
	}
}

func TestCompleteWithSandboxLocal(t *testing.T) {
	// Test Complete with local sandbox (no container runtime needed)
	client := &mockLLMClient{
		completeFunc: func(ctx context.Context, messages []core.Message) (core.LLMResponse, error) {
			return core.LLMResponse{Content: "FINAL(sandbox result)", PromptTokens: 10, CompletionTokens: 5}, nil
		},
	}
	replClient := &mockREPLClient{}

	rlm := New(client, replClient, WithSandboxBackend(sandbox.BackendLocal))

	result, err := rlm.Complete(context.Background(), "test context", "What is the answer?")
	if err != nil {
		t.Fatalf("Complete() error: %v", err)
	}

	if result.Response != "sandbox result" {
		t.Errorf("Response = %q, want %q", result.Response, "sandbox result")
	}
}

func TestCompleteWithSandboxAndCodeExecution(t *testing.T) {
	// Test that code execution works in sandbox mode
	callCount := 0
	client := &mockLLMClient{
		completeFunc: func(ctx context.Context, messages []core.Message) (core.LLMResponse, error) {
			callCount++
			if callCount == 1 {
				// First call: return code block
				return core.LLMResponse{
					Content: "```go\nfmt.Println(\"Hello from sandbox\")\n```",
					PromptTokens: 10,
					CompletionTokens: 20,
				}, nil
			}
			// Second call: return final answer
			return core.LLMResponse{Content: "FINAL(executed)", PromptTokens: 10, CompletionTokens: 5}, nil
		},
	}
	replClient := &mockREPLClient{}

	rlm := New(client, replClient, WithSandboxBackend(sandbox.BackendLocal))

	result, err := rlm.Complete(context.Background(), "test context", "Run some code")
	if err != nil {
		t.Fatalf("Complete() error: %v", err)
	}

	if result.Response != "executed" {
		t.Errorf("Response = %q, want %q", result.Response, "executed")
	}
	if result.Iterations != 2 {
		t.Errorf("Iterations = %d, want 2", result.Iterations)
	}
}

func TestCreateExecutionEnvironmentWithSandbox(t *testing.T) {
	client := &mockLLMClient{}
	replClient := &mockREPLClient{}

	// Test with sandbox enabled
	rlm := New(client, replClient, WithSandboxBackend(sandbox.BackendLocal))

	env, err := rlm.createExecutionEnvironment()
	if err != nil {
		t.Fatalf("createExecutionEnvironment() error: %v", err)
	}
	defer env.Close()

	// Verify it's a sandbox adapter
	if _, ok := env.(*SandboxAdapter); !ok {
		t.Errorf("expected *SandboxAdapter, got %T", env)
	}
}

func TestCreateExecutionEnvironmentWithoutSandbox(t *testing.T) {
	client := &mockLLMClient{}
	replClient := &mockREPLClient{}

	// Test without sandbox (default behavior)
	rlm := New(client, replClient)

	env, err := rlm.createExecutionEnvironment()
	if err != nil {
		t.Fatalf("createExecutionEnvironment() error: %v", err)
	}
	defer env.Close()

	// Verify it's a REPL adapter
	if _, ok := env.(*REPLAdapter); !ok {
		t.Errorf("expected *REPLAdapter, got %T", env)
	}
}

// Tests for Compact History

func TestWithCompactHistory(t *testing.T) {
	client := &mockLLMClient{}
	replClient := &mockREPLClient{}

	rlm := New(client, replClient, WithCompactHistory(true))

	if rlm.config.CompactHistory == nil {
		t.Fatal("CompactHistory should be set")
	}
	if !rlm.config.CompactHistory.Enabled {
		t.Error("CompactHistory should be enabled")
	}
	if !rlm.config.CompactHistory.IncludeFewShot {
		t.Error("IncludeFewShot should be true")
	}
	if rlm.config.CompactHistory.MaxHistoryLength != 10000 {
		t.Errorf("MaxHistoryLength = %d, want 10000", rlm.config.CompactHistory.MaxHistoryLength)
	}
}

func TestWithCompactHistoryConfig(t *testing.T) {
	client := &mockLLMClient{}
	replClient := &mockREPLClient{}

	cfg := CompactHistoryConfig{
		MaxHistoryLength: 5000,
		IncludeFewShot:   false,
	}

	rlm := New(client, replClient, WithCompactHistoryConfig(cfg))

	if rlm.config.CompactHistory == nil {
		t.Fatal("CompactHistory should be set")
	}
	if !rlm.config.CompactHistory.Enabled {
		t.Error("CompactHistory should be enabled")
	}
	if rlm.config.CompactHistory.IncludeFewShot {
		t.Error("IncludeFewShot should be false")
	}
	if rlm.config.CompactHistory.MaxHistoryLength != 5000 {
		t.Errorf("MaxHistoryLength = %d, want 5000", rlm.config.CompactHistory.MaxHistoryLength)
	}
}

func TestWithCompactHistoryConfigDefaults(t *testing.T) {
	client := &mockLLMClient{}
	replClient := &mockREPLClient{}

	// Test with zero MaxHistoryLength - should use default
	cfg := CompactHistoryConfig{
		MaxHistoryLength: 0,
		IncludeFewShot:   true,
	}

	rlm := New(client, replClient, WithCompactHistoryConfig(cfg))

	if rlm.config.CompactHistory.MaxHistoryLength != 10000 {
		t.Errorf("MaxHistoryLength = %d, want 10000 (default)", rlm.config.CompactHistory.MaxHistoryLength)
	}
}

func TestCompleteWithCompactHistory(t *testing.T) {
	client := &mockLLMClient{
		completeFunc: func(ctx context.Context, messages []core.Message) (core.LLMResponse, error) {
			return core.LLMResponse{Content: "FINAL(42)", PromptTokens: 10, CompletionTokens: 5}, nil
		},
	}
	replClient := &mockREPLClient{}

	rlm := New(client, replClient,
		WithCompactHistory(true),
		WithMaxIterations(5),
	)

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
}

func TestCompleteWithCompactHistoryMultipleIterations(t *testing.T) {
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

	rlm := New(client, replClient,
		WithCompactHistory(false), // Without few-shot for cleaner test
		WithMaxIterations(10),
	)

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

func TestTrimHistoryIfNeeded(t *testing.T) {
	tests := []struct {
		name    string
		history string
		maxLen  int
		check   func(result string) bool
	}{
		{
			name:    "short history not trimmed",
			history: "short",
			maxLen:  100,
			check:   func(result string) bool { return result == "short" },
		},
		{
			name:    "long history trimmed",
			history: strings.Repeat("a", 1000),
			maxLen:  100,
			check:   func(result string) bool { return len(result) < 1000 && strings.Contains(result, "[...earlier iterations truncated...]") },
		},
		{
			name:    "preserves iteration markers",
			history: "old content\n--- Iteration 1 ---\nmore content\n--- Iteration 2 ---\nrecent content",
			maxLen:  50,
			check:   func(result string) bool { return strings.Contains(result, "--- Iteration") },
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := trimHistoryIfNeeded(tt.history, tt.maxLen)
			if !tt.check(result) {
				t.Errorf("trimHistoryIfNeeded() = %q, unexpected result", result)
			}
		})
	}
}

func TestAppendIterationHistory(t *testing.T) {
	var history strings.Builder

	appendIterationHistory(&history, 1, "explore", "thinking about it", "fmt.Println(1)", "1\n")

	result := history.String()

	if !strings.Contains(result, "--- Iteration 1 ---") {
		t.Error("should contain iteration marker")
	}
	if !strings.Contains(result, "Action: explore") {
		t.Error("should contain action")
	}
	if !strings.Contains(result, "```go") {
		t.Error("should contain code block")
	}
	if !strings.Contains(result, "Output:") {
		t.Error("should contain output")
	}
}

func TestBuildCompactHistoryPrompt(t *testing.T) {
	client := &mockLLMClient{}
	replClient := &mockREPLClient{}

	rlm := New(client, replClient, WithCompactHistory(true))

	// Test first iteration
	prompt := rlm.buildCompactHistoryPrompt("string, 100 chars", "What is the answer?", "", 0)

	if !strings.Contains(prompt, "Context: string, 100 chars") {
		t.Error("should contain context info")
	}
	if !strings.Contains(prompt, "Query: What is the answer?") {
		t.Error("should contain query")
	}
	if !strings.Contains(prompt, "You have not explored the context yet") {
		t.Error("first iteration should have exploration prompt")
	}
	if !strings.Contains(prompt, "FEW-SHOT EXAMPLES") {
		t.Error("should include few-shot examples when enabled")
	}
}

func TestBuildCompactHistoryPromptWithHistory(t *testing.T) {
	client := &mockLLMClient{}
	replClient := &mockREPLClient{}

	rlm := New(client, replClient, WithCompactHistory(false)) // No few-shot for cleaner test

	history := "--- Iteration 1 ---\nAction: explore\nOutput: hello"
	prompt := rlm.buildCompactHistoryPrompt("string, 100 chars", "What is the answer?", history, 1)

	if !strings.Contains(prompt, "=== PREVIOUS ITERATIONS ===") {
		t.Error("should contain history section header")
	}
	if !strings.Contains(prompt, history) {
		t.Error("should contain the history")
	}
	if !strings.Contains(prompt, "Based on your previous exploration") {
		t.Error("subsequent iteration should have continuation prompt")
	}
}

// Tests for Few-Shot Examples

func TestIterationDemos(t *testing.T) {
	demos := IterationDemos()

	if len(demos) == 0 {
		t.Fatal("expected at least one demo example")
	}

	// Verify first demo has expected structure
	first := demos[0]
	if first.ContextInfo == "" {
		t.Error("first demo should have context info")
	}
	if first.Query == "" {
		t.Error("first demo should have query")
	}
	if first.Reasoning == "" {
		t.Error("first demo should have reasoning")
	}
	if first.Action == "" {
		t.Error("first demo should have action")
	}
}

func TestFormatFewShotExamples(t *testing.T) {
	result := FormatFewShotExamples()

	if !strings.Contains(result, "FEW-SHOT EXAMPLES") {
		t.Error("should contain header")
	}
	if !strings.Contains(result, "EXAMPLE 1:") {
		t.Error("should contain example 1")
	}
	if !strings.Contains(result, "My reasoning:") {
		t.Error("should contain reasoning")
	}
	if !strings.Contains(result, "```go") {
		t.Error("should contain code blocks")
	}
	if !strings.Contains(result, "FINAL(") {
		t.Error("should contain FINAL example")
	}
}

func TestFewShotExampleStructure(t *testing.T) {
	demos := IterationDemos()

	// Find the final answer example
	var hasFinalExample bool
	for _, demo := range demos {
		if demo.Answer != "" {
			hasFinalExample = true
			if demo.Action != "final" {
				t.Errorf("example with answer should have action 'final', got %q", demo.Action)
			}
		}
	}

	if !hasFinalExample {
		t.Error("should have at least one example showing FINAL answer")
	}
}
