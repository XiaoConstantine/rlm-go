package repl

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/XiaoConstantine/rlm-go/pkg/core"
)

// mockLLMClient implements LLMClient for testing
type mockLLMClient struct {
	mu            sync.Mutex
	queryFunc     func(ctx context.Context, prompt string) (QueryResponse, error)
	batchFunc     func(ctx context.Context, prompts []string) ([]QueryResponse, error)
	queryCalls    []string
	batchCalls    [][]string
}

func newMockClient() *mockLLMClient {
	return &mockLLMClient{
		queryFunc: func(ctx context.Context, prompt string) (QueryResponse, error) {
			return QueryResponse{Response: "mock response", PromptTokens: 10, CompletionTokens: 5}, nil
		},
		batchFunc: func(ctx context.Context, prompts []string) ([]QueryResponse, error) {
			results := make([]QueryResponse, len(prompts))
			for i := range prompts {
				results[i] = QueryResponse{Response: fmt.Sprintf("batch response %d", i), PromptTokens: 10, CompletionTokens: 5}
			}
			return results, nil
		},
	}
}

func (m *mockLLMClient) Query(ctx context.Context, prompt string) (QueryResponse, error) {
	m.mu.Lock()
	m.queryCalls = append(m.queryCalls, prompt)
	m.mu.Unlock()
	return m.queryFunc(ctx, prompt)
}

func (m *mockLLMClient) QueryBatched(ctx context.Context, prompts []string) ([]QueryResponse, error) {
	m.mu.Lock()
	m.batchCalls = append(m.batchCalls, prompts)
	m.mu.Unlock()
	return m.batchFunc(ctx, prompts)
}

func TestNew(t *testing.T) {
	client := newMockClient()
	repl := New(client)

	if repl == nil {
		t.Fatal("expected non-nil REPL")
	}
	if repl.interp == nil {
		t.Error("expected non-nil interpreter")
	}
	if repl.stdout == nil {
		t.Error("expected non-nil stdout buffer")
	}
	if repl.stderr == nil {
		t.Error("expected non-nil stderr buffer")
	}
}

func TestExecuteBasic(t *testing.T) {
	tests := []struct {
		name           string
		code           string
		expectedStdout string
		expectedStderr string
	}{
		{
			name:           "simple print",
			code:           `fmt.Println("hello")`,
			expectedStdout: "hello\n",
			expectedStderr: "",
		},
		{
			name:           "variable assignment and print",
			code:           "x := 42\nfmt.Println(x)",
			expectedStdout: "42\n",
			expectedStderr: "",
		},
		{
			name:           "string concatenation",
			code:           `fmt.Println("hello" + " " + "world")`,
			expectedStdout: "hello world\n",
			expectedStderr: "",
		},
		{
			name:           "arithmetic",
			code:           "fmt.Println(2 + 3 * 4)",
			expectedStdout: "14\n",
			expectedStderr: "",
		},
		{
			name:           "no output",
			code:           "x := 1",
			expectedStdout: "",
			expectedStderr: "",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client := newMockClient()
			repl := New(client)

			result, err := repl.Execute(context.Background(), tt.code)
			if err != nil {
				t.Fatalf("Execute() error: %v", err)
			}

			if result.Stdout != tt.expectedStdout {
				t.Errorf("Stdout = %q, want %q", result.Stdout, tt.expectedStdout)
			}
			// Note: stderr might contain additional info, so use Contains
			if tt.expectedStderr != "" && !strings.Contains(result.Stderr, tt.expectedStderr) {
				t.Errorf("Stderr = %q, want to contain %q", result.Stderr, tt.expectedStderr)
			}
		})
	}
}

func TestExecuteWithError(t *testing.T) {
	client := newMockClient()
	repl := New(client)

	// Execute code with undefined variable
	result, err := repl.Execute(context.Background(), "fmt.Println(undefinedVariable)")

	// The Execute method returns nil error, but stderr contains the error
	if err != nil {
		t.Fatalf("Execute() should return nil error, got: %v", err)
	}

	if result.Stderr == "" {
		t.Error("expected stderr to contain error message")
	}
}

func TestExecuteWithSyntaxError(t *testing.T) {
	client := newMockClient()
	repl := New(client)

	result, err := repl.Execute(context.Background(), "invalid go code {{{{")
	if err != nil {
		t.Fatalf("Execute() should return nil error, got: %v", err)
	}

	if result.Stderr == "" {
		t.Error("expected stderr to contain syntax error")
	}
}

func TestLoadContextString(t *testing.T) {
	client := newMockClient()
	repl := New(client)

	// Load string context
	err := repl.LoadContext("test context data")
	if err != nil {
		t.Fatalf("LoadContext() error: %v", err)
	}

	// Verify context is accessible
	result, err := repl.Execute(context.Background(), "fmt.Println(context)")
	if err != nil {
		t.Fatalf("Execute() error: %v", err)
	}

	if !strings.Contains(result.Stdout, "test context data") {
		t.Errorf("context not accessible, stdout = %q", result.Stdout)
	}
}

func TestLoadContextMap(t *testing.T) {
	client := newMockClient()
	repl := New(client)

	// Load map context
	ctx := map[string]any{
		"key1": "value1",
		"key2": 42,
	}
	err := repl.LoadContext(ctx)
	if err != nil {
		t.Fatalf("LoadContext() error: %v", err)
	}

	// Verify context type info
	info := repl.ContextInfo()
	if !strings.Contains(info, "map") {
		t.Errorf("ContextInfo() = %q, expected to contain 'map'", info)
	}
}

func TestLoadContextSlice(t *testing.T) {
	client := newMockClient()
	repl := New(client)

	// Load slice context
	ctx := []any{"item1", "item2", "item3"}
	err := repl.LoadContext(ctx)
	if err != nil {
		t.Fatalf("LoadContext() error: %v", err)
	}

	// Verify context type
	info := repl.ContextInfo()
	// The info might show the slice type
	if info == "context not loaded" {
		t.Error("expected context to be loaded")
	}
}

func TestGetVariable(t *testing.T) {
	client := newMockClient()
	repl := New(client)

	// Define a variable
	_, err := repl.Execute(context.Background(), `answer := "forty-two"`)
	if err != nil {
		t.Fatalf("Execute() error: %v", err)
	}

	// Get the variable
	value, err := repl.GetVariable("answer")
	if err != nil {
		t.Fatalf("GetVariable() error: %v", err)
	}

	if value != "forty-two" {
		t.Errorf("GetVariable() = %q, want %q", value, "forty-two")
	}
}

func TestGetVariableNotFound(t *testing.T) {
	client := newMockClient()
	repl := New(client)

	_, err := repl.GetVariable("nonexistent")
	if err == nil {
		t.Error("expected error for nonexistent variable")
	}
}

func TestGetVariableNumeric(t *testing.T) {
	client := newMockClient()
	repl := New(client)

	_, err := repl.Execute(context.Background(), `count := 42`)
	if err != nil {
		t.Fatalf("Execute() error: %v", err)
	}

	value, err := repl.GetVariable("count")
	if err != nil {
		t.Fatalf("GetVariable() error: %v", err)
	}

	if value != "42" {
		t.Errorf("GetVariable() = %q, want %q", value, "42")
	}
}

func TestLLMQuery(t *testing.T) {
	client := newMockClient()
	client.queryFunc = func(ctx context.Context, prompt string) (QueryResponse, error) {
		return QueryResponse{Response: "LLM says: " + prompt, PromptTokens: 10, CompletionTokens: 5}, nil
	}

	repl := New(client)

	// Execute code that calls Query
	result, err := repl.Execute(context.Background(), `
response := Query("What is 2+2?")
fmt.Println(response)
`)
	if err != nil {
		t.Fatalf("Execute() error: %v", err)
	}

	if !strings.Contains(result.Stdout, "LLM says: What is 2+2?") {
		t.Errorf("unexpected stdout: %q", result.Stdout)
	}

	// Check that the LLM call was recorded
	calls := repl.GetLLMCalls()
	if len(calls) != 1 {
		t.Errorf("expected 1 LLM call, got %d", len(calls))
	}
	if len(calls) > 0 && calls[0].Prompt != "What is 2+2?" {
		t.Errorf("unexpected prompt: %q", calls[0].Prompt)
	}
}

func TestLLMQueryBatched(t *testing.T) {
	client := newMockClient()
	client.batchFunc = func(ctx context.Context, prompts []string) ([]QueryResponse, error) {
		results := make([]QueryResponse, len(prompts))
		for i, p := range prompts {
			results[i] = QueryResponse{Response: "Response to: " + p, PromptTokens: 10, CompletionTokens: 5}
		}
		return results, nil
	}

	repl := New(client)

	result, err := repl.Execute(context.Background(), `
prompts := []string{"Q1", "Q2", "Q3"}
responses := QueryBatched(prompts)
for _, r := range responses {
	fmt.Println(r)
}
`)
	if err != nil {
		t.Fatalf("Execute() error: %v", err)
	}

	if !strings.Contains(result.Stdout, "Response to: Q1") {
		t.Errorf("missing Q1 response in stdout: %q", result.Stdout)
	}

	// Check LLM calls
	calls := repl.GetLLMCalls()
	if len(calls) != 3 {
		t.Errorf("expected 3 LLM calls, got %d", len(calls))
	}
}

func TestLLMQueryError(t *testing.T) {
	client := newMockClient()
	client.queryFunc = func(ctx context.Context, prompt string) (QueryResponse, error) {
		return QueryResponse{}, errors.New("LLM error")
	}

	repl := New(client)

	result, err := repl.Execute(context.Background(), `
response := Query("test")
fmt.Println(response)
`)
	if err != nil {
		t.Fatalf("Execute() error: %v", err)
	}

	if !strings.Contains(result.Stdout, "Error:") {
		t.Errorf("expected error in response, got: %q", result.Stdout)
	}
}

func TestLLMQueryBatchedError(t *testing.T) {
	client := newMockClient()
	client.batchFunc = func(ctx context.Context, prompts []string) ([]QueryResponse, error) {
		return nil, errors.New("batch error")
	}

	repl := New(client)

	result, err := repl.Execute(context.Background(), `
responses := QueryBatched([]string{"Q1", "Q2"})
fmt.Println(responses[0])
`)
	if err != nil {
		t.Fatalf("Execute() error: %v", err)
	}

	if !strings.Contains(result.Stdout, "Error:") {
		t.Errorf("expected error in response, got: %q", result.Stdout)
	}
}

func TestReset(t *testing.T) {
	client := newMockClient()
	repl := New(client)

	// Define a variable
	_, err := repl.Execute(context.Background(), `testVar := "hello"`)
	if err != nil {
		t.Fatalf("Execute() error: %v", err)
	}

	// Verify it exists
	_, err = repl.GetVariable("testVar")
	if err != nil {
		t.Fatalf("variable should exist before reset: %v", err)
	}

	// Reset
	err = repl.Reset()
	if err != nil {
		t.Fatalf("Reset() error: %v", err)
	}

	// Verify variable no longer exists
	_, err = repl.GetVariable("testVar")
	if err == nil {
		t.Error("variable should not exist after reset")
	}
}

func TestGetLocals(t *testing.T) {
	client := newMockClient()
	repl := New(client)

	// Define some standard variable names
	_, err := repl.Execute(context.Background(), `
result := "test result"
answer := "42"
count := 10
`)
	if err != nil {
		t.Fatalf("Execute() error: %v", err)
	}

	locals := repl.GetLocals()

	// Check that expected variables are present
	if _, ok := locals["result"]; !ok {
		t.Error("expected 'result' in locals")
	}
	if _, ok := locals["answer"]; !ok {
		t.Error("expected 'answer' in locals")
	}
	if _, ok := locals["count"]; !ok {
		t.Error("expected 'count' in locals")
	}
}

func TestContextInfo(t *testing.T) {
	tests := []struct {
		name     string
		context  any
		expected string
	}{
		{
			name:     "string context",
			context:  "test string",
			expected: "string",
		},
		{
			name:     "no context",
			context:  nil,
			expected: "context not loaded",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client := newMockClient()
			repl := New(client)

			if tt.context != nil {
				err := repl.LoadContext(tt.context)
				if err != nil {
					t.Fatalf("LoadContext() error: %v", err)
				}
			}

			info := repl.ContextInfo()
			if !strings.Contains(info, tt.expected) {
				t.Errorf("ContextInfo() = %q, expected to contain %q", info, tt.expected)
			}
		})
	}
}

func TestSetContext(t *testing.T) {
	client := newMockClient()
	repl := New(client)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	repl.SetContext(ctx)
	// Just verify it doesn't panic - the context is used internally
}

func TestClearLLMCalls(t *testing.T) {
	client := newMockClient()
	repl := New(client)

	// Make some LLM calls
	_, _ = repl.Execute(context.Background(), `Query("test")`)

	calls := repl.GetLLMCalls()
	if len(calls) == 0 {
		t.Skip("no calls recorded, skipping clear test")
	}

	// Clear and verify
	repl.ClearLLMCalls()
	calls = repl.GetLLMCalls()
	if len(calls) != 0 {
		t.Errorf("expected 0 calls after clear, got %d", len(calls))
	}
}

func TestFormatExecutionResult(t *testing.T) {
	tests := []struct {
		name     string
		result   *core.ExecutionResult
		expected string
	}{
		{
			name: "stdout only",
			result: &core.ExecutionResult{
				Stdout: "hello\n",
				Stderr: "",
			},
			expected: "hello\n",
		},
		{
			name: "stderr only",
			result: &core.ExecutionResult{
				Stdout: "",
				Stderr: "error occurred",
			},
			expected: "error occurred",
		},
		{
			name: "both stdout and stderr",
			result: &core.ExecutionResult{
				Stdout: "output",
				Stderr: "warning",
			},
			expected: "output\n\nwarning",
		},
		{
			name: "no output",
			result: &core.ExecutionResult{
				Stdout: "",
				Stderr: "",
			},
			expected: "No output",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := FormatExecutionResult(tt.result)
			if result != tt.expected {
				t.Errorf("FormatExecutionResult() = %q, want %q", result, tt.expected)
			}
		})
	}
}

func TestExecuteDuration(t *testing.T) {
	client := newMockClient()
	repl := New(client)

	result, err := repl.Execute(context.Background(), `
for i := 0; i < 1000; i++ {
	_ = i * i
}
`)
	if err != nil {
		t.Fatalf("Execute() error: %v", err)
	}

	if result.Duration <= 0 {
		t.Error("expected positive duration")
	}
}

func TestConcurrentExecution(t *testing.T) {
	// Note: Yaegi interpreter is not thread-safe for concurrent Eval calls
	// on the same instance. This test verifies that multiple REPL instances
	// can run concurrently without issues.
	var wg sync.WaitGroup
	errChan := make(chan error, 10)

	// Run multiple executions concurrently, each with its own REPL
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func(n int) {
			defer wg.Done()
			client := newMockClient()
			r := New(client)
			code := fmt.Sprintf(`fmt.Println(%d)`, n)
			result, err := r.Execute(context.Background(), code)
			if err != nil {
				errChan <- err
				return
			}
			expected := fmt.Sprintf("%d\n", n)
			if result.Stdout != expected {
				errChan <- fmt.Errorf("goroutine %d: got %q, want %q", n, result.Stdout, expected)
			}
		}(i)
	}

	wg.Wait()
	close(errChan)

	for err := range errChan {
		t.Errorf("concurrent execution error: %v", err)
	}
}

func TestLLMCallTracking(t *testing.T) {
	client := newMockClient()
	client.queryFunc = func(ctx context.Context, prompt string) (QueryResponse, error) {
		return QueryResponse{
			Response:         "response",
			PromptTokens:     100,
			CompletionTokens: 50,
		}, nil
	}

	repl := New(client)

	_, _ = repl.Execute(context.Background(), `Query("test prompt")`)

	calls := repl.GetLLMCalls()
	if len(calls) != 1 {
		t.Fatalf("expected 1 call, got %d", len(calls))
	}

	call := calls[0]
	if call.Prompt != "test prompt" {
		t.Errorf("Prompt = %q, want %q", call.Prompt, "test prompt")
	}
	if call.Response != "response" {
		t.Errorf("Response = %q, want %q", call.Response, "response")
	}
	if call.PromptTokens != 100 {
		t.Errorf("PromptTokens = %d, want %d", call.PromptTokens, 100)
	}
	if call.CompletionTokens != 50 {
		t.Errorf("CompletionTokens = %d, want %d", call.CompletionTokens, 50)
	}
	if call.Duration <= 0 {
		t.Error("expected positive duration")
	}
}

func TestNewREPLPool(t *testing.T) {
	client := newMockClient()

	// Test without pre-warming
	pool := NewREPLPool(client, 3, false)
	if pool == nil {
		t.Fatal("expected non-nil pool")
	}

	poolSize, created := pool.Stats()
	if poolSize != 0 {
		t.Errorf("expected empty pool without pre-warm, got %d", poolSize)
	}
	if created != 0 {
		t.Errorf("expected 0 created without pre-warm, got %d", created)
	}
}

func TestNewREPLPoolPreWarmed(t *testing.T) {
	client := newMockClient()

	// Test with pre-warming
	pool := NewREPLPool(client, 3, true)
	if pool == nil {
		t.Fatal("expected non-nil pool")
	}

	poolSize, created := pool.Stats()
	if poolSize != 3 {
		t.Errorf("expected pool size 3, got %d", poolSize)
	}
	if created != 3 {
		t.Errorf("expected 3 created with pre-warm, got %d", created)
	}
}

func TestREPLPoolGetPut(t *testing.T) {
	client := newMockClient()
	pool := NewREPLPool(client, 2, true)

	// Get first REPL
	r1 := pool.Get()
	if r1 == nil {
		t.Fatal("expected non-nil REPL from Get")
	}

	poolSize, _ := pool.Stats()
	if poolSize != 1 {
		t.Errorf("expected pool size 1 after Get, got %d", poolSize)
	}

	// Execute something to verify it works
	result, err := r1.Execute(context.Background(), `fmt.Println("hello")`)
	if err != nil {
		t.Fatalf("Execute error: %v", err)
	}
	if !strings.Contains(result.Stdout, "hello") {
		t.Errorf("expected hello in stdout, got %q", result.Stdout)
	}

	// Put it back
	pool.Put(r1)

	// Pool should have 2 again (Put creates a fresh REPL)
	poolSize, _ = pool.Stats()
	if poolSize != 2 {
		t.Errorf("expected pool size 2 after Put, got %d", poolSize)
	}
}

func TestREPLPoolExhausted(t *testing.T) {
	client := newMockClient()
	pool := NewREPLPool(client, 1, true)

	// Exhaust the pool
	r1 := pool.Get()
	if r1 == nil {
		t.Fatal("expected non-nil REPL")
	}

	// Get another - should create new one
	r2 := pool.Get()
	if r2 == nil {
		t.Fatal("expected non-nil REPL even when pool exhausted")
	}

	// Both should work
	result1, _ := r1.Execute(context.Background(), `fmt.Println(1)`)
	result2, _ := r2.Execute(context.Background(), `fmt.Println(2)`)

	if !strings.Contains(result1.Stdout, "1") {
		t.Errorf("r1 stdout wrong: %q", result1.Stdout)
	}
	if !strings.Contains(result2.Stdout, "2") {
		t.Errorf("r2 stdout wrong: %q", result2.Stdout)
	}
}

func TestREPLClose(t *testing.T) {
	client := newMockClient()
	repl := New(client)

	// Execute something
	_, _ = repl.Execute(context.Background(), `Query("test")`)

	// Should have calls
	calls := repl.GetLLMCalls()
	if len(calls) == 0 {
		t.Skip("no calls to verify close clears them")
	}

	// Close should clear state
	repl.Close()

	// GetLLMCalls should return empty now
	calls = repl.GetLLMCalls()
	if len(calls) != 0 {
		t.Errorf("expected 0 calls after Close, got %d", len(calls))
	}
}

func TestREPLResetState(t *testing.T) {
	client := newMockClient()
	repl := New(client)

	// Execute something to put data in buffers
	_, _ = repl.Execute(context.Background(), `
fmt.Println("test output")
Query("test prompt")
`)

	// Reset state
	repl.resetState()

	// Buffers should be clear
	result, _ := repl.Execute(context.Background(), `// no-op`)
	if result.Stdout != "" {
		t.Errorf("expected empty stdout after resetState, got %q", result.Stdout)
	}

	// LLM calls should be clear
	calls := repl.GetLLMCalls()
	if len(calls) != 0 {
		t.Errorf("expected 0 calls after resetState, got %d", len(calls))
	}
}

func TestNewPooled(t *testing.T) {
	client := newMockClient()
	repl := NewPooled(client)

	if repl == nil {
		t.Fatal("expected non-nil REPL from NewPooled")
	}

	// Should work like regular REPL
	result, err := repl.Execute(context.Background(), `fmt.Println("pooled")`)
	if err != nil {
		t.Fatalf("Execute error: %v", err)
	}
	if !strings.Contains(result.Stdout, "pooled") {
		t.Errorf("expected pooled in stdout, got %q", result.Stdout)
	}
}

func TestWithPoolingOption(t *testing.T) {
	client := newMockClient()
	repl := New(client, WithPooling(true))

	if !repl.usePooling {
		t.Error("expected usePooling to be true")
	}
}
