package repl

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/XiaoConstantine/rlm-go/pkg/core"
)

// mockRecursiveLLMClient implements RecursiveLLMClient for testing
type mockRecursiveLLMClient struct {
	mu                sync.Mutex
	queryFunc         func(ctx context.Context, prompt string) (QueryResponse, error)
	batchFunc         func(ctx context.Context, prompts []string) ([]QueryResponse, error)
	queryWithRLMFunc  func(ctx context.Context, prompt string, depth int) (QueryResponse, error)
	currentDepthValue int
	maxDepthValue     int
	queryCalls        []string
	batchCalls        [][]string
	recursiveCalls    []recursiveCallRecord
}

type recursiveCallRecord struct {
	prompt string
	depth  int
}

func newMockRecursiveClient() *mockRecursiveLLMClient {
	return &mockRecursiveLLMClient{
		currentDepthValue: 0,
		maxDepthValue:     2,
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
		queryWithRLMFunc: func(ctx context.Context, prompt string, depth int) (QueryResponse, error) {
			return QueryResponse{
				Response:         fmt.Sprintf("recursive response at depth %d: %s", depth, prompt),
				PromptTokens:     20,
				CompletionTokens: 10,
			}, nil
		},
	}
}

func (m *mockRecursiveLLMClient) Query(ctx context.Context, prompt string) (QueryResponse, error) {
	m.mu.Lock()
	m.queryCalls = append(m.queryCalls, prompt)
	m.mu.Unlock()
	return m.queryFunc(ctx, prompt)
}

func (m *mockRecursiveLLMClient) QueryBatched(ctx context.Context, prompts []string) ([]QueryResponse, error) {
	m.mu.Lock()
	m.batchCalls = append(m.batchCalls, prompts)
	m.mu.Unlock()
	return m.batchFunc(ctx, prompts)
}

func (m *mockRecursiveLLMClient) QueryWithRLM(ctx context.Context, prompt string, depth int) (QueryResponse, error) {
	m.mu.Lock()
	m.recursiveCalls = append(m.recursiveCalls, recursiveCallRecord{prompt: prompt, depth: depth})
	m.mu.Unlock()
	return m.queryWithRLMFunc(ctx, prompt, depth)
}

func (m *mockRecursiveLLMClient) CurrentDepth() int {
	return m.currentDepthValue
}

func (m *mockRecursiveLLMClient) MaxDepth() int {
	return m.maxDepthValue
}

func TestNewRecursiveREPL(t *testing.T) {
	client := newMockRecursiveClient()
	recursionCtx := core.NewRecursionContext(2)
	repl := NewRecursiveREPL(client, recursionCtx)

	if repl == nil {
		t.Fatal("expected non-nil RecursiveREPL")
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
	if repl.client != client {
		t.Error("expected client to be set")
	}
	if repl.recursionContext != recursionCtx {
		t.Error("expected recursion context to be set")
	}
}

func TestRecursiveREPL_ExecuteBasic(t *testing.T) {
	client := newMockRecursiveClient()
	recursionCtx := core.NewRecursionContext(2)
	repl := NewRecursiveREPL(client, recursionCtx)

	tests := []struct {
		name           string
		code           string
		expectedStdout string
	}{
		{
			name:           "simple print",
			code:           `fmt.Println("hello")`,
			expectedStdout: "hello\n",
		},
		{
			name:           "variable assignment and print",
			code:           "x := 42\nfmt.Println(x)",
			expectedStdout: "42\n",
		},
		{
			name:           "use min function",
			code:           "fmt.Println(min(5, 3))",
			expectedStdout: "3\n",
		},
		{
			name:           "use max function",
			code:           "fmt.Println(max(5, 3))",
			expectedStdout: "5\n",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// Reset REPL for each test
			repl.Reset()

			result, err := repl.Execute(context.Background(), tt.code)
			if err != nil {
				t.Fatalf("Execute() error: %v", err)
			}

			if result.Stdout != tt.expectedStdout {
				t.Errorf("Stdout = %q, want %q", result.Stdout, tt.expectedStdout)
			}
		})
	}
}

func TestRecursiveREPL_CurrentDepth(t *testing.T) {
	client := newMockRecursiveClient()
	client.currentDepthValue = 1
	recursionCtx := core.NewRecursionContext(3)
	repl := NewRecursiveREPL(client, recursionCtx)

	result, err := repl.Execute(context.Background(), `
depth := CurrentDepth()
fmt.Println(depth)
`)
	if err != nil {
		t.Fatalf("Execute() error: %v", err)
	}

	if !strings.Contains(result.Stdout, "1") {
		t.Errorf("expected current depth 1, got stdout: %q", result.Stdout)
	}
}

func TestRecursiveREPL_MaxDepth(t *testing.T) {
	client := newMockRecursiveClient()
	client.maxDepthValue = 5
	recursionCtx := core.NewRecursionContext(5)
	repl := NewRecursiveREPL(client, recursionCtx)

	result, err := repl.Execute(context.Background(), `
maxD := MaxDepth()
fmt.Println(maxD)
`)
	if err != nil {
		t.Fatalf("Execute() error: %v", err)
	}

	if !strings.Contains(result.Stdout, "5") {
		t.Errorf("expected max depth 5, got stdout: %q", result.Stdout)
	}
}

func TestRecursiveREPL_CanRecurse(t *testing.T) {
	tests := []struct {
		name         string
		currentDepth int
		maxDepth     int
		expected     string
	}{
		{
			name:         "can recurse",
			currentDepth: 0,
			maxDepth:     2,
			expected:     "true",
		},
		{
			name:         "can recurse at mid depth",
			currentDepth: 1,
			maxDepth:     3,
			expected:     "true",
		},
		{
			name:         "cannot recurse at max",
			currentDepth: 2,
			maxDepth:     2,
			expected:     "false",
		},
		{
			name:         "cannot recurse past max",
			currentDepth: 3,
			maxDepth:     2,
			expected:     "false",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			client := newMockRecursiveClient()
			client.currentDepthValue = tt.currentDepth
			client.maxDepthValue = tt.maxDepth
			recursionCtx := core.NewRecursionContext(tt.maxDepth)
			repl := NewRecursiveREPL(client, recursionCtx)

			result, err := repl.Execute(context.Background(), `
canR := CanRecurse()
fmt.Println(canR)
`)
			if err != nil {
				t.Fatalf("Execute() error: %v", err)
			}

			if !strings.Contains(result.Stdout, tt.expected) {
				t.Errorf("expected %s, got stdout: %q", tt.expected, result.Stdout)
			}
		})
	}
}

func TestRecursiveREPL_QueryWithRLM(t *testing.T) {
	client := newMockRecursiveClient()
	client.currentDepthValue = 0
	client.maxDepthValue = 3
	recursionCtx := core.NewRecursionContext(3)
	repl := NewRecursiveREPL(client, recursionCtx)

	result, err := repl.Execute(context.Background(), `
response := QueryWithRLM("analyze this data", 1)
fmt.Println(response)
`)
	if err != nil {
		t.Fatalf("Execute() error: %v", err)
	}

	if !strings.Contains(result.Stdout, "recursive response at depth 1") {
		t.Errorf("unexpected stdout: %q", result.Stdout)
	}

	// Verify the recursive call was recorded
	if len(client.recursiveCalls) != 1 {
		t.Errorf("expected 1 recursive call, got %d", len(client.recursiveCalls))
	}
	if len(client.recursiveCalls) > 0 {
		call := client.recursiveCalls[0]
		if call.prompt != "analyze this data" {
			t.Errorf("unexpected prompt: %q", call.prompt)
		}
		if call.depth != 1 {
			t.Errorf("unexpected depth: %d", call.depth)
		}
	}

	// Verify recursive calls are tracked
	recursiveCalls := repl.GetRecursiveCalls()
	if len(recursiveCalls) != 1 {
		t.Errorf("expected 1 tracked recursive call, got %d", len(recursiveCalls))
	}
	if len(recursiveCalls) > 0 {
		call := recursiveCalls[0]
		if call.Depth != 1 {
			t.Errorf("tracked call depth = %d, want 1", call.Depth)
		}
		if call.PromptTokens != 20 {
			t.Errorf("tracked call PromptTokens = %d, want 20", call.PromptTokens)
		}
		if call.CompletionTokens != 10 {
			t.Errorf("tracked call CompletionTokens = %d, want 10", call.CompletionTokens)
		}
	}
}

func TestRecursiveREPL_QueryWithRLMDefaultDepth(t *testing.T) {
	client := newMockRecursiveClient()
	client.currentDepthValue = 1
	client.maxDepthValue = 3
	recursionCtx := core.NewRecursionContext(3)
	repl := NewRecursiveREPL(client, recursionCtx)

	// When depth is 0 or negative, it defaults to currentDepth + 1
	result, err := repl.Execute(context.Background(), `
response := QueryWithRLM("test prompt", 0)
fmt.Println(response)
`)
	if err != nil {
		t.Fatalf("Execute() error: %v", err)
	}

	// Should use depth 2 (currentDepth 1 + 1)
	if !strings.Contains(result.Stdout, "recursive response at depth 2") {
		t.Errorf("expected depth 2 response, got stdout: %q", result.Stdout)
	}
}

func TestRecursiveREPL_QueryBatchedWithRLM(t *testing.T) {
	client := newMockRecursiveClient()
	client.currentDepthValue = 0
	client.maxDepthValue = 3
	recursionCtx := core.NewRecursionContext(3)
	repl := NewRecursiveREPL(client, recursionCtx)

	result, err := repl.Execute(context.Background(), `
prompts := []string{"prompt1", "prompt2", "prompt3"}
responses := QueryBatchedWithRLM(prompts, 1)
for i, r := range responses {
	fmt.Printf("Response %d: %s\n", i, r)
}
`)
	if err != nil {
		t.Fatalf("Execute() error: %v", err)
	}

	// Verify all responses are present
	if !strings.Contains(result.Stdout, "Response 0:") {
		t.Error("missing response 0")
	}
	if !strings.Contains(result.Stdout, "Response 1:") {
		t.Error("missing response 1")
	}
	if !strings.Contains(result.Stdout, "Response 2:") {
		t.Error("missing response 2")
	}

	// Verify 3 recursive calls were made
	if len(client.recursiveCalls) != 3 {
		t.Errorf("expected 3 recursive calls, got %d", len(client.recursiveCalls))
	}

	// Verify all recursive calls are tracked
	recursiveCalls := repl.GetRecursiveCalls()
	if len(recursiveCalls) != 3 {
		t.Errorf("expected 3 tracked recursive calls, got %d", len(recursiveCalls))
	}
}

func TestRecursiveREPL_QueryWithRLMError(t *testing.T) {
	client := newMockRecursiveClient()
	client.queryWithRLMFunc = func(ctx context.Context, prompt string, depth int) (QueryResponse, error) {
		return QueryResponse{}, fmt.Errorf("simulated RLM error")
	}
	recursionCtx := core.NewRecursionContext(2)
	repl := NewRecursiveREPL(client, recursionCtx)

	result, err := repl.Execute(context.Background(), `
response := QueryWithRLM("test", 1)
fmt.Println(response)
`)
	if err != nil {
		t.Fatalf("Execute() error: %v", err)
	}

	// Error should be captured in the response
	if !strings.Contains(result.Stdout, "Error:") {
		t.Errorf("expected error in response, got: %q", result.Stdout)
	}
}

func TestRecursiveREPL_LoadContext(t *testing.T) {
	client := newMockRecursiveClient()
	recursionCtx := core.NewRecursionContext(2)
	repl := NewRecursiveREPL(client, recursionCtx)

	err := repl.LoadContext("test context data")
	if err != nil {
		t.Fatalf("LoadContext() error: %v", err)
	}

	result, err := repl.Execute(context.Background(), "fmt.Println(context)")
	if err != nil {
		t.Fatalf("Execute() error: %v", err)
	}

	if !strings.Contains(result.Stdout, "test context data") {
		t.Errorf("context not accessible, stdout = %q", result.Stdout)
	}
}

func TestRecursiveREPL_GetVariable(t *testing.T) {
	client := newMockRecursiveClient()
	recursionCtx := core.NewRecursionContext(2)
	repl := NewRecursiveREPL(client, recursionCtx)

	_, err := repl.Execute(context.Background(), `answer := "forty-two"`)
	if err != nil {
		t.Fatalf("Execute() error: %v", err)
	}

	value, err := repl.GetVariable("answer")
	if err != nil {
		t.Fatalf("GetVariable() error: %v", err)
	}

	if value != "forty-two" {
		t.Errorf("GetVariable() = %q, want %q", value, "forty-two")
	}
}

func TestRecursiveREPL_GetLocals(t *testing.T) {
	client := newMockRecursiveClient()
	recursionCtx := core.NewRecursionContext(2)
	repl := NewRecursiveREPL(client, recursionCtx)

	_, err := repl.Execute(context.Background(), `
result := "test result"
answer := "42"
count := 10
`)
	if err != nil {
		t.Fatalf("Execute() error: %v", err)
	}

	locals := repl.GetLocals()

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

func TestRecursiveREPL_ContextInfo(t *testing.T) {
	client := newMockRecursiveClient()
	recursionCtx := core.NewRecursionContext(2)
	repl := NewRecursiveREPL(client, recursionCtx)

	// Before loading context
	info := repl.ContextInfo()
	if info != "context not loaded" {
		t.Errorf("expected 'context not loaded', got %q", info)
	}

	// After loading string context
	err := repl.LoadContext("test string")
	if err != nil {
		t.Fatalf("LoadContext() error: %v", err)
	}

	info = repl.ContextInfo()
	if !strings.Contains(info, "string") {
		t.Errorf("expected info to contain 'string', got %q", info)
	}
}

func TestRecursiveREPL_Reset(t *testing.T) {
	client := newMockRecursiveClient()
	recursionCtx := core.NewRecursionContext(2)
	repl := NewRecursiveREPL(client, recursionCtx)

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

func TestRecursiveREPL_Close(t *testing.T) {
	client := newMockRecursiveClient()
	recursionCtx := core.NewRecursionContext(2)
	repl := NewRecursiveREPL(client, recursionCtx)

	// Execute some code
	_, _ = repl.Execute(context.Background(), `Query("test")`)
	_, _ = repl.Execute(context.Background(), `QueryWithRLM("test", 1)`)

	// Should have calls
	llmCalls := repl.GetLLMCalls()
	if len(llmCalls) == 0 {
		t.Skip("no LLM calls to verify close clears them")
	}

	// Close should clear state
	repl.Close()

	// GetLLMCalls should return empty now
	llmCalls = repl.GetLLMCalls()
	if len(llmCalls) != 0 {
		t.Errorf("expected 0 LLM calls after Close, got %d", len(llmCalls))
	}

	recursiveCalls := repl.GetRecursiveCalls()
	if len(recursiveCalls) != 0 {
		t.Errorf("expected 0 recursive calls after Close, got %d", len(recursiveCalls))
	}
}

func TestRecursiveREPL_SetContext(t *testing.T) {
	client := newMockRecursiveClient()
	recursionCtx := core.NewRecursionContext(2)
	repl := NewRecursiveREPL(client, recursionCtx)

	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	repl.SetContext(ctx)
	// Verify it doesn't panic
}

func TestRecursiveREPL_StandardQuery(t *testing.T) {
	client := newMockRecursiveClient()
	client.queryFunc = func(ctx context.Context, prompt string) (QueryResponse, error) {
		return QueryResponse{Response: "standard LLM response: " + prompt, PromptTokens: 15, CompletionTokens: 8}, nil
	}
	recursionCtx := core.NewRecursionContext(2)
	repl := NewRecursiveREPL(client, recursionCtx)

	result, err := repl.Execute(context.Background(), `
response := Query("What is 2+2?")
fmt.Println(response)
`)
	if err != nil {
		t.Fatalf("Execute() error: %v", err)
	}

	if !strings.Contains(result.Stdout, "standard LLM response: What is 2+2?") {
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

func TestRecursiveREPL_QueryBatched(t *testing.T) {
	client := newMockRecursiveClient()
	client.batchFunc = func(ctx context.Context, prompts []string) ([]QueryResponse, error) {
		results := make([]QueryResponse, len(prompts))
		for i, p := range prompts {
			results[i] = QueryResponse{Response: "Response to: " + p, PromptTokens: 10, CompletionTokens: 5}
		}
		return results, nil
	}
	recursionCtx := core.NewRecursionContext(2)
	repl := NewRecursiveREPL(client, recursionCtx)

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

	calls := repl.GetLLMCalls()
	if len(calls) != 3 {
		t.Errorf("expected 3 LLM calls, got %d", len(calls))
	}
}

func TestRecursiveREPL_AsyncQuery(t *testing.T) {
	client := newMockRecursiveClient()
	client.queryFunc = func(ctx context.Context, prompt string) (QueryResponse, error) {
		time.Sleep(10 * time.Millisecond)
		return QueryResponse{Response: "async result: " + prompt, PromptTokens: 10, CompletionTokens: 5}, nil
	}
	recursionCtx := core.NewRecursionContext(2)
	repl := NewRecursiveREPL(client, recursionCtx)

	result, err := repl.Execute(context.Background(), `
handleID := QueryAsync("async prompt")
fmt.Println("started:", handleID != "")
result := WaitAsync(handleID)
fmt.Println("result:", result)
`)
	if err != nil {
		t.Fatalf("Execute() error: %v", err)
	}

	if result.Stderr != "" {
		t.Errorf("unexpected stderr: %s", result.Stderr)
	}

	if !strings.Contains(result.Stdout, "started: true") {
		t.Errorf("expected 'started: true' in stdout: %q", result.Stdout)
	}
	if !strings.Contains(result.Stdout, "async result:") {
		t.Errorf("expected async result in stdout: %q", result.Stdout)
	}
}

func TestRecursiveREPL_ExecuteDuration(t *testing.T) {
	client := newMockRecursiveClient()
	recursionCtx := core.NewRecursionContext(2)
	repl := NewRecursiveREPL(client, recursionCtx)

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

func TestRecursiveREPL_ConcurrentREPLs(t *testing.T) {
	// Test that multiple RecursiveREPL instances can run concurrently
	var wg sync.WaitGroup
	errChan := make(chan error, 10)

	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func(n int) {
			defer wg.Done()
			client := newMockRecursiveClient()
			recursionCtx := core.NewRecursionContext(2)
			r := NewRecursiveREPL(client, recursionCtx)
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

func TestRecursiveREPL_RecursiveCallTokenAggregation(t *testing.T) {
	client := newMockRecursiveClient()
	client.queryWithRLMFunc = func(ctx context.Context, prompt string, depth int) (QueryResponse, error) {
		return QueryResponse{
			Response:         fmt.Sprintf("response at depth %d", depth),
			PromptTokens:     100,
			CompletionTokens: 50,
		}, nil
	}
	recursionCtx := core.NewRecursionContext(3)
	repl := NewRecursiveREPL(client, recursionCtx)

	// Make multiple recursive calls
	_, err := repl.Execute(context.Background(), `
r1 := QueryWithRLM("prompt1", 1)
r2 := QueryWithRLM("prompt2", 2)
fmt.Println(r1)
fmt.Println(r2)
`)
	if err != nil {
		t.Fatalf("Execute() error: %v", err)
	}

	recursiveCalls := repl.GetRecursiveCalls()
	if len(recursiveCalls) != 2 {
		t.Fatalf("expected 2 recursive calls, got %d", len(recursiveCalls))
	}

	// Verify token counts
	totalPromptTokens := 0
	totalCompletionTokens := 0
	for _, call := range recursiveCalls {
		totalPromptTokens += call.PromptTokens
		totalCompletionTokens += call.CompletionTokens
	}

	if totalPromptTokens != 200 {
		t.Errorf("total prompt tokens = %d, want 200", totalPromptTokens)
	}
	if totalCompletionTokens != 100 {
		t.Errorf("total completion tokens = %d, want 100", totalCompletionTokens)
	}
}

func TestRecursiveCall_Fields(t *testing.T) {
	call := RecursiveCall{
		Prompt:           "test prompt",
		Response:         "test response",
		Depth:            2,
		Duration:         100 * time.Millisecond,
		PromptTokens:     50,
		CompletionTokens: 25,
	}

	if call.Prompt != "test prompt" {
		t.Errorf("Prompt = %q, want %q", call.Prompt, "test prompt")
	}
	if call.Response != "test response" {
		t.Errorf("Response = %q, want %q", call.Response, "test response")
	}
	if call.Depth != 2 {
		t.Errorf("Depth = %d, want 2", call.Depth)
	}
	if call.Duration != 100*time.Millisecond {
		t.Errorf("Duration = %v, want 100ms", call.Duration)
	}
	if call.PromptTokens != 50 {
		t.Errorf("PromptTokens = %d, want 50", call.PromptTokens)
	}
	if call.CompletionTokens != 25 {
		t.Errorf("CompletionTokens = %d, want 25", call.CompletionTokens)
	}
}
