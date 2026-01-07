package repl

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestQueryAsync_Basic(t *testing.T) {
	client := newMockClient()
	client.queryFunc = func(ctx context.Context, prompt string) (QueryResponse, error) {
		time.Sleep(10 * time.Millisecond)
		return QueryResponse{
			Response:         "async response for: " + prompt,
			PromptTokens:     10,
			CompletionTokens: 5,
		}, nil
	}

	r := New(client)

	handle := r.QueryAsync("test prompt")
	if handle == nil {
		t.Fatal("expected non-nil handle")
	}

	if handle.ID() == "" {
		t.Error("expected non-empty handle ID")
	}

	if handle.Ready() {
		t.Error("expected handle not to be ready immediately")
	}

	result, err := handle.Wait()
	if err != nil {
		t.Fatalf("Wait() error: %v", err)
	}

	if result != "async response for: test prompt" {
		t.Errorf("result = %q, want %q", result, "async response for: test prompt")
	}

	if !handle.Ready() {
		t.Error("expected handle to be ready after Wait()")
	}

	calls := r.GetLLMCalls()
	if len(calls) != 1 {
		t.Fatalf("expected 1 LLM call, got %d", len(calls))
	}
	if !calls[0].Async {
		t.Error("expected call to be marked as async")
	}
}

func TestQueryAsync_WaitMultiple(t *testing.T) {
	var callCount int32
	client := newMockClient()
	client.queryFunc = func(ctx context.Context, prompt string) (QueryResponse, error) {
		atomic.AddInt32(&callCount, 1)
		time.Sleep(20 * time.Millisecond)
		return QueryResponse{
			Response:         "response for: " + prompt,
			PromptTokens:     10,
			CompletionTokens: 5,
		}, nil
	}

	r := New(client)

	h1 := r.QueryAsync("prompt 1")
	h2 := r.QueryAsync("prompt 2")
	h3 := r.QueryAsync("prompt 3")

	if h1.Ready() || h2.Ready() || h3.Ready() {
		t.Error("expected handles not to be ready immediately")
	}

	start := time.Now()

	r1, err1 := h1.Wait()
	r2, err2 := h2.Wait()
	r3, err3 := h3.Wait()

	elapsed := time.Since(start)

	if err1 != nil || err2 != nil || err3 != nil {
		t.Errorf("Wait errors: %v, %v, %v", err1, err2, err3)
	}

	if r1 != "response for: prompt 1" {
		t.Errorf("r1 = %q, want %q", r1, "response for: prompt 1")
	}
	if r2 != "response for: prompt 2" {
		t.Errorf("r2 = %q, want %q", r2, "response for: prompt 2")
	}
	if r3 != "response for: prompt 3" {
		t.Errorf("r3 = %q, want %q", r3, "response for: prompt 3")
	}

	if elapsed > 100*time.Millisecond {
		t.Errorf("expected parallel execution, but took %v", elapsed)
	}

	if atomic.LoadInt32(&callCount) != 3 {
		t.Errorf("expected 3 LLM calls, got %d", atomic.LoadInt32(&callCount))
	}

	calls := r.GetLLMCalls()
	if len(calls) != 3 {
		t.Errorf("expected 3 tracked calls, got %d", len(calls))
	}
	for _, call := range calls {
		if !call.Async {
			t.Error("expected all calls to be marked as async")
		}
	}
}

func TestQueryBatchedAsync(t *testing.T) {
	var callCount int32
	client := newMockClient()
	client.queryFunc = func(ctx context.Context, prompt string) (QueryResponse, error) {
		atomic.AddInt32(&callCount, 1)
		time.Sleep(15 * time.Millisecond)
		return QueryResponse{
			Response:         "batch response: " + prompt,
			PromptTokens:     10,
			CompletionTokens: 5,
		}, nil
	}

	r := New(client)

	prompts := []string{"prompt A", "prompt B", "prompt C"}
	batchHandle := r.QueryBatchedAsync(prompts)

	if batchHandle == nil {
		t.Fatal("expected non-nil batch handle")
	}

	if batchHandle.TotalCount() != 3 {
		t.Errorf("TotalCount = %d, want 3", batchHandle.TotalCount())
	}

	if batchHandle.Ready() {
		t.Error("expected batch not to be ready immediately")
	}

	start := time.Now()
	results, err := batchHandle.WaitAll()
	elapsed := time.Since(start)

	if err != nil {
		t.Fatalf("WaitAll() error: %v", err)
	}

	if len(results) != 3 {
		t.Fatalf("expected 3 results, got %d", len(results))
	}

	for i, r := range results {
		expected := "batch response: " + prompts[i]
		if r != expected {
			t.Errorf("result[%d] = %q, want %q", i, r, expected)
		}
	}

	if elapsed > 100*time.Millisecond {
		t.Errorf("expected parallel execution, but took %v", elapsed)
	}

	if !batchHandle.Ready() {
		t.Error("expected batch to be ready after WaitAll()")
	}

	if batchHandle.CompletedCount() != 3 {
		t.Errorf("CompletedCount = %d, want 3", batchHandle.CompletedCount())
	}
}

func TestAsyncQueryHandle_Ready(t *testing.T) {
	client := newMockClient()
	completeCh := make(chan struct{})
	client.queryFunc = func(ctx context.Context, prompt string) (QueryResponse, error) {
		<-completeCh
		return QueryResponse{Response: "done"}, nil
	}

	r := New(client)

	handle := r.QueryAsync("test")

	time.Sleep(10 * time.Millisecond)
	if handle.Ready() {
		t.Error("expected handle not to be ready before query completes")
	}

	result, ok := handle.Result()
	if ok {
		t.Error("Result() should return false when not ready")
	}
	if result != "" {
		t.Error("Result() should return empty string when not ready")
	}

	close(completeCh)

	finalResult, err := handle.Wait()
	if err != nil {
		t.Fatalf("Wait() error: %v", err)
	}

	if finalResult != "done" {
		t.Errorf("final result = %q, want %q", finalResult, "done")
	}

	if !handle.Ready() {
		t.Error("expected handle to be ready after Wait()")
	}

	resultAfter, ok := handle.Result()
	if !ok {
		t.Error("Result() should return true after completion")
	}
	if resultAfter != "done" {
		t.Errorf("Result() = %q, want %q", resultAfter, "done")
	}
}

func TestAsyncQueryHandle_Error(t *testing.T) {
	client := newMockClient()
	expectedErr := errors.New("simulated LLM error")
	client.queryFunc = func(ctx context.Context, prompt string) (QueryResponse, error) {
		return QueryResponse{}, expectedErr
	}

	r := New(client)

	handle := r.QueryAsync("test")

	_, err := handle.Wait()
	if err == nil {
		t.Error("expected error from Wait()")
	}
	if err != expectedErr {
		t.Errorf("error = %v, want %v", err, expectedErr)
	}

	if handle.Error() != expectedErr {
		t.Errorf("Error() = %v, want %v", handle.Error(), expectedErr)
	}
}

func TestAsyncQueryFromInterpreter(t *testing.T) {
	client := newMockClient()
	client.queryFunc = func(ctx context.Context, prompt string) (QueryResponse, error) {
		time.Sleep(10 * time.Millisecond)
		return QueryResponse{Response: "interpreted async: " + prompt}, nil
	}

	r := New(client)

	code := `
handleID := QueryAsync("from interpreter")
fmt.Println("started:", handleID != "")
result := WaitAsync(handleID)
fmt.Println("result:", result)
`
	result, err := r.Execute(context.Background(), code)
	if err != nil {
		t.Fatalf("Execute error: %v", err)
	}

	if result.Stderr != "" {
		t.Errorf("unexpected stderr: %s", result.Stderr)
	}

	expectedStdout := "started: true\nresult: interpreted async: from interpreter\n"
	if result.Stdout != expectedStdout {
		t.Errorf("stdout = %q, want %q", result.Stdout, expectedStdout)
	}
}

func TestAsyncReadyFromInterpreter(t *testing.T) {
	client := newMockClient()
	completeCh := make(chan struct{})
	client.queryFunc = func(ctx context.Context, prompt string) (QueryResponse, error) {
		<-completeCh
		return QueryResponse{Response: "done"}, nil
	}

	r := New(client)

	code := `
handleID := QueryAsync("test")
ready := AsyncReady(handleID)
fmt.Println("ready before:", ready)
`
	result, err := r.Execute(context.Background(), code)
	if err != nil {
		t.Fatalf("Execute error: %v", err)
	}

	if result.Stderr != "" {
		t.Errorf("unexpected stderr: %s", result.Stderr)
	}

	if result.Stdout != "ready before: false\n" {
		t.Errorf("stdout = %q, want %q", result.Stdout, "ready before: false\n")
	}

	close(completeCh)

	r.WaitAllAsyncQueries()
}

func TestBatchedAsyncFromInterpreter(t *testing.T) {
	client := newMockClient()
	client.queryFunc = func(ctx context.Context, prompt string) (QueryResponse, error) {
		time.Sleep(5 * time.Millisecond)
		return QueryResponse{Response: "resp: " + prompt}, nil
	}

	r := New(client)

	code := `
prompts := []string{"a", "b", "c"}
handleIDs := QueryBatchedAsync(prompts)
fmt.Println("count:", len(handleIDs))
for i, id := range handleIDs {
    result := WaitAsync(id)
    fmt.Printf("result %d: %s\n", i, result)
}
`
	result, err := r.Execute(context.Background(), code)
	if err != nil {
		t.Fatalf("Execute error: %v", err)
	}

	if result.Stderr != "" {
		t.Errorf("unexpected stderr: %s", result.Stderr)
	}

	expected := "count: 3\nresult 0: resp: a\nresult 1: resp: b\nresult 2: resp: c\n"
	if result.Stdout != expected {
		t.Errorf("stdout = %q, want %q", result.Stdout, expected)
	}
}

func TestPendingAsyncQueries(t *testing.T) {
	client := newMockClient()
	blockCh := make(chan struct{})
	client.queryFunc = func(ctx context.Context, prompt string) (QueryResponse, error) {
		<-blockCh
		return QueryResponse{Response: "done"}, nil
	}

	r := New(client)

	if r.PendingAsyncQueries() != 0 {
		t.Error("expected 0 pending queries initially")
	}

	_ = r.QueryAsync("test1")
	_ = r.QueryAsync("test2")

	if r.PendingAsyncQueries() != 2 {
		t.Errorf("expected 2 pending queries, got %d", r.PendingAsyncQueries())
	}

	close(blockCh)

	r.WaitAllAsyncQueries()

	if r.PendingAsyncQueries() != 0 {
		t.Errorf("expected 0 pending queries after wait, got %d", r.PendingAsyncQueries())
	}
}

func TestGetAsyncQuery(t *testing.T) {
	client := newMockClient()

	r := New(client)

	handle := r.QueryAsync("test")

	retrieved, exists := r.GetAsyncQuery(handle.ID())
	if !exists {
		t.Error("expected query to exist")
	}
	if retrieved != handle {
		t.Error("expected to retrieve same handle")
	}

	_, exists = r.GetAsyncQuery("non-existent-id")
	if exists {
		t.Error("expected non-existent query to not exist")
	}
}

func TestAsyncCleanupOnClose(t *testing.T) {
	client := newMockClient()
	blockCh := make(chan struct{})
	client.queryFunc = func(ctx context.Context, prompt string) (QueryResponse, error) {
		<-blockCh
		return QueryResponse{Response: "done"}, nil
	}

	r := New(client)

	_ = r.QueryAsync("test1")
	_ = r.QueryAsync("test2")

	r.Close()

	if len(r.asyncQueries) != 0 {
		t.Errorf("expected async queries to be cleared on Close, got %d", len(r.asyncQueries))
	}

	close(blockCh)
}

func TestAsyncConcurrentAccess(t *testing.T) {
	client := newMockClient()
	client.queryFunc = func(ctx context.Context, prompt string) (QueryResponse, error) {
		time.Sleep(5 * time.Millisecond)
		return QueryResponse{Response: "done"}, nil
	}

	r := New(client)

	var wg sync.WaitGroup
	errCh := make(chan error, 100)

	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func(n int) {
			defer wg.Done()

			handle := r.QueryAsync("prompt")
			result, err := handle.Wait()
			if err != nil {
				errCh <- err
				return
			}
			if result != "done" {
				errCh <- errors.New("unexpected result")
			}
		}(i)
	}

	wg.Wait()
	close(errCh)

	for err := range errCh {
		t.Errorf("concurrent error: %v", err)
	}

	calls := r.GetLLMCalls()
	if len(calls) != 100 {
		t.Errorf("expected 100 LLM calls, got %d", len(calls))
	}
}

func TestAsyncBatchHandleHandles(t *testing.T) {
	client := newMockClient()
	r := New(client)

	prompts := []string{"a", "b"}
	batchHandle := r.QueryBatchedAsync(prompts)

	handles := batchHandle.Handles()
	if len(handles) != 2 {
		t.Fatalf("expected 2 handles, got %d", len(handles))
	}

	for i, h := range handles {
		result, err := h.Wait()
		if err != nil {
			t.Errorf("handle %d error: %v", i, err)
		}
		if result == "" {
			t.Errorf("handle %d has empty result", i)
		}
	}
}
