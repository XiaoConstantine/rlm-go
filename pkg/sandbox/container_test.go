package sandbox

import (
	"context"
	"errors"
	"fmt"
	"os"
	osexec "os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/XiaoConstantine/rlm-go/pkg/contextindex"
)

// MockLLMClient is a mock implementation of LLMClient for testing.
type MockLLMClient struct {
	Responses map[string]string
	CallCount int
	Calls     []string
}

func NewMockLLMClient() *MockLLMClient {
	return &MockLLMClient{
		Responses: make(map[string]string),
	}
}

func (m *MockLLMClient) Query(ctx context.Context, prompt string) (QueryResponse, error) {
	m.CallCount++
	m.Calls = append(m.Calls, prompt)

	response, ok := m.Responses[prompt]
	if !ok {
		response = "Mock response for: " + prompt
	}

	return QueryResponse{
		Response:         response,
		PromptTokens:     len(prompt) / 4,
		CompletionTokens: len(response) / 4,
	}, nil
}

func (m *MockLLMClient) QueryBatched(ctx context.Context, prompts []string) ([]QueryResponse, error) {
	results := make([]QueryResponse, len(prompts))
	for i, prompt := range prompts {
		results[i], _ = m.Query(ctx, prompt)
	}
	return results, nil
}

type contextDeadlineLLMClient struct {
	errCh chan error
}

func newContextDeadlineLLMClient() *contextDeadlineLLMClient {
	return &contextDeadlineLLMClient{errCh: make(chan error, 1)}
}

func (m *contextDeadlineLLMClient) Query(ctx context.Context, prompt string) (QueryResponse, error) {
	<-ctx.Done()
	err := ctx.Err()
	m.errCh <- err
	return QueryResponse{}, err
}

func (m *contextDeadlineLLMClient) QueryBatched(ctx context.Context, prompts []string) ([]QueryResponse, error) {
	<-ctx.Done()
	err := ctx.Err()
	m.errCh <- err
	return nil, err
}

type delayedLLMClient struct {
	started chan struct{}
	release chan struct{}
}

func newDelayedLLMClient() *delayedLLMClient {
	return &delayedLLMClient{
		started: make(chan struct{}),
		release: make(chan struct{}),
	}
}

func (m *delayedLLMClient) Query(ctx context.Context, prompt string) (QueryResponse, error) {
	close(m.started)
	<-m.release
	return QueryResponse{Response: "late"}, nil
}

func (m *delayedLLMClient) QueryBatched(ctx context.Context, prompts []string) ([]QueryResponse, error) {
	close(m.started)
	<-m.release
	results := make([]QueryResponse, len(prompts))
	for i := range results {
		results[i] = QueryResponse{Response: "late"}
	}
	return results, nil
}

type promptBlockingLLMClient struct {
	staleStarted   chan struct{}
	staleRelease   chan struct{}
	barrierStarted chan struct{}
	barrierRelease chan struct{}
	freshStarted   chan struct{}
	freshRelease   chan struct{}
	staleOnce      sync.Once
	barrierOnce    sync.Once
	freshOnce      sync.Once
}

func newPromptBlockingLLMClient() *promptBlockingLLMClient {
	return &promptBlockingLLMClient{
		staleStarted:   make(chan struct{}),
		staleRelease:   make(chan struct{}),
		barrierStarted: make(chan struct{}),
		barrierRelease: make(chan struct{}),
		freshStarted:   make(chan struct{}),
		freshRelease:   make(chan struct{}),
	}
}

func (m *promptBlockingLLMClient) Query(ctx context.Context, prompt string) (QueryResponse, error) {
	switch {
	case strings.Contains(prompt, "stale query"):
		m.staleOnce.Do(func() { close(m.staleStarted) })
		<-m.staleRelease
		return QueryResponse{Response: "stale output"}, nil
	case strings.Contains(prompt, "barrier query"):
		m.barrierOnce.Do(func() { close(m.barrierStarted) })
		<-m.barrierRelease
		return QueryResponse{Response: "barrier output"}, nil
	case strings.Contains(prompt, "fresh query"):
		m.freshOnce.Do(func() { close(m.freshStarted) })
		<-m.freshRelease
		return QueryResponse{Response: "fresh output"}, nil
	default:
		return QueryResponse{Response: "response for " + prompt}, nil
	}
}

func (m *promptBlockingLLMClient) QueryBatched(ctx context.Context, prompts []string) ([]QueryResponse, error) {
	results := make([]QueryResponse, len(prompts))
	for i, prompt := range prompts {
		result, err := m.Query(ctx, prompt)
		if err != nil {
			return nil, err
		}
		results[i] = result
	}
	return results, nil
}

func TestRuntimeDetection(t *testing.T) {
	// Test the runtime detection functions
	podmanAvail := IsRuntimeAvailable("podman")
	dockerAvail := IsRuntimeAvailable("docker")

	t.Logf("Podman available: %v", podmanAvail)
	t.Logf("Docker available: %v", dockerAvail)

	// detectBestBackend should return something
	backend := detectBestBackend()
	t.Logf("Best backend: %s", backend)

	if !podmanAvail && !dockerAvail {
		if backend != BackendLocal {
			t.Errorf("Expected BackendLocal when no container runtime available, got %s", backend)
		}
	}
}

func TestLocalExecutorBasic(t *testing.T) {
	client := NewMockLLMClient()
	cfg := DefaultConfig()
	cfg.Backend = BackendLocal

	exec, err := NewLocalExecutor(client, cfg)
	if err != nil {
		t.Fatalf("Failed to create local executor: %v", err)
	}
	defer func() { _ = exec.Close() }()

	// Test basic code execution
	result, err := exec.Execute(context.Background(), `fmt.Println("Hello, sandbox!")`)
	if err != nil {
		t.Fatalf("Execute failed: %v", err)
	}

	if !strings.Contains(result.Stdout, "Hello, sandbox!") {
		t.Errorf("Expected 'Hello, sandbox!' in output, got: %s", result.Stdout)
	}

	if exec.Backend() != BackendLocal {
		t.Errorf("Expected BackendLocal, got %s", exec.Backend())
	}

	type finalState interface {
		HasFinal() bool
		Final() (string, bool)
		ClearFinal()
	}
	finalEnv, ok := any(exec).(finalState)
	if !ok {
		t.Fatal("LocalExecutor should expose FINAL state")
	}
	if finalEnv.HasFinal() {
		t.Fatal("new LocalExecutor should not have final state")
	}
}

func TestLocalExecutorWithContext(t *testing.T) {
	client := NewMockLLMClient()
	cfg := DefaultConfig()
	cfg.Backend = BackendLocal

	exec, err := NewLocalExecutor(client, cfg)
	if err != nil {
		t.Fatalf("Failed to create local executor: %v", err)
	}
	defer func() { _ = exec.Close() }()

	// Load string context
	err = exec.LoadContext("test context data")
	if err != nil {
		t.Fatalf("LoadContext failed: %v", err)
	}

	// Verify context info
	info := exec.ContextInfo()
	if !strings.Contains(info, "string") {
		t.Errorf("Expected context info to mention 'string', got: %s", info)
	}

	// Execute code that uses the context
	result, err := exec.Execute(context.Background(), `fmt.Println("Context:", context)`)
	if err != nil {
		t.Fatalf("Execute failed: %v", err)
	}

	if !strings.Contains(result.Stdout, "test context data") {
		t.Errorf("Expected context data in output, got: %s", result.Stdout)
	}
}

func TestLocalExecutorContextIndexHelpers(t *testing.T) {
	client := NewMockLLMClient()
	cfg := DefaultConfig()
	cfg.Backend = BackendLocal

	exec, err := NewLocalExecutor(client, cfg)
	if err != nil {
		t.Fatalf("Failed to create local executor: %v", err)
	}
	defer func() { _ = exec.Close() }()

	if err := exec.LoadContext("alpha first\nbeta second\nalpha beta third"); err != nil {
		t.Fatalf("LoadContext failed: %v", err)
	}

	info := exec.ContextInfo()
	for _, want := range []string{"lines=3", "chunks=1"} {
		if !strings.Contains(info, want) {
			t.Fatalf("ContextInfo() = %q, want %q", info, want)
		}
	}

	result, err := exec.Execute(context.Background(), `
fmt.Println("lines", LineCount())
fmt.Println("chunks", ChunkCount())
fmt.Println("range", GetContext(2, 3))
fmt.Println("past", GetContext(99, 99) == "")
fmt.Println("chunk", strings.Contains(GetChunk(0), "alpha first"))
relevant := FindRelevant("beta", 1)
fmt.Println("relevant", len(relevant), strings.Contains(relevant[0], "beta second"))
`)
	if err != nil {
		t.Fatalf("Execute failed: %v", err)
	}
	if result.Stderr != "" {
		t.Fatalf("stderr = %q", result.Stderr)
	}
	for _, want := range []string{
		"lines 3",
		"chunks 1",
		"range beta second\nalpha beta third",
		"past true",
		"chunk true",
		"relevant 1 true",
	} {
		if !strings.Contains(result.Stdout, want) {
			t.Fatalf("stdout = %q, want %q", result.Stdout, want)
		}
	}

	result, err = exec.Execute(context.Background(), `
context = "gamma first\ndelta second"
fmt.Println("mutated lines", LineCount())
fmt.Println("mutated range", GetContext(1, 2))
mutated := FindRelevant("gamma?", 1)
fmt.Println("mutated relevant", len(mutated), strings.Contains(mutated[0], "gamma first"), strings.Contains(mutated[0], "alpha first"))
`)
	if err != nil {
		t.Fatalf("Execute mutated context failed: %v", err)
	}
	if result.Stderr != "" {
		t.Fatalf("mutated stderr = %q", result.Stderr)
	}
	for _, want := range []string{
		"mutated lines 2",
		"mutated range gamma first\ndelta second",
		"mutated relevant 1 true false",
	} {
		if !strings.Contains(result.Stdout, want) {
			t.Fatalf("mutated stdout = %q, want %q", result.Stdout, want)
		}
	}
}

func TestLocalExecutorLoadContextNilIsEmpty(t *testing.T) {
	client := NewMockLLMClient()
	cfg := DefaultConfig()
	cfg.Backend = BackendLocal

	exec, err := NewLocalExecutor(client, cfg)
	if err != nil {
		t.Fatalf("Failed to create local executor: %v", err)
	}
	defer func() { _ = exec.Close() }()

	if err := exec.LoadContext(nil); err != nil {
		t.Fatalf("LoadContext(nil) failed: %v", err)
	}

	result, err := exec.Execute(context.Background(), `
fmt.Println("context", context == "")
fmt.Println("lines", LineCount())
fmt.Println("range", GetContext(1, 1) == "")
fmt.Println("relevant", len(FindRelevant("null", 1)))
`)
	if err != nil {
		t.Fatalf("Execute failed: %v", err)
	}
	if result.Stderr != "" {
		t.Fatalf("stderr = %q", result.Stderr)
	}
	for _, want := range []string{
		"context true",
		"lines 0",
		"range true",
		"relevant 0",
	} {
		if !strings.Contains(result.Stdout, want) {
			t.Fatalf("stdout = %q, want %q", result.Stdout, want)
		}
	}
}

func TestLocalExecutorWithQuery(t *testing.T) {
	client := NewMockLLMClient()
	client.Responses["What is 2+2?"] = "4"

	cfg := DefaultConfig()
	cfg.Backend = BackendLocal

	exec, err := NewLocalExecutor(client, cfg)
	if err != nil {
		t.Fatalf("Failed to create local executor: %v", err)
	}
	defer func() { _ = exec.Close() }()

	// Execute code that uses Query
	result, err := exec.Execute(context.Background(), `
		answer := Query("What is 2+2?")
		fmt.Println("Answer:", answer)
	`)
	if err != nil {
		t.Fatalf("Execute failed: %v", err)
	}

	if !strings.Contains(result.Stdout, "4") {
		t.Errorf("Expected '4' in output, got: %s", result.Stdout)
	}

	// Verify LLM calls were recorded
	calls := exec.GetLLMCalls()
	if len(calls) != 1 {
		t.Errorf("Expected 1 LLM call, got %d", len(calls))
	}

	if len(calls) > 0 && calls[0].Prompt != "What is 2+2?" {
		t.Errorf("Expected prompt 'What is 2+2?', got: %s", calls[0].Prompt)
	}
}

func TestLocalExecutorQueryModes(t *testing.T) {
	client := NewMockLLMClient()
	cfg := DefaultConfig()
	cfg.Backend = BackendLocal

	exec, err := NewLocalExecutor(client, cfg)
	if err != nil {
		t.Fatalf("Failed to create local executor: %v", err)
	}
	defer func() { _ = exec.Close() }()

	if err := exec.LoadContext("full context"); err != nil {
		t.Fatalf("LoadContext failed: %v", err)
	}

	if _, err := exec.Execute(context.Background(), `Query("question")`); err != nil {
		t.Fatalf("Execute Query failed: %v", err)
	}
	if len(client.Calls) != 1 || !strings.Contains(client.Calls[0], "full context") {
		t.Fatalf("Query should prepend full context, calls=%v", client.Calls)
	}
	if calls := exec.GetLLMCalls(); len(calls) != 1 || calls[0].Prompt != client.Calls[0] {
		t.Fatalf("GetLLMCalls should record expanded Query prompt, calls=%+v clientCalls=%v", calls, client.Calls)
	}

	client.Calls = nil
	if _, err := exec.Execute(context.Background(), `context = "mutated context"
Query("mutated question")`); err != nil {
		t.Fatalf("Execute mutated Query failed: %v", err)
	}
	if len(client.Calls) != 1 || !strings.Contains(client.Calls[0], "mutated context") || strings.Contains(client.Calls[0], "full context") {
		t.Fatalf("Query should use current mutable context variable, calls=%v", client.Calls)
	}
	_ = exec.GetLLMCalls()
	if _, err := exec.Execute(context.Background(), `context = "full context"`); err != nil {
		t.Fatalf("Execute restore context failed: %v", err)
	}

	client.Calls = nil
	if _, err := exec.Execute(context.Background(), `QueryRaw("raw question")`); err != nil {
		t.Fatalf("Execute QueryRaw failed: %v", err)
	}
	if len(client.Calls) != 1 || client.Calls[0] != "raw question" {
		t.Fatalf("QueryRaw should send raw prompt, calls=%v", client.Calls)
	}
	if calls := exec.GetLLMCalls(); len(calls) != 1 || calls[0].Prompt != "raw question" {
		t.Fatalf("GetLLMCalls should record raw QueryRaw prompt, calls=%+v", calls)
	}

	client.Calls = nil
	if _, err := exec.Execute(context.Background(), `QueryWith("slice context", "slice question")`); err != nil {
		t.Fatalf("Execute QueryWith failed: %v", err)
	}
	if len(client.Calls) != 1 || !strings.Contains(client.Calls[0], "slice context") || strings.Contains(client.Calls[0], "full context") {
		t.Fatalf("QueryWith should send only selected context, calls=%v", client.Calls)
	}
	if calls := exec.GetLLMCalls(); len(calls) != 1 || calls[0].Prompt != client.Calls[0] {
		t.Fatalf("GetLLMCalls should record expanded QueryWith prompt, calls=%+v clientCalls=%v", calls, client.Calls)
	}

	client.Calls = nil
	if _, err := exec.Execute(context.Background(), `QueryBatched([]string{"batch one", "batch two"})`); err != nil {
		t.Fatalf("Execute QueryBatched failed: %v", err)
	}
	if len(client.Calls) != 2 || !strings.Contains(client.Calls[0], "full context") || !strings.Contains(client.Calls[1], "full context") {
		t.Fatalf("QueryBatched should prepend full context to each prompt, calls=%v", client.Calls)
	}
	if calls := exec.GetLLMCalls(); len(calls) != 2 || calls[0].Prompt != client.Calls[0] || calls[1].Prompt != client.Calls[1] {
		t.Fatalf("GetLLMCalls should record expanded QueryBatched prompts, calls=%+v clientCalls=%v", calls, client.Calls)
	}

	client.Calls = nil
	if err := exec.Reset(); err != nil {
		t.Fatalf("Reset failed: %v", err)
	}
	if _, err := exec.Execute(context.Background(), `Query("after reset")`); err != nil {
		t.Fatalf("Execute Query after reset failed: %v", err)
	}
	if len(client.Calls) != 1 || strings.Contains(client.Calls[0], "full context") {
		t.Fatalf("Query after Reset should not prepend stale context, calls=%v", client.Calls)
	}
}

func TestLocalExecutorMaxFullContextQueryCharsBlocksQuery(t *testing.T) {
	client := NewMockLLMClient()
	cfg := DefaultConfig()
	cfg.Backend = BackendLocal
	cfg.MaxFullContextQueryChars = 5

	exec, err := NewLocalExecutor(client, cfg)
	if err != nil {
		t.Fatalf("Failed to create local executor: %v", err)
	}
	defer func() { _ = exec.Close() }()

	if err := exec.LoadContext("large context"); err != nil {
		t.Fatalf("LoadContext failed: %v", err)
	}

	result, err := exec.Execute(context.Background(), `
fmt.Println(Query("blocked"))
responses := QueryBatched([]string{"one", "two"})
fmt.Println(responses[0])
fmt.Println(responses[1])
fmt.Println(QueryRaw("raw"))
fmt.Println(QueryWith("slice", "allowed"))
`)
	if err != nil {
		t.Fatalf("Execute failed: %v", err)
	}
	if result.Stderr != "" {
		t.Fatalf("stderr = %q", result.Stderr)
	}
	if got := strings.Count(result.Stdout, "exceeding the limit"); got != 3 {
		t.Fatalf("stdout = %q, want 3 guardrail errors", result.Stdout)
	}
	if len(client.Calls) != 2 {
		t.Fatalf("client calls = %v, want only QueryRaw and QueryWith", client.Calls)
	}
	if client.Calls[0] != "raw" {
		t.Fatalf("QueryRaw call = %q, want raw", client.Calls[0])
	}
	if !strings.Contains(client.Calls[1], "slice") || strings.Contains(client.Calls[1], "large context") {
		t.Fatalf("QueryWith call should use selected context only, calls=%v", client.Calls)
	}

	calls := exec.GetLLMCalls()
	if len(calls) != 5 {
		t.Fatalf("GetLLMCalls() returned %d calls, want blocked and allowed calls: %+v", len(calls), calls)
	}
	for i, wantPrompt := range []string{"blocked", "one", "two"} {
		if calls[i].Prompt != wantPrompt || !strings.Contains(calls[i].Response, "exceeding the limit") {
			t.Fatalf("call %d = %+v, want blocked prompt %q", i, calls[i], wantPrompt)
		}
	}
}

func TestLocalExecutorWithBatchedQuery(t *testing.T) {
	client := NewMockLLMClient()
	client.Responses["Q1"] = "A1"
	client.Responses["Q2"] = "A2"

	cfg := DefaultConfig()
	cfg.Backend = BackendLocal

	exec, err := NewLocalExecutor(client, cfg)
	if err != nil {
		t.Fatalf("Failed to create local executor: %v", err)
	}
	defer func() { _ = exec.Close() }()

	// Execute code that uses QueryBatched
	result, err := exec.Execute(context.Background(), `
		prompts := []string{"Q1", "Q2"}
		answers := QueryBatched(prompts)
		for i, a := range answers {
			fmt.Printf("Answer %d: %s\n", i, a)
		}
	`)
	if err != nil {
		t.Fatalf("Execute failed: %v", err)
	}

	if !strings.Contains(result.Stdout, "A1") || !strings.Contains(result.Stdout, "A2") {
		t.Errorf("Expected 'A1' and 'A2' in output, got: %s", result.Stdout)
	}

	// Verify LLM calls were recorded
	calls := exec.GetLLMCalls()
	if len(calls) != 2 {
		t.Errorf("Expected 2 LLM calls, got %d", len(calls))
	}
}

func TestLocalExecutorTimeout(t *testing.T) {
	client := NewMockLLMClient()
	cfg := DefaultConfig()
	cfg.Backend = BackendLocal
	cfg.Timeout = 100 * time.Millisecond

	exec, err := NewLocalExecutor(client, cfg)
	if err != nil {
		t.Fatalf("Failed to create local executor: %v", err)
	}
	defer func() { _ = exec.Close() }()

	// Note: Yaegi doesn't support true cancellation, so this test
	// verifies the timeout configuration is passed correctly.
	// The actual timeout behavior may vary.
	result, err := exec.Execute(context.Background(), `
		fmt.Println("Quick execution")
	`)
	if err != nil {
		t.Fatalf("Execute failed: %v", err)
	}

	if !strings.Contains(result.Stdout, "Quick execution") {
		t.Errorf("Expected output, got: %s", result.Stdout)
	}
}

func TestLocalExecutorTimeoutReturns(t *testing.T) {
	client := NewMockLLMClient()
	cfg := DefaultConfig()
	cfg.Backend = BackendLocal
	cfg.Timeout = 50 * time.Millisecond

	exec, err := NewLocalExecutor(client, cfg)
	if err != nil {
		t.Fatalf("Failed to create local executor: %v", err)
	}
	defer func() { _ = exec.Close() }()

	start := time.Now()
	result, err := exec.Execute(context.Background(), `ch := make(chan struct{})
<-ch`)
	if err != nil {
		t.Fatalf("Execute timeout should return nil error, got: %v", err)
	}
	if time.Since(start) > time.Second {
		t.Fatal("Execute did not return promptly after timeout")
	}
	if !strings.Contains(result.Stderr, "execution timeout exceeded") {
		t.Fatalf("stderr = %q, want timeout", result.Stderr)
	}

	result, err = exec.Execute(context.Background(), `FINAL("fresh")`)
	if err != nil {
		t.Fatalf("Execute after timeout failed: %v", err)
	}
	if final, ok := exec.Final(); !ok || final != "fresh" {
		t.Fatalf("Final after timeout = %q/%v, want fresh/true; stdout=%q", final, ok, result.Stdout)
	}
}

func TestLocalExecutorQueryUsesExecutionTimeout(t *testing.T) {
	client := newContextDeadlineLLMClient()
	cfg := DefaultConfig()
	cfg.Backend = BackendLocal
	cfg.Timeout = 50 * time.Millisecond

	exec, err := NewLocalExecutor(client, cfg)
	if err != nil {
		t.Fatalf("Failed to create local executor: %v", err)
	}
	defer func() { _ = exec.Close() }()

	start := time.Now()
	_, err = exec.Execute(context.Background(), `fmt.Println(Query("wait for timeout"))`)
	if err != nil {
		t.Fatalf("Execute failed: %v", err)
	}
	if time.Since(start) > time.Second {
		t.Fatal("Execute did not return promptly after Query context deadline")
	}

	select {
	case err := <-client.errCh:
		if !errors.Is(err, context.DeadlineExceeded) {
			t.Fatalf("Query context error = %v, want deadline exceeded", err)
		}
	default:
		t.Fatal("Query did not observe the execution timeout context")
	}
}

func TestLocalExecutorStaleQueryDoesNotCallClient(t *testing.T) {
	client := NewMockLLMClient()
	cfg := DefaultConfig()
	cfg.Backend = BackendLocal

	exec, err := NewLocalExecutor(client, cfg)
	if err != nil {
		t.Fatalf("Failed to create local executor: %v", err)
	}
	defer func() { _ = exec.Close() }()

	staleToken := exec.rotateFinalToken()
	exec.expireFinalToken(staleToken)

	response := exec.llmQueryRaw(staleToken, context.Background(), "question")
	if response != "Error: execution expired" {
		t.Fatalf("stale Query response = %q, want execution expired", response)
	}
	if client.CallCount != 0 {
		t.Fatalf("stale Query should not call client, calls=%d", client.CallCount)
	}
}

func TestLocalExecutorTimeoutClearsFinalAfterQueryDeadline(t *testing.T) {
	client := newContextDeadlineLLMClient()
	cfg := DefaultConfig()
	cfg.Backend = BackendLocal
	cfg.Timeout = 25 * time.Millisecond

	exec, err := NewLocalExecutor(client, cfg)
	if err != nil {
		t.Fatalf("Failed to create local executor: %v", err)
	}
	defer func() { _ = exec.Close() }()

	result, err := exec.Execute(context.Background(), `answer := Query("wait for timeout")
FINAL("should not finalize after " + answer)`)
	if err != nil {
		t.Fatalf("Execute failed: %v", err)
	}
	if !strings.Contains(result.Stderr, "execution timeout exceeded") {
		t.Fatalf("stderr = %q, want timeout", result.Stderr)
	}
	if exec.HasFinal() {
		t.Fatal("timeout should clear final state even if code calls FINAL after Query returns")
	}
}

func TestLocalExecutorTimeoutPreservesLLMCalls(t *testing.T) {
	client := NewMockLLMClient()
	client.Responses["before timeout"] = "ok"
	cfg := DefaultConfig()
	cfg.Backend = BackendLocal
	cfg.Timeout = 50 * time.Millisecond

	exec, err := NewLocalExecutor(client, cfg)
	if err != nil {
		t.Fatalf("Failed to create local executor: %v", err)
	}
	defer func() { _ = exec.Close() }()

	result, err := exec.Execute(context.Background(), `Query("before timeout")
ch := make(chan struct{})
<-ch`)
	if err != nil {
		t.Fatalf("Execute timeout should return nil error, got: %v", err)
	}
	if !strings.Contains(result.Stderr, "execution timeout exceeded") {
		t.Fatalf("stderr = %q, want timeout", result.Stderr)
	}
	calls := exec.GetLLMCalls()
	if len(calls) != 1 {
		t.Fatalf("LLM calls after timeout = %d, want 1", len(calls))
	}
	if calls[0].Prompt != "before timeout" || calls[0].Response != "ok" {
		t.Fatalf("LLM call = %+v, want prompt/response before timeout/ok", calls[0])
	}
}

func TestLocalExecutorCanceledContextDoesNotRun(t *testing.T) {
	client := NewMockLLMClient()
	cfg := DefaultConfig()
	cfg.Backend = BackendLocal

	exec, err := NewLocalExecutor(client, cfg)
	if err != nil {
		t.Fatalf("Failed to create local executor: %v", err)
	}
	defer func() { _ = exec.Close() }()

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	result, err := exec.Execute(ctx, `fmt.Println("should not run")
FINAL("should not finalize")`)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("Execute error = %v, want context canceled", err)
	}
	if strings.Contains(result.Stdout, "should not run") {
		t.Fatalf("canceled execution ran code, stdout=%q", result.Stdout)
	}
	if exec.HasFinal() {
		t.Fatal("canceled execution should not set final state")
	}
}

func TestLocalExecutorCanceledContextPreservesState(t *testing.T) {
	client := NewMockLLMClient()
	cfg := DefaultConfig()
	cfg.Backend = BackendLocal

	exec, err := NewLocalExecutor(client, cfg)
	if err != nil {
		t.Fatalf("Failed to create local executor: %v", err)
	}
	defer func() { _ = exec.Close() }()

	if _, err := exec.Execute(context.Background(), `var keep = "state"`); err != nil {
		t.Fatalf("Execute setup failed: %v", err)
	}

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	_, err = exec.Execute(ctx, `var shouldNotRun = true`)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("Execute error = %v, want context canceled", err)
	}

	got, err := exec.GetVariable("keep")
	if err != nil {
		t.Fatalf("state variable missing after canceled Execute: %v", err)
	}
	if got != "state" {
		t.Fatalf("keep = %q, want state", got)
	}
}

func TestLocalExecutorExpiredParentDeadlinePreservesState(t *testing.T) {
	client := NewMockLLMClient()
	cfg := DefaultConfig()
	cfg.Backend = BackendLocal

	exec, err := NewLocalExecutor(client, cfg)
	if err != nil {
		t.Fatalf("Failed to create local executor: %v", err)
	}
	defer func() { _ = exec.Close() }()

	if _, err := exec.Execute(context.Background(), `var keepDeadline = "state"`); err != nil {
		t.Fatalf("Execute setup failed: %v", err)
	}

	ctx, cancel := context.WithDeadline(context.Background(), time.Now().Add(-time.Second))
	defer cancel()
	_, err = exec.Execute(ctx, `var shouldNotRunDeadline = true`)
	if !errors.Is(err, context.DeadlineExceeded) {
		t.Fatalf("Execute error = %v, want deadline exceeded", err)
	}

	got, err := exec.GetVariable("keepDeadline")
	if err != nil {
		t.Fatalf("state variable missing after expired parent deadline: %v", err)
	}
	if got != "state" {
		t.Fatalf("keepDeadline = %q, want state", got)
	}
}

func TestLocalExecutorFinalTokenRefresh(t *testing.T) {
	client := NewMockLLMClient()
	cfg := DefaultConfig()
	cfg.Backend = BackendLocal

	exec, err := NewLocalExecutor(client, cfg)
	if err != nil {
		t.Fatalf("Failed to create local executor: %v", err)
	}
	defer func() { _ = exec.Close() }()

	result, err := exec.Execute(context.Background(), `FINAL("first")`)
	if err != nil {
		t.Fatalf("Execute first FINAL failed: %v", err)
	}
	if result.Stderr != "" {
		t.Fatalf("Execute first FINAL stderr: %s", result.Stderr)
	}
	if final, ok := exec.Final(); !ok || final != "first" {
		t.Fatalf("First final = %q/%v, want first/true", final, ok)
	}

	result, err = exec.Execute(context.Background(), `FINAL("second")`)
	if err != nil {
		t.Fatalf("Execute second FINAL failed: %v", err)
	}
	if result.Stderr != "" {
		t.Fatalf("Execute second FINAL stderr: %s", result.Stderr)
	}
	if final, ok := exec.Final(); !ok || final != "second" {
		t.Fatalf("Second final = %q/%v, want second/true", final, ok)
	}
	if strings.Contains(result.Stdout, finalMarkerPrefix) {
		t.Fatalf("local stdout leaked final marker: %q", result.Stdout)
	}
}

func TestLocalExecutorStaleFinalIgnored(t *testing.T) {
	client := NewMockLLMClient()
	cfg := DefaultConfig()
	cfg.Backend = BackendLocal

	exec, err := NewLocalExecutor(client, cfg)
	if err != nil {
		t.Fatalf("Failed to create local executor: %v", err)
	}
	defer func() { _ = exec.Close() }()

	staleToken := exec.rotateFinalToken()
	exec.expireFinalToken(staleToken)
	currentToken := exec.rotateFinalToken()
	exec.stdout.Begin(1)

	exec.finalAnswer(staleToken, exec.stdout, 1, "stale")
	if final, ok := exec.Final(); ok {
		t.Fatalf("Stale token set final = %q", final)
	}
	if out := exec.stdout.String(); out != "" {
		t.Fatalf("Stale token wrote stdout marker: %q", out)
	}

	exec.finalAnswer(currentToken, exec.stdout, 1, "fresh")
	if final, ok := exec.Final(); !ok || final != "fresh" {
		t.Fatalf("Current token final = %q/%v, want fresh/true", final, ok)
	}
}

func TestLocalOutputRejectsNonEvalGoroutine(t *testing.T) {
	out := new(localOutput)
	out.Begin(1)
	out.AllowCurrent(1)

	done := make(chan struct{})
	go func() {
		_, _ = out.Write([]byte("same-run"))
		close(done)
	}()

	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("write goroutine did not finish")
	}

	if got := out.End(1); got != "same-run" {
		t.Fatalf("allowed child goroutine write = %q, want same-run", got)
	}

	out.Begin(2)
	done = make(chan struct{})
	go func() {
		_, _ = out.Write([]byte("stale"))
		close(done)
	}()

	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("stale write goroutine did not finish")
	}

	if got := out.String(); got != "" {
		t.Fatalf("unowned goroutine write = %q, want empty", got)
	}
	_, _ = out.WriteForRun(1, []byte("fresh"))
	if got := out.String(); got != "" {
		t.Fatalf("stale WriteForRun output = %q, want empty", got)
	}
	_, _ = out.WriteForRun(2, []byte("fresh"))
	if got := out.End(2); got != "fresh" {
		t.Fatalf("current WriteForRun output = %q, want fresh", got)
	}
}

func TestLocalOutputRejectsOlderYaegiGoroutine(t *testing.T) {
	out := new(localOutput)
	out.Begin(1)
	out.rootGID = 100
	out.allowed[100] = struct{}{}

	if !out.allowWrite(101, 100, true, true) {
		t.Fatal("current-run yaegi goroutine should be allowed")
	}
	if out.allowWrite(99, 0, false, true) {
		t.Fatal("older yaegi goroutine should be rejected")
	}

	out.Begin(2)
	out.rootGID = 200
	out.allowed[200] = struct{}{}
	if out.allowWrite(250, 100, true, true) {
		t.Fatal("previous-run yaegi goroutine should not be allowed in later run")
	}
}

func TestLocalExecutorDropsStaleGoroutineOutputDuringNextRun(t *testing.T) {
	client := newPromptBlockingLLMClient()
	cfg := DefaultConfig()
	cfg.Backend = BackendLocal

	exec, err := NewLocalExecutor(client, cfg)
	if err != nil {
		t.Fatalf("Failed to create local executor: %v", err)
	}
	defer func() { _ = exec.Close() }()

	type outcome struct {
		stdout string
		stderr string
		err    error
	}
	firstDone := make(chan outcome, 1)
	go func() {
		result, err := exec.Execute(context.Background(), `go func() {
	fmt.Println(QueryRaw("stale query"))
}()
fmt.Println(QueryRaw("barrier query"))`)
		out := outcome{err: err}
		if result != nil {
			out.stdout = result.Stdout
			out.stderr = result.Stderr
		}
		firstDone <- out
	}()

	select {
	case <-client.staleStarted:
	case <-time.After(time.Second):
		t.Fatal("stale query did not start")
	}
	select {
	case <-client.barrierStarted:
	case <-time.After(time.Second):
		t.Fatal("barrier query did not start")
	}
	close(client.barrierRelease)
	select {
	case out := <-firstDone:
		if out.err != nil {
			t.Fatalf("Execute stale goroutine setup failed: %v", out.err)
		}
		if out.stderr != "" {
			t.Fatalf("setup stderr = %q", out.stderr)
		}
		if strings.Contains(out.stdout, "stale output") {
			t.Fatalf("setup stdout leaked stale output: %q", out.stdout)
		}
		if !strings.Contains(out.stdout, "barrier output") {
			t.Fatalf("setup stdout = %q, want barrier output", out.stdout)
		}
	case <-time.After(time.Second):
		t.Fatal("stale goroutine setup did not finish")
	}

	done := make(chan outcome, 1)
	go func() {
		result, err := exec.Execute(context.Background(), `fmt.Println(QueryRaw("fresh query"))`)
		out := outcome{err: err}
		if result != nil {
			out.stdout = result.Stdout
			out.stderr = result.Stderr
		}
		done <- out
	}()

	select {
	case <-client.freshStarted:
	case <-time.After(time.Second):
		t.Fatal("fresh query did not start")
	}
	close(client.staleRelease)
	time.Sleep(25 * time.Millisecond)
	close(client.freshRelease)

	select {
	case out := <-done:
		if out.err != nil {
			t.Fatalf("Execute second run failed: %v", out.err)
		}
		if out.stderr != "" {
			t.Fatalf("second run stderr = %q", out.stderr)
		}
		if strings.Contains(out.stdout, "stale output") {
			t.Fatalf("stdout leaked stale goroutine output: %q", out.stdout)
		}
		if !strings.Contains(out.stdout, "fresh output") {
			t.Fatalf("stdout = %q, want fresh output", out.stdout)
		}
	case <-time.After(time.Second):
		t.Fatal("second run did not finish")
	}
}

func TestLocalExecutorContextQueryFromGoroutineDuringActiveRun(t *testing.T) {
	client := newPromptBlockingLLMClient()
	cfg := DefaultConfig()
	cfg.Backend = BackendLocal

	exec, err := NewLocalExecutor(client, cfg)
	if err != nil {
		t.Fatalf("Failed to create local executor: %v", err)
	}
	defer func() { _ = exec.Close() }()

	if err := exec.LoadContext("active context"); err != nil {
		t.Fatalf("LoadContext failed: %v", err)
	}

	type outcome struct {
		stdout string
		stderr string
		err    error
	}
	done := make(chan outcome, 1)
	go func() {
		result, err := exec.Execute(context.Background(), `done := make(chan struct{})
go func() {
	fmt.Println(Query("fresh query"))
	close(done)
}()
<-done`)
		out := outcome{err: err}
		if result != nil {
			out.stdout = result.Stdout
			out.stderr = result.Stderr
		}
		done <- out
	}()

	select {
	case <-client.freshStarted:
	case <-time.After(time.Second):
		t.Fatal("fresh query did not start")
	}
	close(client.freshRelease)

	select {
	case out := <-done:
		if out.err != nil {
			t.Fatalf("Execute failed: %v", out.err)
		}
		if out.stderr != "" {
			t.Fatalf("stderr = %q", out.stderr)
		}
		if !strings.Contains(out.stdout, "fresh output") {
			t.Fatalf("stdout = %q, want fresh output", out.stdout)
		}
	case <-time.After(time.Second):
		t.Fatal("Execute did not finish")
	}
}

func TestLocalExecutorCapturesAwaitedGoroutineOutput(t *testing.T) {
	client := NewMockLLMClient()
	cfg := DefaultConfig()
	cfg.Backend = BackendLocal

	exec, err := NewLocalExecutor(client, cfg)
	if err != nil {
		t.Fatalf("Failed to create local executor: %v", err)
	}
	defer func() { _ = exec.Close() }()

	result, err := exec.Execute(context.Background(), `done := make(chan struct{})
go func() {
	fmt.Println("from goroutine")
	close(done)
}()
<-done`)
	if err != nil {
		t.Fatalf("Execute failed: %v", err)
	}
	if result.Stderr != "" {
		t.Fatalf("stderr = %q", result.Stderr)
	}
	if !strings.Contains(result.Stdout, "from goroutine") {
		t.Fatalf("stdout = %q, want awaited goroutine output", result.Stdout)
	}
}

func TestLocalExecutorCapturesNestedAwaitedGoroutineOutput(t *testing.T) {
	client := NewMockLLMClient()
	cfg := DefaultConfig()
	cfg.Backend = BackendLocal

	exec, err := NewLocalExecutor(client, cfg)
	if err != nil {
		t.Fatalf("Failed to create local executor: %v", err)
	}
	defer func() { _ = exec.Close() }()

	result, err := exec.Execute(context.Background(), `done := make(chan struct{})
innerDone := make(chan struct{})
go func() {
	go func() {
		fmt.Println("from nested goroutine")
		close(innerDone)
	}()
	<-innerDone
	close(done)
}()
<-done`)
	if err != nil {
		t.Fatalf("Execute failed: %v", err)
	}
	if result.Stderr != "" {
		t.Fatalf("stderr = %q", result.Stderr)
	}
	if !strings.Contains(result.Stdout, "from nested goroutine") {
		t.Fatalf("stdout = %q, want nested awaited goroutine output", result.Stdout)
	}
}

func TestLocalExecutorGetVariable(t *testing.T) {
	client := NewMockLLMClient()
	cfg := DefaultConfig()
	cfg.Backend = BackendLocal

	exec, err := NewLocalExecutor(client, cfg)
	if err != nil {
		t.Fatalf("Failed to create local executor: %v", err)
	}
	defer func() { _ = exec.Close() }()

	// Execute code that sets a variable
	_, err = exec.Execute(context.Background(), `
		var result = "test value"
	`)
	if err != nil {
		t.Fatalf("Execute failed: %v", err)
	}

	// Get the variable
	val, err := exec.GetVariable("result")
	if err != nil {
		t.Fatalf("GetVariable failed: %v", err)
	}

	if val != "test value" {
		t.Errorf("Expected 'test value', got: %s", val)
	}
}

func TestLocalExecutorGetLocals(t *testing.T) {
	client := NewMockLLMClient()
	cfg := DefaultConfig()
	cfg.Backend = BackendLocal

	exec, err := NewLocalExecutor(client, cfg)
	if err != nil {
		t.Fatalf("Failed to create local executor: %v", err)
	}
	defer func() { _ = exec.Close() }()

	// Execute code that sets common variables
	_, err = exec.Execute(context.Background(), `
		var result = "some result"
		var answer = 42
	`)
	if err != nil {
		t.Fatalf("Execute failed: %v", err)
	}

	// Get locals
	locals := exec.GetLocals()

	if _, ok := locals["result"]; !ok {
		t.Error("Expected 'result' in locals")
	}
	if _, ok := locals["answer"]; !ok {
		t.Error("Expected 'answer' in locals")
	}
}

func TestLocalExecutorReset(t *testing.T) {
	client := NewMockLLMClient()
	cfg := DefaultConfig()
	cfg.Backend = BackendLocal

	exec, err := NewLocalExecutor(client, cfg)
	if err != nil {
		t.Fatalf("Failed to create local executor: %v", err)
	}
	defer func() { _ = exec.Close() }()

	// Execute some code
	_, _ = exec.Execute(context.Background(), `
		var result = "before reset"
	`)

	// Reset
	if err := exec.Reset(); err != nil {
		t.Fatalf("Reset failed: %v", err)
	}

	// After reset, the variable should not exist
	_, err = exec.GetVariable("result")
	if err == nil {
		t.Error("Expected error after reset, variable should not exist")
	}
}

func TestNewExecutorWithAuto(t *testing.T) {
	client := NewMockLLMClient()
	cfg := DefaultConfig()
	cfg.Backend = BackendAuto
	cfg.EnableIPC = false // Disable IPC for basic test to avoid network issues in CI

	exec, err := New(client, cfg)
	if err != nil {
		t.Fatalf("Failed to create executor: %v", err)
	}
	defer func() { _ = exec.Close() }()

	// Should have created some executor
	backend := exec.Backend()
	t.Logf("Auto-selected backend: %s", backend)

	// If container backend was selected, ensure image exists
	if backend == BackendPodman || backend == BackendDocker {
		if containerExec, ok := exec.(*ContainerExecutor); ok {
			if !containerExec.ImageExists() {
				ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
				defer cancel()
				if err := containerExec.PullImage(ctx); err != nil {
					t.Skipf("Container backend selected but failed to pull image: %v", err)
				}
			}
		}
	}

	// Execute basic code
	result, err := exec.Execute(context.Background(), `fmt.Println("Hello from auto backend!")`)
	if err != nil {
		t.Fatalf("Execute failed: %v", err)
	}

	if !strings.Contains(result.Stdout, "Hello from auto backend!") {
		t.Errorf("Expected greeting in output, got stdout: %q, stderr: %q", result.Stdout, result.Stderr)
	}
}

func TestContainerExecutorBasic(t *testing.T) {
	// Skip if no container runtime available
	if !IsRuntimeAvailable("podman") && !IsRuntimeAvailable("docker") {
		t.Skip("No container runtime available")
	}

	client := NewMockLLMClient()
	cfg := DefaultConfig()
	cfg.Backend = BackendAuto
	cfg.EnableIPC = false // Disable IPC for basic test
	cfg.Timeout = 60 * time.Second

	exec, err := NewContainerExecutor(client, cfg, BackendAuto)
	if err != nil {
		t.Fatalf("Failed to create container executor: %v", err)
	}
	defer func() { _ = exec.Close() }()

	t.Logf("Using backend: %s", exec.Backend())

	// Pull image first if needed
	if !exec.ImageExists() {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
		defer cancel()
		if err := exec.PullImage(ctx); err != nil {
			t.Skipf("Failed to pull image: %v", err)
		}
	}

	// Test basic code execution
	result, err := exec.Execute(context.Background(), `fmt.Println("Hello from container!")`)
	if err != nil {
		t.Fatalf("Execute failed: %v", err)
	}

	if result.Stderr != "" && !strings.Contains(result.Stdout, "Hello from container!") {
		t.Errorf("Execution had errors. stdout: %s, stderr: %s", result.Stdout, result.Stderr)
	}
}

func TestContainerExecutorWithContext(t *testing.T) {
	// Skip if no container runtime available
	if !IsRuntimeAvailable("podman") && !IsRuntimeAvailable("docker") {
		t.Skip("No container runtime available")
	}

	client := NewMockLLMClient()
	cfg := DefaultConfig()
	cfg.Backend = BackendAuto
	cfg.EnableIPC = false
	cfg.Timeout = 60 * time.Second

	exec, err := NewContainerExecutor(client, cfg, BackendAuto)
	if err != nil {
		t.Fatalf("Failed to create container executor: %v", err)
	}
	defer func() { _ = exec.Close() }()

	// Pull image first if needed
	if !exec.ImageExists() {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
		defer cancel()
		if err := exec.PullImage(ctx); err != nil {
			t.Skipf("Failed to pull image: %v", err)
		}
	}

	// Load context
	err = exec.LoadContext("container context data")
	if err != nil {
		t.Fatalf("LoadContext failed: %v", err)
	}

	// Verify context info
	info := exec.ContextInfo()
	if !strings.Contains(info, "string") {
		t.Errorf("Expected context info to mention 'string', got: %s", info)
	}
}

func TestContainerExecutorResourceLimits(t *testing.T) {
	// Skip if no container runtime available
	if !IsRuntimeAvailable("podman") && !IsRuntimeAvailable("docker") {
		t.Skip("No container runtime available")
	}

	client := NewMockLLMClient()
	cfg := DefaultConfig()
	cfg.Backend = BackendAuto
	cfg.EnableIPC = false
	cfg.Memory = "256m"
	cfg.CPUs = 0.5
	cfg.Timeout = 30 * time.Second

	exec, err := NewContainerExecutor(client, cfg, BackendAuto)
	if err != nil {
		t.Fatalf("Failed to create container executor: %v", err)
	}
	defer func() { _ = exec.Close() }()

	// Pull image first if needed
	if !exec.ImageExists() {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
		defer cancel()
		if err := exec.PullImage(ctx); err != nil {
			t.Skipf("Failed to pull image: %v", err)
		}
	}

	// Test that resource limits don't break execution
	result, err := exec.Execute(context.Background(), `fmt.Println("Limited resources test")`)
	if err != nil {
		t.Fatalf("Execute failed: %v", err)
	}

	if !strings.Contains(result.Stdout, "Limited resources test") {
		t.Logf("stdout: %s", result.Stdout)
		t.Logf("stderr: %s", result.Stderr)
	}
}

func TestContainerExecutorRuntimeInfo(t *testing.T) {
	// Skip if no container runtime available
	if !IsRuntimeAvailable("podman") && !IsRuntimeAvailable("docker") {
		t.Skip("No container runtime available")
	}

	client := NewMockLLMClient()
	cfg := DefaultConfig()
	cfg.EnableIPC = false

	exec, err := NewContainerExecutor(client, cfg, BackendAuto)
	if err != nil {
		t.Fatalf("Failed to create container executor: %v", err)
	}
	defer func() { _ = exec.Close() }()

	info, err := exec.RuntimeInfo()
	if err != nil {
		t.Fatalf("RuntimeInfo failed: %v", err)
	}

	t.Logf("Runtime info: %s", info)

	if info == "" {
		t.Error("Expected non-empty runtime info")
	}
}

func TestIPCServerBasic(t *testing.T) {
	client := NewMockLLMClient()
	client.Responses["test prompt"] = "test response"

	server, err := NewIPCServer(client, 0) // Auto-assign port
	if err != nil {
		t.Fatalf("Failed to create IPC server: %v", err)
	}
	defer func() { _ = server.Stop() }()

	server.Start()

	port := server.Port()
	if port == 0 {
		t.Error("Expected non-zero port")
	}

	t.Logf("IPC server listening on port %d", port)

	// Create IPC client
	ipcClient, err := NewIPCClient(fmt.Sprintf("127.0.0.1:%d", port))
	if err != nil {
		t.Fatalf("Failed to create IPC client: %v", err)
	}
	defer func() { _ = ipcClient.Close() }()

	// Test Query
	response, tokenUsage, err := ipcClient.Query("test prompt")
	if err != nil {
		t.Fatalf("Query failed: %v", err)
	}

	if response != "test response" {
		t.Errorf("Expected 'test response', got: %s", response)
	}

	if tokenUsage == nil {
		t.Error("Expected token usage")
	}

	// Verify call was recorded
	calls := server.GetCalls()
	if len(calls) != 1 {
		t.Errorf("Expected 1 call recorded, got %d", len(calls))
	}
}

func TestIPCServerBatched(t *testing.T) {
	client := NewMockLLMClient()
	client.Responses["p1"] = "r1"
	client.Responses["p2"] = "r2"

	server, err := NewIPCServer(client, 0)
	if err != nil {
		t.Fatalf("Failed to create IPC server: %v", err)
	}
	defer func() { _ = server.Stop() }()

	server.Start()

	// Create IPC client
	ipcClient, err := NewIPCClient(server.Address())
	if err != nil {
		t.Fatalf("Failed to create IPC client: %v", err)
	}
	defer func() { _ = ipcClient.Close() }()

	// Test QueryBatched
	responses, usages, err := ipcClient.QueryBatched([]string{"p1", "p2"})
	if err != nil {
		t.Fatalf("QueryBatched failed: %v", err)
	}

	if len(responses) != 2 {
		t.Errorf("Expected 2 responses, got %d", len(responses))
	}

	if responses[0] != "r1" || responses[1] != "r2" {
		t.Errorf("Expected ['r1', 'r2'], got: %v", responses)
	}

	if len(usages) != 2 {
		t.Errorf("Expected 2 token usages, got %d", len(usages))
	}
}

func TestIPCServerRecordsBlockedQuery(t *testing.T) {
	client := NewMockLLMClient()
	server, err := NewIPCServer(client, 0)
	if err != nil {
		t.Fatalf("Failed to create IPC server: %v", err)
	}
	defer func() { _ = server.Stop() }()

	execID := server.beginExecution(context.Background())
	defer server.endExecution(execID)
	resp := server.handleQueryBlocked(IPCMessage{
		Type:        MessageQueryBlocked,
		ID:          "blocked",
		ExecutionID: execID,
		Prompts:     []string{"one", "two"},
		Responses:   []string{"Error: one", "Error: two"},
	})
	if resp.Type != MessageResponse {
		t.Fatalf("handleQueryBlocked type = %s, want %s", resp.Type, MessageResponse)
	}
	if client.CallCount != 0 {
		t.Fatalf("client CallCount = %d, want no LLM calls", client.CallCount)
	}
	calls := server.GetCalls()
	if len(calls) != 2 {
		t.Fatalf("GetCalls() = %+v, want 2 blocked calls", calls)
	}
	if calls[0].Prompt != "one" || calls[0].Response != "Error: one" || calls[1].Prompt != "two" || calls[1].Response != "Error: two" {
		t.Fatalf("blocked calls = %+v", calls)
	}
}

func TestIPCServerQueryUsesExecutionContext(t *testing.T) {
	client := newContextDeadlineLLMClient()
	server, err := NewIPCServer(client, 0)
	if err != nil {
		t.Fatalf("Failed to create IPC server: %v", err)
	}
	defer func() { _ = server.Stop() }()

	execCtx, cancel := context.WithTimeout(context.Background(), 25*time.Millisecond)
	defer cancel()
	execID := server.beginExecution(execCtx)
	defer server.endExecution(execID)

	start := time.Now()
	resp := server.handleQuery(IPCMessage{Type: MessageQuery, ID: "q1", ExecutionID: execID, Prompt: "wait"})
	if time.Since(start) > time.Second {
		t.Fatal("IPC query did not return promptly after execution timeout")
	}
	if resp.Type != MessageError {
		t.Fatalf("response type = %s, want error", resp.Type)
	}

	select {
	case err := <-client.errCh:
		if !errors.Is(err, context.DeadlineExceeded) {
			t.Fatalf("query context error = %v, want deadline exceeded", err)
		}
	default:
		t.Fatal("LLM client did not observe execution context deadline")
	}
}

func TestIPCServerDropsCallsAfterExecutionEnds(t *testing.T) {
	client := newDelayedLLMClient()
	server, err := NewIPCServer(client, 0)
	if err != nil {
		t.Fatalf("Failed to create IPC server: %v", err)
	}
	defer func() { _ = server.Stop() }()

	execID := server.beginExecution(context.Background())
	done := make(chan IPCMessage, 1)
	go func() {
		done <- server.handleQuery(IPCMessage{Type: MessageQuery, ID: "q1", ExecutionID: execID, Prompt: "late"})
	}()

	select {
	case <-client.started:
	case <-time.After(time.Second):
		t.Fatal("LLM query did not start")
	}

	server.endExecution(execID)
	close(client.release)

	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("LLM query did not finish")
	}

	if calls := server.GetCalls(); len(calls) != 0 {
		t.Fatalf("late query should not be recorded after execution ended, calls=%+v", calls)
	}
}

func TestIPCServerRejectsQueryAfterExecutionEnds(t *testing.T) {
	client := NewMockLLMClient()
	server, err := NewIPCServer(client, 0)
	if err != nil {
		t.Fatalf("Failed to create IPC server: %v", err)
	}
	defer func() { _ = server.Stop() }()

	execID := server.beginExecution(context.Background())
	server.endExecution(execID)

	resp := server.handleQuery(IPCMessage{Type: MessageQuery, ID: "q1", Prompt: "late"})
	if resp.Type != MessageError {
		t.Fatalf("response type = %s, want error", resp.Type)
	}
	if !strings.Contains(resp.Error, "execution expired") {
		t.Fatalf("error = %q, want execution expired", resp.Error)
	}
	if client.CallCount != 0 {
		t.Fatalf("late query should not call client, calls=%d", client.CallCount)
	}
	if calls := server.GetCalls(); len(calls) != 0 {
		t.Fatalf("late query should not be recorded, calls=%+v", calls)
	}
}

func TestIPCServerRejectsQueryAfterExecutionContextExpires(t *testing.T) {
	client := NewMockLLMClient()
	server, err := NewIPCServer(client, 0)
	if err != nil {
		t.Fatalf("Failed to create IPC server: %v", err)
	}
	defer func() { _ = server.Stop() }()

	execCtx, cancel := context.WithCancel(context.Background())
	execID := server.beginExecution(execCtx)
	cancel()

	resp := server.handleQuery(IPCMessage{Type: MessageQuery, ID: "q1", ExecutionID: execID, Prompt: "late"})
	if resp.Type != MessageError {
		t.Fatalf("response type = %s, want error", resp.Type)
	}
	if !strings.Contains(resp.Error, "execution expired") {
		t.Fatalf("error = %q, want execution expired", resp.Error)
	}
	if client.CallCount != 0 {
		t.Fatalf("expired query should not call client, calls=%d", client.CallCount)
	}
	if calls := server.GetCalls(); len(calls) != 0 {
		t.Fatalf("expired query should not be recorded, calls=%+v", calls)
	}
}

func TestIPCServerRejectsOldExecutionIDDuringNewExecution(t *testing.T) {
	client := NewMockLLMClient()
	server, err := NewIPCServer(client, 0)
	if err != nil {
		t.Fatalf("Failed to create IPC server: %v", err)
	}
	defer func() { _ = server.Stop() }()

	oldExecID := server.beginExecution(context.Background())
	server.endExecution(oldExecID)
	newExecID := server.beginExecution(context.Background())
	defer server.endExecution(newExecID)

	resp := server.handleQuery(IPCMessage{Type: MessageQuery, ID: "q1", ExecutionID: oldExecID, Prompt: "stale"})
	if resp.Type != MessageError {
		t.Fatalf("response type = %s, want error", resp.Type)
	}
	if !strings.Contains(resp.Error, "execution expired") {
		t.Fatalf("error = %q, want execution expired", resp.Error)
	}
	if client.CallCount != 0 {
		t.Fatalf("stale execution query should not call client, calls=%d", client.CallCount)
	}
	if calls := server.GetCalls(); len(calls) != 0 {
		t.Fatalf("stale execution query should not be recorded, calls=%+v", calls)
	}
}

func TestContainerExecutorFinalMarker(t *testing.T) {
	exec := &ContainerExecutor{
		variables: make(map[string]any),
		finalTok:  "token",
	}

	exec.extractFinal("FINAL(fake)"+formatFinalMarker("token", "real"), "token")

	final, ok := exec.Final()
	if !ok {
		t.Fatal("expected final state")
	}
	if final != "real" {
		t.Fatalf("Final = %q, want real", final)
	}

	exec.ClearFinal()
	exec.extractFinal("FINAL(fake)", "token")
	if exec.HasFinal() {
		t.Fatal("fake stdout FINAL should not set final state")
	}
}

func TestContainerExecutorFinalMarkerStrippedAndRotated(t *testing.T) {
	exec := &ContainerExecutor{
		variables: make(map[string]any),
		finalTok:  "old-token",
	}

	stripped := stripFinalMarkers("before"+formatFinalMarker("old-token", "real")+"after\n", "old-token")
	if strings.Contains(stripped, finalMarkerPrefix) {
		t.Fatalf("stripFinalMarkers left marker in stdout: %q", stripped)
	}

	_, _ = exec.Execute(context.Background(), "")
	if exec.finalTok == "old-token" {
		t.Fatal("Execute should rotate container final token")
	}

	exec.extractFinal(formatFinalMarker("old-token", "forged"), exec.finalTok)
	if exec.HasFinal() {
		t.Fatal("old leaked final marker should not set final state after token rotation")
	}
}

func TestContainerExecutorTimeoutDoesNotSetFinal(t *testing.T) {
	tmpDir := t.TempDir()
	runtimePath := filepath.Join(tmpDir, "fake-runtime")
	script := `#!/bin/sh
mount_arg=""
for arg in "$@"; do
	case "$arg" in
		*:/workspace:ro)
			mount_arg="$arg"
			;;
	esac
done
workspace="${mount_arg%%:/workspace:ro}"
token="$(sed -n 's/.*finalToken := "\([^"]*\)".*/\1/p' "$workspace/main.go" | head -n 1)"
printf '\n__RLM_FINAL__%s__dGltZWQgb3V0\n' "$token"
sleep 2
`
	if err := os.WriteFile(runtimePath, []byte(script), 0755); err != nil {
		t.Fatalf("write fake runtime: %v", err)
	}

	exec := &ContainerExecutor{
		config: Config{
			Timeout:   25 * time.Millisecond,
			EnableIPC: false,
		},
		runtime:   runtimePath,
		variables: make(map[string]any),
	}

	result, err := exec.Execute(context.Background(), `FINAL("timed out")`)
	if err != nil {
		t.Fatalf("Execute failed: %v", err)
	}
	if !strings.Contains(result.Stderr, "execution timeout exceeded") {
		t.Fatalf("stderr = %q, want timeout", result.Stderr)
	}
	if exec.HasFinal() {
		t.Fatal("timeout/canceled container execution should not set final state")
	}
	if strings.Contains(result.Stdout, finalMarkerPrefix) {
		t.Fatalf("timeout stdout leaked final marker: %q", result.Stdout)
	}
}

func TestContainerExecutorExecuteClearsFinal(t *testing.T) {
	exec := &ContainerExecutor{
		variables: make(map[string]any),
		finalTok:  "token",
		finalSet:  true,
		finalVal:  "stale",
	}

	_, _ = exec.Execute(context.Background(), "")
	if exec.HasFinal() {
		t.Fatal("Execute should clear stale final state before running")
	}
}

func TestContainerExecutorStringContextIsRaw(t *testing.T) {
	exec := &ContainerExecutor{
		variables: make(map[string]any),
		finalTok:  "token",
	}
	if err := exec.LoadContext("container context data"); err != nil {
		t.Fatalf("LoadContext failed: %v", err)
	}

	code, err := exec.generateProgram("", 0)
	if err != nil {
		t.Fatalf("generateProgram failed: %v", err)
	}
	if !strings.Contains(code, `var context = "container context data"`) {
		t.Fatalf("generated string context should be raw, code=%s", code)
	}
	if strings.Contains(code, `var context = "\"container context data\""`) {
		t.Fatal("generated string context should not be JSON-quoted")
	}
	if !strings.Contains(code, "unicode.IsLetter") || !strings.Contains(code, "unicode.IsNumber") {
		t.Error("no-IPC generated context search should use contextindex-style tokenization")
	}
	if !strings.Contains(code, "runeByteOffsets") || !strings.Contains(code, "context[byteOffsets[start]:byteOffsets[end]]") {
		t.Error("no-IPC generated context chunking should be UTF-8 safe")
	}
	if !strings.Contains(code, "if startLine > len(lines)") {
		t.Error("no-IPC generated GetContext should clamp startLine past EOF")
	}
}

func TestContainerGeneratedContextChunksMatchContextIndex(t *testing.T) {
	raw := generatedChunkingFixture()
	containerExec := &ContainerExecutor{
		variables: make(map[string]any),
		finalTok:  "token",
	}
	if err := containerExec.LoadContext(raw); err != nil {
		t.Fatalf("LoadContext failed: %v", err)
	}

	code, err := containerExec.generateProgram(generatedChunkDumpCode(), 0)
	if err != nil {
		t.Fatalf("generateProgram failed: %v", err)
	}
	output := runGeneratedProgram(t, code)
	assertGeneratedChunksMatchContextIndex(t, output, raw)
}

func TestIPCGeneratedContextChunksMatchContextIndex(t *testing.T) {
	raw := generatedChunkingFixture()
	server, err := NewIPCServer(NewMockLLMClient(), 0)
	if err != nil {
		t.Fatalf("Failed to create IPC server: %v", err)
	}
	defer func() { _ = server.Stop() }()
	server.Start()

	code := GenerateContainerRLMCode(server.Address(), "token", "1", "0")
	code += "\nvar context = " + strconv.Quote(raw) + "\n"
	code += "func main() {\n" + generatedChunkDumpCode() + "\n}\n"

	output := runGeneratedProgram(t, code)
	assertGeneratedChunksMatchContextIndex(t, output, raw)
}

func generatedChunkingFixture() string {
	return strings.Repeat("a", 3999) + "🙂" + string([]byte{0xff}) + strings.Repeat("b", 201)
}

func generatedChunkDumpCode() string {
	return `
fmt.Printf("count=%d\n", ChunkCount())
for i := 0; i < ChunkCount(); i++ {
	fmt.Printf("chunk%d=%x\n", i, []byte(GetChunk(i)))
}
`
}

func runGeneratedProgram(t *testing.T, code string) string {
	t.Helper()

	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "main.go"), []byte(code), 0644); err != nil {
		t.Fatalf("write generated main.go: %v", err)
	}
	goMod := "module generated\n\ngo 1.23\n"
	if err := os.WriteFile(filepath.Join(dir, "go.mod"), []byte(goMod), 0644); err != nil {
		t.Fatalf("write generated go.mod: %v", err)
	}

	cmd := osexec.Command("go", "run", ".")
	cmd.Dir = dir
	output, err := cmd.CombinedOutput()
	if err != nil {
		t.Fatalf("generated program failed: %v\n%s", err, output)
	}
	return string(output)
}

func assertGeneratedChunksMatchContextIndex(t *testing.T, output, raw string) {
	t.Helper()

	idx := contextindex.New(raw)
	countLine := fmt.Sprintf("count=%d", idx.ChunkCount())
	if !strings.Contains(output, countLine) {
		t.Fatalf("generated output = %q, want %q", output, countLine)
	}
	for i := 0; i < idx.ChunkCount(); i++ {
		chunk, ok := idx.GetChunk(i)
		if !ok {
			t.Fatalf("contextindex missing chunk %d", i)
		}
		chunkLine := fmt.Sprintf("chunk%d=%x", i, []byte(chunk))
		if !strings.Contains(output, chunkLine) {
			t.Fatalf("generated output = %q, want %q", output, chunkLine)
		}
	}
}

func TestContainerExecutorLoadContextNilIsEmpty(t *testing.T) {
	exec := &ContainerExecutor{
		variables: make(map[string]any),
		finalTok:  "token",
	}
	if err := exec.LoadContext(nil); err != nil {
		t.Fatalf("LoadContext(nil) failed: %v", err)
	}
	if info := exec.ContextInfo(); info != "context not loaded" {
		t.Fatalf("ContextInfo() = %q, want context not loaded", info)
	}
	code, err := exec.generateProgram("", 0)
	if err != nil {
		t.Fatalf("generateProgram failed: %v", err)
	}
	if !strings.Contains(code, `var context = ""`) {
		t.Fatalf("generated nil context should be empty string, code=%s", code)
	}
}

func TestGenerateContainerRLMCode(t *testing.T) {
	code := GenerateContainerRLMCode("host.containers.internal:12345", "token", "1", "5")

	// Verify the code contains expected elements
	if !strings.Contains(code, "package main") {
		t.Error("Expected 'package main' in generated code")
	}

	if !strings.Contains(code, "func Query(prompt string) string") {
		t.Error("Expected Query function in generated code")
	}

	if !strings.Contains(code, "func QueryBatched(prompts []string) []string") {
		t.Error("Expected QueryBatched function in generated code")
	}

	if !strings.Contains(code, "const maxFullContextQueryChars = 5") {
		t.Error("Expected generated full-context query limit")
	}

	if !strings.Contains(code, `fullContextQueryBlocked("Query")`) || !strings.Contains(code, `fullContextQueryBlocked("QueryBatched")`) {
		t.Error("Expected generated Query/QueryBatched guardrails")
	}

	if !strings.Contains(code, "messageQueryBlocked") || !strings.Contains(code, "recordBlockedQueries") {
		t.Error("Expected generated blocked-query recording")
	}

	if !strings.Contains(code, "func FindRelevant(query string, topK int) []string") {
		t.Error("Expected FindRelevant function in generated code")
	}

	if !strings.Contains(code, "unicode.IsLetter") || !strings.Contains(code, "unicode.IsNumber") {
		t.Error("Expected generated context search tokenization to match contextindex")
	}

	if !strings.Contains(code, "runeByteOffsets") || !strings.Contains(code, "context[byteOffsets[start]:byteOffsets[end]]") {
		t.Error("Expected generated context chunking to be UTF-8 safe")
	}

	if !strings.Contains(code, "func GetContext(startLine, endLine int) string") {
		t.Error("Expected GetContext function in generated code")
	}

	if !strings.Contains(code, "if startLine > len(lines)") {
		t.Error("Expected generated GetContext to clamp startLine past EOF")
	}

	if !strings.Contains(code, "func QueryRaw(prompt string) string") {
		t.Error("Expected QueryRaw function in generated code")
	}

	if !strings.Contains(code, finalMarkerPrefix) {
		t.Error("Expected private FINAL marker in generated code")
	}

	if strings.Contains(code, "const finalToken") {
		t.Error("generated code should not expose final token as package-level constant")
	}

	if !strings.Contains(code, "host.containers.internal:12345") {
		t.Error("Expected IPC address in generated code")
	}
}

func TestDefaultConfig(t *testing.T) {
	cfg := DefaultConfig()

	if cfg.Backend != BackendAuto {
		t.Errorf("Expected BackendAuto, got %s", cfg.Backend)
	}

	if cfg.Image != "golang:1.23-alpine" {
		t.Errorf("Expected 'golang:1.23-alpine', got %s", cfg.Image)
	}

	if cfg.Memory != "512m" {
		t.Errorf("Expected '512m', got %s", cfg.Memory)
	}

	if cfg.CPUs != 1.0 {
		t.Errorf("Expected 1.0, got %f", cfg.CPUs)
	}

	if cfg.Timeout != 60*time.Second {
		t.Errorf("Expected 60s, got %v", cfg.Timeout)
	}

	if cfg.NetworkMode != NetworkNone {
		t.Errorf("Expected NetworkNone, got %s", cfg.NetworkMode)
	}
}

func TestLocalExecutorMinMaxHelpers(t *testing.T) {
	client := NewMockLLMClient()
	cfg := DefaultConfig()
	cfg.Backend = BackendLocal

	exec, err := NewLocalExecutor(client, cfg)
	if err != nil {
		t.Fatalf("Failed to create local executor: %v", err)
	}
	defer func() { _ = exec.Close() }()

	// Test min/max helper functions
	result, err := exec.Execute(context.Background(), `
		a := min(5, 3)
		b := max(5, 3)
		fmt.Printf("min=%d max=%d", a, b)
	`)
	if err != nil {
		t.Fatalf("Execute failed: %v", err)
	}

	if !strings.Contains(result.Stdout, "min=3 max=5") {
		t.Errorf("Expected 'min=3 max=5' in output, got: %s", result.Stdout)
	}
}

func TestLocalExecutorMapContext(t *testing.T) {
	client := NewMockLLMClient()
	cfg := DefaultConfig()
	cfg.Backend = BackendLocal

	exec, err := NewLocalExecutor(client, cfg)
	if err != nil {
		t.Fatalf("Failed to create local executor: %v", err)
	}
	defer func() { _ = exec.Close() }()

	// Load map context (stored as JSON string)
	err = exec.LoadContext(map[string]any{
		"key1": "value1",
		"key2": 42,
	})
	if err != nil {
		t.Fatalf("LoadContext failed: %v", err)
	}

	// Execute code that uses the context as a JSON string
	result, err := exec.Execute(context.Background(), `
		// Context is stored as JSON string
		fmt.Printf("context=%s", context)
	`)
	if err != nil {
		t.Fatalf("Execute failed: %v", err)
	}

	// Context should contain the JSON representation
	if !strings.Contains(result.Stdout, "key1") || !strings.Contains(result.Stdout, "value1") {
		t.Errorf("Expected context JSON to contain 'key1' and 'value1', got: %s", result.Stdout)
	}
}

func TestExecutorInterface(t *testing.T) {
	// Verify that both executors implement the Executor interface
	var _ Executor = (*LocalExecutor)(nil)
	var _ Executor = (*ContainerExecutor)(nil)
}
