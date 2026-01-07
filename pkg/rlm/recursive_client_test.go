package rlm

import (
	"context"
	"errors"
	"sync"
	"testing"

	"github.com/XiaoConstantine/rlm-go/pkg/core"
	"github.com/XiaoConstantine/rlm-go/pkg/repl"
)

// TestNewRecursiveTokenStats verifies RecursiveTokenStats initialization
func TestNewRecursiveTokenStats(t *testing.T) {
	stats := NewRecursiveTokenStats()

	if stats == nil {
		t.Fatal("expected non-nil RecursiveTokenStats")
	}
	if stats.PromptTokens != 0 {
		t.Errorf("PromptTokens = %d, want 0", stats.PromptTokens)
	}
	if stats.CompletionTokens != 0 {
		t.Errorf("CompletionTokens = %d, want 0", stats.CompletionTokens)
	}
	if stats.TotalCalls != 0 {
		t.Errorf("TotalCalls = %d, want 0", stats.TotalCalls)
	}
	if stats.CallsByDepth == nil {
		t.Error("CallsByDepth should be initialized")
	}
}

// TestRecursiveTokenStats_Add verifies token aggregation
func TestRecursiveTokenStats_Add(t *testing.T) {
	stats := NewRecursiveTokenStats()

	// Add tokens at different depths
	stats.Add(0, 100, 50)
	stats.Add(0, 80, 40)
	stats.Add(1, 60, 30)
	stats.Add(2, 40, 20)

	if stats.PromptTokens != 280 {
		t.Errorf("PromptTokens = %d, want 280", stats.PromptTokens)
	}
	if stats.CompletionTokens != 140 {
		t.Errorf("CompletionTokens = %d, want 140", stats.CompletionTokens)
	}
	if stats.TotalCalls != 4 {
		t.Errorf("TotalCalls = %d, want 4", stats.TotalCalls)
	}

	// Verify calls by depth
	if stats.CallsByDepth[0] != 2 {
		t.Errorf("CallsByDepth[0] = %d, want 2", stats.CallsByDepth[0])
	}
	if stats.CallsByDepth[1] != 1 {
		t.Errorf("CallsByDepth[1] = %d, want 1", stats.CallsByDepth[1])
	}
	if stats.CallsByDepth[2] != 1 {
		t.Errorf("CallsByDepth[2] = %d, want 1", stats.CallsByDepth[2])
	}
}

// TestRecursiveTokenStats_GetTotals verifies totals retrieval
func TestRecursiveTokenStats_GetTotals(t *testing.T) {
	stats := NewRecursiveTokenStats()

	stats.Add(0, 100, 50)
	stats.Add(1, 80, 40)

	promptTokens, completionTokens, totalCalls := stats.GetTotals()

	if promptTokens != 180 {
		t.Errorf("promptTokens = %d, want 180", promptTokens)
	}
	if completionTokens != 90 {
		t.Errorf("completionTokens = %d, want 90", completionTokens)
	}
	if totalCalls != 2 {
		t.Errorf("totalCalls = %d, want 2", totalCalls)
	}
}

// TestRecursiveTokenStats_ConcurrentAccess verifies thread safety
func TestRecursiveTokenStats_ConcurrentAccess(t *testing.T) {
	stats := NewRecursiveTokenStats()

	var wg sync.WaitGroup
	numGoroutines := 100

	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(depth int) {
			defer wg.Done()
			stats.Add(depth%3, 10, 5)
		}(i)
	}

	wg.Wait()

	promptTokens, completionTokens, totalCalls := stats.GetTotals()

	if promptTokens != 1000 {
		t.Errorf("promptTokens = %d, want 1000", promptTokens)
	}
	if completionTokens != 500 {
		t.Errorf("completionTokens = %d, want 500", completionTokens)
	}
	if totalCalls != int64(numGoroutines) {
		t.Errorf("totalCalls = %d, want %d", totalCalls, numGoroutines)
	}
}

// TestNewRecursiveClientAdapter verifies adapter creation
func TestNewRecursiveClientAdapter(t *testing.T) {
	client := &mockLLMClient{
		completeFunc: func(ctx context.Context, messages []core.Message) (core.LLMResponse, error) {
			return core.LLMResponse{Content: "test"}, nil
		},
	}
	replClient := &mockREPLClient{}

	rlm := New(client, replClient, WithMaxRecursionDepth(3))

	recursionCtx := core.NewRecursionContext(3)
	tokenStats := NewRecursiveTokenStats()

	adapter := NewRecursiveClientAdapter(rlm, recursionCtx, tokenStats, "test context")

	if adapter == nil {
		t.Fatal("expected non-nil adapter")
	}
	if adapter.rlm != rlm {
		t.Error("rlm not set correctly")
	}
	if adapter.recursionContext != recursionCtx {
		t.Error("recursionContext not set correctly")
	}
	if adapter.tokenStats != tokenStats {
		t.Error("tokenStats not set correctly")
	}
	if adapter.contextPayload != "test context" {
		t.Error("contextPayload not set correctly")
	}
}

// TestRecursiveClientAdapter_CurrentDepth verifies current depth reporting
func TestRecursiveClientAdapter_CurrentDepth(t *testing.T) {
	client := &mockLLMClient{}
	replClient := &mockREPLClient{}
	rlm := New(client, replClient)

	tests := []struct {
		name         string
		currentDepth int
	}{
		{"depth 0", 0},
		{"depth 1", 1},
		{"depth 2", 2},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			recursionCtx := &core.RecursionContext{
				CurrentDepth: tt.currentDepth,
				MaxDepth:     3,
			}
			adapter := NewRecursiveClientAdapter(rlm, recursionCtx, NewRecursiveTokenStats(), nil)

			if adapter.CurrentDepth() != tt.currentDepth {
				t.Errorf("CurrentDepth() = %d, want %d", adapter.CurrentDepth(), tt.currentDepth)
			}
		})
	}
}

// TestRecursiveClientAdapter_MaxDepth verifies max depth reporting
func TestRecursiveClientAdapter_MaxDepth(t *testing.T) {
	client := &mockLLMClient{}
	replClient := &mockREPLClient{}
	rlm := New(client, replClient)

	tests := []struct {
		name     string
		maxDepth int
	}{
		{"max depth 1", 1},
		{"max depth 3", 3},
		{"max depth 5", 5},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			recursionCtx := &core.RecursionContext{
				CurrentDepth: 0,
				MaxDepth:     tt.maxDepth,
			}
			adapter := NewRecursiveClientAdapter(rlm, recursionCtx, NewRecursiveTokenStats(), nil)

			if adapter.MaxDepth() != tt.maxDepth {
				t.Errorf("MaxDepth() = %d, want %d", adapter.MaxDepth(), tt.maxDepth)
			}
		})
	}
}

// TestRecursiveClientAdapter_Query verifies standard query delegation
func TestRecursiveClientAdapter_Query(t *testing.T) {
	client := &mockLLMClient{}
	replClient := &mockREPLClient{
		queryFunc: func(ctx context.Context, prompt string) (repl.QueryResponse, error) {
			return repl.QueryResponse{
				Response:         "standard response: " + prompt,
				PromptTokens:     15,
				CompletionTokens: 8,
			}, nil
		},
	}
	rlm := New(client, replClient)
	recursionCtx := core.NewRecursionContext(2)
	adapter := NewRecursiveClientAdapter(rlm, recursionCtx, NewRecursiveTokenStats(), nil)

	resp, err := adapter.Query(context.Background(), "test prompt")
	if err != nil {
		t.Fatalf("Query() error: %v", err)
	}

	if resp.Response != "standard response: test prompt" {
		t.Errorf("Response = %q, want %q", resp.Response, "standard response: test prompt")
	}
	if resp.PromptTokens != 15 {
		t.Errorf("PromptTokens = %d, want 15", resp.PromptTokens)
	}
	if resp.CompletionTokens != 8 {
		t.Errorf("CompletionTokens = %d, want 8", resp.CompletionTokens)
	}
}

// TestRecursiveClientAdapter_QueryBatched verifies batched query delegation
func TestRecursiveClientAdapter_QueryBatched(t *testing.T) {
	client := &mockLLMClient{}
	replClient := &mockREPLClient{
		batchFunc: func(ctx context.Context, prompts []string) ([]repl.QueryResponse, error) {
			results := make([]repl.QueryResponse, len(prompts))
			for i, p := range prompts {
				results[i] = repl.QueryResponse{
					Response:         "batch: " + p,
					PromptTokens:     10,
					CompletionTokens: 5,
				}
			}
			return results, nil
		},
	}
	rlm := New(client, replClient)
	recursionCtx := core.NewRecursionContext(2)
	adapter := NewRecursiveClientAdapter(rlm, recursionCtx, NewRecursiveTokenStats(), nil)

	prompts := []string{"prompt1", "prompt2", "prompt3"}
	results, err := adapter.QueryBatched(context.Background(), prompts)
	if err != nil {
		t.Fatalf("QueryBatched() error: %v", err)
	}

	if len(results) != 3 {
		t.Fatalf("expected 3 results, got %d", len(results))
	}

	for i, r := range results {
		expected := "batch: " + prompts[i]
		if r.Response != expected {
			t.Errorf("results[%d].Response = %q, want %q", i, r.Response, expected)
		}
	}
}

// TestRecursiveClientAdapter_QueryWithRLM_DepthExceeded verifies depth limiting
func TestRecursiveClientAdapter_QueryWithRLM_DepthExceeded(t *testing.T) {
	client := &mockLLMClient{}
	replClient := &mockREPLClient{}
	rlm := New(client, replClient)

	// Create context at max depth
	recursionCtx := &core.RecursionContext{
		CurrentDepth: 2,
		MaxDepth:     2,
	}
	adapter := NewRecursiveClientAdapter(rlm, recursionCtx, NewRecursiveTokenStats(), nil)

	resp, err := adapter.QueryWithRLM(context.Background(), "test prompt", 3)

	// Should not return an error, but response should contain error message
	if err != nil {
		t.Fatalf("QueryWithRLM() error: %v", err)
	}

	if !contains(resp.Response, "depth exceeded") && !contains(resp.Response, "Error:") {
		t.Errorf("expected depth exceeded error in response, got: %q", resp.Response)
	}
}

// TestRecursiveClientAdapter_OnRecursiveQueryCallback verifies callback invocation
func TestRecursiveClientAdapter_OnRecursiveQueryCallback(t *testing.T) {
	var callbackDepth int
	var callbackPrompt string

	client := &mockLLMClient{
		completeFunc: func(ctx context.Context, messages []core.Message) (core.LLMResponse, error) {
			return core.LLMResponse{Content: "FINAL(done)", PromptTokens: 10, CompletionTokens: 5}, nil
		},
	}
	replClient := &mockREPLClient{}

	callback := func(depth int, prompt string) {
		callbackDepth = depth
		callbackPrompt = prompt
	}

	rlm := New(client, replClient,
		WithMaxRecursionDepth(3),
		WithRecursionCallback(callback),
	)

	recursionCtx := &core.RecursionContext{
		CurrentDepth: 0,
		MaxDepth:     3,
	}
	adapter := NewRecursiveClientAdapter(rlm, recursionCtx, NewRecursiveTokenStats(), "test context")

	_, _ = adapter.QueryWithRLM(context.Background(), "test prompt", 1)

	if callbackDepth != 1 {
		t.Errorf("callback depth = %d, want 1", callbackDepth)
	}
	if callbackPrompt != "test prompt" {
		t.Errorf("callback prompt = %q, want %q", callbackPrompt, "test prompt")
	}
}

// TestNewRecursiveCallTracker verifies call tracker initialization
func TestNewRecursiveCallTracker(t *testing.T) {
	tracker := NewRecursiveCallTracker()

	if tracker == nil {
		t.Fatal("expected non-nil tracker")
	}
	if len(tracker.calls) != 0 {
		t.Errorf("expected empty calls, got %d", len(tracker.calls))
	}
	if tracker.GetCallCount() != 0 {
		t.Errorf("GetCallCount() = %d, want 0", tracker.GetCallCount())
	}
}

// TestRecursiveCallTracker_Track verifies call tracking
func TestRecursiveCallTracker_Track(t *testing.T) {
	tracker := NewRecursiveCallTracker()

	tracker.Track(RecursiveCallInfo{
		ID:               "call-1",
		ParentID:         "",
		Depth:            0,
		Prompt:           "prompt 1",
		Response:         "response 1",
		PromptTokens:     100,
		CompletionTokens: 50,
		Iterations:       3,
		Success:          true,
	})

	tracker.Track(RecursiveCallInfo{
		ID:               "call-2",
		ParentID:         "call-1",
		Depth:            1,
		Prompt:           "prompt 2",
		Response:         "response 2",
		PromptTokens:     80,
		CompletionTokens: 40,
		Iterations:       2,
		Success:          true,
	})

	if tracker.GetCallCount() != 2 {
		t.Errorf("GetCallCount() = %d, want 2", tracker.GetCallCount())
	}

	calls := tracker.GetCalls()
	if len(calls) != 2 {
		t.Fatalf("GetCalls() returned %d calls, want 2", len(calls))
	}

	// Verify first call
	if calls[0].ID != "call-1" {
		t.Errorf("calls[0].ID = %q, want %q", calls[0].ID, "call-1")
	}
	if calls[0].Depth != 0 {
		t.Errorf("calls[0].Depth = %d, want 0", calls[0].Depth)
	}

	// Verify second call
	if calls[1].ParentID != "call-1" {
		t.Errorf("calls[1].ParentID = %q, want %q", calls[1].ParentID, "call-1")
	}
	if calls[1].Depth != 1 {
		t.Errorf("calls[1].Depth = %d, want 1", calls[1].Depth)
	}
}

// TestRecursiveCallTracker_GetCallsByDepth verifies depth-based retrieval
func TestRecursiveCallTracker_GetCallsByDepth(t *testing.T) {
	tracker := NewRecursiveCallTracker()

	// Add calls at different depths
	tracker.Track(RecursiveCallInfo{ID: "d0-1", Depth: 0})
	tracker.Track(RecursiveCallInfo{ID: "d0-2", Depth: 0})
	tracker.Track(RecursiveCallInfo{ID: "d1-1", Depth: 1})
	tracker.Track(RecursiveCallInfo{ID: "d2-1", Depth: 2})
	tracker.Track(RecursiveCallInfo{ID: "d2-2", Depth: 2})

	byDepth := tracker.GetCallsByDepth()

	if len(byDepth[0]) != 2 {
		t.Errorf("depth 0 calls = %d, want 2", len(byDepth[0]))
	}
	if len(byDepth[1]) != 1 {
		t.Errorf("depth 1 calls = %d, want 1", len(byDepth[1]))
	}
	if len(byDepth[2]) != 2 {
		t.Errorf("depth 2 calls = %d, want 2", len(byDepth[2]))
	}
}

// TestRecursiveCallTracker_Clear verifies clearing tracked calls
func TestRecursiveCallTracker_Clear(t *testing.T) {
	tracker := NewRecursiveCallTracker()

	tracker.Track(RecursiveCallInfo{ID: "call-1"})
	tracker.Track(RecursiveCallInfo{ID: "call-2"})

	if tracker.GetCallCount() != 2 {
		t.Fatalf("expected 2 calls before clear, got %d", tracker.GetCallCount())
	}

	tracker.Clear()

	if tracker.GetCallCount() != 0 {
		t.Errorf("GetCallCount() after clear = %d, want 0", tracker.GetCallCount())
	}

	calls := tracker.GetCalls()
	if len(calls) != 0 {
		t.Errorf("GetCalls() after clear returned %d calls, want 0", len(calls))
	}
}

// TestRecursiveCallTracker_ConcurrentAccess verifies thread safety
func TestRecursiveCallTracker_ConcurrentAccess(t *testing.T) {
	tracker := NewRecursiveCallTracker()

	var wg sync.WaitGroup
	numGoroutines := 100

	for i := 0; i < numGoroutines; i++ {
		wg.Add(1)
		go func(n int) {
			defer wg.Done()
			tracker.Track(RecursiveCallInfo{
				ID:    "call-" + string(rune(n)),
				Depth: n % 3,
			})
		}(i)
	}

	wg.Wait()

	if tracker.GetCallCount() != int64(numGoroutines) {
		t.Errorf("GetCallCount() = %d, want %d", tracker.GetCallCount(), numGoroutines)
	}
}

// TestRecursiveCallInfo_Fields verifies RecursiveCallInfo structure
func TestRecursiveCallInfo_Fields(t *testing.T) {
	info := RecursiveCallInfo{
		ID:               "test-id",
		ParentID:         "parent-id",
		Depth:            2,
		Prompt:           "test prompt",
		Response:         "test response",
		PromptTokens:     100,
		CompletionTokens: 50,
		Iterations:       5,
		Success:          true,
		Error:            "",
	}

	if info.ID != "test-id" {
		t.Errorf("ID = %q, want %q", info.ID, "test-id")
	}
	if info.ParentID != "parent-id" {
		t.Errorf("ParentID = %q, want %q", info.ParentID, "parent-id")
	}
	if info.Depth != 2 {
		t.Errorf("Depth = %d, want 2", info.Depth)
	}
	if info.Prompt != "test prompt" {
		t.Errorf("Prompt = %q, want %q", info.Prompt, "test prompt")
	}
	if info.Response != "test response" {
		t.Errorf("Response = %q, want %q", info.Response, "test response")
	}
	if info.PromptTokens != 100 {
		t.Errorf("PromptTokens = %d, want 100", info.PromptTokens)
	}
	if info.CompletionTokens != 50 {
		t.Errorf("CompletionTokens = %d, want 50", info.CompletionTokens)
	}
	if info.Iterations != 5 {
		t.Errorf("Iterations = %d, want 5", info.Iterations)
	}
	if !info.Success {
		t.Error("Success should be true")
	}
	if info.Error != "" {
		t.Errorf("Error = %q, want empty string", info.Error)
	}
}

// TestRecursiveCallInfo_WithError verifies error tracking
func TestRecursiveCallInfo_WithError(t *testing.T) {
	info := RecursiveCallInfo{
		ID:       "error-call",
		Success:  false,
		Error:    "simulated error",
	}

	if info.Success {
		t.Error("Success should be false for error call")
	}
	if info.Error != "simulated error" {
		t.Errorf("Error = %q, want %q", info.Error, "simulated error")
	}
}

// TestRecursiveClientAdapter_QueryError verifies error handling in Query
func TestRecursiveClientAdapter_QueryError(t *testing.T) {
	client := &mockLLMClient{}
	replClient := &mockREPLClient{
		queryFunc: func(ctx context.Context, prompt string) (repl.QueryResponse, error) {
			return repl.QueryResponse{}, errors.New("query failed")
		},
	}
	rlm := New(client, replClient)
	recursionCtx := core.NewRecursionContext(2)
	adapter := NewRecursiveClientAdapter(rlm, recursionCtx, NewRecursiveTokenStats(), nil)

	_, err := adapter.Query(context.Background(), "test")
	if err == nil {
		t.Error("expected error from Query")
	}
}

// TestRecursiveClientAdapter_QueryBatchedError verifies error handling in QueryBatched
func TestRecursiveClientAdapter_QueryBatchedError(t *testing.T) {
	client := &mockLLMClient{}
	replClient := &mockREPLClient{
		batchFunc: func(ctx context.Context, prompts []string) ([]repl.QueryResponse, error) {
			return nil, errors.New("batch failed")
		},
	}
	rlm := New(client, replClient)
	recursionCtx := core.NewRecursionContext(2)
	adapter := NewRecursiveClientAdapter(rlm, recursionCtx, NewRecursiveTokenStats(), nil)

	_, err := adapter.QueryBatched(context.Background(), []string{"p1", "p2"})
	if err == nil {
		t.Error("expected error from QueryBatched")
	}
}

// Helper function
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
