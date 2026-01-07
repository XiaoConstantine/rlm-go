// Package rlm provides recursive client adapter for multi-depth RLM.
package rlm

import (
	"context"
	"fmt"
	"sync"
	"sync/atomic"

	"github.com/XiaoConstantine/rlm-go/pkg/core"
	"github.com/XiaoConstantine/rlm-go/pkg/repl"
	"github.com/google/uuid"
)

// RecursiveClientAdapter wraps an RLM instance to provide recursive query capabilities.
// It implements repl.RecursiveLLMClient and manages nested RLM executions.
type RecursiveClientAdapter struct {
	// rlm is the parent RLM instance used for recursive queries.
	rlm *RLM

	// recursionContext tracks the current recursion state.
	recursionContext *core.RecursionContext

	// tokenStats aggregates token usage across all recursion levels.
	tokenStats *RecursiveTokenStats

	// onRecursiveQuery is called when a recursive query starts.
	onRecursiveQuery func(depth int, prompt string)

	// contextPayload is the context to pass to nested RLM calls.
	contextPayload any
}

// RecursiveTokenStats aggregates token usage across recursion levels.
type RecursiveTokenStats struct {
	mu               sync.Mutex
	PromptTokens     int64
	CompletionTokens int64
	TotalCalls       int64
	CallsByDepth     map[int]int64
}

// NewRecursiveTokenStats creates a new token stats tracker.
func NewRecursiveTokenStats() *RecursiveTokenStats {
	return &RecursiveTokenStats{
		CallsByDepth: make(map[int]int64),
	}
}

// Add adds token usage from a call at the specified depth.
func (s *RecursiveTokenStats) Add(depth, promptTokens, completionTokens int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.PromptTokens += int64(promptTokens)
	s.CompletionTokens += int64(completionTokens)
	s.TotalCalls++
	s.CallsByDepth[depth]++
}

// GetTotals returns the total token counts.
func (s *RecursiveTokenStats) GetTotals() (promptTokens, completionTokens, totalCalls int64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.PromptTokens, s.CompletionTokens, s.TotalCalls
}

// NewRecursiveClientAdapter creates a new recursive client adapter.
func NewRecursiveClientAdapter(
	rlm *RLM,
	recursionCtx *core.RecursionContext,
	tokenStats *RecursiveTokenStats,
	contextPayload any,
) *RecursiveClientAdapter {
	adapter := &RecursiveClientAdapter{
		rlm:              rlm,
		recursionContext: recursionCtx,
		tokenStats:       tokenStats,
		contextPayload:   contextPayload,
	}

	// Copy the callback from config if available
	if rlm.config.Recursion != nil && rlm.config.Recursion.OnRecursiveQuery != nil {
		adapter.onRecursiveQuery = rlm.config.Recursion.OnRecursiveQuery
	}

	return adapter
}

// Query performs a standard (non-recursive) sub-LLM query.
func (a *RecursiveClientAdapter) Query(ctx context.Context, prompt string) (repl.QueryResponse, error) {
	// Delegate to the underlying replClient
	return a.rlm.replClient.Query(ctx, prompt)
}

// QueryBatched performs concurrent standard (non-recursive) sub-LLM queries.
func (a *RecursiveClientAdapter) QueryBatched(ctx context.Context, prompts []string) ([]repl.QueryResponse, error) {
	// Delegate to the underlying replClient
	return a.rlm.replClient.QueryBatched(ctx, prompts)
}

// QueryWithRLM performs a recursive RLM query, spawning a nested RLM execution.
func (a *RecursiveClientAdapter) QueryWithRLM(ctx context.Context, prompt string, depth int) (repl.QueryResponse, error) {
	// Check if we can recurse
	if !a.recursionContext.CanRecurse() {
		return repl.QueryResponse{
			Response: fmt.Sprintf("Error: %v", &core.DepthExceededError{
				CurrentDepth: a.recursionContext.CurrentDepth,
				MaxDepth:     a.recursionContext.MaxDepth,
				Prompt:       prompt,
			}),
		}, nil
	}

	// Notify callback if set
	if a.onRecursiveQuery != nil {
		a.onRecursiveQuery(a.recursionContext.CurrentDepth+1, prompt)
	}

	// Generate a unique ID for this recursive call
	callID := uuid.New().String()

	// Create child recursion context
	childCtx := a.recursionContext.Child(callID)

	// Perform the recursive RLM call
	result, err := a.rlm.CompleteWithRecursion(ctx, a.contextPayload, prompt, childCtx, a.tokenStats)
	if err != nil {
		return repl.QueryResponse{
			Response: fmt.Sprintf("Error: %v", err),
		}, nil
	}

	// Return the result
	return repl.QueryResponse{
		Response:         result.Response,
		PromptTokens:     result.Usage.PromptTokens,
		CompletionTokens: result.Usage.CompletionTokens,
	}, nil
}

// CurrentDepth returns the current recursion depth.
func (a *RecursiveClientAdapter) CurrentDepth() int {
	return a.recursionContext.CurrentDepth
}

// MaxDepth returns the maximum allowed recursion depth.
func (a *RecursiveClientAdapter) MaxDepth() int {
	return a.recursionContext.MaxDepth
}

// RecursiveCallTracker tracks recursive calls for logging and debugging.
type RecursiveCallTracker struct {
	calls     []RecursiveCallInfo
	mu        sync.Mutex
	callCount int64
}

// RecursiveCallInfo contains information about a recursive call.
type RecursiveCallInfo struct {
	ID               string
	ParentID         string
	Depth            int
	Prompt           string
	Response         string
	PromptTokens     int
	CompletionTokens int
	Iterations       int
	Success          bool
	Error            string
}

// NewRecursiveCallTracker creates a new call tracker.
func NewRecursiveCallTracker() *RecursiveCallTracker {
	return &RecursiveCallTracker{
		calls: make([]RecursiveCallInfo, 0),
	}
}

// Track records a recursive call.
func (t *RecursiveCallTracker) Track(info RecursiveCallInfo) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.calls = append(t.calls, info)
	atomic.AddInt64(&t.callCount, 1)
}

// GetCalls returns all tracked calls.
func (t *RecursiveCallTracker) GetCalls() []RecursiveCallInfo {
	t.mu.Lock()
	defer t.mu.Unlock()
	result := make([]RecursiveCallInfo, len(t.calls))
	copy(result, t.calls)
	return result
}

// GetCallCount returns the total number of calls tracked.
func (t *RecursiveCallTracker) GetCallCount() int64 {
	return atomic.LoadInt64(&t.callCount)
}

// GetCallsByDepth returns calls grouped by depth.
func (t *RecursiveCallTracker) GetCallsByDepth() map[int][]RecursiveCallInfo {
	t.mu.Lock()
	defer t.mu.Unlock()
	result := make(map[int][]RecursiveCallInfo)
	for _, call := range t.calls {
		result[call.Depth] = append(result[call.Depth], call)
	}
	return result
}

// Clear removes all tracked calls.
func (t *RecursiveCallTracker) Clear() {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.calls = make([]RecursiveCallInfo, 0)
	atomic.StoreInt64(&t.callCount, 0)
}
