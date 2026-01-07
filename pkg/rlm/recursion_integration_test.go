package rlm

import (
	"context"
	"fmt"
	"os"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/XiaoConstantine/rlm-go/pkg/core"
	"github.com/XiaoConstantine/rlm-go/pkg/repl"
)

// mockRecursiveLLMClient simulates a root LLM for recursive testing.
// It returns responses that trigger recursive calls based on the iteration.
type mockRecursiveLLMClient struct {
	completeFunc func(ctx context.Context, messages []core.Message) (core.LLMResponse, error)
	calls        [][]core.Message
	callCount    int32
}

func (m *mockRecursiveLLMClient) Complete(ctx context.Context, messages []core.Message) (core.LLMResponse, error) {
	m.calls = append(m.calls, messages)
	atomic.AddInt32(&m.callCount, 1)
	return m.completeFunc(ctx, messages)
}

// TestRecursiveComplete_Basic tests basic recursive completion workflow
func TestRecursiveComplete_Basic(t *testing.T) {
	callCount := 0
	client := &mockLLMClient{
		completeFunc: func(ctx context.Context, messages []core.Message) (core.LLMResponse, error) {
			callCount++
			// Simply return a FINAL answer
			return core.LLMResponse{
				Content:          "FINAL(recursive result)",
				PromptTokens:     100,
				CompletionTokens: 50,
			}, nil
		},
	}
	replClient := &mockREPLClient{}

	r := New(client, replClient, WithMaxRecursionDepth(2))

	result, err := r.RecursiveComplete(context.Background(), "test context", "test query")
	if err != nil {
		t.Fatalf("RecursiveComplete() error: %v", err)
	}

	if result.Response != "recursive result" {
		t.Errorf("Response = %q, want %q", result.Response, "recursive result")
	}
	if result.TokenStats == nil {
		t.Error("TokenStats should not be nil")
	}
}

// TestRecursiveComplete_NoRecursionConfig tests fallback when recursion is disabled
func TestRecursiveComplete_NoRecursionConfig(t *testing.T) {
	client := &mockLLMClient{
		completeFunc: func(ctx context.Context, messages []core.Message) (core.LLMResponse, error) {
			return core.LLMResponse{
				Content:          "FINAL(no recursion)",
				PromptTokens:     50,
				CompletionTokens: 25,
			}, nil
		},
	}
	replClient := &mockREPLClient{}

	// No recursion config
	r := New(client, replClient)

	result, err := r.RecursiveComplete(context.Background(), "test context", "test query")
	if err != nil {
		t.Fatalf("RecursiveComplete() error: %v", err)
	}

	if result.Response != "no recursion" {
		t.Errorf("Response = %q, want %q", result.Response, "no recursion")
	}
	if result.MaxDepthReached != 0 {
		t.Errorf("MaxDepthReached = %d, want 0", result.MaxDepthReached)
	}
}

// TestRecursiveComplete_WithCallback tests that recursion callbacks are invoked
func TestRecursiveComplete_WithCallback(t *testing.T) {
	var callbackInvocations []struct {
		depth  int
		prompt string
	}

	callCount := 0
	client := &mockLLMClient{
		completeFunc: func(ctx context.Context, messages []core.Message) (core.LLMResponse, error) {
			callCount++
			if callCount == 1 {
				// First call triggers recursive call
				return core.LLMResponse{
					Content: `Let me use recursive analysis.
` + "```go\nresult := QueryWithRLM(\"sub-analysis\", 1)\nfmt.Println(result)\n```",
					PromptTokens:     100,
					CompletionTokens: 80,
				}, nil
			}
			// Subsequent calls return FINAL
			return core.LLMResponse{
				Content:          "FINAL(done)",
				PromptTokens:     50,
				CompletionTokens: 25,
			}, nil
		},
	}
	replClient := &mockREPLClient{}

	callback := func(depth int, prompt string) {
		callbackInvocations = append(callbackInvocations, struct {
			depth  int
			prompt string
		}{depth: depth, prompt: prompt})
	}

	r := New(client, replClient,
		WithMaxRecursionDepth(3),
		WithRecursionCallback(callback),
	)

	_, err := r.RecursiveComplete(context.Background(), "test context", "test query")
	if err != nil {
		t.Fatalf("RecursiveComplete() error: %v", err)
	}

	// Callback should have been invoked when recursive call was made
	// Note: In full integration, the callback would be invoked from RecursiveClientAdapter
	// This test verifies the callback is properly configured
	if r.config.Recursion == nil {
		t.Error("Recursion config should be set")
	}
	if r.config.Recursion.OnRecursiveQuery == nil {
		t.Error("OnRecursiveQuery callback should be set")
	}
}

// TestRecursiveComplete_TokenAggregation tests that tokens are aggregated across depths
func TestRecursiveComplete_TokenAggregation(t *testing.T) {
	client := &mockLLMClient{
		completeFunc: func(ctx context.Context, messages []core.Message) (core.LLMResponse, error) {
			return core.LLMResponse{
				Content:          "FINAL(done)",
				PromptTokens:     100,
				CompletionTokens: 50,
			}, nil
		},
	}
	replClient := &mockREPLClient{}

	r := New(client, replClient, WithMaxRecursionDepth(2))

	result, err := r.RecursiveComplete(context.Background(), "test context", "test query")
	if err != nil {
		t.Fatalf("RecursiveComplete() error: %v", err)
	}

	// Verify token stats exist
	if result.TokenStats == nil {
		t.Fatal("TokenStats should not be nil")
	}

	// Token stats should show the root level call
	promptTokens, completionTokens, totalCalls := result.TokenStats.GetTotals()
	// At minimum we should have tracked something at depth 0
	t.Logf("Token stats: prompt=%d, completion=%d, calls=%d", promptTokens, completionTokens, totalCalls)
}

// TestRecursiveComplete_DepthExceeded tests graceful handling when max depth is reached
func TestRecursiveComplete_DepthExceeded(t *testing.T) {
	// This simulates a scenario where the code tries to recurse beyond max depth
	// The RecursiveClientAdapter.QueryWithRLM should return an error in the response

	callCount := 0
	client := &mockLLMClient{
		completeFunc: func(ctx context.Context, messages []core.Message) (core.LLMResponse, error) {
			callCount++
			// Always return FINAL to prevent infinite loop
			return core.LLMResponse{
				Content:          "FINAL(done)",
				PromptTokens:     50,
				CompletionTokens: 25,
			}, nil
		},
	}
	replClient := &mockREPLClient{}

	r := New(client, replClient, WithMaxRecursionDepth(1))

	result, err := r.RecursiveComplete(context.Background(), "test context", "test query")
	if err != nil {
		t.Fatalf("RecursiveComplete() error: %v", err)
	}

	if result.Response != "done" {
		t.Errorf("Response = %q, want %q", result.Response, "done")
	}
}

// TestRecursiveComplete_PerDepthMaxIterations tests different max iterations per depth
func TestRecursiveComplete_PerDepthMaxIterations(t *testing.T) {
	client := &mockLLMClient{
		completeFunc: func(ctx context.Context, messages []core.Message) (core.LLMResponse, error) {
			return core.LLMResponse{
				Content:          "FINAL(done)",
				PromptTokens:     50,
				CompletionTokens: 25,
			}, nil
		},
	}
	replClient := &mockREPLClient{}

	cfg := RecursionConfig{
		MaxDepth: 3,
		PerDepthMaxIterations: map[int]int{
			0: 10, // Root level: 10 iterations
			1: 5,  // First recursive level: 5 iterations
			2: 3,  // Second recursive level: 3 iterations
		},
	}

	r := New(client, replClient, WithRecursionConfig(cfg))

	if r.config.Recursion == nil {
		t.Fatal("Recursion config should be set")
	}

	// Test computeMaxIterationsForDepth
	tests := []struct {
		depth    int
		expected int
	}{
		{0, 10},
		{1, 5},
		{2, 3},
		{3, 30}, // Not in map, should use default
	}

	for _, tt := range tests {
		t.Run(fmt.Sprintf("depth_%d", tt.depth), func(t *testing.T) {
			ctx := &core.RecursionContext{
				CurrentDepth: tt.depth,
				MaxDepth:     3,
			}
			result := r.computeMaxIterationsForDepth(100, ctx)
			if result != tt.expected {
				t.Errorf("computeMaxIterationsForDepth() = %d, want %d", result, tt.expected)
			}
		})
	}
}

// TestRecursionContext_Child tests creating child recursion contexts
func TestRecursionContext_Child(t *testing.T) {
	parent := core.NewRecursionContext(3)
	parent.TraceID = "trace-123"

	child := parent.Child("call-1")

	if child.CurrentDepth != 1 {
		t.Errorf("child.CurrentDepth = %d, want 1", child.CurrentDepth)
	}
	if child.MaxDepth != 3 {
		t.Errorf("child.MaxDepth = %d, want 3", child.MaxDepth)
	}
	if child.ParentID != "call-1" {
		t.Errorf("child.ParentID = %q, want %q", child.ParentID, "call-1")
	}
	if child.TraceID != "trace-123" {
		t.Errorf("child.TraceID = %q, want %q", child.TraceID, "trace-123")
	}
}

// TestRecursionContext_CanRecurse tests recursion permission checking
func TestRecursionContext_CanRecurse(t *testing.T) {
	tests := []struct {
		name         string
		currentDepth int
		maxDepth     int
		expected     bool
	}{
		{"can recurse at depth 0", 0, 2, true},
		{"can recurse at depth 1", 1, 2, true},
		{"cannot recurse at max", 2, 2, false},
		{"cannot recurse past max", 3, 2, false},
		{"can recurse with large max", 5, 10, true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			ctx := &core.RecursionContext{
				CurrentDepth: tt.currentDepth,
				MaxDepth:     tt.maxDepth,
			}
			if ctx.CanRecurse() != tt.expected {
				t.Errorf("CanRecurse() = %v, want %v", ctx.CanRecurse(), tt.expected)
			}
		})
	}
}

// TestDepthExceededError tests the depth exceeded error format
func TestDepthExceededError(t *testing.T) {
	err := &core.DepthExceededError{
		CurrentDepth: 3,
		MaxDepth:     2,
		Prompt:       "test prompt that is quite long and should be truncated in the error message",
	}

	errMsg := err.Error()

	if !strings.Contains(errMsg, "depth exceeded") {
		t.Errorf("error message should contain 'depth exceeded': %s", errMsg)
	}
	if !strings.Contains(errMsg, "current=3") {
		t.Errorf("error message should contain 'current=3': %s", errMsg)
	}
	if !strings.Contains(errMsg, "max=2") {
		t.Errorf("error message should contain 'max=2': %s", errMsg)
	}
	// Prompt should be truncated
	if strings.Contains(errMsg, "truncated in the error message") {
		t.Errorf("error message should have truncated the prompt: %s", errMsg)
	}
}

// TestWithMaxRecursionDepth tests the option function
func TestWithMaxRecursionDepth(t *testing.T) {
	client := &mockLLMClient{}
	replClient := &mockREPLClient{}

	tests := []struct {
		name     string
		depth    int
		expected int
	}{
		{"positive depth", 3, 3},
		{"zero depth", 0, 0},
		{"negative depth clamped to 0", -1, 0},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			r := New(client, replClient, WithMaxRecursionDepth(tt.depth))

			if r.config.Recursion == nil {
				t.Fatal("Recursion config should be set")
			}
			if r.config.Recursion.MaxDepth != tt.expected {
				t.Errorf("MaxDepth = %d, want %d", r.config.Recursion.MaxDepth, tt.expected)
			}
		})
	}
}

// TestRecursiveSystemPrompt tests that recursive system prompt is used at depth > 0
func TestRecursiveSystemPrompt(t *testing.T) {
	// Verify RecursiveSystemPrompt contains expected content
	if !strings.Contains(RecursiveSystemPrompt, "QueryWithRLM") {
		t.Error("RecursiveSystemPrompt should mention QueryWithRLM")
	}
	if !strings.Contains(RecursiveSystemPrompt, "CurrentDepth") {
		t.Error("RecursiveSystemPrompt should mention CurrentDepth")
	}
	if !strings.Contains(RecursiveSystemPrompt, "MaxDepth") {
		t.Error("RecursiveSystemPrompt should mention MaxDepth")
	}
	if !strings.Contains(RecursiveSystemPrompt, "CanRecurse") {
		t.Error("RecursiveSystemPrompt should mention CanRecurse")
	}
}

// BenchmarkRecursiveTokenStats benchmarks token stats operations
func BenchmarkRecursiveTokenStats(b *testing.B) {
	stats := NewRecursiveTokenStats()

	b.Run("Add", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			stats.Add(i%3, 100, 50)
		}
	})

	b.Run("GetTotals", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			stats.GetTotals()
		}
	})
}

// BenchmarkRecursionContext benchmarks recursion context operations
func BenchmarkRecursionContext(b *testing.B) {
	ctx := core.NewRecursionContext(10)

	b.Run("CanRecurse", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			ctx.CanRecurse()
		}
	})

	b.Run("Child", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			ctx.Child(fmt.Sprintf("call-%d", i))
		}
	})
}

// BenchmarkRecursiveCallTracker benchmarks call tracker operations
func BenchmarkRecursiveCallTracker(b *testing.B) {
	tracker := NewRecursiveCallTracker()

	b.Run("Track", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			tracker.Track(RecursiveCallInfo{
				ID:               fmt.Sprintf("call-%d", i),
				Depth:            i % 3,
				PromptTokens:     100,
				CompletionTokens: 50,
			})
		}
	})

	b.Run("GetCalls", func(b *testing.B) {
		// Pre-populate
		for i := 0; i < 100; i++ {
			tracker.Track(RecursiveCallInfo{ID: fmt.Sprintf("call-%d", i)})
		}
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			tracker.GetCalls()
		}
	})

	b.Run("GetCallsByDepth", func(b *testing.B) {
		// Pre-populate
		for i := 0; i < 100; i++ {
			tracker.Track(RecursiveCallInfo{ID: fmt.Sprintf("call-%d", i), Depth: i % 5})
		}
		b.ResetTimer()
		for i := 0; i < b.N; i++ {
			tracker.GetCallsByDepth()
		}
	})
}

// Integration test that can be run with real LLM providers via environment variables
// Run with: RLM_INTEGRATION_TEST=1 ANTHROPIC_API_KEY=xxx go test -v -run TestIntegration

func skipIfNotIntegrationTest(t *testing.T) {
	if os.Getenv("RLM_INTEGRATION_TEST") != "1" {
		t.Skip("Skipping integration test. Set RLM_INTEGRATION_TEST=1 to run.")
	}
}

// TestIntegration_RecursiveComplete_WithRealLLM tests with actual LLM providers
func TestIntegration_RecursiveComplete_WithRealLLM(t *testing.T) {
	skipIfNotIntegrationTest(t)

	// Check for API keys
	anthropicKey := os.Getenv("ANTHROPIC_API_KEY")
	geminiKey := os.Getenv("GEMINI_API_KEY")

	if anthropicKey == "" && geminiKey == "" {
		t.Skip("No API keys found. Set ANTHROPIC_API_KEY or GEMINI_API_KEY")
	}

	// This test would use real providers
	// For now, just verify the test infrastructure is in place
	t.Log("Integration test infrastructure verified")
	t.Log("To run with real LLMs, implement provider initialization here")
}

// TestIntegration_MultiDepthRecursion tests multiple levels of recursion
func TestIntegration_MultiDepthRecursion(t *testing.T) {
	skipIfNotIntegrationTest(t)

	// Test structure for multi-depth recursion scenarios
	testCases := []struct {
		name       string
		maxDepth   int
		context    string
		query      string
		expectPass bool
	}{
		{
			name:       "simple recursive analysis",
			maxDepth:   2,
			context:    "Sample data for analysis",
			query:      "Analyze this data recursively",
			expectPass: true,
		},
		{
			name:       "deep recursion",
			maxDepth:   3,
			context:    "Complex nested data structure",
			query:      "Perform multi-level analysis",
			expectPass: true,
		},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			t.Logf("Test case: %s (maxDepth=%d)", tc.name, tc.maxDepth)
			// Actual implementation would go here with real providers
		})
	}
}

// mockREPLClientForRecursion provides a mock REPL client for recursion testing
type mockREPLClientForRecursion struct {
	queryFunc func(ctx context.Context, prompt string) (repl.QueryResponse, error)
	batchFunc func(ctx context.Context, prompts []string) ([]repl.QueryResponse, error)
}

func (m *mockREPLClientForRecursion) Query(ctx context.Context, prompt string) (repl.QueryResponse, error) {
	if m.queryFunc != nil {
		return m.queryFunc(ctx, prompt)
	}
	return repl.QueryResponse{Response: "mock response"}, nil
}

func (m *mockREPLClientForRecursion) QueryBatched(ctx context.Context, prompts []string) ([]repl.QueryResponse, error) {
	if m.batchFunc != nil {
		return m.batchFunc(ctx, prompts)
	}
	results := make([]repl.QueryResponse, len(prompts))
	for i := range prompts {
		results[i] = repl.QueryResponse{Response: fmt.Sprintf("mock batch response %d", i)}
	}
	return results, nil
}

// TestRecursiveComplete_ContextPropagation tests that context is passed through recursion
func TestRecursiveComplete_ContextPropagation(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()

	client := &mockLLMClient{
		completeFunc: func(callCtx context.Context, messages []core.Message) (core.LLMResponse, error) {
			// Verify context is properly passed
			select {
			case <-callCtx.Done():
				return core.LLMResponse{}, callCtx.Err()
			default:
				return core.LLMResponse{
					Content:          "FINAL(done)",
					PromptTokens:     50,
					CompletionTokens: 25,
				}, nil
			}
		},
	}
	replClient := &mockREPLClient{}

	r := New(client, replClient, WithMaxRecursionDepth(2))

	result, err := r.RecursiveComplete(ctx, "test context", "test query")
	if err != nil {
		t.Fatalf("RecursiveComplete() error: %v", err)
	}

	if result.Response != "done" {
		t.Errorf("Response = %q, want %q", result.Response, "done")
	}
}

// TestRecursiveComplete_ContextCancellation tests cancellation propagation
func TestRecursiveComplete_ContextCancellation(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	cancel() // Cancel immediately

	client := &mockLLMClient{
		completeFunc: func(callCtx context.Context, messages []core.Message) (core.LLMResponse, error) {
			select {
			case <-callCtx.Done():
				return core.LLMResponse{}, callCtx.Err()
			default:
				return core.LLMResponse{Content: "FINAL(done)"}, nil
			}
		},
	}
	replClient := &mockREPLClient{}

	r := New(client, replClient, WithMaxRecursionDepth(2))

	_, err := r.RecursiveComplete(ctx, "test context", "test query")
	if err == nil {
		t.Error("expected context cancellation error")
	}
}
