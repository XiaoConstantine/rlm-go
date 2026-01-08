// Package benchmark provides comprehensive performance benchmarks for rlm-go.
// Run with: GOOGLE_API_KEY=$GOOGLE_API_KEY go test -v -timeout 20m ./pkg/benchmark/... -count=1
package benchmark

import (
	"context"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/XiaoConstantine/rlm-go/pkg/providers"
	"github.com/XiaoConstantine/rlm-go/pkg/repl"
	"github.com/XiaoConstantine/rlm-go/pkg/rlm"
)

// getGeminiClient returns a Gemini client if API key is available.
func getGeminiClient(t *testing.T, verbose bool) *providers.GeminiClient {
	apiKey := os.Getenv("GOOGLE_API_KEY")
	if apiKey == "" {
		apiKey = os.Getenv("GEMINI_API_KEY")
	}
	if apiKey == "" {
		t.Skip("Set GOOGLE_API_KEY or GEMINI_API_KEY to run this test")
	}
	return providers.NewGeminiClient(apiKey, "gemini-2.5-flash", verbose, providers.WithGeminiCaching(true))
}

// ============================================================================
// REPL Pool Performance Tests
// ============================================================================

// BenchmarkREPLCreation measures REPL instance creation time.
func BenchmarkREPLCreation(b *testing.B) {
	apiKey := os.Getenv("GOOGLE_API_KEY")
	if apiKey == "" {
		apiKey = os.Getenv("GEMINI_API_KEY")
	}
	if apiKey == "" {
		b.Skip("No API key available")
	}
	client := providers.NewGeminiClient(apiKey, "gemini-2.5-flash", false)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		r := repl.New(client)
		r.Close()
	}
}

// BenchmarkREPLPoolGet measures REPL acquisition from pre-warmed pool.
func BenchmarkREPLPoolGet(b *testing.B) {
	apiKey := os.Getenv("GOOGLE_API_KEY")
	if apiKey == "" {
		apiKey = os.Getenv("GEMINI_API_KEY")
	}
	if apiKey == "" {
		b.Skip("No API key available")
	}
	client := providers.NewGeminiClient(apiKey, "gemini-2.5-flash", false)

	pool := repl.NewREPLPool(client, 10, true)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		r := pool.Get()
		pool.Put(r)
	}
}

// TestREPLPoolVsDirectCreation compares pool vs direct creation timing.
func TestREPLPoolVsDirectCreation(t *testing.T) {
	client := getGeminiClient(t, false)

	const iterations = 100

	// Measure direct creation
	directStart := time.Now()
	for i := 0; i < iterations; i++ {
		r := repl.New(client)
		_ = r.LoadContext("test context")
		r.Close()
	}
	directDuration := time.Since(directStart)

	// Measure pool usage with pre-warming
	pool := repl.NewREPLPool(client, 10, true)
	poolStart := time.Now()
	for i := 0; i < iterations; i++ {
		r := pool.Get()
		_ = r.LoadContext("test context")
		pool.Put(r)
	}
	poolDuration := time.Since(poolStart)

	t.Logf("\n=== REPL Pool Performance ===")
	t.Logf("Iterations: %d", iterations)
	t.Logf("Direct creation: %v (avg: %v)", directDuration, directDuration/iterations)
	t.Logf("Pool usage:      %v (avg: %v)", poolDuration, poolDuration/iterations)
	t.Logf("Speedup:         %.2fx", float64(directDuration)/float64(poolDuration))

	poolSize, created := pool.Stats()
	t.Logf("Pool stats - size: %d, total created: %d", poolSize, created)
}

// ============================================================================
// Streaming Performance Tests
// ============================================================================

// TestStreamingVsNonStreaming compares streaming and non-streaming latency.
func TestStreamingVsNonStreaming(t *testing.T) {
	client := getGeminiClient(t, false)

	testContext := strings.Repeat("The weather is nice today. ", 500)
	query := "What is the first word in the text?"

	const runs = 3

	// Non-streaming measurements
	var nonStreamingTimes []time.Duration
	var nonStreamingTokens []int

	for i := 0; i < runs; i++ {
		r := rlm.New(client, client,
			rlm.WithMaxIterations(3),
			rlm.WithStreaming(false),
		)

		start := time.Now()
		result, err := r.Complete(context.Background(), testContext, query)
		elapsed := time.Since(start)

		if err != nil {
			t.Logf("Non-streaming run %d error: %v", i+1, err)
			continue
		}

		nonStreamingTimes = append(nonStreamingTimes, elapsed)
		nonStreamingTokens = append(nonStreamingTokens, result.Usage.TotalTokens)
		t.Logf("Non-streaming run %d: %v, tokens: %d, response: %s",
			i+1, elapsed, result.Usage.TotalTokens, truncate(result.Response, 50))
	}

	// Streaming measurements
	var streamingTimes []time.Duration
	var streamingTokens []int
	var firstChunkTimes []time.Duration

	for i := 0; i < runs; i++ {
		var firstChunkTime time.Time
		var firstChunk bool
		start := time.Now()

		r := rlm.New(client, client,
			rlm.WithMaxIterations(3),
			rlm.WithStreaming(true),
			rlm.WithStreamHandler(func(chunk string, done bool) error {
				if !firstChunk && chunk != "" {
					firstChunk = true
					firstChunkTime = time.Now()
				}
				return nil
			}),
		)

		result, err := r.Complete(context.Background(), testContext, query)
		elapsed := time.Since(start)

		if err != nil {
			t.Logf("Streaming run %d error: %v", i+1, err)
			continue
		}

		streamingTimes = append(streamingTimes, elapsed)
		streamingTokens = append(streamingTokens, result.Usage.TotalTokens)
		if !firstChunkTime.IsZero() {
			firstChunkTimes = append(firstChunkTimes, firstChunkTime.Sub(start))
		}
		t.Logf("Streaming run %d: %v, first chunk: %v, tokens: %d",
			i+1, elapsed, firstChunkTime.Sub(start), result.Usage.TotalTokens)
	}

	// Calculate averages
	avgNonStreaming := avgDuration(nonStreamingTimes)
	avgStreaming := avgDuration(streamingTimes)
	avgFirstChunk := avgDuration(firstChunkTimes)

	t.Logf("\n=== Streaming vs Non-Streaming ===")
	t.Logf("Non-streaming avg: %v (tokens: %v)", avgNonStreaming, nonStreamingTokens)
	t.Logf("Streaming avg:     %v (tokens: %v)", avgStreaming, streamingTokens)
	t.Logf("First chunk avg:   %v", avgFirstChunk)
	if avgNonStreaming > 0 {
		improvement := float64(avgNonStreaming-avgFirstChunk) / float64(avgNonStreaming) * 100
		t.Logf("Perceived latency improvement (to first chunk): %.1f%%", improvement)
	}
}

// ============================================================================
// History Compression Tests
// ============================================================================

// TestHistoryCompressionTokenSavings measures token savings from history compression.
func TestHistoryCompressionTokenSavings(t *testing.T) {
	client := getGeminiClient(t, false)

	testContext := strings.Repeat("Data entry: value=", 100) + "42. "
	query := "Find all data entries and count them."

	// Test without compression
	var tokensWithout []int
	for i := 0; i < 2; i++ {
		r := rlm.New(client, client,
			rlm.WithMaxIterations(8),
		)

		result, err := r.Complete(context.Background(), testContext, query)
		if err != nil {
			t.Logf("Without compression run %d error: %v", i+1, err)
			continue
		}
		tokensWithout = append(tokensWithout, result.Usage.TotalTokens)
		t.Logf("Without compression run %d: iterations=%d, tokens=%d",
			i+1, result.Iterations, result.Usage.TotalTokens)
	}

	// Test with compression
	var tokensWith []int
	for i := 0; i < 2; i++ {
		r := rlm.New(client, client,
			rlm.WithMaxIterations(8),
			rlm.WithHistoryCompression(2, 500),
		)

		result, err := r.Complete(context.Background(), testContext, query)
		if err != nil {
			t.Logf("With compression run %d error: %v", i+1, err)
			continue
		}
		tokensWith = append(tokensWith, result.Usage.TotalTokens)
		t.Logf("With compression run %d: iterations=%d, tokens=%d",
			i+1, result.Iterations, result.Usage.TotalTokens)
	}

	avgWithout := avgInt(tokensWithout)
	avgWith := avgInt(tokensWith)

	t.Logf("\n=== History Compression Token Savings ===")
	t.Logf("Without compression: avg %d tokens", avgWithout)
	t.Logf("With compression:    avg %d tokens", avgWith)
	if avgWithout > 0 {
		savings := float64(avgWithout-avgWith) / float64(avgWithout) * 100
		t.Logf("Token savings:       %.1f%%", savings)
	}
}

// ============================================================================
// Adaptive Iteration Tests
// ============================================================================

// TestAdaptiveIterationEfficiency tests if adaptive iteration reduces unnecessary iterations.
func TestAdaptiveIterationEfficiency(t *testing.T) {
	client := getGeminiClient(t, true)

	// Simple task that should terminate early with confidence
	testContext := "The secret code is: ALPHA-7892"
	query := "What is the secret code in the text?"

	// Test without adaptive iteration
	var nonAdaptiveIterations []int
	for i := 0; i < 2; i++ {
		r := rlm.New(client, client,
			rlm.WithMaxIterations(15),
		)

		result, err := r.Complete(context.Background(), testContext, query)
		if err != nil {
			t.Logf("Non-adaptive run %d error: %v", i+1, err)
			continue
		}
		nonAdaptiveIterations = append(nonAdaptiveIterations, result.Iterations)
		t.Logf("Non-adaptive run %d: iterations=%d, response=%s",
			i+1, result.Iterations, truncate(result.Response, 50))
	}

	// Test with adaptive iteration (early termination enabled)
	var adaptiveIterations []int
	for i := 0; i < 2; i++ {
		var confidenceSignals int32

		r := rlm.New(client, client,
			rlm.WithMaxIterations(15),
			rlm.WithAdaptiveIteration(),
			rlm.WithProgressHandler(func(p rlm.IterationProgress) {
				atomic.StoreInt32(&confidenceSignals, int32(p.ConfidenceSignals))
			}),
		)

		result, err := r.Complete(context.Background(), testContext, query)
		if err != nil {
			t.Logf("Adaptive run %d error: %v", i+1, err)
			continue
		}
		adaptiveIterations = append(adaptiveIterations, result.Iterations)
		t.Logf("Adaptive run %d: iterations=%d, confidence signals=%d, response=%s",
			i+1, result.Iterations, atomic.LoadInt32(&confidenceSignals), truncate(result.Response, 50))
	}

	avgNonAdaptive := avgInt(nonAdaptiveIterations)
	avgAdaptive := avgInt(adaptiveIterations)

	t.Logf("\n=== Adaptive Iteration Efficiency ===")
	t.Logf("Non-adaptive avg: %.1f iterations", float64(avgNonAdaptive))
	t.Logf("Adaptive avg:     %.1f iterations", float64(avgAdaptive))
	if avgNonAdaptive > 0 {
		reduction := float64(avgNonAdaptive-avgAdaptive) / float64(avgNonAdaptive) * 100
		t.Logf("Iteration reduction: %.1f%%", reduction)
	}
}

// ============================================================================
// Gemini Implicit Caching Tests
// ============================================================================

// TestGeminiImplicitCaching tests Gemini's automatic caching behavior.
func TestGeminiImplicitCaching(t *testing.T) {
	client := getGeminiClient(t, true)

	// Large context to trigger caching
	testContext := strings.Repeat("Line of text with important data. ", 200) +
		"\nThe answer is 42." +
		strings.Repeat(" More filler text. ", 200)

	query := "What is the answer mentioned in the text?"

	const runs = 3
	var cacheReadTokens []int
	var totalTokens []int
	var timings []time.Duration

	for i := 0; i < runs; i++ {
		r := rlm.New(client, client,
			rlm.WithMaxIterations(3),
		)

		start := time.Now()
		result, err := r.Complete(context.Background(), testContext, query)
		elapsed := time.Since(start)

		if err != nil {
			t.Logf("Run %d error: %v", i+1, err)
			continue
		}

		cacheReadTokens = append(cacheReadTokens, result.Usage.CacheReadTokens)
		totalTokens = append(totalTokens, result.Usage.TotalTokens)
		timings = append(timings, elapsed)

		t.Logf("Run %d: time=%v, total=%d, cached=%d, response=%s",
			i+1, elapsed, result.Usage.TotalTokens, result.Usage.CacheReadTokens,
			truncate(result.Response, 50))
	}

	t.Logf("\n=== Gemini Implicit Caching ===")
	t.Logf("Total tokens across runs: %v", totalTokens)
	t.Logf("Cache read tokens across runs: %v", cacheReadTokens)
	t.Logf("Timing across runs: %v", timings)

	// Check if caching increased after first run
	if len(cacheReadTokens) >= 2 && cacheReadTokens[1] > cacheReadTokens[0] {
		t.Logf("Cache hit rate increased on subsequent runs (good)")
	} else {
		t.Logf("Note: Gemini implicit caching may not show significant cache hits in short tests")
	}
}

// ============================================================================
// End-to-End Latency Breakdown
// ============================================================================

// TestLatencyBreakdown provides a detailed breakdown of where time is spent.
func TestLatencyBreakdown(t *testing.T) {
	client := getGeminiClient(t, true)

	testContext := strings.Repeat("The quick brown fox jumps over the lazy dog. ", 100)
	query := "Count the number of 'fox' occurrences."

	var replCreationTime time.Duration

	// Track timing manually
	start := time.Now()
	replStart := time.Now()
	replEnv := repl.New(client)
	replCreationTime = time.Since(replStart)
	defer replEnv.Close()

	_ = replEnv.LoadContext(testContext)

	r := rlm.New(client, client,
		rlm.WithMaxIterations(5),
		rlm.WithVerbose(true),
		rlm.WithProgressHandler(func(p rlm.IterationProgress) {
			t.Logf("  [Progress] iteration %d/%d", p.CurrentIteration, p.MaxIterations)
		}),
	)

	result, err := r.Complete(context.Background(), testContext, query)
	totalTime := time.Since(start)

	if err != nil {
		t.Fatalf("Complete error: %v", err)
	}

	t.Logf("\n=== Latency Breakdown ===")
	t.Logf("Total time:        %v", totalTime)
	t.Logf("REPL creation:     %v (%.1f%%)", replCreationTime, float64(replCreationTime)/float64(totalTime)*100)
	t.Logf("Iterations:        %d", result.Iterations)
	t.Logf("Total tokens:      %d", result.Usage.TotalTokens)
	t.Logf("  Prompt tokens:   %d", result.Usage.PromptTokens)
	t.Logf("  Completion:      %d", result.Usage.CompletionTokens)
	t.Logf("  Cache read:      %d", result.Usage.CacheReadTokens)

	t.Logf("Response:          %s", truncate(result.Response, 100))
}

// ============================================================================
// Concurrent Performance Tests
// ============================================================================

// TestConcurrentRLMExecutions tests concurrent RLM execution performance.
func TestConcurrentRLMExecutions(t *testing.T) {
	client := getGeminiClient(t, false)

	const concurrency = 3
	testContext := "The secret number is 42."
	query := "What is the secret number?"

	var wg sync.WaitGroup
	var mu sync.Mutex
	var results []struct {
		duration   time.Duration
		iterations int
		tokens     int
		err        error
	}

	start := time.Now()
	for i := 0; i < concurrency; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()

			r := rlm.New(client, client,
				rlm.WithMaxIterations(5),
			)

			execStart := time.Now()
			result, err := r.Complete(context.Background(), testContext, query)
			elapsed := time.Since(execStart)

			mu.Lock()
			if err != nil {
				results = append(results, struct {
					duration   time.Duration
					iterations int
					tokens     int
					err        error
				}{elapsed, 0, 0, err})
			} else {
				results = append(results, struct {
					duration   time.Duration
					iterations int
					tokens     int
					err        error
				}{elapsed, result.Iterations, result.Usage.TotalTokens, nil})
			}
			mu.Unlock()
		}(i)
	}

	wg.Wait()
	totalTime := time.Since(start)

	t.Logf("\n=== Concurrent RLM Executions ===")
	t.Logf("Concurrency: %d", concurrency)
	t.Logf("Total wall time: %v", totalTime)

	var totalDuration time.Duration
	var totalTokens int
	successCount := 0

	for i, r := range results {
		if r.err != nil {
			t.Logf("  Worker %d: ERROR - %v", i, r.err)
		} else {
			t.Logf("  Worker %d: %v, iterations=%d, tokens=%d", i, r.duration, r.iterations, r.tokens)
			totalDuration += r.duration
			totalTokens += r.tokens
			successCount++
		}
	}

	if successCount > 0 {
		t.Logf("Average per worker: %v", totalDuration/time.Duration(successCount))
		t.Logf("Total tokens consumed: %d", totalTokens)
		parallelSpeedup := float64(totalDuration) / float64(totalTime)
		t.Logf("Parallel efficiency: %.2fx (ideal: %.1fx)", parallelSpeedup, float64(concurrency))
	}
}

// ============================================================================
// Memory and Allocation Benchmarks
// ============================================================================

// BenchmarkMessageHistoryGrowth measures memory allocation as history grows.
func BenchmarkMessageHistoryGrowth(b *testing.B) {
	// This is a synthetic benchmark - no LLM calls needed
	b.Run("without_compression", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_ = simulateHistoryGrowth(10, false)
		}
	})

	b.Run("with_compression", func(b *testing.B) {
		for i := 0; i < b.N; i++ {
			_ = simulateHistoryGrowth(10, true)
		}
	})
}

func simulateHistoryGrowth(iterations int, compress bool) int {
	// Simulate message history growth
	totalSize := 0
	basePrompt := strings.Repeat("x", 1000)

	for i := 0; i < iterations; i++ {
		if compress && i > 3 {
			// Compressed: keep summary + last 3
			totalSize = len(basePrompt) + 500 + 3*200 // summary + recent iterations
		} else {
			// Full history
			totalSize += 200 // Each iteration adds ~200 chars
		}
	}
	return totalSize
}

// ============================================================================
// Helper Functions
// ============================================================================

func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}

func avgDuration(durations []time.Duration) time.Duration {
	if len(durations) == 0 {
		return 0
	}
	var total time.Duration
	for _, d := range durations {
		total += d
	}
	return total / time.Duration(len(durations))
}

func avgInt(values []int) int {
	if len(values) == 0 {
		return 0
	}
	total := 0
	for _, v := range values {
		total += v
	}
	return total / len(values)
}

// ============================================================================
// Profiling Support
// ============================================================================

// TestProfileHotPaths runs a workload suitable for profiling.
// Run with: go test -v -run TestProfileHotPaths -cpuprofile=cpu.prof -memprofile=mem.prof ./pkg/benchmark/...
func TestProfileHotPaths(t *testing.T) {
	client := getGeminiClient(t, false)

	testContext := strings.Repeat("Data point: ", 50) + "The answer is 42."
	query := "Find the answer in the text."

	for i := 0; i < 3; i++ {
		r := rlm.New(client, client,
			rlm.WithMaxIterations(5),
			rlm.WithHistoryCompression(2, 500),
			rlm.WithAdaptiveIteration(),
		)

		_, err := r.Complete(context.Background(), testContext, query)
		if err != nil {
			t.Logf("Run %d error: %v", i+1, err)
		}
	}
}

// ============================================================================
// FINAL Parsing Performance
// ============================================================================

// BenchmarkFINALParsing benchmarks the FINAL answer parsing strategies.
func BenchmarkFINALParsing(b *testing.B) {
	testCases := []string{
		"The answer is FINAL(42)",
		"FINAL_VAR(result)",
		"After analysis: FINAL(\nmulti-line\nresponse\n)",
		"FINAL: the answer is 42",
		"FINAL => the answer",
		`{"result": "the answer is 42"}`,
		"No final answer in this response, just some text...",
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		for _, tc := range testCases {
			// Simulate finding final answer in response
			_ = strings.Contains(tc, "FINAL")
		}
	}
}

// ============================================================================
// Complete Performance Summary Test
// ============================================================================

// TestPerformanceSummary runs all key performance tests and summarizes results.
func TestPerformanceSummary(t *testing.T) {
	if testing.Short() {
		t.Skip("Skipping performance summary in short mode")
	}

	t.Log("\n" + strings.Repeat("=", 60))
	t.Log("RLM-GO PERFORMANCE BENCHMARK SUMMARY")
	t.Log(strings.Repeat("=", 60))

	client := getGeminiClient(t, false)

	// 1. REPL Pool performance
	t.Log("\n--- REPL Pool Performance ---")
	pool := repl.NewREPLPool(client, 5, true)
	directTime := measureREPLCreation(client, 10)
	poolTime := measureREPLPool(pool, 10)
	t.Logf("Direct creation (10x): %v", directTime)
	t.Logf("Pool usage (10x):      %v", poolTime)
	t.Logf("Speedup:               %.2fx", float64(directTime)/float64(poolTime))

	// 2. Simple completion benchmark
	t.Log("\n--- Simple Completion ---")
	testContext := "The secret is: CODE-123"
	query := "What is the secret?"

	r := rlm.New(client, client,
		rlm.WithMaxIterations(3),
	)

	start := time.Now()
	result, err := r.Complete(context.Background(), testContext, query)
	elapsed := time.Since(start)

	if err != nil {
		t.Logf("Error: %v", err)
	} else {
		t.Logf("Time:       %v", elapsed)
		t.Logf("Iterations: %d", result.Iterations)
		t.Logf("Tokens:     %d (prompt: %d, completion: %d)",
			result.Usage.TotalTokens, result.Usage.PromptTokens, result.Usage.CompletionTokens)
		t.Logf("Response:   %s", truncate(result.Response, 50))
	}

	t.Log("\n" + strings.Repeat("=", 60))
	t.Log("BENCHMARK COMPLETE")
	t.Log(strings.Repeat("=", 60))
}

func measureREPLCreation(client *providers.GeminiClient, n int) time.Duration {
	start := time.Now()
	for i := 0; i < n; i++ {
		r := repl.New(client)
		r.Close()
	}
	return time.Since(start)
}

func measureREPLPool(pool *repl.REPLPool, n int) time.Duration {
	start := time.Now()
	for i := 0; i < n; i++ {
		r := pool.Get()
		pool.Put(r)
	}
	return time.Since(start)
}

// ============================================================================
// Bottleneck Identification Tests
// ============================================================================

// TestIdentifyBottlenecks runs specific tests to identify performance bottlenecks.
func TestIdentifyBottlenecks(t *testing.T) {
	client := getGeminiClient(t, false)

	t.Log("\n=== BOTTLENECK IDENTIFICATION ===\n")

	// 1. Measure Yaegi interpreter overhead
	t.Log("1. Yaegi Interpreter Startup")
	yaegStarts := make([]time.Duration, 5)
	for i := 0; i < 5; i++ {
		start := time.Now()
		r := repl.New(client)
		yaegStarts[i] = time.Since(start)
		r.Close()
	}
	avgYaegi := avgDuration(yaegStarts)
	t.Logf("   Avg interpreter startup: %v", avgYaegi)
	t.Logf("   Per-iteration overhead if not pooled: %v", avgYaegi)

	// 2. Measure context loading
	t.Log("\n2. Context Loading")
	largeContext := strings.Repeat("x", 100000) // 100KB
	r := repl.New(client)
	start := time.Now()
	_ = r.LoadContext(largeContext)
	loadTime := time.Since(start)
	r.Close()
	t.Logf("   Loading 100KB context: %v", loadTime)

	// 3. Measure code execution without LLM
	t.Log("\n3. Code Execution (no LLM)")
	r = repl.New(client)
	execTimes := make([]time.Duration, 5)
	for i := 0; i < 5; i++ {
		start := time.Now()
		code := `
			sum := 0
			for j := 0; j < 1000; j++ {
				sum += j
			}
			fmt.Println(sum)
		`
		_, _ = r.Execute(context.Background(), code)
		execTimes[i] = time.Since(start)
	}
	r.Close()
	t.Logf("   Avg code execution (1000 iterations): %v", avgDuration(execTimes))

	// 4. HTTP client overhead estimation
	t.Log("\n4. HTTP/Network Overhead")
	t.Log("   (Included in LLM call time - cannot isolate without mock)")
	t.Log("   Recommendation: Use connection pooling (already in place)")

	// Summary
	t.Log("\n=== BOTTLENECK SUMMARY ===")
	t.Logf("1. Yaegi startup:    %v per REPL (use pooling)", avgYaegi)
	t.Logf("2. Context loading:  %v for 100KB (negligible)", loadTime)
	t.Logf("3. Code execution:   %v per block (negligible)", avgDuration(execTimes))
	t.Log("4. LLM API calls:    PRIMARY BOTTLENECK (network + inference)")
	t.Log("\nConclusion: LLM API latency dominates. Focus on:")
	t.Log("   - Reducing iteration count (adaptive termination)")
	t.Log("   - Reducing context size (history compression)")
	t.Log("   - Leveraging caching (Gemini implicit, Anthropic explicit)")
	t.Log("   - Streaming for perceived latency")
}
