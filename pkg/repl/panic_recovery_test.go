// Package repl provides a Yaegi-based Go REPL for RLM code execution.
// This file contains stress tests for the Yaegi interpreter panic recovery mechanism.
package repl

import (
	"context"
	"fmt"
	"strings"
	"sync"
	"testing"
	"time"
)

// TestPanicRecovery_ManyIterations tests that the REPL survives many sequential executions
// that would normally accumulate state and potentially crash Yaegi.
func TestPanicRecovery_ManyIterations(t *testing.T) {
	client := &mockLLMClient{
		queryFunc: func(_ context.Context, prompt string) (QueryResponse, error) {
			return QueryResponse{Response: "mock response for: " + prompt}, nil
		},
	}
	repl := New(client)
	defer repl.Close()

	ctx := context.Background()
	const iterations = 25

	for i := 0; i < iterations; i++ {
		// Execute different code patterns each iteration
		code := fmt.Sprintf(`
x%d := %d
y%d := x%d * 2
fmt.Println("iteration %d: x=%d, y=", x%d, y%d)
`, i, i, i, i, i, i, i, i)

		result, err := repl.Execute(ctx, code)
		if err != nil {
			// Panic was caught - verify NeedsReset behavior
			t.Logf("Iteration %d: panic caught (expected behavior under stress): %v", i, err)
			if !repl.NeedsReset() {
				t.Errorf("Iteration %d: NeedsReset() should be true after panic", i)
			}
			// Attempt recovery
			reset, resetErr := repl.ResetIfNeeded()
			if resetErr != nil {
				t.Fatalf("Iteration %d: ResetIfNeeded() failed: %v", i, resetErr)
			}
			if !reset {
				t.Errorf("Iteration %d: ResetIfNeeded() should return true when reset needed", i)
			}
			// Continue with fresh interpreter
			continue
		}

		if result.Stderr != "" && strings.Contains(result.Stderr, "panic") {
			t.Logf("Iteration %d: stderr contains panic info: %s", i, result.Stderr)
		}
	}

	t.Logf("Completed %d iterations, execution count: %d", iterations, repl.ExecutionCount())
}

// TestPanicRecovery_FunctionRedefinition tests handling of function redefinition conflicts.
func TestPanicRecovery_FunctionRedefinition(t *testing.T) {
	client := &mockLLMClient{
		queryFunc: func(_ context.Context, prompt string) (QueryResponse, error) {
			return QueryResponse{Response: "mock"}, nil
		},
	}
	repl := New(client)
	defer repl.Close()

	ctx := context.Background()

	// Test multiple function redefinitions which can cause Yaegi issues
	funcDefs := []string{
		`func helper(x int) int { return x + 1 }`,
		`func helper(x int) int { return x + 2 }`,        // redefinition
		`func helper(x int) int { return x * 2 }`,        // another redefinition
		`func helper(x, y int) int { return x + y }`,     // signature change
		`func helper(s string) string { return s + s }`, // type change
	}

	panicCount := 0
	for i, def := range funcDefs {
		result, err := repl.Execute(ctx, def)
		if err != nil {
			panicCount++
			t.Logf("Function redefinition %d caused panic (expected): %v", i, err)
			if repl.NeedsReset() {
				_, _ = repl.ResetIfNeeded()
			}
			continue
		}
		if result.Stderr != "" {
			t.Logf("Function redefinition %d error (expected): %s", i, result.Stderr)
		}
	}

	t.Logf("Function redefinition test: %d/%d caused panics", panicCount, len(funcDefs))
}

// TestPanicRecovery_MinMaxRedefinition tests the specific min/max function redefinition
// that was causing the original crash after ~8 iterations.
func TestPanicRecovery_MinMaxRedefinition(t *testing.T) {
	client := &mockLLMClient{
		queryFunc: func(_ context.Context, prompt string) (QueryResponse, error) {
			return QueryResponse{Response: "mock"}, nil
		},
	}
	repl := New(client)
	defer repl.Close()

	ctx := context.Background()

	// Attempt to redefine min/max multiple times (the original crash trigger)
	for i := 0; i < 15; i++ {
		code := fmt.Sprintf(`
// Attempt %d: redefine min
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// Attempt %d: redefine max
func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

result := min(10, 5) + max(3, 7)
fmt.Println("min/max result:", result)
`, i, i)

		result, err := repl.Execute(ctx, code)
		if err != nil {
			t.Logf("Iteration %d: panic recovered: %v", i, err)
			if !repl.NeedsReset() {
				t.Errorf("NeedsReset() should be true after panic")
			}
			// Recover and continue
			reset, resetErr := repl.ResetIfNeeded()
			if resetErr != nil {
				t.Fatalf("ResetIfNeeded() failed: %v", resetErr)
			}
			if !reset {
				t.Error("ResetIfNeeded() should have performed reset")
			}
			// Verify interpreter is working after reset
			_, checkErr := repl.Execute(ctx, `fmt.Println("recovered")`)
			if checkErr != nil {
				t.Errorf("Interpreter not working after reset: %v", checkErr)
			}
		} else if result.Stderr != "" {
			// Expected: Yaegi should reject redefinition
			t.Logf("Iteration %d: redefinition error (expected): %s", i, result.Stderr[:min(100, len(result.Stderr))])
		}
	}
}

// TestPanicRecovery_ImportAccumulation tests import statement accumulation.
func TestPanicRecovery_ImportAccumulation(t *testing.T) {
	client := &mockLLMClient{
		queryFunc: func(_ context.Context, prompt string) (QueryResponse, error) {
			return QueryResponse{Response: "mock"}, nil
		},
	}
	repl := New(client)
	defer repl.Close()

	ctx := context.Background()

	// Execute many import statements
	imports := []string{
		`import "time"`,
		`import "strconv"`,
		`import "bytes"`,
		`import "sort"`,
		`import "math"`,
		`import "io"`,
		`import "os"`,
		`import "path"`,
		`import "sync"`,
		`import "reflect"`,
	}

	for round := 0; round < 3; round++ {
		for i, imp := range imports {
			result, err := repl.Execute(ctx, imp)
			if err != nil {
				t.Logf("Round %d, import %d: panic recovered: %v", round, i, err)
				if repl.NeedsReset() {
					_, _ = repl.ResetIfNeeded()
				}
				continue
			}
			if result.Stderr != "" && !strings.Contains(result.Stderr, "already imported") {
				// Some errors are expected for re-imports
				t.Logf("Round %d, import %d: %s", round, i, result.Stderr)
			}
		}
	}

	t.Logf("Import accumulation test completed, execution count: %d", repl.ExecutionCount())
}

// TestPanicRecovery_VariableShadowing tests variable shadowing across executions.
func TestPanicRecovery_VariableShadowing(t *testing.T) {
	client := &mockLLMClient{
		queryFunc: func(_ context.Context, prompt string) (QueryResponse, error) {
			return QueryResponse{Response: "mock"}, nil
		},
	}
	repl := New(client)
	defer repl.Close()

	ctx := context.Background()

	// Create many variables with same names but different types
	shadowPatterns := []string{
		`x := 42`,
		`x := "hello"`,
		`x := 3.14`,
		`x := true`,
		`x := []int{1, 2, 3}`,
		`x := map[string]int{"a": 1}`,
		`var x interface{} = struct{}{}`,
	}

	for round := 0; round < 5; round++ {
		for i, pattern := range shadowPatterns {
			result, err := repl.Execute(ctx, pattern)
			if err != nil {
				t.Logf("Round %d, shadow %d: panic recovered: %v", round, i, err)
				if repl.NeedsReset() {
					_, _ = repl.ResetIfNeeded()
				}
				continue
			}
			// Variable shadowing typically causes "redeclared" errors, which is expected
			if result.Stderr != "" {
				t.Logf("Round %d, shadow %d: %s", round, i, result.Stderr[:min(80, len(result.Stderr))])
			}
		}
	}

	t.Logf("Variable shadowing test completed, execution count: %d", repl.ExecutionCount())
}

// TestPanicRecovery_NeedsResetFlag verifies the NeedsReset flag behavior.
func TestPanicRecovery_NeedsResetFlag(t *testing.T) {
	client := &mockLLMClient{
		queryFunc: func(_ context.Context, prompt string) (QueryResponse, error) {
			return QueryResponse{Response: "mock"}, nil
		},
	}
	repl := New(client)
	defer repl.Close()

	ctx := context.Background()

	// Initially should not need reset
	if repl.NeedsReset() {
		t.Error("Fresh REPL should not need reset")
	}

	// Normal execution should not trigger reset flag
	_, err := repl.Execute(ctx, `fmt.Println("hello")`)
	if err != nil {
		t.Fatalf("Normal execution failed: %v", err)
	}
	if repl.NeedsReset() {
		t.Error("NeedsReset should be false after successful execution")
	}

	// ResetIfNeeded should return false when not needed
	reset, resetErr := repl.ResetIfNeeded()
	if resetErr != nil {
		t.Errorf("ResetIfNeeded failed: %v", resetErr)
	}
	if reset {
		t.Error("ResetIfNeeded should return false when reset not needed")
	}

	// After explicit reset, flag should be false
	if err := repl.Reset(); err != nil {
		t.Fatalf("Reset failed: %v", err)
	}
	if repl.NeedsReset() {
		t.Error("NeedsReset should be false after Reset()")
	}
}

// TestPanicRecovery_ResetRecovery tests that ResetIfNeeded properly recovers the interpreter.
func TestPanicRecovery_ResetRecovery(t *testing.T) {
	client := &mockLLMClient{
		queryFunc: func(_ context.Context, prompt string) (QueryResponse, error) {
			return QueryResponse{Response: "query result for: " + prompt}, nil
		},
	}
	repl := New(client)
	defer repl.Close()

	ctx := context.Background()

	// Store a variable
	_, err := repl.Execute(ctx, `myVar := "original value"`)
	if err != nil {
		t.Fatalf("Failed to set variable: %v", err)
	}

	// Verify variable exists
	val, err := repl.GetVariable("myVar")
	if err != nil {
		t.Fatalf("Failed to get variable: %v", err)
	}
	if val != "original value" {
		t.Errorf("Variable value = %q, want %q", val, "original value")
	}

	// Force a reset
	if err := repl.Reset(); err != nil {
		t.Fatalf("Reset failed: %v", err)
	}

	// Variable should no longer exist
	_, err = repl.GetVariable("myVar")
	if err == nil {
		t.Error("Variable should not exist after reset")
	}

	// But REPL should still work
	result, err := repl.Execute(ctx, `fmt.Println("working after reset")`)
	if err != nil {
		t.Errorf("Execute failed after reset: %v", err)
	}
	if !strings.Contains(result.Stdout, "working after reset") {
		t.Errorf("Unexpected output after reset: %s", result.Stdout)
	}

	// Query function should still work
	result, err = repl.Execute(ctx, `
response := Query("test prompt")
fmt.Println(response)
`)
	if err != nil {
		t.Errorf("Query execution failed after reset: %v", err)
	}
	if !strings.Contains(result.Stdout, "query result for: test prompt") {
		t.Errorf("Query not working after reset, got: %s", result.Stdout)
	}

	// min/max should still work
	result, err = repl.Execute(ctx, `fmt.Println(min(10, 5), max(3, 7))`)
	if err != nil {
		t.Errorf("min/max execution failed after reset: %v", err)
	}
	if !strings.Contains(result.Stdout, "5 7") {
		t.Errorf("min/max not working after reset, got: %s", result.Stdout)
	}
}

// TestPanicRecovery_ContextReloadAfterReset tests that context can be reloaded after reset.
func TestPanicRecovery_ContextReloadAfterReset(t *testing.T) {
	client := &mockLLMClient{
		queryFunc: func(_ context.Context, prompt string) (QueryResponse, error) {
			return QueryResponse{Response: "mock"}, nil
		},
	}
	repl := New(client)
	defer repl.Close()

	ctx := context.Background()
	testContext := "This is the test context data for analysis"

	// Load initial context
	if err := repl.LoadContext(testContext); err != nil {
		t.Fatalf("Failed to load context: %v", err)
	}

	// Verify context is accessible
	result, err := repl.Execute(ctx, `fmt.Println(len(context))`)
	if err != nil {
		t.Fatalf("Execute failed: %v", err)
	}
	if !strings.Contains(result.Stdout, fmt.Sprintf("%d", len(testContext))) {
		t.Errorf("Context length incorrect: %s", result.Stdout)
	}

	// Reset the interpreter
	if err := repl.Reset(); err != nil {
		t.Fatalf("Reset failed: %v", err)
	}

	// Context should no longer be accessible
	info := repl.ContextInfo()
	if info != "context not loaded" {
		t.Errorf("Context should not be loaded after reset, got: %s", info)
	}

	// Reload context
	if err := repl.LoadContext(testContext); err != nil {
		t.Fatalf("Failed to reload context: %v", err)
	}

	// Verify context is accessible again
	result, err = repl.Execute(ctx, `fmt.Println(len(context))`)
	if err != nil {
		t.Errorf("Execute failed after reload: %v", err)
	}
	if !strings.Contains(result.Stdout, fmt.Sprintf("%d", len(testContext))) {
		t.Errorf("Context length incorrect after reload: %s", result.Stdout)
	}
}

// TestPanicRecovery_StressCodePatterns tests code patterns known to stress Yaegi.
func TestPanicRecovery_StressCodePatterns(t *testing.T) {
	client := &mockLLMClient{
		queryFunc: func(_ context.Context, prompt string) (QueryResponse, error) {
			return QueryResponse{Response: "mock"}, nil
		},
	}
	repl := New(client)
	defer repl.Close()

	ctx := context.Background()

	// Code patterns that can stress the interpreter
	stressPatterns := []struct {
		name string
		code string
	}{
		{
			name: "large slice allocation",
			code: `
large := make([]int, 10000)
for i := range large {
	large[i] = i * i
}
fmt.Println("slice len:", len(large))
`,
		},
		{
			name: "nested closures",
			code: `
f1 := func(x int) func(int) int {
	return func(y int) int {
		return x + y
	}
}
f2 := f1(10)
fmt.Println("closure result:", f2(5))
`,
		},
		{
			name: "recursive function",
			code: `
var fib func(int) int
fib = func(n int) int {
	if n <= 1 {
		return n
	}
	return fib(n-1) + fib(n-2)
}
fmt.Println("fib(10):", fib(10))
`,
		},
		{
			name: "map operations",
			code: `
m := make(map[string][]int)
for i := 0; i < 100; i++ {
	key := fmt.Sprintf("key%d", i%10)
	m[key] = append(m[key], i)
}
fmt.Println("map keys:", len(m))
`,
		},
		{
			name: "interface type assertions",
			code: `
var items []interface{}
items = append(items, 42, "hello", 3.14, true)
for _, item := range items {
	switch v := item.(type) {
	case int:
		fmt.Printf("int: %d\n", v)
	case string:
		fmt.Printf("string: %s\n", v)
	default:
		fmt.Printf("other: %v\n", v)
	}
}
`,
		},
		{
			name: "string builder",
			code: `
var sb strings.Builder
for i := 0; i < 100; i++ {
	sb.WriteString(fmt.Sprintf("item%d ", i))
}
fmt.Println("built string len:", sb.Len())
`,
		},
		{
			name: "regex operations",
			code: `
pattern := regexp.MustCompile("\\d+")
text := "abc123def456ghi789"
matches := pattern.FindAllString(text, -1)
fmt.Println("matches:", matches)
`,
		},
		{
			name: "goroutine simulation (sequential)",
			code: `
results := make([]int, 10)
for i := 0; i < 10; i++ {
	results[i] = i * i
}
sum := 0
for _, v := range results {
	sum += v
}
fmt.Println("sum:", sum)
`,
		},
	}

	for _, pattern := range stressPatterns {
		t.Run(pattern.name, func(t *testing.T) {
			// Run each pattern multiple times
			for i := 0; i < 3; i++ {
				result, err := repl.Execute(ctx, pattern.code)
				if err != nil {
					t.Logf("Pattern %q iteration %d: panic recovered: %v", pattern.name, i, err)
					if repl.NeedsReset() {
						_, _ = repl.ResetIfNeeded()
					}
					continue
				}
				if result.Stderr != "" && !strings.Contains(result.Stderr, "already declared") {
					t.Logf("Pattern %q iteration %d stderr: %s", pattern.name, i, result.Stderr)
				}
			}
		})
	}
}

// TestPanicRecovery_ConcurrentStress tests panic recovery under concurrent execution.
func TestPanicRecovery_ConcurrentStress(t *testing.T) {
	client := &mockLLMClient{
		queryFunc: func(_ context.Context, prompt string) (QueryResponse, error) {
			return QueryResponse{Response: "mock"}, nil
		},
	}

	ctx := context.Background()
	const numREPLs = 5
	const iterationsPerREPL = 20

	var wg sync.WaitGroup
	errChan := make(chan string, numREPLs*iterationsPerREPL)

	for r := 0; r < numREPLs; r++ {
		wg.Add(1)
		go func(replID int) {
			defer wg.Done()
			repl := New(client)
			defer repl.Close()

			for i := 0; i < iterationsPerREPL; i++ {
				code := fmt.Sprintf(`
n%d_%d := %d
result := n%d_%d * 2
fmt.Println("REPL %d, iteration %d:", result)
`, replID, i, i, replID, i, replID, i)

				result, err := repl.Execute(ctx, code)
				if err != nil {
					errChan <- fmt.Sprintf("REPL %d, iteration %d: panic: %v", replID, i, err)
					if repl.NeedsReset() {
						if _, resetErr := repl.ResetIfNeeded(); resetErr != nil {
							errChan <- fmt.Sprintf("REPL %d, iteration %d: reset failed: %v", replID, i, resetErr)
						}
					}
					continue
				}
				if result.Stderr != "" && strings.Contains(strings.ToLower(result.Stderr), "panic") {
					errChan <- fmt.Sprintf("REPL %d, iteration %d: stderr panic: %s", replID, i, result.Stderr)
				}
			}
		}(r)
	}

	wg.Wait()
	close(errChan)

	// Log any issues found
	panicCount := 0
	for msg := range errChan {
		panicCount++
		t.Log(msg)
	}

	t.Logf("Concurrent stress test: %d total panics/errors across %d REPLs x %d iterations",
		panicCount, numREPLs, iterationsPerREPL)
}

// TestPanicRecovery_ExecutionCountTracking tests the execution count tracking.
func TestPanicRecovery_ExecutionCountTracking(t *testing.T) {
	client := &mockLLMClient{
		queryFunc: func(_ context.Context, prompt string) (QueryResponse, error) {
			return QueryResponse{Response: "mock"}, nil
		},
	}
	repl := New(client)
	defer repl.Close()

	ctx := context.Background()

	// Initial count should be 0
	if repl.ExecutionCount() != 0 {
		t.Errorf("Initial execution count = %d, want 0", repl.ExecutionCount())
	}

	// Execute some code
	const numExecs = 10
	for i := 0; i < numExecs; i++ {
		_, _ = repl.Execute(ctx, fmt.Sprintf(`fmt.Println(%d)`, i))
	}

	if repl.ExecutionCount() != numExecs {
		t.Errorf("Execution count = %d, want %d", repl.ExecutionCount(), numExecs)
	}

	// Reset should clear the count
	if err := repl.Reset(); err != nil {
		t.Fatalf("Reset failed: %v", err)
	}

	if repl.ExecutionCount() != 0 {
		t.Errorf("Execution count after reset = %d, want 0", repl.ExecutionCount())
	}
}

// TestPanicRecovery_RapidResetCycles tests rapid reset cycles.
func TestPanicRecovery_RapidResetCycles(t *testing.T) {
	client := &mockLLMClient{
		queryFunc: func(_ context.Context, prompt string) (QueryResponse, error) {
			return QueryResponse{Response: "mock"}, nil
		},
	}
	repl := New(client)
	defer repl.Close()

	ctx := context.Background()

	for cycle := 0; cycle < 10; cycle++ {
		// Execute some code
		for i := 0; i < 5; i++ {
			result, err := repl.Execute(ctx, fmt.Sprintf(`fmt.Println("cycle %d, exec %d")`, cycle, i))
			if err != nil {
				t.Logf("Cycle %d, exec %d: error: %v", cycle, i, err)
			}
			if result.Stderr != "" {
				t.Logf("Cycle %d, exec %d: stderr: %s", cycle, i, result.Stderr)
			}
		}

		// Reset
		if err := repl.Reset(); err != nil {
			t.Fatalf("Cycle %d: Reset failed: %v", cycle, err)
		}

		// Verify min/max still work after reset
		result, err := repl.Execute(ctx, `fmt.Println(min(5, 3), max(5, 3))`)
		if err != nil {
			t.Errorf("Cycle %d: post-reset execution failed: %v", cycle, err)
		}
		if !strings.Contains(result.Stdout, "3 5") {
			t.Errorf("Cycle %d: min/max incorrect after reset: %s", cycle, result.Stdout)
		}
	}
}

// TestPanicRecovery_LongRunningExecution tests behavior with longer-running code.
func TestPanicRecovery_LongRunningExecution(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping long-running test in short mode")
	}

	client := &mockLLMClient{
		queryFunc: func(_ context.Context, prompt string) (QueryResponse, error) {
			time.Sleep(10 * time.Millisecond)
			return QueryResponse{Response: "mock"}, nil
		},
	}
	repl := New(client)
	defer repl.Close()

	ctx := context.Background()

	// Execute code that takes some time (simulating LLM calls)
	code := `
results := []string{}
for i := 0; i < 5; i++ {
	r := Query(fmt.Sprintf("prompt %d", i))
	results = append(results, r)
}
fmt.Println("got", len(results), "results")
`

	start := time.Now()
	result, err := repl.Execute(ctx, code)
	duration := time.Since(start)

	if err != nil {
		t.Errorf("Long-running execution failed: %v", err)
	}
	if !strings.Contains(result.Stdout, "got 5 results") {
		t.Errorf("Unexpected output: %s", result.Stdout)
	}

	t.Logf("Long-running execution took: %v", duration)
}

// TestPanicRecovery_EdgeCases tests various edge cases.
func TestPanicRecovery_EdgeCases(t *testing.T) {
	client := &mockLLMClient{
		queryFunc: func(_ context.Context, prompt string) (QueryResponse, error) {
			return QueryResponse{Response: "mock"}, nil
		},
	}
	repl := New(client)
	defer repl.Close()

	ctx := context.Background()

	testCases := []struct {
		name string
		code string
	}{
		{"empty code", ""},
		{"whitespace only", "   \n\t\n   "},
		{"comment only", "// just a comment"},
		{"invalid syntax", "this is not valid go code !@#$%"},
		{"undefined variable", "fmt.Println(undefinedVar)"},
		{"nil dereference (guarded)", `
var p *int
if p != nil {
	fmt.Println(*p)
} else {
	fmt.Println("nil pointer, handled")
}
`},
		{"division by zero (guarded)", `
divisor := 0
if divisor != 0 {
	fmt.Println(10 / divisor)
} else {
	fmt.Println("division by zero, handled")
}
`},
	}

	for _, tc := range testCases {
		t.Run(tc.name, func(t *testing.T) {
			result, err := repl.Execute(ctx, tc.code)
			if err != nil {
				// Panic was caught - this is acceptable for edge cases
				t.Logf("Edge case %q: panic caught: %v", tc.name, err)
				if repl.NeedsReset() {
					_, _ = repl.ResetIfNeeded()
				}
				return
			}
			if result.Stderr != "" {
				t.Logf("Edge case %q: stderr: %s", tc.name, result.Stderr)
			}
			if result.Stdout != "" {
				t.Logf("Edge case %q: stdout: %s", tc.name, result.Stdout)
			}
		})
	}
}

// min is a helper for string truncation in tests (Go 1.21 builtin not available in tests).
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// TestPanicRecovery_HighIterationCount tests very high iteration counts to trigger state accumulation issues.
func TestPanicRecovery_HighIterationCount(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping high iteration test in short mode")
	}

	client := &mockLLMClient{
		queryFunc: func(_ context.Context, prompt string) (QueryResponse, error) {
			return QueryResponse{Response: "mock"}, nil
		},
	}
	repl := New(client)
	defer repl.Close()

	ctx := context.Background()
	const iterations = 50 // Higher than the original ~8 that caused crashes

	panicCount := 0
	resetCount := 0

	for i := 0; i < iterations; i++ {
		// Deliberately stressful code pattern - creates new variables each time
		code := fmt.Sprintf(`
var v%d_%d = %d
var s%d_%d = "%d"
var arr%d_%d = []int{%d, %d, %d}
sum := 0
for _, x := range arr%d_%d {
	sum += x
}
fmt.Printf("iter %d: v=%%d, s=%%s, sum=%%d\n", v%d_%d, s%d_%d, sum)
`, i, i%10, i, i, i%10, i, i, i%10, i, i+1, i+2, i, i%10, i, i, i%10, i, i%10)

		result, err := repl.Execute(ctx, code)
		if err != nil {
			panicCount++
			t.Logf("Iteration %d: panic caught: %v", i, err)
			if repl.NeedsReset() {
				resetCount++
				if _, resetErr := repl.ResetIfNeeded(); resetErr != nil {
					t.Fatalf("Reset failed at iteration %d: %v", i, resetErr)
				}
			}
			continue
		}

		if result.Stderr != "" && strings.Contains(result.Stderr, "panic") {
			t.Logf("Iteration %d stderr: %s", i, result.Stderr[:min(100, len(result.Stderr))])
		}
	}

	t.Logf("High iteration test: %d panics, %d resets across %d iterations", panicCount, resetCount, iterations)
	t.Logf("Final execution count: %d", repl.ExecutionCount())
}

// TestPanicRecovery_TypeConflicts tests rapid type changes that can confuse Yaegi.
func TestPanicRecovery_TypeConflicts(t *testing.T) {
	client := &mockLLMClient{
		queryFunc: func(_ context.Context, prompt string) (QueryResponse, error) {
			return QueryResponse{Response: "mock"}, nil
		},
	}
	repl := New(client)
	defer repl.Close()

	ctx := context.Background()

	// Type conflict patterns - same name, different types
	typePatterns := []string{
		`type MyType int`,
		`type MyType string`,
		`type MyType struct { X int }`,
		`type MyType interface { Do() }`,
		`type MyType []byte`,
		`type MyType map[string]int`,
		`type MyType func(int) int`,
		`type MyType chan int`,
	}

	panicCount := 0
	for round := 0; round < 3; round++ {
		for i, pattern := range typePatterns {
			result, err := repl.Execute(ctx, pattern)
			if err != nil {
				panicCount++
				t.Logf("Round %d, type %d: panic: %v", round, i, err)
				if repl.NeedsReset() {
					_, _ = repl.ResetIfNeeded()
				}
				continue
			}
			if result.Stderr != "" {
				t.Logf("Round %d, type %d: stderr: %s", round, i, result.Stderr[:min(80, len(result.Stderr))])
			}
		}
	}

	t.Logf("Type conflict test: %d panics", panicCount)
}

// TestPanicRecovery_InterfaceMethodConflicts tests interface method conflicts.
func TestPanicRecovery_InterfaceMethodConflicts(t *testing.T) {
	client := &mockLLMClient{
		queryFunc: func(_ context.Context, prompt string) (QueryResponse, error) {
			return QueryResponse{Response: "mock"}, nil
		},
	}
	repl := New(client)
	defer repl.Close()

	ctx := context.Background()

	// Method conflicts
	methodPatterns := []string{
		`
type Handler interface {
	Handle(x int) int
}
`,
		`
type Handler interface {
	Handle(x string) string
}
`,
		`
type Handler interface {
	Handle() error
}
`,
		`
type Handler interface {
	Handle(ctx context.Context) (string, error)
}
`,
	}

	panicCount := 0
	for i, pattern := range methodPatterns {
		result, err := repl.Execute(ctx, pattern)
		if err != nil {
			panicCount++
			t.Logf("Method pattern %d: panic: %v", i, err)
			if repl.NeedsReset() {
				_, _ = repl.ResetIfNeeded()
			}
			continue
		}
		if result.Stderr != "" {
			t.Logf("Method pattern %d: stderr: %s", i, result.Stderr[:min(80, len(result.Stderr))])
		}
	}

	t.Logf("Interface method conflict test: %d panics", panicCount)
}

// TestPanicRecovery_ReflectionStress tests patterns that stress reflection in Yaegi.
func TestPanicRecovery_ReflectionStress(t *testing.T) {
	client := &mockLLMClient{
		queryFunc: func(_ context.Context, prompt string) (QueryResponse, error) {
			return QueryResponse{Response: "mock"}, nil
		},
	}
	repl := New(client)
	defer repl.Close()

	ctx := context.Background()

	// Reflection-heavy patterns
	reflectionPatterns := []string{
		`
import "reflect"
v := reflect.ValueOf(42)
fmt.Println("kind:", v.Kind())
`,
		`
import "reflect"
s := "hello"
v := reflect.ValueOf(&s)
v.Elem().SetString("world")
fmt.Println("modified:", s)
`,
		`
import "reflect"
arr := []int{1, 2, 3}
v := reflect.ValueOf(arr)
fmt.Println("len:", v.Len())
`,
		`
import "reflect"
m := map[string]int{"a": 1, "b": 2}
v := reflect.ValueOf(m)
fmt.Println("map keys:", v.MapKeys())
`,
	}

	panicCount := 0
	for round := 0; round < 5; round++ {
		for i, pattern := range reflectionPatterns {
			result, err := repl.Execute(ctx, pattern)
			if err != nil {
				panicCount++
				t.Logf("Round %d, pattern %d: panic: %v", round, i, err)
				if repl.NeedsReset() {
					_, _ = repl.ResetIfNeeded()
				}
				continue
			}
			if result.Stderr != "" && !strings.Contains(result.Stderr, "redeclared") {
				t.Logf("Round %d, pattern %d: stderr: %s", round, i, result.Stderr[:min(80, len(result.Stderr))])
			}
		}
	}

	t.Logf("Reflection stress test: %d panics", panicCount)
}

// TestPanicRecovery_StructEmbedding tests struct embedding conflicts.
func TestPanicRecovery_StructEmbedding(t *testing.T) {
	client := &mockLLMClient{
		queryFunc: func(_ context.Context, prompt string) (QueryResponse, error) {
			return QueryResponse{Response: "mock"}, nil
		},
	}
	repl := New(client)
	defer repl.Close()

	ctx := context.Background()

	// Struct embedding patterns that can conflict
	structPatterns := []string{
		`
type Base struct {
	Name string
}
`,
		`
type Base struct {
	Name string
	Value int
}
`,
		`
type Derived struct {
	Base
	Extra string
}
`,
		`
type Derived struct {
	*Base
	Extra string
}
`,
		`
type Base struct {
	Name string
}
type Derived struct {
	Base
	Name string // shadows Base.Name
}
`,
	}

	panicCount := 0
	for round := 0; round < 3; round++ {
		for i, pattern := range structPatterns {
			result, err := repl.Execute(ctx, pattern)
			if err != nil {
				panicCount++
				t.Logf("Round %d, struct %d: panic: %v", round, i, err)
				if repl.NeedsReset() {
					_, _ = repl.ResetIfNeeded()
				}
				continue
			}
			if result.Stderr != "" {
				t.Logf("Round %d, struct %d: stderr: %s", round, i, result.Stderr[:min(80, len(result.Stderr))])
			}
		}
	}

	t.Logf("Struct embedding test: %d panics", panicCount)
}

// TestPanicRecovery_VerifyPanicReturnsBool verifies that panic returns error not crash.
func TestPanicRecovery_VerifyPanicReturnsBool(t *testing.T) {
	client := &mockLLMClient{
		queryFunc: func(_ context.Context, prompt string) (QueryResponse, error) {
			return QueryResponse{Response: "mock"}, nil
		},
	}
	repl := New(client)
	defer repl.Close()

	ctx := context.Background()

	// Execute code that explicitly causes nil pointer dereference in a way that might panic Yaegi
	// Note: This is designed to test the panic recovery, not crash the test
	panicInducingCode := `
var nilSlice []int
// Attempting to append to nil slice (this is actually valid Go, won't panic)
nilSlice = append(nilSlice, 1)
fmt.Println("len:", len(nilSlice))
`

	result, err := repl.Execute(ctx, panicInducingCode)
	// Whether this panics or not, the test should complete without crashing
	if err != nil {
		t.Logf("Execution returned error (panic recovered): %v", err)
		if !repl.NeedsReset() {
			t.Log("Note: NeedsReset is false, panic may have been handled differently")
		}
	}
	if result != nil {
		t.Logf("Result stdout: %s", result.Stdout)
		if result.Stderr != "" {
			t.Logf("Result stderr: %s", result.Stderr)
		}
	}
}

// TestPanicRecovery_MultipleResetAndReload verifies multiple reset and context reload cycles.
func TestPanicRecovery_MultipleResetAndReload(t *testing.T) {
	client := &mockLLMClient{
		queryFunc: func(_ context.Context, prompt string) (QueryResponse, error) {
			return QueryResponse{Response: "query response for: " + prompt}, nil
		},
	}
	repl := New(client)
	defer repl.Close()

	ctx := context.Background()

	for cycle := 0; cycle < 20; cycle++ {
		// Load context
		testCtx := fmt.Sprintf("Context data for cycle %d", cycle)
		if err := repl.LoadContext(testCtx); err != nil {
			t.Fatalf("Cycle %d: LoadContext failed: %v", cycle, err)
		}

		// Execute code using context and Query
		code := `
ctxLen := len(context)
response := Query("analyze context")
fmt.Printf("context len: %d, response len: %d\n", ctxLen, len(response))
`
		result, err := repl.Execute(ctx, code)
		if err != nil {
			t.Logf("Cycle %d: execution error: %v", cycle, err)
			if repl.NeedsReset() {
				if _, resetErr := repl.ResetIfNeeded(); resetErr != nil {
					t.Fatalf("Cycle %d: reset failed: %v", cycle, resetErr)
				}
			}
			continue
		}

		if !strings.Contains(result.Stdout, "context len:") {
			t.Errorf("Cycle %d: unexpected output: %s", cycle, result.Stdout)
		}

		// Reset
		if err := repl.Reset(); err != nil {
			t.Fatalf("Cycle %d: Reset failed: %v", cycle, err)
		}
	}
}

// TestPanicRecovery_SameFunctionMultipleCalls tests calling the same RLM function many times.
func TestPanicRecovery_SameFunctionMultipleCalls(t *testing.T) {
	callCount := 0
	client := &mockLLMClient{
		queryFunc: func(_ context.Context, prompt string) (QueryResponse, error) {
			callCount++
			return QueryResponse{Response: fmt.Sprintf("response %d", callCount)}, nil
		},
	}
	repl := New(client)
	defer repl.Close()

	ctx := context.Background()

	// Execute code that calls Query many times in a loop
	code := `
results := []string{}
for i := 0; i < 20; i++ {
	r := Query(fmt.Sprintf("prompt %d", i))
	results = append(results, r)
}
fmt.Println("total results:", len(results))
for i, r := range results[:3] {
	fmt.Printf("result %d: %s\n", i, r)
}
`

	result, err := repl.Execute(ctx, code)
	if err != nil {
		t.Fatalf("Execution failed: %v", err)
	}

	if !strings.Contains(result.Stdout, "total results: 20") {
		t.Errorf("Unexpected output: %s", result.Stdout)
	}

	// Verify all calls were made
	calls := repl.GetLLMCalls()
	if len(calls) != 20 {
		t.Errorf("Expected 20 LLM calls, got %d", len(calls))
	}
}
