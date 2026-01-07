// Package repl provides recursive REPL capabilities for multi-depth RLM.
package repl

import (
	"bytes"
	"context"
	"fmt"
	"reflect"
	"sync"
	"time"

	"github.com/XiaoConstantine/rlm-go/pkg/core"
	"github.com/traefik/yaegi/interp"
	"github.com/traefik/yaegi/stdlib"
)

// RecursiveLLMClient extends LLMClient with recursive RLM capabilities.
// It allows sub-LLMs to spawn their own sub-LLMs up to a maximum depth.
type RecursiveLLMClient interface {
	LLMClient

	// QueryWithRLM performs a recursive RLM query at the specified depth.
	// Returns the final answer from the nested RLM execution.
	QueryWithRLM(ctx context.Context, prompt string, depth int) (QueryResponse, error)

	// CurrentDepth returns the current recursion depth.
	CurrentDepth() int

	// MaxDepth returns the maximum allowed recursion depth.
	MaxDepth() int
}

// RecursiveREPL wraps a standard REPL with recursive RLM capabilities.
// It injects additional functions for multi-depth recursion.
type RecursiveREPL struct {
	interp           *interp.Interpreter
	stdout           *bytes.Buffer
	stderr           *bytes.Buffer
	client           RecursiveLLMClient
	ctx              context.Context
	mu               sync.Mutex
	llmCalls         []LLMCall
	recursiveCalls   []RecursiveCall // Track recursive RLM calls
	asyncQueries     map[string]*AsyncQueryHandle
	asyncMu          sync.RWMutex
	recursionContext *core.RecursionContext
}

// RecursiveCall represents a recursive RLM call made from within the REPL.
type RecursiveCall struct {
	Prompt           string        `json:"prompt"`
	Response         string        `json:"response"`
	Depth            int           `json:"depth"`
	Duration         time.Duration `json:"duration"`
	PromptTokens     int           `json:"prompt_tokens"`
	CompletionTokens int           `json:"completion_tokens"`
}

// NewRecursiveREPL creates a new RecursiveREPL instance with recursive capabilities.
func NewRecursiveREPL(client RecursiveLLMClient, recursionCtx *core.RecursionContext) *RecursiveREPL {
	stdout := new(bytes.Buffer)
	stderr := new(bytes.Buffer)

	i := interp.New(interp.Options{
		Stdout: stdout,
		Stderr: stderr,
	})

	// Load standard library
	if err := i.Use(stdlib.Symbols); err != nil {
		panic(fmt.Sprintf("failed to load stdlib: %v", err))
	}

	r := &RecursiveREPL{
		interp:           i,
		stdout:           stdout,
		stderr:           stderr,
		client:           client,
		ctx:              context.Background(),
		asyncQueries:     make(map[string]*AsyncQueryHandle),
		recursionContext: recursionCtx,
	}

	// Inject RLM functions including recursive ones
	if err := r.injectBuiltins(); err != nil {
		panic(fmt.Sprintf("failed to inject builtins: %v", err))
	}

	return r
}

// injectBuiltins registers all RLM functions including recursive ones.
func (r *RecursiveREPL) injectBuiltins() error {
	symbols := interp.Exports{
		"rlm/rlm": {
			// Standard Query functions
			"Query":             reflect.ValueOf(r.llmQuery),
			"QueryBatched":      reflect.ValueOf(r.llmQueryBatched),
			"QueryAsync":        reflect.ValueOf(r.llmQueryAsync),
			"QueryBatchedAsync": reflect.ValueOf(r.llmQueryBatchedAsync),
			"WaitAsync":         reflect.ValueOf(r.waitAsync),
			"AsyncReady":        reflect.ValueOf(r.asyncReady),
			"AsyncResult":       reflect.ValueOf(r.asyncResult),
			// Recursive RLM functions
			"QueryWithRLM":        reflect.ValueOf(r.queryWithRLM),
			"QueryBatchedWithRLM": reflect.ValueOf(r.queryBatchedWithRLM),
			"CurrentDepth":        reflect.ValueOf(r.currentDepth),
			"MaxDepth":            reflect.ValueOf(r.maxDepth),
			"CanRecurse":          reflect.ValueOf(r.canRecurse),
		},
	}

	if err := r.interp.Use(symbols); err != nil {
		return fmt.Errorf("failed to inject rlm symbols: %w", err)
	}

	// Pre-import common packages and RLM functions
	setupCode := `
import "fmt"
import "strings"
import "regexp"
import . "rlm/rlm"

// min returns the smaller of two integers (Go 1.21 builtin not supported in Yaegi)
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// max returns the larger of two integers (Go 1.21 builtin not supported in Yaegi)
func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}
`
	_, err := r.interp.Eval(setupCode)
	return err
}

// llmQuery makes a single LLM query (standard, non-recursive).
func (r *RecursiveREPL) llmQuery(prompt string) string {
	start := time.Now()
	result, err := r.client.Query(r.ctx, prompt)
	duration := time.Since(start).Seconds()

	response := result.Response
	if err != nil {
		response = fmt.Sprintf("Error: %v", err)
	}

	r.mu.Lock()
	r.llmCalls = append(r.llmCalls, LLMCall{
		Prompt:           prompt,
		Response:         response,
		Duration:         duration,
		PromptTokens:     result.PromptTokens,
		CompletionTokens: result.CompletionTokens,
	})
	r.mu.Unlock()

	return response
}

// llmQueryBatched makes concurrent LLM queries (standard, non-recursive).
func (r *RecursiveREPL) llmQueryBatched(prompts []string) []string {
	start := time.Now()
	results, err := r.client.QueryBatched(r.ctx, prompts)
	duration := time.Since(start).Seconds()

	if err != nil {
		errResults := make([]string, len(prompts))
		for i := range errResults {
			errResults[i] = fmt.Sprintf("Error: %v", err)
		}
		r.mu.Lock()
		for i, p := range prompts {
			r.llmCalls = append(r.llmCalls, LLMCall{
				Prompt:   p,
				Response: errResults[i],
				Duration: duration / float64(len(prompts)),
			})
		}
		r.mu.Unlock()
		return errResults
	}

	responses := make([]string, len(results))
	r.mu.Lock()
	for i, p := range prompts {
		responses[i] = results[i].Response
		r.llmCalls = append(r.llmCalls, LLMCall{
			Prompt:           p,
			Response:         results[i].Response,
			Duration:         duration / float64(len(prompts)),
			PromptTokens:     results[i].PromptTokens,
			CompletionTokens: results[i].CompletionTokens,
		})
	}
	r.mu.Unlock()
	return responses
}

// queryWithRLM performs a recursive RLM query at the specified depth.
// This spawns a new sub-LLM that can itself use Query() and QueryWithRLM().
func (r *RecursiveREPL) queryWithRLM(prompt string, depth int) string {
	start := time.Now()

	// Use the depth from parameter, defaulting to current depth + 1
	targetDepth := depth
	if targetDepth <= 0 {
		targetDepth = r.client.CurrentDepth() + 1
	}

	result, err := r.client.QueryWithRLM(r.ctx, prompt, targetDepth)
	duration := time.Since(start)

	response := result.Response
	if err != nil {
		response = fmt.Sprintf("Error: %v", err)
	}

	r.mu.Lock()
	r.recursiveCalls = append(r.recursiveCalls, RecursiveCall{
		Prompt:           prompt,
		Response:         response,
		Depth:            targetDepth,
		Duration:         duration,
		PromptTokens:     result.PromptTokens,
		CompletionTokens: result.CompletionTokens,
	})
	r.mu.Unlock()

	return response
}

// queryBatchedWithRLM performs multiple recursive RLM queries concurrently.
func (r *RecursiveREPL) queryBatchedWithRLM(prompts []string, depth int) []string {
	results := make([]string, len(prompts))
	var wg sync.WaitGroup
	var mu sync.Mutex

	targetDepth := depth
	if targetDepth <= 0 {
		targetDepth = r.client.CurrentDepth() + 1
	}

	for i, prompt := range prompts {
		wg.Add(1)
		go func(idx int, p string) {
			defer wg.Done()
			result := r.queryWithRLM(p, targetDepth)
			mu.Lock()
			results[idx] = result
			mu.Unlock()
		}(i, prompt)
	}

	wg.Wait()
	return results
}

// currentDepth returns the current recursion depth.
func (r *RecursiveREPL) currentDepth() int {
	return r.client.CurrentDepth()
}

// maxDepth returns the maximum allowed recursion depth.
func (r *RecursiveREPL) maxDepth() int {
	return r.client.MaxDepth()
}

// canRecurse returns true if another level of recursion is allowed.
func (r *RecursiveREPL) canRecurse() bool {
	return r.client.CurrentDepth() < r.client.MaxDepth()
}

// Async query methods (same as regular REPL)

func (r *RecursiveREPL) llmQueryAsync(prompt string) string {
	handle := newAsyncQueryHandle()

	r.asyncMu.Lock()
	r.asyncQueries[handle.id] = handle
	r.asyncMu.Unlock()

	go func() {
		start := time.Now()
		result, err := r.client.Query(r.ctx, prompt)
		duration := time.Since(start).Seconds()

		response := result.Response
		if err != nil {
			response = fmt.Sprintf("Error: %v", err)
		}

		r.mu.Lock()
		r.llmCalls = append(r.llmCalls, LLMCall{
			Prompt:           prompt,
			Response:         response,
			Duration:         duration,
			PromptTokens:     result.PromptTokens,
			CompletionTokens: result.CompletionTokens,
			Async:            true,
		})
		r.mu.Unlock()

		handle.complete(result, err)
	}()

	return handle.id
}

func (r *RecursiveREPL) llmQueryBatchedAsync(prompts []string) []string {
	handleIDs := make([]string, len(prompts))
	for i, prompt := range prompts {
		handleIDs[i] = r.llmQueryAsync(prompt)
	}
	return handleIDs
}

func (r *RecursiveREPL) waitAsync(handleID string) string {
	r.asyncMu.RLock()
	handle, exists := r.asyncQueries[handleID]
	r.asyncMu.RUnlock()

	if !exists {
		return fmt.Sprintf("Error: async query %s not found", handleID)
	}

	result, err := handle.Wait()
	if err != nil {
		return fmt.Sprintf("Error: %v", err)
	}
	return result
}

func (r *RecursiveREPL) asyncReady(handleID string) bool {
	r.asyncMu.RLock()
	handle, exists := r.asyncQueries[handleID]
	r.asyncMu.RUnlock()

	if !exists {
		return true
	}

	return handle.Ready()
}

func (r *RecursiveREPL) asyncResult(handleID string) string {
	r.asyncMu.RLock()
	handle, exists := r.asyncQueries[handleID]
	r.asyncMu.RUnlock()

	if !exists {
		return fmt.Sprintf("Error: async query %s not found", handleID)
	}

	result, ready := handle.Result()
	if !ready {
		return ""
	}
	return result
}

// LoadContext injects the context payload into the interpreter.
func (r *RecursiveREPL) LoadContext(payload any) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	switch v := payload.(type) {
	case string:
		_, err := r.interp.Eval(`var context = ` + fmt.Sprintf("%q", v))
		return err
	default:
		// For other types, serialize to string
		_, err := r.interp.Eval(`var context = ` + fmt.Sprintf("%q", fmt.Sprintf("%v", v)))
		return err
	}
}

// SetContext sets the execution context for LLM calls.
func (r *RecursiveREPL) SetContext(ctx context.Context) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.ctx = ctx
}

// Execute runs Go code in the interpreter and returns the result.
func (r *RecursiveREPL) Execute(ctx context.Context, code string) (*core.ExecutionResult, error) {
	r.mu.Lock()
	r.ctx = ctx
	r.stdout.Reset()
	r.stderr.Reset()
	r.mu.Unlock()

	start := time.Now()

	_, err := r.interp.Eval(code)

	r.mu.Lock()
	result := &core.ExecutionResult{
		Stdout:   r.stdout.String(),
		Stderr:   r.stderr.String(),
		Duration: time.Since(start),
	}
	r.mu.Unlock()

	if err != nil {
		if result.Stderr != "" {
			result.Stderr += "\n"
		}
		result.Stderr += err.Error()
	}

	return result, nil
}

// GetVariable retrieves a variable value from the interpreter.
func (r *RecursiveREPL) GetVariable(name string) (string, error) {
	r.mu.Lock()
	defer r.mu.Unlock()

	v, err := r.interp.Eval(name)
	if err != nil {
		return "", fmt.Errorf("variable %q not found: %w", name, err)
	}

	if !v.IsValid() {
		return "", fmt.Errorf("variable %q is invalid", name)
	}

	return fmt.Sprintf("%v", v.Interface()), nil
}

// GetLLMCalls returns and clears the recorded LLM calls.
func (r *RecursiveREPL) GetLLMCalls() []LLMCall {
	r.mu.Lock()
	defer r.mu.Unlock()
	calls := r.llmCalls
	r.llmCalls = nil
	return calls
}

// GetRecursiveCalls returns and clears the recorded recursive RLM calls.
func (r *RecursiveREPL) GetRecursiveCalls() []RecursiveCall {
	r.mu.Lock()
	defer r.mu.Unlock()
	calls := r.recursiveCalls
	r.recursiveCalls = nil
	return calls
}

// Close releases resources.
func (r *RecursiveREPL) Close() {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.stdout.Reset()
	r.stderr.Reset()
	r.llmCalls = nil
	r.recursiveCalls = nil

	r.asyncMu.Lock()
	r.asyncQueries = make(map[string]*AsyncQueryHandle)
	r.asyncMu.Unlock()
}

// Reset clears the interpreter state.
func (r *RecursiveREPL) Reset() error {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.stdout.Reset()
	r.stderr.Reset()
	r.llmCalls = nil
	r.recursiveCalls = nil

	i := interp.New(interp.Options{
		Stdout: r.stdout,
		Stderr: r.stderr,
	})

	if err := i.Use(stdlib.Symbols); err != nil {
		return fmt.Errorf("failed to load stdlib: %w", err)
	}

	r.interp = i

	return r.injectBuiltins()
}

// ContextInfo returns metadata about the loaded context.
func (r *RecursiveREPL) ContextInfo() string {
	r.mu.Lock()
	defer r.mu.Unlock()

	v, err := r.interp.Eval("context")
	if err != nil {
		return "context not loaded"
	}

	if !v.IsValid() {
		return "context not loaded"
	}

	iface := v.Interface()
	switch ctx := iface.(type) {
	case string:
		return fmt.Sprintf("type=string, len=%d", len(ctx))
	default:
		return fmt.Sprintf("type=%T", ctx)
	}
}

// GetLocals extracts user-defined variables from the interpreter.
func (r *RecursiveREPL) GetLocals() map[string]any {
	r.mu.Lock()
	defer r.mu.Unlock()

	locals := make(map[string]any)

	varNames := []string{
		"context", "result", "answer", "data", "output", "response",
		"analysis", "summary", "final_answer", "count", "total",
		"items", "records", "values", "results", "findings",
	}

	for _, name := range varNames {
		v, err := r.interp.Eval(name)
		if err != nil || !v.IsValid() {
			continue
		}
		locals[name] = v.Interface()
	}

	return locals
}
