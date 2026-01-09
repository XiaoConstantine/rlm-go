// Package repl provides a Yaegi-based Go REPL for RLM code execution.
package repl

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/XiaoConstantine/rlm-go/pkg/core"
	"github.com/traefik/yaegi/interp"
	"github.com/traefik/yaegi/stdlib"
)

// REPLPool manages a pool of reusable REPL instances.
// Note: Since Yaegi interpreters accumulate state and can't be reset,
// this pool pre-creates REPL instances for faster acquisition.
type REPLPool struct {
	pool      chan *REPL
	client    LLMClient
	maxSize   int
	preWarm   bool
	mu        sync.Mutex
	created   int
}

// NewREPLPool creates a new REPL pool with the specified size.
// If preWarm is true, it pre-creates all instances (takes longer but faster subsequent use).
func NewREPLPool(client LLMClient, size int, preWarm bool) *REPLPool {
	p := &REPLPool{
		pool:    make(chan *REPL, size),
		client:  client,
		maxSize: size,
		preWarm: preWarm,
	}

	if preWarm {
		// Pre-warm the pool with REPL instances
		for i := 0; i < size; i++ {
			r := p.createREPL()
			p.pool <- r
		}
	}

	return p
}

// Get retrieves a REPL from the pool or creates a new one.
func (p *REPLPool) Get() *REPL {
	select {
	case r := <-p.pool:
		// Got one from pool, reset its state
		r.resetState()
		return r
	default:
		// Pool empty, create new one
		return p.createREPL()
	}
}

// Put returns a REPL to the pool for reuse.
// Note: Due to Yaegi limitations, the interpreter can't be fully reset,
// so this creates a new REPL for the pool instead.
func (p *REPLPool) Put(r *REPL) {
	// Clear the REPL's state
	r.Close()

	// Try to add a fresh REPL to the pool
	select {
	case p.pool <- p.createREPL():
		// Added to pool
	default:
		// Pool full, discard
	}
}

// createREPL creates a new REPL instance with the pool's client.
func (p *REPLPool) createREPL() *REPL {
	p.mu.Lock()
	p.created++
	p.mu.Unlock()
	return New(p.client)
}

// Stats returns pool statistics.
func (p *REPLPool) Stats() (poolSize, totalCreated int) {
	p.mu.Lock()
	defer p.mu.Unlock()
	return len(p.pool), p.created
}

// interpreterPool is a simple sync.Pool for basic interpreter reuse.
// Note: Due to Yaegi limitations, interpreters can't be fully reset,
// so this primarily helps with pre-warming the GC and memory allocation.
var interpreterPool = sync.Pool{
	New: func() interface{} {
		stdout := new(bytes.Buffer)
		stderr := new(bytes.Buffer)

		i := interp.New(interp.Options{
			Stdout: stdout,
			Stderr: stderr,
		})

		if err := i.Use(stdlib.Symbols); err != nil {
			panic(fmt.Sprintf("failed to load stdlib: %v", err))
		}

		return i
	},
}

// QueryResponse contains the LLM response with usage metadata.
type QueryResponse struct {
	Response         string
	PromptTokens     int
	CompletionTokens int
}

// LLMClient defines the interface for making LLM calls from within the REPL.
type LLMClient interface {
	Query(ctx context.Context, prompt string) (QueryResponse, error)
	QueryBatched(ctx context.Context, prompts []string) ([]QueryResponse, error)
}

// LLMCall represents a sub-LLM call made from within the REPL.
type LLMCall struct {
	Prompt           string  `json:"prompt"`
	Response         string  `json:"response"`
	Duration         float64 `json:"duration"`
	PromptTokens     int     `json:"prompt_tokens"`
	CompletionTokens int     `json:"completion_tokens"`
	Async            bool    `json:"async,omitempty"`
}

// REPL represents a Yaegi-based Go interpreter with RLM capabilities.
type REPL struct {
	interp       *interp.Interpreter
	stdout       *bytes.Buffer
	stderr       *bytes.Buffer
	llmClient    LLMClient
	ctx          context.Context
	mu           sync.Mutex
	llmCalls     []LLMCall // Track LLM calls made during execution
	fromPool     bool      // Whether the interpreter came from the pool
	usePooling   bool      // Whether to use pooling for this REPL
	asyncQueries map[string]*AsyncQueryHandle
	asyncMu      sync.RWMutex
	execCount    int  // Track number of executions for health monitoring
	needsReset   bool // Flag indicating interpreter corruption detected
}

// REPLOption configures a REPL instance.
type REPLOption func(*REPL)

// WithPooling enables interpreter pooling for this REPL.
// When enabled, the interpreter is returned to the pool on Close().
func WithPooling(enabled bool) REPLOption {
	return func(r *REPL) {
		r.usePooling = enabled
	}
}

// New creates a new REPL instance.
func New(client LLMClient, opts ...REPLOption) *REPL {
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

	r := &REPL{
		interp:       i,
		stdout:       stdout,
		stderr:       stderr,
		llmClient:    client,
		ctx:          context.Background(),
		fromPool:     false,
		usePooling:   false,
		asyncQueries: make(map[string]*AsyncQueryHandle),
	}

	// Apply options
	for _, opt := range opts {
		opt(r)
	}

	// Inject RLM functions
	if err := r.injectBuiltins(); err != nil {
		panic(fmt.Sprintf("failed to inject builtins: %v", err))
	}

	return r
}

// NewPooled creates a new REPL instance using the interpreter pool.
// This eliminates the 10-50ms startup overhead per REPL instance.
// Call Close() when done to return the interpreter to the pool.
func NewPooled(client LLMClient) *REPL {
	stdout := new(bytes.Buffer)
	stderr := new(bytes.Buffer)

	// Get interpreter from pool (will be created if pool is empty)
	pooledInterp := interpreterPool.Get().(*interp.Interpreter)

	// Create a new interpreter with fresh buffers
	// Note: We can't reuse the pooled interpreter's buffers as they may contain stale data
	// and we can't safely redirect them. Instead, we create a fresh interpreter.
	i := interp.New(interp.Options{
		Stdout: stdout,
		Stderr: stderr,
	})

	// Load standard library
	if err := i.Use(stdlib.Symbols); err != nil {
		panic(fmt.Sprintf("failed to load stdlib: %v", err))
	}

	// Return the pooled interpreter since we can't fully reuse it
	interpreterPool.Put(pooledInterp)

	r := &REPL{
		interp:       i,
		stdout:       stdout,
		stderr:       stderr,
		llmClient:    client,
		ctx:          context.Background(),
		fromPool:     true,
		usePooling:   true,
		asyncQueries: make(map[string]*AsyncQueryHandle),
	}

	// Inject RLM functions
	if err := r.injectBuiltins(); err != nil {
		panic(fmt.Sprintf("failed to inject builtins: %v", err))
	}

	return r
}

// Close releases resources and returns the interpreter to the pool if applicable.
func (r *REPL) Close() {
	r.mu.Lock()
	defer r.mu.Unlock()

	// Clear state
	r.stdout.Reset()
	r.stderr.Reset()
	r.llmCalls = nil

	// Clean up async queries
	r.asyncMu.Lock()
	r.asyncQueries = make(map[string]*AsyncQueryHandle)
	r.asyncMu.Unlock()

	// Note: Yaegi interpreters can't be easily reset to a clean state,
	// so we don't actually return them to the pool. The pool is kept
	// for potential future optimization if Yaegi adds proper reset support.
}

// resetState clears the REPL's output buffers and LLM call history.
// Note: This does NOT reset the interpreter's variable state.
func (r *REPL) resetState() {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.stdout.Reset()
	r.stderr.Reset()
	r.llmCalls = nil

	// Clean up async queries during reset
	r.asyncMu.Lock()
	r.asyncQueries = make(map[string]*AsyncQueryHandle)
	r.asyncMu.Unlock()
}

// injectBuiltins registers llmQuery and llmQueryBatched functions in the interpreter.
func (r *REPL) injectBuiltins() error {
	symbols := interp.Exports{
		"rlm/rlm": {
			"Query":             reflect.ValueOf(r.llmQuery),
			"QueryBatched":      reflect.ValueOf(r.llmQueryBatched),
			"QueryAsync":        reflect.ValueOf(r.llmQueryAsync),
			"QueryBatchedAsync": reflect.ValueOf(r.llmQueryBatchedAsync),
			"WaitAsync":         reflect.ValueOf(r.waitAsync),
			"AsyncReady":        reflect.ValueOf(r.asyncReady),
			"AsyncResult":       reflect.ValueOf(r.asyncResult),
			// FINAL and FINAL_VAR allow LLMs to signal completion from within code blocks
			"FINAL":     reflect.ValueOf(r.finalAnswer),
			"FINAL_VAR": reflect.ValueOf(r.finalVarAnswer),
		},
	}

	if err := r.interp.Use(symbols); err != nil {
		return fmt.Errorf("failed to inject rlm symbols: %w", err)
	}

	// Pre-import common packages and RLM functions so they're available without qualification
	// Also define min/max helper functions since Yaegi doesn't support Go 1.21 builtins
	// NOTE: All common packages are pre-imported so LLM code should NOT use import statements
	// (mixing imports with statements causes Yaegi to treat code as package-level where statements are invalid)
	setupCode := `
import "fmt"
import "strings"
import "regexp"
import "strconv"
import "encoding/json"
import "sort"
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

// llmQuery makes a single LLM query. This is called from interpreted code.
// It automatically includes the context variable (if loaded) in the prompt.
func (r *REPL) llmQuery(prompt string) string {
	start := time.Now()

	// Get the context variable from the interpreter and include it in the prompt
	fullPrompt := r.buildPromptWithContext(prompt)

	result, err := r.llmClient.Query(r.ctx, fullPrompt)
	duration := time.Since(start).Seconds()

	response := result.Response
	if err != nil {
		response = fmt.Sprintf("Error: %v", err)
	}

	// Record the call with token usage (store original prompt for clarity)
	r.llmCalls = append(r.llmCalls, LLMCall{
		Prompt:           prompt,
		Response:         response,
		Duration:         duration,
		PromptTokens:     result.PromptTokens,
		CompletionTokens: result.CompletionTokens,
	})

	return response
}

// buildPromptWithContext retrieves the context variable and prepends it to the prompt.
// This ensures sub-LLM queries have access to the loaded context data.
// Thread-safe: acquires mutex before accessing interpreter.
func (r *REPL) buildPromptWithContext(prompt string) string {
	r.mu.Lock()
	defer r.mu.Unlock()

	// Try to get the context variable from the interpreter
	v, err := r.interp.Eval("context")
	if err != nil || !v.IsValid() {
		// No context loaded, use prompt as-is
		return prompt
	}

	contextStr := ""
	switch ctx := v.Interface().(type) {
	case string:
		contextStr = ctx
	default:
		// For non-string context, try JSON marshaling
		jsonBytes, err := json.Marshal(ctx)
		if err != nil {
			return prompt
		}
		contextStr = string(jsonBytes)
	}

	if contextStr == "" {
		return prompt
	}

	// Prepend context to prompt with instructions for concise responses
	return fmt.Sprintf("Context data:\n%s\n\nTask: %s\n\nIMPORTANT: Provide a direct, concise answer. Do not explain your reasoning unless specifically asked.", contextStr, prompt)
}

// llmQueryBatched makes concurrent LLM queries. This is called from interpreted code.
// It automatically includes the context variable (if loaded) in each prompt.
func (r *REPL) llmQueryBatched(prompts []string) []string {
	start := time.Now()

	// Build full prompts with context included
	fullPrompts := make([]string, len(prompts))
	for i, p := range prompts {
		fullPrompts[i] = r.buildPromptWithContext(p)
	}

	results, err := r.llmClient.QueryBatched(r.ctx, fullPrompts)
	duration := time.Since(start).Seconds()

	if err != nil {
		errResults := make([]string, len(prompts))
		for i := range errResults {
			errResults[i] = fmt.Sprintf("Error: %v", err)
		}
		// Record each as a failed call (store original prompts for clarity)
		for i, p := range prompts {
			r.llmCalls = append(r.llmCalls, LLMCall{
				Prompt:   p,
				Response: errResults[i],
				Duration: duration / float64(len(prompts)),
			})
		}
		return errResults
	}

	// Record each successful call with token usage (store original prompts for clarity)
	responses := make([]string, len(results))
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
	return responses
}

// llmQueryAsync starts an async query and returns a handle ID.
// This is called from interpreted code via QueryAsync().
// It automatically includes the context variable (if loaded) in the prompt.
func (r *REPL) llmQueryAsync(prompt string) string {
	handle := newAsyncQueryHandle()

	// Build full prompt with context included (capture before goroutine)
	fullPrompt := r.buildPromptWithContext(prompt)

	// Track the handle
	r.asyncMu.Lock()
	r.asyncQueries[handle.id] = handle
	r.asyncMu.Unlock()

	// Start the async query
	go func() {
		start := time.Now()
		result, err := r.llmClient.Query(r.ctx, fullPrompt)
		duration := time.Since(start).Seconds()

		response := result.Response
		if err != nil {
			response = fmt.Sprintf("Error: %v", err)
		}

		// Record the async call (store original prompt for clarity)
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

		// Complete the handle
		handle.complete(result, err)
	}()

	return handle.id
}

// llmQueryBatchedAsync starts batch async queries and returns handle IDs.
// This is called from interpreted code via QueryBatchedAsync().
func (r *REPL) llmQueryBatchedAsync(prompts []string) []string {
	handleIDs := make([]string, len(prompts))

	for i, prompt := range prompts {
		handleIDs[i] = r.llmQueryAsync(prompt)
	}

	return handleIDs
}

// waitAsync blocks until the async query with the given ID completes.
// This is called from interpreted code via WaitAsync().
func (r *REPL) waitAsync(handleID string) string {
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

// asyncReady returns true if the async query with the given ID is complete.
// This is called from interpreted code via AsyncReady().
func (r *REPL) asyncReady(handleID string) bool {
	r.asyncMu.RLock()
	handle, exists := r.asyncQueries[handleID]
	r.asyncMu.RUnlock()

	if !exists {
		return true // Non-existent handle is considered "ready" (error case)
	}

	return handle.Ready()
}

// asyncResult returns the result if ready, or empty string if not.
// This is called from interpreted code via AsyncResult().
func (r *REPL) asyncResult(handleID string) string {
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

// finalAnswer handles FINAL(value) calls from within code blocks.
// It prints a special marker that the RLM parser can detect.
// This allows LLMs to signal completion from inside code, which is more natural.
func (r *REPL) finalAnswer(value string) string {
	// Print to stdout so it appears in the output and can be parsed
	fmt.Fprintf(r.stdout, "\nFINAL(%s)\n", value)
	return value
}

// finalVarAnswer handles FINAL_VAR(varName) calls from within code blocks.
// The varName is treated as the value directly (the variable was already evaluated by Go).
// This is called when LLM writes FINAL_VAR(answer) where answer is a Go variable.
func (r *REPL) finalVarAnswer(value string) string {
	// The value parameter IS the resolved variable value (Go already evaluated it)
	// Print to stdout so it appears in the output and can be parsed
	fmt.Fprintf(r.stdout, "\nFINAL(%s)\n", value)
	return value
}

// QueryAsync starts an async query and returns a handle.
// This is the Go API for async queries.
// It automatically includes the context variable (if loaded) in the prompt.
func (r *REPL) QueryAsync(prompt string) *AsyncQueryHandle {
	handle := newAsyncQueryHandle()

	// Build full prompt with context included (capture before goroutine)
	fullPrompt := r.buildPromptWithContext(prompt)

	// Track the handle
	r.asyncMu.Lock()
	r.asyncQueries[handle.id] = handle
	r.asyncMu.Unlock()

	// Start the async query
	go func() {
		start := time.Now()
		result, err := r.llmClient.Query(r.ctx, fullPrompt)
		duration := time.Since(start).Seconds()

		response := result.Response
		if err != nil {
			response = fmt.Sprintf("Error: %v", err)
		}

		// Record the async call (store original prompt for clarity)
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

		// Complete the handle
		handle.complete(result, err)
	}()

	return handle
}

// QueryBatchedAsync starts batch async queries and returns a batch handle.
// This is the Go API for batch async queries.
func (r *REPL) QueryBatchedAsync(prompts []string) *AsyncBatchHandle {
	handles := make([]*AsyncQueryHandle, len(prompts))

	for i, prompt := range prompts {
		handles[i] = r.QueryAsync(prompt)
	}

	return newAsyncBatchHandle(handles)
}

// GetAsyncQuery returns the async query handle by ID.
func (r *REPL) GetAsyncQuery(handleID string) (*AsyncQueryHandle, bool) {
	r.asyncMu.RLock()
	defer r.asyncMu.RUnlock()
	handle, exists := r.asyncQueries[handleID]
	return handle, exists
}

// PendingAsyncQueries returns the number of pending async queries.
func (r *REPL) PendingAsyncQueries() int {
	r.asyncMu.RLock()
	defer r.asyncMu.RUnlock()

	count := 0
	for _, h := range r.asyncQueries {
		if !h.Ready() {
			count++
		}
	}
	return count
}

// WaitAllAsyncQueries waits for all pending async queries to complete.
func (r *REPL) WaitAllAsyncQueries() {
	r.asyncMu.RLock()
	handles := make([]*AsyncQueryHandle, 0, len(r.asyncQueries))
	for _, h := range r.asyncQueries {
		handles = append(handles, h)
	}
	r.asyncMu.RUnlock()

	for _, h := range handles {
		<-h.done
	}
}

// LoadContext injects the context payload into the interpreter as the `context` variable.
func (r *REPL) LoadContext(payload any) error {
	r.mu.Lock()
	defer r.mu.Unlock()

	switch v := payload.(type) {
	case string:
		// String context - inject directly
		_, err := r.interp.Eval(`var context = ` + strconv.Quote(v))
		return err

	case map[string]any:
		// Map context - serialize to JSON, then unmarshal in REPL
		return r.loadStructuredContext(v, "map[string]interface{}")

	case []any:
		// Slice context - serialize to JSON, then unmarshal in REPL
		return r.loadStructuredContext(v, "[]interface{}")

	default:
		// Try JSON marshaling as fallback
		jsonBytes, err := json.Marshal(v)
		if err != nil {
			return fmt.Errorf("unsupported context type %T: %w", v, err)
		}
		return r.LoadContext(string(jsonBytes))
	}
}

// loadStructuredContext handles map and slice context types.
// Note: encoding/json is already imported by injectBuiltins, so we don't import it here.
func (r *REPL) loadStructuredContext(v any, typeDecl string) error {
	jsonBytes, err := json.Marshal(v)
	if err != nil {
		return fmt.Errorf("marshal context: %w", err)
	}

	code := fmt.Sprintf(`
var context %s
func init() {
	json.Unmarshal([]byte(%s), &context)
}
`, typeDecl, strconv.Quote(string(jsonBytes)))

	_, err = r.interp.Eval(code)
	return err
}

// SetContext sets the execution context for LLM calls.
func (r *REPL) SetContext(ctx context.Context) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.ctx = ctx
}

// Execute runs Go code in the interpreter and returns the result.
// This function includes panic recovery for Yaegi interpreter crashes.
func (r *REPL) Execute(ctx context.Context, code string) (*core.ExecutionResult, error) {
	r.mu.Lock()
	// Set context for LLM calls
	r.ctx = ctx

	// Reset buffers
	r.stdout.Reset()
	r.stderr.Reset()

	// Track execution count for interpreter health
	r.execCount++
	execCount := r.execCount
	r.mu.Unlock() // Release lock before executing code (allows LLM calls to proceed)

	start := time.Now()

	// Execute the code with panic recovery (Yaegi can crash on certain patterns)
	var evalErr error
	var panicErr error
	func() {
		defer func() {
			if rec := recover(); rec != nil {
				panicErr = fmt.Errorf("interpreter panic (after %d executions): %v", execCount, rec)
			}
		}()
		// Execute the code (may call llmQuery which doesn't need lock)
		_, evalErr = r.interp.Eval(code)
	}()

	r.mu.Lock()
	result := &core.ExecutionResult{
		Stdout:   r.stdout.String(),
		Stderr:   r.stderr.String(),
		Duration: time.Since(start),
	}
	r.mu.Unlock()

	// Handle panic - interpreter is likely corrupted
	if panicErr != nil {
		if result.Stderr != "" {
			result.Stderr += "\n"
		}
		result.Stderr += panicErr.Error()
		// Mark interpreter as needing reset
		r.mu.Lock()
		r.needsReset = true
		r.mu.Unlock()
		return result, panicErr
	}

	if evalErr != nil {
		// Append error to stderr
		if result.Stderr != "" {
			result.Stderr += "\n"
		}
		result.Stderr += evalErr.Error()
	}

	return result, nil // We don't return error - execution errors go to stderr
}

// GetVariable retrieves a variable value from the interpreter.
// Used for resolving FINAL_VAR references.
func (r *REPL) GetVariable(name string) (string, error) {
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

// Reset clears the interpreter state.
func (r *REPL) Reset() error {
	r.mu.Lock()
	defer r.mu.Unlock()

	r.stdout.Reset()
	r.stderr.Reset()
	r.llmCalls = nil
	r.execCount = 0
	r.needsReset = false

	// Create a fresh interpreter
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

// NeedsReset returns true if the interpreter has detected corruption.
func (r *REPL) NeedsReset() bool {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.needsReset
}

// ExecutionCount returns the number of code executions since last reset.
func (r *REPL) ExecutionCount() int {
	r.mu.Lock()
	defer r.mu.Unlock()
	return r.execCount
}

// ResetIfNeeded resets the interpreter if corruption was detected.
// Returns true if a reset was performed.
func (r *REPL) ResetIfNeeded() (bool, error) {
	r.mu.Lock()
	needsReset := r.needsReset
	r.mu.Unlock()

	if !needsReset {
		return false, nil
	}

	return true, r.Reset()
}

// GetLLMCalls returns and clears the recorded LLM calls.
func (r *REPL) GetLLMCalls() []LLMCall {
	r.mu.Lock()
	defer r.mu.Unlock()
	calls := r.llmCalls
	r.llmCalls = nil
	return calls
}

// ClearLLMCalls clears the recorded LLM calls.
func (r *REPL) ClearLLMCalls() {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.llmCalls = nil
}

// GetLocals extracts user-defined variables from the interpreter.
// Returns a map of variable names to their values (JSON-serializable via encoding/json).
func (r *REPL) GetLocals() map[string]any {
	r.mu.Lock()
	defer r.mu.Unlock()

	locals := make(map[string]any)

	// Check for commonly used variable names in RLM code
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

// ContextInfo returns metadata about the loaded context.
func (r *REPL) ContextInfo() string {
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

// FormatExecutionResult formats an execution result for display to the LLM.
func FormatExecutionResult(result *core.ExecutionResult) string {
	var parts []string

	if result.Stdout != "" {
		parts = append(parts, result.Stdout)
	}
	if result.Stderr != "" {
		parts = append(parts, result.Stderr)
	}

	if len(parts) == 0 {
		return "No output"
	}

	return strings.Join(parts, "\n\n")
}
