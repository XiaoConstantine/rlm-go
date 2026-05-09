// Package repl provides a Yaegi-based Go REPL for RLM code execution.
package repl

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"reflect"
	"strconv"
	"sync"
	"time"

	"github.com/XiaoConstantine/rlm-go/pkg/contextindex"
	"github.com/XiaoConstantine/rlm-go/pkg/core"
	"github.com/XiaoConstantine/rlm-go/pkg/interpreter"
	"github.com/traefik/yaegi/interp"
	"github.com/traefik/yaegi/stdlib"
)

// REPLPool manages a pool of reusable REPL instances.
// Note: Since Yaegi interpreters accumulate state and can't be reset,
// this pool pre-creates REPL instances for faster acquisition.
type REPLPool struct {
	pool    chan *REPL
	client  LLMClient
	maxSize int
	preWarm bool
	mu      sync.Mutex
	created int
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

// Note: Yaegi interpreters cannot be reset to a clean state, so traditional
// pooling (reusing the same interpreter across requests) is not possible.
// See REPLPool for a pre-creation approach that helps with startup latency.

// QueryResponse is an alias for core.QueryResponse for backward compatibility.
type QueryResponse = core.QueryResponse

// LLMClient defines the interface for making LLM calls from within the REPL.
type LLMClient interface {
	Query(ctx context.Context, prompt string) (core.QueryResponse, error)
	QueryBatched(ctx context.Context, prompts []string) ([]core.QueryResponse, error)
}

// LLMCall is an alias for core.LLMCall for backward compatibility.
type LLMCall = core.LLMCall

// REPL represents a Yaegi-based Go interpreter with RLM capabilities.
type REPL struct {
	interp        *interp.Interpreter
	stdout        *bytes.Buffer
	stderr        *bytes.Buffer
	llmClient     LLMClient
	ctx           context.Context
	mu            sync.Mutex
	llmCalls      []LLMCall // Track LLM calls made during execution
	asyncQueries  map[string]*AsyncQueryHandle
	asyncMu       sync.RWMutex
	finalMu       sync.RWMutex
	finalSet      bool
	finalValue    string
	maxFullQuery  int
	execCount     int  // Track number of executions for health monitoring
	needsReset    bool // Flag indicating interpreter corruption detected
	injectedNames map[string]struct{}
}

// REPLOption configures a REPL instance.
type REPLOption func(*REPL)

// WithMaxFullContextQueryChars blocks Query/QueryBatched when the loaded context
// is larger than max chars. Zero disables the guard.
func WithMaxFullContextQueryChars(max int) REPLOption {
	return func(r *REPL) {
		r.SetMaxFullContextQueryChars(max)
	}
}

// SetMaxFullContextQueryChars updates the full-context Query guard. Zero disables it.
func (r *REPL) SetMaxFullContextQueryChars(max int) {
	if max < 0 {
		max = 0
	}
	r.mu.Lock()
	defer r.mu.Unlock()
	r.maxFullQuery = max
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

// NewPooled creates a new REPL instance.
// Note: Due to Yaegi interpreter limitations (can't be reset to clean state),
// this function simply creates a fresh REPL each time.
// Consider using REPLPool for pre-created REPL instances if startup time matters.
// Deprecated: Use New() directly instead.
func NewPooled(client LLMClient) *REPL {
	return New(client)
}

// Close releases resources and returns the interpreter to the pool if applicable.
func (r *REPL) Close() {
	r.mu.Lock()
	defer r.mu.Unlock()

	// Clear state
	r.stdout.Reset()
	r.stderr.Reset()
	r.llmCalls = nil
	r.clearFinalLocked()

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
	r.clearFinalLocked()

	// Clean up async queries during reset
	r.asyncMu.Lock()
	r.asyncQueries = make(map[string]*AsyncQueryHandle)
	r.asyncMu.Unlock()
}

// injectBuiltins registers llmQuery and llmQueryBatched functions in the interpreter.
func (r *REPL) injectBuiltins() error {
	rlmSymbols := map[string]reflect.Value{
		"Query":             reflect.ValueOf(r.llmQuery),
		"QueryRaw":          reflect.ValueOf(r.llmQueryRaw),
		"QueryWith":         reflect.ValueOf(r.llmQueryWith),
		"QueryBatched":      reflect.ValueOf(r.llmQueryBatched),
		"QueryBatchedRaw":   reflect.ValueOf(r.llmQueryBatchedRaw),
		"FindRelevant":      reflect.ValueOf(r.findRelevant),
		"GetChunk":          reflect.ValueOf(r.getChunk),
		"GetContext":        reflect.ValueOf(r.getContextRange),
		"ChunkCount":        reflect.ValueOf(r.chunkCount),
		"LineCount":         reflect.ValueOf(r.lineCount),
		"QueryAsync":        reflect.ValueOf(r.llmQueryAsync),
		"QueryBatchedAsync": reflect.ValueOf(r.llmQueryBatchedAsync),
		"WaitAsync":         reflect.ValueOf(r.waitAsync),
		"AsyncReady":        reflect.ValueOf(r.asyncReady),
		"AsyncResult":       reflect.ValueOf(r.asyncResult),
		// FINAL and FINAL_VAR allow LLMs to signal completion from within code blocks
		"FINAL":     reflect.ValueOf(r.finalAnswer),
		"FINAL_VAR": reflect.ValueOf(r.finalVarAnswer),
	}

	symbols := interp.Exports{"rlm/rlm": rlmSymbols}

	if err := r.interp.Use(symbols); err != nil {
		return fmt.Errorf("failed to inject rlm symbols: %w", err)
	}

	// Record the names so InjectSymbols can guard against collisions
	// with exactly the builtins this instance actually registered.
	r.injectedNames = make(map[string]struct{}, len(rlmSymbols))
	for name := range rlmSymbols {
		r.injectedNames[name] = struct{}{}
	}

	// Use the shared extended setup code which includes all common imports
	return interpreter.RunSetup(r.interp, interpreter.SetupCodeExtended)
}

// InjectSymbols merges external symbols into the REPL's "rlm/rlm" namespace.
// Call this after REPL creation and context loading, before Execute().
func (r *REPL) InjectSymbols(symbols map[string]reflect.Value) error {
	if len(symbols) == 0 {
		return nil
	}

	for name := range symbols {
		if _, collision := r.injectedNames[name]; collision {
			return fmt.Errorf("inject symbols: %q collides with existing RLM builtin", name)
		}
	}

	exports := interp.Exports{
		"rlm/rlm": make(map[string]reflect.Value, len(symbols)),
	}
	for name, val := range symbols {
		exports["rlm/rlm"][name] = val
	}

	r.mu.Lock()
	defer r.mu.Unlock()

	if err := r.interp.Use(exports); err != nil {
		return fmt.Errorf("inject symbols: failed to merge symbols: %w", err)
	}

	// Refresh dot-imported names so newly injected symbols are available as unqualified identifiers.
	if _, err := r.interp.Eval(`import . "rlm/rlm"`); err != nil {
		return fmt.Errorf("inject symbols: failed to refresh imports: %w", err)
	}

	return nil
}

// llmQuery makes a single LLM query. This is called from interpreted code.
// It automatically includes the context variable (if loaded) in the prompt.
func (r *REPL) llmQuery(prompt string) string {
	start := time.Now()

	if err := r.fullContextQueryBlocked("Query"); err != nil {
		response := fmt.Sprintf("Error: %v", err)
		r.recordLLMCall(prompt, response, time.Since(start).Seconds(), QueryResponse{})
		return response
	}

	// Get the context variable from the interpreter and include it in the prompt
	fullPrompt := r.buildPromptWithContext(prompt)

	result, err := r.llmClient.Query(r.ctx, fullPrompt)
	duration := time.Since(start).Seconds()

	response := result.Response
	if err != nil {
		response = fmt.Sprintf("Error: %v", err)
	}

	// Record the call with token usage (store original prompt for clarity)
	r.recordLLMCall(prompt, response, duration, result)

	return response
}

func (r *REPL) contextString() (string, bool) {
	r.mu.Lock()
	defer r.mu.Unlock()

	// Try to get the context variable from the interpreter
	v, err := r.interp.Eval("context")
	if err != nil || !v.IsValid() {
		// No context loaded, use prompt as-is
		return "", false
	}

	switch ctx := v.Interface().(type) {
	case string:
		return ctx, ctx != ""
	default:
		// For non-string context, try JSON marshaling
		jsonBytes, err := json.Marshal(ctx)
		if err != nil {
			return "", false
		}
		contextStr := string(jsonBytes)
		return contextStr, contextStr != ""
	}
}

// buildPromptWithContext retrieves the context variable and prepends it to the prompt.
// This ensures sub-LLM queries have access to the loaded context data.
func (r *REPL) buildPromptWithContext(prompt string) string {
	contextStr, ok := r.contextString()
	if !ok {
		return prompt
	}

	return r.buildPromptWithProvidedContext(contextStr, prompt)
}

func (r *REPL) buildPromptWithProvidedContext(contextStr, prompt string) string {
	if contextStr == "" {
		return prompt
	}

	// Prepend context to prompt with instructions for concise responses
	return fmt.Sprintf("Context data:\n%s\n\nTask: %s\n\nIMPORTANT: Provide a direct, concise answer. Do not explain your reasoning unless specifically asked.", contextStr, prompt)
}

func (r *REPL) fullContextQueryBlocked(name string) error {
	r.mu.Lock()
	maxChars := r.maxFullQuery
	r.mu.Unlock()

	if maxChars <= 0 {
		return nil
	}

	contextStr, ok := r.contextString()
	if !ok || len(contextStr) <= maxChars {
		return nil
	}

	return fmt.Errorf("%s would prepend the full context (%d chars), exceeding the limit of %d chars; use QueryWith(contextSlice, prompt) or QueryRaw(prompt)", name, len(contextStr), maxChars)
}

func (r *REPL) recordLLMCall(prompt, response string, duration float64, result QueryResponse) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.llmCalls = append(r.llmCalls, LLMCall{
		Prompt:           prompt,
		Response:         response,
		Duration:         duration,
		PromptTokens:     result.PromptTokens,
		CompletionTokens: result.CompletionTokens,
	})
}

func (r *REPL) recordLLMCallAsync(prompt, response string, duration float64, result QueryResponse) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.llmCalls = append(r.llmCalls, LLMCall{
		Prompt:           prompt,
		Response:         response,
		Duration:         duration,
		PromptTokens:     result.PromptTokens,
		CompletionTokens: result.CompletionTokens,
		Async:            true,
	})
}

func (r *REPL) queryBatchError(prompts []string, err error, duration float64) []string {
	errResults := make([]string, len(prompts))
	for i, p := range prompts {
		errResults[i] = fmt.Sprintf("Error: %v", err)
		r.recordLLMCall(p, errResults[i], durationPerPrompt(duration, len(prompts)), QueryResponse{})
	}
	return errResults
}

func durationPerPrompt(duration float64, promptCount int) float64 {
	if promptCount <= 0 {
		return duration
	}
	return duration / float64(promptCount)
}

// llmQueryRaw makes a single LLM query without prepending the loaded context.
func (r *REPL) llmQueryRaw(prompt string) string {
	start := time.Now()

	result, err := r.llmClient.Query(r.ctx, prompt)
	duration := time.Since(start).Seconds()

	response := result.Response
	if err != nil {
		response = fmt.Sprintf("Error: %v", err)
	}
	r.recordLLMCall(prompt, response, duration, result)
	return response
}

// llmQueryWith makes a single LLM query with only the provided context slice.
func (r *REPL) llmQueryWith(contextSlice, prompt string) string {
	start := time.Now()
	fullPrompt := r.buildPromptWithProvidedContext(contextSlice, prompt)

	result, err := r.llmClient.Query(r.ctx, fullPrompt)
	duration := time.Since(start).Seconds()

	response := result.Response
	if err != nil {
		response = fmt.Sprintf("Error: %v", err)
	}
	r.recordLLMCall(prompt, response, duration, result)
	return response
}

// llmQueryBatched makes concurrent LLM queries. This is called from interpreted code.
// It automatically includes the context variable (if loaded) in each prompt.
func (r *REPL) llmQueryBatched(prompts []string) []string {
	start := time.Now()

	if err := r.fullContextQueryBlocked("QueryBatched"); err != nil {
		return r.queryBatchError(prompts, err, time.Since(start).Seconds())
	}

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
			r.recordLLMCall(p, errResults[i], durationPerPrompt(duration, len(prompts)), QueryResponse{})
		}
		return errResults
	}

	// Record each successful call with token usage (store original prompts for clarity)
	responses := make([]string, len(results))
	for i, p := range prompts {
		responses[i] = results[i].Response
		r.recordLLMCall(p, results[i].Response, durationPerPrompt(duration, len(prompts)), results[i])
	}
	return responses
}

// llmQueryBatchedRaw makes concurrent LLM queries without prepending loaded context.
func (r *REPL) llmQueryBatchedRaw(prompts []string) []string {
	start := time.Now()

	results, err := r.llmClient.QueryBatched(r.ctx, prompts)
	duration := time.Since(start).Seconds()

	if err != nil {
		return r.queryBatchError(prompts, err, duration)
	}

	responses := make([]string, len(results))
	for i, p := range prompts {
		responses[i] = results[i].Response
		r.recordLLMCall(p, results[i].Response, durationPerPrompt(duration, len(prompts)), results[i])
	}
	return responses
}

// llmQueryAsync starts an async query and returns a handle ID.
// This is called from interpreted code via QueryAsync().
// It automatically includes the context variable (if loaded) in the prompt.
func (r *REPL) llmQueryAsync(prompt string) string {
	handle := newAsyncQueryHandle()

	if err := r.fullContextQueryBlocked("QueryAsync"); err != nil {
		response := fmt.Sprintf("Error: %v", err)
		r.recordLLMCallAsync(prompt, response, 0, QueryResponse{})
		r.asyncMu.Lock()
		r.asyncQueries[handle.id] = handle
		r.asyncMu.Unlock()
		handle.complete(QueryResponse{}, err)
		return handle.id
	}

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
		r.recordLLMCallAsync(prompt, response, duration, result)

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
func (r *REPL) finalAnswer(value any) string {
	finalValue := fmt.Sprint(value)
	r.setFinal(finalValue)
	// Print to stdout so it appears in the output and can be parsed
	fmt.Fprintf(r.stdout, "\nFINAL(%s)\n", finalValue)
	return finalValue
}

// finalVarAnswer handles FINAL_VAR(varName) calls from within code blocks.
// The varName is treated as the value directly (the variable was already evaluated by Go).
// This is called when LLM writes FINAL_VAR(answer) where answer is a Go variable.
func (r *REPL) finalVarAnswer(value any) string {
	finalValue := fmt.Sprint(value)
	r.setFinal(finalValue)
	// The value parameter IS the resolved variable value (Go already evaluated it)
	// Print to stdout so it appears in the output and can be parsed
	fmt.Fprintf(r.stdout, "\nFINAL(%s)\n", finalValue)
	return finalValue
}

func (r *REPL) setFinal(value string) {
	r.finalMu.Lock()
	defer r.finalMu.Unlock()
	if r.finalSet {
		return
	}
	r.finalSet = true
	r.finalValue = value
}

func (r *REPL) clearFinalLocked() {
	r.finalMu.Lock()
	defer r.finalMu.Unlock()
	r.finalSet = false
	r.finalValue = ""
}

// HasFinal reports whether executed code called FINAL or FINAL_VAR.
func (r *REPL) HasFinal() bool {
	r.finalMu.RLock()
	defer r.finalMu.RUnlock()
	return r.finalSet
}

// Final returns the value provided to FINAL or FINAL_VAR.
func (r *REPL) Final() (string, bool) {
	r.finalMu.RLock()
	defer r.finalMu.RUnlock()
	if !r.finalSet {
		return "", false
	}
	return r.finalValue, true
}

// ClearFinal clears any previous FINAL/FINAL_VAR signal.
func (r *REPL) ClearFinal() {
	r.clearFinalLocked()
}

// QueryAsync starts an async query and returns a handle.
// This is the Go API for async queries.
// It automatically includes the context variable (if loaded) in the prompt.
func (r *REPL) QueryAsync(prompt string) *AsyncQueryHandle {
	handle := newAsyncQueryHandle()

	if err := r.fullContextQueryBlocked("QueryAsync"); err != nil {
		response := fmt.Sprintf("Error: %v", err)
		r.recordLLMCallAsync(prompt, response, 0, QueryResponse{})
		r.asyncMu.Lock()
		r.asyncQueries[handle.id] = handle
		r.asyncMu.Unlock()
		handle.complete(QueryResponse{}, err)
		return handle
	}

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
		r.recordLLMCallAsync(prompt, response, duration, result)

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
	if payload == nil {
		_, err := r.interp.Eval(`var context = ""`)
		return err
	}

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
		contextStr := string(jsonBytes)
		_, err = r.interp.Eval(`var context = ` + strconv.Quote(contextStr))
		return err
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
	r.clearFinalLocked()

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

	for _, name := range core.CommonVarNames {
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
		idx := contextindex.New(ctx)
		chunks := idx.ChunkCount()
		lines := idx.LineCount()
		return fmt.Sprintf("type=string, len=%d, lines=%d, chunks=%d", len(ctx), lines, chunks)
	default:
		return fmt.Sprintf("type=%T", ctx)
	}
}

func (r *REPL) findRelevant(query string, topK int) []string {
	contextStr, ok := r.contextString()
	if !ok {
		return []string{}
	}
	return contextindex.FindRelevant(contextStr, query, topK)
}

func (r *REPL) getChunk(id int) string {
	contextStr, ok := r.contextString()
	if !ok {
		return ""
	}
	return contextindex.GetChunk(contextStr, id)
}

func (r *REPL) getContextRange(startLine, endLine int) string {
	contextStr, ok := r.contextString()
	if !ok {
		return ""
	}
	return contextindex.GetContext(contextStr, startLine, endLine)
}

func (r *REPL) chunkCount() int {
	contextStr, ok := r.contextString()
	if !ok {
		return 0
	}
	return contextindex.ChunkCount(contextStr)
}

func (r *REPL) lineCount() int {
	contextStr, ok := r.contextString()
	if !ok {
		return 0
	}
	return contextindex.LineCount(contextStr)
}

// FormatExecutionResult formats an execution result for display to the LLM.
// This is a convenience wrapper around core.FormatExecutionResult.
func FormatExecutionResult(result *core.ExecutionResult) string {
	return core.FormatExecutionResult(result)
}
