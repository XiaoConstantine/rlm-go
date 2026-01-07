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
}

// REPL represents a Yaegi-based Go interpreter with RLM capabilities.
type REPL struct {
	interp     *interp.Interpreter
	stdout     *bytes.Buffer
	stderr     *bytes.Buffer
	llmClient  LLMClient
	ctx        context.Context
	mu         sync.Mutex
	llmCalls   []LLMCall // Track LLM calls made during execution
	fromPool   bool      // Whether the interpreter came from the pool
	usePooling bool      // Whether to use pooling for this REPL
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
		interp:     i,
		stdout:     stdout,
		stderr:     stderr,
		llmClient:  client,
		ctx:        context.Background(),
		fromPool:   false,
		usePooling: false,
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
		interp:     i,
		stdout:     stdout,
		stderr:     stderr,
		llmClient:  client,
		ctx:        context.Background(),
		fromPool:   true,
		usePooling: true,
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
}

// injectBuiltins registers llmQuery and llmQueryBatched functions in the interpreter.
func (r *REPL) injectBuiltins() error {
	symbols := interp.Exports{
		"rlm/rlm": {
			"Query":        reflect.ValueOf(r.llmQuery),
			"QueryBatched": reflect.ValueOf(r.llmQueryBatched),
		},
	}

	if err := r.interp.Use(symbols); err != nil {
		return fmt.Errorf("failed to inject rlm symbols: %w", err)
	}

	// Pre-import common packages and RLM functions so they're available without qualification
	setupCode := `
import "fmt"
import "strings"
import "regexp"
import . "rlm/rlm"
`
	_, err := r.interp.Eval(setupCode)
	return err
}

// llmQuery makes a single LLM query. This is called from interpreted code.
func (r *REPL) llmQuery(prompt string) string {
	start := time.Now()
	result, err := r.llmClient.Query(r.ctx, prompt)
	duration := time.Since(start).Seconds()

	response := result.Response
	if err != nil {
		response = fmt.Sprintf("Error: %v", err)
	}

	// Record the call with token usage
	r.llmCalls = append(r.llmCalls, LLMCall{
		Prompt:           prompt,
		Response:         response,
		Duration:         duration,
		PromptTokens:     result.PromptTokens,
		CompletionTokens: result.CompletionTokens,
	})

	return response
}

// llmQueryBatched makes concurrent LLM queries. This is called from interpreted code.
func (r *REPL) llmQueryBatched(prompts []string) []string {
	start := time.Now()
	results, err := r.llmClient.QueryBatched(r.ctx, prompts)
	duration := time.Since(start).Seconds()

	if err != nil {
		errResults := make([]string, len(prompts))
		for i := range errResults {
			errResults[i] = fmt.Sprintf("Error: %v", err)
		}
		// Record each as a failed call
		for i, p := range prompts {
			r.llmCalls = append(r.llmCalls, LLMCall{
				Prompt:   p,
				Response: errResults[i],
				Duration: duration / float64(len(prompts)),
			})
		}
		return errResults
	}

	// Record each successful call with token usage
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
func (r *REPL) loadStructuredContext(v any, typeDecl string) error {
	jsonBytes, err := json.Marshal(v)
	if err != nil {
		return fmt.Errorf("marshal context: %w", err)
	}

	code := fmt.Sprintf(`
import "encoding/json"
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
func (r *REPL) Execute(ctx context.Context, code string) (*core.ExecutionResult, error) {
	r.mu.Lock()
	// Set context for LLM calls
	r.ctx = ctx

	// Reset buffers
	r.stdout.Reset()
	r.stderr.Reset()
	r.mu.Unlock() // Release lock before executing code (allows LLM calls to proceed)

	start := time.Now()

	// Execute the code (may call llmQuery which doesn't need lock)
	_, err := r.interp.Eval(code)

	r.mu.Lock()
	result := &core.ExecutionResult{
		Stdout:   r.stdout.String(),
		Stderr:   r.stderr.String(),
		Duration: time.Since(start),
	}
	r.mu.Unlock()

	if err != nil {
		// Append error to stderr
		if result.Stderr != "" {
			result.Stderr += "\n"
		}
		result.Stderr += err.Error()
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
