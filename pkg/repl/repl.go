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

// LLMClient defines the interface for making LLM calls from within the REPL.
type LLMClient interface {
	Query(ctx context.Context, prompt string) (string, error)
	QueryBatched(ctx context.Context, prompts []string) ([]string, error)
}

// REPL represents a Yaegi-based Go interpreter with RLM capabilities.
type REPL struct {
	interp    *interp.Interpreter
	stdout    *bytes.Buffer
	stderr    *bytes.Buffer
	llmClient LLMClient
	ctx       context.Context
	mu        sync.Mutex
}

// New creates a new REPL instance.
func New(client LLMClient) *REPL {
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
		interp:    i,
		stdout:    stdout,
		stderr:    stderr,
		llmClient: client,
		ctx:       context.Background(),
	}

	// Inject RLM functions
	if err := r.injectBuiltins(); err != nil {
		panic(fmt.Sprintf("failed to inject builtins: %v", err))
	}

	return r
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
	result, err := r.llmClient.Query(r.ctx, prompt)
	if err != nil {
		return fmt.Sprintf("Error: %v", err)
	}
	return result
}

// llmQueryBatched makes concurrent LLM queries. This is called from interpreted code.
func (r *REPL) llmQueryBatched(prompts []string) []string {
	results, err := r.llmClient.QueryBatched(r.ctx, prompts)
	if err != nil {
		errResults := make([]string, len(prompts))
		for i := range errResults {
			errResults[i] = fmt.Sprintf("Error: %v", err)
		}
		return errResults
	}
	return results
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
	defer r.mu.Unlock()

	// Set context for LLM calls
	r.ctx = ctx

	// Reset buffers
	r.stdout.Reset()
	r.stderr.Reset()

	start := time.Now()

	// Execute the code
	_, err := r.interp.Eval(code)

	result := &core.ExecutionResult{
		Stdout:   r.stdout.String(),
		Stderr:   r.stderr.String(),
		Duration: time.Since(start),
	}

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
