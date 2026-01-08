package sandbox

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

// LocalExecutor provides in-process code execution using Yaegi.
// This is the fastest option but has the least isolation.
type LocalExecutor struct {
	interp    *interp.Interpreter
	stdout    *bytes.Buffer
	stderr    *bytes.Buffer
	client    LLMClient
	ctx       context.Context
	mu        sync.Mutex
	llmCalls  []LLMCall
	config    Config
}

// NewLocalExecutor creates a new local executor using Yaegi.
func NewLocalExecutor(client LLMClient, cfg Config) (*LocalExecutor, error) {
	stdout := new(bytes.Buffer)
	stderr := new(bytes.Buffer)

	i := interp.New(interp.Options{
		Stdout: stdout,
		Stderr: stderr,
	})

	// Load standard library
	if err := i.Use(stdlib.Symbols); err != nil {
		return nil, fmt.Errorf("failed to load stdlib: %w", err)
	}

	exec := &LocalExecutor{
		interp:   i,
		stdout:   stdout,
		stderr:   stderr,
		client:   client,
		ctx:      context.Background(),
		config:   cfg,
	}

	// Inject RLM functions
	if err := exec.injectBuiltins(); err != nil {
		return nil, fmt.Errorf("failed to inject builtins: %w", err)
	}

	return exec, nil
}

// injectBuiltins registers Query and QueryBatched functions in the interpreter.
func (e *LocalExecutor) injectBuiltins() error {
	symbols := interp.Exports{
		"rlm/rlm": {
			"Query":        reflect.ValueOf(e.llmQuery),
			"QueryBatched": reflect.ValueOf(e.llmQueryBatched),
		},
	}

	if err := e.interp.Use(symbols); err != nil {
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
	_, err := e.interp.Eval(setupCode)
	return err
}

// llmQuery makes a single LLM query.
func (e *LocalExecutor) llmQuery(prompt string) string {
	start := time.Now()
	result, err := e.client.Query(e.ctx, prompt)
	duration := time.Since(start).Seconds()

	response := result.Response
	if err != nil {
		response = fmt.Sprintf("Error: %v", err)
	}

	e.mu.Lock()
	e.llmCalls = append(e.llmCalls, LLMCall{
		Prompt:           prompt,
		Response:         response,
		Duration:         duration,
		PromptTokens:     result.PromptTokens,
		CompletionTokens: result.CompletionTokens,
	})
	e.mu.Unlock()

	return response
}

// llmQueryBatched makes concurrent LLM queries.
func (e *LocalExecutor) llmQueryBatched(prompts []string) []string {
	start := time.Now()
	results, err := e.client.QueryBatched(e.ctx, prompts)
	duration := time.Since(start).Seconds()

	if err != nil {
		errResults := make([]string, len(prompts))
		for i := range errResults {
			errResults[i] = fmt.Sprintf("Error: %v", err)
		}
		e.mu.Lock()
		for i, p := range prompts {
			e.llmCalls = append(e.llmCalls, LLMCall{
				Prompt:   p,
				Response: errResults[i],
				Duration: duration / float64(len(prompts)),
			})
		}
		e.mu.Unlock()
		return errResults
	}

	responses := make([]string, len(results))
	e.mu.Lock()
	for i, p := range prompts {
		responses[i] = results[i].Response
		e.llmCalls = append(e.llmCalls, LLMCall{
			Prompt:           p,
			Response:         results[i].Response,
			Duration:         duration / float64(len(prompts)),
			PromptTokens:     results[i].PromptTokens,
			CompletionTokens: results[i].CompletionTokens,
		})
	}
	e.mu.Unlock()

	return responses
}

// Execute runs Go code and returns the result.
func (e *LocalExecutor) Execute(ctx context.Context, code string) (*core.ExecutionResult, error) {
	e.mu.Lock()
	e.ctx = ctx
	e.stdout.Reset()
	e.stderr.Reset()
	e.mu.Unlock()

	start := time.Now()

	// Create a channel for the result
	done := make(chan struct{})
	var evalErr error

	go func() {
		_, evalErr = e.interp.Eval(code)
		close(done)
	}()

	// Wait for completion or timeout
	timeout := e.config.Timeout
	if timeout == 0 {
		timeout = 60 * time.Second
	}

	select {
	case <-done:
		// Execution completed
	case <-time.After(timeout):
		return &core.ExecutionResult{
			Stderr:   "execution timeout exceeded",
			Duration: time.Since(start),
		}, nil
	case <-ctx.Done():
		return &core.ExecutionResult{
			Stderr:   "execution cancelled",
			Duration: time.Since(start),
		}, ctx.Err()
	}

	e.mu.Lock()
	result := &core.ExecutionResult{
		Stdout:   e.stdout.String(),
		Stderr:   e.stderr.String(),
		Duration: time.Since(start),
	}
	e.mu.Unlock()

	if evalErr != nil {
		if result.Stderr != "" {
			result.Stderr += "\n"
		}
		result.Stderr += evalErr.Error()
	}

	return result, nil
}

// LoadContext injects the context payload into the interpreter.
func (e *LocalExecutor) LoadContext(payload any) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	switch v := payload.(type) {
	case string:
		_, err := e.interp.Eval(`var context = ` + strconv.Quote(v))
		return err

	case map[string]any:
		return e.loadStructuredContext(v, "map[string]interface{}")

	case []any:
		return e.loadStructuredContext(v, "[]interface{}")

	default:
		jsonBytes, err := json.Marshal(v)
		if err != nil {
			return fmt.Errorf("unsupported context type %T: %w", v, err)
		}
		return e.LoadContext(string(jsonBytes))
	}
}

// loadStructuredContext handles map and slice context types.
// It marshals the data to JSON and stores it as a string, which can be parsed by user code.
func (e *LocalExecutor) loadStructuredContext(v any, _ string) error {
	jsonBytes, err := json.Marshal(v)
	if err != nil {
		return fmt.Errorf("marshal context: %w", err)
	}

	// Store as JSON string - user code can parse if needed
	_, err = e.interp.Eval(`var context = ` + strconv.Quote(string(jsonBytes)))
	return err
}

// GetVariable retrieves a variable value from the interpreter.
func (e *LocalExecutor) GetVariable(name string) (string, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	v, err := e.interp.Eval(name)
	if err != nil {
		return "", fmt.Errorf("variable %q not found: %w", name, err)
	}

	if !v.IsValid() {
		return "", fmt.Errorf("variable %q is invalid", name)
	}

	return fmt.Sprintf("%v", v.Interface()), nil
}

// GetLLMCalls returns and clears the recorded LLM calls.
func (e *LocalExecutor) GetLLMCalls() []LLMCall {
	e.mu.Lock()
	defer e.mu.Unlock()
	calls := e.llmCalls
	e.llmCalls = nil
	return calls
}

// GetLocals returns user-defined variables from the interpreter.
func (e *LocalExecutor) GetLocals() map[string]any {
	e.mu.Lock()
	defer e.mu.Unlock()

	locals := make(map[string]any)

	varNames := []string{
		"context", "result", "answer", "data", "output", "response",
		"analysis", "summary", "final_answer", "count", "total",
		"items", "records", "values", "results", "findings",
	}

	for _, name := range varNames {
		v, err := e.interp.Eval(name)
		if err != nil || !v.IsValid() {
			continue
		}
		locals[name] = v.Interface()
	}

	return locals
}

// ContextInfo returns metadata about the loaded context.
func (e *LocalExecutor) ContextInfo() string {
	e.mu.Lock()
	defer e.mu.Unlock()

	v, err := e.interp.Eval("context")
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

// Reset clears the interpreter state.
func (e *LocalExecutor) Reset() error {
	e.mu.Lock()
	defer e.mu.Unlock()

	e.stdout.Reset()
	e.stderr.Reset()
	e.llmCalls = nil

	// Create a fresh interpreter
	i := interp.New(interp.Options{
		Stdout: e.stdout,
		Stderr: e.stderr,
	})

	if err := i.Use(stdlib.Symbols); err != nil {
		return fmt.Errorf("failed to load stdlib: %w", err)
	}

	e.interp = i

	return e.injectBuiltins()
}

// Close releases resources.
func (e *LocalExecutor) Close() error {
	e.mu.Lock()
	defer e.mu.Unlock()

	e.stdout.Reset()
	e.stderr.Reset()
	e.llmCalls = nil

	return nil
}

// Backend returns the backend type.
func (e *LocalExecutor) Backend() Backend {
	return BackendLocal
}

// FormatExecutionResult formats an execution result for display.
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
