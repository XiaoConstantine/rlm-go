package rlm

import (
	"context"

	"github.com/XiaoConstantine/rlm-go/pkg/core"
	"github.com/XiaoConstantine/rlm-go/pkg/logger"
	"github.com/XiaoConstantine/rlm-go/pkg/repl"
	"github.com/XiaoConstantine/rlm-go/pkg/sandbox"
)

// ExecutionEnvironment defines the interface for code execution environments.
// Both repl.REPL and sandbox.Executor can implement this interface.
type ExecutionEnvironment interface {
	// Execute runs Go code and returns the result.
	Execute(ctx context.Context, code string) (*core.ExecutionResult, error)

	// LoadContext injects the context payload into the execution environment.
	LoadContext(payload any) error

	// GetVariable retrieves a variable value from the execution environment.
	GetVariable(name string) (string, error)

	// GetLLMCalls returns the LLM calls made during execution.
	// Returns a slice that can be converted to logger.RLMCallEntry.
	GetLLMCalls() []LLMCallRecord

	// GetLocals returns user-defined variables.
	GetLocals() map[string]any

	// ContextInfo returns metadata about the loaded context.
	ContextInfo() string

	// Close releases resources.
	Close()
}

// LLMCallRecord represents an LLM call that can be logged.
type LLMCallRecord struct {
	Prompt           string
	Response         string
	Duration         float64
	PromptTokens     int
	CompletionTokens int
	Async            bool
}

// REPLAdapter adapts a repl.REPL to the ExecutionEnvironment interface.
type REPLAdapter struct {
	repl *repl.REPL
}

// NewREPLAdapter creates a new REPLAdapter.
func NewREPLAdapter(r *repl.REPL) *REPLAdapter {
	return &REPLAdapter{repl: r}
}

// Execute runs Go code in the REPL.
func (a *REPLAdapter) Execute(ctx context.Context, code string) (*core.ExecutionResult, error) {
	return a.repl.Execute(ctx, code)
}

// LoadContext injects the context payload.
func (a *REPLAdapter) LoadContext(payload any) error {
	return a.repl.LoadContext(payload)
}

// GetVariable retrieves a variable value.
func (a *REPLAdapter) GetVariable(name string) (string, error) {
	return a.repl.GetVariable(name)
}

// GetLLMCalls returns the LLM calls made during execution.
func (a *REPLAdapter) GetLLMCalls() []LLMCallRecord {
	calls := a.repl.GetLLMCalls()
	records := make([]LLMCallRecord, len(calls))
	for i, c := range calls {
		records[i] = LLMCallRecord{
			Prompt:           c.Prompt,
			Response:         c.Response,
			Duration:         c.Duration,
			PromptTokens:     c.PromptTokens,
			CompletionTokens: c.CompletionTokens,
			Async:            c.Async,
		}
	}
	return records
}

// GetLocals returns user-defined variables.
func (a *REPLAdapter) GetLocals() map[string]any {
	return a.repl.GetLocals()
}

// ContextInfo returns metadata about the loaded context.
func (a *REPLAdapter) ContextInfo() string {
	return a.repl.ContextInfo()
}

// Close releases resources.
func (a *REPLAdapter) Close() {
	a.repl.Close()
}

// ResetIfNeeded resets the interpreter if corruption was detected.
func (a *REPLAdapter) ResetIfNeeded() (bool, error) {
	return a.repl.ResetIfNeeded()
}

// SandboxAdapter adapts a sandbox.Executor to the ExecutionEnvironment interface.
type SandboxAdapter struct {
	executor sandbox.Executor
}

// NewSandboxAdapter creates a new SandboxAdapter.
func NewSandboxAdapter(exec sandbox.Executor) *SandboxAdapter {
	return &SandboxAdapter{executor: exec}
}

// Execute runs Go code in the sandbox.
func (a *SandboxAdapter) Execute(ctx context.Context, code string) (*core.ExecutionResult, error) {
	return a.executor.Execute(ctx, code)
}

// LoadContext injects the context payload.
func (a *SandboxAdapter) LoadContext(payload any) error {
	return a.executor.LoadContext(payload)
}

// GetVariable retrieves a variable value.
func (a *SandboxAdapter) GetVariable(name string) (string, error) {
	return a.executor.GetVariable(name)
}

// GetLLMCalls returns the LLM calls made during execution.
func (a *SandboxAdapter) GetLLMCalls() []LLMCallRecord {
	calls := a.executor.GetLLMCalls()
	records := make([]LLMCallRecord, len(calls))
	for i, c := range calls {
		records[i] = LLMCallRecord{
			Prompt:           c.Prompt,
			Response:         c.Response,
			Duration:         c.Duration,
			PromptTokens:     c.PromptTokens,
			CompletionTokens: c.CompletionTokens,
			Async:            c.Async,
		}
	}
	return records
}

// GetLocals returns user-defined variables.
func (a *SandboxAdapter) GetLocals() map[string]any {
	return a.executor.GetLocals()
}

// ContextInfo returns metadata about the loaded context.
func (a *SandboxAdapter) ContextInfo() string {
	return a.executor.ContextInfo()
}

// Close releases resources.
func (a *SandboxAdapter) Close() {
	_ = a.executor.Close()
}

// sandboxLLMClientAdapter adapts repl.LLMClient to sandbox.LLMClient.
type sandboxLLMClientAdapter struct {
	client repl.LLMClient
}

// Query makes a single LLM query.
func (a *sandboxLLMClientAdapter) Query(ctx context.Context, prompt string) (sandbox.QueryResponse, error) {
	resp, err := a.client.Query(ctx, prompt)
	if err != nil {
		return sandbox.QueryResponse{}, err
	}
	return sandbox.QueryResponse{
		Response:         resp.Response,
		PromptTokens:     resp.PromptTokens,
		CompletionTokens: resp.CompletionTokens,
	}, nil
}

// QueryBatched makes concurrent LLM queries.
func (a *sandboxLLMClientAdapter) QueryBatched(ctx context.Context, prompts []string) ([]sandbox.QueryResponse, error) {
	results, err := a.client.QueryBatched(ctx, prompts)
	if err != nil {
		return nil, err
	}
	responses := make([]sandbox.QueryResponse, len(results))
	for i, r := range results {
		responses[i] = sandbox.QueryResponse{
			Response:         r.Response,
			PromptTokens:     r.PromptTokens,
			CompletionTokens: r.CompletionTokens,
		}
	}
	return responses, nil
}

// createExecutionEnvironment creates the appropriate execution environment based on config.
func (r *RLM) createExecutionEnvironment() (ExecutionEnvironment, error) {
	// Check if sandbox is enabled
	if r.config.Sandbox != nil && r.config.Sandbox.Enabled {
		// Create sandbox executor
		cfg := sandbox.DefaultConfig()
		if r.config.Sandbox.Config != nil {
			cfg = *r.config.Sandbox.Config
		}
		cfg.Verbose = r.config.Verbose

		// Adapt the REPL client for sandbox
		sandboxClient := &sandboxLLMClientAdapter{client: r.replClient}

		exec, err := sandbox.New(sandboxClient, cfg)
		if err != nil {
			return nil, err
		}

		return NewSandboxAdapter(exec), nil
	}

	// Use REPL environment
	var replEnv *repl.REPL
	if r.config.REPLPool != nil {
		replEnv = r.config.REPLPool.Get()
	} else {
		replEnv = repl.New(r.replClient)
	}

	return NewREPLAdapter(replEnv), nil
}

// convertCallsToLoggerEntries converts LLMCallRecords to logger.RLMCallEntry.
func convertCallsToLoggerEntries(calls []LLMCallRecord) []logger.RLMCallEntry {
	entries := make([]logger.RLMCallEntry, len(calls))
	for i, c := range calls {
		entries[i] = logger.RLMCallEntry{
			Prompt:           c.Prompt,
			Response:         c.Response,
			PromptTokens:     c.PromptTokens,
			CompletionTokens: c.CompletionTokens,
			ExecutionTime:    c.Duration,
		}
	}
	return entries
}
