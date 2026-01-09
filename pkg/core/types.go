// Package core provides core types for rlm-go.
package core

import (
	"fmt"
	"strings"
	"time"
)

// ExecutionResult represents the result of executing code in the REPL.
type ExecutionResult struct {
	Stdout   string
	Stderr   string
	Duration time.Duration
}

// CodeBlock represents an extracted and executed code block.
type CodeBlock struct {
	Code   string
	Result ExecutionResult
}

// FinalAnswerType indicates whether the answer is direct or a variable reference.
type FinalAnswerType string

const (
	FinalTypeDirect   FinalAnswerType = "FINAL"
	FinalTypeVariable FinalAnswerType = "FINAL_VAR"
)

// FinalAnswer represents a detected FINAL or FINAL_VAR signal.
type FinalAnswer struct {
	Type    FinalAnswerType
	Content string
}

// Iteration represents a single iteration of the RLM loop.
type Iteration struct {
	Response    string
	CodeBlocks  []CodeBlock
	FinalAnswer *FinalAnswer
	Duration    time.Duration
}

// Message represents a chat message.
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// UsageStats tracks token usage.
type UsageStats struct {
	PromptTokens          int
	CompletionTokens      int
	TotalTokens           int
	CacheCreationTokens   int
	CacheReadTokens       int
}

// LLMResponse represents the response from an LLM call with usage metadata.
type LLMResponse struct {
	Content               string
	PromptTokens          int
	CompletionTokens      int
	CacheCreationTokens   int
	CacheReadTokens       int
}

// CompletionResult represents the final result of an RLM completion.
type CompletionResult struct {
	Response   string
	Iterations int
	Duration   time.Duration
	Usage      UsageStats
}

// RecursionContext tracks the current recursion state for multi-depth RLM.
type RecursionContext struct {
	// CurrentDepth is the current recursion depth (0 = root).
	CurrentDepth int

	// MaxDepth is the maximum allowed recursion depth.
	MaxDepth int

	// ParentID identifies the parent RLM call for tracing/logging.
	ParentID string

	// TraceID is the root trace identifier shared across all recursive calls.
	TraceID string
}

// NewRecursionContext creates a new RecursionContext with the given max depth.
func NewRecursionContext(maxDepth int) *RecursionContext {
	return &RecursionContext{
		CurrentDepth: 0,
		MaxDepth:     maxDepth,
		ParentID:     "",
		TraceID:      "",
	}
}

// Child creates a child RecursionContext with incremented depth.
func (rc *RecursionContext) Child(parentID string) *RecursionContext {
	return &RecursionContext{
		CurrentDepth: rc.CurrentDepth + 1,
		MaxDepth:     rc.MaxDepth,
		ParentID:     parentID,
		TraceID:      rc.TraceID,
	}
}

// CanRecurse returns true if another level of recursion is allowed.
func (rc *RecursionContext) CanRecurse() bool {
	return rc.CurrentDepth < rc.MaxDepth
}

// DepthExceededError is returned when MaxDepth is exceeded.
type DepthExceededError struct {
	CurrentDepth int
	MaxDepth     int
	Prompt       string
}

// Error implements the error interface.
func (e *DepthExceededError) Error() string {
	return fmt.Sprintf("recursion depth exceeded: current=%d, max=%d, prompt=%q",
		e.CurrentDepth, e.MaxDepth, Truncate(e.Prompt, 50))
}

// Truncate shortens a string to maxLen characters, adding "..." suffix if truncated.
// This is a utility function used for logging and error messages.
func Truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}

// QueryResponse contains the LLM response with usage metadata.
// This is used for sub-LLM calls from REPL/sandbox execution.
type QueryResponse struct {
	Response         string
	PromptTokens     int
	CompletionTokens int
}

// LLMCall represents a sub-LLM call made during code execution.
type LLMCall struct {
	Prompt           string  `json:"prompt"`
	Response         string  `json:"response"`
	Duration         float64 `json:"duration"`
	PromptTokens     int     `json:"prompt_tokens"`
	CompletionTokens int     `json:"completion_tokens"`
	Async            bool    `json:"async,omitempty"`
}

// CommonVarNames is the list of commonly used variable names in RLM code.
// Used by GetLocals implementations to extract user-defined variables.
var CommonVarNames = []string{
	"context", "result", "answer", "data", "output", "response",
	"analysis", "summary", "final_answer", "count", "total",
	"items", "records", "values", "results", "findings",
}

// FormatExecutionResult formats an execution result for display to the LLM.
func FormatExecutionResult(result *ExecutionResult) string {
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
