// Package core provides core types for rlm-go.
package core

import "time"

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
