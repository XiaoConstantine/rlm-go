package rlm

import (
	"context"
	"fmt"
	"time"

	"github.com/XiaoConstantine/rlm-go/pkg/core"
	"github.com/XiaoConstantine/rlm-go/pkg/parsing"
	"github.com/XiaoConstantine/rlm-go/pkg/repl"
)

// LLMClient defines the interface for the root LLM.
type LLMClient interface {
	// Complete generates a completion for the given messages.
	Complete(ctx context.Context, messages []core.Message) (string, error)
}

// Config holds RLM configuration.
type Config struct {
	// MaxIterations is the maximum number of iteration loops (default: 30).
	MaxIterations int

	// SystemPrompt overrides the default system prompt.
	SystemPrompt string

	// Verbose enables verbose logging.
	Verbose bool
}

// DefaultConfig returns the default RLM configuration.
func DefaultConfig() Config {
	return Config{
		MaxIterations: 30,
		SystemPrompt:  SystemPrompt,
	}
}

// RLM is the main Recursive Language Model implementation.
type RLM struct {
	client    LLMClient
	replClient repl.LLMClient
	config    Config
}

// New creates a new RLM instance.
// client is used for the root LLM orchestration.
// replClient is used for sub-LLM calls from within the REPL.
func New(client LLMClient, replClient repl.LLMClient, opts ...Option) *RLM {
	cfg := DefaultConfig()
	for _, opt := range opts {
		opt(&cfg)
	}

	return &RLM{
		client:     client,
		replClient: replClient,
		config:     cfg,
	}
}

// Option configures the RLM.
type Option func(*Config)

// WithMaxIterations sets the maximum number of iterations.
func WithMaxIterations(n int) Option {
	return func(c *Config) {
		c.MaxIterations = n
	}
}

// WithSystemPrompt sets a custom system prompt.
func WithSystemPrompt(prompt string) Option {
	return func(c *Config) {
		c.SystemPrompt = prompt
	}
}

// WithVerbose enables verbose logging.
func WithVerbose(v bool) Option {
	return func(c *Config) {
		c.Verbose = v
	}
}

// Complete runs an RLM completion.
// contextPayload is the context data (string, map, or slice).
// query is the user's question.
func (r *RLM) Complete(ctx context.Context, contextPayload any, query string) (*core.CompletionResult, error) {
	start := time.Now()

	// Create REPL environment
	replEnv := repl.New(r.replClient)

	// Load context into REPL
	if err := replEnv.LoadContext(contextPayload); err != nil {
		return nil, fmt.Errorf("failed to load context: %w", err)
	}

	// Build initial message history
	messages := r.buildInitialMessages(replEnv, query)

	// Iteration loop
	for i := 0; i < r.config.MaxIterations; i++ {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		if r.config.Verbose {
			fmt.Printf("[RLM] Iteration %d/%d\n", i+1, r.config.MaxIterations)
		}

		// Add iteration-specific user prompt
		currentMessages := r.appendIterationPrompt(messages, i)

		// Get LLM response
		response, err := r.client.Complete(ctx, currentMessages)
		if err != nil {
			return nil, fmt.Errorf("iteration %d: llm completion failed: %w", i, err)
		}

		if r.config.Verbose {
			fmt.Printf("[RLM] Response: %s\n", truncate(response, 200))
		}

		// Extract and execute code blocks
		codeBlocks := parsing.FindCodeBlocks(response)
		var execResults []core.CodeBlock

		for _, code := range codeBlocks {
			if r.config.Verbose {
				fmt.Printf("[RLM] Executing code:\n%s\n", truncate(code, 200))
			}

			result, _ := replEnv.Execute(ctx, code)
			execResults = append(execResults, core.CodeBlock{
				Code:   code,
				Result: *result,
			})

			if r.config.Verbose && result.Stdout != "" {
				fmt.Printf("[RLM] Output: %s\n", truncate(result.Stdout, 200))
			}
			if r.config.Verbose && result.Stderr != "" {
				fmt.Printf("[RLM] Stderr: %s\n", truncate(result.Stderr, 200))
			}
		}

		// Check for final answer
		if final := parsing.FindFinalAnswer(response); final != nil {
			answer := final.Content

			// Resolve variable if FINAL_VAR
			if final.Type == core.FinalTypeVariable {
				varValue, err := replEnv.GetVariable(final.Content)
				if err != nil {
					// Fall back to content if variable not found
					if r.config.Verbose {
						fmt.Printf("[RLM] Warning: could not resolve variable %q: %v\n", final.Content, err)
					}
				} else {
					answer = varValue
				}
			}

			return &core.CompletionResult{
				Response:   answer,
				Iterations: i + 1,
				Duration:   time.Since(start),
			}, nil
		}

		// Append iteration results to history
		messages = r.appendIterationToHistory(messages, response, execResults)
	}

	// Max iterations exhausted - force final answer
	return r.forceDefaultAnswer(ctx, messages, start)
}

// buildInitialMessages creates the initial message history.
func (r *RLM) buildInitialMessages(replEnv *repl.REPL, query string) []core.Message {
	contextInfo := replEnv.ContextInfo()
	userPrompt := fmt.Sprintf(UserPromptTemplate, contextInfo, query)

	return []core.Message{
		{Role: "system", Content: r.config.SystemPrompt},
		{Role: "user", Content: userPrompt},
	}
}

// appendIterationPrompt adds the appropriate user prompt for the current iteration.
func (r *RLM) appendIterationPrompt(messages []core.Message, iteration int) []core.Message {
	if iteration == 0 {
		// First iteration - messages already include the initial user prompt
		return messages
	}

	// Subsequent iterations - add continuation prompt
	return append(messages, core.Message{
		Role:    "user",
		Content: IterationPromptTemplate,
	})
}

// appendIterationToHistory adds the LLM response and execution results to message history.
func (r *RLM) appendIterationToHistory(messages []core.Message, response string, blocks []core.CodeBlock) []core.Message {
	// Add assistant response
	messages = append(messages, core.Message{
		Role:    "assistant",
		Content: response,
	})

	// Add execution results as user messages
	for _, block := range blocks {
		content := fmt.Sprintf(
			"Code executed:\n```go\n%s\n```\n\nREPL output:\n%s",
			block.Code,
			repl.FormatExecutionResult(&block.Result),
		)
		messages = append(messages, core.Message{
			Role:    "user",
			Content: truncateString(content, 20000),
		})
	}

	return messages
}

// forceDefaultAnswer forces the LLM to provide a final answer.
func (r *RLM) forceDefaultAnswer(ctx context.Context, messages []core.Message, start time.Time) (*core.CompletionResult, error) {
	messages = append(messages, core.Message{
		Role:    "user",
		Content: DefaultAnswerPrompt,
	})

	response, err := r.client.Complete(ctx, messages)
	if err != nil {
		return nil, fmt.Errorf("default answer: llm completion failed: %w", err)
	}

	// Try to extract FINAL from response
	answer := response
	if final := parsing.FindFinalAnswer(response); final != nil {
		answer = final.Content
	}

	return &core.CompletionResult{
		Response:   answer,
		Iterations: r.config.MaxIterations,
		Duration:   time.Since(start),
	}, nil
}

// truncate shortens a string for logging.
func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}

// truncateString shortens a string with a suffix indicator.
func truncateString(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "\n... (truncated)"
}

// ContextMetadata returns a string describing the context.
func ContextMetadata(payload any) string {
	switch v := payload.(type) {
	case string:
		return fmt.Sprintf("string, %d chars", len(v))
	case []any:
		return fmt.Sprintf("array, %d items", len(v))
	case map[string]any:
		return fmt.Sprintf("object, %d keys", len(v))
	default:
		return fmt.Sprintf("%T", v)
	}
}
