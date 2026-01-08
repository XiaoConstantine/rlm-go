package rlm

import (
	"context"
	"fmt"
	"strings"
	"time"

	"github.com/XiaoConstantine/rlm-go/pkg/core"
	"github.com/XiaoConstantine/rlm-go/pkg/logger"
	"github.com/XiaoConstantine/rlm-go/pkg/parsing"
	"github.com/XiaoConstantine/rlm-go/pkg/repl"
	"github.com/XiaoConstantine/rlm-go/pkg/sandbox"
)

// LLMClient defines the interface for the root LLM.
type LLMClient interface {
	// Complete generates a completion for the given messages.
	// Returns LLMResponse with content and token usage.
	Complete(ctx context.Context, messages []core.Message) (core.LLMResponse, error)
}

// StreamHandler is called for each chunk of streamed content.
type StreamHandler func(chunk string, done bool) error

// StreamingLLMClient extends LLMClient with streaming support.
type StreamingLLMClient interface {
	LLMClient
	// CompleteStream generates a streaming completion.
	// The handler is called for each chunk of content as it arrives.
	CompleteStream(ctx context.Context, messages []core.Message, handler StreamHandler) (core.LLMResponse, error)
}

// Config holds RLM configuration.
type Config struct {
	// MaxIterations is the maximum number of iteration loops (default: 30).
	MaxIterations int

	// SystemPrompt overrides the default system prompt.
	SystemPrompt string

	// Verbose enables verbose logging.
	Verbose bool

	// Logger is the optional JSONL logger.
	Logger *logger.Logger

	// EnableStreaming enables streaming for root LLM calls (default: false).
	// When enabled, uses SSE streaming for lower perceived latency.
	EnableStreaming bool

	// OnStreamChunk is called for each chunk when streaming is enabled.
	// Can be used to display streaming output to users.
	OnStreamChunk StreamHandler

	// REPLPool is an optional pool of REPL instances for reduced startup overhead.
	// When set, REPLs will be acquired from and returned to this pool.
	REPLPool *repl.REPLPool

	// HistoryCompression configures incremental history compression.
	// When enabled, older iterations are summarized to reduce context size.
	HistoryCompression *HistoryCompressionConfig

	// AdaptiveIteration configures adaptive iteration strategy.
	// When enabled, max iterations are dynamically calculated based on context size.
	AdaptiveIteration *AdaptiveIterationConfig

	// OnProgress is called at the start of each iteration with progress info.
	// Can be used to display progress to users or implement custom termination logic.
	OnProgress func(progress IterationProgress)

	// Recursion configures multi-depth recursion behavior.
	// When enabled, sub-LLMs can spawn their own sub-LLMs.
	Recursion *RecursionConfig

	// Sandbox configures isolated execution for code blocks.
	// When enabled, code runs in Podman/Docker containers instead of in-process.
	// This provides better security isolation at the cost of execution speed.
	Sandbox *SandboxConfig
}

// SandboxConfig configures sandboxed code execution.
type SandboxConfig struct {
	// Enabled turns on sandbox execution (default: false).
	Enabled bool

	// Config contains the detailed sandbox configuration.
	// If nil when Enabled is true, DefaultConfig() is used.
	Config *sandbox.Config
}

// HistoryCompressionConfig configures how message history is compressed.
type HistoryCompressionConfig struct {
	// Enabled turns on history compression (default: false).
	Enabled bool

	// VerbatimIterations is the number of recent iterations to keep verbatim.
	// Older iterations will be summarized. Default: 3.
	VerbatimIterations int

	// MaxSummaryTokens is the approximate maximum tokens for summarized history.
	// Default: 500.
	MaxSummaryTokens int
}

// AdaptiveIterationConfig configures adaptive iteration behavior.
type AdaptiveIterationConfig struct {
	// Enabled turns on adaptive iteration (default: false).
	Enabled bool

	// BaseIterations is the base number of iterations before context scaling.
	// Default: 10.
	BaseIterations int

	// MaxIterations caps the total iterations regardless of context size.
	// Default: 50.
	MaxIterations int

	// ContextScaleFactor determines how much context size increases iterations.
	// iterations = BaseIterations + (contextSize / ContextScaleFactor)
	// Default: 100000 (100KB per additional iteration).
	ContextScaleFactor int

	// EnableEarlyTermination allows early exit when model signals confidence.
	// Default: true.
	EnableEarlyTermination bool

	// ConfidenceThreshold is the number of confidence signals needed for early termination.
	// Default: 1.
	ConfidenceThreshold int
}

// RecursionConfig configures multi-depth recursion behavior.
type RecursionConfig struct {
	// MaxDepth is the maximum recursion depth allowed.
	// Depth 0 = no recursion, 1 = one level of sub-RLM, etc.
	// Default: 0 (disabled).
	MaxDepth int

	// OnRecursiveQuery is called when a recursive query is initiated.
	// Can be used for logging/tracing.
	OnRecursiveQuery func(depth int, prompt string)

	// PerDepthMaxIterations optionally sets different max iterations per depth level.
	// If nil or missing an entry, uses the default MaxIterations.
	PerDepthMaxIterations map[int]int
}

// IterationProgress tracks progress and confidence during iteration.
type IterationProgress struct {
	// CurrentIteration is the current iteration number (1-indexed).
	CurrentIteration int

	// MaxIterations is the computed maximum iterations for this request.
	MaxIterations int

	// ConfidenceSignals counts how many times the model has signaled confidence.
	ConfidenceSignals int

	// HasFinalAttempt indicates the model tried to give a final answer.
	HasFinalAttempt bool

	// ContextSize is the size of the input context in bytes.
	ContextSize int
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

// WithLogger sets the JSONL logger.
func WithLogger(l *logger.Logger) Option {
	return func(c *Config) {
		c.Logger = l
	}
}

// WithStreaming enables streaming mode for root LLM calls.
// When enabled, the LLM client must implement StreamingLLMClient.
func WithStreaming(enabled bool) Option {
	return func(c *Config) {
		c.EnableStreaming = enabled
	}
}

// WithStreamHandler sets the handler for streaming chunks.
// Only used when streaming is enabled.
func WithStreamHandler(handler StreamHandler) Option {
	return func(c *Config) {
		c.OnStreamChunk = handler
	}
}

// WithREPLPool sets a REPL pool for reduced REPL startup overhead.
// When set, REPLs will be acquired from and returned to this pool.
func WithREPLPool(pool *repl.REPLPool) Option {
	return func(c *Config) {
		c.REPLPool = pool
	}
}

// WithHistoryCompression enables incremental history compression.
// verbatimIterations is how many recent iterations to keep in full (default: 3).
// maxSummaryTokens is the approximate max tokens for the summary (default: 500).
func WithHistoryCompression(verbatimIterations, maxSummaryTokens int) Option {
	return func(c *Config) {
		if verbatimIterations <= 0 {
			verbatimIterations = 3
		}
		if maxSummaryTokens <= 0 {
			maxSummaryTokens = 500
		}
		c.HistoryCompression = &HistoryCompressionConfig{
			Enabled:            true,
			VerbatimIterations: verbatimIterations,
			MaxSummaryTokens:   maxSummaryTokens,
		}
	}
}

// WithAdaptiveIteration enables adaptive iteration strategy.
// This dynamically adjusts max iterations based on context size and enables
// early termination when the model signals confidence.
func WithAdaptiveIteration() Option {
	return func(c *Config) {
		c.AdaptiveIteration = &AdaptiveIterationConfig{
			Enabled:                true,
			BaseIterations:         10,
			MaxIterations:          50,
			ContextScaleFactor:     100000, // 100KB per additional iteration
			EnableEarlyTermination: true,
			ConfidenceThreshold:    1,
		}
	}
}

// WithAdaptiveIterationConfig enables adaptive iteration with custom configuration.
func WithAdaptiveIterationConfig(cfg AdaptiveIterationConfig) Option {
	return func(c *Config) {
		cfg.Enabled = true
		if cfg.BaseIterations <= 0 {
			cfg.BaseIterations = 10
		}
		if cfg.MaxIterations <= 0 {
			cfg.MaxIterations = 50
		}
		if cfg.ContextScaleFactor <= 0 {
			cfg.ContextScaleFactor = 100000
		}
		if cfg.ConfidenceThreshold <= 0 {
			cfg.ConfidenceThreshold = 1
		}
		c.AdaptiveIteration = &cfg
	}
}

// WithProgressHandler sets a callback for iteration progress updates.
func WithProgressHandler(handler func(IterationProgress)) Option {
	return func(c *Config) {
		c.OnProgress = handler
	}
}

// WithMaxRecursionDepth enables multi-depth recursion with the specified max depth.
// Depth 0 = disabled, 1 = one level of sub-RLM, 2 = sub-RLM can spawn sub-RLM, etc.
func WithMaxRecursionDepth(depth int) Option {
	return func(c *Config) {
		if depth < 0 {
			depth = 0
		}
		c.Recursion = &RecursionConfig{
			MaxDepth: depth,
		}
	}
}

// WithRecursionConfig enables multi-depth recursion with custom configuration.
func WithRecursionConfig(cfg RecursionConfig) Option {
	return func(c *Config) {
		if cfg.MaxDepth < 0 {
			cfg.MaxDepth = 0
		}
		c.Recursion = &cfg
	}
}

// WithRecursionCallback sets a callback for when recursive queries are initiated.
func WithRecursionCallback(callback func(depth int, prompt string)) Option {
	return func(c *Config) {
		if c.Recursion == nil {
			c.Recursion = &RecursionConfig{}
		}
		c.Recursion.OnRecursiveQuery = callback
	}
}

// WithSandbox enables sandboxed code execution using Podman (preferred) or Docker.
// This provides better security isolation at the cost of execution speed.
// Uses default sandbox configuration (auto-detect runtime, 512MB memory, 60s timeout).
func WithSandbox() Option {
	return func(c *Config) {
		cfg := sandbox.DefaultConfig()
		c.Sandbox = &SandboxConfig{
			Enabled: true,
			Config:  &cfg,
		}
	}
}

// WithSandboxConfig enables sandboxed code execution with custom configuration.
// Allows fine-grained control over the sandbox environment.
func WithSandboxConfig(cfg sandbox.Config) Option {
	return func(c *Config) {
		c.Sandbox = &SandboxConfig{
			Enabled: true,
			Config:  &cfg,
		}
	}
}

// WithSandboxBackend enables sandboxed execution with a specific backend.
// Supported backends: sandbox.BackendPodman, sandbox.BackendDocker, sandbox.BackendAuto.
func WithSandboxBackend(backend sandbox.Backend) Option {
	return func(c *Config) {
		cfg := sandbox.DefaultConfig()
		cfg.Backend = backend
		c.Sandbox = &SandboxConfig{
			Enabled: true,
			Config:  &cfg,
		}
	}
}

// confidencePhrases are phrases that indicate the model is confident in its answer.
var confidencePhrases = []string{
	"i'm confident",
	"i am confident",
	"i'm certain",
	"i am certain",
	"the answer is definitely",
	"the final answer is",
	"based on my analysis, the answer is",
	"after thorough analysis",
	"i have found the answer",
	"the definitive answer",
	"i can confirm that",
	"with certainty",
	"conclusively",
}

// detectConfidence checks if the response contains confidence signals.
func detectConfidence(response string) bool {
	lower := strings.ToLower(response)
	for _, phrase := range confidencePhrases {
		if strings.Contains(lower, phrase) {
			return true
		}
	}
	return false
}

// computeMaxIterations calculates the dynamic max iterations based on context size.
func (r *RLM) computeMaxIterations(contextSize int) int {
	if r.config.AdaptiveIteration == nil || !r.config.AdaptiveIteration.Enabled {
		return r.config.MaxIterations
	}

	cfg := r.config.AdaptiveIteration
	additionalIterations := contextSize / cfg.ContextScaleFactor
	computed := cfg.BaseIterations + additionalIterations

	if computed > cfg.MaxIterations {
		return cfg.MaxIterations
	}
	return computed
}

// getContextSize returns the size of the context payload in bytes.
func getContextSize(payload any) int {
	switch v := payload.(type) {
	case string:
		return len(v)
	case []byte:
		return len(v)
	default:
		// For complex types, estimate based on JSON representation
		// This is a rough approximation
		return len(fmt.Sprintf("%v", v))
	}
}

// shouldTerminateEarly checks if we should terminate early based on confidence signals.
// Early termination only happens when:
// 1. Adaptive iteration is enabled with early termination
// 2. Confidence threshold is met
// 3. There are no pending code blocks (the model isn't waiting for execution results)
func (r *RLM) shouldTerminateEarly(confidenceSignals, pendingCodeBlocks int) bool {
	if r.config.AdaptiveIteration == nil || !r.config.AdaptiveIteration.Enabled {
		return false
	}
	if !r.config.AdaptiveIteration.EnableEarlyTermination {
		return false
	}
	if pendingCodeBlocks > 0 {
		return false // Don't terminate while code is pending execution
	}
	return confidenceSignals >= r.config.AdaptiveIteration.ConfidenceThreshold
}

// Complete runs an RLM completion.
// contextPayload is the context data (string, map, or slice).
// query is the user's question.
func (r *RLM) Complete(ctx context.Context, contextPayload any, query string) (*core.CompletionResult, error) {
	start := time.Now()

	// Create execution environment (REPL or sandbox based on config)
	execEnv, err := r.createExecutionEnvironment()
	if err != nil {
		return nil, fmt.Errorf("failed to create execution environment: %w", err)
	}
	defer execEnv.Close()

	// Load context into execution environment
	if err := execEnv.LoadContext(contextPayload); err != nil {
		return nil, fmt.Errorf("failed to load context: %w", err)
	}

	// Build initial message history
	messages := r.buildInitialMessagesFromEnv(execEnv, query)

	// Track total token usage across iterations
	var totalPromptTokens, totalCompletionTokens int
	var totalCacheCreationTokens, totalCacheReadTokens int

	// Compute max iterations (adaptive or fixed)
	contextSize := getContextSize(contextPayload)
	maxIterations := r.computeMaxIterations(contextSize)

	// Track confidence signals for early termination
	var confidenceSignals int

	// Iteration loop
	for i := 0; i < maxIterations; i++ {
		iterStart := time.Now()

		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		// Report progress if handler is set
		if r.config.OnProgress != nil {
			r.config.OnProgress(IterationProgress{
				CurrentIteration:  i + 1,
				MaxIterations:     maxIterations,
				ConfidenceSignals: confidenceSignals,
				HasFinalAttempt:   false,
				ContextSize:       contextSize,
			})
		}

		if r.config.Verbose {
			fmt.Printf("[RLM] Iteration %d/%d\n", i+1, maxIterations)
		}

		// Add iteration-specific user prompt
		currentMessages := r.appendIterationPrompt(messages, i, query)

		// Get LLM response (streaming or non-streaming)
		llmResp, err := r.completeWithOptionalStreaming(ctx, currentMessages)
		if err != nil {
			return nil, fmt.Errorf("iteration %d: llm completion failed: %w", i, err)
		}
		response := llmResp.Content

		// Aggregate root LLM tokens
		totalPromptTokens += llmResp.PromptTokens
		totalCompletionTokens += llmResp.CompletionTokens
		totalCacheCreationTokens += llmResp.CacheCreationTokens
		totalCacheReadTokens += llmResp.CacheReadTokens

		if r.config.Verbose {
			if llmResp.CacheCreationTokens > 0 || llmResp.CacheReadTokens > 0 {
				fmt.Printf("[RLM] Cache stats this iteration: created=%d, read=%d (totals: created=%d, read=%d)\n",
					llmResp.CacheCreationTokens, llmResp.CacheReadTokens,
					totalCacheCreationTokens, totalCacheReadTokens)
			}
			fmt.Printf("[RLM] Response: %s\n", truncate(response, 200))
		}

		// Extract and execute code blocks
		codeBlocks := parsing.FindCodeBlocks(response)
		var execResults []core.CodeBlock
		var interpreterPanic bool

		for _, code := range codeBlocks {
			if r.config.Verbose {
				fmt.Printf("[RLM] Executing code:\n%s\n", truncate(code, 200))
			}

			result, execErr := execEnv.Execute(ctx, code)
			if execErr != nil {
				// Interpreter panic detected - log and attempt recovery
				if r.config.Verbose {
					fmt.Printf("[RLM] Interpreter error: %v\n", execErr)
				}
				interpreterPanic = true
				// Try to reset the interpreter for subsequent code blocks
				if resetter, ok := execEnv.(interface{ ResetIfNeeded() (bool, error) }); ok {
					if reset, resetErr := resetter.ResetIfNeeded(); reset {
						if r.config.Verbose {
							if resetErr != nil {
								fmt.Printf("[RLM] Interpreter reset failed: %v\n", resetErr)
							} else {
								fmt.Printf("[RLM] Interpreter reset successful\n")
							}
						}
					}
				}
			}

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

		// If interpreter panic occurred, we may need to reload context
		if interpreterPanic {
			if r.config.Verbose {
				fmt.Printf("[RLM] Reloading context after interpreter reset\n")
			}
			// Attempt to reload context (may fail if interpreter is still broken)
			_ = execEnv.LoadContext(contextPayload)
		}

		// Get LLM calls made during code execution and aggregate tokens
		llmCalls := execEnv.GetLLMCalls()
		rlmCalls := convertCallsToLoggerEntries(llmCalls)
		for _, call := range llmCalls {
			// Aggregate token usage
			totalPromptTokens += call.PromptTokens
			totalCompletionTokens += call.CompletionTokens
		}

		// Get locals from execution environment for logging
		locals := execEnv.GetLocals()

		// Detect confidence signals for adaptive early termination
		if detectConfidence(response) {
			confidenceSignals++
			if r.config.Verbose {
				fmt.Printf("[RLM] Confidence signal detected (total: %d)\n", confidenceSignals)
			}
		}

		// Check for final answer - BUT only if there were NO code blocks in this response.
		// If there are code blocks, we need to wait for the next iteration so the model
		// can see the execution results before providing a final answer.
		var finalAnswer any // nil, string, or []string{varname, value}
		var resultResponse string
		if len(codeBlocks) == 0 && parsing.FindFinalAnswer(response) != nil {
			// No code blocks and has final answer - process it
			final := parsing.FindFinalAnswer(response)
			varName := final.Content
			varValue := varName // Default to content itself

			// Resolve variable if FINAL_VAR
			if final.Type == core.FinalTypeVariable {
				resolved, err := execEnv.GetVariable(varName)
				if err != nil {
					if r.config.Verbose {
						fmt.Printf("[RLM] Warning: could not resolve variable %q: %v\n", varName, err)
					}
				} else {
					varValue = resolved
				}
				// FINAL_VAR: output as [varname, value] tuple
				finalAnswer = []string{varName, varValue}
			} else {
				// FINAL: output as string directly
				finalAnswer = varValue
			}
			resultResponse = varValue

			// Log iteration before returning
			if r.config.Logger != nil {
				_ = r.config.Logger.LogIteration(i+1, currentMessages, response, execResults, rlmCalls, locals, finalAnswer, time.Since(iterStart))
			}

			return &core.CompletionResult{
				Response:   resultResponse,
				Iterations: i + 1,
				Duration:   time.Since(start),
				Usage: core.UsageStats{
					PromptTokens:          totalPromptTokens,
					CompletionTokens:      totalCompletionTokens,
					TotalTokens:           totalPromptTokens + totalCompletionTokens,
					CacheCreationTokens:   totalCacheCreationTokens,
					CacheReadTokens:       totalCacheReadTokens,
				},
			}, nil
		}

		// Log iteration (no final answer)
		if r.config.Logger != nil {
			_ = r.config.Logger.LogIteration(i+1, currentMessages, response, execResults, rlmCalls, locals, nil, time.Since(iterStart))
		}

		// Check for early termination based on confidence signals
		if r.shouldTerminateEarly(confidenceSignals, len(codeBlocks)) {
			if r.config.Verbose {
				fmt.Printf("[RLM] Early termination triggered (confidence signals: %d)\n", confidenceSignals)
			}
			// Force immediate final answer due to high confidence
			return r.forceDefaultAnswer(ctx, messages, start, totalPromptTokens, totalCompletionTokens, totalCacheCreationTokens, totalCacheReadTokens)
		}

		// Append iteration results to history
		messages = r.appendIterationToHistory(messages, response, execResults)

		// Apply history compression if enabled and we have enough iterations
		if r.config.HistoryCompression != nil && r.config.HistoryCompression.Enabled {
			messages = r.compressHistory(messages, i+1)
		}
	}

	// Max iterations exhausted - force final answer
	return r.forceDefaultAnswer(ctx, messages, start, totalPromptTokens, totalCompletionTokens, totalCacheCreationTokens, totalCacheReadTokens)
}

// buildInitialMessages creates the initial message history.
func (r *RLM) buildInitialMessages(replEnv *repl.REPL, query string) []core.Message {
	contextInfo := replEnv.ContextInfo()
	userPrompt := fmt.Sprintf(UserPromptTemplate, contextInfo, query) + FirstIterationSuffix

	return []core.Message{
		{Role: "system", Content: r.config.SystemPrompt},
		{Role: "user", Content: userPrompt},
	}
}

// buildInitialMessagesFromEnv creates the initial message history using the ExecutionEnvironment interface.
func (r *RLM) buildInitialMessagesFromEnv(execEnv ExecutionEnvironment, query string) []core.Message {
	contextInfo := execEnv.ContextInfo()
	userPrompt := fmt.Sprintf(UserPromptTemplate, contextInfo, query) + FirstIterationSuffix

	return []core.Message{
		{Role: "system", Content: r.config.SystemPrompt},
		{Role: "user", Content: userPrompt},
	}
}

// appendIterationPrompt adds the appropriate user prompt for the current iteration.
func (r *RLM) appendIterationPrompt(messages []core.Message, iteration int, query string) []core.Message {
	if iteration == 0 {
		// First iteration - messages already include the initial user prompt
		return messages
	}

	// Subsequent iterations - add continuation prompt with query reminder
	return append(messages, core.Message{
		Role:    "user",
		Content: fmt.Sprintf(IterationPromptTemplate, query),
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
			sandbox.FormatExecutionResult(&block.Result),
		)
		messages = append(messages, core.Message{
			Role:    "user",
			Content: truncateString(content, 20000),
		})
	}

	return messages
}

// compressHistory compresses older iterations in the message history.
// It keeps the system message and initial user prompt, plus the most recent N iterations verbatim.
// Older iterations are summarized into a single message.
func (r *RLM) compressHistory(messages []core.Message, currentIteration int) []core.Message {
	cfg := r.config.HistoryCompression
	if cfg == nil || !cfg.Enabled {
		return messages
	}

	// Calculate how many messages constitute one iteration:
	// - 1 assistant message (LLM response)
	// - 1+ user messages (execution results, iteration prompts)
	// For simplicity, we estimate each iteration adds ~2-3 messages on average.

	// We keep:
	// - messages[0]: system prompt
	// - messages[1]: initial user prompt with context
	// - Last N iterations of messages

	if len(messages) <= 2 {
		return messages // Nothing to compress
	}

	// Estimate messages per iteration (assistant + user messages for code results)
	// This is approximate - each iteration has at least 2 messages
	messagesPerIteration := 2

	// Calculate how many messages to keep verbatim at the end
	verbatimMessageCount := cfg.VerbatimIterations * messagesPerIteration
	if verbatimMessageCount <= 0 {
		verbatimMessageCount = 6 // Default: keep last 3 iterations (2 msgs each)
	}

	// If we don't have enough messages to compress, return as-is
	totalIterationMessages := len(messages) - 2 // Subtract system + initial user
	if totalIterationMessages <= verbatimMessageCount {
		return messages
	}

	// Calculate split point
	splitIdx := len(messages) - verbatimMessageCount
	if splitIdx <= 2 {
		return messages // Not enough to compress
	}

	// Build compressed history
	result := make([]core.Message, 0, 3+verbatimMessageCount)

	// Keep system and initial user prompts
	result = append(result, messages[0], messages[1])

	// Summarize messages from index 2 to splitIdx
	toCompress := messages[2:splitIdx]
	if len(toCompress) > 0 {
		summary := r.summarizeIterations(toCompress, cfg.MaxSummaryTokens)
		result = append(result, core.Message{
			Role:    "user",
			Content: summary,
		})
	}

	// Append verbatim recent messages
	result = append(result, messages[splitIdx:]...)

	if r.config.Verbose {
		fmt.Printf("[RLM] Compressed history: %d -> %d messages (summarized %d iteration messages)\n",
			len(messages), len(result), len(toCompress))
	}

	return result
}

// summarizeIterations creates a concise summary of older iteration messages.
func (r *RLM) summarizeIterations(messages []core.Message, maxTokens int) string {
	var summary strings.Builder
	summary.WriteString("[Previous iterations summary]\n")

	// Extract key information from each iteration
	iterCount := 0
	for i := 0; i < len(messages); i++ {
		msg := messages[i]

		if msg.Role == "assistant" {
			iterCount++

			// Extract code block presence and any FINAL mentions
			hasCode := strings.Contains(msg.Content, "```go")
			hasFinal := strings.Contains(msg.Content, "FINAL")

			summary.WriteString(fmt.Sprintf("- Iteration %d: ", iterCount))
			if hasCode {
				summary.WriteString("executed code")
			}
			if hasFinal {
				summary.WriteString(" (mentioned FINAL)")
			}
			summary.WriteString("\n")
		} else if msg.Role == "user" && strings.Contains(msg.Content, "REPL output:") {
			// Summarize execution results
			if strings.Contains(msg.Content, "Error:") || strings.Contains(msg.Content, "error") {
				summary.WriteString("  -> execution had errors\n")
			} else if strings.Contains(msg.Content, "No output") {
				summary.WriteString("  -> no output\n")
			} else {
				// Extract first line of output
				lines := strings.Split(msg.Content, "\n")
				for _, line := range lines {
					if strings.HasPrefix(line, "REPL output:") {
						continue
					}
					if strings.TrimSpace(line) != "" && !strings.HasPrefix(line, "Code executed:") && !strings.HasPrefix(line, "```") {
						outputPreview := line
						if len(outputPreview) > 80 {
							outputPreview = outputPreview[:80] + "..."
						}
						summary.WriteString(fmt.Sprintf("  -> output: %s\n", outputPreview))
						break
					}
				}
			}
		}
	}

	result := summary.String()

	// Rough token estimation: ~4 chars per token
	maxChars := maxTokens * 4
	if len(result) > maxChars {
		result = result[:maxChars] + "\n[...truncated]"
	}

	return result
}

// forceDefaultAnswer forces the LLM to provide a final answer.
func (r *RLM) forceDefaultAnswer(ctx context.Context, messages []core.Message, start time.Time, promptTokens, completionTokens, cacheCreationTokens, cacheReadTokens int) (*core.CompletionResult, error) {
	messages = append(messages, core.Message{
		Role:    "user",
		Content: DefaultAnswerPrompt,
	})

	llmResp, err := r.client.Complete(ctx, messages)
	if err != nil {
		return nil, fmt.Errorf("default answer: llm completion failed: %w", err)
	}

	// Add tokens from this final call
	promptTokens += llmResp.PromptTokens
	completionTokens += llmResp.CompletionTokens
	cacheCreationTokens += llmResp.CacheCreationTokens
	cacheReadTokens += llmResp.CacheReadTokens

	// Try to extract FINAL from response
	answer := llmResp.Content
	if final := parsing.FindFinalAnswer(llmResp.Content); final != nil {
		answer = final.Content
	}

	return &core.CompletionResult{
		Response:   answer,
		Iterations: r.config.MaxIterations,
		Duration:   time.Since(start),
		Usage: core.UsageStats{
			PromptTokens:          promptTokens,
			CompletionTokens:      completionTokens,
			TotalTokens:           promptTokens + completionTokens,
			CacheCreationTokens:   cacheCreationTokens,
			CacheReadTokens:       cacheReadTokens,
		},
	}, nil
}

// completeWithOptionalStreaming calls the LLM with streaming if enabled and supported.
func (r *RLM) completeWithOptionalStreaming(ctx context.Context, messages []core.Message) (core.LLMResponse, error) {
	// Check if streaming is enabled and the client supports it
	if r.config.EnableStreaming {
		if streamClient, ok := r.client.(StreamingLLMClient); ok {
			return streamClient.CompleteStream(ctx, messages, r.config.OnStreamChunk)
		}
		// Fall back to non-streaming if client doesn't support it
		if r.config.Verbose {
			fmt.Println("[RLM] Streaming requested but client doesn't support it, falling back to non-streaming")
		}
	}
	return r.client.Complete(ctx, messages)
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

// CompleteWithRecursion runs an RLM completion with multi-depth recursion support.
// This method is called internally for nested RLM executions.
func (r *RLM) CompleteWithRecursion(
	ctx context.Context,
	contextPayload any,
	query string,
	recursionCtx *core.RecursionContext,
	tokenStats *RecursiveTokenStats,
) (*core.CompletionResult, error) {
	start := time.Now()

	// Select system prompt based on recursion capability
	systemPrompt := r.config.SystemPrompt
	if recursionCtx != nil && recursionCtx.MaxDepth > 0 && recursionCtx.CanRecurse() {
		systemPrompt = RecursiveSystemPrompt
	}

	// Create recursive client adapter for nested calls
	var replEnv interface {
		LoadContext(any) error
		Execute(context.Context, string) (*core.ExecutionResult, error)
		GetVariable(string) (string, error)
		GetLLMCalls() []repl.LLMCall
		GetLocals() map[string]any
		ContextInfo() string
		Close()
	}

	if recursionCtx != nil && recursionCtx.MaxDepth > 0 {
		// Create recursive adapter
		adapter := NewRecursiveClientAdapter(r, recursionCtx, tokenStats, contextPayload)
		replEnv = repl.NewRecursiveREPL(adapter, recursionCtx)
	} else {
		// Standard REPL
		if r.config.REPLPool != nil {
			replEnv = r.config.REPLPool.Get()
		} else {
			replEnv = repl.New(r.replClient)
		}
	}

	defer replEnv.Close()

	// Load context into REPL
	if err := replEnv.LoadContext(contextPayload); err != nil {
		return nil, fmt.Errorf("failed to load context: %w", err)
	}

	// Build initial message history with appropriate system prompt
	contextInfo := replEnv.ContextInfo()
	userPrompt := fmt.Sprintf(UserPromptTemplate, contextInfo, query) + FirstIterationSuffix

	messages := []core.Message{
		{Role: "system", Content: systemPrompt},
		{Role: "user", Content: userPrompt},
	}

	// Track total token usage across iterations
	var totalPromptTokens, totalCompletionTokens int
	var totalCacheCreationTokens, totalCacheReadTokens int

	// Compute max iterations (may differ by depth)
	contextSize := getContextSize(contextPayload)
	maxIterations := r.computeMaxIterationsForDepth(contextSize, recursionCtx)

	// Track confidence signals for early termination
	var confidenceSignals int

	// Iteration loop
	for i := 0; i < maxIterations; i++ {
		iterStart := time.Now()

		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		default:
		}

		// Report progress if handler is set
		if r.config.OnProgress != nil {
			depth := 0
			if recursionCtx != nil {
				depth = recursionCtx.CurrentDepth
			}
			r.config.OnProgress(IterationProgress{
				CurrentIteration:  i + 1,
				MaxIterations:     maxIterations,
				ConfidenceSignals: confidenceSignals,
				HasFinalAttempt:   false,
				ContextSize:       contextSize,
			})
			_ = depth // Used for potential future logging
		}

		if r.config.Verbose {
			depth := 0
			if recursionCtx != nil {
				depth = recursionCtx.CurrentDepth
			}
			fmt.Printf("[RLM] Depth %d, Iteration %d/%d\n", depth, i+1, maxIterations)
		}

		// Add iteration-specific user prompt
		currentMessages := r.appendIterationPrompt(messages, i, query)

		// Get LLM response (streaming or non-streaming)
		llmResp, err := r.completeWithOptionalStreaming(ctx, currentMessages)
		if err != nil {
			return nil, fmt.Errorf("iteration %d: llm completion failed: %w", i, err)
		}
		response := llmResp.Content

		// Aggregate root LLM tokens
		totalPromptTokens += llmResp.PromptTokens
		totalCompletionTokens += llmResp.CompletionTokens
		totalCacheCreationTokens += llmResp.CacheCreationTokens
		totalCacheReadTokens += llmResp.CacheReadTokens

		// Track in token stats if provided
		if tokenStats != nil && recursionCtx != nil {
			tokenStats.Add(recursionCtx.CurrentDepth, llmResp.PromptTokens, llmResp.CompletionTokens)
		}

		if r.config.Verbose {
			if llmResp.CacheCreationTokens > 0 || llmResp.CacheReadTokens > 0 {
				fmt.Printf("[RLM] Cache stats this iteration: created=%d, read=%d\n",
					llmResp.CacheCreationTokens, llmResp.CacheReadTokens)
			}
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

		// Get RLM calls made during code execution and aggregate tokens
		var rlmCalls []logger.RLMCallEntry
		for _, call := range replEnv.GetLLMCalls() {
			rlmCalls = append(rlmCalls, logger.RLMCallEntry{
				Prompt:           call.Prompt,
				Response:         call.Response,
				PromptTokens:     call.PromptTokens,
				CompletionTokens: call.CompletionTokens,
				ExecutionTime:    call.Duration,
			})
			totalPromptTokens += call.PromptTokens
			totalCompletionTokens += call.CompletionTokens
		}

		// Log iteration
		if r.config.Logger != nil {
			locals := replEnv.GetLocals()
			_ = r.config.Logger.LogIteration(i+1, currentMessages, response, execResults, rlmCalls, locals, nil, time.Since(iterStart))
		}

		// Detect confidence signals for adaptive early termination
		if detectConfidence(response) {
			confidenceSignals++
			if r.config.Verbose {
				fmt.Printf("[RLM] Confidence signal detected (total: %d)\n", confidenceSignals)
			}
		}

		// Check for final answer
		if len(codeBlocks) == 0 && parsing.FindFinalAnswer(response) != nil {
			final := parsing.FindFinalAnswer(response)
			varName := final.Content
			varValue := varName

			if final.Type == core.FinalTypeVariable {
				resolved, err := replEnv.GetVariable(varName)
				if err != nil {
					if r.config.Verbose {
						fmt.Printf("[RLM] Warning: could not resolve variable %q: %v\n", varName, err)
					}
				} else {
					varValue = resolved
				}
			}

			return &core.CompletionResult{
				Response:   varValue,
				Iterations: i + 1,
				Duration:   time.Since(start),
				Usage: core.UsageStats{
					PromptTokens:        totalPromptTokens,
					CompletionTokens:    totalCompletionTokens,
					TotalTokens:         totalPromptTokens + totalCompletionTokens,
					CacheCreationTokens: totalCacheCreationTokens,
					CacheReadTokens:     totalCacheReadTokens,
				},
			}, nil
		}

		// Check for early termination based on confidence signals
		if r.shouldTerminateEarly(confidenceSignals, len(codeBlocks)) {
			if r.config.Verbose {
				fmt.Printf("[RLM] Early termination triggered (confidence signals: %d)\n", confidenceSignals)
			}
			return r.forceDefaultAnswer(ctx, messages, start, totalPromptTokens, totalCompletionTokens, totalCacheCreationTokens, totalCacheReadTokens)
		}

		// Append iteration results to history
		messages = r.appendIterationToHistory(messages, response, execResults)

		// Apply history compression if enabled
		if r.config.HistoryCompression != nil && r.config.HistoryCompression.Enabled {
			messages = r.compressHistory(messages, i+1)
		}
	}

	// Max iterations exhausted - force final answer
	return r.forceDefaultAnswer(ctx, messages, start, totalPromptTokens, totalCompletionTokens, totalCacheCreationTokens, totalCacheReadTokens)
}

// computeMaxIterationsForDepth calculates max iterations considering recursion depth.
func (r *RLM) computeMaxIterationsForDepth(contextSize int, recursionCtx *core.RecursionContext) int {
	// Check if there's a per-depth setting
	if recursionCtx != nil && r.config.Recursion != nil && r.config.Recursion.PerDepthMaxIterations != nil {
		if maxIter, ok := r.config.Recursion.PerDepthMaxIterations[recursionCtx.CurrentDepth]; ok {
			return maxIter
		}
	}

	// Fall back to adaptive or default calculation
	return r.computeMaxIterations(contextSize)
}

// RecursiveComplete runs an RLM completion with multi-depth recursion enabled.
// This is the public API for recursive completions.
func (r *RLM) RecursiveComplete(ctx context.Context, contextPayload any, query string) (*RecursiveCompletionResult, error) {
	// Validate recursion config
	if r.config.Recursion == nil || r.config.Recursion.MaxDepth <= 0 {
		// Fall back to regular Complete
		result, err := r.Complete(ctx, contextPayload, query)
		if err != nil {
			return nil, err
		}
		return &RecursiveCompletionResult{
			CompletionResult: *result,
			TokenStats:       NewRecursiveTokenStats(),
			MaxDepthReached:  0,
		}, nil
	}

	// Create token stats tracker
	tokenStats := NewRecursiveTokenStats()

	// Create root recursion context
	recursionCtx := core.NewRecursionContext(r.config.Recursion.MaxDepth)

	// Run with recursion support
	result, err := r.CompleteWithRecursion(ctx, contextPayload, query, recursionCtx, tokenStats)
	if err != nil {
		return nil, err
	}

	// Calculate max depth reached
	maxDepth := 0
	for depth := range tokenStats.CallsByDepth {
		if depth > maxDepth {
			maxDepth = depth
		}
	}

	return &RecursiveCompletionResult{
		CompletionResult: *result,
		TokenStats:       tokenStats,
		MaxDepthReached:  maxDepth,
	}, nil
}

// RecursiveCompletionResult extends CompletionResult with recursion-specific data.
type RecursiveCompletionResult struct {
	core.CompletionResult

	// TokenStats aggregates token usage across all recursion levels.
	TokenStats *RecursiveTokenStats

	// MaxDepthReached is the maximum recursion depth that was used.
	MaxDepthReached int
}
