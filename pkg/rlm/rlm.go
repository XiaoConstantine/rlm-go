package rlm

import (
	"context"
	"errors"
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

// CostEstimator estimates total USD cost for the current usage.
// It is used to enforce MaxBudgetUSD when configured.
type CostEstimator func(usage core.UsageStats) (float64, error)

// Config holds RLM configuration.
type Config struct {
	// MaxIterations is the maximum number of iteration loops (default: 30).
	MaxIterations int

	// MaxBudgetUSD is the maximum allowed total cost in USD for one completion.
	// Disabled when <= 0.
	MaxBudgetUSD float64

	// MaxTimeout is the maximum wall-clock duration for one completion.
	// Disabled when <= 0.
	MaxTimeout time.Duration

	// MaxTokens is the maximum allowed total tokens (prompt + completion).
	// Disabled when <= 0.
	MaxTokens int

	// MaxErrors is the maximum allowed consecutive code execution errors.
	// Disabled when <= 0.
	MaxErrors int

	// CostEstimator estimates total USD cost from cumulative usage.
	// Required when MaxBudgetUSD is enabled.
	CostEstimator CostEstimator

	// SystemPrompt overrides the default system prompt.
	SystemPrompt string

	// PromptPolicy optionally appends model-specific operating guidance to the
	// root prompt without replacing the base RLM protocol.
	PromptPolicy *PromptPolicy

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

	// OnProgress is emitted after each root LLM call with progress info.
	// Can be used to display progress to users or implement custom termination logic.
	OnProgress func(progress IterationProgress)

	// REPLSetup is called after REPL creation and context loading, but before
	// the iteration loop starts. Use this to inject additional REPL symbols.
	REPLSetup func(replEnv *repl.REPL) error

	// MaxFullContextQueryChars blocks Query/QueryBatched from prepending the
	// entire loaded context when it is larger than this many chars.
	// Disabled when <= 0. Use QueryWith or QueryRaw for large contexts.
	MaxFullContextQueryChars int

	maxFullContextQueryCharsSet bool

	// Recursion configures multi-depth recursion behavior.
	// When enabled, sub-LLMs can spawn their own sub-LLMs.
	Recursion *RecursionConfig

	// Sandbox configures isolated execution for code blocks.
	// When enabled, code runs in Podman/Docker containers instead of in-process.
	// This provides better security isolation at the cost of execution speed.
	Sandbox *SandboxConfig

	// CompactHistory configures compact string-based history tracking.
	// When enabled, uses a more token-efficient history format.
	CompactHistory *CompactHistoryConfig

	// AutoCompactHistoryThreshold enables compact history automatically when
	// context size is at least this many bytes. Disabled when <= 0.
	AutoCompactHistoryThreshold int
}

// SandboxConfig configures sandboxed code execution.
type SandboxConfig struct {
	// Enabled turns on sandbox execution (default: false).
	Enabled bool

	// Config contains the detailed sandbox configuration.
	// If nil when Enabled is true, DefaultConfig() is used.
	Config *sandbox.Config

	configFromUser bool
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

// CompactHistoryConfig configures compact string-based history tracking.
// When enabled, iteration history is tracked as a compact string instead of
// separate messages, reducing token usage significantly.
type CompactHistoryConfig struct {
	// Enabled turns on compact history mode (default: false).
	Enabled bool

	// MaxHistoryLength is the maximum length of the history string in characters.
	// Older entries are trimmed when this limit is exceeded. Default: 10000.
	MaxHistoryLength int

	// IncludeFewShot includes few-shot examples in the prompt. Default: true.
	IncludeFewShot bool
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

	// RootPromptTokens is the prompt token count for this iteration's root LLM call.
	// This is the per-call value (not cumulative).
	RootPromptTokens int
}

// PromptPolicy contains model-specific RLM operating guidance.
type PromptPolicy struct {
	// Name identifies the policy in the root prompt.
	Name string

	// SubLLMContextChars is the recommended maximum context size for one sub-call.
	SubLLMContextChars int

	// BatchChars is the recommended target size for batched sub-call chunks.
	BatchChars int

	// MaxSubCalls is an optional soft budget for generated plans.
	MaxSubCalls int

	// ExtraInstructions contains additional model-specific guidance.
	ExtraInstructions []string
}

const (
	// DefaultMaxFullContextQueryChars is the default guardrail for Query and
	// QueryBatched calls that would prepend the entire loaded context.
	DefaultMaxFullContextQueryChars = 200000

	// DefaultAutoCompactHistoryThreshold is the default context size at which
	// Complete switches to compact history.
	DefaultAutoCompactHistoryThreshold = 200000
)

// DefaultConfig returns the default RLM configuration.
func DefaultConfig() Config {
	return Config{
		MaxIterations:               30,
		MaxFullContextQueryChars:    DefaultMaxFullContextQueryChars,
		AutoCompactHistoryThreshold: DefaultAutoCompactHistoryThreshold,
		SystemPrompt:                SystemPrompt,
	}
}

// PromptPolicyForModel returns model-specific RLM guidance for known model families.
func PromptPolicyForModel(model string) (PromptPolicy, bool) {
	lower := strings.ToLower(model)
	switch {
	case strings.Contains(lower, "qwen"):
		return PromptPolicy{
			Name:               "qwen-conservative",
			SubLLMContextChars: 200000,
			BatchChars:         100000,
			MaxSubCalls:        100,
			ExtraInstructions: []string{
				"Use QueryWith or QueryBatchedRaw over selected slices; avoid full-context Query on large inputs.",
				"Batch examples before sub-calling; do not recurse per row unless pairwise or multi-step analysis requires it.",
				"Keep Go code short and verify syntax before launching many recursive calls.",
			},
		}, true
	case strings.Contains(lower, "mini") || strings.Contains(lower, "flash") || strings.Contains(lower, "8b"):
		return PromptPolicy{
			Name:               "small-model-conservative",
			SubLLMContextChars: 200000,
			BatchChars:         100000,
			MaxSubCalls:        64,
			ExtraInstructions: []string{
				"Prefer deterministic slicing plus batched sub-calls over deep recursion.",
				"Keep code and FINAL outputs short to preserve output budget.",
			},
		}, true
	default:
		return PromptPolicy{}, false
	}
}

// RLM is the main Recursive Language Model implementation.
type RLM struct {
	client         LLMClient
	replClient     repl.LLMClient
	config         Config
	limitConfigErr error
}

// logf is a conditional logging helper that only prints when verbose mode is enabled.
func (r *RLM) logf(format string, args ...any) {
	if r.config.Verbose {
		fmt.Printf("[RLM] "+format+"\n", args...)
	}
}

// logCacheStats logs cache statistics if there are any.
func (r *RLM) logCacheStats(llmResp core.LLMResponse, state *iterationState) {
	if llmResp.CacheCreationTokens > 0 || llmResp.CacheReadTokens > 0 {
		r.logf("Cache stats this iteration: created=%d, read=%d (totals: created=%d, read=%d)",
			llmResp.CacheCreationTokens, llmResp.CacheReadTokens,
			state.totalCacheCreationTokens, state.totalCacheReadTokens)
	}
}

func validateLimitConfig(cfg Config) error {
	if cfg.MaxBudgetUSD > 0 && cfg.CostEstimator == nil {
		return &BudgetEstimatorNotConfiguredError{
			BudgetUSD: cfg.MaxBudgetUSD,
		}
	}
	return nil
}

func (r *RLM) checkCancellation(ctx context.Context, state *iterationState) error {
	select {
	case <-ctx.Done():
		return &CancellationError{
			Cause:   ctx.Err(),
			partial: state.buildPartialResult(),
		}
	default:
		return nil
	}
}

func (r *RLM) checkTimeout(state *iterationState, completedIterations int) error {
	if r.config.MaxTimeout <= 0 {
		return nil
	}
	elapsed := time.Since(state.start)
	if elapsed <= r.config.MaxTimeout {
		return nil
	}
	return &TimeoutExceededError{
		CompletedIterations: completedIterations,
		Elapsed:             elapsed,
		Timeout:             r.config.MaxTimeout,
		partial:             state.buildPartialResult(),
		cause:               context.DeadlineExceeded,
	}
}

func (r *RLM) withRemainingTimeout(ctx context.Context, state *iterationState, completedIterations int) (context.Context, context.CancelFunc, error) {
	if r.config.MaxTimeout <= 0 {
		return ctx, func() {}, nil
	}
	remaining := r.config.MaxTimeout - time.Since(state.start)
	if remaining <= 0 {
		return nil, nil, &TimeoutExceededError{
			CompletedIterations: completedIterations,
			Elapsed:             time.Since(state.start),
			Timeout:             r.config.MaxTimeout,
			partial:             state.buildPartialResult(),
			cause:               context.DeadlineExceeded,
		}
	}
	callCtx, cancel := context.WithTimeout(ctx, remaining)
	return callCtx, cancel, nil
}

func (r *RLM) wrapCallError(err error, state *iterationState, completedIterations int) error {
	if err == nil {
		return nil
	}
	if errors.Is(err, context.DeadlineExceeded) && r.config.MaxTimeout > 0 {
		return &TimeoutExceededError{
			CompletedIterations: completedIterations,
			Elapsed:             time.Since(state.start),
			Timeout:             r.config.MaxTimeout,
			partial:             state.buildPartialResult(),
			cause:               err,
		}
	}
	if errors.Is(err, context.Canceled) || errors.Is(err, context.DeadlineExceeded) {
		return &CancellationError{
			Cause:   err,
			partial: state.buildPartialResult(),
		}
	}
	return err
}

func (r *RLM) updateIterationStateAndCheckLimits(state *iterationState, execResults []core.CodeBlock, iteration int) error {
	if r.config.MaxErrors > 0 {
		iterationHadError := false
		for _, codeBlock := range execResults {
			if strings.TrimSpace(codeBlock.Result.Stderr) != "" {
				iterationHadError = true
				state.lastError = codeBlock.Result.Stderr
				break
			}
		}

		if iterationHadError {
			state.consecutiveErrors++
		} else {
			state.consecutiveErrors = 0
			state.lastError = ""
		}

		if state.consecutiveErrors >= r.config.MaxErrors {
			return &ErrorThresholdExceededError{
				Iteration:  iteration,
				ErrorCount: state.consecutiveErrors,
				Threshold:  r.config.MaxErrors,
				LastError:  state.lastError,
				partial:    state.buildPartialResult(),
			}
		}
	}

	if r.config.MaxTokens > 0 {
		totalTokens := state.totalPromptTokens + state.totalCompletionTokens
		if totalTokens > r.config.MaxTokens {
			return &TokenLimitExceededError{
				Iteration:  iteration,
				TokensUsed: totalTokens,
				TokenLimit: r.config.MaxTokens,
				partial:    state.buildPartialResult(),
			}
		}
	}

	if r.config.MaxBudgetUSD > 0 {
		totalCostUSD, err := r.config.CostEstimator(state.usageStats())
		if err != nil {
			return fmt.Errorf("iteration %d: cost estimation failed: %w", iteration, err)
		}
		state.totalCostUSD = totalCostUSD

		if totalCostUSD > r.config.MaxBudgetUSD {
			return &BudgetExceededError{
				Iteration: iteration,
				SpentUSD:  totalCostUSD,
				BudgetUSD: r.config.MaxBudgetUSD,
				partial:   state.buildPartialResult(),
			}
		}
	}

	return nil
}

// tryResetInterpreter attempts to reset the interpreter if it supports the ResetIfNeeded interface.
func (r *RLM) tryResetInterpreter(execEnv ExecutionEnvironment) {
	if resetter, ok := execEnv.(interface{ ResetIfNeeded() (bool, error) }); ok {
		if reset, resetErr := resetter.ResetIfNeeded(); reset {
			if resetErr != nil {
				r.logf("Interpreter reset failed: %v", resetErr)
			} else {
				r.logf("Interpreter reset successful")
			}
		}
	}
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
		client:         client,
		replClient:     replClient,
		config:         cfg,
		limitConfigErr: validateLimitConfig(cfg),
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

// WithMaxBudgetUSD sets the maximum allowed budget in USD.
func WithMaxBudgetUSD(budget float64) Option {
	return func(c *Config) {
		if budget <= 0 {
			budget = 0
		}
		c.MaxBudgetUSD = budget
	}
}

// WithMaxTimeout sets the maximum wall-clock completion duration.
func WithMaxTimeout(timeout time.Duration) Option {
	return func(c *Config) {
		if timeout <= 0 {
			timeout = 0
		}
		c.MaxTimeout = timeout
	}
}

// WithMaxTokens sets the maximum total tokens (prompt + completion).
func WithMaxTokens(tokens int) Option {
	return func(c *Config) {
		if tokens <= 0 {
			tokens = 0
		}
		c.MaxTokens = tokens
	}
}

// WithMaxErrors sets the maximum consecutive code execution errors.
func WithMaxErrors(maxErrors int) Option {
	return func(c *Config) {
		if maxErrors <= 0 {
			maxErrors = 0
		}
		c.MaxErrors = maxErrors
	}
}

// WithCostEstimator sets the estimator used for budget enforcement.
func WithCostEstimator(estimator CostEstimator) Option {
	return func(c *Config) {
		c.CostEstimator = estimator
	}
}

// WithSystemPrompt sets a custom system prompt.
func WithSystemPrompt(prompt string) Option {
	return func(c *Config) {
		c.SystemPrompt = prompt
	}
}

// WithPromptPolicy appends model-specific RLM operating guidance to the system prompt.
func WithPromptPolicy(policy PromptPolicy) Option {
	return func(c *Config) {
		c.PromptPolicy = &policy
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

// WithREPLSetup sets a callback that runs after REPL creation + context load
// and before the iteration loop begins.
func WithREPLSetup(fn func(replEnv *repl.REPL) error) Option {
	return func(c *Config) {
		c.REPLSetup = fn
	}
}

// WithMaxFullContextQueryChars blocks Query/QueryBatched when they would
// prepend a loaded context larger than max chars. Zero disables the guard.
func WithMaxFullContextQueryChars(max int) Option {
	return func(c *Config) {
		if max < 0 {
			max = 0
		}
		c.MaxFullContextQueryChars = max
		c.maxFullContextQueryCharsSet = true
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
			Enabled:        true,
			Config:         &cfg,
			configFromUser: true,
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

// WithCompactHistory enables compact string-based history tracking.
// This approach is more token-efficient than the default message array history.
// includeFewShot controls whether few-shot examples are included in the prompt.
func WithCompactHistory(includeFewShot bool) Option {
	return func(c *Config) {
		c.CompactHistory = &CompactHistoryConfig{
			Enabled:          true,
			MaxHistoryLength: 10000,
			IncludeFewShot:   includeFewShot,
		}
	}
}

// WithCompactHistoryConfig enables compact history with custom configuration.
func WithCompactHistoryConfig(cfg CompactHistoryConfig) Option {
	return func(c *Config) {
		cfg.Enabled = true
		if cfg.MaxHistoryLength <= 0 {
			cfg.MaxHistoryLength = 10000
		}
		c.CompactHistory = &cfg
	}
}

// WithAutoCompactHistoryThreshold sets the context size at which Complete uses
// compact history automatically. Set threshold <= 0 to disable auto compact history.
func WithAutoCompactHistoryThreshold(threshold int) Option {
	return func(c *Config) {
		if threshold < 0 {
			threshold = 0
		}
		c.AutoCompactHistoryThreshold = threshold
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

func (r *RLM) compactHistoryConfigForContext(contextSize int) *CompactHistoryConfig {
	if r.config.CompactHistory != nil {
		if r.config.CompactHistory.Enabled {
			return r.config.CompactHistory
		}
		return nil
	}
	if r.config.AutoCompactHistoryThreshold <= 0 || contextSize < r.config.AutoCompactHistoryThreshold {
		return nil
	}
	return &CompactHistoryConfig{
		Enabled:          true,
		MaxHistoryLength: defaultMaxHistoryLen,
		IncludeFewShot:   true,
	}
}

type finalStateEnvironment interface {
	HasFinal() bool
	Final() (string, bool)
	ClearFinal()
}

type finalStateSupporter interface {
	SupportsFinalState() bool
}

func clearFinalState(execEnv any) {
	if finalEnv, ok := execEnv.(finalStateEnvironment); ok {
		finalEnv.ClearFinal()
	}
}

func getFinalState(execEnv any) (string, bool) {
	finalEnv, ok := execEnv.(finalStateEnvironment)
	if !ok || !finalEnv.HasFinal() {
		return "", false
	}
	return finalEnv.Final()
}

func supportsFinalState(execEnv any) bool {
	if supporter, ok := execEnv.(finalStateSupporter); ok {
		return supporter.SupportsFinalState()
	}
	_, ok := execEnv.(finalStateEnvironment)
	return ok
}

// Complete runs an RLM completion.
// contextPayload is the context data (string, map, or slice).
// query is the user's question.
func (r *RLM) Complete(ctx context.Context, contextPayload any, query string) (*core.CompletionResult, error) {
	// Use compact history mode when explicitly enabled or automatically for
	// large contexts.
	if compactCfg := r.compactHistoryConfigForContext(getContextSize(contextPayload)); compactCfg != nil {
		compactRLM := *r
		compactRLM.config.CompactHistory = compactCfg
		return compactRLM.CompleteWithCompactHistory(ctx, contextPayload, query)
	}

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
	if err := r.runREPLSetup(execEnv); err != nil {
		return nil, err
	}

	// Build initial message history
	messages := r.buildInitialMessagesFromEnv(execEnv, query)

	// Initialize iteration state
	contextSize := getContextSize(contextPayload)
	maxIterations := r.computeMaxIterations(contextSize)
	state := newIterationState(contextSize, maxIterations)
	if r.limitConfigErr != nil {
		return nil, r.limitConfigErr
	}

	// Iteration loop
	for i := 0; i < maxIterations; i++ {
		iterStart := time.Now()

		if err := r.checkCancellation(ctx, state); err != nil {
			return nil, err
		}
		if err := r.checkTimeout(state, i); err != nil {
			return nil, err
		}

		r.logf("Iteration %d/%d", i+1, maxIterations)

		// Add iteration-specific user prompt
		currentMessages := r.appendIterationPrompt(messages, i, query)

		callCtx, cancelCall, err := r.withRemainingTimeout(ctx, state, i)
		if err != nil {
			return nil, err
		}

		// Get LLM response (streaming or non-streaming)
		llmResp, err := r.completeWithOptionalStreaming(callCtx, currentMessages)
		cancelCall()
		if err != nil {
			wrappedErr := r.wrapCallError(err, state, i)
			if wrappedErr != err {
				return nil, wrappedErr
			}
			return nil, fmt.Errorf("iteration %d: llm completion failed: %w", i, err)
		}
		response := llmResp.Content

		// Report progress after the root LLM call so per-iteration tokens are available.
		if r.config.OnProgress != nil {
			r.config.OnProgress(IterationProgress{
				CurrentIteration:  i + 1,
				MaxIterations:     maxIterations,
				ConfidenceSignals: state.confidenceSignals,
				HasFinalAttempt:   false,
				ContextSize:       contextSize,
				RootPromptTokens:  llmResp.PromptTokens,
			})
		}

		// Aggregate tokens
		state.aggregateTokens(llmResp)
		state.recordLatestPartial(response, i+1)
		r.logCacheStats(llmResp, state)
		r.logf("Response: %s", truncate(response, truncateLenShort))

		// Extract and execute code blocks
		codeBlocks := parsing.FindCodeBlocks(response)
		clearFinalState(execEnv)
		execResult := r.executeCodeBlocks(ctx, execEnv, codeBlocks, contextPayload)

		// Get LLM calls made during code execution and aggregate tokens
		llmCalls := execEnv.GetLLMCalls()
		rlmCalls := convertCallsToLoggerEntries(llmCalls)
		state.aggregateLLMCallTokens(llmCalls)

		if err := r.updateIterationStateAndCheckLimits(state, execResult.execResults, i+1); err != nil {
			return nil, err
		}

		// Get locals from execution environment for logging
		locals := execEnv.GetLocals()

		// Detect confidence signals for adaptive early termination
		if detectConfidence(response) {
			state.confidenceSignals++
			r.logf("Confidence signal detected (total: %d)", state.confidenceSignals)
		}

		// Prefer explicit REPL completion state set by FINAL/FINAL_VAR calls in code.
		if finalValue, ok := getFinalState(execEnv); ok {
			r.logf("Found FINAL state: %s", truncate(finalValue, truncateLenPreview))

			if r.config.Logger != nil {
				_ = r.config.Logger.LogIteration(i+1, currentMessages, response, execResult.execResults, rlmCalls, locals, finalValue, time.Since(iterStart))
			}
			return state.buildResult(finalValue, i+1), nil
		}

		// Fall back to output parsing only for environments without out-of-band final state.
		if !supportsFinalState(execEnv) {
			if final := parsing.FindFinalAnswer(execResult.allOutput.String()); final != nil {
				resultResponse := final.Content
				r.logf("Found FINAL in execution output: %s", truncate(resultResponse, truncateLenPreview))

				if r.config.Logger != nil {
					_ = r.config.Logger.LogIteration(i+1, currentMessages, response, execResult.execResults, rlmCalls, locals, resultResponse, time.Since(iterStart))
				}
				return state.buildResult(resultResponse, i+1), nil
			}
		}

		// Check for final answer - BUT only if there were NO code blocks in this response.
		if len(codeBlocks) == 0 {
			if final := parsing.FindFinalAnswer(response); final != nil {
				finalAnswer, resultResponse, _ := r.processFinalAnswer(final, execEnv)

				// Log iteration before returning
				if r.config.Logger != nil {
					_ = r.config.Logger.LogIteration(i+1, currentMessages, response, execResult.execResults, rlmCalls, locals, finalAnswer, time.Since(iterStart))
				}

				return state.buildResult(resultResponse, i+1), nil
			}
		}

		// Log iteration (no final answer)
		if r.config.Logger != nil {
			_ = r.config.Logger.LogIteration(i+1, currentMessages, response, execResult.execResults, rlmCalls, locals, nil, time.Since(iterStart))
		}

		// Check for early termination based on confidence signals
		if r.shouldTerminateEarly(state.confidenceSignals, len(codeBlocks)) {
			r.logf("Early termination triggered (confidence signals: %d)", state.confidenceSignals)
			return r.forceDefaultAnswer(ctx, messages, state, i+1)
		}

		// Append iteration results to history
		messages = r.appendIterationToHistory(messages, response, execResult.execResults)

		// Apply history compression if enabled and we have enough iterations
		if r.config.HistoryCompression != nil && r.config.HistoryCompression.Enabled {
			messages = r.compressHistory(messages, i+1)
		}
	}

	// Max iterations exhausted - force final answer
	return r.forceDefaultAnswer(ctx, messages, state, maxIterations)
}

func (r *RLM) systemPromptFor(recursionCtx *core.RecursionContext) string {
	systemPrompt := r.config.SystemPrompt
	if recursionCtx != nil && recursionCtx.MaxDepth > 0 && recursionCtx.CanRecurse() {
		systemPrompt = RecursiveSystemPrompt
	}
	return appendPromptPolicy(systemPrompt, r.config.PromptPolicy)
}

func appendPromptPolicy(systemPrompt string, policy *PromptPolicy) string {
	if policy == nil || policy.Name == "" {
		return systemPrompt
	}

	var b strings.Builder
	b.WriteString(systemPrompt)
	b.WriteString("\n\nMODEL-SPECIFIC RLM POLICY:\n")
	b.WriteString("- Profile: ")
	b.WriteString(policy.Name)
	b.WriteByte('\n')
	if policy.SubLLMContextChars > 0 {
		fmt.Fprintf(&b, "- Recommended sub-LLM context cap: %d characters per call.\n", policy.SubLLMContextChars)
	}
	if policy.BatchChars > 0 {
		fmt.Fprintf(&b, "- Recommended batch chunk target: %d characters.\n", policy.BatchChars)
	}
	if policy.MaxSubCalls > 0 {
		fmt.Fprintf(&b, "- Soft sub-call budget: at most %d sub-calls unless the task explicitly requires more.\n", policy.MaxSubCalls)
	}
	for _, instruction := range policy.ExtraInstructions {
		if strings.TrimSpace(instruction) == "" {
			continue
		}
		b.WriteString("- ")
		b.WriteString(instruction)
		b.WriteByte('\n')
	}
	return b.String()
}

// buildInitialMessages creates the initial message history.
func (r *RLM) buildInitialMessages(replEnv *repl.REPL, query string) []core.Message {
	contextInfo := replEnv.ContextInfo()
	userPrompt := fmt.Sprintf(UserPromptTemplate, contextInfo, query) + FirstIterationSuffix

	return []core.Message{
		{Role: "system", Content: r.systemPromptFor(nil)},
		{Role: "user", Content: userPrompt},
	}
}

// buildInitialMessagesFromEnv creates the initial message history using the ExecutionEnvironment interface.
func (r *RLM) buildInitialMessagesFromEnv(execEnv ExecutionEnvironment, query string) []core.Message {
	contextInfo := execEnv.ContextInfo()
	userPrompt := fmt.Sprintf(UserPromptTemplate, contextInfo, query) + FirstIterationSuffix

	return []core.Message{
		{Role: "system", Content: r.systemPromptFor(nil)},
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
			formatExecutionResultForHistory(&block.Result),
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

	r.logf("Compressed history: %d -> %d messages (summarized %d iteration messages)",
		len(messages), len(result), len(toCompress))

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
						if len(outputPreview) > summaryPreviewLen {
							outputPreview = outputPreview[:summaryPreviewLen] + "..."
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

func (r *RLM) runREPLSetup(execEnv ExecutionEnvironment) error {
	if r.config.REPLSetup == nil {
		return nil
	}

	replAdapter, ok := execEnv.(*REPLAdapter)
	if !ok || replAdapter == nil || replAdapter.repl == nil {
		return fmt.Errorf("REPL setup hook requires standard REPL execution environment")
	}

	if err := r.config.REPLSetup(replAdapter.repl); err != nil {
		return fmt.Errorf("REPL setup hook failed: %w", err)
	}
	return nil
}

// forceDefaultAnswer forces the LLM to provide a final answer.
func (r *RLM) forceDefaultAnswer(ctx context.Context, messages []core.Message, state *iterationState, iterations int) (*core.CompletionResult, error) {
	messages = append(messages, core.Message{
		Role:    "user",
		Content: DefaultAnswerPrompt,
	})

	llmResp, err := r.client.Complete(ctx, messages)
	if err != nil {
		return nil, r.wrapCallError(fmt.Errorf("default answer: llm completion failed: %w", err), state, iterations)
	}

	// Add tokens from this final call
	state.aggregateTokens(llmResp)

	// Try to extract FINAL from response
	answer := llmResp.Content
	if final := parsing.FindFinalAnswer(llmResp.Content); final != nil {
		answer = final.Content
	}
	state.recordLatestPartial(answer, iterations)

	return state.buildResult(answer, iterations), nil
}

// completeWithOptionalStreaming calls the LLM with streaming if enabled and supported.
func (r *RLM) completeWithOptionalStreaming(ctx context.Context, messages []core.Message) (core.LLMResponse, error) {
	// Check if streaming is enabled and the client supports it
	if r.config.EnableStreaming {
		if streamClient, ok := r.client.(StreamingLLMClient); ok {
			return streamClient.CompleteStream(ctx, messages, r.config.OnStreamChunk)
		}
		// Fall back to non-streaming if client doesn't support it
		r.logf("Streaming requested but client doesn't support it, falling back to non-streaming")
	}
	return r.client.Complete(ctx, messages)
}

// truncate shortens a string for logging (uses core.Truncate).
func truncate(s string, maxLen int) string {
	return core.Truncate(s, maxLen)
}

// truncateString shortens a string with a suffix indicator for message content.
func truncateString(s string, maxLen int) string {
	if maxLen <= 0 {
		maxLen = truncateLenLong // Default to long truncation
	}
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
		adapter := NewRecursiveClientAdapter(r, recursionCtx, tokenStats, contextPayload)
		replEnv = repl.NewRecursiveREPL(adapter, recursionCtx)
	} else {
		if r.config.REPLPool != nil {
			replEnv = r.config.REPLPool.Get()
		} else {
			replEnv = repl.New(r.replClient)
		}
	}
	defer replEnv.Close()

	if err := replEnv.LoadContext(contextPayload); err != nil {
		return nil, fmt.Errorf("failed to load context: %w", err)
	}
	if r.config.REPLSetup != nil {
		if baseREPL, ok := replEnv.(*repl.REPL); ok {
			if err := r.config.REPLSetup(baseREPL); err != nil {
				return nil, fmt.Errorf("REPL setup hook failed: %w", err)
			}
		} else {
			return nil, fmt.Errorf("REPL setup hook requires standard REPL execution environment")
		}
	}

	contextInfo := replEnv.ContextInfo()
	userPrompt := fmt.Sprintf(UserPromptTemplate, contextInfo, query) + FirstIterationSuffix
	messages := []core.Message{
		{Role: "system", Content: r.systemPromptFor(recursionCtx)},
		{Role: "user", Content: userPrompt},
	}

	contextSize := getContextSize(contextPayload)
	maxIterations := r.computeMaxIterationsForDepth(contextSize, recursionCtx)
	state := newIterationState(contextSize, maxIterations)
	if r.limitConfigErr != nil {
		return nil, r.limitConfigErr
	}

	for i := 0; i < maxIterations; i++ {
		iterStart := time.Now()

		if err := r.checkCancellation(ctx, state); err != nil {
			return nil, err
		}
		if err := r.checkTimeout(state, i); err != nil {
			return nil, err
		}

		depth := 0
		if recursionCtx != nil {
			depth = recursionCtx.CurrentDepth
		}
		r.logf("Depth %d, Iteration %d/%d", depth, i+1, maxIterations)

		currentMessages := r.appendIterationPrompt(messages, i, query)
		callCtx, cancelCall, err := r.withRemainingTimeout(ctx, state, i)
		if err != nil {
			return nil, err
		}
		llmResp, err := r.completeWithOptionalStreaming(callCtx, currentMessages)
		cancelCall()
		if err != nil {
			wrappedErr := r.wrapCallError(err, state, i)
			if wrappedErr != err {
				return nil, wrappedErr
			}
			return nil, fmt.Errorf("iteration %d: llm completion failed: %w", i, err)
		}
		response := llmResp.Content

		if r.config.OnProgress != nil {
			r.config.OnProgress(IterationProgress{
				CurrentIteration:  i + 1,
				MaxIterations:     maxIterations,
				ConfidenceSignals: state.confidenceSignals,
				HasFinalAttempt:   false,
				ContextSize:       contextSize,
				RootPromptTokens:  llmResp.PromptTokens,
			})
		}

		state.aggregateTokens(llmResp)
		state.recordLatestPartial(response, i+1)

		if tokenStats != nil && recursionCtx != nil {
			tokenStats.Add(recursionCtx.CurrentDepth, llmResp.PromptTokens, llmResp.CompletionTokens)
		}

		r.logCacheStats(llmResp, state)
		r.logf("Response: %s", truncate(response, 200))

		codeBlocks := parsing.FindCodeBlocks(response)
		var execResults []core.CodeBlock
		clearFinalState(replEnv)
		for _, code := range codeBlocks {
			r.logf("Executing code:\n%s", truncate(code, 200))
			result, _ := replEnv.Execute(ctx, code)
			execResults = append(execResults, core.CodeBlock{
				Code:   code,
				Result: *result,
			})
			if result.Stdout != "" {
				r.logf("Output: %s", truncate(result.Stdout, 200))
			}
			if result.Stderr != "" {
				r.logf("Stderr: %s", truncate(result.Stderr, 200))
			}
			if _, ok := getFinalState(replEnv); ok {
				break
			}
			if !supportsFinalState(replEnv) && parsing.FindFinalAnswer(sandbox.FormatExecutionResult(result)) != nil {
				break
			}
		}

		replCalls := replEnv.GetLLMCalls()
		llmCalls := make([]LLMCallRecord, len(replCalls))
		for i, call := range replCalls {
			llmCalls[i] = LLMCallRecord{
				Prompt:           call.Prompt,
				Response:         call.Response,
				Duration:         call.Duration,
				PromptTokens:     call.PromptTokens,
				CompletionTokens: call.CompletionTokens,
				Async:            call.Async,
			}
		}
		rlmCalls := convertCallsToLoggerEntries(llmCalls)
		state.aggregateLLMCallTokens(llmCalls)

		if err := r.updateIterationStateAndCheckLimits(state, execResults, i+1); err != nil {
			return nil, err
		}

		locals := map[string]any(nil)

		if detectConfidence(response) {
			state.confidenceSignals++
			r.logf("Confidence signal detected (total: %d)", state.confidenceSignals)
		}

		if finalValue, ok := getFinalState(replEnv); ok {
			r.logf("Found FINAL state: %s", truncate(finalValue, truncateLenPreview))
			if r.config.Logger != nil {
				if locals == nil {
					locals = replEnv.GetLocals()
				}
				_ = r.config.Logger.LogIteration(i+1, currentMessages, response, execResults, rlmCalls, locals, finalValue, time.Since(iterStart))
			}
			return state.buildResult(finalValue, i+1), nil
		}

		if !supportsFinalState(replEnv) {
			for _, block := range execResults {
				if final := parsing.FindFinalAnswer(sandbox.FormatExecutionResult(&block.Result)); final != nil {
					resultResponse := final.Content
					r.logf("Found FINAL in execution output: %s", truncate(resultResponse, truncateLenPreview))
					if r.config.Logger != nil {
						if locals == nil {
							locals = replEnv.GetLocals()
						}
						_ = r.config.Logger.LogIteration(i+1, currentMessages, response, execResults, rlmCalls, locals, resultResponse, time.Since(iterStart))
					}
					return state.buildResult(resultResponse, i+1), nil
				}
			}
		}

		if len(codeBlocks) == 0 && parsing.FindFinalAnswer(response) != nil {
			final := parsing.FindFinalAnswer(response)
			varName := final.Content
			varValue := varName
			if final.Type == core.FinalTypeVariable {
				resolved, err := replEnv.GetVariable(varName)
				if err != nil {
					r.logf("Warning: could not resolve variable %q: %v", varName, err)
				} else {
					varValue = resolved
				}
			}
			if r.config.Logger != nil {
				if locals == nil {
					locals = replEnv.GetLocals()
				}
				_ = r.config.Logger.LogIteration(i+1, currentMessages, response, execResults, rlmCalls, locals, varValue, time.Since(iterStart))
			}
			return state.buildResult(varValue, i+1), nil
		}

		if r.config.Logger != nil {
			if locals == nil {
				locals = replEnv.GetLocals()
			}
			_ = r.config.Logger.LogIteration(i+1, currentMessages, response, execResults, rlmCalls, locals, nil, time.Since(iterStart))
		}

		if r.shouldTerminateEarly(state.confidenceSignals, len(codeBlocks)) {
			r.logf("Early termination triggered (confidence signals: %d)", state.confidenceSignals)
			return r.forceDefaultAnswer(ctx, messages, state, i+1)
		}

		messages = r.appendIterationToHistory(messages, response, execResults)
		if r.config.HistoryCompression != nil && r.config.HistoryCompression.Enabled {
			messages = r.compressHistory(messages, i+1)
		}
	}

	return r.forceDefaultAnswer(ctx, messages, state, maxIterations)
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

// iterationState holds all shared state during an iteration loop.
type iterationState struct {
	// Token tracking
	totalPromptTokens        int
	totalCompletionTokens    int
	totalCacheCreationTokens int
	totalCacheReadTokens     int
	totalCostUSD             float64

	// Iteration tracking
	confidenceSignals int
	maxIterations     int
	contextSize       int
	consecutiveErrors int
	lastError         string

	// Timing
	start time.Time

	// Latest non-empty partial response seen so far.
	latestPartialResponse   string
	latestPartialIterations int
}

// newIterationState creates a new iteration state.
func newIterationState(contextSize, maxIterations int) *iterationState {
	return &iterationState{
		start:         time.Now(),
		contextSize:   contextSize,
		maxIterations: maxIterations,
	}
}

// aggregateTokens adds token usage from an LLM response.
func (s *iterationState) aggregateTokens(resp core.LLMResponse) {
	s.totalPromptTokens += resp.PromptTokens
	s.totalCompletionTokens += resp.CompletionTokens
	s.totalCacheCreationTokens += resp.CacheCreationTokens
	s.totalCacheReadTokens += resp.CacheReadTokens
}

// aggregateLLMCallTokens adds token usage from LLM calls made during code execution.
func (s *iterationState) aggregateLLMCallTokens(calls []LLMCallRecord) {
	for _, call := range calls {
		s.totalPromptTokens += call.PromptTokens
		s.totalCompletionTokens += call.CompletionTokens
	}
}

func (s *iterationState) recordLatestPartial(response string, iterations int) {
	if strings.TrimSpace(response) == "" {
		return
	}
	s.latestPartialResponse = response
	s.latestPartialIterations = iterations
}

func (s *iterationState) usageStats() core.UsageStats {
	return core.UsageStats{
		PromptTokens:        s.totalPromptTokens,
		CompletionTokens:    s.totalCompletionTokens,
		TotalTokens:         s.totalPromptTokens + s.totalCompletionTokens,
		CacheCreationTokens: s.totalCacheCreationTokens,
		CacheReadTokens:     s.totalCacheReadTokens,
	}
}

// buildResult creates a CompletionResult from the current state.
func (s *iterationState) buildResult(response string, iterations int) *core.CompletionResult {
	return &core.CompletionResult{
		Response:   response,
		Iterations: iterations,
		Duration:   time.Since(s.start),
		Usage:      s.usageStats(),
	}
}

func (s *iterationState) buildPartialResult() *core.CompletionResult {
	if strings.TrimSpace(s.latestPartialResponse) == "" {
		return nil
	}
	return s.buildResult(s.latestPartialResponse, s.latestPartialIterations)
}

// codeExecutionResult holds the results of executing code blocks.
type codeExecutionResult struct {
	execResults      []core.CodeBlock
	allOutput        strings.Builder // For compact history mode
	interpreterPanic bool
}

// executeCodeBlocks executes all code blocks and returns the results.
func (r *RLM) executeCodeBlocks(ctx context.Context, execEnv ExecutionEnvironment, codeBlocks []string, contextPayload any) *codeExecutionResult {
	result := &codeExecutionResult{}

	for _, code := range codeBlocks {
		r.logf("Executing code:\n%s", truncate(code, truncateLenShort))

		execResult, execErr := execEnv.Execute(ctx, code)
		if execErr != nil {
			r.logf("Interpreter error: %v", execErr)
			result.interpreterPanic = true
			r.tryResetInterpreter(execEnv)
		}

		result.execResults = append(result.execResults, core.CodeBlock{
			Code:   code,
			Result: *execResult,
		})

		// Collect output for compact history mode
		if execResult.Stdout != "" {
			result.allOutput.WriteString(execResult.Stdout)
			r.logf("Output: %s", truncate(execResult.Stdout, truncateLenShort))
		}
		if execResult.Stderr != "" {
			result.allOutput.WriteString("Error: ")
			result.allOutput.WriteString(execResult.Stderr)
			r.logf("Stderr: %s", truncate(execResult.Stderr, truncateLenShort))
		}
		if _, ok := getFinalState(execEnv); ok {
			break
		}
		if !supportsFinalState(execEnv) && parsing.FindFinalAnswer(sandbox.FormatExecutionResult(execResult)) != nil {
			break
		}
	}

	// Reload context if interpreter panicked
	if result.interpreterPanic {
		r.logf("Reloading context after interpreter reset")
		_ = execEnv.LoadContext(contextPayload)
	}

	return result
}

// processFinalAnswer processes a FINAL/FINAL_VAR signal and returns the resolved value.
// Returns (finalAnswer, resultResponse, found).
func (r *RLM) processFinalAnswer(final *core.FinalAnswer, execEnv ExecutionEnvironment) (any, string, bool) {
	if final == nil {
		return nil, "", false
	}

	varName := final.Content
	varValue := varName

	if final.Type == core.FinalTypeVariable {
		resolved, err := execEnv.GetVariable(varName)
		if err != nil {
			r.logf("Warning: could not resolve variable %q: %v", varName, err)
		} else {
			varValue = resolved
		}
		return []string{varName, varValue}, varValue, true
	}

	return varValue, varValue, true
}

// Constants for logging and history management.
const (
	// Truncation lengths for different contexts
	truncateLenShort   = 200   // For logging responses, code, output
	truncateLenMedium  = 500   // For history output in compact mode
	truncateLenLong    = 20000 // For message content truncation
	truncateLenPreview = 100   // For short previews (e.g., FINAL output)

	// History limits
	defaultMaxHistoryLen           = 10000 // Default max length for compact history string
	historyOutputMetadataThreshold = 4000  // Above this, keep metadata rather than raw output in root history.

	// Summary output preview limit
	summaryPreviewLen = 80
)

func formatExecutionResultForHistory(result *core.ExecutionResult) string {
	formatted := sandbox.FormatExecutionResult(result)
	if len(formatted) <= historyOutputMetadataThreshold {
		return formatted
	}

	var parts []string
	if result.Stdout != "" {
		parts = append(parts, fmt.Sprintf("stdout: %d chars; preview:\n%s", len(result.Stdout), truncate(result.Stdout, truncateLenMedium)))
	}
	if result.Stderr != "" {
		parts = append(parts, fmt.Sprintf("stderr: %d chars; preview:\n%s", len(result.Stderr), truncate(result.Stderr, truncateLenMedium)))
	}
	parts = append(parts, "[large REPL output omitted from root history; keep full data in REPL variables and print compact summaries]")
	return strings.Join(parts, "\n")
}

// appendIterationHistory appends a formatted iteration entry to the compact history string.
// This is more token-efficient than using separate messages for each iteration.
func appendIterationHistory(history *strings.Builder, iteration int, action, reasoning, code, output string) {
	fmt.Fprintf(history, "\n--- Iteration %d ---\n", iteration)
	fmt.Fprintf(history, "Action: %s\n", action)
	if reasoning != "" {
		fmt.Fprintf(history, "Reasoning: %s\n", truncate(reasoning, truncateLenShort))
	}
	if code != "" {
		fmt.Fprintf(history, "Code:\n```go\n%s\n```\n", code)
		if output != "" {
			fmt.Fprintf(history, "Output:\n%s\n", truncate(output, truncateLenMedium))
		}
	}
}

// buildCompactHistoryPrompt builds a prompt that includes compact history.
// This is more token-efficient than using separate messages for each iteration.
func (r *RLM) buildCompactHistoryPrompt(contextInfo, query, history string, iteration int) string {
	var prompt strings.Builder

	// Include few-shot examples if enabled
	if r.config.CompactHistory != nil && r.config.CompactHistory.IncludeFewShot {
		prompt.WriteString(FormatFewShotExamples())
		prompt.WriteString("\n")
	}

	// Add context info
	prompt.WriteString(fmt.Sprintf("Context: %s\n\n", contextInfo))
	prompt.WriteString(fmt.Sprintf("Query: %s\n", query))

	// Add history if we have any
	if history != "" {
		prompt.WriteString("\n=== PREVIOUS ITERATIONS ===\n")
		prompt.WriteString(history)
		prompt.WriteString("\n=== END HISTORY ===\n")
	}

	// Add iteration-specific instructions
	if iteration == 0 {
		prompt.WriteString("\nYou have not explored the context yet. Your first action should be to write Go code to:\n")
		prompt.WriteString("1. Check the context size: fmt.Println(len(context))\n")
		prompt.WriteString("2. Preview the content: fmt.Println(context[:min(1000, len(context))])\n")
		prompt.WriteString("3. Use QueryWith() over selected slices or QueryBatchedRaw with selected data\n\n")
		prompt.WriteString("Write your code now in a go code block:")
	} else {
		prompt.WriteString("\nBased on your previous exploration, continue working toward the answer.\n\n")
		prompt.WriteString("If you have found the answer and stored it in a variable:\n")
		prompt.WriteString("- Write FINAL_VAR(varName) on its own line (not in code)\n\n")
		prompt.WriteString("If you need more analysis:\n")
		prompt.WriteString("- Write more Go code to explore or query\n\n")
		prompt.WriteString(fmt.Sprintf("Remember: The original query was: %s\n\n", query))
		prompt.WriteString("Your next action:")
	}

	return prompt.String()
}

// trimHistoryIfNeeded trims the history string if it exceeds the maximum length.
// It removes older entries from the beginning while preserving complete iteration blocks.
func trimHistoryIfNeeded(history string, maxLen int) string {
	if len(history) <= maxLen {
		return history
	}

	// Find a good break point - look for an iteration marker
	// Start from where we need to cut
	cutPoint := len(history) - maxLen
	markerPrefix := "\n--- Iteration "

	// Look for the next iteration marker after the cut point
	nextMarker := strings.Index(history[cutPoint:], markerPrefix)
	if nextMarker != -1 {
		cutPoint += nextMarker
	}

	return "[...earlier iterations truncated...]\n" + history[cutPoint:]
}

// CompleteWithCompactHistory runs an RLM completion using compact string-based history.
// This is an alternative to the standard Complete method that uses less tokens for history.
func (r *RLM) CompleteWithCompactHistory(ctx context.Context, contextPayload any, query string) (*core.CompletionResult, error) {
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
	if err := r.runREPLSetup(execEnv); err != nil {
		return nil, err
	}

	// Initialize iteration state
	contextSize := getContextSize(contextPayload)
	maxIterations := r.computeMaxIterations(contextSize)
	state := newIterationState(contextSize, maxIterations)
	if r.limitConfigErr != nil {
		return nil, r.limitConfigErr
	}

	// Build compact history string
	var history strings.Builder
	contextInfo := execEnv.ContextInfo()

	// Get max history length from config
	maxHistoryLen := defaultMaxHistoryLen
	if r.config.CompactHistory != nil && r.config.CompactHistory.MaxHistoryLength > 0 {
		maxHistoryLen = r.config.CompactHistory.MaxHistoryLength
	}

	// Iteration loop
	for i := 0; i < maxIterations; i++ {
		iterStart := time.Now()

		if err := r.checkCancellation(ctx, state); err != nil {
			return nil, err
		}
		if err := r.checkTimeout(state, i); err != nil {
			return nil, err
		}

		r.logf("Iteration %d/%d (compact history)", i+1, maxIterations)

		// Trim history if needed
		historyStr := trimHistoryIfNeeded(history.String(), maxHistoryLen)

		// Build messages with compact history in user prompt
		userPrompt := r.buildCompactHistoryPrompt(contextInfo, query, historyStr, i)
		messages := []core.Message{
			{Role: "system", Content: r.systemPromptFor(nil)},
			{Role: "user", Content: userPrompt},
		}

		callCtx, cancelCall, err := r.withRemainingTimeout(ctx, state, i)
		if err != nil {
			return nil, err
		}

		// Get LLM response (streaming or non-streaming)
		llmResp, err := r.completeWithOptionalStreaming(callCtx, messages)
		cancelCall()
		if err != nil {
			wrappedErr := r.wrapCallError(err, state, i)
			if wrappedErr != err {
				return nil, wrappedErr
			}
			return nil, fmt.Errorf("iteration %d: llm completion failed: %w", i, err)
		}
		response := llmResp.Content

		// Report progress after the root LLM call so per-iteration tokens are available.
		if r.config.OnProgress != nil {
			r.config.OnProgress(IterationProgress{
				CurrentIteration:  i + 1,
				MaxIterations:     maxIterations,
				ConfidenceSignals: state.confidenceSignals,
				HasFinalAttempt:   false,
				ContextSize:       contextSize,
				RootPromptTokens:  llmResp.PromptTokens,
			})
		}

		// Aggregate tokens
		state.aggregateTokens(llmResp)
		state.recordLatestPartial(response, i+1)
		r.logCacheStats(llmResp, state)
		r.logf("Response: %s", truncate(response, truncateLenShort))

		// Extract and execute code blocks
		codeBlocks := parsing.FindCodeBlocks(response)
		clearFinalState(execEnv)
		execResult := r.executeCodeBlocks(ctx, execEnv, codeBlocks, contextPayload)

		// Get LLM calls made during code execution and aggregate tokens
		llmCalls := execEnv.GetLLMCalls()
		rlmCalls := convertCallsToLoggerEntries(llmCalls)
		state.aggregateLLMCallTokens(llmCalls)

		if err := r.updateIterationStateAndCheckLimits(state, execResult.execResults, i+1); err != nil {
			return nil, err
		}

		// CRITICAL FIX: Add Query results to output so the LLM can see them in subsequent iterations.
		for _, call := range llmCalls {
			execResult.allOutput.WriteString(fmt.Sprintf("\n[Query] %s\n[Result] %s\n",
				truncate(call.Prompt, truncateLenShort),
				call.Response))
			r.logf("Query result: %s", truncate(call.Response, truncateLenShort))
		}

		// Get locals from execution environment for logging
		locals := execEnv.GetLocals()

		// Detect confidence signals for adaptive early termination
		if detectConfidence(response) {
			state.confidenceSignals++
			r.logf("Confidence signal detected (total: %d)", state.confidenceSignals)
		}

		// Determine action type for history
		action := determineActionType(codeBlocks, response)

		// Check for final answer in BOTH the LLM response AND the execution output.
		outputStr := execResult.allOutput.String()

		// First check explicit REPL completion state from code-called FINAL/FINAL_VAR functions.
		if finalValue, ok := getFinalState(execEnv); ok {
			r.logf("Found FINAL state: %s", truncate(finalValue, truncateLenPreview))

			if r.config.Logger != nil {
				_ = r.config.Logger.LogIteration(i+1, messages, response, execResult.execResults, rlmCalls, locals, finalValue, time.Since(iterStart))
			}
			return state.buildResult(finalValue, i+1), nil
		}

		// Then check execution output for FINAL as a sandbox/legacy fallback.
		if !supportsFinalState(execEnv) {
			if final := parsing.FindFinalAnswer(outputStr); final != nil {
				resultResponse := final.Content
				r.logf("Found FINAL in execution output: %s", truncate(resultResponse, truncateLenPreview))

				if r.config.Logger != nil {
					_ = r.config.Logger.LogIteration(i+1, messages, response, execResult.execResults, rlmCalls, locals, resultResponse, time.Since(iterStart))
				}
				return state.buildResult(resultResponse, i+1), nil
			}
		}

		// Then check LLM response (only if no code blocks)
		if len(codeBlocks) == 0 {
			if final := parsing.FindFinalAnswer(response); final != nil {
				finalAnswer, resultResponse, _ := r.processFinalAnswer(final, execEnv)

				if r.config.Logger != nil {
					_ = r.config.Logger.LogIteration(i+1, messages, response, execResult.execResults, rlmCalls, locals, finalAnswer, time.Since(iterStart))
				}
				return state.buildResult(resultResponse, i+1), nil
			}
		}

		// Log iteration (no final answer)
		if r.config.Logger != nil {
			_ = r.config.Logger.LogIteration(i+1, messages, response, execResult.execResults, rlmCalls, locals, nil, time.Since(iterStart))
		}

		// Check for early termination based on confidence signals
		if r.shouldTerminateEarly(state.confidenceSignals, len(codeBlocks)) {
			r.logf("Early termination triggered (confidence signals: %d)", state.confidenceSignals)
			finalMessages := r.buildCompactHistoryMessages(contextInfo, query, history.String(), i)
			return r.forceDefaultAnswer(ctx, finalMessages, state, i+1)
		}

		// Append to compact history
		codeStr := ""
		if len(codeBlocks) > 0 {
			codeStr = codeBlocks[0] // Just include first code block for brevity
		}
		appendIterationHistory(&history, i+1, action, "", codeStr, outputStr)
	}

	// Max iterations exhausted - force final answer
	finalMessages := r.buildCompactHistoryMessages(contextInfo, query, history.String(), maxIterations-1)
	return r.forceDefaultAnswer(ctx, finalMessages, state, maxIterations)
}

// determineActionType returns the action type string for compact history.
func determineActionType(codeBlocks []string, response string) string {
	if len(codeBlocks) == 0 {
		return "thinking"
	}
	if strings.Contains(response, "Query(") || strings.Contains(response, "QueryBatched(") {
		return "query"
	}
	return "explore"
}

// buildCompactHistoryMessages builds the messages array for compact history mode.
func (r *RLM) buildCompactHistoryMessages(contextInfo, query, historyStr string, iteration int) []core.Message {
	return []core.Message{
		{Role: "system", Content: r.systemPromptFor(nil)},
		{Role: "user", Content: r.buildCompactHistoryPrompt(contextInfo, query, historyStr, iteration)},
	}
}
