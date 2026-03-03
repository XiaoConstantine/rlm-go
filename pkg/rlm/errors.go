package rlm

import (
	"fmt"
	"time"

	"github.com/XiaoConstantine/rlm-go/pkg/core"
)

// PartialResultError is implemented by errors that expose the best partial result.
type PartialResultError interface {
	error
	PartialResult() *core.CompletionResult
}

// BudgetEstimatorNotConfiguredError is returned when budget enforcement is enabled
// without a configured cost estimator.
type BudgetEstimatorNotConfiguredError struct {
	BudgetUSD float64
}

func (e *BudgetEstimatorNotConfiguredError) Error() string {
	return fmt.Sprintf("max budget set to $%.6f but no cost estimator is configured", e.BudgetUSD)
}

// BudgetExceededError indicates execution spent more than the configured budget.
type BudgetExceededError struct {
	Iteration int
	SpentUSD  float64
	BudgetUSD float64
	partial   *core.CompletionResult
	cause     error
}

func (e *BudgetExceededError) Error() string {
	return fmt.Sprintf(
		"budget exceeded after iteration %d: spent $%.6f of $%.6f",
		e.Iteration,
		e.SpentUSD,
		e.BudgetUSD,
	)
}

func (e *BudgetExceededError) PartialResult() *core.CompletionResult {
	return e.partial
}

func (e *BudgetExceededError) Unwrap() error {
	return e.cause
}

// TimeoutExceededError indicates execution exceeded the configured wall-clock timeout.
type TimeoutExceededError struct {
	CompletedIterations int
	Elapsed             time.Duration
	Timeout             time.Duration
	partial             *core.CompletionResult
	cause               error
}

func (e *TimeoutExceededError) Error() string {
	return fmt.Sprintf(
		"timeout exceeded after iteration %d: %s of %s",
		e.CompletedIterations,
		e.Elapsed.Round(time.Millisecond),
		e.Timeout.Round(time.Millisecond),
	)
}

func (e *TimeoutExceededError) PartialResult() *core.CompletionResult {
	return e.partial
}

func (e *TimeoutExceededError) Unwrap() error {
	return e.cause
}

// TokenLimitExceededError indicates execution exceeded the configured token limit.
type TokenLimitExceededError struct {
	Iteration  int
	TokensUsed int
	TokenLimit int
	partial    *core.CompletionResult
	cause      error
}

func (e *TokenLimitExceededError) Error() string {
	return fmt.Sprintf(
		"token limit exceeded after iteration %d: %d of %d tokens",
		e.Iteration,
		e.TokensUsed,
		e.TokenLimit,
	)
}

func (e *TokenLimitExceededError) PartialResult() *core.CompletionResult {
	return e.partial
}

func (e *TokenLimitExceededError) Unwrap() error {
	return e.cause
}

// ErrorThresholdExceededError indicates too many consecutive execution errors.
type ErrorThresholdExceededError struct {
	Iteration  int
	ErrorCount int
	Threshold  int
	LastError  string
	partial    *core.CompletionResult
	cause      error
}

func (e *ErrorThresholdExceededError) Error() string {
	return fmt.Sprintf(
		"error threshold exceeded after iteration %d: %d consecutive errors (limit: %d)",
		e.Iteration,
		e.ErrorCount,
		e.Threshold,
	)
}

func (e *ErrorThresholdExceededError) PartialResult() *core.CompletionResult {
	return e.partial
}

func (e *ErrorThresholdExceededError) Unwrap() error {
	return e.cause
}

// CancellationError wraps context cancellation and preserves partial progress.
type CancellationError struct {
	Cause   error
	partial *core.CompletionResult
}

func (e *CancellationError) Error() string {
	if e.Cause == nil {
		return "execution cancelled"
	}
	return fmt.Sprintf("execution cancelled: %v", e.Cause)
}

func (e *CancellationError) Unwrap() error {
	return e.Cause
}

func (e *CancellationError) PartialResult() *core.CompletionResult {
	return e.partial
}
