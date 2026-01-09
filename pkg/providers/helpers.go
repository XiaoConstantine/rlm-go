// Package providers contains shared helpers for LLM provider implementations.
package providers

import (
	"context"
	"fmt"
	"sync"

	"github.com/XiaoConstantine/rlm-go/pkg/core"
)

// SingleQueryFunc is a function that performs a single LLM query.
type SingleQueryFunc func(ctx context.Context, prompt string) (core.QueryResponse, error)

// QueryBatchedConcurrent executes multiple prompts concurrently using the provided query function.
// This is a shared implementation used by all provider clients.
func QueryBatchedConcurrent(ctx context.Context, prompts []string, queryFn SingleQueryFunc) ([]core.QueryResponse, error) {
	results := make([]core.QueryResponse, len(prompts))
	var wg sync.WaitGroup

	for i, prompt := range prompts {
		wg.Add(1)
		go func(idx int, p string) {
			defer wg.Done()
			result, err := queryFn(ctx, p)
			if err != nil {
				results[idx] = core.QueryResponse{Response: fmt.Sprintf("Error: %v", err)}
			} else {
				results[idx] = result
			}
		}(i, prompt)
	}

	wg.Wait()
	return results, nil
}
