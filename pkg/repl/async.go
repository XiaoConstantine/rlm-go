// Package repl provides async query execution for RLM sub-LLM calls.
package repl

import (
	"sync"
	"sync/atomic"
	"time"

	"github.com/google/uuid"
)

// AsyncQueryHandle represents a pending async query.
type AsyncQueryHandle struct {
	id       string
	resultCh chan QueryResponse
	done     chan struct{}
	result   *QueryResponse
	err      error
	mu       sync.Mutex
	started  time.Time
}

// newAsyncQueryHandle creates a new async query handle.
func newAsyncQueryHandle() *AsyncQueryHandle {
	return &AsyncQueryHandle{
		id:       uuid.New().String(),
		resultCh: make(chan QueryResponse, 1),
		done:     make(chan struct{}),
		started:  time.Now(),
	}
}

// ID returns the unique identifier for this async query.
func (h *AsyncQueryHandle) ID() string {
	return h.id
}

// Wait blocks until the query completes and returns the result.
func (h *AsyncQueryHandle) Wait() (string, error) {
	<-h.done

	h.mu.Lock()
	defer h.mu.Unlock()

	if h.err != nil {
		return "", h.err
	}
	if h.result != nil {
		return h.result.Response, nil
	}
	return "", nil
}

// Ready returns true if the result is available.
func (h *AsyncQueryHandle) Ready() bool {
	select {
	case <-h.done:
		return true
	default:
		return false
	}
}

// Result returns the result if ready, or empty string if not.
func (h *AsyncQueryHandle) Result() (string, bool) {
	if !h.Ready() {
		return "", false
	}

	h.mu.Lock()
	defer h.mu.Unlock()

	if h.result != nil {
		return h.result.Response, true
	}
	return "", true
}

// Error returns any error that occurred during the query.
func (h *AsyncQueryHandle) Error() error {
	h.mu.Lock()
	defer h.mu.Unlock()
	return h.err
}

// Duration returns the time elapsed since the query was started.
// If the query is complete, returns the total duration.
func (h *AsyncQueryHandle) Duration() time.Duration {
	h.mu.Lock()
	defer h.mu.Unlock()
	return time.Since(h.started)
}

// complete marks the handle as complete with the given result.
func (h *AsyncQueryHandle) complete(resp QueryResponse, err error) {
	h.mu.Lock()
	if err != nil {
		h.err = err
	} else {
		h.result = &resp
	}
	h.mu.Unlock()

	close(h.done)
}

// AsyncBatchHandle represents a batch of pending async queries.
type AsyncBatchHandle struct {
	handles    []*AsyncQueryHandle
	allDone    chan struct{}
	doneOnce   sync.Once
	completed  int32
	totalCount int32
}

// newAsyncBatchHandle creates a new batch handle for the given handles.
func newAsyncBatchHandle(handles []*AsyncQueryHandle) *AsyncBatchHandle {
	bh := &AsyncBatchHandle{
		handles:    handles,
		allDone:    make(chan struct{}),
		totalCount: int32(len(handles)),
	}

	// Start a goroutine to track completion
	go bh.trackCompletion()

	return bh
}

// trackCompletion monitors all handles and closes allDone when all complete.
func (bh *AsyncBatchHandle) trackCompletion() {
	for _, h := range bh.handles {
		go func(handle *AsyncQueryHandle) {
			<-handle.done
			if atomic.AddInt32(&bh.completed, 1) == bh.totalCount {
				bh.doneOnce.Do(func() {
					close(bh.allDone)
				})
			}
		}(h)
	}
}

// WaitAll blocks until all queries complete and returns all results.
func (bh *AsyncBatchHandle) WaitAll() ([]string, error) {
	<-bh.allDone

	results := make([]string, len(bh.handles))
	var firstErr error

	for i, h := range bh.handles {
		result, err := h.Wait()
		if err != nil && firstErr == nil {
			firstErr = err
		}
		results[i] = result
	}

	return results, firstErr
}

// Ready returns true if all queries have completed.
func (bh *AsyncBatchHandle) Ready() bool {
	select {
	case <-bh.allDone:
		return true
	default:
		return false
	}
}

// Handles returns the individual query handles.
func (bh *AsyncBatchHandle) Handles() []*AsyncQueryHandle {
	return bh.handles
}

// CompletedCount returns the number of completed queries.
func (bh *AsyncBatchHandle) CompletedCount() int {
	return int(atomic.LoadInt32(&bh.completed))
}

// TotalCount returns the total number of queries in the batch.
func (bh *AsyncBatchHandle) TotalCount() int {
	return int(bh.totalCount)
}
