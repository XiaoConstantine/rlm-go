package sandbox

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"reflect"
	"runtime"
	"strconv"
	"sync"
	"sync/atomic"
	"time"

	"github.com/XiaoConstantine/rlm-go/pkg/contextindex"
	"github.com/XiaoConstantine/rlm-go/pkg/core"
	"github.com/XiaoConstantine/rlm-go/pkg/interpreter"
	"github.com/traefik/yaegi/interp"
	"github.com/traefik/yaegi/stdlib"
)

// LocalExecutor provides in-process code execution using Yaegi.
// This is the fastest option but has the least isolation.
type LocalExecutor struct {
	interp   *interp.Interpreter
	stdout   *localOutput
	stderr   *localOutput
	client   LLMClient
	ctx      context.Context
	mu       sync.Mutex
	llmCalls []LLMCall
	finalMu  sync.RWMutex
	finalSet bool
	finalVal string
	finalTok string
	context  string
	runSeq   uint64
	config   Config
}

type localOutput struct {
	mu          sync.Mutex
	buf         bytes.Buffer
	activeRun   uint64
	rootGID     uint64
	allowed     map[uint64]struct{}
	preexisting map[uint64]struct{}
}

// localOutput favors dropping ambiguous late goroutine writes over leaking
// stale output into a later execution. Awaited child output is captured only
// when the child was created after the current run began.
func (o *localOutput) Write(p []byte) (int, error) {
	var stack [4096]byte
	n := runtime.Stack(stack[:], false)
	gid, ok := goroutineID(stack[:n])
	parentID, hasParent := goroutineParentID(stack[:n])
	yaegiStack := bytes.Contains(stack[:n], []byte("github.com/traefik/yaegi/interp."))
	var parents map[uint64]uint64
	if yaegiStack {
		parents = goroutineParents(allGoroutineStacks())
	}

	o.mu.Lock()
	defer o.mu.Unlock()
	if !ok || !o.allowWriteLocked(gid, parentID, hasParent, yaegiStack, parents) {
		return len(p), nil
	}
	return o.buf.Write(p)
}

func (o *localOutput) AllowCurrent(runID uint64) {
	var stack [256]byte
	n := runtime.Stack(stack[:], false)
	gid, ok := goroutineID(stack[:n])
	if !ok {
		return
	}

	o.mu.Lock()
	defer o.mu.Unlock()
	if runID == 0 || runID != o.activeRun {
		return
	}
	if o.rootGID == 0 {
		o.rootGID = gid
	}
	o.allowed[gid] = struct{}{}
}

func (o *localOutput) allowWrite(gid, parentID uint64, hasParent, yaegiStack bool) bool {
	o.mu.Lock()
	defer o.mu.Unlock()
	return o.allowWriteLocked(gid, parentID, hasParent, yaegiStack, nil)
}

func (o *localOutput) allowWriteLocked(gid, parentID uint64, hasParent, yaegiStack bool, parents map[uint64]uint64) bool {
	if o.activeRun == 0 || gid == 0 {
		return false
	}
	if _, allowed := o.allowed[gid]; allowed {
		return true
	}
	if _, existed := o.preexisting[gid]; existed {
		return false
	}
	_, parentAllowed := o.allowed[parentID]
	if hasParent && parentAllowed {
		o.allowed[gid] = struct{}{}
		return true
	}
	if yaegiStack && o.rootGID != 0 && goroutineDescendsFrom(gid, o.rootGID, parents, o.preexisting) {
		o.allowed[gid] = struct{}{}
		return true
	}
	return false
}

func (o *localOutput) WriteForRun(runID uint64, p []byte) (int, error) {
	o.mu.Lock()
	defer o.mu.Unlock()
	if runID == 0 || runID != o.activeRun {
		return len(p), nil
	}
	return o.buf.Write(p)
}

func (o *localOutput) Begin(runID uint64, snapshots ...map[uint64]struct{}) {
	var preexisting map[uint64]struct{}
	if len(snapshots) > 0 {
		preexisting = snapshots[0]
	} else {
		preexisting = goroutineIDs(allGoroutineStacks())
	}

	o.mu.Lock()
	defer o.mu.Unlock()
	o.buf.Reset()
	o.activeRun = runID
	o.rootGID = 0
	o.allowed = make(map[uint64]struct{})
	o.preexisting = preexisting
}

func (o *localOutput) End(runID uint64) string {
	o.mu.Lock()
	defer o.mu.Unlock()
	if runID != 0 && runID == o.activeRun {
		o.activeRun = 0
		o.rootGID = 0
		o.allowed = nil
		o.preexisting = nil
	}
	return o.buf.String()
}

func (o *localOutput) Clear() {
	o.mu.Lock()
	defer o.mu.Unlock()
	o.activeRun = 0
	o.rootGID = 0
	o.allowed = nil
	o.preexisting = nil
	o.buf.Reset()
}

func (o *localOutput) Reset() {
	o.Clear()
}

func (o *localOutput) String() string {
	o.mu.Lock()
	defer o.mu.Unlock()
	return o.buf.String()
}

func goroutineID(stack []byte) (uint64, bool) {
	const prefix = "goroutine "
	if !bytes.HasPrefix(stack, []byte(prefix)) {
		return 0, false
	}
	stack = stack[len(prefix):]
	i := bytes.IndexByte(stack, ' ')
	if i < 0 {
		return 0, false
	}
	id, err := strconv.ParseUint(string(stack[:i]), 10, 64)
	if err != nil {
		return 0, false
	}
	return id, true
}

func goroutineParentID(stack []byte) (uint64, bool) {
	const marker = " in goroutine "
	i := bytes.LastIndex(stack, []byte(marker))
	if i < 0 {
		return 0, false
	}
	stack = stack[i+len(marker):]
	end := bytes.IndexByte(stack, '\n')
	if end >= 0 {
		stack = stack[:end]
	}
	id, err := strconv.ParseUint(string(stack), 10, 64)
	if err != nil {
		return 0, false
	}
	return id, true
}

func goroutineParents(stacks []byte) map[uint64]uint64 {
	parents := make(map[uint64]uint64)
	for _, chunk := range goroutineStackChunks(stacks) {
		gid, ok := goroutineID(chunk)
		if !ok {
			continue
		}
		parentID, ok := goroutineParentID(chunk)
		if ok {
			parents[gid] = parentID
		}
	}
	return parents
}

func goroutineIDs(stacks []byte) map[uint64]struct{} {
	ids := make(map[uint64]struct{})
	for _, chunk := range goroutineStackChunks(stacks) {
		gid, ok := goroutineID(chunk)
		if ok {
			ids[gid] = struct{}{}
		}
	}
	return ids
}

func goroutineStackChunks(stacks []byte) [][]byte {
	rawChunks := bytes.Split(stacks, []byte("\ngoroutine "))
	chunks := make([][]byte, 0, len(rawChunks))
	for _, chunk := range rawChunks {
		if len(chunk) == 0 {
			continue
		}
		if !bytes.HasPrefix(chunk, []byte("goroutine ")) {
			chunk = append([]byte("goroutine "), chunk...)
		}
		chunks = append(chunks, chunk)
	}
	return chunks
}

func allGoroutineStacks() []byte {
	size := 1 << 20
	for {
		buf := make([]byte, size)
		n := runtime.Stack(buf, true)
		if n < len(buf) {
			return buf[:n]
		}
		size *= 2
	}
}

func newYaegiGoroutines(preexisting map[uint64]struct{}) []uint64 {
	var ids []uint64
	for _, chunk := range goroutineStackChunks(allGoroutineStacks()) {
		gid, ok := goroutineID(chunk)
		if !ok {
			continue
		}
		if _, existed := preexisting[gid]; existed {
			continue
		}
		if bytes.Contains(chunk, []byte("github.com/traefik/yaegi/")) {
			ids = append(ids, gid)
		}
	}
	return ids
}

func goroutineDescendsFrom(gid, root uint64, parents map[uint64]uint64, preexisting map[uint64]struct{}) bool {
	for i := 0; i < 64; i++ {
		if gid == root {
			return true
		}
		if _, existed := preexisting[gid]; existed {
			return false
		}
		parentID, ok := parents[gid]
		if !ok || parentID == 0 || parentID == gid {
			return false
		}
		gid = parentID
	}
	return false
}

// NewLocalExecutor creates a new local executor using Yaegi.
func NewLocalExecutor(client LLMClient, cfg Config) (*LocalExecutor, error) {
	stdout := new(localOutput)
	stderr := new(localOutput)

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
		finalTok: newFinalToken(),
		config:   cfg,
	}

	// Inject RLM functions
	if err := exec.injectBuiltins(); err != nil {
		return nil, fmt.Errorf("failed to inject builtins: %w", err)
	}

	return exec, nil
}

// useBuiltins registers Query and QueryBatched functions in the interpreter.
func (e *LocalExecutor) useBuiltins(token string, runCtx context.Context, stdout *localOutput, runID uint64) error {
	symbols := interp.Exports{
		"rlm/rlm": {
			"QueryRaw":              reflect.ValueOf(func(prompt string) string { return e.llmQueryRaw(token, runCtx, prompt) }),
			"QueryBatchedRaw":       reflect.ValueOf(func(prompts []string) []string { return e.llmQueryBatchedRaw(token, runCtx, prompts) }),
			"FindRelevantInContext": reflect.ValueOf(contextindex.FindRelevant),
			"GetChunkInContext":     reflect.ValueOf(contextindex.GetChunk),
			"GetContextInContext":   reflect.ValueOf(contextindex.GetContext),
			"ChunkCountInContext":   reflect.ValueOf(contextindex.ChunkCount),
			"LineCountInContext":    reflect.ValueOf(contextindex.LineCount),
			"MaxFullContextQueryChars": reflect.ValueOf(func() int {
				return e.config.MaxFullContextQueryChars
			}),
			"FINAL":     reflect.ValueOf(func(value any) string { return e.finalAnswer(token, stdout, runID, value) }),
			"FINAL_VAR": reflect.ValueOf(func(value any) string { return e.finalAnswer(token, stdout, runID, value) }),
		},
	}

	if err := e.interp.Use(symbols); err != nil {
		return fmt.Errorf("failed to inject rlm symbols: %w", err)
	}

	return nil
}

const localRLMSetupCode = `
var context = ""
var Query func(string) string
var QueryWith func(string, string) string
var QueryBatched func([]string) []string

func buildPromptWithProvidedContext(contextStr, prompt string) string {
	if contextStr == "" {
		return prompt
	}
	return fmt.Sprintf("Context data:\n%s\n\nTask: %s\n\nIMPORTANT: Provide a direct, concise answer. Do not explain your reasoning unless specifically asked.", contextStr, prompt)
}

func FindRelevant(query string, topK int) []string {
	return FindRelevantInContext(context, query, topK)
}

func GetChunk(id int) string {
	return GetChunkInContext(context, id)
}

func GetContext(startLine, endLine int) string {
	return GetContextInContext(context, startLine, endLine)
}

func ChunkCount() int {
	return ChunkCountInContext(context)
}

func LineCount() int {
	return LineCountInContext(context)
}

func fullContextQueryBlocked(name string) string {
	maxChars := MaxFullContextQueryChars()
	if maxChars <= 0 || len(context) <= maxChars {
		return ""
	}
	return fmt.Sprintf("%s would prepend the full context (%d chars), exceeding the limit of %d chars; use QueryWith(contextSlice, prompt) or QueryRaw(prompt)", name, len(context), maxChars)
}
	`

const localRLMWrapperCode = `
Query = func(prompt string) string {
	if err := fullContextQueryBlocked("Query"); err != "" {
		return "Error: " + err
	}
	return QueryRaw(buildPromptWithProvidedContext(context, prompt))
}

QueryWith = func(contextSlice, prompt string) string {
	return QueryRaw(buildPromptWithProvidedContext(contextSlice, prompt))
}

QueryBatched = func(prompts []string) []string {
	if err := fullContextQueryBlocked("QueryBatched"); err != "" {
		results := make([]string, len(prompts))
		for i := range results {
			results[i] = "Error: " + err
		}
		return results
	}
	fullPrompts := make([]string, len(prompts))
	for i, prompt := range prompts {
		fullPrompts[i] = buildPromptWithProvidedContext(context, prompt)
	}
	return QueryBatchedRaw(fullPrompts)
}
`

// injectBuiltins registers Query and QueryBatched functions in the interpreter.
func (e *LocalExecutor) injectBuiltins() error {
	if err := e.useBuiltins(e.finalTok, context.Background(), e.stdout, 0); err != nil {
		return err
	}

	// Use the shared setup code for basic imports
	if err := interpreter.RunSetup(e.interp, interpreter.SetupCode); err != nil {
		return err
	}
	if _, err := e.interp.Eval(localRLMSetupCode); err != nil {
		return err
	}
	return e.refreshLocalRLMWrappers()
}

func (e *LocalExecutor) refreshBuiltins(token string, runCtx context.Context, runID uint64) error {
	if err := e.useBuiltins(token, runCtx, e.stdout, runID); err != nil {
		return err
	}
	if _, err := e.interp.Eval(`import . "rlm/rlm"`); err != nil {
		return fmt.Errorf("failed to refresh rlm imports: %w", err)
	}
	if err := e.refreshLocalRLMWrappers(); err != nil {
		return fmt.Errorf("failed to refresh rlm wrappers: %w", err)
	}
	return nil
}

func (e *LocalExecutor) refreshLocalRLMWrappers() error {
	_, err := e.interp.Eval(localRLMWrapperCode)
	return err
}

func (e *LocalExecutor) llmQueryRaw(token string, ctx context.Context, prompt string) string {
	return e.query(token, ctx, prompt, prompt)
}

func (e *LocalExecutor) query(token string, ctx context.Context, fullPrompt, recordedPrompt string) string {
	if !e.tokenActive(token) {
		return "Error: execution expired"
	}
	start := time.Now()
	if ctx == nil {
		ctx = context.Background()
	}
	result, err := e.client.Query(ctx, fullPrompt)
	duration := time.Since(start).Seconds()

	response := result.Response
	if err != nil {
		response = fmt.Sprintf("Error: %v", err)
	}

	if !e.tokenActive(token) {
		return response
	}
	e.mu.Lock()
	e.llmCalls = append(e.llmCalls, LLMCall{
		Prompt:           recordedPrompt,
		Response:         response,
		Duration:         duration,
		PromptTokens:     result.PromptTokens,
		CompletionTokens: result.CompletionTokens,
	})
	e.mu.Unlock()

	return response
}

func (e *LocalExecutor) llmQueryBatchedRaw(token string, ctx context.Context, prompts []string) []string {
	if !e.tokenActive(token) {
		return queryBatchError(prompts, fmt.Errorf("execution expired"))
	}
	start := time.Now()
	if ctx == nil {
		ctx = context.Background()
	}
	results, err := e.client.QueryBatched(ctx, prompts)
	duration := time.Since(start).Seconds()
	if err != nil {
		return e.recordBatchError(token, prompts, err, duration)
	}
	responses := make([]string, len(results))
	if !e.tokenActive(token) {
		for i, result := range results {
			responses[i] = result.Response
		}
		return responses
	}
	e.mu.Lock()
	for i, p := range prompts {
		responses[i] = results[i].Response
		e.llmCalls = append(e.llmCalls, LLMCall{
			Prompt:           p,
			Response:         results[i].Response,
			Duration:         durationPerPrompt(duration, len(prompts)),
			PromptTokens:     results[i].PromptTokens,
			CompletionTokens: results[i].CompletionTokens,
		})
	}
	e.mu.Unlock()
	return responses
}

func (e *LocalExecutor) finalAnswer(token string, stdout *localOutput, runID uint64, value any) string {
	finalValue := fmt.Sprint(value)
	if e.setFinal(token, finalValue) {
		_, _ = stdout.WriteForRun(runID, []byte(formatFinalMarker(token, finalValue)))
	}
	return finalValue
}

func (e *LocalExecutor) tokenActive(token string) bool {
	e.finalMu.RLock()
	defer e.finalMu.RUnlock()
	return token != "" && token == e.finalTok
}

func (e *LocalExecutor) setFinal(token, value string) bool {
	e.finalMu.Lock()
	defer e.finalMu.Unlock()
	if token == "" || token != e.finalTok || e.finalSet {
		return false
	}
	e.finalSet = true
	e.finalVal = value
	return true
}

func (e *LocalExecutor) recordBatchError(token string, prompts []string, err error, duration float64) []string {
	results := queryBatchError(prompts, err)
	if !e.tokenActive(token) {
		return results
	}
	e.mu.Lock()
	for i, prompt := range prompts {
		e.llmCalls = append(e.llmCalls, LLMCall{
			Prompt:   prompt,
			Response: results[i],
			Duration: durationPerPrompt(duration, len(prompts)),
		})
	}
	e.mu.Unlock()
	return results
}

func queryBatchError(prompts []string, err error) []string {
	results := make([]string, len(prompts))
	for i := range results {
		results[i] = fmt.Sprintf("Error: %v", err)
	}
	return results
}

func durationPerPrompt(duration float64, promptCount int) float64 {
	if promptCount <= 0 {
		return duration
	}
	return duration / float64(promptCount)
}

// HasFinal reports whether executed code called FINAL or FINAL_VAR.
func (e *LocalExecutor) HasFinal() bool {
	e.finalMu.RLock()
	defer e.finalMu.RUnlock()
	return e.finalSet
}

// Final returns the value supplied to FINAL or FINAL_VAR.
func (e *LocalExecutor) Final() (string, bool) {
	e.finalMu.RLock()
	defer e.finalMu.RUnlock()
	if !e.finalSet {
		return "", false
	}
	return e.finalVal, true
}

// ClearFinal clears any previous FINAL/FINAL_VAR signal.
func (e *LocalExecutor) ClearFinal() {
	e.finalMu.Lock()
	defer e.finalMu.Unlock()
	e.finalSet = false
	e.finalVal = ""
}

func (e *LocalExecutor) rotateFinalToken() string {
	e.finalMu.Lock()
	defer e.finalMu.Unlock()
	e.finalTok = newFinalToken()
	e.finalSet = false
	e.finalVal = ""
	return e.finalTok
}

func (e *LocalExecutor) expireFinalToken(token string) {
	e.finalMu.Lock()
	defer e.finalMu.Unlock()
	if token != "" && token == e.finalTok {
		e.finalTok = newFinalToken()
	}
}

func (e *LocalExecutor) resetAbandonedInterpreter() error {
	stdout := new(localOutput)
	stderr := new(localOutput)
	i := interp.New(interp.Options{
		Stdout: stdout,
		Stderr: stderr,
	})

	if err := i.Use(stdlib.Symbols); err != nil {
		return fmt.Errorf("failed to load stdlib: %w", err)
	}

	e.mu.Lock()
	e.interp = i
	e.stdout = stdout
	e.stderr = stderr
	e.finalMu.Lock()
	e.finalTok = newFinalToken()
	e.finalMu.Unlock()
	contextStr := e.context
	err := e.injectBuiltins()
	if err == nil && contextStr != "" {
		_, err = e.interp.Eval(`context = ` + strconv.Quote(contextStr))
	}
	e.mu.Unlock()
	return err
}

func (e *LocalExecutor) evalWithContext(ctx context.Context, code string, runID uint64) (error, bool, bool) {
	if ctx == nil {
		ctx = context.Background()
	}
	if err := ctx.Err(); err != nil {
		return err, false, false
	}
	e.stdout.AllowCurrent(runID)
	e.stderr.AllowCurrent(runID)
	_, err := e.interp.EvalWithContext(ctx, code)
	if ctxErr := ctx.Err(); ctxErr != nil && errors.Is(err, ctxErr) {
		return err, true, false
	}
	return err, true, true
}

// Execute runs Go code and returns the result.
func (e *LocalExecutor) Execute(ctx context.Context, code string) (*core.ExecutionResult, error) {
	start := time.Now()
	timeout := e.config.Timeout
	if timeout == 0 {
		timeout = 60 * time.Second
	}

	evalCtx, cancel := context.WithTimeout(ctx, timeout)
	var sandboxTimedOut atomic.Bool
	timeoutTimer := time.AfterFunc(timeout, func() {
		sandboxTimedOut.Store(true)
	})
	defer func() {
		timeoutTimer.Stop()
		cancel()
	}()

	token := e.rotateFinalToken()
	e.mu.Lock()
	e.runSeq++
	runID := e.runSeq
	if runID == 0 {
		e.runSeq++
		runID = e.runSeq
	}
	preexistingGoroutines := goroutineIDs(allGoroutineStacks())
	e.ctx = evalCtx
	e.stdout.Begin(runID, preexistingGoroutines)
	e.stderr.Begin(runID, preexistingGoroutines)
	refreshErr := e.refreshBuiltins(token, evalCtx, runID)
	e.mu.Unlock()
	if refreshErr != nil {
		e.expireFinalToken(token)
		e.mu.Lock()
		e.stdout.Clear()
		e.stderr.Clear()
		e.ctx = context.Background()
		e.mu.Unlock()
		return &core.ExecutionResult{
			Stderr:   refreshErr.Error(),
			Duration: time.Since(start),
		}, nil
	}

	evalErr, evalStarted, evalCompleted := e.evalWithContext(evalCtx, code, runID)
	timeoutTimer.Stop()
	timedOut := sandboxTimedOut.Load()
	if !evalCompleted && evalCtx.Err() != nil {
		evalErr = evalCtx.Err()
	}
	escapedGoroutines := newYaegiGoroutines(preexistingGoroutines)
	cancel()
	e.expireFinalToken(token)

	e.mu.Lock()
	stdout := e.stdout.End(runID)
	result := &core.ExecutionResult{
		Stdout:   stripFinalMarkers(stdout, token),
		Stderr:   e.stderr.End(runID),
		Duration: time.Since(start),
	}
	e.ctx = context.Background()
	e.mu.Unlock()

	if evalErr != nil {
		if result.Stderr != "" {
			result.Stderr += "\n"
		}
		result.Stderr += evalErr.Error()
	}
	if !evalCompleted && ctx.Err() != nil && !timedOut {
		e.ClearFinal()
		if result.Stderr != "" {
			result.Stderr += "\n"
		}
		result.Stderr += "execution cancelled"
		if evalStarted {
			if resetErr := e.resetAbandonedInterpreter(); resetErr != nil {
				result.Stderr += "\n" + resetErr.Error()
			}
		}
		return result, ctx.Err()
	}
	if timedOut || (!evalCompleted && errors.Is(evalErr, context.DeadlineExceeded)) {
		e.ClearFinal()
		if result.Stderr != "" {
			result.Stderr += "\n"
		}
		result.Stderr += "execution timeout exceeded"
		if resetErr := e.resetAbandonedInterpreter(); resetErr != nil {
			result.Stderr += "\n" + resetErr.Error()
		}
		return result, nil
	}
	if len(escapedGoroutines) > 0 {
		if resetErr := e.resetAbandonedInterpreter(); resetErr != nil {
			if result.Stderr != "" {
				result.Stderr += "\n"
			}
			result.Stderr += resetErr.Error()
		}
	}

	return result, nil
}

// LoadContext injects the context payload into the interpreter.
func (e *LocalExecutor) LoadContext(payload any) error {
	e.mu.Lock()
	defer e.mu.Unlock()
	if payload == nil {
		e.context = ""
		_, err := e.interp.Eval(`context = ""`)
		return err
	}

	switch v := payload.(type) {
	case string:
		e.context = v
		_, err := e.interp.Eval(`context = ` + strconv.Quote(v))
		return err

	case map[string]any:
		return e.loadStructuredContext(v)

	case []any:
		return e.loadStructuredContext(v)

	default:
		jsonBytes, err := json.Marshal(v)
		if err != nil {
			return fmt.Errorf("unsupported context type %T: %w", v, err)
		}
		e.context = string(jsonBytes)
		_, err = e.interp.Eval(`context = ` + strconv.Quote(string(jsonBytes)))
		return err
	}
}

// loadStructuredContext handles map and slice context types.
// It marshals the data to JSON and stores it as a string, which can be parsed by user code.
func (e *LocalExecutor) loadStructuredContext(v any) error {
	jsonBytes, err := json.Marshal(v)
	if err != nil {
		return fmt.Errorf("marshal context: %w", err)
	}

	// Store as JSON string - user code can parse if needed
	e.context = string(jsonBytes)
	_, err = e.interp.Eval(`context = ` + strconv.Quote(string(jsonBytes)))
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

	for _, name := range core.CommonVarNames {
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
		idx := contextindex.New(ctx)
		chunks := idx.ChunkCount()
		lines := idx.LineCount()
		return fmt.Sprintf("type=string, len=%d, lines=%d, chunks=%d", len(ctx), lines, chunks)
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
	e.context = ""
	e.rotateFinalToken()

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
	e.rotateFinalToken()

	return nil
}

// Backend returns the backend type.
func (e *LocalExecutor) Backend() Backend {
	return BackendLocal
}

// FormatExecutionResult formats an execution result for display.
// This is a convenience wrapper around core.FormatExecutionResult.
func FormatExecutionResult(result *core.ExecutionResult) string {
	return core.FormatExecutionResult(result)
}
