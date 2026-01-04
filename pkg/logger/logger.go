// Package logger provides JSONL logging for RLM sessions.
package logger

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"time"

	"github.com/XiaoConstantine/rlm-go/pkg/core"
)

// Logger writes RLM session logs in JSONL format compatible with the Python implementation.
type Logger struct {
	file      *os.File
	startTime time.Time
}

// Config holds logger configuration.
type Config struct {
	RootModel     string
	MaxIterations int
	Backend       string
	BackendKwargs map[string]any
	Context       string // The context payload (truncated for display)
	Query         string // The user query
}

// MetadataEntry represents the first line of a JSONL log file.
type MetadataEntry struct {
	Type              string         `json:"type"`
	Timestamp         string         `json:"timestamp"`
	RootModel         string         `json:"root_model"`
	MaxDepth          int            `json:"max_depth"`
	MaxIterations     int            `json:"max_iterations"`
	Backend           string         `json:"backend"`
	BackendKwargs     map[string]any `json:"backend_kwargs"`
	EnvironmentType   string         `json:"environment_type"`
	EnvironmentKwargs map[string]any `json:"environment_kwargs"`
	OtherBackends     any            `json:"other_backends"`
	Context           string         `json:"context,omitempty"`
	Query             string         `json:"query,omitempty"`
}

// IterationEntry represents a single iteration in the log.
type IterationEntry struct {
	Type          string           `json:"type"`
	Iteration     int              `json:"iteration"`
	Timestamp     string           `json:"timestamp"`
	Prompt        []core.Message   `json:"prompt"`
	Response      string           `json:"response"`
	CodeBlocks    []CodeBlockEntry `json:"code_blocks"`
	FinalAnswer   any              `json:"final_answer"` // string or [varname, value] tuple
	IterationTime float64          `json:"iteration_time"`
}

// CodeBlockEntry represents an executed code block in the log.
type CodeBlockEntry struct {
	Code   string            `json:"code"`
	Result CodeResultEntry   `json:"result"`
}

// CodeResultEntry represents the result of code execution.
type CodeResultEntry struct {
	Stdout        string         `json:"stdout"`
	Stderr        string         `json:"stderr"`
	Locals        map[string]any `json:"locals"`
	ExecutionTime float64        `json:"execution_time"`
	RLMCalls      []RLMCallEntry `json:"rlm_calls"`
}

// RLMCallEntry represents a sub-LLM call made from within REPL code.
// Fields match the visualizer's RLMChatCompletion interface.
type RLMCallEntry struct {
	Prompt           string  `json:"prompt"`
	Response         string  `json:"response"`
	PromptTokens     int     `json:"prompt_tokens"`
	CompletionTokens int     `json:"completion_tokens"`
	ExecutionTime    float64 `json:"execution_time"`
}

// New creates a new Logger and writes the metadata entry.
func New(logDir string, cfg Config) (*Logger, error) {
	if err := os.MkdirAll(logDir, 0755); err != nil {
		return nil, fmt.Errorf("create log directory: %w", err)
	}

	now := time.Now()
	filename := fmt.Sprintf("rlm_%s_%s.jsonl",
		now.Format("2006-01-02_15-04-05"),
		generateSessionID(),
	)
	path := filepath.Join(logDir, filename)

	file, err := os.Create(path)
	if err != nil {
		return nil, fmt.Errorf("create log file: %w", err)
	}

	l := &Logger{
		file:      file,
		startTime: now,
	}

	// Truncate context for display (first 500 chars)
	contextPreview := cfg.Context
	if len(contextPreview) > 500 {
		contextPreview = contextPreview[:500] + "..."
	}

	// Write metadata entry
	metadata := MetadataEntry{
		Type:              "metadata",
		Timestamp:         now.Format(time.RFC3339Nano),
		RootModel:         cfg.RootModel,
		MaxDepth:          1,
		MaxIterations:     cfg.MaxIterations,
		Backend:           cfg.Backend,
		BackendKwargs:     cfg.BackendKwargs,
		EnvironmentType:   "local",
		EnvironmentKwargs: map[string]any{},
		OtherBackends:     nil,
		Context:           contextPreview,
		Query:             cfg.Query,
	}

	if err := l.writeEntry(metadata); err != nil {
		file.Close()
		return nil, fmt.Errorf("write metadata: %w", err)
	}

	return l, nil
}

// LogIteration logs a single RLM iteration.
// finalAnswer can be:
//   - nil: no final answer
//   - string: direct FINAL answer
//   - []string{varname, value}: FINAL_VAR answer as tuple
func (l *Logger) LogIteration(
	iteration int,
	prompt []core.Message,
	response string,
	codeBlocks []core.CodeBlock,
	rlmCalls []RLMCallEntry,
	locals map[string]any,
	finalAnswer any,
	iterationTime time.Duration,
) error {
	blocks := make([]CodeBlockEntry, len(codeBlocks))
	for i, cb := range codeBlocks {
		blocks[i] = CodeBlockEntry{
			Code: cb.Code,
			Result: CodeResultEntry{
				Stdout:        cb.Result.Stdout,
				Stderr:        cb.Result.Stderr,
				Locals:        locals,
				ExecutionTime: cb.Result.Duration.Seconds(),
				RLMCalls:      rlmCalls,
			},
		}
	}

	entry := IterationEntry{
		Type:          "iteration",
		Iteration:     iteration,
		Timestamp:     time.Now().Format(time.RFC3339Nano),
		Prompt:        prompt,
		Response:      response,
		CodeBlocks:    blocks,
		FinalAnswer:   finalAnswer,
		IterationTime: iterationTime.Seconds(),
	}

	return l.writeEntry(entry)
}

// Close closes the log file.
func (l *Logger) Close() error {
	if l.file != nil {
		return l.file.Close()
	}
	return nil
}

// Path returns the path to the log file.
func (l *Logger) Path() string {
	if l.file != nil {
		return l.file.Name()
	}
	return ""
}

func (l *Logger) writeEntry(entry any) error {
	data, err := json.Marshal(entry)
	if err != nil {
		return err
	}
	_, err = l.file.Write(append(data, '\n'))
	return err
}

func generateSessionID() string {
	return fmt.Sprintf("%08x", time.Now().UnixNano()&0xFFFFFFFF)
}
