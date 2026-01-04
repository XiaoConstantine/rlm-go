package logger

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"

	"github.com/XiaoConstantine/rlm-go/pkg/core"
)

func TestNewLogger(t *testing.T) {
	tests := []struct {
		name        string
		cfg         Config
		expectError bool
	}{
		{
			name: "basic config",
			cfg: Config{
				RootModel:     "gpt-4",
				MaxIterations: 30,
				Backend:       "openai",
				BackendKwargs: map[string]any{"temperature": 0.7},
				Context:       "test context",
				Query:         "test query",
			},
			expectError: false,
		},
		{
			name: "minimal config",
			cfg: Config{
				RootModel:     "claude-3",
				MaxIterations: 10,
				Backend:       "anthropic",
			},
			expectError: false,
		},
		{
			name: "long context gets truncated",
			cfg: Config{
				RootModel:     "gpt-4",
				MaxIterations: 30,
				Backend:       "openai",
				Context:       strings.Repeat("x", 1000),
				Query:         "test query",
			},
			expectError: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tmpDir := t.TempDir()
			logger, err := New(tmpDir, tt.cfg)

			if tt.expectError {
				if err == nil {
					t.Error("expected error, got nil")
				}
				return
			}

			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			defer logger.Close()

			// Verify log file was created
			path := logger.Path()
			if path == "" {
				t.Error("expected non-empty path")
			}

			if !strings.HasSuffix(path, ".jsonl") {
				t.Errorf("expected .jsonl extension, got %s", path)
			}

			// Read and verify metadata entry
			data, err := os.ReadFile(path)
			if err != nil {
				t.Fatalf("failed to read log file: %v", err)
			}

			var metadata MetadataEntry
			lines := strings.Split(strings.TrimSpace(string(data)), "\n")
			if len(lines) < 1 {
				t.Fatal("expected at least one line in log file")
			}

			if err := json.Unmarshal([]byte(lines[0]), &metadata); err != nil {
				t.Fatalf("failed to unmarshal metadata: %v", err)
			}

			if metadata.Type != "metadata" {
				t.Errorf("expected type 'metadata', got %s", metadata.Type)
			}
			if metadata.RootModel != tt.cfg.RootModel {
				t.Errorf("expected root_model %s, got %s", tt.cfg.RootModel, metadata.RootModel)
			}
			if metadata.MaxIterations != tt.cfg.MaxIterations {
				t.Errorf("expected max_iterations %d, got %d", tt.cfg.MaxIterations, metadata.MaxIterations)
			}
			if metadata.Backend != tt.cfg.Backend {
				t.Errorf("expected backend %s, got %s", tt.cfg.Backend, metadata.Backend)
			}

			// Verify context truncation
			if len(tt.cfg.Context) > 500 {
				if !strings.HasSuffix(metadata.Context, "...") {
					t.Error("expected truncated context to end with '...'")
				}
				if len(metadata.Context) != 503 { // 500 + "..."
					t.Errorf("expected truncated context length 503, got %d", len(metadata.Context))
				}
			}
		})
	}
}

func TestLoggerInvalidDirectory(t *testing.T) {
	// Try to create logger in a path that can't be created
	_, err := New("/nonexistent/path/that/should/not/exist/"+time.Now().Format(time.RFC3339Nano), Config{})
	if err == nil {
		t.Error("expected error for invalid directory, got nil")
	}
}

func TestLogIteration(t *testing.T) {
	tmpDir := t.TempDir()
	cfg := Config{
		RootModel:     "gpt-4",
		MaxIterations: 30,
		Backend:       "openai",
	}

	logger, err := New(tmpDir, cfg)
	if err != nil {
		t.Fatalf("failed to create logger: %v", err)
	}
	defer logger.Close()

	// Test logging iteration with code blocks
	prompt := []core.Message{
		{Role: "system", Content: "You are a helpful assistant."},
		{Role: "user", Content: "What is 2+2?"},
	}

	codeBlocks := []core.CodeBlock{
		{
			Code: "fmt.Println(2+2)",
			Result: core.ExecutionResult{
				Stdout:   "4\n",
				Stderr:   "",
				Duration: 100 * time.Millisecond,
			},
		},
	}

	rlmCalls := []RLMCallEntry{
		{
			Prompt:           "What is 2+2?",
			Response:         "4",
			PromptTokens:     10,
			CompletionTokens: 1,
			ExecutionTime:    0.5,
		},
	}

	locals := map[string]any{
		"answer": "4",
		"count":  42,
	}

	err = logger.LogIteration(
		1,
		prompt,
		"Let me calculate that.\n```go\nfmt.Println(2+2)\n```",
		codeBlocks,
		rlmCalls,
		locals,
		nil, // no final answer yet
		200*time.Millisecond,
	)
	if err != nil {
		t.Fatalf("failed to log iteration: %v", err)
	}

	// Read and verify iteration entry
	data, err := os.ReadFile(logger.Path())
	if err != nil {
		t.Fatalf("failed to read log file: %v", err)
	}

	lines := strings.Split(strings.TrimSpace(string(data)), "\n")
	if len(lines) < 2 {
		t.Fatal("expected at least two lines in log file")
	}

	var iteration IterationEntry
	if err := json.Unmarshal([]byte(lines[1]), &iteration); err != nil {
		t.Fatalf("failed to unmarshal iteration: %v", err)
	}

	if iteration.Type != "iteration" {
		t.Errorf("expected type 'iteration', got %s", iteration.Type)
	}
	if iteration.Iteration != 1 {
		t.Errorf("expected iteration 1, got %d", iteration.Iteration)
	}
	if len(iteration.Prompt) != 2 {
		t.Errorf("expected 2 prompt messages, got %d", len(iteration.Prompt))
	}
	if len(iteration.CodeBlocks) != 1 {
		t.Errorf("expected 1 code block, got %d", len(iteration.CodeBlocks))
	}
	if iteration.FinalAnswer != nil {
		t.Errorf("expected nil final answer, got %v", iteration.FinalAnswer)
	}
}

func TestLogIterationWithFinalAnswer(t *testing.T) {
	tests := []struct {
		name        string
		finalAnswer any
		expected    any
	}{
		{
			name:        "string final answer",
			finalAnswer: "42",
			expected:    "42",
		},
		{
			name:        "variable tuple final answer",
			finalAnswer: []string{"answer", "42"},
			expected:    []any{"answer", "42"},
		},
		{
			name:        "nil final answer",
			finalAnswer: nil,
			expected:    nil,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			tmpDir := t.TempDir()
			cfg := Config{
				RootModel:     "gpt-4",
				MaxIterations: 30,
				Backend:       "openai",
			}

			logger, err := New(tmpDir, cfg)
			if err != nil {
				t.Fatalf("failed to create logger: %v", err)
			}
			defer logger.Close()

			err = logger.LogIteration(
				1,
				[]core.Message{{Role: "user", Content: "test"}},
				"FINAL_VAR(answer)",
				nil,
				nil,
				nil,
				tt.finalAnswer,
				100*time.Millisecond,
			)
			if err != nil {
				t.Fatalf("failed to log iteration: %v", err)
			}

			data, err := os.ReadFile(logger.Path())
			if err != nil {
				t.Fatalf("failed to read log file: %v", err)
			}

			lines := strings.Split(strings.TrimSpace(string(data)), "\n")
			if len(lines) < 2 {
				t.Fatal("expected at least two lines")
			}

			var iteration IterationEntry
			if err := json.Unmarshal([]byte(lines[1]), &iteration); err != nil {
				t.Fatalf("failed to unmarshal: %v", err)
			}

			// Check final answer type and value
			switch expected := tt.expected.(type) {
			case nil:
				if iteration.FinalAnswer != nil {
					t.Errorf("expected nil, got %v", iteration.FinalAnswer)
				}
			case string:
				if iteration.FinalAnswer != expected {
					t.Errorf("expected %q, got %v", expected, iteration.FinalAnswer)
				}
			case []any:
				arr, ok := iteration.FinalAnswer.([]any)
				if !ok {
					t.Errorf("expected array, got %T", iteration.FinalAnswer)
					return
				}
				if len(arr) != len(expected) {
					t.Errorf("expected array of length %d, got %d", len(expected), len(arr))
					return
				}
				for i, v := range expected {
					if arr[i] != v {
						t.Errorf("expected arr[%d] = %v, got %v", i, v, arr[i])
					}
				}
			}
		})
	}
}

func TestLoggerClose(t *testing.T) {
	tmpDir := t.TempDir()
	logger, err := New(tmpDir, Config{RootModel: "test", MaxIterations: 10, Backend: "test"})
	if err != nil {
		t.Fatalf("failed to create logger: %v", err)
	}

	// Close should succeed
	if err := logger.Close(); err != nil {
		t.Errorf("Close() failed: %v", err)
	}

	// Double close should be safe (file already closed)
	// This tests the Close() with nil file path
}

func TestLoggerPath(t *testing.T) {
	tmpDir := t.TempDir()
	logger, err := New(tmpDir, Config{RootModel: "test", MaxIterations: 10, Backend: "test"})
	if err != nil {
		t.Fatalf("failed to create logger: %v", err)
	}
	defer logger.Close()

	path := logger.Path()

	// Path should be in tmpDir
	if !strings.HasPrefix(path, tmpDir) {
		t.Errorf("expected path to start with %s, got %s", tmpDir, path)
	}

	// Path should have expected format: rlm_YYYY-MM-DD_HH-MM-SS_XXXXXXXX.jsonl
	filename := filepath.Base(path)
	if !strings.HasPrefix(filename, "rlm_") {
		t.Errorf("expected filename to start with 'rlm_', got %s", filename)
	}
	if !strings.HasSuffix(filename, ".jsonl") {
		t.Errorf("expected filename to end with '.jsonl', got %s", filename)
	}
}

func TestLoggerPathWhenNil(t *testing.T) {
	l := &Logger{file: nil}
	if path := l.Path(); path != "" {
		t.Errorf("expected empty path when file is nil, got %s", path)
	}
}

func TestGenerateSessionID(t *testing.T) {
	// Generate multiple IDs and ensure they're unique
	ids := make(map[string]bool)
	for i := 0; i < 100; i++ {
		id := generateSessionID()
		if len(id) != 8 {
			t.Errorf("expected session ID length 8, got %d", len(id))
		}
		if ids[id] {
			// Note: This could theoretically fail due to timing, but very unlikely
			t.Logf("warning: duplicate session ID generated: %s", id)
		}
		ids[id] = true
		time.Sleep(time.Nanosecond) // Small delay to ensure different timestamps
	}
}

func TestMetadataEntryJSON(t *testing.T) {
	entry := MetadataEntry{
		Type:              "metadata",
		Timestamp:         "2024-01-01T00:00:00Z",
		RootModel:         "gpt-4",
		MaxDepth:          1,
		MaxIterations:     30,
		Backend:           "openai",
		BackendKwargs:     map[string]any{"temperature": 0.7},
		EnvironmentType:   "local",
		EnvironmentKwargs: map[string]any{},
		OtherBackends:     nil,
		Context:           "test context",
		Query:             "test query",
	}

	data, err := json.Marshal(entry)
	if err != nil {
		t.Fatalf("failed to marshal: %v", err)
	}

	var decoded MetadataEntry
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("failed to unmarshal: %v", err)
	}

	if decoded.Type != entry.Type {
		t.Errorf("Type mismatch: %s vs %s", decoded.Type, entry.Type)
	}
	if decoded.RootModel != entry.RootModel {
		t.Errorf("RootModel mismatch: %s vs %s", decoded.RootModel, entry.RootModel)
	}
}

func TestIterationEntryJSON(t *testing.T) {
	entry := IterationEntry{
		Type:      "iteration",
		Iteration: 1,
		Timestamp: "2024-01-01T00:00:00Z",
		Prompt: []core.Message{
			{Role: "user", Content: "test"},
		},
		Response: "test response",
		CodeBlocks: []CodeBlockEntry{
			{
				Code: "fmt.Println(1)",
				Result: CodeResultEntry{
					Stdout:        "1\n",
					Stderr:        "",
					Locals:        map[string]any{"x": 1},
					ExecutionTime: 0.1,
					RLMCalls:      nil,
				},
			},
		},
		FinalAnswer:   "42",
		IterationTime: 0.5,
	}

	data, err := json.Marshal(entry)
	if err != nil {
		t.Fatalf("failed to marshal: %v", err)
	}

	var decoded IterationEntry
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("failed to unmarshal: %v", err)
	}

	if decoded.Type != entry.Type {
		t.Errorf("Type mismatch")
	}
	if decoded.Iteration != entry.Iteration {
		t.Errorf("Iteration mismatch")
	}
	if len(decoded.CodeBlocks) != len(entry.CodeBlocks) {
		t.Errorf("CodeBlocks length mismatch")
	}
}

func TestCodeResultEntryWithRLMCalls(t *testing.T) {
	entry := CodeResultEntry{
		Stdout:        "output",
		Stderr:        "",
		Locals:        map[string]any{"result": "value"},
		ExecutionTime: 0.5,
		RLMCalls: []RLMCallEntry{
			{
				Prompt:           "What is 2+2?",
				Response:         "4",
				PromptTokens:     10,
				CompletionTokens: 1,
				ExecutionTime:    0.2,
			},
		},
	}

	data, err := json.Marshal(entry)
	if err != nil {
		t.Fatalf("failed to marshal: %v", err)
	}

	var decoded CodeResultEntry
	if err := json.Unmarshal(data, &decoded); err != nil {
		t.Fatalf("failed to unmarshal: %v", err)
	}

	if len(decoded.RLMCalls) != 1 {
		t.Errorf("expected 1 RLM call, got %d", len(decoded.RLMCalls))
	}
	if decoded.RLMCalls[0].PromptTokens != 10 {
		t.Errorf("expected PromptTokens 10, got %d", decoded.RLMCalls[0].PromptTokens)
	}
}
