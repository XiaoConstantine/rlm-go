// Package main provides a benchmark tool for comparing RLM vs baseline LLM.
package main

import (
	"bytes"
	"context"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"net/http"
	"os"
	"regexp"
	"strings"
	"sync"
	"time"

	"github.com/XiaoConstantine/rlm-go/pkg/core"
	"github.com/XiaoConstantine/rlm-go/pkg/logger"
	"github.com/XiaoConstantine/rlm-go/pkg/repl"
	"github.com/XiaoConstantine/rlm-go/pkg/rlm"
)

// Task represents a benchmark task.
type Task struct {
	TaskID   string `json:"task_id"`
	Context  string `json:"context"`
	Question string `json:"question"`
	Answer   string `json:"answer"`
}

// BenchmarkResult represents the result of a single benchmark run.
type BenchmarkResult struct {
	TaskID        string  `json:"task_id"`
	UseRLM        bool    `json:"use_rlm"`
	Expected      string  `json:"expected"`
	Got           string  `json:"got"`
	IsCorrect     bool    `json:"is_correct"`
	ExecutionTime float64 `json:"execution_time"`
	InputTokens   int     `json:"input_tokens"`
	OutputTokens  int     `json:"output_tokens"`
	Error         string  `json:"error,omitempty"`
}

// BenchmarkSummary provides aggregate statistics.
type BenchmarkSummary struct {
	TotalTasks    int     `json:"total_tasks"`
	CorrectTasks  int     `json:"correct_tasks"`
	Accuracy      float64 `json:"accuracy"`
	TotalTime     float64 `json:"total_time"`
	AvgTimePerTask float64 `json:"avg_time_per_task"`
	TotalInputTokens  int `json:"total_input_tokens"`
	TotalOutputTokens int `json:"total_output_tokens"`
}

// AnthropicClient implements both rlm.LLMClient and repl.LLMClient.
type AnthropicClient struct {
	apiKey     string
	model      string
	maxTokens  int
	httpClient *http.Client

	// Track usage for baseline calls
	mu              sync.Mutex
	lastInputTokens  int
	lastOutputTokens int
}

func NewAnthropicClient(apiKey, model string) *AnthropicClient {
	return &AnthropicClient{
		apiKey:    apiKey,
		model:     model,
		maxTokens: 4096,
		httpClient: &http.Client{
			Timeout: 180 * time.Second,
			Transport: &http.Transport{
				MaxIdleConns:        100,
				MaxIdleConnsPerHost: 10,
				MaxConnsPerHost:     10,
				IdleConnTimeout:     90 * time.Second,
			},
		},
	}
}

type anthropicRequest struct {
	Model     string             `json:"model"`
	MaxTokens int                `json:"max_tokens"`
	Messages  []anthropicMessage `json:"messages"`
	System    string             `json:"system,omitempty"`
}

type anthropicMessage struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

type anthropicResponse struct {
	Content []struct {
		Type string `json:"type"`
		Text string `json:"text"`
	} `json:"content"`
	Usage struct {
		InputTokens  int `json:"input_tokens"`
		OutputTokens int `json:"output_tokens"`
	} `json:"usage"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error,omitempty"`
}

// Complete implements rlm.LLMClient.
func (c *AnthropicClient) Complete(ctx context.Context, messages []core.Message) (core.LLMResponse, error) {
	var systemPrompt string
	var apiMessages []anthropicMessage

	for _, msg := range messages {
		if msg.Role == "system" {
			systemPrompt = msg.Content
		} else {
			apiMessages = append(apiMessages, anthropicMessage{
				Role:    msg.Role,
				Content: msg.Content,
			})
		}
	}

	reqBody := anthropicRequest{
		Model:     c.model,
		MaxTokens: c.maxTokens,
		Messages:  apiMessages,
		System:    systemPrompt,
	}

	text, inputTokens, outputTokens, err := c.doRequest(ctx, reqBody)
	if err != nil {
		return core.LLMResponse{}, err
	}

	c.mu.Lock()
	c.lastInputTokens = inputTokens
	c.lastOutputTokens = outputTokens
	c.mu.Unlock()

	return core.LLMResponse{
		Content:          text,
		PromptTokens:     inputTokens,
		CompletionTokens: outputTokens,
	}, nil
}

// Query implements repl.LLMClient.
func (c *AnthropicClient) Query(ctx context.Context, prompt string) (repl.QueryResponse, error) {
	reqBody := anthropicRequest{
		Model:     c.model,
		MaxTokens: c.maxTokens,
		Messages: []anthropicMessage{
			{Role: "user", Content: prompt},
		},
	}

	text, inputTokens, outputTokens, err := c.doRequest(ctx, reqBody)
	return repl.QueryResponse{
		Response:         text,
		PromptTokens:     inputTokens,
		CompletionTokens: outputTokens,
	}, err
}

// QueryBatched implements repl.LLMClient.
func (c *AnthropicClient) QueryBatched(ctx context.Context, prompts []string) ([]repl.QueryResponse, error) {
	results := make([]repl.QueryResponse, len(prompts))
	var wg sync.WaitGroup

	for i, prompt := range prompts {
		wg.Add(1)
		go func(idx int, p string) {
			defer wg.Done()
			result, err := c.Query(ctx, p)
			if err != nil {
				results[idx] = repl.QueryResponse{Response: fmt.Sprintf("Error: %v", err)}
			} else {
				results[idx] = result
			}
		}(i, prompt)
	}

	wg.Wait()
	return results, nil
}

// GetLastUsage returns the last recorded token usage.
func (c *AnthropicClient) GetLastUsage() (int, int) {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.lastInputTokens, c.lastOutputTokens
}

func (c *AnthropicClient) doRequest(ctx context.Context, reqBody anthropicRequest) (string, int, int, error) {
	jsonBody, err := json.Marshal(reqBody)
	if err != nil {
		return "", 0, 0, fmt.Errorf("marshal request: %w", err)
	}

	req, err := http.NewRequestWithContext(ctx, "POST", "https://api.anthropic.com/v1/messages", bytes.NewReader(jsonBody))
	if err != nil {
		return "", 0, 0, fmt.Errorf("create request: %w", err)
	}

	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", c.apiKey)
	req.Header.Set("anthropic-version", "2023-06-01")

	resp, err := c.httpClient.Do(req)
	if err != nil {
		return "", 0, 0, fmt.Errorf("http request: %w", err)
	}
	defer func() { _ = resp.Body.Close() }()

	body, err := io.ReadAll(resp.Body)
	if err != nil {
		return "", 0, 0, fmt.Errorf("read response: %w", err)
	}

	if resp.StatusCode != http.StatusOK {
		return "", 0, 0, fmt.Errorf("api error (status %d): %s", resp.StatusCode, string(body))
	}

	var apiResp anthropicResponse
	if err := json.Unmarshal(body, &apiResp); err != nil {
		return "", 0, 0, fmt.Errorf("unmarshal response: %w", err)
	}

	if apiResp.Error != nil {
		return "", 0, 0, fmt.Errorf("api error: %s", apiResp.Error.Message)
	}

	var texts []string
	for _, block := range apiResp.Content {
		if block.Type == "text" {
			texts = append(texts, block.Text)
		}
	}

	return strings.Join(texts, ""), apiResp.Usage.InputTokens, apiResp.Usage.OutputTokens, nil
}

// checkAnswer determines if the model's answer matches the expected answer.
// Uses the same logic as Python's check_answer for consistency.
func checkAnswer(expected, actual string) bool {
	// Normalize expected - handle list notation like "['incorrect']"
	expectedNorm := strings.ToLower(strings.TrimSpace(expected))
	if strings.HasPrefix(expectedNorm, "[") && strings.HasSuffix(expectedNorm, "]") {
		inner := strings.TrimSpace(expectedNorm[1 : len(expectedNorm)-1])
		if (strings.HasPrefix(inner, "'") && strings.HasSuffix(inner, "'")) ||
			(strings.HasPrefix(inner, "\"") && strings.HasSuffix(inner, "\"")) {
			inner = inner[1 : len(inner)-1]
		}
		expectedNorm = inner
	}

	actualNorm := strings.ToLower(strings.TrimSpace(actual))
	responseLen := len(actualNorm)

	// Exact match
	if expectedNorm == actualNorm {
		return true
	}

	// For SHORT responses (< 50 chars), allow flexible matching
	if responseLen < 50 {
		// Word boundary match
		pattern := `(?:^|[\s'":=-])` + regexp.QuoteMeta(expectedNorm) + `(?:$|[\s'".,;:=-])`
		if matched, _ := regexp.MatchString(pattern, actualNorm); matched {
			return true
		}

		// Numeric match
		if isNumeric(expectedNorm) {
			cleaned := regexp.MustCompile(`[^\d]`).ReplaceAllString(actualNorm, "")
			if cleaned == expectedNorm {
				return true
			}
		}

		return false
	}

	// For LONG responses (>= 50 chars), only check the LAST LINE
	lines := strings.Split(strings.TrimSpace(actualNorm), "\n")
	lastLine := ""
	if len(lines) > 0 {
		lastLine = strings.TrimSpace(lines[len(lines)-1])
	}

	// Match structured formats: "Label: X", "Answer: X", "User: X"
	structuredPattern := `^\s*(?:the\s+)?(?:answer|label|result|user)\s*(?:is)?[:=]\s*["']?([^"'\n,]+)["']?\s*$`
	re := regexp.MustCompile(structuredPattern)
	if match := re.FindStringSubmatch(lastLine); len(match) > 1 {
		extracted := strings.Trim(strings.TrimSpace(match[1]), ".,;:")
		if expectedNorm == extracted {
			return true
		}
	}

	// If last line is short, check if it equals the answer
	if len(lastLine) < 30 {
		cleaned := strings.Trim(lastLine, ".,;:\"'")
		if expectedNorm == cleaned {
			return true
		}
	}

	// Numeric in structured format
	if isNumeric(expectedNorm) {
		numPattern := `^\s*(?:answer|result|user)?[:=]?\s*(\d+)\s*$`
		numRe := regexp.MustCompile(numPattern)
		if match := numRe.FindStringSubmatch(lastLine); len(match) > 1 && match[1] == expectedNorm {
			return true
		}
	}

	return false
}

func isNumeric(s string) bool {
	for _, c := range s {
		if c < '0' || c > '9' {
			return false
		}
	}
	return len(s) > 0
}

// buildPrompt creates the prompt for baseline LLM calls.
func buildPrompt(task Task) string {
	return fmt.Sprintf(`Read the following text carefully and answer the question that follows.

TEXT:
%s

QUESTION: %s

Please provide a concise answer based only on the information in the text above. If the answer is a code, number, password, or specific value, just provide that value without additional explanation.`, task.Context, task.Question)
}

// runBaseline runs a task with direct LLM call.
func runBaseline(ctx context.Context, task Task, client *AnthropicClient) BenchmarkResult {
	prompt := buildPrompt(task)

	start := time.Now()
	messages := []core.Message{
		{Role: "user", Content: prompt},
	}

	llmResp, err := client.Complete(ctx, messages)
	duration := time.Since(start)

	if err != nil {
		return BenchmarkResult{
			TaskID:        task.TaskID,
			UseRLM:        false,
			Expected:      task.Answer,
			Got:           "",
			IsCorrect:     false,
			ExecutionTime: duration.Seconds(),
			InputTokens:   0,
			OutputTokens:  0,
			Error:         err.Error(),
		}
	}

	isCorrect := checkAnswer(task.Answer, llmResp.Content)

	return BenchmarkResult{
		TaskID:        task.TaskID,
		UseRLM:        false,
		Expected:      task.Answer,
		Got:           llmResp.Content,
		IsCorrect:     isCorrect,
		ExecutionTime: duration.Seconds(),
		InputTokens:   llmResp.PromptTokens,
		OutputTokens:  llmResp.CompletionTokens,
	}
}

// RLMOptions holds configuration for RLM runs.
type RLMOptions struct {
	LogDir          string
	Verbose         bool
	Streaming       bool
	Pool            *repl.REPLPool
	Compression     bool
	VerbatimIters   int
}

// runRLM runs a task with RLM.
func runRLM(ctx context.Context, task Task, client *AnthropicClient, opts RLMOptions) BenchmarkResult {
	var log *logger.Logger
	var err error

	if opts.LogDir != "" {
		log, err = logger.New(opts.LogDir, logger.Config{
			RootModel:     client.model,
			MaxIterations: 30,
			Backend:       "anthropic",
			BackendKwargs: map[string]any{"model_name": client.model},
			Context:       task.Context,
			Query:         task.Question,
		})
		if err != nil {
			fmt.Printf("Warning: could not create logger: %v\n", err)
		} else {
			defer func() { _ = log.Close() }()
		}
	}

	// Build RLM options
	rlmOpts := []rlm.Option{
		rlm.WithMaxIterations(30),
		rlm.WithVerbose(opts.Verbose),
		rlm.WithLogger(log),
	}

	if opts.Streaming {
		rlmOpts = append(rlmOpts, rlm.WithStreaming(true))
	}

	if opts.Pool != nil {
		rlmOpts = append(rlmOpts, rlm.WithREPLPool(opts.Pool))
	}

	if opts.Compression {
		rlmOpts = append(rlmOpts, rlm.WithHistoryCompression(opts.VerbatimIters, 500))
	}

	r := rlm.New(client, client, rlmOpts...)

	start := time.Now()
	result, err := r.Complete(ctx, task.Context, task.Question)
	duration := time.Since(start)

	if err != nil {
		return BenchmarkResult{
			TaskID:        task.TaskID,
			UseRLM:        true,
			Expected:      task.Answer,
			Got:           "",
			IsCorrect:     false,
			ExecutionTime: duration.Seconds(),
			Error:         err.Error(),
		}
	}

	isCorrect := checkAnswer(task.Answer, result.Response)

	return BenchmarkResult{
		TaskID:        task.TaskID,
		UseRLM:        true,
		Expected:      task.Answer,
		Got:           result.Response,
		IsCorrect:     isCorrect,
		ExecutionTime: duration.Seconds(),
		InputTokens:   result.Usage.PromptTokens,
		OutputTokens:  result.Usage.CompletionTokens,
	}
}

// computeSummary calculates aggregate statistics.
func computeSummary(results []BenchmarkResult) BenchmarkSummary {
	if len(results) == 0 {
		return BenchmarkSummary{}
	}

	var correct int
	var totalTime float64
	var totalInput, totalOutput int

	for _, r := range results {
		if r.IsCorrect {
			correct++
		}
		totalTime += r.ExecutionTime
		totalInput += r.InputTokens
		totalOutput += r.OutputTokens
	}

	return BenchmarkSummary{
		TotalTasks:        len(results),
		CorrectTasks:      correct,
		Accuracy:          float64(correct) / float64(len(results)),
		TotalTime:         totalTime,
		AvgTimePerTask:    totalTime / float64(len(results)),
		TotalInputTokens:  totalInput,
		TotalOutputTokens: totalOutput,
	}
}

// loadTasks loads benchmark tasks from a JSON file.
func loadTasks(path string) ([]Task, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read file: %w", err)
	}

	var tasks []Task
	if err := json.Unmarshal(data, &tasks); err != nil {
		return nil, fmt.Errorf("unmarshal: %w", err)
	}

	return tasks, nil
}

// truncate truncates a string to maxLen characters.
func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}

func main() {
	var (
		tasksFile       = flag.String("tasks", "", "Path to tasks JSON file")
		model           = flag.String("model", "claude-sonnet-4-5-20250929", "Model to use")
		numTasks        = flag.Int("num-tasks", 10, "Number of tasks to run")
		logDir          = flag.String("log-dir", "./logs", "Directory for RLM logs")
		outputFile      = flag.String("output", "", "Output JSON file for results")
		verbose         = flag.Bool("verbose", false, "Enable verbose RLM output")
		enableStreaming = flag.Bool("streaming", false, "Enable streaming for root LLM calls")
		enablePooling   = flag.Bool("pooling", false, "Enable REPL instance pooling")
		poolSize        = flag.Int("pool-size", 5, "REPL pool size (requires -pooling)")
		enableCompression = flag.Bool("compression", false, "Enable history compression")
		verbatimIters   = flag.Int("verbatim-iters", 3, "Keep last N iterations verbatim (requires -compression)")
	)
	flag.Parse()

	apiKey := os.Getenv("ANTHROPIC_API_KEY")
	if apiKey == "" {
		fmt.Println("Error: ANTHROPIC_API_KEY environment variable not set")
		os.Exit(1)
	}

	// Load tasks
	var tasks []Task
	if *tasksFile != "" {
		var err error
		tasks, err = loadTasks(*tasksFile)
		if err != nil {
			fmt.Printf("Error loading tasks: %v\n", err)
			os.Exit(1)
		}
	} else {
		// Use sample OOLONG-like tasks for testing
		tasks = generateSampleTasks()
	}

	// Limit tasks
	if *numTasks > 0 && *numTasks < len(tasks) {
		tasks = tasks[:*numTasks]
	}

	fmt.Println(strings.Repeat("=", 60))
	fmt.Println("RLM-GO BENCHMARK")
	fmt.Println(strings.Repeat("=", 60))
	fmt.Printf("Model: %s\n", *model)
	fmt.Printf("Tasks: %d\n", len(tasks))
	if *enableStreaming {
		fmt.Println("Streaming: enabled")
	}
	if *enablePooling {
		fmt.Printf("REPL Pooling: enabled (size=%d)\n", *poolSize)
	}
	if *enableCompression {
		fmt.Printf("History Compression: enabled (verbatim=%d)\n", *verbatimIters)
	}
	fmt.Println(strings.Repeat("-", 60))

	client := NewAnthropicClient(apiKey, *model)
	ctx := context.Background()

	// Create REPL pool if enabled
	var pool *repl.REPLPool
	if *enablePooling {
		pool = repl.NewREPLPool(client, *poolSize, true) // pre-warm
		fmt.Printf("REPL pool initialized with %d instances\n", *poolSize)
	}

	// Build RLM options
	rlmOpts := RLMOptions{
		LogDir:        *logDir,
		Verbose:       *verbose,
		Streaming:     *enableStreaming,
		Pool:          pool,
		Compression:   *enableCompression,
		VerbatimIters: *verbatimIters,
	}

	var baselineResults []BenchmarkResult
	var rlmResults []BenchmarkResult

	for i, task := range tasks {
		fmt.Printf("\nTask %d/%d: %s\n", i+1, len(tasks), task.TaskID)

		// Run baseline
		fmt.Println("  Running baseline (direct LLM)...")
		baselineResult := runBaseline(ctx, task, client)
		baselineResults = append(baselineResults, baselineResult)

		status := "CORRECT"
		if !baselineResult.IsCorrect {
			status = "WRONG"
		}
		if baselineResult.Error != "" {
			status = "ERROR: " + baselineResult.Error
		}
		fmt.Printf("    Baseline: %s (%.2fs)\n", status, baselineResult.ExecutionTime)
		fmt.Printf("      Expected: %s\n", task.Answer)
		fmt.Printf("      Got:      %s\n", truncate(baselineResult.Got, 100))

		// Run RLM
		fmt.Println("  Running RLM...")
		rlmResult := runRLM(ctx, task, client, rlmOpts)
		rlmResults = append(rlmResults, rlmResult)

		status = "CORRECT"
		if !rlmResult.IsCorrect {
			status = "WRONG"
		}
		if rlmResult.Error != "" {
			status = "ERROR: " + rlmResult.Error
		}
		fmt.Printf("    RLM:      %s (%.2fs)\n", status, rlmResult.ExecutionTime)
		fmt.Printf("      Expected: %s\n", task.Answer)
		fmt.Printf("      Got:      %s\n", truncate(rlmResult.Got, 100))
	}

	// Compute summaries
	baselineSummary := computeSummary(baselineResults)
	rlmSummary := computeSummary(rlmResults)

	// Print comparison
	fmt.Println()
	fmt.Println(strings.Repeat("=", 60))
	fmt.Println("BENCHMARK COMPARISON")
	fmt.Println(strings.Repeat("=", 60))
	fmt.Printf("Model: %s\n", *model)
	fmt.Printf("Total tasks: %d\n", len(tasks))
	fmt.Println(strings.Repeat("-", 60))
	fmt.Printf("%-30s %12s %12s\n", "Metric", "Baseline", "RLM")
	fmt.Println(strings.Repeat("-", 60))
	fmt.Printf("%-30s %11.1f%% %11.1f%%\n", "Accuracy", baselineSummary.Accuracy*100, rlmSummary.Accuracy*100)
	fmt.Printf("%-30s %12d %12d\n", "Correct Tasks", baselineSummary.CorrectTasks, rlmSummary.CorrectTasks)
	fmt.Printf("%-30s %12.2f %12.2f\n", "Total Time (s)", baselineSummary.TotalTime, rlmSummary.TotalTime)
	fmt.Printf("%-30s %12.2f %12.2f\n", "Avg Time/Task (s)", baselineSummary.AvgTimePerTask, rlmSummary.AvgTimePerTask)
	fmt.Printf("%-30s %12d %12d\n", "Total Input Tokens", baselineSummary.TotalInputTokens, rlmSummary.TotalInputTokens)
	fmt.Printf("%-30s %12d %12d\n", "Total Output Tokens", baselineSummary.TotalOutputTokens, rlmSummary.TotalOutputTokens)
	fmt.Println(strings.Repeat("=", 60))

	// Highlight winner
	if rlmSummary.Accuracy > baselineSummary.Accuracy {
		improvement := (rlmSummary.Accuracy - baselineSummary.Accuracy) * 100
		fmt.Printf("\nRLM outperforms baseline by %.1f percentage points!\n", improvement)
	} else if baselineSummary.Accuracy > rlmSummary.Accuracy {
		diff := (baselineSummary.Accuracy - rlmSummary.Accuracy) * 100
		fmt.Printf("\nBaseline outperforms RLM by %.1f percentage points.\n", diff)
	} else {
		fmt.Println("\nRLM and baseline have equal accuracy.")
	}

	// Save results if output file specified
	if *outputFile != "" {
		output := map[string]any{
			"model":            *model,
			"baseline_results": baselineResults,
			"rlm_results":      rlmResults,
			"baseline_summary": baselineSummary,
			"rlm_summary":      rlmSummary,
		}

		data, err := json.MarshalIndent(output, "", "  ")
		if err != nil {
			fmt.Printf("Error marshaling output: %v\n", err)
		} else if err := os.WriteFile(*outputFile, data, 0644); err != nil {
			fmt.Printf("Error writing output: %v\n", err)
		} else {
			fmt.Printf("\nResults saved to: %s\n", *outputFile)
		}
	}

	fmt.Println("\nBenchmark complete!")
}

// generateSampleTasks creates sample OOLONG-like tasks for testing.
func generateSampleTasks() []Task {
	// Generate a simple needle-in-haystack context
	filler := strings.Repeat("The weather today is pleasant and the temperature is moderate. Many people enjoy outdoor activities during this season. ", 100)

	return []Task{
		{
			TaskID:   "sample_0",
			Context:  filler + "\n\nThe secret code is: ALPHA-7892.\n\n" + filler,
			Question: "What is the secret code mentioned in the text?",
			Answer:   "ALPHA-7892",
		},
		{
			TaskID:   "sample_1",
			Context:  filler + "\n\nThe password to access the system is: sunshine42.\n\n" + filler,
			Question: "What is the password mentioned in the text?",
			Answer:   "sunshine42",
		},
		{
			TaskID:   "sample_2",
			Context:  filler + "\n\nThe special number you need to remember is: 847293.\n\n" + filler,
			Question: "What is the special number mentioned in the text?",
			Answer:   "847293",
		},
	}
}
