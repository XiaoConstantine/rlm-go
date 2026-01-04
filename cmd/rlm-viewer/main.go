// Package main provides an enhanced CLI viewer for RLM JSONL log files.
package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"os/signal"
	"regexp"
	"sort"
	"strings"
	"syscall"
	"time"

	"golang.org/x/term"
)

// ANSI color codes (variables so they can be disabled)
var (
	Reset      = "\033[0m"
	Bold       = "\033[1m"
	Dim        = "\033[2m"
	Italic     = "\033[3m"
	Cyan       = "\033[36m"
	Green      = "\033[32m"
	Yellow     = "\033[33m"
	Blue       = "\033[34m"
	Magenta    = "\033[35m"
	Red        = "\033[31m"
	BoldCyan   = "\033[1;36m"
	BoldGreen  = "\033[1;32m"
	BoldYellow = "\033[1;33m"
	BoldBlue   = "\033[1;34m"
	BoldRed    = "\033[1;31m"
	BgBlue     = "\033[44m"
	BgGreen    = "\033[42m"
	White      = "\033[37m"
)

// Metadata represents the metadata entry in JSONL.
type Metadata struct {
	Type          string         `json:"type"`
	Timestamp     string         `json:"timestamp"`
	RootModel     string         `json:"root_model"`
	MaxIterations int            `json:"max_iterations"`
	Backend       string         `json:"backend"`
	Context       string         `json:"context"`
	Query         string         `json:"query"`
	BackendKwargs map[string]any `json:"backend_kwargs"`
}

// Iteration represents an iteration entry in JSONL.
type Iteration struct {
	Type          string      `json:"type"`
	Iteration     int         `json:"iteration"`
	Timestamp     string      `json:"timestamp"`
	Prompt        []Message   `json:"prompt"`
	Response      string      `json:"response"`
	CodeBlocks    []CodeBlock `json:"code_blocks"`
	FinalAnswer   any         `json:"final_answer"`
	IterationTime float64     `json:"iteration_time"`
}

// Message represents a chat message.
type Message struct {
	Role    string `json:"role"`
	Content string `json:"content"`
}

// CodeBlock represents an executed code block.
type CodeBlock struct {
	Code   string     `json:"code"`
	Result CodeResult `json:"result"`
}

// CodeResult represents code execution results.
type CodeResult struct {
	Stdout        string         `json:"stdout"`
	Stderr        string         `json:"stderr"`
	Locals        map[string]any `json:"locals"`
	ExecutionTime float64        `json:"execution_time"`
	RLMCalls      []RLMCall      `json:"rlm_calls"`
}

// RLMCall represents a sub-LLM call.
type RLMCall struct {
	Prompt           string  `json:"prompt"`
	Response         string  `json:"response"`
	PromptTokens     int     `json:"prompt_tokens"`
	CompletionTokens int     `json:"completion_tokens"`
	ExecutionTime    float64 `json:"execution_time"`
}

// LogData holds all parsed log data.
type LogData struct {
	Filename   string
	Metadata   *Metadata
	Iterations []Iteration
}

// ViewerConfig holds viewer configuration.
type ViewerConfig struct {
	Compact     bool
	NoColor     bool
	Interactive bool
	Watch       bool
	Iteration   int  // -1 means all
	ErrorsOnly  bool
	FinalOnly   bool
	Search      string
	Stats       bool
	Export      string
}

func main() {
	cfg := ViewerConfig{Iteration: -1}

	flag.BoolVar(&cfg.Compact, "compact", false, "Compact output (hide full responses)")
	flag.BoolVar(&cfg.Compact, "c", false, "Compact output (shorthand)")
	flag.BoolVar(&cfg.NoColor, "no-color", false, "Disable colored output")
	flag.BoolVar(&cfg.Interactive, "interactive", false, "Interactive navigation mode")
	flag.BoolVar(&cfg.Interactive, "i", false, "Interactive mode (shorthand)")
	flag.BoolVar(&cfg.Watch, "watch", false, "Watch file for changes (live tail)")
	flag.BoolVar(&cfg.Watch, "w", false, "Watch mode (shorthand)")
	flag.IntVar(&cfg.Iteration, "iter", -1, "Show only specific iteration (1-indexed)")
	flag.BoolVar(&cfg.ErrorsOnly, "errors", false, "Show only iterations with errors")
	flag.BoolVar(&cfg.FinalOnly, "final", false, "Show only the final answer")
	flag.StringVar(&cfg.Search, "search", "", "Search for text in responses/code")
	flag.StringVar(&cfg.Search, "s", "", "Search (shorthand)")
	flag.BoolVar(&cfg.Stats, "stats", false, "Show detailed statistics only")
	flag.StringVar(&cfg.Export, "export", "", "Export to file (supports .md)")
	flag.Parse()

	if flag.NArg() < 1 {
		printUsage()
		os.Exit(1)
	}

	filename := flag.Arg(0)

	if cfg.NoColor {
		disableColors()
	}

	if cfg.Watch {
		if err := watchLog(filename, cfg); err != nil {
			fmt.Fprintf(os.Stderr, "Error: %v\n", err)
			os.Exit(1)
		}
		return
	}

	data, err := parseLog(filename)
	if err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}

	if cfg.Export != "" {
		if err := exportLog(data, cfg); err != nil {
			fmt.Fprintf(os.Stderr, "Error exporting: %v\n", err)
			os.Exit(1)
		}
		fmt.Printf("Exported to: %s\n", cfg.Export)
		return
	}

	if cfg.Interactive {
		runInteractive(data, cfg)
		return
	}

	viewLog(data, cfg)
}

func printUsage() {
	fmt.Fprintf(os.Stderr, `%sRLM Log Viewer%s - Enhanced CLI viewer for RLM JSONL logs

%sUsage:%s
  rlm-viewer [options] <file.jsonl>

%sOptions:%s
  -c, --compact      Compact output (hide full responses)
  -i, --interactive  Interactive navigation mode
  -w, --watch        Watch file for changes (live tail)
  --iter N           Show only iteration N (1-indexed)
  --errors           Show only iterations with errors
  --final            Show only the final answer
  -s, --search TEXT  Search for text in responses/code
  --stats            Show detailed statistics only
  --export FILE      Export to markdown file
  --no-color         Disable colored output

%sInteractive Mode Keys:%s
  j/↓    Next iteration
  k/↑    Previous iteration
  g      Go to first iteration
  G      Go to last iteration
  /      Search
  n      Next search result
  e      Toggle expand/compact
  q      Quit

%sExamples:%s
  rlm-viewer session.jsonl              # View log
  rlm-viewer -i session.jsonl           # Interactive mode
  rlm-viewer -w session.jsonl           # Watch live
  rlm-viewer --stats session.jsonl      # Statistics only
  rlm-viewer --iter 3 session.jsonl     # Show iteration 3
  rlm-viewer -s "error" session.jsonl   # Search for "error"

`, BoldCyan, Reset, Bold, Reset, Bold, Reset, Bold, Reset, Bold, Reset)
}

func parseLog(filename string) (*LogData, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("open file: %w", err)
	}
	defer func() { _ = file.Close() }()

	data := &LogData{Filename: filename}
	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 1024*1024), 10*1024*1024)

	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			continue
		}

		var entry struct {
			Type string `json:"type"`
		}
		if err := json.Unmarshal([]byte(line), &entry); err != nil {
			continue
		}

		switch entry.Type {
		case "metadata":
			var m Metadata
			if err := json.Unmarshal([]byte(line), &m); err == nil {
				data.Metadata = &m
			}
		case "iteration":
			var iter Iteration
			if err := json.Unmarshal([]byte(line), &iter); err == nil {
				data.Iterations = append(data.Iterations, iter)
			}
		}
	}

	return data, scanner.Err()
}

func viewLog(data *LogData, cfg ViewerConfig) {
	// Apply filters
	iterations := filterIterations(data.Iterations, cfg)

	if cfg.FinalOnly {
		printFinalOnly(data, iterations)
		return
	}

	if cfg.Stats {
		printDetailedStats(data)
		return
	}

	// Print header
	printHeader(data.Filename, data.Metadata)

	// Print iterations
	var totalTokens struct{ prompt, completion int }
	for _, iter := range iterations {
		tokens := printIteration(iter, cfg.Compact, cfg.Search)
		totalTokens.prompt += tokens.prompt
		totalTokens.completion += tokens.completion
	}

	// Print summary
	printSummary(iterations, totalTokens.prompt, totalTokens.completion)
}

func filterIterations(iterations []Iteration, cfg ViewerConfig) []Iteration {
	var result []Iteration

	for _, iter := range iterations {
		// Specific iteration filter
		if cfg.Iteration > 0 && iter.Iteration != cfg.Iteration {
			continue
		}

		// Errors only filter
		if cfg.ErrorsOnly {
			hasError := false
			for _, block := range iter.CodeBlocks {
				if block.Result.Stderr != "" {
					hasError = true
					break
				}
			}
			if !hasError {
				continue
			}
		}

		// Search filter
		if cfg.Search != "" {
			if !searchInIteration(iter, cfg.Search) {
				continue
			}
		}

		result = append(result, iter)
	}

	return result
}

func searchInIteration(iter Iteration, query string) bool {
	query = strings.ToLower(query)

	// Search in response
	if strings.Contains(strings.ToLower(iter.Response), query) {
		return true
	}

	// Search in code blocks
	for _, block := range iter.CodeBlocks {
		if strings.Contains(strings.ToLower(block.Code), query) {
			return true
		}
		if strings.Contains(strings.ToLower(block.Result.Stdout), query) {
			return true
		}
		if strings.Contains(strings.ToLower(block.Result.Stderr), query) {
			return true
		}
	}

	return false
}

func printFinalOnly(data *LogData, iterations []Iteration) {
	fmt.Printf("\n%s%s RLM Final Answer %s\n", BoldGreen, "═══", Reset)

	if data.Metadata != nil && data.Metadata.Query != "" {
		fmt.Printf("%sQuery:%s %s\n\n", Dim, Reset, data.Metadata.Query)
	}

	for _, iter := range iterations {
		if iter.FinalAnswer != nil {
			switch v := iter.FinalAnswer.(type) {
			case string:
				fmt.Printf("%s\n", v)
			case []any:
				if len(v) == 2 {
					fmt.Printf("%s%v%s = %v\n", Cyan, v[0], Reset, v[1])
				}
			default:
				fmt.Printf("%v\n", v)
			}
			fmt.Printf("\n%s(Iteration %d, %.2fs)%s\n", Dim, iter.Iteration, iter.IterationTime, Reset)
			return
		}
	}

	fmt.Printf("%sNo final answer found%s\n", Dim, Reset)
}

func printHeader(filename string, meta *Metadata) {
	fmt.Printf("\n%s%s RLM Log Viewer %s\n", BoldCyan, "═══", Reset)
	fmt.Printf("%sFile:%s %s\n", Dim, Reset, filename)

	if meta != nil {
		fmt.Printf("%sModel:%s %s\n", Dim, Reset, meta.RootModel)
		fmt.Printf("%sBackend:%s %s\n", Dim, Reset, meta.Backend)
		fmt.Printf("%sMax Iterations:%s %d\n", Dim, Reset, meta.MaxIterations)

		if meta.Query != "" {
			fmt.Printf("%sQuery:%s %s\n", Dim, Reset, truncate(meta.Query, 100))
		}
		if meta.Context != "" {
			fmt.Printf("%sContext:%s %s\n", Dim, Reset, truncate(meta.Context, 100))
		}

		if ts, err := time.Parse(time.RFC3339Nano, meta.Timestamp); err == nil {
			fmt.Printf("%sStarted:%s %s\n", Dim, Reset, ts.Format("2006-01-02 15:04:05"))
		}
	}
	fmt.Println()
}

func printIteration(iter Iteration, compact bool, searchQuery string) struct{ prompt, completion int } {
	var tokens struct{ prompt, completion int }

	// Iteration header
	fmt.Printf("%s┌─ Iteration %d %s", BoldYellow, iter.Iteration, Reset)
	if iter.IterationTime > 0 {
		fmt.Printf("%s(%.2fs)%s", Dim, iter.IterationTime, Reset)
	}

	// Final answer indicator
	if iter.FinalAnswer != nil {
		fmt.Printf(" %s[FINAL]%s", BoldGreen, Reset)
	}

	// Error indicator
	hasError := false
	for _, block := range iter.CodeBlocks {
		if block.Result.Stderr != "" {
			hasError = true
			break
		}
	}
	if hasError {
		fmt.Printf(" %s[ERROR]%s", BoldRed, Reset)
	}

	fmt.Println()

	// Response preview
	if !compact && iter.Response != "" {
		fmt.Printf("%s│%s %sResponse:%s\n", Yellow, Reset, Dim, Reset)
		response := iter.Response
		if searchQuery != "" {
			response = highlightSearch(response, searchQuery)
		}
		printIndented(response, "│   ", 500)
	}

	// Code blocks
	for i, block := range iter.CodeBlocks {
		fmt.Printf("%s│%s\n", Yellow, Reset)
		fmt.Printf("%s├─ Code Block #%d%s", BoldBlue, i+1, Reset)
		if block.Result.ExecutionTime > 0 {
			fmt.Printf(" %s(%.2fs)%s", Dim, block.Result.ExecutionTime, Reset)
		}
		fmt.Println()

		// Code
		fmt.Printf("%s│%s  %s┌─ Code:%s\n", Yellow, Reset, Blue, Reset)
		code := block.Code
		if searchQuery != "" {
			code = highlightSearch(code, searchQuery)
		}
		printCodeBlock(code, "│  │ ")

		// Output
		if block.Result.Stdout != "" {
			fmt.Printf("%s│%s  %s├─ Output:%s\n", Yellow, Reset, Green, Reset)
			stdout := block.Result.Stdout
			if searchQuery != "" {
				stdout = highlightSearch(stdout, searchQuery)
			}
			printIndented(stdout, "│  │ ", 300)
		}
		if block.Result.Stderr != "" {
			fmt.Printf("%s│%s  %s├─ Stderr:%s\n", Yellow, Reset, Red, Reset)
			printIndented(block.Result.Stderr, "│  │ ", 300)
		}

		// Locals
		if len(block.Result.Locals) > 0 {
			fmt.Printf("%s│%s  %s├─ Locals:%s\n", Yellow, Reset, Magenta, Reset)
			for k, v := range block.Result.Locals {
				vStr := fmt.Sprintf("%v", v)
				fmt.Printf("%s│%s  │   %s%s%s = %s\n", Yellow, Reset, Cyan, k, Reset, truncate(vStr, 80))
			}
		}

		// Sub-LLM calls
		if len(block.Result.RLMCalls) > 0 {
			fmt.Printf("%s│%s  %s└─ Sub-LLM Calls (%d):%s\n", Yellow, Reset, Magenta, len(block.Result.RLMCalls), Reset)
			for j, call := range block.Result.RLMCalls {
				tokens.prompt += call.PromptTokens
				tokens.completion += call.CompletionTokens

				fmt.Printf("%s│%s      %s[%d]%s ", Yellow, Reset, Dim, j+1, Reset)
				fmt.Printf("%s%.2fs%s", Dim, call.ExecutionTime, Reset)
				if call.PromptTokens > 0 || call.CompletionTokens > 0 {
					fmt.Printf(" %s(%d→%d tokens)%s", Dim, call.PromptTokens, call.CompletionTokens, Reset)
				}
				fmt.Println()

				if !compact {
					fmt.Printf("%s│%s        %sPrompt:%s %s\n", Yellow, Reset, Dim, Reset, truncate(call.Prompt, 100))
					fmt.Printf("%s│%s        %sResponse:%s %s\n", Yellow, Reset, Dim, Reset, truncate(call.Response, 100))
				}
			}
		}
	}

	// Final answer
	if iter.FinalAnswer != nil {
		fmt.Printf("%s│%s\n", Yellow, Reset)
		fmt.Printf("%s└─ Final Answer:%s\n", BoldGreen, Reset)
		switch v := iter.FinalAnswer.(type) {
		case string:
			printIndented(v, "   ", 500)
		case []any:
			if len(v) == 2 {
				fmt.Printf("   %s%v%s = %s\n", Cyan, v[0], Reset, truncate(fmt.Sprintf("%v", v[1]), 200))
			}
		default:
			fmt.Printf("   %v\n", v)
		}
	}

	fmt.Println()
	return tokens
}

func highlightSearch(text, query string) string {
	if query == "" {
		return text
	}
	re := regexp.MustCompile("(?i)" + regexp.QuoteMeta(query))
	return re.ReplaceAllStringFunc(text, func(match string) string {
		return BgGreen + Bold + match + Reset
	})
}

func printDetailedStats(data *LogData) {
	fmt.Printf("\n%s%s RLM Session Statistics %s\n\n", BoldCyan, "═══", Reset)

	if data.Metadata != nil {
		fmt.Printf("%sModel:%s %s\n", Dim, Reset, data.Metadata.RootModel)
		fmt.Printf("%sBackend:%s %s\n", Dim, Reset, data.Metadata.Backend)
		if data.Metadata.Query != "" {
			fmt.Printf("%sQuery:%s %s\n", Dim, Reset, truncate(data.Metadata.Query, 80))
		}
		fmt.Println()
	}

	var totalTime float64
	var totalCodeBlocks, totalLLMCalls int
	var totalPromptTokens, totalCompletionTokens int
	var iterTimes []float64
	var llmCallTimes []float64

	for _, iter := range data.Iterations {
		totalTime += iter.IterationTime
		iterTimes = append(iterTimes, iter.IterationTime)
		totalCodeBlocks += len(iter.CodeBlocks)

		for _, block := range iter.CodeBlocks {
			for _, call := range block.Result.RLMCalls {
				totalLLMCalls++
				totalPromptTokens += call.PromptTokens
				totalCompletionTokens += call.CompletionTokens
				llmCallTimes = append(llmCallTimes, call.ExecutionTime)
			}
		}
	}

	// Overview
	fmt.Printf("%s── Overview ──%s\n", Bold, Reset)
	fmt.Printf("  Iterations:     %d\n", len(data.Iterations))
	fmt.Printf("  Code Blocks:    %d\n", totalCodeBlocks)
	fmt.Printf("  Sub-LLM Calls:  %d\n", totalLLMCalls)
	fmt.Printf("  Total Time:     %.2fs\n", totalTime)
	fmt.Println()

	// Token usage
	if totalPromptTokens > 0 || totalCompletionTokens > 0 {
		fmt.Printf("%s── Token Usage ──%s\n", Bold, Reset)
		fmt.Printf("  Prompt Tokens:     %d\n", totalPromptTokens)
		fmt.Printf("  Completion Tokens: %d\n", totalCompletionTokens)
		fmt.Printf("  Total Tokens:      %d\n", totalPromptTokens+totalCompletionTokens)
		if totalLLMCalls > 0 {
			fmt.Printf("  Avg per Call:      %.0f\n", float64(totalPromptTokens+totalCompletionTokens)/float64(totalLLMCalls))
		}
		fmt.Println()
	}

	// Timing analysis
	fmt.Printf("%s── Timing Analysis ──%s\n", Bold, Reset)
	if len(iterTimes) > 0 {
		sort.Float64s(iterTimes)
		fmt.Printf("  Iteration Time:\n")
		fmt.Printf("    Min:    %.2fs\n", iterTimes[0])
		fmt.Printf("    Max:    %.2fs\n", iterTimes[len(iterTimes)-1])
		fmt.Printf("    Median: %.2fs\n", iterTimes[len(iterTimes)/2])
		fmt.Printf("    Avg:    %.2fs\n", totalTime/float64(len(iterTimes)))
	}

	if len(llmCallTimes) > 0 {
		sort.Float64s(llmCallTimes)
		var sum float64
		for _, t := range llmCallTimes {
			sum += t
		}
		fmt.Printf("  Sub-LLM Call Time:\n")
		fmt.Printf("    Min:    %.2fs\n", llmCallTimes[0])
		fmt.Printf("    Max:    %.2fs\n", llmCallTimes[len(llmCallTimes)-1])
		fmt.Printf("    Avg:    %.2fs\n", sum/float64(len(llmCallTimes)))
	}
	fmt.Println()

	// Per-iteration breakdown
	fmt.Printf("%s── Per-Iteration Breakdown ──%s\n", Bold, Reset)
	fmt.Printf("  %s%-5s %-8s %-6s %-6s %-12s%s\n", Dim, "Iter", "Time", "Code", "Calls", "Tokens", Reset)
	for _, iter := range data.Iterations {
		var iterPrompt, iterCompletion int
		var callCount int
		for _, block := range iter.CodeBlocks {
			for _, call := range block.Result.RLMCalls {
				callCount++
				iterPrompt += call.PromptTokens
				iterCompletion += call.CompletionTokens
			}
		}

		marker := ""
		if iter.FinalAnswer != nil {
			marker = BoldGreen + " ✓" + Reset
		}

		fmt.Printf("  %-5d %-8.2fs %-6d %-6d %-12s%s\n",
			iter.Iteration,
			iter.IterationTime,
			len(iter.CodeBlocks),
			callCount,
			fmt.Sprintf("%d→%d", iterPrompt, iterCompletion),
			marker,
		)
	}
	fmt.Println()

	// Timeline visualization
	if len(data.Iterations) > 0 && len(data.Iterations) <= 20 {
		fmt.Printf("%s── Timeline ──%s\n", Bold, Reset)
		maxTime := iterTimes[len(iterTimes)-1]
		for _, iter := range data.Iterations {
			barLen := int((iter.IterationTime / maxTime) * 40)
			if barLen < 1 {
				barLen = 1
			}
			bar := strings.Repeat("█", barLen)
			color := Yellow
			if iter.FinalAnswer != nil {
				color = Green
			}
			fmt.Printf("  %2d %s%s%s %.2fs\n", iter.Iteration, color, bar, Reset, iter.IterationTime)
		}
		fmt.Println()
	}
}

func printSummary(iterations []Iteration, promptTokens, completionTokens int) {
	var totalTime float64
	var totalCodeBlocks, totalLLMCalls int

	for _, iter := range iterations {
		totalTime += iter.IterationTime
		totalCodeBlocks += len(iter.CodeBlocks)
		for _, block := range iter.CodeBlocks {
			totalLLMCalls += len(block.Result.RLMCalls)
		}
	}

	fmt.Printf("%s%s Summary %s\n", BoldCyan, "═══", Reset)
	fmt.Printf("  Iterations: %d\n", len(iterations))
	fmt.Printf("  Code Blocks: %d\n", totalCodeBlocks)
	fmt.Printf("  Sub-LLM Calls: %d\n", totalLLMCalls)
	if promptTokens > 0 || completionTokens > 0 {
		fmt.Printf("  Tokens: %d prompt + %d completion = %d total\n",
			promptTokens, completionTokens, promptTokens+completionTokens)
	}
	fmt.Printf("  Total Time: %.2fs\n", totalTime)
	fmt.Println()
}

// Interactive mode
func runInteractive(data *LogData, cfg ViewerConfig) {
	if len(data.Iterations) == 0 {
		fmt.Println("No iterations to display")
		return
	}

	// Save terminal state
	oldState, err := term.MakeRaw(int(os.Stdin.Fd()))
	if err != nil {
		fmt.Printf("Error entering raw mode: %v\n", err)
		viewLog(data, cfg)
		return
	}
	defer func() { _ = term.Restore(int(os.Stdin.Fd()), oldState) }()

	currentIdx := 0
	expanded := !cfg.Compact
	searchQuery := cfg.Search
	searchResults := []int{}
	searchIdx := 0

	if searchQuery != "" {
		searchResults = findSearchMatches(data.Iterations, searchQuery)
		if len(searchResults) > 0 {
			currentIdx = searchResults[0]
		}
	}

	clearScreen()
	printInteractiveIteration(data, currentIdx, expanded, searchQuery, len(searchResults), searchIdx)

	buf := make([]byte, 3)
	for {
		n, err := os.Stdin.Read(buf)
		if err != nil {
			break
		}

		switch n {
		case 1:
			switch buf[0] {
			case 'q', 3: // q or Ctrl+C
				clearScreen()
				return
			case 'j':
				if currentIdx < len(data.Iterations)-1 {
					currentIdx++
				}
			case 'k':
				if currentIdx > 0 {
					currentIdx--
				}
			case 'g':
				currentIdx = 0
			case 'G':
				currentIdx = len(data.Iterations) - 1
			case 'e':
				expanded = !expanded
			case 'n':
				if len(searchResults) > 0 {
					searchIdx = (searchIdx + 1) % len(searchResults)
					currentIdx = searchResults[searchIdx]
				}
			case 'N':
				if len(searchResults) > 0 {
					searchIdx = (searchIdx - 1 + len(searchResults)) % len(searchResults)
					currentIdx = searchResults[searchIdx]
				}
			case '/':
				_ = term.Restore(int(os.Stdin.Fd()), oldState)
				fmt.Print("\r\033[K/")
				reader := bufio.NewReader(os.Stdin)
				query, _ := reader.ReadString('\n')
				searchQuery = strings.TrimSpace(query)
				searchResults = findSearchMatches(data.Iterations, searchQuery)
				searchIdx = 0
				if len(searchResults) > 0 {
					currentIdx = searchResults[0]
				}
				_, _ = term.MakeRaw(int(os.Stdin.Fd()))
			case 's':
				// Show stats
				_ = term.Restore(int(os.Stdin.Fd()), oldState)
				clearScreen()
				printDetailedStats(data)
				fmt.Print("\nPress any key to continue...")
				_, _ = os.Stdin.Read(make([]byte, 1))
				_, _ = term.MakeRaw(int(os.Stdin.Fd()))
			case '?':
				// Show help
				_ = term.Restore(int(os.Stdin.Fd()), oldState)
				clearScreen()
				printInteractiveHelp()
				fmt.Print("\nPress any key to continue...")
				_, _ = os.Stdin.Read(make([]byte, 1))
				_, _ = term.MakeRaw(int(os.Stdin.Fd()))
			}
		case 3:
			// Arrow keys
			if buf[0] == 27 && buf[1] == 91 {
				switch buf[2] {
				case 65: // Up
					if currentIdx > 0 {
						currentIdx--
					}
				case 66: // Down
					if currentIdx < len(data.Iterations)-1 {
						currentIdx++
					}
				}
			}
		}

		clearScreen()
		printInteractiveIteration(data, currentIdx, expanded, searchQuery, len(searchResults), searchIdx)
	}
}

func findSearchMatches(iterations []Iteration, query string) []int {
	var matches []int
	for i, iter := range iterations {
		if searchInIteration(iter, query) {
			matches = append(matches, i)
		}
	}
	return matches
}

func printInteractiveIteration(data *LogData, idx int, expanded bool, searchQuery string, matchCount, matchIdx int) {
	iter := data.Iterations[idx]

	// Status bar
	fmt.Printf("%s%s Iteration %d/%d %s", BgBlue, White, idx+1, len(data.Iterations), Reset)
	if matchCount > 0 {
		fmt.Printf(" %s[Match %d/%d]%s", Dim, matchIdx+1, matchCount, Reset)
	}
	if iter.FinalAnswer != nil {
		fmt.Printf(" %s[FINAL]%s", BoldGreen, Reset)
	}
	fmt.Printf(" %s[e]xpand [/]search [s]tats [?]help [q]uit%s\n", Dim, Reset)
	fmt.Println(strings.Repeat("─", 60))

	// Print the iteration
	if expanded {
		printIteration(iter, false, searchQuery)
	} else {
		printIteration(iter, true, searchQuery)
	}

	// Navigation hints
	fmt.Printf("\n%s← k/↑  j/↓ →  g=first G=last  n/N=search%s\n", Dim, Reset)
}

func printInteractiveHelp() {
	fmt.Printf(`
%s%s Interactive Mode Help %s

%sNavigation:%s
  j, ↓       Next iteration
  k, ↑       Previous iteration
  g          Go to first iteration
  G          Go to last iteration

%sSearch:%s
  /          Enter search query
  n          Next search result
  N          Previous search result

%sDisplay:%s
  e          Toggle expand/compact mode
  s          Show detailed statistics

%sOther:%s
  ?          Show this help
  q, Ctrl+C  Quit

`, BoldCyan, "═══", Reset, Bold, Reset, Bold, Reset, Bold, Reset, Bold, Reset)
}

func clearScreen() {
	fmt.Print("\033[2J\033[H")
}

// Watch mode
func watchLog(filename string, cfg ViewerConfig) error {
	fmt.Printf("%s%s Watching: %s %s\n", BoldCyan, "═══", filename, Reset)
	fmt.Printf("%sPress Ctrl+C to stop%s\n\n", Dim, Reset)

	lastSize := int64(0)
	lastIterCount := 0

	// Handle Ctrl+C gracefully
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)

	ticker := time.NewTicker(500 * time.Millisecond)
	defer ticker.Stop()

	for {
		select {
		case <-sigCh:
			fmt.Printf("\n%sStopped watching%s\n", Dim, Reset)
			return nil
		case <-ticker.C:
			info, err := os.Stat(filename)
			if err != nil {
				continue
			}

			if info.Size() != lastSize {
				lastSize = info.Size()

				data, err := parseLog(filename)
				if err != nil {
					continue
				}

				// Only print new iterations
				for i := lastIterCount; i < len(data.Iterations); i++ {
					if i == 0 && data.Metadata != nil {
						printHeader(filename, data.Metadata)
					}
					printIteration(data.Iterations[i], cfg.Compact, "")
				}
				lastIterCount = len(data.Iterations)
			}
		}
	}
}

// Export functionality
func exportLog(data *LogData, cfg ViewerConfig) error {
	if !strings.HasSuffix(cfg.Export, ".md") {
		return fmt.Errorf("only markdown (.md) export is supported")
	}

	var sb strings.Builder

	// Header
	sb.WriteString("# RLM Session Log\n\n")

	if data.Metadata != nil {
		sb.WriteString("## Metadata\n\n")
		sb.WriteString(fmt.Sprintf("- **Model:** %s\n", data.Metadata.RootModel))
		sb.WriteString(fmt.Sprintf("- **Backend:** %s\n", data.Metadata.Backend))
		sb.WriteString(fmt.Sprintf("- **Max Iterations:** %d\n", data.Metadata.MaxIterations))
		if data.Metadata.Query != "" {
			sb.WriteString(fmt.Sprintf("- **Query:** %s\n", data.Metadata.Query))
		}
		if data.Metadata.Context != "" {
			sb.WriteString(fmt.Sprintf("- **Context:** %s\n", truncate(data.Metadata.Context, 200)))
		}
		sb.WriteString("\n")
	}

	// Iterations
	for _, iter := range data.Iterations {
		finalMarker := ""
		if iter.FinalAnswer != nil {
			finalMarker = " ✅"
		}
		sb.WriteString(fmt.Sprintf("## Iteration %d%s\n\n", iter.Iteration, finalMarker))
		sb.WriteString(fmt.Sprintf("*Time: %.2fs*\n\n", iter.IterationTime))

		if iter.Response != "" {
			sb.WriteString("### Response\n\n")
			sb.WriteString(iter.Response)
			sb.WriteString("\n\n")
		}

		for i, block := range iter.CodeBlocks {
			sb.WriteString(fmt.Sprintf("### Code Block %d\n\n", i+1))
			sb.WriteString("```go\n")
			sb.WriteString(block.Code)
			sb.WriteString("\n```\n\n")

			if block.Result.Stdout != "" {
				sb.WriteString("**Output:**\n```\n")
				sb.WriteString(block.Result.Stdout)
				sb.WriteString("\n```\n\n")
			}

			if block.Result.Stderr != "" {
				sb.WriteString("**Errors:**\n```\n")
				sb.WriteString(block.Result.Stderr)
				sb.WriteString("\n```\n\n")
			}

			if len(block.Result.RLMCalls) > 0 {
				sb.WriteString(fmt.Sprintf("**Sub-LLM Calls (%d):**\n\n", len(block.Result.RLMCalls)))
				for j, call := range block.Result.RLMCalls {
					sb.WriteString(fmt.Sprintf("%d. *%.2fs, %d→%d tokens*\n", j+1, call.ExecutionTime, call.PromptTokens, call.CompletionTokens))
					sb.WriteString(fmt.Sprintf("   - Prompt: %s\n", truncate(call.Prompt, 100)))
					sb.WriteString(fmt.Sprintf("   - Response: %s\n", truncate(call.Response, 100)))
				}
				sb.WriteString("\n")
			}
		}

		if iter.FinalAnswer != nil {
			sb.WriteString("### Final Answer\n\n")
			switch v := iter.FinalAnswer.(type) {
			case string:
				sb.WriteString(v)
			case []any:
				if len(v) == 2 {
					sb.WriteString(fmt.Sprintf("`%v` = %v", v[0], v[1]))
				}
			default:
				sb.WriteString(fmt.Sprintf("%v", v))
			}
			sb.WriteString("\n\n")
		}
	}

	// Summary
	var totalTime float64
	var totalCodeBlocks, totalLLMCalls, totalPrompt, totalCompletion int
	for _, iter := range data.Iterations {
		totalTime += iter.IterationTime
		totalCodeBlocks += len(iter.CodeBlocks)
		for _, block := range iter.CodeBlocks {
			for _, call := range block.Result.RLMCalls {
				totalLLMCalls++
				totalPrompt += call.PromptTokens
				totalCompletion += call.CompletionTokens
			}
		}
	}

	sb.WriteString("## Summary\n\n")
	sb.WriteString(fmt.Sprintf("- **Iterations:** %d\n", len(data.Iterations)))
	sb.WriteString(fmt.Sprintf("- **Code Blocks:** %d\n", totalCodeBlocks))
	sb.WriteString(fmt.Sprintf("- **Sub-LLM Calls:** %d\n", totalLLMCalls))
	sb.WriteString(fmt.Sprintf("- **Total Tokens:** %d (%d prompt + %d completion)\n", totalPrompt+totalCompletion, totalPrompt, totalCompletion))
	sb.WriteString(fmt.Sprintf("- **Total Time:** %.2fs\n", totalTime))

	return os.WriteFile(cfg.Export, []byte(sb.String()), 0644)
}

func printIndented(text, prefix string, maxLen int) {
	text = truncateContent(text, maxLen)
	lines := strings.Split(text, "\n")
	for _, line := range lines {
		fmt.Printf("%s%s%s\n", Yellow, prefix, Reset+line)
	}
}

func printCodeBlock(code, prefix string) {
	lines := strings.Split(strings.TrimSpace(code), "\n")
	maxLines := 15
	if len(lines) > maxLines {
		for i := 0; i < maxLines-1; i++ {
			fmt.Printf("%s%s%s\n", Yellow, prefix, Reset+lines[i])
		}
		fmt.Printf("%s%s%s... (%d more lines)%s\n", Yellow, prefix, Dim, len(lines)-maxLines+1, Reset)
	} else {
		for _, line := range lines {
			fmt.Printf("%s%s%s\n", Yellow, prefix, Reset+line)
		}
	}
}

func truncate(s string, maxLen int) string {
	s = strings.ReplaceAll(s, "\n", " ")
	s = strings.Join(strings.Fields(s), " ")
	if len(s) > maxLen {
		return s[:maxLen] + "..."
	}
	return s
}

func truncateContent(s string, maxLen int) string {
	if len(s) > maxLen {
		return s[:maxLen] + "\n... (truncated)"
	}
	return s
}

func disableColors() {
	Reset = ""
	Bold = ""
	Dim = ""
	Italic = ""
	Cyan = ""
	Green = ""
	Yellow = ""
	Blue = ""
	Magenta = ""
	Red = ""
	BoldCyan = ""
	BoldGreen = ""
	BoldYellow = ""
	BoldBlue = ""
	BoldRed = ""
	BgBlue = ""
	BgGreen = ""
	White = ""
}
