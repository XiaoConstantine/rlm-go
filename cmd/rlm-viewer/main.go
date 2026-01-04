// Package main provides a CLI viewer for RLM JSONL log files.
package main

import (
	"bufio"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"strings"
	"time"
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

func main() {
	compact := flag.Bool("compact", false, "Compact output (hide full responses)")
	noColor := flag.Bool("no-color", false, "Disable colored output")
	flag.Parse()

	if flag.NArg() < 1 {
		fmt.Fprintf(os.Stderr, "Usage: rlm-viewer [options] <file.jsonl>\n")
		flag.PrintDefaults()
		os.Exit(1)
	}

	filename := flag.Arg(0)
	if err := viewLog(filename, *compact, *noColor); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}

func viewLog(filename string, compact, noColor bool) error {
	file, err := os.Open(filename)
	if err != nil {
		return fmt.Errorf("open file: %w", err)
	}
	defer file.Close()

	// Disable colors if requested
	if noColor {
		disableColors()
	}

	scanner := bufio.NewScanner(file)
	scanner.Buffer(make([]byte, 1024*1024), 10*1024*1024) // 10MB max line

	var metadata *Metadata
	var iterations []Iteration
	var totalTokens struct {
		prompt, completion int
	}

	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			continue
		}

		// Check type first
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
				metadata = &m
			}
		case "iteration":
			var iter Iteration
			if err := json.Unmarshal([]byte(line), &iter); err == nil {
				iterations = append(iterations, iter)
			}
		}
	}

	if err := scanner.Err(); err != nil {
		return fmt.Errorf("scan file: %w", err)
	}

	// Print header
	printHeader(filename, metadata)

	// Print iterations
	for _, iter := range iterations {
		tokens := printIteration(iter, compact)
		totalTokens.prompt += tokens.prompt
		totalTokens.completion += tokens.completion
	}

	// Print summary
	printSummary(iterations, totalTokens.prompt, totalTokens.completion)

	return nil
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

func printIteration(iter Iteration, compact bool) struct{ prompt, completion int } {
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
	fmt.Println()

	// Response preview
	if !compact && iter.Response != "" {
		fmt.Printf("%s│%s %sResponse:%s\n", Yellow, Reset, Dim, Reset)
		printIndented(iter.Response, "│   ", 500)
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
		printCodeBlock(block.Code, "│  │ ")

		// Output
		if block.Result.Stdout != "" {
			fmt.Printf("%s│%s  %s├─ Output:%s\n", Yellow, Reset, Green, Reset)
			printIndented(block.Result.Stdout, "│  │ ", 300)
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

func printIndented(text, prefix string, maxLen int) {
	text = truncate(text, maxLen)
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
}
