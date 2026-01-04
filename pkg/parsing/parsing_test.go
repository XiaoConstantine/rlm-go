package parsing

import (
	"testing"

	"github.com/XiaoConstantine/rlm-go/pkg/core"
)

func TestFindCodeBlocks(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected []string
	}{
		{
			name:     "empty input",
			input:    "",
			expected: nil,
		},
		{
			name:     "no code blocks",
			input:    "This is just plain text without any code blocks.",
			expected: nil,
		},
		{
			name:     "single go code block",
			input:    "Here is some code:\n```go\nfmt.Println(\"hello\")\n```",
			expected: []string{`fmt.Println("hello")`},
		},
		{
			name:     "single repl code block",
			input:    "Here is some code:\n```repl\nfmt.Println(\"hello\")\n```",
			expected: []string{`fmt.Println("hello")`},
		},
		{
			name:     "multiple code blocks",
			input:    "First:\n```go\nx := 1\n```\n\nSecond:\n```go\ny := 2\n```",
			expected: []string{"x := 1", "y := 2"},
		},
		{
			name:     "mixed go and repl blocks",
			input:    "```go\na := 1\n```\n\n```repl\nb := 2\n```",
			expected: []string{"a := 1", "b := 2"},
		},
		{
			name:     "code block with multiple lines",
			input:    "```go\nfmt.Println(\"line1\")\nfmt.Println(\"line2\")\nfmt.Println(\"line3\")\n```",
			expected: []string{"fmt.Println(\"line1\")\nfmt.Println(\"line2\")\nfmt.Println(\"line3\")"},
		},
		{
			name:     "ignores other language blocks",
			input:    "```python\nprint('hello')\n```\n\n```go\nfmt.Println(\"hello\")\n```",
			expected: []string{`fmt.Println("hello")`},
		},
		{
			name:     "code block with leading/trailing whitespace",
			input:    "```go\n   x := 1   \n```",
			expected: []string{"x := 1"},
		},
		{
			name:     "empty code block",
			input:    "```go\n   \n```",
			expected: nil,
		},
		{
			name:     "code block with special characters",
			input:    "```go\nregex := regexp.MustCompile(`\\d+`)\n```",
			expected: []string{"regex := regexp.MustCompile(`\\d+`)"},
		},
		{
			name:     "code block with nested backticks",
			input:    "```go\nfmt.Printf(\"`quoted`\")\n```",
			expected: []string{"fmt.Printf(\"`quoted`\")"},
		},
		{
			name:     "code block without trailing newline before closing",
			input:    "```go\nx := 1\n```",
			expected: []string{"x := 1"},
		},
		{
			name: "multiline code with blank lines",
			input: `Some text
` + "```go" + `
x := 1

y := 2
` + "```",
			expected: []string{"x := 1\n\ny := 2"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := FindCodeBlocks(tt.input)
			if len(result) != len(tt.expected) {
				t.Errorf("FindCodeBlocks() got %d blocks, want %d", len(result), len(tt.expected))
				t.Errorf("Got: %v", result)
				t.Errorf("Want: %v", tt.expected)
				return
			}
			for i := range result {
				if result[i] != tt.expected[i] {
					t.Errorf("FindCodeBlocks()[%d] = %q, want %q", i, result[i], tt.expected[i])
				}
			}
		})
	}
}

func TestFindFinalAnswer(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected *core.FinalAnswer
	}{
		{
			name:     "empty input",
			input:    "",
			expected: nil,
		},
		{
			name:     "no final answer",
			input:    "This is just some text without any final answer.",
			expected: nil,
		},
		{
			name:     "FINAL with simple string",
			input:    "FINAL(42)",
			expected: &core.FinalAnswer{Type: core.FinalTypeDirect, Content: "42"},
		},
		{
			name:     "FINAL with double quoted string",
			input:    `FINAL("the answer is 42")`,
			expected: &core.FinalAnswer{Type: core.FinalTypeDirect, Content: "the answer is 42"},
		},
		{
			name:     "FINAL with single quoted string",
			input:    `FINAL('the answer is 42')`,
			expected: &core.FinalAnswer{Type: core.FinalTypeDirect, Content: "the answer is 42"},
		},
		{
			name:     "FINAL with backtick quoted string",
			input:    "FINAL(`the answer is 42`)",
			expected: &core.FinalAnswer{Type: core.FinalTypeDirect, Content: "the answer is 42"},
		},
		{
			name:     "FINAL with leading whitespace",
			input:    "   FINAL(result)",
			expected: &core.FinalAnswer{Type: core.FinalTypeDirect, Content: "result"},
		},
		{
			name:     "FINAL with complex content",
			input:    "FINAL(40% positive, 40% negative, 20% neutral)",
			expected: &core.FinalAnswer{Type: core.FinalTypeDirect, Content: "40% positive, 40% negative, 20% neutral"},
		},
		{
			name:     "FINAL_VAR with simple identifier",
			input:    "FINAL_VAR(answer)",
			expected: &core.FinalAnswer{Type: core.FinalTypeVariable, Content: "answer"},
		},
		{
			name:     "FINAL_VAR with underscore identifier",
			input:    "FINAL_VAR(final_answer)",
			expected: &core.FinalAnswer{Type: core.FinalTypeVariable, Content: "final_answer"},
		},
		{
			name:     "FINAL_VAR with leading whitespace",
			input:    "  FINAL_VAR(result)",
			expected: &core.FinalAnswer{Type: core.FinalTypeVariable, Content: "result"},
		},
		{
			name:     "FINAL_VAR with numeric suffix",
			input:    "FINAL_VAR(answer123)",
			expected: &core.FinalAnswer{Type: core.FinalTypeVariable, Content: "answer123"},
		},
		{
			name:     "FINAL_VAR takes precedence over FINAL",
			input:    "FINAL_VAR(x)\nFINAL(y)",
			expected: &core.FinalAnswer{Type: core.FinalTypeVariable, Content: "x"},
		},
		{
			name:     "FINAL in middle of text",
			input:    "Some text\nFINAL(answer)\nMore text",
			expected: &core.FinalAnswer{Type: core.FinalTypeDirect, Content: "answer"},
		},
		{
			name:     "FINAL_VAR in middle of text",
			input:    "Some text\nFINAL_VAR(myVar)\nMore text",
			expected: &core.FinalAnswer{Type: core.FinalTypeVariable, Content: "myVar"},
		},
		{
			name:     "FINAL not at start of line (still matches with leading spaces)",
			input:    "	FINAL(tabbed)",
			expected: &core.FinalAnswer{Type: core.FinalTypeDirect, Content: "tabbed"},
		},
		{
			name:     "FINAL with spaces inside parentheses",
			input:    "FINAL(  answer  )",
			expected: &core.FinalAnswer{Type: core.FinalTypeDirect, Content: "answer"},
		},
		{
			name:     "FINAL_VAR with spaces inside parentheses",
			input:    "FINAL_VAR(  varName  )",
			expected: &core.FinalAnswer{Type: core.FinalTypeVariable, Content: "varName"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := FindFinalAnswer(tt.input)
			if tt.expected == nil {
				if result != nil {
					t.Errorf("FindFinalAnswer() = %+v, want nil", result)
				}
				return
			}
			if result == nil {
				t.Errorf("FindFinalAnswer() = nil, want %+v", tt.expected)
				return
			}
			if result.Type != tt.expected.Type {
				t.Errorf("FindFinalAnswer().Type = %v, want %v", result.Type, tt.expected.Type)
			}
			if result.Content != tt.expected.Content {
				t.Errorf("FindFinalAnswer().Content = %q, want %q", result.Content, tt.expected.Content)
			}
		})
	}
}

func TestStripQuotes(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected string
	}{
		{
			name:     "no quotes",
			input:    "hello",
			expected: "hello",
		},
		{
			name:     "double quotes",
			input:    `"hello"`,
			expected: "hello",
		},
		{
			name:     "single quotes",
			input:    `'hello'`,
			expected: "hello",
		},
		{
			name:     "backticks",
			input:    "`hello`",
			expected: "hello",
		},
		{
			name:     "mismatched quotes",
			input:    `"hello'`,
			expected: `"hello'`,
		},
		{
			name:     "empty string",
			input:    "",
			expected: "",
		},
		{
			name:     "single character",
			input:    "a",
			expected: "a",
		},
		{
			name:     "empty quotes",
			input:    `""`,
			expected: "",
		},
		{
			name:     "nested quotes preserved",
			input:    `"'hello'"`,
			expected: "'hello'",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := stripQuotes(tt.input)
			if result != tt.expected {
				t.Errorf("stripQuotes(%q) = %q, want %q", tt.input, result, tt.expected)
			}
		})
	}
}

// TestFindCodeBlocksBoundary tests boundary conditions
func TestFindCodeBlocksBoundary(t *testing.T) {
	tests := []struct {
		name     string
		input    string
		expected []string
	}{
		{
			name:     "very long code block",
			input:    "```go\n" + string(make([]byte, 10000)) + "x := 1\n```",
			expected: []string{string(make([]byte, 10000)) + "x := 1"},
		},
		{
			name:     "code block at very beginning",
			input:    "```go\nx := 1\n```",
			expected: []string{"x := 1"},
		},
		{
			name:     "code block at very end",
			input:    "text\n```go\nx := 1\n```",
			expected: []string{"x := 1"},
		},
		{
			name:     "consecutive code blocks",
			input:    "```go\na := 1\n```\n```go\nb := 2\n```",
			expected: []string{"a := 1", "b := 2"},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := FindCodeBlocks(tt.input)
			if len(result) != len(tt.expected) {
				t.Errorf("FindCodeBlocks() got %d blocks, want %d", len(result), len(tt.expected))
				return
			}
			for i := range result {
				if result[i] != tt.expected[i] {
					t.Errorf("FindCodeBlocks()[%d] len=%d, want len=%d", i, len(result[i]), len(tt.expected[i]))
				}
			}
		})
	}
}
