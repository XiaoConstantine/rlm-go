// Package parsing provides utilities for extracting code blocks and final answers from LLM responses.
package parsing

import (
	"encoding/json"
	"fmt"
	"regexp"
	"strconv"
	"strings"

	"github.com/XiaoConstantine/rlm-go/pkg/core"
)

var (
	// goCodeBlockRe matches ```go or ```repl code blocks.
	// (?s) enables DOTALL mode so . matches newlines.
	goCodeBlockRe = regexp.MustCompile("(?s)```(?:go|repl)\\s*\\n(.*?)\\n```")

	// finalVarRe matches FINAL_VAR(identifier) at start of line.
	finalVarRe = regexp.MustCompile(`(?m)^\s*FINAL_VAR\(\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\)`)

	// finalRe matches FINAL(...) at start of line.
	finalRe = regexp.MustCompile(`(?m)^\s*FINAL\((.+?)\)\s*$`)

	// Multi-line FINAL patterns for enhanced parsing
	// finalMultilineRe matches FINAL( with content spanning multiple lines until )
	finalMultilineRe = regexp.MustCompile(`(?s)^\s*FINAL\((.*?)\)\s*$`)

	// finalColonRe matches "FINAL:" or "Final:" followed by content
	finalColonRe = regexp.MustCompile(`(?im)^\s*FINAL\s*:\s*(.+)$`)

	// finalArrowRe matches "FINAL =>" or "FINAL ->" followed by content
	finalArrowRe = regexp.MustCompile(`(?im)^\s*FINAL\s*(?:=>|->)\s*(.+)$`)

	// JSON result patterns
	jsonResultRe = regexp.MustCompile(`(?s)\{\s*"(?:result|answer|final|output)"\s*:\s*(.+?)\s*\}`)
)

// FindCodeBlocks extracts all ```go or ```repl code blocks from the LLM response.
func FindCodeBlocks(text string) []string {
	matches := goCodeBlockRe.FindAllStringSubmatch(text, -1)
	if matches == nil {
		return nil
	}

	results := make([]string, 0, len(matches))
	for _, match := range matches {
		if len(match) > 1 {
			code := strings.TrimSpace(match[1])
			if code != "" {
				results = append(results, code)
			}
		}
	}
	return results
}

// FindFinalAnswer detects FINAL() or FINAL_VAR() signals in the LLM response.
// Returns nil if no final answer is found.
// Note: Code blocks are filtered out first to avoid false positives when
// FINAL appears in code examples.
//
// Uses multi-stage parsing with fallback strategies:
// 1. FINAL_VAR(varName) - variable reference
// 2. FINAL(value) - single-line direct value
// 3. FINAL(multi-line value) - multi-line content
// 4. FINAL: value - colon-style format
// 5. FINAL => value or FINAL -> value - arrow-style format
// 6. {"result": value} or {"answer": value} - JSON structured output
func FindFinalAnswer(text string) *core.FinalAnswer {
	// Remove code blocks first to avoid false positives from FINAL in examples
	textWithoutCode := goCodeBlockRe.ReplaceAllString(text, "")

	// Stage 1: Check FINAL_VAR first (most specific pattern)
	if match := finalVarRe.FindStringSubmatch(textWithoutCode); match != nil {
		return &core.FinalAnswer{
			Type:    core.FinalTypeVariable,
			Content: strings.TrimSpace(match[1]),
		}
	}

	// Stage 2: Check standard FINAL(value) - single line
	if match := finalRe.FindStringSubmatch(textWithoutCode); match != nil {
		content := strings.TrimSpace(match[1])
		content = stripQuotes(content)
		return &core.FinalAnswer{
			Type:    core.FinalTypeDirect,
			Content: content,
		}
	}

	// Stage 3: Check multi-line FINAL(value) - for complex multi-line answers
	if result := findMultilineFinal(textWithoutCode); result != nil {
		return result
	}

	// Stage 4: Check FINAL: value format
	if match := finalColonRe.FindStringSubmatch(textWithoutCode); match != nil {
		content := strings.TrimSpace(match[1])
		content = stripQuotes(content)
		return &core.FinalAnswer{
			Type:    core.FinalTypeDirect,
			Content: content,
		}
	}

	// Stage 5: Check FINAL => value or FINAL -> value format
	if match := finalArrowRe.FindStringSubmatch(textWithoutCode); match != nil {
		content := strings.TrimSpace(match[1])
		content = stripQuotes(content)
		return &core.FinalAnswer{
			Type:    core.FinalTypeDirect,
			Content: content,
		}
	}

	// Stage 6: Check JSON structured output
	if result := findJSONFinalAnswer(text); result != nil {
		return result
	}

	return nil
}

// findMultilineFinal handles FINAL() with content that spans multiple lines.
func findMultilineFinal(text string) *core.FinalAnswer {
	// Look for lines that start with FINAL(
	lines := strings.Split(text, "\n")
	for i, line := range lines {
		trimmed := strings.TrimSpace(line)
		if strings.HasPrefix(strings.ToUpper(trimmed), "FINAL(") {
			// Found start of FINAL, now find the closing )
			content := extractParenContent(lines, i)
			if content != "" {
				content = stripQuotes(content)
				return &core.FinalAnswer{
					Type:    core.FinalTypeDirect,
					Content: content,
				}
			}
		}
	}
	return nil
}

// extractParenContent extracts content from FINAL(...) that may span multiple lines.
func extractParenContent(lines []string, startLine int) string {
	// Build the full text from startLine onwards
	var builder strings.Builder
	for i := startLine; i < len(lines); i++ {
		if i > startLine {
			builder.WriteString("\n")
		}
		builder.WriteString(lines[i])
	}
	fullText := builder.String()

	// Find FINAL( and extract content until matching )
	trimmed := strings.TrimSpace(fullText)
	upperTrimmed := strings.ToUpper(trimmed)
	if !strings.HasPrefix(upperTrimmed, "FINAL(") {
		return ""
	}

	// Find the opening ( after FINAL
	openIdx := strings.Index(trimmed, "(")
	if openIdx == -1 {
		return ""
	}

	// Track parenthesis depth to find matching close
	depth := 0
	var content strings.Builder
	inContent := false

	for i := openIdx; i < len(trimmed); i++ {
		ch := trimmed[i]
		if ch == '(' {
			if depth > 0 {
				content.WriteByte(ch)
			}
			depth++
			inContent = true
		} else if ch == ')' {
			depth--
			if depth == 0 {
				// Found matching close
				return strings.TrimSpace(content.String())
			}
			content.WriteByte(ch)
		} else if inContent {
			content.WriteByte(ch)
		}
	}

	return "" // Unmatched parentheses
}

// findJSONFinalAnswer looks for JSON structured output containing result/answer.
func findJSONFinalAnswer(text string) *core.FinalAnswer {
	// First try to find and parse a complete JSON object
	if idx := strings.Index(text, "{"); idx != -1 {
		// Find potential JSON objects
		remaining := text[idx:]
		if endIdx := findJSONEnd(remaining); endIdx > 0 {
			jsonStr := remaining[:endIdx+1]
			if result := parseJSONResult(jsonStr); result != nil {
				return result
			}
		}
	}

	// Fallback: use regex for simpler patterns
	if match := jsonResultRe.FindStringSubmatch(text); match != nil {
		content := strings.TrimSpace(match[1])
		// Handle if the value is a quoted string
		if len(content) > 0 && content[0] == '"' {
			var str string
			if err := json.Unmarshal([]byte(content), &str); err == nil {
				return &core.FinalAnswer{
					Type:    core.FinalTypeDirect,
					Content: str,
				}
			}
		}
		content = stripQuotes(content)
		return &core.FinalAnswer{
			Type:    core.FinalTypeDirect,
			Content: content,
		}
	}

	return nil
}

// findJSONEnd finds the end index of a JSON object starting at position 0.
func findJSONEnd(text string) int {
	if len(text) == 0 || text[0] != '{' {
		return -1
	}

	depth := 0
	inString := false
	escaped := false

	for i, ch := range text {
		if escaped {
			escaped = false
			continue
		}

		if ch == '\\' && inString {
			escaped = true
			continue
		}

		if ch == '"' {
			inString = !inString
			continue
		}

		if !inString {
			if ch == '{' {
				depth++
			} else if ch == '}' {
				depth--
				if depth == 0 {
					return i
				}
			}
		}
	}

	return -1
}

// parseJSONResult attempts to parse JSON and extract result/answer/final/output field.
func parseJSONResult(jsonStr string) *core.FinalAnswer {
	var data map[string]any
	if err := json.Unmarshal([]byte(jsonStr), &data); err != nil {
		return nil
	}

	// Check for known result keys
	for _, key := range []string{"result", "answer", "final", "output"} {
		if val, ok := data[key]; ok {
			var content string
			switch v := val.(type) {
			case string:
				content = v
			case float64:
				// Format number cleanly
				content = formatFloat(v)
			case bool:
				if v {
					content = "true"
				} else {
					content = "false"
				}
			case nil:
				content = "null"
			default:
				// For complex types, convert to JSON string
				if bytes, err := json.Marshal(v); err == nil {
					content = string(bytes)
				} else {
					continue
				}
			}
			return &core.FinalAnswer{
				Type:    core.FinalTypeDirect,
				Content: content,
			}
		}
	}

	return nil
}

// formatFloat formats a float64 to a clean string representation.
func formatFloat(f float64) string {
	// Check if it's a whole number
	if f == float64(int64(f)) {
		return fmt.Sprintf("%d", int64(f))
	}
	// Use strconv for proper formatting, then clean up trailing zeros
	s := strconv.FormatFloat(f, 'f', -1, 64)
	// Remove unnecessary trailing zeros after decimal point
	if strings.Contains(s, ".") {
		s = strings.TrimRight(s, "0")
		s = strings.TrimRight(s, ".")
	}
	return s
}

// stripQuotes removes surrounding quotes from a string if present.
func stripQuotes(s string) string {
	if len(s) >= 2 {
		first, last := s[0], s[len(s)-1]
		if (first == '"' && last == '"') ||
			(first == '\'' && last == '\'') ||
			(first == '`' && last == '`') {
			return s[1 : len(s)-1]
		}
	}
	return s
}
