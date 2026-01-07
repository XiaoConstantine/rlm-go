package analyzer

import (
	"strings"
	"testing"
)

func TestAnalyze(t *testing.T) {
	tests := []struct {
		name         string
		content      any
		expectedType ContextType
		expectedSize int
	}{
		{
			name:         "empty string",
			content:      "",
			expectedType: TypePlainText,
			expectedSize: 0,
		},
		{
			name:         "plain text",
			content:      "This is just some plain text without any special formatting.",
			expectedType: TypePlainText,
			expectedSize: 60,
		},
		{
			name:         "json object",
			content:      `{"name": "test", "value": 42}`,
			expectedType: TypeJSON,
			expectedSize: 29,
		},
		{
			name:         "json array",
			content:      `[1, 2, 3, 4, 5]`,
			expectedType: TypeJSON,
			expectedSize: 15,
		},
		{
			name: "markdown with headers",
			content: `# Title

This is some content.

## Section 1

More content here.

### Subsection

Even more content.
`,
			expectedType: TypeMarkdown,
		},
		{
			name: "code content",
			content: `package main

import "fmt"

func main() {
	fmt.Println("Hello, World!")
}
`,
			expectedType: TypeCode,
		},
		{
			name: "log content",
			content: `2024-01-15 10:30:45 INFO Starting application
2024-01-15 10:30:46 DEBUG Initializing components
2024-01-15 10:30:47 ERROR Failed to connect
`,
			expectedType: TypeLog,
		},
		{
			name: "csv content",
			content: `name,age,city
John,30,New York
Jane,25,Los Angeles
Bob,35,Chicago
Alice,28,Seattle
`,
			expectedType: TypeCSV,
		},
		{
			name:         "xml content",
			content:      `<?xml version="1.0"?><root><item>test</item></root>`,
			expectedType: TypeXML,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := Analyze(tt.content)

			if result.Type != tt.expectedType {
				t.Errorf("Type = %v, want %v", result.Type, tt.expectedType)
			}

			if tt.expectedSize > 0 && result.Size != tt.expectedSize {
				t.Errorf("Size = %d, want %d", result.Size, tt.expectedSize)
			}
		})
	}
}

func TestAnalyzeStructure(t *testing.T) {
	tests := []struct {
		name               string
		content            string
		expectedHasHeaders bool
		expectedHasCode    bool
		expectedHasLists   bool
	}{
		{
			name:               "plain text",
			content:            "Just some text",
			expectedHasHeaders: false,
			expectedHasCode:    false,
			expectedHasLists:   false,
		},
		{
			name:               "markdown with headers",
			content:            "# Header 1\n## Header 2\n### Header 3",
			expectedHasHeaders: true,
			expectedHasCode:    false,
			expectedHasLists:   false,
		},
		{
			name:               "markdown with code blocks",
			content:            "Text\n```go\ncode here\n```\nMore text",
			expectedHasHeaders: false,
			expectedHasCode:    true,
			expectedHasLists:   false,
		},
		{
			name:               "markdown with lists",
			content:            "Items:\n- Item 1\n- Item 2\n- Item 3",
			expectedHasHeaders: false,
			expectedHasCode:    false,
			expectedHasLists:   true,
		},
		{
			name:               "full markdown",
			content:            "# Title\n\n- list item\n\n```go\ncode\n```",
			expectedHasHeaders: true,
			expectedHasCode:    true,
			expectedHasLists:   true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := analyzeStructure(tt.content)

			if result.HasHeaders != tt.expectedHasHeaders {
				t.Errorf("HasHeaders = %v, want %v", result.HasHeaders, tt.expectedHasHeaders)
			}
			if result.HasCodeBlocks != tt.expectedHasCode {
				t.Errorf("HasCodeBlocks = %v, want %v", result.HasCodeBlocks, tt.expectedHasCode)
			}
			if result.HasLists != tt.expectedHasLists {
				t.Errorf("HasLists = %v, want %v", result.HasLists, tt.expectedHasLists)
			}
		})
	}
}

func TestCalculateNestingDepth(t *testing.T) {
	tests := []struct {
		name     string
		content  string
		expected int
	}{
		{
			name:     "no nesting",
			content:  "plain text",
			expected: 0,
		},
		{
			name:     "single level object",
			content:  `{"key": "value"}`,
			expected: 1,
		},
		{
			name:     "nested object",
			content:  `{"outer": {"inner": "value"}}`,
			expected: 2,
		},
		{
			name:     "deeply nested",
			content:  `{"a": {"b": {"c": {"d": "value"}}}}`,
			expected: 4,
		},
		{
			name:     "array nesting",
			content:  `[[[1, 2], [3, 4]], [[5, 6]]]`,
			expected: 3,
		},
		{
			name:     "mixed nesting",
			content:  `{"arr": [{"nested": [1, 2, 3]}]}`,
			expected: 4,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := calculateNestingDepth(tt.content)
			if result != tt.expected {
				t.Errorf("calculateNestingDepth() = %d, want %d", result, tt.expected)
			}
		})
	}
}

func TestDetermineChunkStrategy(t *testing.T) {
	tests := []struct {
		name             string
		size             int
		contentType      ContextType
		expectedStrategy ChunkStrategy
	}{
		{
			name:             "small content",
			size:             10000,
			contentType:      TypePlainText,
			expectedStrategy: StrategyNone,
		},
		{
			name:             "medium JSON",
			size:             100000,
			contentType:      TypeJSON,
			expectedStrategy: StrategyHierarchical,
		},
		{
			name:             "medium markdown with headers",
			size:             100000,
			contentType:      TypeMarkdown,
			expectedStrategy: StrategyDelimiter,
		},
		{
			name:             "large code",
			size:             300000,
			contentType:      TypeCode,
			expectedStrategy: StrategySemantic,
		},
		{
			name:             "large logs",
			size:             300000,
			contentType:      TypeLog,
			expectedStrategy: StrategyDelimiter,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			analysis := &ContextAnalysis{
				Size: tt.size,
				Type: tt.contentType,
				Structure: StructureHints{
					HasHeaders: tt.contentType == TypeMarkdown,
				},
			}
			strategy, _ := determineChunkStrategy(analysis)
			if strategy != tt.expectedStrategy {
				t.Errorf("strategy = %v, want %v", strategy, tt.expectedStrategy)
			}
		})
	}
}

func TestIsJSON(t *testing.T) {
	tests := []struct {
		name     string
		content  string
		expected bool
	}{
		{
			name:     "valid object",
			content:  `{"key": "value"}`,
			expected: true,
		},
		{
			name:     "valid array",
			content:  `[1, 2, 3]`,
			expected: true,
		},
		{
			name:     "nested JSON",
			content:  `{"nested": {"key": [1, 2, 3]}}`,
			expected: true,
		},
		{
			name:     "invalid JSON",
			content:  `{key: value}`,
			expected: false,
		},
		{
			name:     "plain text",
			content:  `This is not JSON`,
			expected: false,
		},
		{
			name:     "empty string",
			content:  ``,
			expected: false,
		},
		{
			name:     "with whitespace",
			content:  `  {"key": "value"}  `,
			expected: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := isJSON(tt.content)
			if result != tt.expected {
				t.Errorf("isJSON() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestIsCSV(t *testing.T) {
	tests := []struct {
		name     string
		content  string
		expected bool
	}{
		{
			name: "valid CSV",
			content: `a,b,c
1,2,3
4,5,6
7,8,9`,
			expected: true,
		},
		{
			name: "tab separated",
			content: "a\tb\tc\n1\t2\t3\n4\t5\t6\n7\t8\t9",
			expected: true,
		},
		{
			name:     "single line",
			content:  `a,b,c`,
			expected: false,
		},
		{
			name:     "plain text",
			content:  "This is not CSV\nJust regular text",
			expected: false,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := isCSV(tt.content)
			if result != tt.expected {
				t.Errorf("isCSV() = %v, want %v", result, tt.expected)
			}
		})
	}
}

func TestFindDelimiters(t *testing.T) {
	tests := []struct {
		name        string
		contentType ContextType
		wantNonEmpty bool
	}{
		{name: "JSON", contentType: TypeJSON, wantNonEmpty: true},
		{name: "Markdown", contentType: TypeMarkdown, wantNonEmpty: true},
		{name: "Code", contentType: TypeCode, wantNonEmpty: true},
		{name: "Log", contentType: TypeLog, wantNonEmpty: true},
		{name: "CSV", contentType: TypeCSV, wantNonEmpty: true},
		{name: "PlainText", contentType: TypePlainText, wantNonEmpty: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := findDelimiters("test content", tt.contentType)
			if tt.wantNonEmpty && len(result) == 0 {
				t.Errorf("findDelimiters() returned empty slice, want non-empty")
			}
		})
	}
}

func TestGenerateLLMHint(t *testing.T) {
	tests := []struct {
		name       string
		analysis   *ContextAnalysis
		wantPrefix string
	}{
		{
			name: "JSON content",
			analysis: &ContextAnalysis{
				Type:            TypeJSON,
				EstimatedTokens: 1000,
				Structure:       StructureHints{NestingDepth: 2},
			},
			wantPrefix: "CONTEXT ANALYSIS:",
		},
		{
			name: "large content",
			analysis: &ContextAnalysis{
				Type:            TypePlainText,
				EstimatedTokens: 150000,
				Structure:       StructureHints{},
			},
			wantPrefix: "CONTEXT ANALYSIS:",
		},
		{
			name: "markdown with headers",
			analysis: &ContextAnalysis{
				Type:            TypeMarkdown,
				EstimatedTokens: 1000,
				Structure:       StructureHints{HasHeaders: true},
			},
			wantPrefix: "CONTEXT ANALYSIS:",
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := generateLLMHint(tt.analysis)
			if !strings.HasPrefix(result, tt.wantPrefix) {
				t.Errorf("generateLLMHint() = %q, want prefix %q", result, tt.wantPrefix)
			}
		})
	}
}

func TestEstimateTokens(t *testing.T) {
	tests := []struct {
		name       string
		content    string
		minTokens  int
		maxTokens  int
	}{
		{
			name:      "empty",
			content:   "",
			minTokens: 0,
			maxTokens: 0,
		},
		{
			name:      "single word",
			content:   "hello",
			minTokens: 1,
			maxTokens: 3,
		},
		{
			name:      "sentence",
			content:   "This is a simple test sentence.",
			minTokens: 5,
			maxTokens: 15,
		},
		{
			name:      "with punctuation",
			content:   "Hello! How are you? I'm fine, thanks.",
			minTokens: 5,
			maxTokens: 20,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			result := estimateTokens(tt.content)
			if result < tt.minTokens || result > tt.maxTokens {
				t.Errorf("estimateTokens() = %d, want between %d and %d", result, tt.minTokens, tt.maxTokens)
			}
		})
	}
}

func TestAnalyzeWithConfig(t *testing.T) {
	cfg := Config{
		SmallContextThreshold: 100000,
		TokenEstimateRatio:    4.0,
	}

	content := strings.Repeat("test ", 1000) // ~5000 bytes

	result := AnalyzeWithConfig(content, cfg)

	if result.RecommendedStrategy != StrategyNone {
		t.Errorf("RecommendedStrategy = %v, want %v", result.RecommendedStrategy, StrategyNone)
	}
}

func TestDefaultConfig(t *testing.T) {
	cfg := DefaultConfig()

	if cfg.SmallContextThreshold != 50000 {
		t.Errorf("SmallContextThreshold = %d, want 50000", cfg.SmallContextThreshold)
	}
	if cfg.TokenEstimateRatio != 4.0 {
		t.Errorf("TokenEstimateRatio = %f, want 4.0", cfg.TokenEstimateRatio)
	}
}

func TestAnalyzeComplexPayload(t *testing.T) {
	// Test with map payload
	mapPayload := map[string]any{
		"key1": "value1",
		"key2": 42,
		"nested": map[string]any{
			"inner": "data",
		},
	}

	result := Analyze(mapPayload)

	if result.Size == 0 {
		t.Error("Size should be > 0 for map payload")
	}
	if result.Type != TypeJSON {
		t.Errorf("Type = %v, want %v for map payload", result.Type, TypeJSON)
	}
}

func TestAnalyzeBytesPayload(t *testing.T) {
	content := []byte(`{"test": "data"}`)
	result := Analyze(content)

	if result.Type != TypeJSON {
		t.Errorf("Type = %v, want %v for byte slice JSON", result.Type, TypeJSON)
	}
	if result.Size != len(content) {
		t.Errorf("Size = %d, want %d", result.Size, len(content))
	}
}
