package contextindex

import (
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"unicode"
)

const (
	defaultChunkChars   = 4000
	defaultChunkOverlap = 200
)

// Chunk is a searchable slice of the loaded context.
type Chunk struct {
	ID        int
	Content   string
	StartChar int
	EndChar   int
	StartLine int
	EndLine   int
}

// Index provides deterministic chunk, line-range, and keyword search helpers.
type Index struct {
	raw    string
	lines  []string
	chunks []Chunk
}

// Stringify converts an arbitrary context payload into the string used by the index.
func Stringify(payload any) string {
	switch v := payload.(type) {
	case string:
		return v
	case []byte:
		return string(v)
	default:
		data, err := json.Marshal(v)
		if err != nil {
			return fmt.Sprintf("%v", v)
		}
		return string(data)
	}
}

// New builds an index over raw context text.
func New(raw string) *Index {
	idx := &Index{
		raw:   raw,
		lines: strings.Split(raw, "\n"),
	}
	idx.chunks = buildChunks(raw, idx.lines)
	return idx
}

// FindRelevant searches a raw context string without requiring callers to retain an Index.
func FindRelevant(raw, query string, topK int) []string {
	return New(raw).FindRelevant(query, topK)
}

// GetChunk returns a chunk from a raw context string.
func GetChunk(raw string, id int) string {
	chunk, _ := New(raw).GetChunk(id)
	return chunk
}

// GetContext returns a 1-indexed inclusive line range from a raw context string.
func GetContext(raw string, startLine, endLine int) string {
	return New(raw).GetContext(startLine, endLine)
}

// ChunkCount returns the number of chunks in a raw context string.
func ChunkCount(raw string) int {
	return New(raw).ChunkCount()
}

// LineCount returns the number of lines in a raw context string.
func LineCount(raw string) int {
	return New(raw).LineCount()
}

func buildChunks(raw string, lines []string) []Chunk {
	if raw == "" {
		return nil
	}

	var chunks []Chunk
	for start := 0; start < len(raw); {
		end := start + defaultChunkChars
		if end > len(raw) {
			end = len(raw)
		}
		startLine := lineForChar(raw, start)
		endLine := lineForChar(raw, end)
		chunks = append(chunks, Chunk{
			ID:        len(chunks),
			Content:   raw[start:end],
			StartChar: start,
			EndChar:   end,
			StartLine: startLine,
			EndLine:   endLine,
		})
		if end == len(raw) {
			break
		}
		start = end - defaultChunkOverlap
		if start < 0 {
			start = 0
		}
	}
	if len(chunks) == 0 && len(lines) > 0 {
		chunks = append(chunks, Chunk{ID: 0, Content: raw, StartLine: 1, EndLine: len(lines), EndChar: len(raw)})
	}
	return chunks
}

func lineForChar(raw string, charOffset int) int {
	if charOffset <= 0 {
		return 1
	}
	if charOffset > len(raw) {
		charOffset = len(raw)
	}
	return strings.Count(raw[:charOffset], "\n") + 1
}

// FindRelevant returns the top matching chunks by simple keyword score.
func (idx *Index) FindRelevant(query string, topK int) []string {
	if idx == nil || len(idx.chunks) == 0 {
		return []string{}
	}
	if topK <= 0 {
		topK = 3
	}
	if topK > len(idx.chunks) {
		topK = len(idx.chunks)
	}

	terms := terms(query)
	if len(terms) == 0 {
		return chunkContents(idx.chunks[:topK])
	}

	type scored struct {
		chunk Chunk
		score int
	}
	scoredChunks := make([]scored, len(idx.chunks))
	for i, chunk := range idx.chunks {
		lower := strings.ToLower(chunk.Content)
		score := 0
		for _, term := range terms {
			score += strings.Count(lower, term)
		}
		scoredChunks[i] = scored{chunk: chunk, score: score}
	}
	sort.SliceStable(scoredChunks, func(i, j int) bool {
		if scoredChunks[i].score == scoredChunks[j].score {
			return scoredChunks[i].chunk.ID < scoredChunks[j].chunk.ID
		}
		return scoredChunks[i].score > scoredChunks[j].score
	})
	if scoredChunks[0].score == 0 {
		return chunkContents(idx.chunks[:topK])
	}

	results := make([]string, 0, topK)
	for _, item := range scoredChunks {
		if item.score == 0 {
			break
		}
		results = append(results, item.chunk.Content)
		if len(results) == topK {
			break
		}
	}
	if len(results) == 0 {
		return chunkContents(idx.chunks[:topK])
	}
	return results
}

func terms(query string) []string {
	fields := strings.FieldsFunc(strings.ToLower(query), func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsNumber(r)
	})
	out := make([]string, 0, len(fields))
	seen := make(map[string]struct{}, len(fields))
	for _, field := range fields {
		if len(field) < 2 {
			continue
		}
		if _, ok := seen[field]; ok {
			continue
		}
		seen[field] = struct{}{}
		out = append(out, field)
	}
	return out
}

func chunkContents(chunks []Chunk) []string {
	results := make([]string, len(chunks))
	for i, chunk := range chunks {
		results[i] = chunk.Content
	}
	return results
}

// GetChunk returns the chunk content for id.
func (idx *Index) GetChunk(id int) (string, bool) {
	if idx == nil || id < 0 || id >= len(idx.chunks) {
		return "", false
	}
	return idx.chunks[id].Content, true
}

// GetContext returns a 1-indexed inclusive line range.
func (idx *Index) GetContext(startLine, endLine int) string {
	if idx == nil || len(idx.lines) == 0 {
		return ""
	}
	if startLine < 1 {
		startLine = 1
	}
	if startLine > len(idx.lines) {
		return ""
	}
	if endLine < startLine {
		endLine = startLine
	}
	if endLine > len(idx.lines) {
		endLine = len(idx.lines)
	}
	return strings.Join(idx.lines[startLine-1:endLine], "\n")
}

// ChunkCount returns the number of indexed chunks.
func (idx *Index) ChunkCount() int {
	if idx == nil {
		return 0
	}
	return len(idx.chunks)
}

// LineCount returns the number of context lines.
func (idx *Index) LineCount() int {
	if idx == nil || idx.raw == "" {
		return 0
	}
	return len(idx.lines)
}
