package contextindex

import (
	"strings"
	"testing"
	"unicode/utf8"
)

func TestIndexLineAndChunkHelpers(t *testing.T) {
	raw := "alpha first\nbeta second\nalpha beta third"
	idx := New(raw)

	if got := idx.LineCount(); got != 3 {
		t.Fatalf("LineCount() = %d, want 3", got)
	}
	if got := idx.ChunkCount(); got != 1 {
		t.Fatalf("ChunkCount() = %d, want 1", got)
	}
	if got := idx.GetContext(2, 20); got != "beta second\nalpha beta third" {
		t.Fatalf("GetContext() = %q", got)
	}
	if got := idx.GetContext(0, 0); got != "alpha first" {
		t.Fatalf("clamped GetContext() = %q", got)
	}
	if got := idx.GetContext(99, 99); got != "" {
		t.Fatalf("out-of-range GetContext() = %q, want empty", got)
	}
	chunk, ok := idx.GetChunk(0)
	if !ok || chunk != raw {
		t.Fatalf("GetChunk(0) = %q/%v, want raw/true", chunk, ok)
	}
	if _, ok := idx.GetChunk(1); ok {
		t.Fatal("GetChunk(1) should be absent")
	}
}

func TestFindRelevantScoresAndFallsBack(t *testing.T) {
	raw := strings.Repeat("alpha filler\n", 360) + "needle target target\n" + strings.Repeat("omega filler\n", 360)
	idx := New(raw)

	results := idx.FindRelevant("target?", 1)
	if len(results) != 1 {
		t.Fatalf("FindRelevant returned %d results, want 1", len(results))
	}
	if !strings.Contains(results[0], "needle target target") {
		t.Fatalf("FindRelevant result = %q, want target chunk", results[0])
	}

	fallback := idx.FindRelevant("missing", 2)
	if len(fallback) != 2 {
		t.Fatalf("fallback result count = %d, want 2", len(fallback))
	}
	if fallback[0] == "" || fallback[1] == "" {
		t.Fatalf("fallback should return non-empty chunks: %q", fallback)
	}

	if got := FindRelevant(raw, "needle,target", 1); len(got) != 1 || !strings.Contains(got[0], "needle target") {
		t.Fatalf("package FindRelevant() = %q, want punctuation-insensitive target chunk", got)
	}
	if got := LineCount("a\nb"); got != 2 {
		t.Fatalf("package LineCount() = %d, want 2", got)
	}
	if got := GetContext("a\nb\nc", 2, 3); got != "b\nc" {
		t.Fatalf("package GetContext() = %q", got)
	}
	if got := GetContext("a\nb\nc", 99, 99); got != "" {
		t.Fatalf("package out-of-range GetContext() = %q, want empty", got)
	}
}

func TestChunksDoNotSplitUTF8Runes(t *testing.T) {
	raw := strings.Repeat("a", defaultChunkChars-1) + "🙂" + strings.Repeat("b", defaultChunkOverlap+1)
	idx := New(raw)

	if got := idx.ChunkCount(); got < 2 {
		t.Fatalf("ChunkCount() = %d, want at least 2", got)
	}
	for i := 0; i < idx.ChunkCount(); i++ {
		chunk, ok := idx.GetChunk(i)
		if !ok {
			t.Fatalf("GetChunk(%d) not found", i)
		}
		if !utf8.ValidString(chunk) {
			t.Fatalf("GetChunk(%d) is invalid UTF-8: %q", i, chunk)
		}
	}

	if idx.chunks[0].EndChar != defaultChunkChars {
		t.Fatalf("first chunk EndChar = %d, want %d rune offset", idx.chunks[0].EndChar, defaultChunkChars)
	}
	if !strings.Contains(idx.chunks[0].Content, "🙂") {
		t.Fatalf("first chunk = %q, want complete emoji", idx.chunks[0].Content)
	}
}

func TestStringify(t *testing.T) {
	if got := Stringify("plain"); got != "plain" {
		t.Fatalf("Stringify(string) = %q", got)
	}
	if got := Stringify([]byte("bytes")); got != "bytes" {
		t.Fatalf("Stringify([]byte) = %q", got)
	}
	if got := Stringify(map[string]any{"b": 2}); got != `{"b":2}` {
		t.Fatalf("Stringify(map) = %q", got)
	}
}
