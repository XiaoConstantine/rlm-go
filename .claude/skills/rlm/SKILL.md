---
name: rlm
description: Recursive Language Model for processing large contexts (>50KB). Use for complex analysis tasks where token efficiency matters. Achieves 40% token savings by letting the LLM programmatically explore context via Query() and FINAL() patterns.
allowed-tools:
  - Bash
---

# RLM - Recursive Language Model

**RLM** is an inference-time scaling strategy that enables LLMs to handle arbitrarily long contexts by treating prompts as external objects that can be programmatically examined and recursively processed.

- **License:** MIT
- **Repository:** https://github.com/XiaoConstantine/rlm-go

## When to Use

Use `rlm` instead of direct LLM calls when:
- Processing **large contexts** (>50KB of text)
- Token efficiency is important (40% savings on large contexts)
- The task requires **iterative exploration** of data
- Complex analysis that benefits from sub-queries

## Do NOT Use When

- Context is small (<10KB) - overhead not worth it
- Simple single-turn questions
- Tasks that don't require data exploration

## Command Usage

```bash
# Basic usage with context file
~/.local/bin/rlm -context <file> -query "<query>" -verbose

# With inline context
~/.local/bin/rlm -context-string "data" -query "<query>"

# Pipe context from stdin
cat largefile.txt | ~/.local/bin/rlm -query "<query>"

# JSON output for programmatic use
~/.local/bin/rlm -context <file> -query "<query>" -json
```

## Options

| Flag | Description | Default |
|------|-------------|---------|
| `-context` | Path to context file | - |
| `-context-string` | Context string directly | - |
| `-query` | Query to run against context | Required |
| `-model` | LLM model to use | claude-sonnet-4-20250514 |
| `-max-iterations` | Maximum iterations | 30 |
| `-verbose` | Enable verbose output | false |
| `-json` | Output result as JSON | false |
| `-log-dir` | Directory for JSONL logs | - |

## How It Works

RLM uses a Go REPL environment where LLM-generated code can:

1. **Access context** as a string variable
2. **Make recursive sub-LLM calls** via `Query()` for focused analysis
3. **Use standard Go operations** for text processing
4. **Signal completion** with `FINAL()` when done

### The Query() Pattern

```go
// LLM generates code like this inside the REPL:
chunk := context[0:10000]
summary := Query("Summarize the key findings in this text: " + chunk)
// ... iterate through more chunks
FINAL(combinedResult)
```

### The FINAL() Pattern

The LLM signals completion by calling:
- `FINAL("answer")` - Return a string answer
- `FINAL_VAR(variableName)` - Return value of a variable

## Token Efficiency Benefits

For large contexts (>50KB), RLM typically achieves **40% token savings** by:
- Only sending relevant context chunks to sub-queries
- Avoiding repeated full-context processing
- Using programmatic iteration instead of full-context reasoning

## Examples

### Analyze Log Files
```bash
rlm -context server.log -query "Find all unique error patterns and their frequencies"
```

### Process JSON Data
```bash
rlm -context data.json -query "Extract all user IDs with failed transactions" -verbose
```

### Code Analysis
```bash
cat src/*.go | rlm -query "Identify all exported functions and their purposes"
```

## Requirements

- `ANTHROPIC_API_KEY` environment variable must be set
- Binary installed at `~/.local/bin/rlm`

## Installation

```bash
# Quick install
curl -fsSL https://raw.githubusercontent.com/XiaoConstantine/rlm-go/main/install.sh | bash

# Or with Go
go install github.com/XiaoConstantine/rlm-go/cmd/rlm@latest
```
