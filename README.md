# rlm-go

[![CI](https://github.com/XiaoConstantine/rlm-go/actions/workflows/go.yml/badge.svg)](https://github.com/XiaoConstantine/rlm-go/actions/workflows/go.yml)
[![Go Report Card](https://goreportcard.com/badge/github.com/XiaoConstantine/rlm-go)](https://goreportcard.com/report/github.com/XiaoConstantine/rlm-go)
[![Go Reference](https://pkg.go.dev/badge/github.com/XiaoConstantine/rlm-go.svg)](https://pkg.go.dev/github.com/XiaoConstantine/rlm-go)

A Go implementation of [Recursive Language Models (RLM)](https://github.com/alexzhang13/rlm) - an inference-time scaling strategy that enables LLMs to handle arbitrarily long contexts by treating prompts as external objects that can be programmatically examined and recursively processed.

## Overview

RLM-go provides a Go REPL environment where LLM-generated code can:
- Access context stored as a variable
- Make recursive sub-LLM calls via `Query()` and `QueryBatched()`
- Use standard Go operations for text processing
- Signal completion with `FINAL()` or `FINAL_VAR()`

### Key Design Decisions

Unlike the Python RLM which uses socket IPC, rlm-go uses **direct function injection** via [Yaegi](https://github.com/traefik/yaegi) - a Go interpreter. This eliminates:
- Socket server overhead
- Serialization/deserialization
- Process boundaries

The result is ~100x less latency per sub-LLM call compared to socket IPC.

## Requirements

- Go 1.23 or later (for building from source)
- An LLM API key:
  - `ANTHROPIC_API_KEY` for Claude models (default)
  - `GEMINI_API_KEY` for Gemini models
  - `OPENAI_API_KEY` for OpenAI models
- (Optional) Podman or Docker for isolated sandbox execution

## Supported Models

| Provider | Models | Env Variable |
|----------|--------|--------------|
| Anthropic | claude-sonnet-4-20250514, claude-opus-4-20250514, etc. | `ANTHROPIC_API_KEY` |
| Google | gemini-3-flash-preview, gemini-3-pro-preview | `GEMINI_API_KEY` |
| OpenAI | gpt-5, gpt-5-mini | `OPENAI_API_KEY` |

The provider is auto-detected based on model name. Anthropic is the default.

## Installation

### Quick Install (Recommended)

```bash
# Download and install the latest release
curl -fsSL https://raw.githubusercontent.com/XiaoConstantine/rlm-go/main/install.sh | bash
```

This installs the `rlm` binary to `~/.local/bin/rlm`.

### Go Install

```bash
go install github.com/XiaoConstantine/rlm-go/cmd/rlm@latest
```

### From Source

```bash
git clone https://github.com/XiaoConstantine/rlm-go.git
cd rlm-go
go build -o rlm ./cmd/rlm
```

### As a Library

```bash
go get github.com/XiaoConstantine/rlm-go
```

## Claude Code Integration

RLM includes a skill for [Claude Code](https://claude.com/claude-code) that provides documentation and usage guidance for large context processing.

### Install the Skill

```bash
rlm install-claude-code
```

This creates a skill at `~/.claude/skills/rlm/SKILL.md` that teaches Claude Code:
- When to use RLM (contexts >50KB, token efficiency needed)
- Command usage and options
- The Query() and FINAL() patterns
- Token efficiency benefits (40% savings on large contexts)

After installation, restart Claude Code to activate the skill.

## CLI Usage

```bash
# Basic usage with Anthropic (default)
rlm -context file.txt -query "Summarize the key points"

# Use Gemini
rlm -model gemini-3-flash-preview -context file.txt -query "Analyze this data"

# Use OpenAI
rlm -model gpt-5-mini -context file.txt -query "Summarize this"

# Verbose output with iteration details
rlm -context logs.json -query "Find all errors" -verbose

# JSON output for programmatic use
rlm -context data.csv -query "Extract anomalies" -json

# Pipe context from stdin
cat largefile.txt | rlm -query "What patterns do you see?"
```

### CLI Options

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

## Quick Start (Library)

```go
package main

import (
    "context"
    "fmt"
    "log"
    "os"

    "github.com/XiaoConstantine/rlm-go/pkg/rlm"
)

func main() {
    // Create your LLM client (implements rlm.LLMClient and repl.LLMClient)
    client := NewAnthropicClient(os.Getenv("ANTHROPIC_API_KEY"), "claude-sonnet-4-20250514")

    // Create RLM instance
    r := rlm.New(client, client,
        rlm.WithMaxIterations(10),
        rlm.WithVerbose(true),
    )

    // Run completion with long context
    result, err := r.Complete(context.Background(), longDocument, "What are the key findings?")
    if err != nil {
        log.Fatal(err)
    }

    fmt.Printf("Answer: %s\n", result.Response)
    fmt.Printf("Iterations: %d\n", result.Iterations)
    fmt.Printf("Total Tokens: %d\n", result.TotalUsage.TotalTokens)
}
```

## Architecture

```
┌──────────────────────────────────────────────┐
│              Single Go Process               │
│                                              │
│  ┌──────────────┐    ┌──────────────────┐   │
│  │    Yaegi     │───►│   LLM Client     │   │
│  │  Interpreter │    │ (your impl)      │   │
│  │              │    └──────────────────┘   │
│  │ - context    │                           │
│  │ - Query()    │◄── direct function call   │
│  │ - fmt.*      │                           │
│  └──────────────┘                           │
│         ▲                                   │
│         │ Eval(code)                        │
│  ┌──────┴──────┐                            │
│  │  RLM Loop   │                            │
│  └─────────────┘                            │
└──────────────────────────────────────────────┘

No sockets. No IPC. No subprocess.
```

## Interfaces

You need to implement two interfaces:

```go
// For root LLM orchestration
type LLMClient interface {
    Complete(ctx context.Context, messages []core.Message) (core.LLMResponse, error)
}

// For sub-LLM calls from REPL
type REPLClient interface {
    Query(ctx context.Context, prompt string) (repl.QueryResponse, error)
    QueryBatched(ctx context.Context, prompts []string) ([]repl.QueryResponse, error)
}
```

See [examples/basic](examples/basic/main.go) for a complete Anthropic client implementation.

## How It Works

1. **Context Loading**: Your context is injected into a Yaegi interpreter as a `context` variable
2. **Iteration Loop**: The root LLM generates Go code in ` ```go ` blocks
3. **Code Execution**: Yaegi executes the code with access to `Query()`, `QueryBatched()`, `fmt`, `strings`, `regexp`
4. **Sub-LLM Calls**: `Query()` calls your LLM client directly (no IPC)
5. **Completion**: LLM signals done with `FINAL("answer")` or `FINAL_VAR(varName)`

## Available in REPL

```go
// Pre-imported packages
fmt, strings, regexp

// RLM functions
Query(prompt string) string              // Single sub-LLM call
QueryBatched(prompts []string) []string  // Concurrent sub-LLM calls

// Multi-depth recursion (when enabled)
QueryWithRLM(prompt string, depth int) string  // Spawn nested RLM
CurrentDepth() int                              // Get current recursion depth
MaxDepth() int                                  // Get max allowed depth
CanRecurse() bool                               // Check if more recursion allowed

// Your context
context  // string variable with your data
```

### Multi-Depth Recursion

Enable sub-LLMs to spawn their own sub-LLMs for complex decomposition tasks:

```go
r := rlm.New(client, replClient,
    rlm.WithMaxRecursionDepth(3),  // Allow 3 levels of nesting
    rlm.WithRecursionCallback(func(depth int, prompt string) {
        log.Printf("Recursive call at depth %d", depth)
    }),
)
```

In the REPL, use `QueryWithRLM()` to spawn a nested RLM that can itself use `Query()`:
```go
// Depth 0 (root)
result := QueryWithRLM("Analyze each section in detail", 1)

// The sub-RLM (depth 1) can use Query() or QueryWithRLM() up to MaxDepth
```

## Configuration

```go
rlm.New(client, replClient,
    rlm.WithMaxIterations(30),      // Default: 30
    rlm.WithSystemPrompt(custom),   // Override system prompt
    rlm.WithVerbose(true),          // Enable console logging
    rlm.WithLogger(logger),         // Attach JSONL logger for session recording
)
```

## Sandbox Execution (Podman/Docker)

By default, rlm-go executes LLM-generated code in-process using Yaegi for maximum performance. For production environments or when running untrusted code, you can enable isolated sandbox execution using Podman (recommended) or Docker.

### Why Sandbox?

| Mode | Isolation | Latency | Security |
|------|-----------|---------|----------|
| Local (default) | None | ~0ms | Trusted code only |
| Podman/Docker | Full container | 50-200ms | Untrusted code safe |

### Setup

**Podman (Recommended - Open Source, Daemonless)**
```bash
# macOS
brew install podman
podman machine init
podman machine start

# Linux (Fedora/RHEL)
sudo dnf install podman

# Linux (Ubuntu/Debian)
sudo apt install podman
```

**Docker (Alternative)**
```bash
# macOS
brew install --cask docker
# Start Docker Desktop

# Linux
sudo apt install docker.io
sudo systemctl start docker
```

### Usage

```go
import (
    "github.com/XiaoConstantine/rlm-go/pkg/rlm"
    "github.com/XiaoConstantine/rlm-go/pkg/sandbox"
)

// Auto-detect best available backend (podman > docker > local)
r := rlm.New(client, replClient, rlm.WithSandbox())

// Use specific backend
r := rlm.New(client, replClient,
    rlm.WithSandboxBackend(sandbox.BackendPodman))

// Custom configuration
cfg := sandbox.Config{
    Backend:     sandbox.BackendPodman,
    Image:       "golang:1.23-alpine",  // Container image
    Memory:      "512m",                // Memory limit
    CPUs:        1.0,                   // CPU limit
    Timeout:     60 * time.Second,      // Execution timeout
    NetworkMode: sandbox.NetworkNone,   // Disable network (secure)
}
r := rlm.New(client, replClient, rlm.WithSandboxConfig(cfg))
```

### Container Behavior

When sandbox mode is enabled:
- Code runs in an isolated container with resource limits
- Network is disabled by default (`--network=none`)
- Containers are auto-removed after execution (`--rm`)
- `Query()` and `QueryBatched()` work via JSON IPC protocol
- First execution pulls the Go image (~250MB, cached after)

### Verifying Setup

```bash
# Check if Podman/Docker is available
podman --version  # or: docker --version

# Test container execution
podman run --rm golang:1.23-alpine go version

# Run sandbox tests
go test ./pkg/sandbox/... -v -run TestContainer
```

## Token Tracking

RLM-go provides complete token usage accounting across all LLM calls:

```go
result, _ := r.Complete(ctx, context, query)

// Aggregated token usage
fmt.Printf("Prompt tokens: %d\n", result.TotalUsage.PromptTokens)
fmt.Printf("Completion tokens: %d\n", result.TotalUsage.CompletionTokens)
fmt.Printf("Total tokens: %d\n", result.TotalUsage.TotalTokens)
```

Token counts include both root LLM calls and all sub-LLM calls made via `Query()` and `QueryBatched()`.

## JSONL Logging

Record sessions for analysis and visualization:

```go
import "github.com/XiaoConstantine/rlm-go/pkg/logger"

// Create logger
log, _ := logger.New("./logs", "session-001")
defer log.Close()

// Attach to RLM
r := rlm.New(client, replClient, rlm.WithLogger(log))
```

Log format includes:
- Session metadata (model, max iterations, context info)
- Per-iteration details (prompts, responses, executed code)
- Sub-LLM call records with token counts
- Compatible with the [Python RLM visualizer](https://github.com/alexzhang13/rlm)

## CLI Tools

### Benchmark Tool

Compare RLM accuracy against baseline direct LLM calls:

```bash
go run ./cmd/benchmark/main.go \
  -tasks tasks.json \
  -model claude-sonnet-4-20250514 \
  -num-tasks 10 \
  -log-dir ./logs \
  -output results.json \
  -verbose
```

Features:
- Load tasks from JSON or generate samples
- Track accuracy, execution time, token usage
- Flexible answer matching (exact, word-boundary, numeric)

### Log Viewer

Interactive CLI viewer for JSONL session logs:

```bash
go run ./cmd/rlm-viewer/main.go ./logs/session.jsonl

# Watch mode for real-time viewing
go run ./cmd/rlm-viewer/main.go -watch ./logs/session.jsonl

# Filter by iteration
go run ./cmd/rlm-viewer/main.go -iter 3 ./logs/session.jsonl

# Interactive navigation mode
go run ./cmd/rlm-viewer/main.go -interactive ./logs/session.jsonl
```

Features:
- Color-coded output (system, user, assistant messages)
- Code block display with execution results
- Token usage tracking per LLM call
- Interactive navigation

## Package Structure

```
rlm-go/
├── pkg/
│   ├── core/      # Core types (Message, CompletionResult, UsageStats)
│   ├── rlm/       # Main RLM orchestration engine
│   ├── repl/      # Yaegi-based Go interpreter
│   ├── sandbox/   # Isolated execution (Podman/Docker)
│   ├── parsing/   # LLM response parsing utilities
│   └── logger/    # JSONL session logging
├── cmd/
│   ├── benchmark/ # RLM vs baseline comparison tool
│   └── rlm-viewer/ # JSONL log viewer
└── examples/
    └── basic/     # Complete Anthropic client example
```

## Testing

```bash
# Run all tests
go test -v ./...

# Run with race detection and coverage
go test -race -v ./... -coverprofile coverage.txt
```

## References

- [Recursive Language Models Paper](https://arxiv.org/abs/2512.24601) (Zhang, Kraska, Khattab - MIT CSAIL)
- [Python RLM Implementation](https://github.com/alexzhang13/rlm)
- [Yaegi Go Interpreter](https://github.com/traefik/yaegi)

## License

MIT License - see [LICENSE](LICENSE)
