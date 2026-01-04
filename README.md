# rlm-go

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

## Installation

```bash
go get github.com/XiaoConstantine/rlm-go
```

## Quick Start

```go
package main

import (
    "context"
    "fmt"

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
    result, err := r.Complete(ctx, longDocument, "What are the key findings?")
    if err != nil {
        log.Fatal(err)
    }

    fmt.Printf("Answer: %s\n", result.Response)
    fmt.Printf("Iterations: %d\n", result.Iterations)
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
    Complete(ctx context.Context, messages []core.Message) (string, error)
}

// For sub-LLM calls from REPL
type REPLClient interface {
    Query(ctx context.Context, prompt string) (string, error)
    QueryBatched(ctx context.Context, prompts []string) ([]string, error)
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
Query(prompt string) string           // Single sub-LLM call
QueryBatched(prompts []string) []string  // Concurrent sub-LLM calls

// Your context
context  // string variable with your data
```

## Configuration

```go
rlm.New(client, replClient,
    rlm.WithMaxIterations(30),      // Default: 30
    rlm.WithSystemPrompt(custom),   // Override system prompt
    rlm.WithVerbose(true),          // Enable logging
)
```

## References

- [Recursive Language Models Paper](https://arxiv.org/abs/2512.24601) (Zhang, Kraska, Khattab - MIT CSAIL)
- [Python RLM Implementation](https://github.com/alexzhang13/rlm)
- [Yaegi Go Interpreter](https://github.com/traefik/yaegi)

## License

MIT License - see [LICENSE](LICENSE)
