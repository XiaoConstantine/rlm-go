// Package sandbox provides isolated execution environments for RLM code.
// It supports local execution via Yaegi and containerized execution via Podman/Docker.
package sandbox

import (
	"context"
	"time"

	"github.com/XiaoConstantine/rlm-go/pkg/core"
)

// Backend specifies the sandbox execution backend.
type Backend string

const (
	// BackendLocal uses Yaegi for in-process execution (fastest, least isolation).
	BackendLocal Backend = "local"

	// BackendPodman uses Podman for container execution (rootless, secure).
	BackendPodman Backend = "podman"

	// BackendDocker uses Docker for container execution (fallback).
	BackendDocker Backend = "docker"

	// BackendAuto automatically selects the best available backend.
	// Priority: podman > docker > local
	BackendAuto Backend = "auto"
)

// NetworkMode specifies the container network configuration.
type NetworkMode string

const (
	// NetworkNone disables networking (most secure).
	NetworkNone NetworkMode = "none"

	// NetworkBridge enables bridged networking (for IPC callbacks).
	NetworkBridge NetworkMode = "bridge"

	// NetworkHost uses host networking (least secure, fastest).
	NetworkHost NetworkMode = "host"
)

// Config configures a sandbox executor.
type Config struct {
	// Backend specifies which execution backend to use.
	// Default: "auto" (tries podman, then docker, then local).
	Backend Backend

	// Image is the container image to use (e.g., "golang:1.23-alpine").
	// Only used for container backends.
	// Default: "golang:1.23-alpine"
	Image string

	// Memory is the memory limit (e.g., "512m", "1g").
	// Only used for container backends.
	// Default: "512m"
	Memory string

	// CPUs is the CPU limit (e.g., 0.5, 1.0, 2.0).
	// Only used for container backends.
	// Default: 1.0
	CPUs float64

	// Timeout is the maximum execution time per code block.
	// Default: 60s
	Timeout time.Duration

	// NetworkMode configures container networking.
	// Default: "none" (no network access)
	NetworkMode NetworkMode

	// WorkDir is the working directory inside the container.
	// Default: "/workspace"
	WorkDir string

	// EnableIPC enables JSON-based IPC for Query() callbacks.
	// When enabled, the host starts an IPC server and the container can call back.
	// Default: true for container backends
	EnableIPC bool

	// IPCPort is the port for IPC communication.
	// Only used when EnableIPC is true.
	// Default: 0 (auto-assign)
	IPCPort int

	// Verbose enables verbose logging of sandbox operations.
	Verbose bool
}

// DefaultConfig returns a Config with sensible defaults.
func DefaultConfig() Config {
	return Config{
		Backend:     BackendAuto,
		Image:       "golang:1.23-alpine",
		Memory:      "512m",
		CPUs:        1.0,
		Timeout:     60 * time.Second,
		NetworkMode: NetworkNone,
		WorkDir:     "/workspace",
		EnableIPC:   true,
		IPCPort:     0,
		Verbose:     false,
	}
}

// LLMClient defines the interface for making LLM calls from within the sandbox.
// This mirrors repl.LLMClient for compatibility.
type LLMClient interface {
	Query(ctx context.Context, prompt string) (QueryResponse, error)
	QueryBatched(ctx context.Context, prompts []string) ([]QueryResponse, error)
}

// QueryResponse contains the LLM response with usage metadata.
type QueryResponse struct {
	Response         string
	PromptTokens     int
	CompletionTokens int
}

// LLMCall represents a sub-LLM call made from within the sandbox.
type LLMCall struct {
	Prompt           string  `json:"prompt"`
	Response         string  `json:"response"`
	Duration         float64 `json:"duration"`
	PromptTokens     int     `json:"prompt_tokens"`
	CompletionTokens int     `json:"completion_tokens"`
	Async            bool    `json:"async,omitempty"`
}

// Executor defines the interface for sandbox execution.
// All sandbox backends (local, podman, docker) implement this interface.
type Executor interface {
	// Execute runs Go code and returns the result.
	// The code may contain Query() calls that will be intercepted and routed to the LLM client.
	Execute(ctx context.Context, code string) (*core.ExecutionResult, error)

	// LoadContext injects the context payload into the execution environment.
	// The context is available as the `context` variable in executed code.
	LoadContext(payload any) error

	// GetVariable retrieves a variable value from the execution environment.
	GetVariable(name string) (string, error)

	// GetLLMCalls returns the LLM calls made during the last execution.
	GetLLMCalls() []LLMCall

	// GetLocals returns user-defined variables from the execution environment.
	GetLocals() map[string]any

	// ContextInfo returns metadata about the loaded context.
	ContextInfo() string

	// Reset clears the execution state.
	Reset() error

	// Close releases resources.
	Close() error

	// Backend returns the backend type being used.
	Backend() Backend
}

// New creates a new Executor based on the configuration.
// If Backend is "auto", it probes for available backends in order:
// podman > docker > local
func New(client LLMClient, cfg Config) (Executor, error) {
	backend := cfg.Backend
	if backend == BackendAuto {
		backend = detectBestBackend()
	}

	switch backend {
	case BackendPodman, BackendDocker:
		return NewContainerExecutor(client, cfg, backend)
	case BackendLocal:
		return NewLocalExecutor(client, cfg)
	default:
		// Default to local
		return NewLocalExecutor(client, cfg)
	}
}

// detectBestBackend probes the system for available container runtimes.
func detectBestBackend() Backend {
	if IsRuntimeAvailable("podman") {
		return BackendPodman
	}
	if IsRuntimeAvailable("docker") {
		return BackendDocker
	}
	return BackendLocal
}
