package sandbox

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"time"

	"github.com/XiaoConstantine/rlm-go/pkg/core"
)

// ContainerExecutor provides isolated code execution using Podman or Docker.
type ContainerExecutor struct {
	client      LLMClient
	config      Config
	backend     Backend
	runtime     string // "podman" or "docker" command name
	ipcServer   *IPCServer
	contextData any
	mu          sync.Mutex
	variables   map[string]any
}

// NewContainerExecutor creates a new container-based executor.
// If backend is BackendAuto, it probes for available runtimes.
func NewContainerExecutor(client LLMClient, cfg Config, backend Backend) (*ContainerExecutor, error) {
	// Determine the runtime command
	runtime := ""
	actualBackend := backend

	if backend == BackendAuto {
		actualBackend, runtime = detectContainerRuntime()
		if actualBackend == BackendLocal {
			return nil, fmt.Errorf("no container runtime (podman or docker) found")
		}
	} else {
		switch backend {
		case BackendPodman:
			if !IsRuntimeAvailable("podman") {
				return nil, fmt.Errorf("podman is not available")
			}
			runtime = "podman"
		case BackendDocker:
			if !IsRuntimeAvailable("docker") {
				return nil, fmt.Errorf("docker is not available")
			}
			runtime = "docker"
		default:
			return nil, fmt.Errorf("unsupported backend: %s", backend)
		}
	}

	// Start IPC server if enabled
	var ipcServer *IPCServer
	if cfg.EnableIPC {
		var err error
		ipcServer, err = NewIPCServer(client, cfg.IPCPort)
		if err != nil {
			return nil, fmt.Errorf("failed to start IPC server: %w", err)
		}
		ipcServer.Start()
	}

	return &ContainerExecutor{
		client:    client,
		config:    cfg,
		backend:   actualBackend,
		runtime:   runtime,
		ipcServer: ipcServer,
		variables: make(map[string]any),
	}, nil
}

// detectContainerRuntime probes for available container runtimes.
func detectContainerRuntime() (Backend, string) {
	if IsRuntimeAvailable("podman") {
		return BackendPodman, "podman"
	}
	if IsRuntimeAvailable("docker") {
		return BackendDocker, "docker"
	}
	return BackendLocal, ""
}

// IsRuntimeAvailable checks if a container runtime is installed and working.
func IsRuntimeAvailable(runtime string) bool {
	cmd := exec.Command(runtime, "--version")
	return cmd.Run() == nil
}

// Execute runs Go code in a container and returns the result.
func (e *ContainerExecutor) Execute(ctx context.Context, code string) (*core.ExecutionResult, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	start := time.Now()

	// Create temporary directory for code
	tmpDir, err := os.MkdirTemp("", "rlm-sandbox-*")
	if err != nil {
		return nil, fmt.Errorf("failed to create temp dir: %w", err)
	}
	defer func() { _ = os.RemoveAll(tmpDir) }()

	// Generate the full Go program
	program, err := e.generateProgram(code)
	if err != nil {
		return &core.ExecutionResult{
			Stderr:   fmt.Sprintf("failed to generate program: %v", err),
			Duration: time.Since(start),
		}, nil
	}

	// Write the program to a file
	mainFile := filepath.Join(tmpDir, "main.go")
	if err := os.WriteFile(mainFile, []byte(program), 0644); err != nil {
		return nil, fmt.Errorf("failed to write main.go: %w", err)
	}

	// Write go.mod
	goMod := "module sandbox\n\ngo 1.23\n"
	goModFile := filepath.Join(tmpDir, "go.mod")
	if err := os.WriteFile(goModFile, []byte(goMod), 0644); err != nil {
		return nil, fmt.Errorf("failed to write go.mod: %w", err)
	}

	// Build container command
	args := e.buildContainerArgs(tmpDir)

	// Create the command with timeout context
	timeout := e.config.Timeout
	if timeout == 0 {
		timeout = 60 * time.Second
	}
	cmdCtx, cancel := context.WithTimeout(ctx, timeout)
	defer cancel()

	cmd := exec.CommandContext(cmdCtx, e.runtime, args...)

	var stdout, stderr bytes.Buffer
	cmd.Stdout = &stdout
	cmd.Stderr = &stderr

	if e.config.Verbose {
		fmt.Printf("[sandbox] Running: %s %s\n", e.runtime, strings.Join(args, " "))
	}

	// Run the container
	runErr := cmd.Run()

	result := &core.ExecutionResult{
		Stdout:   stdout.String(),
		Stderr:   stderr.String(),
		Duration: time.Since(start),
	}

	// Check for timeout
	if cmdCtx.Err() == context.DeadlineExceeded {
		result.Stderr = "execution timeout exceeded\n" + result.Stderr
	} else if runErr != nil {
		if result.Stderr != "" {
			result.Stderr += "\n"
		}
		result.Stderr += runErr.Error()
	}

	// Extract variables from stdout (we encode them as JSON at the end)
	e.extractVariables(result.Stdout)

	return result, nil
}

// generateProgram creates a complete Go program from the code snippet.
func (e *ContainerExecutor) generateProgram(code string) (string, error) {
	var sb strings.Builder

	// Add IPC code if enabled
	if e.config.EnableIPC && e.ipcServer != nil {
		ipcAddr := e.getIPCAddress()
		sb.WriteString(GenerateContainerRLMCode(ipcAddr))
	} else {
		// Add stub functions if IPC is disabled
		sb.WriteString(`package main

import "fmt"

// Query stub - IPC disabled
func Query(prompt string) string {
	return "Error: IPC disabled, cannot call Query()"
}

// QueryBatched stub - IPC disabled
func QueryBatched(prompts []string) []string {
	results := make([]string, len(prompts))
	for i := range results {
		results[i] = "Error: IPC disabled, cannot call QueryBatched()"
	}
	return results
}
`)
	}

	// Add context variable
	sb.WriteString("\n// Context data\n")
	if e.contextData != nil {
		contextJSON, err := json.Marshal(e.contextData)
		if err != nil {
			return "", fmt.Errorf("failed to marshal context: %w", err)
		}
		sb.WriteString(fmt.Sprintf("var context = %s\n", strconv.Quote(string(contextJSON))))
	} else {
		sb.WriteString("var context = \"\"\n")
	}

	// Add helper functions
	sb.WriteString(`
// min returns the smaller of two integers
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// max returns the larger of two integers
func max(a, b int) int {
	if a > b {
		return a
	}
	return b
}

`)

	// Add the user code wrapped in main
	sb.WriteString("func main() {\n")

	// Indent the user code
	lines := strings.Split(code, "\n")
	for _, line := range lines {
		sb.WriteString("\t")
		sb.WriteString(line)
		sb.WriteString("\n")
	}

	sb.WriteString("}\n")

	return sb.String(), nil
}

// getIPCAddress returns the address for IPC communication.
func (e *ContainerExecutor) getIPCAddress() string {
	if e.ipcServer == nil {
		return ""
	}

	// For container access, we need to use host.docker.internal or host.containers.internal
	// depending on the runtime. Alternatively, we can use the host IP.
	port := e.ipcServer.Port()

	switch e.backend {
	case BackendPodman:
		// Podman uses host.containers.internal for host access
		return fmt.Sprintf("host.containers.internal:%d", port)
	case BackendDocker:
		// Docker uses host.docker.internal for host access
		return fmt.Sprintf("host.docker.internal:%d", port)
	default:
		// Fallback to localhost (may not work in all container configurations)
		return fmt.Sprintf("127.0.0.1:%d", port)
	}
}

// buildContainerArgs constructs the container run command arguments.
func (e *ContainerExecutor) buildContainerArgs(tmpDir string) []string {
	args := []string{"run", "--rm"}

	// Add resource limits
	if e.config.Memory != "" {
		args = append(args, "--memory", e.config.Memory)
	}
	if e.config.CPUs > 0 {
		args = append(args, "--cpus", fmt.Sprintf("%.2f", e.config.CPUs))
	}

	// Add network mode
	switch e.config.NetworkMode {
	case NetworkNone:
		if !e.config.EnableIPC {
			// Only disable network if we don't need IPC
			args = append(args, "--network", "none")
		}
	case NetworkBridge:
		args = append(args, "--network", "bridge")
	case NetworkHost:
		args = append(args, "--network", "host")
	}

	// Add host access for IPC if enabled
	if e.config.EnableIPC {
		switch e.backend {
		case BackendPodman:
			// Podman automatically provides host.containers.internal
			args = append(args, "--add-host", "host.containers.internal:host-gateway")
		case BackendDocker:
			// Docker automatically provides host.docker.internal on Mac/Windows
			// For Linux, we may need to add it manually
			args = append(args, "--add-host", "host.docker.internal:host-gateway")
		}
	}

	// Mount the code directory
	args = append(args, "-v", fmt.Sprintf("%s:/workspace:ro", tmpDir))

	// Set working directory
	args = append(args, "-w", "/workspace")

	// Add the image
	image := e.config.Image
	if image == "" {
		image = "golang:1.23-alpine"
	}
	args = append(args, image)

	// Add the command to run
	args = append(args, "go", "run", "main.go")

	return args
}

// extractVariables parses output to extract variable values.
func (e *ContainerExecutor) extractVariables(output string) {
	// Look for JSON-encoded variable exports at the end of output
	// Format: __RLM_VARS__{"name": "value", ...}
	marker := "__RLM_VARS__"
	idx := strings.LastIndex(output, marker)
	if idx == -1 {
		return
	}

	jsonStr := output[idx+len(marker):]
	var vars map[string]any
	if err := json.Unmarshal([]byte(jsonStr), &vars); err != nil {
		return
	}

	for k, v := range vars {
		e.variables[k] = v
	}
}

// LoadContext stores the context payload for injection into the container.
func (e *ContainerExecutor) LoadContext(payload any) error {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.contextData = payload
	return nil
}

// GetVariable retrieves a variable value.
func (e *ContainerExecutor) GetVariable(name string) (string, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	if v, ok := e.variables[name]; ok {
		return fmt.Sprintf("%v", v), nil
	}
	return "", fmt.Errorf("variable %q not found", name)
}

// GetLLMCalls returns and clears the recorded LLM calls.
func (e *ContainerExecutor) GetLLMCalls() []LLMCall {
	if e.ipcServer == nil {
		return nil
	}
	return e.ipcServer.GetCalls()
}

// GetLocals returns user-defined variables.
func (e *ContainerExecutor) GetLocals() map[string]any {
	e.mu.Lock()
	defer e.mu.Unlock()

	result := make(map[string]any)
	for k, v := range e.variables {
		result[k] = v
	}
	return result
}

// ContextInfo returns metadata about the loaded context.
func (e *ContainerExecutor) ContextInfo() string {
	e.mu.Lock()
	defer e.mu.Unlock()

	if e.contextData == nil {
		return "context not loaded"
	}

	switch v := e.contextData.(type) {
	case string:
		return fmt.Sprintf("type=string, len=%d", len(v))
	case []any:
		return fmt.Sprintf("type=array, len=%d", len(v))
	case map[string]any:
		return fmt.Sprintf("type=object, keys=%d", len(v))
	default:
		return fmt.Sprintf("type=%T", v)
	}
}

// Reset clears the execution state.
func (e *ContainerExecutor) Reset() error {
	e.mu.Lock()
	defer e.mu.Unlock()

	e.variables = make(map[string]any)
	e.contextData = nil

	if e.ipcServer != nil {
		e.ipcServer.ClearCalls()
	}

	return nil
}

// Close releases resources.
func (e *ContainerExecutor) Close() error {
	e.mu.Lock()
	defer e.mu.Unlock()

	if e.ipcServer != nil {
		return e.ipcServer.Stop()
	}
	return nil
}

// Backend returns the backend type.
func (e *ContainerExecutor) Backend() Backend {
	return e.backend
}

// PullImage pulls the container image if not already present.
func (e *ContainerExecutor) PullImage(ctx context.Context) error {
	image := e.config.Image
	if image == "" {
		image = "golang:1.23-alpine"
	}

	cmd := exec.CommandContext(ctx, e.runtime, "pull", image)
	return cmd.Run()
}

// ImageExists checks if the container image exists locally.
func (e *ContainerExecutor) ImageExists() bool {
	image := e.config.Image
	if image == "" {
		image = "golang:1.23-alpine"
	}

	cmd := exec.Command(e.runtime, "image", "inspect", image)
	return cmd.Run() == nil
}

// RuntimeInfo returns information about the container runtime.
func (e *ContainerExecutor) RuntimeInfo() (string, error) {
	cmd := exec.Command(e.runtime, "--version")
	output, err := cmd.Output()
	if err != nil {
		return "", err
	}
	return strings.TrimSpace(string(output)), nil
}
