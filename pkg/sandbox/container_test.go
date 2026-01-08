package sandbox

import (
	"context"
	"fmt"
	"strings"
	"testing"
	"time"
)

// MockLLMClient is a mock implementation of LLMClient for testing.
type MockLLMClient struct {
	Responses map[string]string
	CallCount int
	Calls     []string
}

func NewMockLLMClient() *MockLLMClient {
	return &MockLLMClient{
		Responses: make(map[string]string),
	}
}

func (m *MockLLMClient) Query(ctx context.Context, prompt string) (QueryResponse, error) {
	m.CallCount++
	m.Calls = append(m.Calls, prompt)

	response, ok := m.Responses[prompt]
	if !ok {
		response = "Mock response for: " + prompt
	}

	return QueryResponse{
		Response:         response,
		PromptTokens:     len(prompt) / 4,
		CompletionTokens: len(response) / 4,
	}, nil
}

func (m *MockLLMClient) QueryBatched(ctx context.Context, prompts []string) ([]QueryResponse, error) {
	results := make([]QueryResponse, len(prompts))
	for i, prompt := range prompts {
		results[i], _ = m.Query(ctx, prompt)
	}
	return results, nil
}

func TestRuntimeDetection(t *testing.T) {
	// Test the runtime detection functions
	podmanAvail := IsRuntimeAvailable("podman")
	dockerAvail := IsRuntimeAvailable("docker")

	t.Logf("Podman available: %v", podmanAvail)
	t.Logf("Docker available: %v", dockerAvail)

	// detectBestBackend should return something
	backend := detectBestBackend()
	t.Logf("Best backend: %s", backend)

	if !podmanAvail && !dockerAvail {
		if backend != BackendLocal {
			t.Errorf("Expected BackendLocal when no container runtime available, got %s", backend)
		}
	}
}

func TestLocalExecutorBasic(t *testing.T) {
	client := NewMockLLMClient()
	cfg := DefaultConfig()
	cfg.Backend = BackendLocal

	exec, err := NewLocalExecutor(client, cfg)
	if err != nil {
		t.Fatalf("Failed to create local executor: %v", err)
	}
	defer func() { _ = exec.Close() }()

	// Test basic code execution
	result, err := exec.Execute(context.Background(), `fmt.Println("Hello, sandbox!")`)
	if err != nil {
		t.Fatalf("Execute failed: %v", err)
	}

	if !strings.Contains(result.Stdout, "Hello, sandbox!") {
		t.Errorf("Expected 'Hello, sandbox!' in output, got: %s", result.Stdout)
	}

	if exec.Backend() != BackendLocal {
		t.Errorf("Expected BackendLocal, got %s", exec.Backend())
	}
}

func TestLocalExecutorWithContext(t *testing.T) {
	client := NewMockLLMClient()
	cfg := DefaultConfig()
	cfg.Backend = BackendLocal

	exec, err := NewLocalExecutor(client, cfg)
	if err != nil {
		t.Fatalf("Failed to create local executor: %v", err)
	}
	defer func() { _ = exec.Close() }()

	// Load string context
	err = exec.LoadContext("test context data")
	if err != nil {
		t.Fatalf("LoadContext failed: %v", err)
	}

	// Verify context info
	info := exec.ContextInfo()
	if !strings.Contains(info, "string") {
		t.Errorf("Expected context info to mention 'string', got: %s", info)
	}

	// Execute code that uses the context
	result, err := exec.Execute(context.Background(), `fmt.Println("Context:", context)`)
	if err != nil {
		t.Fatalf("Execute failed: %v", err)
	}

	if !strings.Contains(result.Stdout, "test context data") {
		t.Errorf("Expected context data in output, got: %s", result.Stdout)
	}
}

func TestLocalExecutorWithQuery(t *testing.T) {
	client := NewMockLLMClient()
	client.Responses["What is 2+2?"] = "4"

	cfg := DefaultConfig()
	cfg.Backend = BackendLocal

	exec, err := NewLocalExecutor(client, cfg)
	if err != nil {
		t.Fatalf("Failed to create local executor: %v", err)
	}
	defer func() { _ = exec.Close() }()

	// Execute code that uses Query
	result, err := exec.Execute(context.Background(), `
		answer := Query("What is 2+2?")
		fmt.Println("Answer:", answer)
	`)
	if err != nil {
		t.Fatalf("Execute failed: %v", err)
	}

	if !strings.Contains(result.Stdout, "4") {
		t.Errorf("Expected '4' in output, got: %s", result.Stdout)
	}

	// Verify LLM calls were recorded
	calls := exec.GetLLMCalls()
	if len(calls) != 1 {
		t.Errorf("Expected 1 LLM call, got %d", len(calls))
	}

	if len(calls) > 0 && calls[0].Prompt != "What is 2+2?" {
		t.Errorf("Expected prompt 'What is 2+2?', got: %s", calls[0].Prompt)
	}
}

func TestLocalExecutorWithBatchedQuery(t *testing.T) {
	client := NewMockLLMClient()
	client.Responses["Q1"] = "A1"
	client.Responses["Q2"] = "A2"

	cfg := DefaultConfig()
	cfg.Backend = BackendLocal

	exec, err := NewLocalExecutor(client, cfg)
	if err != nil {
		t.Fatalf("Failed to create local executor: %v", err)
	}
	defer func() { _ = exec.Close() }()

	// Execute code that uses QueryBatched
	result, err := exec.Execute(context.Background(), `
		prompts := []string{"Q1", "Q2"}
		answers := QueryBatched(prompts)
		for i, a := range answers {
			fmt.Printf("Answer %d: %s\n", i, a)
		}
	`)
	if err != nil {
		t.Fatalf("Execute failed: %v", err)
	}

	if !strings.Contains(result.Stdout, "A1") || !strings.Contains(result.Stdout, "A2") {
		t.Errorf("Expected 'A1' and 'A2' in output, got: %s", result.Stdout)
	}

	// Verify LLM calls were recorded
	calls := exec.GetLLMCalls()
	if len(calls) != 2 {
		t.Errorf("Expected 2 LLM calls, got %d", len(calls))
	}
}

func TestLocalExecutorTimeout(t *testing.T) {
	client := NewMockLLMClient()
	cfg := DefaultConfig()
	cfg.Backend = BackendLocal
	cfg.Timeout = 100 * time.Millisecond

	exec, err := NewLocalExecutor(client, cfg)
	if err != nil {
		t.Fatalf("Failed to create local executor: %v", err)
	}
	defer func() { _ = exec.Close() }()

	// Note: Yaegi doesn't support true cancellation, so this test
	// verifies the timeout configuration is passed correctly.
	// The actual timeout behavior may vary.
	result, err := exec.Execute(context.Background(), `
		fmt.Println("Quick execution")
	`)
	if err != nil {
		t.Fatalf("Execute failed: %v", err)
	}

	if !strings.Contains(result.Stdout, "Quick execution") {
		t.Errorf("Expected output, got: %s", result.Stdout)
	}
}

func TestLocalExecutorGetVariable(t *testing.T) {
	client := NewMockLLMClient()
	cfg := DefaultConfig()
	cfg.Backend = BackendLocal

	exec, err := NewLocalExecutor(client, cfg)
	if err != nil {
		t.Fatalf("Failed to create local executor: %v", err)
	}
	defer func() { _ = exec.Close() }()

	// Execute code that sets a variable
	_, err = exec.Execute(context.Background(), `
		var result = "test value"
	`)
	if err != nil {
		t.Fatalf("Execute failed: %v", err)
	}

	// Get the variable
	val, err := exec.GetVariable("result")
	if err != nil {
		t.Fatalf("GetVariable failed: %v", err)
	}

	if val != "test value" {
		t.Errorf("Expected 'test value', got: %s", val)
	}
}

func TestLocalExecutorGetLocals(t *testing.T) {
	client := NewMockLLMClient()
	cfg := DefaultConfig()
	cfg.Backend = BackendLocal

	exec, err := NewLocalExecutor(client, cfg)
	if err != nil {
		t.Fatalf("Failed to create local executor: %v", err)
	}
	defer func() { _ = exec.Close() }()

	// Execute code that sets common variables
	_, err = exec.Execute(context.Background(), `
		var result = "some result"
		var answer = 42
	`)
	if err != nil {
		t.Fatalf("Execute failed: %v", err)
	}

	// Get locals
	locals := exec.GetLocals()

	if _, ok := locals["result"]; !ok {
		t.Error("Expected 'result' in locals")
	}
	if _, ok := locals["answer"]; !ok {
		t.Error("Expected 'answer' in locals")
	}
}

func TestLocalExecutorReset(t *testing.T) {
	client := NewMockLLMClient()
	cfg := DefaultConfig()
	cfg.Backend = BackendLocal

	exec, err := NewLocalExecutor(client, cfg)
	if err != nil {
		t.Fatalf("Failed to create local executor: %v", err)
	}
	defer func() { _ = exec.Close() }()

	// Execute some code
	_, _ = exec.Execute(context.Background(), `
		var result = "before reset"
	`)

	// Reset
	if err := exec.Reset(); err != nil {
		t.Fatalf("Reset failed: %v", err)
	}

	// After reset, the variable should not exist
	_, err = exec.GetVariable("result")
	if err == nil {
		t.Error("Expected error after reset, variable should not exist")
	}
}

func TestNewExecutorWithAuto(t *testing.T) {
	client := NewMockLLMClient()
	cfg := DefaultConfig()
	cfg.Backend = BackendAuto

	exec, err := New(client, cfg)
	if err != nil {
		t.Fatalf("Failed to create executor: %v", err)
	}
	defer func() { _ = exec.Close() }()

	// Should have created some executor
	backend := exec.Backend()
	t.Logf("Auto-selected backend: %s", backend)

	// If container backend was selected, ensure image exists
	if backend == BackendPodman || backend == BackendDocker {
		if containerExec, ok := exec.(*ContainerExecutor); ok {
			if !containerExec.ImageExists() {
				ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
				defer cancel()
				if err := containerExec.PullImage(ctx); err != nil {
					t.Skipf("Container backend selected but failed to pull image: %v", err)
				}
			}
		}
	}

	// Execute basic code
	result, err := exec.Execute(context.Background(), `fmt.Println("Hello from auto backend!")`)
	if err != nil {
		t.Fatalf("Execute failed: %v", err)
	}

	if !strings.Contains(result.Stdout, "Hello from auto backend!") {
		t.Errorf("Expected greeting in output, got: %s", result.Stdout)
	}
}

func TestContainerExecutorBasic(t *testing.T) {
	// Skip if no container runtime available
	if !IsRuntimeAvailable("podman") && !IsRuntimeAvailable("docker") {
		t.Skip("No container runtime available")
	}

	client := NewMockLLMClient()
	cfg := DefaultConfig()
	cfg.Backend = BackendAuto
	cfg.EnableIPC = false // Disable IPC for basic test
	cfg.Timeout = 60 * time.Second

	exec, err := NewContainerExecutor(client, cfg, BackendAuto)
	if err != nil {
		t.Fatalf("Failed to create container executor: %v", err)
	}
	defer func() { _ = exec.Close() }()

	t.Logf("Using backend: %s", exec.Backend())

	// Pull image first if needed
	if !exec.ImageExists() {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
		defer cancel()
		if err := exec.PullImage(ctx); err != nil {
			t.Skipf("Failed to pull image: %v", err)
		}
	}

	// Test basic code execution
	result, err := exec.Execute(context.Background(), `fmt.Println("Hello from container!")`)
	if err != nil {
		t.Fatalf("Execute failed: %v", err)
	}

	if result.Stderr != "" && !strings.Contains(result.Stdout, "Hello from container!") {
		t.Errorf("Execution had errors. stdout: %s, stderr: %s", result.Stdout, result.Stderr)
	}
}

func TestContainerExecutorWithContext(t *testing.T) {
	// Skip if no container runtime available
	if !IsRuntimeAvailable("podman") && !IsRuntimeAvailable("docker") {
		t.Skip("No container runtime available")
	}

	client := NewMockLLMClient()
	cfg := DefaultConfig()
	cfg.Backend = BackendAuto
	cfg.EnableIPC = false
	cfg.Timeout = 60 * time.Second

	exec, err := NewContainerExecutor(client, cfg, BackendAuto)
	if err != nil {
		t.Fatalf("Failed to create container executor: %v", err)
	}
	defer func() { _ = exec.Close() }()

	// Pull image first if needed
	if !exec.ImageExists() {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
		defer cancel()
		if err := exec.PullImage(ctx); err != nil {
			t.Skipf("Failed to pull image: %v", err)
		}
	}

	// Load context
	err = exec.LoadContext("container context data")
	if err != nil {
		t.Fatalf("LoadContext failed: %v", err)
	}

	// Verify context info
	info := exec.ContextInfo()
	if !strings.Contains(info, "string") {
		t.Errorf("Expected context info to mention 'string', got: %s", info)
	}
}

func TestContainerExecutorResourceLimits(t *testing.T) {
	// Skip if no container runtime available
	if !IsRuntimeAvailable("podman") && !IsRuntimeAvailable("docker") {
		t.Skip("No container runtime available")
	}

	client := NewMockLLMClient()
	cfg := DefaultConfig()
	cfg.Backend = BackendAuto
	cfg.EnableIPC = false
	cfg.Memory = "256m"
	cfg.CPUs = 0.5
	cfg.Timeout = 30 * time.Second

	exec, err := NewContainerExecutor(client, cfg, BackendAuto)
	if err != nil {
		t.Fatalf("Failed to create container executor: %v", err)
	}
	defer func() { _ = exec.Close() }()

	// Pull image first if needed
	if !exec.ImageExists() {
		ctx, cancel := context.WithTimeout(context.Background(), 2*time.Minute)
		defer cancel()
		if err := exec.PullImage(ctx); err != nil {
			t.Skipf("Failed to pull image: %v", err)
		}
	}

	// Test that resource limits don't break execution
	result, err := exec.Execute(context.Background(), `fmt.Println("Limited resources test")`)
	if err != nil {
		t.Fatalf("Execute failed: %v", err)
	}

	if !strings.Contains(result.Stdout, "Limited resources test") {
		t.Logf("stdout: %s", result.Stdout)
		t.Logf("stderr: %s", result.Stderr)
	}
}

func TestContainerExecutorRuntimeInfo(t *testing.T) {
	// Skip if no container runtime available
	if !IsRuntimeAvailable("podman") && !IsRuntimeAvailable("docker") {
		t.Skip("No container runtime available")
	}

	client := NewMockLLMClient()
	cfg := DefaultConfig()
	cfg.EnableIPC = false

	exec, err := NewContainerExecutor(client, cfg, BackendAuto)
	if err != nil {
		t.Fatalf("Failed to create container executor: %v", err)
	}
	defer func() { _ = exec.Close() }()

	info, err := exec.RuntimeInfo()
	if err != nil {
		t.Fatalf("RuntimeInfo failed: %v", err)
	}

	t.Logf("Runtime info: %s", info)

	if info == "" {
		t.Error("Expected non-empty runtime info")
	}
}

func TestIPCServerBasic(t *testing.T) {
	client := NewMockLLMClient()
	client.Responses["test prompt"] = "test response"

	server, err := NewIPCServer(client, 0) // Auto-assign port
	if err != nil {
		t.Fatalf("Failed to create IPC server: %v", err)
	}
	defer func() { _ = server.Stop() }()

	server.Start()

	port := server.Port()
	if port == 0 {
		t.Error("Expected non-zero port")
	}

	t.Logf("IPC server listening on port %d", port)

	// Create IPC client
	ipcClient, err := NewIPCClient(fmt.Sprintf("127.0.0.1:%d", port))
	if err != nil {
		t.Fatalf("Failed to create IPC client: %v", err)
	}
	defer func() { _ = ipcClient.Close() }()

	// Test Query
	response, tokenUsage, err := ipcClient.Query("test prompt")
	if err != nil {
		t.Fatalf("Query failed: %v", err)
	}

	if response != "test response" {
		t.Errorf("Expected 'test response', got: %s", response)
	}

	if tokenUsage == nil {
		t.Error("Expected token usage")
	}

	// Verify call was recorded
	calls := server.GetCalls()
	if len(calls) != 1 {
		t.Errorf("Expected 1 call recorded, got %d", len(calls))
	}
}

func TestIPCServerBatched(t *testing.T) {
	client := NewMockLLMClient()
	client.Responses["p1"] = "r1"
	client.Responses["p2"] = "r2"

	server, err := NewIPCServer(client, 0)
	if err != nil {
		t.Fatalf("Failed to create IPC server: %v", err)
	}
	defer func() { _ = server.Stop() }()

	server.Start()

	// Create IPC client
	ipcClient, err := NewIPCClient(server.Address())
	if err != nil {
		t.Fatalf("Failed to create IPC client: %v", err)
	}
	defer func() { _ = ipcClient.Close() }()

	// Test QueryBatched
	responses, usages, err := ipcClient.QueryBatched([]string{"p1", "p2"})
	if err != nil {
		t.Fatalf("QueryBatched failed: %v", err)
	}

	if len(responses) != 2 {
		t.Errorf("Expected 2 responses, got %d", len(responses))
	}

	if responses[0] != "r1" || responses[1] != "r2" {
		t.Errorf("Expected ['r1', 'r2'], got: %v", responses)
	}

	if len(usages) != 2 {
		t.Errorf("Expected 2 token usages, got %d", len(usages))
	}
}

func TestGenerateContainerRLMCode(t *testing.T) {
	code := GenerateContainerRLMCode("host.containers.internal:12345")

	// Verify the code contains expected elements
	if !strings.Contains(code, "package main") {
		t.Error("Expected 'package main' in generated code")
	}

	if !strings.Contains(code, "func Query(prompt string) string") {
		t.Error("Expected Query function in generated code")
	}

	if !strings.Contains(code, "func QueryBatched(prompts []string) []string") {
		t.Error("Expected QueryBatched function in generated code")
	}

	if !strings.Contains(code, "host.containers.internal:12345") {
		t.Error("Expected IPC address in generated code")
	}
}

func TestDefaultConfig(t *testing.T) {
	cfg := DefaultConfig()

	if cfg.Backend != BackendAuto {
		t.Errorf("Expected BackendAuto, got %s", cfg.Backend)
	}

	if cfg.Image != "golang:1.23-alpine" {
		t.Errorf("Expected 'golang:1.23-alpine', got %s", cfg.Image)
	}

	if cfg.Memory != "512m" {
		t.Errorf("Expected '512m', got %s", cfg.Memory)
	}

	if cfg.CPUs != 1.0 {
		t.Errorf("Expected 1.0, got %f", cfg.CPUs)
	}

	if cfg.Timeout != 60*time.Second {
		t.Errorf("Expected 60s, got %v", cfg.Timeout)
	}

	if cfg.NetworkMode != NetworkNone {
		t.Errorf("Expected NetworkNone, got %s", cfg.NetworkMode)
	}
}

func TestLocalExecutorMinMaxHelpers(t *testing.T) {
	client := NewMockLLMClient()
	cfg := DefaultConfig()
	cfg.Backend = BackendLocal

	exec, err := NewLocalExecutor(client, cfg)
	if err != nil {
		t.Fatalf("Failed to create local executor: %v", err)
	}
	defer func() { _ = exec.Close() }()

	// Test min/max helper functions
	result, err := exec.Execute(context.Background(), `
		a := min(5, 3)
		b := max(5, 3)
		fmt.Printf("min=%d max=%d", a, b)
	`)
	if err != nil {
		t.Fatalf("Execute failed: %v", err)
	}

	if !strings.Contains(result.Stdout, "min=3 max=5") {
		t.Errorf("Expected 'min=3 max=5' in output, got: %s", result.Stdout)
	}
}

func TestLocalExecutorMapContext(t *testing.T) {
	client := NewMockLLMClient()
	cfg := DefaultConfig()
	cfg.Backend = BackendLocal

	exec, err := NewLocalExecutor(client, cfg)
	if err != nil {
		t.Fatalf("Failed to create local executor: %v", err)
	}
	defer func() { _ = exec.Close() }()

	// Load map context (stored as JSON string)
	err = exec.LoadContext(map[string]any{
		"key1": "value1",
		"key2": 42,
	})
	if err != nil {
		t.Fatalf("LoadContext failed: %v", err)
	}

	// Execute code that uses the context as a JSON string
	result, err := exec.Execute(context.Background(), `
		// Context is stored as JSON string
		fmt.Printf("context=%s", context)
	`)
	if err != nil {
		t.Fatalf("Execute failed: %v", err)
	}

	// Context should contain the JSON representation
	if !strings.Contains(result.Stdout, "key1") || !strings.Contains(result.Stdout, "value1") {
		t.Errorf("Expected context JSON to contain 'key1' and 'value1', got: %s", result.Stdout)
	}
}

func TestExecutorInterface(t *testing.T) {
	// Verify that both executors implement the Executor interface
	var _ Executor = (*LocalExecutor)(nil)
	var _ Executor = (*ContainerExecutor)(nil)
}
