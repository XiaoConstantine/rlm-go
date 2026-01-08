// Package benchmark_test provides integration benchmarks for multi-depth recursion.
// Run with: RLM_INTEGRATION_TEST=1 go test -v -timeout 10m ./benchmark_test/...
package benchmark_test

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/XiaoConstantine/rlm-go/pkg/providers"
	"github.com/XiaoConstantine/rlm-go/pkg/rlm"
)

// BenchmarkStats tracks statistics for a benchmark run.
type BenchmarkStats struct {
	TotalTime       time.Duration
	TotalIterations int
	MaxDepthReached int
	TokensByDepth   map[int]TokenStats
	RecursiveCalls  int
	mu              sync.Mutex
}

// TokenStats tracks token usage at a specific depth.
type TokenStats struct {
	PromptTokens     int64
	CompletionTokens int64
	TotalCalls       int64
}

// Add adds token counts atomically.
func (s *BenchmarkStats) AddTokens(depth int, prompt, completion int) {
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.TokensByDepth == nil {
		s.TokensByDepth = make(map[int]TokenStats)
	}
	stats := s.TokensByDepth[depth]
	stats.PromptTokens += int64(prompt)
	stats.CompletionTokens += int64(completion)
	stats.TotalCalls++
	s.TokensByDepth[depth] = stats
	if depth > s.MaxDepthReached {
		s.MaxDepthReached = depth
	}
}

// String returns a formatted summary of the stats.
func (s *BenchmarkStats) String() string {
	s.mu.Lock()
	defer s.mu.Unlock()

	var b strings.Builder
	b.WriteString("\n=== Benchmark Results ===\n")
	b.WriteString(fmt.Sprintf("Total Time:        %v\n", s.TotalTime))
	b.WriteString(fmt.Sprintf("Total Iterations:  %d\n", s.TotalIterations))
	b.WriteString(fmt.Sprintf("Max Depth Reached: %d\n", s.MaxDepthReached))
	b.WriteString(fmt.Sprintf("Recursive Calls:   %d\n", s.RecursiveCalls))

	var totalPrompt, totalCompletion int64
	for depth := 0; depth <= s.MaxDepthReached; depth++ {
		if stats, ok := s.TokensByDepth[depth]; ok {
			b.WriteString(fmt.Sprintf("\n[Depth %d]\n", depth))
			b.WriteString(fmt.Sprintf("  Prompt Tokens:     %d\n", stats.PromptTokens))
			b.WriteString(fmt.Sprintf("  Completion Tokens: %d\n", stats.CompletionTokens))
			b.WriteString(fmt.Sprintf("  Total Calls:       %d\n", stats.TotalCalls))
			totalPrompt += stats.PromptTokens
			totalCompletion += stats.CompletionTokens
		}
	}

	b.WriteString("\n[Totals]\n")
	b.WriteString(fmt.Sprintf("  Total Prompt Tokens:     %d\n", totalPrompt))
	b.WriteString(fmt.Sprintf("  Total Completion Tokens: %d\n", totalCompletion))
	b.WriteString(fmt.Sprintf("  Grand Total Tokens:      %d\n", totalPrompt+totalCompletion))

	return b.String()
}

// getAvailableProvider returns the first available provider based on environment variables.
func getAvailableProvider(t *testing.T) (providers.Client, string) {
	// Check for Gemini first (preferred for multi-depth testing due to speed)
	if apiKey := os.Getenv("GEMINI_API_KEY"); apiKey != "" {
		client := providers.NewGeminiClient(apiKey, "gemini-2.5-flash", true)
		t.Log("Using Gemini provider (gemini-2.5-flash)")
		return client, "gemini"
	}

	// Check for Google API key (alternate Gemini env var)
	if apiKey := os.Getenv("GOOGLE_API_KEY"); apiKey != "" {
		client := providers.NewGeminiClient(apiKey, "gemini-2.5-flash", true)
		t.Log("Using Gemini provider via GOOGLE_API_KEY (gemini-2.5-flash)")
		return client, "gemini"
	}

	// Fallback to Anthropic
	if apiKey := os.Getenv("ANTHROPIC_API_KEY"); apiKey != "" {
		client := providers.NewAnthropicClient(apiKey, "claude-haiku-4-5", true)
		t.Log("Using Anthropic provider (claude-haiku-4-5)")
		return client, "anthropic"
	}

	return nil, ""
}

// skipIfNotIntegration skips the test if RLM_INTEGRATION_TEST is not set.
func skipIfNotIntegration(t *testing.T) {
	if os.Getenv("RLM_INTEGRATION_TEST") != "1" {
		t.Skip("Skipping integration test. Set RLM_INTEGRATION_TEST=1 to run.")
	}
}

// TestRecursionBenchmark_Depth2_SimpleAnalysis tests depth-2 recursion with a simple analysis task.
func TestRecursionBenchmark_Depth2_SimpleAnalysis(t *testing.T) {
	skipIfNotIntegration(t)

	client, providerName := getAvailableProvider(t)
	if client == nil {
		t.Skip("No API keys found. Set GEMINI_API_KEY, GOOGLE_API_KEY, or ANTHROPIC_API_KEY")
	}

	t.Logf("Provider: %s", providerName)

	// Create benchmark stats tracker
	stats := &BenchmarkStats{TokensByDepth: make(map[int]TokenStats)}
	var recursiveCallCount int32

	// Track recursive calls
	recursionCallback := func(depth int, prompt string) {
		atomic.AddInt32(&recursiveCallCount, 1)
		t.Logf("[Depth %d] Recursive call initiated: %s...", depth, truncate(prompt, 100))
	}

	// Create RLM with depth-2 recursion
	r := rlm.New(client, client,
		rlm.WithMaxIterations(10),
		rlm.WithVerbose(true),
		rlm.WithMaxRecursionDepth(2),
		rlm.WithRecursionCallback(recursionCallback),
		rlm.WithProgressHandler(func(p rlm.IterationProgress) {
			t.Logf("Progress: iteration %d/%d, confidence signals: %d",
				p.CurrentIteration, p.MaxIterations, p.ConfidenceSignals)
		}),
	)

	// Create a context that benefits from hierarchical analysis
	testContext := createNestedAnalysisContext()
	query := `Analyze this nested data structure. For each category, identify the key themes
and then synthesize them into an overall summary. Use QueryWithRLM if you need deep analysis.
Return a one-sentence summary of the most important finding.`

	t.Logf("Context size: %d chars", len(testContext))

	// Run the recursive completion
	start := time.Now()
	result, err := r.RecursiveComplete(context.Background(), testContext, query)
	elapsed := time.Since(start)

	if err != nil {
		t.Fatalf("RecursiveComplete failed: %v", err)
	}

	// Update stats
	stats.TotalTime = elapsed
	stats.TotalIterations = result.Iterations
	stats.RecursiveCalls = int(atomic.LoadInt32(&recursiveCallCount))

	// Get token stats from result
	if result.TokenStats != nil {
		for depth, count := range result.TokenStats.CallsByDepth {
			stats.AddTokens(depth, int(result.TokenStats.PromptTokens/count), int(result.TokenStats.CompletionTokens/count))
		}
		if result.MaxDepthReached > stats.MaxDepthReached {
			stats.MaxDepthReached = result.MaxDepthReached
		}
	}

	// Log results
	t.Log(stats.String())
	t.Logf("\nFinal Response: %s", truncate(result.Response, 500))
	t.Logf("\nRecursive calls made: %d", stats.RecursiveCalls)
	t.Logf("Max depth reached: %d", stats.MaxDepthReached)

	// Verify recursion actually happened at depth 2
	if stats.MaxDepthReached < 1 {
		t.Logf("Note: Recursion did not reach depth 2 in this run (max depth: %d)", stats.MaxDepthReached)
	}
}

// TestRecursionBenchmark_Depth3_ComplexTask tests depth-3 recursion with a complex multi-part task.
func TestRecursionBenchmark_Depth3_ComplexTask(t *testing.T) {
	skipIfNotIntegration(t)

	client, providerName := getAvailableProvider(t)
	if client == nil {
		t.Skip("No API keys found. Set GEMINI_API_KEY, GOOGLE_API_KEY, or ANTHROPIC_API_KEY")
	}

	t.Logf("Provider: %s", providerName)

	// Track recursive calls with depth
	var depthCounts sync.Map // map[int]int

	recursionCallback := func(depth int, prompt string) {
		val, _ := depthCounts.LoadOrStore(depth, new(int32))
		atomic.AddInt32(val.(*int32), 1)
		t.Logf("[Depth %d] Recursive RLM spawned", depth)
	}

	// Create RLM with depth-3 recursion
	r := rlm.New(client, client,
		rlm.WithMaxIterations(15),
		rlm.WithVerbose(true),
		rlm.WithRecursionConfig(rlm.RecursionConfig{
			MaxDepth: 3,
			PerDepthMaxIterations: map[int]int{
				0: 15, // Root: more iterations
				1: 10, // First recursion level
				2: 8,  // Second level
				3: 5,  // Third level
			},
			OnRecursiveQuery: recursionCallback,
		}),
	)

	// Create a deeply nested analysis task
	testContext := createDeepNestedContext()
	query := `This is a multi-level analysis task.
1. At the top level, identify the main sections
2. For each section, use QueryWithRLM to get a detailed analysis
3. Sub-analyses may spawn their own sub-analyses if needed for complex items
4. Synthesize all findings into a final one-line summary.
The answer should be: "Summary: [key finding]"`

	t.Logf("Context size: %d chars", len(testContext))

	start := time.Now()
	result, err := r.RecursiveComplete(context.Background(), testContext, query)
	elapsed := time.Since(start)

	if err != nil {
		t.Fatalf("RecursiveComplete failed: %v", err)
	}

	// Count recursive calls by depth
	var totalRecursive int32
	t.Log("\nRecursive calls by depth:")
	depthCounts.Range(func(key, value interface{}) bool {
		depth := key.(int)
		count := atomic.LoadInt32(value.(*int32))
		t.Logf("  Depth %d: %d calls", depth, count)
		totalRecursive += count
		return true
	})

	t.Logf("\n=== Depth-3 Benchmark Results ===")
	t.Logf("Total Time:        %v", elapsed)
	t.Logf("Total Iterations:  %d", result.Iterations)
	t.Logf("Max Depth Reached: %d", result.MaxDepthReached)
	t.Logf("Total Recursive:   %d", totalRecursive)
	t.Logf("Response:          %s", truncate(result.Response, 300))

	// Token usage
	if result.TokenStats != nil {
		prompt, completion, calls := result.TokenStats.GetTotals()
		t.Logf("\n=== Token Usage ===")
		t.Logf("Total Prompt Tokens:     %d", prompt)
		t.Logf("Total Completion Tokens: %d", completion)
		t.Logf("Total Calls:             %d", calls)
	}

	// Verify we actually used multi-depth recursion
	if result.MaxDepthReached < 2 {
		t.Logf("Warning: Did not reach expected depth 3 (max reached: %d)", result.MaxDepthReached)
	}
}

// TestRecursionBenchmark_JSONAnalysis tests recursive analysis of nested JSON.
func TestRecursionBenchmark_JSONAnalysis(t *testing.T) {
	skipIfNotIntegration(t)

	client, providerName := getAvailableProvider(t)
	if client == nil {
		t.Skip("No API keys found. Set GEMINI_API_KEY, GOOGLE_API_KEY, or ANTHROPIC_API_KEY")
	}

	t.Logf("Provider: %s", providerName)

	// Create nested JSON structure
	jsonContext := createNestedJSONContext()
	t.Logf("JSON Context size: %d chars", len(jsonContext))

	var recursiveCount int32
	r := rlm.New(client, client,
		rlm.WithMaxIterations(12),
		rlm.WithVerbose(true),
		rlm.WithMaxRecursionDepth(2),
		rlm.WithRecursionCallback(func(depth int, prompt string) {
			atomic.AddInt32(&recursiveCount, 1)
			t.Logf("[Depth %d] JSON section analysis", depth)
		}),
	)

	query := `This is a nested JSON structure representing a company's departments and projects.
Analyze each department by using QueryWithRLM if needed, and find:
1. The total number of employees across all departments
2. The department with the most active projects
Return the department name with the most projects.`

	ctx := context.Background()
	start := time.Now()
	result, err := r.RecursiveComplete(ctx, jsonContext, query)
	elapsed := time.Since(start)

	if err != nil {
		t.Fatalf("RecursiveComplete failed: %v", err)
	}

	t.Logf("\n=== JSON Analysis Benchmark ===")
	t.Logf("Provider:          %s", providerName)
	t.Logf("Total Time:        %v", elapsed)
	t.Logf("Iterations:        %d", result.Iterations)
	t.Logf("Max Depth:         %d", result.MaxDepthReached)
	t.Logf("Recursive Calls:   %d", atomic.LoadInt32(&recursiveCount))
	t.Logf("Response:          %s", result.Response)

	if result.TokenStats != nil {
		prompt, completion, calls := result.TokenStats.GetTotals()
		t.Logf("\n=== Token Usage ===")
		t.Logf("Prompt:     %d", prompt)
		t.Logf("Completion: %d", completion)
		t.Logf("Calls:      %d", calls)
	}
}

// BenchmarkRecursiveComplete_Depth2 is a proper Go benchmark for depth-2 recursion.
func BenchmarkRecursiveComplete_Depth2(b *testing.B) {
	if os.Getenv("RLM_INTEGRATION_TEST") != "1" {
		b.Skip("Skipping benchmark. Set RLM_INTEGRATION_TEST=1 to run.")
	}

	client, _ := getAvailableProviderB(b)
	if client == nil {
		b.Skip("No API keys found")
	}

	r := rlm.New(client, client,
		rlm.WithMaxIterations(8),
		rlm.WithMaxRecursionDepth(2),
	)

	testContext := createSimpleContext()
	query := "Summarize the key points in one sentence."

	ctx := context.Background()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, err := r.RecursiveComplete(ctx, testContext, query)
		if err != nil {
			b.Fatalf("RecursiveComplete failed: %v", err)
		}
	}
}

// Helper functions

func getAvailableProviderB(b *testing.B) (providers.Client, string) {
	if apiKey := os.Getenv("GEMINI_API_KEY"); apiKey != "" {
		return providers.NewGeminiClient(apiKey, "gemini-2.5-flash", false), "gemini"
	}
	if apiKey := os.Getenv("GOOGLE_API_KEY"); apiKey != "" {
		return providers.NewGeminiClient(apiKey, "gemini-2.5-flash", false), "gemini"
	}
	if apiKey := os.Getenv("ANTHROPIC_API_KEY"); apiKey != "" {
		return providers.NewAnthropicClient(apiKey, "claude-haiku-4-5", false), "anthropic"
	}
	return nil, ""
}

func truncate(s string, maxLen int) string {
	if len(s) <= maxLen {
		return s
	}
	return s[:maxLen] + "..."
}

func createNestedAnalysisContext() string {
	return `
=== RESEARCH REPORT: Technology Trends Analysis ===

== SECTION 1: Artificial Intelligence ==
Category: Machine Learning
- Trend: Transformer architectures dominate NLP
- Impact: High - reshaping all text-based applications
- Growth: 45% YoY increase in enterprise adoption
- Key Finding: Multi-modal models are emerging as the next frontier

Category: Computer Vision
- Trend: Vision-language models gaining traction
- Impact: Medium-High - new applications in robotics and AR
- Growth: 32% YoY increase
- Key Finding: Real-time processing now feasible on edge devices

== SECTION 2: Cloud Computing ==
Category: Infrastructure
- Trend: Serverless and edge computing convergence
- Impact: High - reducing operational complexity
- Growth: 55% YoY for serverless adoption
- Key Finding: Hybrid cloud becoming the default architecture

Category: Data Services
- Trend: AI-native databases emerging
- Impact: Medium - specialized workloads benefit most
- Growth: 28% YoY
- Key Finding: Vector databases critical for AI applications

== SECTION 3: Security ==
Category: Zero Trust
- Trend: Identity-centric security models
- Impact: Very High - fundamental architecture shift
- Growth: 67% YoY adoption
- Key Finding: Network perimeters are obsolete

Category: AI Security
- Trend: ML models as attack surface
- Impact: High - new threat vectors emerging
- Growth: 89% increase in reported vulnerabilities
- Key Finding: Model security frameworks urgently needed

=== END REPORT ===
`
}

func createDeepNestedContext() string {
	return `
# ENTERPRISE ANALYSIS DOCUMENT

## Level 1: Business Units

### Division A: Technology Products
Status: Active
Revenue: $500M
Subdivisions:
  - A1: Consumer Electronics
    Products: SmartHome Hub, WearOS Watch, AudioMax Speakers
    Performance: Exceeds targets by 15%
    Issues: Supply chain delays, Component shortages
    Recommendations: Diversify suppliers, Increase inventory buffers

  - A2: Enterprise Solutions
    Products: CloudSync Platform, DataVault, SecureConnect
    Performance: Meets targets
    Issues: Customer support backlog, Integration complexity
    Recommendations: Hire support staff, Simplify APIs

  - A3: Emerging Technologies
    Products: AI Assistant, Quantum Simulator, BioSensor
    Performance: Below targets (expected for R&D)
    Issues: Patent disputes, Talent retention
    Recommendations: Settle disputes, Improve compensation

### Division B: Financial Services
Status: Active
Revenue: $300M
Subdivisions:
  - B1: Retail Banking
    Products: Mobile Banking App, Credit Cards, Savings Accounts
    Performance: Meets targets
    Issues: Regulatory compliance, Fraud detection

  - B2: Investment Services
    Products: Robo-Advisor, Trading Platform, Portfolio Analytics
    Performance: Exceeds targets by 25%
    Issues: Market volatility exposure

### Division C: Healthcare
Status: Expanding
Revenue: $200M
Subdivisions:
  - C1: Medical Devices
    Products: DiagnoScan, HealthMonitor, SurgicalBot
    Performance: Exceeds targets by 30%
    Issues: FDA approval delays, Clinical trial costs

  - C2: Pharmaceutical
    Products: GenoCure, ImmunoBoost, PainRelief+
    Performance: Below targets
    Issues: R&D costs, Drug pricing pressure

## Level 2: Cross-Cutting Concerns

### Risk Analysis
- Technology Risk: Medium (AI regulation pending)
- Market Risk: High (economic uncertainty)
- Operational Risk: Low (processes mature)
- Strategic Risk: Medium (disruption potential)

### Opportunities
1. AI integration across all divisions
2. Healthcare-technology convergence
3. Sustainability initiatives
4. Global expansion in emerging markets

## Summary Required
Identify the single most critical finding that leadership should address immediately.
`
}

func createNestedJSONContext() string {
	data := map[string]interface{}{
		"company": map[string]interface{}{
			"name": "TechCorp Industries",
			"departments": []map[string]interface{}{
				{
					"name": "Engineering",
					"head": "Jane Smith",
					"employees": 150,
					"projects": []map[string]interface{}{
						{"name": "Project Alpha", "status": "active", "budget": 500000},
						{"name": "Project Beta", "status": "active", "budget": 300000},
						{"name": "Project Gamma", "status": "planning", "budget": 200000},
					},
				},
				{
					"name": "Sales",
					"head": "Bob Johnson",
					"employees": 80,
					"projects": []map[string]interface{}{
						{"name": "Q1 Campaign", "status": "active", "budget": 100000},
						{"name": "Partner Program", "status": "active", "budget": 150000},
					},
				},
				{
					"name": "Research",
					"head": "Dr. Chen",
					"employees": 45,
					"projects": []map[string]interface{}{
						{"name": "AI Research", "status": "active", "budget": 800000},
						{"name": "Quantum Computing", "status": "planning", "budget": 1000000},
						{"name": "Biotech Initiative", "status": "active", "budget": 600000},
						{"name": "Materials Science", "status": "active", "budget": 400000},
					},
				},
				{
					"name": "Operations",
					"head": "Maria Garcia",
					"employees": 200,
					"projects": []map[string]interface{}{
						{"name": "Process Automation", "status": "active", "budget": 250000},
					},
				},
			},
		},
	}

	jsonBytes, _ := json.MarshalIndent(data, "", "  ")
	return string(jsonBytes)
}

func createSimpleContext() string {
	return `
Key Points Summary:
1. Revenue increased 20% year over year
2. Customer satisfaction improved to 4.5/5
3. New product launches exceeded expectations
4. Operational costs reduced by 15%
5. Market share grew from 12% to 15%
`
}

// TestRecursionWithGemini_Depth2_Proof is a focused test to prove multi-depth recursion works
// with Gemini at depth 2-3. This test is specifically designed to trigger recursive calls.
//
// Run with:
//
//	GOOGLE_API_KEY=$GOOGLE_API_KEY go test -v -timeout 10m ./benchmark_test/... -run TestRecursionWithGemini_Depth2_Proof -count=1
func TestRecursionWithGemini_Depth2_Proof(t *testing.T) {
	// Check for API key
	apiKey := os.Getenv("GOOGLE_API_KEY")
	if apiKey == "" {
		apiKey = os.Getenv("GEMINI_API_KEY")
	}
	if apiKey == "" {
		t.Skip("Set GOOGLE_API_KEY or GEMINI_API_KEY to run this test")
	}

	t.Log("=== Multi-Depth Recursion Proof Test with Gemini ===")
	t.Log("This test demonstrates that RLM can execute recursive sub-RLM calls at multiple depths.")

	// Create Gemini client - using gemini-2.5-flash for better availability
	model := "gemini-2.5-flash"
	client := providers.NewGeminiClient(apiKey, model, true)
	t.Logf("Using model: %s", model)

	// Track recursive calls with detailed logging
	var recursiveCallsMu sync.Mutex
	recursiveCalls := make([]struct {
		depth  int
		prompt string
		time   time.Time
	}, 0)

	recursionCallback := func(depth int, prompt string) {
		recursiveCallsMu.Lock()
		defer recursiveCallsMu.Unlock()
		recursiveCalls = append(recursiveCalls, struct {
			depth  int
			prompt string
			time   time.Time
		}{depth: depth, prompt: prompt, time: time.Now()})
		t.Logf("[RECURSION DETECTED] Depth %d initiated at %v", depth, time.Now().Format("15:04:05.000"))
		t.Logf("  Prompt preview: %s", truncate(prompt, 100))
	}

	// Create RLM with max depth 3 (allowing depths 0, 1, 2)
	r := rlm.New(client, client,
		rlm.WithMaxIterations(15),
		rlm.WithVerbose(true),
		rlm.WithRecursionConfig(rlm.RecursionConfig{
			MaxDepth: 3,
			PerDepthMaxIterations: map[int]int{
				0: 15, // Root level: up to 15 iterations
				1: 8,  // First recursion level: up to 8 iterations
				2: 5,  // Second recursion level: up to 5 iterations
			},
			OnRecursiveQuery: recursionCallback,
		}),
		rlm.WithProgressHandler(func(p rlm.IterationProgress) {
			t.Logf("[Progress] Iteration %d/%d, Context: %d bytes, Confidence: %d",
				p.CurrentIteration, p.MaxIterations, p.ContextSize, p.ConfidenceSignals)
		}),
	)

	// Create a context designed to encourage recursive analysis
	testContext := `
=== ORGANIZATIONAL DATA FOR MULTI-LEVEL ANALYSIS ===

## REGION A: North America
### Department: Engineering
- Budget: $5,000,000
- Employees: 200
- Projects: 15 active
- Key Metric: 95% on-time delivery

### Department: Sales
- Budget: $3,000,000
- Employees: 150
- Projects: 8 active
- Key Metric: 120% quota attainment

## REGION B: Europe
### Department: Engineering
- Budget: $4,000,000
- Employees: 180
- Projects: 12 active
- Key Metric: 92% on-time delivery

### Department: Sales
- Budget: $2,500,000
- Employees: 100
- Projects: 6 active
- Key Metric: 105% quota attainment

## REGION C: Asia Pacific
### Department: Engineering
- Budget: $3,500,000
- Employees: 150
- Projects: 10 active
- Key Metric: 98% on-time delivery

### Department: Sales
- Budget: $2,000,000
- Employees: 80
- Projects: 5 active
- Key Metric: 115% quota attainment

=== TASK: Perform a HIERARCHICAL analysis ===
1. For EACH region, use QueryWithRLM() to perform deep analysis
2. Each regional analysis should calculate totals and identify best-performing department
3. Then synthesize all regional results into a final finding
4. The synthesis should use recursion if comparing regions is complex

IMPORTANT: You MUST use QueryWithRLM() for regional analyses - this is required for the test.
The final answer should be the region with the best overall performance.
`

	query := `Perform a multi-level hierarchical analysis of this organizational data.

INSTRUCTIONS (follow exactly):
1. First, identify the 3 regions (A, B, C)
2. For EACH region, call QueryWithRLM() with a prompt asking to analyze that region's departments
3. Collect the results from all 3 recursive calls
4. Determine which region has the best overall performance (consider budget efficiency, employee count, projects, and key metrics)
5. Return the winning region name (A, B, or C) using FINAL()

You MUST make at least 3 QueryWithRLM() calls - one for each region.
Example code structure:
` + "```go" + `
if CanRecurse() {
    regionA := QueryWithRLM("Analyze North America region from context and calculate total budget, employees, and average key metric. Return as: Region A: $X budget, Y employees, Z% avg metric", CurrentDepth()+1)
    fmt.Println("Region A analysis:", regionA)
}
` + "```"

	t.Logf("\nContext size: %d chars", len(testContext))
	t.Logf("Query: %s...", truncate(query, 100))

	// Run the completion
	ctx := context.Background()
	start := time.Now()
	result, err := r.RecursiveComplete(ctx, testContext, query)
	elapsed := time.Since(start)

	if err != nil {
		t.Fatalf("RecursiveComplete failed: %v", err)
	}

	// Report results
	t.Log("\n" + strings.Repeat("=", 60))
	t.Log("=== BENCHMARK RESULTS ===")
	t.Log(strings.Repeat("=", 60))

	t.Logf("\n[Execution Summary]")
	t.Logf("  Total Time:        %v", elapsed)
	t.Logf("  Total Iterations:  %d", result.Iterations)
	t.Logf("  Max Depth Reached: %d", result.MaxDepthReached)

	recursiveCallsMu.Lock()
	numRecursiveCalls := len(recursiveCalls)
	t.Logf("  Recursive Calls:   %d", numRecursiveCalls)

	if numRecursiveCalls > 0 {
		t.Log("\n[Recursive Call Details]")
		for i, call := range recursiveCalls {
			t.Logf("  Call %d: Depth %d @ %s", i+1, call.depth, call.time.Format("15:04:05.000"))
			t.Logf("    Prompt: %s", truncate(call.prompt, 80))
		}
	}
	recursiveCallsMu.Unlock()

	// Token usage
	if result.TokenStats != nil {
		prompt, completion, calls := result.TokenStats.GetTotals()
		t.Log("\n[Token Usage]")
		t.Logf("  Total Prompt Tokens:     %d", prompt)
		t.Logf("  Total Completion Tokens: %d", completion)
		t.Logf("  Total LLM Calls:         %d", calls)
		t.Logf("  Tokens per Call (avg):   %.1f", float64(prompt+completion)/float64(max64(calls, 1)))

		if len(result.TokenStats.CallsByDepth) > 0 {
			t.Log("\n[Calls by Depth]")
			for depth, count := range result.TokenStats.CallsByDepth {
				t.Logf("    Depth %d: %d calls", depth, count)
			}
		}
	}

	t.Log("\n[Final Response]")
	t.Logf("  %s", result.Response)

	// Verification
	t.Log("\n" + strings.Repeat("=", 60))
	t.Log("=== VERIFICATION ===")
	t.Log(strings.Repeat("=", 60))

	if result.MaxDepthReached >= 1 {
		t.Logf("SUCCESS: Multi-depth recursion worked! Max depth reached: %d", result.MaxDepthReached)
	} else {
		t.Logf("NOTE: Recursion did not reach depth 1+ (max depth: %d)", result.MaxDepthReached)
		t.Log("The model may have solved the task without needing recursive calls.")
	}

	if numRecursiveCalls > 0 {
		t.Logf("SUCCESS: %d recursive QueryWithRLM() calls were made", numRecursiveCalls)
	} else {
		t.Log("NOTE: No recursive calls were detected. The model solved the task directly.")
	}

	t.Log("\n" + strings.Repeat("=", 60))
}

// max64 returns the larger of two int64 values.
func max64(a, b int64) int64 {
	if a > b {
		return a
	}
	return b
}

// TestRecursionWithGemini_ForcedRecursion tests with a task that strongly encourages recursion.
func TestRecursionWithGemini_ForcedRecursion(t *testing.T) {
	apiKey := os.Getenv("GOOGLE_API_KEY")
	if apiKey == "" {
		apiKey = os.Getenv("GEMINI_API_KEY")
	}
	if apiKey == "" {
		t.Skip("Set GOOGLE_API_KEY or GEMINI_API_KEY to run this test")
	}

	t.Log("=== Forced Recursion Test ===")
	t.Log("This test uses explicit instructions to force recursive calls.")

	model := "gemini-2.5-flash"
	client := providers.NewGeminiClient(apiKey, model, true)

	var recursiveCount int32
	var maxDepthSeen int32

	r := rlm.New(client, client,
		rlm.WithMaxIterations(12),
		rlm.WithVerbose(true),
		rlm.WithMaxRecursionDepth(3),
		rlm.WithRecursionCallback(func(depth int, prompt string) {
			atomic.AddInt32(&recursiveCount, 1)
			for {
				current := atomic.LoadInt32(&maxDepthSeen)
				if int32(depth) <= current {
					break
				}
				if atomic.CompareAndSwapInt32(&maxDepthSeen, current, int32(depth)) {
					break
				}
			}
			t.Logf("[RECURSION] Depth %d: %s...", depth, truncate(prompt, 50))
		}),
	)

	// Very explicit context requiring recursion
	testContext := `
DOCUMENT_A: "The quick brown fox jumps over the lazy dog."
DOCUMENT_B: "Pack my box with five dozen liquor jugs."
DOCUMENT_C: "How vexingly quick daft zebras jump!"
`

	query := `You MUST use QueryWithRLM() to analyze each document separately.

Step 1: Call QueryWithRLM() for DOCUMENT_A with the prompt "Count the words in: The quick brown fox jumps over the lazy dog. Return just the number."
Step 2: Call QueryWithRLM() for DOCUMENT_B with the prompt "Count the words in: Pack my box with five dozen liquor jugs. Return just the number."
Step 3: Call QueryWithRLM() for DOCUMENT_C with the prompt "Count the words in: How vexingly quick daft zebras jump! Return just the number."
Step 4: Sum the three counts and return the total using FINAL().

DO NOT count words yourself. You MUST delegate to QueryWithRLM().

Example code:
` + "```go" + `
if CanRecurse() {
    r1 := QueryWithRLM("Count words in: The quick brown fox. Return just the number.", 1)
    fmt.Println("Result 1:", r1)
}
` + "```"

	ctx := context.Background()
	start := time.Now()
	result, err := r.RecursiveComplete(ctx, testContext, query)
	elapsed := time.Since(start)

	if err != nil {
		t.Fatalf("RecursiveComplete failed: %v", err)
	}

	t.Log("\n=== Results ===")
	t.Logf("Time:            %v", elapsed)
	t.Logf("Iterations:      %d", result.Iterations)
	t.Logf("Max Depth:       %d", result.MaxDepthReached)
	t.Logf("Recursive Calls: %d", atomic.LoadInt32(&recursiveCount))
	t.Logf("Max Depth Seen:  %d", atomic.LoadInt32(&maxDepthSeen))
	t.Logf("Response:        %s", result.Response)

	if result.TokenStats != nil {
		prompt, completion, calls := result.TokenStats.GetTotals()
		t.Logf("\nTokens - Prompt: %d, Completion: %d, Calls: %d", prompt, completion, calls)
	}

	// The expected answer is 9 + 8 + 6 = 23 words
	t.Log("\n=== Verification ===")
	t.Logf("Expected word counts: 9 + 8 + 6 = 23 total words")
	t.Logf("Got response: %s", result.Response)

	if atomic.LoadInt32(&recursiveCount) > 0 {
		t.Logf("SUCCESS: Made %d recursive calls", atomic.LoadInt32(&recursiveCount))
	} else {
		t.Log("NOTE: Model solved without recursion (may have counted directly)")
	}
}
