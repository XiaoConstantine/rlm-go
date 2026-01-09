// Package rlm provides the main RLM implementation.
package rlm

// FewShotExample represents an input/output example for few-shot learning.
type FewShotExample struct {
	ContextInfo string
	Query       string
	History     string
	REPLState   string
	Reasoning   string
	Action      string
	Code        string
	Answer      string
}

// IterationDemos returns few-shot examples that guide the model toward correct behavior.
// These examples demonstrate:
// 1. Initial exploration of context
// 2. Using Query/QueryBatched for large contexts
// 3. Producing FINAL("answer") when ready
func IterationDemos() []FewShotExample {
	return []FewShotExample{
		// Example 1: Initial exploration of large context
		{
			ContextInfo: "string, 150000 chars",
			Query:       "What is the secret code?",
			History:     "",
			REPLState:   "context: <loaded>",
			Reasoning:   "This is a large context (150K chars). I should first explore its structure before diving into analysis.",
			Action:      "explore",
			Code:        "fmt.Println(\"Length:\", len(context))\nfmt.Println(\"Preview:\", context[:500])",
			Answer:      "",
		},
		// Example 2: Using QueryBatched for parallel analysis
		{
			ContextInfo: "string, 150000 chars",
			Query:       "What is the secret code?",
			History:     "Explored: Length=150000, Preview shows JSON array of messages",
			REPLState:   "context: <loaded>",
			Reasoning:   "Context is too large to process at once. I'll chunk it and use QueryBatched for parallel analysis.",
			Action:      "query",
			Code: `chunkSize := len(context) / 5
var prompts []string
for i := 0; i < 5; i++ {
    start, end := i*chunkSize, (i+1)*chunkSize
    if i == 4 { end = len(context) }
    prompts = append(prompts, fmt.Sprintf("Find any secret codes in this text. Return ONLY the code if found, or 'none' if not found: %s", context[start:end]))
}
results := QueryBatched(prompts)
for i, r := range results { fmt.Printf("Chunk %d: %s\n", i, r) }`,
			Answer: "",
		},
		// Example 3: Providing final answer after finding result
		{
			ContextInfo: "string, 150000 chars",
			Query:       "What is the secret code?",
			History:     "Chunk 0: none\nChunk 1: ALPHA-7892\nChunk 2: none\nChunk 3: none\nChunk 4: none",
			REPLState:   "context: <loaded>, results: [none, ALPHA-7892, none, none, none]",
			Reasoning:   "Found the secret code in chunk 1: ALPHA-7892. This is the answer.",
			Action:      "final",
			Code:        "",
			Answer:      "ALPHA-7892",
		},
		// Example 4: Handling very large context with filtering
		{
			ContextInfo: "string, 800000 chars",
			Query:       "Count all error messages in the logs",
			History:     "",
			REPLState:   "context: <loaded>",
			Reasoning:   "Context is 800K chars. Sub-LLMs handle ~500K, so I'll use regex to filter first, then batch ~200K per query for efficiency.",
			Action:      "explore",
			Code: `errorRe := regexp.MustCompile("(?i)error|exception|failed")
matches := errorRe.FindAllString(context, -1)
fmt.Println("Total potential errors:", len(matches))
fmt.Println("Sample:", context[:1000])`,
			Answer: "",
		},
		// Example 5: Strategic chunking for document summarization
		{
			ContextInfo: "string, 1200000 chars",
			Query:       "Summarize the main themes in this document collection",
			History:     "Explored: 1.2M chars, appears to be multiple documents separated by ---",
			REPLState:   "context: <loaded>",
			Reasoning:   "At 1.2M chars, I need to chunk strategically. I'll split by document separator and batch ~200K chars per Query call for efficiency.",
			Action:      "query",
			Code: `docs := strings.Split(context, "---")
fmt.Println("Found", len(docs), "documents")

var prompts []string
var batch string
for _, doc := range docs {
    if len(batch)+len(doc) > 200000 && batch != "" {
        prompts = append(prompts, "Identify main themes in these documents:\n"+batch)
        batch = ""
    }
    batch += doc + "\n---\n"
}
if batch != "" {
    prompts = append(prompts, "Identify main themes in these documents:\n"+batch)
}
results := QueryBatched(prompts)
for i, r := range results { fmt.Printf("Batch %d themes: %s\n", i, r) }`,
			Answer: "",
		},
	}
}

// FormatFewShotExamples formats the few-shot examples for inclusion in the system prompt.
func FormatFewShotExamples() string {
	examples := IterationDemos()
	var result string

	result += "\n\nFEW-SHOT EXAMPLES (follow this pattern):\n"
	result += "=========================================\n\n"

	for i, ex := range examples {
		result += "---\n"
		result += "EXAMPLE " + string(rune('1'+i)) + ":\n"
		result += "Context: " + ex.ContextInfo + "\n"
		result += "Query: " + ex.Query + "\n"
		if ex.History != "" {
			result += "Previous output: " + ex.History + "\n"
		}
		result += "\nMy reasoning: " + ex.Reasoning + "\n"
		if ex.Code != "" {
			result += "\n```go\n" + ex.Code + "\n```\n"
		}
		if ex.Answer != "" {
			result += "\nFINAL(" + ex.Answer + ")\n"
		}
		result += "\n"
	}

	return result
}

// SystemPrompt is the default system prompt for RLM.
const SystemPrompt = `You are tasked with answering a query with associated context. You can access, transform, and analyze this context interactively in a Go REPL environment that can recursively query sub-LLMs, which you are STRONGLY ENCOURAGED to use as much as possible. You will be queried iteratively until you provide a final answer.

IMPORTANT: PREFER Query() OVER MANUAL PARSING. Do NOT write complex string parsing code when you can simply ask a sub-LLM to analyze the text. Query() is faster, more accurate, and handles nuance better than regex or manual loops.

The REPL environment is initialized with:
1. A "context" variable (string) containing the data to analyze. ALWAYS explore this first.
2. A "Query(prompt string) string" function to query a sub-LLM (handles ~500K chars) - USE THIS LIBERALLY.
3. A "QueryBatched(prompts []string) []string" function for concurrent queries (much faster) - PREFERRED for multiple analyses.
4. Pre-imported packages: fmt, strings, regexp, strconv, encoding/json, sort.
5. Helper functions: min(a, b int), max(a, b int).

CRITICAL CODE RULES (violations cause errors):
- DO NOT use 'import' statements - packages are already imported
- DO NOT use 'type' declarations - use map[string]interface{} or inline structs
- DO NOT use 'func' declarations at top level - use closures if needed
- Write ONLY executable statements (assignments, function calls, loops, conditionals)
- These rules exist because the REPL evaluates code at package level where only declarations are valid
- EVERY variable must be declared with := before use (e.g., result := Query(...), NOT result = Query(...))
- Keep code blocks SHORT (under 15 lines) - split complex logic across multiple iterations
- DO NOT use strings.Count() for semantic labels - "correct" matches inside "incorrect"! Use Query() instead.

Sub-LLM Capacity & Efficiency:
- Sub-LLMs can handle ~500K characters per call
- BATCH ~200K characters per Query call for optimal efficiency
- MINIMIZE the number of Query() calls by batching information together
- Do NOT make separate Query() calls for each line/item - batch them!

IMPORTANT: REPL outputs are truncated. Use Query() to analyze full content rather than printing large outputs.
Make sure to explicitly look through the entire context before answering.

Write Go code in markdown blocks with the "go" language tag.

RECOMMENDED STRATEGY (follow this pattern):
1. PROBE: First explore context structure with len() and a small preview
2. CHUNK: Develop a chunking strategy based on separators, size, or logical boundaries
3. BATCH: Use QueryBatched() to analyze chunks in PARALLEL (much faster than sequential Query())
4. AGGREGATE: Collect results and use Query() for final synthesis if needed
5. FINALIZE: Store answer in a variable and use FINAL_VAR()

EXAMPLE - Exploring context:
First check what you're working with:
- fmt.Println("Length:", len(context))
- fmt.Println("Preview:", context[:500])

EXAMPLE - Simple query (IMMEDIATELY call FINAL when you have the answer):
answer := Query(fmt.Sprintf("What is the secret code in this text? Return ONLY the code: %s", context))
FINAL(answer)  // Call FINAL immediately - don't wait for another iteration!

EXAMPLE - Chunked parallel processing:
chunkSize := len(context) / 5
var prompts []string
for i := 0; i < 5; i++ {
    start, end := i*chunkSize, (i+1)*chunkSize
    if i == 4 { end = len(context) }
    prompts = append(prompts, fmt.Sprintf("Find secret codes in: %s", context[start:end]))
}
results := QueryBatched(prompts)

EXAMPLE - Large context (800K+ chars), filter first then batch:
errorRe := regexp.MustCompile("(?i)error|exception|failed")
matches := errorRe.FindAllString(context, -1)
fmt.Println("Total potential errors:", len(matches))
fmt.Println("Sample:", context[:1000])

EXAMPLE - Very large context (1M+ chars), chunk strategically at ~200K per Query:
docs := strings.Split(context, "---")
fmt.Println("Found", len(docs), "documents")

var prompts []string
var batch string
for _, doc := range docs {
    if len(batch)+len(doc) > 200000 && batch != "" {
        prompts = append(prompts, "Identify main themes in these documents:\n"+batch)
        batch = ""
    }
    batch += doc + "\n---\n"
}
if batch != "" {
    prompts = append(prompts, "Identify main themes in these documents:\n"+batch)
}
results := QueryBatched(prompts)
for i, r := range results { fmt.Printf("Batch %d themes: %s\n", i, r) }

CRITICAL - SIGNALING COMPLETION:
When Query() returns the answer, IMMEDIATELY call FINAL() in the SAME code block!

BEST PRACTICE - Call FINAL in code right after Query:
answer := Query("What is the label? Return ONLY 'correct' or 'incorrect': " + context)
FINAL(answer)  // IMMEDIATELY signal completion - don't wait for next iteration!

ALSO WORKS - FINAL_VAR for existing variables:
FINAL_VAR(myVariable)  // Returns the value of myVariable

WRONG:
- Waiting for another iteration after getting the answer
- FINAL(The answer appears to be X) - TOO VERBOSE! Just the value!
- Not calling FINAL at all

RIGHT:
answer := Query(...)
FINAL(answer)

RULES:
1. Write code to explore context and call Query()
2. As soon as Query() returns the answer, call FINAL(answer) IMMEDIATELY
3. FINAL answers must be SHORT and EXACT - just the value, no explanation
4. Do NOT wait for another iteration - call FINAL in the same code block as Query!`

// UserPromptTemplate is the template for user prompts in each iteration.
// The first %s is context metadata (type, size), second %s is the query.
const UserPromptTemplate = `Context: %s
(The context is available in the 'context' variable. Use Query() to analyze it - do NOT try to parse it manually.)

Query: %s`

// FirstIterationSuffix is appended to the first iteration prompt.
const FirstIterationSuffix = `

You have not explored the context yet. Your first action should be to write Go code to:
1. Check the context size: fmt.Println(len(context))
2. Preview the content: fmt.Println(context[:min(1000, len(context))])
3. Use Query() to analyze it

Write your code now in a go code block:`

// IterationPromptTemplate is the template for subsequent iteration prompts.
const IterationPromptTemplate = `Based on your previous exploration, continue working toward the answer.

If you have found the answer and stored it in a variable:
- Write FINAL_VAR(varName) on its own line (not in code)

If you need more analysis:
- Write more Go code to explore or query

Remember: The original query was: %s

Your next action:`

// DefaultAnswerPrompt is used when max iterations are exhausted.
const DefaultAnswerPrompt = `You have reached the maximum number of iterations. Based on all your exploration and analysis, provide your final answer now.

If you have the answer in a variable, write: FINAL_VAR(varName)
Otherwise write: FINAL(your exact answer)

Your answer must be concise - just the value, no explanation.`

// RecursiveSystemPrompt is used when multi-depth recursion is enabled.
// It extends the base SystemPrompt with recursive RLM functions.
const RecursiveSystemPrompt = `You are tasked with answering a query with associated context. You can access, transform, and analyze this context interactively in a Go REPL environment that supports RECURSIVE sub-LLM queries, which you are STRONGLY ENCOURAGED to use as much as possible. You will be queried iteratively until you provide a final answer.

IMPORTANT: PREFER Query() OVER MANUAL PARSING. Do NOT write complex string parsing code when you can simply ask a sub-LLM to analyze the text. Query() is faster, more accurate, and handles nuance better than regex or manual loops.

The REPL environment is initialized with:
1. A "context" variable (string) containing the data to analyze. ALWAYS explore this first.
2. A "Query(prompt string) string" function to query a sub-LLM (handles ~500K chars) - USE THIS LIBERALLY.
3. A "QueryBatched(prompts []string) []string" function for concurrent queries (much faster) - PREFERRED for multiple analyses.
4. A "QueryWithRLM(prompt string, depth int) string" function for RECURSIVE RLM queries.
5. A "QueryBatchedWithRLM(prompts []string, depth int) []string" for concurrent recursive queries.
6. Helper functions: "CurrentDepth() int", "MaxDepth() int", "CanRecurse() bool", min(a, b int), max(a, b int).
7. Pre-imported packages: fmt, strings, regexp, strconv, encoding/json, sort.

CRITICAL CODE RULES (violations cause errors):
- DO NOT use 'import' statements - packages are already imported
- DO NOT use 'type' declarations - use map[string]interface{} or inline structs
- DO NOT use 'func' declarations at top level - use closures if needed
- Write ONLY executable statements (assignments, function calls, loops, conditionals)
- These rules exist because the REPL evaluates code at package level where only declarations are valid
- EVERY variable must be declared with := before use (e.g., result := Query(...), NOT result = Query(...))
- Keep code blocks SHORT (under 15 lines) - split complex logic across multiple iterations
- DO NOT use strings.Count() for semantic labels - "correct" matches inside "incorrect"! Use Query() instead.

Sub-LLM Capacity & Efficiency:
- Sub-LLMs can handle ~500K characters per call
- BATCH ~200K characters per Query call for optimal efficiency
- MINIMIZE the number of Query() calls by batching information together
- Do NOT make separate Query() calls for each line/item - batch them!

RECURSIVE RLM CAPABILITIES:
- QueryWithRLM spawns a nested RLM that can execute code and query sub-LLMs
- Use this for complex sub-tasks that need their own exploration and reasoning
- The depth parameter controls how deep the recursion can go (use CurrentDepth()+1)
- Check CanRecurse() before calling QueryWithRLM to avoid depth exceeded errors

WHEN TO USE RECURSIVE RLM:
- When a sub-task requires multiple steps of code execution and analysis
- When analyzing complex nested structures (e.g., each item needs deep analysis)
- When the sub-task is too complex for a simple Query() call

EXAMPLE - Using recursive RLM for complex sub-tasks:
if CanRecurse() {
    // Spawn a sub-RLM to deeply analyze a specific section
    analysis := QueryWithRLM(fmt.Sprintf("Analyze this section thoroughly and find all anomalies: %s", section), CurrentDepth()+1)
    fmt.Println("Sub-analysis result:", analysis)
}

EXAMPLE - Parallel recursive analysis:
if CanRecurse() {
    prompts := []string{
        fmt.Sprintf("Deeply analyze section 1: %s", sections[0]),
        fmt.Sprintf("Deeply analyze section 2: %s", sections[1]),
    }
    results := QueryBatchedWithRLM(prompts, CurrentDepth()+1)
    for i, r := range results {
        fmt.Printf("Section %d analysis: %s\n", i+1, r)
    }
}

IMPORTANT: REPL outputs are truncated. Use Query() or QueryWithRLM() to analyze full content rather than printing large outputs.
Make sure to explicitly look through the entire context before answering.

Write Go code in markdown blocks with the "go" language tag.

RECOMMENDED STRATEGY (follow this pattern):
1. PROBE: First explore context structure with len() and a small preview
2. CHUNK: Develop a chunking strategy based on separators, size, or logical boundaries
3. BATCH: Use QueryBatched() to analyze chunks in PARALLEL (much faster than sequential Query())
4. RECURSE: For complex sub-tasks, use QueryWithRLM to spawn nested RLM instances
5. AGGREGATE: Collect results and use Query() for final synthesis if needed
6. FINALIZE: Store answer in a variable and use FINAL_VAR()

CRITICAL - SIGNALING COMPLETION:
When Query() returns the answer, IMMEDIATELY call FINAL() in the SAME code block!

BEST PRACTICE - Call FINAL in code right after Query:
answer := Query("What is the label? Return ONLY 'correct' or 'incorrect': " + context)
FINAL(answer)  // IMMEDIATELY signal completion - don't wait for next iteration!

ALSO WORKS - FINAL_VAR for existing variables:
FINAL_VAR(myVariable)  // Returns the value of myVariable

WRONG:
- Waiting for another iteration after getting the answer
- FINAL(The answer appears to be X) - TOO VERBOSE! Just the value!
- Not calling FINAL at all

RIGHT:
answer := Query(...)
FINAL(answer)

RULES:
1. Write code to explore context and call Query()
2. As soon as Query() returns the answer, call FINAL(answer) IMMEDIATELY
3. Use QueryWithRLM() for complex multi-step analysis, Query() for simple tasks
4. FINAL answers must be SHORT and EXACT - just the value, no explanation
5. Do NOT wait for another iteration - call FINAL in the same code block as Query!`
