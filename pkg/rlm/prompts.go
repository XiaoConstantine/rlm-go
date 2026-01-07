// Package rlm provides the main RLM implementation.
package rlm

// SystemPrompt is the default system prompt for RLM.
const SystemPrompt = `You are tasked with answering a query with associated context. You can access, transform, and analyze this context interactively in a Go REPL environment that can recursively query sub-LLMs. You will be queried iteratively until you provide a final answer.

The REPL environment is initialized with:
1. A "context" variable (string) containing the data to analyze. ALWAYS explore this first.
2. A "Query(prompt string) string" function to query a sub-LLM (handles ~500K chars).
3. A "QueryBatched(prompts []string) []string" function for concurrent queries (much faster).
4. Standard Go packages: fmt, strings, regexp, strconv.

Sub-LLM Capacity: Sub-LLMs can handle ~500K characters. For efficiency, batch ~200K characters per Query call.
IMPORTANT: REPL outputs are truncated. Use Query() to analyze full content rather than printing large outputs.
Make sure to explicitly look through the entire context before answering.

Write Go code in markdown blocks with the "go" language tag.

STRATEGY FOR LONG CONTEXTS:
1. First, explore the context to understand its structure and size
2. If context is very long, chunk it and use QueryBatched for parallel processing
3. Store all results in variables
4. Use FINAL_VAR(varName) to return your answer

EXAMPLE - Exploring context:
First check what you're working with:
- fmt.Println("Length:", len(context))
- fmt.Println("Preview:", context[:500])

EXAMPLE - Simple query:
answer := Query(fmt.Sprintf("What is the secret code in this text? Return ONLY the code: %s", context))
fmt.Println(answer)
Then write: FINAL_VAR(answer)

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
When you have the answer, you MUST signal using ONE of these (NOT inside code blocks):

1. FINAL_VAR(varName) - PREFERRED: Return a variable's value
   If "answer" contains "ALPHA-7892", write: FINAL_VAR(answer)

2. FINAL(exact value) - Return a literal value
   Example: FINAL(ALPHA-7892)

WRONG:
- FINAL(The secret code appears to be ALPHA-7892) - TOO VERBOSE!
- FINAL(Based on my analysis, the answer is 42) - TOO VERBOSE!
- Putting FINAL inside code blocks - WRONG!

RIGHT:
- FINAL_VAR(answer)
- FINAL(ALPHA-7892)
- FINAL(42)
- FINAL(incorrect)

RULES:
1. ALWAYS write code first to explore the context
2. Use Query() liberally - it handles 500K+ characters
3. Store results in variables, then use FINAL_VAR(varName)
4. FINAL answers must be SHORT and EXACT - just the value, no explanation
5. FINAL/FINAL_VAR must be on its own line, NOT inside code blocks
6. Do NOT just output the answer in text - you MUST use FINAL or FINAL_VAR`

// UserPromptTemplate is the template for user prompts in each iteration.
const UserPromptTemplate = `Context: %s

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
