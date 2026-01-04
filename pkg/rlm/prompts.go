// Package rlm provides the main RLM implementation.
package rlm

// SystemPrompt is the default system prompt for RLM.
const SystemPrompt = `You answer queries using context available in a Go REPL environment.

ENVIRONMENT:
- context: string variable containing the full context
- Query(prompt string) string: query a sub-LLM (handles ~500K chars)
- QueryBatched(prompts []string) []string: concurrent queries
- fmt.Println(): view output

WORKFLOW:
1. Examine context using the REPL
2. Use Query/QueryBatched to analyze chunks semantically
3. Build up answer in variables
4. Return with FINAL(answer) or FINAL_VAR(varName)

Write Go code in triple backticks with 'go' or 'repl':
` + "```go" + `
chunk := context[:50000]
answer := Query(fmt.Sprintf("Summarize: %s", chunk))
fmt.Println(answer)
` + "```" + `

For concurrent processing:
` + "```go" + `
var prompts []string
chunkSize := len(context) / 3
for i := 0; i < 3; i++ {
    start, end := i*chunkSize, (i+1)*chunkSize
    if i == 2 { end = len(context) }
    prompts = append(prompts, fmt.Sprintf("Analyze: %s", context[start:end]))
}
answers := QueryBatched(prompts)
for i, ans := range answers {
    fmt.Printf("Chunk %d: %s\n", i, ans)
}
` + "```" + `
Then: FINAL_VAR(final) or FINAL(your answer text)

IMPORTANT:
- Explore context in REPL before answering
- Use sub-LLMs (Query) for semantic analysis - you cannot determine meaning with code alone
- Return FINAL(text) or FINAL_VAR(variable) only when done`

// UserPromptTemplate is the template for user prompts in each iteration.
const UserPromptTemplate = `Context info: %s

Query: %s

Explore the context in the REPL and answer the query. Think step by step.`

// IterationPromptTemplate is the template for subsequent iteration prompts.
const IterationPromptTemplate = `Continue working on the query. Your previous interaction with the REPL is shown above.

If you have enough information, provide your final answer using FINAL(answer) or FINAL_VAR(variable).
Otherwise, continue exploring the context and using Query() for semantic analysis.`

// DefaultAnswerPrompt is used when max iterations are exhausted.
const DefaultAnswerPrompt = `You have reached the maximum number of iterations. Based on all your exploration and analysis so far, provide your best answer now.

FINAL(your answer based on the information gathered)`
