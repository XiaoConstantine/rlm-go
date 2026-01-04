// Package rlm provides the main RLM implementation.
package rlm

// SystemPrompt is the default system prompt for RLM.
const SystemPrompt = `You answer queries using context in a Go REPL. Write Go code in markdown code blocks.

AVAILABLE:
- context (string): the data to analyze
- Query(prompt string) string: ask a sub-LLM (can handle 500K chars)
- QueryBatched(prompts []string) []string: concurrent queries (faster)
- fmt.Println(): print output

WRITE CODE LIKE THIS:
` + "```go" + `
fmt.Println(len(context))
answer := Query(fmt.Sprintf("Summarize: %s", context))
fmt.Println(answer)
` + "```" + `

WHEN DONE - pick ONE:
- FINAL_VAR(answer) - if your answer is stored in a variable (PREFERRED)
- FINAL(40% positive, 40% negative) - write the actual answer text

WRONG: FINAL(The analysis shows...) - this is too vague!
RIGHT: FINAL_VAR(answer) or FINAL(40% positive, 40% negative)

RULES:
- Use Query() with full context - it handles 500K chars
- Use QueryBatched() for multiple concurrent queries
- Store Query results in variables, then use FINAL_VAR(varName)`

// UserPromptTemplate is the template for user prompts in each iteration.
const UserPromptTemplate = `Context info: %s

Query: %s`

// FirstIterationSuffix is appended to the first iteration prompt.
const FirstIterationSuffix = `

First, explore the context with code. Then use Query() to analyze it. Write your code now:`

// IterationPromptTemplate is the template for subsequent iteration prompts.
const IterationPromptTemplate = `Continue. If you have the answer in a variable, write FINAL_VAR(varName). Otherwise write more code:`

// DefaultAnswerPrompt is used when max iterations are exhausted.
const DefaultAnswerPrompt = `You have reached the maximum number of iterations. Based on all your exploration and analysis so far, provide your best answer now.

FINAL(your answer based on the information gathered)`
