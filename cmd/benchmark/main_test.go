package main

import (
	"context"
	"strings"
	"testing"

	"github.com/XiaoConstantine/rlm-go/pkg/core"
	"github.com/XiaoConstantine/rlm-go/pkg/repl"
)

type benchmarkMockClient struct {
	systemPrompt string
}

func (c *benchmarkMockClient) Complete(ctx context.Context, messages []core.Message) (core.LLMResponse, error) {
	systemPrompt := ""
	if len(messages) > 0 {
		systemPrompt = messages[0].Content
	}
	c.systemPrompt = systemPrompt
	if strings.Contains(systemPrompt, "QueryWithRLM") {
		return core.LLMResponse{
			Content:          "FINAL(recursive path)",
			PromptTokens:     11,
			CompletionTokens: 7,
		}, nil
	}
	return core.LLMResponse{
		Content:          "FINAL(plain path)",
		PromptTokens:     5,
		CompletionTokens: 3,
	}, nil
}

func (c *benchmarkMockClient) Query(ctx context.Context, prompt string) (repl.QueryResponse, error) {
	return repl.QueryResponse{Response: "query response"}, nil
}

func (c *benchmarkMockClient) QueryBatched(ctx context.Context, prompts []string) ([]repl.QueryResponse, error) {
	results := make([]repl.QueryResponse, len(prompts))
	for i := range results {
		results[i] = repl.QueryResponse{Response: "query response"}
	}
	return results, nil
}

func TestRunRLMUsesRecursiveCompleteWhenDepthEnabled(t *testing.T) {
	task := Task{
		TaskID:   "recursive",
		Context:  "context",
		Question: "question",
		Answer:   "recursive path",
	}
	result := runRLM(context.Background(), task, &benchmarkMockClient{}, RLMOptions{
		MaxIters:       3,
		RecursionDepth: 1,
	})
	if result.Error != "" {
		t.Fatalf("runRLM() error = %q", result.Error)
	}
	if result.Got != "recursive path" {
		t.Fatalf("runRLM() got = %q, want recursive path", result.Got)
	}
	if !result.IsCorrect {
		t.Fatal("runRLM() should mark recursive answer correct")
	}
}

func TestRunRLMUsesPlainCompleteWhenDepthDisabled(t *testing.T) {
	task := Task{
		TaskID:   "plain",
		Context:  "context",
		Question: "question",
		Answer:   "plain path",
	}
	result := runRLM(context.Background(), task, &benchmarkMockClient{}, RLMOptions{
		MaxIters:       3,
		RecursionDepth: 0,
	})
	if result.Error != "" {
		t.Fatalf("runRLM() error = %q", result.Error)
	}
	if result.Got != "plain path" {
		t.Fatalf("runRLM() got = %q, want plain path", result.Got)
	}
	if !result.IsCorrect {
		t.Fatal("runRLM() should mark plain answer correct")
	}
}
