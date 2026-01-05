// Package providers implements LLM client providers for various backends.
package providers

import (
	"github.com/XiaoConstantine/rlm-go/pkg/repl"
	"github.com/XiaoConstantine/rlm-go/pkg/rlm"
)

// Provider identifies an LLM provider.
type Provider string

const (
	Anthropic Provider = "anthropic"
	Gemini    Provider = "gemini"
	OpenAI    Provider = "openai"
)

// Client combines both rlm.LLMClient and repl.LLMClient interfaces.
type Client interface {
	rlm.LLMClient
	repl.LLMClient
}

// SupportedModels maps model names to their providers.
var SupportedModels = map[string]Provider{
	// Gemini models
	"gemini-3-flash-preview": Gemini,
	"gemini-3-pro-preview":   Gemini,
	// OpenAI models
	"gpt-5":      OpenAI,
	"gpt-5-mini": OpenAI,
}

// GetProvider returns the provider for a given model name.
// Returns Anthropic as default for unknown models.
func GetProvider(model string) Provider {
	if p, ok := SupportedModels[model]; ok {
		return p
	}
	return Anthropic
}

// EnvKey returns the environment variable name for the provider's API key.
func (p Provider) EnvKey() string {
	switch p {
	case Gemini:
		return "GEMINI_API_KEY"
	case OpenAI:
		return "OPENAI_API_KEY"
	default:
		return "ANTHROPIC_API_KEY"
	}
}
