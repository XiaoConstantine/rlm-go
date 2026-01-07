package providers

import (
	"testing"
)

func TestGetProvider(t *testing.T) {
	tests := []struct {
		model    string
		expected Provider
	}{
		// Gemini models
		{"gemini-2.5-flash", Gemini},
		{"gemini-2.5-pro", Gemini},
		{"gemini-2.0-flash", Gemini},
		{"gemini-3-flash-preview", Gemini},
		{"gemini-3-pro-preview", Gemini},
		// OpenAI models
		{"gpt-5", OpenAI},
		{"gpt-5-mini", OpenAI},
		// Anthropic (default)
		{"claude-sonnet-4-20250514", Anthropic},
		{"claude-opus-4-20250514", Anthropic},
		{"unknown-model", Anthropic},
		{"", Anthropic},
	}

	for _, tt := range tests {
		t.Run(tt.model, func(t *testing.T) {
			got := GetProvider(tt.model)
			if got != tt.expected {
				t.Errorf("GetProvider(%q) = %v, want %v", tt.model, got, tt.expected)
			}
		})
	}
}

func TestProviderEnvKey(t *testing.T) {
	tests := []struct {
		provider Provider
		expected string
	}{
		{Anthropic, "ANTHROPIC_API_KEY"},
		{Gemini, "GEMINI_API_KEY"},
		{OpenAI, "OPENAI_API_KEY"},
		{Provider("unknown"), "ANTHROPIC_API_KEY"}, // default
	}

	for _, tt := range tests {
		t.Run(string(tt.provider), func(t *testing.T) {
			got := tt.provider.EnvKey()
			if got != tt.expected {
				t.Errorf("%v.EnvKey() = %q, want %q", tt.provider, got, tt.expected)
			}
		})
	}
}

func TestSupportedModels(t *testing.T) {
	// Verify all expected models are in the map
	expectedModels := map[string]Provider{
		"gemini-2.5-flash":       Gemini,
		"gemini-2.5-pro":         Gemini,
		"gemini-2.0-flash":       Gemini,
		"gemini-3-flash-preview": Gemini,
		"gemini-3-pro-preview":   Gemini,
		"gpt-5":                  OpenAI,
		"gpt-5-mini":             OpenAI,
	}

	for model, provider := range expectedModels {
		if got, ok := SupportedModels[model]; !ok {
			t.Errorf("SupportedModels missing model %q", model)
		} else if got != provider {
			t.Errorf("SupportedModels[%q] = %v, want %v", model, got, provider)
		}
	}
}

func TestNewAnthropicClient(t *testing.T) {
	client := NewAnthropicClient("test-key", "claude-sonnet-4-20250514", false)
	if client == nil {
		t.Fatal("NewAnthropicClient returned nil")
	}
	if client.apiKey != "test-key" {
		t.Errorf("apiKey = %q, want %q", client.apiKey, "test-key")
	}
	if client.model != "claude-sonnet-4-20250514" {
		t.Errorf("model = %q, want %q", client.model, "claude-sonnet-4-20250514")
	}
	if client.verbose != false {
		t.Error("verbose should be false")
	}
	if client.httpClient == nil {
		t.Error("httpClient should not be nil")
	}
}

func TestNewGeminiClient(t *testing.T) {
	client := NewGeminiClient("test-key", "gemini-2.5-flash", true)
	if client == nil {
		t.Fatal("NewGeminiClient returned nil")
	}
	if client.apiKey != "test-key" {
		t.Errorf("apiKey = %q, want %q", client.apiKey, "test-key")
	}
	if client.model != "gemini-2.5-flash" {
		t.Errorf("model = %q, want %q", client.model, "gemini-2.5-flash")
	}
	if client.verbose != true {
		t.Error("verbose should be true")
	}
	if client.httpClient == nil {
		t.Error("httpClient should not be nil")
	}
}

func TestNewOpenAIClient(t *testing.T) {
	client := NewOpenAIClient("test-key", "gpt-5", false)
	if client == nil {
		t.Fatal("NewOpenAIClient returned nil")
	}
	if client.apiKey != "test-key" {
		t.Errorf("apiKey = %q, want %q", client.apiKey, "test-key")
	}
	if client.model != "gpt-5" {
		t.Errorf("model = %q, want %q", client.model, "gpt-5")
	}
	if client.verbose != false {
		t.Error("verbose should be false")
	}
	if client.httpClient == nil {
		t.Error("httpClient should not be nil")
	}
}

func TestClientInterface(t *testing.T) {
	// Verify all clients implement the Client interface
	var _ Client = (*AnthropicClient)(nil)
	var _ Client = (*GeminiClient)(nil)
	var _ Client = (*OpenAIClient)(nil)
}
