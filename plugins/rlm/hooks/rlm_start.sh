#!/bin/bash
# RLM plugin startup check

# Check if rlm binary exists
RLM_BIN="${HOME}/.local/bin/rlm"

if [ -x "$RLM_BIN" ]; then
    echo "rlm: Binary found at $RLM_BIN"

    # Verify ANTHROPIC_API_KEY is set
    if [ -z "$ANTHROPIC_API_KEY" ]; then
        echo "rlm: Warning - ANTHROPIC_API_KEY not set. RLM requires this to function."
    else
        echo "rlm: Ready (API key configured)"
    fi
else
    echo "rlm: Binary not found. Install with:"
    echo "  curl -fsSL https://raw.githubusercontent.com/XiaoConstantine/rlm-go/main/install.sh | bash"
    echo "  # or"
    echo "  go install github.com/XiaoConstantine/rlm-go/cmd/rlm@latest"
fi
