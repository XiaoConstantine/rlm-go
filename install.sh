#!/bin/bash
# RLM installer script
# Downloads and installs the latest release of rlm to ~/.local/bin

set -e

REPO="XiaoConstantine/rlm-go"
INSTALL_DIR="${HOME}/.local/bin"
BINARY_NAME="rlm"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

# Detect OS
detect_os() {
    local os
    os=$(uname -s | tr '[:upper:]' '[:lower:]')
    case "$os" in
        linux*)  echo "linux" ;;
        darwin*) echo "darwin" ;;
        *)       error "Unsupported operating system: $os" ;;
    esac
}

# Detect architecture
detect_arch() {
    local arch
    arch=$(uname -m)
    case "$arch" in
        x86_64)  echo "amd64" ;;
        amd64)   echo "amd64" ;;
        arm64)   echo "arm64" ;;
        aarch64) echo "arm64" ;;
        *)       error "Unsupported architecture: $arch" ;;
    esac
}

# Get latest release version
get_latest_version() {
    local version
    version=$(curl -sL "https://api.github.com/repos/${REPO}/releases/latest" | grep '"tag_name":' | sed -E 's/.*"([^"]+)".*/\1/')
    if [ -z "$version" ]; then
        error "Could not determine latest version. Check your internet connection."
    fi
    echo "$version"
}

# Main installation
main() {
    info "RLM Installer"
    info "============="

    local os arch version download_url binary_name

    os=$(detect_os)
    arch=$(detect_arch)
    version=$(get_latest_version)

    info "Detected OS: $os"
    info "Detected architecture: $arch"
    info "Latest version: $version"

    binary_name="rlm-${os}-${arch}"
    download_url="https://github.com/${REPO}/releases/download/${version}/${binary_name}"

    info "Downloading from: $download_url"

    # Create install directory
    mkdir -p "$INSTALL_DIR"

    # Download binary
    if command -v curl &> /dev/null; then
        curl -fsSL "$download_url" -o "${INSTALL_DIR}/${BINARY_NAME}"
    elif command -v wget &> /dev/null; then
        wget -q "$download_url" -O "${INSTALL_DIR}/${BINARY_NAME}"
    else
        error "Neither curl nor wget found. Please install one of them."
    fi

    # Make executable
    chmod +x "${INSTALL_DIR}/${BINARY_NAME}"

    info "Installed to: ${INSTALL_DIR}/${BINARY_NAME}"

    # Check if install dir is in PATH
    if [[ ":$PATH:" != *":${INSTALL_DIR}:"* ]]; then
        warn "~/.local/bin is not in your PATH"
        warn "Add it by running:"
        echo ""
        echo "  export PATH=\"\$HOME/.local/bin:\$PATH\""
        echo ""
        warn "Add this line to your shell profile (~/.bashrc, ~/.zshrc, etc.) for persistence"
    fi

    # Verify installation
    if "${INSTALL_DIR}/${BINARY_NAME}" --help &> /dev/null; then
        info "Installation successful!"
        echo ""
        info "Quick start:"
        echo "  rlm -context <file> -query \"your question\""
        echo ""
        info "Install Claude Code skill:"
        echo "  rlm install-claude-code"
    else
        error "Installation verification failed"
    fi
}

main "$@"
