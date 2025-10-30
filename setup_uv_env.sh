#!/bin/bash

# UV Environment Setup Script for Qwen Hebrew Fine-tuning Project
# This script installs UV and sets up the virtual environment

set -e

echo "🚀 Setting up UV environment for Qwen Hebrew Fine-tuning..."

# Check if UV is installed
if ! command -v uv &> /dev/null; then
    echo "📦 Installing UV..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    
    # Add UV to PATH for current session
    export PATH="$HOME/.cargo/bin:$PATH"
    
    # Add UV to shell profile for persistence
    if [[ "$SHELL" == *"zsh"* ]]; then
        echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.zshrc
    else
        echo 'export PATH="$HOME/.cargo/bin:$PATH"' >> ~/.bashrc
    fi
    
    echo "✅ UV installed successfully!"
else
    echo "✅ UV is already installed"
fi

# Load environment variables from .env if it exists
if [ -f ".env" ]; then
    echo "🔧 Loading environment variables from .env..."
    set -a
    source .env
    set +a
    echo "✅ Environment variables loaded"
fi

# Create UV project if pyproject.toml doesn't exist
if [ ! -f "pyproject.toml" ]; then
    echo "📝 Creating pyproject.toml..."
    uv init --name qwen-hebrew-finetuning --python 3.10
else
    echo "✅ pyproject.toml already exists"
fi

# Create virtual environment
echo "🐍 Creating virtual environment with UV..."
uv venv

# Activate virtual environment
echo "🔄 Activating virtual environment..."
source .venv/bin/activate

# Install dependencies
if [ -f "requirements.txt" ]; then
    echo "📦 Installing dependencies from requirements.txt..."
    uv pip install -r requirements.txt
fi

# Install development dependencies
echo "📦 Installing development dependencies..."
uv add --dev pytest black flake8 mypy jupyter

# Sync environment
echo "🔄 Syncing environment..."
uv sync

echo ""
echo "🎉 UV environment setup complete!"
echo ""
echo "To activate the environment in the future, run:"
echo "  source .venv/bin/activate"
echo ""
echo "To install new packages, use:"
echo "  uv add package_name"
echo ""
echo "To run commands in the virtual environment:"
echo "  uv run python script.py"
echo ""
