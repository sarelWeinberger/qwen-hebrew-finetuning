#!/bin/bash

# Qwen Hebrew Fine-tuning Setup Script
# This script sets up everything needed to run the Qwen fine-tuning pipeline

set -e  # Exit on any error

echo "========================================="
echo "Qwen Hebrew Fine-tuning Setup"
echo "========================================="

# Get the directory where this script is located
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "Setting up in: $SCRIPT_DIR"

# Create necessary directories
echo "Creating necessary directories..."
mkdir -p logs
mkdir -p datasets
mkdir -p output

# Check if uv is installed
if ! command -v uv &> /dev/null; then
    echo "Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source $HOME/.local/bin/env
else
    echo "uv is already installed"
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    uv venv
else
    echo "Virtual environment already exists"
fi

# Activate virtual environment
echo "Activating virtual environment..."
source .venv/bin/activate

# Install requirements
echo "Installing Python dependencies..."
uv pip install -r requirements.txt

# Set up HuggingFace transfer for faster downloads
export HF_HUB_ENABLE_HF_TRANSFER=1

echo ""
echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Activate the virtual environment:"
echo "   source .venv/bin/activate"
echo ""
echo "2. Login to services:"
echo "   wandb login"
echo "   huggingface-cli login"
echo ""
echo "3. Set up AWS credentials in ~/.aws/credentials:"
echo "   [default]"
echo "   aws_access_key_id = YOUR_ACCESS_KEY"
echo "   aws_secret_access_key = YOUR_SECRET_KEY"
echo ""
echo "4. Download models (optional, will be downloaded automatically if needed):"
echo "   huggingface-cli download Qwen/Qwen3-30B-A3B-Base"
echo "   huggingface-cli download Qwen/Qwen3-32B"
echo ""
echo "5. Place your dataset files in the datasets/ directory"
echo ""
echo "6. Start training:"
echo "   nohup ./train_all.sh > logs/train.log 2>&1 &"
echo ""
echo "7. Monitor training:"
echo "   tail -f logs/train.log"
echo ""
