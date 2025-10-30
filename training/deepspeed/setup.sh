#!/bin/bash

# Qwen Hebrew Fine-tuning Setup Script
# This script sets up everything needed to run the Qwen fine-tuning pipeline

set -e  # Exit on any error

echo "========================================="
echo "Qwen Hebrew Fine-tuning Setup"
echo "========================================="

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
    echo "Creating virtual environment & installing reqs..."
    uv venv
    uv pip install -r requirements.txt
else
    source .venv/bin/activate
    echo "Virtual environment already exists"
fi

# Log in to huggingface
if hf auth whoami | grep -q "Not logged in"; then
  hf auth login
fi

wandb login 

if aws configure list | grep -q "access_key : <not set>"; then
    aws configure
fi

mkdir -p datasets/
mkdir -p logs/

# save storage & speed up by having the models saved to the NVME
export HF_HOME=/opt/dlami/nvme/.cache/huggingface

echo ""
echo "========================================="
echo "Setup Complete!"
echo "========================================="
echo ""
echo "Next steps:"
echo "1. Place your dataset files in the datasets/ directory"
echo ""
echo "2. Start training:"
echo "   nohup ./train_all.sh > logs/train.log 2>&1 &"
echo ""
echo "3. Monitor training:"
echo "   tail -f logs/train.log"
echo ""