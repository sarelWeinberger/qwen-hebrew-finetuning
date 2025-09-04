#!/bin/bash

# Test script to verify setup is working correctly

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "========================================="
echo "Testing Qwen Fine-tuning Setup"
echo "========================================="

# Test 1: Check if virtual environment exists and works
echo "Test 1: Virtual Environment"
if [ -d ".venv" ]; then
    echo "✓ Virtual environment exists"
    source .venv/bin/activate
    
    # Test if python works
    if python -c "import sys; print(f'Python {sys.version}')" 2>/dev/null; then
        echo "✓ Python environment is working"
    else
        echo "✗ Python environment has issues"
        exit 1
    fi
else
    echo "✗ Virtual environment not found. Run ./setup.sh first"
    exit 1
fi

# Test 2: Check required directories
echo ""
echo "Test 2: Directory Structure"
for dir in "logs" "datasets" "output"; do
    if [ -d "$dir" ]; then
        echo "✓ $dir/ directory exists"
    else
        echo "✗ $dir/ directory missing"
        exit 1
    fi
done

# Test 3: Check key Python packages
echo ""
echo "Test 3: Python Dependencies"
REQUIRED_PACKAGES=("torch" "transformers" "accelerate" "deepspeed" "datasets")

for package in "${REQUIRED_PACKAGES[@]}"; do
    if python -c "import $package" 2>/dev/null; then
        VERSION=$(python -c "import $package; print($package.__version__)" 2>/dev/null || echo "unknown")
        echo "✓ $package ($VERSION)"
    else
        echo "✗ $package not installed or not working"
        exit 1
    fi
done

# Test 4: Check configuration files
echo ""
echo "Test 4: Configuration Files"
CONFIG_FILES=("cpt_config.json" "deepspeed_zero3.yaml" "train.py")

for file in "${CONFIG_FILES[@]}"; do
    if [ -f "$file" ]; then
        echo "✓ $file exists"
    else
        echo "✗ $file missing"
        exit 1
    fi
done

# Test 5: Check executable scripts
echo ""
echo "Test 5: Executable Scripts"
SCRIPTS=("setup.sh" "train_all.sh" "monitor.sh")

for script in "${SCRIPTS[@]}"; do
    if [ -x "$script" ]; then
        echo "✓ $script is executable"
    else
        echo "✗ $script is not executable"
        exit 1
    fi
done

# Test 6: Check CUDA availability (if on GPU machine)
echo ""
echo "Test 6: CUDA/GPU Check"
if command -v nvidia-smi &> /dev/null; then
    echo "✓ nvidia-smi available"
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    echo "✓ Found $GPU_COUNT GPU(s)"
    
    if python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Device count: {torch.cuda.device_count()}')" 2>/dev/null; then
        echo "✓ PyTorch can access CUDA"
    else
        echo "⚠ PyTorch CUDA support may have issues"
    fi
else
    echo "⚠ nvidia-smi not available (CPU-only mode)"
fi

echo ""
echo "========================================="
echo "✓ Setup verification complete!"
echo "========================================="
echo ""
echo "Your environment is ready for training."
echo ""
echo "Next steps:"
echo "1. Place your JSONL datasets in the datasets/ directory"
echo "2. Run: nohup ./train_all.sh > logs/train.log 2>&1 &"
echo "3. Monitor: ./monitor.sh --log"
echo ""
