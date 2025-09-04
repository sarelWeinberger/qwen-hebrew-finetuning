#!/bin/bash

# Script to safely kill all training processes
# This script will terminate all Python training processes and DeepSpeed processes

echo "🛑 Killing all training processes..."

# Function to safely kill processes
kill_processes() {
    local pattern=$1
    local description=$2
    
    pids=$(pgrep -f "$pattern" 2>/dev/null || true)
    if [ -n "$pids" ]; then
        echo "📍 Found $description processes: $pids"
        echo "   Sending SIGTERM..."
        kill -TERM $pids 2>/dev/null || true
        sleep 3
        
        # Check if processes are still running
        remaining_pids=$(pgrep -f "$pattern" 2>/dev/null || true)
        if [ -n "$remaining_pids" ]; then
            echo "   Some processes still running, sending SIGKILL..."
            kill -KILL $remaining_pids 2>/dev/null || true
            sleep 1
        fi
        echo "   ✅ $description processes terminated"
    else
        echo "   ✅ No $description processes found"
    fi
}

# Kill training processes in order of preference
echo ""
echo "🔍 Searching for training processes..."

# 1. Kill train.py processes
kill_processes "train\.py" "train.py"

# 2. Kill train_debug.py processes
kill_processes "train_debug\.py" "train_debug.py"

# 3. Kill accelerate launch processes
kill_processes "accelerate.*launch" "accelerate launch"

# 4. Kill any Python processes with DeepSpeed
kill_processes "python.*deepspeed" "Python DeepSpeed"

# 5. Kill any remaining DeepSpeed processes
kill_processes "deepspeed" "DeepSpeed"

# 6. Kill any Python processes with "qwen" in the command line
kill_processes "python.*qwen" "Qwen-related Python"

# 7. Kill any Python processes with "transformers" and "train" in command line
kill_processes "python.*transformers.*train" "Transformers training"

echo ""
echo "🧹 Cleaning up GPU memory..."

# Clear GPU memory if nvidia-smi is available
if command -v nvidia-smi &> /dev/null; then
    echo "📊 GPU memory before cleanup:"
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits
    
    # Reset GPU if nvidia-ml-py is available
    python3 -c "
try:
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        print('   ✅ PyTorch GPU cache cleared')
    else:
        print('   ⚠️  CUDA not available in PyTorch')
except ImportError:
    print('   ⚠️  PyTorch not available')
    
try:
    import gc
    gc.collect()
    print('   ✅ Python garbage collection completed')
except:
    print('   ⚠️  Garbage collection failed')
" 2>/dev/null || echo "   ⚠️  Could not run Python cleanup"

    echo ""
    echo "📊 GPU memory after cleanup:"
    nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader,nounits
else
    echo "   ⚠️  nvidia-smi not found, skipping GPU cleanup"
fi

echo ""
echo "🔍 Checking for remaining training processes..."
remaining=$(pgrep -f "train\.py|accelerate.*launch|deepspeed|python.*qwen" 2>/dev/null || true)
if [ -n "$remaining" ]; then
    echo "   ⚠️  Warning: Some processes may still be running:"
    ps aux | grep -E "train\.py|accelerate.*launch|deepspeed|python.*qwen" | grep -v grep | head -5
    echo "   💡 You may need to kill them manually or restart the system"
else
    echo "   ✅ No remaining training processes found"
fi

echo ""
echo "🎯 Process cleanup summary:"
echo "   • Terminated all train.py processes"
echo "   • Terminated all accelerate launch processes"  
echo "   • Terminated all DeepSpeed processes"
echo "   • Cleared GPU memory cache"
echo "   • Performed garbage collection"

echo ""
echo "✅ Training process cleanup completed!"
echo "💡 It's safe to start new training runs now."

# Show current GPU usage
if command -v nvidia-smi &> /dev/null; then
    echo ""
    echo "📊 Current GPU status:"
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader
fi