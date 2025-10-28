#!/bin/bash
CONFIG="cpt_config.json"
DEEPSPEED="deepspeed_zero3.yaml"
LOCKFILE="/tmp/train_remaining.lock"

# Function to clean GPU memory
clean_gpu_memory() {
    echo "Cleaning GPU memory..."
    
    # Kill processes using NVIDIA devices
    sudo lsof /dev/nvidia* | grep python | awk '{print $2}' | xargs -r sudo kill -9
    pkill -f "train.py"
    pkill -f "accelerate"
    
    # Use nvidia-smi to reset GPU if available
    if command -v nvidia-smi &> /dev/null; then
        nvidia-smi --gpu-reset-default-mode || true
    fi
    
    # Clear CUDA cache using Python
    python3 -c "
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    for i in range(torch.cuda.device_count()):
        torch.cuda.set_device(i)
        torch.cuda.empty_cache()
    print('CUDA cache cleared')
else:
    print('CUDA not available')
" 2>/dev/null || echo "Python CUDA cleanup failed"
    
    sleep 5
}

# Check if another instance is already running
if [ -f "$LOCKFILE" ]; then
    echo "Another training script is already running. Exiting..."
    exit 1
fi

# Create lock file
echo $$ > "$LOCKFILE"
trap "rm -f $LOCKFILE" EXIT

# List of remaining datasets to process
DATASETS=(
    "datasets/BIU.jsonl"
    "datasets/BooksNLI2.jsonl" 
    "datasets/Geektime.jsonl"
    "datasets/Oscar.jsonl"
)

echo "Starting training on remaining datasets..."
echo "Total datasets to process: ${#DATASETS[@]}"

for DATASET in "${DATASETS[@]}"; do
  if [ -f "$DATASET" ]; then
    BASENAME=$(basename "$DATASET" .jsonl)
    WANDB_NAME="qwen-hebrew-finetuning-$BASENAME"
    echo "================================================"
    echo "Preparing to train on $DATASET"
    
    # Clean GPU memory before starting
    clean_gpu_memory
    
    # Wait for cleanup and check if any training processes are still running
    sleep 15
    if pgrep -f "train.py" > /dev/null; then
        echo "Warning: Training processes still running, waiting longer..."
        sleep 30
        clean_gpu_memory
    fi
    
    echo "Training on $DATASET with wandb run name $WANDB_NAME"
    echo "Started at: $(date)"
    accelerate launch --config_file=$DEEPSPEED train.py --dataset_path "$DATASET" --wandb_name $WANDB_NAME
    echo "Completed $DATASET at: $(date)"
    echo "================================================"
  else
    echo "Warning: Dataset $DATASET not found, skipping..."
  fi
done

echo "Cleaning GPU memory after all trainings finished"
clean_gpu_memory
echo "All training completed at: $(date)"
