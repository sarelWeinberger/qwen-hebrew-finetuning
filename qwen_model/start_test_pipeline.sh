#!/bin/bash

# Script to test the training pipeline with a small subset of data using nohup

# Create logs directory
mkdir -p qwen_model/logs

# Get timestamp for log files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="qwen_model/logs/test_pipeline_${TIMESTAMP}.log"
PID_FILE="qwen_model/logs/test_pipeline_${TIMESTAMP}.pid"

# Default values
DATASET_PATH="qwen_model/data/dataset/dataset"
CONFIG_PATH="qwen_model/finetuning/training_config.json"
OUTPUT_DIR="qwen_model/test_run"
MAX_SAMPLES=10
MAX_STEPS=5
MAX_SEQ_LENGTH=128
SEED=42
SKIP_WANDB=""
SINGLE_DEVICE=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --dataset_path)
      DATASET_PATH="$2"
      shift
      shift
      ;;
    --config)
      CONFIG_PATH="$2"
      shift
      shift
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift
      shift
      ;;
    --max_samples)
      MAX_SAMPLES="$2"
      shift
      shift
      ;;
    --max_steps)
      MAX_STEPS="$2"
      shift
      shift
      ;;
    --max_seq_length)
      MAX_SEQ_LENGTH="$2"
      shift
      shift
      ;;
    --seed)
      SEED="$2"
      shift
      shift
      ;;
    --skip_wandb)
      SKIP_WANDB="--skip_wandb"
      shift
      ;;
    --single_device)
      SINGLE_DEVICE="--single_device"
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Print test configuration
echo "=== Test Pipeline Configuration ==="
echo "Dataset path: $DATASET_PATH"
echo "Config path: $CONFIG_PATH"
echo "Output directory: $OUTPUT_DIR"
echo "Max samples: $MAX_SAMPLES"
echo "Max steps: $MAX_STEPS"
echo "Max sequence length: $MAX_SEQ_LENGTH"
echo "Seed: $SEED"
echo "Skip W&B: ${SKIP_WANDB:-false}"
echo "Single device: ${SINGLE_DEVICE:-false}"
echo "=================================="

# Set environment variables to avoid memory fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Set environment variables for better GPU utilization
export NCCL_DEBUG=WARN  # Less verbose NCCL logging

# Set environment variable to avoid tokenizer parallelism warnings
export TOKENIZERS_PARALLELISM=false

# Start test pipeline with single GPU first to test
echo "Starting test pipeline..."
echo "Logs will be saved to: $LOG_FILE"

# Run the test pipeline
nohup python qwen_model/test_pipeline.py \
  --dataset_path $DATASET_PATH \
  --config $CONFIG_PATH \
  --output_dir $OUTPUT_DIR \
  --max_samples $MAX_SAMPLES \
  --max_steps $MAX_STEPS \
  --max_seq_length $MAX_SEQ_LENGTH \
  --seed $SEED \
  $SKIP_WANDB \
  $SINGLE_DEVICE \
  > $LOG_FILE 2>&1 &

# Save the process ID
echo $! > $PID_FILE
echo "Test pipeline process started with PID: $!"
echo "PID saved to: $PID_FILE"
echo "To monitor progress: tail -f $LOG_FILE"
echo "To stop the process: kill -9 \$(cat $PID_FILE)"
echo ""
echo "The test pipeline will continue even if you disconnect from SSH."