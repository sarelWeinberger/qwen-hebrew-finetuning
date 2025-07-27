#!/bin/bash

# Script to start training with the best hyperparameters from tuning results using torchrun

# Default values
CONFIG_PATH="qwen_model/finetuning/training_config.json"
DATASET_PATH="qwen_model/data/dataset/dataset"
DEEPSPEED_CONFIG="qwen_model/deepspeed_config.json"
WANDB_PROJECT="qwen-30b-dataset-quality-benchmark"
WANDB_ENTITY="llm_train_mafat"
WANDB_NAME="qwen-hebrew-finetuning-$(date +"%Y%m%d_%H%M%S")"
SEED=42
NUM_GPUS=8

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --config)
      CONFIG_PATH="$2"
      shift
      shift
      ;;
    --dataset_path)
      DATASET_PATH="$2"
      shift
      shift
      ;;
    --deepspeed_config)
      DEEPSPEED_CONFIG="$2"
      shift
      shift
      ;;
    --wandb_project)
      WANDB_PROJECT="$2"
      shift
      shift
      ;;
    --wandb_entity)
      WANDB_ENTITY="$2"
      shift
      shift
      ;;
    --wandb_name)
      WANDB_NAME="$2"
      shift
      shift
      ;;
    --seed)
      SEED="$2"
      shift
      shift
      ;;
    --num_gpus)
      NUM_GPUS="$2"
      shift
      shift
      ;;
    *)
      echo "Unknown option: $1"
      echo "Valid options: --config, --dataset_path, --deepspeed_config, --wandb_project, --wandb_entity, --wandb_name, --seed, --num_gpus"
      exit 1
      ;;
  esac
done

# Validate NUM_GPUS
if [ "$NUM_GPUS" -lt 1 ] || [ "$NUM_GPUS" -gt 8 ]; then
  echo "Error: NUM_GPUS must be between 1 and 8, got: $NUM_GPUS"
  exit 1
fi

# Check if the configuration file exists
if [ ! -f "$CONFIG_PATH" ]; then
  echo "Configuration file not found: $CONFIG_PATH"
  echo "Running extract_best_params.sh to generate it..."
  
  # Run the extraction script
  ./qwen_model/extract_best_params.sh \
    --output_path "$CONFIG_PATH" \
    --dataset_path "$DATASET_PATH" \
    --deepspeed_config "$DEEPSPEED_CONFIG" \
    --wandb_project "$WANDB_PROJECT" \
    ${WANDB_ENTITY:+--wandb_entity "$WANDB_ENTITY"}
  
  # Check if the extraction was successful
  if [ ! -f "$CONFIG_PATH" ]; then
    echo "Failed to create training configuration"
    exit 1
  fi
fi

# Validate dataset path
if [ ! -d "$DATASET_PATH" ]; then
  echo "Error: Dataset path does not exist: $DATASET_PATH"
  exit 1
fi

# Validate DeepSpeed config
if [ ! -f "$DEEPSPEED_CONFIG" ]; then
  echo "Error: DeepSpeed config file not found: $DEEPSPEED_CONFIG"
  exit 1
fi

echo "=== Training Configuration ==="
echo "Configuration: $CONFIG_PATH"
echo "Dataset: $DATASET_PATH"
echo "DeepSpeed config: $DEEPSPEED_CONFIG"
echo "Number of GPUs: $NUM_GPUS"
echo "W&B project: $WANDB_PROJECT"
echo "W&B run name: $WANDB_NAME"
echo "Seed: $SEED"
echo "=============================="

# Create a timestamp for logs
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="qwen_model/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/train_${TIMESTAMP}.log"
ERROR_LOG_FILE="$LOG_DIR/train_${TIMESTAMP}_error.log"
PID_FILE="$LOG_DIR/train_${TIMESTAMP}.pid"

# Build the training command
TRAIN_ARGS="--config \"$CONFIG_PATH\" --dataset_path \"$DATASET_PATH\" --deepspeed \"$DEEPSPEED_CONFIG\" --wandb_project \"$WANDB_PROJECT\" --wandb_name \"$WANDB_NAME\" --seed $SEED"

# Add optional arguments
if [ -n "$WANDB_ENTITY" ]; then
  TRAIN_ARGS="$TRAIN_ARGS --wandb_entity \"$WANDB_ENTITY\""
fi

# Determine training command based on number of GPUs
if [ "$NUM_GPUS" -eq 1 ]; then
  echo "Running single-GPU training..."
  TRAINING_CMD="python qwen_model/train.py $TRAIN_ARGS"
else
  echo "Running distributed training with $NUM_GPUS GPUs using torchrun..."
  TRAINING_CMD="torchrun --nproc_per_node=$NUM_GPUS qwen_model/train.py $TRAIN_ARGS"
fi

echo "Training command: $TRAINING_CMD"
echo ""

# Start training with nohup to keep it running in the background
echo "Starting training process..."
echo "Main log: $LOG_FILE"
echo "Error log: $ERROR_LOG_FILE"
echo "Rank logs: /tmp/train_rank_*.log"

# Set environment variables for better distributed training
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_LAUNCH_BLOCKING=1
export PYTHONUNBUFFERED=1

# Clear any existing rank logs
rm -f /tmp/train_rank_*.log

nohup bash -c "
  echo \"Starting training at: \$(date)\"
  echo \"Command: $TRAINING_CMD\"
  echo \"Environment: PYTORCH_CUDA_ALLOC_CONF=\$PYTORCH_CUDA_ALLOC_CONF\"
  echo \"===========================================\"
  
  # Run the training command
  $TRAINING_CMD
  
  EXIT_CODE=\$?
  echo \"===========================================\"
  echo \"Training finished at: \$(date)\"
  echo \"Exit code: \$EXIT_CODE\"
  
  if [ \$EXIT_CODE -eq 0 ]; then
    echo \"Training completed successfully!\"
  else
    echo \"Training failed with exit code: \$EXIT_CODE\"
  fi
" > "$LOG_FILE" 2> "$ERROR_LOG_FILE" &

# Save the process ID
TRAIN_PID=$!
echo $TRAIN_PID > "$PID_FILE"

echo ""
echo "=== Training Started ==="
echo "Process ID: $TRAIN_PID"
echo "PID saved to: $PID_FILE"
echo "========================"
echo ""
echo "=== Monitoring Commands ==="
echo "Main progress: tail -f $LOG_FILE"
echo "Error logs: tail -f $ERROR_LOG_FILE"
echo "Rank logs: tail -f /tmp/train_rank_*.log"
echo "GPU usage: watch -n 5 nvidia-smi"
echo "Check process: ps aux | grep $TRAIN_PID"
echo "Stop training: kill \$(cat $PID_FILE)"
echo "=========================="
echo ""
echo "Training is now running in the background and will continue after disconnection."
echo "You can safely close your terminal or SSH session."

# Optional: Show initial log output for a few seconds
echo ""
echo "=== Initial Log Output (first 10 seconds) ==="
sleep 2
if [ -f "$LOG_FILE" ]; then
  timeout 8 tail -f "$LOG_FILE" || true
  echo ""
  echo "=== Use 'tail -f $LOG_FILE' to continue monitoring ==="
else
  echo "Log file not created yet, check in a moment with: tail -f $LOG_FILE"
fi