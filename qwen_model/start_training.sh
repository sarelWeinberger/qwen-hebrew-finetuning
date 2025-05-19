#!/bin/bash

# Script to start training with the best hyperparameters from tuning results

# Default values
CONFIG_PATH="qwen_model/finetuning/training_config.json"
DATASET_PATH="qwen_model/data/dataset/dataset"
DEEPSPEED_CONFIG="qwen_model/deepspeed_config.json"
WANDB_PROJECT="qwen-hebrew-finetuning"
WANDB_ENTITY=""
WANDB_NAME="qwen-hebrew-finetuning-$(date +"%Y%m%d_%H%M%S")"
SEED=42

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
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

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

echo "Starting training with configuration: $CONFIG_PATH"
echo "Dataset: $DATASET_PATH"
echo "DeepSpeed config: $DEEPSPEED_CONFIG"
echo "W&B project: $WANDB_PROJECT"
echo "W&B run name: $WANDB_NAME"

# Create a timestamp for logs
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_DIR="qwen_model/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/train_${TIMESTAMP}.log"
PID_FILE="$LOG_DIR/train_${TIMESTAMP}.pid"

# Start training with nohup to keep it running in the background
echo "Starting training process..."
echo "Logs will be saved to: $LOG_FILE"

nohup python qwen_model/train.py \
  --config "$CONFIG_PATH" \
  --dataset_path "$DATASET_PATH" \
  --deepspeed "$DEEPSPEED_CONFIG" \
  --wandb_project "$WANDB_PROJECT" \
  --wandb_name "$WANDB_NAME" \
  --seed "$SEED" \
  ${WANDB_ENTITY:+--wandb_entity "$WANDB_ENTITY"} \
  > "$LOG_FILE" 2>&1 &

# Save the process ID
echo $! > "$PID_FILE"
echo "Training process started with PID: $!"
echo "PID saved to: $PID_FILE"
echo "To monitor progress: tail -f $LOG_FILE"
echo "To stop the process: kill \$(cat $PID_FILE)"