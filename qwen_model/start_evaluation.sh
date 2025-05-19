#!/bin/bash

# Script to start Hebrew LLM leaderboard evaluation with nohup to ensure it continues after SSH disconnection

# Create logs directory
mkdir -p qwen_model/logs

# Get timestamp for log files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="qwen_model/logs/evaluate_${TIMESTAMP}.log"
PID_FILE="qwen_model/logs/evaluate_${TIMESTAMP}.pid"

# Default values
MODEL_PATH="qwen_model/finetuned"
DATASET_PATH=""
WANDB_PROJECT="qwen-hebrew-evaluation"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --model_path)
      MODEL_PATH="$2"
      shift
      shift
      ;;
    --dataset_path)
      DATASET_PATH="$2"
      shift
      shift
      ;;
    --wandb_project)
      WANDB_PROJECT="$2"
      shift
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Validate required arguments
if [ -z "$MODEL_PATH" ]; then
  echo "Error: --model_path is required"
  exit 1
fi

# Build the command
COMMAND="python qwen_model/evaluate_hebrew.py --model_path $MODEL_PATH --wandb_project $WANDB_PROJECT"
if [ ! -z "$DATASET_PATH" ]; then
  COMMAND="$COMMAND --dataset_path $DATASET_PATH"
fi

# Start evaluation
echo "Starting Hebrew LLM leaderboard evaluation with nohup..."
echo "Logs will be saved to: $LOG_FILE"

nohup $COMMAND > $LOG_FILE 2>&1 &

# Save the process ID
echo $! > $PID_FILE
echo "Evaluation process started with PID: $!"
echo "PID saved to: $PID_FILE"
echo "To monitor progress: tail -f $LOG_FILE"
echo "To stop the process: kill -9 \$(cat $PID_FILE)"
echo ""
echo "The evaluation will continue even if you disconnect from SSH."