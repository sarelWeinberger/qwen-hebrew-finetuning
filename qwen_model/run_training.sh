#!/bin/bash

# Script to run Qwen model training with nohup to continue after disconnection

# Parse command line arguments
DATASET_PATH=""
CONFIG_PATH="qwen_model/finetuning/training_config.json"
DEEPSPEED_CONFIG="qwen_model/finetuning/h100_ds_config.json"
NUM_GPUS=8
MODE="train"  # Options: train, hp_tune, evaluate
OUTPUT_DIR=""
NUM_TRIALS=10
MODEL_PATH=""

# Parse arguments
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
    --deepspeed)
      DEEPSPEED_CONFIG="$2"
      shift
      shift
      ;;
    --num_gpus)
      NUM_GPUS="$2"
      shift
      shift
      ;;
    --mode)
      MODE="$2"
      shift
      shift
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift
      shift
      ;;
    --num_trials)
      NUM_TRIALS="$2"
      shift
      shift
      ;;
    --model_path)
      MODEL_PATH="$2"
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
if [ -z "$DATASET_PATH" ]; then
  echo "Error: --dataset_path is required"
  exit 1
fi

# Create logs directory
LOGS_DIR="qwen_model/logs"
mkdir -p $LOGS_DIR

# Generate timestamp for log files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# Function to run training
run_training() {
  echo "Starting training with nohup..."
  echo "Logs will be saved to: $LOGS_DIR/train_${TIMESTAMP}.log"
  echo "To monitor progress: tail -f $LOGS_DIR/train_${TIMESTAMP}.log"
  
  nohup torchrun --nproc_per_node=$NUM_GPUS qwen_model/train.py \
    --dataset_path $DATASET_PATH \
    --config $CONFIG_PATH \
    --deepspeed $DEEPSPEED_CONFIG \
    > $LOGS_DIR/train_${TIMESTAMP}.log 2>&1 &
  
  echo "Training process started with PID: $!"
  echo "To stop the process: kill $!"
  echo "Process ID saved to: $LOGS_DIR/train_${TIMESTAMP}.pid"
  echo $! > $LOGS_DIR/train_${TIMESTAMP}.pid
}

# Function to run hyperparameter tuning
run_hp_tuning() {
  if [ -z "$OUTPUT_DIR" ]; then
    OUTPUT_DIR="qwen_model/finetuning/hp_tuning"
  fi
  
  echo "Starting hyperparameter tuning with nohup..."
  echo "Logs will be saved to: $LOGS_DIR/hp_tune_${TIMESTAMP}.log"
  echo "To monitor progress: tail -f $LOGS_DIR/hp_tune_${TIMESTAMP}.log"
  
  nohup python qwen_model/hp_tuning.py \
    --dataset_path $DATASET_PATH \
    --output_dir $OUTPUT_DIR \
    --num_trials $NUM_TRIALS \
    > $LOGS_DIR/hp_tune_${TIMESTAMP}.log 2>&1 &
  
  echo "Hyperparameter tuning process started with PID: $!"
  echo "To stop the process: kill $!"
  echo "Process ID saved to: $LOGS_DIR/hp_tune_${TIMESTAMP}.pid"
  echo $! > $LOGS_DIR/hp_tune_${TIMESTAMP}.pid
}

# Function to run evaluation
run_evaluation() {
  if [ -z "$MODEL_PATH" ]; then
    echo "Error: --model_path is required for evaluation mode"
    exit 1
  fi
  
  echo "Starting evaluation with nohup..."
  echo "Logs will be saved to: $LOGS_DIR/evaluate_${TIMESTAMP}.log"
  echo "To monitor progress: tail -f $LOGS_DIR/evaluate_${TIMESTAMP}.log"
  
  nohup python qwen_model/evaluate_hebrew.py \
    --model_path $MODEL_PATH \
    --dataset_path $DATASET_PATH \
    > $LOGS_DIR/evaluate_${TIMESTAMP}.log 2>&1 &
  
  echo "Evaluation process started with PID: $!"
  echo "To stop the process: kill $!"
  echo "Process ID saved to: $LOGS_DIR/evaluate_${TIMESTAMP}.pid"
  echo $! > $LOGS_DIR/evaluate_${TIMESTAMP}.pid
}

# Run the appropriate mode
case $MODE in
  train)
    run_training
    ;;
  hp_tune)
    run_hp_tuning
    ;;
  evaluate)
    run_evaluation
    ;;
  *)
    echo "Error: Unknown mode '$MODE'. Valid options are: train, hp_tune, evaluate"
    exit 1
    ;;
esac

echo "Process is now running in the background and will continue after disconnection."
echo "You can safely close your terminal or SSH session."