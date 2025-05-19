#!/bin/bash

mkdir -p qwen_model/logs

TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="qwen_model/logs/hp_tune_${TIMESTAMP}.log"
PID_FILE="qwen_model/logs/hp_tune_${TIMESTAMP}.pid"

NUM_TRIALS=10
OUTPUT_DIR="qwen_model/finetuning/hp_tuning"
DATASET_PATH="qwen_model/data/dataset/dataset"
DEEPSPEED_CONFIG="qwen_model/deepspeed_config.json"  # או כל מיקום שתבחר

while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --dataset_path)
      DATASET_PATH="$2"
      shift
      shift
      ;;
    --num_trials)
      NUM_TRIALS="$2"
      shift
      shift
      ;;
    --output_dir)
      OUTPUT_DIR="$2"
      shift
      shift
      ;;
    --deepspeed_config)
      DEEPSPEED_CONFIG="$2"
      shift
      shift
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

echo "Starting hyperparameter tuning with DeepSpeed..."
echo "Logs: $LOG_FILE"

nohup deepspeed qwen_model/hp_tuning.py \
  --dataset_path $DATASET_PATH \
  --output_dir $OUTPUT_DIR \
  --num_trials $NUM_TRIALS \
  --deepspeed $DEEPSPEED_CONFIG \
  > $LOG_FILE 2>&1 &

echo $! > $PID_FILE
echo "PID: $! | Saved to: $PID_FILE"
echo "Monitor with: tail -f $LOG_FILE"
echo "To stop the process, use: kill \$(cat $PID_FILE)"