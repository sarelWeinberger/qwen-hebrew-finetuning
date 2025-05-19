#!/bin/bash

# Script to extract the best hyperparameters from tuning results and generate a training config

# Default values
HP_TUNING_DIR="qwen_model/finetuning/hp_tuning"
OUTPUT_PATH="qwen_model/finetuning/training_config.json"
MODEL_PATH="qwen_model/model"
DATASET_PATH="qwen_model/data/dataset/dataset"
DEEPSPEED_CONFIG="qwen_model/deepspeed_config.json"
WANDB_PROJECT="qwen-hebrew-hp-tuning"
WANDB_ENTITY=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --hp_tuning_dir)
      HP_TUNING_DIR="$2"
      shift
      shift
      ;;
    --output_path)
      OUTPUT_PATH="$2"
      shift
      shift
      ;;
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
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

echo "Extracting best hyperparameters from tuning results..."
echo "HP Tuning Directory: $HP_TUNING_DIR"
echo "Output Path: $OUTPUT_PATH"

# Run the extraction script
python qwen_model/extract_best_params.py \
  --hp_tuning_dir "$HP_TUNING_DIR" \
  --output_path "$OUTPUT_PATH" \
  --model_path "$MODEL_PATH" \
  --dataset_path "$DATASET_PATH" \
  --deepspeed_config "$DEEPSPEED_CONFIG" \
  --wandb_project "$WANDB_PROJECT" \
  ${WANDB_ENTITY:+--wandb_entity "$WANDB_ENTITY"}

# Check if the extraction was successful
if [ -f "$OUTPUT_PATH" ]; then
  echo "Successfully created training configuration at $OUTPUT_PATH"
  echo "To start training with the best hyperparameters, run:"
  echo "python qwen_model/train.py --config $OUTPUT_PATH"
else
  echo "Failed to create training configuration"
fi