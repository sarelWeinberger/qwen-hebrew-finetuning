#!/bin/bash
CONFIG="cpt_config.json"
DEEPSPEED="deepspeed_zero3.yaml"

# Create logs directory if it doesn't exist
mkdir -p logs

# Get the current script directory and set up virtual environment path
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="$SCRIPT_DIR/.venv"
PYTHON="$VENV_PATH/bin/python"
ACCELERATE="$VENV_PATH/bin/accelerate"

for DATASET in datasets/*.jsonl; do
  BASENAME=$(basename "$DATASET" .jsonl)
  LOGFILE="logs/train.log"
  TIMESTAMP=$(date +%Y%m%d_%H%M%S)
  ARCHIVE_LOGFILE="train_${BASENAME}_${TIMESTAMP}.log"

  echo "Cleaning GPU memory before training on $DATASET"
  $PYTHON clean_cuda.py

  echo "Training on $DATASET - wandb name will be auto-generated from model + dataset"
  echo "To trace the current log output: tail -f $LOGFILE"
  $ACCELERATE launch --config_file=$DEEPSPEED train.py --dataset_path "$DATASET" --validation_split 0.1 | tee $LOGFILE

  echo "Cleaning GPU memory after training on $DATASET"
  $PYTHON clean_cuda.py

done