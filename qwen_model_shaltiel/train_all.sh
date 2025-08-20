#!/bin/bash
CONFIG="cpt_config.json"
DEEPSPEED="deepspeed_zero3.yaml"

# Set paths to virtual environment executables
VENV_PATH="/home/ec2-user/qwen-hebrew-finetuning/qwen_model_shaltiel/.venv"
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

  cp $LOGFILE $ARCHIVE_LOGFILE

done