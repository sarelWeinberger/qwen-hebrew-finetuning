#!/bin/bash
CONFIG="cpt_config.json"
DEEPSPEED="deepspeed_zero3.yaml"

for DATASET in datasets/*.jsonl; do
  BASENAME=$(basename "$DATASET" .jsonl)
  WANDB_NAME="qwen-hebrew-finetuning-$BASENAME-with-packing"
  LOGFILE="train_all.log"
  TIMESTAMP=$(date +%Y%m%d_%H%M%S)
  ARCHIVE_LOGFILE="train_${BASENAME}_${TIMESTAMP}.log"

  echo "Cleaning GPU memory before training on $DATASET"
  python clean_cuda.py

  echo "Training on $DATASET with wandb run name $WANDB_NAME"
  echo "To trace the current log output: tail -f $LOGFILE"
  accelerate launch --config_file=$DEEPSPEED train.py --dataset_path "$DATASET" --wandb_name "$WANDB_NAME" | tee $LOGFILE

  echo "Cleaning GPU memory after training on $DATASET"
  python clean_cuda.py

  cp $LOGFILE $ARCHIVE_LOGFILE

done
