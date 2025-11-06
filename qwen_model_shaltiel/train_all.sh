#!/bin/bash
CONFIG="cpt_config.json"
DEEPSPEED="deepspeed_zero3.yaml"

for DATASET in datasets/*.jsonl; do
  BASENAME=$(basename "$DATASET" .jsonl)sudo lsof /dev/nvidia* | grep python | awk '{print $2}' | xargs -r sudo kill -9
  WANDB_NAME="qwen-hebrew-finetuning-$BASENAME"
  echo "Cleaning GPU memory before training on $DATASET"
  
  echo "Training on $DATASET with wandb run name $WANDB_NAME"
  accelerate launch --config_file=$DEEPSPEED train.py --dataset_path "$DATASET" --wandb_name $WANDB_NAME
done

echo "Cleaning GPU memory after all trainings finished"
sudo lsof /dev/nvidia* | grep python | awk '{print $2}' | xargs -r sudo kill -9
