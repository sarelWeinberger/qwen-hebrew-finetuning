#!/bin/bash
CONFIG="cpt_config.json"
DEEPSPEED="deepspeed_zero3.yaml"

# Create logs directory if it doesn't exist
mkdir -p logs

# Function to train with retry logic
train() {
    local dataset="$1"
    local logfile="$2"
    
    # Start training
    echo "Starting training on $dataset, to trace the current log output: tail -f $logfile"
    accelerate launch --config_file=$DEEPSPEED train.py --dataset_path "$dataset" --validation_split 0.001 | tee "$logfile"
    
    # Check exit status
    if [ $? -eq 0 ]; then
      echo "✅ Training completed successfully for $dataset" | tee "$logfile"
    else
      echo "❌ Training failed for $dataset" | tee "$logfile"
    fi
}

for DATASET in datasets/*.jsonl; do
  BASENAME=$(basename "$DATASET" .jsonl)
  LOGFILE="logs/train.log"
  TIMESTAMP=$(date +%Y%m%d_%H%M%S)
  ARCHIVE_LOGFILE="train_${BASENAME}_${TIMESTAMP}.log"

  echo "========================================="
  echo "Processing dataset: $DATASET"
  echo "========================================="

  train "$DATASET" "$LOGFILE"
done

echo "🎉 All datasets completed successfully!"