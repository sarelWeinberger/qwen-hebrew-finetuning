#!/bin/bash
CONFIG="cpt_config.json"
DEEPSPEED="deepspeed_zero3.yaml"
MAX_RETRIES=3

# Create logs directory if it doesn't exist
mkdir -p logs

# Get the current script directory and set up virtual environment path
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_PATH="$SCRIPT_DIR/.venv"
PYTHON="$VENV_PATH/bin/python"
ACCELERATE="$VENV_PATH/bin/accelerate"

# Function to train with retry logic
train_with_retry() {
  local dataset="$1"
  local basename="$2"
  local logfile="$3"
  local attempt=1
  
  while [ $attempt -le $MAX_RETRIES ]; do
    echo "Training attempt $attempt/$MAX_RETRIES on $dataset"
    
    # Clean GPU before each attempt
    echo "Cleaning GPU memory before training attempt $attempt"
    $PYTHON clean_cuda.py
    
    # Start training
    echo "Starting training on $dataset (attempt $attempt)"
    echo "To trace the current log output: tail -f $logfile"
    
    $ACCELERATE launch --config_file=$DEEPSPEED train.py --dataset_path "$dataset" --validation_split 0.001 | tee "$logfile"
    
    # Check exit status
    if [ $? -eq 0 ]; then
      echo "✅ Training completed successfully for $dataset"
      return 0
    else
      echo "❌ Training failed for $dataset (attempt $attempt)"
      
      if [ $attempt -lt $MAX_RETRIES ]; then
        echo "⏳ Retrying in 30 seconds..."
        sleep 30
      fi
    fi
    
    attempt=$((attempt + 1))
  done
  
  echo "💥 Failed to train $dataset after $MAX_RETRIES attempts"
  return 1
}

for DATASET in datasets/*.jsonl; do
  BASENAME=$(basename "$DATASET" .jsonl)
  LOGFILE="logs/train.log"
  TIMESTAMP=$(date +%Y%m%d_%H%M%S)
  ARCHIVE_LOGFILE="train_${BASENAME}_${TIMESTAMP}.log"

  echo "========================================="
  echo "Processing dataset: $DATASET"
  echo "========================================="

  if train_with_retry "$DATASET" "$BASENAME" "$LOGFILE"; then
    echo "✅ Successfully completed $DATASET"
  else
    echo "❌ Failed to complete $DATASET after all retries"
    echo "Check logs in $LOGFILE for details"
    exit 1
  fi

  echo "Cleaning GPU memory after completing $DATASET"
  $PYTHON clean_cuda.py
  echo ""

done

echo "🎉 All datasets completed successfully!"