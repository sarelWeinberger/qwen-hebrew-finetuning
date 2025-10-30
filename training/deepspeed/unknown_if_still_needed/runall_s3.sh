#!/bin/bash
CONFIG="cpt_config.json"
DEEPSPEED="deepspeed_zero3.yaml"
S3_BUCKET="s3://gepeta-checkpoints"
S3_PREFIX="qwen-hebrew-finetuning"

# Function to sync checkpoints to S3
sync_to_s3() {
    local s3_path=$1
    local output_dir=$3
    
        # Create S3 path if it doesn't exist and sync output directory
    if [ -d "$output_dir" ]; then
        echo "Uploading model checkpoints to: $s3_path"
        aws s3 sync "$output_dir" "$s3_path/model" --delete
        
        if [ $? -eq 0 ]; then
            echo "✓ Successfully synced checkpoints to S3: $s3_path"
        else
            echo "✗ Failed to sync checkpoints to S3"
            return 1
        fi
    else
        echo "Warning: Output directory $output_dir not found, skipping S3 sync"
    fi
    
    # Upload log file to S3
    if [ -f "$ARCHIVE_LOGFILE" ]; then
        aws s3 cp "$ARCHIVE_LOGFILE" "$s3_path/logs/" --metadata "dataset=$dataset_name,timestamp=$timestamp"
        echo "✓ Uploaded log file to S3: $s3_path/logs/$ARCHIVE_LOGFILE"
    fi
    
    # Create and upload training manifest
    create_training_manifest "$dataset_name" "$timestamp" "$s3_path"
}

# Function to create training manifest
create_training_manifest() {
    local dataset_name=$1
    local timestamp=$2
    local s3_path=$3
    local manifest_file="training_manifest_${dataset_name}_${timestamp}.json"
    
    cat > "$manifest_file" << EOF
{
    "training_info": {
        "dataset_name": "$dataset_name",
        "timestamp": "$timestamp",
        "start_time": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
        "model": "$(jq -r '.model_name_or_path' $CONFIG)",
        "s3_location": "$s3_path",
        "config_file": "$CONFIG",
        "deepspeed_config": "$DEEPSPEED"
    },
    "hyperparameters": $(cat $CONFIG),
    "system_info": {
        "hostname": "$(hostname)",
        "gpu_count": "$(nvidia-smi -L | wc -l)",
        "aws_region": "$(aws configure get region || echo 'unknown')"
    }
}
EOF

    # Upload manifest to S3
    aws s3 cp "$manifest_file" "$s3_path/" --metadata "type=training-manifest"
    rm "$manifest_file"
    echo "✓ Created and uploaded training manifest"
}

# Function to check S3 access
check_s3_access() {
    echo "Checking S3 access to $S3_BUCKET..."
    aws s3 ls "$S3_BUCKET" >/dev/null 2>&1
    if [ $? -eq 0 ]; then
        echo "✓ S3 access confirmed"
        return 0
    else
        echo "✗ Cannot access S3 bucket: $S3_BUCKET"
        echo "Please check AWS credentials and bucket permissions"
        return 1
    fi
}

# Check S3 access before starting
if ! check_s3_access; then
    echo "Exiting due to S3 access issues"
    exit 1
fi

for DATASET in datasets/*.jsonl; do
  BASENAME=$(basename "$DATASET" .jsonl)
  WANDB_NAME="qwen-hebrew-finetuning-$BASENAME-with-qwen-30B"
  TIMESTAMP=$(date +%Y%m%d_%H%M%S)
  ARCHIVE_LOGFILE="train_${BASENAME}_${TIMESTAMP}.log"
  S3_LOC="${S3_BUCKET}/${S3_PREFIX}/${dataset_name}/${timestamp}"
  OUTPUT_DIR=$(jq -r '.output_dir' "$CONFIG")
  

  echo "================================================"
  echo "Training on $DATASET with wandb run name $WANDB_NAME"
  echo "Timestamp: $TIMESTAMP"
  echo "S3 Sync Location: $S3_LOC"
  echo "================================================"

  echo "Starting training, to trace the current log output: tail -f $ARCHIVE_LOGFILE"
  accelerate launch --config_file=$DEEPSPEED train.py --dataset_path "$DATASET" --wandb_name "$WANDB_NAME" | tee $ARCHIVE_LOGFILE

  echo "Training completed for $DATASET"
  
  # Sync to S3
  echo "Syncing results to S3..."
  sync_to_s3 "$S3_LOC" "$OUTPUT_DIR"
  
  echo "Completed processing $DATASET"
  echo "================================================"

done

echo "All training jobs completed!"
echo "Check S3 for checkpoints: aws s3 ls $S3_BUCKET/$S3_PREFIX/ --recursive"