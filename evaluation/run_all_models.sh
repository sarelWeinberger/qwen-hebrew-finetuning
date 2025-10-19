#!/bin/bash

MODEL_PATHS=(
    "/home/ec2-user/models/aya-23-8B"
    "/home/ec2-user/models/aya-23-35B"
    "/home/ec2-user/models/aya-expanse-32b"
    "/home/ec2-user/models/model260000"
    "/home/ec2-user/models/Qwen3-30B-A3B"
    "/home/ec2-user/models/Qwen3-14B"
    "/home/ec2-user/models/Qwen3-8B"
)
echo "Starting evaluation for all models..."
echo "$MODEL_PATHS"
for MODEL in "${MODEL_PATHS[@]}"; do
    if [[ ! -d "$MODEL" ]]; then
        echo "Warning: Model path '$MODEL' does not exist. Skipping..."
        continue
    fi

    echo "Running evaluation for model: $MODEL"
    /home/ec2-user/qwen-hebrew-finetuning/evaluation/run_cli_eval.sh --model-path "$MODEL"
done
# nohup ./run_all_models.sh > "run_all_models_$(date +%Y%m%d_%H%M%S).log" 2>&1 &