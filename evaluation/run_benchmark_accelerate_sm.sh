#!/bin/bash
set -e  # Exit on any error

# ==========================================
# SageMaker-Compatible Benchmark Runner
# ==========================================

# Default values
DATASET_NAME="arc_ai2_heb,copa_heb,hellaswag_heb,mmlu_heb,psychometric_heb_math,psychometric_heb_analogies"
MAX_SAMPLES=3000
DEVICE="cuda:0"
BATCH_SIZE=8
PAD_TOKEN_ID=151643
DTYPE="bfloat16"
TOP_K=1
TEMPERATURE=1.0
BACKEND="vllm"

# SageMaker paths
SAGEMAKER_BASE="/opt/ml"
OUTPUT_DIR="${SAGEMAKER_BASE}/model/benchmark_results"
CODE_DIR="${SAGEMAKER_BASE}/input/data/code"
BENCHMARKS_DIR="${SAGEMAKER_BASE}/input/data/benchmarks"

# Model configuration - can be overridden by environment variables
MODEL_SOURCE="${MODEL_SOURCE:-huggingface}"  # Options: "huggingface" or "s3"
MODEL_NAME="${MODEL_NAME:-CohereLabs/aya-expanse-8b}"
S3_MODEL_PATH="${S3_MODEL_PATH:-}"  # Only used if MODEL_SOURCE=s3

# Local model path (will be downloaded to here)
LOCAL_MODEL_DIR="${SAGEMAKER_BASE}/model_cache"
mkdir -p "$LOCAL_MODEL_DIR"

echo "========================================="
echo "SageMaker LightEval Benchmark Runner"
echo "========================================="
echo "Model Source: $MODEL_SOURCE"
echo "Model Name/Path: $MODEL_NAME"
echo "Dataset: $DATASET_NAME"
echo "Max Samples: $MAX_SAMPLES"
echo "Backend: $BACKEND"
echo "========================================="

# ==========================================
# Function: Download model from HuggingFace
# ==========================================
download_from_huggingface() {
    local hf_model_name=$1
    local local_path=$2

    echo "📥 Downloading model from HuggingFace: $hf_model_name"

    # Use huggingface-cli to download
    python3 -c "
from huggingface_hub import snapshot_download
import os

model_path = snapshot_download(
    repo_id='$hf_model_name',
    cache_dir='$local_path',
    resume_download=True,
    local_files_only=False
)
print(f'Model downloaded to: {model_path}')

# Create a symlink for easier access
symlink_path = '$local_path/current_model'
if os.path.exists(symlink_path):
    os.remove(symlink_path)
os.symlink(model_path, symlink_path)
print(f'Symlink created at: {symlink_path}')
"

    echo "✅ Model downloaded successfully"
    MODEL_PATH="$local_path/current_model"
}

# ==========================================
# Function: Download model from S3
# ==========================================
download_from_s3() {
    local s3_path=$1
    local local_path=$2

    echo "📥 Downloading model from S3: $s3_path"

    # Create target directory
    mkdir -p "$local_path/s3_model"

    # Download from S3
    aws s3 sync "$s3_path" "$local_path/s3_model" --quiet

    if [ $? -eq 0 ]; then
        echo "✅ Model downloaded successfully from S3"
        MODEL_PATH="$local_path/s3_model"
    else
        echo "❌ Failed to download model from S3"
        exit 1
    fi
}

# ==========================================
# Download Model Based on Source
# ==========================================
if [ "$MODEL_SOURCE" == "huggingface" ]; then
    download_from_huggingface "$MODEL_NAME" "$LOCAL_MODEL_DIR"
elif [ "$MODEL_SOURCE" == "s3" ]; then
    if [ -z "$S3_MODEL_PATH" ]; then
        echo "❌ Error: S3_MODEL_PATH must be set when MODEL_SOURCE=s3"
        exit 1
    fi
    download_from_s3 "$S3_MODEL_PATH" "$LOCAL_MODEL_DIR"
else
    echo "❌ Error: MODEL_SOURCE must be 'huggingface' or 's3'"
    exit 1
fi

echo "Model ready at: $MODEL_PATH"

# ==========================================
# Setup Environment
# ==========================================
export HEB_BENCHMARKS_DIR_PATH="$BENCHMARKS_DIR"
export PYTHONPATH="$CODE_DIR:$PYTHONPATH"

echo "HEB_BENCHMARKS_DIR_PATH: $HEB_BENCHMARKS_DIR_PATH"
echo "PYTHONPATH: $PYTHONPATH"

# Determine custom tasks directory based on backend
if [[ "$BACKEND" == "vllm" ]]; then
    CUSTOM_TASKS="$CODE_DIR/custom_tasks_new_version"
elif [[ "$BACKEND" == "accelerate" ]]; then
    CUSTOM_TASKS="$CODE_DIR/custom_tasks"
fi

echo "Custom Tasks Dir: $CUSTOM_TASKS"

# ==========================================
# Create Output Structure
# ==========================================
RUN_TIMESTAMP=$(date +%Y-%m-%dT%H-%M-%S)
OUTPUT_RUN_DIR="$OUTPUT_DIR/$RUN_TIMESTAMP"
mkdir -p "$OUTPUT_RUN_DIR"

# Save run metadata
cat > "$OUTPUT_RUN_DIR/run_metadata.json" << EOF
{
    "model_source": "$MODEL_SOURCE",
    "model_name": "$MODEL_NAME",
    "s3_model_path": "$S3_MODEL_PATH",
    "datasets": "$DATASET_NAME",
    "max_samples": $MAX_SAMPLES,
    "backend": "$BACKEND",
    "timestamp": "$RUN_TIMESTAMP",
    "dtype": "$DTYPE",
    "batch_size": $BATCH_SIZE
}
EOF

# ==========================================
# Start Benchmark Execution
# ==========================================
START_TIME=$(date +%s)

IFS=',' read -ra DATASET_LIST <<< "$DATASET_NAME"

for DS_NAME in "${DATASET_LIST[@]}"; do
    TASK_CONFIG="community|${DS_NAME}|5|0"

    echo ""
    echo "========================================="
    echo "🚀 Running benchmark: $DS_NAME"
    echo "========================================="

    RESULTS_PATH_TEMPLATE="$OUTPUT_RUN_DIR/${DS_NAME}"

    if [[ "$BACKEND" == "vllm" ]]; then
        echo "Using VLLM backend"
        export VLLM_CACHE_DIR="$OUTPUT_RUN_DIR/vllm_cache"
        mkdir -p "$VLLM_CACHE_DIR"

        MODEL_CONFIG="model_name=$MODEL_PATH,tensor_parallel_size=4,gpu_memory_utilization=0.85,dtype=$DTYPE,generation_parameters={top_k:$TOP_K,temperature:$TEMPERATURE},max_model_length=2048"

        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        VLLM_WORKER_MULTIPROC_METHOD=spawn \
        python -m lighteval vllm \
            "$MODEL_CONFIG" \
            "$TASK_CONFIG" \
            --custom-tasks "$CUSTOM_TASKS" \
            --output-dir "$OUTPUT_RUN_DIR" \
            --save-details \
            --results-path-template "$RESULTS_PATH_TEMPLATE" \
            --max-samples "$MAX_SAMPLES"

    elif [[ "$BACKEND" == "accelerate" ]]; then
        echo "Using Accelerate backend"
        MODEL_CONFIG="model_name=$MODEL_PATH,model_parallel=True,device=$DEVICE,batch_size=$BATCH_SIZE,dtype=$DTYPE,generation_parameters={top_k:$TOP_K,temperature:$TEMPERATURE}"

        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        python -m lighteval accelerate \
            "$MODEL_CONFIG" \
            "$TASK_CONFIG" \
            --custom-tasks "$CUSTOM_TASKS" \
            --output-dir "$OUTPUT_RUN_DIR" \
            --save-details \
            --results-path-template "$RESULTS_PATH_TEMPLATE" \
            --max-samples "$MAX_SAMPLES"
    else
        echo "❌ Error: Unsupported backend '$BACKEND'"
        exit 1
    fi

    if [[ $? -eq 0 ]]; then
        echo "✅ Benchmark $DS_NAME completed successfully"
    else
        echo "❌ Benchmark $DS_NAME failed"
        exit 1
    fi
done

# ==========================================
# Finalize and Upload Results
# ==========================================
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo ""
echo "========================================="
echo "✅ All benchmarks completed!"
echo "Total duration: ${DURATION} seconds"
echo "========================================="

echo "${DURATION}" > "$OUTPUT_RUN_DIR/run_duration_seconds.txt"

# Upload results to S3
echo "📤 Uploading results to S3..."
if [ -f "$CODE_DIR/s3_upload.py" ]; then
    python "$CODE_DIR/s3_upload.py" \
        "$OUTPUT_RUN_DIR" \
        "gepeta-datasets" \
        --prefix "benchmark_results/heb_bench_results/"
fi

# Summarize results
echo "📊 Generating summary..."
if [ -f "$CODE_DIR/extract_benchmark_results.py" ]; then
    python -c "
import sys
sys.path.insert(0, '$CODE_DIR')
from extract_benchmark_results import summarize_benchmark_runs
summarize_benchmark_runs('$OUTPUT_DIR')
"
fi

# ==========================================
# Update Gradio Leaderboard (if provided)
# ==========================================
if [ -n "${LEADERBOARD_UPDATE_SCRIPT:-}" ] && [ -f "$CODE_DIR/$LEADERBOARD_UPDATE_SCRIPT" ]; then
    echo "📋 Updating Gradio leaderboard..."
    python "$CODE_DIR/$LEADERBOARD_UPDATE_SCRIPT" \
        --results-s3-path "s3://gepeta-datasets/benchmark_results/heb_bench_results/" \
        --model-name "$MODEL_NAME" \
        --timestamp "$RUN_TIMESTAMP"

    if [[ $? -eq 0 ]]; then
        echo "✅ Leaderboard updated successfully"
    else
        echo "⚠️  Warning: Leaderboard update failed"
    fi
fi

echo ""
echo "========================================="
echo "🎉 Benchmark run completed successfully!"
echo "Results saved to: $OUTPUT_RUN_DIR"
echo "========================================="