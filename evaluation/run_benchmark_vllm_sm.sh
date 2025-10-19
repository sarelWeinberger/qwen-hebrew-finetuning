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

# Detect environment - SageMaker vs EC2
if [ -d "/opt/ml/model" ]; then
    echo "🔍 Detected: SageMaker Training Job"
    OUTPUT_DIR="/opt/ml/model/benchmark_results"
    LOCAL_MODEL_DIR="/opt/ml/model/model_cache"

    # Check if custom_tasks were uploaded
    if [ -d "/opt/ml/input/data/code" ]; then
        CODE_DIR="/opt/ml/input/data/code"
    else
        # No custom code uploaded - will fail if needed
        CODE_DIR="/opt/ml/code"
    fi

    # Benchmarks path
    BENCHMARKS_DIR="${HEB_BENCHMARKS_DIR_PATH:-/opt/ml/input/data/benchmarks}"
else
    echo "🔍 Detected: EC2 / Local environment"
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    OUTPUT_DIR="${SCRIPT_DIR}/benchmark_results"
    CODE_DIR="${SCRIPT_DIR}"
    BENCHMARKS_DIR="${BENCHMARKS_DIR:-${SCRIPT_DIR}/benchmarks}"
    LOCAL_MODEL_DIR="/tmp/model_cache"
fi

mkdir -p "$LOCAL_MODEL_DIR"
mkdir -p "$OUTPUT_DIR"

echo "📁 Paths configured:"
echo "   OUTPUT_DIR: $OUTPUT_DIR"
echo "   CODE_DIR: $CODE_DIR"
echo "   BENCHMARKS_DIR: $BENCHMARKS_DIR"
echo "   LOCAL_MODEL_DIR: $LOCAL_MODEL_DIR"

# Model configuration - can be overridden by environment variables
MODEL_SOURCE="${MODEL_SOURCE:-huggingface}"  # Options: "huggingface" or "s3"
MODEL_NAME="${MODEL_NAME:-CohereLabs/aya-expanse-8b}"
S3_MODEL_PATH="${S3_MODEL_PATH:-}"  # Only used if MODEL_SOURCE=s3

# HuggingFace Token - REQUIRED for gated models
HF_TOKEN="${HF_TOKEN:-}"  # Add your token here

# Export for Python to use
export HF_TOKEN
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"

echo "========================================="
echo "Environment Configuration"
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

    # Ensure the local path exists
    mkdir -p "$local_path"

    # Use system Python instead of venv Python
    # Check if we're in SageMaker and use the appropriate Python
    if [ -f "/opt/conda/bin/python" ]; then
        PYTHON_CMD="/opt/conda/bin/python"
    elif [ -f "/usr/bin/python3" ]; then
        PYTHON_CMD="/usr/bin/python3"
    else
        PYTHON_CMD="python3"
    fi

    echo "Using Python: $PYTHON_CMD"

    # Install dependencies - try multiple approaches
    echo "📦 Installing required dependencies..."

    # Try 1: Use --break-system-packages (safe in containers/SageMaker)
    if $PYTHON_CMD -m pip install --break-system-packages huggingface_hub transformers accelerate --quiet 2>/dev/null; then
        echo "✅ Dependencies installed with --break-system-packages"
    # Try 2: Create and use a venv
    elif $PYTHON_CMD -m venv "$local_path/venv" 2>/dev/null && source "$local_path/venv/bin/activate"; then
        echo "📦 Created virtual environment"
        pip install --upgrade pip --quiet
        pip install huggingface_hub transformers accelerate --quiet
        PYTHON_CMD="$local_path/venv/bin/python"
        echo "✅ Dependencies installed in venv"
    # Try 3: Use conda if available
    elif command -v conda &> /dev/null; then
        echo "📦 Using conda to install dependencies"
        conda install -y -c conda-forge huggingface_hub transformers accelerate --quiet
        echo "✅ Dependencies installed with conda"
    else
        echo "⚠️ Trying fallback installation method..."
        $PYTHON_CMD -m pip install --user --break-system-packages huggingface_hub transformers accelerate --quiet
        echo "✅ Dependencies installed successfully"
    fi

    # Use huggingface_hub to download the model
    $PYTHON_CMD - <<PYCODE
from huggingface_hub import snapshot_download
try:
    from huggingface_hub import login
except ImportError:
    # Fallback for older versions
    login = None
import os

hf_model_name = "$hf_model_name"
local_path = "$local_path"

# Try to authenticate with HuggingFace token
hf_token = os.environ.get('HF_TOKEN') or os.environ.get('HUGGING_FACE_HUB_TOKEN')
if hf_token:
    print("🔐 Authenticating with HuggingFace...")
    if login:
        login(token=hf_token)
        print("✅ Authentication successful")
    else:
        print("⚠️  login() not available, passing token directly to download")
else:
    print("⚠️  No HF_TOKEN found - trying without authentication...")

print(f"📦 Starting download for: {hf_model_name}")

model_path = snapshot_download(
    repo_id=hf_model_name,
    cache_dir=local_path,
    resume_download=True,
    local_files_only=False,
    token=hf_token
)
print(f"✅ Model downloaded to: {model_path}")

# Create a symlink for easier access
symlink_path = os.path.join(local_path, "current_model")
if os.path.exists(symlink_path):
    os.remove(symlink_path)
os.symlink(model_path, symlink_path)
print(f"🔗 Symlink created at: {symlink_path}")
PYCODE

    echo "✅ Model downloaded and linked successfully"
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

# Setup Python path for custom tasks
if [ -d "/opt/ml/input/data/code" ]; then
    # Add the parent directory to PYTHONPATH so Python can import the module
    export PYTHONPATH="/opt/ml/input/data:$PYTHONPATH"
    CUSTOM_TASKS="code"  # This is the module name

    echo "========================================="
    echo "DEBUG: Custom Tasks Module Check"
    echo "========================================="
    echo "PYTHONPATH: $PYTHONPATH"
    echo ""
    echo "Files in /opt/ml/input/data/code:"
    ls -la /opt/ml/input/data/code/
    echo ""
    echo "Python sys.path:"
    python3 -c "import sys; import pprint; pprint.pprint(sys.path)"
    echo ""
    echo "Trying to import 'code' module:"
    python3 -c "
import sys
try:
    import code as custom_module
    print('✅ Module imported successfully')
    print('Module file:', custom_module.__file__ if hasattr(custom_module, '__file__') else 'No __file__')
    print('Has TASKS_TABLE:', hasattr(custom_module, 'TASKS_TABLE'))
    if hasattr(custom_module, 'TASKS_TABLE'):
        print('TASKS_TABLE type:', type(custom_module.TASKS_TABLE))
        print('Number of tasks:', len(custom_module.TASKS_TABLE) if hasattr(custom_module.TASKS_TABLE, '__len__') else 'N/A')
    print('Module attributes:', dir(custom_module))
except Exception as e:
    print('❌ Failed to import module')
    print('Error:', str(e))
    import traceback
    traceback.print_exc()
"
    echo "========================================="
    echo ""
else
    CUSTOM_TASKS=""
    echo "⚠️  No custom tasks found at /opt/ml/input/data/code"
fi

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

    # Build lighteval command based on whether we have custom tasks
    if [ -n "$CUSTOM_TASKS" ]; then
        CUSTOM_TASKS_ARG="--custom-tasks $CUSTOM_TASKS"
        echo "Using custom tasks: $CUSTOM_TASKS"
    else
        CUSTOM_TASKS_ARG=""
        echo "No custom tasks - using default tasks"
    fi

    if [[ "$BACKEND" == "vllm" ]]; then
        echo "Using VLLM backend"
        export VLLM_CACHE_DIR="$OUTPUT_RUN_DIR/vllm_cache"
        mkdir -p "$VLLM_CACHE_DIR"

        MODEL_CONFIG="model_name=$MODEL_PATH,tensor_parallel_size=1,gpu_memory_utilization=0.85,dtype=$DTYPE,generation_parameters={top_k:$TOP_K,temperature:$TEMPERATURE},max_model_length=2048"

        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        VLLM_WORKER_MULTIPROC_METHOD=spawn \
        python -m lighteval vllm \
            "$MODEL_CONFIG" \
            "$TASK_CONFIG" \
            $CUSTOM_TASKS_ARG \
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
            $CUSTOM_TASKS_ARG \
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