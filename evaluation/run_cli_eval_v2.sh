#!/bin/bash
# Default values
# pip install lighteval[vllm] emoji boto3
## check s3 
# python /home/ec2-user/qwen-hebrew-finetuning/evaluation/s3_utils.py

DEVICE="cuda:0"

DTYPE="bfloat16"
TOP_K=1
TEMPERATURE=1.0
FEW_SHOTS=5
MODEL_PATH="/home/ec2-user/models/qwen8-20-billion/long-step-517"
OUTPUT_DIR="./hebrew_benchmark_results"
RESULTS_PATH_TEMPLATE="$OUTPUT_DIR/scores_sum"
PYTHONPATH_DIR="/home/ec2-user/qwen-hebrew-finetuning/evaluation/heb_bnch"
HEB_BENCHMARKS_DIR_PATH="/home/ec2-user/qwen-hebrew-finetuning/evaluation/heb_bnch"
# export to env variable
export HEB_BENCHMARKS_DIR_PATH
echo "HEB_BENCHMARKS_DIR_PATH is set to: $HEB_BENCHMARKS_DIR_PATH"

CUSTOM_TASKS="custom_tasks_new_version"

# Function to display usage
which python
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -d, --dataset-name DATASET        Dataset name (default: $DATASET_NAME)"
    echo "  -p, --model-path PATH             Model path (default: $MODEL_PATH)"
    echo "  --device DEVICE                   Device (default: $DEVICE)"
    echo "  --batch-size SIZE                 Batch size (default: $BATCH_SIZE)"
    echo "  --dtype DTYPE                     Data type (default: $DTYPE)"
    echo "  --top-k K                         Top K value (default: $TOP_K)"
    echo "  --temperature TEMP                Temperature (default: $TEMPERATURE)"
    echo "  -o, --output-dir DIR              Output directory (default: $OUTPUT_DIR)"
    echo "  -r, --results-template TEMPLATE   Results path template (default: $RESULTS_PATH_TEMPLATE)"
    echo "  --pythonpath PATH                 Python path (default: $PYTHONPATH_DIR)"
    echo "  -h, --help                        Show this help message"
    echo ""
    echo "Example:"
    echo "  $0 --dataset-name hellaswag_heb --max-samples 50"
}

if [[ ! -d "$MODEL_PATH" ]]; then
    echo "Warning: Model path '$MODEL_PATH' does not exist"
fi

# Display configuration
echo "=== LightEval Configuration ==="
echo "Dataset Name: $DATASET_NAME"
echo "Max Samples: $MAX_SAMPLES"
echo "Model Path: $MODEL_PATH"
echo "Device: $DEVICE"
echo "Data Type: $DTYPE"
echo "Top K: $TOP_K"
echo "Temperature: $TEMPERATURE"
echo "Output Directory: $OUTPUT_DIR"
echo "Results Template: $RESULTS_PATH_TEMPLATE"
echo "Python Path: $PYTHONPATH_DIR"
echo "Custom Tasks: $CUSTOM_TASKS"
echo "Task Config: $TASK_CONFIG"
echo "Backend: $BACKEND"
echo "==============================="
echo ""

# Create a timestamped output subdirectory for this run
RUN_TIMESTAMP=$(date +%Y-%m-%dT%H-%M-%S)
OUTPUT_RUN_DIR="$OUTPUT_DIR/scores_sum/$RUN_TIMESTAMP"
mkdir -p "$OUTPUT_RUN_DIR"

# Start timing
START_TIME=$(date +%s)


mkdir -p "$OUTPUT_RUN_DIR"

RESULTS_PATH_TEMPLATE="$OUTPUT_RUN_DIR/$DS_NAME"

export VLLM_CACHE_DIR="$OUTPUT_RUN_DIR/vllm_cache"
mkdir -p "$VLLM_CACHE_DIR"
MODEL_CONFIG="model_name=$MODEL_PATH,override_chat_template=false,tensor_parallel_size=4,gpu_memory_utilization=0.85,dtype=$DTYPE,generation_parameters={top_k:$TOP_K,temperature:$TEMPERATURE},max_model_length=4096"
# MODEL_CONFIG="model_name=$MODEL_PATH,override_chat_template=false,tensor_parallel_size=1,gpu_memory_utilization=0.9,dtype=$DTYPE,max_num_seqs=128,generation_parameters={top_k:$TOP_K,temperature:$TEMPERATURE},max_model_length=4096,max_num_batched_tokens=8192"
# MODEL_CONFIG="model_name=$MODEL_PATH,override_chat_template=false,tensor_parallel_size=4,gpu_memory_utilization=0.85,dtype=$DTYPE,max_num_seqs=64,enforce_eager=false,enable_chunked_prefill=true,enable_prefix_caching=true,generation_parameters={top_k:$TOP_K,temperature:$TEMPERATURE},max_model_length=4096"

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
VLLM_WORKER_MULTIPROC_METHOD=spawn \
PYTHONPATH="$PYTHONPATH_DIR" \
python -m lighteval vllm \
    "$MODEL_CONFIG" \
    tasks.txt \
    --custom-tasks "$CUSTOM_TASKS" \
    --output-dir "$OUTPUT_RUN_DIR" \
    --save-details \
    --results-path-template "$RESULTS_PATH_TEMPLATE" \

# Check exit status
if [[ $? -eq 0 ]]; then
    echo ""
    echo "✅ LightEval has completed successfully!"
    echo "Results saved to: $OUTPUT_RUN_DIR"
else
    echo ""
    echo "❌ LightEval failed with exit code $?"
    exit 1
fi

# End timing and save duration
END_TIME=$(date +%s)
DURATION=$((END_TIME - START_TIME))
echo "Total run duration: ${DURATION} seconds"
echo "${DURATION}" > "$OUTPUT_RUN_DIR/run_duration_seconds.txt"
# Summarize results
# PYTHONPATH_DIR="/home/ec2-user/qwen-hebrew-finetuning/evaluation"
# export PYTHONPATH="$PYTHONPATH_DIR"
# Ensure we run helper scripts from this script's directory so calls work regardless of current cwd
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# If PYTHONPATH_DIR set earlier, keep it; otherwise default to the evaluation dir (script dir)
PYTHONPATH_DIR="${PYTHONPATH_DIR:-$SCRIPT_DIR}"
export PYTHONPATH="$PYTHONPATH_DIR"

# Run s3 upload and summarizer using absolute paths to the helper scripts
python "$SCRIPT_DIR/s3_upload.py" "$OUTPUT_RUN_DIR" gepeta-datasets --prefix benchmark_results/heb_benc_results/
python -c "from extract_benchmark_results import summarize_benchmark_runs; summarize_benchmark_runs('$OUTPUT_DIR/scores_sum')"
