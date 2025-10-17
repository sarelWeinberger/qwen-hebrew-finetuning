#!/bin/bash
# Default values
# pip install lighteval[vllm] emoji boto3
## check s3 
# python /home/ec2-user/qwen-hebrew-finetuning/evaluation/s3_utils.py
# DATASET_NAME="hellaswag_heb,mmlu_heb,copa_heb,arc_ai2_heb"
# DATASET_NAME="arc_ai2_heb,copa_heb,hellaswag_heb,mmlu_heb,psychometric_heb_math,psychometric_heb_analogies,psychometric_heb_restatement,psychometric_heb_sentence_complete_english,psychometric_heb_sentence_complete_hebrew,psychometric_heb_sentence_text_english,psychometric_heb_sentence_text_hebrew,psychometric_heb_understanding_hebrew"
DATASET_NAME="arc_ai2_heb,copa_heb,hellaswag_heb,mmlu_heb,psychometric_heb_math,psychometric_heb_analogies,psychometric_heb_restatement,psychometric_heb_sentence_complete_english,psychometric_heb_sentence_complete_hebrew"
# TODO: activate those benchmarks: gsm8k_heb isn't configured yet and all the psychometric bench below caused content length issue in vllm backend (RuntimeError: Worker failed with error 'Sampled token IDs exceed the max model length. Total number of tokens: 2048 > max_model_len: 2047)
# DATASET_NAME="gsm8k_heb,psychometric_heb_sentence_text_english,psychometric_heb_sentence_text_hebrew,psychometric_heb_understanding_hebrew"
# TODO: call all tasks togther will save a lot of time - currently each task is called separately and the backend uploading in each task the model to gpu
MAX_SAMPLES=3000

MODEL_PATH="/home/ec2-user/models/Qwen3-14B" #"/home/ec2-user/qwen-hebrew-finetuning/model260000" #"Qwen/Qwen3-30B-A3B-Base" #"/home/ec2-user/qwen-hebrew-finetuning/model260000"
DEVICE="cuda:0"
BATCH_SIZE=8
PAD_TOKEN_ID=151643 #qwen 151643, 
DTYPE="bfloat16"
TOP_K=1
TEMPERATURE=1.0
FEW_SHOTS=5
OUTPUT_DIR="./hebrew_benchmark_results"
RESULTS_PATH_TEMPLATE="$OUTPUT_DIR/scores_sum"
PYTHONPATH_DIR="/home/ec2-user/qwen-hebrew-finetuning/evaluation/heb_bnch"
HEB_BENCHMARKS_DIR_PATH="/home/ec2-user/qwen-hebrew-finetuning/evaluation/heb_bnch"
# export to env variable
export HEB_BENCHMARKS_DIR_PATH
echo "HEB_BENCHMARKS_DIR_PATH is set to: $HEB_BENCHMARKS_DIR_PATH"
# CUSTOM_TASKS="/home/ec2-user/noam/heb_bnch/custom_tasks" #tried to be suitable for new lighteval version
# CUSTOM_TASKS="custom_tasks"
# TASK_CONFIG="lighteval|$DATASET_NAME|$FEW_SHOTS"
# TASK_CONFIG="leaderboard|$DATASET_NAME|$FEW_SHOTS"
BACKEND="vllm"  # Options: "vllm" or "accelerate"
# TASK_CONFIG="community|mmlu_heb|5|0,community|copa_heb|5|0,community|hellaswag_heb|5|0"
if [[ "$BACKEND" == "vllm" ]]; then
    CUSTOM_TASKS="custom_tasks_new_version"
elif [[ "$BACKEND" == "accelerate" ]]; then
    echo "Using Accelerate backend"
    CUSTOM_TASKS="custom_tasks"
fi
# Function to display usage
which python
usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  -d, --dataset-name DATASET        Dataset name (default: $DATASET_NAME)"
    echo "  -m, --max-samples SAMPLES         Maximum samples (default: $MAX_SAMPLES)"
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

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--dataset-name)
            if [[ -n "$2" && ! "$2" =~ ^- ]]; then
                DATASET_NAME="$2"
                shift 2
            else
                echo "Error: --dataset-name requires a value"
                exit 1
            fi
            ;;
        -m|--max-samples)
            if [[ -n "$2" && ! "$2" =~ ^- ]]; then
                MAX_SAMPLES="$2"
                shift 2
            else
                echo "Error: --max-samples requires a value"
                exit 1
            fi
            ;;
        -p|--model-path)
            if [[ -n "$2" && "$2" != -* ]]; then
                MODEL_PATH="$2"
                shift 2
            else
                echo "Error: --model-path requires a value"
                exit 1
            fi
            ;;
        --device)
            if [[ -n "$2" && ! "$2" =~ ^- ]]; then
                DEVICE="$2"
                shift 2
            else
                echo "Error: --device requires a value"
                exit 1
            fi
            ;;
        --batch-size)
            if [[ -n "$2" && ! "$2" =~ ^- ]]; then
                BATCH_SIZE="$2"
                shift 2
            else
                echo "Error: --batch-size requires a value"
                exit 1
            fi
            ;;
        --dtype)
            if [[ -n "$2" && ! "$2" =~ ^- ]]; then
                DTYPE="$2"
                shift 2
            else
                echo "Error: --dtype requires a value"
                exit 1
            fi
            ;;
        --top-k)
            if [[ -n "$2" && ! "$2" =~ ^- ]]; then
                TOP_K="$2"
                shift 2
            else
                echo "Error: --top-k requires a value"
                exit 1
            fi
            ;;
        --temperature)
            if [[ -n "$2" && ! "$2" =~ ^- ]]; then
                TEMPERATURE="$2"
                shift 2
            else
                echo "Error: --temperature requires a value"
                exit 1
            fi
            ;;
        -o|--output-dir)
            if [[ -n "$2" && ! "$2" =~ ^- ]]; then
                OUTPUT_DIR="$2"
                shift 2
            else
                echo "Error: --output-dir requires a value"
                exit 1
            fi
            ;;
        -r|--results-template)
            if [[ -n "$2" && ! "$2" =~ ^- ]]; then
                RESULTS_PATH_TEMPLATE="$2"
                shift 2
            else
                echo "Error: --results-template requires a value"
                exit 1
            fi
            ;;
        --pythonpath)
            if [[ -n "$2" && ! "$2" =~ ^- ]]; then
                PYTHONPATH_DIR="$2"
                shift 2
            else
                echo "Error: --pythonpath requires a value"
                exit 1
            fi
            ;;
        --custom-tasks)
            if [[ -n "$2" && ! "$2" =~ ^- ]]; then
                CUSTOM_TASKS="$2"
                shift 2
            else
                echo "Error: --custom-tasks requires a value"
                exit 1
            fi
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            usage
            exit 1
            ;;
    esac
done

# Validate required parameters
if [[ -z "$DATASET_NAME" ]]; then
    echo "Error: Dataset name is required"
    exit 1
fi

if [[ ! "$MAX_SAMPLES" =~ ^[0-9]+$ ]]; then
    echo "Error: Max samples must be a positive integer"
    exit 1
fi

if [[ ! -d "$MODEL_PATH" ]]; then
    echo "Warning: Model path '$MODEL_PATH' does not exist"
fi

# Display configuration
echo "=== LightEval Configuration ==="
echo "Dataset Name: $DATASET_NAME"
echo "Max Samples: $MAX_SAMPLES"
echo "Model Path: $MODEL_PATH"
echo "Device: $DEVICE"
echo "Batch Size: $BATCH_SIZE"
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

# Accept comma-separated datasets
IFS=',' read -ra DATASET_LIST <<< "$DATASET_NAME"

for DS_NAME in "${DATASET_LIST[@]}"; do
    # TODO: make sure to call all the tasks together to save time, see https://huggingface.co/docs/lighteval/en/quicktour?utm_source=chatgpt.com#running-multiple-tasks
    TASK_CONFIG="community|${DS_NAME}|${FEW_SHOTS}"
    echo "\n=== Running LightEval for dataset: $DS_NAME ==="
    echo "Task Config: $TASK_CONFIG"
    echo "==============================="
    echo ""
    # Create output directory if it doesn't exist
    mkdir -p "$OUTPUT_RUN_DIR"

    # Set results path template to be inside the timestamped directory
    RESULTS_PATH_TEMPLATE="$OUTPUT_RUN_DIR/$DS_NAME"

    # Execute the command
    echo "Executing lighteval for $DS_NAME..."
    echo ""
    if [[ "$BACKEND" == "vllm" ]]; then
        echo "Using VLLM backend"
        export VLLM_CACHE_DIR="$OUTPUT_RUN_DIR/vllm_cache"
        mkdir -p "$VLLM_CACHE_DIR"
        MODEL_CONFIG="model_name=$MODEL_PATH,override_chat_template=false,tensor_parallel_size=4,gpu_memory_utilization=0.85,dtype=$DTYPE,generation_parameters={top_k:$TOP_K,temperature:$TEMPERATURE},max_model_length=2047"

        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        VLLM_WORKER_MULTIPROC_METHOD=spawn \
        PYTHONPATH="$PYTHONPATH_DIR" \
        python -m lighteval vllm \
            "$MODEL_CONFIG" \
            "$TASK_CONFIG" \
            --custom-tasks "$CUSTOM_TASKS" \
            --output-dir "$OUTPUT_RUN_DIR" \
            --save-details \
            --results-path-template "$RESULTS_PATH_TEMPLATE" \
            --max-samples "$MAX_SAMPLES"

    elif [[ "$BACKEND" == "accelerate" ]]; then
        MODEL_CONFIG="model_name=$MODEL_PATH,model_parallel=True,device=$DEVICE,batch_size=$BATCH_SIZE,dtype=$DTYPE,generation_parameters={top_k:$TOP_K,temperature:$TEMPERATURE}"
        # MODEL_CONFIG="model_name=$MODEL_PATH,model_parallel=True,batch_size=$BATCH_SIZE,dtype=$DTYPE,generation_parameters={top_k:$TOP_K,temperature:$TEMPERATURE}"

        echo "Using Accelerate backend"
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        PYTHONPATH="$PYTHONPATH_DIR" \
        python -m lighteval accelerate \
            "$MODEL_CONFIG" \
            "$TASK_CONFIG" \
            --custom-tasks "$CUSTOM_TASKS" \
            --output-dir "$OUTPUT_RUN_DIR" \
            --save-details \
            --results-path-template "$RESULTS_PATH_TEMPLATE" \
            --max-samples "$MAX_SAMPLES"

    else
        echo "Error: Unsupported backend '$BACKEND'. Use 'vllm' or 'accelerate'."
        exit 1
    fi

    # Check exit status
    if [[ $? -eq 0 ]]; then
        echo ""
        echo "✅ LightEval for $DS_NAME completed successfully!"
        echo "Results saved to: $OUTPUT_RUN_DIR"
    else
        echo ""
        echo "❌ LightEval for $DS_NAME failed with exit code $?"
        exit 1
    fi

done

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

#consider using a loop to eval multiple models
# echo "Starting batch evaluation for multiple models..."
# echo make sure you have all these models downloaded locally or change to the model you want to evaluate   
# MODEL_PATHS=(
#     "/home/ec2-user/models/aya-23-8B"
#     "/home/ec2-user/models/aya-23-35B"
#     "/home/ec2-user/models/aya-expanse-32B"
#     "/home/ec2-user/models/model260000"
#     "/home/ec2-user/models/Qwen3-30B-A3B"
#     "/home/ec2-user/models/Qwen3-32B"
#     "/home/ec2-user/models/Qwen3-14B"
#     "/home/ec2-user/models/Qwen3-8B"
# )
# check that the model paths exist
# for MODEL in "${MODEL_PATHS[@]}"; do
#     if [[ ! -d "$MODEL" ]]; then
#         echo "Warning: Model path '$MODEL' does not exist. Skipping..."
#         continue
#     fi
# done

# for MODEL in "${MODEL_PATHS[@]}"; do
#     echo "Running evaluation for model: $MODEL"
#     /home/ec2-user/qwen-hebrew-finetuning/evaluation/run_cli_eval.sh --model-path "$MODEL"
# done

# lighteval vllm "model_name=openai-community/gpt2,tensor_parallel_size=4,gpu_memory_utilization=0.85,dtype=bfloat16,generation_parameters={top_k:1,temperature:1.0},max_model_length=1024" "leaderboard|truthfulqa:mc|0" --max-samples 10 --output-dir ./test_output --save-details --results-path-template ./test_output/truthfulqa_test
# lighteval vllm "model_name=openai-community/gpt2,tensor_parallel_size=4,gpu_memory_utilization=0.85,dtype=bfloat16,generation_parameters={top_k:1,temperature:1.0},max_model_length=1024" "leaderboard|mmlu:college_chemistry|5" --max-samples 10 --output-dir ./test_output --save-details --results-path-template ./test_output/mmlu_test