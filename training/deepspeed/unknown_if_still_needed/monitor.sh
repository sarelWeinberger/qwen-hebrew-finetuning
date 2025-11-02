#!/bin/bash

# Training Monitor Script
# Provides various ways to monitor the training process

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_FILE="$SCRIPT_DIR/logs/train.log"

show_help() {
    echo "Training Monitor - Monitor your Qwen fine-tuning progress"
    echo ""
    echo "Usage: $0 [OPTION]"
    echo ""
    echo "Options:"
    echo "  -h, --help     Show this help message"
    echo "  -l, --log      Show live log (tail -f)"
    echo "  -s, --status   Show training status"
    echo "  -p, --process  Show running processes"
    echo "  -g, --gpu      Show GPU utilization"
    echo "  -d, --disk     Show disk usage"
    echo ""
    echo "Examples:"
    echo "  $0 -l          # Follow the live training log"
    echo "  $0 -s          # Check if training is running"
    echo "  $0 -g          # Monitor GPU usage"
}

show_log() {
    if [ -f "$LOG_FILE" ]; then
        echo "Following training log: $LOG_FILE"
        echo "Press Ctrl+C to stop following..."
        echo "========================================="
        tail -f "$LOG_FILE"
    else
        echo "Log file not found: $LOG_FILE"
        echo "Training may not have started yet."
    fi
}

show_status() {
    echo "Training Status Check"
    echo "======================================"
    
    # Check for training processes
    TRAIN_PROCESSES=$(ps aux | grep -E "(train\.py|train_all\.sh)" | grep -v grep)
    if [ -n "$TRAIN_PROCESSES" ]; then
        echo "✓ Training processes found:"
        echo "$TRAIN_PROCESSES"
    else
        echo "✗ No training processes found"
    fi
    
    echo ""
    
    # Check log file
    if [ -f "$LOG_FILE" ]; then
        echo "✓ Log file exists: $LOG_FILE"
        echo "Last 5 lines:"
        tail -5 "$LOG_FILE"
    else
        echo "✗ Log file not found: $LOG_FILE"
    fi
    
    echo ""
    
    # Check GPU usage
    if command -v nvidia-smi &> /dev/null; then
        echo "GPU Status:"
        nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
    else
        echo "nvidia-smi not available"
    fi
}

show_processes() {
    echo "Training-related Processes"
    echo "======================================"
    ps aux | grep -E "(python|accelerate|train)" | grep -v grep
}

show_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        echo "GPU Utilization (refreshes every 2 seconds, press Ctrl+C to stop)"
        echo "========================================="
        watch -n 2 nvidia-smi
    else
        echo "nvidia-smi not available"
    fi
}

show_disk() {
    echo "Disk Usage"
    echo "======================================"
    df -h "$SCRIPT_DIR"
    echo ""
    echo "Directory sizes:"
    du -sh "$SCRIPT_DIR"/{logs,output,datasets,.venv} 2>/dev/null | sort -hr
}

# Parse command line arguments
case "${1:-}" in
    -h|--help)
        show_help
        ;;
    -l|--log)
        show_log
        ;;
    -s|--status)
        show_status
        ;;
    -p|--process)
        show_processes
        ;;
    -g|--gpu)
        show_gpu
        ;;
    -d|--disk)
        show_disk
        ;;
    "")
        show_status
        ;;
    *)
        echo "Unknown option: $1"
        echo "Use -h or --help for usage information"
        exit 1
        ;;
esac
