import os
import argparse
import subprocess
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="Download and prepare Qwen3-30B-A3B-Base model for fine-tuning")
    parser.add_argument(
        "--action",
        type=str,
        choices=["download", "prepare", "all"],
        default="all",
        help="Action to perform: download the model, prepare for fine-tuning, or both"
    )
    parser.add_argument(
        "--skip_download",
        action="store_true",
        help="Skip downloading the model (use if already downloaded)"
    )
    parser.add_argument(
        "--skip_prepare",
        action="store_true",
        help="Skip preparing the model for fine-tuning"
    )
    return parser.parse_args()

def check_gpu_memory():
    """Check if there's enough GPU memory for full checkpoint fine-tuning"""
    try:
        import torch
        if not torch.cuda.is_available():
            print("WARNING: No CUDA-capable GPU detected. Fine-tuning will be extremely slow or impossible.")
            return False
        
        # Get GPU memory information
        gpu_count = torch.cuda.device_count()
        total_memory_gb = 0
        
        for i in range(gpu_count):
            memory_gb = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            total_memory_gb += memory_gb
            print(f"GPU {i}: {memory_gb:.2f} GB")
        
        print(f"Total GPU memory: {total_memory_gb:.2f} GB")
        
        # Check if there's enough memory for full checkpoint fine-tuning
        # A 30B parameter model requires approximately 60GB in 16-bit precision
        if total_memory_gb < 60:
            print("\nWARNING: You may not have enough GPU memory for full checkpoint fine-tuning.")
            print("Consider using a machine with more GPU memory or using parameter-efficient fine-tuning methods.")
            return False
        
        return True
    
    except ImportError:
        print("WARNING: Could not import torch to check GPU memory.")
        return False
    except Exception as e:
        print(f"WARNING: Error checking GPU memory: {e}")
        return False

def download_model():
    """Download the Qwen3-30B-A3B-Base model"""
    print("\n=== Downloading Qwen3-30B-A3B-Base model ===\n")
    
    # Check if model directory exists
    model_path = os.path.join(os.getcwd(), "qwen_model/model")
    if os.path.exists(os.path.join(model_path, "config.json")):
        print(f"Model config already exists at {model_path}")
        proceed = input("Do you want to continue downloading (may resume if incomplete)? (y/n): ")
        if proceed.lower() != 'y':
            print("Skipping download.")
            return
    
    # Run the download script
    try:
        subprocess.run([sys.executable, "qwen_model/download_model.py"], check=True)
        print("\nModel download completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\nError downloading model: {e}")
        sys.exit(1)

def prepare_for_finetuning():
    """Prepare the model for fine-tuning"""
    print("\n=== Preparing Qwen3-30B-A3B-Base model for fine-tuning ===\n")
    
    # Check if model exists
    model_path = os.path.join(os.getcwd(), "qwen_model/model")
    if not os.path.exists(os.path.join(model_path, "config.json")):
        print(f"Model not found at {model_path}. Please download the model first.")
        return
    
    # Check GPU memory
    check_gpu_memory()
    
    # Run the preparation script
    try:
        subprocess.run([sys.executable, "qwen_model/prepare_for_finetuning.py"], check=True)
        print("\nModel preparation completed successfully!")
    except subprocess.CalledProcessError as e:
        print(f"\nError preparing model: {e}")
        sys.exit(1)

def print_next_steps():
    """Print instructions for next steps"""
    print("\n=== Next Steps ===\n")
    print("1. Prepare your dataset:")
    print("   python qwen_model/prepare_dataset.py --input_file your_data.json --format [text|chat|instruction]")
    print("\n2. Start fine-tuning:")
    print("   python qwen_model/train.py --dataset_path qwen_model/finetuning/dataset/dataset --config qwen_model/finetuning/training_config.json")
    print("\nNote: Full checkpoint fine-tuning of a 30B parameter model requires:")
    print("- Multiple high-memory GPUs (recommended: at least 4x A100 80GB or equivalent)")
    print("- DeepSpeed or FSDP for distributed training")
    print("- Gradient checkpointing for memory efficiency")
    print("\nAll these are configured in the training script, but you need the hardware resources.")

def main():
    args = parse_args()
    
    print("\n=== Qwen3-30B-A3B-Base Model Setup ===\n")
    
    # Create directories
    os.makedirs("qwen_model/model", exist_ok=True)
    os.makedirs("qwen_model/finetuning", exist_ok=True)
    
    if args.action in ["download", "all"] and not args.skip_download:
        download_model()
    
    if args.action in ["prepare", "all"] and not args.skip_prepare:
        prepare_for_finetuning()
    
    print_next_steps()

if __name__ == "__main__":
    main()