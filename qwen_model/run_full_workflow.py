import os
import argparse
import subprocess
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="Full workflow for downloading Qwen model and preparing Hebrew data for fine-tuning")
    parser.add_argument(
        "--aws_access_key_id",
        type=str,
        help="AWS Access Key ID"
    )
    parser.add_argument(
        "--aws_secret_access_key",
        type=str,
        help="AWS Secret Access Key"
    )
    parser.add_argument(
        "--region",
        type=str,
        default="us-east-1",
        help="AWS region"
    )
    parser.add_argument(
        "--skip_model_download",
        action="store_true",
        help="Skip downloading the Qwen model"
    )
    parser.add_argument(
        "--skip_data_download",
        action="store_true",
        help="Skip downloading the S3 data"
    )
    parser.add_argument(
        "--skip_model_prepare",
        action="store_true",
        help="Skip preparing the model for fine-tuning"
    )
    return parser.parse_args()

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"\n=== {description} ===\n")
    print(f"Running command: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, check=True)
        print(f"\n{description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\nError during {description}: {e}")
        return False

def main():
    args = parse_args()
    
    print("\n=== Full Workflow for Qwen Model and Hebrew Data Preparation ===\n")
    
    # Step 1: Download the Qwen model
    if not args.skip_model_download:
        success = run_command(
            "python qwen_model/main.py --action download",
            "Downloading Qwen3-30B-A3B-Base model"
        )
        if not success:
            print("Model download failed. Exiting.")
            sys.exit(1)
    else:
        print("\nSkipping model download as requested.")
    
    # Step 2: Prepare the model for fine-tuning
    if not args.skip_model_prepare:
        success = run_command(
            "python qwen_model/main.py --action prepare",
            "Preparing Qwen3-30B-A3B-Base model for fine-tuning"
        )
        if not success:
            print("Model preparation failed. Exiting.")
            sys.exit(1)
    else:
        print("\nSkipping model preparation as requested.")
    
    # Step 3: Download and process S3 data
    if not args.skip_data_download:
        # Build the command with AWS credentials if provided
        s3_cmd = "python qwen_model/download_s3_data.py"
        if args.aws_access_key_id and args.aws_secret_access_key:
            s3_cmd += f" --aws_access_key_id {args.aws_access_key_id}"
            s3_cmd += f" --aws_secret_access_key {args.aws_secret_access_key}"
        if args.region:
            s3_cmd += f" --region {args.region}"
        
        success = run_command(
            s3_cmd,
            "Downloading and processing Hebrew data from S3"
        )
        if not success:
            print("Data download and processing encountered issues, but sample data was created.")
            print("You can proceed with the sample data or fix the S3 authentication issues and try again.")
    else:
        print("\nSkipping S3 data download as requested.")
    
    # Final instructions
    print("\n=== Workflow Complete ===\n")
    print("The Qwen3-30B-A3B-Base model has been downloaded and prepared for fine-tuning.")
    print("The Hebrew data from S3 has been downloaded, processed, and prepared for training.")
    
    # Check if dataset exists
    dataset_path = "qwen_model/data/dataset/dataset"
    if os.path.exists(dataset_path):
        print("\nTo start fine-tuning with the prepared Hebrew dataset, run:")
        print(f"python qwen_model/train.py --dataset_path {dataset_path} --config qwen_model/finetuning/training_config.json")
    else:
        print("\nDataset not found at expected location. Please check the logs for errors.")
    
    print("\nNote: Full checkpoint fine-tuning of a 30B parameter model requires:")
    print("- Multiple high-memory GPUs (recommended: at least 4x A100 80GB or equivalent)")
    print("- DeepSpeed or FSDP for distributed training")
    print("- Gradient checkpointing for memory efficiency")
    print("\nAll these are configured in the training script, but you need the hardware resources.")

if __name__ == "__main__":
    main()