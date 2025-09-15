#!/usr/bin/env python3
import os
import sys
from pathlib import Path

# Add to Python path
sys.path.insert(0, "/home/ec2-user/qwen-hebrew-finetuning/heb_bnch")

# Set environment variables - CORRECT PATH
os.environ["HEB_BNCH_DATA_PATH"] = "/home/ec2-user/qwen-hebrew-finetuning/heb_bnch/bnch_data"
os.environ["PYTHONPATH"] = "/home/ec2-user/qwen-hebrew-finetuning/heb_bnch"

print("Checking data files...")
data_path = Path("/home/ec2-user/qwen-hebrew-finetuning/heb_bnch/bnch_data")
benchmarks = ["arc_ai2_heb", "mmlu_heb", "copa_heb", "hellaswag_heb", "psychometric_heb_math"]

for benchmark in benchmarks:
    # Handle different folder structures
    if benchmark.startswith("psychometric_heb_"):
        subject = benchmark.replace("psychometric_heb_", "")
        benchmark_dir = data_path / "psychometric_heb" / subject
    else:
        benchmark_dir = data_path / benchmark
        
    train_file = benchmark_dir / "train.jsonl"
    validation_file = benchmark_dir / "validation.jsonl"
    
    print(f"\n{benchmark}:")
    print(f"  Directory exists: {benchmark_dir.exists()}")
    print(f"  train.jsonl exists: {train_file.exists()}")
    print(f"  validation.jsonl exists: {validation_file.exists()}")
    
    if benchmark_dir.exists():
        files = list(benchmark_dir.glob("*"))
        print(f"  Files in directory: {[f.name for f in files]}")

print("\nTrying to import custom_tasks...")
try:
    import custom_tasks
    print("✓ Successfully imported custom_tasks")
    
    # Try to access task definitions
    if hasattr(custom_tasks, '__dict__'):
        tasks = [attr for attr in dir(custom_tasks) if not attr.startswith('_')]
        print(f"Available tasks: {tasks}")
        
except ImportError as e:
    print(f"✗ Failed to import custom_tasks: {e}")

print("\nDone!")
