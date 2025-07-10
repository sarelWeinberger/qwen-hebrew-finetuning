#!/usr/bin/env python3
"""
SageMaker Training Script for Qwen Hebrew Fine-tuning
Fixed version with manual DeepSpeed initialization
"""

import os
import json
import argparse
import torch
import wandb
import time
import logging
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed,
    TrainerCallback
)
from datasets import load_from_disk, load_dataset
import deepspeed
import boto3
from datetime import datetime
from torch.optim import AdamW

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SageMakerMetricsCallback:
    """Custom callback for SageMaker metrics collection and W&B logging."""
    
    def __init__(self, instance_type, max_seq_length=2048):
        self.instance_type = instance_type
        self.max_seq_length = max_seq_length
        self.total_tokens = 0
        self.steps = 0
        self.start_time = time.time()
        self.step_times = []
        
    def on_step_end(self, step, loss, model_engine, args):
        """Track performance metrics for each step."""
        current_time = time.time()
        step_time = current_time - getattr(self, 'last_step_time', current_time)
        self.last_step_time = current_time
        self.step_times.append(step_time)
        
        # Calculate tokens processed in this step
        total_batch_size = args.per_device_train_batch_size * args.world_size
        if args.gradient_accumulation_steps > 0:
            total_batch_size *= args.gradient_accumulation_steps
            
        step_tokens = total_batch_size * self.max_seq_length
        self.total_tokens += step_tokens
        self.steps += 1
        
        # Calculate throughput
        if len(self.step_times) > 0:
            avg_step_time = sum(self.step_times[-10:]) / min(len(self.step_times), 10)
            tokens_per_second = step_tokens / avg_step_time if avg_step_time > 0 else 0
        else:
            tokens_per_second = 0
        
        # Log metrics to W&B
        metrics = {
            "training/tokens": self.total_tokens,
            "training/step_num": self.steps,
            "training/tokens_per_second": tokens_per_second,
            "training/step_time": step_time,
            "training/instance_type": self.instance_type,
            "training/global_step": step,
            "training/loss": loss
        }
        
        # Add GPU metrics if available
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                metrics[f"gpu_{i}/memory_allocated_gb"] = torch.cuda.memory_allocated(i) / (1024**3)
                metrics[f"gpu_{i}/memory_reserved_gb"] = torch.cuda.memory_reserved(i) / (1024**3)
        
        wandb.log(metrics, step=step)

def parse_args():
    parser = argparse.ArgumentParser(description="SageMaker Qwen Hebrew Fine-tuning")
    
    parser.add_argument('command', choices=['train', 'serve'], help='Command to run (train or serve)')
    
    # SageMaker specific arguments
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    
    train_channel = os.environ.get('SM_CHANNEL_TRAINING', '/opt/ml/input/data/training')
    if train_channel and not train_channel.startswith('/'):
        train_path = f'/opt/ml/input/data/{train_channel}'
    else:
        train_path = train_channel
    
    parser.add_argument('--train', type=str, default=train_path)
    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output/data'))
    parser.add_argument('--num-gpus', type=int, default=int(os.environ.get('SM_NUM_GPUS', '8')))
    parser.add_argument('--current-host', type=str, default=os.environ.get('SM_CURRENT_HOST', 'algo-1'))
    parser.add_argument('--hosts', type=str, default=os.environ.get('SM_HOSTS', '["algo-1"]'))
    
    # Instance type detection
    parser.add_argument('--instance-type', type=str, default=os.environ.get('SM_CURRENT_INSTANCE_TYPE', 'ml.p4d.24xlarge'))
    
    # Training arguments
    parser.add_argument('--model-name', type=str, default='Qwen/Qwen3-30B-A3B-Base')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-seq-length', type=int, default=2048)
    
    # W&B arguments
    parser.add_argument('--wandb-project', type=str, default='qwen-hebrew-sagemaker')
    parser.add_argument('--wandb-entity', type=str, default=None)
    parser.add_argument('--wandb-run-name', type=str, default=None)
    
    # Performance testing
    parser.add_argument('--benchmark-mode', action='store_true', help='Run in benchmark mode with performance metrics')
    parser.add_argument('--max-steps', type=int, default=None, help='Maximum training steps for benchmarking')
    
    return parser.parse_args()

def load_instance_config(instance_type):
    """Load instance-specific configuration"""
    
    default_configs = {
        'ml.p4d.24xlarge': {
            "batch_size_per_gpu": 1,
            "gradient_accumulation_steps": 1,
            "deepspeed_config": "configs/deepspeed/p4d_deepspeed_config.json",
            "gpu_count": 8,
            "gpu_type": "A100",
            "gpu_memory": "40GB",
            "estimated_hourly_cost": 32.77
        },
        'ml.p4de.24xlarge': {
            "batch_size_per_gpu": 1,
            "gradient_accumulation_steps": 1,
            "deepspeed_config": "configs/deepspeed/p4de_deepspeed_config.json",
            "gpu_count": 8,
            "gpu_type": "A100",
            "gpu_memory": "80GB",
            "estimated_hourly_cost": 40.96
        },
        'ml.p5.48xlarge': {
            "batch_size_per_gpu": 1,
            "gradient_accumulation_steps": 1,
            "deepspeed_config": "configs/deepspeed/p5_deepspeed_config.json",
            "gpu_count": 8,
            "gpu_type": "H100",
            "gpu_memory": "80GB",
            "estimated_hourly_cost": 98.32
        }
    }
    
    if instance_type in default_configs:
        logger.info(f"Using built-in configuration for {instance_type}")
        return default_configs[instance_type]
    else:
        logger.warning(f"Unknown instance type {instance_type}, using default P4d config")
        return default_configs['ml.p4d.24xlarge']

def create_deepspeed_config():
    """Create DeepSpeed configuration that works"""
    config = {
        "fp16": {"enabled": True},
        "zero_optimization": {
            "stage": 2,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "reduce_bucket_size": 500000000
        },
        "train_micro_batch_size_per_gpu": 1,
        "gradient_accumulation_steps": 1,
        "gradient_clipping": 1.0,
        "wall_clock_breakdown": False
    }
    
    # Save the config
    os.makedirs("configs/deepspeed", exist_ok=True)
    config_path = "configs/deepspeed/working_deepspeed_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    return config_path, config

def setup_wandb(args, instance_config):
    """Initialize Weights & Biases with comprehensive configuration"""
    run_name = args.wandb_run_name or f"qwen-hebrew-{args.instance_type}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Calculate estimated cost
    hourly_cost = instance_config.get('estimated_hourly_cost', 32.77)
    estimated_total_cost = hourly_cost * args.epochs * 8  # Rough estimate
    
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        config={
            # Instance information
            "instance_type": args.instance_type,
            "gpu_count": instance_config.get("gpu_count", 8),
            "gpu_type": instance_config.get("gpu_type", "A100"),
            "gpu_memory": instance_config.get("gpu_memory", "40GB"),
            "estimated_hourly_cost": hourly_cost,
            "estimated_total_cost": estimated_total_cost,
            
            # Training configuration
            "model_name": args.model_name,
            "epochs": args.epochs,
            "max_seq_length": args.max_seq_length,
            "seed": args.seed,
            "batch_size_per_gpu": instance_config.get("batch_size_per_gpu", 1),
            "gradient_accumulation_steps": instance_config.get("gradient_accumulation_steps", 1),
            
            # SageMaker information
            "sagemaker_job": True,
            "current_host": args.current_host,
            "num_gpus": args.num_gpus,
            "benchmark_mode": args.benchmark_mode,
            
            # Timestamp
            "start_time": datetime.now().isoformat()
        }
    )
    
    logger.info(f"Initialized W&B run: {run_name}")
    return run_name

def load_and_prepare_dataset(data_path, tokenizer, max_seq_length):
    """Load and prepare dataset for training"""
    logger.info(f"Loading dataset from {data_path}")
    
    if not os.path.exists(data_path):
        logger.error(f"Dataset path does not exist: {data_path}")
        alternative_paths = [
            '/opt/ml/input/data/training',
            '/opt/ml/input/data/train',
            '/opt/ml/input/data/'
        ]
        
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                logger.info(f"Found alternative path: {alt_path}")
                data_path = alt_path
                break
        else:
            raise FileNotFoundError(f"No valid dataset path found. Tried: {[data_path] + alternative_paths}")
    
    try:
        files = os.listdir(data_path)
        logger.info(f"Files in {data_path}: {files[:10]}")
    except Exception as e:
        logger.warning(f"Could not list files in {data_path}: {e}")
    
    try:
        # Try loading as a saved dataset first (Hugging Face format)
        if os.path.exists(os.path.join(data_path, 'dataset_info.json')):
            dataset = load_from_disk(data_path)
            logger.info("Loaded dataset from disk (Hugging Face format)")
        else:
            files = os.listdir(data_path)
            json_files = [f for f in files if f.endswith('.json') or f.endswith('.jsonl')]
            parquet_files = [f for f in files if f.endswith('.parquet')]
            
            if json_files:
                logger.info(f"Found JSON files: {json_files[:5]}")
                json_path = os.path.join(data_path, "*.json*")
                dataset = load_dataset("json", data_files=json_path)
                logger.info("Loaded dataset from JSON files")
            elif parquet_files:
                logger.info(f"Found Parquet files: {parquet_files[:5]}")
                parquet_path = os.path.join(data_path, "*.parquet")
                dataset = load_dataset("parquet", data_files=parquet_path)
                logger.info("Loaded dataset from Parquet files")
            else:
                raise ValueError(f"No supported dataset files found in {data_path}. Files: {files}")
                
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        raise
    
    # Tokenization function
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_seq_length,
            return_tensors="pt"
        )
    
    # Tokenize dataset
    logger.info("Tokenizing dataset...")
    
    if isinstance(dataset, dict) and "train" in dataset:
        train_dataset = dataset["train"]
    else:
        train_dataset = dataset
    
    tokenized_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=train_dataset.column_names
    )
    
    logger.info(f"Dataset tokenized. Train samples: {len(tokenized_dataset)}")
    return tokenized_dataset

def manual_deepspeed_training(model, tokenized_dataset, tokenizer, args, instance_config, metrics_callback):
    """Manual DeepSpeed training loop - based on working notebook version"""
    
    logger.info("=== Starting Manual DeepSpeed Training ===")
    
    # Create DeepSpeed config
    deepspeed_config_path, ds_config = create_deepspeed_config()
    logger.info(f"DeepSpeed config created: {list(ds_config.keys())}")
    
    # Create optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=1e-5,
        betas=[0.9, 0.999],
        eps=1e-8,
        weight_decay=0.01
    )
    logger.info("Optimizer created")
    
    # Initialize DeepSpeed
    try:
        model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
            model=model,
            optimizer=optimizer,
            config=ds_config
        )
        logger.info(f"✅ DeepSpeed engine created: {type(model_engine)}")
        
    except Exception as e:
        logger.error(f"❌ DeepSpeed initialization failed: {e}")
        raise
    
    # Create data collator and dataloader
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    from torch.utils.data import DataLoader
    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=instance_config.get("batch_size_per_gpu", 1),
        collate_fn=data_collator,
        shuffle=True
    )
    
    logger.info(f"DataLoader created with {len(dataloader)} batches")
    
    # Training loop
    model_engine.train()
    global_step = 0
    
    # Create mock TrainingArguments for metrics
    class MockArgs:
        def __init__(self, instance_config, args):
            self.per_device_train_batch_size = instance_config.get("batch_size_per_gpu", 1)
            self.gradient_accumulation_steps = instance_config.get("gradient_accumulation_steps", 1)
            self.world_size = args.num_gpus
    
    mock_args = MockArgs(instance_config, args)
    
    logger.info(f"Starting training for {args.epochs} epochs...")
    
    for epoch in range(args.epochs):
        logger.info(f"=== Epoch {epoch + 1}/{args.epochs} ===")
        
        for step, batch in enumerate(dataloader):
            # Check max_steps
            if args.max_steps and global_step >= args.max_steps:
                logger.info(f"Reached max_steps ({args.max_steps}), stopping training")
                break
            
            # Forward pass
            outputs = model_engine(**batch)
            loss = outputs.loss
            
            # Backward and step
            model_engine.backward(loss)
            model_engine.step()
            
            # Metrics and logging
            if global_step % 10 == 0:
                logger.info(f"Epoch {epoch + 1}, Step {global_step}, Loss: {loss.item():.4f}")
            
            # Callback for metrics
            metrics_callback.on_step_end(global_step, loss.item(), model_engine, mock_args)
            
            global_step += 1
        
        # Check if we should stop after epoch
        if args.max_steps and global_step >= args.max_steps:
            break
    
    logger.info(f"✅ Training completed! Total steps: {global_step}")
    return model_engine, global_step

def main():
    args = parse_args()
    
    if args.command != 'train':
        logger.error(f"This script only supports 'train' command, got: {args.command}")
        return
    
    set_seed(args.seed)
    
    logger.info(f"Starting SageMaker training on {args.instance_type}")
    logger.info(f"Dataset path: {args.train}")
    logger.info(f"Number of GPUs: {args.num_gpus}")
    logger.info(f"Current host: {args.current_host}")
    
    # Load instance-specific configuration
    instance_config = load_instance_config(args.instance_type)
    logger.info(f"Instance config: {instance_config}")
    
    # Setup W&B
    run_name = setup_wandb(args, instance_config)
    
    # Load tokenizer
    logger.info(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model
    logger.info(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        use_cache=False,
        low_cpu_mem_usage=True
    )

    # Move model to GPU
    if torch.cuda.is_available():
        model = model.to("cuda:0")
        logger.info(f"Model moved to CUDA device 0. Available GPUs: {torch.cuda.device_count()}")
    else:
        logger.info("CUDA not available, using CPU")
    
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()

    # Load and prepare dataset
    tokenized_dataset = load_and_prepare_dataset(
        data_path=args.train,
        tokenizer=tokenizer, 
        max_seq_length=args.max_seq_length
    )
    
    # Initialize metrics callback
    metrics_callback = SageMakerMetricsCallback(
        instance_type=args.instance_type,
        max_seq_length=args.max_seq_length
    )
    
    # Start training
    start_time = time.time()
    training_successful = False

    try:
        # Manual DeepSpeed training (this is what worked!)
        model_engine, total_steps = manual_deepspeed_training(
            model=model,
            tokenized_dataset=tokenized_dataset,
            tokenizer=tokenizer,
            args=args,
            instance_config=instance_config,
            metrics_callback=metrics_callback
        )
        training_successful = True
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        training_successful = False
        raise
    
    finally:
        end_time = time.time()
        total_time = end_time - start_time
        
        # Log final metrics
        final_metrics = {
            "training/total_time_seconds": total_time,
            "training/total_time_hours": total_time / 3600,
            "training/training_successful": training_successful,
            "training/cost_estimate": instance_config.get('estimated_hourly_cost', 32.77) * (total_time / 3600)
        }
        wandb.log(final_metrics)
        
        logger.info(f"Training completed in {total_time/3600:.2f} hours")
    
    # Save model
    if training_successful:
        logger.info(f"Saving model to {args.model_dir}")
        
        # Save the underlying model (not the DeepSpeed engine)
        if hasattr(model_engine, 'module'):
            model_to_save = model_engine.module
        else:
            model_to_save = model_engine
            
        model_to_save.save_pretrained(args.model_dir)
        tokenizer.save_pretrained(args.model_dir)
        
        # Save training metrics
        metrics_file = os.path.join(args.output_data_dir, "training_metrics.json")
        os.makedirs(args.output_data_dir, exist_ok=True)
        
        with open(metrics_file, 'w') as f:
            json.dump({
                "instance_type": args.instance_type,
                "total_time_hours": total_time / 3600,
                "total_tokens": metrics_callback.total_tokens,
                "avg_tokens_per_second": metrics_callback.total_tokens / total_time,
                "estimated_cost": instance_config.get('estimated_hourly_cost', 32.77) * (total_time / 3600),
                "training_successful": training_successful,
                "total_steps": total_steps
            }, f, indent=2)
        
        logger.info(f"Training metrics saved to {metrics_file}")
    
    # Finish W&B run
    wandb.finish()
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()