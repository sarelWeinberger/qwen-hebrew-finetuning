#!/usr/bin/env python3
"""
SageMaker Training Script for Qwen Hebrew Fine-tuning
Supports P4d, P4de, and P5 instances with automatic configuration
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
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed,
    TrainerCallback
)
from datasets import load_from_disk, load_dataset
import deepspeed
import boto3
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SageMakerMetricsCallback(TrainerCallback):
    """Custom callback for SageMaker metrics collection and W&B logging."""
    
    def __init__(self, instance_type, max_seq_length=2048):
        super().__init__()
        self.instance_type = instance_type
        self.max_seq_length = max_seq_length
        self.total_tokens = 0
        self.steps = 0
        self.start_time = time.time()
        self.step_times = []
        
    def on_step_end(self, args, state, control, **kwargs):
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
            avg_step_time = sum(self.step_times[-10:]) / min(len(self.step_times), 10)  # Last 10 steps
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
            "training/global_step": state.global_step
        }
        
        # Add GPU metrics if available
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                metrics[f"gpu_{i}/memory_allocated_gb"] = torch.cuda.memory_allocated(i) / (1024**3)
                metrics[f"gpu_{i}/memory_reserved_gb"] = torch.cuda.memory_reserved(i) / (1024**3)
                metrics[f"gpu_{i}/utilization"] = torch.cuda.utilization(i) if hasattr(torch.cuda, 'utilization') else 0
        
        wandb.log(metrics, step=state.global_step)
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log training metrics to W&B."""
        if logs is None:
            return
        
        # Log loss and other training metrics
        wandb_logs = {}
        for key, value in logs.items():
            if key.startswith(('train_', 'eval_')):
                wandb_logs[f"training/{key}"] = value
            elif key == 'loss':
                wandb_logs["training/loss"] = value
                # Calculate perplexity
                wandb_logs["training/perplexity"] = torch.exp(torch.tensor(value)).item()
        
        if wandb_logs:
            wandb.log(wandb_logs, step=state.global_step)
        
        # Log final statistics
        if "train_runtime" in logs:
            total_time = time.time() - self.start_time
            wandb.log({
                "training/total_tokens": self.total_tokens,
                "training/total_time_hours": total_time / 3600,
                "training/avg_tokens_per_second": self.total_tokens / total_time,
                "training/final_step": self.steps
            }, step=state.global_step)

def parse_args():
    parser = argparse.ArgumentParser(description="SageMaker Qwen Hebrew Fine-tuning")
    
    # SageMaker specific arguments
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING', '/opt/ml/input/data/training'))
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
    config_map = {
        'ml.p4de.24xlarge': 'configs/instance_configs/p4de_config.json',
        'ml.p5.48xlarge': 'configs/instance_configs/p5_config.json',
        'ml.p5e.48xlarge': 'configs/instance_configs/p5e_config.json',
        'ml.p5en.48xlarge': 'configs/instance_configs/p5en_config.json'
    }
    
    config_path = config_map.get(instance_type)
    if not config_path or not os.path.exists(config_path):
        logger.warning(f"Config not found for {instance_type}, using default P4d config")
        config_path = 'configs/instance_configs/p4d_config.json'
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Loaded configuration for {instance_type}")
        return config
    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        # Return default configuration
        return {
            "batch_size_per_gpu": 2,
            "gradient_accumulation_steps": 4,
            "deepspeed_config": "configs/deepspeed/p4d_deepspeed_config.json"
        }

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
            "batch_size_per_gpu": instance_config.get("batch_size_per_gpu", 2),
            "gradient_accumulation_steps": instance_config.get("gradient_accumulation_steps", 4),
            
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
    
    try:
        # Try loading as a saved dataset first
        if os.path.exists(os.path.join(data_path, 'dataset_info.json')):
            dataset = load_from_disk(data_path)
            logger.info("Loaded dataset from disk")
        else:
            # Try loading as JSON files
            dataset = load_dataset("json", data_files=f"{data_path}/*.json")
            logger.info("Loaded dataset from JSON files")
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
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=dataset["train"].column_names if "train" in dataset else dataset.column_names
    )
    
    logger.info(f"Dataset tokenized. Train samples: {len(tokenized_dataset['train']) if 'train' in tokenized_dataset else len(tokenized_dataset)}")
    return tokenized_dataset

def main():
    args = parse_args()
    set_seed(args.seed)
    
    logger.info(f"Starting SageMaker training on {args.instance_type}")
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
    
    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    
    # Load and prepare dataset
    dataset = load_and_prepare_dataset(args.train, tokenizer, args.max_seq_length)
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Setup training arguments
    deepspeed_config = instance_config.get("deepspeed_config", "configs/deepspeed/p4d_deepspeed_config.json")
    
    training_args = TrainingArguments(
        output_dir=args.model_dir,
        per_device_train_batch_size=instance_config.get("batch_size_per_gpu", 2),
        gradient_accumulation_steps=instance_config.get("gradient_accumulation_steps", 4),
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        fp16=True,
        deepspeed=deepspeed_config if os.path.exists(deepspeed_config) else None,
        logging_steps=10,
        save_steps=500,
        save_total_limit=3,
        report_to=["wandb"],
        dataloader_num_workers=4,
        remove_unused_columns=False,
        gradient_checkpointing=True,
        warmup_ratio=0.03,
        learning_rate=1e-5,
        weight_decay=0.01,
        max_grad_norm=1.0,
        logging_first_step=True,
        load_best_model_at_end=False,  # Disable for benchmark mode
        metric_for_best_model="loss",
        greater_is_better=False
    )
    
    logger.info(f"Training arguments configured for {args.instance_type}")
    
    # Initialize custom callback
    metrics_callback = SageMakerMetricsCallback(
        instance_type=args.instance_type,
        max_seq_length=args.max_seq_length
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"] if "train" in dataset else dataset,
        eval_dataset=dataset.get("validation"),
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[metrics_callback]
    )
    
    logger.info("Trainer initialized, starting training...")
    
    # Start training
    start_time = time.time()
    try:
        trainer.train()
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
        trainer.save_model()
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
                "training_successful": training_successful
            }, f, indent=2)
        
        logger.info(f"Training metrics saved to {metrics_file}")
    
    # Finish W&B run
    wandb.finish()
    logger.info("Training completed successfully!")

if __name__ == "__main__":
    main()