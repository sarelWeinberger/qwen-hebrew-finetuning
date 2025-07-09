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
    
    # 🔧 SageMaker מעביר 'train' כארגומנט ראשון - positional argument!
    parser.add_argument('command', choices=['train', 'serve'], help='Command to run (train or serve)')
    
    # SageMaker specific arguments
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    
    # 🔧 תיקון לנתיב הדאטא 
    train_channel = os.environ.get('SM_CHANNEL_TRAINING', '/opt/ml/input/data/training')
    if train_channel and not train_channel.startswith('/'):
        # אם זה רק שם הערוץ (כמו "training"), תבנה את הנתיב המלא
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
    
    # מפה של קונפיגורציות ברירת מחדל - עם gradient_accumulation_steps = 1
    default_configs = {
        'ml.p4d.24xlarge': {
            "batch_size_per_gpu": 2,
            "gradient_accumulation_steps": 1,  # ← שונה מ-4 ל-1
            "deepspeed_config": "configs/deepspeed/p4d_deepspeed_config.json",
            "gpu_count": 8,
            "gpu_type": "A100",
            "gpu_memory": "40GB",
            "estimated_hourly_cost": 32.77
        },
        'ml.p4de.24xlarge': {
            "batch_size_per_gpu": 4,
            "gradient_accumulation_steps": 1,  # ← שונה מ-2 ל-1
            "deepspeed_config": "configs/deepspeed/p4de_deepspeed_config.json",
            "gpu_count": 8,
            "gpu_type": "A100",
            "gpu_memory": "80GB",
            "estimated_hourly_cost": 40.96
        },
        'ml.p5.48xlarge': {
            "batch_size_per_gpu": 6,
            "gradient_accumulation_steps": 1,  # ← שונה מ-2 ל-1
            "deepspeed_config": "configs/deepspeed/p5_deepspeed_config.json",
            "gpu_count": 8,
            "gpu_type": "H100",
            "gpu_memory": "80GB",
            "estimated_hourly_cost": 98.32
        },
        'ml.p5e.48xlarge': {
            "batch_size_per_gpu": 6,
            "gradient_accumulation_steps": 1,  # ← שונה מ-2 ל-1
            "deepspeed_config": "configs/deepspeed/p5e_deepspeed_config.json",
            "gpu_count": 8,
            "gpu_type": "H100",
            "gpu_memory": "80GB",
            "estimated_hourly_cost": 98.32
        },
        'ml.p5en.48xlarge': {
            "batch_size_per_gpu": 6,
            "gradient_accumulation_steps": 1,  # ← שונה מ-2 ל-1
            "deepspeed_config": "configs/deepspeed/p5en_deepspeed_config.json",
            "gpu_count": 8,
            "gpu_type": "H100",
            "gpu_memory": "80GB",
            "estimated_hourly_cost": 98.32
        }
    }
    
    # נסה לטעון מקובץ קודם
    config_map = {
        'ml.p4d.24xlarge': 'configs/instance_configs/p4d_config.json',
        'ml.p4de.24xlarge': 'configs/instance_configs/p4de_config.json',
        'ml.p5.48xlarge': 'configs/instance_configs/p5_config.json',
        'ml.p5e.48xlarge': 'configs/instance_configs/p5e_config.json',
        'ml.p5en.48xlarge': 'configs/instance_configs/p5en_config.json'
    }
    
    config_path = config_map.get(instance_type)
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            logger.info(f"Loaded configuration from file for {instance_type}")
            return config
        except Exception as e:
            logger.warning(f"Failed to load config from file: {e}")
    
    # השתמש בקונפיגורציה מובנית
    if instance_type in default_configs:
        logger.info(f"Using built-in configuration for {instance_type}")
        return default_configs[instance_type]
    else:
        logger.warning(f"Unknown instance type {instance_type}, using default P4d config")
        return default_configs['ml.p4d.24xlarge']

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
    
    # 🔧 וודא שהנתיב קיים
    if not os.path.exists(data_path):
        logger.error(f"Dataset path does not exist: {data_path}")
        # נסה נתיבים אלטרנטיביים
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
    
    # 🔧 הצג תוכן הדירקטוריה לדיבוג
    try:
        files = os.listdir(data_path)
        logger.info(f"Files in {data_path}: {files[:10]}")  # הראה את 10 הקבצים הראשונים
    except Exception as e:
        logger.warning(f"Could not list files in {data_path}: {e}")
    
    try:
        # Try loading as a saved dataset first (Hugging Face format)
        if os.path.exists(os.path.join(data_path, 'dataset_info.json')):
            dataset = load_from_disk(data_path)
            logger.info("Loaded dataset from disk (Hugging Face format)")
        else:
            # 🔧 בדוק איזה סוג קבצים יש בדירקטוריה
            files = os.listdir(data_path)
            json_files = [f for f in files if f.endswith('.json') or f.endswith('.jsonl')]
            parquet_files = [f for f in files if f.endswith('.parquet')]
            
            if json_files:
                logger.info(f"Found JSON files: {json_files[:5]}")
                # Try loading as JSON files
                json_path = os.path.join(data_path, "*.json*")
                dataset = load_dataset("json", data_files=json_path)
                logger.info("Loaded dataset from JSON files")
            elif parquet_files:
                logger.info(f"Found Parquet files: {parquet_files[:5]}")
                # Try loading as Parquet files
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
    
    # 🔧 התמודד עם פורמטים שונים של דאטאסט
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
    return {"train": tokenized_dataset}

def init_distributed_training(args):
    """Initialize distributed training for multi-GPU setups"""
    logger.info("=== Distributed Training Initialization ===")
    
    # Check if we have multiple GPUs
    if args.num_gpus > 1 and torch.cuda.is_available():
        logger.info(f"Multi-GPU detected ({args.num_gpus} GPUs), initializing distributed training...")
        
        # Get environment variables that SageMaker might set
        rank = int(os.environ.get('RANK', 0))
        world_size = int(os.environ.get('WORLD_SIZE', args.num_gpus))
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        logger.info(f"RANK: {rank}, WORLD_SIZE: {world_size}, LOCAL_RANK: {local_rank}")
        
        # Set master address and port if not already set
        if 'MASTER_ADDR' not in os.environ:
            os.environ['MASTER_ADDR'] = 'localhost'
            logger.info("Set MASTER_ADDR to localhost")
        
        if 'MASTER_PORT' not in os.environ:
            os.environ['MASTER_PORT'] = '12355'
            logger.info("Set MASTER_PORT to 12355")
        
        # Initialize distributed training if not already initialized
        if not torch.distributed.is_initialized():
            try:
                torch.distributed.init_process_group(
                    backend='nccl',
                    init_method='env://',
                    rank=rank,
                    world_size=world_size,
                    timeout=datetime.timedelta(minutes=30)
                )
                
                # Set the current device
                torch.cuda.set_device(local_rank)
                
                logger.info(f"✓ Distributed training initialized successfully")
                logger.info(f"  Rank: {torch.distributed.get_rank()}")
                logger.info(f"  World size: {torch.distributed.get_world_size()}")
                logger.info(f"  Device: cuda:{local_rank}")
                
            except Exception as e:
                logger.warning(f"Failed to initialize distributed training: {e}")
                logger.warning("Falling back to single-GPU training")
        else:
            logger.info("✓ Distributed training already initialized")
            logger.info(f"  Rank: {torch.distributed.get_rank()}")
            logger.info(f"  World size: {torch.distributed.get_world_size()}")
    else:
        logger.info("Single GPU or CPU training - no distributed initialization needed")
    
    logger.info("=" * 50)

def main():
    args = parse_args()
    
    # 🔧 בדוק שהפקודה היא train
    if args.command != 'train':
        logger.error(f"This script only supports 'train' command, got: {args.command}")
        return
    
    # 🔧 NEW: Initialize distributed training FIRST
    init_distributed_training(args)
    
    set_seed(args.seed)
    
    logger.info(f"Starting SageMaker training on {args.instance_type}")
    logger.info(f"Dataset path: {args.train}")
    logger.info(f"Number of GPUs: {args.num_gpus}")
    logger.info(f"Current host: {args.current_host}")
    
    # 🔧 דיבוג נתיבי SageMaker
    logger.info("=== SageMaker Environment ===")
    logger.info(f"SM_CHANNEL_TRAINING: {os.environ.get('SM_CHANNEL_TRAINING', 'NOT_SET')}")
    logger.info(f"SM_MODEL_DIR: {os.environ.get('SM_MODEL_DIR', 'NOT_SET')}")
    logger.info(f"Working directory: {os.getcwd()}")
    logger.info("===============================")
    
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

    # הוסף מיד אחרי:
    if torch.cuda.is_available():
        model = model.to("cuda:0")  # Move to specific GPU
        logger.info(f"Model moved to CUDA device 0. Available GPUs: {torch.cuda.device_count()}")
    else:
        logger.info("CUDA not available, using CPU")
    
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()

    # Load and prepare dataset
    dataset = load_and_prepare_dataset(
        data_path=args.train,
        tokenizer=tokenizer, 
        max_seq_length=args.max_seq_length
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Setup training arguments
    deepspeed_config = instance_config.get("deepspeed_config", f"configs/deepspeed/p4de_deepspeed_config.json")
    
    # 🔧 תיקון: וודא שmax_steps לא None אם לא רוצים להשתמש בו
    max_steps_value = args.max_steps if args.max_steps is not None and args.max_steps > 0 else -1
    deepspeed_training_config = deepspeed_config if os.path.exists(deepspeed_config) else None
    logger.info(f'deepspeed_training_config: {deepspeed_training_config}')
    
    training_args_dict = {
        "output_dir": args.model_dir,
        "fp16": True,
        "gradient_accumulation_steps": 1,  # קונקרטי
        "per_device_train_batch_size": 2,  # קונקרטי  
        "learning_rate": 1e-5,
        "weight_decay": 0.01,
        "num_train_epochs": args.epochs,
        "save_steps": 500,
        "save_total_limit": 3,
        "logging_steps": 10,
        "max_grad_norm": 1.0,
        "warmup_ratio": 0.03,
        "deepspeed": deepspeed_config if os.path.exists(deepspeed_config) else None,
        "report_to": ["wandb"],
        "remove_unused_columns": False,
        "dataloader_num_workers": 4,
        "gradient_checkpointing": True,
        "ddp_backend": "nccl" if torch.cuda.is_available() else None,
        "dataloader_pin_memory": False,
        "ddp_find_unused_parameters": False,
    }

    # Add max_steps if specified
    if args.max_steps and args.max_steps > 0:
        training_args_dict["max_steps"] = args.max_steps

    training_args = TrainingArguments(**training_args_dict)   

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

    #Debug logs
    # 🔧 Force DeepSpeed initialization 
    logger.info("=== DeepSpeed Initialization Debug ===")

    # Check if we're in distributed mode
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        logger.info(f"✓ Distributed initialized - rank: {torch.distributed.get_rank()}")
    else:
        logger.info("⚠ Distributed not initialized")

    # Force accelerator initialization
    if hasattr(trainer, 'accelerator'):
        logger.info(f"Accelerator: {trainer.accelerator}")
        
        # Try to access DeepSpeed engine before training
        if hasattr(trainer.accelerator, 'deepspeed_engine_wrapped'):
            engine = trainer.accelerator.deepspeed_engine_wrapped
            if engine is None:
                logger.error("✗ DeepSpeed engine is None - trying to reinitialize...")
                
                # Try to force re-initialization
                try:
                    trainer.accelerator.prepare_model(model)
                    logger.info("Attempted to re-prepare model")
                except Exception as e:
                    logger.error(f"Re-preparation failed: {e}")
            else:
                logger.info(f"✓ DeepSpeed engine: {type(engine)}")
        else:
            logger.error("✗ No deepspeed_engine_wrapped attribute")
    else:
        logger.error("✗ No accelerator found")

    logger.info("=" * 50)

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