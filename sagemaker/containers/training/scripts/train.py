#!/usr/bin/env python3
"""
SageMaker Training Script for Qwen Hebrew Fine-tuning
Fixed version with proper multi-GPU setup and memory optimization
"""

import os
import json
import argparse
import torch
import torch.distributed as dist
import wandb
import time
import logging
import signal
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    set_seed
)
from datasets import load_from_disk, load_dataset
import deepspeed
from datetime import datetime, timedelta
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SageMakerMetricsCallback:
    """Custom callback for SageMaker metrics collection and W&B logging."""
    
    def __init__(self, instance_type, max_seq_length=4096):
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
    parser.add_argument('--instance-type', type=str, default=os.environ.get('SM_CURRENT_INSTANCE_TYPE', 'ml.p4de.24xlarge'))
    
    # Training arguments
    parser.add_argument('--model-name', type=str, default='Qwen/Qwen3-30B-A3B-Base')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max-seq-length', type=int, default=4096)
    
    # W&B arguments
    parser.add_argument('--wandb-project', type=str, default='qwen-hebrew-sagemaker')
    parser.add_argument('--wandb-entity', type=str, default=None)
    parser.add_argument('--wandb-run-name', type=str, default=None)
    
    # Performance testing
    parser.add_argument('--benchmark-mode', action='store_true', help='Run in benchmark mode with performance metrics')
    parser.add_argument('--max-steps', type=int, default=None, help='Maximum training steps for benchmarking')
    
    return parser.parse_args()

def load_instance_config(instance_type):
    """Load instance-specific configuration with memory optimizations"""
    
    default_configs = {
        'ml.p4d.24xlarge': {
            "batch_size_per_gpu": 1,
            "gradient_accumulation_steps": 16,
            "gpu_count": 8,
            "gpu_type": "A100",
            "gpu_memory": "40GB",
            "estimated_hourly_cost": 32.77
        },
        'ml.p4de.24xlarge': {
            "batch_size_per_gpu": 1,
            "gradient_accumulation_steps": 8,
            "gpu_count": 8,
            "gpu_type": "A100",
            "gpu_memory": "80GB",
            "estimated_hourly_cost": 40.96
        },
        'ml.p5.48xlarge': {
            "batch_size_per_gpu": 1,
            "gradient_accumulation_steps": 4,
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
        logger.warning(f"Unknown instance type {instance_type}, using default P4de config")
        return default_configs['ml.p4de.24xlarge']

def create_deepspeed_config(instance_config):
    """Create DeepSpeed configuration optimized for multi-GPU"""
    config = {
        "fp16": {
            "enabled": True,
            "loss_scale": 0,
            "loss_scale_window": 1000,
            "initial_scale_power": 16,
            "hysteresis": 2,
            "min_loss_scale": 1
        },
        "zero_optimization": {
            "stage": 2,
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "contiguous_gradients": True,
            "cpu_offload": False
        },
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": 1e-5,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "weight_decay": 0.01
            }
        },
        "scheduler": {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0,
                "warmup_max_lr": 1e-5,
                "warmup_num_steps": 100
            }
        },
        "train_micro_batch_size_per_gpu": instance_config.get("batch_size_per_gpu", 1),
        "gradient_accumulation_steps": instance_config.get("gradient_accumulation_steps", 8),
        "gradient_clipping": 1.0,
        "steps_per_print": 10,
        "wall_clock_breakdown": True,
        "dump_state": False
    }
    
    # Save the config
    os.makedirs("configs/deepspeed", exist_ok=True)
    config_path = "configs/deepspeed/multi_gpu_deepspeed_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    return config_path, config

def setup_distributed():
    """Setup distributed training properly for SageMaker with timeout handling"""
    
    # בדיקה אם distributed כבר מופעל
    if dist.is_initialized():
        logger.info("Distributed training already initialized")
        return True
    
    # הגדרת משתני הסביבה הנדרשים
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    
    # עבור SageMaker
    if 'SM_HOSTS' in os.environ:
        hosts = json.loads(os.environ['SM_HOSTS'])
        current_host = os.environ['SM_CURRENT_HOST']
        host_rank = hosts.index(current_host)
        num_gpus = int(os.environ.get('SM_NUM_GPUS', '8'))
        
        # Single node, multi-GPU (SageMaker בדרך כלל רק node אחד)
        master_addr = "127.0.0.1"
        master_port = "29500"
        world_size = num_gpus
        rank = local_rank
        
        # Multi-node setup (רק אם יש יותר מ-host אחד)
        if len(hosts) > 1:
            master_addr = hosts[0]
            master_port = "29500"
            world_size = len(hosts) * num_gpus
            rank = host_rank * num_gpus + local_rank
    else:
        # Fallback for local development
        master_addr = "127.0.0.1"
        master_port = "29500"
        world_size = torch.cuda.device_count() if torch.cuda.is_available() else 1
        rank = local_rank
    
    # הגדרת משתני סביבה
    os.environ['MASTER_ADDR'] = master_addr
    os.environ['MASTER_PORT'] = master_port
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(rank)
    os.environ['LOCAL_RANK'] = str(local_rank)
    
    # הגדרת timeout
    os.environ['NCCL_TIMEOUT'] = '600'
    os.environ['NCCL_DEBUG'] = 'INFO'
    
    logger.info(f"Distributed setup:")
    logger.info(f"  MASTER_ADDR: {master_addr}")
    logger.info(f"  MASTER_PORT: {master_port}")
    logger.info(f"  WORLD_SIZE: {world_size}")
    logger.info(f"  RANK: {rank}")
    logger.info(f"  LOCAL_RANK: {local_rank}")
    
    # הגדרת CUDA device
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        logger.info(f"Set CUDA device to: {local_rank}")
    
    # אתחול distributed עם timeout
    try:
        def timeout_handler(signum, frame):
            raise TimeoutError("Distributed initialization timed out")
        
        # הגדרת timeout של 60 שניות
        signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(60)
        
        logger.info("Attempting to initialize distributed training...")
        
        dist.init_process_group(
            backend='nccl',
            init_method=f'tcp://{master_addr}:{master_port}',
            world_size=world_size,
            rank=rank,
            timeout=timedelta(seconds=300)
        )
        
        signal.alarm(0)  # ביטול timeout
        
        logger.info(f"✅ Distributed training initialized successfully")
        logger.info(f"  World size: {dist.get_world_size()}")
        logger.info(f"  Rank: {dist.get_rank()}")
        logger.info(f"  Local rank: {local_rank}")
        
        return True
        
    except (TimeoutError, Exception) as e:
        signal.alarm(0)  # ביטול timeout
        logger.warning(f"⚠️ Failed to initialize distributed training: {e}")
        logger.warning("Continuing with single GPU training...")
        
        # מנקה משתני סביבה
        for env_var in ['MASTER_ADDR', 'MASTER_PORT', 'WORLD_SIZE', 'RANK']:
            if env_var in os.environ:
                del os.environ[env_var]
        
        return False

def setup_wandb(args, instance_config):
    """Initialize Weights & Biases with comprehensive configuration"""
    # Only initialize W&B on rank 0
    if dist.is_initialized() and dist.get_rank() != 0:
        return None
    
    if not dist.is_initialized() and int(os.environ.get('LOCAL_RANK', '0')) != 0:
        return None
    
    run_name = args.wandb_run_name or f"qwen-hebrew-{args.instance_type}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    # Calculate estimated cost
    hourly_cost = instance_config.get('estimated_hourly_cost', 40.96)
    estimated_total_cost = hourly_cost * args.epochs * 8
    
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=run_name,
        config={
            # Instance information
            "instance_type": args.instance_type,
            "gpu_count": instance_config.get("gpu_count", 8),
            "gpu_type": instance_config.get("gpu_type", "A100"),
            "gpu_memory": instance_config.get("gpu_memory", "80GB"),
            "estimated_hourly_cost": hourly_cost,
            "estimated_total_cost": estimated_total_cost,
            
            # Training configuration
            "model_name": args.model_name,
            "epochs": args.epochs,
            "max_seq_length": args.max_seq_length,
            "seed": args.seed,
            "batch_size_per_gpu": instance_config.get("batch_size_per_gpu", 1),
            "gradient_accumulation_steps": instance_config.get("gradient_accumulation_steps", 8),
            
            # Distributed training info
            "world_size": dist.get_world_size() if dist.is_initialized() else 1,
            "distributed_training": dist.is_initialized(),
            
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
    
    # Process in smaller chunks to save memory
    tokenized_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        batch_size=100,
        num_proc=4,
        remove_columns=train_dataset.column_names
    )
    
    logger.info(f"Dataset tokenized. Train samples: {len(tokenized_dataset)}")
    return tokenized_dataset

def manual_deepspeed_training(model, tokenized_dataset, tokenizer, args, instance_config, metrics_callback):
    """Manual DeepSpeed training loop with proper multi-GPU support"""
    
    logger.info("=== Starting Manual DeepSpeed Training ===")
    
    # Create DeepSpeed config
    deepspeed_config_path, ds_config = create_deepspeed_config(instance_config)
    logger.info(f"DeepSpeed config created: {list(ds_config.keys())}")
    
    # Initialize DeepSpeed
    try:
        # בדיקה אם distributed מופעל
        dist_init_required = not dist.is_initialized()
        
        model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
            model=model,
            config=ds_config,
            dist_init_required=dist_init_required
        )
        logger.info(f"✅ DeepSpeed engine created successfully")
        logger.info(f"Model device: {next(model_engine.parameters()).device}")
        logger.info(f"DeepSpeed world size: {model_engine.world_size}")
        logger.info(f"DeepSpeed local rank: {model_engine.local_rank}")
        
    except Exception as e:
        logger.error(f"❌ DeepSpeed initialization failed: {e}")
        raise
    
    # Create data collator and dataloader
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Use DistributedSampler only if we have multiple processes
    if model_engine.world_size > 1:
        sampler = DistributedSampler(
            tokenized_dataset,
            num_replicas=model_engine.world_size,
            rank=model_engine.global_rank,
            shuffle=True
        )
        logger.info(f"Using DistributedSampler with {model_engine.world_size} replicas")
    else:
        sampler = None
        logger.info("Using regular DataLoader (single GPU)")
    
    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=instance_config.get("batch_size_per_gpu", 1),
        collate_fn=data_collator,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=2,
        pin_memory=True
    )
    
    logger.info(f"DataLoader created with {len(dataloader)} batches")
    
    # Training loop
    model_engine.train()
    global_step = 0
    
    # Create mock TrainingArguments for metrics
    class MockArgs:
        def __init__(self, instance_config, world_size):
            self.per_device_train_batch_size = instance_config.get("batch_size_per_gpu", 1)
            self.gradient_accumulation_steps = instance_config.get("gradient_accumulation_steps", 8)
            self.world_size = world_size
    
    mock_args = MockArgs(instance_config, model_engine.world_size)
    
    logger.info(f"Starting training for {args.epochs} epochs...")
    logger.info(f"World size: {model_engine.world_size}")
    logger.info(f"Gradient accumulation steps: {mock_args.gradient_accumulation_steps}")
    
    for epoch in range(args.epochs):
        logger.info(f"=== Epoch {epoch + 1}/{args.epochs} ===")
        
        # Set epoch for DistributedSampler
        if sampler is not None:
            sampler.set_epoch(epoch)
        
        for step, batch in enumerate(dataloader):
            # Check max_steps
            if args.max_steps and global_step >= args.max_steps:
                logger.info(f"Reached max_steps ({args.max_steps}), stopping training")
                break
            
            # Move batch to device
            batch = {k: v.to(model_engine.device) for k, v in batch.items()}
            
            # Forward pass
            outputs = model_engine(**batch)
            loss = outputs.loss
            
            # Backward and step
            model_engine.backward(loss)
            model_engine.step()
            
            # Metrics and logging (only on rank 0)
            if (global_step % 10 == 0) and (model_engine.global_rank == 0):
                logger.info(f"Epoch {epoch + 1}, Step {global_step}, Loss: {loss.item():.4f}")
                
                # Callback for metrics
                if metrics_callback:
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
    
    # Setup distributed training first
    distributed_success = setup_distributed()
    
    # Get rank info
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    global_rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1
    
    logger.info(f"Starting SageMaker training on {args.instance_type}")
    logger.info(f"Local rank: {local_rank}, Global rank: {global_rank}, World size: {world_size}")
    logger.info(f"Distributed training: {'✅ Success' if distributed_success else '❌ Failed, using single GPU'}")
    logger.info(f"Dataset path: {args.train}")
    logger.info(f"Number of GPUs: {args.num_gpus}")
    logger.info(f"Current host: {args.current_host}")
    logger.info(f"Max sequence length: {args.max_seq_length}")
    
    # Load instance-specific configuration
    instance_config = load_instance_config(args.instance_type)
    logger.info(f"Instance config: {instance_config}")
    
    # Setup W&B (only on rank 0)
    run_name = setup_wandb(args, instance_config)
    
    # Load tokenizer
    logger.info(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with memory optimizations
    logger.info(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        use_cache=False,
        low_cpu_mem_usage=True,
        device_map=None  # Let DeepSpeed handle device placement
    )
    
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    
    logger.info(f"Model loaded. Parameter count: {model.num_parameters():,}")

    # Load dataset (each rank loads independently)
    tokenized_dataset = load_and_prepare_dataset(
        data_path=args.train,
        tokenizer=tokenizer, 
        max_seq_length=args.max_seq_length
    )
    
    # Initialize metrics callback (only on rank 0)
    metrics_callback = None
    if global_rank == 0:
        metrics_callback = SageMakerMetricsCallback(
            instance_type=args.instance_type,
            max_seq_length=args.max_seq_length
        )
    
    # Start training
    start_time = time.time()
    training_successful = False

    try:
        # Manual DeepSpeed training
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
        
        # Log final metrics (only on rank 0)
        if global_rank == 0 and run_name:
            final_metrics = {
                "training/total_time_seconds": total_time,
                "training/total_time_hours": total_time / 3600,
                "training/training_successful": training_successful,
                "training/cost_estimate": instance_config.get('estimated_hourly_cost', 40.96) * (total_time / 3600),
                "training/world_size": world_size
            }
            wandb.log(final_metrics)
            
            logger.info(f"Training completed in {total_time/3600:.2f} hours")
    
    # Save model (only on rank 0)
    if training_successful and global_rank == 0:
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
                "total_tokens": metrics_callback.total_tokens if metrics_callback else 0,
                "avg_tokens_per_second": (metrics_callback.total_tokens / total_time) if metrics_callback and total_time > 0 else 0,
                "estimated_cost": instance_config.get('estimated_hourly_cost', 40.96) * (total_time / 3600),
                "training_successful": training_successful,
                "total_steps": total_steps,
                "world_size": world_size,
                "distributed_training": distributed_success
            }, f, indent=2)
        
        logger.info(f"Training metrics saved to {metrics_file}")
    
    # Wait for all processes to complete
    if dist.is_initialized():
        dist.barrier()
    
    # Finish W&B run (only on rank 0)
    if global_rank == 0 and run_name:
        wandb.finish()
    
    logger.info("Training completed!")

if __name__ == "__main__":
    main()