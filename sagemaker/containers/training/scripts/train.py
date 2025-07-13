#!/usr/bin/env python3
"""
SageMaker Training Script for Qwen Hebrew Fine-tuning
Complete final version with all optimizations and fixes applied
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
    
    def __init__(self, instance_type, max_seq_length=512):
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
    parser.add_argument('--max-seq-length', type=int, default=512)  # Reduced for memory efficiency
    
    # W&B arguments
    parser.add_argument('--wandb-project', type=str, default='qwen-hebrew-deepspeed')
    parser.add_argument('--wandb-entity', type=str, default=None)
    parser.add_argument('--wandb-run-name', type=str, default=None)
    
    # Performance testing
    parser.add_argument('--benchmark-mode', action='store_true', help='Run in benchmark mode with performance metrics')
    parser.add_argument('--max-steps', type=int, default=1000, help='Maximum training steps')
    
    return parser.parse_args()

def setup_distributed_training():
    """Setup distributed training environment for SageMaker"""
    
    # Check if we're in a distributed environment
    world_size = int(os.environ.get('WORLD_SIZE', '1'))
    rank = int(os.environ.get('RANK', '0'))
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    
    logger.info(f"Distributed setup - World size: {world_size}, Rank: {rank}, Local rank: {local_rank}")
    
    # Set up environment variables if not set by SageMaker
    if 'RANK' not in os.environ:
        os.environ['RANK'] = '0'
    if 'WORLD_SIZE' not in os.environ:
        os.environ['WORLD_SIZE'] = '1'
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = '0'
    if 'MASTER_ADDR' not in os.environ:
        os.environ['MASTER_ADDR'] = 'localhost'
    if 'MASTER_PORT' not in os.environ:
        os.environ['MASTER_PORT'] = '12355'
    
    # Log the environment variables
    logger.info(f"Environment setup:")
    logger.info(f"  RANK: {os.environ.get('RANK')}")
    logger.info(f"  WORLD_SIZE: {os.environ.get('WORLD_SIZE')}")
    logger.info(f"  LOCAL_RANK: {os.environ.get('LOCAL_RANK')}")
    logger.info(f"  MASTER_ADDR: {os.environ.get('MASTER_ADDR')}")
    logger.info(f"  MASTER_PORT: {os.environ.get('MASTER_PORT')}")
    
    return {
        'world_size': int(os.environ['WORLD_SIZE']),
        'rank': int(os.environ['RANK']),
        'local_rank': int(os.environ['LOCAL_RANK'])
    }

def load_instance_config(instance_type):
    """Load instance-specific configuration with optimized settings for 30B model"""
    
    default_configs = {
        'ml.p4d.24xlarge': {
            "batch_size_per_gpu": 1,
            "gradient_accumulation_steps": 4,
            "deepspeed_config": "configs/deepspeed/p4d_deepspeed_config.json",
            "gpu_count": 8,
            "gpu_type": "A100",
            "gpu_memory": "40GB",
            "estimated_hourly_cost": 32.77,
            "zero_stage": 3
        },
        'ml.p4de.24xlarge': {
            "batch_size_per_gpu": 1,
            "gradient_accumulation_steps": 2,
            "deepspeed_config": "configs/deepspeed/p4de_deepspeed_config.json",
            "gpu_count": 8,
            "gpu_type": "A100",
            "gpu_memory": "80GB",
            "estimated_hourly_cost": 40.96,
            "zero_stage": 3
        },
        'ml.p5.48xlarge': {
            "batch_size_per_gpu": 1,
            "gradient_accumulation_steps": 8,
            "gpu_count": 8,
            "gpu_type": "H100",
            "gpu_memory": "80GB",
            "estimated_hourly_cost": 98.32,
            "zero_stage": 3
        }
    }
    
    if instance_type in default_configs:
        logger.info(f"Using built-in configuration for {instance_type}")
        return default_configs[instance_type]
    else:
        logger.warning(f"Unknown instance type {instance_type}, using default P4DE config")
        return default_configs['ml.p4de.24xlarge']

def create_deepspeed_config(instance_config, distributed_info):
    """Create DeepSpeed configuration optimized for 30B model"""
    
    # Use ZeRO Stage 3 for large models
    zero_stage = instance_config.get("zero_stage", 3)
    
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
            "stage": zero_stage,
            "overlap_comm": True,
            "contiguous_gradients": True,
            "sub_group_size": 1e9,
            "reduce_bucket_size": "auto",
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto",
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "stage3_gather_16bit_weights_on_model_save": True
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
        "gradient_accumulation_steps": instance_config.get("gradient_accumulation_steps", 1),
        "gradient_clipping": 1.0,
        "wall_clock_breakdown": False,
        "memory_breakdown": False
    }
    
    # Additional optimizations for large models
    if zero_stage == 3:
        config["zero_optimization"]["offload_optimizer"] = {
            "device": "cpu",
            "pin_memory": True
        }
        config["zero_optimization"]["offload_param"] = {
            "device": "cpu",
            "pin_memory": True
        }
    
    # Save the config
    os.makedirs("configs/deepspeed", exist_ok=True)
    config_path = f"configs/deepspeed/optimized_deepspeed_config_stage{zero_stage}.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Created DeepSpeed config with ZeRO Stage {zero_stage}")
    return config_path, config

def setup_wandb(args, instance_config, distributed_info):
    """Initialize Weights & Biases with comprehensive configuration"""
    
    # Only initialize W&B on rank 0
    if distributed_info['rank'] != 0:
        logger.info(f"Rank {distributed_info['rank']}: Skipping W&B initialization")
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
            "gradient_accumulation_steps": instance_config.get("gradient_accumulation_steps", 1),
            "zero_stage": instance_config.get("zero_stage", 3),
            
            # Distributed training info
            "world_size": distributed_info['world_size'],
            "rank": distributed_info['rank'],
            
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
        # Try loading as a saved dataset first
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

def manual_deepspeed_training(model, tokenized_dataset, tokenizer, args, instance_config, metrics_callback, distributed_info):
    """Manual DeepSpeed training loop with proper device handling"""
    
    logger.info("=== Starting Manual DeepSpeed Training with Device Management ===")
    logger.info(f"Distributed info: {distributed_info}")
    
    # Set device for this process
    device = torch.device(f"cuda:{distributed_info['local_rank']}")
    logger.info(f"Process rank {distributed_info['rank']} using device: {device}")
    
    # Clear GPU cache before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        if distributed_info['rank'] == 0:  # Only log from rank 0
            for i in range(torch.cuda.device_count()):
                logger.info(f"GPU {i} memory before training: {torch.cuda.memory_allocated(i) / (1024**3):.2f} GB")
    
    # Create DeepSpeed config
    deepspeed_config_path, ds_config = create_deepspeed_config(instance_config, distributed_info)
    logger.info(f"DeepSpeed config created: {list(ds_config.keys())}")
    
    # Initialize DeepSpeed with proper distributed setup
    try:
        # Check if distributed is already initialized (SageMaker might do this)
        if not torch.distributed.is_initialized():
            logger.info("Initializing distributed training...")
            torch.distributed.init_process_group(
                backend="nccl",
                init_method="env://",
                world_size=distributed_info['world_size'],
                rank=distributed_info['rank']
            )
            logger.info("✅ Distributed training initialized")
        else:
            logger.info("✅ Distributed training already initialized")
        
        # Set local rank for current process
        torch.cuda.set_device(distributed_info['local_rank'])
        
        # Initialize DeepSpeed with the model
        model_engine, optimizer, _, lr_scheduler = deepspeed.initialize(
            model=model,
            config=ds_config,
            dist_init_required=False  # We already initialized
        )
        
        logger.info(f"✅ DeepSpeed engine created successfully")
        logger.info(f"DeepSpeed ZeRO stage: {ds_config['zero_optimization']['stage']}")
        
        # Log memory usage after DeepSpeed initialization (only from rank 0)
        if torch.cuda.is_available() and distributed_info['rank'] == 0:
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i) / (1024**3)
                reserved = torch.cuda.memory_reserved(i) / (1024**3)
                logger.info(f"GPU {i} after DeepSpeed init - Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB")
        
    except Exception as e:
        logger.error(f"❌ DeepSpeed initialization failed: {e}")
        # Log detailed GPU memory info for debugging
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                logger.error(f"GPU {i} memory info:")
                logger.error(f"  Allocated: {torch.cuda.memory_allocated(i) / (1024**3):.2f} GB")
                logger.error(f"  Reserved: {torch.cuda.memory_reserved(i) / (1024**3):.2f} GB")
                logger.error(f"  Total: {torch.cuda.get_device_properties(i).total_memory / (1024**3):.2f} GB")
        raise
    
    # Create data collator and dataloader
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    from torch.utils.data import DataLoader
    from torch.utils.data.distributed import DistributedSampler
    
    # Use DistributedSampler for multi-GPU training
    sampler = DistributedSampler(
        tokenized_dataset,
        num_replicas=distributed_info['world_size'],
        rank=distributed_info['rank'],
        shuffle=True
    ) if distributed_info['world_size'] > 1 else None
    
    dataloader = DataLoader(
        tokenized_dataset,
        batch_size=instance_config.get("batch_size_per_gpu", 1),
        collate_fn=data_collator,
        sampler=sampler,
        shuffle=(sampler is None),  # Only shuffle if no sampler
        num_workers=2,
        pin_memory=True
    )
    
    logger.info(f"DataLoader created with {len(dataloader)} batches")
    if sampler:
        logger.info(f"Using DistributedSampler for rank {distributed_info['rank']}")
    
    # Training loop
    model_engine.train()
    global_step = 0
    
    # Create mock TrainingArguments for metrics
    class MockArgs:
        def __init__(self, instance_config, args, distributed_info):
            self.per_device_train_batch_size = instance_config.get("batch_size_per_gpu", 1)
            self.gradient_accumulation_steps = instance_config.get("gradient_accumulation_steps", 1)
            self.world_size = distributed_info['world_size']
    
    mock_args = MockArgs(instance_config, args, distributed_info)
    
    logger.info(f"Starting training for {args.epochs} epochs...")
    logger.info(f"World size: {model_engine.world_size}")
    logger.info(f"Gradient accumulation steps: {mock_args.gradient_accumulation_steps}")
    logger.info(f"Effective batch size: {mock_args.per_device_train_batch_size * mock_args.world_size * mock_args.gradient_accumulation_steps}")
    
    for epoch in range(args.epochs):
        if distributed_info['rank'] == 0:
            logger.info(f"=== Epoch {epoch + 1}/{args.epochs} ===")
        
        # Set epoch for DistributedSampler
        if sampler:
            sampler.set_epoch(epoch)
        
        for step, batch in enumerate(dataloader):
            # Check max_steps
            if args.max_steps and global_step >= args.max_steps:
                if distributed_info['rank'] == 0:
                    logger.info(f"Reached max_steps ({args.max_steps}), stopping training")
                break
            
            try:
                # CRITICAL FIX: Direct device movement for each tensor
                if distributed_info['rank'] == 0 and global_step < 2:
                    logger.info("BEFORE moving to device:")
                    for key, value in batch.items():
                        if isinstance(value, torch.Tensor):
                            logger.info(f"  {key}: {value.device} (shape: {value.shape})")
                
                # Move each tensor explicitly to the device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(device, non_blocking=True)
                
                if distributed_info['rank'] == 0 and global_step < 2:
                    logger.info("AFTER moving to device:")
                    for key, value in batch.items():
                        if isinstance(value, torch.Tensor):
                            logger.info(f"  {key}: {value.device} (shape: {value.shape})")
                
                # Forward pass
                outputs = model_engine(**batch)
                loss = outputs.loss
                
                # Backward and step
                model_engine.backward(loss)
                model_engine.step()
                
                # Metrics and logging (only from rank 0)
                if distributed_info['rank'] == 0 and global_step % 10 == 0:
                    logger.info(f"✅ Epoch {epoch + 1}, Step {global_step}, Loss: {loss.item():.4f}")
                    
                    # Log memory usage periodically
                    if torch.cuda.is_available() and global_step % 100 == 0:
                        for i in range(torch.cuda.device_count()):
                            allocated = torch.cuda.memory_allocated(i) / (1024**3)
                            logger.info(f"GPU {i} memory at step {global_step}: {allocated:.2f} GB")
                
                # Callback for metrics (only from rank 0)
                if distributed_info['rank'] == 0:
                    metrics_callback.on_step_end(global_step, loss.item(), model_engine, mock_args)
                
                global_step += 1
                
            except torch.cuda.OutOfMemoryError as e:
                logger.error(f"CUDA OOM at step {global_step} on rank {distributed_info['rank']}: {e}")
                torch.cuda.empty_cache()
                raise
            except RuntimeError as e:
                if "device" in str(e).lower():
                    logger.error(f"Device error at step {global_step}: {e}")
                    logger.error(f"Current device: {device}")
                    if hasattr(model_engine, 'module'):
                        model_device = next(model_engine.module.parameters()).device
                    else:
                        model_device = next(model_engine.parameters()).device
                    logger.error(f"Model device: {model_device}")
                    for key, value in batch.items():
                        if isinstance(value, torch.Tensor):
                            logger.error(f"Batch '{key}' device: {value.device}")
                raise
        
        # Check if we should stop after epoch
        if args.max_steps and global_step >= args.max_steps:
            break
    
    if distributed_info['rank'] == 0:
        logger.info(f"✅ Training completed! Total steps: {global_step}")
    
    return model_engine, global_step

def main():
    args = parse_args()
    
    if args.command != 'train':
        logger.error(f"This script only supports 'train' command, got: {args.command}")
        return
    
    # Set memory management environment variables
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    # Setup distributed training
    distributed_info = setup_distributed_training()
    
    set_seed(args.seed)
    
    logger.info(f"🚀 Starting SageMaker training on {args.instance_type}")
    logger.info(f"📁 Dataset path: {args.train}")
    logger.info(f"🔧 Number of GPUs: {args.num_gpus}")
    logger.info(f"🖥️  Current host: {args.current_host}")
    logger.info(f"💾 CUDA memory management: max_split_size_mb=512")
    logger.info(f"🌐 Distributed training: {distributed_info}")
    
    # Load instance-specific configuration
    instance_config = load_instance_config(args.instance_type)
    logger.info(f"⚙️  Instance config: {instance_config}")
    
    # Setup W&B (only on rank 0)
    run_name = setup_wandb(args, instance_config, distributed_info)
    
    # Load tokenizer
    logger.info(f"🔤 Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with optimized settings for large models
    logger.info(f"🤖 Loading model: {args.model_name}")
    
    # DO NOT move model to GPU manually - let DeepSpeed handle it
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        use_cache=False,
        low_cpu_mem_usage=True,
        device_map=None  # Let DeepSpeed handle device placement
    )
    
    logger.info(f"📱 Model loaded on CPU. Available GPUs: {torch.cuda.device_count()}")
    
    # Enable gradient checkpointing for memory efficiency
    model.gradient_checkpointing_enable()
    logger.info("✅ Gradient checkpointing enabled")
    
    # Load and prepare dataset
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
        # Manual DeepSpeed training with device handling
        logger.info("🎯 Starting training with all optimizations applied")
        model_engine, total_steps = manual_deepspeed_training(
            model=model,
            tokenized_dataset=tokenized_dataset,
            tokenizer=tokenizer,
            args=args,
            instance_config=instance_config,
            metrics_callback=metrics_callback,
            distributed_info=distributed_info
        )
        training_successful = True
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        training_successful = False
        
        # Log failure to W&B
        if global_rank == 0 and run_name:
            wandb.log({"training/training_failed": True, "training/error": str(e)})
        raise
    
    finally:
        end_time = time.time()
        total_time = end_time - start_time
        
        # Log final metrics (only from rank 0)
        if distributed_info['rank'] == 0:
            final_metrics = {
                "training/total_time_seconds": total_time,
                "training/total_time_hours": total_time / 3600,
                "training/training_successful": training_successful,
                "training/cost_estimate": instance_config.get('estimated_hourly_cost', 32.77) * (total_time / 3600)
            }
            if run_name:  # Only if W&B was initialized
                wandb.log(final_metrics)
            
            logger.info(f"⏱️  Training completed in {total_time/3600:.2f} hours")
    
    # Save model (only from rank 0)
    if training_successful and distributed_info['rank'] == 0:
        logger.info(f"💾 Saving model to {args.model_dir}")
        
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
                "distributed_info": distributed_info
            }, f, indent=2)
        
        logger.info(f"📊 Training metrics saved to {metrics_file}")
    
    # Finish W&B run (only from rank 0)
    if distributed_info['rank'] == 0 and run_name:
        wandb.finish()
        logger.info("🎉 Training completed successfully!")

if __name__ == "__main__":
    main()