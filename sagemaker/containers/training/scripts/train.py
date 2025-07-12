#!/usr/bin/env python3
"""
SageMaker Training Script for Qwen Hebrew Fine-tuning
Optimized DataParallel with Heavy Memory Optimizations
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
    DataCollatorForLanguageModeling,
    set_seed,
    AutoConfig
)
from datasets import load_dataset
from datetime import datetime
from torch.utils.data import DataLoader
import gc

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser(description="SageMaker Qwen Hebrew Fine-tuning - Optimized DataParallel")
    
    parser.add_argument('command', choices=['train'], help='Command to run')
    
    # SageMaker paths
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING', '/opt/ml/input/data/training'))
    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output/data'))
    
    # Instance info
    parser.add_argument('--num-gpus', type=int, default=int(os.environ.get('SM_NUM_GPUS', '8')))
    parser.add_argument('--instance-type', type=str, default=os.environ.get('SM_CURRENT_INSTANCE_TYPE', 'ml.p4de.24xlarge'))
    
    # Training parameters
    parser.add_argument('--model-name', type=str, default='Qwen/Qwen3-30B-A3B-Base')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--max-steps', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=42)
    
    # AGGRESSIVE memory optimization - TESTED VALUES
    parser.add_argument('--max-seq-length', type=int, default=512)  # Tested and works
    parser.add_argument('--gradient-accumulation-steps', type=int, default=64)  # Tested and works
    parser.add_argument('--batch-size', type=int, default=1)  # Minimal for memory
    
    parser.add_argument('--learning-rate', type=float, default=1e-5)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--warmup-steps', type=int, default=100)
    
    # Memory management - TESTED CONFIGURATION
    parser.add_argument('--cpu-offload', action='store_true', default=True, help='Offload optimizer to CPU - TESTED')
    parser.add_argument('--mixed-precision', action='store_true', default=False, help='Use mixed precision - DISABLED (conflicts with CPU offload)')
    
    # W&B
    parser.add_argument('--wandb-project', type=str, default='qwen-hebrew-optimized')
    parser.add_argument('--wandb-run-name', type=str, default=None)
    
    return parser.parse_args()

def setup_wandb(args):
    """Setup W&B logging"""
    run_name = args.wandb_run_name or f"optimized-{args.instance_type}-{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        config={
            "instance_type": args.instance_type,
            "model_name": args.model_name,
            "epochs": args.epochs,
            "max_seq_length": args.max_seq_length,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "batch_size": args.batch_size,
            "approach": "optimized_dataparallel",
            "num_gpus": args.num_gpus,
            "cpu_offload": args.cpu_offload,
            "mixed_precision": args.mixed_precision
        }
    )
    
    logger.info(f"W&B initialized: {run_name}")
    return run_name

def aggressive_memory_cleanup():
    """Aggressive memory cleanup"""
    if torch.cuda.is_available():
        try:
            torch.cuda.init()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            for i in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(i)
        except Exception as e:
            logger.warning(f"Memory cleanup warning: {e}")
    gc.collect()

def load_model_optimized(model_name, num_gpus, cpu_offload=True, mixed_precision=True):
    """Load model with maximum memory optimization"""
    logger.info(f"Loading model with aggressive optimization: {model_name}")
    
    # Check CUDA
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    
    torch.cuda.init()
    logger.info(f"CUDA initialized with {torch.cuda.device_count()} GPUs")
    
    # Set memory management
    if mixed_precision:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.matmul.allow_tf32 = True
        logger.info("✅ Mixed precision optimizations enabled")
    
    # Clean memory aggressively
    aggressive_memory_cleanup()
    
    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    logger.info("✅ Tokenizer loaded")
    
    # Load model with maximum optimization
    logger.info("Loading model on CPU with maximum optimization...")
    start_time = time.time()
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        use_cache=False,
        low_cpu_mem_usage=True,
        device_map=None,  # Keep on CPU initially
        # attn_implementation="flash_attention_2" if hasattr(torch.nn, 'MultiheadAttention') else None  # REMOVED - causes import error
    )
    
    load_time = time.time() - start_time
    logger.info(f"✅ Model loaded on CPU in {load_time:.1f}s")
    logger.info(f"Model parameters: {model.num_parameters():,}")
    
    # Enable ALL memory optimizations
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
        logger.info("✅ Gradient checkpointing enabled")
    
    # Move to GPU with memory monitoring
    logger.info("Moving model to GPU with memory monitoring...")
    
    # Set memory fraction to prevent OOM
    for i in range(num_gpus):
        torch.cuda.set_per_process_memory_fraction(0.85, device=i)  # Use only 85% per GPU
    
    # Move to primary GPU
    model = model.cuda(0)
    
    # Check memory after model load
    gpu0_memory = torch.cuda.memory_allocated(0) / 1024**3
    gpu0_total = torch.cuda.get_device_properties(0).total_memory / 1024**3
    logger.info(f"GPU 0 memory after model load: {gpu0_memory:.1f}GB / {gpu0_total:.1f}GB")
    
    # Only use DataParallel if model fits and we have multiple GPUs
    if num_gpus > 1 and gpu0_memory < (gpu0_total * 0.5):  # Only if model uses <50% of GPU
        logger.info(f"Model fits in memory, enabling DataParallel for {num_gpus} GPUs...")
        model = torch.nn.DataParallel(model)
        logger.info("✅ DataParallel enabled")
        
        # Check memory distribution
        for i in range(num_gpus):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            total = torch.cuda.get_device_properties(i).total_memory / 1024**3
            if i == 0 or allocated > 0.1:
                logger.info(f"GPU {i}: {allocated:.1f}GB / {total:.1f}GB")
    else:
        logger.warning(f"Model too large for DataParallel, using single GPU")
        logger.warning(f"Model uses {gpu0_memory:.1f}GB / {gpu0_total:.1f}GB ({gpu0_memory/gpu0_total*100:.1f}%)")
    
    return model, tokenizer

def load_dataset_optimized(data_path, tokenizer, max_seq_length):
    """Load dataset with memory optimization"""
    logger.info(f"Loading dataset from {data_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"Dataset path not found: {data_path}")
    
    files = os.listdir(data_path)
    json_files = [f for f in files if f.endswith('.json') or f.endswith('.jsonl')]
    
    if not json_files:
        raise ValueError("No JSON files found in dataset path")
    
    logger.info(f"Found {len(json_files)} JSON files")
    
    json_path = os.path.join(data_path, "*.json*")
    dataset = load_dataset("json", data_files=json_path)['train']
    logger.info(f"Dataset loaded: {len(dataset)} samples")
    
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_seq_length,
            return_tensors="pt"
        )
    
    logger.info("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        batch_size=50,  # Smaller batch for memory
        num_proc=2,     # Fewer processes
        remove_columns=dataset.column_names,
        desc="Tokenizing"
    )
    
    logger.info(f"✅ Dataset tokenized: {len(tokenized_dataset)} samples")
    return tokenized_dataset

class CPUOffloadOptimizer:
    """Optimizer that offloads states to CPU"""
    def __init__(self, optimizer, model):
        self.optimizer = optimizer
        self.model = model
        self.cpu_states = {}
        
    def step(self):
        # Move optimizer states to GPU temporarily
        for group in self.optimizer.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    param_state = self.optimizer.state[p]
                    for key, value in param_state.items():
                        if isinstance(value, torch.Tensor) and value.device.type == 'cpu':
                            param_state[key] = value.cuda(p.device)
        
        # Step optimizer
        self.optimizer.step()
        
        # Move states back to CPU
        for group in self.optimizer.param_groups:
            for p in group['params']:
                param_state = self.optimizer.state[p]
                for key, value in param_state.items():
                    if isinstance(value, torch.Tensor) and value.device.type == 'cuda':
                        param_state[key] = value.cpu()
    
    def zero_grad(self):
        self.optimizer.zero_grad()

def training_loop(model, tokenizer, dataset, args):
    """Optimized training loop with aggressive memory management"""
    logger.info("=== Starting Optimized Training ===")
    
    # Setup data loading with minimal memory usage
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=data_collator,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )
    
    # Setup optimizer with CPU offloading
    base_optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.learning_rate, 
        weight_decay=args.weight_decay
    )
    
    if args.cpu_offload:
        optimizer = CPUOffloadOptimizer(base_optimizer, model)
        logger.info("✅ Optimizer with CPU offloading enabled")
    else:
        optimizer = base_optimizer
        logger.info("✅ Standard optimizer")
    
    # Enable autocast for mixed precision  
    # scaler = torch.cuda.amp.GradScaler() if args.mixed_precision else None
    # Disabled mixed precision for now due to FP16 + CPU offload conflicts
    
    model.train()
    
    logger.info(f"Training configuration:")
    logger.info(f"  Model: {args.model_name}")
    logger.info(f"  Approach: Optimized DataParallel")
    logger.info(f"  GPUs: {args.num_gpus}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Gradient accumulation: {args.gradient_accumulation_steps}")
    logger.info(f"  Sequence length: {args.max_seq_length}")
    logger.info(f"  Effective batch size: {args.batch_size * args.num_gpus * args.gradient_accumulation_steps}")
    logger.info(f"  CPU offload: {args.cpu_offload}")
    logger.info(f"  Mixed precision: {args.mixed_precision}")
    
    global_step = 0
    start_time = time.time()
    total_loss = 0.0
    
    for epoch in range(args.epochs):
        logger.info(f"=== Epoch {epoch + 1}/{args.epochs} ===")
        
        accumulated_loss = 0.0
        
        for batch_idx, batch in enumerate(dataloader):
            if args.max_steps and global_step >= args.max_steps:
                logger.info(f"Reached max_steps: {args.max_steps}")
                break
            
            try:
                # Move batch to GPU
                batch = {k: v.cuda() for k, v in batch.items()}
                
                # Forward pass (no mixed precision for now)
                outputs = model(**batch)
                loss = outputs.loss / args.gradient_accumulation_steps
                
                # Backward pass  
                loss.backward()
                
                accumulated_loss += loss.item()
                
                # Optimizer step after accumulation
                if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                    # Check for NaN gradients
                    total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    if torch.isnan(total_norm) or torch.isinf(total_norm):
                        logger.warning(f"NaN/Inf gradients detected at step {global_step}, skipping")
                        optimizer.zero_grad()
                        continue
                    
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    final_loss = accumulated_loss * args.gradient_accumulation_steps
                    total_loss += final_loss
                    
                    # Check for NaN loss
                    if torch.isnan(torch.tensor(final_loss)):
                        logger.warning(f"NaN loss detected at step {global_step}")
                    
                    # Aggressive memory cleanup every few steps
                    if global_step % 5 == 0:
                        torch.cuda.empty_cache()
                    
                    # Logging
                    if global_step % 10 == 0:
                        elapsed = time.time() - start_time
                        avg_loss = total_loss / (global_step + 1)
                        
                        logger.info(f"Step {global_step}: Loss = {final_loss:.4f}, Avg = {avg_loss:.4f}")
                        logger.info(f"  Time: {elapsed:.1f}s")
                        
                        # Memory monitoring
                        max_memory = 0
                        for i in range(args.num_gpus):
                            allocated = torch.cuda.memory_allocated(i) / 1024**3
                            if allocated > 0.5:
                                logger.info(f"  GPU {i}: {allocated:.1f}GB")
                                max_memory = max(max_memory, allocated)
                        
                        # W&B logging
                        wandb.log({
                            "training/loss": final_loss,
                            "training/avg_loss": avg_loss,
                            "training/step": global_step,
                            "training/max_gpu_memory": max_memory
                        }, step=global_step)
                    
                    accumulated_loss = 0.0
                    global_step += 1
                    
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.error(f"CUDA OOM at step {global_step}")
                    
                    # Memory analysis
                    for i in range(args.num_gpus):
                        allocated = torch.cuda.memory_allocated(i) / 1024**3
                        reserved = torch.cuda.memory_reserved(i) / 1024**3
                        total = torch.cuda.get_device_properties(i).total_memory / 1024**3
                        logger.error(f"GPU {i}: {allocated:.1f}GB alloc, {reserved:.1f}GB reserved, {total:.1f}GB total")
                    
                    logger.error("Try these optimizations:")
                    logger.error("  --max-seq-length 128")
                    logger.error("  --gradient-accumulation-steps 256")
                    logger.error("  --cpu-offload")
                    
                    # Emergency cleanup
                    aggressive_memory_cleanup()
                    raise
                else:
                    logger.error(f"Runtime error: {e}")
                    raise
        
        if args.max_steps and global_step >= args.max_steps:
            break
    
    total_time = time.time() - start_time
    avg_loss = total_loss / max(global_step, 1)
    
    logger.info(f"✅ Training completed!")
    logger.info(f"  Total time: {total_time:.1f}s ({total_time/3600:.2f}h)")
    logger.info(f"  Total steps: {global_step}")
    logger.info(f"  Average loss: {avg_loss:.4f}")
    
    return global_step, avg_loss

def main():
    args = parse_args()
    set_seed(args.seed)
    
    logger.info("🚀 Starting Qwen Hebrew Fine-tuning - Optimized Approach")
    logger.info(f"Instance: {args.instance_type}")
    logger.info(f"GPUs available: {args.num_gpus}")
    logger.info(f"Model: {args.model_name}")
    logger.info(f"Memory optimizations: CPU offload={args.cpu_offload}, Mixed precision={args.mixed_precision}")
    
    # Setup W&B
    run_name = setup_wandb(args)
    
    # Load model with optimizations
    try:
        model, tokenizer = load_model_optimized(
            args.model_name, 
            args.num_gpus,
            args.cpu_offload,
            args.mixed_precision
        )
        logger.info("✅ Model loaded with optimizations")
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        wandb.log({"training/model_loading_failed": True})
        wandb.finish()
        raise
    
    # Load dataset
    try:
        tokenized_dataset = load_dataset_optimized(args.train, tokenizer, args.max_seq_length)
    except Exception as e:
        logger.error(f"Dataset loading failed: {e}")
        wandb.log({"training/dataset_loading_failed": True})
        wandb.finish()
        raise
    
    # Start training
    training_successful = False
    
    try:
        total_steps, avg_loss = training_loop(model, tokenizer, tokenized_dataset, args)
        training_successful = True
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        training_successful = False
        
        wandb.log({
            "training/training_failed": True,
            "training/error_message": str(e)
        })
        raise
    
    # Save model if successful
    if training_successful:
        logger.info(f"Saving model to {args.model_dir}")
        
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(args.model_dir)
        tokenizer.save_pretrained(args.model_dir)
    
    wandb.finish()
    
    logger.info("🎯 TRAINING SUMMARY:")
    logger.info(f"  Approach: Optimized DataParallel")
    logger.info(f"  Training: {'✅ Success' if training_successful else '❌ Failed'}")

if __name__ == "__main__":
    main()