import os
import json
import argparse
import torch
import wandb
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed,
    TrainerCallback
)
import numpy as np
from typing import Dict, List, Union
from datasets import load_dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen3-30B-A3B-Base model")
    parser.add_argument(
        "--config",
        type=str,
        default="qwen_model/finetuning/training_config.json",
        help="Path to the training configuration file"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the dataset directory or Hugging Face dataset name"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for initialization"
    )
    parser.add_argument(
        "--deepspeed",
        type=str,
        default=None,
        help="Path to deepspeed config file for distributed training"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="qwen-hebrew-finetuning",
        help="Weights & Biases project name"
    )
    parser.add_argument(
        "--wandb_name",
        type=str,
        default=None,
        help="Weights & Biases run name"
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="Weights & Biases entity name"
    )
    return parser.parse_args()

class WandbMetricsCallback(TrainerCallback):
    """Custom callback for logging metrics to Weights & Biases."""
    
    def __init__(self, max_seq_length=2048):
        super().__init__()
        self.total_tokens = 0
        self.max_seq_length = max_seq_length
        self.steps = 0
    
    def on_step_end(self, args, state, control, **kwargs):
        """Track steps for token counting."""
        self.steps += 1
        
        # Calculate tokens processed in this step
        # Total batch size = per_device_batch_size * num_devices * gradient_accumulation_steps
        total_batch_size = args.per_device_train_batch_size * args.world_size
        if args.gradient_accumulation_steps > 0:
            total_batch_size *= args.gradient_accumulation_steps
            
        step_tokens = total_batch_size * self.max_seq_length
        self.total_tokens += step_tokens
        
        # Log metrics for every step
        wandb.log({
            "training/tokens": self.total_tokens,
            "training/step_num": self.steps,
            "training/batch_num": state.global_step,
            "training/epoch_num": state.epoch if hasattr(state, "epoch") else 0
        }, step=state.global_step)
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log metrics to W&B on each logging step."""
        if logs is None:
            return
        
        # Log loss
        if "loss" in logs:
            wandb.log({"training/loss": logs["loss"]}, step=state.global_step)
        
        # Calculate and log number of tokens processed
        if "train_runtime" in logs:
            # At the end of training, log final token count and statistics
            wandb.log({
                "training/total_tokens": self.total_tokens,
                "training/tokens_per_second": self.total_tokens / logs["train_runtime"],
                "training/final_step": self.steps
            }, step=state.global_step)
            
            # Print token statistics for easy reference
            print(f"\n=== Training Token Statistics ===")
            print(f"Total tokens processed: {self.total_tokens:,}")
            print(f"Tokens per second: {self.total_tokens / logs['train_runtime']:.2f}")
            print(f"Total steps: {self.steps}")
            print(f"Sequence length: {self.max_seq_length}")
            print(f"Batch size per step: {args.per_device_train_batch_size * args.world_size}")
            print(f"===================================\n")
        
        # Log perplexity (derived from loss)
        if "loss" in logs:
            perplexity = np.exp(logs["loss"])
            wandb.log({"training/perplexity": perplexity}, step=state.global_step)
            
            # For language models, we can use perplexity as a proxy for accuracy
            # Lower perplexity = higher accuracy
            accuracy_proxy = max(0, 100 * (1 - (perplexity / 100)))
            accuracy_proxy = min(accuracy_proxy, 100)  # Cap at 100%
            wandb.log({"training/accuracy_proxy": accuracy_proxy}, step=state.global_step)

def create_deepspeed_config():
    """Create a basic DeepSpeed configuration for ZeRO-3"""
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
            "stage": 3,
            "offload_optimizer": {
                "device": "cpu",
                "pin_memory": True
            },
            "offload_param": {
                "device": "cpu",
                "pin_memory": True
            },
            "overlap_comm": True,
            "contiguous_gradients": True,
            "sub_group_size": 1e9,
            "reduce_bucket_size": "auto",
            "stage3_prefetch_bucket_size": "auto",
            "stage3_param_persistence_threshold": "auto",
            "stage3_max_live_parameters": 1e9,
            "stage3_max_reuse_distance": 1e9,
            "gather_16bit_weights_on_model_save": True
        },
        "gradient_accumulation_steps": 8,
        "gradient_clipping": 1.0,
        "steps_per_print": 100,
        "train_micro_batch_size_per_gpu": 1,
        "wall_clock_breakdown": False
    }
    
    # Save the config
    os.makedirs("qwen_model/finetuning", exist_ok=True)
    config_path = "qwen_model/finetuning/deepspeed_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    return config_path

def train():
    args = parse_args()
    set_seed(args.seed)
    
    # Initialize Weights & Biases
    run_name = args.wandb_name or f"qwen-hebrew-{args.seed}"
    wandb.init(
        project=args.wandb_project,
        name=run_name,
        entity=args.wandb_entity,
        config={
            "seed": args.seed,
            "dataset_path": args.dataset_path,
            "model_name": "Qwen3-30B-A3B-Base"
        }
    )
    print(f"Initialized Weights & Biases run: {run_name}")
    
    # Load training configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Add config to wandb
    wandb.config.update(config)
    
    # Create DeepSpeed config if not provided
    if args.deepspeed is None:
        args.deepspeed = create_deepspeed_config()
        print(f"Created DeepSpeed config at: {args.deepspeed}")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"])
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with gradient checkpointing for memory efficiency
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name_or_path"],
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        use_cache=False,  # Disable KV cache for training
        device_map=None,  # Ensure all parameters are on CPU initially
        low_cpu_mem_usage=True  # Optimize memory usage during loading
    )
    
    # Don't move model to GPU - let DeepSpeed handle device placement
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    
    # Load dataset
    print(f"Loading dataset from {args.dataset_path}...")
    try:
        # First try loading as a local disk dataset
        try:
            from datasets import load_from_disk
            dataset = load_from_disk(args.dataset_path)
            print("Successfully loaded dataset from disk")
        except Exception as e:
            print(f"Failed to load dataset from disk: {e}")
            # Try loading as a Hugging Face dataset
            dataset = load_dataset(args.dataset_path)
    except Exception as e:
        print(f"Failed to load dataset as HF dataset: {e}")
        # Try loading as a local JSON dataset
        try:
            dataset = load_dataset("json", data_files=args.dataset_path)
        except Exception as e:
            print(f"Failed to load dataset as JSON: {e}")
            raise ValueError(f"Could not load dataset from {args.dataset_path}. Please check the path and format.")
    
    # Print dataset info
    print(f"Dataset loaded: {dataset}")
    
    # Tokenize the dataset
    def tokenize_function(examples):
        # Tokenize the texts with padding to max length
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=2048,
            return_tensors="pt"
        )
    
    print("Tokenizing dataset...")
    # Check if dataset has the expected structure
    if "train" not in dataset:
        raise ValueError(f"Dataset does not have a 'train' split. Available splits: {dataset.keys()}")
    
    # Check if the dataset has the expected columns
    print(f"Dataset train split columns: {dataset['train'].column_names}")
    if "text" not in dataset["train"].column_names:
        raise ValueError(f"Dataset does not have a 'text' column. Available columns: {dataset['train'].column_names}")
    
    # Adjust num_proc based on dataset size to avoid warnings
    num_proc = min(4, len(dataset["train"]))
    
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        num_proc=num_proc,
        remove_columns=["text"] if "text" in dataset["train"].column_names else None
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # We're doing causal language modeling, not masked
    )
    
    # Training arguments optimized for 8x H100 GPUs
    print("Setting up training arguments...")
    
    # Use only parameters that are supported by all transformers versions
    training_args_dict = {
        "output_dir": config["output_dir"],
        "fp16": True,  # Use mixed precision
        "gradient_accumulation_steps": 1,  # Can use smaller value with 8x H100s
        "per_device_train_batch_size": 2,  # Can use larger batch size with H100s
        "learning_rate": config["learning_rate"],
        "weight_decay": config["weight_decay"],
        "num_train_epochs": config["num_train_epochs"],
        "save_steps": config["save_steps"],
        "save_total_limit": config["save_total_limit"],
        "logging_steps": config["logging_steps"],
        "max_grad_norm": 1.0 if "max_grad_norm" not in config else config["max_grad_norm"],
        "warmup_ratio": config["warmup_ratio"],
        "deepspeed": args.deepspeed,  # Use the DeepSpeed config from command line
        "report_to": ["tensorboard", "wandb"],
        "remove_unused_columns": False,
        "dataloader_num_workers": 4,  # Can use more workers with H100s
        "gradient_checkpointing": True  # Still use gradient checkpointing for efficiency
    }
    
    # Add checkpoint-related parameters if they exist in the config
    if "resume_from_checkpoint" in config:
        print(f"Setting up checkpoint directory: {config['resume_from_checkpoint']}")
        os.makedirs(config["resume_from_checkpoint"], exist_ok=True)
        training_args_dict["resume_from_checkpoint"] = config["resume_from_checkpoint"]
    
    if "save_strategy" in config:
        training_args_dict["save_strategy"] = config["save_strategy"]
    
    if "load_best_model_at_end" in config:
        training_args_dict["load_best_model_at_end"] = config["load_best_model_at_end"]
        
        # If we're loading the best model at the end, we need an evaluation strategy
        if config["load_best_model_at_end"]:
            training_args_dict["evaluation_strategy"] = "steps"
            training_args_dict["eval_steps"] = config.get("eval_steps", 500)
    
    if "metric_for_best_model" in config:
        training_args_dict["metric_for_best_model"] = config["metric_for_best_model"]
        
    if "greater_is_better" in config:
        training_args_dict["greater_is_better"] = config["greater_is_better"]
    
    # Create the training arguments
    training_args = TrainingArguments(**training_args_dict)
    
    print(f"Training arguments set up successfully: {training_args}")
    
    # Initialize custom W&B metrics callback with the correct sequence length
    max_seq_length = 2048  # This should match the max_length in tokenize_function
    wandb_metrics_callback = WandbMetricsCallback(max_seq_length=max_seq_length)
    
    # Prepare validation dataset if it exists
    eval_dataset = tokenized_dataset["validation"] if "validation" in tokenized_dataset else None
    
    # Initialize Trainer with simple configuration
    print("Initializing Trainer...")
    
    # Ensure model parameters are properly initialized for DeepSpeed
    # Move model to first GPU to ensure all parameters are on the same device
    if torch.cuda.is_available():
        model = model.to("cuda:0")
        print("Model moved to CUDA device")
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
        callbacks=[wandb_metrics_callback]
        # Remove model_init lambda to avoid device placement conflicts
    )
    
    print("Trainer initialized successfully")
    
    # Create checkpoint directory if specified
    if "resume_from_checkpoint" in config:
        checkpoint_dir = config["resume_from_checkpoint"]
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"Created checkpoint directory: {checkpoint_dir}")
    
    # Start training
    print("Starting training...")
    try:
        trainer.train()
    except Exception as e:
        print(f"Training error: {e}")
        # Save checkpoint even if training fails
        if "resume_from_checkpoint" in config:
            print(f"Attempting to save checkpoint after error...")
            try:
                trainer.save_model(config["resume_from_checkpoint"] + "/error_checkpoint")
                print(f"Saved checkpoint after error to {config['resume_from_checkpoint']}/error_checkpoint")
            except Exception as save_error:
                print(f"Failed to save checkpoint after error: {save_error}")
        raise e
    
    # Save the final model
    print(f"Saving model to {config['output_dir']}")
    trainer.save_model()
    tokenizer.save_pretrained(config["output_dir"])
    
    print("Training complete!")
    
    # Finish wandb run
    wandb.finish()

if __name__ == "__main__":
    train()