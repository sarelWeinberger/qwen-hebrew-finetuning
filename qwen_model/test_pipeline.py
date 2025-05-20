import os
import argparse
import torch
import json
import random
from datasets import load_dataset, load_from_disk, Dataset, DatasetDict
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed
)
import wandb

def parse_args():
    parser = argparse.ArgumentParser(description="Test the training pipeline with a small subset of data")
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the dataset directory or Hugging Face dataset name"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="qwen_model/finetuning/training_config.json",
        help="Path to the training configuration file"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to the model (defaults to the one in config)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="qwen_model/test_run",
        help="Output directory for the test run"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=10,
        help="Maximum number of samples to use for testing"
    )
    parser.add_argument(
        "--max_steps",
        type=int,
        default=5,
        help="Maximum number of training steps to run"
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=128,
        help="Maximum sequence length for tokenization (use a small value for testing)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="qwen-hebrew-test",
        help="Weights & Biases project name"
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="Weights & Biases entity name"
    )
    parser.add_argument(
        "--skip_wandb",
        action="store_true",
        help="Skip Weights & Biases logging"
    )
    # Add arguments for distributed training
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training (-1 means not distributed)"
    )
    parser.add_argument(
        "--deepspeed",
        type=str,
        default=None,
        help="Path to deepspeed config file"
    )
    parser.add_argument(
        "--single_device",
        action="store_true",
        help="Force using a single device even with multiple GPUs available"
    )
    return parser.parse_args()

def load_and_sample_dataset(dataset_path, max_samples, seed):
    """Load dataset and sample a small subset for testing."""
    print(f"Loading dataset from {dataset_path}...")
    
    # Set a streaming flag to avoid loading the entire dataset
    streaming = True if max_samples <= 100 else False
    print(f"Using streaming mode: {streaming} (max_samples={max_samples})")
    
    try:
        # First try loading as a local disk dataset
        try:
            if streaming:
                # For disk datasets, we'll still load it normally but sample immediately
                dataset = load_from_disk(dataset_path)
                print("Successfully loaded dataset from disk")
            else:
                dataset = load_from_disk(dataset_path)
                print("Successfully loaded dataset from disk")
        except Exception as e:
            print(f"Failed to load dataset from disk: {e}")
            # Try loading as a Hugging Face dataset with streaming
            if streaming:
                dataset = load_dataset(dataset_path, streaming=True)
                # Convert streaming dataset to regular dataset with limited samples
                train_samples = []
                for i, sample in enumerate(dataset["train"]):
                    if i >= max_samples:
                        break
                    train_samples.append(sample)
                
                val_samples = []
                if "validation" in dataset:
                    for i, sample in enumerate(dataset["validation"]):
                        if i >= max_samples // 2:
                            break
                        val_samples.append(sample)
                
                # Create new dataset from collected samples
                train_dataset = Dataset.from_dict({k: [sample[k] for sample in train_samples] for k in train_samples[0].keys()})
                if val_samples:
                    val_dataset = Dataset.from_dict({k: [sample[k] for sample in val_samples] for k in val_samples[0].keys()})
                    dataset = DatasetDict({"train": train_dataset, "validation": val_dataset})
                else:
                    dataset = DatasetDict({"train": train_dataset})
            else:
                dataset = load_dataset(dataset_path)
    except Exception as e:
        print(f"Failed to load dataset as HF dataset: {e}")
        # Try loading as a local JSON dataset with streaming
        try:
            if streaming:
                dataset = load_dataset("json", data_files=dataset_path, streaming=True)
                # Convert streaming dataset to regular dataset with limited samples
                train_samples = []
                for i, sample in enumerate(dataset["train"]):
                    if i >= max_samples:
                        break
                    train_samples.append(sample)
                
                # Create new dataset from collected samples
                train_dataset = Dataset.from_dict({k: [sample[k] for sample in train_samples] for k in train_samples[0].keys()})
                dataset = DatasetDict({"train": train_dataset})
            else:
                dataset = load_dataset("json", data_files=dataset_path)
        except Exception as e:
            print(f"Failed to load dataset as JSON: {e}")
            raise ValueError(f"Could not load dataset from {dataset_path}")
    
    # Print dataset info
    print(f"Original dataset: {dataset}")
    
    # Check if dataset has the expected structure
    if "train" not in dataset:
        raise ValueError(f"Dataset does not have a 'train' split. Available splits: {dataset.keys()}")
    
    # Sample a small subset for testing
    random.seed(seed)
    
    # Sample from train split
    train_indices = random.sample(range(len(dataset["train"])), min(max_samples, len(dataset["train"])))
    train_subset = dataset["train"].select(train_indices)
    
    # Sample from validation split if it exists
    if "validation" in dataset:
        val_indices = random.sample(range(len(dataset["validation"])), min(max_samples // 2, len(dataset["validation"])))
        val_subset = dataset["validation"].select(val_indices)
        test_dataset = DatasetDict({"train": train_subset, "validation": val_subset})
    else:
        # Create a small validation split from the train subset
        splits = train_subset.train_test_split(test_size=0.2, seed=seed)
        test_dataset = DatasetDict({"train": splits["train"], "validation": splits["test"]})
    
    print(f"Sampled test dataset: {test_dataset}")
    return test_dataset

def setup_distributed(args):
    """Initialize distributed training environment"""
    if args.local_rank == -1:
        # Not using distributed mode
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args.world_size = 1
        args.n_gpu = torch.cuda.device_count() if not args.single_device else 1
        args.is_master = True
        print(f"Running in non-distributed mode with {args.n_gpu} GPUs available")
    else:
        # Distributed mode
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        
        # Initialize process group if not using DeepSpeed (DeepSpeed handles this internally)
        if not args.deepspeed:
            if not torch.distributed.is_initialized():
                torch.distributed.init_process_group(backend="nccl")
        
        args.world_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
        args.n_gpu = 1  # In distributed mode, each process has 1 GPU
        args.is_master = args.local_rank in [-1, 0]  # Only rank 0 is master
        
        print(f"Running in distributed mode with world_size={args.world_size}, local_rank={args.local_rank}")
    
    return device

def main():
    args = parse_args()
    
    # Setup distributed training
    device = setup_distributed(args)
    
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Create output directory (only on master process)
    if args.is_master:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Load configuration
    with open(args.config, 'r') as f:
        config = json.load(f)
    
    # Override model path if provided
    if args.model_path:
        config["model_name_or_path"] = args.model_path
    
    # Initialize W&B if not skipped (only on master process)
    if not args.skip_wandb and args.is_master:
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=f"test-run-{args.seed}",
            config={
                "seed": args.seed,
                "dataset_path": args.dataset_path,
                "max_samples": args.max_samples,
                "max_steps": args.max_steps,
                "max_seq_length": args.max_seq_length,
                "model_name": config["model_name_or_path"],
                "distributed": args.local_rank != -1,
                "world_size": args.world_size,
                "n_gpu": args.n_gpu,
                "single_device": args.single_device
            }
        )
    
    # Load and sample dataset
    dataset = load_and_sample_dataset(args.dataset_path, args.max_samples, args.seed)
    
    # Load tokenizer
    print(f"Loading tokenizer from {config['model_name_or_path']}...")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"])
    tokenizer.pad_token = tokenizer.eos_token
    
    # Tokenize the dataset with a small sequence length
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=args.max_seq_length,
            return_tensors="pt"
        )
    
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"] if "text" in dataset["train"].column_names else None
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Load model with optimizations for multi-GPU setup
    print(f"Loading model from {config['model_name_or_path']}...")
    
    # Set environment variable to avoid memory fragmentation
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    
    # Fix for device mismatch error: use a single device for testing
    if args.single_device or args.local_rank != -1:
        # Use a specific device
        if args.local_rank != -1:
            # In distributed mode, use the assigned GPU
            device = torch.device(f"cuda:{args.local_rank}")
        else:
            # In single device mode, use the first GPU
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        
        print(f"Loading model on single device: {device}")
        model = AutoModelForCausalLM.from_pretrained(
            config["model_name_or_path"],
            torch_dtype=torch.float16,
            trust_remote_code=True,
            use_cache=False,
            low_cpu_mem_usage=True
        ).to(device)
    else:
        # Try auto device mapping for multi-GPU setup
        try:
            print("Attempting to load model with device_map='auto'...")
            model = AutoModelForCausalLM.from_pretrained(
                config["model_name_or_path"],
                torch_dtype=torch.float16,
                trust_remote_code=True,
                use_cache=False,
                device_map="auto"
            )
            print("Successfully loaded model with device_map='auto'")
            
            # Check if model is distributed across multiple devices
            if hasattr(model, 'hf_device_map') and len(set(model.hf_device_map.values())) > 1:
                print("WARNING: Model is distributed across multiple devices. This may cause device mismatch errors.")
                print("Consider using --single_device flag to force loading on a single device.")
        except Exception as e:
            print(f"Error loading model with device_map='auto': {e}")
            print("Falling back to single device loading...")
            
            # Fall back to single device
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            model = AutoModelForCausalLM.from_pretrained(
                config["model_name_or_path"],
                torch_dtype=torch.float16,
                trust_remote_code=True,
                use_cache=False,
                low_cpu_mem_usage=True
            ).to(device)
    
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    
    # Training arguments optimized for multi-GPU setup
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        fp16=True,
        per_device_train_batch_size=1,  # Keep small for testing
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=4,  # Reduced from 8 to avoid memory issues
        learning_rate=config["learning_rate"],
        weight_decay=config["weight_decay"],
        max_steps=args.max_steps,  # Override epochs with max_steps for quick testing
        logging_steps=1,
        save_steps=args.max_steps,
        save_total_limit=1,
        report_to=["wandb"] if not args.skip_wandb else [],
        remove_unused_columns=False,
        dataloader_num_workers=4,  # Increased for better data loading
        gradient_checkpointing=True,
        ddp_find_unused_parameters=False,  # Optimize DDP
        deepspeed=args.deepspeed,  # Will be configured if DeepSpeed is available
        local_rank=args.local_rank
    )
    
    # Try to import and configure DeepSpeed if available and not already configured
    if not args.deepspeed:
        try:
            import deepspeed
            print("DeepSpeed is available, configuring DeepSpeed...")
            
            # Create a basic DeepSpeed config
            ds_config = {
                "fp16": {
                    "enabled": True
                },
                "zero_optimization": {
                    "stage": 2,
                    "offload_optimizer": {
                        "device": "cpu"
                    },
                    "contiguous_gradients": True,
                    "overlap_comm": True
                },
                "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
                "gradient_clipping": training_args.max_grad_norm,
                "train_batch_size": training_args.per_device_train_batch_size * max(1, torch.cuda.device_count()) * training_args.gradient_accumulation_steps
            }
            
            # Save DeepSpeed config to a file
            ds_config_path = os.path.join(args.output_dir, "ds_config.json")
            with open(ds_config_path, 'w') as f:
                json.dump(ds_config, f)
            
            # Set DeepSpeed config path in training arguments
            training_args.deepspeed = ds_config_path
            print(f"DeepSpeed config saved to {ds_config_path}")
        except ImportError:
            print("DeepSpeed not available, continuing without it")
    
    # Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=tokenized_dataset["validation"],
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    
    # Start training
    print("Starting test training run...")
    try:
        train_result = trainer.train()
        
        # Print training stats
        print("\n=== Training Statistics ===")
        print(f"Steps completed: {trainer.state.global_step}")
        print(f"Training loss: {train_result.training_loss}")
        
        # Try evaluation
        try:
            print("\nRunning evaluation...")
            eval_result = trainer.evaluate()
            print(f"Evaluation loss: {eval_result['eval_loss']}")
        except Exception as e:
            print(f"Evaluation failed: {e}")
        
        # Save a tiny model checkpoint
        print(f"\nSaving test model to {args.output_dir}")
        trainer.save_model()
    except Exception as e:
        print(f"Training failed: {e}")
        print("This may be due to device mismatch errors. Try running with --single_device flag.")
    
    # Calculate token statistics
    num_train_samples = len(tokenized_dataset["train"])
    num_val_samples = len(tokenized_dataset["validation"])
    total_samples = num_train_samples + num_val_samples
    total_tokens = total_samples * args.max_seq_length
    
    print("\n=== Token Statistics ===")
    print(f"Number of training samples: {num_train_samples}")
    print(f"Number of validation samples: {num_val_samples}")
    print(f"Sequence length: {args.max_seq_length}")
    print(f"Total tokens processed: {total_tokens}")
    
    # Finish W&B run (only on master process)
    if not args.skip_wandb and args.is_master:
        wandb.finish()
    
    # Only print final messages on master process
    if args.is_master:
        print("\nTest pipeline completed!")
        print("You can now proceed with the full training run.")

if __name__ == "__main__":
    main()