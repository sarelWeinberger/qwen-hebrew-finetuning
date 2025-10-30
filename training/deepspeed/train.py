import os
import json
import argparse
import subprocess
import torch
import wandb
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed
)
from trl import pack_dataset
import deepspeed
from accelerate import Accelerator
import numpy as np
from typing import Dict, List, Union
from datasets import load_dataset
from datetime import datetime
from verbose_utils import print_model_architecture
from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen3-30B-A3B-Base model")
    parser.add_argument(
        "--config",
        type=str,
        default="cpt_config.json",
        help="Path to the training configuration file"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=False,
        default='wikipedia_he_part_002.jsonl',
        help="Path to the HuggingFace dataset or local JSON/L file (for now)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for initialization"
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
        default="sarel-weinbergerbiu-bar-ilan-university",
        help="Weights & Biases entity name"
    )
    parser.add_argument(
        "--validation_split",
        type=float,
        default=0.0,
        help="Fraction of data to use for validation (default: 0.0 = no validation)"
    )
    parser.add_argument(
        "--validation_seed",
        type=int,
        default=42,
        help="Seed for validation split"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print verbose output with info like model architecture"
    )
    parser.add_argument(
        "--dynamic_packing",
        action="store_true",
        help="Dynamically pack dataset after loading"
    )
    return parser.parse_args()

def check_for_sparse_moe_block(model):
    for module in model.modules():
        if isinstance(module, Qwen3MoeSparseMoeBlock):
            return True
    return False

def create_tokenize_function(tokenizer, max_seq_length, truncation):
    def tokenize_function(examples):
        # Tokenize the texts with truncation but no padding here
        # Let the DataCollator handle padding during batch creation
        return tokenizer(
            examples["text"],
            truncation=truncation,
            max_length=max_seq_length,
            padding='longest'
            # Remove return_tensors="pt" - let DataCollator handle tensor conversion
        )
    return tokenize_function

def detect_dataset_name(args):
    """
    Detect dataset name from datasets directory or use provided dataset_path.
    """
    datasets_dir = "./datasets"
    
    # If a specific dataset is provided in args, use it
    if args.dataset_path and args.dataset_path != 'wikipedia_he_part_002.jsonl':
        return os.path.splitext(os.path.basename(args.dataset_path))[0]
    
    # Otherwise, look in datasets directory
    if os.path.exists(datasets_dir):
        dataset_files = [f for f in os.listdir(datasets_dir) if f.endswith(('.jsonl', '.json'))]
        if dataset_files:
            dataset_files.sort()  # Sort for consistency
            if len(dataset_files) == 1:
                return os.path.splitext(dataset_files[0])[0]
            else:
                # Multiple datasets, use first one with "_et_al" suffix
                return f"{os.path.splitext(dataset_files[0])[0]}_et_al"
    
    # Fallback to the provided dataset path
    return os.path.splitext(os.path.basename(args.dataset_path))[0]

def create_run_identifier(config, args):
    """
    Create a simple, consistent run identifier based on model and dataset (no timestamp).
    """
    # Extract model name (last part after /)
    model_name = config["model_name_or_path"].split("/")[-1]
    
    # Detect dataset name intelligently
    dataset_name = detect_dataset_name(args)
    
    # Simple, consistent naming without timestamp
    return f"{model_name}_{dataset_name}"

def train():
    args = parse_args()
    set_seed(args.seed)
    
    # create a training arguments to load in the deepspeed plugin before we create an Accelerator below (which doesn't default to deepspeed)
    _ = TrainingArguments('.')

    # load in the training config
    with open(args.config, 'r', encoding='utf8') as f:
        config = json.loads(f.read())

    # Initialize Weights & Biases - only on the main process
    accelerator = Accelerator()
    if accelerator.is_local_main_process:
        wandb_run_name = args.wandb_name or f"{create_run_identifier(config, args)}_{create_run_identifier(config, args)}"        
        wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=wandb_run_name,
            config=dict(config=config, args=args.__dict__),
            tags=["qwen", "hebrew", "fine-tuning", "deepspeed"] + (["validation"] if args.validation_split > 0 else []),
            notes=f"Training {config['model_name_or_path']} on {args.dataset_path}" + (f" with {args.validation_split:.1%} validation split" if args.validation_split > 0 else "")
        )
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"])
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load in the model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name_or_path"],
        dtype=torch.bfloat16,
        use_cache=False,  # Disable KV cache for training
        device_map=None,  # Automatically mapped using accelerate 
        low_cpu_mem_usage=True  # Unsure whether this matters
    )    

    # Print detailed model architecture (only verbose)
    if args.verbose and accelerator.is_local_main_process:
        print_model_architecture(model)

    # Workaround for a known bug, relevant for qwen 30B only
    # Known bug with DeepSpeed freezing on the first forward: https://huggingface.co/posts/stas/984424866637646
    # This is the workaround :)
    if check_for_sparse_moe_block(model):
        deepspeed.utils.set_z3_leaf_modules(model, [Qwen3MoeSparseMoeBlock])

    # Load dataset - either from a local file or from the huggingface hub
    print(f"Loading dataset from {args.dataset_path}...")
    if os.path.isfile(args.dataset_path):
        dataset = load_dataset('json', data_files=args.dataset_path, split="train")
    else:
        dataset = load_dataset(args.dataset_path, split="train")
    
    # Print dataset info & Check if the dataset has the expected columns (which is, text)
    print(f"Dataset loaded: {dataset}")
    if args.verbose: print(f"Dataset train split columns: {dataset.column_names}")
    if "text" not in dataset.column_names:
        raise ValueError(f"Dataset does not have a 'text' column. Available columns: {dataset.column_names}")
    
    # Create train/validation split
    if args.validation_split > 0:
        if args.verbose: print(f"Creating train/validation split with {args.validation_split:.1%} validation data...")
        split_dataset = dataset.train_test_split(
            test_size=args.validation_split,
            seed=args.validation_seed,
            shuffle=True
        )
        train_dataset = split_dataset['train']
        val_dataset = split_dataset['test']
        if args.verbose: 
            print(f"Train dataset size: {len(train_dataset)}")
            print(f"Validation dataset size: {len(val_dataset)}")
    else:
        if args.verbose: 
            print("No validation split created (validation_split=0)")
        train_dataset = dataset
        val_dataset = None
    
    print("Tokenizing datasets...")
    
    # Tokenize training dataset
    tokenized_train_dataset = train_dataset.map(
        create_tokenize_function(tokenizer, config["max_seq_length"], truncation=not args.dynamic_packing),
        batched=True,
        num_proc=64,
        remove_columns=train_dataset.column_names,  # Remove all original columns
    )
    
    # Tokenize validation dataset if it exists
    tokenized_val_dataset = None
    if val_dataset is not None:
        tokenized_val_dataset = val_dataset.map(
            create_tokenize_function(tokenizer, config["max_seq_length"], truncation=not args.dynamic_packing),
            batched=True,
            num_proc=64,
            remove_columns=val_dataset.column_names,  # Remove all original columns
        )

    # pack?
    if args.dynamic_packing:
        print("Packing...")
        tokenized_train_dataset = pack_dataset(tokenized_train_dataset, seq_length=config["max_seq_length"], strategy="bfd")
        if tokenized_val_dataset is not None:
            tokenized_val_dataset = pack_dataset(tokenized_val_dataset, seq_length=config["max_seq_length"], strategy="bfd")

    # Print some statistics about the tokenized datasets
    print(f"Final train dataset size: {len(tokenized_train_dataset)}")
    if tokenized_val_dataset is not None:
        print(f"Final validation dataset size: {len(tokenized_val_dataset)}")

    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config.get("per_device_eval_batch_size", config["per_device_train_batch_size"]),
        learning_rate=config.get("learning_rate", 1e-5),
        logging_steps=config.get("logging_steps", 10),
        num_train_epochs=config.get("num_train_epochs", 1),
        max_steps=config.get("max_steps", 0),
        save_steps=config.get("save_steps", 500),
        save_total_limit=config.get("save_total_limit", 5),
        warmup_ratio=config.get('warmup_ratio', None),
        warmup_steps=config.get('warmup_steps', 100) if not 'warmup_ratio' in config else 0,
        # Evaluation settings
        eval_strategy="steps" if tokenized_val_dataset is not None else "no",
        eval_steps=config.get("eval_steps", 100),
        # Defaults:
        report_to=['wandb'],
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': True},
        save_strategy="steps",
        bf16=True,
        lr_scheduler_type="cosine",
        logging_first_step=True,
        remove_unused_columns=False,
        dataloader_num_workers=4,
        push_to_hub=False,
        load_best_model_at_end=tokenized_val_dataset is not None,
        metric_for_best_model="eval_loss" if tokenized_val_dataset is not None else None,
        greater_is_better=False if tokenized_val_dataset is not None else None,
    )
    
    # Handle checkpoint resumption - SIMPLE LOCAL ONLY
    resume_from_checkpoint = False
    if config.get("resume_from_checkpoint") == "auto":
        # Check only the local checkpoint directory
        if os.path.exists(config["output_dir"]):
            for item in os.listdir(config["output_dir"]):
                item_path = os.path.join(config["output_dir"], item)
                if os.path.isdir(item_path) and item.startswith("checkpoint-"):
                    resume_from_checkpoint = True
    elif isinstance(config.get("resume_from_checkpoint"), str):
        resume_from_checkpoint = config.get("resume_from_checkpoint")
        print(f"Resuming from: {resume_from_checkpoint}")

    # Create data collator with proper padding
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False,
        pad_to_multiple_of=8,  # Pad to multiple of 8 for efficiency
        return_tensors="pt"
    )

    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,  # This will be None if no validation split
        data_collator=data_collator
    )
    
    # Log dataset information to W&B
    if accelerator.is_local_main_process and wandb.run:
        dataset_info = {
            "dataset_path": args.dataset_path,
            "train_size": len(tokenized_train_dataset),
            "validation_size": len(tokenized_val_dataset) if tokenized_val_dataset else 0,
            "validation_split": args.validation_split,
            "max_seq_length": config["max_seq_length"],
            "has_validation": tokenized_val_dataset is not None
        }
        wandb.log(dataset_info)

    # Start training
    try:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    except Exception as e:
        print(f"Training error: {e}")
        # Create error checkpoint directory if it doesn't exist
        error_checkpoint_dir = os.path.join(config["output_dir"], "step-error")
        os.makedirs(error_checkpoint_dir, exist_ok=True)
        # Try to save with minimal data only
        try:
            trainer.save_model(error_checkpoint_dir)
        except Exception as save_error:
            print(f"Could not save error checkpoint: {save_error}")
            # Save just the state dict as a fallback
            try:
                torch.save(trainer.model.state_dict(), 
                          os.path.join(error_checkpoint_dir, "pytorch_model.bin"))
            except Exception as fallback_error:
                print(f"Could not save fallback checkpoint: {fallback_error}")
        raise e

    # Save the final model
    print(f"Saving final model to {config['output_dir']}")
    final_model_dir = os.path.join(config["output_dir"], "step-final")
    trainer.save_model(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    print("Training complete!")
    
if __name__ == "__main__":
    train()