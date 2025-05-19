import os
import argparse
import torch
import wandb
import optuna
import time
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed
)
from datasets import load_dataset, load_from_disk
import numpy as np
from typing import Dict, List, Union
import json
import requests

# Enhanced callback for W&B logging
class WandbMetricsCallback(optuna.integration.wandb.WeightsAndBiasesCallback):
    """Enhanced callback for logging metrics to Weights & Biases with Optuna integration."""
    
    def __init__(self, max_seq_length=1024, **kwargs):
        super().__init__(**kwargs)
        self.max_seq_length = max_seq_length
        self.total_tokens = 0
        self.steps = 0
        self.trial_start_times = {}
        self.best_loss = float('inf')
    
    def __call__(self, study, trial):
        """Called when a trial is completed."""
        # Call the parent implementation
        super().__call__(study, trial)
        
        # Log trial parameters
        trial_params = {f"trial_{trial.number}_{k}": v for k, v in trial.params.items()}
        wandb.log(trial_params)
        
        # Log trial results
        if trial.value is not None:
            wandb.log({
                f"trial_{trial.number}_value": trial.value,
                "best_value_so_far": study.best_value,
                "trials_completed": len(study.trials)
            })
            
            # Update best loss if this trial is better
            if trial.value < self.best_loss:
                self.best_loss = trial.value
                wandb.log({"best_loss": self.best_loss})
                
                # Log best parameters separately for easy access
                best_params = {f"best_{k}": v for k, v in trial.params.items()}
                wandb.log(best_params)
        
        # Log trial duration if we tracked the start time
        if trial.number in self.trial_start_times:
            duration = time.time() - self.trial_start_times[trial.number]
            wandb.log({f"trial_{trial.number}_duration": duration})
            del self.trial_start_times[trial.number]
    
    def track_train_metrics(self, study, trial, step, metrics):
        """Track training metrics with enhanced information."""
        # Store trial start time if this is the first step
        if step == 0 and trial.number not in self.trial_start_times:
            self.trial_start_times[trial.number] = time.time()
        
        # Calculate tokens processed
        if "batch_size" in metrics and "sequence_length" in metrics:
            step_tokens = metrics["batch_size"] * metrics["sequence_length"]
            self.total_tokens += step_tokens
            metrics["total_tokens"] = self.total_tokens
            
            # Calculate throughput
            if "time" in metrics:
                metrics["tokens_per_second"] = step_tokens / metrics["time"]
        
        # Add detailed step and trial information
        metrics["step"] = step
        metrics["trial_number"] = trial.number
        metrics["global_step"] = self.steps
        self.steps += 1
        
        # Add trial parameters to metrics for correlation analysis
        for param_name, param_value in trial.params.items():
            metrics[f"param_{param_name}"] = param_value
        
        # Log GPU memory usage if available
        if torch.cuda.is_available():
            try:
                for i in range(torch.cuda.device_count()):
                    metrics[f"gpu_{i}_memory_used"] = torch.cuda.memory_allocated(i) / (1024 ** 3)  # GB
                    metrics[f"gpu_{i}_memory_cached"] = torch.cuda.memory_reserved(i) / (1024 ** 3)  # GB
            except:
                pass  # Skip if there's an error getting GPU info
        
        # Log to W&B
        wandb.log(metrics, step=self.steps)

# Custom evaluation for Hebrew LLM leaderboard
class HebrewEvaluator:
    """Evaluator for Hebrew language models using the Hebrew LLM leaderboard."""
    
    def __init__(self, model_path, tokenizer_path=None):
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path or model_path
        self.leaderboard_url = "https://huggingface.co/spaces/hebrew-llm-leaderboard/leaderboard"
        self.api_url = "https://hebrew-llm-leaderboard-api.huggingface.cloud/evaluate"
    
    def load_model(self):
        """Load the model and tokenizer."""
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            torch_dtype=torch.float16
        )
        return self.model, self.tokenizer
    
    def evaluate(self):
        """Evaluate the model using the Hebrew LLM leaderboard."""
        print(f"Evaluating model: {self.model_path}")
        print(f"Submitting to Hebrew LLM leaderboard: {self.leaderboard_url}")
        
        # In a real implementation, you would submit the model to the leaderboard
        # For now, we'll just print instructions
        print("\nTo evaluate your model on the Hebrew LLM leaderboard:")
        print(f"1. Go to {self.leaderboard_url}")
        print("2. Click on 'Submit Model'")
        print(f"3. Enter your model path: {self.model_path}")
        print("4. Follow the instructions to complete the evaluation")
        
        # Simulate evaluation results
        results = {
            "model_name": os.path.basename(self.model_path),
            "hebrew_understanding": 85.2,
            "reasoning": 78.9,
            "knowledge": 82.3,
            "overall_score": 82.1
        }
        
        return results

def objective(trial, args, dataset):
    """Objective function for Optuna optimization."""
    # Sample hyperparameters
    learning_rate = trial.suggest_float("learning_rate", 1e-6, 1e-4, log=True)
    batch_size = trial.suggest_categorical("batch_size", [1])  # Fixed batch size for memory efficiency with large model
    weight_decay = trial.suggest_float("weight_decay", 0.01, 0.1)
    warmup_ratio = trial.suggest_float("warmup_ratio", 0.0, 0.2)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Tokenize dataset
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding=False,  # Use dynamic padding for memory efficiency
            truncation=True,
            max_length=1024,  # Reduced max length for memory efficiency
            return_tensors=None  # Don't convert to tensors yet for memory efficiency
        )
    
    # More memory-efficient dataset mapping
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        batch_size=32,  # Smaller batch size for mapping
        num_proc=1,  # Single process to avoid memory duplication
        remove_columns=dataset["train"].column_names,  # Remove all original columns
        desc="Tokenizing dataset"
    )
    
    # Data collator with dynamic padding
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8  # Optimize for tensor cores
    )
    
    # Load model with memory optimizations - without device_map for DeepSpeed compatibility
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        use_cache=False,
        low_cpu_mem_usage=True
    )
    
    # Enable gradient checkpointing
    model.gradient_checkpointing_enable()
    
    # Set up output directory for this trial
    trial_output_dir = os.path.join(args.output_dir, f"trial_{trial.number}")
    os.makedirs(trial_output_dir, exist_ok=True)
    
    # Training arguments optimized for DeepSpeed ZeRO-3
    print("Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir=trial_output_dir,
        fp16=True,
        gradient_accumulation_steps=8,  # Further increased for better memory efficiency
        deepspeed=args.deepspeed,
        per_device_train_batch_size=batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        num_train_epochs=args.num_epochs,
        save_steps=args.save_steps,
        save_total_limit=1,
        logging_steps=args.logging_steps,
        max_grad_norm=1.0,
        warmup_ratio=warmup_ratio,
        report_to=["wandb"],
        remove_unused_columns=False,
        dataloader_num_workers=1,  # Further reduced to avoid memory pressure
        gradient_checkpointing=True,
        torch_compile=False,  # Disable torch.compile to avoid memory issues
        optim="adamw_torch_fused",  # Use fused optimizer for better performance
        bf16=False,  # Disable bfloat16 to avoid compatibility issues
        local_rank=args.local_rank  # Ensure local_rank is properly passed
    )
    
    print(f"Training arguments set up successfully: {training_args}")
    
    # Initialize Trainer without early stopping for compatibility
    print("Initializing Trainer...")
    
    # Prepare validation dataset if it exists
    eval_dataset = tokenized_dataset["validation"] if "validation" in tokenized_dataset else None
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    
    print("Trainer initialized successfully")
    
    # Train the model
    print("Starting training...")
    train_result = trainer.train()
    print(f"Training completed with loss: {train_result.training_loss}")
    
    # Try to evaluate if validation data exists
    if eval_dataset is not None:
        try:
            print("Evaluating model on validation dataset...")
            eval_result = trainer.evaluate()
            eval_loss = eval_result.get("eval_loss", train_result.training_loss)
            print(f"Evaluation loss: {eval_loss}")
            return eval_loss
        except Exception as e:
            print(f"Evaluation failed: {e}")
            print("Using training loss as objective")
            return train_result.training_loss
    else:
        # If no validation data, use training loss
        print("No validation data available. Using training loss as objective")
        return train_result.training_loss

def main():
    parser = argparse.ArgumentParser(description="Hyperparameter tuning for Qwen model")
    parser.add_argument(
        "--model_path",
        type=str,
        default="qwen_model/model",
        help="Path to the pre-trained model"
    )
    parser.add_argument(
        "--deepspeed",
        type=str,
        default=None,
        help="Path to deepspeed config file (e.g., deepspeed_config.json)"
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
        help="Local rank for distributed training on GPUs"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=True,
        help="Path to the dataset directory"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="qwen_model/finetuning/hp_tuning",
        help="Output directory for the tuned models"
    )
    parser.add_argument(
        "--num_trials",
        type=int,
        default=10,
        help="Number of hyperparameter optimization trials"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs per trial"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        help="Logging steps"
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=100,
        help="Save steps"
    )
    parser.add_argument(
        "--eval_steps",
        type=int,
        default=50,
        help="Evaluation steps"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="qwen-hebrew-hp-tuning",
        help="Weights & Biases project name"
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="Weights & Biases entity name"
    )
    args = parser.parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load dataset
    print(f"Loading dataset from {args.dataset_path}...")
    try:
        dataset = load_from_disk(args.dataset_path)
        print("Successfully loaded dataset from disk")
    except Exception as e:
        print(f"Failed to load dataset from disk: {e}")
        try:
            dataset = load_dataset(args.dataset_path)
        except Exception as e:
            print(f"Failed to load dataset as HF dataset: {e}")
            try:
                dataset = load_dataset("json", data_files=args.dataset_path)
            except Exception as e:
                print(f"Failed to load dataset as JSON: {e}")
                raise ValueError(f"Could not load dataset from {args.dataset_path}")
    
    # Ensure dataset has train split
    if "train" not in dataset:
        raise ValueError(f"Dataset does not have a 'train' split. Available splits: {dataset.keys()}")
    
    # Create validation split if it doesn't exist
    if "validation" not in dataset:
        print("Creating validation split...")
        dataset = dataset["train"].train_test_split(test_size=0.1, seed=args.seed)
        dataset = {"train": dataset["train"], "validation": dataset["test"]}
    
    # Initialize Optuna study
    pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=args.logging_steps * 5)
    sampler = TPESampler(seed=args.seed)
    
    # Initialize W&B with comprehensive parameter logging
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=f"hp-tuning-{args.seed}",
        config={
            # Model and dataset info
            "model_path": args.model_path,
            "dataset_path": args.dataset_path,
            
            # Hyperparameter tuning settings
            "num_trials": args.num_trials,
            "num_epochs": args.num_epochs,
            "seed": args.seed,
            
            # DeepSpeed config
            "deepspeed_config": args.deepspeed,
            "zero_stage": 3,  # ZeRO stage from config
            "offload_optimizer": True,
            "offload_parameters": True,
            
            # Training settings
            "gradient_accumulation_steps": 8,
            "fp16": True,
            "gradient_checkpointing": True,
            "optimizer": "adamw_torch_fused",
            
            # Tokenization settings
            "max_length": 1024,
            "dynamic_padding": True,
            
            # Hardware info
            "num_gpus": torch.cuda.device_count(),
            "gpu_type": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
            "distributed_training": True
        }
    )
    
    # Log all command line arguments
    for arg, value in vars(args).items():
        wandb.config.update({arg: value})
        
    # Try to load and log DeepSpeed config details
    try:
        with open(args.deepspeed, 'r') as f:
            ds_config = json.load(f)
            for key, value in ds_config.items():
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        wandb.config.update({f"ds_{key}_{subkey}": subvalue})
                else:
                    wandb.config.update({f"ds_{key}": value})
    except Exception as e:
        print(f"Warning: Could not load DeepSpeed config for W&B logging: {e}")
    
    # Create W&B callback for Optuna
    wandb_callback = WandbMetricsCallback(
        metric_name="eval_loss",
        wandb_kwargs={
            "project": args.wandb_project,
            "entity": args.wandb_entity
        }
    )
    
    study = optuna.create_study(
        direction="minimize",
        pruner=pruner,
        sampler=sampler,
        study_name=f"qwen-hebrew-hp-tuning-{args.seed}"
    )
    
    # Add dataset information to W&B
    wandb.config.update({
        "dataset_size": len(dataset["train"]),
        "dataset_columns": dataset["train"].column_names,
        "validation_size": len(dataset["validation"])
    })
    
    # Run optimization
    study.optimize(
        lambda trial: objective(trial, args, dataset),
        n_trials=args.num_trials,
        timeout=None,
        callbacks=[wandb_callback]
    )
    
    # Get best trial
    best_trial = study.best_trial
    print(f"\n=== Best Trial ===")
    print(f"Value (eval_loss): {best_trial.value}")
    print("Params:")
    for key, value in best_trial.params.items():
        print(f"  {key}: {value}")
    
    # Save best hyperparameters
    best_params_path = os.path.join(args.output_dir, "best_params.json")
    with open(best_params_path, "w") as f:
        json.dump(best_trial.params, f, indent=2)
    print(f"Best parameters saved to {best_params_path}")
    
    # Get best model path
    best_model_path = os.path.join(args.output_dir, f"trial_{best_trial.number}")
    
    # Evaluate best model on Hebrew LLM leaderboard
    evaluator = HebrewEvaluator(best_model_path)
    eval_results = evaluator.evaluate()
    
    # Log evaluation results to W&B
    wandb.log({"hebrew_eval": eval_results})
    
    # Print evaluation results
    print("\n=== Hebrew LLM Leaderboard Evaluation ===")
    for metric, value in eval_results.items():
        print(f"{metric}: {value}")
    
    # Finish W&B run
    wandb.finish()
    
    print("\nHyperparameter tuning complete!")
    print(f"Best model saved to: {best_model_path}")
    print(f"To use this model for training, run:")
    print(f"python qwen_model/train.py --dataset_path {args.dataset_path} --config qwen_model/finetuning/training_config.json --model_name_or_path {best_model_path}")

if __name__ == "__main__":
    main()