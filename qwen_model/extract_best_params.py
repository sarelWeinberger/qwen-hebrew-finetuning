#!/usr/bin/env python3
"""
Script to extract the best hyperparameters from hyperparameter tuning results
and generate a training configuration file.
"""

import os
import json
import argparse
import glob
import wandb
from datetime import datetime

def find_best_params_file(hp_tuning_dir):
    """Find the best_params.json file in the hyperparameter tuning directory."""
    best_params_path = os.path.join(hp_tuning_dir, "best_params.json")
    if os.path.exists(best_params_path):
        print(f"Found best_params.json at {best_params_path}")
        return best_params_path
    
    # If best_params.json doesn't exist, look for it in trial subdirectories
    trial_dirs = glob.glob(os.path.join(hp_tuning_dir, "trial_*"))
    for trial_dir in trial_dirs:
        best_params_path = os.path.join(trial_dir, "best_params.json")
        if os.path.exists(best_params_path):
            print(f"Found best_params.json at {best_params_path}")
            return best_params_path
    
    return None

def get_best_params_from_wandb(project_name, entity_name=None):
    """Get the best hyperparameters from W&B."""
    try:
        # Initialize W&B
        wandb.login()
        
        # Get the best run from the project
        api = wandb.Api()
        runs = api.runs(f"{entity_name}/{project_name}" if entity_name else project_name)
        
        best_run = None
        best_loss = float('inf')
        
        for run in runs:
            # Check if the run has a best_loss metric
            if 'best_loss' in run.summary:
                loss = run.summary['best_loss']
                if loss < best_loss:
                    best_loss = loss
                    best_run = run
        
        if best_run:
            # Extract the best hyperparameters
            best_params = {}
            for key, value in best_run.config.items():
                if key.startswith('best_'):
                    param_name = key[5:]  # Remove 'best_' prefix
                    best_params[param_name] = value
            
            if best_params:
                print(f"Found best hyperparameters from W&B run {best_run.name}")
                return best_params
            
            # If no best_ parameters found, try to find trial parameters
            for key, value in best_run.config.items():
                if key.startswith('param_'):
                    param_name = key[6:]  # Remove 'param_' prefix
                    best_params[param_name] = value
            
            if best_params:
                print(f"Found hyperparameters from W&B run {best_run.name}")
                return best_params
        
        print("Could not find best hyperparameters in W&B")
        return None
    
    except Exception as e:
        print(f"Error accessing W&B: {e}")
        return None

def create_training_config(best_params, output_path, model_path, dataset_path, deepspeed_config=None):
    """Create a training configuration file with the best hyperparameters."""
    # Default training parameters
    training_config = {
        "model_name_or_path": model_path,
        "dataset_path": dataset_path,
        "output_dir": f"qwen_model/finetuning/training_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        "num_epochs": 3,
        "gradient_accumulation_steps": 8,
        "fp16": True,
        "gradient_checkpointing": True,
        "per_device_train_batch_size": 1,
        "logging_steps": 10,
        "save_steps": 100,
        "save_total_limit": 3,
        "evaluation_strategy": "steps",
        "eval_steps": 500,
        "dataloader_num_workers": 1,
        "optim": "adamw_torch_fused"
    }
    
    # Add DeepSpeed configuration if provided
    if deepspeed_config:
        training_config["deepspeed"] = deepspeed_config
    
    # Update with best hyperparameters
    for key, value in best_params.items():
        if key in ["learning_rate", "weight_decay", "warmup_ratio"]:
            training_config[key] = value
    
    # Write the configuration to a file
    with open(output_path, 'w') as f:
        json.dump(training_config, f, indent=2)
    
    print(f"Training configuration saved to {output_path}")
    return training_config

def main():
    parser = argparse.ArgumentParser(description="Extract best hyperparameters and create training config")
    parser.add_argument(
        "--hp_tuning_dir",
        type=str,
        default="qwen_model/finetuning/hp_tuning",
        help="Directory containing hyperparameter tuning results"
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="qwen_model/finetuning/training_config.json",
        help="Path to save the training configuration"
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="qwen_model/model",
        help="Path to the model"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="qwen_model/data/dataset/dataset",
        help="Path to the dataset"
    )
    parser.add_argument(
        "--deepspeed_config",
        type=str,
        default="qwen_model/deepspeed_config.json",
        help="Path to DeepSpeed configuration file"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="qwen-hebrew-hp-tuning",
        help="W&B project name"
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="W&B entity name"
    )
    args = parser.parse_args()
    
    # Find the best parameters file
    best_params_path = find_best_params_file(args.hp_tuning_dir)
    
    best_params = None
    if best_params_path:
        # Load the best parameters from the file
        with open(best_params_path, 'r') as f:
            best_params = json.load(f)
        print(f"Loaded best parameters from {best_params_path}")
    else:
        # Try to get the best parameters from W&B
        print("Could not find best_params.json, trying to get parameters from W&B...")
        best_params = get_best_params_from_wandb(args.wandb_project, args.wandb_entity)
    
    if not best_params:
        print("Could not find best hyperparameters. Make sure hyperparameter tuning has completed.")
        return
    
    # Create the training configuration
    training_config = create_training_config(
        best_params,
        args.output_path,
        args.model_path,
        args.dataset_path,
        args.deepspeed_config
    )
    
    # Print command to run training
    print("\nTo run training with the best hyperparameters, use:")
    print(f"python qwen_model/train.py --config {args.output_path}")

if __name__ == "__main__":
    main()