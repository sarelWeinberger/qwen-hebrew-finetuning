import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def prepare_model_for_finetuning():
    """
    Prepare the Qwen3-30B-A3B-Base model for full checkpoint fine-tuning using 16-bit mixed precision
    """
    print("Preparing model for full checkpoint fine-tuning with 16-bit mixed precision...")
    
    # Set the model path
    model_path = os.path.join(os.getcwd(), "qwen_model/model")
    
    # Check if model exists
    if not os.path.exists(os.path.join(model_path, "config.json")):
        print(f"Model not found at {model_path}. Please download the model first.")
        return
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    print("Loading model in 16-bit mixed precision (this may take a while)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,  # Use 16-bit precision
        device_map="auto",          # Automatically distribute model across available GPUs
        trust_remote_code=True
    )
    
    # Print model information
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel loaded successfully with {total_params/1e9:.2f} billion parameters")
    print(f"Model dtype: {model.dtype}")
    
    # Save model configuration for training
    config_path = os.path.join(os.getcwd(), "qwen_model/finetuning/training_config.json")
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    # Create a sample training configuration
    training_config = {
        "model_name_or_path": model_path,
        "output_dir": "qwen_model/finetuned",
        "fp16": True,                  # Enable mixed precision training
        "gradient_accumulation_steps": 8,
        "per_device_train_batch_size": 1,
        "learning_rate": 1e-5,
        "weight_decay": 0.01,
        "num_train_epochs": 3,
        "save_strategy": "steps",
        "save_steps": 500,
        "save_total_limit": 3,
        "logging_steps": 100,
        "max_grad_norm": 1.0,
        "warmup_ratio": 0.03
    }
    
    # Save the configuration
    import json
    with open(config_path, 'w') as f:
        json.dump(training_config, f, indent=2)
    
    print("\nModel prepared for full checkpoint fine-tuning with 16-bit mixed precision!")
    print(f"\nTraining configuration saved to {config_path}")
    print("\nTo fine-tune the model, you will need:")
    print("1. A dataset in the appropriate format")
    print("2. A training script that uses the prepared model")
    print("\nSample training command:")
    print("python qwen_model/train.py --config qwen_model/finetuning/training_config.json --dataset_path your_dataset")
    
    print("\nNote: Full checkpoint fine-tuning of a 30B parameter model requires:")
    print("- Multiple high-memory GPUs (recommended: at least 4x A100 80GB or equivalent)")
    print("- DeepSpeed or FSDP for distributed training")
    print("- Gradient checkpointing for memory efficiency")

if __name__ == "__main__":
    prepare_model_for_finetuning()