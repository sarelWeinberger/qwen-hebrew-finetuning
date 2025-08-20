import os
import json
import argparse
import subprocess
import torch
import wandb
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed,
    TrainerCallback
)
import deepspeed
from accelerate import Accelerator
import numpy as np
from typing import Dict, List, Union
from datasets import load_dataset
from datetime import datetime

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
        default="llm_train_mafat",
        help="Weights & Biases entity name"
    )
    return parser.parse_args()

# Copied from QLoRA original code
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    print(
        f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param}"
    )


def print_model_architecture(model):
    """
    Print detailed model architecture information.
    """
    print("\n" + "="*80)
    print("DETAILED MODEL ARCHITECTURE")
    print("="*80)
    
    # Basic model info
    print(f"Model class: {model.__class__.__name__}")
    print(f"Model dtype: {model.dtype}")
    
    # Configuration details
    if hasattr(model, 'config'):
        config = model.config
        print(f"\nModel Configuration:")
        print(f"  - Hidden size: {getattr(config, 'hidden_size', 'N/A')}")
        print(f"  - Number of layers: {getattr(config, 'num_hidden_layers', 'N/A')}")
        print(f"  - Number of attention heads: {getattr(config, 'num_attention_heads', 'N/A')}")
        print(f"  - Vocabulary size: {getattr(config, 'vocab_size', 'N/A')}")
        print(f"  - Max position embeddings: {getattr(config, 'max_position_embeddings', 'N/A')}")
        print(f"  - Intermediate size: {getattr(config, 'intermediate_size', 'N/A')}")
        if hasattr(config, 'num_experts'):
            print(f"  - Number of experts (MoE): {config.num_experts}")
        if hasattr(config, 'num_experts_per_tok'):
            print(f"  - Experts per token: {config.num_experts_per_tok}")
    
    # Layer structure
    print(f"\nModel Layers:")
    for name, module in model.named_children():
        module_class = module.__class__.__name__
        if hasattr(module, '__len__'):
            try:
                layer_count = len(module)
                print(f"  {name}: {module_class} (contains {layer_count} layers)")
            except:
                print(f"  {name}: {module_class}")
        else:
            print(f"  {name}: {module_class}")
    
    # Parameter analysis
    print(f"\nParameter Analysis:")
    total_params = 0
    trainable_params = 0
    param_by_type = {}
    
    for name, param in model.named_parameters():
        num_params = param.numel()
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel
            
        total_params += num_params
        if param.requires_grad:
            trainable_params += num_params
            
        # Group by parameter type
        param_type = name.split('.')[0] if '.' in name else name
        if param_type not in param_by_type:
            param_by_type[param_type] = 0
        param_by_type[param_type] += num_params
    
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Trainable percentage: {100 * trainable_params / total_params:.2f}%")
    
    print(f"\nParameters by component:")
    for param_type, count in sorted(param_by_type.items(), key=lambda x: x[1], reverse=True):
        percentage = 100 * count / total_params
        print(f"  {param_type}: {count:,} ({percentage:.1f}%)")
    
    # Memory information
    if torch.cuda.is_available():
        print(f"\nGPU Memory Usage:")
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"  GPU {i}: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")
    
    print("="*80 + "\n")


def create_tokenize_function(tokenizer, max_seq_length):
    def tokenize_function(examples):
        # Tokenize the texts with truncation but no padding here
        # Let the DataCollator handle padding during batch creation
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_length,
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

def create_s3_checkpoint_path(run_id, step=None):
    """
    Create S3 path for checkpoints.
    """
    base_path = f"s3://gepeta-checkpoints/{run_id}"
    if step is not None:
        if isinstance(step, int):
            checkpoint_name = f"step-{step}"
        else:
            checkpoint_name = f"step-{step}" if not step.startswith("step-") else step
        return f"{base_path}/{checkpoint_name}"
    return base_path

class S3CheckpointCallback(TrainerCallback):
    """Simple callback to sync checkpoints to S3 after saving."""
    
    def __init__(self, run_id):
        self.run_id = run_id
        self.uploaded_steps = set()
    
    def on_save(self, args, state, control, **kwargs):
        """Upload checkpoint to S3 after local save."""
        if state.global_step > 0 and state.global_step not in self.uploaded_steps:
            checkpoint_dir_trainer = f"checkpoint-{state.global_step}"
            checkpoint_dir_ours = f"step-{state.global_step}"
            
            local_checkpoint_path_trainer = os.path.join(args.output_dir, checkpoint_dir_trainer)
            local_checkpoint_path_ours = os.path.join(args.output_dir, checkpoint_dir_ours)
            
            # Wait a moment to ensure all files are written
            import time
            time.sleep(2)
            
            # Check if Trainer created checkpoint-{step}, and rename to step-{step}
            if os.path.exists(local_checkpoint_path_trainer) and not os.path.exists(local_checkpoint_path_ours):
                try:
                    import shutil
                    print(f"Renaming checkpoint: {checkpoint_dir_trainer} -> {checkpoint_dir_ours}")
                    shutil.move(local_checkpoint_path_trainer, local_checkpoint_path_ours)
                except Exception as e:
                    print(f"Warning: Failed to rename checkpoint: {e}")
                    # Use the original name if rename fails
                    local_checkpoint_path_ours = local_checkpoint_path_trainer
            
            # Upload whichever directory exists
            upload_path = local_checkpoint_path_ours if os.path.exists(local_checkpoint_path_ours) else local_checkpoint_path_trainer
            
            if os.path.exists(upload_path):
                s3_checkpoint_path = create_s3_checkpoint_path(self.run_id, state.global_step)
                print(f"Uploading checkpoint {os.path.basename(upload_path)} to S3...")
                try:
                    if sync_to_s3(upload_path, s3_checkpoint_path):
                        self.uploaded_steps.add(state.global_step)
                        print(f"✅ Successfully uploaded checkpoint step-{state.global_step}")
                    else:
                        print(f"❌ Failed to upload checkpoint step-{state.global_step}")
                except Exception as e:
                    print(f"❌ Error uploading checkpoint step-{state.global_step}: {e}")
            else:
                print(f"⚠️ Checkpoint directory not found: {upload_path}")

def sync_to_s3(local_path, s3_path):
    """
    Sync checkpoint from local directory to S3 using boto3.
    """
    try:
        s3 = boto3.client('s3')
        
        # Parse S3 path
        if not s3_path.startswith('s3://'):
            raise ValueError(f"Invalid S3 path: {s3_path}")
        
        path_parts = s3_path[5:].split('/', 1)
        bucket = path_parts[0]
        prefix = path_parts[1] if len(path_parts) > 1 else ""
        
        print(f"Syncing to S3: {local_path} -> {s3_path}")
        
        if not os.path.exists(local_path):
            print(f"Local path does not exist: {local_path}")
            return False
        
        # Walk through local directory and upload files
        for root, dirs, files in os.walk(local_path):
            for file in files:
                local_file_path = os.path.join(root, file)
                
                # Calculate S3 key
                relative_path = os.path.relpath(local_file_path, local_path)
                s3_key = f"{prefix}/{relative_path}".replace('\\', '/') if prefix else relative_path.replace('\\', '/')
                
                # Upload file
                s3.upload_file(local_file_path, bucket, s3_key)
                print(f"Uploaded: {local_file_path} -> s3://{bucket}/{s3_key}")
        
        return True
    except (ClientError, NoCredentialsError) as e:
        print(f"Failed to sync to S3: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error syncing to S3: {e}")
        return False

def train():
    args = parse_args()
    set_seed(args.seed)
    
    # create a training arguments to load in the deepspeed plugin before we create an Accelerator below (which doesn't default to deepspeed)
    _ = TrainingArguments('.')

    # load in the training config
    with open(args.config, 'r', encoding='utf8') as f:
        config = json.loads(f.read())

    # Create run identifier for this training session (consistent, no timestamp)
    run_id = create_run_identifier(config, args)
    print(f"Run ID: {run_id}")
    
    # Set up simple checkpoint directory - always use "local"
    local_output_dir = "./checkpoints/local"
    s3_base_path = f"s3://gepeta-checkpoints/{run_id}"
    
    # Update config output_dir to use our structured approach
    config["output_dir"] = local_output_dir
    os.makedirs(local_output_dir, exist_ok=True)

    # Initialize Weights & Biases - only on the main process
    accelerator = Accelerator()
    if accelerator.is_local_main_process:
        # Create descriptive wandb run name with timestamp for uniqueness
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        wandb_run_name = args.wandb_name or f"{run_id}_{timestamp}"
        wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=wandb_run_name,
            config=dict(
                args=args,
                config=config,
                run_id=run_id
            )
        )
        print(f"Initialized Weights & Biases run: {wandb_run_name}")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"])
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load in the model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name_or_path"],
        torch_dtype=torch.bfloat16,
        use_cache=False,  # Disable KV cache for training
        device_map=None,  # Automatically mapped using accelerate 
        low_cpu_mem_usage=True  # Unsure whether this matters
    )    

    # Print detailed model architecture
    print_model_architecture(model)

    # Print model architecture - layers only
    print("\n" + "="*50)
    print("MODEL LAYERS")
    print("="*50)
    print(f"Model class: {model.__class__.__name__}")
    print(f"Model dtype: {model.dtype}")
    
    # Print model layer structure
    print("\nModel layers:")
    for name, module in model.named_children():
        module_class = module.__class__.__name__
        if hasattr(module, '__len__'):
            try:
                layer_count = len(module)
                print(f"  {name}: {module_class} (contains {layer_count} layers)")
                
                # If it's a transformer layer container, show some sublayers
                if layer_count > 0 and hasattr(module, '__getitem__'):
                    try:
                        first_layer = module[0]
                        print(f"    Example sublayers in {name}[0]:")
                        for subname, submodule in first_layer.named_children():
                            print(f"      {subname}: {submodule.__class__.__name__}")
                    except:
                        pass
            except:
                print(f"  {name}: {module_class}")
        else:
            print(f"  {name}: {module_class}")
    
    print("="*50 + "\n")

    # Workaround for a known bug, relevant for qwen 30B only
    # Known bug with DeepSpeed freezing on the first forward: https://huggingface.co/posts/stas/984424866637646
    # This is the workaround :)
    # from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock
    # deepspeed.utils.set_z3_leaf_modules(model, [Qwen3MoeSparseMoeBlock])

    # Don't move model to GPU - let DeepSpeed handle device placement
    # Enable gradient checkpointing
    # Disabled - gradient checkpointing is done in the trainer
    # model.gradient_checkpointing_enable()
    
    # Load dataset - either from a local file or from the hug
    print(f"Loading dataset from {args.dataset_path}...")
    if os.path.isfile(args.dataset_path):
        dataset = load_dataset('json', data_files=args.dataset_path, split="train")
    else:
        dataset = load_dataset(args.dataset_path, split="train")
    
    # Print dataset info & Check if the dataset has the expected columns (which is, text)
    print(f"Dataset loaded: {dataset}")
    print(f"Dataset train split columns: {dataset.column_names}")
    if "text" not in dataset.column_names:
        raise ValueError(f"Dataset does not have a 'text' column. Available columns: {dataset.column_names}")
    
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        create_tokenize_function(tokenizer, config["max_seq_length"]),
        batched=True,
        num_proc=64,
        remove_columns=dataset.column_names,  # Remove all original columns
    )
    
    # Add debugging info about the tokenized dataset
    print(f"Tokenized dataset: {tokenized_dataset}")
    print(f"Tokenized dataset features: {tokenized_dataset.features}")
    if len(tokenized_dataset) > 0:
        print(f"Sample tokenized item: {tokenized_dataset[0]}")
        print(f"Input IDs shape: {len(tokenized_dataset[0]['input_ids'])}")
        if 'attention_mask' in tokenized_dataset[0]:
            print(f"Attention mask shape: {len(tokenized_dataset[0]['attention_mask'])}")
    
    # Training arguments optimized for 8x H100 GPUs
    print("Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir=config.get("output_dir", local_output_dir),
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        learning_rate=config.get("learning_rate", 1e-5),
        logging_steps=config.get("logging_steps", 10),
        num_train_epochs=config.get("num_train_epochs", 1) if not 'max_steps' in config else 0, 
        max_steps=config.get("max_steps", 0), # default to 1 epoch 
        save_steps=config.get("save_steps", 500),
        save_total_limit=config.get("save_total_limit", 5),
        warmup_ratio=config.get('warmup_ratio', None), # default to 100 steps
        warmup_steps=config.get('warmup_steps', 100) if not 'warmup_ratio' in config else 0,
        # Defaults:
        report_to=['wandb'],
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': True},
        save_strategy="steps",
        bf16=True,
        lr_scheduler_type="cosine",
        eval_strategy="no", # evaluation we'll do separately?
        logging_first_step=True,
        remove_unused_columns=False,
        dataloader_num_workers=4,  # Can use more workers with H100s
        push_to_hub=False, # When running fo real, set to True and specify a larger save_steps
    )
    print(f"Training arguments set up successfully: {training_args}")
    
    # Handle checkpoint resumption - SIMPLE LOCAL ONLY
    resume_from_checkpoint = None
    
    if config.get("resume_from_checkpoint") == "auto":
        print("🔍 Auto-resume mode: searching for latest LOCAL checkpoint...")
        
        # Check only the local checkpoint directory
        if os.path.exists(local_output_dir):
            print(f"📁 Contents of {local_output_dir}:")
            local_checkpoints = []
            for item in os.listdir(local_output_dir):
                item_path = os.path.join(local_output_dir, item)
                print(f"  - {item} ({'directory' if os.path.isdir(item_path) else 'file'})")
                if item.startswith(('step-', 'checkpoint-')) and os.path.isdir(item_path):
                    # Extract step number for sorting
                    try:
                        step_num = int(item.split("-")[1]) if item.split("-")[1].isdigit() else 0
                        local_checkpoints.append((step_num, item_path))
                        print(f"🔍 Found checkpoint: {item_path} (step {step_num})")
                    except:
                        pass
            
            if local_checkpoints:
                # Sort by step number and get the latest
                local_checkpoints.sort(key=lambda x: x[0])
                latest_checkpoint = local_checkpoints[-1][1]
                resume_from_checkpoint = latest_checkpoint
                print(f"✅ Will resume from: {latest_checkpoint} (step {local_checkpoints[-1][0]})")
            else:
                print("❌ No checkpoints found - starting from scratch")
        else:
            print(f"📁 Directory {local_output_dir} does not exist - starting from scratch")
            
    elif isinstance(config.get("resume_from_checkpoint"), str) and config["resume_from_checkpoint"] != "auto":
        # Use specified checkpoint path (local only)
        checkpoint_path = config["resume_from_checkpoint"]
        print(f"🔍 Using specified checkpoint: {checkpoint_path}")
        
        if os.path.exists(checkpoint_path):
            resume_from_checkpoint = checkpoint_path
            print(f"✅ Will resume from: {checkpoint_path}")
        else:
            print(f"❌ Specified checkpoint path does not exist: {checkpoint_path}")

    # Debug: Show final resume decision
    print(f"\n🎯 FINAL RESUME DECISION:")
    if resume_from_checkpoint:
        print(f"✅ Will resume from: {resume_from_checkpoint}")
    else:
        print("❌ Starting from scratch")

    # Prepare validation dataset if it exists
    # Commented out for now - if we do go with this, then eval_strategy will have to be updated
    # in training_args, and the dataset loading has to handle the splits, right now we load just
    # the train split directly
    eval_dataset = None # tokenized_dataset["validation"] if "validation" in tokenized_dataset else None
    
    # Initialize Trainer with our configuration
    # No need to move parameters to GPU, all handled automatically with accelerate
    print("Initializing Trainer...")
    
    # Create data collator with proper padding
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False,
        pad_to_multiple_of=8,  # Pad to multiple of 8 for efficiency
        return_tensors="pt"
    )

    print("model architecture:")
    print(model)

    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[S3CheckpointCallback(run_id)]  # Simple S3 upload callback
    )
    print("Trainer initialized successfully")
    
    # Start training
    print("Starting training...")
    print_trainable_parameters(model)
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
        
        # Try to upload error checkpoint to S3
        try:
            s3_error_path = create_s3_checkpoint_path(run_id, "error")
            sync_to_s3(error_checkpoint_dir, s3_error_path)
            print(f"Uploaded error checkpoint to S3: {s3_error_path}")
        except Exception as s3_error:
            print(f"Could not upload error checkpoint to S3: {s3_error}")
        
        raise e

    # Save the final model
    print(f"Saving final model to {config['output_dir']}")
    final_model_dir = os.path.join(config["output_dir"], "step-final")
    trainer.save_model(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    
    # Upload final model to S3
    final_model_s3_path = create_s3_checkpoint_path(run_id, "final")
    print(f"Uploading final model to S3: {final_model_s3_path}")
    sync_to_s3(final_model_dir, final_model_s3_path)

    print("Upload finished.")
    print("Training complete!")
    print(f"All checkpoints available in S3 under: s3://gepeta-checkpoints/{run_id}/")
    
    # Finish wandb run
    wandb.finish()

if __name__ == "__main__":
    train()