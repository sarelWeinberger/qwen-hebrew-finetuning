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
    return parser.parse_args()

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
            padding='max_length'
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

class WandbMetricsCallback(TrainerCallback):
    """Enhanced W&B callback for comprehensive training and validation monitoring"""
    
    def __init__(self):
        self.step_count = 0
        self.optimizer_params = {}
        
    def on_step_begin(self, args, state, control, model=None, optimizer=None, **kwargs):
        """Called at the beginning of each training step"""
        if not wandb.run or optimizer is None:
            return
        
        # Track optimizer parameters for each parameter group
        if hasattr(optimizer, 'param_groups'):
            for i, param_group in enumerate(optimizer.param_groups):
                # Basic optimizer parameters
                self.optimizer_params[f'optimizer/lr_group_{i}'] = param_group.get('lr', 0)
                self.optimizer_params[f'optimizer/weight_decay_group_{i}'] = param_group.get('weight_decay', 0)
                self.optimizer_params[f'optimizer/eps_group_{i}'] = param_group.get('eps', 0)
                
                # Adam/AdamW specific parameters
                if 'betas' in param_group:
                    self.optimizer_params[f'optimizer/beta1_group_{i}'] = param_group['betas'][0]
                    self.optimizer_params[f'optimizer/beta2_group_{i}'] = param_group['betas'][1]
                
                # Parameter group size
                self.optimizer_params[f'optimizer/param_count_group_{i}'] = len(param_group['params'])
        
        # Track optimizer state statistics (for debugging convergence)
        if hasattr(optimizer, 'state') and self.step_count % 50 == 0:  # Every 50 steps
            try:
                total_params = 0
                exp_avg_norms = []
                exp_avg_sq_norms = []
                
                for param_group in optimizer.param_groups:
                    for param in param_group['params']:
                        if param in optimizer.state:
                            state = optimizer.state[param]
                            total_params += 1
                            
                            # Track momentum and second moment statistics
                            if 'exp_avg' in state:
                                exp_avg_norms.append(torch.norm(state['exp_avg']).item())
                            if 'exp_avg_sq' in state:
                                exp_avg_sq_norms.append(torch.norm(state['exp_avg_sq']).item())
                
                if exp_avg_norms:
                    self.optimizer_params['optimizer/momentum_norm_mean'] = np.mean(exp_avg_norms)
                    self.optimizer_params['optimizer/momentum_norm_std'] = np.std(exp_avg_norms)
                if exp_avg_sq_norms:
                    self.optimizer_params['optimizer/second_moment_norm_mean'] = np.mean(exp_avg_sq_norms)
                    self.optimizer_params['optimizer/second_moment_norm_std'] = np.std(exp_avg_sq_norms)
                    
                self.optimizer_params['optimizer/total_tracked_params'] = total_params
                
            except Exception as e:
                print(f"Warning: Could not track optimizer state: {e}")

    def on_step_end(self, args, state, control, model=None, optimizer=None, lr_scheduler=None, **kwargs):
        """Called at the end of each training step"""
        if not wandb.run:
            return
        
        self.step_count += 1
        
        # Consolidate all metrics into a single log call to avoid step conflicts
        all_metrics = {}
        
        # Add optimizer parameters
        if self.optimizer_params:
            all_metrics.update(self.optimizer_params)
            self.optimizer_params.clear()
        
        # Track learning rate scheduler details
        if lr_scheduler is not None:
            try:
                if hasattr(lr_scheduler, 'get_last_lr'):
                    last_lrs = lr_scheduler.get_last_lr()
                    for i, lr in enumerate(last_lrs):
                        all_metrics[f'scheduler/lr_group_{i}'] = lr
                
                # Track scheduler state if available
                if hasattr(lr_scheduler, '_last_lr'):
                    all_metrics['scheduler/last_lr_avg'] = np.mean(lr_scheduler._last_lr)
                
                # Track step count for schedulers
                if hasattr(lr_scheduler, '_step_count'):
                    all_metrics['scheduler/step_count'] = lr_scheduler._step_count
                    
            except Exception as e:
                print(f"Warning: Could not track scheduler details: {e}")
        
        # Track layer-wise gradient norms and update patterns (every 10 steps)
        if model is not None and self.step_count % 10 == 0:
            layer_metrics = self._compute_layer_metrics(model)
            if layer_metrics:
                all_metrics.update(layer_metrics)
        
        # Compute and log update rhythms (every 20 steps)
        if model is not None and optimizer is not None and self.step_count % 20 == 0:
            update_metrics = self._compute_update_rhythms(model, optimizer)
            if update_metrics:
                all_metrics.update(update_metrics)
        
        # Log all metrics in a single call to prevent step conflicts
        if all_metrics:
            wandb.log(all_metrics, step=state.global_step)

    def on_evaluate(self, args, state, control, model=None, logs=None, **kwargs):
        """Called after validation evaluation"""
        if not wandb.run or logs is None:
            return
        
        # Log validation metrics with special prefix
        validation_metrics = {}
        for key, value in logs.items():
            if key.startswith('eval_'):
                # Remove eval_ prefix and add validation_ prefix
                clean_key = key.replace('eval_', '')
                validation_metrics[f'validation/{clean_key}'] = value
        
        if validation_metrics:
            wandb.log(validation_metrics, step=state.global_step)
            print(f"Validation metrics logged: {validation_metrics}")

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when the trainer logs training metrics"""
        if not wandb.run or logs is None:
            return
        
        # Log training metrics (loss, learning_rate, epoch, etc.)
        training_metrics = {}
        for key, value in logs.items():
            # Skip eval metrics as they're handled in on_evaluate
            if not key.startswith('eval_'):
                training_metrics[f'train/{key}'] = value
        
        if training_metrics:
            wandb.log(training_metrics, step=state.global_step)

    def _compute_layer_metrics(self, model):
        """Compute layer-wise gradient norms and parameter statistics"""
        layer_metrics = {}
        
        try:
            # Group parameters by layer type
            layer_groups = {
                'attention': [],
                'mlp': [],
                'embedding': [],
                'norm': [],
                'other': []
            }
            
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = torch.norm(param.grad).item()
                    param_norm = torch.norm(param).item()
                    
                    # Categorize layers
                    if 'attn' in name.lower() or 'attention' in name.lower():
                        layer_type = 'attention'
                    elif 'mlp' in name.lower() or 'feed_forward' in name.lower() or 'ffn' in name.lower():
                        layer_type = 'mlp'
                    elif 'embed' in name.lower():
                        layer_type = 'embedding'
                    elif 'norm' in name.lower() or 'ln' in name.lower():
                        layer_type = 'norm'
                    else:
                        layer_type = 'other'
                    
                    layer_groups[layer_type].append({
                        'grad_norm': grad_norm,
                        'param_norm': param_norm,
                        'name': name
                    })
            
            # Compute statistics for each layer type
            for layer_type, params in layer_groups.items():
                if params:
                    grad_norms = [p['grad_norm'] for p in params]
                    param_norms = [p['param_norm'] for p in params]
                    
                    layer_metrics[f'layer_gradients/{layer_type}_grad_norm_mean'] = np.mean(grad_norms)
                    layer_metrics[f'layer_gradients/{layer_type}_grad_norm_std'] = np.std(grad_norms)
                    layer_metrics[f'layer_gradients/{layer_type}_grad_norm_max'] = np.max(grad_norms)
                    layer_metrics[f'layer_gradients/{layer_type}_param_norm_mean'] = np.mean(param_norms)
                    layer_metrics[f'layer_gradients/{layer_type}_param_norm_std'] = np.std(param_norms)
                    
                    # Update rhythm - ratio of gradient norm to parameter norm
                    update_ratios = [g/max(p, 1e-8) for g, p in zip(grad_norms, param_norms)]
                    layer_metrics[f'layer_updates/{layer_type}_update_ratio_mean'] = np.mean(update_ratios)
                    layer_metrics[f'layer_updates/{layer_type}_update_ratio_std'] = np.std(update_ratios)
            
            # Global gradient statistics
            all_grad_norms = []
            all_param_norms = []
            for params in layer_groups.values():
                all_grad_norms.extend([p['grad_norm'] for p in params])
                all_param_norms.extend([p['param_norm'] for p in params])
            
            if all_grad_norms:
                layer_metrics['gradients/global_grad_norm_mean'] = np.mean(all_grad_norms)
                layer_metrics['gradients/global_grad_norm_std'] = np.std(all_grad_norms)
                layer_metrics['gradients/global_grad_norm_max'] = np.max(all_grad_norms)
                layer_metrics['gradients/global_param_norm_mean'] = np.mean(all_param_norms)
                
                # Global update rhythm
                global_update_ratios = [g/max(p, 1e-8) for g, p in zip(all_grad_norms, all_param_norms)]
                layer_metrics['updates/global_update_ratio_mean'] = np.mean(global_update_ratios)
                layer_metrics['updates/global_update_ratio_std'] = np.std(global_update_ratios)
        
        except Exception as e:
            print(f"Warning: Could not compute layer metrics: {e}")
            
        return layer_metrics

    def _compute_update_rhythms(self, model, optimizer):
        """Compute advanced update rhythm metrics"""
        update_metrics = {}
        
        try:
            param_updates = []
            param_magnitudes = []
            
            for param_group in optimizer.param_groups:
                for param in param_group['params']:
                    if param.grad is not None:
                        # Approximate parameter update magnitude
                        lr = param_group['lr']
                        grad_norm = torch.norm(param.grad).item()
                        param_norm = torch.norm(param).item()
                        
                        # Estimated update magnitude
                        update_magnitude = lr * grad_norm
                        param_updates.append(update_magnitude)
                        param_magnitudes.append(param_norm)
            
            if param_updates:
                update_metrics['rhythm/update_magnitude_mean'] = np.mean(param_updates)
                update_metrics['rhythm/update_magnitude_std'] = np.std(param_updates)
                update_metrics['rhythm/update_magnitude_max'] = np.max(param_updates)
                
                # Relative update size
                relative_updates = [u/max(p, 1e-8) for u, p in zip(param_updates, param_magnitudes)]
                update_metrics['rhythm/relative_update_mean'] = np.mean(relative_updates)
                update_metrics['rhythm/relative_update_std'] = np.std(relative_updates)
                
        except Exception as e:
            print(f"Warning: Could not compute update rhythms: {e}")
        
        return update_metrics



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
        
        # Enhanced W&B configuration
        wandb_config = dict(
            # Training arguments
            **{k: v for k, v in args.__dict__.items()},
            # Model and training config
            **config,
            # Additional metadata
            model_name=config["model_name_or_path"],
            gpu_count=torch.cuda.device_count() if torch.cuda.is_available() else 0,
            distributed_training=True,
            deepspeed_enabled=True,
            validation_enabled=args.validation_split > 0,
            # Environment info
            torch_version=torch.__version__,
        )
        
        wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=wandb_run_name,
            config=wandb_config,
            tags=["qwen", "hebrew", "fine-tuning", "deepspeed"] + (["validation"] if args.validation_split > 0 else []),
            notes=f"Training {config['model_name_or_path']} on {args.dataset_path}" + (f" with {args.validation_split:.1%} validation split" if args.validation_split > 0 else "")
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
        dtype=torch.bfloat16,
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
    if config["model_name_or_path"].startswith("Qwen/Qwen3-30B"):
        print("🔧 Applying DeepSpeed workaround for Qwen3-30B model...")
        from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock
        deepspeed.utils.set_z3_leaf_modules(model, [Qwen3MoeSparseMoeBlock])
        print("✅ DeepSpeed workaround applied successfully")
    else:
        print(f"ℹ️  Model {config['model_name_or_path']} does not require DeepSpeed workaround")

    # Don't move model to GPU - let DeepSpeed handle device placement
    # Enable gradient checkpointing
    # Disabled - gradient checkpointing is done in the trainer
    # model.gradient_checkpointing_enable()
    
    # Load dataset - either from a local file or from the huggingface hub
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
    
    # Create train/validation split
    if args.validation_split > 0:
        print(f"Creating train/validation split with {args.validation_split:.1%} validation data...")
        split_dataset = dataset.train_test_split(
            test_size=args.validation_split,
            seed=args.validation_seed,
            shuffle=True
        )
        train_dataset = split_dataset['train']
        val_dataset = split_dataset['test']
        print(f"Train dataset size: {len(train_dataset)}")
        print(f"Validation dataset size: {len(val_dataset)}")
    else:
        print("No validation split created (validation_split=0)")
        train_dataset = dataset
        val_dataset = None
    
    print("Tokenizing datasets...")
    
    # Tokenize training dataset
    tokenized_train_dataset = train_dataset.map(
        create_tokenize_function(tokenizer, config["max_seq_length"]),
        batched=True,
        num_proc=64,
        remove_columns=train_dataset.column_names,  # Remove all original columns
    )
    
    # Tokenize validation dataset if it exists
    tokenized_val_dataset = None
    if val_dataset is not None:
        tokenized_val_dataset = val_dataset.map(
            create_tokenize_function(tokenizer, config["max_seq_length"]),
            batched=True,
            num_proc=64,
            remove_columns=val_dataset.column_names,  # Remove all original columns
        )
        print(f"Tokenized validation dataset: {tokenized_val_dataset}")

    # Print some statistics about the tokenized datasets
    print(f"Tokenized train dataset size: {len(tokenized_train_dataset)}")
    if tokenized_val_dataset is not None:
        print(f"Tokenized validation dataset size: {len(tokenized_val_dataset)}")
    
    # Training arguments optimized for 8x H100 GPUs
    print("Setting up training arguments...")
    
    # Set evaluation strategy based on whether we have validation data
    eval_strategy = "steps" if tokenized_val_dataset is not None else "no"
    eval_steps = config.get("eval_steps", 100) if tokenized_val_dataset is not None else None

    training_args = TrainingArguments(
        output_dir=config.get("output_dir", local_output_dir),
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
        eval_strategy=eval_strategy,
        eval_steps=eval_steps,
        eval_accumulation_steps=config.get("eval_accumulation_steps", None),
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

    # Initialize custom W&B callback for enhanced monitoring
    wandb_callback = WandbMetricsCallback()

    trainer = Trainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,  # This will be None if no validation split
        data_collator=data_collator,
        callbacks=[wandb_callback]
    )
    print("Trainer initialized successfully")
    
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
    print("Starting training...")
    try:
        trainer.train(resume_from_checkpoint=False)
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
    
    # Finish wandb run
    wandb.finish()

if __name__ == "__main__":
    train()