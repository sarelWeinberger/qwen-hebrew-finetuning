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
import deepspeed
from accelerate import Accelerator
import numpy as np
from typing import Dict, List, Union
from datasets import load_dataset, Dataset

# --- Custom Data Collator for Document Packing ---
class DataCollatorForDocumentPacking(DataCollatorForLanguageModeling):
    """
    Data Collator that creates a document-aware causal attention mask and
    document-relative position_ids for packed sequences.
    """
    def __init__(self, tokenizer, **kwargs):
        super().__init__(tokenizer=tokenizer, mlm=False, **kwargs)
        self.tokenizer = tokenizer
        self.eos_token_id = tokenizer.eos_token_id

    def torch_call(self, examples: list[dict]) -> dict:
        batch = super().torch_call(examples)
        
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        
        if not isinstance(input_ids, torch.Tensor):
            input_ids = torch.stack(input_ids)
        
        batch_size, sequence_length = input_ids.shape
        device = input_ids.device
        
        # Initialize attention_mask and position_ids
        attention_mask = torch.zeros(
            (batch_size, sequence_length, sequence_length), dtype=torch.bool, device=device
        )
        position_ids = torch.zeros_like(input_ids, device=device)
        
        # Build the document-aware causal mask and position ids
        for i in range(batch_size):
            sequence_ids = input_ids[i]
            # Find the indices of the EOS tokens
            eos_indices = (sequence_ids == self.eos_token_id).nonzero(as_tuple=False).squeeze(-1)
            
            doc_start_indices = [0] + (eos_indices[:-1] + 1).tolist()
            doc_end_indices = eos_indices.tolist()

            current_pos = 0
            for doc_start, doc_end in zip(doc_start_indices, doc_end_indices):
                doc_len = doc_end - doc_start + 1
                
                # Update position_ids for the current document
                position_ids[i, doc_start : doc_end + 1] = torch.arange(doc_len, device=device)
                
                # Apply causal mask within the document boundaries
                attention_mask[i, doc_start : doc_end + 1, doc_start : doc_end + 1] = torch.tril(
                    torch.ones(doc_len, doc_len, dtype=torch.bool, device=device)
                )

        # The model's attention mechanism expects a 2D mask.
        # We can either provide a 3D mask if the model supports it, or flatten it.
        # Here we provide a causal mask that is `True` for all non-padded tokens.
        # This simplification might need adjustment based on the model's exact implementation.
        batch["attention_mask"] = attention_mask.any(dim=1).to(device)
        batch["position_ids"] = position_ids
        
        return batch

# --- Your original code starts here ---

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
        
        # Log validation metrics with val/ prefix to match your expectation
        validation_metrics = {}
        for key, value in logs.items():
            if key.startswith('eval_'):
                # Remove eval_ prefix and add val/ prefix
                clean_key = key.replace('eval_', '')
                validation_metrics[f'val/{clean_key}'] = value
        
        if validation_metrics:
            wandb.log(validation_metrics, step=state.global_step)
            print(f"Validation metrics logged: {validation_metrics}")

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Called when the trainer logs training metrics"""
        if not wandb.run or logs is None:
            return
        
        # Since report_to=["wandb"] is enabled, basic metrics (loss, learning_rate, etc.) 
        # are automatically logged by the built-in integration.
        # This callback now focuses on additional metrics and validation prefixing
        
        # Log additional training metrics with prefixes (but skip basic ones to avoid duplicates)
        additional_metrics = {}
        basic_metrics = {'loss', 'learning_rate', 'epoch', 'grad_norm', 'train_loss'}
        
        for key, value in logs.items():
            # Skip eval metrics as they're handled in on_evaluate
            # Skip basic metrics that are already logged by built-in integration
            if not key.startswith('eval_') and key not in basic_metrics:
                additional_metrics[f'train/{key}'] = value
        
        if additional_metrics:
            wandb.log(additional_metrics, step=state.global_step)

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
        default='datasets/BIU.jsonl',
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
    parser.add_argument(
        "--validation_split",
        type=float,
        default=0.1,
        help="Fraction of data to use for validation (default: 0.1 = 10%)"
    )
    parser.add_argument(
        "--validation_seed",
        type=int,
        default=42,
        help="Seed for validation split"
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

# New function for packing
def group_texts(examples, block_size):
    """
    Concatenates all texts and chunks them into fixed-size blocks.
    """
    # Concatenate all texts from the 'input_ids' and 'attention_mask' lists
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])

    # We drop the last chunk if it's smaller than the block_size
    total_length = (total_length // block_size) * block_size

    # Split by chunks of block_size
    result = {
        k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
        for k, t in concatenated_examples.items()
    }

    # Create the labels by shifting the input_ids
    result["labels"] = result["input_ids"].copy()
    return result

# Modified tokenize function for packing
def create_tokenize_function_for_packing(tokenizer):
    """
    Tokenizes the texts without truncation.
    """
    def tokenize_function(examples):
        # Tokenize the texts without truncation and return attention_mask
        return tokenizer(
            examples["text"],
            truncation=False,
            return_attention_mask=True
        )
    return tokenize_function

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
        run_name = args.wandb_name or f"qwen-hebrew-{args.seed}"
        
        # Enhanced W&B configuration
        wandb_config = dict(
            # Training arguments
            **{k: v for k, v in args.__dict__.items()},
            # Model and training config
            **config,
            # Additional metadata
            model_name=config["model_name_or_path"],
            total_parameters="30.53B",
            gpu_count=torch.cuda.device_count() if torch.cuda.is_available() else 0,
            distributed_training=True,
            deepspeed_enabled=True,
            validation_enabled=args.validation_split > 0,
            # Environment info
            torch_version=torch.__version__,
            transformers_version="4.47.1",  # Update as needed
            python_version="3.12",
        )
        
        wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=run_name,
            config=wandb_config,
            tags=["qwen", "hebrew", "fine-tuning", "deepspeed", "validation"],
            notes=f"Training {config['model_name_or_path']} on {args.dataset_path} with {args.validation_split:.1%} validation split"
        )
        print(f"Initialized Weights & Biases run: {run_name}")
        
        # Log model architecture info
        if wandb.run:
            wandb.run.summary["model_parameters"] = "30.53B"
            wandb.run.summary["model_architecture"] = "Qwen3-30B-A3B-Base"
    
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

    # Known bug with DeepSpeed freezing on the first forward: https://huggingface.co/posts/stas/984424866637646
    # This is the workaround :)
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock
    deepspeed.utils.set_z3_leaf_modules(model, [Qwen3MoeSparseMoeBlock])

    # Don't move model to GPU - let DeepSpeed handle device placement
    # Enable gradient checkpointing
    # Disabled - gradient checkpointing is done in the trainer
    # model.gradient_checkpointing_enable()
    
    # Load dataset - either from a local file or from huggingface
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
    
    print("Tokenizing and packing datasets...")

    block_size = config["max_seq_length"]
    tokenize_function = create_tokenize_function_for_packing(tokenizer)
    
    # Tokenize and pack training dataset
    tokenized_train_dataset = train_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=64,
        remove_columns=["text", "id", "metadata"]  # Remove all non-tokenization columns
    )
    tokenized_train_dataset = tokenized_train_dataset.map(
        lambda examples: group_texts(examples, block_size),
        batched=True,
        num_proc=64,
        remove_columns=tokenized_train_dataset.column_names, # Remove old columns after packing
    )

    # Tokenize and pack validation dataset if it exists
    tokenized_val_dataset = None
    if val_dataset is not None:
        tokenized_val_dataset = val_dataset.map(
            tokenize_function,
            batched=True,
            num_proc=64,
            remove_columns=["text", "id", "metadata"]
        )
        tokenized_val_dataset = tokenized_val_dataset.map(
            lambda examples: group_texts(examples, block_size),
            batched=True,
            num_proc=64,
            remove_columns=tokenized_val_dataset.column_names
        )
        print(f"Tokenized and packed validation dataset: {tokenized_val_dataset}")
    
    # Add debugging info about the tokenized datasets
    print(f"Tokenized and packed train dataset: {tokenized_train_dataset}")
    print(f"Tokenized and packed train dataset features: {tokenized_train_dataset.features}")
    if len(tokenized_train_dataset) > 0:
        print(f"Sample tokenized train item: {tokenized_train_dataset[0]}")
        print(f"Train input IDs shape: {len(tokenized_train_dataset[0]['input_ids'])}")
        if 'attention_mask' in tokenized_train_dataset[0]:
            print(f"Train attention mask shape: {len(tokenized_train_dataset[0]['attention_mask'])}")
    
    if tokenized_val_dataset is not None and len(tokenized_val_dataset) > 0:
        print(f"Sample tokenized validation item: {tokenized_val_dataset[0]}")
        print(f"Validation input IDs shape: {len(tokenized_val_dataset[0]['input_ids'])}")
        if 'attention_mask' in tokenized_val_dataset[0]:
            print(f"Validation attention mask shape: {len(tokenized_val_dataset[0]['attention_mask'])}")
    
    # Training arguments optimized for 8x H100 GPUs
    print("Setting up training arguments...")
    
    # Set evaluation strategy based on whether we have validation data
    eval_strategy = "steps" if tokenized_val_dataset is not None else "no"
    eval_steps = config.get("eval_steps", 100) if tokenized_val_dataset is not None else None
    
    training_args = TrainingArguments(
        output_dir=config.get("output_dir", "./output"),
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        per_device_eval_batch_size=config.get("per_device_eval_batch_size", config["per_device_train_batch_size"]),
        learning_rate=config.get("learning_rate", 1e-4),
        logging_steps=config.get("logging_steps", 10),
        num_train_epochs=config.get("num_train_epochs", 1) if not 'max_steps' in config else 0, 
        max_steps=config.get("max_steps", 0), # default to 1 epoch 
        save_steps=config.get("save_steps", 500),
        save_total_limit=config.get("save_total_limit", 5),
        warmup_ratio=config.get('warmup_ratio', None), # default to 100 steps
        warmup_steps=config.get('warmup_steps', 100) if not 'warmup_ratio' in config else 0,
        # Evaluation settings
        eval_strategy=eval_strategy,
        eval_steps=eval_steps,
        eval_accumulation_steps=config.get("eval_accumulation_steps", None),
        # Defaults:
        report_to=["wandb"],  # Enable W&B reporting for basic metrics
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': True},
        save_strategy="steps",
        bf16=True,
        lr_scheduler_type="cosine",
        logging_first_step=True,
        remove_unused_columns=False,
        dataloader_num_workers=4,  # Can use more workers with H100s
        push_to_hub=False, # When running fo real, set to True and specify a larger save_steps
        # Load best model at end if we have validation
        load_best_model_at_end=tokenized_val_dataset is not None,
        metric_for_best_model="eval_loss" if tokenized_val_dataset is not None else None,
        greater_is_better=False if tokenized_val_dataset is not None else None,
    )
    print(f"Training arguments set up successfully: {training_args}")
    
    # Initialize Trainer with our configuration
    # Note: We no longer need DataCollatorForLanguageModeling because our data is already packed.
    # All sequences are the same length.
    print("Initializing Trainer...")
    
    # Create the callback object
    wandb_callback = WandbMetricsCallback()
    
    # The new DataCollatorForDocumentPacking will handle the attention mask and position_ids for packed data.
    data_collator = DataCollatorForDocumentPacking(tokenizer)

    trainer = Trainer(
        model=model,
        processing_class=tokenizer, # renamed from tokenizer
        args=training_args,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,  # This can be None
        data_collator=data_collator, # Pass the new custom collator
        callbacks=[wandb_callback]  # Add our custom callback
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
    print("Trainer initialized successfully")
    
    # Start training
    print("Starting training...")
    print_trainable_parameters(model)
    try:
        trainer.train(resume_from_checkpoint=config.get("resume_from_checkpoint", False))
    except Exception as e:
        print(f"Training error: {e}")
        # Save checkpoint even if training fails
        trainer.save_model(config["output_dir"] + "/error_checkpoint")
        print(f"Saved checkpoint after error to {config['output_dir']}/error_checkpoint")
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