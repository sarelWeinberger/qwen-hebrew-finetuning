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
from datasets import load_dataset

class WandbMetricsCallback(TrainerCallback):
    def __init__(self):
        self.loss_history = []
        self.layer_grad_norms = {}
        self.optimizer_params = {}
        self.step_count = 0

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
        
        # Log optimizer parameters
        if self.optimizer_params:
            wandb.log(self.optimizer_params)
            self.optimizer_params.clear()
        
        # Track learning rate scheduler details
        if lr_scheduler is not None:
            try:
                if hasattr(lr_scheduler, 'get_last_lr'):
                    last_lrs = lr_scheduler.get_last_lr()
                    for i, lr in enumerate(last_lrs):
                        wandb.log({f'scheduler/lr_group_{i}': lr}, step=state.global_step)
                
                # Track scheduler state if available
                if hasattr(lr_scheduler, '_last_lr'):
                    wandb.log({'scheduler/last_lr_avg': np.mean(lr_scheduler._last_lr)})
                
                # Track step count for schedulers
                if hasattr(lr_scheduler, '_step_count'):
                    wandb.log({'scheduler/step_count': lr_scheduler._step_count})
                    
            except Exception as e:
                print(f"Warning: Could not track scheduler details: {e}")
        
        # Track layer-wise gradient norms and update patterns
        if model is not None and self.step_count % 10 == 0:  # Every 10 steps to avoid overhead
            layer_metrics = self._compute_layer_metrics(model)
            if layer_metrics:
                wandb.log(layer_metrics)
        
        # Compute and log update rhythms (every 20 steps)
        if model is not None and optimizer is not None and self.step_count % 20 == 0:
            update_metrics = self._compute_update_rhythms(model, optimizer)
            if update_metrics:
                wandb.log(update_metrics)

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
        """Compute detailed update rhythm metrics"""
        update_metrics = {}
        
        try:
            # Track parameter update magnitudes relative to current parameter values
            total_param_norm = 0
            total_grad_norm = 0
            layer_update_norms = {}
            
            for name, param in model.named_parameters():
                if param.grad is not None:
                    param_norm = torch.norm(param).item()
                    grad_norm = torch.norm(param.grad).item()
                    
                    total_param_norm += param_norm ** 2
                    total_grad_norm += grad_norm ** 2
                    
                    # Group by layer depth (if possible to extract from name)
                    layer_depth = self._extract_layer_depth(name)
                    if layer_depth is not None:
                        if layer_depth not in layer_update_norms:
                            layer_update_norms[layer_depth] = {'grad_norms': [], 'param_norms': []}
                        layer_update_norms[layer_depth]['grad_norms'].append(grad_norm)
                        layer_update_norms[layer_depth]['param_norms'].append(param_norm)
            
            # Global update rhythm
            total_param_norm = total_param_norm ** 0.5
            total_grad_norm = total_grad_norm ** 0.5
            
            if total_param_norm > 0:
                update_metrics['rhythm/global_update_ratio'] = total_grad_norm / total_param_norm
                update_metrics['rhythm/global_grad_norm'] = total_grad_norm
                update_metrics['rhythm/global_param_norm'] = total_param_norm
            
            # Layer depth update patterns
            if layer_update_norms:
                depth_ratios = []
                for depth, norms in layer_update_norms.items():
                    grad_norm_sum = sum(norms['grad_norms'])
                    param_norm_sum = sum(norms['param_norms'])
                    if param_norm_sum > 0:
                        ratio = grad_norm_sum / param_norm_sum
                        update_metrics[f'rhythm/layer_depth_{depth}_update_ratio'] = ratio
                        depth_ratios.append(ratio)
                
                if depth_ratios:
                    update_metrics['rhythm/depth_ratio_variance'] = np.var(depth_ratios)
                    update_metrics['rhythm/depth_ratio_range'] = max(depth_ratios) - min(depth_ratios)
        
        except Exception as e:
            print(f"Warning: Could not compute update rhythms: {e}")
            
        return update_metrics
    
    def _extract_layer_depth(self, param_name):
        """Extract layer depth from parameter name"""
        try:
            # Look for patterns like "layers.0", "layer.12", "h.5", etc.
            import re
            patterns = [
                r'layers?\.(\d+)',
                r'h\.(\d+)',
                r'blocks?\.(\d+)',
                r'transformer\.h\.(\d+)'
            ]
            for pattern in patterns:
                match = re.search(pattern, param_name)
                if match:
                    return int(match.group(1))
            return None
        except:
            return None

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is None or not wandb.run:
            return

        # Loss slope tracking
        if 'loss' in logs:
            self.loss_history.append(logs['loss'])
            if len(self.loss_history) >= 2:
                slope = self.loss_history[-1] - self.loss_history[-2]
                wandb.log({"loss_slope": slope})
            
            # Keep only last 100 loss values to compute moving average slope
            if len(self.loss_history) > 100:
                self.loss_history = self.loss_history[-100:]
                
            # Compute moving average slope over last 10 steps
            if len(self.loss_history) >= 10:
                recent_losses = self.loss_history[-10:]
                x = np.arange(len(recent_losses))
                slope_10 = np.polyfit(x, recent_losses, 1)[0]
                wandb.log({"loss_slope_10step": slope_10})

        # Log additional training metrics
        additional_metrics = {}
        if 'learning_rate' in logs:
            additional_metrics['training/learning_rate'] = logs['learning_rate']
        if 'grad_norm' in logs:
            additional_metrics['training/grad_norm'] = logs['grad_norm']
        if 'epoch' in logs:
            additional_metrics['training/epoch'] = logs['epoch']
            
        if additional_metrics:
            wandb.log(additional_metrics)

            # Loss volatility
            if len(self.loss_history) >= 10:
                volatility = float(np.std(self.loss_history[-10:]))
                wandb.log({"loss_volatility": volatility})

        # Gradient norm (approx)
        grad_norm = None
        try:
            parameters = [p for p in kwargs['model'].parameters() if p.grad is not None]
            total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in parameters]), 2)
            grad_norm = total_norm.item()
            wandb.log({"grad_norm": grad_norm})
        except:
            pass  # ignore if gradient not available

        # Token count (approx)
        if 'input_ids' in kwargs.get('inputs', {}):
            token_count = kwargs['inputs']['input_ids'].ne(tokenizer.pad_token_id).sum().item()
            wandb.log({"train_tokens": token_count})

        # Effective step
        eff_step = state.global_step / args.gradient_accumulation_steps
        wandb.log({"effective_step": eff_step})
        
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
        run_name = args.wandb_name or f"qwen-hebrew-finetuning-{args.dataset_name}"
        
        # Enhanced W&B configuration with optimizer and layer tracking
        wandb_config = dict(
            args=args,
            config=config,
            # Add model architecture details
            model_architecture="Qwen3-30B-A3B-Base",
            total_parameters="30.53B",
            # Training setup details for tracking
            optimizer_type="AdamW",  # Default HF optimizer
            scheduler_type=config.get("lr_scheduler_type", "cosine"),
            gradient_checkpointing=True,
            mixed_precision="bf16",
            distributed_training="DeepSpeed ZeRO-3",
            # Layer tracking configuration
            layer_metrics_frequency=10,  # Log layer metrics every 10 steps
            track_gradients=True,
            track_optimizer_params=True,
            track_update_rhythms=True
        )
        
        wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=run_name,
            config=wandb_config,
            # Enable gradient and parameter tracking
            settings=wandb.Settings(
                _disable_stats=False,  # Enable system stats
                _disable_meta=False   # Enable metadata tracking
            )
        )
        print(f"Initialized Weights & Biases run: {run_name}")
    
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
    
    # Load dataset - either from a local file or from the hug
    print(f"Loading dataset from {args.dataset_path}...")
    if os.path.isfile(args.dataset_path):
        dataset = load_dataset('json', data_files=args.dataset_path, split="train")
    else:
        dataset = load_dataset(args.dataset_path, split="train")
    
    # Print dataset info & Check if the dataset has the expected columns (which is, `text`)
    print(f"Dataset loaded: {dataset}")
    print(f"Dataset train split columns: {dataset.column_names}")
    if "text" not in dataset.column_names:
        raise ValueError(f"Dataset does not have a 'text' column. Available columns: {dataset.column_names}")
    
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        create_tokenize_function(tokenizer, config["max_seq_length"]),
        batched=True,
        num_proc=64,
        remove_columns=["text", "id", "metadata"]  # Remove all non-tokenization columns
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
        output_dir=config.get("output_dir", "./output"),
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
    
    trainer = Trainer(
        model=model,
        processing_class=tokenizer, # renamed from tokenizer
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        callbacks=[WandbMetricsCallback()]
    )
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