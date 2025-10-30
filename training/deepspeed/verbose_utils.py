from transformers import TrainerCallback

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

