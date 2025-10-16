import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import logging
from datetime import datetime

# Setup logging to file only
log_filename = f"moe_all_layers_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename)
    ]
)

logger = logging.getLogger(__name__)

model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"

print(f"Starting ALL LAYERS MoE analysis... Results will be saved to: {log_filename}")
logger.info("=== ALL LAYERS MoE Analysis Started ===")
logger.info("Loading model and tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="auto",
    device_map="auto",
    output_router_logits=True
)

# Smaller sample for ALL layers analysis
sample_prompts = {
    "English": [
        "Explain how neural networks work in machine learning.",
        "Write a short story about a robot learning to paint.", 
        "How was your day today?"
    ],
    "Hebrew": [
        "הסבר איך רשתות נוירליות עובדות בלמידת מכונה.",
        "כתוב סיפור קצר על רובוט הלומד לצייר.",
        "איך היה לך היום?"
    ]
}

def analyze_routing_patterns_all_layers(model, inputs, prompt_type, prompt_idx):
    """Analyze routing patterns in ALL MoE layers"""
    logger.info(f"\n=== MoE Routing Analysis for {prompt_type} - Prompt {prompt_idx} ===")
    
    with torch.no_grad():
        outputs = model(**inputs, output_router_logits=True)
    
    if hasattr(outputs, 'router_logits') and outputs.router_logits is not None:
        logger.info(f"Analyzing ALL {len(outputs.router_logits)} MoE layers")
        routing_stats = {}
        
        # Analyze EVERY layer (no filtering)
        for layer_idx, router_logits in enumerate(outputs.router_logits):
            if router_logits is not None:
                # Convert logits to probabilities
                router_probs = torch.softmax(router_logits, dim=-1)
                
                # Calculate routing statistics
                seq_len, num_experts = router_probs.shape[-2:]
                
                # Expert selection (top-1 routing)
                top_experts = torch.argmax(router_probs, dim=-1)
                
                # Expert usage distribution
                expert_counts = torch.bincount(top_experts.flatten(), minlength=num_experts)
                expert_usage = expert_counts.float() / top_experts.numel()
                
                # Routing entropy (measure of routing diversity)
                entropy = -torch.sum(router_probs * torch.log(router_probs + 1e-10), dim=-1)
                avg_entropy = entropy.mean()
                
                # Load balancing metric
                load_variance = torch.var(expert_usage)
                
                # Expert concentration (Gini coefficient approximation)
                sorted_usage, _ = torch.sort(expert_usage)
                n = len(sorted_usage)
                cumsum = torch.cumsum(sorted_usage, dim=0)
                gini = (n + 1 - 2 * torch.sum(cumsum) / torch.sum(sorted_usage)) / n
                
                # Store statistics
                layer_stats = {
                    'avg_entropy': avg_entropy.item(),
                    'load_variance': load_variance.item(),
                    'gini_coefficient': gini.item(),
                    'most_used_expert': torch.argmax(expert_usage).item(),
                    'max_usage': expert_usage.max().item(),
                    'num_active_experts': (expert_usage > 0.001).sum().item(),
                    'top_3_experts': torch.topk(expert_usage, 3).indices.tolist(),
                    'top_3_usage': torch.topk(expert_usage, 3).values.tolist()
                }
                
                routing_stats[f'layer_{layer_idx}'] = layer_stats
                
                # Log every single layer
                logger.info(f"Layer {layer_idx:2d}: Entropy={avg_entropy:.4f}, Variance={load_variance:.6f}, Gini={gini:.4f}, MaxUsage={expert_usage.max().item():.3f}, ActiveExperts={layer_stats['num_active_experts']}")
        
        return routing_stats
    else:
        logger.info("No router logits found in model output")
        return {}

# Run analysis on smaller sample but ALL layers
all_results = {"English": {}, "Hebrew": {}}

for lang, prompts in sample_prompts.items():
    print(f"Processing {lang} prompts (all layers)...")
    all_results[lang] = {}
    
    for idx, prompt in enumerate(prompts):
        print(f"  Prompt {idx+1}/{len(prompts)}: {prompt[:50]}...")
        logger.info(f"\nPrompt {idx+1}: {prompt}")
        
        # Prepare input
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        logger.info(f"Input tokens: {model_inputs.input_ids.shape[1]}")
        
        # Analyze routing for ALL layers
        stats = analyze_routing_patterns_all_layers(model, model_inputs, lang, idx+1)
        all_results[lang][idx+1] = stats

# Layer progression analysis
print("Analyzing layer progression...")
logger.info(f"\n{'='*80}")
logger.info("LAYER PROGRESSION ANALYSIS")
logger.info(f"{'='*80}")

# Compare how metrics change across ALL layers
for prompt_idx in range(1, len(sample_prompts["English"]) + 1):
    if prompt_idx in all_results["English"] and prompt_idx in all_results["Hebrew"]:
        logger.info(f"\nPROMPT {prompt_idx} LAYER PROGRESSION:")
        logger.info(f"EN:  {sample_prompts['English'][prompt_idx-1]}")
        logger.info(f"HEB: {sample_prompts['Hebrew'][prompt_idx-1]}")
        logger.info(f"{'Layer':<5} {'EN_Entropy':<10} {'HEB_Entropy':<11} {'EN_Variance':<12} {'HEB_Variance':<13} {'Overlap':<7}")
        logger.info("-" * 70)
        
        en_stats = all_results["English"][prompt_idx]
        heb_stats = all_results["Hebrew"][prompt_idx]
        
        for layer_num in range(48):  # All 48 layers
            layer_key = f'layer_{layer_num}'
            if layer_key in en_stats and layer_key in heb_stats:
                en_layer = en_stats[layer_key]
                heb_layer = heb_stats[layer_key]
                
                # Calculate expert overlap
                en_top = set(en_layer['top_3_experts'])
                heb_top = set(heb_layer['top_3_experts'])
                overlap = len(en_top & heb_top) / len(en_top | heb_top) if (en_top | heb_top) else 0
                
                logger.info(f"{layer_num:<5} {en_layer['avg_entropy']:<10.4f} {heb_layer['avg_entropy']:<11.4f} "
                           f"{en_layer['load_variance']:<12.6f} {heb_layer['load_variance']:<13.6f} {overlap:<7.3f}")

logger.info(f"\n{'='*80}")
logger.info("=== ALL LAYERS MoE Analysis Complete ===")

print(f"All layers analysis complete! Results saved to: {log_filename}")
