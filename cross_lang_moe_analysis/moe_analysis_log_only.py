import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import logging
from datetime import datetime

# Setup logging to file only
log_filename = f"moe_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename)
    ]
)

logger = logging.getLogger(__name__)

model_name = "Qwen/Qwen3-30B-A3B-Base"

print(f"Starting MoE analysis... Results will be saved to: {log_filename}")
logger.info("=== MoE Analysis Started ===")
logger.info("Loading model and tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="auto",
    device_map="auto",
    output_router_logits=True
)

def analyze_routing_patterns(model, inputs, prompt_type):
    """Analyze routing patterns in MoE layers"""
    logger.info(f"\n=== MoE Routing Analysis for {prompt_type} ===")
    
    with torch.no_grad():
        outputs = model(**inputs, output_router_logits=True)
    
    if hasattr(outputs, 'router_logits') and outputs.router_logits is not None:
        logger.info(f"Found {len(outputs.router_logits)} MoE layers")
        
        routing_stats = {}
        
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
                ideal_load = 1.0 / num_experts
                load_variance = torch.var(expert_usage)
                
                # Store statistics
                layer_stats = {
                    'num_experts': num_experts,
                    'sequence_length': seq_len,
                    'expert_usage': expert_usage.cpu().numpy(),
                    'avg_entropy': avg_entropy.item(),
                    'load_variance': load_variance.item(),
                    'most_used_expert': torch.argmax(expert_usage).item(),
                    'least_used_expert': torch.argmin(expert_usage).item()
                }
                
                routing_stats[f'layer_{layer_idx}'] = layer_stats
                
                logger.info(f"\nLayer {layer_idx}:")
                logger.info(f"  Experts: {num_experts}, Sequence length: {seq_len}")
                logger.info(f"  Average routing entropy: {avg_entropy:.4f}")
                logger.info(f"  Load balancing variance: {load_variance:.6f}")
                logger.info(f"  Most used expert: {layer_stats['most_used_expert']} ({expert_usage[layer_stats['most_used_expert']]:.3f})")
                logger.info(f"  Least used expert: {layer_stats['least_used_expert']} ({expert_usage[layer_stats['least_used_expert']]:.3f})")
                
                # Show top 3 most used experts
                top_3_experts = torch.topk(expert_usage, 3)
                logger.info(f"  Top 3 experts: {top_3_experts.indices.tolist()} with usage {top_3_experts.values.tolist()}")
        
        return routing_stats
    else:
        logger.info("No router logits found in model output")
        return {}

def compare_routing_patterns(stats_en, stats_heb):
    """Compare routing patterns between English and Hebrew"""
    logger.info(f"\n{'='*60}")
    logger.info("ROUTING PATTERN COMPARISON")
    logger.info(f"{'='*60}")
    
    common_layers = set(stats_en.keys()) & set(stats_heb.keys())
    
    # Summary statistics
    en_entropies = [stats_en[layer]['avg_entropy'] for layer in common_layers]
    heb_entropies = [stats_heb[layer]['avg_entropy'] for layer in common_layers]
    en_variances = [stats_en[layer]['load_variance'] for layer in common_layers]
    heb_variances = [stats_heb[layer]['load_variance'] for layer in common_layers]
    
    logger.info(f"\nSUMMARY STATISTICS:")
    logger.info(f"Average Entropy - EN: {np.mean(en_entropies):.4f}, HEB: {np.mean(heb_entropies):.4f}")
    logger.info(f"Average Load Variance - EN: {np.mean(en_variances):.6f}, HEB: {np.mean(heb_variances):.6f}")
    
    similarities = []
    
    for layer in sorted(common_layers):
        en_stats = stats_en[layer]
        heb_stats = stats_heb[layer]
        
        logger.info(f"\n{layer.upper()}:")
        logger.info(f"  Entropy - EN: {en_stats['avg_entropy']:.4f}, HEB: {heb_stats['avg_entropy']:.4f}")
        logger.info(f"  Load variance - EN: {en_stats['load_variance']:.6f}, HEB: {heb_stats['load_variance']:.6f}")
        logger.info(f"  Most used expert - EN: {en_stats['most_used_expert']}, HEB: {heb_stats['most_used_expert']}")
        
        # Calculate routing overlap
        en_usage = en_stats['expert_usage']
        heb_usage = heb_stats['expert_usage']
        
        # Cosine similarity between expert usage patterns
        cos_sim = np.dot(en_usage, heb_usage) / (np.linalg.norm(en_usage) * np.linalg.norm(heb_usage))
        logger.info(f"  Expert usage similarity: {cos_sim:.4f}")
        similarities.append(cos_sim)
    
    logger.info(f"\nOVERALL SIMILARITY STATISTICS:")
    logger.info(f"Mean similarity across layers: {np.mean(similarities):.4f}")
    logger.info(f"Min similarity: {np.min(similarities):.4f}")
    logger.info(f"Max similarity: {np.max(similarities):.4f}")
    logger.info(f"Std similarity: {np.std(similarities):.4f}")

# Test prompts
prompts = {
    "English": "Give me a short introduction to large language model.",
    "Hebrew": "תן לי הקדמה קצרה למודל שפה גדול."
}

all_stats = {}

for lang, prompt in prompts.items():
    print(f"Processing {lang} prompt...")
    logger.info(f"\n{'='*50}")
    logger.info(f"Processing {lang}: {prompt}")
    
    # Prepare input
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    
    logger.info(f"Input tokens: {model_inputs.input_ids.shape[1]}")
    
    # Analyze routing
    stats = analyze_routing_patterns(model, model_inputs, lang)
    all_stats[lang] = stats

# Compare patterns
if 'English' in all_stats and 'Hebrew' in all_stats:
    print("Comparing routing patterns...")
    compare_routing_patterns(all_stats['English'], all_stats['Hebrew'])

logger.info(f"\n{'='*60}")
logger.info("=== MoE Analysis Complete ===")
print(f"Analysis complete! Results saved to: {log_filename}")
