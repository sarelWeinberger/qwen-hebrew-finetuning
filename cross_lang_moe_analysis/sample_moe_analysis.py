import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import logging
from datetime import datetime
import json

# clean the GPU cache
torch.cuda.empty_cache()

# Setup logging to file only
log_filename = f"moe_sample_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename)
    ]
)

logger = logging.getLogger(__name__)

model_name = "Qwen/Qwen3-30B-A3B-Base"

print(f"Starting sample MoE analysis... Results will be saved to: {log_filename}")
logger.info("=== Sample MoE Analysis Started ===")
logger.info("Loading model and tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    output_router_logits=True
)

# Sample test prompts - Diverse but manageable set
sample_prompts = {
    "English": [
        "Explain how neural networks work in machine learning.",
        "Write a short story about a robot learning to paint.", 
        "Analyze the pros and cons of remote work.",
        "How was your day today?",
        "Explain quantum physics to a 10-year-old.",
        "What is artificial intelligence and how does it work?",
        "Describe the process of photosynthesis in plants.",
        "Write a poem about the ocean.",
        "Compare different programming languages.",
        "What are the benefits of exercise for health?"
    ],
    "Hebrew": [
        "הסבר איך רשתות נוירליות עובדות בלמידת מכונה.",
        "כתוב סיפור קצר על רובוט הלומד לצייר.",
        "נתח את היתרונות והחסרונות של עבודה מרחוק.",
        "איך היה לך היום?", 
        "הסבר פיזיקה קוונטית לילד בן 10.",
        "מה זה בינה מלאכותית ואיך היא עובדת?",
        "תאר את התהליך של פוטוסינתזה בצמחים.",
        "כתוב שיר על הים.",
        "השווה בין שפות תכנות שונות.",
        "מה היתרונות של פעילות גופנית לבריאות?"
    ]
}

def analyze_routing_patterns(model, inputs, prompt_type, prompt_idx):
    """Analyze routing patterns in MoE layers"""
    logger.info(f"\n=== MoE Routing Analysis for {prompt_type} - Prompt {prompt_idx} ===")
    
    with torch.no_grad():
        outputs = model(**inputs, output_router_logits=True)
    
    if hasattr(outputs, 'router_logits') and outputs.router_logits is not None:
        routing_stats = {}
        
        # Focus on key layers for efficiency
        key_layers = [0, 5, 10, 15, 20, 25, 30, 35, 40, 45, 47]  # Sample across depth
        
        for layer_idx, router_logits in enumerate(outputs.router_logits):
            if router_logits is not None and layer_idx in key_layers:
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
                    'num_active_experts': (expert_usage > 0.001).sum().item(),  # Experts with >0.1% usage
                    'top_3_experts': torch.topk(expert_usage, 3).indices.tolist(),
                    'top_3_usage': torch.topk(expert_usage, 3).values.tolist()
                }
                
                routing_stats[f'layer_{layer_idx}'] = layer_stats
                
                logger.info(f"Layer {layer_idx}: Entropy={avg_entropy:.4f}, Variance={load_variance:.6f}, Gini={gini:.4f}")
        
        return routing_stats
    else:
        logger.info("No router logits found in model output")
        return {}

def calculate_layer_similarity(en_stats, heb_stats, layer):
    """Calculate similarity between English and Hebrew for a specific layer"""
    if layer not in en_stats or layer not in heb_stats:
        return None
    
    en_layer = en_stats[layer]
    heb_layer = heb_stats[layer]
    
    # Compare top experts overlap
    en_top = set(en_layer['top_3_experts'])
    heb_top = set(heb_layer['top_3_experts'])
    overlap = len(en_top & heb_top) / len(en_top | heb_top)
    
    return {
        'expert_overlap': overlap,
        'entropy_diff': abs(en_layer['avg_entropy'] - heb_layer['avg_entropy']),
        'variance_diff': abs(en_layer['load_variance'] - heb_layer['load_variance']),
        'gini_diff': abs(en_layer['gini_coefficient'] - heb_layer['gini_coefficient'])
    }

# Run analysis
all_results = {"English": {}, "Hebrew": {}}

for lang, prompts in sample_prompts.items():
    print(f"Processing {lang} prompts...")
    all_results[lang] = {}
    
    for idx, prompt in enumerate(prompts):
        print(f"  Prompt {idx+1}/{len(prompts)}: {prompt[:50]}...")
        logger.info(f"\nPrompt {idx+1}: {prompt}")
        
        # Prepare input
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        logger.info(f"Input tokens: {model_inputs.input_ids.shape[1]}")
        
        # Analyze routing
        stats = analyze_routing_patterns(model, model_inputs, lang, idx+1)
        all_results[lang][idx+1] = stats

# Comprehensive comparison
print("Analyzing cross-language patterns...")
logger.info(f"\n{'='*80}")
logger.info("CROSS-LANGUAGE ROUTING ANALYSIS")
logger.info(f"{'='*80}")

# Compare each English prompt with corresponding Hebrew prompt
similarities_by_prompt = {}
for prompt_idx in range(1, len(sample_prompts["English"]) + 1):
    if prompt_idx in all_results["English"] and prompt_idx in all_results["Hebrew"]:
        en_stats = all_results["English"][prompt_idx]
        heb_stats = all_results["Hebrew"][prompt_idx]
        
        logger.info(f"\nPROMPT {prompt_idx} COMPARISON:")
        logger.info(f"EN:  {sample_prompts['English'][prompt_idx-1]}")
        logger.info(f"HEB: {sample_prompts['Hebrew'][prompt_idx-1]}")
        
        prompt_similarities = {}
        for layer in en_stats.keys():
            similarity = calculate_layer_similarity(en_stats, heb_stats, layer)
            if similarity:
                prompt_similarities[layer] = similarity
                logger.info(f"{layer}: Overlap={similarity['expert_overlap']:.3f}, "
                           f"ΔEntropy={similarity['entropy_diff']:.4f}, "
                           f"ΔVariance={similarity['variance_diff']:.6f}")
        
        similarities_by_prompt[prompt_idx] = prompt_similarities

# Aggregate statistics
logger.info(f"\n{'='*80}")
logger.info("AGGREGATE STATISTICS")
logger.info(f"{'='*80}")

# Calculate overall statistics
all_overlaps = []
all_entropy_diffs = []
all_variance_diffs = []
all_gini_diffs = []

for prompt_similarities in similarities_by_prompt.values():
    for layer_sim in prompt_similarities.values():
        all_overlaps.append(layer_sim['expert_overlap'])
        all_entropy_diffs.append(layer_sim['entropy_diff'])
        all_variance_diffs.append(layer_sim['variance_diff'])
        all_gini_diffs.append(layer_sim['gini_diff'])

logger.info(f"\nOVERALL LANGUAGE SIMILARITY:")
logger.info(f"Mean Expert Overlap: {np.mean(all_overlaps):.4f} (±{np.std(all_overlaps):.4f})")
logger.info(f"Mean Entropy Difference: {np.mean(all_entropy_diffs):.4f} (±{np.std(all_entropy_diffs):.4f})")
logger.info(f"Mean Variance Difference: {np.mean(all_variance_diffs):.6f} (±{np.std(all_variance_diffs):.6f})")
logger.info(f"Mean Gini Difference: {np.mean(all_gini_diffs):.4f} (±{np.std(all_gini_diffs):.4f})")

# Layer-wise analysis
logger.info(f"\nLAYER-WISE PATTERNS:")
layer_stats = {}
for layer in ["layer_0", "layer_10", "layer_20", "layer_30", "layer_40", "layer_47"]:
    layer_overlaps = []
    layer_entropy_diffs = []
    
    for prompt_similarities in similarities_by_prompt.values():
        if layer in prompt_similarities:
            layer_overlaps.append(prompt_similarities[layer]['expert_overlap'])
            layer_entropy_diffs.append(prompt_similarities[layer]['entropy_diff'])
    
    if layer_overlaps:
        logger.info(f"{layer}: Overlap={np.mean(layer_overlaps):.4f}, ΔEntropy={np.mean(layer_entropy_diffs):.4f}")

logger.info(f"\n{'='*80}")
logger.info("=== Sample MoE Analysis Complete ===")

print(f"Sample analysis complete! Results saved to: {log_filename}")
