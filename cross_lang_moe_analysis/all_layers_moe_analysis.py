import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import pandas as pd
import logging
from datetime import datetime
import os

# Data to use
# data_File = '/home/sagemaker-user/qwen-hebrew-finetuning/cross_lang_moe_analysis/moe_analysis_data/arc_en_he_500.csv'
# data_File = '/home/sagemaker-user/qwen-hebrew-finetuning/cross_lang_moe_analysis/moe_analysis_data/gsm_en_he_500.csv'
# data_File = '/home/sagemaker-user/qwen-hebrew-finetuning/cross_lang_moe_analysis/moe_analysis_data/mmlu_en_he_500.csv'
# data_File = '/home/sagemaker-user/qwen-hebrew-finetuning/cross_lang_moe_analysis/moe_analysis_data/copa_en_he_500.csv'
# data_File = '/home/sagemaker-user/qwen-hebrew-finetuning/cross_lang_moe_analysis/moe_analysis_data/nli_en_he_460.csv'
data_File = '/home/sagemaker-user/qwen-hebrew-finetuning/cross_lang_moe_analysis/moe_analysis_data/ted_he_en_chunks.csv'
num_of_samples = 500  # (Eng, Heb) pairess

add_to_logger_name = data_File.split('/')[-1].replace('.csv', '')

# Setup logging to file only
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
log_filename = f"log_files/moe_all_layers_analysis_{add_to_logger_name}_{timestamp}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename)
    ]
)

logger = logging.getLogger(__name__)

model_name = "Qwen/Qwen3-30B-A3B-Base"

print(f"Starting ALL LAYERS MoE analysis... Results will be saved to: {log_filename}")
logger.info("=== ALL LAYERS MoE Analysis Started ===")

# Smaller sample for ALL layers analysis
data_df = dict(pd.read_csv(data_File)[['en_prompt', 'he_prompt']].head(num_of_samples))

sample_prompts = {}
sample_prompts['English'] = list(data_df['en_prompt'])
sample_prompts['Hebrew'] = list(data_df['he_prompt'])

logger.info("Loading model and tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
    output_router_logits=True
)

def analyze_routing_patterns_all_layers(model, inputs, prompt_type, prompt_idx):
    """Analyze routing patterns in ALL MoE layers"""
    logger.info(f"\n=== MoE Routing Analysis for {prompt_type} - Prompt {prompt_idx} ===")
    
    with torch.no_grad():
        outputs = model(**inputs, output_router_logits=True)
    
    if hasattr(outputs, 'router_logits') and outputs.router_logits is not None:
        logger.info(f"Analyzing ALL {len(outputs.router_logits)} MoE layers")
        routing_stats = {}
        layer_probs_for_csv = {}
        layer_activations_for_csv = {} # NEW: To store 1-0 activation vectors

        for layer_idx, router_logits in enumerate(outputs.router_logits):
            if router_logits is not None:
                if router_logits.dim() == 3 and router_logits.shape[0] == 1:
                    router_logits = router_logits.squeeze(0)

                router_probs = torch.softmax(router_logits, dim=-1)
                layer_probs_for_csv[layer_idx] = router_probs.cpu()
                
                # NEW: Calculate top-8 expert activations (1-0 vector)
                top_k_experts = torch.topk(router_probs, k=8, dim=-1).indices
                activations = torch.zeros_like(router_probs)
                activations.scatter_(-1, top_k_experts, 1)
                layer_activations_for_csv[layer_idx] = activations.cpu() # NEW: Store CPU tensor
                
                avg_router_probs = torch.mean(router_probs, dim=0)
                
                seq_len, num_experts = router_probs.shape[-2:]
                top_experts = torch.argmax(router_probs, dim=-1)
                expert_counts = torch.bincount(top_experts.flatten(), minlength=num_experts)
                expert_usage = expert_counts.float() / top_experts.numel()
                
                entropy = -torch.sum(router_probs * torch.log(router_probs + 1e-10), dim=-1)
                avg_entropy = entropy.mean()
                load_variance = torch.var(expert_usage)
                
                sorted_usage, _ = torch.sort(expert_usage)
                n = len(sorted_usage)
                cumsum = torch.cumsum(sorted_usage, dim=0)
                gini = (n + 1 - 2 * torch.sum(cumsum) / torch.sum(sorted_usage)) / n
                
                layer_stats = {
                    'avg_entropy': avg_entropy.item(),
                    'load_variance': load_variance.item(),
                    'gini_coefficient': gini.item(),
                    'most_used_expert': torch.argmax(expert_usage).item(),
                    'max_usage': expert_usage.max().item(),
                    'num_active_experts': (expert_usage > 0.001).sum().item(),
                    'top_3_experts': torch.topk(expert_usage, 3).indices.tolist(),
                    'top_8_experts': torch.topk(expert_usage, 8).indices.tolist(),
                    'avg_router_distribution': avg_router_probs
                }
                
                routing_stats[f'layer_{layer_idx}'] = layer_stats
                logger.info(f"Layer {layer_idx:2d}: Entropy={avg_entropy:.4f}, Variance={load_variance:.6f}, Gini={gini:.4f}, MaxUsage={expert_usage.max().item():.3f}, ActiveExperts={layer_stats['num_active_experts']}")
        
        return routing_stats, layer_probs_for_csv, layer_activations_for_csv # MODIFIED: Return activations as well
    else:
        logger.info("No router logits found in model output")
        return {}, {}, {} # MODIFIED: Return empty dict for activations

# --- Main Analysis Execution ---
all_results = {"English": {}, "Hebrew": {}}
# Dictionary to aggregate token-level distributions for CSV export
lang_distributions = {
    "English": {i: [] for i in range(48)},
    "Hebrew": {i: [] for i in range(48)}
}
# NEW: Dictionary to aggregate token-level activations for CSV export
lang_activations = {
    "English": {i: [] for i in range(48)},
    "Hebrew": {i: [] for i in range(48)}
}

for lang, prompts in sample_prompts.items():
    print(f"Processing {lang} prompts (all layers)...")
    all_results[lang] = {}
    
    for idx, prompt in enumerate(prompts):
        print(f"  Prompt {idx+1}/{len(prompts)}: {prompt[:50]}...")
        logger.info(f"\nPrompt {idx+1}: {prompt}")
        
        messages = [{"role": "user", "content": prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
        
        logger.info(f"Input tokens: {model_inputs.input_ids.shape[1]}")
        
        # MODIFIED: Unpack three return values
        stats, layer_probs, layer_activations = analyze_routing_patterns_all_layers(model, model_inputs, lang, idx+1)
        all_results[lang][idx+1] = stats

        # Aggregate raw probabilities for later averaging
        for layer_idx, probs_tensor in layer_probs.items():
            lang_distributions[lang][layer_idx].append(probs_tensor)
            
        # NEW: Aggregate raw activations for later averaging
        for layer_idx, activation_tensor in layer_activations.items():
            lang_activations[lang][layer_idx].append(activation_tensor)

# --- Comparative Layer Progression Analysis ---
# (This section remains unchanged)
print("Analyzing layer progression...")
logger.info(f"\n{'='*80}")
logger.info("LAYER PROGRESSION ANALYSIS")
logger.info(f"{'='*80}")

for prompt_idx in range(1, len(sample_prompts["English"]) + 1):
    if prompt_idx in all_results["English"] and prompt_idx in all_results["Hebrew"]:
        logger.info(f"\nPROMPT {prompt_idx} LAYER PROGRESSION:")
        logger.info(f"EN:  {sample_prompts['English'][prompt_idx-1]}")
        logger.info(f"HEB: {sample_prompts['Hebrew'][prompt_idx-1]}")
        logger.info(f"{'Layer':<5} {'EN_Entropy':<10} {'HEB_Entropy':<11} {'EN_Variance':<12} {'HEB_Variance':<13} {'Overlap_3':<10} {'Overlap_8':<10} {'Cosine_Dist':<12}")
        logger.info("-" * 98)
        
        en_stats = all_results["English"][prompt_idx]
        heb_stats = all_results["Hebrew"][prompt_idx]
        
        for layer_num in range(48):
            layer_key = f'layer_{layer_num}'
            if layer_key in en_stats and layer_key in heb_stats:
                en_layer = en_stats[layer_key]
                heb_layer = heb_stats[layer_key]
                
                en_top_3 = set(en_layer['top_3_experts'])
                heb_top_3 = set(heb_layer['top_3_experts'])
                overlap_3 = len(en_top_3 & heb_top_3) / len(en_top_3 | heb_top_3) if (en_top_3 | heb_top_3) else 0
                
                en_top_8 = set(en_layer['top_8_experts'])
                heb_top_8 = set(heb_layer['top_8_experts'])
                overlap_8 = len(en_top_8 & heb_top_8) / len(en_top_8 | heb_top_8) if (en_top_8 | heb_top_8) else 0
                
                en_dist_vec = en_layer['avg_router_distribution']
                heb_dist_vec = heb_layer['avg_router_distribution']
                cosine_dist = 1.0 - torch.nn.functional.cosine_similarity(en_dist_vec, heb_dist_vec, dim=0).item()
                
                logger.info(f"{layer_num:<5} {en_layer['avg_entropy']:<10.4f} {heb_layer['avg_entropy']:<11.4f} "
                           f"{en_layer['load_variance']:<12.6f} {heb_layer['load_variance']:<13.6f} {overlap_3:<10.3f} {overlap_8:<10.3f} "
                           f"{cosine_dist:<12.4f}")

# --- Aggregating and Saving Average Distributions to CSV ---
print("Aggregating and saving average distributions to CSV...")
logger.info(f"\n{'='*80}")
logger.info("AGGREGATED AVERAGE DISTRIBUTIONS")
logger.info(f"{'='*80}")

csv_output_folder = "moe_csv_results"
os.makedirs(csv_output_folder, exist_ok=True)
csv_filename = os.path.join(csv_output_folder, f"moe_all_layers_analysis_{add_to_logger_name}_{timestamp}_avg_dist.csv")

csv_data = []
num_experts = 128 # Assuming 128 experts from the model architecture

for lang in ["English", "Hebrew"]:
    for layer_idx in range(48):
        all_token_probs = lang_distributions[lang][layer_idx]
        if not all_token_probs:
            continue
            
        concatenated_probs = torch.cat(all_token_probs, dim=0)
        final_avg_dist = torch.mean(concatenated_probs, dim=0)
        
        row = {'Layer': layer_idx, 'Language': lang}
        for i in range(num_experts):
            row[f'Expert_{i}'] = final_avg_dist[i].item()
        csv_data.append(row)

df = pd.DataFrame(csv_data)
df.to_csv(csv_filename, index=False)

logger.info(f"Saved aggregated average distributions to {csv_filename}")

# --- NEW: Aggregating and Saving Average ACTIVATIONS to CSV ---
print("Aggregating and saving average activations to CSV...")
logger.info(f"\n{'='*80}")
logger.info("AGGREGATED AVERAGE ACTIVATIONS (TOP-8)")
logger.info(f"{'='*80}")

csv_activation_filename = os.path.join(csv_output_folder, f"moe_all_layers_analysis_{add_to_logger_name}_{timestamp}_avg_activation.csv")

csv_activation_data = []
# num_experts is already defined above

for lang in ["English", "Hebrew"]:
    for layer_idx in range(48):
        # Concatenate all token-level activation vectors from all prompts for this layer
        all_token_activations = lang_activations[lang][layer_idx]
        if not all_token_activations:
            continue
            
        concatenated_activations = torch.cat(all_token_activations, dim=0)
        
        # Calculate the final average activation across all tokens
        final_avg_activation = torch.mean(concatenated_activations, dim=0)
        
        # Prepare data for DataFrame
        row = {'Layer': layer_idx, 'Language': lang}
        for i in range(num_experts):
            row[f'Expert_{i}'] = final_avg_activation[i].item()
        csv_activation_data.append(row)

# Create and save the DataFrame for activations
df_activation = pd.DataFrame(csv_activation_data)
df_activation.to_csv(csv_activation_filename, index=False)

logger.info(f"Saved aggregated average activations to {csv_activation_filename}")
logger.info(f"\n{'='*80}")
logger.info("=== ALL LAYERS MoE Analysis Complete ===")

print(f"All layers analysis complete! Log saved to: {log_filename}")
print(f"Aggregated distributions saved to: {csv_filename}")
print(f"Aggregated activations saved to: {csv_activation_filename}") # NEW: Print new filename