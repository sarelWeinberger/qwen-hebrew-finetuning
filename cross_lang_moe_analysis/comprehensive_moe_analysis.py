import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np
import logging
from datetime import datetime
import json

# Setup logging to file only
log_filename = f"moe_comprehensive_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(message)s',
    handlers=[
        logging.FileHandler(log_filename)
    ]
)

logger = logging.getLogger(__name__)

model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"

print(f"Starting comprehensive MoE analysis... Results will be saved to: {log_filename}")
logger.info("=== Comprehensive MoE Analysis Started ===")
logger.info("Loading model and tokenizer...")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="auto",
    device_map="auto",
    output_router_logits=True
)

# Comprehensive test prompts - Multiple categories
test_prompts = {
    "English": {
        "tech": [
            "Explain how neural networks work in machine learning.",
            "What is the difference between supervised and unsupervised learning?",
            "Describe the architecture of a transformer model.",
            "How does gradient descent optimization work?",
            "What are the advantages of using GPUs for deep learning?"
        ],
        "creative": [
            "Write a short story about a robot learning to paint.",
            "Create a poem about the beauty of mathematics.",
            "Describe a futuristic city where humans and AI coexist.",
            "Write a dialogue between two AI systems discussing consciousness.",
            "Compose a haiku about machine learning."
        ],
        "analytical": [
            "Analyze the pros and cons of remote work.",
            "Compare renewable energy sources and their efficiency.",
            "Evaluate the impact of social media on society.",
            "Discuss the ethical implications of artificial intelligence.",
            "Examine the causes and effects of climate change."
        ],
        "conversational": [
            "How was your day today?",
            "What's your favorite book and why?",
            "Can you recommend a good restaurant?",
            "Tell me a joke about programmers.",
            "What's the weather like in your opinion?"
        ],
        "educational": [
            "Explain quantum physics to a 10-year-old.",
            "Teach me the basics of programming in Python.",
            "How do vaccines work in the human body?",
            "What causes earthquakes and how are they measured?",
            "Explain the water cycle and its importance."
        ]
    },
    "Hebrew": {
        "tech": [
            "הסבר איך רשתות נוירליות עובדות בלמידת מכונה.",
            "מה ההבדל בין למידה מונחית ללמידה לא מונחית?",
            "תאר את הארכיטקטורה של מודל טרנספורמר.",
            "איך עובד אופטימיזציה של ירידה בגרדיאנט?",
            "מה היתרונות של שימוש ב-GPU ללמידה עמוקה?"
        ],
        "creative": [
            "כתוב סיפור קצר על רובוט הלומד לצייר.",
            "צור שיר על היופי של מתמטיקה.",
            "תאר עיר עתידנית שבה בני אדם ובינה מלאכותית חיים יחד.",
            "כתוב דיאלוג בין שני מערכות בינה מלאכותית הדנות בתודעה.",
            "חבר האיקו על למידת מכונה."
        ],
        "analytical": [
            "נתח את היתרונות והחסרונות של עבודה מרחוק.",
            "השווה בין מקורות אנרגיה מתחדשת והיעילות שלהם.",
            "העריך את השפעת הרשתות החברתיות על החברה.",
            "דון בהשלכות האתיות של בינה מלאכותית.",
            "בחן את הסיבות וההשפעות של שינויי האקלים."
        ],
        "conversational": [
            "איך היה לך היום?",
            "איזה ספר הכי אהוב עליך ולמה?",
            "אתה יכול להמליץ על מסעדה טובה?",
            "ספר לי בדיחה על מתכנתים.",
            "איך מזג האוויר לדעתך?"
        ],
        "educational": [
            "הסבר פיזיקה קוונטית לילד בן 10.",
            "למד אותי את היסודות של תכנות בפייתון.",
            "איך חיסונים עובדים בגוף האדם?",
            "מה גורם לרעידות אדמה ואיך מודדים אותן?",
            "הסבר את מחזור המים וחשיבותו."
        ]
    }
}

def analyze_routing_patterns(model, inputs, prompt_type, category, prompt_idx):
    """Analyze routing patterns in MoE layers"""
    logger.info(f"\n=== MoE Routing Analysis for {prompt_type} - {category} - Prompt {prompt_idx} ===")
    
    with torch.no_grad():
        outputs = model(**inputs, output_router_logits=True)
    
    if hasattr(outputs, 'router_logits') and outputs.router_logits is not None:
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
                load_variance = torch.var(expert_usage)
                
                # Store statistics
                layer_stats = {
                    'num_experts': num_experts,
                    'sequence_length': seq_len,
                    'expert_usage': expert_usage.cpu().numpy(),
                    'avg_entropy': avg_entropy.item(),
                    'load_variance': load_variance.item(),
                    'most_used_expert': torch.argmax(expert_usage).item(),
                    'least_used_expert': torch.argmin(expert_usage).item(),
                    'top_k_usage': torch.topk(expert_usage, min(5, num_experts)).values.cpu().numpy().tolist()
                }
                
                routing_stats[f'layer_{layer_idx}'] = layer_stats
        
        return routing_stats
    else:
        logger.info("No router logits found in model output")
        return {}

def aggregate_stats_by_category(all_results):
    """Aggregate statistics by language and category"""
    logger.info(f"\n{'='*80}")
    logger.info("CATEGORY-WISE ANALYSIS")
    logger.info(f"{'='*80}")
    
    for lang in ["English", "Hebrew"]:
        logger.info(f"\n{lang.upper()} LANGUAGE ANALYSIS:")
        logger.info("-" * 40)
        
        for category in all_results[lang]:
            category_entropies = []
            category_variances = []
            category_similarities = []
            
            logger.info(f"\n{category.upper()} Category:")
            
            for prompt_idx, stats in all_results[lang][category].items():
                if stats:  # If we have routing stats
                    entropies = [layer_data['avg_entropy'] for layer_data in stats.values()]
                    variances = [layer_data['load_variance'] for layer_data in stats.values()]
                    
                    category_entropies.extend(entropies)
                    category_variances.extend(variances)
                    
                    logger.info(f"  Prompt {prompt_idx}: Avg Entropy={np.mean(entropies):.4f}, Avg Variance={np.mean(variances):.6f}")
            
            if category_entropies:
                logger.info(f"  Category Summary:")
                logger.info(f"    Mean Entropy: {np.mean(category_entropies):.4f} (±{np.std(category_entropies):.4f})")
                logger.info(f"    Mean Load Variance: {np.mean(category_variances):.6f} (±{np.std(category_variances):.6f})")

def compare_languages_comprehensive(all_results):
    """Comprehensive comparison between languages"""
    logger.info(f"\n{'='*80}")
    logger.info("COMPREHENSIVE LANGUAGE COMPARISON")
    logger.info(f"{'='*80}")
    
    # Compare by category
    for category in test_prompts["English"].keys():
        logger.info(f"\n{category.upper()} CATEGORY COMPARISON:")
        logger.info("-" * 50)
        
        en_entropies = []
        heb_entropies = []
        en_variances = []
        heb_variances = []
        
        # Collect all entropies and variances for this category
        for prompt_idx in all_results["English"][category]:
            if all_results["English"][category][prompt_idx]:
                stats = all_results["English"][category][prompt_idx]
                entropies = [layer_data['avg_entropy'] for layer_data in stats.values()]
                variances = [layer_data['load_variance'] for layer_data in stats.values()]
                en_entropies.extend(entropies)
                en_variances.extend(variances)
        
        for prompt_idx in all_results["Hebrew"][category]:
            if all_results["Hebrew"][category][prompt_idx]:
                stats = all_results["Hebrew"][category][prompt_idx]
                entropies = [layer_data['avg_entropy'] for layer_data in stats.values()]
                variances = [layer_data['load_variance'] for layer_data in stats.values()]
                heb_entropies.extend(entropies)
                heb_variances.extend(variances)
        
        if en_entropies and heb_entropies:
            logger.info(f"Entropy - EN: {np.mean(en_entropies):.4f}±{np.std(en_entropies):.4f}, HEB: {np.mean(heb_entropies):.4f}±{np.std(heb_entropies):.4f}")
            logger.info(f"Variance - EN: {np.mean(en_variances):.6f}±{np.std(en_variances):.6f}, HEB: {np.mean(heb_variances):.6f}±{np.std(heb_variances):.6f}")
            
            # Statistical significance test (simple t-test approximation)
            entropy_diff = abs(np.mean(en_entropies) - np.mean(heb_entropies))
            variance_diff = abs(np.mean(en_variances) - np.mean(heb_variances))
            
            logger.info(f"Entropy difference: {entropy_diff:.4f}")
            logger.info(f"Variance difference: {variance_diff:.6f}")

# Run comprehensive analysis
all_results = {"English": {}, "Hebrew": {}}

total_prompts = sum(len(prompts) for lang_prompts in test_prompts.values() for prompts in lang_prompts.values())
current_prompt = 0

for lang in ["English", "Hebrew"]:
    all_results[lang] = {}
    
    for category, prompts in test_prompts[lang].items():
        print(f"Processing {lang} - {category} category...")
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {lang} - {category.upper()} Category")
        logger.info(f"{'='*60}")
        
        all_results[lang][category] = {}
        
        for idx, prompt in enumerate(prompts):
            current_prompt += 1
            print(f"Progress: {current_prompt}/{total_prompts} - {lang} {category} prompt {idx+1}")
            
            logger.info(f"\nPrompt {idx+1}: {prompt}")
            
            # Prepare input
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
            
            logger.info(f"Input tokens: {model_inputs.input_ids.shape[1]}")
            
            # Analyze routing
            stats = analyze_routing_patterns(model, model_inputs, lang, category, idx+1)
            all_results[lang][category][idx+1] = stats

# Perform comprehensive analysis
print("Analyzing results...")
aggregate_stats_by_category(all_results)
compare_languages_comprehensive(all_results)

# Save results as JSON for further analysis
results_filename = f"moe_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
with open(results_filename, 'w', encoding='utf-8') as f:
    # Convert numpy arrays to lists for JSON serialization
    json_results = {}
    for lang in all_results:
        json_results[lang] = {}
        for category in all_results[lang]:
            json_results[lang][category] = {}
            for prompt_idx in all_results[lang][category]:
                if all_results[lang][category][prompt_idx]:
                    json_results[lang][category][prompt_idx] = {}
                    for layer in all_results[lang][category][prompt_idx]:
                        layer_data = all_results[lang][category][prompt_idx][layer].copy()
                        layer_data['expert_usage'] = layer_data['expert_usage'].tolist()
                        json_results[lang][category][prompt_idx][layer] = layer_data
    
    json.dump(json_results, f, indent=2, ensure_ascii=False)

logger.info(f"\n{'='*80}")
logger.info("=== Comprehensive MoE Analysis Complete ===")
logger.info(f"Log saved to: {log_filename}")
logger.info(f"JSON results saved to: {results_filename}")

print(f"Comprehensive analysis complete!")
print(f"Log file: {log_filename}")
print(f"JSON results: {results_filename}")
