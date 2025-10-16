import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import numpy as np

model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"

# load the tokenizer and the model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype="auto",
    device_map="auto",
    output_router_logits=True  # Enable router logits output for MoE analysis
)

def analyze_moe_metrics(model, inputs, prompt_type):
    """Analyze MoE gate, router, and auxiliary loss metrics"""
    print(f"\n=== Analysis for {prompt_type} prompt ===")
    
    with torch.no_grad():
        outputs = model(**inputs, output_router_logits=True)
    
    # Extract router logits if available (for MoE models)
    if hasattr(outputs, 'router_logits') and outputs.router_logits is not None:
        router_logits = outputs.router_logits
        print(f"Router logits available: {len(router_logits)} layers")
        
        for i, layer_logits in enumerate(router_logits):
            if layer_logits is not None:
                # Analyze gate probabilities
                gate_probs = torch.softmax(layer_logits, dim=-1)
                
                # Calculate entropy (diversity measure)
                entropy = -torch.sum(gate_probs * torch.log(gate_probs + 1e-10), dim=-1)
                
                # Expert utilization
                expert_usage = torch.mean(gate_probs, dim=[0, 1])  # Average usage per expert
                
                print(f"  Layer {i}:")
                print(f"    Gate entropy (avg): {entropy.mean().item():.4f}")
                print(f"    Gate entropy (std): {entropy.std().item():.4f}")
                print(f"    Expert utilization (mean): {expert_usage.mean().item():.4f}")
                print(f"    Expert utilization (std): {expert_usage.std().item():.4f}")
                print(f"    Most used expert: {torch.argmax(expert_usage).item()}")
                print(f"    Least used expert: {torch.argmin(expert_usage).item()}")
                
                # Load balancing loss calculation
                if gate_probs.shape[-1] > 1:  # Only if multiple experts
                    aux_loss = calculate_auxiliary_loss(gate_probs)
                    print(f"    Auxiliary loss: {aux_loss:.6f}")
    else:
        print("No router logits found - this might not be a MoE model or router outputs are disabled")
    
    return outputs

def calculate_auxiliary_loss(gate_probs):
    """Calculate auxiliary loss for load balancing"""
    # Simple auxiliary loss calculation
    num_experts = gate_probs.shape[-1]
    
    # Calculate the fraction of tokens routed to each expert
    fraction_routed = torch.mean(gate_probs, dim=[0, 1])  # [num_experts]
    
    # Calculate the expected fraction (should be 1/num_experts for perfect balance)
    expected_fraction = 1.0 / num_experts
    
    # Auxiliary loss encourages balanced routing
    aux_loss = torch.sum((fraction_routed - expected_fraction) ** 2)
    
    return aux_loss.item()

# Prepare prompts
prompt_en = "Give me a short introduction to large language model."
prompt_heb = "תן לי הקדמה קצרה למודל שפה גדול."

prompts = {
    "English": prompt_en,
    "Hebrew": prompt_heb
}

results = {}

for lang, prompt in prompts.items():
    print(f"\n{'='*50}")
    print(f"Processing {lang} prompt: {prompt}")
    print(f"{'='*50}")
    
    # Prepare messages
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    
    # Tokenize
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)
    print(f"Input tokens: {model_inputs.input_ids.shape[1]}")
    
    # Analyze MoE metrics
    outputs = analyze_moe_metrics(model, model_inputs, lang)
    
    # Generate response
    print(f"\nGenerating response for {lang}...")
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=256,  # Reduced for faster comparison
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id
        # Removed output_router_logits=True to avoid tensor mismatch during generation
    )
    
    output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
    content = tokenizer.decode(output_ids, skip_special_tokens=True)
    
    print(f"\n{lang} Response:")
    print(f"Content: {content}")
    
    results[lang] = {
        'input_length': model_inputs.input_ids.shape[1],
        'output_length': len(output_ids),
        'content': content
    }

print(f"\n{'='*60}")
print("SUMMARY COMPARISON")
print(f"{'='*60}")
for lang, result in results.items():
    print(f"{lang}:")
    print(f"  Input length: {result['input_length']} tokens")
    print(f"  Output length: {result['output_length']} tokens")
    print(f"  Response preview: {result['content'][:100]}...")
    print()