# cleaners/llm_cleaner.py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from text_cleaning.cleaners.base_cleaner import BaseCleaner
import wandb
import psutil
import GPUtil
from datetime import datetime

class LLMCleaner(BaseCleaner):
    def __init__(self, model_id, processor, few_shot_prompt: list, raw_text: str, torch_dtype=torch.float16, wandb_project="text-cleaning"):
        self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch_dtype)
        self.tokenizer = processor
        self.few_shot_prompt = few_shot_prompt
        self.raw_text = raw_text
        
        # Initialize wandb
        self.wandb_project = wandb_project
        self.run_name = f"cleaning-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        wandb.init(project=self.wandb_project, name=self.run_name)
        
        # Log model configuration
        wandb.config.update({
            "model_id": model_id,
            "torch_dtype": str(torch_dtype),
            "device": str(self.model.device)
        })

    def _log_metrics(self, input_length, output_length, processing_time):
        """Log metrics to wandb including GPU usage"""
        metrics = {
            "input_length": input_length,
            "output_length": output_length,
            "processing_time": processing_time,
            "compression_ratio": output_length / input_length if input_length > 0 else 0
        }
        
        # Log GPU metrics if available
        try:
            gpus = GPUtil.getGPUs()
            for i, gpu in enumerate(gpus):
                metrics.update({
                    f"gpu_{i}_memory_used": gpu.memoryUsed,
                    f"gpu_{i}_memory_total": gpu.memoryTotal,
                    f"gpu_{i}_utilization": gpu.load * 100
                })
        except Exception as e:
            print(f"Could not log GPU metrics: {e}")
        
        # Log CPU metrics
        metrics.update({
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent
        })
        
        wandb.log(metrics)

    def clean(self, raw_text: str) -> str:
        import time
        start_time = time.time()
        
        full_chat = self.few_shot_prompt + [{"role": "user", "content": f"טקסט: {raw_text}"}]
        prompt_input = self.tokenizer.apply_chat_template(full_chat, add_generation_prompt=True, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(**prompt_input, max_new_tokens=200)
            response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        cleaned_text = self.extract_response(response)
        
        # Log metrics
        processing_time = time.time() - start_time
        self._log_metrics(
            input_length=len(raw_text),
            output_length=len(cleaned_text),
            processing_time=processing_time
        )
        
        return cleaned_text

    def extract_response(self, text) -> str:
        return text.split("assistant")[1].strip() if "assistant" in text else text

    def __del__(self):
        """Clean up wandb run when the cleaner is destroyed"""
        if wandb.run is not None:
            wandb.finish()