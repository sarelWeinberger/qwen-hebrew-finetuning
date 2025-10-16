# cleaners/llm_cleaner.py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from .base_cleaner import BaseCleaner
import wandb
import psutil
import GPUtil
from datetime import datetime
from utils.logger import logger
import pandas as pd
import time
import json
import os
from typing import Dict, Any, List

class LLMCleaner(BaseCleaner):
    def __init__(self, model_id, processor, few_shot_prompt: list,
                raw_text: str,
                torch_dtype=torch.float16,
                wandb_project="text-cleaning",
                save_samples: bool = True,
                sample_percentage: float = 0.05):
        super().__init__(save_samples=save_samples, sample_percentage=sample_percentage)
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
        
        logger.info(f"Initialized LLMCleaner")

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
            logger.warning(f"Could not log GPU metrics: {e}")
        
        # Log CPU metrics
        metrics.update({
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent
        })
        
        wandb.log(metrics)
        
        # Update base cleaner stats
        self.stats['total_rows_processed'] += 1
        self.stats['characters_removed'] += max(0, input_length - output_length)
        self.stats['characters_added'] += max(0, output_length - input_length)
        self.stats['execution_time'] += processing_time
        
        if input_length != output_length:
            self.stats['rows_modified'] += 1

    def _clean_implementation(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the DataFrame using LLM processing.
        
        Args:
            df: Input DataFrame with 'text' column
            
        Returns:
            Cleaned DataFrame with 'text' and 'n_words' columns
        """
        cleaned_texts = []
        n_words = []
        
        for idx, row in df.iterrows():
            raw_text = row['text']
            
            try:
                cleaned_text = self._clean_single_text(raw_text)
                cleaned_texts.append(cleaned_text)
                n_words.append(len(cleaned_text.split()))
                    
            except Exception as e:
                logger.error(f"Error processing row {idx}: {str(e)}")
                # Keep original text if cleaning fails
                cleaned_texts.append(raw_text)
                n_words.append(len(raw_text.split()))
        
        # Create result DataFrame
        result_df = pd.DataFrame({
            'text': cleaned_texts,
            'n_words': n_words
        })
        
        return result_df

    def _clean_single_text(self, raw_text: str) -> str:
        """
        Clean a single text using LLM.
        
        Args:
            raw_text: Input text to clean
            
        Returns:
            Cleaned text
        """
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