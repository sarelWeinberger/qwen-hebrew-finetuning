# cleaners/llm_cleaner.py
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from text_cleaning.cleaners.base_cleaner import BaseCleaner

class LLMCleaner(BaseCleaner):
    def __init__(self, model_id, processor, few_shot_prompt: list,raw_text: str, torch_dtype=torch.float16):
        self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch_dtype)
        self.tokenizer = processor
        self.few_shot_prompt = few_shot_prompt
        self.raw_text = raw_text

    def clean(self, raw_text: str) -> str:
        full_chat = self.few_shot_prompt + [{"role": "user", "content": f"טקסט: {raw_text}"}]
        prompt_input = self.tokenizer.apply_chat_template(full_chat, add_generation_prompt=True, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output_ids = self.model.generate(**prompt_input, max_new_tokens=200)
            response = self.tokenizer.decode(output_ids[0], skip_special_tokens=True)

        return self.extract_response(response)

    def extract_response(self, text) -> str:
        return text.split("assistant")[1].strip() if "assistant" in text else text