#!/usr/bin/env python3

import sys
import os
import json
sys.path.append('/home/ec2-user/qwen-hebrew-finetuning/qwen_model_shaltiel')

from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling
import torch

def create_tokenize_function(tokenizer, max_seq_length):
    def tokenize_function(examples):
        # Tokenize the texts with truncation but no padding here
        # Let the DataCollator handle padding during batch creation
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_seq_length,
            # Remove return_tensors="pt" - let DataCollator handle tensor conversion
        )
    return tokenize_function

def test_tokenization():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B-Base", trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    print("Loading small sample from BIU dataset...")
    dataset = load_dataset('json', data_files='datasets/BIU.jsonl', split="train")
    sample_dataset = dataset.select(range(min(10, len(dataset))))
    
    print(f"Sample dataset: {sample_dataset}")
    print(f"Sample dataset columns: {sample_dataset.column_names}")
    print(f"First item before tokenization: {sample_dataset[0]}")
    
    print("Tokenizing dataset...")
    tokenized_dataset = sample_dataset.map(
        create_tokenize_function(tokenizer, 2048),
        batched=True,
        num_proc=1,
        remove_columns=["text", "id", "metadata"]  # Remove all non-tokenization columns
    )
    
    print(f"Tokenized dataset: {tokenized_dataset}")
    print(f"Tokenized dataset features: {tokenized_dataset.features}")
    print(f"First tokenized item: {tokenized_dataset[0]}")
    
    print("Testing DataCollator...")
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
        pad_to_multiple_of=8
    )
    
    # Test with a small batch
    batch = [tokenized_dataset[i] for i in range(min(3, len(tokenized_dataset)))]
    print(f"Batch before collating: {batch}")
    
    try:
        collated_batch = data_collator(batch)
        print("✅ DataCollator succeeded!")
        print(f"Collated batch keys: {collated_batch.keys()}")
        print(f"Input IDs shape: {collated_batch['input_ids'].shape}")
        print(f"Attention mask shape: {collated_batch['attention_mask'].shape}")
        print("✅ Tokenization fix successful!")
        return True
    except Exception as e:
        print(f"❌ DataCollator failed: {e}")
        return False

if __name__ == "__main__":
    test_tokenization()
