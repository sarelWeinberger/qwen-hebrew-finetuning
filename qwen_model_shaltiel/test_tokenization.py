#!/usr/bin/env python3
"""
Quick test to verify tokenization is working correctly
"""
import json
from datasets import load_dataset
from transformers import AutoTokenizer, DataCollatorForLanguageModeling

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
    print("Testing tokenization with BIU dataset...")
    
    # Load a small sample
    dataset = load_dataset('json', data_files='datasets/BIU.jsonl', split="train")
    print(f"Dataset loaded: {len(dataset)} samples")
    
    # Take only first 10 samples for testing
    small_dataset = dataset.select(range(min(10, len(dataset))))
    print(f"Using {len(small_dataset)} samples for testing")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B-Base")
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Tokenize
    print("Tokenizing...")
    
    # Remove all original columns, keep only tokenizer outputs
    columns_to_remove = small_dataset.column_names
    print(f"Removing columns: {columns_to_remove}")
    
    tokenized_dataset = small_dataset.map(
        create_tokenize_function(tokenizer, 2048),
        batched=True,
        remove_columns=columns_to_remove
    )
    
    print(f"Tokenized dataset: {tokenized_dataset}")
    print(f"Features: {tokenized_dataset.features}")
    
    if len(tokenized_dataset) > 0:
        sample = tokenized_dataset[0]
        print(f"Sample keys: {sample.keys()}")
        print(f"Input IDs length: {len(sample['input_ids'])}")
        print(f"Attention mask length: {len(sample['attention_mask'])}")
    
    # Test data collator
    print("\nTesting data collator...")
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, 
        mlm=False,
        pad_to_multiple_of=8,
        return_tensors="pt"
    )
    
    # Create a small batch
    batch_samples = [tokenized_dataset[i] for i in range(min(3, len(tokenized_dataset)))]
    print(f"Creating batch from {len(batch_samples)} samples")
    
    try:
        batch = data_collator(batch_samples)
        print("✅ Data collator successful!")
        print(f"Batch keys: {batch.keys()}")
        if 'input_ids' in batch:
            print(f"Batch input_ids shape: {batch['input_ids'].shape}")
        if 'attention_mask' in batch:
            print(f"Batch attention_mask shape: {batch['attention_mask'].shape}")
        if 'labels' in batch:
            print(f"Batch labels shape: {batch['labels'].shape}")
    except Exception as e:
        print(f"❌ Data collator failed: {e}")
        return False
    
    print("\n✅ All tokenization tests passed!")
    return True

if __name__ == "__main__":
    test_tokenization()
