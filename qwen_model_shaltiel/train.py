import os
import json
import argparse
import torch
import wandb
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed,
    TrainerCallback
)
import deepspeed
from accelerate import Accelerator
import numpy as np
from typing import Dict, List, Union
from datasets import load_dataset

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune Qwen3-30B-A3B-Base model")
    parser.add_argument(
        "--config",
        type=str,
        default="cpt_config.json",
        help="Path to the training configuration file"
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        required=False,
        default='he_wiki_sample.jsonl',
        help="Path to the HuggingFace dataset or local JSON/L file (for now)"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for initialization"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="qwen-hebrew-finetuning",
        help="Weights & Biases project name"
    )
    parser.add_argument(
        "--wandb_name",
        type=str,
        default=None,
        help="Weights & Biases run name"
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default="llm_train_mafat",
        help="Weights & Biases entity name"
    )
    return parser.parse_args()

# Copied from QLoRA original code
def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
    print(
        f"trainable params: {trainable_params:,d} || all params: {all_param:,d} || trainable%: {100 * trainable_params / all_param}"
    )


def create_tokenize_function(tokenizer, max_seq_length):
    def tokenize_function(examples):
        # Tokenize the texts with padding to max length
        return tokenizer(
            examples["text"],
            padding="max_length",
            truncation=True,
            max_length=max_seq_length,
            return_tensors="pt"
        )
    return tokenize_function

def train():
    args = parse_args()
    set_seed(args.seed)
    
    # create a training arguments to load in the deepspeed plugin before we create an Accelerator below (which doesn't default to deepspeed)
    _ = TrainingArguments('.')

    # load in the training config
    with open(args.config, 'r', encoding='utf8') as f:
        config = json.loads(f.read())

    # Initialize Weights & Biases - only on the main process
    accelerator = Accelerator()
    if accelerator.is_local_main_process:
        run_name = args.wandb_name or f"qwen-hebrew-{args.seed}"
        wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=run_name,
            config=dict(
                args=args,
                config=config
            )
        )
        print(f"Initialized Weights & Biases run: {run_name}")
    
    # Load tokenizer
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(config["model_name_or_path"])
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Load in the model
    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        config["model_name_or_path"],
        torch_dtype=torch.bfloat16,
        use_cache=False,  # Disable KV cache for training
        device_map=None,  # Automatically mapped using accelerate 
        low_cpu_mem_usage=True  # Unsure whether this matters
    )    

    # Known bug with DeepSpeed freezing on the first forward: https://huggingface.co/posts/stas/984424866637646
    # This is the workaround :)
    from transformers.models.qwen3_moe.modeling_qwen3_moe import Qwen3MoeSparseMoeBlock
    deepspeed.utils.set_z3_leaf_modules(model, [Qwen3MoeSparseMoeBlock])

    # Don't move model to GPU - let DeepSpeed handle device placement
    # Enable gradient checkpointing
    # Disabled - gradient checkpointing is done in the trainer
    # model.gradient_checkpointing_enable()
    
    # Load dataset - either from a local file or from the hug
    print(f"Loading dataset from {args.dataset_path}...")
    if os.path.isfile(args.dataset_path):
        dataset = load_dataset('json', data_files=args.dataset_path, split="train")
    else:
        dataset = load_dataset(args.dataset_path, split="train")
    
    # Print dataset info & Check if the dataset has the expected columns (which is, `text`)
    print(f"Dataset loaded: {dataset}")
    print(f"Dataset train split columns: {dataset.column_names}")
    if "text" not in dataset.column_names:
        raise ValueError(f"Dataset does not have a 'text' column. Available columns: {dataset.column_names}")
    
    print("Tokenizing dataset...")
    tokenized_dataset = dataset.map(
        create_tokenize_function(tokenizer, config["max_seq_length"]),
        batched=True,
        num_proc=64,
        remove_columns=["text"]
    )
    
    # Training arguments optimized for 8x H100 GPUs
    print("Setting up training arguments...")
    training_args = TrainingArguments(
        output_dir=config.get("output_dir", "./output"),
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        per_device_train_batch_size=config["per_device_train_batch_size"],
        learning_rate=config.get("learning_rate", 1e-5),
        logging_steps=config.get("logging_steps", 10),
        num_train_epochs=config.get("num_train_epochs", 1) if not 'max_steps' in config else 0, 
        max_steps=config.get("max_steps", 0), # default to 1 epoch 
        save_steps=config.get("save_steps", 500),
        save_total_limit=config.get("save_total_limit", 5),
        warmup_ratio=config.get('warmup_ratio', None), # default to 100 steps
        warmup_steps=config.get('warmup_steps', 100) if not 'warmup_ratio' in config else 0,
        # Defaults:
        report_to=['wandb'],
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': True},
        save_strategy="steps",
        bf16=True,
        lr_scheduler_type="cosine",
        eval_strategy="no", # evaluation we'll do separately?
        logging_first_step=True,
        remove_unused_columns=False,
        dataloader_num_workers=4,  # Can use more workers with H100s
        push_to_hub=False, # When running fo real, set to True and specify a larger save_steps
    )
    print(f"Training arguments set up successfully: {training_args}")
    
    # Prepare validation dataset if it exists
    # Commented out for now - if we do go with this, then eval_strategy will have to be updated
    # in training_args, and the dataset loading has to handle the splits, right now we load just
    # the train split directly
    eval_dataset = None # tokenized_dataset["validation"] if "validation" in tokenized_dataset else None
    
    # Initialize Trainer with our configuration
    # No need to move parameters to GPU, all handled automatically with accelerate
    print("Initializing Trainer...")
    trainer = Trainer(
        model=model,
        processing_class=tokenizer, # renamed from tokenizer
        args=training_args,
        train_dataset=tokenized_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False)
    )
    print("Trainer initialized successfully")
    
    # Start training
    print("Starting training...")
    print_trainable_parameters(model)
    try:
        trainer.train(resume_from_checkpoint=config.get("resume_from_checkpoint", False))
    except Exception as e:
        print(f"Training error: {e}")
        # Save checkpoint even if training fails
        trainer.save_model(config["output_dir"] + "/error_checkpoint")
        print(f"Saved checkpoint after error to {config['resume_from_checkpoint']}/error_checkpoint")
        raise e
    
    # Save the final model
    print(f"Saving model to {config['output_dir']}")
    trainer.save_model()
    tokenizer.save_pretrained(config["output_dir"])
    
    print("Training complete!")
    
    # Finish wandb run
    wandb.finish()

if __name__ == "__main__":
    train()