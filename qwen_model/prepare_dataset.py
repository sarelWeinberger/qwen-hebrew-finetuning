import os
import json
import argparse
from datasets import Dataset, DatasetDict

def parse_args():
    parser = argparse.ArgumentParser(description="Prepare dataset for fine-tuning Qwen3-30B-A3B-Base model")
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Path to the input data file (can be .txt, .json, or .jsonl)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="qwen_model/finetuning/dataset",
        help="Directory to save the processed dataset"
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["text", "chat", "instruction"],
        default="text",
        help="Format of the dataset (text, chat, or instruction)"
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2048,
        help="Maximum sequence length for training"
    )
    return parser.parse_args()

def format_text_data(text, max_length=2048):
    """Format raw text data into chunks of max_length tokens (approximate)"""
    # Simple chunking by characters as a rough approximation
    # In practice, you'd want to use the tokenizer for proper chunking
    chunks = []
    words = text.split()
    current_chunk = []
    current_length = 0
    
    for word in words:
        # Rough approximation: 1 word ≈ 1.3 tokens on average
        word_length = len(word) * 1.3
        if current_length + word_length > max_length:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = word_length
        else:
            current_chunk.append(word)
            current_length += word_length
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return [{"text": chunk} for chunk in chunks]

def format_chat_data(data):
    """Format chat data into a format suitable for training"""
    formatted_data = []
    
    for conversation in data:
        if "messages" not in conversation:
            continue
            
        formatted_text = ""
        for message in conversation["messages"]:
            if "role" not in message or "content" not in message:
                continue
                
            role = message["role"]
            content = message["content"]
            
            if role == "system":
                formatted_text += f"<|system|>\n{content}\n"
            elif role == "user":
                formatted_text += f"<|user|>\n{content}\n"
            elif role == "assistant":
                formatted_text += f"<|assistant|>\n{content}\n"
        
        formatted_data.append({"text": formatted_text})
    
    return formatted_data

def format_instruction_data(data):
    """Format instruction data into a format suitable for training"""
    formatted_data = []
    
    for item in data:
        if "instruction" not in item:
            continue
            
        instruction = item["instruction"]
        input_text = item.get("input", "")
        output = item.get("output", "")
        
        if input_text:
            formatted_text = f"<|user|>\n{instruction}\n\n{input_text}\n<|assistant|>\n{output}"
        else:
            formatted_text = f"<|user|>\n{instruction}\n<|assistant|>\n{output}"
        
        formatted_data.append({"text": formatted_text})
    
    return formatted_data

def prepare_dataset():
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data based on file extension
    file_extension = os.path.splitext(args.input_file)[1].lower()
    
    if file_extension == '.txt':
        with open(args.input_file, 'r', encoding='utf-8') as f:
            text = f.read()
        
        if args.format == "text":
            data = format_text_data(text, args.max_length)
        else:
            print(f"Error: {args.format} format not supported for .txt files. Use .json or .jsonl for chat/instruction formats.")
            return
    
    elif file_extension in ['.json', '.jsonl']:
        if file_extension == '.json':
            with open(args.input_file, 'r', encoding='utf-8') as f:
                raw_data = json.load(f)
        else:  # .jsonl
            raw_data = []
            with open(args.input_file, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        raw_data.append(json.loads(line))
        
        if args.format == "text":
            # Assume each item has a "text" field
            data = [{"text": item["text"]} for item in raw_data if "text" in item]
        elif args.format == "chat":
            data = format_chat_data(raw_data)
        elif args.format == "instruction":
            data = format_instruction_data(raw_data)
    
    else:
        print(f"Error: Unsupported file extension {file_extension}. Use .txt, .json, or .jsonl.")
        return
    
    # Check if we have any data
    if not data:
        print("Warning: No data found in the input file.")
        print("Creating a minimal sample dataset to avoid schema inference errors.")
        
        # Create a minimal sample dataset with Hebrew text
        data = [{"text": "זוהי דוגמה לטקסט בעברית. נוצר כדי למנוע שגיאות סכמה."}]
    
    # Create dataset
    dataset = Dataset.from_list(data)
    
    # Split into train/validation sets (90/10 split)
    # For very small datasets, ensure we have at least one example in each split
    if len(dataset) <= 1:
        # If only one example, duplicate it for both train and validation
        dataset_dict = DatasetDict({
            "train": dataset,
            "validation": dataset
        })
    else:
        dataset = dataset.train_test_split(test_size=0.1)
        dataset_dict = DatasetDict({
            "train": dataset["train"],
            "validation": dataset["test"]
        })
    
    # Save dataset
    dataset_path = os.path.join(args.output_dir, "dataset")
    dataset_dict.save_to_disk(dataset_path)
    
    print(f"Dataset prepared and saved to {dataset_path}")
    print(f"Train set size: {len(dataset_dict['train'])}")
    print(f"Validation set size: {len(dataset_dict['validation'])}")
    print("\nExample data format:")
    print(dataset_dict["train"][0])
    
    # Create a sample command for training
    print("\nTo use this dataset for training, run:")
    print(f"python qwen_model/train.py --dataset_path {dataset_path} --config qwen_model/finetuning/training_config.json")

if __name__ == "__main__":
    prepare_dataset()