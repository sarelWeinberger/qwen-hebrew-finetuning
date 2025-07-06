#!/usr/bin/env python3
"""
SageMaker Data Preparation Script for Hebrew Text Processing
Runs on CPU instances (m5.xlarge) for cost-effective preprocessing
"""

import os
import json
import argparse
import boto3
import pandas as pd
import logging
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm
import re
from typing import List, Dict, Any

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def setup_logging():
    """Setup logging configuration"""
    return logger

def parse_args():
    parser = argparse.ArgumentParser(description="SageMaker Data Preparation for Hebrew Text")
    
    # SageMaker specific arguments
    parser.add_argument('--input-data', type=str, default=os.environ.get('SM_CHANNEL_RAW_DATA', '/opt/ml/input/data/raw-data'))
    parser.add_argument('--output-data', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR', '/opt/ml/output/data'))
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR', '/opt/ml/model'))
    
    # Processing arguments
    parser.add_argument('--model-name', type=str, default='Qwen/Qwen3-30B-A3B-Base')
    parser.add_argument('--max-length', type=int, default=2048)
    parser.add_argument('--train-split', type=float, default=0.9)
    parser.add_argument('--min-text-length', type=int, default=50)
    parser.add_argument('--max-text-length', type=int, default=10000)
    
    # S3 arguments for downloading data
    parser.add_argument('--s3-bucket', type=str, default='israllm-datasets')
    parser.add_argument('--s3-prefix', type=str, default='hebrew-text/')
    parser.add_argument('--aws-access-key-id', type=str, default=None)
    parser.add_argument('--aws-secret-access-key', type=str, default=None)
    
    return parser.parse_args()

class HebrewTextProcessor:
    """Hebrew text processing utilities"""
    
    def __init__(self, min_length=50, max_length=10000):
        self.min_length = min_length
        self.max_length = max_length
        
        # Hebrew character range
        self.hebrew_pattern = re.compile(r'[\u0590-\u05FF]+')
        
    def is_hebrew_text(self, text: str) -> bool:
        """Check if text contains Hebrew characters"""
        if not text or len(text.strip()) < self.min_length:
            return False
        
        hebrew_chars = len(self.hebrew_pattern.findall(text))
        total_chars = len(re.findall(r'[a-zA-Z\u0590-\u05FF]', text))
        
        # At least 70% Hebrew characters
        return total_chars > 0 and (hebrew_chars / total_chars) >= 0.7
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize Hebrew text"""
        if not text:
            return ""
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove very long lines (likely corrupted data)
        lines = text.split('\n')
        cleaned_lines = [line.strip() for line in lines if len(line.strip()) < 1000]
        text = '\n'.join(cleaned_lines)
        
        # Remove text that's too short or too long
        if len(text) < self.min_length or len(text) > self.max_length:
            return ""
        
        return text.strip()
    
    def process_text_batch(self, texts: List[str]) -> List[str]:
        """Process a batch of texts"""
        processed = []
        for text in texts:
            cleaned = self.clean_text(text)
            if cleaned and self.is_hebrew_text(cleaned):
                processed.append(cleaned)
        return processed

class S3DataDownloader:
    """Download Hebrew data from S3"""
    
    def __init__(self, bucket_name, aws_access_key_id=None, aws_secret_access_key=None):
        self.bucket_name = bucket_name
        
        # Setup S3 client
        if aws_access_key_id and aws_secret_access_key:
            self.s3_client = boto3.client(
                's3',
                aws_access_key_id=aws_access_key_id,
                aws_secret_access_key=aws_secret_access_key
            )
        else:
            # Use default credentials (IAM role, etc.)
            self.s3_client = boto3.client('s3')
    
    def download_files(self, prefix, local_dir):
        """Download files from S3 to local directory"""
        os.makedirs(local_dir, exist_ok=True)
        
        try:
            # List objects in S3
            response = self.s3_client.list_objects_v2(
                Bucket=self.bucket_name,
                Prefix=prefix
            )
            
            if 'Contents' not in response:
                logger.warning(f"No files found in s3://{self.bucket_name}/{prefix}")
                return []
            
            downloaded_files = []
            for obj in response['Contents']:
                key = obj['Key']
                local_file = os.path.join(local_dir, os.path.basename(key))
                
                logger.info(f"Downloading {key} to {local_file}")
                self.s3_client.download_file(self.bucket_name, key, local_file)
                downloaded_files.append(local_file)
            
            logger.info(f"Downloaded {len(downloaded_files)} files from S3")
            return downloaded_files
            
        except Exception as e:
            logger.error(f"Failed to download from S3: {e}")
            return []

def create_sample_hebrew_data(output_dir: str, num_samples: int = 1000):
    """Create sample Hebrew data for testing"""
    logger.info("Creating sample Hebrew data for testing...")
    
    sample_texts = [
        "שלום עולם! זוהי דוגמה לטקסט בעברית. אנחנו בודקים את המערכת שלנו.",
        "הטכנולוgia החדשה מאפשרת לנו לעבד טקסט בעברית בצורה יעילה יותר.",
        "בישראל יש הרבה חברות טכנולוגיה מתקדמות שעובדות על בינה מלאכותית.",
        "המודל שלנו לומד לעבד טקסט בעברית ולהבין את המשמעות שלו.",
        "זהו פרויקט מחקר חשוב שיכול לשפר את הטכנולוגיה בישראל.",
    ]
    
    # Generate more samples by combining and varying the base texts
    extended_samples = []
    for i in range(num_samples):
        # Combine 2-3 sample texts
        combined = " ".join(sample_texts[i % len(sample_texts):i % len(sample_texts) + 2])
        extended_samples.append(f"{combined} מספר דוגמה: {i+1}")
    
    # Save as JSON
    output_file = os.path.join(output_dir, "sample_hebrew_data.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        for text in extended_samples:
            json.dump({"text": text}, f, ensure_ascii=False)
            f.write('\n')
    
    logger.info(f"Created {num_samples} sample texts in {output_file}")
    return [output_file]

def load_and_process_data(input_files: List[str], processor: HebrewTextProcessor) -> List[Dict[str, str]]:
    """Load and process data from input files"""
    all_texts = []
    
    for file_path in input_files:
        logger.info(f"Processing file: {file_path}")
        
        try:
            if file_path.endswith('.json') or file_path.endswith('.jsonl'):
                # Load JSON/JSONL files
                with open(file_path, 'r', encoding='utf-8') as f:
                    for line in f:
                        try:
                            data = json.loads(line.strip())
                            if 'text' in data:
                                all_texts.append(data['text'])
                        except json.JSONDecodeError:
                            continue
            
            elif file_path.endswith('.csv'):
                # Load CSV files
                df = pd.read_csv(file_path)
                text_columns = ['text', 'content', 'body', 'message']
                
                for col in text_columns:
                    if col in df.columns:
                        all_texts.extend(df[col].dropna().astype(str).tolist())
                        break
            
            elif file_path.endswith('.txt'):
                # Load text files
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    # Split by paragraphs
                    paragraphs = content.split('\n\n')
                    all_texts.extend(paragraphs)
            
        except Exception as e:
            logger.error(f"Failed to process {file_path}: {e}")
            continue
    
    logger.info(f"Loaded {len(all_texts)} raw texts")
    
    # Process texts in batches
    batch_size = 1000
    processed_texts = []
    
    for i in tqdm(range(0, len(all_texts), batch_size), desc="Processing texts"):
        batch = all_texts[i:i + batch_size]
        processed_batch = processor.process_text_batch(batch)
        processed_texts.extend(processed_batch)
    
    logger.info(f"Processed {len(processed_texts)} valid Hebrew texts")
    
    # Convert to dataset format
    dataset_texts = [{"text": text} for text in processed_texts]
    return dataset_texts

def tokenize_and_save_dataset(texts: List[Dict[str, str]], tokenizer, output_dir: str, max_length: int, train_split: float):
    """Tokenize texts and save as dataset"""
    logger.info("Creating dataset from processed texts...")
    
    # Create dataset
    dataset = Dataset.from_list(texts)
    
    # Split into train/validation
    if train_split < 1.0:
        dataset = dataset.train_test_split(test_size=1.0 - train_split, seed=42)
        train_dataset = dataset['train']
        val_dataset = dataset['test']
    else:
        train_dataset = dataset
        val_dataset = None
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    if val_dataset:
        logger.info(f"Validation dataset size: {len(val_dataset)}")
    
    # Tokenization function
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            padding=False,  # Use dynamic padding during training
            truncation=True,
            max_length=max_length,
            return_tensors=None
        )
    
    # Tokenize datasets
    logger.info("Tokenizing train dataset...")
    tokenized_train = train_dataset.map(
        tokenize_function,
        batched=True,
        num_proc=4,
        remove_columns=["text"],
        desc="Tokenizing train data"
    )
    
    tokenized_val = None
    if val_dataset:
        logger.info("Tokenizing validation dataset...")
        tokenized_val = val_dataset.map(
            tokenize_function,
            batched=True,
            num_proc=4,
            remove_columns=["text"],
            desc="Tokenizing validation data"
        )
    
    # Save tokenized datasets
    dataset_dir = os.path.join(output_dir, "dataset")
    os.makedirs(dataset_dir, exist_ok=True)
    
    if tokenized_val:
        final_dataset = {
            "train": tokenized_train,
            "validation": tokenized_val
        }
    else:
        final_dataset = {"train": tokenized_train}
    
    # Save as dataset
    from datasets import DatasetDict
    dataset_dict = DatasetDict(final_dataset)
    dataset_dict.save_to_disk(dataset_dir)
    
    logger.info(f"Dataset saved to {dataset_dir}")
    
    # Save dataset info
    info_file = os.path.join(output_dir, "dataset_info.json")
    with open(info_file, 'w') as f:
        json.dump({
            "train_size": len(tokenized_train),
            "validation_size": len(tokenized_val) if tokenized_val else 0,
            "max_length": max_length,
            "tokenizer": tokenizer.name_or_path,
            "total_texts_processed": len(texts)
        }, f, indent=2)
    
    return dataset_dir

def main():
    logger = setup_logging()
    args = parse_args()
    
    logger.info("Starting SageMaker data preparation...")
    logger.info(f"Input data: {args.input_data}")
    logger.info(f"Output data: {args.output_data}")
    
    # Create output directory
    os.makedirs(args.output_data, exist_ok=True)
    
    # Initialize text processor
    processor = HebrewTextProcessor(
        min_length=args.min_text_length,
        max_length=args.max_text_length
    )
    
    # Download data from S3 or use local data
    input_files = []
    
    if args.s3_bucket:
        logger.info(f"Downloading data from S3: s3://{args.s3_bucket}/{args.s3_prefix}")
        downloader = S3DataDownloader(
            args.s3_bucket,
            args.aws_access_key_id,
            args.aws_secret_access_key
        )
        
        temp_dir = "/tmp/s3_data"
        downloaded_files = downloader.download_files(args.s3_prefix, temp_dir)
        
        if downloaded_files:
            input_files = downloaded_files
        else:
            logger.warning("No files downloaded from S3, creating sample data")
            input_files = create_sample_hebrew_data(args.output_data)
    
    # Also check for local input files
    if os.path.exists(args.input_data):
        for root, dirs, files in os.walk(args.input_data):
            for file in files:
                if file.endswith(('.json', '.jsonl', '.csv', '.txt')):
                    input_files.append(os.path.join(root, file))
    
    # If no input files found, create sample data
    if not input_files:
        logger.warning("No input files found, creating sample Hebrew data")
        input_files = create_sample_hebrew_data(args.output_data)
    
    logger.info(f"Processing {len(input_files)} input files")
    
    # Load tokenizer
    logger.info(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Process data
    processed_texts = load_and_process_data(input_files, processor)
    
    if not processed_texts:
        logger.error("No valid texts found after processing")
        return
    
    # Tokenize and save dataset
    dataset_dir = tokenize_and_save_dataset(
        processed_texts,
        tokenizer,
        args.output_data,
        args.max_length,
        args.train_split
    )
    
    logger.info("Data preparation completed successfully!")
    logger.info(f"Processed dataset saved to: {dataset_dir}")

if __name__ == "__main__":
    main()