import boto3

def get_directory_size(bucket, prefix):
    s3 = boto3.client('s3')
    total_size = 0
    
    paginator = s3.get_paginator('list_objects_v2')
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        if 'Contents' in page:
            for obj in page['Contents']:
                total_size += obj['Size']
    
    return total_size

def compare_directory_sizes():
    s3 = boto3.client('s3')
    bucket = 'gepeta-datasets'
    base_prefix = 'processed_cleaned_filtered/run_4/'
    
    total_output_size = 0
    total_removed_size = 0
    
    # List all directories in run_4
    response = s3.list_objects_v2(Bucket=bucket, Prefix=base_prefix, Delimiter='/')
    
    for prefix in response.get('CommonPrefixes', []):
        dir_name = prefix['Prefix'].split('/')[-2]
        output_size = get_directory_size(bucket, f"{prefix['Prefix']}filtering/output/")
        removed_size = get_directory_size(bucket, f"{prefix['Prefix']}filtering/removed/")
        
        total_size = output_size + removed_size
        filter_percent = (removed_size / total_size * 100) if total_size else 0
        
        print(f"Dataset: {dir_name}")
        print(f"  Output size: {output_size / (1024 * 1024):.2f} MB")
        print(f"  Removed size: {removed_size / (1024 * 1024):.2f} MB")
        print(f"  Filter percentage: {filter_percent:.2f}%")
        print()
        
        total_output_size += output_size
        total_removed_size += removed_size
    
    total_size = total_output_size + total_removed_size
    overall_filter_percent = (total_removed_size / total_size * 100) if total_size else 0
    
    print("Overall Statistics:")
    print(f"  Total Output size: {total_output_size / (1024 * 1024):.2f} MB")
    print(f"  Total Removed size: {total_removed_size / (1024 * 1024):.2f} MB")
    print(f"  Total size: {total_size / (1024 * 1024):.2f} MB")
    print(f"  Overall Filter percentage: {overall_filter_percent:.2f}%")



import json
import gzip
import multiprocessing
import os
import boto3
from functools import partial
from tqdm import tqdm
from transformers import AutoTokenizer
import io
from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
from typing import List, Tuple, Iterator, Set
import gc
from botocore.exceptions import ClientError

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global tokenizer - will be initialized once per process
tokenizer = None
CHUNK_SIZE = 1024 * 1024  # 1MB chunks for streaming
BATCH_SIZE = 100  # Number of texts to tokenize at once

def init_worker():
    """Initialize tokenizer in each worker process"""
    global tokenizer
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B", trust_remote_code=True)
        logger.info(f"Tokenizer initialized in process {os.getpid()}")

def stream_jsonl_from_s3(s3_client, bucket: str, key: str) -> Iterator[str]:
    """Stream and decompress JSONL file from S3 in chunks"""
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        
        # Create a streaming decompressor
        decompressor = gzip.GzipFile(fileobj=response['Body'])
        
        # Stream in chunks and yield complete lines
        buffer = ""
        while True:
            chunk = decompressor.read(CHUNK_SIZE)
            if not chunk:
                break
                
            chunk_str = chunk.decode('utf-8', errors='ignore')
            buffer += chunk_str
            
            # Split into lines and yield complete ones
            lines = buffer.split('\n')
            buffer = lines[-1]  # Keep incomplete line in buffer
            
            for line in lines[:-1]:
                if line.strip():
                    yield line.strip()
        
        # Yield any remaining content in buffer
        if buffer.strip():
            yield buffer.strip()
            
    except Exception as e:
        logger.error(f"Error streaming file {key}: {str(e)}")
        raise

def count_tokens_batch(texts: List[str]) -> Tuple[int, int]:
    """Count tokens and words for a batch of texts"""
    global tokenizer
    
    if not texts:
        return 0, 0
    
    # Count words efficiently
    word_count = sum(len(text.split()) for text in texts)
    
    # Batch tokenize for efficiency
    try:
        # Use batch encoding for better performance
        encodings = tokenizer(texts, 
                            add_special_tokens=False, 
                            padding=False, 
                            truncation=False, 
                            return_attention_mask=False,
                            return_token_type_ids=False)
        
        token_count = sum(len(encoding) for encoding in encodings['input_ids'])
        
    except Exception as e:
        logger.warning(f"Batch tokenization failed, falling back to individual: {str(e)}")
        # Fallback to individual tokenization
        token_count = 0
        for text in texts:
            try:
                token_count += len(tokenizer.encode(text, add_special_tokens=False))
            except Exception:
                continue  # Skip problematic texts
    
    return word_count, token_count

def process_file_optimized(bucket: str, file_key: str) -> Tuple[str, int, int]:
    """Process a single file with optimized streaming and batching"""
    global tokenizer
    
    # Create S3 client with optimized configuration
    s3 = boto3.client('s3', 
                      config=boto3.session.Config(
                          region_name='us-east-1',  # Adjust as needed
                          retries={'max_attempts': 3},
                          max_pool_connections=50
                      ))
    
    total_word_count = 0
    total_token_count = 0
    batch_texts = []
    
    try:
        logger.info(f"Processing file: {file_key}")
        
        # Stream the file and process in batches
        for line in stream_jsonl_from_s3(s3, bucket, file_key):
            try:
                data = json.loads(line)
                text = data.get('text', '').strip()
                
                if text:
                    batch_texts.append(text)
                    
                    # Process batch when it reaches BATCH_SIZE
                    if len(batch_texts) >= BATCH_SIZE:
                        word_count, token_count = count_tokens_batch(batch_texts)
                        total_word_count += word_count
                        total_token_count += token_count
                        batch_texts = []
                        
                        # Force garbage collection periodically
                        if total_token_count % (BATCH_SIZE * 10) == 0:
                            gc.collect()
                            
            except json.JSONDecodeError:
                continue  # Skip malformed JSON lines
        
        # Process remaining texts in the last batch
        if batch_texts:
            word_count, token_count = count_tokens_batch(batch_texts)
            total_word_count += word_count
            total_token_count += token_count
        
        logger.info(f"Completed {file_key}: {total_word_count} words, {total_token_count} tokens")
        return file_key, total_word_count, total_token_count
        
    except Exception as e:
        logger.error(f"Error processing file {file_key}: {str(e)}")
        return file_key, 0, 0
    finally:
        # Clean up
        gc.collect()

def check_existing_output(s3_client, bucket: str, token_counts_prefix: str) -> Set[str]:
    """Check which output files already exist in S3 and return set of completed sources"""
    existing_files = set()
    
    try:
        logger.info(f"Checking for existing output files in s3://{bucket}/{token_counts_prefix}")
        paginator = s3_client.get_paginator('list_objects_v2')
        
        for page in paginator.paginate(Bucket=bucket, Prefix=token_counts_prefix):
            for obj in page.get('Contents', []):
                key = obj['Key']
                if key.endswith('_counts.txt'):
                    # Extract source name from filename
                    filename = key.split('/')[-1]
                    source_name = filename.replace('_counts.txt', '')
                    existing_files.add(source_name)
                    logger.info(f"Found existing output: {filename}")
        
        if existing_files:
            logger.info(f"Found {len(existing_files)} existing output files: {sorted(existing_files)}")
        else:
            logger.info("No existing output files found")
            
    except ClientError as e:
        if e.response['Error']['Code'] == 'NoSuchBucket':
            logger.warning(f"Bucket {bucket} does not exist")
        else:
            logger.warning(f"Error checking existing files: {str(e)}")
    
    return existing_files

def ensure_s3_directory(s3_client, bucket: str, prefix: str):
    """Ensure S3 directory exists by creating a placeholder if needed"""
    try:
        # Check if directory exists by listing objects with the prefix
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=1)
        
        if 'Contents' not in response:
            # Directory doesn't exist, create it by uploading a placeholder
            placeholder_key = f"{prefix.rstrip('/')}/.placeholder"
            s3_client.put_object(Bucket=bucket, Key=placeholder_key, Body=b'')
            logger.info(f"Created S3 directory: s3://{bucket}/{prefix}")
        else:
            logger.info(f"S3 directory already exists: s3://{bucket}/{prefix}")
            
    except ClientError as e:
        logger.error(f"Error ensuring S3 directory: {str(e)}")
        raise

def upload_file_to_s3(s3_client, local_file_path: str, bucket: str, s3_key: str):
    """Upload a file to S3 with error handling"""
    try:
        logger.info(f"Uploading {local_file_path} to s3://{bucket}/{s3_key}")
        s3_client.upload_file(local_file_path, bucket, s3_key)
        logger.info(f"Successfully uploaded to S3: {s3_key}")
    except ClientError as e:
        logger.error(f"Failed to upload {local_file_path} to S3: {str(e)}")
        raise

def process_source_optimized(bucket: str, source_prefix: str, max_processes: int, 
                           token_counts_prefix: str, existing_files: Set[str]):
    """Process all files in a source with optimized multiprocessing and S3 backup"""
    s3 = boto3.client('s3')
    
    # Extract source name for checking if already processed
    source_name = source_prefix.strip('/').split('/')[-1]
    
    # Check if this source was already processed
    if source_name in existing_files:
        logger.info(f"Source {source_name} already processed, skipping...")
        return
    
    # List all files in the source directory
    logger.info(f"Listing files in {source_prefix}...")
    paginator = s3.get_paginator('list_objects_v2')
    files = []
    
    for page in paginator.paginate(Bucket=bucket, Prefix=f"{source_prefix}filtering/output/"):
        for obj in page.get('Contents', []):
            if obj['Key'].endswith('.jsonl.gz'):
                files.append(obj['Key'])
    
    logger.info(f"Found {len(files)} files to process for source {source_name}")
    
    if not files:
        logger.warning(f"No files found to process in {source_prefix}")
        return
    
    # Process files using ProcessPoolExecutor for better control
    results = []
    with ProcessPoolExecutor(max_workers=max_processes, initializer=init_worker) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(process_file_optimized, bucket, file_key): file_key 
            for file_key in files
        }
        
        # Process completed tasks with progress bar
        with tqdm(total=len(files), desc=f"Processing {source_name}") as pbar:
            for future in as_completed(future_to_file):
                file_key = future_to_file[future]
                try:
                    result = future.result(timeout=3600)  # 1 hour timeout per file
                    results.append(result)
                except Exception as e:
                    logger.error(f"Failed to process {file_key}: {str(e)}")
                    results.append((file_key, 0, 0))
                finally:
                    pbar.update(1)
    
    # Write results to local file
    output_file = f"{source_name}_counts.txt"
    logger.info(f"Writing results to {output_file}")
    
    with open(output_file, 'w') as f:
        f.write("file_key,word_count,token_count\n")  # Header
        for file_key, word_count, token_count in sorted(results):
            f.write(f"{file_key},{word_count},{token_count}\n")
    
    # Upload to S3
    s3_key = f"{token_counts_prefix}{output_file}"
    try:
        upload_file_to_s3(s3, output_file, bucket, s3_key)
        logger.info(f"Successfully backed up {output_file} to S3")
    except Exception as e:
        logger.error(f"Failed to upload {output_file} to S3, but local file is available: {str(e)}")
    
    # Calculate and log totals
    total_words = sum(result[1] for result in results)
    total_tokens = sum(result[2] for result in results)
    processed_files = len([r for r in results if r[1] > 0 or r[2] > 0])
    logger.info(f"Source {source_name} completed: {processed_files}/{len(files)} files processed, "
                f"{total_words:,} words, {total_tokens:,} tokens")

def count_tokens(bucket, base_prefix):
    """Main function with optimized configuration and S3 backup"""

    token_counts_prefix = f"{base_prefix}token_counts/"
    
    # Optimize process count based on system resources
    cpu_count = os.cpu_count()
    max_processes = min(cpu_count - 3, 8)  # Cap at 8 to avoid overwhelming S3
    max_processes = max(1, max_processes)  # Ensure at least 1 process
    
    logger.info(f"Using {max_processes} processes (CPU count: {cpu_count})")
    logger.info(f"Token counts will be saved to: s3://{bucket}/{token_counts_prefix}")
    
    # Initialize tokenizer in main process for validation
    logger.info("Initializing tokenizer...")
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B", trust_remote_code=True)
    logger.info("Tokenizer initialized successfully")
    
    s3 = boto3.client('s3')
    
    try:
        # Ensure token_counts directory exists in S3
        ensure_s3_directory(s3, bucket, token_counts_prefix)
        
        # Check for existing output files
        existing_files = check_existing_output(s3, bucket, token_counts_prefix)
        
        # Get list of source prefixes to process
        response = s3.list_objects_v2(Bucket=bucket, Prefix=base_prefix, Delimiter='/')
        prefixes = response.get('CommonPrefixes', [])
        
        if not prefixes:
            logger.warning(f"No source prefixes found under {base_prefix}")
            return
        
        # Filter out already processed sources
        sources_to_process = []
        for prefix in prefixes:
            source_prefix = prefix['Prefix']
            source_name = source_prefix.strip('/').split('/')[-1]
            if source_name not in existing_files and not "token_counts" in source_name:
                sources_to_process.append(source_prefix)
            else:
                logger.info(f"Skipping already processed source: {source_name}")
        
        if not sources_to_process:
            logger.info("All sources have already been processed!")
            return
        
        logger.info(f"Found {len(sources_to_process)} source prefixes to process "
                   f"({len(existing_files)} already completed)")
        
        # Process each source
        for i, source_prefix in enumerate(reversed(sources_to_process), 1):
            source_name = source_prefix.strip('/').split('/')[-1]
            logger.info(f"Processing source {i}/{len(sources_to_process)}: {source_name}")
            
            try:
                process_source_optimized(bucket, source_prefix, max_processes, 
                                       token_counts_prefix, existing_files)
                logger.info(f"Completed source {i}/{len(sources_to_process)}: {source_name}")
                
                # Add to existing files set to avoid reprocessing if script is interrupted and restarted
                existing_files.add(source_name)
                
            except Exception as e:
                logger.error(f"Failed to process source {source_name}: {str(e)}")
                continue  # Continue with next source
        
        logger.info("All sources processed successfully!")
        
        # Generate summary report
        final_existing = check_existing_output(s3, bucket, token_counts_prefix)
        logger.info(f"Final summary: {len(final_existing)} output files in S3:")
        for filename in sorted(final_existing):
            logger.info(f"  - {filename}_counts.txt")
        
    except Exception as e:
        logger.error(f"Error in main processing: {str(e)}")
        raise
import pandas as pd
import boto3
from io import StringIO

def summarize_counts(s3_bucket, prefix):
    s3 = boto3.client('s3')
    
    # List all objects in the bucket with the given prefix
    response = s3.list_objects_v2(Bucket=s3_bucket, Prefix=prefix)
    
    summaries = {}
    
    for obj in response['Contents']:
        if obj['Key'].endswith('_counts.txt'):
            # Read the file content
            file_content = s3.get_object(Bucket=s3_bucket, Key=obj['Key'])['Body'].read().decode('utf-8')
            
            # Parse the content as CSV
            df = pd.read_csv(StringIO(file_content))
            
            # Extract the data source name from the file key
            data_source = obj['Key'].split('/')[-1].replace("_counts.txt","")
            
            # Sum up the word and token counts
            total_words = df['word_count'].sum()
            total_tokens = df['token_count'].sum()
            
            # Store the summary
            summaries[data_source] = {
                'total_words': total_words,
                'total_tokens': total_tokens
            }
    
    # Create a summary DataFrame
    summary_df = pd.DataFrame.from_dict(summaries, orient='index')
    # Calculate the sum of each column
    total_row = summary_df.sum()

    # Calculate the total in millions
    total_millions_row = total_row / 1_000_000

    # Calculate percentages
    summary_df['percent_words'] = (summary_df['total_words'] / total_row['total_words']) * 100
    summary_df['percent_tokens'] = (summary_df['total_tokens'] / total_row['total_tokens']) * 100

    # Add the total row
    summary_df.loc['Total'] = total_row

    # Add the total in millions row
    summary_df.loc['Total (millions)'] = total_millions_row

    # Round the values
    summary_df['total_words'] = summary_df['total_words'].round(0).astype(int)
    summary_df['total_tokens'] = summary_df['total_tokens'].round(0).astype(int)
    summary_df['percent_words'] = summary_df['percent_words'].round(2)
    summary_df['percent_tokens'] = summary_df['percent_tokens'].round(2)

    # Adjust the 'Total' row for percentage columns
    summary_df.loc['Total', 'percent_words'] = 100
    summary_df.loc['Total', 'percent_tokens'] = 100

    # Set percentage columns to NaN for 'Total (millions)' row
    summary_df.loc['Total (millions)', ['percent_words', 'percent_tokens']] = pd.NA

    # Reorder columns
    summary_df = summary_df[['total_words', 'percent_words', 'total_tokens', 'percent_tokens']]

    # Write the summary to a file in S3
    summary_csv = summary_df.to_csv()
    print(summary_csv)
    s3.put_object(Bucket=s3_bucket, Key=f'{prefix}/counts_summary.csv', Body=summary_csv)
    
    print(f"Summary has been written to {s3_bucket}/{prefix}/counts_summary.csv")


if __name__ == "__main__":
    # compare_directory_sizes()
    # count_tokens(bucket, base_prefix)
    # Usage
    s3_bucket = 'gepeta-datasets'
    prefix = 'processed_cleaned_filtered/run_4/'
    token_counts_prefix = 'processed_cleaned_filtered/run_4/token_counts'

    summarize_counts(s3_bucket, token_counts_prefix)