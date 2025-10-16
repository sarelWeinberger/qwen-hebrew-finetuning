import json
import gzip
import multiprocessing
import os
import boto3
from functools import partial
from tqdm import tqdm
from transformers import AutoTokenizer
import io
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import logging
from typing import List, Tuple, Iterator, Set, Dict, Any
import gc
import threading
import queue
import time
from botocore.exceptions import ClientError
from botocore.config import Config
import psutil
import asyncio
import aiofiles
from dataclasses import dataclass
import pickle
import tempfile
import shutil

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Global tokenizer - will be initialized once per process
tokenizer = None
CHUNK_SIZE = 1024 * 1024 * 50  # 50MB chunks for better throughput
BATCH_SIZE = 500  # Larger batch size for better GPU/CPU utilization
MAX_QUEUE_SIZE = 1000  # Maximum queue size for memory management

@dataclass
class ProcessingResult:
    file_key: str
    word_count: int
    token_count: int
    processing_time: float
    error: str = None

class TokenizerPool:
    """Thread-safe tokenizer pool to avoid reinitialization overhead"""
    def __init__(self, pool_size: int = 4):
        self.pool = queue.Queue(maxsize=pool_size)
        self.pool_size = pool_size
        self._initialize_pool()
    
    def _initialize_pool(self):
        for _ in range(self.pool_size):
            tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B", trust_remote_code=True)
            self.pool.put(tokenizer)
    
    def get_tokenizer(self):
        return self.pool.get()
    
    def return_tokenizer(self, tokenizer):
        self.pool.put(tokenizer)

# Global tokenizer pool
tokenizer_pool = None

def init_worker():
    """Initialize resources in each worker process"""
    global tokenizer_pool
    if tokenizer_pool is None:
        # Create a smaller pool per process to manage memory
        tokenizer_pool = TokenizerPool(pool_size=2)
        logger.info(f"Tokenizer pool initialized in process {os.getpid()}")

def get_optimal_process_count() -> Tuple[int, int]:
    """Calculate optimal process and thread counts based on system resources"""
    cpu_count = os.cpu_count() or 4
    memory_gb = psutil.virtual_memory().total / (1024**3)
    
    # More aggressive CPU utilization
    process_count = max(1, min(cpu_count, 32))  # Use most CPUs but cap at 32
    
    # Thread pool for I/O operations
    io_thread_count = min(process_count * 2, 64)  # More threads for I/O
    
    logger.info(f"System: {cpu_count} CPUs, {memory_gb:.1f}GB RAM")
    logger.info(f"Using: {process_count} processes, {io_thread_count} I/O threads")
    
    return process_count, io_thread_count

def create_optimized_s3_client():
    """Create S3 client with optimized configuration for high concurrency"""
    config = Config(
        region_name='us-east-1',
        retries={'max_attempts': 3, 'mode': 'adaptive'},
        max_pool_connections=100,  # Higher connection pool
        read_timeout=300,
        connect_timeout=60,
        tcp_keepalive=True
    )
    return boto3.client('s3', config=config)

def stream_jsonl_from_s3_optimized(s3_client, bucket: str, key: str) -> Iterator[Dict[str, Any]]:
    """Optimized streaming with better error handling and memory management"""
    try:
        response = s3_client.get_object(Bucket=bucket, Key=key)
        
        with gzip.GzipFile(fileobj=response['Body']) as gz_file:
            buffer = ""
            
            while True:
                chunk = gz_file.read(CHUNK_SIZE)
                if not chunk:
                    break
                
                chunk_str = chunk.decode('utf-8', errors='ignore')
                buffer += chunk_str
                
                lines = buffer.split('\n')
                buffer = lines[-1]
                
                for line in lines[:-1]:
                    line = line.strip()
                    if line:
                        try:
                            yield json.loads(line)
                        except json.JSONDecodeError:
                            continue
            
            # Process remaining buffer
            if buffer.strip():
                try:
                    yield json.loads(buffer.strip())
                except json.JSONDecodeError:
                    pass
                    
    except Exception as e:
        logger.error(f"Error streaming file {key}: {str(e)}")
        raise

def count_tokens_batch_optimized(texts: List[str]) -> Tuple[int, int]:
    """Optimized batch tokenization with better memory management"""
    global tokenizer_pool
    
    if not texts:
        return 0, 0
    
    # Quick word count
    word_count = sum(len(text.split()) for text in texts if text)
    
    # Get tokenizer from pool
    tokenizer = tokenizer_pool.get_tokenizer()
    
    try:
        # Filter out empty texts
        valid_texts = [text for text in texts if text.strip()]
        if not valid_texts:
            return word_count, 0
        
        # Batch tokenize with optimized settings
        encodings = tokenizer(
            valid_texts,
            add_special_tokens=False,
            padding=False,
            truncation=False,
            return_attention_mask=False,
            return_token_type_ids=False,
            return_tensors=None  # Return lists instead of tensors for memory efficiency
        )
        
        token_count = sum(len(encoding) for encoding in encodings['input_ids'])
        
    except Exception as e:
        logger.warning(f"Batch tokenization failed: {str(e)}")
        # Fallback to individual tokenization
        token_count = 0
        for text in valid_texts:
            try:
                tokens = tokenizer.encode(text, add_special_tokens=False)
                token_count += len(tokens)
            except Exception:
                continue
    finally:
        # Return tokenizer to pool
        tokenizer_pool.return_tokenizer(tokenizer)
    
    return word_count, token_count

def process_file_chunks(file_data: Tuple[str, str]) -> ProcessingResult:
    """Process a file in chunks with optimized memory usage"""
    bucket, file_key = file_data
    start_time = time.time()
    
    s3_client = create_optimized_s3_client()
    
    total_word_count = 0
    total_token_count = 0
    batch_texts = []
    
    try:
        logger.debug(f"Processing file: {file_key}")
        
        # Stream and process file
        for data in stream_jsonl_from_s3_optimized(s3_client, bucket, file_key):
            text = data.get('text', '').strip()
            
            if text:
                batch_texts.append(text)
                
                # Process batch when it reaches target size
                if len(batch_texts) >= BATCH_SIZE:
                    word_count, token_count = count_tokens_batch_optimized(batch_texts)
                    total_word_count += word_count
                    total_token_count += token_count
                    batch_texts = []
                    
                    # Periodic garbage collection
                    if total_token_count % (BATCH_SIZE * 20) == 0:
                        gc.collect()
        
        # Process remaining texts
        if batch_texts:
            word_count, token_count = count_tokens_batch_optimized(batch_texts)
            total_word_count += word_count
            total_token_count += token_count
        
        processing_time = time.time() - start_time
        logger.debug(f"Completed {file_key}: {total_word_count} words, {total_token_count} tokens in {processing_time:.2f}s")
        
        return ProcessingResult(
            file_key=file_key,
            word_count=total_word_count,
            token_count=total_token_count,
            processing_time=processing_time
        )
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"Error processing file {file_key}: {str(e)}")
        return ProcessingResult(
            file_key=file_key,
            word_count=0,
            token_count=0,
            processing_time=processing_time,
            error=str(e)
        )
    finally:
        gc.collect()

def get_all_files_parallel(bucket: str, source_prefixes: List[str], max_threads: int) -> Dict[str, List[str]]:
    """Get all files for all sources in parallel using threading"""
    def list_files_for_source(source_prefix: str) -> Tuple[str, List[str]]:
        s3_client = create_optimized_s3_client()
        files = []
        
        try:
            paginator = s3_client.get_paginator('list_objects_v2')
            for page in paginator.paginate(Bucket=bucket, Prefix=f"{source_prefix}filtering/output/"):
                for obj in page.get('Contents', []):
                    if obj['Key'].endswith('.jsonl.gz'):
                        files.append(obj['Key'])
        except Exception as e:
            logger.error(f"Error listing files for {source_prefix}: {str(e)}")
        
        return source_prefix, files
    
    source_files = {}
    
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        future_to_source = {
            executor.submit(list_files_for_source, source_prefix): source_prefix
            for source_prefix in source_prefixes
        }
        
        for future in as_completed(future_to_source):
            source_prefix, files = future.result()
            source_name = source_prefix.strip('/').split('/')[-1]
            source_files[source_name] = files
            logger.info(f"Found {len(files)} files for source {source_name}")
    
    return source_files

def process_all_files_mega_parallel(bucket: str, source_files: Dict[str, List[str]], 
                                  max_processes: int, token_counts_prefix: str) -> Dict[str, List[ProcessingResult]]:
    """Process all files from all sources in a single mega-parallel operation"""
    
    # Flatten all files with source tracking
    all_file_data = []
    file_to_source = {}
    
    for source_name, files in source_files.items():
        for file_key in files:
            all_file_data.append((bucket, file_key))
            file_to_source[file_key] = source_name
    
    logger.info(f"Processing {len(all_file_data)} files across all sources with {max_processes} processes")
    
    # Process all files in mega-parallel
    results_by_source = {source: [] for source in source_files.keys()}
    
    with ProcessPoolExecutor(max_workers=max_processes, initializer=init_worker) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(process_file_chunks, file_data): file_data[1]
            for file_data in all_file_data
        }
        
        # Collect results with progress tracking
        with tqdm(total=len(all_file_data), desc="Processing all files") as pbar:
            for future in as_completed(future_to_file):
                file_key = future_to_file[future]
                try:
                    result = future.result(timeout=1800)  # 30 min timeout per file
                    source_name = file_to_source[file_key]
                    results_by_source[source_name].append(result)
                except Exception as e:
                    logger.error(f"Failed to process {file_key}: {str(e)}")
                    source_name = file_to_source[file_key]
                    error_result = ProcessingResult(
                        file_key=file_key,
                        word_count=0,
                        token_count=0,
                        processing_time=0,
                        error=str(e)
                    )
                    results_by_source[source_name].append(error_result)
                finally:
                    pbar.update(1)
    
    return results_by_source

def save_results_parallel(bucket: str, results_by_source: Dict[str, List[ProcessingResult]], 
                         token_counts_prefix: str, max_threads: int):
    """Save all results to S3 in parallel"""
    def save_source_results(source_data: Tuple[str, List[ProcessingResult]]) -> str:
        source_name, results = source_data
        
        # Create temporary file
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix=f'_{source_name}_counts.txt') as f:
            f.write("file_key,word_count,token_count,processing_time,error\n")
            for result in sorted(results, key=lambda x: x.file_key):
                f.write(f"{result.file_key},{result.word_count},{result.token_count},"
                       f"{result.processing_time:.2f},{result.error or ''}\n")
            temp_path = f.name
        
        # Upload to S3
        s3_client = create_optimized_s3_client()
        s3_key = f"{token_counts_prefix}/{source_name}_counts.txt"
        
        try:
            s3_client.upload_file(temp_path, bucket, s3_key)
            logger.info(f"Uploaded results for {source_name} to S3")
            
            # Calculate totals
            total_words = sum(r.word_count for r in results)
            total_tokens = sum(r.token_count for r in results)
            successful_files = len([r for r in results if r.error is None])
            
            logger.info(f"Source {source_name}: {successful_files}/{len(results)} files, "
                       f"{total_words:,} words, {total_tokens:,} tokens")
            
            return f"{source_name}: SUCCESS"
        
        except Exception as e:
            logger.error(f"Failed to upload results for {source_name}: {str(e)}")
            return f"{source_name}: FAILED - {str(e)}"
        
        finally:
            # Clean up temp file
            try:
                os.unlink(temp_path)
            except Exception:
                pass
    
    # Save all results in parallel
    with ThreadPoolExecutor(max_workers=max_threads) as executor:
        futures = [
            executor.submit(save_source_results, (source_name, results))
            for source_name, results in results_by_source.items()
        ]
        
        for future in as_completed(futures):
            result = future.result()
            logger.info(f"Save result: {result}")

def count_tokens_mega_parallel(bucket: str, base_prefix: str):
    """Ultra-optimized parallel processing using all available resources"""
    
    # Get optimal resource allocation
    max_processes, io_threads = get_optimal_process_count()
    
    token_counts_prefix = f"{base_prefix}token_counts"
    
    logger.info(f"Starting mega-parallel token counting")
    logger.info(f"Configuration: {max_processes} processes, {io_threads} I/O threads")
    logger.info(f"Batch size: {BATCH_SIZE}, Chunk size: {CHUNK_SIZE//1024//1024}MB")
    
    # Initialize tokenizer in main process
    logger.info("Pre-loading tokenizer...")
    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-30B-A3B", trust_remote_code=True)
    logger.info("Tokenizer loaded successfully")
    
    s3_client = create_optimized_s3_client()
    
    try:
        # Ensure output directory exists
        try:
            s3_client.put_object(Bucket=bucket, Key=f"{token_counts_prefix}/.placeholder", Body=b'')
        except Exception:
            pass
        
        # Check existing files
        existing_files = check_existing_output(s3_client, bucket, token_counts_prefix)
        
        # Get all source prefixes
        response = s3_client.list_objects_v2(Bucket=bucket, Prefix=base_prefix, Delimiter='/')
        prefixes = response.get('CommonPrefixes', [])
        
        if not prefixes:
            logger.warning(f"No source prefixes found under {base_prefix}")
            return
        
        # Filter sources to process
        sources_to_process = []
        for prefix in prefixes:
            source_prefix = prefix['Prefix']
            source_name = source_prefix.strip('/').split('/')[-1]
            if source_name not in existing_files and "token_counts" not in source_name:
                sources_to_process.append(source_prefix)
        
        if not sources_to_process:
            logger.info("All sources already processed!")
            return
        
        logger.info(f"Processing {len(sources_to_process)} sources")
        
        # Get all files in parallel
        logger.info("Discovering files...")
        start_time = time.time()
        source_files = get_all_files_parallel(bucket, sources_to_process, io_threads)
        discovery_time = time.time() - start_time
        
        total_files = sum(len(files) for files in source_files.values())
        logger.info(f"Found {total_files} files in {discovery_time:.2f}s")
        
        if total_files == 0:
            logger.warning("No files found to process")
            return
        
        # Process all files in mega-parallel
        logger.info("Starting mega-parallel processing...")
        processing_start = time.time()
        
        results_by_source = process_all_files_mega_parallel(
            bucket, source_files, max_processes, token_counts_prefix
        )
        
        processing_time = time.time() - processing_start
        logger.info(f"Processing completed in {processing_time:.2f}s")
        
        # Save results in parallel
        logger.info("Saving results...")
        save_start = time.time()
        
        save_results_parallel(bucket, results_by_source, token_counts_prefix, io_threads)
        
        save_time = time.time() - save_start
        total_time = time.time() - start_time
        
        # Final summary
        total_words = sum(
            sum(r.word_count for r in results)
            for results in results_by_source.values()
        )
        total_tokens = sum(
            sum(r.token_count for r in results)
            for results in results_by_source.values()
        )
        successful_files = sum(
            len([r for r in results if r.error is None])
            for results in results_by_source.values()
        )
        
        logger.info("="*80)
        logger.info("MEGA-PARALLEL PROCESSING COMPLETE")
        logger.info(f"Total time: {total_time:.2f}s")
        logger.info(f"File discovery: {discovery_time:.2f}s")
        logger.info(f"Processing: {processing_time:.2f}s ({successful_files/processing_time:.1f} files/sec)")
        logger.info(f"Saving: {save_time:.2f}s")
        logger.info(f"Files processed: {successful_files}/{total_files}")
        logger.info(f"Total words: {total_words:,}")
        logger.info(f"Total tokens: {total_tokens:,}")
        logger.info(f"Throughput: {total_tokens/processing_time:.0f} tokens/sec")
        logger.info("="*80)
        
    except Exception as e:
        logger.error(f"Error in mega-parallel processing: {str(e)}")
        raise

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
                    filename = key.split('/')[-1]
                    source_name = filename.replace('_counts.txt', '')
                    existing_files.add(source_name)
        
        if existing_files:
            logger.info(f"Found {len(existing_files)} existing output files")
        else:
            logger.info("No existing output files found")
            
    except ClientError as e:
        logger.warning(f"Error checking existing files: {str(e)}")
    
    return existing_files

if __name__ == "__main__":
    # Configuration
    bucket = 'gepeta-datasets'
    base_prefix = 'processed_cleaned_filtered/run_5/'
    
    # Run mega-parallel processing
    count_tokens_mega_parallel(bucket, base_prefix)