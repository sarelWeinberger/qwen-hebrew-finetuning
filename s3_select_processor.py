#!/usr/bin/env python3
"""
S3 Select JSONL to CSV Processor

This program processes JSONL files directly in S3 using S3 Select,
converting them to CSV without downloading the files.
"""

import boto3
import pandas as pd
import json
import io
from typing import List, Dict, Any, Generator
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class S3SelectProcessor:
    def __init__(self, bucket_name: str, input_prefix: str, output_prefix: str = "csv_output"):
        """
        Initialize the S3 Select processor.
        
        Args:
            bucket_name: S3 bucket name
            input_prefix: S3 prefix containing JSONL files
            output_prefix: S3 prefix for output CSV files
        """
        self.s3_client = boto3.client('s3')
        self.bucket_name = bucket_name
        self.input_prefix = input_prefix.rstrip('/')
        self.output_prefix = output_prefix.rstrip('/')
        
        logger.info(f"Initialized S3 Select processor for s3://{bucket_name}/{input_prefix}")
        logger.info(f"Output prefix: {output_prefix}")
    
    def list_jsonl_files(self) -> List[str]:
        """
        List all JSONL files in the input prefix.
        
        Returns:
            List of S3 keys for JSONL files
        """
        logger.info(f"Listing JSONL files in s3://{self.bucket_name}/{self.input_prefix}")
        
        jsonl_files = []
        paginator = self.s3_client.get_paginator('list_objects_v2')
        
        for page in paginator.paginate(Bucket=self.bucket_name, Prefix=self.input_prefix):
            if 'Contents' in page:
                for obj in page['Contents']:
                    if obj['Key'].endswith('.jsonl'):
                        jsonl_files.append(obj['Key'])
        
        logger.info(f"Found {len(jsonl_files)} JSONL files")
        return jsonl_files
    
    def process_jsonl_with_s3_select(self, jsonl_key: str, max_size_mb: int = 100) -> List[str]:
        """
        Process a JSONL file using S3 Select and convert to CSV chunks.
        
        Args:
            jsonl_key: S3 key of the JSONL file
            max_size_mb: Maximum size in MB for each CSV file
            
        Returns:
            List of S3 keys for uploaded CSV files
        """
        logger.info(f"Processing {jsonl_key} with S3 Select")
        
        # S3 Select query to extract text and calculate word count
        # Note: S3 Select has limitations with JSON parsing, so we'll use a simpler approach
        query = """
        SELECT 
            s.text,
            s.n_words
        FROM s3object s
        WHERE s.text IS NOT NULL
        """
        
        uploaded_files = []
        chunk_index = 0
        current_chunk = []
        current_size = 0
        max_bytes = max_size_mb * 1024 * 1024
        
        try:
            # Use S3 Select to process the file in chunks
            response = self.s3_client.select_object_content(
                Bucket=self.bucket_name,
                Key=jsonl_key,
                Expression=query,
                ExpressionType='SQL',
                InputSerialization={
                    'JSON': {
                        'Type': 'LINES'
                    }
                },
                OutputSerialization={
                    'CSV': {
                        'FieldDelimiter': ',',
                        'RecordDelimiter': '\n'
                    }
                }
            )
            
            # Process the streaming response
            for event in response['Payload']:
                if 'Records' in event:
                    records = event['Records']['Payload'].decode('utf-8')
                    
                    # Process each line
                    for line in records.split('\n'):
                        if line.strip():
                            # Parse CSV line back to dict
                            parts = line.split(',', 1)  # Split on first comma only
                            if len(parts) == 2:
                                text = parts[0].strip('"')
                                n_words = int(parts[1])
                                
                                record = {
                                    'text': text,
                                    'n_words': n_words
                                }
                                
                                # Estimate record size
                                record_size = len(json.dumps(record)) + 50
                                
                                # Check if we need to upload current chunk
                                if current_size + record_size > max_bytes and current_chunk:
                                    output_key = self.upload_csv_chunk(current_chunk, chunk_index, jsonl_key)
                                    if output_key:
                                        uploaded_files.append(output_key)
                                    current_chunk = []
                                    current_size = 0
                                    chunk_index += 1
                                
                                current_chunk.append(record)
                                current_size += record_size
            
            # Upload final chunk if any
            if current_chunk:
                output_key = self.upload_csv_chunk(current_chunk, chunk_index, jsonl_key)
                if output_key:
                    uploaded_files.append(output_key)
                    
        except Exception as e:
            logger.error(f"Error processing {jsonl_key} with S3 Select: {str(e)}")
            # Fallback to traditional processing if S3 Select fails
            logger.info("Falling back to traditional processing method")
            return self.process_jsonl_traditional(jsonl_key, max_size_mb)
        
        logger.info(f"Processed {jsonl_key}, uploaded {len(uploaded_files)} CSV files")
        return uploaded_files
    
    def process_jsonl_traditional(self, jsonl_key: str, max_size_mb: int = 100) -> List[str]:
        """
        Fallback method to process JSONL file by streaming it.
        
        Args:
            jsonl_key: S3 key of the JSONL file
            max_size_mb: Maximum size in MB for each CSV file
            
        Returns:
            List of S3 keys for uploaded CSV files
        """
        logger.info(f"Processing {jsonl_key} with traditional streaming method")
        
        uploaded_files = []
        chunk_index = 0
        current_chunk = []
        current_size = 0
        max_bytes = max_size_mb * 1024 * 1024
        
        try:
            # Stream the JSONL file from S3
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=jsonl_key)
            
            for line in response['Body'].iter_lines():
                if line:
                    try:
                        json_data = json.loads(line.decode('utf-8'))
                        
                        # Extract text field
                        text = None
                        if 'text' in json_data:
                            text = json_data['text']
                        elif 'content' in json_data:
                            text = json_data['content']
                        elif 'message' in json_data:
                            text = json_data['message']
                        else:
                            # Try to find text field
                            for key, value in json_data.items():
                                if isinstance(value, str) and len(value) > 10:
                                    text = value
                                    break
                        
                        if text:
                            n_words = len(text.split())
                            record = {
                                'text': text,
                                'n_words': n_words
                            }
                            
                            # Estimate record size
                            record_size = len(json.dumps(record)) + 50
                            
                            # Check if we need to upload current chunk
                            if current_size + record_size > max_bytes and current_chunk:
                                output_key = self.upload_csv_chunk(current_chunk, chunk_index, jsonl_key)
                                if output_key:
                                    uploaded_files.append(output_key)
                                current_chunk = []
                                current_size = 0
                                chunk_index += 1
                            
                            current_chunk.append(record)
                            current_size += record_size
                            
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON in {jsonl_key}: {str(e)}")
                        continue
            
            # Upload final chunk if any
            if current_chunk:
                output_key = self.upload_csv_chunk(current_chunk, chunk_index, jsonl_key)
                if output_key:
                    uploaded_files.append(output_key)
                    
        except Exception as e:
            logger.error(f"Error processing {jsonl_key}: {str(e)}")
            raise
        
        logger.info(f"Processed {jsonl_key}, uploaded {len(uploaded_files)} CSV files")
        return uploaded_files
    
    def upload_csv_chunk(self, chunk: List[Dict[str, Any]], chunk_index: int, source_file: str) -> str:
        """
        Convert a data chunk to CSV and upload to S3.
        
        Args:
            chunk: List of data dictionaries
            chunk_index: Index of the chunk for naming
            source_file: Source file name for naming
            
        Returns:
            S3 key of the uploaded file
        """
        if not chunk:
            logger.warning(f"Empty chunk {chunk_index}, skipping upload")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(chunk)
        
        # Create CSV in memory
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        csv_data = csv_buffer.getvalue()
        
        # Generate output filename
        source_name = Path(source_file).stem
        output_filename = f"{source_name}_chunk_{chunk_index:04d}.csv"
        output_key = f"{self.output_prefix}/{output_filename}"
        
        try:
            # Upload to S3
            logger.info(f"Uploading chunk {chunk_index} with {len(chunk)} records to s3://{self.bucket_name}/{output_key}")
            
            self.s3_client.put_object(
                Bucket=self.bucket_name,
                Key=output_key,
                Body=csv_data,
                ContentType='text/csv'
            )
            
            file_size_mb = len(csv_data) / (1024 * 1024)
            logger.info(f"Successfully uploaded {output_key} ({file_size_mb:.2f}MB)")
            
            return output_key
            
        except Exception as e:
            logger.error(f"Error uploading chunk {chunk_index}: {str(e)}")
            raise
    
    def process_all_files(self, max_size_mb: int = 100) -> List[str]:
        """
        Process all JSONL files in the input prefix.
        
        Args:
            max_size_mb: Maximum size in MB for each CSV file
            
        Returns:
            List of all uploaded CSV file keys
        """
        logger.info("Starting processing of all JSONL files")
        
        jsonl_files = self.list_jsonl_files()
        if not jsonl_files:
            logger.warning("No JSONL files found to process")
            return []
        
        all_uploaded_files = []
        
        for jsonl_file in tqdm(jsonl_files, desc="Processing files"):
            uploaded_files = self.process_jsonl_with_s3_select(jsonl_file, max_size_mb)
            all_uploaded_files.extend(uploaded_files)
        
        logger.info(f"Completed processing. Total uploaded files: {len(all_uploaded_files)}")
        return all_uploaded_files

if __name__ == "__main__":
    processor = S3SelectProcessor(
        bucket_name='israllm-datasets',
        input_prefix='raw-datasets/jsonl',  # Assuming JSONL files are already extracted
        output_prefix='israllm-datasets/raw-datasets/csv_output'
    )
    uploaded_files = processor.process_all_files() 