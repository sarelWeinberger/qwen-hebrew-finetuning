#!/usr/bin/env python3
"""
RAR to CSV Converter

This program downloads a RAR file from S3, extracts JSONL files from it,
and converts them to CSV files with a maximum size of 100MB each.
The CSV files contain 'text' and 'n_words' columns.
"""

import boto3
import pandas as pd
import rarfile
import json
import io
import os
import tempfile
from pathlib import Path
from typing import List, Dict, Any
import logging
from tqdm import tqdm
import math

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RARToCSVConverter:
    def __init__(self, bucket_name: str, rar_file_key: str, output_prefix: str = "csv_output"):
        """
        Initialize the RAR to CSV converter.
        
        Args:
            bucket_name: S3 bucket name containing the RAR file
            rar_file_key: S3 key of the RAR file
            output_prefix: S3 prefix for output CSV files
        """
        self.s3_client = boto3.client('s3')
        self.bucket_name = bucket_name
        self.rar_file_key = rar_file_key
        self.output_prefix = output_prefix.rstrip('/')
        
        # Set rarfile to use unrar executable (needs to be installed)
        rarfile.UNRAR_TOOL = "unrar"
        
        logger.info(f"Initialized converter for s3://{bucket_name}/{rar_file_key}")
        logger.info(f"Output prefix: {output_prefix}")
    
    def download_rar_file(self) -> str:
        """
        Download the RAR file from S3 to a temporary location.
        
        Returns:
            Path to the downloaded RAR file
        """
        logger.info(f"Downloading RAR file from s3://{self.bucket_name}/{self.rar_file_key}")
        
        # Create temporary file
        temp_dir = tempfile.mkdtemp()
        rar_file_path = os.path.join(temp_dir, "data.rar")
        
        try:
            # Download file with progress bar
            response = self.s3_client.get_object(Bucket=self.bucket_name, Key=self.rar_file_key)
            file_size = response['ContentLength']
            
            with open(rar_file_path, 'wb') as f:
                with tqdm(total=file_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                    for chunk in response['Body'].iter_chunks(chunk_size=8192):
                        f.write(chunk)
                        pbar.update(len(chunk))
            
            logger.info(f"Successfully downloaded RAR file to {rar_file_path}")
            return rar_file_path
            
        except Exception as e:
            logger.error(f"Error downloading RAR file: {str(e)}")
            raise
    
    def extract_jsonl_files(self, rar_file_path: str) -> List[str]:
        """
        Extract JSONL files from the RAR archive.
        
        Args:
            rar_file_path: Path to the RAR file
            
        Returns:
            List of paths to extracted JSONL files
        """
        logger.info(f"Extracting files from {rar_file_path}")
        
        temp_dir = os.path.dirname(rar_file_path)
        extracted_files = []
        
        try:
            with rarfile.RarFile(rar_file_path, 'r') as rf:
                # List all files in the archive
                file_list = rf.namelist()
                logger.info(f"Found {len(file_list)} files in RAR archive")
                
                # Extract JSONL files
                for file_info in rf.infolist():
                    if file_info.filename.endswith('.jsonl'):
                        logger.info(f"Extracting {file_info.filename}")
                        rf.extract(file_info, temp_dir)
                        extracted_path = os.path.join(temp_dir, file_info.filename)
                        extracted_files.append(extracted_path)
                
                if not extracted_files:
                    logger.warning("No JSONL files found in RAR archive")
                    
        except Exception as e:
            logger.error(f"Error extracting RAR file: {str(e)}")
            raise
        
        logger.info(f"Extracted {len(extracted_files)} JSONL files")
        return extracted_files
    
    def process_jsonl_file(self, jsonl_file_path: str) -> List[Dict[str, Any]]:
        """
        Process a JSONL file and extract text and word count.
        
        Args:
            jsonl_file_path: Path to the JSONL file
            
        Returns:
            List of dictionaries with 'text' and 'n_words' keys
        """
        logger.info(f"Processing JSONL file: {jsonl_file_path}")
        
        data = []
        line_count = 0
        
        try:
            with open(jsonl_file_path, 'r', encoding='utf-8') as f:
                for line in tqdm(f, desc="Processing JSONL"):
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        json_data = json.loads(line)
                        
                        # Extract text field (handle different possible field names)
                        text = None
                        if 'text' in json_data:
                            text = json_data['text']
                        elif 'content' in json_data:
                            text = json_data['content']
                        elif 'message' in json_data:
                            text = json_data['message']
                        else:
                            # If no obvious text field, try to find one
                            for key, value in json_data.items():
                                if isinstance(value, str) and len(value) > 10:
                                    text = value
                                    break
                        
                        if text:
                            # Calculate word count
                            n_words = len(text.split())
                            
                            data.append({
                                'text': text,
                                'n_words': n_words
                            })
                        
                        line_count += 1
                        
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON on line {line_count}: {str(e)}")
                        continue
                        
        except Exception as e:
            logger.error(f"Error processing JSONL file {jsonl_file_path}: {str(e)}")
            raise
        
        logger.info(f"Processed {line_count} lines, extracted {len(data)} valid records")
        return data
    
    def split_data_into_chunks(self, data: List[Dict[str, Any]], max_size_mb: int = 100) -> List[List[Dict[str, Any]]]:
        """
        Split data into chunks that will result in CSV files of approximately max_size_mb.
        
        Args:
            data: List of data dictionaries
            max_size_mb: Maximum size in MB for each CSV file
            
        Returns:
            List of data chunks
        """
        logger.info(f"Splitting {len(data)} records into ~{max_size_mb}MB chunks")
        
        if not data:
            return []
        
        # Estimate size per record (rough approximation)
        sample_df = pd.DataFrame(data[:1000])
        sample_csv = sample_df.to_csv(index=False)
        bytes_per_record = len(sample_csv) / len(sample_df)
        
        # Calculate records per chunk
        max_bytes = max_size_mb * 1024 * 1024
        records_per_chunk = int(max_bytes / bytes_per_record)
        
        # Split data into chunks
        chunks = []
        for i in range(0, len(data), records_per_chunk):
            chunk = data[i:i + records_per_chunk]
            chunks.append(chunk)
        
        logger.info(f"Split data into {len(chunks)} chunks of ~{records_per_chunk} records each")
        return chunks
    
    def upload_csv_chunk(self, chunk: List[Dict[str, Any]], chunk_index: int) -> str:
        """
        Convert a data chunk to CSV and upload to S3.
        
        Args:
            chunk: List of data dictionaries
            chunk_index: Index of the chunk for naming
            
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
        output_filename = f"chunk_{chunk_index:04d}.csv"
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
    
    def convert(self, max_size_mb: int = 100) -> List[str]:
        """
        Main conversion method.
        
        Args:
            max_size_mb: Maximum size in MB for each CSV file
            
        Returns:
            List of S3 keys for uploaded CSV files
        """
        logger.info("Starting RAR to CSV conversion process")
        
        try:
            # Step 1: Download RAR file
            rar_file_path = self.download_rar_file()
            
            # Step 2: Extract JSONL files
            jsonl_files = self.extract_jsonl_files(rar_file_path)
            
            if not jsonl_files:
                logger.error("No JSONL files found to process")
                return []
            
            # Step 3: Process all JSONL files
            all_data = []
            for jsonl_file in jsonl_files:
                data = self.process_jsonl_file(jsonl_file)
                all_data.extend(data)
            
            logger.info(f"Total records extracted: {len(all_data)}")
            
            # Step 4: Split data into chunks
            chunks = self.split_data_into_chunks(all_data, max_size_mb)
            
            # Step 5: Upload chunks as CSV files
            uploaded_files = []
            for i, chunk in enumerate(chunks):
                output_key = self.upload_csv_chunk(chunk, i)
                if output_key:
                    uploaded_files.append(output_key)
            
            # Cleanup temporary files
            try:
                os.remove(rar_file_path)
                for jsonl_file in jsonl_files:
                    if os.path.exists(jsonl_file):
                        os.remove(jsonl_file)
                os.rmdir(os.path.dirname(rar_file_path))
            except Exception as e:
                logger.warning(f"Error cleaning up temporary files: {str(e)}")
            
            logger.info(f"Conversion completed successfully. Uploaded {len(uploaded_files)} CSV files.")
            return uploaded_files
            
        except Exception as e:
            logger.error(f"Error during conversion: {str(e)}")
            raise


def main():
    """
    Main function with example usage.
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Convert RAR file from S3 to CSV files')
    parser.add_argument('--bucket', required=True, help='S3 bucket name')
    parser.add_argument('--rar-key', required=True, help='S3 key of the RAR file')
    parser.add_argument('--output-prefix', default='csv_output', help='S3 prefix for output files')
    parser.add_argument('--max-size-mb', type=int, default=100, help='Maximum size in MB for each CSV file')
    
    args = parser.parse_args()
    
    # Create converter and run conversion
    converter = RARToCSVConverter(
        bucket_name=args.bucket,
        rar_file_key=args.rar_key,
        output_prefix=args.output_prefix
    )
    
    try:
        uploaded_files = converter.convert(max_size_mb=args.max_size_mb)
        print(f"\nConversion completed! Uploaded {len(uploaded_files)} files:")
        for file_key in uploaded_files:
            print(f"  s3://{args.bucket}/{file_key}")
    except Exception as e:
        print(f"Error: {str(e)}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main()) 