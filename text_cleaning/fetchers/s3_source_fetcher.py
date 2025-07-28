import boto3
import pandas as pd
import io
from .base_fetcher import BaseFetcher
from typing import List
from pathlib import Path
import time
from utils.logger import logger
import json
import tempfile
import os
import rarfile

class S3SourceFetcher(BaseFetcher):
    def __init__(self, bucket_name: str, prefix: str, source_name: str, output_prefix: str, 
                 output_bucket_name: str):
        super().__init__(source_name)
        self.s3 = boto3.client("s3")
        self.bucket_name = bucket_name
        self.prefix = prefix.rstrip("/") + "/"
        self.output_prefix = output_prefix
        self.output_bucket_name = output_bucket_name
        logger.info(f"Initialized S3SourceFetcher")

    def get_files_to_process(self) -> List[str]:
        """
        Get list of S3 keys that need to be processed.
        Excludes files that already have cleaned versions in the output directory.
        Supports .jsonl files (direct JSONL), .rar files (containing JSONL), and .csv files.
        """
        paginator = self.s3.get_paginator('list_objects_v2')
        files_keys = []
        total_size = 0
        
        # Get list of existing cleaned files in output directory
        existing_cleaned_files = set()
        try:
            output_paginator = self.s3.get_paginator('list_objects_v2')
            for page in output_paginator.paginate(Bucket=self.output_bucket_name, Prefix=self.output_prefix):
                for obj in page.get('Contents', []):
                    output_key = obj['Key']
                    # Extract the original filename from cleaned filename
                    # cleaned filename format: original_stem_cleaned.csv
                    output_filename = output_key.split('/')[-1]
                    if output_filename.endswith('_cleaned.csv'):
                        original_stem = output_filename.replace('_cleaned.csv', '')
                        existing_cleaned_files.add(original_stem)
        except Exception as e:
            logger.warning(f"Could not check existing cleaned files: {str(e)}")
        
        for page in paginator.paginate(Bucket=self.bucket_name, Prefix=self.prefix):
            for obj in page.get('Contents', []):
                key = obj['Key']
                size = obj['Size']
                # Fetch files that start with source_name and end with .jsonl, .rar, or .csv
                filename = key.split('/')[-1]
                files_jsonl = filename.startswith(self.source_name) and filename.endswith('.jsonl')
                files_rar = filename.startswith(self.source_name) and filename.endswith('.rar')
                files_csv = filename.startswith(self.source_name) and filename.endswith('.csv')
                
                if files_jsonl or files_rar or files_csv:
                    # Check if cleaned version already exists
                    file_stem = Path(filename).stem
                    if file_stem not in existing_cleaned_files:
                        files_keys.append(key)
                        total_size += size
                    else:
                        logger.info(f"Skipping {filename} - cleaned version already exists")

        return files_keys

    def _read_jsonl_data_streaming(self, response_body) -> pd.DataFrame:
        """
        Read JSONL data line by line from S3 response body.
        
        Args:
            response_body: S3 response body object
            
        Returns:
            DataFrame with parsed JSONL data
        """
        try:
            all_data = []
            line_num = 0
            
            # Read line by line from the response body
            for line in response_body.iter_lines():
                line_num += 1
                if line:  # Skip empty lines
                    try:
                        # Decode the line
                        line_text = line.decode('utf-8').strip()
                        if line_text:  # Skip empty lines after decoding
                            data = json.loads(line_text)
                            all_data.append(data)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON at line {line_num}: {e}")
                    except Exception as e:
                        logger.warning(f"Error processing line {line_num}: {e}")
            
            if not all_data:
                logger.warning("No valid JSON data found in file")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(all_data)
            
            # If the JSONL contains text data, ensure we have a 'text' column
            if 'text' not in df.columns:
                # Try to find a suitable text column
                text_columns = [col for col in df.columns if 'text' in col.lower() or 'content' in col.lower()]
                if text_columns:
                    df['text'] = df[text_columns[0]]
                else:
                    # If no text column found, use the first column as text
                    df['text'] = df.iloc[:, 0].astype(str)
            
            # Add word count column if not present
            if 'n_words' not in df.columns:
                df['n_words'] = df['text'].str.split().str.len()
            
            return df
            
        except Exception as e:
            logger.error(f"Error reading JSONL data: {str(e)}")
            return pd.DataFrame()

    def _read_jsonl_data(self, file_data: bytes) -> pd.DataFrame:
        """
        Read JSONL data from bytes (fallback method).
        
        Args:
            file_data: Raw bytes of the JSONL file
            
        Returns:
            DataFrame with parsed JSONL data
        """
        try:
            text = file_data.decode('utf-8')
            all_data = []
            
            for line_num, line in enumerate(text.split('\n'), 1):
                line = line.strip()
                if line:  # Skip empty lines
                    try:
                        data = json.loads(line)
                        all_data.append(data)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Invalid JSON at line {line_num}: {e}")
                    except Exception as e:
                        logger.warning(f"Error processing line {line_num}: {e}")
            
            if not all_data:
                logger.warning("No valid JSON data found in file")
                return pd.DataFrame()
            
            # Convert to DataFrame
            df = pd.DataFrame(all_data)
            
            # If the JSONL contains text data, ensure we have a 'text' column
            if 'text' not in df.columns:
                # Try to find a suitable text column
                text_columns = [col for col in df.columns if 'text' in col.lower() or 'content' in col.lower()]
                if text_columns:
                    df['text'] = df[text_columns[0]]
                else:
                    # If no text column found, use the first column as text
                    df['text'] = df.iloc[:, 0].astype(str)
            
            # Add word count column if not present
            if 'n_words' not in df.columns:
                df['n_words'] = df['text'].str.split().str.len()
            
            return df
            
        except Exception as e:
            logger.error(f"Error reading JSONL data: {str(e)}")
            return pd.DataFrame()

    def _extract_rar_and_read_jsonl(self, rar_data: bytes) -> pd.DataFrame:
        """
        Extract RAR file and read JSONL content from it.
        
        Args:
            rar_data: Raw bytes of the RAR file
            
        Returns:
            DataFrame with extracted JSONL data
        """
        try:
            # Create a temporary directory for extraction
            with tempfile.TemporaryDirectory() as temp_dir:
                # Save RAR file to temporary location
                rar_path = os.path.join(temp_dir, "temp.rar")
                with open(rar_path, 'wb') as f:
                    f.write(rar_data)
                
                # Extract RAR file
                with rarfile.RarFile(rar_path, 'r') as rf:
                    # Find JSONL files in the archive
                    jsonl_files = [f for f in rf.namelist() if f.endswith('.jsonl')]
                    
                    if not jsonl_files:
                        logger.warning("No JSONL files found in RAR archive")
                        return pd.DataFrame()
                    
                    all_data = []
                    
                    for jsonl_file in jsonl_files:
                        logger.info(f"Processing JSONL file: {jsonl_file}")
                        
                        # Read JSONL content
                        with rf.open(jsonl_file) as f:
                            for line_num, line in enumerate(f, 1):
                                try:
                                    line = line.decode('utf-8').strip()
                                    if line:  # Skip empty lines
                                        data = json.loads(line)
                                        all_data.append(data)
                                except json.JSONDecodeError as e:
                                    logger.warning(f"Invalid JSON at line {line_num} in {jsonl_file}: {e}")
                                except Exception as e:
                                    logger.warning(f"Error processing line {line_num} in {jsonl_file}: {e}")
                    
                    if not all_data:
                        logger.warning("No valid JSON data found in RAR archive")
                        return pd.DataFrame()
                    
                    # Convert to DataFrame
                    df = pd.DataFrame(all_data)
                    
                    # If the JSONL contains text data, ensure we have a 'text' column
                    if 'text' not in df.columns:
                        # Try to find a suitable text column
                        text_columns = [col for col in df.columns if 'text' in col.lower() or 'content' in col.lower()]
                        if text_columns:
                            df['text'] = df[text_columns[0]]
                        else:
                            # If no text column found, use the first column as text
                            df['text'] = df.iloc[:, 0].astype(str)
                    
                    # Add word count column if not present
                    if 'n_words' not in df.columns:
                        df['n_words'] = df['text'].str.split().str.len()
                    
                    return df
                    
        except Exception as e:
            logger.error(f"Error extracting RAR file: {str(e)}")
            return pd.DataFrame()

    def fetch_single_file(self, file_path: str) -> pd.DataFrame:
        """
        Fetch data from a single S3 file.
        Supports .jsonl files (direct JSONL), .rar files (containing JSONL), and .csv files.
        Uses streaming for JSONL files to improve performance.
        """
    
        try:
            # Get object data
            response = self.s3.get_object(Bucket=self.bucket_name, Key=file_path)
            
            if file_path.endswith('.jsonl'):
                # Handle extracted JSONL files directly using streaming
                logger.info(f"Processing JSONL file with streaming: {file_path}")
                df = self._read_jsonl_data_streaming(response["Body"])
            elif file_path.endswith('.rar'):
                # Handle RAR files containing JSONL data
                logger.info(f"Processing RAR file: {file_path}")
                file_data = response["Body"].read()
                df = self._extract_rar_and_read_jsonl(file_data)
            elif file_path.endswith('.csv'):
                # Handle CSV files
                logger.info(f"Processing CSV file: {file_path}")
                df = pd.read_csv(io.BytesIO(response["Body"].read()), header=None, names=["text", "n_words"])
            else:
                logger.warning(f"Unsupported file format: {file_path}")
                return pd.DataFrame()

            return df
            
        except Exception as e:
            error_msg = f"Error fetching data from s3://{self.bucket_name}/{file_path}: {str(e)}"
            logger.error(error_msg)
            self.stats['errors'].append(error_msg)
            return pd.DataFrame()

    def save_cleaned_data(self, df: pd.DataFrame, source_name: str, original_file_path: str):
        """
        Save cleaned data to S3, preserving the original filename structure.
        """
        try:
            # Get the original filename and create the output key
            original_filename = original_file_path.split('/')[-1]
            output_filename = f"{Path(original_filename).stem}_cleaned.csv"
            output_key = f"{self.output_prefix.rstrip('/')}/{output_filename}"
            
            # Prepare data for upload
            csv_buffer = io.StringIO()
            df.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()
            
            # Upload to S3
            self.s3.put_object(
                Bucket=self.output_bucket_name,
                Key=output_key,
                Body=csv_data,
                ContentType='text/csv'
            )
            
        except Exception as e:
            error_msg = f"Error saving cleaned data to s3://{self.bucket_name}/{output_key}: {str(e)}"
            logger.error(error_msg)
            self.stats['errors'].append(error_msg)