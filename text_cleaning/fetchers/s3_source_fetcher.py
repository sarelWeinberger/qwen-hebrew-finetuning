import boto3
import pandas as pd
import io
from .base_fetcher import BaseFetcher
from typing import List
from pathlib import Path
import time
from ..logger import logger

class S3SourceFetcher(BaseFetcher):
    def __init__(self, bucket_name: str, prefix: str, source_name: str, output_prefix: str):
        super().__init__(source_name)
        self.s3 = boto3.client("s3")
        self.bucket_name = bucket_name
        self.prefix = prefix.rstrip("/") + "/"
        self.output_prefix = output_prefix
        logger.info(f"Initialized S3SourceFetcher for bucket: {bucket_name}")
        logger.info(f"Input prefix: {self.prefix}")
        logger.info(f"Output prefix: {self.output_prefix}")

    def get_files_to_process(self) -> List[str]:
        """
        Get list of S3 keys that need to be processed.
        """
        paginator = self.s3.get_paginator('list_objects_v2')
        csv_keys = []
        total_size = 0

        logger.info(f"Listing objects in s3://{self.bucket_name}/{self.prefix}")
        
        for page in paginator.paginate(Bucket=self.bucket_name, Prefix=self.prefix):
            for obj in page.get('Contents', []):
                key = obj['Key']
                size = obj['Size']
                # Fetch files that start with source_name and end with .csv
                filename = key.split('/')[-1]
                if filename.startswith(self.source_name) and filename.endswith('.csv'):
                    csv_keys.append(key)
                    total_size += size
                    logger.debug(f"Found matching file: {key} (Size: {size} bytes)")

        logger.info(f"Found {len(csv_keys)} matching files (Total size: {total_size} bytes)")
        return csv_keys

    def fetch_single_file(self, file_path: str) -> pd.DataFrame:
        """
        Fetch data from a single S3 file.
        """
        start_time = time.time()
        file_stats = {
            'rows': 0,
            'size_bytes': 0,
            'processing_time': 0
        }
        
        try:
            # Get object metadata
            head_response = self.s3.head_object(Bucket=self.bucket_name, Key=file_path)
            file_stats['size_bytes'] = head_response['ContentLength']
            
            logger.info(f"Fetching s3://{self.bucket_name}/{file_path} (Size: {file_stats['size_bytes']} bytes)")
            
            # Get object data
            response = self.s3.get_object(Bucket=self.bucket_name, Key=file_path)
            df = pd.read_csv(io.BytesIO(response["Body"].read()))
            file_stats['rows'] = len(df)
            
            # Update statistics
            self.stats['total_files_processed'] += 1
            self.stats['total_rows_fetched'] += len(df)
            self.stats['total_bytes_read'] += file_stats['size_bytes']
            
            logger.info(f"Successfully fetched {len(df)} rows from s3://{self.bucket_name}/{file_path}")
            return df
            
        except Exception as e:
            error_msg = f"Error fetching data from s3://{self.bucket_name}/{file_path}: {str(e)}"
            logger.error(error_msg)
            self.stats['errors'].append(error_msg)
            return pd.DataFrame()
            
        finally:
            file_stats['processing_time'] = time.time() - start_time
            self.stats['file_stats'][file_path] = file_stats

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
            logger.info(f"Uploading {len(df)} rows to s3://{self.bucket_name}/{output_key}")
            self.s3.put_object(
                Bucket=self.bucket_name,
                Key=output_key,
                Body=csv_data,
                ContentType='text/csv'
            )
            
            logger.info(f"Successfully saved cleaned data to s3://{self.bucket_name}/{output_key}")
            logger.info(f"Output size: {len(csv_data)} bytes")
            
        except Exception as e:
            error_msg = f"Error saving cleaned data to s3://{self.bucket_name}/{output_key}: {str(e)}"
            logger.error(error_msg)
            self.stats['errors'].append(error_msg)