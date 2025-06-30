import boto3
import pandas as pd
import io
from .base_fetcher import BaseFetcher
from typing import List
from pathlib import Path
import time
from utils.logger import logger

class S3SourceFetcher(BaseFetcher):
    def __init__(self, bucket_name: str, prefix: str, source_name: str, output_prefix: str):
        super().__init__(source_name)
        self.s3 = boto3.client("s3")
        self.bucket_name = bucket_name
        self.prefix = prefix.rstrip("/") + "/"
        self.output_prefix = output_prefix
        logger.info(f"Initialized S3SourceFetcher")

    def get_files_to_process(self) -> List[str]:
        """
        Get list of S3 keys that need to be processed.
        """
        paginator = self.s3.get_paginator('list_objects_v2')
        csv_keys = []
        total_size = 0
        
        for page in paginator.paginate(Bucket=self.bucket_name, Prefix=self.prefix):
            for obj in page.get('Contents', []):
                key = obj['Key']
                size = obj['Size']
                # Fetch files that start with source_name and end with .csv
                filename = key.split('/')[-1]
                if filename.startswith(self.source_name) and filename.endswith('.csv'):
                    csv_keys.append(key)
                    total_size += size

        return csv_keys

    def fetch_single_file(self, file_path: str) -> pd.DataFrame:
        """
        Fetch data from a single S3 file.
        """
    
        try:
            # Get object metadata
            head_response = self.s3.head_object(Bucket=self.bucket_name, Key=file_path)            
            # Get object data
            response = self.s3.get_object(Bucket=self.bucket_name, Key=file_path)
            df = pd.read_csv(io.BytesIO(response["Body"].read()), header=None, names=["text", "n_words"])

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
                Bucket=self.bucket_name,
                Key=output_key,
                Body=csv_data,
                ContentType='text/csv'
            )
            
        except Exception as e:
            error_msg = f"Error saving cleaned data to s3://{self.bucket_name}/{output_key}: {str(e)}"
            logger.error(error_msg)
            self.stats['errors'].append(error_msg)