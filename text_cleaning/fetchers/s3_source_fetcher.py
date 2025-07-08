import boto3
import pandas as pd
import io
from .base_fetcher import BaseFetcher
from typing import List
from pathlib import Path
import time
from utils.logger import logger

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
                # Fetch files that start with source_name and end with .csv or .parquet
                filename = key.split('/')[-1]
                files_csv = filename.startswith(self.source_name) and filename.endswith('.csv')
                files_parquet = filename.startswith(self.source_name) and filename.endswith('.parquet')
                
                if files_csv or files_parquet:
                    # Check if cleaned version already exists
                    file_stem = Path(filename).stem
                    if file_stem not in existing_cleaned_files:
                        files_keys.append(key)
                        total_size += size
                    else:
                        logger.info(f"Skipping {filename} - cleaned version already exists")

        return files_keys

    def fetch_single_file(self, file_path: str) -> pd.DataFrame:
        """
        Fetch data from a single S3 file.
        """
    
        try:
            # Get object metadata
            head_response = self.s3.head_object(Bucket=self.bucket_name, Key=file_path)            
            # Get object data
            response = self.s3.get_object(Bucket=self.bucket_name, Key=file_path)
            if file_path.endswith('.csv'):
                df = pd.read_csv(io.BytesIO(response["Body"].read()), header=None, names=["text", "n_words"])
            elif file_path.endswith('.parquet'):
                df = pd.read_parquet(io.BytesIO(response["Body"].read()))

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