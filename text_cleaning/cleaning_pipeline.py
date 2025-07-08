# pipelines/cleaning_pipeline.py
import pandas as pd
import numpy as np
from pathlib import Path
from utils.logger import logger
import random

class CleaningPipeline:
    def __init__(self, fetcher, cleaner, source_name: str):
        self.fetcher = fetcher
        self.cleaner = cleaner
        self.source_name = source_name

    def run(self):
        # Get list of files to process
        files = self.fetcher.get_files_to_process()
        total_files = len(files)
        
        logger.info(f"Starting to process {total_files} files...")
        
        for i, file_path in enumerate(files, 1):
            # Process single file
            df = self.fetcher.fetch_single_file(file_path)
            if not df.empty:
                # Extract file name for sample naming
                file_name = Path(file_path).stem
                
                # Clean the data (this will handle sample saving if enabled)
                df = self.cleaner.clean(df)
                
                # Save cleaned data
                self.fetcher.save_cleaned_data(df, self.source_name, file_path)
                logger.info(f"Done processing file {i}/{total_files}")
            else:
                logger.warning(f"Skipped empty file {i}/{total_files}")
        
        logger.info(f"Cleaning pipeline completed for all {total_files} files.")

    def run_sample_mode(self, custom_output_prefix=None, custom_bucket_name=None):
        """
        Sample mode: randomly select up to 10 files, sample ceil(100/n) texts from each, combine, clean, and save as {data_source}.csv to S3.
        custom_output_prefix: if provided, use this as the S3 output prefix (path)
        custom_bucket_name: if provided, use this as the S3 bucket name
        """
        files = self.fetcher.get_files_to_process()
        if not files:
            logger.warning("No files found for sampling.")
            return
        n_files = min(10, len(files))
        selected_files = random.sample(files, n_files)
        n_per_file = int(np.ceil(100 / n_files))
        logger.info(f"Sample mode: {n_files} files selected, {n_per_file} samples per file (total ≈ 100)")
        print("Selected files:")
        for f in selected_files:
            print(f"  {f}")
        sampled_rows = []
        for file_path in selected_files:
            print(f"Processing file: {file_path}")
            df = self.fetcher.fetch_single_file(file_path)
            if df.empty:
                print(f"  Skipped empty file: {file_path}")
                continue
            if 'text' not in df.columns:
                logger.warning(f"File {file_path} missing 'text' column, skipping.")
                continue
            sample_n = min(n_per_file, len(df))
            sampled = df.sample(n=sample_n, random_state=42) if len(df) > sample_n else df.copy()
            sampled_rows.append(sampled[['text']].copy())
        if not sampled_rows:
            logger.warning("No samples collected from any file.")
            return
        sample_df = pd.concat(sampled_rows, ignore_index=True)
        logger.info(f"Collected {len(sample_df)} samples. Running cleaner...")
        cleaned_df = self.cleaner.clean(sample_df.copy())  # No file_name argument
        # Save only original_text and cleaned_text columns
        result_df = pd.DataFrame({
            'original_text': sample_df['text'],
            'cleaned_text': cleaned_df['text'] if 'text' in cleaned_df.columns else cleaned_df.iloc[:,0]
        })
        # Print S3 bucket and output prefix
        bucket = custom_bucket_name if custom_bucket_name else getattr(self.fetcher, 'bucket_name', None)
        output_prefix = custom_output_prefix if custom_output_prefix else getattr(self.fetcher, 'output_prefix', None)
        print(f"Saving to S3 bucket: {bucket}, path: {output_prefix}")
        # Save to S3 as {data_source}.csv, using custom_output_prefix and custom_bucket_name if provided
        old_prefix = self.fetcher.output_prefix
        old_bucket = self.fetcher.bucket_name
        if custom_output_prefix:
            self.fetcher.output_prefix = custom_output_prefix
        if custom_bucket_name:
            self.fetcher.bucket_name = custom_bucket_name
        self.fetcher.save_cleaned_data(result_df, self.source_name, f"{self.source_name}.csv")
        # Restore
        self.fetcher.output_prefix = old_prefix
        self.fetcher.bucket_name = old_bucket
        logger.info(f"Sampled and cleaned data saved as {self.source_name}.csv to S3.")