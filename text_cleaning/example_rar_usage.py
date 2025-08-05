#!/usr/bin/env python3
"""
Example script demonstrating how to use the expanded S3 fetcher with RAR files.
This script shows how to set up and run a cleaning pipeline for RAR files containing JSONL data.
"""

import os
import sys
from pathlib import Path

# Add the text_cleaning directory to the path
sys.path.append(str(Path(__file__).parent))

from fetchers.s3_source_fetcher import S3SourceFetcher
from cleaners.composite_cleaner import CompositeCleaner
from cleaning_pipeline import CleaningPipeline
from utils.logger import logger

def setup_rar_cleaning_pipeline():
    """
    Set up a cleaning pipeline for RAR files containing JSONL data.
    
    This example assumes you have RAR files in S3 with the following structure:
    - Bucket: your-data-bucket
    - Prefix: raw-data/
    - Files: source_name_*.rar (containing JSONL files inside)
    """
    
    # S3 configuration for input data (RAR files)
    input_bucket = "your-data-bucket"
    input_prefix = "raw-data/"
    source_name = "hebrew_corpus"  # Your source name prefix
    
    # S3 configuration for output (cleaned data)
    output_bucket = "your-cleaned-data-bucket"
    output_prefix = "cleaned-data/"
    
    # Create the S3 fetcher
    fetcher = S3SourceFetcher(
        bucket_name=input_bucket,
        prefix=input_prefix,
        source_name=source_name,
        output_prefix=output_prefix,
        output_bucket_name=output_bucket
    )
    
    # Create a composite cleaner (you can customize this based on your needs)
    cleaner = CompositeCleaner()
    
    # Create the cleaning pipeline
    pipeline = CleaningPipeline(
        fetcher=fetcher,
        cleaner=cleaner,
        source_name=source_name
    )
    
    return pipeline

def run_sample_mode_example():
    """
    Example of running the pipeline in sample mode to test with a small subset.
    """
    logger.info("Setting up RAR cleaning pipeline...")
    
    try:
        pipeline = setup_rar_cleaning_pipeline()
        
        logger.info("Running in sample mode to test with small subset...")
        
        # Run in sample mode - this will:
        # 1. Find up to 10 RAR files in S3
        # 2. Extract JSONL data from each RAR file
        # 3. Sample ~100 total texts across all files
        # 4. Clean the sampled data
        # 5. Save as {source_name}.csv to S3
        pipeline.run_sample_mode(
            custom_output_prefix="test-output/",
            custom_bucket_name="your-test-bucket"
        )
        
        logger.info("Sample mode completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in sample mode: {str(e)}")
        raise

def run_full_pipeline_example():
    """
    Example of running the full pipeline on all RAR files.
    """
    logger.info("Setting up RAR cleaning pipeline...")
    
    try:
        pipeline = setup_rar_cleaning_pipeline()
        
        logger.info("Running full pipeline on all RAR files...")
        
        # Run the full pipeline - this will:
        # 1. Find all RAR files in S3 matching the source_name pattern
        # 2. Extract JSONL data from each RAR file
        # 3. Clean all the data
        # 4. Save cleaned data as {original_filename}_cleaned.csv to S3
        pipeline.run()
        
        logger.info("Full pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in full pipeline: {str(e)}")
        raise

def check_rar_files_in_s3():
    """
    Example of checking what RAR files are available in S3.
    """
    logger.info("Checking RAR files in S3...")
    
    try:
        pipeline = setup_rar_cleaning_pipeline()
        files = pipeline.fetcher.get_files_to_process()
        
        if files:
            logger.info(f"Found {len(files)} RAR files to process:")
            for i, file_path in enumerate(files, 1):
                logger.info(f"  {i}. {file_path}")
        else:
            logger.info("No RAR files found matching the criteria.")
            
    except Exception as e:
        logger.error(f"Error checking RAR files: {str(e)}")
        raise

if __name__ == "__main__":
    print("=== RAR File Cleaning Pipeline Example ===\n")
    
    # Check what RAR files are available
    check_rar_files_in_s3()
    print()
    
    # Ask user what to do
    print("Choose an option:")
    print("1. Run sample mode (test with small subset)")
    print("2. Run full pipeline (process all files)")
    print("3. Just check available files (already done above)")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        run_sample_mode_example()
    elif choice == "2":
        run_full_pipeline_example()
    elif choice == "3":
        print("File check already completed above.")
    else:
        print("Invalid choice. Exiting.") 