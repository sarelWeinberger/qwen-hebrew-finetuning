#!/usr/bin/env python3
"""
Example script for using SpaceFixCleaner with tracking functionality.
This script demonstrates how to enable tracking and get before/after comparison data.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cleaners.spacefix_cleaner import SpaceFixCleaner
from fetchers.s3_source_fetcher import S3SourceFetcher
import pandas as pd
from utils.logger import logger

def run_spacefix_with_tracking_example():
    """
    Example of running space fixer with tracking enabled.
    """
    logger.info("Starting space fixer with tracking example...")
    
    # Initialize S3 fetcher
    fetcher = S3SourceFetcher(
        bucket_name="gepeta-datasets",
        source_name="test_source",
        output_prefix="partly-processed/round_2_test_examples/spacefix_tracking"
    )
    
    # Get files to process
    files = fetcher.get_files_to_process()
    if not files:
        logger.warning("No files found for processing.")
        return None
    
    # Take first few files for example
    sample_files = files[:3]  # Process first 3 files
    logger.info(f"Processing {len(sample_files)} files for space fixer tracking...")
    
    # Initialize space fixer with tracking enabled
    space_fixer = SpaceFixCleaner(enable_tracking=True)
    
    all_tracking_data = []
    
    for file_path in sample_files:
        logger.info(f"Processing file: {file_path}")
        
        # Fetch data
        df = fetcher.fetch_single_file(file_path)
        if df.empty:
            logger.warning(f"Skipped empty file: {file_path}")
            continue
            
        if 'text' not in df.columns:
            logger.warning(f"File {file_path} missing 'text' column, skipping.")
            continue
        
        # Take a small sample for demonstration
        sample_df = df.sample(n=min(20, len(df)), random_state=42) if len(df) > 20 else df.copy()
        
        # Clear previous tracking data
        space_fixer.clear_tracking_data()
        
        # Process the data (this will collect tracking data)
        cleaned_df = space_fixer.clean(sample_df, file_name=file_path)
        
        # Get tracking data for this file
        file_tracking_data = space_fixer.get_tracking_data()
        if not file_tracking_data.empty:
            all_tracking_data.append(file_tracking_data)
            logger.info(f"Collected tracking data for {len(file_tracking_data)} samples from {file_path}")
    
    # Combine all tracking data
    if all_tracking_data:
        combined_tracking_df = pd.concat(all_tracking_data, ignore_index=True)
        
        # Save to S3
        fetcher.save_cleaned_data(combined_tracking_df, "spacefix_tracking_analysis", "spacefix_tracking_results.csv")
        
        # Print summary
        print(f"\n=== Space Fixer Tracking Results ===")
        print(f"Total samples processed: {len(combined_tracking_df)}")
        
        changed_samples = combined_tracking_df['has_changes'].sum()
        print(f"Samples with changes: {changed_samples} ({changed_samples/len(combined_tracking_df)*100:.1f}%)")
        
        total_spaces_added = combined_tracking_df['spaces_added'].sum()
        print(f"Total spaces added: {total_spaces_added}")
        
        if changed_samples > 0:
            avg_spaces_per_changed = combined_tracking_df[combined_tracking_df['has_changes']]['spaces_added'].mean()
            print(f"Average spaces per changed sample: {avg_spaces_per_changed:.2f}")
        
        # Show some examples
        print(f"\n=== Example Changes ===")
        changed_examples = combined_tracking_df[combined_tracking_df['has_changes'] == True]
        if not changed_examples.empty:
            for idx, row in changed_examples.head(3).iterrows():
                print(f"\nSample from {row['file_name']}:")
                print(f"Original: '{row['original_text'][:80]}...'")
                print(f"Fixed:    '{row['fixed_text'][:80]}...'")
                print(f"Spaces added: {row['spaces_added']}")
        
        # Save local copy
        combined_tracking_df.to_csv("spacefix_tracking_results.csv", index=False)
        print(f"\nDetailed results saved to: spacefix_tracking_results.csv")
        
        return combined_tracking_df
    else:
        logger.warning("No tracking data was collected.")
        return None

def analyze_tracking_results(tracking_df):
    """
    Perform detailed analysis on the tracking results.
    
    Args:
        tracking_df: DataFrame with space fixer tracking results
    """
    if tracking_df is None or tracking_df.empty:
        print("No data to analyze.")
        return
        
    print(f"\n=== Detailed Tracking Analysis ===")
    
    # File-level analysis
    print("Analysis by file:")
    file_stats = tracking_df.groupby('file_name').agg({
        'has_changes': ['count', 'sum'],
        'spaces_added': ['sum', 'mean'],
        'word_count_change': ['sum', 'mean']
    }).round(2)
    
    print(file_stats)
    
    # Change distribution
    print(f"\nChange Distribution:")
    space_changes = tracking_df['spaces_added'].value_counts().sort_index()
    print("Spaces added distribution:")
    for spaces, count in space_changes.head(10).items():
        print(f"  {spaces} spaces: {count} samples")
    
    # Text length analysis
    tracking_df['original_length'] = tracking_df['original_text'].str.len()
    tracking_df['fixed_length'] = tracking_df['fixed_text'].str.len()
    tracking_df['length_change'] = tracking_df['fixed_length'] - tracking_df['original_length']
    
    print(f"\nLength Analysis:")
    print(f"Average original text length: {tracking_df['original_length'].mean():.1f}")
    print(f"Average fixed text length: {tracking_df['fixed_length'].mean():.1f}")
    print(f"Average length change: {tracking_df['length_change'].mean():.1f}")

if __name__ == "__main__":
    print("=== Space Fixer Tracking Example ===")
    
    # Run the tracking example
    tracking_df = run_spacefix_with_tracking_example()
    
    # Perform detailed analysis if we have results
    if tracking_df is not None:
        analyze_tracking_results(tracking_df)
    
    print("\n=== Tracking Analysis Complete ===") 