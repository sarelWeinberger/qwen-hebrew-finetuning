#!/usr/bin/env python3
"""
Example script for running space fixer analysis on a sample of data.
This script demonstrates how to use the new run_spacefix_sample_mode functionality.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cleaning_pipeline import CleaningPipeline
from fetchers.s3_source_fetcher import S3SourceFetcher
from cleaners.composite_cleaner import CompositeCleaner
from utils.logger import logger

def setup_spacefix_analysis_pipeline(source_name="test_source"):
    """
    Set up a cleaning pipeline for space fixer analysis.
    
    Args:
        source_name: Name of the data source to analyze
        
    Returns:
        CleaningPipeline: Configured pipeline for space fixer analysis
    """
    # Initialize S3 fetcher
    fetcher = S3SourceFetcher(
        bucket_name="gepeta-datasets",
        source_name=source_name,
        output_prefix="partly-processed/round_2_test_examples/spacefix_analysis"
    )
    
    # Initialize composite cleaner (we'll only use space fixer in the analysis)
    cleaner = CompositeCleaner()
    
    # Create pipeline
    pipeline = CleaningPipeline(fetcher, cleaner, source_name)
    
    return pipeline

def run_spacefix_analysis_example():
    """
    Example of running space fixer analysis on a sample.
    """
    logger.info("Starting space fixer analysis example...")
    
    # Set up pipeline
    pipeline = setup_spacefix_analysis_pipeline("test_source")
    
    # Run space fixer analysis on sample
    comparison_df = pipeline.run_spacefix_sample_mode(
        sample_size=50,  # Analyze 50 samples
        custom_output_prefix="partly-processed/round_2_test_examples/spacefix_analysis",
        custom_bucket_name="gepeta-datasets"
    )
    
    if not comparison_df.empty:
        logger.info("Space fixer analysis completed successfully!")
        logger.info(f"Analysis DataFrame shape: {comparison_df.shape}")
        
        # Display some example results
        print("\n=== Example Results ===")
        changed_samples = comparison_df[comparison_df['has_changes'] == True]
        if not changed_samples.empty:
            print("Sample of texts with changes:")
            for idx, row in changed_samples.head(3).iterrows():
                print(f"\nSample {row['sample_index']}:")
                print(f"Original: '{row['original_text'][:100]}...'")
                print(f"Fixed:    '{row['fixed_text'][:100]}...'")
                print(f"Spaces added: {row['spaces_added']}")
        else:
            print("No samples had changes in this analysis.")
            
        return comparison_df
    else:
        logger.warning("No data was processed in the analysis.")
        return None

def analyze_spacefix_results(comparison_df):
    """
    Perform additional analysis on the space fixer results.
    
    Args:
        comparison_df: DataFrame with space fixer comparison results
    """
    if comparison_df is None or comparison_df.empty:
        print("No data to analyze.")
        return
        
    print("\n=== Detailed Space Fixer Analysis ===")
    
    # Basic statistics
    total_samples = len(comparison_df)
    changed_samples = comparison_df['has_changes'].sum()
    unchanged_samples = total_samples - changed_samples
    
    print(f"Total samples: {total_samples}")
    print(f"Changed samples: {changed_samples} ({changed_samples/total_samples*100:.1f}%)")
    print(f"Unchanged samples: {unchanged_samples} ({unchanged_samples/total_samples*100:.1f}%)")
    
    # Space statistics
    total_spaces_added = comparison_df['spaces_added'].sum()
    avg_spaces_per_changed = comparison_df[comparison_df['has_changes']]['spaces_added'].mean()
    max_spaces_added = comparison_df['spaces_added'].max()
    
    print(f"\nSpace Statistics:")
    print(f"Total spaces added: {total_spaces_added}")
    print(f"Average spaces per changed sample: {avg_spaces_per_changed:.2f}")
    print(f"Maximum spaces added to a single sample: {max_spaces_added}")
    
    # Word count statistics
    total_word_changes = comparison_df['word_count_change'].sum()
    avg_word_change = comparison_df['word_count_change'].mean()
    
    print(f"\nWord Count Statistics:")
    print(f"Total word count change: {total_word_changes}")
    print(f"Average word count change per sample: {avg_word_change:.2f}")
    
    # Distribution of changes
    print(f"\nChange Distribution:")
    space_changes = comparison_df['spaces_added'].value_counts().sort_index()
    print("Spaces added distribution:")
    for spaces, count in space_changes.head(10).items():
        print(f"  {spaces} spaces: {count} samples")

if __name__ == "__main__":
    print("=== Space Fixer Analysis Example ===")
    
    # Run the analysis
    comparison_df = run_spacefix_analysis_example()
    
    # Perform additional analysis if we have results
    if comparison_df is not None:
        analyze_spacefix_results(comparison_df)
        
        # Save local copy for further analysis
        comparison_df.to_csv("spacefix_analysis_results.csv", index=False)
        print(f"\nDetailed results saved to: spacefix_analysis_results.csv")
    
    print("\n=== Analysis Complete ===") 