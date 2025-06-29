#!/usr/bin/env python3
"""
Example script demonstrating the enhanced cleaners with sample saving functionality.

This script shows how to use each cleaner with before/after sample saving
and how to configure the sample percentage.
"""

import pandas as pd
from cleaners.regex_cleaner import RegExCleaner
from cleaners.duplicate_remove_cleaner import DuplicateRemoverCleaner
from cleaners.quality_cleaner import QualityCleaner
from cleaners.composite_cleaner import CompositeCleaner
from utils.logger import logger
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

def create_sample_data():
    """Create sample data for testing."""
    sample_texts = [
        "This is a sample text with some HTML tags <b>bold</b> and <i>italic</i>.",
        "Another sample with duplicate lines.\nThis line appears twice.\nThis line appears twice.",
        "Text with special characters: @#$%^&*() and numbers 12345.",
        "Simple text without any special formatting.",
        "Text with multiple spaces    and    tabs\t\t\there.",
        "Another sample with duplicate lines.\nThis line appears twice.\nThis line appears twice.",
        "Text with Hebrew characters: שלום עולם!",
        "Text with mixed content: English עברית 123 @#$%",
        "Very short text.",
        "This is a very long text that contains many words and should be processed by the quality cleaner to calculate various metrics about the text quality and characteristics."
    ]
    
    df = pd.DataFrame({
        'text': sample_texts,
        'n_words': [len(text.split()) for text in sample_texts]
    })
    
    return df

def example_regex_cleaner():
    """Example of using RegExCleaner with sample saving."""
    print("\n=== RegExCleaner Example ===")
    
    # Define regex patterns
    patterns = [
        (r'<[^>]+>', ''),  # Remove HTML tags
        (r'\s+', ' '),     # Replace multiple whitespace with single space
        (r'[^\w\s]', ''),  # Remove special characters
    ]
    
    # Create cleaner with 10% sampling
    cleaner = RegExCleaner(
        patterns=patterns,
        save_samples=True,
        sample_percentage=0.1  # 10% sampling
    )
    
    # Create sample data
    df = create_sample_data()
    print(f"Original data: {len(df)} rows")
    
    # Clean the data
    cleaned_df = cleaner.clean(df, file_name="regex_example")
    
    print(f"Cleaned data: {len(cleaned_df)} rows")
    print("Sample saving enabled - check text_cleaning/samples/RegExCleaner/")
    
    return cleaned_df

def example_duplicate_remover():
    """Example of using DuplicateRemoverCleaner with sample saving."""
    print("\n=== DuplicateRemoverCleaner Example ===")
    
    # Create cleaner with 15% sampling
    cleaner = DuplicateRemoverCleaner(
        save_samples=True,
        sample_percentage=0.15  # 15% sampling
    )
    
    # Create sample data
    df = create_sample_data()
    print(f"Original data: {len(df)} rows")
    
    # Clean the data
    cleaned_df = cleaner.clean(df, file_name="duplicate_remover_example")
    
    print(f"Cleaned data: {len(cleaned_df)} rows")
    print("Sample saving enabled - check text_cleaning/samples/DuplicateRemoverCleaner/")
    
    return cleaned_df

def example_quality_cleaner():
    """Example of using QualityCleaner with sample saving."""
    print("\n=== QualityCleaner Example ===")
    
    # Create cleaner with 5% sampling (default)
    cleaner = QualityCleaner(
        save_samples=True,
        sample_percentage=0.05  # 5% sampling
    )
    
    # Add a custom metric
    def word_length_metric(text):
        words = text.split()
        if not words:
            return 0
        return sum(len(word) for word in words) / len(words)
    
    cleaner.add_metric("avg_word_length", word_length_metric)
    
    # Create sample data
    df = create_sample_data()
    print(f"Original data: {len(df)} rows")
    
    # Clean the data
    cleaned_df = cleaner.clean(df, file_name="quality_example")
    
    print(f"Cleaned data: {len(cleaned_df)} rows")
    print("Sample saving enabled - check text_cleaning/samples/QualityCleaner/")
    
    return cleaned_df

def example_composite_cleaner():
    """Example of using CompositeCleaner with sample saving."""
    print("\n=== CompositeCleaner Example ===")
    
    # Create individual cleaners
    regex_patterns = [
        (r'<[^>]+>', ''),  # Remove HTML tags
        (r'\s+', ' '),     # Replace multiple whitespace with single space
    ]
    
    regex_cleaner = RegExCleaner(
        patterns=regex_patterns,
        save_samples=True,
        sample_percentage=0.1
    )
    
    duplicate_cleaner = DuplicateRemoverCleaner(
        save_samples=True,
        sample_percentage=0.1
    )
    
    quality_cleaner = QualityCleaner(
        save_samples=True,
        sample_percentage=0.1
    )
    
    # Create composite cleaner
    composite_cleaner = CompositeCleaner(
        cleaners=[regex_cleaner, duplicate_cleaner, quality_cleaner],
        save_samples=True,
        sample_percentage=0.1
    )
    
    # Create sample data
    df = create_sample_data()
    print(f"Original data: {len(df)} rows")
    
    # Clean the data
    cleaned_df = composite_cleaner.clean(df, file_name="composite_example")
    
    print(f"Cleaned data: {len(cleaned_df)} rows")
    print("Sample saving enabled for each step - check text_cleaning/samples/")
    
    return cleaned_df

def example_disable_sample_saving():
    """Example of disabling sample saving."""
    print("\n=== Disable Sample Saving Example ===")
    
    # Create cleaner with sample saving disabled
    cleaner = RegExCleaner(
        patterns=[(r'\s+', ' ')],
        save_samples=False  # Disable sample saving
    )
    
    # Create sample data
    df = create_sample_data()
    print(f"Original data: {len(df)} rows")
    
    # Clean the data
    cleaned_df = cleaner.clean(df, file_name="no_samples_example")
    
    print(f"Cleaned data: {len(cleaned_df)} rows")
    print("Sample saving disabled")
    
    return cleaned_df

def example_change_sample_percentage():
    """Example of changing sample percentage after creation."""
    print("\n=== Change Sample Percentage Example ===")
    
    # Create cleaner with default 5% sampling
    cleaner = RegExCleaner(
        patterns=[(r'\s+', ' ')],
        save_samples=True,
        sample_percentage=0.05
    )
    
    # Change to 20% sampling
    cleaner.set_sample_saving(enabled=True, sample_percentage=0.2)
    
    # Create sample data
    df = create_sample_data()
    print(f"Original data: {len(df)} rows")
    
    # Clean the data
    cleaned_df = cleaner.clean(df, file_name="changed_percentage_example")
    
    print(f"Cleaned data: {len(cleaned_df)} rows")
    print("Sample saving with 20% sampling - check text_cleaning/samples/RegExCleaner/")
    
    return cleaned_df

def main():
    """Run all examples."""
    print("Enhanced Cleaners with Sample Saving - Examples")
    print("=" * 60)
    
    # Run examples
    example_regex_cleaner()
    example_duplicate_remover()
    example_quality_cleaner()
    example_composite_cleaner()
    example_disable_sample_saving()
    example_change_sample_percentage()
    
    print("\n" + "=" * 60)
    print("All examples completed!")
    print("\nSample files have been saved to:")
    print("- text_cleaning/samples/before/ - Original samples")
    print("- text_cleaning/samples/after/ - Cleaned samples")
    print("- text_cleaning/samples/comparison/ - Side-by-side comparisons")
    print("\nEach cleaner has its own subdirectory for organization.")

if __name__ == "__main__":
    main() 