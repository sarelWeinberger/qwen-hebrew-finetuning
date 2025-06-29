#!/usr/bin/env python3
"""
Example usage of the RAR to CSV converter.

This script demonstrates how to use the RARToCSVConverter class
to convert RAR files from S3 to CSV files.
"""

from rar_to_csv_converter import RARToCSVConverter
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

def example_basic_usage():
    """
    Basic usage example.
    """
    print("=== Basic Usage Example ===")
    
    # Initialize converter
    converter = RARToCSVConverter(
        bucket_name="your-bucket-name",
        rar_file_key="path/to/your/file.rar",
        output_prefix="csv_output"
    )
    
    # Convert with default 100MB chunk size
    try:
        uploaded_files = converter.convert(max_size_mb=100)
        print(f"Successfully uploaded {len(uploaded_files)} files")
    except Exception as e:
        print(f"Error: {e}")

def example_custom_chunk_size():
    """
    Example with custom chunk size.
    """
    print("\n=== Custom Chunk Size Example ===")
    
    converter = RARToCSVConverter(
        bucket_name="your-bucket-name",
        rar_file_key="path/to/your/file.rar",
        output_prefix="csv_output_50mb"
    )
    
    # Convert with 50MB chunk size
    try:
        uploaded_files = converter.convert(max_size_mb=50)
        print(f"Successfully uploaded {len(uploaded_files)} files with 50MB chunks")
    except Exception as e:
        print(f"Error: {e}")

def example_with_different_output_prefix():
    """
    Example with different output prefix.
    """
    print("\n=== Different Output Prefix Example ===")
    
    converter = RARToCSVConverter(
        bucket_name="your-bucket-name",
        rar_file_key="path/to/your/file.rar",
        output_prefix="processed_data/hebrew_texts"
    )
    
    try:
        uploaded_files = converter.convert(max_size_mb=100)
        print(f"Successfully uploaded {len(uploaded_files)} files to processed_data/hebrew_texts/")
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    print("RAR to CSV Converter - Example Usage")
    print("=" * 50)
    
    # Note: These examples require actual S3 bucket and file paths
    # Uncomment and modify the examples below with your actual values
    
    # example_basic_usage()
    # example_custom_chunk_size()
    # example_with_different_output_prefix()
    
    print("\nTo use these examples:")
    print("1. Replace 'your-bucket-name' with your actual S3 bucket name")
    print("2. Replace 'path/to/your/file.rar' with the actual S3 key of your RAR file")
    print("3. Uncomment the example function calls above")
    print("4. Make sure you have AWS credentials configured")
    print("5. Install the required dependencies: pip install -r requirements.txt")
    print("6. Install unrar: brew install unrar (on macOS) or apt-get install unrar (on Ubuntu)") 