#!/usr/bin/env python3
"""
Test script for RAR to CSV conversion with text cleaning sampling.

This script demonstrates how to test the cleaning pipeline on different sample sizes
before running on the full dataset.
"""

from rar_to_csv_with_cleaning import RARToCSVWithCleaning
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def test_small_sample():
    """Test with a small sample (1000 records) to verify the pipeline works."""
    print("🧪 TEST 1: Small Sample (1000 records)")
    print("=" * 50)
    
    converter = RARToCSVWithCleaning(
        bucket_name='israllm-datasets',
        rar_file_key='raw-datasets/rar/AllHebNLIFiles.tar.gz',
        output_prefix='israllm-datasets/raw-datasets/rar/test_small_sample'
    )
    
    try:
        uploaded_files = converter.convert(
            sample_size=1000,  # Only process 1000 records
            enable_cleaning=True,
            cleaning_sample_percentage=0.1  # Save 10% of samples for inspection
        )
        
        print(f"✅ Small sample test completed successfully!")
        print(f"📁 Uploaded {len(uploaded_files)} files")
        print("📋 Check the samples in text_cleaning/samples/ for cleaning results")
        
    except Exception as e:
        print(f"❌ Small sample test failed: {str(e)}")

def test_medium_sample():
    """Test with a medium sample (10000 records) to get better statistics."""
    print("\n🧪 TEST 2: Medium Sample (10000 records)")
    print("=" * 50)
    
    converter = RARToCSVWithCleaning(
        bucket_name='israllm-datasets',
        rar_file_key='raw-datasets/rar/AllHebNLIFiles.tar.gz',
        output_prefix='israllm-datasets/raw-datasets/rar/test_medium_sample'
    )
    
    try:
        uploaded_files = converter.convert(
            sample_size=10000,  # Process 10000 records
            enable_cleaning=True,
            cleaning_sample_percentage=0.05  # Save 5% of samples
        )
        
        print(f"✅ Medium sample test completed successfully!")
        print(f"📁 Uploaded {len(uploaded_files)} files")
        print("📊 This gives better statistics for cleaning quality assessment")
        
    except Exception as e:
        print(f"❌ Medium sample test failed: {str(e)}")

def test_without_cleaning():
    """Test without cleaning to compare results."""
    print("\n🧪 TEST 3: Without Cleaning (for comparison)")
    print("=" * 50)
    
    converter = RARToCSVWithCleaning(
        bucket_name='israllm-datasets',
        rar_file_key='raw-datasets/rar/AllHebNLIFiles.tar.gz',
        output_prefix='israllm-datasets/raw-datasets/rar/test_no_cleaning'
    )
    
    try:
        uploaded_files = converter.convert(
            sample_size=5000,  # Process 5000 records
            enable_cleaning=False  # No cleaning
        )
        
        print(f"✅ No cleaning test completed successfully!")
        print(f"📁 Uploaded {len(uploaded_files)} files")
        print("📊 Compare this with cleaned results to see the difference")
        
    except Exception as e:
        print(f"❌ No cleaning test failed: {str(e)}")

def test_custom_cleaning():
    """Test with custom cleaning parameters."""
    print("\n🧪 TEST 4: Custom Cleaning Parameters")
    print("=" * 50)
    
    converter = RARToCSVWithCleaning(
        bucket_name='israllm-datasets',
        rar_file_key='raw-datasets/rar/AllHebNLIFiles.tar.gz',
        output_prefix='israllm-datasets/raw-datasets/rar/test_custom_cleaning'
    )
    
    try:
        uploaded_files = converter.convert(
            sample_size=3000,  # Process 3000 records
            enable_cleaning=True,
            cleaning_sample_percentage=0.15  # Save 15% of samples for detailed inspection
        )
        
        print(f"✅ Custom cleaning test completed successfully!")
        print(f"📁 Uploaded {len(uploaded_files)} files")
        print("📋 Higher sample percentage for detailed cleaning analysis")
        
    except Exception as e:
        print(f"❌ Custom cleaning test failed: {str(e)}")

def run_all_tests():
    """Run all test scenarios."""
    print("🚀 Starting RAR to CSV Cleaning Tests")
    print("This will test different sample sizes and cleaning configurations")
    print("=" * 60)
    
    # Run tests in order of increasing complexity
    test_small_sample()
    test_without_cleaning()
    test_medium_sample()
    test_custom_cleaning()
    
    print("\n" + "=" * 60)
    print("🎉 All tests completed!")
    print("\n📋 Next steps:")
    print("1. Check the uploaded CSV files in S3")
    print("2. Review cleaning samples in text_cleaning/samples/")
    print("3. Compare results between different configurations")
    print("4. Choose the best configuration for full processing")

if __name__ == "__main__":
    # You can run individual tests or all tests
    run_all_tests()
    
    # Or run individual tests:
    # test_small_sample()
    # test_medium_sample()
    # test_without_cleaning()
    # test_custom_cleaning() 