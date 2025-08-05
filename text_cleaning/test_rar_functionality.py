#!/usr/bin/env python3
"""
Test script to verify RAR file handling functionality in the S3 fetcher.
This script tests the RAR extraction and JSONL parsing capabilities.
"""

import tempfile
import os
import json
import pandas as pd
from fetchers.s3_source_fetcher import S3SourceFetcher

def create_test_rar_with_jsonl():
    """
    Create a test RAR file with JSONL content for testing purposes.
    This is a helper function for testing - in real usage, you'd have actual RAR files in S3.
    """
    try:
        import rarfile
        
        # Create test JSONL data
        test_data = [
            {"text": "This is the first test sentence.", "id": 1, "category": "test"},
            {"text": "This is the second test sentence with more words.", "id": 2, "category": "test"},
            {"text": "Third sentence for testing purposes.", "id": 3, "category": "sample"}
        ]
        
        # Create temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create JSONL file
            jsonl_path = os.path.join(temp_dir, "test_data.jsonl")
            with open(jsonl_path, 'w', encoding='utf-8') as f:
                for item in test_data:
                    f.write(json.dumps(item, ensure_ascii=False) + '\n')
            
            # Create RAR file
            rar_path = os.path.join(temp_dir, "test_data.rar")
            with rarfile.RarFile(rar_path, 'w') as rf:
                rf.write(jsonl_path, "test_data.jsonl")
            
            # Read the RAR file back
            with open(rar_path, 'rb') as f:
                rar_data = f.read()
            
            return rar_data
            
    except ImportError:
        print("rarfile module not available. Please install it: pip install rarfile")
        return None
    except Exception as e:
        print(f"Error creating test RAR file: {e}")
        return None

def test_rar_extraction():
    """
    Test the RAR extraction functionality.
    """
    print("Testing RAR extraction functionality...")
    
    # Create test RAR data
    rar_data = create_test_rar_with_jsonl()
    if rar_data is None:
        print("Could not create test RAR data. Skipping test.")
        return False
    
    try:
        # Create a mock S3 fetcher instance (we won't actually use S3)
        fetcher = S3SourceFetcher(
            bucket_name="test-bucket",
            prefix="test-prefix",
            source_name="test-source",
            output_prefix="test-output",
            output_bucket_name="test-output-bucket"
        )
        
        # Test the RAR extraction method
        df = fetcher._extract_rar_and_read_jsonl(rar_data)
        
        if df.empty:
            print("❌ RAR extraction failed - returned empty DataFrame")
            return False
        
        print(f"✅ RAR extraction successful! Extracted {len(df)} rows")
        print(f"DataFrame columns: {list(df.columns)}")
        print(f"First few rows:")
        print(df.head())
        
        # Verify we have the expected columns
        if 'text' not in df.columns:
            print("❌ Missing 'text' column in extracted data")
            return False
        
        if 'n_words' not in df.columns:
            print("❌ Missing 'n_words' column in extracted data")
            return False
        
        print("✅ All required columns present")
        return True
        
    except Exception as e:
        print(f"❌ Error during RAR extraction test: {e}")
        return False

def test_jsonl_reading():
    """
    Test the JSONL reading functionality.
    """
    print("Testing JSONL reading functionality...")
    
    try:
        # Create a mock S3 fetcher instance
        fetcher = S3SourceFetcher(
            bucket_name="test-bucket",
            prefix="test-prefix",
            source_name="test-source",
            output_prefix="test-output",
            output_bucket_name="test-output-bucket"
        )
        
        # Test with valid JSONL data
        valid_jsonl = b'{"text": "First line", "id": 1}\n{"text": "Second line", "id": 2}\n{"text": "Third line", "id": 3}'
        
        # Test JSONL reading
        df = fetcher._read_jsonl_data(valid_jsonl)
        
        if df.empty:
            print("❌ JSONL reading failed - returned empty DataFrame")
            return False
        
        print(f"✅ JSONL reading successful! Extracted {len(df)} rows")
        print(f"DataFrame columns: {list(df.columns)}")
        
        # Verify we have the expected columns
        if 'text' not in df.columns:
            print("❌ Missing 'text' column in JSONL data")
            return False
        
        if 'n_words' not in df.columns:
            print("❌ Missing 'n_words' column in JSONL data")
            return False
        
        print("✅ All required columns present in JSONL data")
        return True
        
    except Exception as e:
        print(f"❌ Error during JSONL reading test: {e}")
        return False

def test_system_dependencies():
    """
    Test if system dependencies for RAR extraction are available.
    """
    print("Checking system dependencies for RAR extraction...")
    
    try:
        import rarfile
        
        # Check if unrar is available
        if rarfile.UNRAR_TOOL:
            print(f"✅ unrar tool found at: {rarfile.UNRAR_TOOL}")
        else:
            print("⚠️  unrar tool not found. You may need to install it:")
            print("   - On macOS: brew install unrar")
            print("   - On Ubuntu/Debian: sudo apt-get install unrar")
            print("   - On CentOS/RHEL: sudo yum install unrar")
        
        return True
        
    except ImportError:
        print("❌ rarfile module not available. Please install it:")
        print("   pip install rarfile")
        return False

if __name__ == "__main__":
    print("=== RAR and JSONL Functionality Test ===\n")
    
    # Test system dependencies
    deps_ok = test_system_dependencies()
    print()
    
    if deps_ok:
        # Test RAR extraction
        extraction_ok = test_rar_extraction()
        print()
        
        # Test JSONL reading
        jsonl_ok = test_jsonl_reading()
        print()
        
        if extraction_ok and jsonl_ok:
            print("🎉 All tests passed! RAR and JSONL functionality is working correctly.")
        else:
            print("❌ Some tests failed.")
    else:
        print("❌ System dependencies not met. Please install required packages.") 