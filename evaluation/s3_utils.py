import boto3
import os
import tempfile
import json
from botocore.exceptions import ClientError

def parse_s3_path(s3_path):
    """Parse S3 path into bucket and key components."""
    if not s3_path.startswith('s3://'):
        raise ValueError("S3 path must start with 's3://'")
    
    path_parts = s3_path[5:].split('/', 1)  # Remove 's3://' and split
    bucket = path_parts[0]
    key = path_parts[1] if len(path_parts) > 1 else ''
    
    return bucket, key

def download_s3_directory(s3_path, local_dir):
    """
    Download an entire S3 directory to a local directory.
    
    Args:
        s3_path (str): S3 path in format 's3://bucket/key/'
        local_dir (str): Local directory path where files will be downloaded
    
    Returns:
        str: Path to the local directory containing downloaded files
    """
    bucket, key_prefix = parse_s3_path(s3_path)
    
    # Ensure key_prefix ends with '/' for directory listing
    if key_prefix and not key_prefix.endswith('/'):
        key_prefix += '/'
    
    s3_client = boto3.client('s3')
    
    try:
        # Create local directory if it doesn't exist
        os.makedirs(local_dir, exist_ok=True)
        
        # List all objects with the given prefix
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=bucket, Prefix=key_prefix)
        
        downloaded_files = []
        
        for page in pages:
            if 'Contents' in page:
                for obj in page['Contents']:
                    s3_key = obj['Key']
                    
                    # Skip if it's just a directory marker
                    if s3_key.endswith('/') or s3_key.endswith('csv'):
                        continue
                    
                    # Calculate relative path from the prefix
                    relative_path = s3_key[len(key_prefix):]
                    local_file_path = os.path.join(local_dir, relative_path)
                    
                    # Create subdirectories if needed
                    local_file_dir = os.path.dirname(local_file_path)
                    if local_file_dir:
                        os.makedirs(local_file_dir, exist_ok=True)
                    
                    # Download the file
                    s3_client.download_file(bucket, s3_key, local_file_path)
                    downloaded_files.append(local_file_path)
        
        print(f"Downloaded {len(downloaded_files)} files from {s3_path} to {local_dir}")
        return local_dir
        
    except ClientError as e:
        print(f"Error downloading from S3: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error: {e}")
        raise

def create_temp_directory():
    """Create a temporary directory for S3 downloads."""
    return tempfile.mkdtemp(prefix="s3_benchmark_")

def cleanup_temp_directory(temp_dir):
    """Clean up temporary directory."""
    import shutil
    try:
        shutil.rmtree(temp_dir)
        print(f"Cleaned up temporary directory: {temp_dir}")
    except Exception as e:
        print(f"Warning: Could not clean up temporary directory {temp_dir}: {e}")

def is_s3_path(path):
    """Check if a path is an S3 path."""
    return isinstance(path, str) and path.startswith('s3://')

def test_s3_connection():
    """Test S3 connection and credentials."""
    try:
        s3_client = boto3.client('s3')
        s3_client.list_buckets()
        print("S3 connection successful")
        return True
    except Exception as e:
        print(f"S3 connection failed: {e}")
        return False

if __name__ == "__main__":
    # Test the S3 connection
    test_s3_connection()
