import boto3
import os
import tempfile
import json
import re
from botocore.exceptions import ClientError

def sanitize_filename(filename):
    """
    Sanitize filename to be compatible with Windows and other OS.
    Replaces invalid characters with underscores.
    
    Args:
        filename (str): Original filename
        
    Returns:
        str: Sanitized filename safe for Windows and Unix systems
    """
    # Characters that are invalid in Windows filenames
    invalid_chars = r'[<>:"|?*]'
    
    # Replace invalid characters with underscore
    sanitized = re.sub(invalid_chars, '_', filename)
    
    # Also handle pipe character which can cause issues
    sanitized = sanitized.replace('|', '_')
    
    # Remove any trailing dots or spaces (invalid in Windows)
    sanitized = sanitized.rstrip('. ')
    
    return sanitized

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
    Compatible with Windows and Unix systems.
    
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
                    
                    # Split the path into components and sanitize each part
                    path_parts = relative_path.split('/')
                    sanitized_parts = [sanitize_filename(part) for part in path_parts]
                    
                    # Reconstruct the path using os.path.join for OS compatibility
                    local_file_path = os.path.join(local_dir, *sanitized_parts)
                    
                    # Create subdirectories if needed
                    local_file_dir = os.path.dirname(local_file_path)
                    if local_file_dir:
                        os.makedirs(local_file_dir, exist_ok=True)
                    
                    # Download the file
                    try:
                        s3_client.download_file(bucket, s3_key, local_file_path)
                        downloaded_files.append(local_file_path)
                        print(f"Downloaded: {s3_key} -> {local_file_path}")
                    except Exception as e:
                        print(f"Error downloading {s3_key}: {e}")
                        continue
        
        print(f"Successfully downloaded {len(downloaded_files)} files from {s3_path} to {local_dir}")
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

def download_s3_file(s3_path, local_path=None, s3_client=None):
    """
    Download a single S3 object to a local file.
    Compatible with Windows and Unix systems.

    Args:
        s3_path (str): S3 path in format 's3://bucket/key'
        local_path (str|None): Local file path or directory. If None, uses basename(key) in CWD.
        s3_client (boto3.client|None): Optional boto3 S3 client to use.

    Returns:
        str: Path to the downloaded local file.
    """
    bucket, key = parse_s3_path(s3_path)

    if not key or key.endswith('/'):
        raise ValueError("S3 path must point to a file (not a bucket or directory).")

    if s3_client is None:
        s3_client = boto3.client('s3')

    # Determine local path
    if local_path is None:
        # Sanitize the basename for Windows compatibility
        basename = os.path.basename(key) or "downloaded_s3_object"
        local_path = sanitize_filename(basename)

    # If local_path is a directory path (ends with separator) or an existing dir, place file inside it
    if local_path.endswith(os.sep) or local_path.endswith('/') or os.path.isdir(local_path):
        os.makedirs(local_path, exist_ok=True)
        basename = sanitize_filename(os.path.basename(key))
        local_path = os.path.join(local_path, basename)
    else:
        # Sanitize the full path
        dir_path = os.path.dirname(local_path)
        file_name = os.path.basename(local_path)
        sanitized_file_name = sanitize_filename(file_name)
        local_path = os.path.join(dir_path, sanitized_file_name) if dir_path else sanitized_file_name

    # Ensure parent directories exist
    parent_dir = os.path.dirname(local_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    try:
        s3_client.download_file(bucket, key, local_path)
        print(f"Downloaded s3://{bucket}/{key} -> {local_path}")
        return local_path
    except ClientError as e:
        print(f"Error downloading s3://{bucket}/{key}: {e}")
        raise
    except Exception as e:
        print(f"Unexpected error downloading s3://{bucket}/{key}: {e}")
        raise
    
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
    if test_s3_connection():
        print("Ready to download from S3")
    else:
        print("Please check your AWS credentials")    # download_s3_directory("s3://gepeta-checkpoints/qwen8-20-billion/step-22064/","/home/ec2-user/models/qwen8-20-billion/step-22064/")
    # download_s3_directory("s3://gepeta-checkpoints/qwen8-20-billion/mid-step-15617/","/home/ec2-user/models/qwen8-20-billion/mid-step-15617/")
    # download_s3_directory("s3://gepeta-checkpoints/qwen8-20-billion/low-step-6499/","/home/ec2-user/models/qwen8-20-billion/low-step-6499/")
    download_s3_directory("s3://gepeta-checkpoints/qwen8-20-billion/long-step-517/","/home/ec2-user/models/qwen8-20-billion/long-step-517/")
