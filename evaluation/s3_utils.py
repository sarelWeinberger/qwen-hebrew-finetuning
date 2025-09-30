"""
S3 utilities for downloading and managing S3 data
"""
import os
import tempfile
import shutil
import boto3
from pathlib import Path


def is_s3_path(path):
    """
    Check if a path is an S3 path.

    Args:
        path (str): Path to check

    Returns:
        bool: True if path starts with 's3://', False otherwise
    """
    return isinstance(path, str) and path.startswith('s3://')


def parse_s3_path(s3_path):
    """
    Parse S3 path into bucket and key.

    Args:
        s3_path (str): S3 path in format 's3://bucket/key/path'

    Returns:
        tuple: (bucket_name, key_path)
    """
    if not is_s3_path(s3_path):
        raise ValueError(f"Invalid S3 path: {s3_path}")

    # Remove 's3://' prefix
    path_without_prefix = s3_path[5:]

    # Split into bucket and key
    parts = path_without_prefix.split('/', 1)
    bucket = parts[0]
    key = parts[1] if len(parts) > 1 else ''

    return bucket, key


def create_temp_directory(prefix='lighteval_'):
    """
    Create a temporary directory.

    Args:
        prefix (str): Prefix for the temporary directory name

    Returns:
        str: Path to the created temporary directory
    """
    temp_dir = tempfile.mkdtemp(prefix=prefix)
    return temp_dir


def cleanup_temp_directory(temp_dir):
    """
    Remove a temporary directory and all its contents.

    Args:
        temp_dir (str): Path to the temporary directory to remove
    """
    if os.path.exists(temp_dir):
        try:
            shutil.rmtree(temp_dir)
            print(f"Cleaned up temporary directory: {temp_dir}")
        except Exception as e:
            print(f"Warning: Failed to clean up {temp_dir}: {e}")


def download_s3_directory(s3_path, local_path):
    """
    Download an entire S3 directory to a local path.

    Args:
        s3_path (str): S3 path in format 's3://bucket/key/path'
        local_path (str): Local directory path to download to

    Returns:
        str: Local path where files were downloaded
    """
    bucket, key = parse_s3_path(s3_path)

    # Create local directory if it doesn't exist
    os.makedirs(local_path, exist_ok=True)

    # Initialize S3 client
    s3_client = boto3.client('s3')

    # List all objects with the given prefix
    paginator = s3_client.get_paginator('list_objects_v2')
    pages = paginator.paginate(Bucket=bucket, Prefix=key)

    downloaded_count = 0

    for page in pages:
        if 'Contents' not in page:
            continue

        for obj in page['Contents']:
            s3_key = obj['Key']

            # Skip if it's just a directory marker
            if s3_key.endswith('/'):
                continue

            # Calculate relative path
            relative_path = os.path.relpath(s3_key, key)
            local_file_path = os.path.join(local_path, relative_path)

            # Create local directory structure
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)

            # Download file
            try:
                s3_client.download_file(bucket, s3_key, local_file_path)
                downloaded_count += 1
                if downloaded_count % 10 == 0:
                    print(f"Downloaded {downloaded_count} files...")
            except Exception as e:
                print(f"Error downloading {s3_key}: {e}")

    print(f"Downloaded {downloaded_count} files from {s3_path} to {local_path}")
    return local_path


def upload_file_to_s3(local_file, s3_path):
    """
    Upload a single file to S3.

    Args:
        local_file (str): Path to local file
        s3_path (str): S3 destination path in format 's3://bucket/key/path'

    Returns:
        bool: True if upload successful, False otherwise
    """
    try:
        bucket, key = parse_s3_path(s3_path)
        s3_client = boto3.client('s3')
        s3_client.upload_file(local_file, bucket, key)
        print(f"Uploaded {local_file} to {s3_path}")
        return True
    except Exception as e:
        print(f"Error uploading {local_file} to {s3_path}: {e}")
        return False


def upload_directory_to_s3(local_path, s3_path):
    """
    Upload an entire directory to S3.

    Args:
        local_path (str): Local directory path
        s3_path (str): S3 destination path in format 's3://bucket/key/path'

    Returns:
        int: Number of files uploaded
    """
    bucket, key_prefix = parse_s3_path(s3_path)
    s3_client = boto3.client('s3')

    uploaded_count = 0

    for root, dirs, files in os.walk(local_path):
        for file in files:
            local_file_path = os.path.join(root, file)

            # Calculate relative path
            relative_path = os.path.relpath(local_file_path, local_path)
            s3_key = os.path.join(key_prefix, relative_path).replace('\\', '/')

            try:
                s3_client.upload_file(local_file_path, bucket, s3_key)
                uploaded_count += 1
                if uploaded_count % 10 == 0:
                    print(f"Uploaded {uploaded_count} files...")
            except Exception as e:
                print(f"Error uploading {local_file_path}: {e}")

    print(f"Uploaded {uploaded_count} files from {local_path} to {s3_path}")
    return uploaded_count


def file_exists_in_s3(s3_path):
    """
    Check if a file exists in S3.

    Args:
        s3_path (str): S3 path in format 's3://bucket/key/path'

    Returns:
        bool: True if file exists, False otherwise
    """
    try:
        bucket, key = parse_s3_path(s3_path)
        s3_client = boto3.client('s3')
        s3_client.head_object(Bucket=bucket, Key=key)
        return True
    except:
        return False


def list_s3_objects(s3_path, max_keys=1000):
    """
    List objects in an S3 path.

    Args:
        s3_path (str): S3 path in format 's3://bucket/key/path'
        max_keys (int): Maximum number of keys to return

    Returns:
        list: List of object keys
    """
    bucket, prefix = parse_s3_path(s3_path)
    s3_client = boto3.client('s3')

    response = s3_client.list_objects_v2(
        Bucket=bucket,
        Prefix=prefix,
        MaxKeys=max_keys
    )

    if 'Contents' not in response:
        return []

    return [obj['Key'] for obj in response['Contents']]