#!/usr/bin/env python3

import os
import sys
import argparse
from pathlib import Path
import boto3
from botocore.exceptions import ClientError, NoCredentialsError
from tqdm import tqdm
import mimetypes

def get_content_type(file_path):
    """Get the content type for a file based on its extension."""
    content_type, _ = mimetypes.guess_type(file_path)
    return content_type or 'binary/octet-stream'

def get_all_files(directory):
    """Recursively get all files in a directory."""
    files = []
    for root, dirs, filenames in os.walk(directory):
        for filename in filenames:
            files.append(os.path.join(root, filename))
    return files

def upload_file_to_s3(s3_client, local_file, bucket, s3_key):
    """Upload a single file to S3."""
    try:
        content_type = get_content_type(local_file)
        extra_args = {'ContentType': content_type}
        
        s3_client.upload_file(
            local_file, 
            bucket, 
            s3_key,
            ExtraArgs=extra_args
        )
        return True
    except ClientError as e:
        print(f"Error uploading {local_file}: {e}")
        return False

def upload_directory_to_s3(local_directory, bucket_name, s3_prefix="", 
                          aws_profile=None, dry_run=False, preserve_dir_name=True):
    """
    Upload a directory and its subdirectories to S3.
    
    Args:
        local_directory (str): Path to local directory
        bucket_name (str): S3 bucket name
        s3_prefix (str): S3 prefix (folder path)
        aws_profile (str): AWS profile name (optional)
        dry_run (bool): If True, only show what would be uploaded
        preserve_dir_name (bool): If True, include the directory name in S3 path
    """
    
    # Initialize S3 client
    try:
        if aws_profile:
            session = boto3.Session(profile_name=aws_profile)
            s3_client = session.client('s3')
        else:
            s3_client = boto3.client('s3')
    except NoCredentialsError:
        print("Error: AWS credentials not found. Please configure your credentials.")
        return False
    except Exception as e:
        print(f"Error initializing AWS client: {e}")
        return False
    
    # Validate local directory
    if not os.path.isdir(local_directory):
        print(f"Error: Directory '{local_directory}' does not exist.")
        return False
    
    # Validate S3 bucket
    try:
        s3_client.head_bucket(Bucket=bucket_name)
    except ClientError as e:
        error_code = int(e.response['Error']['Code'])
        if error_code == 404:
            print(f"Error: Bucket '{bucket_name}' does not exist.")
        elif error_code == 403:
            print(f"Error: Access denied to bucket '{bucket_name}'.")
        else:
            print(f"Error accessing bucket '{bucket_name}': {e}")
        return False
    
    # Get all files to upload
    local_directory = os.path.abspath(local_directory)
    all_files = get_all_files(local_directory)
    
    if not all_files:
        print("No files found to upload.")
        return True
    
    # Prepare S3 prefix
    if s3_prefix and not s3_prefix.endswith('/'):
        s3_prefix += '/'
    
    # Add directory name to S3 prefix if preserve_dir_name is True
    if preserve_dir_name:
        dir_name = os.path.basename(local_directory)
        s3_prefix = f"{s3_prefix}{dir_name}/"
    
    print(f"Local directory: {local_directory}")
    print(f"S3 destination: s3://{bucket_name}/{s3_prefix}")
    print(f"Files to upload: {len(all_files)}")
    print("-" * 50)
    
    # Upload files
    successful_uploads = 0
    failed_uploads = 0
    
    # Create progress bar
    with tqdm(total=len(all_files), desc="Uploading", unit="file") as pbar:
        for local_file in all_files:
            # Calculate relative path from local directory
            relative_path = os.path.relpath(local_file, local_directory)
            
            # Create S3 key
            s3_key = f"{s3_prefix}{relative_path}".replace(os.path.sep, '/')
            
            if dry_run:
                print(f"Would upload: {local_file} -> s3://{bucket_name}/{s3_key}")
                successful_uploads += 1
            else:
                # Upload file
                if upload_file_to_s3(s3_client, local_file, bucket_name, s3_key):
                    successful_uploads += 1
                    pbar.set_description(f"Uploaded: {os.path.basename(local_file)}")
                else:
                    failed_uploads += 1
                    pbar.set_description(f"Failed: {os.path.basename(local_file)}")
            
            pbar.update(1)
    
    # Summary
    print("-" * 50)
    if dry_run:
        print("✅ Dry run completed!")
        print(f"Files that would be uploaded: {successful_uploads}")
    else:
        if failed_uploads == 0:
            print("✅ Upload completed successfully!")
        else:
            print("⚠️  Upload completed with some failures!")
        
        print(f"Successfully uploaded: {successful_uploads}")
        if failed_uploads > 0:
            print(f"Failed uploads: {failed_uploads}")
    
    return failed_uploads == 0

def main():
    parser = argparse.ArgumentParser(
        description="Upload a directory and its subdirectories to Amazon S3",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Upload with directory name preserved (default behavior)
  python s3_upload.py /path/to/mydir my-bucket-name --prefix backup/
  # Result: s3://my-bucket-name/backup/mydir/...

  # Upload directory contents only (without directory name)
  python s3_upload.py /path/to/mydir my-bucket-name --prefix backup/ --no-preserve-dir
  # Result: s3://my-bucket-name/backup/...

  # Your specific example:
  python s3_upload.py /home/ec2-user/qwen-hebrew-finetuning/hebrew_benchmark_results/scores_sum/2025-09-18T02-17-43 gepeta-datasets --prefix benchmark_results/heb_benc_results/
  # Result: s3://gepeta-datasets/benchmark_results/heb_benc_results/2025-09-18T02-17-43/...
        """
    )
    
    parser.add_argument('local_directory', 
                       help='Local directory to upload')
    parser.add_argument('bucket_name', 
                       help='S3 bucket name')
    parser.add_argument('--prefix', '-p', 
                       default='',
                       help='S3 prefix (folder path in bucket)')
    parser.add_argument('--profile', 
                       help='AWS profile name')
    parser.add_argument('--dry-run', 
                       action='store_true',
                       help='Show what would be uploaded without actually uploading')
    parser.add_argument('--no-preserve-dir', 
                       action='store_true',
                       help='Upload directory contents only, without preserving directory name')
    
    args = parser.parse_args()
    
    # Upload directory
    success = upload_directory_to_s3(
        local_directory=args.local_directory,
        bucket_name=args.bucket_name,
        s3_prefix=args.prefix,
        aws_profile=args.profile,
        dry_run=args.dry_run,
        preserve_dir_name=not args.no_preserve_dir  # Default is True (preserve)
    )
    
    sys.exit(0 if success else 1)

print("s3_upload.py loaded.")
if __name__ == "__main__":
    main()
# /home/ec2-user/qwen-hebrew-finetuning/hebrew_benchmark_results/scores_sum/2025-09-18T02-17-43
# python s3_upload.py /home/ec2-user/qwen-hebrew-finetuning/hebrew_benchmark_results/scores_sum/2025-09-18T02-17-43 gepeta-datasets --prefix benchmark_results/heb_benc_results/