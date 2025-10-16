#!/usr/bin/env python3
"""
Script to read a CSV file from S3 and calculate the sum of the n_words column.
"""

import boto3
import pandas as pd
import io
import argparse
from typing import Optional


def read_csv_from_s3(bucket_name: str, file_key: str, has_header: bool = True) -> pd.DataFrame:
    """
    Read a CSV file from S3 and return as a DataFrame.
    
    Args:
        bucket_name: S3 bucket name
        file_key: S3 object key (file path)
        has_header: Whether the CSV has a header row
        
    Returns:
        DataFrame with the CSV data
    """
    try:
        # Initialize S3 client
        s3 = boto3.client('s3')
        
        # Get the object from S3
        response = s3.get_object(Bucket=bucket_name, Key=file_key)
        
        # Read CSV data
        if has_header:
            df = pd.read_csv(io.BytesIO(response["Body"].read()))
        else:
            # If no header, assume columns are "text" and "n_words"
            df = pd.read_csv(io.BytesIO(response["Body"].read()), header=None, names=["text", "n_words"])
        
        return df
        
    except Exception as e:
        print(f"Error reading file from S3: {str(e)}")
        return pd.DataFrame()


def calculate_n_words_sum(df: pd.DataFrame) -> Optional[int]:
    """
    Calculate the sum of the n_words column.
    
    Args:
        df: DataFrame containing the data
        
    Returns:
        Sum of n_words column, or None if column doesn't exist
    """
    if 'n_words' not in df.columns:
        print("Error: 'n_words' column not found in the CSV file.")
        print(f"Available columns: {list(df.columns)}")
        return None
    
    # Calculate sum
    total_words = df['n_words'].sum()
    return int(total_words)


def main():
        
    # Read the CSV file
    df = read_csv_from_s3(bucket_name='gepeta-datasets', file_key='processed_and_cleaned/BIU/AllBIUDriveDocs-MD-Deduped.forgpt.jsonl_cleaned.csv')
    
    if df.empty:
        print("Failed to read CSV file or file is empty.")
        return
    
    print(f"Successfully read CSV file with {len(df)} rows and {len(df.columns)} columns")
    print(f"Columns: {list(df.columns)}")
        
    # Calculate and display the sum
    total_words = calculate_n_words_sum(df)
    
    if total_words is not None:
        print(f"\nTotal number of words: {total_words:,}")
        
        # Additional statistics
        if len(df) > 0:
            avg_words = df['n_words'].mean()
            min_words = df['n_words'].min()
            max_words = df['n_words'].max()
            
            print(f"Average words per row: {avg_words:.2f}")
            print(f"Minimum words in a row: {min_words}")
            print(f"Maximum words in a row: {max_words}")


if __name__ == "__main__":
    main() 