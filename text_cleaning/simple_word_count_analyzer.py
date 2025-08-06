#!/usr/bin/env python3
"""
Simple script to analyze word counts for raw and cleaned data from the registry.
This script works with both active and commented sources in the registry.
"""
from io import StringIO
import re
import csv
import sys
import os
import json
import io
import tempfile
import gzip

# Try to import required packages, with fallbacks
import boto3
import pandas as pd
import rarfile


def create_registry():
    """
    Create registry with all sources from main.py (including commented ones)
    """
    registry = {
        'AllHebNLI': {
            'bucket_name': 'israllm-datasets',
            'prefix': 'raw-datasets/nli/csv_output/',
            'source_name': 'AllHebNLI',
            'output_prefix': 'processed_and_cleaned/test',
            'output_bucket_name': 'gepeta-datasets'
        },
        'AllOfHEOscarData-Combined-Deduped-DC4.forgpt': {
            'bucket_name': 'israllm-datasets',
            'prefix': 'raw-datasets/rar/csv_output/',
            'source_name': 'AllOfHEOscarData-Combined-Deduped-DC4.forgpt',
            'output_prefix': 'processed_and_cleaned/AllOfHEOscarData',
            'output_bucket_name': 'gepeta-datasets'
        },
        'AllTzenzuraData-Combined-Deduped-DC4.forgpt': {
            'bucket_name': 'israllm-datasets',
            'prefix': 'raw-datasets/rar/csv_output/',
            'source_name': 'AllTzenzuraData-Combined-Deduped-DC4.forgpt',
            'output_prefix': 'processed_and_cleaned/AllTzenzuraData',
            'output_bucket_name': 'gepeta-datasets'
        },
        'BooksNLI2-Combined-Deduped.forgpt': {
            'bucket_name': 'israllm-datasets',
            'prefix': 'raw-datasets/rar/csv_output/',
            'source_name': 'BooksNLI2-Combined-Deduped.forgpt',
            'output_prefix': 'processed_and_cleaned/BooksNLI2',
            'output_bucket_name': 'gepeta-datasets'
        },
        'GeektimeCorpus-Combined-Deduped.forgpt': {
            'bucket_name': 'israllm-datasets',
            'prefix': 'raw-datasets/rar/csv_output/',
            'source_name': 'GeektimeCorpus-Combined-Deduped.forgpt',
            'output_prefix': 'processed_and_cleaned/GeektimeCorpus',
            'output_bucket_name': 'gepeta-datasets'
        },
        'hebrew_tweets_text_clean_full-Deduped.forgpt': {
            'bucket_name': 'israllm-datasets',
            'prefix': 'raw-datasets/rar/csv_output/',
            'source_name': 'hebrew_tweets_text_clean_full-Deduped.forgpt',
            'output_prefix': 'processed_and_cleaned/hebrew_tweets',
            'output_bucket_name': 'gepeta-datasets'
        },
        'HeC4DictaCombined-Clean-Deduped.forgpt': {
            'bucket_name': 'israllm-datasets',
            'prefix': 'raw-datasets/rar/csv_output/',
            'source_name': 'HeC4DictaCombined-Clean-Deduped.forgpt',
            'output_prefix': 'processed_and_cleaned/HeC4DictaCombined',
            'output_bucket_name': 'gepeta-datasets'
        },
        'YifatDataBatch2-Round3-Deduped.forgpt': {
            'bucket_name': 'israllm-datasets',
            'prefix': 'raw-datasets/rar/csv_output/',
            'source_name': 'YifatDataBatch2-Round3-Deduped.forgpt',
            'output_prefix': 'processed_and_cleaned/YifatDataBatch2',
            'output_bucket_name': 'gepeta-datasets'
        },
        'YifatDataRound2-Deduped.forgpt': {
            'bucket_name': 'israllm-datasets',
            'prefix': 'raw-datasets/rar/csv_output/',
            'source_name': 'YifatDataRound2-Deduped.forgpt',
            'output_prefix': 'processed_and_cleaned/YifatDataRound2',
            'output_bucket_name': 'gepeta-datasets'
        },
        'YifatToCombine-Deduped.forgpt': {
            'bucket_name': 'israllm-datasets',
            'prefix': 'raw-datasets/rar/csv_output/',
            'source_name': 'YifatToCombine-Deduped.forgpt',
            'output_prefix': 'processed_and_cleaned/YifatToCombine',
            'output_bucket_name': 'gepeta-datasets'
        },
        'YisraelHayomData-Combined-Deduped.forgpt': {
            'bucket_name': 'israllm-datasets',
            'prefix': 'raw-datasets/rar/csv_output/',
            'source_name': 'YisraelHayomData-Combined-Deduped.forgpt',
            'output_prefix': 'processed_and_cleaned/YisraelHayomData',
            'output_bucket_name': 'gepeta-datasets'
        },
        'FineWeb2': {
            'bucket_name': 'israllm-datasets',
            'prefix': 'raw-datasets/fineweb2',
            'source_name': 'batch',
            'output_prefix': 'processed_and_cleaned/FineWeb2/',
            'output_bucket_name': 'gepeta-datasets'
        },
        'HeC4': {
            'bucket_name': 'israllm-datasets',
            'prefix': 'raw-datasets/HeC4',
            'source_name': 'part',
            'output_prefix': 'processed_and_cleaned/HeC4-HF',
            'output_bucket_name': 'gepeta-datasets'
        },
        'SupremeCourtOfIsrael': {
            'bucket_name': 'israllm-datasets',
            'prefix': 'raw-datasets/SupremeCourtOfIsrael/text_extraction/',
            'source_name': 'batch',
            'output_prefix': 'processed_and_cleaned/SupremeCourtOfIsrael',
            'output_bucket_name': 'gepeta-datasets'
        },
        'YifatDataBatch2-Round4': {
            'bucket_name': 'israllm-datasets',
            'prefix': 'raw-datasets/Yifat4+5/csv_output',
            'source_name': 'YifatDataBatch2-Round4',
            'output_prefix': 'processed_and_cleaned/YifatDataBatch2-Round4',
            'output_bucket_name': 'gepeta-datasets'
        },
        'YifatDataBatch3-Round5': {
            'bucket_name': 'israllm-datasets',
            'prefix': 'raw-datasets/Yifat4+5/csv_output',
            'source_name': 'YifatDataBatch3-Round5',
            'output_prefix': 'test_spacefix/YifatDataBatch2-Round5',
            'output_bucket_name': 'gepeta-datasets'
        },
        'OcrTau': {
            'bucket_name': 'israllm-datasets',
            'prefix': 'tau_clean/',
            'source_name': 'HQ',
            'output_prefix': 'processed_and_cleaned/TauOCR',
            'output_bucket_name': 'gepeta-datasets'
        },
        'TauDigital': {
            'bucket_name': 'israllm-datasets',
            'prefix': 'tau_clean/',
            'source_name': 'DigitalMarkdown_Tables.jsonl',
            'output_prefix': 'processed_and_cleaned/TauDigital',
            'output_bucket_name': 'gepeta-datasets'
        },
        'BIU': {
            'bucket_name': 'israllm-datasets',
            'prefix': 'raw-datasets/biu-drive/',
            'source_name': 'AllBIUDriveDocs-MD-Deduped.forgpt.jsonl.gz',
            'output_prefix': 'processed_and_cleaned/BIU',
            'output_bucket_name': 'gepeta-datasets'
        },
        'sefaria': {
            'bucket_name': 'israllm-datasets',
            'prefix': 'raw-datasets/sefaria',
            'source_name': 'AllOfSefariaTexts',
            'output_prefix': 'processed_and_cleaned/sefaria',
            'output_bucket_name': 'gepeta-datasets'
        }
    }
    return registry


def count_words_in_text(text):
    """Count words in text"""
    if not text or not isinstance(text, str):
        return 0
    return len(text.split())





def read_jsonl_data(file_data):
    """Read JSONL data from bytes"""
    try:
        text = file_data.decode('utf-8')
        total_words = 0
        
        for line in text.split('\n'):
            line = line.strip()
            if line:
                try:
                    data = json.loads(line)
                    # Look for text field
                    if 'text' in data:
                        total_words += count_words_in_text(data['text'])
                    elif 'content' in data:
                        total_words += count_words_in_text(data['content'])
                    else:
                        # Use first field as text
                        first_value = list(data.values())[0] if data else ""
                        total_words += count_words_in_text(str(first_value))
                except:
                    continue
        
        return total_words
    except Exception as e:
        print(f"Error reading JSONL: {e}")
        return 0


def read_csv_data(file_data):
    """Read CSV data from bytes"""
    try:
        text = file_data.decode('utf-8')
        total_words = 0
        
        # Try reading with header first
        try:
            df = pd.read_csv(StringIO(text))
            if 'n_words' in df.columns:
                total_words = df['n_words'].sum()
                return total_words
            elif 'text' in df.columns:
                total_words = df['text'].apply(count_words_in_text).sum()
                return total_words
        except:
            pass
        
        # Try reading without header and add headers
        try:
            df = pd.read_csv(StringIO(text), header=None)
            if len(df.columns) > 0:
                # Add headers: text, n_words (or just text if only one column)
                if len(df.columns) == 1:
                    df.columns = ['text']
                elif len(df.columns) == 2:
                    df.columns = ['text', 'n_words']
                
                # Now check for n_words or text columns
                if 'n_words' in df.columns:
                    total_words = df['n_words'].sum()
                    return total_words
                elif 'text' in df.columns:
                    total_words = df['text'].apply(count_words_in_text).sum()
                    return total_words
        except:
            pass
        
        print("Could not read CSV data properly")
        return 0
        
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return 0


def read_parquet_data(file_data):
    """Read parquet data from bytes"""
    try:
        df = pd.read_parquet(io.BytesIO(file_data))
        total_words = 0
        
        # Try with named columns first
        if 'n_words' in df.columns:
            total_words = df['n_words'].sum()
            return total_words
        elif 'text' in df.columns:
            # Count words in text column
            total_words = df['text'].apply(count_words_in_text).sum()
            return total_words
        
        # If no named columns, add headers and try again
        if len(df.columns) > 0:
            # Add headers: text, n_words (or just text if only one column)
            if len(df.columns) == 1:
                df.columns = ['text']
            elif len(df.columns) == 2:
                df.columns = ['text', 'n_words']

            # Now check for n_words or text columns
            if 'n_words' in df.columns:
                total_words = df['n_words'].sum()
                return total_words
            elif 'text' in df.columns:
                total_words = df['text'].apply(count_words_in_text).sum()
                return total_words
        
        print("No usable columns found in parquet")
        return 0
        
    except Exception as e:
        print(f"Error reading parquet: {e}")
        return 0


def count_words_in_source(bucket_name, prefix, source_name):
    """Count total words in all files for a source before cleaning."""
    
    try:
        s3 = boto3.client("s3")
        paginator = s3.get_paginator('list_objects_v2')
        total_words = 0
        file_count = 0
        
        for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
            for obj in page.get('Contents', []):
                key = obj['Key']
                filename = key.split('/')[-1]
                print(f'filename: {filename}')
                if filename.startswith(source_name) and (filename.endswith('.jsonl') or filename.endswith('.csv') or filename.endswith('.gz') or filename.endswith('.parquet')):
                    try:
                        response = s3.get_object(Bucket=bucket_name, Key=key)
                        file_data = response["Body"].read()
                        
                        if key.endswith('.jsonl'):
                            words = read_jsonl_data(file_data)
                        elif key.endswith('.csv'):
                            words = read_csv_data(file_data)
                        elif key.endswith('.parquet'):
                            words = read_parquet_data(file_data)
                        elif key.endswith('.gz'):
                            # Handle GZ compressed files
                            decompressed_data = gzip.decompress(file_data)
                            base_filename = filename.replace('.gz', '')
                            if base_filename.endswith('.jsonl'):
                                words = read_jsonl_data(decompressed_data)
                            elif base_filename.endswith('.csv'):
                                words = read_csv_data(decompressed_data)
                            elif base_filename.endswith('.parquet'):
                                words = read_parquet_data(decompressed_data)
                            else:
                                continue
                        else:
                            continue
                        
                        total_words += words
                        file_count += 1
                        
                    except Exception as e:
                        print(f"Error processing {key}: {str(e)}")
        
        return total_words, file_count
    except Exception as e:
        print(f"Error accessing S3: {e}")
        return 0, 0


def count_words_after_cleaning(output_bucket_name, output_prefix):
    """Count total words in all cleaned files for a source."""
    
    try:
        s3 = boto3.client("s3")
        paginator = s3.get_paginator('list_objects_v2')
        total_words = 0
        file_count = 0
        
        for page in paginator.paginate(Bucket=output_bucket_name, Prefix=output_prefix):
            for obj in page.get('Contents', []):
                key = obj['Key']
                filename = key.split('/')[-1]
                
                if filename.endswith('_cleaned.csv') or filename.endswith('_cleaned.parquet'):
                    try:
                        response = s3.get_object(Bucket=output_bucket_name, Key=key)
                        file_data = response["Body"].read()
                        
                        if key.endswith('_cleaned.csv'):
                            words = read_csv_data(file_data)
                        elif key.endswith('_cleaned.parquet'):
                            words = read_parquet_data(file_data)
                        else:
                            continue
                            
                        total_words += words
                        file_count += 1
                        
                    except Exception as e:
                        print(f"Error processing cleaned {key}: {str(e)}")
        
        return total_words, file_count
    except Exception as e:
        print(f"Error accessing S3: {e}")
        return 0, 0


def main():
    """Main function to analyze word counts for all sources."""
    print("Creating registry with all sources...")
    registry = create_registry()
    
    print(f"Found {len(registry)} sources to analyze")
    
    results = []
    
    for source_name, config in registry.items():
        print(f"\nProcessing source: {source_name}")
        s3_path = f"s3://{config['bucket_name']}/{config['prefix']}"
        print(f"  Raw data: {s3_path}")
        print(f"  Cleaned data: s3://{config['output_bucket_name']}/{config['output_prefix']}")
        
        try:
            # Count words in raw data
            raw_words, raw_files = count_words_in_source(
                bucket_name=config['bucket_name'],
                prefix=config['prefix'],
                source_name=config['source_name']
            )
            
            # Count words in cleaned data
            cleaned_words, cleaned_files = count_words_after_cleaning(
                output_bucket_name=config['output_bucket_name'],
                output_prefix=config['output_prefix']
            )
            
            results.append({
                'source_name': source_name,
                's3_path': s3_path,
                'n_words_raw': raw_words,
                'n_words_cleaned': cleaned_words
            })
            
            print(f"  Raw: {raw_words:,} words ({raw_files} files)")
            print(f"  Cleaned: {cleaned_words:,} words ({cleaned_files} files)")
            
            if raw_words > 0:
                reduction = ((raw_words - cleaned_words) / raw_words * 100)
                print(f"  Reduction: {reduction:.1f}%")
            
        except Exception as e:
            print(f"  Error processing {source_name}: {str(e)}")
            results.append({
                'source_name': source_name,
                's3_path': s3_path,
                'n_words_raw': 0,
                'n_words_cleaned': 0
            })
    
    # Save results to CSV
    output_file = 'word_count_analysis.csv'
    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['source_name', 's3_path', 'n_words_raw', 'n_words_cleaned']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for result in results:
            writer.writerow(result)
    
    print("\n" + "="*100)
    print("WORD COUNT ANALYSIS SUMMARY")
    print("="*100)
    
    # Print results in a table format
    print(f"{'Source Name':<30} {'S3 Path':<50} {'Raw Words':<12} {'Cleaned Words':<15}")
    print("-" * 100)
    
    for result in results:
        print(f"{result['source_name']:<30} {result['s3_path']:<50} {result['n_words_raw']:<12} {result['n_words_cleaned']:<15}")
    
    print("="*100)
    print(f"\nResults saved to: {output_file}")
    
    # Summary statistics
    total_raw = sum(r['n_words_raw'] for r in results)
    total_cleaned = sum(r['n_words_cleaned'] for r in results)
    total_reduction = ((total_raw - total_cleaned) / total_raw * 100) if total_raw > 0 else 0
    
    print(f"\nTOTAL SUMMARY:")
    print(f"Total raw words: {total_raw:,}")
    print(f"Total cleaned words: {total_cleaned:,}")
    print(f"Total reduction: {total_reduction:.1f}%")


if __name__ == "__main__":
    main() 
