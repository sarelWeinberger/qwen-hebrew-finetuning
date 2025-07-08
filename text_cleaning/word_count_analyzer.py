import pandas as pd
import boto3
import io
from main import create_registry_regex_only
from utils.logger import logger


def count_words_in_source(bucket_name, prefix, source_name):
    """Count total words in all files for a source before cleaning."""
    s3 = boto3.client("s3")
    paginator = s3.get_paginator('list_objects_v2')
    total_words = 0
    
    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        for obj in page.get('Contents', []):
            key = obj['Key']
            filename = key.split('/')[-1]
            
            if filename.startswith(source_name) and (filename.endswith('.csv') or filename.endswith('.parquet')):
                try:
                    response = s3.get_object(Bucket=bucket_name, Key=key)
                    
                    if key.endswith('.csv'):
                        df = pd.read_csv(io.BytesIO(response["Body"].read()))
                    elif key.endswith('.parquet'):
                        df = pd.read_parquet(io.BytesIO(response["Body"].read()))
                    
                    if 'n_words' in df.columns:
                        total_words += df['n_words'].sum()
                    elif 'n_count' in df.columns:
                        total_words += df['n_count'].sum()
                    else:
                        if 'text' in df.columns:
                            total_words += sum(len(str(text).split()) for text in df['text'] if pd.notna(text))
                    
                except Exception as e:
                    logger.error(f"Error processing {key}: {str(e)}")
    
    return int(total_words)


def count_words_after_cleaning(output_bucket_name, output_prefix):
    """Count total words in all cleaned files for a source."""
    s3 = boto3.client("s3")
    paginator = s3.get_paginator('list_objects_v2')
    total_words = 0
    
    for page in paginator.paginate(Bucket=output_bucket_name, Prefix=output_prefix):
        for obj in page.get('Contents', []):
            key = obj['Key']
            filename = key.split('/')[-1]
            
            if filename.endswith('_cleaned.csv'):
                try:
                    response = s3.get_object(Bucket=output_bucket_name, Key=key)
                    df = pd.read_csv(io.BytesIO(response["Body"].read()))
                    
                    if 'n_words' in df.columns:
                        total_words += df['n_words'].sum()
                    elif 'n_count' in df.columns:
                        total_words += df['n_count'].sum()
                    else:
                        if 'text' in df.columns:
                            total_words += sum(len(str(text).split()) for text in df['text'] if pd.notna(text))
                    
                except Exception as e:
                    logger.error(f"Error processing cleaned {key}: {str(e)}")
    
    return int(total_words)


def main():
    registry = create_registry_regex_only()
    results = []
    
    for source_name, components in registry.items():
        logger.info(f"Processing source: {source_name}")
        
        fetcher = components['fetcher']
        
        before_clean = count_words_in_source(
            bucket_name=fetcher.bucket_name,
            prefix=fetcher.prefix,
            source_name=fetcher.source_name
        )
        
        after_clean = count_words_after_cleaning(
            output_bucket_name=fetcher.output_bucket_name,
            output_prefix=fetcher.output_prefix
        )
        
        reduction_percent = ((before_clean - after_clean) / before_clean * 100) if before_clean > 0 else 0
        
        results.append({
            'source': source_name,
            'before_clean': before_clean,
            'after_clean': after_clean,
            'reduction_percent': round(reduction_percent, 2)
        })
        
        logger.info(f"{source_name}: {before_clean:,} -> {after_clean:,} words ({reduction_percent:.1f}% reduction)")
    
    df = pd.DataFrame(results)
    
    output_file = 'word_count_analysis.csv'
    df.to_csv(output_file, index=False)
    logger.info(f"Results saved to {output_file}")
    
    print("\n" + "="*80)
    print("WORD COUNT ANALYSIS SUMMARY")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    main() 