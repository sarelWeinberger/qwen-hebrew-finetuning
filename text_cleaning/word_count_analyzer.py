import pandas as pd
import boto3
import io
import json
import tempfile
import os
import rarfile
from main import create_registry_regex_only
from utils.logger import logger


def _extract_rar_and_read_jsonl(rar_data: bytes) -> pd.DataFrame:
    """
    Extract RAR file and read JSONL content from it.
    
    Args:
        rar_data: Raw bytes of the RAR file
        
    Returns:
        DataFrame with extracted JSONL data
    """
    try:
        # Create a temporary directory for extraction
        with tempfile.TemporaryDirectory() as temp_dir:
            # Save RAR file to temporary location
            rar_path = os.path.join(temp_dir, "temp.rar")
            with open(rar_path, 'wb') as f:
                f.write(rar_data)
            
            # Extract RAR file
            with rarfile.RarFile(rar_path, 'r') as rf:
                # Find JSONL files in the archive
                jsonl_files = [f for f in rf.namelist() if f.endswith('.jsonl')]
                
                if not jsonl_files:
                    logger.warning("No JSONL files found in RAR archive")
                    return pd.DataFrame()
                
                all_data = []
                
                for jsonl_file in jsonl_files:
                    logger.info(f"Processing JSONL file: {jsonl_file}")
                    
                    # Read JSONL content
                    with rf.open(jsonl_file) as f:
                        for line_num, line in enumerate(f, 1):
                            try:
                                line = line.decode('utf-8').strip()
                                if line:  # Skip empty lines
                                    data = json.loads(line)
                                    all_data.append(data)
                            except json.JSONDecodeError as e:
                                logger.warning(f"Invalid JSON at line {line_num} in {jsonl_file}: {e}")
                            except Exception as e:
                                logger.warning(f"Error processing line {line_num} in {jsonl_file}: {e}")
                
                if not all_data:
                    logger.warning("No valid JSON data found in RAR archive")
                    return pd.DataFrame()
                
                # Convert to DataFrame
                df = pd.DataFrame(all_data)
                
                # If the JSONL contains text data, ensure we have a 'text' column
                if 'text' not in df.columns:
                    # Try to find a suitable text column
                    text_columns = [col for col in df.columns if 'text' in col.lower() or 'content' in col.lower()]
                    if text_columns:
                        df['text'] = df[text_columns[0]]
                    else:
                        # If no text column found, use the first column as text
                        df['text'] = df.iloc[:, 0].astype(str)
                
                # Add word count column if not present
                if 'n_words' not in df.columns:
                    df['n_words'] = df['text'].str.split().str.len()
                
                return df
                
    except Exception as e:
        logger.error(f"Error extracting RAR file: {str(e)}")
        return pd.DataFrame()



def _read_jsonl_data(file_data: bytes) -> pd.DataFrame:
    """
    Read JSONL data from bytes.
    
    Args:
        file_data: Raw bytes of the JSONL file
        
    Returns:
        DataFrame with parsed JSONL data
    """
    try:
        text = file_data.decode('utf-8')
        all_data = []
        
        for line_num, line in enumerate(text.split('\n'), 1):
            line = line.strip()
            if line:  # Skip empty lines
                try:
                    data = json.loads(line)
                    all_data.append(data)
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON at line {line_num}: {e}")
                except Exception as e:
                    logger.warning(f"Error processing line {line_num}: {e}")
        
        if not all_data:
            logger.warning("No valid JSON data found in file")
            return pd.DataFrame()
        
        # Convert to DataFrame
        df = pd.DataFrame(all_data)
        
        # If the JSONL contains text data, ensure we have a 'text' column
        if 'text' not in df.columns:
            # Try to find a suitable text column
            text_columns = [col for col in df.columns if 'text' in col.lower() or 'content' in col.lower()]
            if text_columns:
                df['text'] = df[text_columns[0]]
            else:
                # If no text column found, use the first column as text
                df['text'] = df.iloc[:, 0].astype(str)
        
        # Add word count column if not present
        if 'n_words' not in df.columns:
            df['n_words'] = df['text'].str.split().str.len()
        
        return df
        
    except Exception as e:
        logger.error(f"Error reading JSONL data: {str(e)}")
        return pd.DataFrame()

def count_words_in_source(bucket_name, prefix, source_name):
    """Count total words in all files for a source before cleaning."""
    s3 = boto3.client("s3")
    paginator = s3.get_paginator('list_objects_v2')
    total_words = 0
    
    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        for obj in page.get('Contents', []):
            key = obj['Key']
            filename = key.split('/')[-1]
            
            if filename.startswith(source_name) and (filename.endswith('.jsonl') or filename.endswith('.rar') or filename.endswith('.csv')):
                try:
                    response = s3.get_object(Bucket=bucket_name, Key=key)
                    file_data = response["Body"].read()
                    
                    if key.endswith('.jsonl'):
                        # Handle extracted JSONL files directly
                        df = _read_jsonl_data(file_data)
                    elif key.endswith('.rar'):
                        # Handle RAR files containing JSONL data
                        df = _extract_rar_and_read_jsonl(file_data)
                    elif key.endswith('.csv'):
                        # Handle CSV files
                        df = pd.read_csv(io.BytesIO(file_data))
                    else:
                        logger.warning(f"Unsupported file format: {key}")
                        continue
                    
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