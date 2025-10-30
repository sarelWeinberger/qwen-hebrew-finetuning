import pandas as pd
import os
from dashboard.config import Config

# Clean model names - remove the long path prefix
def clean_model_name(model_name):
    if pd.isna(model_name):
        return model_name
    model_name = str(model_name)
    if model_name.startswith('/home/ec2-user/models/'):
        return model_name.replace('/home/ec2-user/models/', '')
    if model_name.startswith('/home/ec2-user/qwen-hebrew-finetuning/'):
        return model_name.replace('/home/ec2-user/qwen-hebrew-finetuning/', '')
    return model_name

def load_data():
    # Load your CSV file
    # df = pd.read_csv('benchmark_results_summary.csv')
    #  check if the file exists
    file_path = os.path.join(Config.local_save_directory, Config.csv_filename)
    if not os.path.exists(file_path):
        print( f"CSV file not found: {file_path}",f"CSV file not found: {file_path}")
        return f"CSV file not found: {file_path}",f"CSV file not found: {file_path}"
    df = pd.read_csv(file_path)

    # Clean the data - remove rows with missing model names or timestamps
    df = df.dropna(subset=['model_name', 'timestamp'])
    df = df[df['model_name'] != '']
    df['model_name'] = df['model_name'].apply(clean_model_name)
    
    # Fix timestamp format - replace hyphens with colons in time portion
    def fix_timestamp(timestamp_str):
        if pd.isna(timestamp_str):
            return timestamp_str
        # Convert to string if it's not already
        timestamp_str = str(timestamp_str)
        # Replace hyphens with colons in the time portion (after T)
        # Pattern: YYYY-MM-DDTHH-MM-SS -> YYYY-MM-DDTHH:MM:SS
        if 'T' in timestamp_str:
            date_part, time_part = timestamp_str.split('T', 1)
            time_part = time_part.replace('-', ':')
            return f"{date_part}T{time_part}"
        return timestamp_str
    
    # Apply timestamp fixing
    df['timestamp'] = df['timestamp'].apply(fix_timestamp)
    
    # Convert timestamp to datetime
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    except Exception as e:
        print(f"Error parsing timestamps: {e}")
        # If parsing fails, create a dummy timestamp
        df['timestamp'] = pd.to_datetime('2024-01-01')
    
    # Remove rows where timestamp conversion failed
    df = df.dropna(subset=['timestamp'])
    
    # Add a run identifier (combination of model and timestamp)
    df['run_id'] = df['model_name'] + ' - ' + df['timestamp'].dt.strftime('%Y-%m-%d %H:%M')
    
    # RENAME COLUMNS TO FIX MISTAKES
    df = rename_benchmark_columns(df)
    # Step 2: Replace psychometric_heb with Ψ (Greek Psi character)
    rename_mapping = {}
    for col in df.columns:
        if 'psychometric_heb' in col:
            rename_mapping[col] = col.replace('psychometric_heb', 'Ψ')

    df = df.rename(columns=rename_mapping)

    # Get score columns (excluding std columns and metadata)
    score_columns = [col for col in df.columns if col.endswith('_score')]
    df = df.reset_index(drop=True).round(3)
    return df, score_columns


def get_base_model_name(model_name):
    """Extract base model name from model path"""
    if pd.isna(model_name):
        return None
    # Remove step information and trailing slashes
    parts = model_name.split('/')
    if len(parts) > 1:
        # Return the base model name (first part)
        return parts[0]
    return model_name

def rename_benchmark_columns(df):
    # RENAME COLUMNS TO FIX MISTAKES
    # Step 1: Merge data from mistaken columns into correct columns
    # For psychometric_heb_restatement -> psychometric_heb_restatement_english
    for suffix in ['_score', '_std', '_details']:
        old_col = f'psychometric_heb_restatement{suffix}'
        new_col = f'psychometric_heb_restatement_english{suffix}'
        
        if old_col in df.columns and new_col in df.columns:
            # Fill NaN values in new_col with values from old_col
            df[new_col] = df[new_col].fillna(df[old_col])
            # Drop the old column
            df = df.drop(columns=[old_col])
        elif old_col in df.columns:
            # If new_col doesn't exist, just rename old_col
            df = df.rename(columns={old_col: new_col})

    # For psychometric_heb_analogies -> psychometric_heb_analogies_hebrew
    for suffix in ['_score', '_std', '_details']:
        old_col = f'psychometric_heb_analogies{suffix}'
        new_col = f'psychometric_heb_analogies_hebrew{suffix}'
        
        if old_col in df.columns and new_col in df.columns:
            # Fill NaN values in new_col with values from old_col
            df[new_col] = df[new_col].fillna(df[old_col])
            # Drop the old column
            df = df.drop(columns=[old_col])
        elif old_col in df.columns:
            # If new_col doesn't exist, just rename old_col
            df = df.rename(columns={old_col: new_col})
    return df
