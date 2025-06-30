import pandas as pd
import os
import glob
from typing import List, Dict, Tuple
import numpy as np
from Levenshtein import distance as levenshtein_distance

# Import cleaners
from cleaners.duplicate_remove_cleaner import DuplicateRemoverCleaner
from cleaners.regex_cleaner import RegExCleaner
from utils.cleaner_constants import CLEANUP_RULES


def normalize_levenshtein_distance(str1: str, str2: str) -> float:
    """
    Calculate normalized Levenshtein distance between two strings.
    
    Args:
        str1: First string
        str2: Second string
        
    Returns:
        Normalized Levenshtein distance (0-1, where 0 is identical, 1 is completely different)
    """
    if not str1 and not str2:
        return 0.0
    if not str1 or not str2:
        return 1.0
    
    max_len = max(len(str1), len(str2))
    if max_len == 0:
        return 0.0
    
    return levenshtein_distance(str1, str2) / max_len


def load_benchmark_data(file_path: str) -> pd.DataFrame:
    """
    Load benchmark data from CSV file.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame with original_text and manual_clean columns
    """
    df = pd.read_csv(file_path)
    
    # Ensure required columns exist
    if 'original_text' not in df.columns or 'manual_clean' not in df.columns:
        raise ValueError(f"File {file_path} must contain 'original_text' and 'manual_clean' columns")
    
    # Handle NaN values by converting them to empty strings
    df['original_text'] = df['original_text'].fillna('')
    df['manual_clean'] = df['manual_clean'].fillna('')
    
    return df


def apply_cleaners(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Apply duplicate and regex cleaners to the data.
    
    Args:
        df: Input DataFrame with 'original_text' column
        
    Returns:
        Dictionary with cleaned DataFrames for each step
    """
    results = {}
    
    # Prepare data for cleaners (they expect 'text' column)
    working_df = df.copy()
    working_df['text'] = working_df['original_text']
    working_df['n_words'] = working_df['text'].str.split().str.len().fillna(0)
    
    # Apply duplicate remover
    print("Applying duplicate remover...")
    duplicate_cleaner = DuplicateRemoverCleaner()
    duplicate_cleaned = duplicate_cleaner.clean(working_df)
    results['clean1'] = duplicate_cleaned
    
    # Apply regex cleaner
    print("Applying regex cleaner...")
    # Extract patterns from CLEANUP_RULES
    patterns = [(rule['regex'][0], rule['regex'][1]) for rule in CLEANUP_RULES]
    regex_cleaner = RegExCleaner(patterns=patterns, debug_mode=False, save_cleaned_data=False)
    regex_cleaned = regex_cleaner.clean(duplicate_cleaned)
    results['clean2'] = regex_cleaned
    
    return results


def calculate_levenshtein_metrics(df: pd.DataFrame, cleaned_dfs: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    """
    Calculate normalized Levenshtein distances.
    
    Args:
        df: Original DataFrame with manual_clean column
        cleaned_dfs: Dictionary of cleaned DataFrames
        
    Returns:
        Dictionary with Levenshtein distance metrics
    """
    metrics = {}
    
    # Calculate distance from original to manual clean
    original_to_manual_distances = []
    for orig_text, manual_clean in zip(df['original_text'], df['manual_clean']):
        if pd.isna(orig_text) or pd.isna(manual_clean):
            continue
        dist = normalize_levenshtein_distance(str(orig_text), str(manual_clean))
        original_to_manual_distances.append(dist)
    
    metrics['levenshtein_original_to_manual'] = np.mean(original_to_manual_distances) if original_to_manual_distances else 0.0
    
    # Calculate distances from each cleaning step to original
    for step_name, cleaned_df in cleaned_dfs.items():
        step_to_original_distances = []
        for i, (orig_text, cleaned_text) in enumerate(zip(df['original_text'], cleaned_df['text'])):
            if i >= len(cleaned_df) or pd.isna(orig_text) or pd.isna(cleaned_text):
                continue
            dist = normalize_levenshtein_distance(str(orig_text), str(cleaned_text))
            step_to_original_distances.append(dist)
        
        metrics[f'levenshtein_{step_name}_to_original'] = np.mean(step_to_original_distances) if step_to_original_distances else 0.0
    
    return metrics


def process_benchmark_file(file_path: str) -> Tuple[str, Dict[str, float], pd.DataFrame]:
    """
    Process a single benchmark file.
    
    Args:
        file_path: Path to the benchmark CSV file
        
    Returns:
        Tuple of (file_name, metrics_dict, result_dataframe)
    """
    print(f"Processing {file_path}...")
    
    # Load data
    df = load_benchmark_data(file_path)
    
    # Apply cleaners
    cleaned_dfs = apply_cleaners(df)
    
    # Calculate metrics
    metrics = calculate_levenshtein_metrics(df, cleaned_dfs)
    
    # Create result DataFrame
    result_df = df.copy()
    
    # Add cleaned text columns
    for step_name, cleaned_df in cleaned_dfs.items():
        # Ensure we have the same number of rows
        if len(cleaned_df) == len(result_df):
            result_df[f'cleaned_{step_name}'] = cleaned_df['text']
        else:
            # If lengths don't match, pad with NaN
            cleaned_texts = list(cleaned_df['text'])
            while len(cleaned_texts) < len(result_df):
                cleaned_texts.append(np.nan)
            result_df[f'cleaned_{step_name}'] = cleaned_texts[:len(result_df)]
    
    # Extract file name
    file_name = os.path.basename(file_path).replace('.csv', '')
    
    return file_name, metrics, result_df


def main():
    """
    Main function to process all benchmark files.
    """
    benchmark_dir = "banchmark"
    benchmark_files = glob.glob(os.path.join(benchmark_dir, "data-clean-banchmark - *.csv"))
    # Only process files that do NOT end with '_with_cleaned.csv'
    benchmark_files = [f for f in benchmark_files if not f.endswith('_with_cleaned.csv')]
    
    if not benchmark_files:
        print("No benchmark files found!")
        return
    
    print(f"Found {len(benchmark_files)} benchmark files")
    
    # Process each file
    all_metrics = []
    all_results = {}
    
    for file_path in benchmark_files:
        try:
            file_name, metrics, result_df = process_benchmark_file(file_path)
            
            # Add file name to metrics
            metrics['file_name'] = file_name
            all_metrics.append(metrics)
            
            # Store result DataFrame
            all_results[file_name] = result_df
            
            print(f"Completed processing {file_name}")
            
        except Exception as e:
            print(f"Error processing {file_path}: {str(e)}")
            continue
    
    # Create summary DataFrame
    summary_df = pd.DataFrame(all_metrics)
    
    # Reorder columns to put file_name first
    cols = ['file_name'] + [col for col in summary_df.columns if col != 'file_name']
    summary_df = summary_df[cols]
    
    # Save results
    summary_df.to_csv('banchmark/benchmark_cleaning_summary.csv', index=False)
    print(f"Saved summary to banchmark/benchmark_cleaning_summary.csv")
    
    # Save individual result files
    for file_name, result_df in all_results.items():
        output_path = f"banchmark/{file_name}_with_cleaned.csv"
        result_df.to_csv(output_path, index=False)
        print(f"Saved {file_name} results to {output_path}")
    
    # Print summary
    print("\n=== BENCHMARK CLEANING SUMMARY ===")
    print(summary_df.to_string(index=False))
    
    # Calculate overall averages
    numeric_cols = [col for col in summary_df.columns if col.startswith('levenshtein')]
    if numeric_cols:
        print(f"\n=== OVERALL AVERAGES ===")
        for col in numeric_cols:
            avg_val = summary_df[col].mean()
            print(f"{col}: {avg_val:.4f}")


if __name__ == "__main__":
    main() 