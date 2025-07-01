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


def get_cleaners() -> List[Tuple[str, object]]:
    """
    Define the list of cleaners to apply in order.
    Each cleaner should be a tuple of (name, cleaner_instance).
    
    Returns:
        List of (cleaner_name, cleaner_instance) tuples
    """
    cleaners = [
        ("duplicate_remover", DuplicateRemoverCleaner()),
        ("regex_cleaner", RegExCleaner(
            patterns=[(rule['regex'][0], rule['regex'][1]) for rule in CLEANUP_RULES],
            debug_mode=False, 
            save_cleaned_data=False
        ))
    ]
    
    # To add more cleaners in the future, simply add them here:
    # cleaners.append(("new_cleaner_name", NewCleaner()))
    
    return cleaners


def apply_cleaners_step_by_step(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Apply each cleaner individually and return results after each step.
    
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
    
    # Apply each cleaner individually
    cleaners = get_cleaners()
    for cleaner_name, cleaner in cleaners:
        print(f"Applying {cleaner_name}...")
        cleaned_df = cleaner.clean(working_df)
        results[cleaner_name] = cleaned_df.copy()
        working_df = cleaned_df  # Use result as input for next cleaner
    
    return results


def calculate_levenshtein_metrics(df: pd.DataFrame, cleaned_dfs: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    """
    Calculate normalized Levenshtein distances for each cleaning step (cumulative).
    
    Args:
        df: Original DataFrame with manual_clean column
        cleaned_dfs: Dictionary of cleaned DataFrames for each step (cumulative)
        
    Returns:
        Dictionary with Levenshtein distance metrics
    """
    metrics = {}
    
    # Distance from original to manual
    orig_to_manual = [normalize_levenshtein_distance(str(orig), str(manual))
                      for orig, manual in zip(df['original_text'], df['manual_clean'])
                      if not (pd.isna(orig) or pd.isna(manual))]
    metrics['levenshtein_dist_original_to_manual'] = round(np.mean(orig_to_manual), 3) if orig_to_manual else 0.0
    
    # Cumulative: duplicate_remover, then duplicate_and_regex
    step_names = list(cleaned_dfs.keys())
    prev_df = df
    for i, step_name in enumerate(step_names):
        cleaned_df = cleaned_dfs[step_name]
        # Distance from this step's output to manual
        step_to_manual = [normalize_levenshtein_distance(str(cleaned), str(manual))
                          for cleaned, manual in zip(cleaned_df['text'], df['manual_clean'])
                          if not (pd.isna(cleaned) or pd.isna(manual))]
        if i == 0:
            metrics[f'levenshtein_dist_{step_name}_to_manual'] = round(np.mean(step_to_manual), 3) if step_to_manual else 0.0
        else:
            # Cumulative name for all up to this step
            cum_name = '_and_'.join(step_names[:i+1])
            metrics[f'levenshtein_dist_{cum_name}_to_manual'] = round(np.mean(step_to_manual), 3) if step_to_manual else 0.0
        prev_df = cleaned_df
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
    
    # Apply cleaners step by step
    cleaned_dfs = apply_cleaners_step_by_step(df)
    
    # Calculate metrics for each step
    metrics = calculate_levenshtein_metrics(df, cleaned_dfs)
    
    # Create result DataFrame
    result_df = df.copy()
    
    # Determine the next clean column number
    existing_clean_cols = [col for col in result_df.columns if col.startswith('clean') and col[5:].isdigit()]
    if existing_clean_cols:
        # Find the highest number and increment
        max_num = max(int(col[5:]) for col in existing_clean_cols)
        next_clean_num = max_num + 1
    else:
        next_clean_num = 1
    
    # Add cleaned text columns for each step
    for step_name, cleaned_df in cleaned_dfs.items():
        clean_col_name = f'clean{next_clean_num}_{step_name}'
        
        if len(cleaned_df) == len(result_df):
            result_df[clean_col_name] = cleaned_df['text']
        else:
            # If lengths don't match, pad with NaN
            cleaned_texts = list(cleaned_df['text'])
            while len(cleaned_texts) < len(result_df):
                cleaned_texts.append(np.nan)
            result_df[clean_col_name] = cleaned_texts[:len(result_df)]
    
    # Extract file name
    file_name = os.path.basename(file_path).replace('.csv', '')
    
    return file_name, metrics, result_df


def main():
    """
    Main function to process all benchmark files.
    """
    benchmark_dir = "banchmark"
    # Get all CSV files in the benchmark directory
    all_csv_files = glob.glob(os.path.join('text_cleaning' , os.path.join(benchmark_dir, "*.csv")))
    
    # Filter to only include original benchmark files (not the ones we've already processed)
    benchmark_files = []
    for file_path in all_csv_files:
        file_name = os.path.basename(file_path)
        # Only include files that start with "data-clean-banchmark - " and don't end with "_with_cleaned.csv"
        if file_name.startswith("data-clean-banchmark - ") and not file_name.endswith("_with_cleaned.csv"):
            benchmark_files.append(file_path)
    
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
    summary_df.to_csv('text_cleaning/banchmark/benchmark_cleaning_summary.csv', index=False)
    print(f"Saved summary to text_cleaning/banchmark/benchmark_cleaning_summary.csv")
    
    # Save individual result files
    for file_name, result_df in all_results.items():
        output_path = f"text_cleaning/banchmark/{file_name}_with_cleaned.csv"
        result_df.to_csv(output_path, index=False)
        print(f"Saved {file_name} results to {output_path}")
    
    # Print summary
    print("\n=== BENCHMARK CLEANING SUMMARY ===")
    print(summary_df.to_string(index=False))
    
    # Calculate overall averages
    numeric_cols = [col for col in summary_df.columns if col.startswith('levenshtein_dist_')]
    if numeric_cols:
        print(f"\n=== OVERALL AVERAGES ===")
        for col in numeric_cols:
            avg_val = summary_df[col].mean()
            print(f"{col}: {avg_val:.4f}")


if __name__ == "__main__":
    main() 