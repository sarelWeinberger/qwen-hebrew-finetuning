import pandas as pd
import os
import glob
from typing import List, Dict, Tuple
import numpy as np
from Levenshtein import distance as levenshtein_distance, editops

# Import cleaners
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from cleaners.duplicate_remove_cleaner import DuplicateRemoverCleaner
from cleaners.regex_cleaner import RegExCleaner
from utils.cleaner_constants import CLEANUP_RULES


def calculate_levenshtein_components(str1: str, str2: str) -> Tuple[int, int, int]:
    """
    Calculate the individual components of Levenshtein distance: deletions, insertions.
    
    Args:
        str1: First string
        str2: Second string
        
    Returns:
        Tuple of (deletions, insertions)
    """
    if not str1 and not str2:
        return 0, 0, 0
    if not str1:
        return 0, len(str2), 0
    if not str2:
        return len(str1), 0, 0
    
    # Get edit operations
    ops = editops(str1, str2)
    
    deletions = sum(1 for op in ops if op[0] == 'delete')
    insertions = sum(1 for op in ops if op[0] == 'insert')
    
    return deletions, insertions


def normalize_levenshtein_components(str1: str, str2: str) -> Tuple[float, float, float]:
    """
    Calculate normalized Levenshtein components between two strings.
    
    Args:
        str1: First string
        str2: Second string
        
    Returns:
        Tuple of normalized (deletions, insertions) (0-1 each)
    """
    if not str1 and not str2:
        return 0.0, 0.0, 0.0
    if not str1:
        return 0.0, 1.0, 0.0
    if not str2:
        return 1.0, 0.0, 0.0
    
    max_len = max(len(str1), len(str2))
    if max_len == 0:
        return 0.0, 0.0, 0.0
    
    deletions, insertions = calculate_levenshtein_components(str1, str2)
    
    norm_deletions = deletions / max_len
    norm_insertions = insertions / max_len

    
    return norm_deletions, norm_insertions


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


def has_cleaned_text_column(file_path: str) -> bool:
    """
    Check if a file already has a cleaned text column (cleaned_text, model_cleaned, etc.).
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        True if the file has a cleaned text column, False otherwise
    """
    try:
        df = pd.read_csv(file_path)
        # Check for various cleaned text column names (case insensitive)
        cleaned_text_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['cleaned_text', 'model_cleaned', 'cleaned'])]
        return len(cleaned_text_cols) > 0
    except Exception as e:
        print(f"Error checking columns in {file_path}: {str(e)}")
        return False


def load_benchmark_data(file_path: str) -> pd.DataFrame:
    """
    Load benchmark data from CSV file.
    
    Args:
        file_path: Path to the CSV file
        
    Returns:
        DataFrame with original_text and manual_clean columns
    """
    df = pd.read_csv(file_path)
    
    # Ensure original_text column exists
    if 'original_text' not in df.columns:
        raise ValueError(f"File {file_path} must contain 'original_text' column")
    
    # Find manual clean column (supports various names)
    manual_clean_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['manual_clean', 'manual_cleaned'])]
    if not manual_clean_cols:
        raise ValueError(f"File {file_path} must contain a manual clean column (manual_clean or manual_cleaned)")
    
    manual_clean_col = manual_clean_cols[0]
    print(f"Using manual clean column: {manual_clean_col}")
    
    # Handle NaN values by converting them to empty strings
    df['original_text'] = df['original_text'].fillna('')
    df[manual_clean_col] = df[manual_clean_col].fillna('')
    
    # Rename the manual clean column to 'manual_clean' for consistency
    df = df.rename(columns={manual_clean_col: 'manual_clean'})
    
    return df


def get_cleaners() -> List[Tuple[str, object]]:
    """
    Define the list of cleaners to apply in order.
    Each cleaner should be a tuple of (name, cleaner_instance).
    
    Returns:
        List of (cleaner_name, cleaner_instance) tuples
    """
    cleaners = [
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
    Calculate normalized Levenshtein distances and components for each cleaning step (cumulative).
    
    Args:
        df: Original DataFrame with manual_clean column
        cleaned_dfs: Dictionary of cleaned DataFrames for each step (cumulative)
        
    Returns:
        Dictionary with Levenshtein distance and component metrics
    """
    metrics = {}
    
    # Distance from original to manual
    orig_to_manual = [normalize_levenshtein_distance(str(orig), str(manual))
                      for orig, manual in zip(df['original_text'], df['manual_clean'])
                      if not (pd.isna(orig) or pd.isna(manual))]
    metrics['levenshtein_dist_original_to_manual'] = round(np.mean(orig_to_manual), 3) if orig_to_manual else 0.0
    
    # Components from original to manual
    orig_to_manual_components = [normalize_levenshtein_components(str(orig), str(manual))
                                for orig, manual in zip(df['original_text'], df['manual_clean'])
                                if not (pd.isna(orig) or pd.isna(manual))]
    if orig_to_manual_components:
        avg_deletions, avg_insertions = zip(*orig_to_manual_components)
        metrics['levenshtein_deletions_original_to_manual'] = round(np.mean(avg_deletions), 3)
        metrics['levenshtein_insertions_original_to_manual'] = round(np.mean(avg_insertions), 3)
    else:
        metrics['levenshtein_deletions_original_to_manual'] = 0.0
        metrics['levenshtein_insertions_original_to_manual'] = 0.0
    
    # Cumulative: duplicate_remover, then duplicate_and_regex
    step_names = list(cleaned_dfs.keys())
    prev_df = df
    for i, step_name in enumerate(step_names):
        cleaned_df = cleaned_dfs[step_name]
        # Distance from this step's output to manual
        step_to_manual = [normalize_levenshtein_distance(str(cleaned), str(manual))
                          for cleaned, manual in zip(cleaned_df['text'], df['manual_clean'])
                          if not (pd.isna(cleaned) or pd.isna(manual))]
        
        # Components from this step's output to manual
        step_to_manual_components = [normalize_levenshtein_components(str(cleaned), str(manual))
                                    for cleaned, manual in zip(cleaned_df['text'], df['manual_clean'])
                                    if not (pd.isna(cleaned) or pd.isna(manual))]
        
        if i == 0:
            metrics[f'levenshtein_dist_{step_name}_to_manual'] = round(np.mean(step_to_manual), 3) if step_to_manual else 0.0
            if step_to_manual_components:
                avg_deletions, avg_insertions = zip(*step_to_manual_components)
                metrics[f'levenshtein_deletions_{step_name}_to_manual'] = round(np.mean(avg_deletions), 3)
                metrics[f'levenshtein_insertions_{step_name}_to_manual'] = round(np.mean(avg_insertions), 3)
            else:
                metrics[f'levenshtein_deletions_{step_name}_to_manual'] = 0.0
                metrics[f'levenshtein_insertions_{step_name}_to_manual'] = 0.0
        else:
            # Cumulative name for all up to this step
            cum_name = '_and_'.join(step_names[:i+1])
            metrics[f'levenshtein_dist_{cum_name}_to_manual'] = round(np.mean(step_to_manual), 3) if step_to_manual else 0.0
            if step_to_manual_components:
                avg_deletions, avg_insertions = zip(*step_to_manual_components)
                metrics[f'levenshtein_deletions_{cum_name}_to_manual'] = round(np.mean(avg_deletions), 3)
                metrics[f'levenshtein_insertions_{cum_name}_to_manual'] = round(np.mean(avg_insertions), 3)
            else:
                metrics[f'levenshtein_deletions_{cum_name}_to_manual'] = 0.0
                metrics[f'levenshtein_insertions_{cum_name}_to_manual'] = 0.0
        prev_df = cleaned_df
    return metrics


def calculate_metrics_for_precleaned_file(df: pd.DataFrame) -> Dict[str, float]:
    """
    Calculate metrics for files that already have a cleaned text column.
    
    Args:
        df: DataFrame with original_text, manual_clean, and a cleaned text column
        
    Returns:
        Dictionary with Levenshtein distance and component metrics
    """
    metrics = {}
    
    # Find the cleaned text column (supports various names)
    cleaned_text_cols = [col for col in df.columns if any(keyword in col.lower() for keyword in ['cleaned_text', 'model_cleaned', 'cleaned'])]
    if not cleaned_text_cols:
        raise ValueError("No cleaned text column found in pre-cleaned file")
    
    # Use the first cleaned text column found
    cleaned_text_col = cleaned_text_cols[0]
    print(f"Using pre-cleaned column: {cleaned_text_col}")
    
    # Handle NaN values in cleaned text column
    df[cleaned_text_col] = df[cleaned_text_col].fillna('')
    
    # Distance from original to manual
    orig_to_manual = [normalize_levenshtein_distance(str(orig), str(manual))
                      for orig, manual in zip(df['original_text'], df['manual_clean'])
                      if not (pd.isna(orig) or pd.isna(manual))]
    metrics['levenshtein_dist_original_to_manual'] = round(np.mean(orig_to_manual), 3) if orig_to_manual else 0.0
    
    # Components from original to manual
    orig_to_manual_components = [normalize_levenshtein_components(str(orig), str(manual))
                                for orig, manual in zip(df['original_text'], df['manual_clean'])
                                if not (pd.isna(orig) or pd.isna(manual))]
    if orig_to_manual_components:
        avg_deletions, avg_insertions = zip(*orig_to_manual_components)
        metrics['levenshtein_deletions_original_to_manual'] = round(np.mean(avg_deletions), 3)
        metrics['levenshtein_insertions_original_to_manual'] = round(np.mean(avg_insertions), 3)
    else:
        metrics['levenshtein_deletions_original_to_manual'] = 0.0
        metrics['levenshtein_insertions_original_to_manual'] = 0.0

    
    # Distance from cleaned text to manual
    cleaned_to_manual = [normalize_levenshtein_distance(str(cleaned), str(manual))
                         for cleaned, manual in zip(df[cleaned_text_col], df['manual_clean'])
                         if not (pd.isna(cleaned) or pd.isna(manual))]
    metrics[f'levenshtein_dist_{cleaned_text_col}_to_manual'] = round(np.mean(cleaned_to_manual), 3) if cleaned_to_manual else 0.0
    
    # Components from cleaned text to manual
    cleaned_to_manual_components = [normalize_levenshtein_components(str(cleaned), str(manual))
                                   for cleaned, manual in zip(df[cleaned_text_col], df['manual_clean'])
                                   if not (pd.isna(cleaned) or pd.isna(manual))]
    if cleaned_to_manual_components:
        avg_deletions, avg_insertions = zip(*cleaned_to_manual_components)
        metrics[f'levenshtein_deletions_{cleaned_text_col}_to_manual'] = round(np.mean(avg_deletions), 3)
        metrics[f'levenshtein_insertions_{cleaned_text_col}_to_manual'] = round(np.mean(avg_insertions), 3)
    else:
        metrics[f'levenshtein_deletions_{cleaned_text_col}_to_manual'] = 0.0
        metrics[f'levenshtein_insertions_{cleaned_text_col}_to_manual'] = 0.0    
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
    
    # Check if file already has cleaned_text column
    is_precleaned = has_cleaned_text_column(file_path)
    
    if is_precleaned:
        print(f"File {file_path} already has cleaned_text column - skipping cleaning process")
        # Load data
        df = load_benchmark_data(file_path)
        
        # Calculate metrics for pre-cleaned file
        metrics = calculate_metrics_for_precleaned_file(df)
        
        # Return the original DataFrame as result (no additional cleaning needed)
        result_df = df.copy()
        
    else:
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


def demonstrate_improved_metrics():
    """
    Demonstrate how the new component-based metrics better distinguish between cleaning approaches.
    This function shows the example from the email discussion about HTML tag cleaning.
    """
    print("\n=== DEMONSTRATION OF IMPROVED METRICS ===")
    print("Example from email discussion: HTML tag cleaning")
    
    # Original text with broken HTML tag
    original = "<p>הספר מעולה. אני <b>לא<b> חושבת שיש לו תחליף.</p>"
    gold = "הספר מעולה. אני לא חושבת שיש לו תחליף."
    
    # Three different cleaning approaches
    script_a = "הספר מעולה. אני <b>לא<b> חושבת שיש לו תחליף."  # Conservative
    script_b = "הספר מעולה. אני לא<b> חושבת שיש לו תחליף."      # Incomplete
    script_c = "הספר מעולה. אני חושבת שיש לו תחליף."            # Aggressive
    
    print(f"\nOriginal: {original}")
    print(f"Gold: {gold}")
    print(f"Script A (Conservative): {script_a}")
    print(f"Script B (Incomplete): {script_b}")
    print(f"Script C (Aggressive): {script_c}")
    
    # Calculate metrics for each approach
    approaches = [
        ("Script A (Conservative)", script_a),
        ("Script B (Incomplete)", script_b),
        ("Script C (Aggressive)", script_c)
    ]
    
    print(f"\n--- METRICS COMPARISON ---")
    print(f"{'Approach':<25} {'Distance':<10} {'Deletions':<10} {'Insertions':<10}")
    print("-" * 75)
    
    for name, cleaned in approaches:
        # Calculate components
        deletions, insertions = calculate_levenshtein_components(cleaned, gold)
        distance = deletions + insertions
        
        # Normalize by max length
        max_len = max(len(cleaned), len(gold))
        norm_distance = distance / max_len if max_len > 0 else 0
        norm_deletions = deletions / max_len if max_len > 0 else 0
        norm_insertions = insertions / max_len if max_len > 0 else 0
        
        print(f"{name:<25} {norm_distance:<10.3f} {norm_deletions:<10.3f} {norm_insertions:<10.3f}")
    
    print(f"\n--- INTERPRETATION ---")
    print("• Script A: High insertions (HTML tags remain) - under-cleaning")
    print("• Script B: Medium insertions (partial HTML cleanup) - incomplete")
    print("• Script C: High deletions (content removed) - over-cleaning")
    print("\nThe new metrics reveal that Script C is the most problematic due to high deletions,")
    print("even though it has the lowest overall distance score!")


def main():
    """
    Main function to process all benchmark files.
    """
    # Demonstrate the improved metrics first
    demonstrate_improved_metrics()
    
    benchmark_dir = "banchmark"
    # Get all CSV files in the benchmark directory
    all_csv_files = glob.glob(os.path.join('text_cleaning' , os.path.join(benchmark_dir, "*.csv")))
    
    # Filter to include both original benchmark files and pre-cleaned files
    benchmark_files = []
    for file_path in all_csv_files:
        file_name = os.path.basename(file_path)
        # Include files that start with "data-clean-banchmark - " and don't end with "_with_cleaned.csv"
        # OR files that have cleaned_text column (regardless of name)
        if (file_name.startswith("data-clean-banchmark - ") and not file_name.endswith("_with_cleaned.csv")) or has_cleaned_text_column(file_path):
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
    
    # Calculate overall averages for different metric types
    print(f"\n=== OVERALL AVERAGES ===")
    
    # Distance metrics
    distance_cols = [col for col in summary_df.columns if col.startswith('levenshtein_dist_')]
    if distance_cols:
        print(f"\n--- Distance Metrics ---")
        for col in distance_cols:
            avg_val = summary_df[col].mean()
            print(f"{col}: {avg_val:.4f}")
    
    # Deletion metrics (under-cleaning - less problematic)
    deletion_cols = [col for col in summary_df.columns if col.startswith('levenshtein_deletions_')]
    if deletion_cols:
        print(f"\n--- Deletion Metrics (Under-cleaning) ---")
        for col in deletion_cols:
            avg_val = summary_df[col].mean()
            print(f"{col}: {avg_val:.4f}")
    
    # Insertion metrics (over-cleaning - more problematic)
    insertion_cols = [col for col in summary_df.columns if col.startswith('levenshtein_insertions_')]
    if insertion_cols:
        print(f"\n--- Insertion Metrics (Over-cleaning) ---")
        for col in insertion_cols:
            avg_val = summary_df[col].mean()
            print(f"{col}: {avg_val:.4f}")
    
    # Print interpretation guide
    print(f"\n=== INTERPRETATION GUIDE ===")
    print("• Deletions (Under-cleaning): Lower values are better - indicates less content was removed")
    print("• Insertions (Over-cleaning): Lower values are better - indicates less content was added")
    print("• Overall Distance: Lower values are better - indicates closer match to manual cleaning")


if __name__ == "__main__":
    main() 