import os
import json
import re
import pandas as pd
import tempfile
import shutil
from s3_utils import download_s3_directory, create_temp_directory, cleanup_temp_directory, is_s3_path

def summarize_benchmark_runs(scores_sum_dir, local_save_dir=None,csv_filename='benchmark_results_summary.csv'):
    """
    Summarize benchmark runs from either local directory or S3 path.
    
    Args:
        scores_sum_dir (str): Path to scores directory (local path or S3 path starting with 's3://')
        local_save_dir (str, optional): Local directory to save the CSV. If None, saves to current directory.
    
    Returns:
        pd.DataFrame: DataFrame containing the benchmark results summary
    """
    temp_dir = None
    original_path = scores_sum_dir
    
    try:
        # Handle S3 path
        if is_s3_path(scores_sum_dir):
            print(f"Detected S3 path: {scores_sum_dir}")
            temp_dir = create_temp_directory()
            print(f"Created temporary directory: {temp_dir}")
            scores_sum_dir = download_s3_directory(scores_sum_dir, temp_dir)
            print(f"Downloaded S3 data to: {scores_sum_dir}")
        
        # Process the directory (same logic as before)
        run_dirs = [os.path.join(scores_sum_dir, d) for d in os.listdir(scores_sum_dir) if os.path.isdir(os.path.join(scores_sum_dir, d))]
        all_benchmarks = set()
        rows = []
        
        for run_dir in run_dirs:
            run_info = {}
            run_info['timestamp'] = os.path.basename(run_dir)
            samples_number = None
            model_name = None
            
            # Search for JSON files in all subdirectories of run_dir
            for subdir in os.listdir(run_dir):
                subdir_path = os.path.join(run_dir, subdir)
                if os.path.isdir(subdir_path):
                    for file in os.listdir(subdir_path):
                        if file.endswith('.json'):
                            with open(os.path.join(subdir_path, file), 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                results_key = next(iter(data['results']))
                                benchmark_name = data['config_tasks'][results_key].get('name', results_key)
                                acc = data['results'][results_key].get('acc')
                                acc_stderr = data['results'][results_key].get('acc_stderr')
                                model_name = data['config_general'].get('model_name')
                                samples_number = data['config_general'].get('max_samples')
                                run_info[f'{benchmark_name}_score'] = acc
                                run_info[f'{benchmark_name}_std'] = acc_stderr
                                all_benchmarks.add(benchmark_name)
            
            run_info['samples_number'] = samples_number
            run_info['model_name'] = model_name
            rows.append(run_info)
        
        # Build columns
        columns = ['timestamp', 'samples_number', 'model_name']
        for bench in sorted(all_benchmarks):
            columns.append(f'{bench}_score')
            columns.append(f'{bench}_std')
        
        df = pd.DataFrame(rows)
        df = df[columns]
        
        print(f"Extracted {len(rows)} runs.")
        print("Columns:", df.columns.tolist())
        print(df)
        
        # Determine where to save the CSV
        if local_save_dir is None:
            local_save_dir = os.getcwd()  # Current directory

        # Ensure local save directory exists
        os.makedirs(local_save_dir, exist_ok=True)

        csv_path = os.path.join(local_save_dir, csv_filename)

        # Remove existing file if it exists to ensure complete override
        if os.path.exists(csv_path):
            os.remove(csv_path)
            print(f"Removed existing file: {csv_path}")

        # Save the new CSV file
        df.to_csv(csv_path, index=False)
        print(f"Saved summary to {csv_path}")

        return df
        
    finally:
        # Clean up temporary directory if it was created
        if temp_dir:
            cleanup_temp_directory(temp_dir)

if __name__ == "__main__":
    # Example usage with local path
    # scores_sum_directory = '/home/ec2-user/qwen-hebrew-finetuning/hebrew_benchmark_results/scores_sum'
    
    # Example usage with S3 path
    # scores_sum_directory = 's3://your-bucket/hebrew_benchmark_results/scores_sum'
    
    # Use the original local path as default
    scores_sum_directory = 's3://gepeta-datasets/benchmark_results/heb_benc_results/'
    
    # You can specify a custom local directory to save the CSV
    local_save_directory = None  # Will save to current directory
    
    summarize_benchmark_runs(scores_sum_directory, local_save_directory)

# Example bash commands to run this function:
# For local path:
# python -c "from extract_benchmark_results import summarize_benchmark_runs; summarize_benchmark_runs('/home/ec2-user/qwen-hebrew-finetuning/hebrew_benchmark_results/scores_sum')"

# For S3 path:
# python -c "from extract_benchmark_results import summarize_benchmark_runs; summarize_benchmark_runs('s3://your-bucket/hebrew_benchmark_results/scores_sum')"

# For S3 path with custom local save directory:
# python -c "from extract_benchmark_results import summarize_benchmark_runs; summarize_benchmark_runs('s3://your-bucket/hebrew_benchmark_results/scores_sum', '/path/to/save/csv')"
