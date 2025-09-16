import os
import json
import re
import pandas as pd

def summarize_benchmark_runs(scores_sum_dir):
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
    print(df.columns)
    print(df)
    df.to_csv(scores_sum_dir+'/benchmark_results_summary.csv', index=False)
    print(f"Saved summary to {scores_sum_dir+'/benchmark_results_summary.csv'}")
    return df

if __name__ == "__main__":
    scores_sum_directory = '/home/ec2-user/qwen-hebrew-finetuning/hebrew_benchmark_results/scores_sum'
    summarize_benchmark_runs(scores_sum_directory)
# Example bash command to run this function:
# python -c "from extract_benchmark_results import summarize_benchmark_runs; summarize_benchmark_runs('/home/ec2-user/qwen-hebrew-finetuning/hebrew_benchmark_results/scores_sum')"
