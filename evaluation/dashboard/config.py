import os
# Load and process the data
local_save_directory = os.path.dirname(os.path.abspath(__file__)) 
local_save_directory = "/".join(local_save_directory.split('/')[:-1]) 
csv_filename = 'benchmark_results_summary.csv'
BUCKET_NAME = 'gepeta-datasets'
scores_sum_directory = f's3://{BUCKET_NAME}/benchmark_results/heb_benc_results/'

class Config:
    local_save_directory = local_save_directory
    csv_filename = csv_filename
    BUCKET_NAME = BUCKET_NAME
    scores_sum_directory = scores_sum_directory

if __name__ == "__main__":
    print("Config settings:")
    print(f"Local Save Directory: {Config.local_save_directory}")
    print(f"CSV Filename: {Config.csv_filename}")
    print(f"S3 Bucket Name: {Config.BUCKET_NAME}")
    print(f"Scores Summary Directory: {Config.scores_sum_directory}")