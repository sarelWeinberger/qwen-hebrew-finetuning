from fineweb_filtering_pipeline import fineweb_filtering_pipeline_run
from S3_files_tokens_sizes import summarize_counts, count_tokens
from count_tokens import count_tokens_mega_parallel
if __name__ == '__main__':

    s3_bucket, prefix = fineweb_filtering_pipeline_run()

    s3_bucket = 'gepeta-datasets'
    prefix= 'processed_cleaned_filtered/run_5/'
    token_counts_prefix = f'{prefix}token_counts'
    count_tokens_mega_parallel(s3_bucket, prefix)
    summarize_counts(s3_bucket, token_counts_prefix)