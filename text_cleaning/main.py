from cleaning_pipeline import CleaningPipeline
from fetchers.s3_source_fetcher import S3SourceFetcher
from cleaners.regex_cleaner import RegExCleaner
from utils.cleaner_constants import SOURCES, CLEANUP_RULES
import argparse


def create_registry_regex_only(debug_mode: bool = False):
    """
    Create registry with only RegExCleaner for all sources.
    """
    registry = {
        source_name: {
            'fetcher': S3SourceFetcher(
                bucket_name='israllm-datasets',
                prefix='raw-datasets/rar/csv_output/',
                source_name=source_name,
                output_prefix='test_clean_round_2'
            ),
            'cleaner': RegExCleaner(
                patterns=[(rule['regex'][0], rule['regex'][1]) for rule in CLEANUP_RULES],
                debug_mode=debug_mode,
                save_cleaned_data=False
            )
        }
        for source_name in SOURCES
    }
    return registry


def run_all_samples(debug_mode: bool = False):
    """
    Run the sample cleaning pipeline for all sources using only RegExCleaner.
    """
    registry = create_registry_regex_only(debug_mode=debug_mode)
    for source_name, components in registry.items():
        print(f"\nProcessing sample for source: {source_name}")
        try:
            pipeline = CleaningPipeline(
                fetcher=components["fetcher"],
                cleaner=components["cleaner"],
                source_name=source_name
            )
            custom_output_prefix = "partly-processed/round_2_dataset_examples/"
            pipeline.run_sample_mode(custom_output_prefix=custom_output_prefix,
                                     custom_bucket_name='gepeta-datasets')


            print(f"Successfully processed sample for {source_name}")
        except Exception as e:
            print(f"Error processing sample for {source_name}: {str(e)}")


if __name__ == "__main__":
    run_all_samples(debug_mode=True)