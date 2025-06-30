from cleaning_pipeline import CleaningPipeline
from fetchers.s3_source_fetcher import S3SourceFetcher
from utils.cleaner_config import create_cleaner_registries
from utils.cleaner_constants import SOURCES
import argparse

def create_registry(debug_mode: bool = False):
    """
    Create registry with all sources.
    
    Args:
        debug_mode: Whether to enable debug mode for regex cleaner
    """
    # Create cleaner registries with debug mode
    cleaner_registries = create_cleaner_registries(debug_mode=debug_mode)
    
    registry = {
        source_name: {
            'fetcher': S3SourceFetcher(
                bucket_name='israllm-datasets',
                prefix='raw-datasets/rar/csv_output/',
                source_name=source_name,
                output_prefix='test_clean_round_2'
            ),
            'cleaner': cleaner_registries['composite_cleaner_registry'][source_name]
        }
        for source_name in SOURCES
    }
    
    return registry

def run_all(debug_mode: bool = False):
    """
    Run the cleaning pipeline for all sources in the registry.
    
    Args:
        debug_mode: Whether to enable debug mode for regex cleaner
    """
    registry = create_registry(debug_mode=debug_mode)
    
    for source_name, components in registry.items():
        print(f"\nProcessing source: {source_name}")
        try:
            pipeline = CleaningPipeline(
                fetcher=components["fetcher"],
                cleaner=components["cleaner"],
                source_name=source_name
            )
            pipeline.run()
            print(f"Successfully processed {source_name}")
        except Exception as e:
            print(f"Error processing {source_name}: {str(e)}")

if __name__ == "__main__":

    run_all(debug_mode=True)