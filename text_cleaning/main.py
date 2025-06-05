from .cleaning_pipeline import CleaningPipeline
from .fetchers.s3_source_fetcher import S3SourceFetcher
from .utils import composite_cleaner_registry, SOURCES

# Create registry with all sources
registry = {
    source_name: {
        'fetcher': S3SourceFetcher(
            bucket_name='israllm-datasets',
            prefix='raw-datasets/rar/csv_output/',
            source_name=source_name,
            output_prefix='cleaned_data'
        ),
        'cleaner': composite_cleaner_registry[source_name]
    }
    for source_name in SOURCES
}

def run_all():
    """
    Run the cleaning pipeline for all sources in the registry.
    """
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
    run_all()