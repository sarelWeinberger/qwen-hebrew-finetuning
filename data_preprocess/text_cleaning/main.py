from cleaning_pipeline import CleaningPipeline
from fetchers.s3_source_fetcher import S3SourceFetcher
from cleaners.regex_cleaner import RegExCleaner
from cleaners.spacefix_cleaner import SpaceFixCleaner  # Added import
from utils.cleaner_constants import CLEANUP_RULES
import pandas as pd
from utils.regex_registry import REGISTRY as regex_registry
from utils.spacefix_registry import REGISTRY as spacefix_registry


def run_all_samples(registry, debug_mode=False):
    """
    Run the sample cleaning pipeline for all sources using only RegExCleaner.
    """
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


def run_full_cleaning(registry):
    """
    Run the full cleaning pipeline for all sources using only RegExCleaner.
    """
    for source_name, components in registry.items():
        print(f"\nProcessing full cleaning for source: {source_name}")
        try:
            pipeline = CleaningPipeline(
                fetcher=components["fetcher"],
                cleaner=components["cleaner"],
                source_name=source_name
            )
            pipeline.run()
            print(f"Successfully processed full cleaning for {source_name}")
        except Exception as e:
            print(f"Error processing full cleaning for {source_name}: {str(e)}")


def count_words_for_all_sources(registry):
    """
    Count words before and after cleaning for all sources and save comprehensive results.
    """
    from simple_word_count_analyzer import count_words_in_source, count_words_after_cleaning
    
    results = []
    
    print("Counting words for all sources...")
    
    for source_name, components in registry.items():
        print(f"\nProcessing source: {source_name}")
        
        try:
            fetcher = components["fetcher"]
            
            # Count words in raw data
            raw_words, raw_files = count_words_in_source(
                bucket_name=fetcher.bucket_name,
                prefix=fetcher.prefix,
                source_name=fetcher.source_name
            )
            
            # Count words in cleaned data
            cleaned_words, cleaned_files = count_words_after_cleaning(
                output_bucket_name=fetcher.output_bucket_name,
                output_prefix=fetcher.output_prefix
            )
            
            # Calculate reduction percentage
            reduction_percent = ((raw_words - cleaned_words) / raw_words * 100) if raw_words > 0 else 0
            
            results.append({
                'source_name': source_name,
                'raw_words': raw_words,
                'cleaned_words': cleaned_words,
                'reduction_percent': reduction_percent,
                'raw_files': raw_files,
                'cleaned_files': cleaned_files,
                'raw_s3_path': f"s3://{fetcher.bucket_name}/{fetcher.prefix}",
                'cleaned_s3_path': f"s3://{fetcher.output_bucket_name}/{fetcher.output_prefix}"
            })
            
            print(f"  Raw: {raw_words:,} words ({raw_files} files)")
            print(f"  Cleaned: {cleaned_words:,} words ({cleaned_files} files)")
            print(f"  Reduction: {reduction_percent:.2f}%")
            
        except Exception as e:
            print(f"  Error processing {source_name}: {str(e)}")
            results.append({
                'source_name': source_name,
                'raw_words': 0,
                'cleaned_words': 0,
                'reduction_percent': 0,
                'raw_files': 0,
                'cleaned_files': 0,
                'raw_s3_path': f"s3://{components['fetcher'].bucket_name}/{components['fetcher'].prefix}",
                'cleaned_s3_path': f"s3://{components['fetcher'].output_bucket_name}/{components['fetcher'].output_prefix}"
            })
    
    # Create comprehensive summary
    total_raw = sum(r['raw_words'] for r in results)
    total_cleaned = sum(r['cleaned_words'] for r in results)    
    summary = f"""n words before cleaning: {total_raw:,}
n words after cleaning: {total_cleaned:,}

Individual sources:
"""
    
    for result in results:
        summary += f"""{result['source_name']}:
n words before cleaning: {result['raw_words']:,}
n words after cleaning: {result['cleaned_words']:,}

"""
    
    # Save to a common location
    try:
        import boto3
        s3 = boto3.client("s3")
        
        filename = f"word_count_all_sources.txt"
        
        # Save to gepeta-datasets bucket in a common location
        output_key = f"word_count_analyses/{filename}"
        
        s3.put_object(
            Bucket='gepeta-datasets',
            Key=output_key,
            Body=summary.encode('utf-8'),
            ContentType='text/plain'
        )
        
        print(f"\nComprehensive analysis saved to: s3://gepeta-datasets/{output_key}")
        
    except Exception as e:
        print(f"Error saving comprehensive analysis: {str(e)}")
    
    return results


if __name__ == "__main__":
    # Default: run full cleaning. To run samples, call run_all_samples()
    # To count words for all sources, call count_words_for_all_sources()
    registry = regex_registry
    run_full_cleaning(registry)
    
