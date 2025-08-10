from cleaning_pipeline import CleaningPipeline
from fetchers.s3_source_fetcher import S3SourceFetcher
from cleaners.regex_cleaner import RegExCleaner
from cleaners.spacefix_cleaner import SpaceFixCleaner  # Added import
from utils.cleaner_constants import CLEANUP_RULES
import pandas as pd



def create_registry(debug_mode):
    """
    """

    registry = {
        # 'AllHebNLI': {
        #     'fetcher': S3SourceFetcher(
        #         bucket_name='israllm-datasets',
        #         prefix='raw-datasets/nli/csv_output/',
        #         source_name='AllHebNLI',
        #         output_prefix='processed_and_cleaned/test',
        #         output_bucket_name='gepeta-datasets'
        #     ),
        #     'cleaner': SpaceFixCleaner()  # Use SpaceFixCleaner for debug
        # },
        # 'AllOfHEOscarData-Combined-Deduped-DC4.forgpt': {
        #     'fetcher': S3SourceFetcher(
        #         bucket_name='israllm-datasets',
        #         prefix='raw-datasets/rar/csv_output/',
        #         source_name='AllOfHEOscarData-Combined-Deduped-DC4.forgpt',
        #         output_prefix='processed_and_cleaned/AllOfHEOscarData',
        #         output_bucket_name='gepeta-datasets'
        #     ),
        #     'cleaner': RegExCleaner(
        #         patterns=[(rule['regex'][0], rule['regex'][1]) for rule in CLEANUP_RULES],
        #         debug_mode=debug_mode,
        #         save_cleaned_data=False
        #     )
        # },
        # 'AllTzenzuraData-Combined-Deduped-DC4.forgpt': {
        #     'fetcher': S3SourceFetcher(
        #         bucket_name='israllm-datasets',
        #         prefix='raw-datasets/rar/csv_output/',
        #         source_name='AllTzenzuraData-Combined-Deduped-DC4.forgpt',
        #         output_prefix='processed_and_cleaned/AllTzenzuraData',
        #         output_bucket_name='gepeta-datasets'
        #     ),
        #     'cleaner': RegExCleaner(
        #         patterns=[(rule['regex'][0], rule['regex'][1]) for rule in CLEANUP_RULES],
        #         debug_mode=debug_mode,
        #         save_cleaned_data=False
        #     )
        # },
        # 'BooksNLI2-Combined-Deduped.forgpt': {
        #     'fetcher': S3SourceFetcher(
        #         bucket_name='israllm-datasets',
        #         prefix='raw-datasets/rar/csv_output/',
        #         source_name='BooksNLI2-Combined-Deduped.forgpt',
        #         output_prefix='processed_and_cleaned/BooksNLI2',
        #         output_bucket_name='gepeta-datasets'
        #     ),
        #     'cleaner': RegExCleaner(
        #         patterns=[(rule['regex'][0], rule['regex'][1]) for rule in CLEANUP_RULES],
        #         debug_mode=debug_mode,
        #         save_cleaned_data=False
        #     )
        # },
        # 'GeektimeCorpus-Combined-Deduped.forgpt': {
        #     'fetcher': S3SourceFetcher(
        #         bucket_name='israllm-datasets',
        #         prefix='raw-datasets/rar/csv_output/',
        #         source_name='GeektimeCorpus-Combined-Deduped.forgpt',
        #         output_prefix='processed_and_cleaned/GeektimeCorpus',
        #         output_bucket_name='gepeta-datasets'
        #     ),
        #     'cleaner': RegExCleaner(
        #         patterns=[(rule['regex'][0], rule['regex'][1]) for rule in CLEANUP_RULES],
        #         debug_mode=debug_mode,
        #         save_cleaned_data=False
        #     )
        # },
        # 'hebrew_tweets_text_clean_full-Deduped.forgpt': {
        #     'fetcher': S3SourceFetcher(
        #         bucket_name='israllm-datasets',
        #         prefix='raw-datasets/rar/csv_output/',
        #         source_name='hebrew_tweets_text_clean_full-Deduped.forgpt',
        #         output_prefix='processed_and_cleaned/hebrew_tweets',
        #         output_bucket_name='gepeta-datasets'
        #     ),
        #     'cleaner': RegExCleaner(
        #         patterns=[(rule['regex'][0], rule['regex'][1]) for rule in CLEANUP_RULES],
        #         debug_mode=debug_mode,
        #         save_cleaned_data=False
        #     )
        # },
        # 'HeC4DictaCombined-Clean-Deduped.forgpt': {
        #     'fetcher': S3SourceFetcher(
        #         bucket_name='israllm-datasets',
        #         prefix='raw-datasets/rar/csv_output/',
        #         source_name='HeC4DictaCombined-Clean-Deduped.forgpt',
        #         output_prefix='processed_and_cleaned/HeC4DictaCombined',
        #         output_bucket_name='gepeta-datasets'
        #     ),
        #     'cleaner': RegExCleaner(
        #         patterns=[(rule['regex'][0], rule['regex'][1]) for rule in CLEANUP_RULES],
        #         debug_mode=debug_mode,
        #         save_cleaned_data=False
        #     )
        # },
        # 'YifatDataBatch2-Round3-Deduped.forgpt': {
        #     'fetcher': S3SourceFetcher(
        #         bucket_name='israllm-datasets',
        #         prefix='raw-datasets/rar/csv_output/',
        #         source_name='YifatDataBatch2-Round3-Deduped.forgpt',
        #         output_prefix='processed_and_cleaned/YifatDataBatch2',
        #         output_bucket_name='gepeta-datasets'
        #     ),
        #     'cleaner': RegExCleaner(
        #         patterns=[(rule['regex'][0], rule['regex'][1]) for rule in CLEANUP_RULES],
        #         debug_mode=debug_mode,
        #         save_cleaned_data=False
        #     )
        # },
        # 'YifatDataRound2-Deduped.forgpt': {
        #     'fetcher': S3SourceFetcher(
        #         bucket_name='israllm-datasets',
        #         prefix='raw-datasets/rar/csv_output/',
        #         source_name='YifatDataRound2-Deduped.forgpt',
        #         output_prefix='processed_and_cleaned/YifatDataRound2',
        #         output_bucket_name='gepeta-datasets'
        #     ),
        #     'cleaner': RegExCleaner(
        #         patterns=[(rule['regex'][0], rule['regex'][1]) for rule in CLEANUP_RULES],
        #         debug_mode=debug_mode,
        #         save_cleaned_data=False
        #     )
        # },
        # 'YifatToCombine-Deduped.forgpt': {
        #     'fetcher': S3SourceFetcher(
        #         bucket_name='israllm-datasets',
        #         prefix='raw-datasets/rar/csv_output/',
        #         source_name='YifatToCombine-Deduped.forgpt',
        #         output_prefix='processed_and_cleaned/YifatToCombine',
        #         output_bucket_name='gepeta-datasets'
        #     ),
        #     'cleaner': RegExCleaner(
        #         patterns=[(rule['regex'][0], rule['regex'][1]) for rule in CLEANUP_RULES],
        #         debug_mode=debug_mode,
        #         save_cleaned_data=False
        #     )
        # },
        # 'YisraelHayomData-Combined-Deduped.forgpt': {
        #     'fetcher': S3SourceFetcher(
        #         bucket_name='israllm-datasets',
        #         prefix='raw-datasets/rar/csv_output/',
        #         source_name='YisraelHayomData-Combined-Deduped.forgpt',
        #         output_prefix='processed_and_cleaned/YisraelHayomData',
        #         output_bucket_name='gepeta-datasets'
        #     ),
        #     'cleaner': RegExCleaner(
        #         patterns=[(rule['regex'][0], rule['regex'][1]) for rule in CLEANUP_RULES],
        #         debug_mode=debug_mode,
        #         save_cleaned_data=False
        #     )
        # },
        # 'FineWeb2': {
        #     'fetcher': S3SourceFetcher(
        #         bucket_name='israllm-datasets',
        #         prefix='raw-datasets/fineweb2',
        #         source_name='batch',
        #         output_prefix='processed_and_cleaned/FineWeb2/',
        #         output_bucket_name='gepeta-datasets'
        #     ),
        #     'cleaner': RegExCleaner(
        #         patterns=[(rule['regex'][0], rule['regex'][1]) for rule in CLEANUP_RULES],
        #         debug_mode=debug_mode,
        #         save_cleaned_data=False
        #     )
        # },
        # 'HeC4': {
        #     'fetcher': S3SourceFetcher(
        #         bucket_name='israllm-datasets',
        #         prefix='raw-datasets/HeC4',
        #         source_name='part',
        #         output_prefix='processed_and_cleaned/HeC4-HF',
        #         output_bucket_name='gepeta-datasets'
        #     ),
        #     'cleaner': RegExCleaner(
        #         patterns=[(rule['regex'][0], rule['regex'][1]) for rule in CLEANUP_RULES],
        #         debug_mode=debug_mode,
        #         save_cleaned_data=False
        #     )
        # },
        # 'SupremeCourtOfIsrael': {
        #     'fetcher': S3SourceFetcher(
        #         bucket_name='israllm-datasets',
        #         prefix='raw-datasets/SupremeCourtOfIsrael/text_extraction/',
        #         source_name='batch',
        #         output_prefix='processed_and_cleaned/SupremeCourtOfIsrael',
        #         output_bucket_name='gepeta-datasets'
        #     ),
        #     'cleaner': RegExCleaner(
        #         patterns=[(rule['regex'][0], rule['regex'][1]) for rule in CLEANUP_RULES],
        #         debug_mode=debug_mode,
        #         save_cleaned_data=False
        #     )
        # },
        # 'YifatDataBatch2-Round4': {
        #     'fetcher': S3SourceFetcher(
        #         bucket_name='israllm-datasets',
        #         prefix='raw-datasets/Yifat4+5/csv_output',
        #         source_name='YifatDataBatch2-Round4',
        #         output_prefix='processed_and_cleaned/YifatDataBatch2-Round4',
        #         output_bucket_name='gepeta-datasets'
        #     ),
        #     'cleaner': RegExCleaner(
        #         patterns=[(rule['regex'][0], rule['regex'][1]) for rule in CLEANUP_RULES],
        #         debug_mode=debug_mode,
        #         save_cleaned_data=False
        #     )
        # },
        # 'YifatDataBatch3-Round5': {
        #     'fetcher': S3SourceFetcher(
        #         bucket_name='israllm-datasets',
        #         prefix='raw-datasets/Yifat4+5/csv_output',
        #         source_name='YifatDataBatch3-Round5',
        #         output_prefix='test_spacefix/YifatDataBatch2-Round5',
        #         output_bucket_name='gepeta-datasets'
        #     ),
        #     'cleaner': SpaceFixCleaner() 
        #     # 'cleaner': RegExCleaner(
        #     #     patterns=[(rule['regex'][0], rule['regex'][1]) for rule in CLEANUP_RULES],
        #     #     debug_mode=debug_mode,
        #     #     save_cleaned_data=False
        #     # )
        # },
        # 'OcrTau': {
        #     'fetcher': S3SourceFetcher(
        #         bucket_name='israllm-datasets',
        #         prefix='tau_clean/',
        #         source_name='HQ',
        #         output_prefix = 'processed_and_cleaned/TauOCR',
        #         output_bucket_name='gepeta-datasets'
        #     ),
        #     'cleaner': RegExCleaner(
        #         patterns=[(rule['regex'][0], rule['regex'][1]) for rule in CLEANUP_RULES],
        #         save_word_changes=False
        #     )
        # },
        # 'TauDigital': {
        #     'fetcher': S3SourceFetcher(
        #         bucket_name='israllm-datasets',
        #         prefix='tau_clean/',
        #         source_name='DigitalMarkdown_Tables.jsonl',
        #         output_prefix = 'processed_and_cleaned/TauDigital',
        #         output_bucket_name='gepeta-datasets'
        #     ),
        #     'cleaner': RegExCleaner(
        #         patterns=[(rule['regex'][0], rule['regex'][1]) for rule in CLEANUP_RULES],
        #         save_word_changes=False
        #     )
        # },
        # 'BIU': {
        #     'fetcher': S3SourceFetcher(
        #         bucket_name='israllm-datasets',
        #         prefix='raw-datasets/biu-drive/',
        #         source_name='AllBIUDriveDocs-MD-Deduped.forgpt.jsonl.gz',
        #         output_prefix='processed_and_cleaned/BIU',
        #         output_bucket_name='gepeta-datasets'
        #     ),
        #     'cleaner': RegExCleaner(
        #         patterns=[(rule['regex'][0], rule['regex'][1]) for rule in CLEANUP_RULES],
        #     )
        # },
        # 'sefaria': {
        #     'fetcher': S3SourceFetcher(
        #         bucket_name='israllm-datasets',
        #         prefix='raw-datasets/sefaria',
        #         source_name='AllOfSefariaTexts',
        #         output_prefix='processed_and_cleaned/sefaria',
        #         output_bucket_name='gepeta-datasets'
        #     ),
        #     'cleaner': RegExCleaner(
        #         patterns=[(rule['regex'][0], rule['regex'][1]) for rule in CLEANUP_RULES],
        #     )
        # },
        'COGNI': {
            'fetcher': S3SourceFetcher(
                bucket_name='israllm-datasets',
                prefix='raw-datasets/other_documents/parsed/',
                source_name='COGNI-IDF3_texts.deduped',
                output_prefix='processed_and_cleaned/COGNI',
                output_bucket_name='gepeta-datasets'
            ),
            'cleaner': RegExCleaner(
                patterns=[(rule['regex'][0], rule['regex'][1]) for rule in CLEANUP_RULES],
            )
        },
        'kohelet': {
            'fetcher': S3SourceFetcher(
                bucket_name='israllm-datasets',
                prefix='raw-datasets/other_documents/parsed/',
                source_name='kohelet_texts.deduped',
                output_prefix='processed_and_cleaned/kohelet',
                output_bucket_name='gepeta-datasets'
            ),
            'cleaner': RegExCleaner(
                patterns=[(rule['regex'][0], rule['regex'][1]) for rule in CLEANUP_RULES],
            )
        },
        'RAMA': {
            'fetcher': S3SourceFetcher(
                bucket_name='israllm-datasets',
                prefix='raw-datasets/other_documents/parsed/',
                source_name='RAMA_texts.deduped',
                output_prefix='processed_and_cleaned/RAMA',
                output_bucket_name='gepeta-datasets'
            ),
            'cleaner': RegExCleaner(
                patterns=[(rule['regex'][0], rule['regex'][1]) for rule in CLEANUP_RULES],
            )
        },
        'SecuritiesAuthority': {
            'fetcher': S3SourceFetcher(
                bucket_name='israllm-datasets',
                prefix='raw-datasets/other_documents/parsed/',
                source_name='SecuritiesAuthority_texts.deduped',
                output_prefix='processed_and_cleaned/SecuritiesAuthority',
                output_bucket_name='gepeta-datasets'
            ),
            'cleaner': RegExCleaner(
                patterns=[(rule['regex'][0], rule['regex'][1]) for rule in CLEANUP_RULES],
            )
        },
        'StateComptrollerReports': {
            'fetcher': S3SourceFetcher(
                bucket_name='israllm-datasets',
                prefix='raw-datasets/other_documents/parsed/',
                source_name='StateComptrollerReports_texts.deduped',
                output_prefix='processed_and_cleaned/StateComptrollerReports',
                output_bucket_name='gepeta-datasets'
            ),
            'cleaner': RegExCleaner(
                patterns=[(rule['regex'][0], rule['regex'][1]) for rule in CLEANUP_RULES],
            )
        }
    }
    return registry


def run_all_samples(debug_mode=False):
    """
    Run the sample cleaning pipeline for all sources using only RegExCleaner.
    """
    registry = create_registry(debug_mode=debug_mode)
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


def run_full_cleaning(debug_mode=False):
    """
    Run the full cleaning pipeline for all sources using only RegExCleaner.
    """
    registry = create_registry(debug_mode=debug_mode)
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


def count_words_for_all_sources():
    """
    Count words before and after cleaning for all sources and save comprehensive results.
    """
    from simple_word_count_analyzer import count_words_in_source, count_words_after_cleaning
    
    registry = create_registry(debug_mode=False)
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
    total_reduction = ((total_raw - total_cleaned) / total_raw * 100) if total_raw > 0 else 0
    
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
    run_full_cleaning()
