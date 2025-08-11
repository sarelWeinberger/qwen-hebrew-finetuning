"""
Registry for SpaceFixCleaner sources
"""

from fetchers.s3_source_fetcher import S3SourceFetcher
from cleaners.spacefix_cleaner import SpaceFixCleaner

REGISTRY = {
    'AllHebNLI': {
        'fetcher': S3SourceFetcher(
            bucket_name='gepeta-datasets',
            prefix='processed_cleaned_filtered/run_4/AllHebNLI/filtering/output/',
            source_name='000',
            output_prefix='processed_and_cleaned_spacefixed/AllHebNLI',
            output_bucket_name='gepeta-datasets'
        ),
        'cleaner': SpaceFixCleaner()
    },
    'AllOfHEOscarData-Combined-Deduped-DC4.forgpt': {
        'fetcher': S3SourceFetcher(
            bucket_name='gepeta-datasets',
            prefix='processed_cleaned_filtered/run_4/AllOfHEOscarData/filtering/output/',
            source_name='000',
            output_prefix='processed_and_cleaned_spacefixed/AllOfHEOscarData',
            output_bucket_name='gepeta-datasets'
        ),
        'cleaner': SpaceFixCleaner()
    },
    # 'AllTzenzuraData-Combined-Deduped-DC4.forgpt': {
    #     'fetcher': S3SourceFetcher(
    #         bucket_name='israllm-datasets',
    #         prefix='raw-datasets/rar/csv_output/',
    #         source_name='AllTzenzuraData-Combined-Deduped-DC4.forgpt',
    #         output_prefix='processed_and_cleaned/AllTzenzuraData',
    #         output_bucket_name='gepeta-datasets'
    #     ),
    #     'cleaner': SpaceFixCleaner()
    # },
    # 'BooksNLI2-Combined-Deduped.forgpt': {
    #     'fetcher': S3SourceFetcher(
    #         bucket_name='israllm-datasets',
    #         prefix='raw-datasets/rar/csv_output/',
    #         source_name='BooksNLI2-Combined-Deduped.forgpt',
    #         output_prefix='processed_and_cleaned/BooksNLI2',
    #         output_bucket_name='gepeta-datasets'
    #     ),
    #     'cleaner': SpaceFixCleaner()
    # },
    # 'GeektimeCorpus-Combined-Deduped.forgpt': {
    #     'fetcher': S3SourceFetcher(
    #         bucket_name='israllm-datasets',
    #         prefix='raw-datasets/rar/csv_output/',
    #         source_name='GeektimeCorpus-Combined-Deduped.forgpt',
    #         output_prefix='processed_and_cleaned/GeektimeCorpus',
    #         output_bucket_name='gepeta-datasets'
    #     ),
    #     'cleaner': SpaceFixCleaner()
    # },
    # 'hebrew_tweets_text_clean_full-Deduped.forgpt': {
    #     'fetcher': S3SourceFetcher(
    #         bucket_name='israllm-datasets',
    #         prefix='raw-datasets/rar/csv_output/',
    #         source_name='hebrew_tweets_text_clean_full-Deduped.forgpt',
    #         output_prefix='processed_and_cleaned/hebrew_tweets',
    #         output_bucket_name='gepeta-datasets'
    #     ),
    #     'cleaner': SpaceFixCleaner()
    # },
    # 'HeC4DictaCombined-Clean-Deduped.forgpt': {
    #     'fetcher': S3SourceFetcher(
    #         bucket_name='israllm-datasets',
    #         prefix='raw-datasets/rar/csv_output/',
    #         source_name='HeC4DictaCombined-Clean-Deduped.forgpt',
    #         output_prefix='processed_and_cleaned/HeC4DictaCombined',
    #         output_bucket_name='gepeta-datasets'
    #     ),
    #     'cleaner': SpaceFixCleaner()
    # },
    # 'YifatDataBatch2-Round3-Deduped.forgpt': {
    #     'fetcher': S3SourceFetcher(
    #         bucket_name='israllm-datasets',
    #         prefix='raw-datasets/rar/csv_output/',
    #         source_name='YifatDataBatch2-Round3-Deduped.forgpt',
    #         output_prefix='processed_and_cleaned/YifatDataBatch2',
    #         output_bucket_name='gepeta-datasets'
    #     ),
    #     'cleaner': SpaceFixCleaner()
    # },
    # 'YifatDataRound2-Deduped.forgpt': {
    #     'fetcher': S3SourceFetcher(
    #         bucket_name='israllm-datasets',
    #         prefix='raw-datasets/rar/csv_output/',
    #         source_name='YifatDataRound2-Deduped.forgpt',
    #         output_prefix='processed_and_cleaned/YifatDataRound2',
    #         output_bucket_name='gepeta-datasets'
    #     ),
    #     'cleaner': SpaceFixCleaner()
    # },
    # 'YifatToCombine-Deduped.forgpt': {
    #     'fetcher': S3SourceFetcher(
    #         bucket_name='israllm-datasets',
    #         prefix='raw-datasets/rar/csv_output/',
    #         source_name='YifatToCombine-Deduped.forgpt',
    #         output_prefix='processed_and_cleaned/YifatToCombine',
    #         output_bucket_name='gepeta-datasets'
    #     ),
    #     'cleaner': SpaceFixCleaner()
    # },
    # 'YisraelHayomData-Combined-Deduped.forgpt': {
    #     'fetcher': S3SourceFetcher(
    #         bucket_name='israllm-datasets',
    #         prefix='raw-datasets/rar/csv_output/',
    #         source_name='YisraelHayomData-Combined-Deduped.forgpt',
    #         output_prefix='processed_and_cleaned/YisraelHayomData',
    #         output_bucket_name='gepeta-datasets'
    #     ),
    #     'cleaner': SpaceFixCleaner()
    # },
    # 'FineWeb2': {
    #     'fetcher': S3SourceFetcher(
    #         bucket_name='israllm-datasets',
    #         prefix='raw-datasets/fineweb2',
    #         source_name='batch',
    #         output_prefix='processed_and_cleaned/FineWeb2/',
    #         output_bucket_name='gepeta-datasets'
    #     ),
    #     'cleaner': SpaceFixCleaner()
    # },
    # 'HeC4': {
    #     'fetcher': S3SourceFetcher(
    #         bucket_name='israllm-datasets',
    #         prefix='raw-datasets/HeC4',
    #         source_name='part',
    #         output_prefix='processed_and_cleaned/HeC4-HF',
    #         output_bucket_name='gepeta-datasets'
    #     ),
    #     'cleaner': SpaceFixCleaner()
    # },
    # 'SupremeCourtOfIsrael': {
    #     'fetcher': S3SourceFetcher(
    #         bucket_name='israllm-datasets',
    #         prefix='raw-datasets/SupremeCourtOfIsrael/text_extraction/',
    #         source_name='batch',
    #         output_prefix='processed_and_cleaned/SupremeCourtOfIsrael',
    #         output_bucket_name='gepeta-datasets'
    #     ),
    #     'cleaner': SpaceFixCleaner()
    # },
    # 'YifatDataBatch2-Round4': {
    #     'fetcher': S3SourceFetcher(
    #         bucket_name='israllm-datasets',
    #         prefix='raw-datasets/Yifat4+5/csv_output',
    #         source_name='YifatDataBatch2-Round4',
    #         output_prefix='processed_and_cleaned/YifatDataBatch2-Round4',
    #         output_bucket_name='gepeta-datasets'
    #     ),
    #     'cleaner': SpaceFixCleaner()
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
    # },
    # 'OcrTau': {
    #     'fetcher': S3SourceFetcher(
    #         bucket_name='israllm-datasets',
    #         prefix='tau_clean/',
    #         source_name='HQ',
    #         output_prefix = 'processed_and_cleaned/TauOCR',
    #         output_bucket_name='gepeta-datasets'
    #     ),
    #     'cleaner': SpaceFixCleaner()
    # },
    # 'TauDigital': {
    #     'fetcher': S3SourceFetcher(
    #         bucket_name='israllm-datasets',
    #         prefix='tau_clean/',
    #         source_name='DigitalMarkdown_Tables.jsonl',
    #         output_prefix = 'processed_and_cleaned/TauDigital',
    #         output_bucket_name='gepeta-datasets'
    #     ),
    #     'cleaner': SpaceFixCleaner()
    # },
    # 'BIU': {
    #     'fetcher': S3SourceFetcher(
    #         bucket_name='israllm-datasets',
    #         prefix='raw-datasets/biu-drive/',
    #         source_name='AllBIUDriveDocs-MD-Deduped.forgpt.jsonl.gz',
    #         output_prefix='processed_and_cleaned/BIU',
    #         output_bucket_name='gepeta-datasets'
    #     ),
    #     'cleaner': SpaceFixCleaner()
    # },
    # 'sefaria': {
    #     'fetcher': S3SourceFetcher(
    #         bucket_name='israllm-datasets',
    #         prefix='raw-datasets/sefaria',
    #         source_name='AllOfSefariaTexts',
    #         output_prefix='processed_and_cleaned/sefaria',
    #         output_bucket_name='gepeta-datasets'
    #     ),
    #     'cleaner': SpaceFixCleaner()
    # },
    # 'COGNI': {
    #     'fetcher': S3SourceFetcher(
    #         bucket_name='israllm-datasets',
    #         prefix='raw-datasets/other_documents/parsed/',
    #         source_name='COGNI-IDF3_texts.deduped',
    #         output_prefix='processed_and_cleaned/COGNI',
    #         output_bucket_name='gepeta-datasets'
    #     ),
    #     'cleaner': SpaceFixCleaner()
    # },
    # 'kohelet': {
    #     'fetcher': S3SourceFetcher(
    #         bucket_name='israllm-datasets',
    #         prefix='raw-datasets/other_documents/parsed/',
    #         source_name='kohelet_texts.deduped',
    #         output_prefix='processed_and_cleaned/kohelet',
    #         output_bucket_name='gepeta-datasets'
    #     ),
    #     'cleaner': SpaceFixCleaner()
    # },
    # 'RAMA': {
    #     'fetcher': S3SourceFetcher(
    #         bucket_name='israllm-datasets',
    #         prefix='raw-datasets/other_documents/parsed/',
    #         source_name='RAMA_texts.deduped',
    #         output_prefix='processed_and_cleaned/RAMA',
    #         output_bucket_name='gepeta-datasets'
    #     ),
    #     'cleaner': SpaceFixCleaner()
    # },
    # 'SecuritiesAuthority': {
    #     'fetcher': S3SourceFetcher(
    #         bucket_name='israllm-datasets',
    #         prefix='raw-datasets/other_documents/parsed/',
    #         source_name='SecuritiesAuthority_texts.deduped',
    #         output_prefix='processed_and_cleaned/SecuritiesAuthority',
    #         output_bucket_name='gepeta-datasets'
    #     ),
    #     'cleaner': SpaceFixCleaner()
    # },
    # 'StateComptrollerReports': {
    #     'fetcher': S3SourceFetcher(
    #         bucket_name='israllm-datasets',
    #         prefix='raw-datasets/other_documents/parsed/',
    #         source_name='StateComptrollerReports_texts.deduped',
    #         output_prefix='processed_and_cleaned/StateComptrollerReports',
    #         output_bucket_name='gepeta-datasets'
    #     ),
    #     'cleaner': SpaceFixCleaner()
    # }
}
