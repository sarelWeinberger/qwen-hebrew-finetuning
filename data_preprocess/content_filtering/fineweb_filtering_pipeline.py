#!/usr/bin/env python3
# pip install datatrove fs spacy  pandas s3fs transformers orjson openpyxl
import argparse
import yaml
import json
from openpyxl.workbook import Workbook
import fsspec
import pandas as pd
import numpy as np
from pathlib import Path
from filter_analysis import filter_analysis_pipeline

from datatrove.executor.local import LocalPipelineExecutor
from datatrove.pipeline.readers.csv import CsvReader
from datatrove.pipeline.readers.jsonl import JsonlReader
import csv
import sys
csv.field_size_limit(sys.maxsize)  
from datatrove.pipeline.writers.jsonl import JsonlWriter
from datatrove.pipeline.filters import (
    FineWebQualityFilter,
    GopherQualityFilter,
    GopherRepetitionFilter,
    URLFilter,
)
from datatrove.pipeline.filters.base_filter import BaseFilter
from datatrove.utils.text import PUNCTUATION_SET, split_into_words
from datatrove.pipeline.writers.disk_base import DiskWriter
from datatrove.pipeline.stats.token_stats import TokenStats
from datatrove.utils.typeshelper import Languages
import os
import re
from transformers import AutoModelForCausalLM, AutoTokenizer
from datatrove.data import Document
from datatrove.utils.typeshelper import Languages
from functools import partial

language = Languages.hebrew__hebr


class MinDocFilter(BaseFilter):
    name = "< Gopher MinDocFilter"

    def __init__(
        self,
        min_doc_words: int | None = 50,
        language: str = Languages.english,
        exclusion_writer: DiskWriter = None,

    ):
        """
        Filter to apply Gopher's quality heuristic rules.
        Reference: https://arxiv.org/pdf/2112.11446.pdf

        Args:
            min_doc_words:

        """
        super().__init__(exclusion_writer)
        self.min_doc_words = min_doc_words
        self.language = language

    def filter(self, doc: Document) -> bool | tuple[bool, str]:
        """

        Args:
            doc: Applies the heuristics rules to decide if a document should be REMOVED

        Returns: False if sample.text does not pass any of the the heuristic tests

        """
        text = doc.text
        words = split_into_words(text, self.language)
        n_words = len(words)

        non_symbol_words = [w for w in words if any(ch not in PUNCTUATION_SET for ch in w)]
        n_non_symbol_words_words = len(non_symbol_words)

        # words < min_doc_words or words > max_doc_words
        if self.min_doc_words and n_non_symbol_words_words < self.min_doc_words:
            return False, "gopher_short_doc"
        return True
    
# --- Configuration Parameters (set here, not via CLI) ---
BUCKET_NAME = "gepeta-datasets"
INPUT_BASE = f"s3://{BUCKET_NAME}/processed_and_cleaned"
RUN_VERSION = "run_6_test"
PREFIX = f"processed_cleaned_filtered/{RUN_VERSION}"
OUTPUT_BASE = f"s3://{BUCKET_NAME}/{PREFIX}"
CONFIG_PATH = "heb_Hebr.yml"

TASKS = 6 #90 #100

# For debugging: if True, only process one part file per dataset
DEBUG_ONE_PART = False
# Maximum number of rows per part file (None for full file)
MAX_ROWS = -1

# -------------------------------------------------------

def get_param(param_name: str,config: dict, dataset_name: str):
    """
    Gets a parameter from the config.
    
    It first checks for a dataset-specific override. If one exists, it's returned.
    Otherwise, it returns the general default value.
    
    Args:
        config: The main configuration dictionary (from the YAML).
        dataset_name: The name of the current dataset being processed.
        param_name: The name of the parameter to retrieve.
        
    Returns:
        The appropriate parameter value.
    """
    # Look for the parameter in dataset_overrides -> dataset_name -> param_name
    # The .get(key, {}) pattern safely handles cases where keys don't exist.
    return config.get('dataset_overrides', {}).get(dataset_name, {}).get(param_name, config[param_name])

def list_s3_directories(bucket_name, path=""):
    """
    List all directories in an S3 bucket using fsspec.
    
    Args:
        bucket_name (str): Name of the S3 bucket
        path (str): Optional path within the bucket (default: root)
    
    Returns:
        list: List of directory paths in the bucket
    """
    try:
        # Initialize fsspec S3 filesystem
        fs = fsspec.filesystem('s3')
        
        # Construct the full path
        if path:
            full_path = f"{bucket_name}/{path.strip('/')}"
        else:
            full_path = bucket_name
        
        # List all items in the path
        all_items = fs.ls(full_path, detail=False)
        
        # Filter for directories only
        directories = []
        for item in all_items:
            if item == INPUT_BASE.replace('s3://',''):
                continue
            if fs.isdir(item):
                # Remove bucket name and base path, keep only directory name
                clean_path = item.replace(f"{bucket_name}/", "")
                if path:
                    # Remove the base path and get only the directory name
                    base_path = path.strip('/') + '/'
                    if clean_path.startswith(base_path):
                        dir_name = clean_path[len(base_path):].split('/')[-1]
                        if dir_name and dir_name not in directories:
                            directories.append(dir_name)
                else:
                    # Get only the top-level directory name
                    dir_name = clean_path.split('/')[-1]
                    if dir_name and dir_name not in directories:
                        directories.append(dir_name)
        
        return directories
        
    except Exception as e:
        print(f"Error accessing S3 bucket: {e}")
        return []


def gather_exclusion_logs(output_base, dataset_name, filter_names):
    records = []
    fs = fsspec.filesystem('s3')
    for fname in filter_names:
        pattern = f"{output_base}/{dataset_name}/filtering/removed/{fname}/*.jsonl"
        for path in fs.glob(pattern):
            with fs.open(path, 'r') as f:
                for line in f:
                    records.append(json.loads(line))
    return pd.DataFrame(records)


def remove_illegal_characters(data):
    """Remove characters not supported by Excel."""
    ILLEGAL_CHARACTERS_RE = re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F]')
    if isinstance(data, str):
        return ILLEGAL_CHARACTERS_RE.sub("", data)
    return data

def clean_dataframe(df):
    """Apply illegal character cleaning to all string columns in the DataFrame."""
    for column in df.select_dtypes(include=['object']):
        df[column] = df[column].apply(remove_illegal_characters)
    return df

def process_stats_file(
    stats_path: str,
    dataset_name: str,
    main_output_path: str,
    filter_config,
    drop_per_list = {}
):
    summary_data = {}

    overall_samples_dropped = 0
    overall_total_samples = 0
    overall_tokens_dropped = 0
    overall_total_tokens = 0

    # Load structured stats
    stats = json.load(fsspec.open(stats_path).open())

    # Initialize
    actual_examples = 0
    source_tokens_num = 0
    source_remove_tokens_num = 0
    dropped_records = []

    # Process each pipeline stage
    for stage in stats:
        name = stage.get("name", "")
        s = stage.get("stats", {})

        # --- Reader ---
        if "READER" in name:
            actual_examples = s.get("documents", {}).get("total", 0)
            source_tokens_num = s.get("doc_len", {}).get("total", 0)

        # --- Writer ---
        elif "WRITER" in name:
            pass  # Optional: validate output stats here

        # --- Filters ---
        elif "FILTER" in name and s:
            for k, v in s.items():
                if k.startswith("dropped_") and k != "dropped":
                    dropped_records.extend([{"reason": k}] * v)

    # Compute drop rate
    drop_per = len(dropped_records) / actual_examples if actual_examples else 0
    drop_per_list[dataset_name] = drop_per
    print(f"% samples dropped: {drop_per:.2%}")

    # Process DataFrame
    if dropped_records:
        df = pd.DataFrame(dropped_records)
        reason_counts = df['reason'].value_counts()
        reason_probs = df['reason'].value_counts(normalize=True)

        # sampled_df = df.groupby('reason').apply(
        #     lambda x: x.sample(n=min(len(x), 20), random_state=42)
        # ).reset_index(drop=True)
        sampled_df = df.groupby('reason', group_keys=False).apply(
        lambda x: x.sample(n=min(len(x), filter_config['min_doc_words']), random_state=42)
    ).reset_index(drop=True)
        sampled_df['reason_filter_prob'] = sampled_df['reason'].apply(
            lambda r: reason_counts[r] / actual_examples
        )

        # Write Excel
        try:
            sampled_df.to_excel(os.path.join(main_output_path, f"{dataset_name}_dropped_records_filtered.xlsx"), index=False)
        except Exception as e:
            print(f"Excel write failed: {e}")
            cleaned_df = clean_dataframe(sampled_df)
            cleaned_df.to_excel(os.path.join(main_output_path, f"{dataset_name}_dropped_records_filtered.xlsx"), index=False)
    else:
        reason_counts = pd.Series()
        reason_probs = pd.Series({'Nothing dropped': 0.0})

    # Create summary
    summary_data[dataset_name] = {
        'total_samples': actual_examples,
        'samples_dropped': len(dropped_records),
        'drop_percentage': drop_per * 100,
        'tokens_before': source_tokens_num,
        'tokens_after': source_tokens_num,  # No token-level drops yet
        'tokens_dropped': source_remove_tokens_num,
        'tokens_drop_percentage': 0.0,  # No token info from filters
        'top_reasons': list(reason_counts.head().items())
    }

    fs = fsspec.filesystem('s3')  # This will use s3fs under the hood but through fsspec

    # Replace your existing code with:
    output_file = os.path.join(main_output_path, f"{dataset_name}_reason_counts.txt")
    with fs.open(output_file, 'w') as f:
        f.write(f"Source Name: {dataset_name}\nNumber of samples: {actual_examples}\n\n")
        f.write(f"Percent of samples dropped: {np.round(drop_per * 100, 3)}%\n")
        f.write(f"Percent of tokens dropped: 0.0%\n")  # Placeholder
        f.write("\nReason Counts:\n")
        for reason, count in reason_counts.items():
            f.write(f"{reason}: {count} ({count / actual_examples:.4f} of total)\n")
        f.write("\nReason Probabilities:\n")
        for reason, prob in reason_probs.items():
            f.write(f"{reason}: {prob:.4f} (of dropped)\n")
    # Update overall stats (return if needed externally)
    overall_samples_dropped += len(dropped_records)
    overall_total_samples += actual_examples
    overall_tokens_dropped += source_remove_tokens_num
    overall_total_tokens += source_tokens_num
    print(dataset_name)
    print(summary_data)
    return {
        "summary_data": summary_data,
        "overall": {
            "overall_drop_percentage": (overall_samples_dropped / overall_total_samples) * 100 if overall_total_samples else 0,
            "overall_tokens_drop_percentage": (overall_tokens_dropped / overall_total_tokens) * 100 if overall_total_tokens else 0
        },
        "drop_per_list": drop_per_list
    }


def summarize_dataset(df_dropped, tokens_before, tokens_after):
    total = tokens_before + len(df_dropped)
    dropped = len(df_dropped)
    reasons = df_dropped['reason'].value_counts()
    return {
        'total_samples': total,
        'samples_dropped': dropped,
        'drop_percentage': (dropped/total)*100 if total>0 else 0,
        'tokens_before': tokens_before,
        'tokens_after': tokens_after,
        'tokens_dropped': tokens_before - tokens_after,
        'tokens_drop_percentage': ((tokens_before-tokens_after)/tokens_before)*100 if tokens_before>0 else 0,
        'top_reasons': reasons.head(5).to_dict(),
    }


def write_reports(output_base, dataset_name, summary, df_dropped, sample_n=20):
    fs = fsspec.filesystem('s3')
    with fs.open(f"{output_base}/{dataset_name}/summary.json", 'w') as f:
        json.dump(summary, f, indent=2)
    # sample_df = df_dropped.groupby('reason').apply(
    #     lambda x: x.sample(n=min(len(x), sample_n), random_state=42)
    # ).reset_index(drop=True)
    sample_df = df_dropped.groupby('reason').apply(
    lambda x: x[x.columns.drop('reason')].sample(n=min(len(x), sample_n), random_state=42)
    ).reset_index(drop=True)
    with fs.open(f"{output_base}/{dataset_name}/sampled_dropped.xlsx", 'wb') as f:
        sample_df.to_excel(f, index=False)


def make_pipeline(filter_config, input_base, output_base, dataset_name, max_rows=None, debug_one_part=False,use_jsonl_reader=False):
    fs = fsspec.filesystem('s3')
    # Determine input files
    dataset_path = input_base+"/"+dataset_name
    pattern = f"{dataset_path}/*.csv" if not use_jsonl_reader else f"{dataset_path}/*.jsonl"
    print(f'Reading {pattern}')
    all_paths = sorted(fs.glob(pattern))
    if debug_one_part and all_paths:
        part_paths = [all_paths[0]]
    else:
        part_paths = all_paths
    part_paths = ["s3://"+f for f in part_paths]
    if not fs.exists(dataset_path):
        raise ValueError(f"Directory does not exist: {dataset_path}")

    if not use_jsonl_reader:
        reader = CsvReader(
            data_folder=dataset_path,
            glob_pattern=f"*.csv",
            text_key="text",
            id_key="id",
            recursive=False,
            limit=max_rows,
        )
    else:
        reader = JsonlReader(
            data_folder=dataset_path,
            glob_pattern=f"*.jsonl",
            text_key="text",
            id_key="id",
            recursive=False,
            limit=max_rows,
        )
    
    if not fs.exists(reader.data_folder.path):
        raise ValueError(f"Directory does not exist: {dataset_path}")
    steps = [reader]

    # partial function to get param with dataset overrides
    pr = partial(get_param, config = filter_config, dataset_name = dataset_name)

    steps.append(MinDocFilter(
        min_doc_words=pr('min_doc_words'),
        language=language,
        exclusion_writer=JsonlWriter(f"{output_base}/{dataset_name}/filtering/removed/min_doc"),
    ))
    steps.append(GopherRepetitionFilter(
        dup_line_frac=pr('dup_line_frac'),
        top_n_grams=pr('top_n_grams'),
        dup_n_grams=pr('dup_n_grams'),
        exclusion_writer=JsonlWriter(f"{output_base}/{dataset_name}/filtering/removed/repetition"),
    ))
    steps.append(FineWebQualityFilter(
        language=language,
        short_line_thr=999,
        char_duplicates_ratio=0.1,
        line_punct_exclude_zero=pr('line_punct_exclude_zero'),
        line_punct_thr=pr("line_punct_thr"),
        new_line_ratio=pr("new_line_ratio"),
        exclusion_writer=JsonlWriter(f"{output_base}/{dataset_name}/filtering/removed/fineweb"),
    ))
    steps.append(GopherQualityFilter(
        language=language,
        min_doc_words=pr('min_doc_words'),
        max_doc_words=pr('max_doc_words'),
        max_avg_word_length=pr('max_avg_word_length'),
        min_avg_word_length=pr('min_avg_word_length'),
        stop_words=pr('stopwords'),
        max_non_alpha_words_ratio=pr('max_non_alpha_words_ratio'),
        min_stop_words=pr('min_number_of_stopwords'),
        max_bullet_lines_ratio=pr('max_bullet_lines_ratio'),
        max_ellipsis_lines_ratio=pr('max_ellipsis_lines_ratio'),
        exclusion_writer=JsonlWriter(f"{output_base}/{dataset_name}/filtering/removed/gopher_qual"),
    ))
    # Writer: passed docs
    steps.append(JsonlWriter(f"{output_base}/{dataset_name}/filtering/output"))
    return steps



def fineweb_filtering_pipeline_run():
    # Load filter config
    with open(CONFIG_PATH) as f:
        filter_config = yaml.safe_load(f)

    overall = {'samples': 0, 'dropped': 0, 'tokens_in': 0, 'tokens_out': 0}
    DATASETS = list_s3_directories(INPUT_BASE)

    print(f'DATSETS:{DATASETS}')
    drop_per_list = {}
    for ds in DATASETS:
        if ds !='sefaria':
            continue
        print('_'*100+'\n'+'='*100)
        print(ds)
        if ds=='HeC4DictaCombined':
            continue
        pipeline = make_pipeline(
            filter_config,
            INPUT_BASE,
            OUTPUT_BASE,
            ds,
            max_rows=MAX_ROWS,
            debug_one_part=DEBUG_ONE_PART,
            use_jsonl_reader = True if ds=='wikipedia' else False,
        )

        executor = LocalPipelineExecutor(
            pipeline=pipeline,
            logging_dir=f"{OUTPUT_BASE}/{ds}/logs",
            tasks=TASKS,
        )
        executor.run()

        overall = process_stats_file(f"{OUTPUT_BASE}/{ds}/logs/stats.json",
                                     dataset_name=ds,
                                     drop_per_list=drop_per_list,
                                     main_output_path=f"{OUTPUT_BASE}/{ds}/logs",
                                     filter_config=filter_config)

        filter_analysis_pipeline(dataset_name=ds, bucket_name=BUCKET_NAME, root_path=PREFIX)
    with fsspec.open(f"{OUTPUT_BASE}/overall_summary.json", 'w') as f:
        json.dump(overall, f, indent=2)
    print('FINISH')
    print(overall)

    return BUCKET_NAME, PREFIX

if __name__ == '__main__':
    BUCKET_NAME, PREFIX = fineweb_filtering_pipeline_run()

