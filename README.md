# Qwen Hebrew Fine-tuning

This repository contains scripts and tools for fine-tuning the Qwen3-30B-A3B-Base model on Hebrew language data.

## Overview

The project provides a complete workflow for fine-tuning large language models on Hebrew data, with a focus on the Qwen3-30B-A3B-Base model. It includes:

- Scripts for downloading and preparing the model
- Data processing utilities for Hebrew text
- Fine-tuning with DeepSpeed for distributed training
- Hyperparameter optimization with Optuna
- Evaluation on the Hebrew LLM Leaderboard
- Comprehensive logging and monitoring with Weights & Biases
- **Advanced text cleaning pipeline** with modular cleaners and fetchers

## Hardware Requirements

- 8x NVIDIA H100 GPUs (80GB each)
- Sufficient CPU RAM and disk space for model and dataset storage

## Directory Structure

```
qwen_model/
├── data/                  # Data storage and processing
├── finetuning/            # Fine-tuning configurations and outputs
├── logs/                  # Training and evaluation logs
├── model/                 # Model files (downloaded separately)
├── download_model.py      # Script to download the Qwen model
├── prepare_for_finetuning.py # Prepare model for fine-tuning
├── prepare_dataset.py     # Dataset preparation utilities
├── download_s3_data.py    # Download Hebrew data from S3
├── train.py               # Main training script
├── hp_tuning.py           # Hyperparameter tuning with Optuna
├── evaluate_hebrew.py     # Evaluation on Hebrew LLM Leaderboard
├── test_pipeline.py       # Test pipeline with small subset of data
├── run_full_workflow.py   # Run the complete workflow
├── start_training.sh      # Shell script to start training with nohup
├── start_hp_tuning.sh     # Shell script to start hyperparameter tuning
├── start_evaluation.sh    # Shell script to start evaluation
├── start_test_pipeline.sh # Shell script to test the pipeline
└── TUNING_AND_EVALUATION.md # Detailed documentation

text_cleaning/             # Advanced text cleaning system
├── cleaners/              # Text cleaning modules
│   ├── base_cleaner.py    # Base cleaner class
│   ├── regex_cleaner.py   # Regex-based text cleaning
│   ├── spacefix_cleaner.py # Hebrew-specific space fixing
│   ├── duplicate_remove_cleaner.py # Duplicate removal
│   ├── quality_cleaner.py # Quality-based filtering
│   ├── llm_cleaner.py     # LLM-powered cleaning
│   └── composite_cleaner.py # Composite cleaning pipeline
├── fetchers/              # Data source modules
│   ├── base_fetcher.py    # Base fetcher class
│   ├── s3_source_fetcher.py # S3 data source fetcher
│   └── local_source_fetcher.py # Local file fetcher
├── utils/                 # Utilities and configuration
│   ├── cleaner_constants.py # Cleaning rules and patterns
│   ├── regex_registry.py  # Regex pattern registry
│   ├── spacefix_registry.py # Space fixing patterns
│   ├── cleaner_config.py  # Configuration management
│   └── logger.py          # Logging utilities
├── cleaning_pipeline.py   # Main cleaning pipeline
├── main.py                # Entry point for cleaning operations
├── simple_word_count_analyzer.py # Word counting analysis
└── run_benchmark_cleaning.py # Benchmark cleaning performance
```

## Text Cleaning System

The repository includes a sophisticated text cleaning system designed specifically for Hebrew language data processing. This system provides modular, extensible components for cleaning and preparing text data before fine-tuning.

### Architecture Overview

The text cleaning system follows a modular architecture with three main components:

1. **Fetchers**: Responsible for retrieving data from various sources
2. **Cleaners**: Process and clean the retrieved data
3. **Pipeline**: Orchestrates the cleaning workflow

### Fetchers

Fetchers handle data retrieval from different sources and implement the `BaseFetcher` interface:

#### BaseFetcher
- Abstract base class defining the fetcher interface
- Provides statistics tracking and logging capabilities
- Methods: `get_files_to_process()`, `fetch_single_file()`, `save_cleaned_data()`

#### S3SourceFetcher
- Retrieves data from AWS S3 buckets
- Supports multiple file formats: `.jsonl`, `.csv`, `.rar`, `.gz`
- Handles incremental processing (skips already cleaned files)
- Features:
  - Streaming data processing for large files
  - Automatic file format detection
  - Error handling and retry logic
  - Progress tracking and statistics

#### LocalSourceFetcher
- Processes local file system data
- Supports various file formats
- Useful for testing and development

### Cleaners

Cleaners implement text processing logic and inherit from `BaseCleaner`:

#### BaseCleaner
- Abstract base class with common functionality
- Tracks cleaning statistics (rows processed, characters modified, execution time)
- Provides logging and reporting capabilities

#### RegExCleaner
- Applies regex patterns for text transformation
- Supports both string and callable replacements
- Features:
  - Pattern compilation for performance
  - Word-level change tracking
  - Batch processing optimization
  - Statistics collection

#### SpaceFixCleaner
- Specialized for Hebrew text space handling
- Fixes common Hebrew spacing issues
- Handles Hebrew-specific punctuation and diacritics
- Features:
  - Hebrew-aware space normalization
  - Nikud (vowel points) handling
  - Special character processing

#### DuplicateRemoveCleaner
- Identifies and removes duplicate content
- Configurable similarity thresholds
- Memory-efficient processing for large datasets

#### QualityCleaner
- Filters text based on quality metrics
- Removes low-quality or problematic content
- Configurable quality thresholds

#### LLMCleaner
- Uses language models for advanced text cleaning
- Handles complex text transformations
- Requires API access to LLM services

#### CompositeCleaner
- Combines multiple cleaners in sequence
- Configurable cleaning pipeline
- Maintains statistics from all component cleaners

### Cleaning Pipeline

The `CleaningPipeline` class orchestrates the entire cleaning workflow:

#### Features
- **Modular Design**: Easy to swap fetchers and cleaners
- **Incremental Processing**: Skips already processed files
- **Statistics Tracking**: Comprehensive metrics collection
- **Word Count Analysis**: Before/after comparison
- **Error Handling**: Robust error recovery
- **Sample Mode**: Test cleaning on small data subsets

#### Pipeline Workflow
1. **File Discovery**: Fetcher identifies files to process
2. **Data Retrieval**: Files are loaded from source
3. **Cleaning**: Text is processed through cleaner(s)
4. **Storage**: Cleaned data is saved to output location
5. **Analysis**: Word counts and statistics are generated

### Configuration and Constants

#### Cleaner Constants (`cleaner_constants.py`)
Defines comprehensive cleaning rules including:
- HTML tag removal
- PII (Personal Identifiable Information) masking
- Hebrew-specific text normalization
- Markdown table preservation
- Special character handling

#### Regex Registry (`regex_registry.py`)
Pre-configured regex patterns for common cleaning tasks:
- Email address detection and masking
- IP address anonymization
- HTML entity decoding
- Hebrew text normalization

#### SpaceFix Registry (`spacefix_registry.py`)
Hebrew-specific spacing patterns:
- Nikud (vowel points) handling
- Hebrew punctuation spacing
- Special character normalization

### Usage Examples

#### Basic Cleaning Pipeline
```python
from cleaning_pipeline import CleaningPipeline
from fetchers.s3_source_fetcher import S3SourceFetcher
from cleaners.regex_cleaner import RegExCleaner

# Create fetcher and cleaner
fetcher = S3SourceFetcher(
    bucket_name="my-bucket",
    prefix="raw-data/",
    source_name="hebrew_text",
    output_prefix="cleaned-data/",
    output_bucket_name="my-bucket"
)

cleaner = RegExCleaner(patterns=[
    (r'<[^>]+>', ''),  # Remove HTML tags
    (r'\s+', ' ')      # Normalize whitespace
])

# Run pipeline
pipeline = CleaningPipeline(fetcher, cleaner, "hebrew_text")
pipeline.run()
```

#### Composite Cleaning
```python
from cleaners.composite_cleaner import CompositeCleaner
from cleaners.regex_cleaner import RegExCleaner
from cleaners.spacefix_cleaner import SpaceFixCleaner

# Create composite cleaner
cleaner = CompositeCleaner([
    RegExCleaner(patterns=[(r'<[^>]+>', '')]),
    SpaceFixCleaner(),
    DuplicateRemoveCleaner()
])
```

#### Sample Mode Testing
```python
# Test cleaning on small subset
pipeline.run_sample_mode(
    custom_output_prefix="test-samples/",
    custom_bucket_name="test-bucket"
)
```

### Word Count Analysis

The system includes comprehensive word counting capabilities:

#### Features
- **Before/After Comparison**: Tracks word count changes
- **Source Analysis**: Per-source statistics
- **Reduction Metrics**: Percentage of content removed
- **File-level Tracking**: Detailed file-by-file analysis

#### Usage
```python
from simple_word_count_analyzer import count_words_in_source, count_words_after_cleaning

# Count words in raw data
raw_words, raw_files = count_words_in_source(
    bucket_name="source-bucket",
    prefix="raw-data/",
    source_name="hebrew_text"
)

# Count words in cleaned data
cleaned_words, cleaned_files = count_words_after_cleaning(
    output_bucket_name="output-bucket",
    output_prefix="cleaned-data/"
)

# Calculate reduction
reduction_percent = ((raw_words - cleaned_words) / raw_words * 100)
```

### Benchmark and Evaluation

The system includes benchmarking tools for evaluating cleaning performance:

#### Features
- **Performance Metrics**: Processing speed and efficiency
- **Quality Assessment**: Cleaning effectiveness evaluation
- **Resource Usage**: Memory and CPU utilization tracking
- **Comparative Analysis**: Multiple cleaner comparison

#### Usage
```bash
python run_benchmark_cleaning.py --source hebrew_text --cleaner regex
```

## Getting Started

### 1. Clone the Repository

```bash
git clone <repository-url>
cd <repository-directory>
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Installation Process and Environment Setup

#### Environment Requirements
- Python 3.8+ is required
- CUDA 11.8+ is recommended for GPU acceleration
- At least 500GB of disk space for model and data storage

#### Dependencies Installation
The requirements.txt file includes:
- Core ML libraries: PyTorch, Transformers, DeepSpeed
- Data processing: Pandas, NumPy, TQDM
- Monitoring: Tensorboard, Weights & Biases
- Optimization: Optuna for hyperparameter tuning
- **Text cleaning**: regex, boto3, rarfile

#### Weights & Biases Setup
For logging and monitoring, you need to set up Weights & Biases:

```bash
# Install wandb
pip install wandb

# Login with your API key
wandb login YOUR_API_KEY
```

### 4. Data and Model Preparation

The repository includes a comprehensive workflow script that handles downloading the model from Hugging Face and preparing Hebrew data from S3:

```bash
python qwen_model/run_full_workflow.py --aws_access_key_id YOUR_AWS_ACCESS_KEY --aws_secret_access_key YOUR_AWS_SECRET_KEY
```

#### Model Download Process
The workflow downloads the Qwen3-30B-A3B-Base model from Hugging Face using:

```python
# First downloads config files
snapshot_download(
    repo_id="Qwen/Qwen3-30B-A3B-Base",
    local_dir=model_path,
    ignore_patterns=["*.bin", "*.safetensors"]
)

# Then downloads model weights
snapshot_download(
    repo_id="Qwen/Qwen3-30B-A3B-Base",
    local_dir=model_path,
    ignore_patterns=["*.md", "*.txt"],
    resume_download=True
)
```

#### S3 Data Processing
The workflow processes Hebrew data from S3 using the following steps:

1. **S3 Data Download**:
   - Connects to AWS S3 using provided credentials
   - Downloads Hebrew text datasets from the "israllm-datasets" bucket
   - Includes robust error handling with fallback to sample data

2. **Data Processing**:
   - Extracts text from CSV files using pandas
   - Cleans and normalizes Hebrew text
   - Converts to JSONL format suitable for fine-tuning
   - Creates training dataset with appropriate formatting

### 5. Text Cleaning Setup

Before fine-tuning, you can use the text cleaning system to prepare your data:

#### Configure AWS Credentials
```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
```

#### Run Text Cleaning Pipeline
```bash
cd text_cleaning
python main.py --source hebrew_text --cleaner regex
```

#### Test with Sample Data
```bash
python main.py --source hebrew_text --cleaner regex --sample-mode
```

### 6. Test the Pipeline

Before running a full training job, test the pipeline with a small subset of data:

```bash
# For single GPU testing
python qwen_model/test_pipeline.py --dataset_path qwen_model/data/dataset/dataset --max_samples 100

# For multi-GPU testing with DeepSpeed
deepspeed --num_gpus=8 qwen_model/test_pipeline.py --dataset_path qwen_model/data/dataset/dataset --max_samples 100 --single_device --deepspeed qwen_model/deepspeed_config.json
```

### 7. Run Hyperparameter Tuning

```bash
./qwen_model/start_hp_tuning.sh --dataset_path qwen_model/data/dataset/dataset --num_trials 10
```

### 8. Train the Model

```bash
./qwen_model/start_training.sh
```

### 8. Evaluate the Model
# Demo
### 9. Evaluate the Model

```bash
./qwen_model/start_evaluation.sh --model_path qwen_model/finetuned
```
# Install LightEval directly from GitHub using uv
uv pip install 'git+https://github.com/EleutherAI/lighteval.git'
bash
# Run LightEval with the desired configuration
python -m lighteval accelerate \
  "model_name=Qwen/Qwen3-30B-A3B-Base" \
  "leaderboard|arc:challenge|0|0" \
  --output-dir ./subset_test_results_30B_fewshots \
  --save-details \
  --results-path-template "subset_test_30B_fewshots.json" \
  --max-samples 50 \
  --num-fewshot-seeds 25

## Documentation

For detailed information about hyperparameter tuning, training configuration, and evaluation, see [TUNING_AND_EVALUATION.md](qwen_model/TUNING_AND_EVALUATION.md).

## Features

- **Distributed Training**: Optimized for 8x H100 GPUs using DeepSpeed ZeRO-3
- **Hyperparameter Optimization**: Automated tuning with Optuna
- **Robust Logging**: Comprehensive metrics tracking with Weights & Biases
- **Uninterrupted Training**: All scripts use nohup to continue after SSH disconnection
- **Pipeline Testing**: Test the full pipeline with a small subset before full training
- **Hebrew Evaluation**: Integration with the Hebrew LLM Leaderboard
- **Advanced Text Cleaning**: Modular, extensible text processing system
- **Multi-Source Data Support**: S3, local files, and custom data sources
- **Hebrew-Specific Processing**: Specialized handling for Hebrew text characteristics
- **Quality Assurance**: Comprehensive cleaning validation and benchmarking

## License

This project is licensed under the terms of the license included with the Qwen model.
