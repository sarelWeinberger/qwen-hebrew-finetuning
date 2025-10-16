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
qwen-hebrew-finetuning/
├── cross_lang_moe_analysis/    # Cross-language MoE (Mixture of Experts) analysis
│   ├── all_layers_moe_analysis.py
│   ├── comprehensive_moe_analysis.py
│   ├── moe_analysis.py
│   ├── sample_moe_analysis.py
│   ├── deploy.py
│   └── README.md
├── data_preprocess/           # Data preprocessing and cleaning pipeline
│   ├── content_filtering/     # Content filtering utilities
│   │   ├── count_tokens.py
│   │   ├── filter_analysis.py
│   │   ├── fineweb_filtering_pipeline.py
│   │   └── run_filtering.py
│   ├── extract_data/          # Data extraction from various sources
│   │   ├── clean_gcp_ocr.ipynb
│   │   ├── load_wikipedia/    # Wikipedia data processing
│   │   └── pdf_batching.py
│   ├── minhash/               # MinHash deduplication
│   │   ├── cluster.yaml
│   │   └── minhash.py
│   └── text_cleaning/         # Advanced text cleaning system
│       ├── cleaners/          # Text cleaning modules
│       ├── fetchers/          # Data source modules
│       ├── utils/             # Utilities and configuration
│       ├── cleaning_pipeline.py
│       ├── main.py
│       └── run_benchmark_cleaning.py
├── evaluation/                # Model evaluation and benchmarking
│   └── benchmark_results/     # Evaluation results storage
├── training/                  # Training scripts and configurations
│   ├── deepspeed/            # DeepSpeed training setup
│   │   ├── accelerate_config.yaml
│   │   ├── deepspeed_zero3.yaml
│   │   ├── train.py
│   │   ├── train_debug.py
│   │   └── requirements.txt
│   └── nemo/                  # NeMo training setup
│       ├── train.py
│       └── README.md
├── translation/               # Translation and benchmark preparation
│   ├── prompts/              # Benchmark prompts
│   ├── src/                  # Translation utilities
│   └── plots/                # Analysis plots
├── s3_select_processor.py    # S3 data processing utilities
├── sagemaker-lighteval/      # SageMaker LightEval integration
└── requirements.txt          # Project dependencies
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
cd data_preprocess/text_cleaning
python run_benchmark_cleaning.py --source hebrew_text --cleaner regex
```

## Project Components

### Cross-Language MoE Analysis (`cross_lang_moe_analysis/`)

This directory contains analysis tools for understanding how Mixture of Experts (MoE) models behave across different languages, particularly focusing on Hebrew language processing.

#### Key Features
- **Layer-wise Analysis**: Comprehensive analysis of all model layers
- **Sample Analysis**: Detailed examination of specific model samples
- **Deployment Tools**: Utilities for deploying and testing MoE models
- **Logging**: Comprehensive logging for analysis tracking

#### Usage
```bash
cd cross_lang_moe_analysis
python comprehensive_moe_analysis.py
```

### Data Preprocessing (`data_preprocess/`)

A comprehensive data preprocessing pipeline with multiple specialized components:

#### Content Filtering (`content_filtering/`)
- **Token Counting**: Analyze dataset token distributions
- **Filter Analysis**: Evaluate content filtering effectiveness
- **FineWeb Pipeline**: Process FineWeb datasets with Hebrew-specific filtering
- **S3 Integration**: Process data directly from S3 buckets

#### Data Extraction (`extract_data/`)
- **Wikipedia Processing**: Extract and clean Hebrew Wikipedia data
- **OCR Cleaning**: Process and clean OCR-extracted text
- **PDF Batching**: Efficient processing of PDF documents

#### MinHash Deduplication (`minhash/`)
- **Clustering**: Group similar documents using MinHash
- **Deduplication**: Remove duplicate content across datasets
- **Scalable Processing**: Handle large-scale deduplication tasks

### Training (`training/`)

Comprehensive training infrastructure supporting multiple frameworks:

#### DeepSpeed Training (`deepspeed/`)
- **Distributed Training**: Multi-GPU training with DeepSpeed ZeRO-3
- **Hyperparameter Optimization**: Automated tuning with Optuna
- **Debug Tools**: Testing and debugging utilities
- **Configuration Management**: Flexible training configurations

#### NeMo Training (`nemo/`)
- **NVIDIA NeMo Integration**: Alternative training framework
- **Model Conversion**: Tools for model format conversion
- **Training Scripts**: NeMo-specific training implementations

### Translation (`translation/`)

Tools for translating and preparing Hebrew benchmarks:

#### Features
- **Benchmark Translation**: Translate standard benchmarks to Hebrew
- **Prompt Engineering**: Hebrew-specific prompt templates
- **Quality Assessment**: Evaluate translation quality
- **Visualization**: Generate analysis plots and charts

#### Supported Benchmarks
- **ARC**: AI2 Reasoning Challenge
- **GSM8K**: Grade School Math problems
- **MMLU**: Massive Multitask Language Understanding
- **HellaSwag**: Commonsense reasoning
- **COPA**: Choice of Plausible Alternatives

### Evaluation (`evaluation/`)

Model evaluation and benchmarking infrastructure:

#### Features
- **LightEval Integration**: Standard evaluation framework
- **Hebrew Benchmarks**: Specialized Hebrew language benchmarks
- **Result Storage**: Organized storage of evaluation results
- **Performance Tracking**: Monitor model performance over time

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

The repository includes comprehensive data preprocessing tools for preparing Hebrew data:

#### Data Preprocessing
```bash
# Process content filtering
cd data_preprocess/content_filtering
python run_filtering.py

# Clean text data
cd data_preprocess/text_cleaning
python main.py --source hebrew_text --cleaner regex

# Extract Wikipedia data
cd data_preprocess/extract_data/load_wikipedia
python wiki_to_jsonl_to_s3.py
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
cd data_preprocess/text_cleaning
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
python training/deepspeed/test_debug.py --dataset_path data/dataset/dataset --max_samples 100

# For multi-GPU testing with DeepSpeed
deepspeed --num_gpus=8 training/deepspeed/test_debug.py --dataset_path data/dataset/dataset --max_samples 100 --single_device --deepspeed training/deepspeed/deepspeed_zero3.yaml
```

### 7. Run Hyperparameter Tuning

```bash
cd training/deepspeed
python optuna_search.py --dataset_path data/dataset/dataset --num_trials 10
```

### 8. Train the Model

```bash
cd training/deepspeed
python train.py
```

### 9. Evaluate the Model

```bash
# Using LightEval for evaluation
python -m lighteval accelerate \
  "model_name=Qwen/Qwen3-30B-A3B-Base" \
  "leaderboard|arc:challenge|0|0" \
  --output-dir ./evaluation/benchmark_results \
  --save-details \
  --results-path-template "hebrew_evaluation_results.json" \
  --max-samples 50 \
  --num-fewshot-seeds 25
```

## Documentation

For detailed information about specific components, see the README files in each directory:

- **Training**: `training/deepspeed/README.md` and `training/nemo/README.md`
- **Cross-Language MoE Analysis**: `cross_lang_moe_analysis/README.md`
- **Data Preprocessing**: Individual component documentation in `data_preprocess/` subdirectories

## Features

- **Multi-Framework Training**: Support for both DeepSpeed and NeMo training frameworks
- **Distributed Training**: Optimized for 8x H100 GPUs using DeepSpeed ZeRO-3
- **Hyperparameter Optimization**: Automated tuning with Optuna
- **Cross-Language MoE Analysis**: Comprehensive analysis of Mixture of Experts models
- **Advanced Data Preprocessing**: Multi-stage data cleaning and preparation pipeline
- **Content Filtering**: Specialized filtering for Hebrew language content
- **Deduplication**: MinHash-based deduplication for large-scale datasets
- **Benchmark Translation**: Tools for translating and adapting benchmarks to Hebrew
- **Evaluation Infrastructure**: LightEval integration for comprehensive model evaluation
- **Robust Logging**: Comprehensive metrics tracking with Weights & Biases
- **Hebrew-Specific Processing**: Specialized handling for Hebrew text characteristics
- **Quality Assurance**: Comprehensive cleaning validation and benchmarking
- **Multi-Source Data Support**: S3, local files, Wikipedia, and custom data sources

## License

This project is licensed under the terms of the license included with the Qwen model.
