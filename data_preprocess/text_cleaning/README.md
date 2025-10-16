# Text Cleaning System

A sophisticated, modular text cleaning system designed specifically for Hebrew language data processing. This system provides extensible components for cleaning and preparing text data before fine-tuning large language models.

## Overview

The text cleaning system follows a modular architecture with three main components:
1. **Fetchers**: Responsible for retrieving data from various sources
2. **Cleaners**: Process and clean the retrieved data
3. **Pipeline**: Orchestrates the cleaning workflow

## Architecture

```
text_cleaning/
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
├── clean_wikipedia/       # Wikipedia-specific cleaning
│   ├── wiki_article_finder.py
│   ├── wiki_batch_processor.py
│   ├── wiki_pipeline_interface.py
│   └── wiki_text_cleaner.py
├── cleaning_pipeline.py   # Main cleaning pipeline
├── main.py                # Entry point for cleaning operations
├── simple_word_count_analyzer.py # Word counting analysis
├── run_benchmark_cleaning.py # Benchmark cleaning performance
└── benchmark_evaluation.ipynb # Jupyter notebook for evaluation
```

## Core Components

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

## Configuration and Constants

### Cleaner Constants (`cleaner_constants.py`)
Defines comprehensive cleaning rules including:
- HTML tag removal
- PII (Personal Identifiable Information) masking
- Hebrew-specific text normalization
- Markdown table preservation
- Special character handling

### Regex Registry (`regex_registry.py`)
Pre-configured regex patterns for common cleaning tasks:
- Email address detection and masking
- IP address anonymization
- HTML entity decoding
- Hebrew text normalization

### SpaceFix Registry (`spacefix_registry.py`)
Hebrew-specific spacing patterns:
- Nikud (vowel points) handling
- Hebrew punctuation spacing
- Special character normalization

## Usage Examples

### Basic Cleaning Pipeline

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

### Composite Cleaning

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

### Sample Mode Testing

```python
# Test cleaning on small subset
pipeline.run_sample_mode(
    custom_output_prefix="test-samples/",
    custom_bucket_name="test-bucket"
)
```

## Word Count Analysis

The system includes comprehensive word counting capabilities:

### Features
- **Before/After Comparison**: Tracks word count changes
- **Source Analysis**: Per-source statistics
- **Reduction Metrics**: Percentage of content removed
- **File-level Tracking**: Detailed file-by-file analysis

### Usage

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

## Benchmark and Evaluation

The system includes benchmarking tools for evaluating cleaning performance:

### Features
- **Performance Metrics**: Processing speed and efficiency
- **Quality Assessment**: Cleaning effectiveness evaluation
- **Resource Usage**: Memory and CPU utilization tracking
- **Comparative Analysis**: Multiple cleaner comparison

### Usage

```bash
python run_benchmark_cleaning.py --source hebrew_text --cleaner regex
```

## Wikipedia-Specific Cleaning

The `clean_wikipedia/` directory contains specialized tools for processing Wikipedia data:

### Components
- **WikiArticleFinder**: Locates and retrieves Wikipedia articles
- **WikiBatchProcessor**: Processes articles in batches
- **WikiPipelineInterface**: Interface for Wikipedia processing pipeline
- **WikiTextCleaner**: Specialized cleaner for Wikipedia text

### Usage

```python
from clean_wikipedia.wiki_pipeline_interface import WikiPipelineInterface

# Initialize Wikipedia processing
wiki_pipeline = WikiPipelineInterface()
wiki_pipeline.process_articles(batch_size=100)
```

## Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure AWS Credentials

```bash
export AWS_ACCESS_KEY_ID=your_access_key
export AWS_SECRET_ACCESS_KEY=your_secret_key
```

### 3. Run Text Cleaning Pipeline

```bash
# Basic cleaning
python main.py --source hebrew_text --cleaner regex

# Test with sample data
python main.py --source hebrew_text --cleaner regex --sample-mode

# Run benchmark evaluation
python run_benchmark_cleaning.py --source hebrew_text --cleaner regex
```

### 4. Word Count Analysis

```bash
python simple_word_count_analyzer.py
```

## Advanced Usage

### Custom Cleaner Development

```python
from cleaners.base_cleaner import BaseCleaner

class CustomCleaner(BaseCleaner):
    def clean(self, df: pd.DataFrame, file_name: str = "unknown") -> pd.DataFrame:
        # Implement your custom cleaning logic
        # Update self.stats as needed
        return df
```

### Custom Fetcher Development

```python
from fetchers.base_fetcher import BaseFetcher

class CustomFetcher(BaseFetcher):
    def get_files_to_process(self):
        # Implement your file discovery logic
        pass
    
    def fetch_single_file(self, file_path):
        # Implement your file loading logic
        pass
    
    def save_cleaned_data(self, df, source_name, original_file_path):
        # Implement your data saving logic
        pass
```

## Performance Optimization

### Memory Management
- Use streaming processing for large files
- Implement batch processing for memory efficiency
- Monitor memory usage during processing

### Parallel Processing
- Process multiple files in parallel
- Use multiprocessing for CPU-intensive tasks
- Implement async processing for I/O operations

### Caching
- Cache processed files to avoid reprocessing
- Implement incremental processing
- Use file modification time for change detection

## Troubleshooting

### Common Issues

1. **Memory Issues**: Reduce batch size or use streaming processing
2. **S3 Connection Issues**: Check AWS credentials and network connectivity
3. **Pattern Matching Issues**: Verify regex patterns and test with sample data
4. **Encoding Issues**: Ensure proper UTF-8 encoding for Hebrew text

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Run with verbose output
python main.py --source hebrew_text --cleaner regex --debug
```

## Contributing

When adding new cleaners or fetchers:

1. Inherit from the appropriate base class
2. Implement all required methods
3. Update statistics tracking
4. Add comprehensive tests
5. Update documentation

## License

This text cleaning system is part of the Qwen Hebrew Fine-tuning project and follows the same license terms.
