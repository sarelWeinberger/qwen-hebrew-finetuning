# RAR and JSONL File Support for S3 Fetcher

This document explains how to use the expanded S3 fetcher to process JSONL format data, whether it's already extracted as JSONL files or still compressed in RAR archives.

## Overview

The S3 fetcher has been expanded to support RAR files that contain JSONL (JSON Lines) format data. This allows you to process compressed datasets stored in S3 without having to download and extract them locally.

## Features

- **Dual Format Support**: Handles both extracted JSONL files and RAR archives containing JSONL
- **RAR File Support**: Extracts and parses JSONL files from within RAR archives
- **Direct JSONL Support**: Processes extracted `.jsonl` files directly
- **Automatic Text Column Detection**: Intelligently finds text content in JSONL data
- **Word Count Calculation**: Automatically calculates word counts for extracted text
- **Error Handling**: Robust error handling for malformed JSON or corrupted archives
- **Sample Mode Support**: Works with the existing sample mode for testing

## Prerequisites

### System Dependencies

You need to install the `unrar` system tool:

**macOS:**
```bash
brew install unrar
```

**Ubuntu/Debian:**
```bash
sudo apt-get install unrar
```

**CentOS/RHEL:**
```bash
sudo yum install unrar
```

### Python Dependencies

The required Python packages are already included in `requirements.txt`:
- `rarfile>=4.0` - For RAR file handling
- `boto3>=1.38.28` - For S3 access

## Usage

### Basic Setup

```python
from fetchers.s3_source_fetcher import S3SourceFetcher
from cleaners.composite_cleaner import CompositeCleaner
from cleaning_pipeline import CleaningPipeline

# Create S3 fetcher for RAR files
fetcher = S3SourceFetcher(
    bucket_name="your-data-bucket",
    prefix="raw-data/",
    source_name="hebrew_corpus",  # Files should start with this prefix
    output_prefix="cleaned-data/",
    output_bucket_name="your-output-bucket"
)

# Create cleaner
cleaner = CompositeCleaner()

# Create pipeline
pipeline = CleaningPipeline(
    fetcher=fetcher,
    cleaner=cleaner,
    source_name="hebrew_corpus"
)
```

### File Naming Convention

Your files in S3 should follow this naming pattern:
```
{source_name}_*.{extension}
```

For example, if `source_name="hebrew_corpus"`, the fetcher will look for files like:
- **RAR files**: `hebrew_corpus_part1.rar`, `hebrew_corpus_dataset.rar`
- **JSONL files**: `hebrew_corpus_extracted.jsonl`, `hebrew_corpus_data.jsonl`

The fetcher processes JSONL files directly and extracts JSONL from RAR files.

### RAR File Structure

Each RAR file should contain one or more JSONL files. The fetcher will automatically find and process all `.jsonl` files within the archive.

Example RAR file structure:
```
my_data.rar
├── data_part1.jsonl
├── data_part2.jsonl
└── metadata.jsonl
```

### JSONL Format

Each line in the JSONL files should be a valid JSON object. The fetcher will automatically:

1. **Find text content**: Look for columns named `text`, `content`, or similar
2. **Calculate word counts**: Add `n_words` column if not present
3. **Handle missing data**: Skip invalid JSON lines with warnings
4. **Detect format**: Automatically detect JSONL format even without `.jsonl` extension

Example JSONL content:
```json
{"text": "This is the first sentence.", "id": 1, "category": "news"}
{"text": "This is the second sentence with more words.", "id": 2, "category": "blog"}
{"text": "Third sentence for processing.", "id": 3, "category": "article"}
```

### Supported Formats

The fetcher handles two specific scenarios:

1. **RAR files containing JSONL**: `data.rar` → extracts JSONL files inside
2. **Direct JSONL files**: `data.jsonl` → processes directly

Both formats are processed using the same JSONL parsing logic.

## Running the Pipeline

### Sample Mode (Recommended for Testing)

```python
# Test with a small subset of data
pipeline.run_sample_mode(
    custom_output_prefix="test-output/",
    custom_bucket_name="your-test-bucket"
)
```

This will:
- Select up to 10 RAR files randomly
- Sample ~100 total texts across all files
- Clean the sampled data
- Save as `{source_name}.csv` to S3

### Full Pipeline

```python
# Process all RAR files
pipeline.run()
```

This will:
- Process all JSONL and RAR files matching the source name pattern
- Extract JSONL data from RAR files or process JSONL files directly
- Clean all the data
- Save cleaned data as `{original_filename}_cleaned.csv` to S3

## Error Handling

The fetcher includes comprehensive error handling:

- **Invalid JSON**: Lines with invalid JSON are skipped with warnings
- **Missing JSONL files**: Archives without JSONL files are skipped
- **Corrupted archives**: Damaged RAR files are handled gracefully
- **Missing text columns**: Automatically tries to find suitable text content
- **Unsupported formats**: Files with other extensions are skipped with warnings

## Testing

Run the test script to verify RAR and JSONL functionality:

```bash
cd text_cleaning
python test_rar_functionality.py
```

This will:
- Check system dependencies
- Test RAR extraction and JSONL parsing
- Test JSONL format detection
- Verify the output format

## Example Script

See `example_rar_usage.py` for a complete working example that demonstrates:
- Setting up the pipeline
- Checking available files
- Running in sample mode
- Running the full pipeline

## Troubleshooting

### Common Issues

1. **"unrar tool not found"**
   - Install the `unrar` system package (see Prerequisites)

2. **"No JSONL files found in RAR archive"**
   - Ensure your RAR files contain `.jsonl` files
   - Check the file extensions are lowercase

3. **"Invalid JSON at line X"**
   - Check your JSONL format - each line should be valid JSON
   - Ensure proper UTF-8 encoding

4. **"Missing 'text' column"**
   - The fetcher will try to find text content automatically
   - If it can't find a suitable column, it will use the first column

5. **"Unsupported file format"**
   - Only `.jsonl` and `.rar` files are supported
   - Other file formats will be skipped with a warning

### Debug Mode

Enable debug logging to see detailed processing information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Performance Considerations

- **Memory Usage**: Large RAR files are processed in memory - ensure sufficient RAM
- **Processing Time**: RAR extraction adds overhead compared to direct CSV/Parquet files
- **S3 Costs**: Each RAR file is downloaded completely for processing

## Integration with Existing Workflow

The RAR support is fully integrated with the existing cleaning pipeline:

- Works with all existing cleaners
- Supports sample mode for testing
- Compatible with word count analysis
- Follows the same output format as other file types

## Migration from Other Formats

If you're migrating from CSV/Parquet to RAR or JSONL files:

1. **Update file naming**: Ensure files follow the `{source_name}_*.{extension}` pattern
2. **Convert data**: Package your JSONL data into RAR archives or extract as JSONL files
3. **Upload to S3**: Upload files to your S3 bucket
4. **Update configuration**: Use the same S3 fetcher configuration
5. **Test**: Run sample mode to verify everything works

The cleaning pipeline will automatically detect and handle the format! 