# Enhanced Cleaners with Sample Saving

This document explains the enhanced sample saving functionality that has been added to all cleaners in the text cleaning pipeline.

## Overview

Each cleaner now has the ability to save before and after samples of the data it processes, allowing you to:

- **Analyze the impact** of each cleaning step
- **Quality control** the cleaning process
- **Debug issues** by examining specific examples
- **Track changes** in text length, word count, and content

## Features

### Automatic Sample Saving
- **Configurable sampling rate**: Default 5%, adjustable from 0% to 100%
- **Random sampling**: Ensures representative samples
- **Organized output**: Each cleaner has its own directory structure
- **Comprehensive metadata**: Includes statistics and processing information

### Sample Types Generated
1. **Before samples**: Original data before cleaning
2. **After samples**: Data after cleaning
3. **Comparison files**: Side-by-side comparison with change indicators
4. **Metadata files**: Processing statistics and configuration

### Directory Structure
```
text_cleaning/samples/
├── before/
│   ├── RegExCleaner/
│   ├── DuplicateRemoverCleaner/
│   ├── QualityCleaner/
│   └── CompositeCleaner/
├── after/
│   ├── RegExCleaner/
│   ├── DuplicateRemoverCleaner/
│   ├── QualityCleaner/
│   └── CompositeCleaner/
└── comparison/
    ├── RegExCleaner/
    ├── DuplicateRemoverCleaner/
    ├── QualityCleaner/
    └── CompositeCleaner/
```

## Usage

### Basic Usage with Sample Saving

```python
from cleaners.regex_cleaner import RegExCleaner

# Create cleaner with default 5% sampling
cleaner = RegExCleaner(
    patterns=[(r'<[^>]+>', '')],  # Remove HTML tags
    save_samples=True,             # Enable sample saving
    sample_percentage=0.05         # 5% sampling rate
)

# Clean data
cleaned_df = cleaner.clean(df, file_name="my_data")
```

### Custom Sampling Rate

```python
# Use 10% sampling
cleaner = RegExCleaner(
    patterns=[(r'\s+', ' ')],
    save_samples=True,
    sample_percentage=0.1  # 10% sampling
)
```

### Disable Sample Saving

```python
# Disable sample saving for performance
cleaner = RegExCleaner(
    patterns=[(r'\s+', ' ')],
    save_samples=False  # Disable sample saving
)
```

### Change Sampling Rate After Creation

```python
cleaner = RegExCleaner(patterns=[(r'\s+', ' ')])
cleaner.set_sample_saving(enabled=True, sample_percentage=0.15)  # 15% sampling
```

## Sample Output Files

### Before/After Samples
CSV files containing the sampled data with columns:
- `text`: The text content
- `n_words`: Word count
- `original_index`: Original row index for tracking

### Comparison Files
CSV files with side-by-side comparison:
- `original_index`: Original row index
- `text_before`: Original text
- `text_after`: Cleaned text
- `n_words_before`: Original word count
- `n_words_after`: Cleaned word count
- `text_changed`: Boolean indicating if text changed
- `word_count_changed`: Boolean indicating if word count changed
- `chars_before`: Original character count
- `chars_after`: Cleaned character count
- `chars_removed`: Characters removed (negative if added)

### Metadata Files
JSON files containing:
- Cleaner name and configuration
- Processing statistics
- Sample information
- File paths for all generated files

## Example Output

### Sample Summary Log
```
2024-01-15 10:30:15 - INFO - Saved samples for RegExCleaner:
2024-01-15 10:30:15 - INFO -   - Before sample: text_cleaning/samples/before/RegExCleaner/data_20240115_103015_before.csv
2024-01-15 10:30:15 - INFO -   - After sample: text_cleaning/samples/after/RegExCleaner/data_20240115_103015_after.csv
2024-01-15 10:30:15 - INFO -   - Comparison: text_cleaning/samples/comparison/RegExCleaner/data_20240115_103015_comparison.csv
2024-01-15 10:30:15 - INFO - Sample summary for RegExCleaner:
2024-01-15 10:30:15 - INFO -   - Text changed: 8/10 (80.0%)
2024-01-15 10:30:15 - INFO -   - Word count changed: 3/10 (30.0%)
2024-01-15 10:30:15 - INFO -   - Total chars removed: 45
```

### Comparison CSV Example
```csv
original_index,text_before,text_after,n_words_before,n_words_after,text_changed,word_count_changed,chars_before,chars_after,chars_removed
0,"This is a sample text with some HTML tags <b>bold</b> and <i>italic</i>.","This is a sample text with some HTML tags bold and italic.",12,12,True,False,67,59,8
1,"Another sample with duplicate lines.\nThis line appears twice.\nThis line appears twice.","Another sample with duplicate lines.\nThis line appears twice.",8,6,True,True,89,67,22
```

## Composite Cleaner

When using the `CompositeCleaner`, each individual cleaner saves its own samples:

```python
from cleaners.composite_cleaner import CompositeCleaner

# Create individual cleaners with sample saving
regex_cleaner = RegExCleaner(patterns=[(r'<[^>]+>', '')], save_samples=True)
duplicate_cleaner = DuplicateRemoverCleaner(save_samples=True)
quality_cleaner = QualityCleaner(save_samples=True)

# Create composite cleaner
composite = CompositeCleaner(
    cleaners=[regex_cleaner, duplicate_cleaner, quality_cleaner],
    save_samples=True  # Also saves overall before/after
)

# Each step will save samples
cleaned_df = composite.clean(df, file_name="pipeline_example")
```

This creates samples for:
- Overall pipeline (before → after all steps)
- Each individual step (before → after that specific cleaner)

## Performance Considerations

### Memory Usage
- Sample saving adds minimal memory overhead
- Only sampled data is kept in memory during processing
- Temporary files are cleaned up automatically

### Storage Usage
- Sample files are typically small (5% of data)
- Can be disabled for large datasets if needed
- Files are organized by cleaner and timestamp

### Processing Time
- Sample saving adds minimal processing time
- Random sampling is efficient
- File I/O is optimized

## Configuration Options

### Sample Percentage
- **Range**: 0.0 to 1.0 (0% to 100%)
- **Default**: 0.05 (5%)
- **Recommendation**: 5-10% for most use cases

### Output Directory
- **Default**: `text_cleaning/samples/`
- **Customizable**: Pass `sample_output_dir` to base cleaner

### File Naming
- **Format**: `{file_name}_{timestamp}_{type}.csv`
- **Example**: `my_data_20240115_103015_before.csv`

## Troubleshooting

### Common Issues

1. **"No samples saved"**
   - Check if `save_samples=True`
   - Verify output directory permissions
   - Check log files for errors

2. **"Sample percentage too high"**
   - Reduce `sample_percentage` for large datasets
   - Consider disabling for very large files

3. **"Missing comparison files"**
   - Ensure before and after DataFrames have same length
   - Check for processing errors in logs

### Debug Mode
Enable debug logging to see detailed sample saving information:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Best Practices

1. **Start with 5% sampling** for initial testing
2. **Increase to 10-15%** for detailed analysis
3. **Disable for production** if performance is critical
4. **Review samples regularly** to ensure cleaning quality
5. **Use meaningful file names** for easier organization

## Example Workflow

1. **Test with samples enabled**:
   ```python
   cleaner = RegExCleaner(save_samples=True, sample_percentage=0.1)
   cleaned_df = cleaner.clean(df, file_name="test_run")
   ```

2. **Review sample files**:
   - Check comparison CSV for unexpected changes
   - Verify cleaning statistics in metadata
   - Ensure quality metrics are reasonable

3. **Adjust cleaning parameters** based on sample analysis

4. **Run full pipeline** with samples disabled for performance:
   ```python
   cleaner = RegExCleaner(save_samples=False)
   cleaned_df = cleaner.clean(df, file_name="production_run")
   ```

## Integration with Existing Code

The enhanced sample saving is backward compatible. Existing code will work without changes, but you can enable sample saving by adding parameters:

```python
# Old code (still works)
cleaner = RegExCleaner(patterns=[(r'<[^>]+>', '')])

# Enhanced code with sample saving
cleaner = RegExCleaner(
    patterns=[(r'<[^>]+>', '')],
    save_samples=True,
    sample_percentage=0.05
)
``` 