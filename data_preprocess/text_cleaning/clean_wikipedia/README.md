# Wikipedia Hebrew Text Cleaning Pipeline

A comprehensive system for processing and cleaning Hebrew Wikipedia dump files, extracting clean text suitable for training language models.

## Overview

This system takes raw Wikipedia XML dump files as input and applies sophisticated text cleaning to produce high-quality training data. It supports multiple processing modes from single article extraction to full dump processing with S3 integration.

## Features

- 🧹 **Comprehensive Text Cleaning**: Removes Wikipedia markup, HTML codes, PII, and formatting artifacts
- 📊 **Batch Processing**: Process entire Wikipedia dumps efficiently
- 🔍 **Single Article Finder**: Extract and clean specific articles by title
- 🏗️ **Simple Interface**: Easy-to-use pipeline for basic text cleaning
- ☁️ **S3 Integration**: Automatic upload to AWS S3 for large-scale processing
- 📈 **Statistics Tracking**: Detailed cleaning statistics and progress monitoring
- 💾 **Example Collection**: Save cleaning examples for analysis and validation

## Components

### Core Components

| File | Purpose |
|------|---------|
| `wiki_text_cleaner.py` | Core cleaning engine with all text processing rules |
| `wiki_batch_processor.py` | Full Wikipedia dump processing with S3 upload |
| `wiki_article_finder.py` | Find and clean specific articles from dump |
| `wiki_pipeline_interface.py` | Simple interface for basic text cleaning |

### Cleaning Rules

The system applies the following cleaning operations:

1. **HTML Escape Codes** - Convert `&quot;`, `&#34;`, `&#39;` etc.
2. **Newlines and Spaces** - Normalize line breaks and whitespace
3. **Multiple Spaces** - Collapse consecutive spaces
4. **Whitespace Trimming** - Remove leading/trailing whitespace
5. **PII Detection** - Remove IP addresses and sensitive information
6. **Empty Bullet Lines** - Clean empty bullet point lines
7. **Separator Lines** - Remove horizontal separator lines
8. **CSS/Table Cleanup** - Extract content from Wikipedia tables
9. **Wiki Templates** - Remove Wikipedia templates and markup
10. **Media Descriptions** - Clean image and media descriptions
11. **Headers** - Process and clean section headers
12. **Citations** - Clean and remove citation markup

## Prerequisites

**Required modules:**
```bash
pip install mwparserfromhell boto3 tqdm
```

**Wikipedia Dump:**
Download the latest Hebrew Wikipedia dump from: https://dumps.wikimedia.org/hewiki/latest/hewiki-latest-pages-articles.xml.bz2

## Usage

### 1. Process Entire Wikipedia Dump
```bash
python wiki_batch_processor.py
```

### 2. Find Specific Article
```bash
python wiki_article_finder.py
```

### 3. Simple Text Cleaning Interface
```bash
python wiki_pipeline_interface.py
```

### 4. Use as Library in Your Code

```python
from wiki_pipeline_interface import SimpleWikipediaCleaner

cleaner = SimpleWikipediaCleaner()
clean_text = cleaner.clean_text(raw_wikipedia_text)
```

## Output Format

**Main Output:** JSONL format with `text`, `word_count`, `byte_count`, and `title` fields.

## Directory Structure

```
clean_wikipedia/
├── wiki_text_cleaner.py          # Core cleaning engine
├── wiki_batch_processor.py       # Full dump processing
├── wiki_article_finder.py        # Single article extraction
├── wiki_pipeline_interface.py    # Simple interface
├── README.md                      # This file
├── examples/                      # Generated cleaning examples
└── temp_output/                   # Temporary processing files
```


