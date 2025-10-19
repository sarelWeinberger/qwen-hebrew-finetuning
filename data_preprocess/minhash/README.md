# MinHash Deduplication System

A scalable, distributed deduplication system using MinHash algorithms to identify and remove duplicate content from large-scale Hebrew text datasets. This system is designed to handle massive datasets efficiently using Ray for distributed processing.

## Overview

The MinHash deduplication system uses locality-sensitive hashing to identify near-duplicate documents in large datasets. It's particularly effective for Hebrew text where exact string matching might miss duplicates due to minor variations in formatting, spacing, or encoding.

## Architecture

The system operates in four distinct stages:

```
Stage 1: Signature Generation
├── Input: Raw text documents
├── Process: Generate MinHash signatures
└── Output: Document signatures per bucket

Stage 2: Bucket Processing  
├── Input: MinHash signatures
├── Process: Find candidate duplicates within buckets
└── Output: Potential duplicate pairs

Stage 3: Clustering
├── Input: Duplicate pairs from all buckets
├── Process: Create clusters of duplicates
└── Output: Duplicate clusters with representative IDs

Stage 4: Filtering
├── Input: Original documents + duplicate clusters
├── Process: Remove duplicates, keep one representative per cluster
└── Output: Deduplicated dataset
```

## Key Features

- **Scalable Processing**: Handles datasets with millions of documents
- **Distributed Computing**: Uses Ray for parallel processing across multiple nodes
- **Hebrew Language Support**: Optimized for Hebrew text characteristics
- **Configurable Precision**: Adjustable similarity thresholds
- **Memory Efficient**: Streaming processing for large files
- **S3 Integration**: Native support for AWS S3 storage
- **Progress Tracking**: Comprehensive logging and monitoring

## Configuration

### MinHash Configuration

```python
minhash_config = MinhashConfig(
    hash_config=HashConfig(precision=64),  # Higher precision = fewer false positives
    num_buckets=14,                        # Number of hash buckets
    hashes_per_bucket=8,                  # Hashes per bucket
)
```

### Ray Cluster Configuration

The system uses a Ray cluster for distributed processing:

```yaml
# cluster.yaml
cluster_name: minihash-ray-cluster
provider:
    type: aws
    region: us-east-1
max_workers: 100

head_node_type: head_node
available_node_types:
    head_node:
        node_config:
            InstanceType: m5.large
    worker_node:
        node_config:
            InstanceType: c5.large
        min_workers: 2
        max_workers: 90
```

## Usage

### Basic Setup

1. **Install Dependencies**:
```bash
pip install ray[default]
pip install datatrove[ray,s3,processing] boto3 s3fs orjson spacy
```

2. **Configure AWS Credentials**:
```bash
aws configure set aws_access_key_id YOUR_ACCESS_KEY
aws configure set aws_secret_access_key YOUR_SECRET_KEY
aws configure set default.region us-east-1
```

3. **Initialize Ray Cluster**:
```bash
ray up cluster.yaml
```

### Running Deduplication

```python
from minhash import main

# Run the complete deduplication pipeline
main()
```

### Custom Configuration

```python
# Custom MinHash configuration
minhash_config = MinhashConfig(
    hash_config=HashConfig(precision=32),  # Lower precision for faster processing
    num_buckets=10,                       # Fewer buckets
    hashes_per_bucket=6,                 # Fewer hashes per bucket
)

# Custom input/output paths
S3_MINHASH_BASE_PATH = "s3://your-bucket/dedupe/run1"
INPUT_READER = JsonlReader(
    data_folder="s3://your-bucket/input-data",
    recursive=True,
    compression="gzip",
    glob_pattern="*.jsonl.gz",
)
```

## Processing Stages

### Stage 1: Signature Generation

Generates MinHash signatures for each document:

```python
stage1 = RayPipelineExecutor(
    pipeline=[
        INPUT_READER,
        MinhashDedupSignature(
            output_folder=f"{S3_MINHASH_BASE_PATH}/signatures",
            config=minhash_config,
            language=Languages.hebrew__hebr
        ),
    ],
    tasks=TOTAL_TASKS,
    workers=-1,
    mem_per_cpu_gb=2,
    cpus_per_task=1,
)
```

**Parameters**:
- `tasks`: Number of parallel tasks (typically 1000 for large datasets)
- `workers`: Number of Ray workers (-1 for auto-detection)
- `mem_per_cpu_gb`: Memory allocation per CPU
- `language`: Hebrew language configuration

### Stage 2: Bucket Processing

Finds potential duplicates within each hash bucket:

```python
stage2 = RayPipelineExecutor(
    pipeline=[
        MinhashDedupBuckets(
            input_folder=f"{S3_MINHASH_BASE_PATH}/signatures",
            output_folder=f"{S3_MINHASH_BASE_PATH}/buckets",
            config=minhash_config,
        ),
    ],
    tasks=minhash_config.num_buckets,
    depends=stage1,
)
```

**Parameters**:
- `tasks`: Number of buckets (matches `num_buckets` in config)
- `depends`: Waits for Stage 1 completion

### Stage 3: Clustering

Creates clusters of duplicate documents:

```python
stage3 = RayPipelineExecutor(
    pipeline=[
        MinhashDedupCluster(
            input_folder=f"{S3_MINHASH_BASE_PATH}/buckets",
            output_folder=f"{S3_MINHASH_BASE_PATH}/remove_ids",
            config=minhash_config,
        ),
    ],
    tasks=1,
    mem_per_cpu_gb=30,
    cpus_per_task=4,
    workers=2,
    depends=stage2,
)
```

**Parameters**:
- `tasks`: Single task (clustering is sequential)
- `mem_per_cpu_gb`: High memory requirement for clustering
- `workers`: Limited workers due to memory requirements

### Stage 4: Filtering

Removes duplicates and outputs clean dataset:

```python
stage4 = RayPipelineExecutor(
    pipeline=[
        INPUT_READER,
        TokensCounter(),  # Track token counts
        MinhashDedupFilter(
            input_folder=f"{S3_MINHASH_BASE_PATH}/remove_ids",
            exclusion_writer=JsonlWriter(f"{S3_MINHASH_BASE_PATH}/removed"),
        ),
        JsonlWriter(output_folder=f"{S3_MINHASH_BASE_PATH}/deduplicated_output"),
    ],
    tasks=TOTAL_TASKS,
    workers=15,
    cpus_per_task=3,
    mem_per_cpu_gb=8,
    depends=stage3,
)
```

**Parameters**:
- `tasks`: Must match Stage 1 task count
- `exclusion_writer`: Saves removed duplicates for analysis
- `workers`: Higher worker count for I/O intensive operations

## Performance Tuning

### Memory Optimization

```python
# For large datasets, increase memory allocation
stage1 = RayPipelineExecutor(
    mem_per_cpu_gb=4,  # Increase from default 2GB
    cpus_per_task=2,  # More CPUs per task
)

# For clustering stage
stage3 = RayPipelineExecutor(
    mem_per_cpu_gb=60,  # High memory for clustering
    cpus_per_task=8,   # More CPUs for processing
)
```

### Parallel Processing

```python
# Increase parallelism for faster processing
stage1 = RayPipelineExecutor(
    tasks=2000,  # More tasks for better parallelization
    workers=50,  # More workers
)

stage4 = RayPipelineExecutor(
    workers=30,  # More workers for I/O operations
    cpus_per_task=4,  # More CPUs per task
)
```

### Precision vs Performance

```python
# High precision (fewer false positives, slower)
high_precision_config = MinhashConfig(
    hash_config=HashConfig(precision=128),
    num_buckets=20,
    hashes_per_bucket=10,
)

# Balanced (good precision, reasonable speed)
balanced_config = MinhashConfig(
    hash_config=HashConfig(precision=64),
    num_buckets=14,
    hashes_per_bucket=8,
)

# Fast processing (more false positives, faster)
fast_config = MinhashConfig(
    hash_config=HashConfig(precision=32),
    num_buckets=10,
    hashes_per_bucket=6,
)
```

## Monitoring and Logging

### Log Locations

```python
S3_LOGS_FOLDER = f"{S3_MINHASH_BASE_PATH}/logs"
LOCAL_LOGS_FOLDER = "logs/minhash"
```

### Progress Tracking

The system provides detailed logging for each stage:
- **Stage 1**: Signature generation progress
- **Stage 2**: Bucket processing statistics
- **Stage 3**: Clustering results and cluster sizes
- **Stage 4**: Filtering statistics and token counts

### Token Counting

```python
# TokensCounter provides before/after statistics
TokensCounter()  # Automatically tracks token counts
```

## Output Analysis

### Deduplication Results

The system outputs:
- **Deduplicated Dataset**: Clean dataset with duplicates removed
- **Removed Documents**: Log of all removed duplicates
- **Statistics**: Token counts, duplicate ratios, processing times

### Quality Metrics

```python
# Calculate deduplication effectiveness
original_tokens = tokens_before_dedup
final_tokens = tokens_after_dedup
duplicate_ratio = (original_tokens - final_tokens) / original_tokens
```

## Troubleshooting

### Common Issues

1. **Memory Errors**: Increase `mem_per_cpu_gb` or reduce `tasks`
2. **S3 Connection Issues**: Check AWS credentials and permissions
3. **Ray Cluster Issues**: Verify cluster configuration and worker availability
4. **Task Failures**: Check logs in S3 logs folder

### Debug Mode

```python
# Enable detailed logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Run with smaller dataset for testing
TOTAL_TASKS = 100  # Reduce for testing
```

### Performance Issues

1. **Slow Processing**: Increase workers or reduce precision
2. **Memory Issues**: Optimize memory allocation per task
3. **Network Issues**: Check S3 connectivity and bandwidth

## Advanced Usage

### Custom Similarity Thresholds

```python
# Adjust similarity detection
minhash_config = MinhashConfig(
    hash_config=HashConfig(precision=64),
    num_buckets=14,
    hashes_per_bucket=8,
    similarity_threshold=0.8,  # Custom threshold
)
```

### Multi-Language Support

```python
# Support multiple languages
language_config = Languages.hebrew__hebr | Languages.english__eng
```

### Custom Input Formats

```python
# Support different input formats
INPUT_READER = JsonlReader(
    data_folder="s3://your-bucket/data",
    recursive=True,
    compression="gzip",
    glob_pattern="*.jsonl.gz",
    # Custom parsing logic
    text_key="content",  # Field containing text
    id_key="id",         # Document ID field
)
```

## Best Practices

### Dataset Preparation

1. **Pre-filter Data**: Remove obviously low-quality content before deduplication
2. **Normalize Text**: Ensure consistent encoding and formatting
3. **Batch Processing**: Process data in manageable chunks
4. **Backup Original**: Always keep original data as backup

### Resource Management

1. **Monitor Memory Usage**: Watch for memory spikes during clustering
2. **Optimize Workers**: Balance between parallelism and resource usage
3. **S3 Costs**: Consider S3 request costs for large datasets
4. **Network Bandwidth**: Ensure sufficient bandwidth for S3 operations

### Quality Assurance

1. **Sample Verification**: Manually verify sample duplicates
2. **False Positive Check**: Review removed documents for false positives
3. **Token Count Validation**: Verify token count changes are reasonable
4. **Cross-Validation**: Compare results with different similarity thresholds

## Integration with Text Cleaning

The MinHash deduplication system integrates seamlessly with the text cleaning pipeline:

```python
# Typical workflow
1. Text Cleaning (data_preprocess/text_cleaning)
2. MinHash Deduplication (data_preprocess/minhash)
3. Final Dataset Preparation
```

## License

This MinHash deduplication system is part of the Qwen Hebrew Fine-tuning project and follows the same license terms.
