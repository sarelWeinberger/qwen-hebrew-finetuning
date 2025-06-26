# SageMaker Implementation for Qwen Hebrew Fine-tuning

This directory contains a complete SageMaker implementation for running Qwen Hebrew fine-tuning with automated P5 instance performance comparison across **3 instance types**: P5, P5e, and P5en.

## 🚀 Quick Start

### 1. Prerequisites

- AWS CLI configured with appropriate credentials
- Docker installed and running
- SageMaker execution role with necessary permissions
- S3 bucket for data and model storage

### 2. Find Available Regions

Before starting, check which regions support your desired P5 instances:

```bash
# Check region availability for all P5 instances
python scripts/find_available_regions.py

# Check specific instances only
python scripts/find_available_regions.py \
    --instances ml.p5e.48xlarge ml.p5en.48xlarge

# Perform live availability check (slower but more accurate)
python scripts/find_available_regions.py --check-live
```

**Recommended Regions for Full P5 Support:**
- `us-east-1` (N. Virginia) - Best availability and pricing
- `us-east-2` (Ohio) - Good alternative
- `us-west-2` (Oregon) - West coast option

### 3. Build and Push Docker Containers

```bash
# Navigate to training container directory
cd containers/training/

# Build and push training container
chmod +x build_and_push.sh
./build_and_push.sh
```

### 4. Prepare Your Data

```bash
# Submit data preparation job (runs on CPU instance for cost efficiency)
python scripts/data_preparation.py \
    --s3-bucket your-bucket-name \
    --s3-prefix hebrew-text/ \
    --output-data s3://your-bucket-name/processed-data/
```

### 5. Run Performance Benchmark

```bash
# Run automated benchmark across ALL 3 P5 instances
python scripts/benchmark_runner.py \
    --role-arn arn:aws:iam::YOUR_ACCOUNT:role/SageMakerExecutionRole \
    --bucket-name your-bucket-name \
    --dataset-path s3://your-bucket-name/processed-data/dataset/ \
    --epochs 1 \
    --max-steps 100

# Test specific instance types only
python scripts/benchmark_runner.py \
    --role-arn arn:aws:iam::YOUR_ACCOUNT:role/SageMakerExecutionRole \
    --bucket-name your-bucket-name \
    --dataset-path s3://your-bucket-name/processed-data/dataset/ \
    --instance-types ml.p5e.48xlarge ml.p5en.48xlarge \
    --epochs 1
```

### 6. Use Jupyter Notebook (Recommended)

For a complete interactive experience, use the provided Jupyter notebook:

```bash
jupyter notebook notebooks/setup_and_run_benchmark.ipynb
```

## 📁 Directory Structure

```
sagemaker/
├── containers/
│   └── training/
│       ├── Dockerfile              # Training container definition
│       ├── requirements.txt        # Python dependencies
│       └── build_and_push.sh      # Container build script
├── scripts/
│   ├── train.py                   # Main training script
│   ├── data_preparation.py        # Data preprocessing script
│   ├── benchmark_runner.py        # Automated benchmark runner
│   └── find_available_regions.py  # Region availability checker
├── configs/
│   ├── instance_configs/          # Instance-specific configurations
│   │   ├── p5_config.json        # P5.48xlarge configuration
│   │   ├── p5e_config.json       # P5e.48xlarge configuration
│   │   └── p5en_config.json      # P5en.48xlarge configuration
│   └── deepspeed/                 # DeepSpeed configurations
│       ├── p5_deepspeed_config.json
│       ├── p5e_deepspeed_config.json   # P5e optimizations
│       └── p5en_deepspeed_config.json  # P5en optimizations
├── infrastructure/
│   └── sagemaker_jobs.py          # High-level job management
├── notebooks/
│   └── setup_and_run_benchmark.ipynb  # Interactive benchmark notebook
└── README.md                      # This file
```

## 🏗️ Architecture Overview

### Data Preparation Pipeline
- **Instance Type**: m5.xlarge (CPU-only for cost efficiency)
- **Purpose**: Download, clean, and tokenize Hebrew text data
- **Output**: Processed dataset ready for training

### Training Pipeline
- **Instance Types**: P4d.24xlarge, P4de.24xlarge, P5.48xlarge
- **Purpose**: Fine-tune Qwen model with instance-specific optimizations
- **Features**: DeepSpeed integration, W&B logging, performance metrics

### Benchmark System
- **Automated Job Submission**: Submits training jobs across all instance types
- **Performance Monitoring**: Collects comprehensive metrics
- **Cost Analysis**: Calculates cost-effectiveness for each instance type
- **Recommendations**: Provides data-driven instance selection guidance

## 🔧 Configuration Details

### Instance Configurations

#### P5.48xlarge (H100 80GB)
- **GPUs**: 8x H100 80GB
- **Strategy**: Maximum performance optimization
- **Best For**: Time-critical projects, maximum throughput
- **Estimated Cost**: ~$98.32/hour

#### P5e.48xlarge (H100 80GB + NVMe SSD) ✨ NEW
- **GPUs**: 8x H100 80GB
- **Storage**: 8x7.6TB NVMe SSD
- **Strategy**: Storage-optimized with async checkpointing
- **Best For**: Storage-intensive workloads, large model checkpoints
- **Estimated Cost**: ~$98.32/hour

#### P5en.48xlarge (H100 80GB + EFA) ✨ NEW
- **GPUs**: 8x H100 80GB
- **Networking**: Enhanced with EFA (Elastic Fabric Adapter)
- **Strategy**: Network-optimized for distributed training
- **Best For**: Multi-node training, network-intensive workloads
- **Estimated Cost**: ~$98.32/hour

### DeepSpeed Optimizations

Each instance type has optimized DeepSpeed configurations:

- **P5**: ZeRO-2 with minimal offloading for maximum performance
- **P5e**: ZeRO-2 with NVMe-accelerated checkpointing and async writes
- **P5en**: ZeRO-2 with EFA-optimized communication and multi-node support

## 📊 Performance Metrics

The benchmark system collects comprehensive metrics:

### Training Metrics
- Tokens processed per second
- Training time per epoch
- GPU utilization and memory usage
- Loss convergence rates

### Cost Metrics
- Total training cost
- Cost per token processed
- Cost-effectiveness scores
- ROI calculations

### System Metrics
- GPU memory utilization
- CPU utilization
- Network I/O
- Storage performance (for P4de)

## 🎯 Usage Scenarios

### Region Availability Check
```bash
# Check which regions support your desired instances
python scripts/find_available_regions.py

# Check specific instances
python scripts/find_available_regions.py \
    --instances ml.p5e.48xlarge ml.p5en.48xlarge

# Live availability check (more accurate but slower)
python scripts/find_available_regions.py --check-live
```

### Quick Benchmark (Recommended First Step)
```bash
# Test ALL 3 P5 instances (default)
python scripts/benchmark_runner.py \
    --role-arn arn:aws:iam::YOUR_ACCOUNT:role/SageMakerExecutionRole \
    --bucket-name your-bucket-name \
    --dataset-path s3://your-bucket-name/processed-data/dataset/ \
    --epochs 1 \
    --max-steps 100

# Test specific instances only
python scripts/benchmark_runner.py \
    --role-arn arn:aws:iam::YOUR_ACCOUNT:role/SageMakerExecutionRole \
    --bucket-name your-bucket-name \
    --dataset-path s3://your-bucket-name/processed-data/dataset/ \
    --instance-types ml.p5e.48xlarge ml.p5en.48xlarge \
    --epochs 1
```

### Production Training
```python
from infrastructure.sagemaker_jobs import SageMakerJobManager

job_manager = SageMakerJobManager(
    role_arn='arn:aws:iam::YOUR_ACCOUNT:role/SageMakerExecutionRole',
    bucket_name='your-bucket-name'
)

job_name = job_manager.submit_training_job(
    job_name='qwen-hebrew-production',
    instance_type='ml.p5e.48xlarge',  # Based on benchmark results
    dataset_path='s3://your-bucket-name/processed-data/dataset/',
    epochs=3,
    wandb_project='qwen-hebrew-production'
)
```

### Hyperparameter Tuning
```python
tuning_job = job_manager.submit_hyperparameter_tuning_job(
    tuning_job_name='qwen-hebrew-hp-tuning',
    instance_type='ml.p5.48xlarge',  # Use P5 baseline for tuning
    dataset_path='s3://your-bucket-name/processed-data/dataset/',
    max_jobs=20,
    max_parallel_jobs=4
)
```

## 🔍 Monitoring and Debugging

### View Job Status
```python
status = job_manager.get_job_status('your-job-name')
print(f"Status: {status['status']}")
```

### Get Job Logs
```python
logs = job_manager.get_job_logs('your-job-name')
print(logs)
```

### Monitor with W&B
All training jobs automatically log to Weights & Biases for comprehensive monitoring:
- Training metrics and loss curves
- GPU utilization and system metrics
- Cost tracking and performance analysis
- Instance type comparison dashboards

## 💡 Best Practices

### Cost Optimization
1. **Start with benchmarking**: Always run the benchmark first to understand performance characteristics
2. **Use appropriate instance types**:
   - P5 for maximum compute performance and standard workloads
   - P5e for storage-intensive workloads with NVMe acceleration
   - P5en for multi-node distributed training with enhanced networking
3. **Monitor costs**: Set up billing alerts and use spot instances when possible

### Performance Optimization
1. **Batch size tuning**: Each instance type has optimized batch sizes
2. **Gradient accumulation**: Configured per instance for memory efficiency
3. **Checkpointing**: Use S3 or NVMe (P4de) for efficient checkpointing
4. **Data loading**: Optimize S3 data loading with appropriate prefetching

### Reliability
1. **Checkpointing**: Enable regular checkpointing for long training jobs
2. **Monitoring**: Use CloudWatch and W&B for comprehensive monitoring
3. **Error handling**: Implement retry logic for transient failures
4. **Resource limits**: Set appropriate stopping conditions

## 🚨 Troubleshooting

### Common Issues

#### Container Build Failures
```bash
# Ensure Docker is running and you're logged into ECR
aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com
```

#### Permission Errors
Ensure your SageMaker execution role has:
- S3 access to your bucket
- ECR access to pull containers
- CloudWatch logs access
- SageMaker training job permissions

#### Out of Memory Errors
- Reduce batch size in instance configuration
- Enable gradient checkpointing
- Use CPU offloading (already configured for P4d)

#### Slow Training
- Check GPU utilization in CloudWatch
- Verify data loading is not a bottleneck
- Consider using larger instance types

### Getting Help

1. **Check CloudWatch logs** for detailed error messages
2. **Monitor W&B dashboards** for training metrics
3. **Review instance configurations** for optimization opportunities
4. **Use the benchmark results** to guide instance selection

## 📈 Expected Results

Based on typical performance characteristics:

### P4d.24xlarge
- **Performance**: 85-90% of P5 throughput
- **Cost**: 33% of P5 cost
- **Best For**: Budget-conscious projects, development

### P4de.24xlarge
- **Performance**: 90-95% of P5 throughput
- **Cost**: 42% of P5 cost
- **Best For**: Balanced performance/cost, large datasets

### P5.48xlarge
- **Performance**: 100% (baseline)
- **Cost**: 100% (highest)
- **Best For**: Maximum performance, time-critical projects

## 🔄 Next Steps

After running the benchmark:

1. **Analyze results** using the generated reports and visualizations
2. **Choose optimal instance type** based on your requirements and budget
3. **Submit production training jobs** with the recommended configuration
4. **Scale training** based on performance characteristics
5. **Optimize costs** by selecting the most cost-effective option for your use case

## 📝 License

This implementation follows the same license terms as the Qwen model and your project's licensing requirements.