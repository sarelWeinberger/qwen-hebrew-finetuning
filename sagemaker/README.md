# SageMaker Implementation for Qwen Hebrew Fine-tuning

This directory contains a complete SageMaker implementation for running Qwen Hebrew fine-tuning with automated P-type instance performance comparison across **4 instance types**: P4de, P5, P5e, and P5en, plus EC2 P5e availability checking.

## 🚀 Quick Start

### 1. Prerequisites

- AWS CLI configured with appropriate credentials
- Docker installed and running
- SageMaker execution role with necessary permissions
- S3 bucket for data and model storage

### 2. Find Available Regions

Before starting, check which regions support your desired P-type instances (includes both SageMaker and EC2 P5e availability):

```bash
# Check region availability for all P-type instances (includes EC2 P5e checking)
python scripts/find_available_regions.py

# Check specific instances only
python scripts/find_available_regions.py \
    --instances ml.p4de.24xlarge ml.p5e.48xlarge ml.p5en.48xlarge

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
./build_and_push.sh us-east-1 # choose from Recommended Regions
```

### 4. Prepare Your Data

```bash
# Submit data preparation job (runs on CPU instance for cost efficiency) ex: [make sure you have s3 premissens on your IAM]
python scripts/data_preperation_sagemaker.py \
    --s3-bucket gepeta-datasets \
    --s3-prefix processed/wikipedia/wikipedia_he_part_001.jsonl \
    --output-data s3://gepeta-datasets/fortrain/wiki/ \
    --min-length 10 \
    --max-length 5000
```

### 5. Run Performance Benchmark

```bash
# Run automated benchmark across ALL 4 P-type instances (P4de, P5, P5e, P5en)
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
    --instance-types ml.p4de.24xlarge ml.p5e.48xlarge ml.p5en.48xlarge \
    --epochs 1
```

### 6. Use Jupyter Notebook (Recommended)

For a complete interactive experience, use the provided Jupyter notebook:

```bash
jupyter notebook notebooks/setup_and_run_benchmark.ipynb
```

## 🚀 How to Run SageMaker Training

### Step-by-Step Guide

#### Step 1: Set Up AWS Environment

1. **Configure AWS CLI**:
   ```bash
   aws configure
   # Enter your AWS Access Key ID, Secret Access Key, and region
   ```

2. **Create SageMaker Execution Role**:
   ```bash
   # Create a role with SageMaker, S3, and ECR permissions
   aws iam create-role --role-name SageMakerExecutionRole --assume-role-policy-document '{
     "Version": "2012-10-17",
     "Statement": [
       {
         "Effect": "Allow",
         "Principal": {
           "Service": "sagemaker.amazonaws.com"
         },
         "Action": "sts:AssumeRole"
       }
     ]
   }'
   
   # Attach necessary policies
   aws iam attach-role-policy --role-name SageMakerExecutionRole --policy-arn arn:aws:iam::aws:policy/AmazonSageMakerFullAccess
   aws iam attach-role-policy --role-name SageMakerExecutionRole --policy-arn arn:aws:iam::aws:policy/AmazonS3FullAccess
   aws iam attach-role-policy --role-name SageMakerExecutionRole --policy-arn arn:aws:iam::aws:policy/AmazonEC2ContainerRegistryFullAccess
   ```

3. **Create S3 Bucket**:
   ```bash
   aws s3 mb s3://your-qwen-hebrew-bucket --region us-east-1
   ```

#### Step 2: Build and Deploy Containers

1. **Navigate to container directory**:
   ```bash
   cd sagemaker/containers/training/
   ```

2. **Make build script executable**:
   ```bash
   chmod +x build_and_push.sh
   ```

3. **Build and push to ECR**:
   ```bash
   ./build_and_push.sh
   ```

#### Step 3: Prepare Your Data

1. **Option A: Use existing Hebrew dataset**:
   ```bash
   # Upload your Hebrew text files to S3
   aws s3 cp your-hebrew-data.json s3://your-qwen-hebrew-bucket/raw-data/
   ```

2. **Option B: Run data preparation job**:
   ```bash
   python scripts/data_preparation.py \
       --s3-bucket your-qwen-hebrew-bucket \
       --s3-prefix raw-data/ \
       --output-data s3://your-qwen-hebrew-bucket/processed-data/
   ```

#### Step 4: Check Region Availability

```bash
# Find best regions for P5 instances
python scripts/find_available_regions.py

# Output will show:
# ✅ us-east-1 (recommended)
# ✅ us-east-2 (alternative)
# ✅ us-west-2 (west coast)
```

#### Step 5: Run Performance Benchmark

```bash
# Get your role ARN
ROLE_ARN=$(aws iam get-role --role-name SageMakerExecutionRole --query 'Role.Arn' --output text)

# Run benchmark across all P5 instances
python scripts/benchmark_runner.py \
    --role-arn $ROLE_ARN \
    --bucket-name your-qwen-hebrew-bucket \
    --dataset-path s3://your-qwen-hebrew-bucket/processed-data/dataset/ \
    --region us-east-1 \
    --epochs 1 \
    --max-steps 100
```

#### Step 6: Monitor Training Jobs

1. **Check job status**:
   ```bash
   aws sagemaker list-training-jobs --status-equals InProgress
   ```

2. **View logs in CloudWatch**:
   ```bash
   aws logs describe-log-streams --log-group-name /aws/sagemaker/TrainingJobs
   ```

3. **Monitor in SageMaker Console**:
   - Go to AWS Console → SageMaker → Training Jobs
   - Click on your job to see metrics and logs

#### Step 7: Analyze Results

The benchmark will generate:
- **Console output**: Summary of performance and costs
- **CSV report**: Detailed metrics comparison
- **W&B dashboard**: Real-time training metrics
- **S3 artifacts**: Model checkpoints and logs

### Production Training Workflow

Once you've identified the best instance type from benchmarking:

```bash
# Submit production training job
python -c "
from sagemaker.infrastructure.sagemaker_jobs import SageMakerJobManager
import datetime

job_manager = SageMakerJobManager(
    role_arn='$ROLE_ARN',
    bucket_name='your-qwen-hebrew-bucket'
)

job_name = job_manager.submit_training_job(
    job_name=f'qwen-hebrew-production-{datetime.datetime.now().strftime(\"%Y%m%d-%H%M%S\")}',
    instance_type='ml.p5e.48xlarge',  # Based on benchmark results
    dataset_path='s3://your-qwen-hebrew-bucket/processed-data/dataset/',
    epochs=3,
    wandb_project='qwen-hebrew-production'
)

print(f'Production job submitted: {job_name}')
"
```

### Cost Management

1. **Set up billing alerts**:
   ```bash
   aws budgets create-budget --account-id YOUR_ACCOUNT_ID --budget '{
     "BudgetName": "SageMaker-Training-Budget",
     "BudgetLimit": {
       "Amount": "1000",
       "Unit": "USD"
     },
     "TimeUnit": "MONTHLY",
     "BudgetType": "COST"
   }'
   ```

2. **Monitor costs**:
   ```bash
   aws ce get-cost-and-usage \
       --time-period Start=2024-01-01,End=2024-01-31 \
       --granularity MONTHLY \
       --metrics BlendedCost \
       --group-by Type=DIMENSION,Key=SERVICE
   ```

3. **Stop running jobs if needed**:
   ```bash
   aws sagemaker stop-training-job --training-job-name YOUR_JOB_NAME
   ```

### Troubleshooting

#### Common Issues:

1. **Permission Errors**:
   ```bash
   # Check role permissions
   aws iam list-attached-role-policies --role-name SageMakerExecutionRole
   ```

2. **Container Build Failures**:
   ```bash
   # Login to ECR
   aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin YOUR_ACCOUNT.dkr.ecr.us-east-1.amazonaws.com
   ```

3. **Instance Unavailability**:
   ```bash
   # Check instance limits
   aws service-quotas get-service-quota --service-code sagemaker --quota-code L-1194F1E0
   ```

4. **Out of Memory Errors**:
   - Reduce batch size in instance configuration
   - Enable gradient checkpointing
   - Use smaller model or sequence length

### Advanced Usage

#### Multi-Region Training:
```bash
# Run benchmarks in multiple regions
for region in us-east-1 us-west-2 eu-west-1; do
    python scripts/benchmark_runner.py \
        --role-arn $ROLE_ARN \
        --bucket-name your-qwen-hebrew-bucket \
        --dataset-path s3://your-qwen-hebrew-bucket/processed-data/dataset/ \
        --region $region \
        --epochs 1
done
```

#### Hyperparameter Tuning:
```bash
python -c "
from sagemaker.infrastructure.sagemaker_jobs import SageMakerJobManager

job_manager = SageMakerJobManager(
    role_arn='$ROLE_ARN',
    bucket_name='your-qwen-hebrew-bucket'
)

tuning_job = job_manager.submit_hyperparameter_tuning_job(
    tuning_job_name='qwen-hebrew-hp-tuning',
    instance_type='ml.p5.48xlarge',
    dataset_path='s3://your-qwen-hebrew-bucket/processed-data/dataset/',
    max_jobs=20,
    max_parallel_jobs=4
)
"
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
│   │   ├── p4de_config.json      # P4de.24xlarge configuration
│   │   ├── p5_config.json        # P5.48xlarge configuration
│   │   ├── p5e_config.json       # P5e.48xlarge configuration
│   │   └── p5en_config.json      # P5en.48xlarge configuration
│   └── deepspeed/                 # DeepSpeed configurations
│       ├── p4de_deepspeed_config.json  # P4de optimizations
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

#### P4de.24xlarge (A100 80GB) ✨ RE-ADDED
- **GPUs**: 8x A100 80GB
- **Strategy**: Cost-effective high performance
- **Best For**: Budget-conscious projects with high performance needs
- **Estimated Cost**: ~$40.96/hour

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

- **P4de**: ZeRO-2 with A100-optimized settings for cost-effective performance
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
# Test ALL 4 P-type instances (P4de, P5, P5e, P5en) - default
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
    --instance-types ml.p4de.24xlarge ml.p5e.48xlarge ml.p5en.48xlarge \
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
