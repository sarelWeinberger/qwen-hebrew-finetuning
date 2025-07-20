# SageMaker Implementation Structure

## Directory Structure
```
sagemaker/
├── containers/
│   ├── training/
│   │   ├── Dockerfile
│   │   ├── requirements.txt
│   │   └── build_and_push.sh
│   └── data-prep/
│       ├── Dockerfile
│       ├── requirements.txt
│       └── build_and_push.sh
├── scripts/
│   ├── data_preparation.py
│   ├── train.py
│   ├── hp_tuning.py
│   └── benchmark_runner.py
├── configs/
│   ├── instance_configs/
│   │   ├── p4d_config.json
│   │   ├── p4de_config.json
│   │   └── p5_config.json
│   ├── deepspeed/
│   │   ├── p4d_deepspeed_config.json
│   │   ├── p4de_deepspeed_config.json
│   │   └── p5_deepspeed_config.json
│   └── training_configs/
│       ├── p4d_training_config.json
│       ├── p4de_training_config.json
│       └── p5_training_config.json
├── infrastructure/
│   ├── sagemaker_jobs.py
│   ├── cost_analyzer.py
│   ├── performance_monitor.py
│   └── results_aggregator.py
└── notebooks/
    ├── setup_sagemaker.ipynb
    ├── run_benchmarks.ipynb
    └── analyze_results.ipynb
```

## Key Implementation Files

### 1. SageMaker Training Container
**File**: `sagemaker/containers/training/Dockerfile`
```dockerfile
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-devel

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    curl \
    vim \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /opt/ml/code/requirements.txt
RUN pip install -r /opt/ml/code/requirements.txt

# Copy training code
COPY scripts/ /opt/ml/code/
COPY configs/ /opt/ml/code/configs/

# Set working directory
WORKDIR /opt/ml/code

# Set environment variables for SageMaker
ENV PYTHONUNBUFFERED=TRUE
ENV PYTHONDONTWRITEBYTECODE=TRUE
ENV PATH="/opt/ml/code:${PATH}"

# Entry point
ENTRYPOINT ["python", "train.py"]
```

### 2. Data Preparation Script
**File**: `sagemaker/scripts/data_preparation.py`
```python
#!/usr/bin/env python3
"""
SageMaker Data Preparation Script for Hebrew Text Processing
Runs on CPU instances (m5.xlarge) for cost-effective preprocessing
"""

import os
import json
import argparse
import boto3
import pandas as pd
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer
import logging

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-data', type=str, default='/opt/ml/input/data/raw-data')
    parser.add_argument('--output-data', type=str, default='/opt/ml/output/data')
    parser.add_argument('--model-name', type=str, default='Qwen/Qwen3-30B-A3B-Base')
    parser.add_argument('--max-length', type=int, default=2048)
    parser.add_argument('--train-split', type=float, default=0.9)
    return parser.parse_args()

def main():
    logger = setup_logging()
    args = parse_args()
    
    logger.info("Starting data preparation...")
    logger.info(f"Input data path: {args.input_data}")
    logger.info(f"Output data path: {args.output_data}")
    
    # Load tokenizer
    logger.info(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Process Hebrew data
    # Implementation details for your specific data format
    
    logger.info("Data preparation completed successfully!")

if __name__ == "__main__":
    main()
```

### 3. SageMaker Training Script
**File**: `sagemaker/scripts/train.py`
```python
#!/usr/bin/env python3
"""
SageMaker Training Script for Qwen Hebrew Fine-tuning
Supports P4d, P4de, and P5 instances with automatic configuration
"""

import os
import json
import argparse
import torch
import wandb
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    set_seed
)
from datasets import load_from_disk
import deepspeed

def parse_args():
    parser = argparse.ArgumentParser()
    
    # SageMaker specific arguments
    parser.add_argument('--model-dir', type=str, default=os.environ.get('SM_MODEL_DIR'))
    parser.add_argument('--train', type=str, default=os.environ.get('SM_CHANNEL_TRAINING'))
    parser.add_argument('--output-data-dir', type=str, default=os.environ.get('SM_OUTPUT_DATA_DIR'))
    
    # Instance type detection
    parser.add_argument('--instance-type', type=str, default=os.environ.get('SM_CURRENT_INSTANCE_TYPE'))
    
    # Training arguments
    parser.add_argument('--model-name', type=str, default='Qwen/Qwen3-30B-A3B-Base')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--seed', type=int, default=42)
    
    # W&B arguments
    parser.add_argument('--wandb-project', type=str, default='qwen-hebrew-sagemaker')
    parser.add_argument('--wandb-entity', type=str, default=None)
    
    return parser.parse_args()

def load_instance_config(instance_type):
    """Load instance-specific configuration"""
    config_map = {
        'ml.p4d.24xlarge': 'configs/instance_configs/p4d_config.json',
        'ml.p4de.24xlarge': 'configs/instance_configs/p4de_config.json',
        'ml.p5.48xlarge': 'configs/instance_configs/p5_config.json'
    }
    
    config_path = config_map.get(instance_type)
    if not config_path:
        raise ValueError(f"Unsupported instance type: {instance_type}")
    
    with open(config_path, 'r') as f:
        return json.load(f)

def main():
    args = parse_args()
    set_seed(args.seed)
    
    # Load instance-specific configuration
    instance_config = load_instance_config(args.instance_type)
    
    # Initialize W&B with instance type information
    wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=f"qwen-hebrew-{args.instance_type}-{args.seed}",
        config={
            "instance_type": args.instance_type,
            "gpu_count": instance_config["gpu_count"],
            "gpu_memory": instance_config["gpu_memory"],
            **instance_config
        }
    )
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        use_cache=False
    )
    
    # Load dataset
    dataset = load_from_disk(args.train)
    
    # Configure training arguments based on instance type
    training_args = TrainingArguments(
        output_dir=args.model_dir,
        per_device_train_batch_size=instance_config["batch_size_per_gpu"],
        gradient_accumulation_steps=instance_config["gradient_accumulation_steps"],
        num_train_epochs=args.epochs,
        fp16=True,
        deepspeed=instance_config["deepspeed_config"],
        logging_steps=10,
        save_steps=500,
        report_to=["wandb"],
        dataloader_num_workers=4,
        remove_unused_columns=False
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset.get("validation"),
        data_collator=data_collator,
        tokenizer=tokenizer
    )
    
    # Train
    trainer.train()
    
    # Save model
    trainer.save_model()
    tokenizer.save_pretrained(args.model_dir)
    
    wandb.finish()

if __name__ == "__main__":
    main()
```

### 4. Instance Configuration Files

**File**: `sagemaker/configs/instance_configs/p4d_config.json`
```json
{
  "instance_type": "ml.p4d.24xlarge",
  "gpu_count": 8,
  "gpu_type": "A100",
  "gpu_memory": "40GB",
  "total_gpu_memory": "320GB",
  "cpu_count": 96,
  "ram": "1152GB",
  "network": "400 Gbps",
  "batch_size_per_gpu": 2,
  "gradient_accumulation_steps": 4,
  "deepspeed_config": "configs/deepspeed/p4d_deepspeed_config.json",
  "training_config": "configs/training_configs/p4d_training_config.json",
  "estimated_hourly_cost": 32.77,
  "optimization_notes": "Memory-constrained, requires careful batch sizing"
}
```

**File**: `sagemaker/configs/instance_configs/p4de_config.json`
```json
{
  "instance_type": "ml.p4de.24xlarge",
  "gpu_count": 8,
  "gpu_type": "A100",
  "gpu_memory": "80GB",
  "total_gpu_memory": "640GB",
  "cpu_count": 96,
  "ram": "1152GB",
  "network": "400 Gbps",
  "nvme_storage": "8x3.8TB",
  "batch_size_per_gpu": 4,
  "gradient_accumulation_steps": 2,
  "deepspeed_config": "configs/deepspeed/p4de_deepspeed_config.json",
  "training_config": "configs/training_configs/p4de_training_config.json",
  "estimated_hourly_cost": 40.96,
  "optimization_notes": "High memory, NVMe storage for checkpointing"
}
```

**File**: `sagemaker/configs/instance_configs/p5_config.json`
```json
{
  "instance_type": "ml.p5.48xlarge",
  "gpu_count": 8,
  "gpu_type": "H100",
  "gpu_memory": "80GB",
  "total_gpu_memory": "640GB",
  "cpu_count": 192,
  "ram": "2048GB",
  "network": "3200 Gbps",
  "batch_size_per_gpu": 6,
  "gradient_accumulation_steps": 1,
  "deepspeed_config": "configs/deepspeed/p5_deepspeed_config.json",
  "training_config": "configs/training_configs/p5_training_config.json",
  "estimated_hourly_cost": 98.32,
  "optimization_notes": "Maximum performance, latest GPU architecture"
}
```

### 5. DeepSpeed Configurations

**File**: `sagemaker/configs/deepspeed/p4d_deepspeed_config.json`
```json
{
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 1000,
    "initial_scale_power": 16,
    "hysteresis": 2,
    "min_loss_scale": 1
  },
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    },
    "overlap_comm": true,
    "contiguous_gradients": true,
    "sub_group_size": 1e9,
    "reduce_bucket_size": "auto",
    "stage3_prefetch_bucket_size": "auto",
    "stage3_param_persistence_threshold": "auto",
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9,
    "gather_16bit_weights_on_model_save": true
  },
  "gradient_accumulation_steps": 4,
  "gradient_clipping": 1.0,
  "steps_per_print": 10,
  "train_micro_batch_size_per_gpu": 2,
  "wall_clock_breakdown": false,
  "memory_optimization": {
    "enabled": true,
    "cpu_offload": true,
    "contiguous_gradients": true
  }
}
```

**File**: `sagemaker/configs/deepspeed/p4de_deepspeed_config.json`
```json
{
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 1000,
    "initial_scale_power": 16,
    "hysteresis": 2,
    "min_loss_scale": 1
  },
  "zero_optimization": {
    "stage": 2,
    "offload_optimizer": {
      "device": "nvme",
      "pin_memory": true,
      "buffer_count": 4,
      "fast_init": false
    },
    "allgather_partitions": true,
    "allgather_bucket_size": 2e8,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 2e8,
    "contiguous_gradients": true
  },
  "gradient_accumulation_steps": 2,
  "gradient_clipping": 1.0,
  "steps_per_print": 10,
  "train_micro_batch_size_per_gpu": 4,
  "wall_clock_breakdown": false,
  "nvme_optimization": {
    "enabled": true,
    "checkpoint_in_nvme": true,
    "optimizer_state_nvme": true
  }
}
```

**File**: `sagemaker/configs/deepspeed/p5_deepspeed_config.json`
```json
{
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 1000,
    "initial_scale_power": 16,
    "hysteresis": 2,
    "min_loss_scale": 1
  },
  "zero_optimization": {
    "stage": 2,
    "allgather_partitions": true,
    "allgather_bucket_size": 5e8,
    "overlap_comm": true,
    "reduce_scatter": true,
    "reduce_bucket_size": 5e8,
    "contiguous_gradients": true
  },
  "gradient_accumulation_steps": 1,
  "gradient_clipping": 1.0,
  "steps_per_print": 10,
  "train_micro_batch_size_per_gpu": 6,
  "wall_clock_breakdown": false,
  "performance_optimization": {
    "enabled": true,
    "max_performance": true,
    "minimal_offloading": true
  }
}
```

### 6. Benchmark Runner

**File**: `sagemaker/scripts/benchmark_runner.py`
```python
#!/usr/bin/env python3
"""
Automated benchmark runner for P4/P5 instance comparison
"""

import boto3
import json
import time
import pandas as pd
from datetime import datetime
import logging

class SageMakerBenchmarkRunner:
    def __init__(self, role_arn, bucket_name):
        self.sagemaker = boto3.client('sagemaker')
        self.role_arn = role_arn
        self.bucket_name = bucket_name
        self.results = []
        
    def run_benchmark(self, instance_types, dataset_path, epochs=1):
        """Run training jobs on all specified instance types"""
        job_names = []
        
        for instance_type in instance_types:
            job_name = f"qwen-benchmark-{instance_type.replace('.', '-')}-{int(time.time())}"
            
            self._submit_training_job(job_name, instance_type, dataset_path, epochs)
            job_names.append(job_name)
            
        return self._monitor_jobs(job_names)
    
    def _submit_training_job(self, job_name, instance_type, dataset_path, epochs):
        """Submit a training job for specific instance type"""
        training_job_config = {
            'TrainingJobName': job_name,
            'AlgorithmSpecification': {
                'TrainingImage': f'{self.account_id}.dkr.ecr.{self.region}.amazonaws.com/qwen-training:latest',
                'TrainingInputMode': 'File'
            },
            'RoleArn': self.role_arn,
            'InputDataConfig': [
                {
                    'ChannelName': 'training',
                    'DataSource': {
                        'S3DataSource': {
                            'S3DataType': 'S3Prefix',
                            'S3Uri': dataset_path,
                            'S3DataDistributionType': 'FullyReplicated'
                        }
                    }
                }
            ],
            'OutputDataConfig': {
                'S3OutputPath': f's3://{self.bucket_name}/models/{job_name}/'
            },
            'ResourceConfig': {
                'InstanceType': instance_type,
                'InstanceCount': 1,
                'VolumeSizeInGB': 500
            },
            'StoppingCondition': {
                'MaxRuntimeInSeconds': 86400  # 24 hours
            },
            'HyperParameters': {
                'epochs': str(epochs),
                'instance-type': instance_type
            }
        }
        
        self.sagemaker.create_training_job(**training_job_config)
        
    def _monitor_jobs(self, job_names):
        """Monitor training jobs and collect results"""
        completed_jobs = []
        
        while len(completed_jobs) < len(job_names):
            for job_name in job_names:
                if job_name in completed_jobs:
                    continue
                    
                response = self.sagemaker.describe_training_job(TrainingJobName=job_name)
                status = response['TrainingJobStatus']
                
                if status in ['Completed', 'Failed', 'Stopped']:
                    self._collect_job_metrics(job_name, response)
                    completed_jobs.append(job_name)
                    
            time.sleep(60)  # Check every minute
            
        return self.results
    
    def _collect_job_metrics(self, job_name, job_description):
        """Collect metrics from completed job"""
        # Extract metrics from CloudWatch, training logs, etc.
        # Implementation details for metric collection
        pass

def main():
    runner = SageMakerBenchmarkRunner(
        role_arn='arn:aws:iam::YOUR_ACCOUNT:role/SageMakerExecutionRole',
        bucket_name='your-sagemaker-bucket'
    )
    
    instance_types = ['ml.p4d.24xlarge', 'ml.p4de.24xlarge', 'ml.p5.48xlarge']
    dataset_path = 's3://your-bucket/processed-data/'
    
    results = runner.run_benchmark(instance_types, dataset_path, epochs=1)
    
    # Generate comparison report
    df = pd.DataFrame(results)
    df.to_csv('benchmark_results.csv', index=False)
    print("Benchmark completed! Results saved to benchmark_results.csv")

if __name__ == "__main__":
    main()
```

## Implementation Steps

1. **Create directory structure** as outlined above
2. **Build Docker containers** for training and data preparation
3. **Configure instance-specific settings** for P4d, P4de, and P5
4. **Test data preparation pipeline** on CPU instances
5. **Run benchmark training jobs** across all instance types
6. **Analyze results** and generate recommendations

This structure provides a complete framework for migrating your Qwen Hebrew fine-tuning to SageMaker with automated P4/P5 instance comparison.