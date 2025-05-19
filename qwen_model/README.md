# Qwen Model Fine-tuning for Hebrew

This repository contains scripts for downloading the Qwen3-30B-A3B-Base model and fine-tuning it on Hebrew language data.

## Overview

The workflow consists of the following steps:

1. Download the Qwen3-30B-A3B-Base model
2. Prepare the model for fine-tuning
3. Download and process Hebrew data from S3
4. Prepare the dataset for training
5. Fine-tune the model

## Requirements

- Python 3.8+
- PyTorch 2.0+
- Transformers 4.30+
- Datasets
- DeepSpeed (for distributed training)
- AWS credentials (for accessing S3 data)

## Usage

### Full Workflow

To run the complete workflow, use the `run_full_workflow.py` script:

```bash
python qwen_model/run_full_workflow.py --aws_access_key_id <YOUR_AWS_ACCESS_KEY> --aws_secret_access_key <YOUR_AWS_SECRET_KEY>
```

This will:
1. Download the Qwen model
2. Prepare it for fine-tuning
3. Download and process Hebrew data from S3
4. Prepare the dataset for training

### Individual Steps

You can also run each step individually:

#### 1. Download the Model

```bash
python qwen_model/main.py --action download
```

#### 2. Prepare the Model for Fine-tuning

```bash
python qwen_model/main.py --action prepare
```

#### 3. Download and Process S3 Data

```bash
python qwen_model/download_s3_data.py --aws_access_key_id <YOUR_AWS_ACCESS_KEY> --aws_secret_access_key <YOUR_AWS_SECRET_KEY>
```

#### 4. Fine-tune the Model

```bash
python qwen_model/train.py --dataset_path qwen_model/data/dataset/dataset --config qwen_model/finetuning/training_config.json
```

You can also use Weights & Biases for experiment tracking:

```bash
python qwen_model/train.py --dataset_path qwen_model/data/dataset/dataset --wandb_project your-project-name --wandb_entity your-username
```

## Handling S3 Authentication Issues

If you encounter S3 authentication issues (403 Forbidden, SignatureDoesNotMatch, etc.), the script will automatically create sample Hebrew data for testing purposes. This allows you to test the workflow without needing valid S3 credentials.

To resolve S3 authentication issues:

1. Verify that your AWS credentials are correct
2. Ensure that your IAM user has the necessary permissions to access the S3 bucket
3. Check that the region is correct (default is us-east-1)
4. Verify that the bucket name is correct (default is israllm-datasets)

## Dataset Format

The dataset should be in JSONL format with a "text" field containing the Hebrew text. For example:

```json
{"text": "שלום עולם! זוהי דוגמה לטקסט בעברית."}
```

## Hardware Requirements for Fine-tuning

Fine-tuning a 30B parameter model requires significant hardware resources:

- Multiple high-memory GPUs (recommended: at least 4x A100 80GB or equivalent)
- DeepSpeed or FSDP for distributed training
- Gradient checkpointing for memory efficiency

All these are configured in the training script, but you need the appropriate hardware resources.

## Troubleshooting

### Empty Dataset

If the dataset is empty after processing, check:
- S3 authentication issues (the script will create sample data in this case)
- CSV parsing errors (the script includes robust error handling for various CSV formats)

### Training Errors

If you encounter errors during training:
- Ensure the dataset has the expected format with a "text" column
- Check that you have sufficient GPU memory
- Verify that the DeepSpeed configuration is appropriate for your hardware

## Experiment Tracking with Weights & Biases

The training script integrates with [Weights & Biases](https://wandb.ai/) for experiment tracking. To use this feature:

1. Sign up for a free account at [wandb.ai](https://wandb.ai/)
2. Install the wandb package: `pip install wandb`
3. Log in to wandb: `wandb login`
4. Run the training script with wandb parameters:

```bash
python qwen_model/train.py --dataset_path qwen_model/data/dataset/dataset \
    --wandb_project your-project-name \
    --wandb_entity your-username \
    --wandb_name your-run-name
```

This will track metrics, model checkpoints, and training configuration in your Weights & Biases dashboard.

## License

This project is licensed under the terms of the license included with the Qwen model.