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

## Hardware Requirements

- 8x NVIDIA H100 GPUs (80GB each)
- Sufficient CPU RAM and disk space for model and dataset storage

## Directory Structure

```
qwen_model/
├── data/                  # Data storage and processing
├── finetuning/            # Fine-tuning configurations and outputs
├── logs/                  # Training and evaluation logs
├── model/                 # Model files (downloaded separately)
├── download_model.py      # Script to download the Qwen model
├── prepare_for_finetuning.py # Prepare model for fine-tuning
├── prepare_dataset.py     # Dataset preparation utilities
├── download_s3_data.py    # Download Hebrew data from S3
├── train.py               # Main training script
├── hp_tuning.py           # Hyperparameter tuning with Optuna
├── evaluate_hebrew.py     # Evaluation on Hebrew LLM Leaderboard
├── test_pipeline.py       # Test pipeline with small subset of data
├── run_full_workflow.py   # Run the complete workflow
├── start_training.sh      # Shell script to start training with nohup
├── start_hp_tuning.sh     # Shell script to start hyperparameter tuning
├── start_evaluation.sh    # Shell script to start evaluation
├── start_test_pipeline.sh # Shell script to test the pipeline
└── TUNING_AND_EVALUATION.md # Detailed documentation
```

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

### 3. Test the Pipeline

Before running a full training job, test the pipeline with a small subset of data:

```bash
./qwen_model/start_test_pipeline.sh --dataset_path qwen_model/data/dataset/dataset
```

### 4. Run Hyperparameter Tuning

```bash
./qwen_model/start_hp_tuning.sh --dataset_path qwen_model/data/dataset/dataset --num_trials 10
```

### 5. Train the Model

```bash
./qwen_model/start_training.sh
```

### 6. Evaluate the Model

```bash
./qwen_model/start_evaluation.sh --model_path qwen_model/finetuned
```

## Documentation

For detailed information about hyperparameter tuning, training configuration, and evaluation, see [TUNING_AND_EVALUATION.md](qwen_model/TUNING_AND_EVALUATION.md).

## Features

- **Distributed Training**: Optimized for 8x H100 GPUs using DeepSpeed ZeRO-3
- **Hyperparameter Optimization**: Automated tuning with Optuna
- **Robust Logging**: Comprehensive metrics tracking with Weights & Biases
- **Uninterrupted Training**: All scripts use nohup to continue after SSH disconnection
- **Pipeline Testing**: Test the full pipeline with a small subset before full training
- **Hebrew Evaluation**: Integration with the Hebrew LLM Leaderboard

## License

This project is licensed under the terms of the license included with the Qwen model.