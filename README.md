# Qwen Hebrew Fine-tuning

This repository contains scripts and tools for fine-tuning the Qwen3-30B-A3B-Base model on Hebrew language data.

## Quick Start

```bash
# 1. Clone repository
git clone https://github.com/sarelWeinberger/qwen-hebrew-finetuning.git
cd qwen-hebrew-finetuning

# 2. Get your tokens
# - Hugging Face: https://huggingface.co/settings/tokens
# - GitHub: https://github.com/settings/tokens

# 3. Set up environment (automated)
chmod +x setup_uv_env.sh
./setup_uv_env.sh

# 4. Configure tokens
cat > .env << EOF
GITHUB_TOKEN=your_github_token_here
HF_TOKEN=your_huggingface_token_here
EOF

# 5. Activate environment and verify
source activate_env.sh
python -c "import torch; import transformers; print('Setup complete!')"

# 6. Run pipeline test
python qwen_model/test_pipeline.py --max_samples 10
```

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

# Environment and Setup Files
├── .env                   # Environment variables (create this)
├── .gitignore            # Git ignore patterns
├── pyproject.toml        # UV/Python project configuration
├── requirements.txt      # Python dependencies
├── Makefile              # Development commands
├── setup_uv_env.sh       # Automated environment setup
├── activate_env.sh       # Quick environment activation
├── git_setup.sh          # Git repository setup
└── UV_SETUP.md           # UV environment documentation

# Text Processing Components
text_cleaning/
├── main.py               # Main text cleaning pipeline
├── cleaning_pipeline.py  # Text processing pipeline
├── cleaners/             # Various text cleaning modules
├── fetchers/             # Data source fetchers
└── utils/                # Utilities and configurations

batch_cleaning/
├── build_and_deploy.py   # SageMaker batch processing
├── inference.py          # Inference script for batch processing
├── Dockerfile            # Docker configuration
└── requirements.txt      # Batch processing dependencies

clean_with_gemma_api/
├── gemini_api.py         # Google Gemini API integration
├── gemma_api_quota_aware.py # Quota-aware API calls
└── gemini_api_orchestrator/ # Distributed processing orchestrator
```

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/sarelWeinberger/qwen-hebrew-finetuning.git
cd qwen-hebrew-finetuning
```

### 2. Environment Setup and Authentication

#### Prerequisites
- Python 3.10+ is required
- CUDA 11.8+ is recommended for GPU acceleration
- At least 500GB of disk space for model and data storage
- UV package manager (installed automatically by setup script)

#### Required Tokens
Before starting, you'll need to obtain the following tokens:

1. **Hugging Face Token**: 
   - Go to [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
   - Create a new token with "Read" permissions
   - Copy the token (starts with `hf_`)

2. **GitHub Token** (for development):
   - Go to [https://github.com/settings/tokens](https://github.com/settings/tokens)
   - Create a Personal Access Token with `repo` permissions
   - Copy the token (starts with `github_pat_`)

#### Quick Setup with UV

We provide an automated setup script that installs UV (ultra-fast Python package manager) and sets up the virtual environment:

```bash
# Make the setup script executable and run it
chmod +x setup_uv_env.sh
./setup_uv_env.sh
```

This script will:
- Install UV package manager if not present
- Create a virtual environment with Python 3.10
- Install all project dependencies
- Set up development tools (pytest, black, mypy, jupyter)

#### Manual Environment Setup

If you prefer manual setup or the automated script fails:

```bash
# Install UV
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.cargo/bin:$PATH"

# Create virtual environment
uv venv
source .venv/bin/activate

# Install dependencies
uv pip install -r requirements.txt

# Install additional packages
uv pip install lighteval  # For model evaluation
```

#### Configure Environment Variables

Create a `.env` file with your tokens (this file is already gitignored for security):

```bash
# Create .env file with your actual tokens
cat > .env << EOF
# Authentication tokens
GITHUB_TOKEN=your_github_token_here
HF_TOKEN=your_huggingface_token_here

# UV and Python environment settings
UV_PYTHON=3.10
VIRTUAL_ENV=.venv
PYTHONPATH=.

# Project settings
PROJECT_NAME=qwen-hebrew-finetuning
PYTHON_VERSION=3.10
EOF
```

**OR** set environment variables directly:

```bash
# Export tokens for current session
export GITHUB_TOKEN="your_github_token_here"
export HF_TOKEN="your_huggingface_token_here"

# Make them persistent (add to ~/.bashrc or ~/.zshrc)
echo 'export GITHUB_TOKEN="your_github_token_here"' >> ~/.bashrc
echo 'export HF_TOKEN="your_huggingface_token_here"' >> ~/.bashrc
source ~/.bashrc
```

#### Activate Environment

For future sessions, activate the environment using:

```bash
# Using the provided activation script
source activate_env.sh

# OR manually
source .venv/bin/activate
```

#### Using Make Commands

We provide a Makefile for common development tasks:

```bash
# Show available commands
make help

# Setup environment (alternative to setup_uv_env.sh)
make setup

# Install dependencies
make install

# Install development dependencies
make dev

# Format code
make format

# Run tests
make test

# Clean environment
make clean

# Start training
make run-train

# Start evaluation
make run-eval
```

### 3. Verify Installation

Test that everything is set up correctly:

```bash
# Activate environment
source activate_env.sh

# Check Python and UV versions
python --version  # Should show Python 3.10+
uv --version      # Should show UV version

# Test Hugging Face authentication
python -c "from huggingface_hub import HfApi; api = HfApi(); print('HF Authentication successful!')"

# Test package imports
python -c "import torch; import transformers; import lighteval; print('All packages imported successfully!')"
```

### 4. Git Setup (Optional)

If you want to push changes back to GitHub:

```bash
# Configure git (the script uses environment variables)
./git_setup.sh your_github_username

# This will:
# - Configure git with your credentials
# - Add files to git
# - Create initial commit
# - Set up remote repository
# - Provide instructions for pushing
```

### 5. Data and Model Preparation

#### Weights & Biases Setup (Optional but Recommended)

For logging and monitoring, you can set up Weights & Biases:

```bash
# Install wandb (already included in dependencies)
# Login with your API key
wandb login YOUR_API_KEY
```

#### AWS Credentials (Required for S3 Data)

If you plan to use the S3 data pipeline, configure AWS credentials:

```bash
# Option 1: Environment variables
export AWS_ACCESS_KEY_ID="your_aws_access_key"
export AWS_SECRET_ACCESS_KEY="your_aws_secret_key"
export AWS_DEFAULT_REGION="us-east-1"

# Option 2: AWS CLI configuration
aws configure
```

#### Model and Data Download

The repository includes a comprehensive workflow script that handles downloading the model from Hugging Face and preparing Hebrew data from S3:

```bash
# Ensure environment is activated
source activate_env.sh

# Run the full workflow with your AWS credentials
python qwen_model/run_full_workflow.py \
  --aws_access_key_id YOUR_AWS_ACCESS_KEY \
  --aws_secret_access_key YOUR_AWS_SECRET_KEY
```

#### Model Download Process
The workflow downloads the Qwen3-30B-A3B-Base model from Hugging Face using:

```python
# First downloads config files
snapshot_download(
    repo_id="Qwen/Qwen3-30B-A3B-Base",
    local_dir=model_path,
    ignore_patterns=["*.bin", "*.safetensors"]
)

# Then downloads model weights
snapshot_download(
    repo_id="Qwen/Qwen3-30B-A3B-Base",
    local_dir=model_path,
    ignore_patterns=["*.md", "*.txt"],
    resume_download=True
)
```

#### S3 Data Processing
The workflow processes Hebrew data from S3 using the following steps:

1. **S3 Data Download**:
   - Connects to AWS S3 using provided credentials
   - Downloads Hebrew text datasets from the "israllm-datasets" bucket
   - Includes robust error handling with fallback to sample data

2. **Data Processing**:
   - Extracts text from CSV files using pandas
   - Cleans and normalizes Hebrew text
   - Converts to JSONL format suitable for fine-tuning
   - Creates training dataset with appropriate formatting

### 6. Test the Pipeline

Before running a full training job, test the pipeline with a small subset of data:

```bash
# Ensure environment is activated
source activate_env.sh

# For single GPU testing
python qwen_model/test_pipeline.py \
  --dataset_path qwen_model/data/dataset/dataset \
  --max_samples 100

# For multi-GPU testing with DeepSpeed
deepspeed --num_gpus=8 qwen_model/test_pipeline.py \
  --dataset_path qwen_model/data/dataset/dataset \
  --max_samples 100 \
  --single_device \
  --deepspeed qwen_model/deepspeed_config.json
```

### 7. Run Hyperparameter Tuning

```bash
# Ensure environment is activated
source activate_env.sh

# Using shell script
./qwen_model/start_hp_tuning.sh \
  --dataset_path qwen_model/data/dataset/dataset \
  --num_trials 10

# OR using Make
make run-eval
```

### 8. Train the Model

```bash
# Ensure environment is activated
source activate_env.sh

# Using shell script
./qwen_model/start_training.sh

# OR using Make
make run-train
```

### 9. Evaluate the Model

```bash
# Ensure environment is activated
source activate_env.sh

# Using shell script
./qwen_model/start_evaluation.sh \
  --model_path qwen_model/finetuned

# OR using Make
make run-eval
```

## Troubleshooting

### Common Issues

1. **Disk Space Issues**:
   ```bash
   # Check disk space
   df -h
   
   # Clean UV cache if needed
   uv cache clean
   ```

2. **Token Authentication Errors**:
   ```bash
   # Verify tokens are set
   echo $HF_TOKEN
   echo $GITHUB_TOKEN
   
   # Test HF authentication
   python -c "from huggingface_hub import HfApi; HfApi().whoami()"
   ```

3. **CUDA/GPU Issues**:
   ```bash
   # Check CUDA availability
   python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
   python -c "import torch; print(f'GPU count: {torch.cuda.device_count()}')"
   ```

4. **Environment Issues**:
   ```bash
   # Clean and recreate environment
   make clean
   make setup
   ```

### Dependencies

The project includes these key dependencies:
- **Core ML**: PyTorch, Transformers, DeepSpeed, Accelerate
- **Data Processing**: Pandas, NumPy, Datasets
- **Evaluation**: LightEval, ROUGE Score
- **Monitoring**: Weights & Biases, TensorBoard
- **Optimization**: Optuna for hyperparameter tuning
- **Cloud**: boto3 for AWS S3 integration
- **Development**: pytest, black, mypy, jupyter

### Package Management

We use UV for fast, reliable Python package management:
- Lightning-fast dependency resolution
- Efficient caching and downloading
- Better dependency conflict resolution
- Easy virtual environment management

For more details on UV commands, see: [UV Documentation](https://github.com/astral-sh/uv)

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