# Qwen Model Hyperparameter Tuning and Evaluation

This document describes the hyperparameter tuning and evaluation features added to the Qwen Hebrew fine-tuning project.

## Table of Contents

1. [Test Pipeline](#test-pipeline)
2. [Hyperparameter Tuning with Optuna](#hyperparameter-tuning-with-optuna)
3. [Training Configuration](#training-configuration)
4. [Hebrew LLM Leaderboard Evaluation](#hebrew-llm-leaderboard-evaluation)
5. [Usage Examples](#usage-examples)

## Test Pipeline

Before running a full training job on your entire dataset, it's recommended to test the pipeline with a small subset of data. This helps identify any issues with the training setup before committing to a full training run.

### Features

- Tests the full training pipeline with a minimal number of samples
- Uses a small sequence length to reduce memory requirements
- Runs for a limited number of steps to quickly verify functionality
- Validates that the model, tokenizer, and dataset work together correctly
- Checks that the training and evaluation loops run without errors

### How to Run

```bash
# Run the test pipeline with default settings
./qwen_model/start_test_pipeline.sh

# Run with custom settings
./qwen_model/start_test_pipeline.sh \
  --dataset_path qwen_model/data/dataset/dataset \
  --max_samples 10 \
  --max_steps 5 \
  --max_seq_length 128
```

### Configuration Options

- `--dataset_path`: Path to the dataset to sample from
- `--config`: Path to the training configuration file
- `--output_dir`: Directory to save the test model
- `--max_samples`: Maximum number of samples to use for testing
- `--max_steps`: Maximum number of training steps to run
- `--max_seq_length`: Maximum sequence length for tokenization
- `--seed`: Random seed for reproducibility
- `--skip_wandb`: Skip Weights & Biases logging

## Hyperparameter Tuning with Optuna

The project now includes hyperparameter tuning capabilities using Optuna, a powerful hyperparameter optimization framework. This allows you to automatically find the best hyperparameters for fine-tuning the Qwen model on your Hebrew dataset.

### Features

- Automated hyperparameter search using Bayesian optimization (TPE sampler)
- Pruning of unpromising trials using MedianPruner
- Integration with Weights & Biases for experiment tracking
- Support for early stopping to prevent overfitting
- Optimization of key hyperparameters:
  - Learning rate
  - Batch size
  - Weight decay
  - Warmup ratio

### How It Works

The hyperparameter tuning process:

1. Creates an Optuna study with the specified number of trials
2. For each trial, samples hyperparameters from the defined search space
3. Trains the model with the sampled hyperparameters
4. Evaluates the model on the validation set
5. Reports the evaluation loss to Optuna
6. Saves the best model and hyperparameters

## Training Configuration

The training is configured with the following parameters in the training configuration:

```json
{
  "model_name_or_path": "/home/ec2-user/qwen_model/model",
  "output_dir": "qwen_model/finetuned",
  "fp16": true,
  "gradient_accumulation_steps": 8,
  "per_device_train_batch_size": 1,
  "learning_rate": 1e-05,
  "weight_decay": 0.01,
  "num_train_epochs": 3,
  "save_steps": 500,
  "save_total_limit": 3,
  "logging_steps": 100,
  "max_grad_norm": 1.0,
  "warmup_ratio": 0.03
}
```

These parameters are compatible with all versions of the transformers library and provide a good starting point for fine-tuning the Qwen model.

## Hebrew LLM Leaderboard Evaluation

A new evaluation script has been added to evaluate fine-tuned models using the Hebrew LLM Leaderboard metrics. This allows you to compare your fine-tuned model against other Hebrew language models.

### Evaluation Categories

The evaluation covers various aspects of language model performance:

- Hebrew understanding
- Reasoning
- Knowledge
- Instruction following
- Math
- Coding
- Factuality
- Safety

### Integration with Weights & Biases

Evaluation results are logged to Weights & Biases, allowing you to:

- Track model performance across different fine-tuning runs
- Compare different models and hyperparameter configurations
- Visualize performance metrics
- Share results with your team

## Usage Examples

### Running the Test Pipeline

```bash
# Run with nohup to continue after SSH disconnection
./qwen_model/start_test_pipeline.sh \
  --dataset_path qwen_model/data/dataset/dataset \
  --max_samples 10 \
  --max_steps 5 \
  --max_seq_length 128
```

### Running Hyperparameter Tuning

```bash
# Run with nohup to continue after SSH disconnection
./qwen_model/start_hp_tuning.sh \
  --dataset_path qwen_model/data/dataset/dataset \
  --num_trials 10 \
  --output_dir qwen_model/finetuning/hp_tuning
```

### Training with Full Dataset

```bash
# Run with nohup to continue after SSH disconnection
./qwen_model/start_training.sh
```

### Evaluating on Hebrew LLM Leaderboard

```bash
# Run with nohup to continue after SSH disconnection
./qwen_model/start_evaluation.sh \
  --model_path qwen_model/finetuned
```

## Tips for Best Results

1. **Always test the pipeline first**: Run the test pipeline with a small subset of data to verify that everything works correctly before starting a full training run.

2. **Create a good validation split**: Make sure your validation set is representative of the data you want your model to perform well on.

3. **Balance exploration and exploitation**: For hyperparameter tuning, start with a wide search space and then narrow down based on promising regions.

4. **Monitor training with W&B**: Keep an eye on the learning curves to detect overfitting or other issues early.

5. **Use multiple evaluation metrics**: While the optimization focuses on minimizing validation loss, consider other metrics like perplexity and accuracy for a more complete picture.

6. **Leverage all 8 H100 GPUs**: The distributed training setup is optimized for your hardware configuration, allowing for efficient training of the 30B parameter model.

7. **Use nohup for long-running jobs**: All scripts are designed to work with nohup, ensuring that your training continues even if your SSH connection is lost.