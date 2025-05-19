# Hyperparameter Tuning and Training Guide

This guide explains how to use the hyperparameter tuning results for training the Qwen model.

## Overview

The workflow consists of the following steps:

1. Run hyperparameter tuning with `start_hp_tuning.sh`
2. Extract the best hyperparameters with `extract_best_params.sh`
3. Train the model with the best hyperparameters using `start_training.sh`

## 1. Hyperparameter Tuning

The hyperparameter tuning process tries different combinations of hyperparameters to find the optimal configuration for your model and dataset.

To start hyperparameter tuning:

```bash
./qwen_model/start_hp_tuning.sh --dataset_path qwen_model/data/dataset/dataset --num_trials 10
```

This will run multiple trials with different hyperparameter configurations and track the results in Weights & Biases.

## 2. Extracting Best Hyperparameters

After the hyperparameter tuning is complete, you can extract the best hyperparameters and generate a training configuration file:

```bash
./qwen_model/extract_best_params.sh --hp_tuning_dir qwen_model/finetuning/hp_tuning --output_path qwen_model/finetuning/training_config.json
```

This script will:
- Look for the best_params.json file in the hyperparameter tuning directory
- If not found, try to get the best parameters from Weights & Biases
- Generate a training configuration file with the best hyperparameters

### Options

- `--hp_tuning_dir`: Directory containing hyperparameter tuning results (default: qwen_model/finetuning/hp_tuning)
- `--output_path`: Path to save the training configuration (default: qwen_model/finetuning/training_config.json)
- `--model_path`: Path to the model (default: qwen_model/model)
- `--dataset_path`: Path to the dataset (default: qwen_model/data/dataset/dataset)
- `--deepspeed_config`: Path to DeepSpeed configuration file (default: qwen_model/deepspeed_config.json)
- `--wandb_project`: W&B project name (default: qwen-hebrew-hp-tuning)
- `--wandb_entity`: W&B entity name (optional)

## 3. Training with Best Hyperparameters

Once you have the training configuration file, you can start training with the best hyperparameters:

```bash
./qwen_model/start_training.sh --config qwen_model/finetuning/training_config.json --dataset_path qwen_model/data/dataset/dataset
```

This script will:
- Check if the configuration file exists
- If not, run extract_best_params.sh to generate it
- Start training with the best hyperparameters
- Log the training process to a file

### Options

- `--config`: Path to the training configuration file (default: qwen_model/finetuning/training_config.json)
- `--dataset_path`: Path to the dataset (default: qwen_model/data/dataset/dataset)
- `--deepspeed_config`: Path to DeepSpeed configuration file (default: qwen_model/deepspeed_config.json)
- `--wandb_project`: W&B project name (default: qwen-hebrew-finetuning)
- `--wandb_entity`: W&B entity name (optional)
- `--wandb_name`: W&B run name (default: qwen-hebrew-finetuning-TIMESTAMP)
- `--seed`: Random seed (default: 42)

## Monitoring Training

The training process runs in the background with nohup. You can monitor the progress by checking the log file:

```bash
tail -f qwen_model/logs/train_TIMESTAMP.log
```

You can also monitor the training in the Weights & Biases dashboard.

## Stopping Training

To stop the training process:

```bash
kill $(cat qwen_model/logs/train_TIMESTAMP.pid)
```

## Best Hyperparameters

The best hyperparameters typically include:

- `learning_rate`: The learning rate for the optimizer
- `weight_decay`: The weight decay for regularization
- `warmup_ratio`: The ratio of warmup steps to total training steps

These hyperparameters are automatically extracted from the tuning results and used for training.