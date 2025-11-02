
# Qwen3-30B-A3B Fine-tuning

A working reproduction of single-node distributed continuous-pretraining of a MoE model, using HF Accelerate + DeepSpeed, with no CPU-offloading. 

## Quick Start

### AWS Details

This code was proven to work with the following settings - it may work in other settings as well. 

Machine Type: `p4de.24xlarge` or `p5en.48xlarge`

AMI: `Deep Learning OSS Nvidia Driver AMI GPU PyTorch 2.8 (Ubuntu 22.04)`

Storage: If the checkpoints are going to persist when turning the machine on & off, then add a large sum. Otherwise, 100GB is enough and in the setup HF_HOME & checkpoints will be set to the NVME drive. If the data needs to be persisted, make sure to update the `cpt_config.json` so that the checkpoints will be saved to EBS and not NVME. 

### 1. One-Command Setup

```bash
# Clone the repository
git clone https://github.com/sarelWeinberger/qwen-hebrew-finetuning.git
cd qwen-hebrew-finetuning/training/deepspeed/

# Run the setup script (you may be prompted to log in to HF / WandB / S3)
source ./setup.sh
```

### 2. Add Your Datasets

Place your JSONL dataset files in the `datasets/` directory:

```json
{"text": "text of document one"}
{"text": "text of document two"}
...
```

### 3. Start Training

```bash
# Launch training (runs in background)
nohup ./train_all.sh

# Monitor progress
tail -f logs/train.log

# Terminate a running training process
./kill_train_all.sh
```

### Resuming on Existing Machine

When logging back into a machine after a stop, just run setup again:

```bash
cd qwen-hebrew-finetuning/training/deepspeed/
source setup.sh
```

### Advanced Training Options

### Configuration

The script by default reads in the file [cpt_config.json](./cpt_config.json) for the configuration, but you can specify any other JSON file. 

These are the parameters (currently) supported:

- `model_name_or_path`: (*Required*) - The path to the model we want to train. For the first phase of training it should be `Qwen/Qwen3-30B-A3B-Base`, and for the next phases we want to point it to the intermediate models (e.g., long context)

- `max_seq_length`: (*Required*) - The maximum sequence length to train on - e.g., 512, 2048, 4096, etc. 

- `per_device_train_batch_size`: (*Required*) - The per-GPU batch size to run (also known as micro batch-size).

- `gradient_accumulation_steps`: (*Required*) - The number of gradient accumulation steps. This is the number of steps to compute gradients for before performing an update to the parameters. 

    > **IMPORTANT**: The value here *must* match the value in the deepspeed config passed into the script. The instructions here use the file [deepspeed_zero3.yaml](./deepspeed_zero3.yaml). 

- `output_dir`: (*Required*) - The directory to save all the intermediate checkpoints. Currently set to the NVME, but should be updated to EBS if the data should persist after shutdown. 

- `learning_rate`: (*Optional*, defaults to `1e-5`) - The (max) learning rate to use for training. 

- `logging_steps`: (*Optional*, defaults to `10`) - Log the results to wandb every N steps. 

- `num_train_epochs` (*Optional*, defaults to `1`) - The number of epochs to run on the data. Ignored if `max_steps` is provided. 

- `max_steps` (*Optional*, defaults to `None`) - The number of training steps to run on the data. If this is set, it overrides the `num_train_epochs` field. 

- `save_steps` (*Optional*, defaults to `500`) - Save a checkpoint of the model every N steps. 

- `save_total_limit` (*Optional*, defaults to `5`) - The maximum number of checkpoints to save to disk before deleting old ones. 

- `warmup_ratio` (*Optional*, defaults to `None`) - What percent of the training should be used for LR warmup. E.g., `0.1` means warmup should be 10% of the train. If this is set, it overrides the `warmup_steps` field. 

- `warmup_steps` (*Optional*, defaults to `100`) - The number of LR warmup steps.  Ignored if `warmup_ratio` is provided. 

### DeepSpeed

The script uses `DeepSpeed` to accelerate training, the default configuration used here can be found in [deepspeed_zero3.yaml](./deepspeed_zero3.yaml). 

The only important to note here is that the `gradient_accumulation_steps` **must** be set to the same value specified in the JSON configuration. 

#### Single Training Run

For a single training run with custom parameters:

```bash
accelerate launch --config_file=deepspeed_zero3.yaml train.py --wandb_name run-name-for-wandb
```

Other parameters to set:

- `--config /path/to/alternative/config.json` (Path to an alternative [Configuration](#configuration))
- `--dataset_path /path/to/data/jsonl` (Path to alternative [Data](#data))
- `--seed 111` (Specifying a different random seed)
- `--wandb_project a_different_wandb_project` (For separating runs into a different WandB project)
- `--verbose` (Produces more verbose output, and prints things like model architecture)
- `--dynamic_packing` (Dynamically packs the dataset before training - this should only be done on small datasets, as it is very costly and takes time. Can be used for test runs on sample data, but for larger data there should be a different solution)

## Monitoring Training

### Manual Monitoring

```bash
# Follow training logs
tail -f logs/train.log

# Check if training is running
ps aux | grep train

# Monitor GPU usage
nvidia-smi

# Check disk space
df -h
```

### Stopping Training

```bash
# Stop training using the provided script
./kill_train_all.sh
```

## Troubleshooting

- **CUDA out of memory**
   - Reduce `per_device_train_batch_size` in `cpt_config.json`
   - Increase `gradient_accumulation_steps` to maintain effective batch size

## Next Steps

We have a working script here that trains, utilizes all GPUs with DeepSpeed+Zero3, and the output loss is reasonable (aka, correct tokenization)

That being said, there are still many improvements that must be done to the script for large-scale training:

- **Packing**: Update: We added the `pack_dataset` from `TRL` which packs a given a dataset during the run. This works for smaller datasets, but should be avoided on larger datasets since it takes a long time. 
> In order to maximize utilization we want to avoid padding, and instead we pack sequences together. See [here](https://lweitkamp.github.io/posts/packing/index.html) for more details. This is crucial for successfully training the LLM. 

- **Large Dataset**: The current script assumes a small sample, and loads the full dataset into memory tokenizing ahead of time. When working with very large datasets, we can't have them all in memory at once - instead we want to pre-tokenize and stream them to avoid any bottlenecks. 

- **GPU Utilization**: Update: Works 100% for non MoE models.
> Right now we have a reasonable GPU utilization, average at ~60%. That's decent, but we want to be nearing 99%. This can definitely improve with hardware, but there are software changes that should be done. 

- **Hyperparameter Tuning**: We want to optimize the hyperparameters we're using - currently there is no support for `optuna`, that needs to be added. 

- **Checkpoint Uploading**: When ready, we should open up an organization in HuggingFace where the checkpoints will be uploaded to as the model trains. This allows us to test them out, evlauate them, etc. This requires both setting up an organization, and also configuring the `push_to_hub` option in the script. 

- **Evaluation**: There are two types of evaluation that still need to be done:

    1. Perplexity - this can be done within the `train.py` script, we just need to update the script accordingly and set `eval_dataset` to the relevant dataset. 

    2. Quality - this will involve pulling the checkpoints from the hub as they're uploaded, and then evaluting them using a library like `lighteval`. 
