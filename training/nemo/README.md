# NeMo PWC Israel-LLM Project Training

Minimal steps to preprocess data and launch training in the NVIDIA NeMo 25.07 container.

## Setting up VS-Code Tunneling:

Run this command:

```bash
sudo apt update && sudo apt install screen curl -y && screen -dmS vscode_tunnel sh -c 'curl -Lk "https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64" --output vscode_cli.tar.gz; tar -xf vscode_cli.tar.gz && ./code tunnel --random-name'
screen -r
```

Then, navigate to `GitHub Account` and follow the instructions. Once done, press `Ctrl+a`, then `d` to disconnect from the screen and leave it running in the background.



## Links

* Guide: [https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/quickstart.html](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/quickstart.html)
* Code:  [https://github.com/NVIDIA/NeMo](https://github.com/NVIDIA/NeMo)

## Prereqs

* Tested on AWS 8xH200 Machine - p5en.48xlarge
* Tested on AWS SageMaker Hyperpod - 2 x g6e.48xLarge
* NVIDIA GPU + drivers, Docker with GPU support.
* This image: `nvcr.io/nvidia/nemo:25.07.nemotron-nano-v2`.

## Single Node Instructions

### Pull & Run

```bash
docker pull nvcr.io/nvidia/nemo:25.07.nemotron-nano-v2

docker run --gpus all \
  -e HF_HOME=/home/ubuntu/nvme/.cache/huggingface/ \
  -e NEMO_HOME=/home/ubuntu/nvme/.cache/nemo/ \
  -it --shm-size=10gb --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v $(pwd):/home/ubuntu/nemo_training/ \
  -v /opt/dlami/nvme:/home/ubuntu/nvme/ \
  -w /home/ubuntu/nemo_training/ \
  nvcr.io/nvidia/nemo:25.07.nemotron-nano-v2
```

### Data

For my test run, I placed the source JSONL in `./data`. Example used:

* `./data/hewiki-data.jsonl`

#### Build tokenized dataset (produces two files - .bin / .idx)

> Important Note: The tokenizing does *not* require a machine with a GPU, can be run on a CPU machine. If you are on a CPU machine, make sure to remove the `--gpus all` from the launch command. 

```bash
mkdir tok-data
python /opt/NeMo/scripts/nlp_language_modeling/preprocess_data_for_megatron.py \
  --input ./data \
  --keep-newlines \
  --tokenizer-library huggingface \
  --tokenizer-type Qwen/Qwen3-8B \
  --append-eod \
  --output-prefix ./tok-data/hebdata_hewiki \
  --workers 128 \
  --preproc-folder \
  --files-filter '*' \
  --log-interval 50000
```

Outputs: `hebdata_hewiki_text_document.bin` and `.idx` (under the created preprocess folder). You can change the output prefix as you wish, to help distinguish the different corpora. 

### Import the model (inside the container)

```bash
pip install -U huggingface_hub
hf download Qwen/Qwen3-30B-A3B-Base
python import.py
```

> If you want to import a different model, then you need to update the following:
> 1. The `hf download` command should be updated
> 2. In `import.py`, make sure to update the import line. NOTE: You need to update the source name to point to the correct HF model, and **ALSO** update the model config. E.g., for Qwen3-8B, you need to add an import to Qwen3Config8B in line 2, and update it in line 5 as well. You can view the full list of configurations in the source code [here](https://github.com/NVIDIA/NeMo).

### Training *on a single node*

```bash
python train.py --run_name RUN_NAME [--use_fp8]
```

> Training config lives in `train.py` lines **22ΓÇô29**. Adjust as needed. 

> Note, when cancelling a run the rest of the processes will still be hogging the GPU, make sure to kill them:

> ```bash
> ps ux | grep 'nemo_run.core.runners.fdl_runner' | awk '{print $2}' | xargs kill -9
> ```

## Cluster Instructions
connect to cluster:
```bash
./easy-ssh.sh -c login-group ml-cluster --region us-east-2
```


### Home Directory

Whenever you open a shell, navigate to the home directory (this will change once we move to FSx for Lustre on S3):

```bash
export HOME_DIR=/fsx/ubuntu/qwen-hebrew-finetuning/training/nemo
cd $HOME_DIR
```

### Set up the containers

On the controller node, run (replace the *2* without the actual number of nodes in the cluster):

```bash
srun -N 2 docker pull nvcr.io/nvidia/nemo:25.07.nemotron-nano-v2
srun -N 2 sudo chmod -R 1777 /opt/sagemaker/tmp/
docker pull nvcr.io/nvidia/nemo:25.07.nemotron-nano-v2
chmod o+rx /fsx/ubuntu
```

### Data

> IMPORTANT: This assumes that you've read and understood the [data](#data) section above, to understand what the tokenized format is. 

It is *not* recommended to tokenize on these machines - these are heavy compute machines which aren't utilized well. Ideally, just copy the data into the data directory (once we move to FSx for Lustre - you can just point it to the other directory):

```bash
export DATA_DIR=$HOME_DIR/data
mkdir -p $DATA_DIR
sudo chmod -R 777 $DATA_DIR
cp ... $DATA_DIR/
```

Important: Once you copy all the data in, make sure to run this command:

```bash
sudo chmod -R 777 $DATA_DIR
```

### Importing a model 

As we are running continuous pre-training, we want to import the model from HuggingFace to NeMo. This should only be done once per `HOME_DIR`, since the models will be stored in the cache in the fsx. 

This process needs *a* GPU to do, so we will run it from one of the workers:

```bash
sinfo
# pick one of the nodes -> e.g., ip-10-1-1-1
ssh -t ip-10-1-1-1 "HOME_DIR='$HOME_DIR' bash"

# navigate to the our home dir & launch the container 
cd $HOME_DIR
bash launch_docker.sh 
```

From here, follow along from the section in single node [here](#import-the-model-inside-the-container)

Once done, exit back to the default node. 

### Running the actual training!

#### Prerequisites

Let's just go over the prerequisites:

1. You set the value of `HOME_DIR` [here](#home-directory)

2. You have a directory with the data files (.bin/.idx), as instructed [here](#data-1). You also need to update [train.py](./train.py) under `pretrain.data = run.Config(PreTrainingDataModule, ....` - make sure all the paths there point to the relevant data directory. If it's under the `HOME_DIR`, then you can use `/workspace/...`. Otherwise, you can use the full `/fsx/..` path. 

#### Launching & setting up the container

Start by launching the docker, and then setting it up for slurm run:

```bash
bash launch_docker.sh
```

Set up: 

```bash
pip install -U huggingface_hub git+https://github.com/NVIDIA-NeMo/Run
apt update && apt install -y libmunge2 && adduser --disabled-password --gecos "" slurm
export PATH="$PATH:/opt/slurm/bin"
```

Set up WandB:

```bash
export WANDB_API_KEY=[insert the api key here]
```

#### Start the training! 

> NOTE: It by default resumes training if there is a checkpoint in the specified directory. 

Just run the python script, and let the magic work! (Perhaps run this inside a screen, so it never quits):

```bash
python train.py --checkpoints_path /fsx/test_runs/checkpoints-8b --run_name qwen3-8b-nemo-mn-test --use_fp8 --num_nodes 2 --model Qwen3_8B
```

## Incomplete TODO list / Next steps

* Tokenize everything (cover all corpora beyond `hewiki-data.jsonl`).
* Export checkpoints script.
* Connect to S3 for saving/export (credentials + sync strategy).

## Diamonds in the Ruff

- Answers to all your EFA-related prayers: [https://github.com/aws/aws-ofi-nccl/blob/master/doc/efa-env-var.md](https://github.com/aws/aws-ofi-nccl/blob/master/doc/efa-env-var.md)


