# NeMo PWC Israel-LLM Project Training

Minimal steps to preprocess data and launch training in the NVIDIA NeMo 25.07 container.

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
  --tokenizer-type Qwen/Qwen3-30B-A3B-Base \
  --append-eod \
  --output-prefix ./tok-data/hebdata_hewiki \
  --workers 128 \
  --preproc-folder \
  --files-filter '*' \
  --log-interval 50000
```

Outputs: `hebdata_hewiki_text_document.bin` and `.idx` (under the created preprocess folder).

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

> Training config lives in `train.py` lines **22–29**. Adjust as needed. 

> Note, when cancelling a run the rest of the processes will still be hogging the GPU, make sure to kill them:

> ```bash
> ps ux | grep 'nemo_run.core.runners.fdl_runner' | awk '{print $2}' | xargs kill -9
> ```

## Cluster Instructions

```bash
docker run \
  -e HF_HOME=/workspace/.cache/huggingface/ \
  -e NEMO_HOME=/workspace/.cache/nemo/ \
  -e SLURM_CONF=/opt/slurm/etc/slurm.conf \
  -e NEMORUN_HOME=/fsx/.nemo_run \
  -e WORKSPACE_DIR=$(pwd) \
  -it --shm-size=10gb --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --network host \
  -v /fsx:/fsx \
  -v /opt/slurm:/opt/slurm:ro \
  -v /var/run/munge:/var/run/munge:rw \
  -v /var/log/aws:/var/log/aws \
  -v $(pwd):/workspace \
  -w /workspace \
  nvcr.io/nvidia/nemo:25.07.nemotron-nano-v2 \ 
  adduser slurm && bash
```

> apt update && apt install -y slurm-client && apt purge -y slurm-client

Pull the container on all the nodes

```bash
srun -N 2 docker pull nvcr.io/nvidia/nemo:dev
```

Run `ip a`, and find the relevant interface (not lo/docker0/veth...), and run, for example:

```bash
export NCCL_SOCKET_IFNAME=ens6
```

Run `ip a` on all the nodes, to find the relevant interface:
```bash
srun -N 2 ip a
```

In `train.py`, set it:
```python
env_vars=dict(
    NCCL_SOCKET_IFNAME='enp137s0', # for example
    ...
)
```

Also, set the environment variable:
```bash
export WANDB_API_KEY=[WANDB_API_KEY]
```

> Not sure where this fits in:
```bash
mkdir -p /workspace/tok-data/hebdata_hewiki_text_document/cache/
chmod -R 777 /workspace/tok-data/hebdata_hewiki_text_document/cache/
```

## Incomplete TODO list / Next steps

* Tokenize everything (cover all corpora beyond `hewiki-data.jsonl`).
* Export checkpoints script.
* Connect to S3 for saving/export (credentials + sync strategy).
* Multinode training.
* SageMaker (TBD).
* Resuming (handle timestamped output dirs like `./2025-xxxx` and point trainer to the right checkpoint).
