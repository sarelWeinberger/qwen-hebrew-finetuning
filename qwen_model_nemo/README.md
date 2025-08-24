# NeMo Qwen3-30B-A3B Training

Minimal steps to preprocess data and launch training in the NVIDIA NeMo 25.07 container.

## Links

* Guide: [https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/quickstart.html](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemo-2.0/quickstart.html)
* Code:  [https://github.com/NVIDIA/NeMo](https://github.com/NVIDIA/NeMo)

## Prereqs

* Tested on AWS 8xH200 Machine - p5en.48xlarge
* NVIDIA GPU + drivers, Docker with GPU support.
* This image: `nvcr.io/nvidia/nemo:25.07.nemotron-nano-v2`.

## Pull & Run

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

## Data

For my test run, I placed the source JSONL in `./data`. Example used:

* `./data/hewiki-data.jsonl`

### Build tokenized dataset (produces two files - .bin / .idx)

```bash
python /opt/NeMo/scripts/nlp_language_modeling/preprocess_data_for_megatron.py \
  --input ./data/hewiki-data.jsonl \
  --keep-newlines \
  --tokenizer-library huggingface \
  --tokenizer-type Qwen/Qwen3-30B-A3B \
  --append-eod \
  --output-prefix hebdata_hewiki \
  --workers 128 \
  --preproc-folder \
  --files-filter '*' \
  --log-interval 50000
```

Outputs: `hebdata_hewiki_text_document.bin` and `.idx` (under the created preprocess folder).

## Import the model (inside the container)

```bash
pip install -U huggingface_hub
hf download Qwen/Qwen3-30B-A3B-Base
python import.py
```

## Train

```bash
python train.py --run_name RUN_NAME [--use_fp8]
```

> Training config lives in `train.py` lines **22–29**. Adjust as needed. 

> Note, when cancelling a run the rest of the processes will still be hogging the GPU, make sure to kill them:

> ```bash
> ps ux | grep 'nemo_run.core.runners.fdl_runner' | awk '{print $2}' | xargs kill -9
> ```

## Incomplete TODO list / Next steps

* Tokenize everything (cover all corpora beyond `hewiki-data.jsonl`).
* Export checkpoints script.
* Connect to S3 for saving/export (credentials + sync strategy).
* Multinode training.
* SageMaker (TBD).
* Resuming (handle timestamped output dirs like `./2025-xxxx` and point trainer to the right checkpoint).
