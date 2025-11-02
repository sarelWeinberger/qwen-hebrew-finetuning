# NeMo to Hugging Face Checkpoint Conversion

This tool converts an NVIDIA NeMo checkpoint into the Hugging Face (`.safetensors`) format. It includes a Python script that automatically "patches" the NeMo checkpoint config to allow conversion on a single-GPU machine.

## Prerequisites

- **You must run this on an instance with at least 1 NVIDIA GPU.**
- The host machine must have Docker and the **NVIDIA Container Toolkit** installed (this allows Docker to use the GPU).

## Instructions

### 1. Pull the Docker Image

Pull the specific NeMo container that includes all necessary dependencies.

```bash
sudo docker pull nvcr.io/nvidia/nemo:25.07.nemotron-nano-v2
```

### 3. Run the Container

Start an interactive shell inside the Docker container. This command mounts your current directory (`$(pwd)`) into the `/workspace` directory inside the container.

```bash
sudo docker run --gpus all --rm -it \
  --shm-size=16g \
  -v $(pwd):/workspace \
  -v /fsx:/fsx \
  nvcr.io/nvidia/nemo:25.07.nemotron-nano-v2 \
  bash

cd /workspace
 ```

### 4. Run the Conversion

Once you are inside the container's shell, run the `convert.py` script. You must specify the name of your NeMo checkpoint folder and the desired name for the new Hugging Face output folder.

```bash
python convert.py --nemo_model <nemo-checkpoint-full-path> --hf_model <hf-checkpoint-name>
```

**Example:**

If your checkpoint is located at `/fsx/test_runs/step-1234`, you would run:

```bash
python convert.py --nemo_model /fsx/test_runs/step-1234 --hf_model step-1234
```

The script will:
1. Read from `/fsx/test_runs/step-1234`
2. Save the converted model to `/workspace/checkpoints/hf/step-1234`