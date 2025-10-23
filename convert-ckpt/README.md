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

### 2. Place Your Checkpoint

Download your unpacked NeMo checkpoint directory into the local `./checkpoints/nemo/` folder. The script assumes the following directory structure on your host machine:

```
/home/ubuntu/qwen-hebrew-finetuning/convert-ckpt/
├── checkpoints/
│   ├── nemo/
│   │   └── <nemo-checkpoint-name>/  <--- YOUR CHECKPOINT GOES HERE
│   │       ├── context/
│   │       └── weights/
│   └── hf/
│
└── convert.py
```

### 3. Run the Container

Start an interactive shell inside the Docker container. This command mounts your project directory (`/home/ubuntu/qwen-hebrew-finetuning/convert-ckpt`) into the `/workspace` directory inside the container.

```bash
sudo docker run --gpus all --rm -it \
  --shm-size=16g \
  -v /home/ubuntu/qwen-hebrew-finetuning/convert-ckpt:/workspace \
  sudo docker pull nvcr.io/nvidia/nemo:25.07.nemotron-nano-v2 \
  bash
```

### 4. Run the Conversion

Once you are inside the container's shell, run the `convert.py` script. You must specify the name of your NeMo checkpoint folder and the desired name for the new Hugging Face output folder.

```bash
python convert.py --nemo_model <nemo-checkpoint-name> --hf_model <hf-checkpoint-name>
```

**Example:**

If your checkpoint is located at `/workspace/checkpoints/nemo/nemo-ckpt-v1`, you would run:

```bash
python convert.py --nemo_model nemo-ckpt-v1 --hf_model my-hf-model
```

The script will:
1. Read from `/workspace/checkpoints/nemo/nemo-ckpt-v1`
2. Save the converted model to `/workspace/checkpoints/hf/my-hf-model`