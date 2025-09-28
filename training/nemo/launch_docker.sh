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
  nvcr.io/nvidia/nemo:25.07.nemotron-nano-v2
