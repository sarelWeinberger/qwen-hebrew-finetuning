if command -v nvidia-smi &>/dev/null && nvidia-smi -L &>/dev/null; then
  GPUS="--gpus all"
else
  GPUS=""
fi

docker run \
  $GPUS \
  -e HF_HOME=/workspace/.cache/huggingface/ \
  -e NEMO_HOME=/workspace/.cache/nemo/ \
  -e SLURM_CONF=/opt/slurm/etc/slurm.conf \
  -e NEMORUN_HOME=/fsx/.nemo_run \
  -e WORKSPACE_DIR=$(pwd) \
  -e NCCL_SOCKET_IFNAME=ens,enp \
  -it --shm-size=10gb --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  --network host \
  -v /fsx:/fsx \
  -v /opt/amazon/efa:/opt/amazon/efa:ro \
  -v /opt/amazon/ofi-nccl:/opt/amazon/ofi-nccl:ro \
  -v /opt/slurm:/opt/slurm:ro \
  -v /var/run/munge:/var/run/munge:rw \
  -v /var/log/aws:/var/log/aws \
  -v /usr/local/share/pyxis:/usr/local/share/pyxis:ro \
  -v /usr/local/lib/slurm:/usr/local/lib/slurm:ro \
  -v $(pwd):/workspace \
  -w /workspace \
  nvcr.io/nvidia/nemo:25.07.nemotron-nano-v2