import argparse
import tempfile
import os

def slurm_executor(nodes: int = 1, container_image: str = 'dockerd://nvcr.io/nvidia/nemo:25.07.nemotron-nano-v2'):
    import nemo_run as run
    local_tunnel = run.LocalTunnel(job_dir=os.path.join(os.environ['NEMORUN_HOME'], "experiments"))

    return run.SlurmExecutor(
        # Most of these parameters are specific to slurm
        account="pwc",
        partition="dev",
        ntasks_per_node=8,
        gpus_per_node=8,
        nodes=nodes,
        tunnel=local_tunnel,
        container_image=container_image,
        time="0", # unlimited
        mem="0",
        env_vars=dict(
            FI_PROVIDER="efa",
            FI_EFA_USE_DEVICE_RDMA="1",
            WANDB_API_KEY=os.environ['WANDB_API_KEY'],
            NCCL_SOCKET_IFNAME='ens,enp',
            LD_LIBRARY_PATH="/opt/amazon/efa/lib:/opt/amazon/aws-ofi-nccl/lib:" + os.environ.get("LD_LIBRARY_PATH",""),
            LD_PRELOAD="/opt/amazon/efa/lib/libfabric.so.1",
            ENROOT_MOUNTS="tmpfs:/dev/shm:rw,size=64G,x-create=dir"
        ),
        container_env=[
            'LD_LIBRARY_PATH',
            'LD_PRELOAD'
        ],
        container_mounts=[
            '/fsx:/fsx',
            '/opt/slurm:/opt/slurm:ro',
            '/var/run/munge:/var/run/munge:rw',
            '/var/log/aws:/var/log/aws',
            '/usr/local/share/pyxis:/usr/local/share/pyxis:ro',
            '/usr/local/lib/slurm:/usr/local/lib/slurm:ro',
            '/opt/amazon/efa:/opt/amazon/efa:ro',
            '/opt/amazon/ofi-nccl:/opt/amazon/ofi-nccl:ro',
            "/dev/infiniband:/dev/infiniband",
            f'{os.environ['WORKSPACE_DIR']}:/workspace',
        ],
        packager=run.Packager(),
    )


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Script to train a Qwen3 model.")
    parser.add_argument("--checkpoints_path", required=True, type=str, help="Path to store checkpoints")
    parser.add_argument("--run_name", required=True, type=str, help="Name of the run for the WandB")
    parser.add_argument("--use_fp8", action='store_true', help="Use the FP8 precision module")
    parser.add_argument("--num_nodes", type=int, required=False, default=0, help="Number of nodes to train on - set to 0 (default) for local executor")
    parser.add_argument("--model", type=str, required=False, default='Qwen3_8B', choices=['Qwen3_8B', 'Qwen3_30B_A3B_Base'], help="Which model to train - assumes you already imported the base model from huggingface")

    args = parser.parse_args()

    # imports take time, so prefer to import them after arg parse
    import nemo_run as run
    from nemo.collections import llm
    from nemo.collections.llm.recipes.log.default import default_log, wandb_logger, default_resume
    from nemo.collections.llm.gpt.data.pre_training import PreTrainingDataModule
    from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
    from nemo.collections.llm.recipes.precision.mixed_precision import bf16_with_fp8_mixed, bf16_mixed
    from nemo.collections.llm.recipes.finetune_default import nemo_resume
    from nemo.collections.llm.gpt.model.qwen3 import Qwen3Model, Qwen3Config30B_A3B
    
    checkpoint_path = args.checkpoints_path
    seq_length = 4096
    global_bs = 256
    max_lr = 2e-5
    max_steps = 6500
    wandb_entity = 'llm_train_mafat'

    if args.model == 'Qwen3_8B':
        model = llm.qwen3_8b
        tp, cp, pp = 2, 1, 1
        micro_bs = 4
        model_name = 'Qwen/Qwen3-8B' # NOT BASE - BE AWARE
    elif args.model == 'Qwen3_30B_A3B_Base':
        model = llm.qwen3_30b_a3b
        tp, cp, pp = 2, 1, 2
        micro_bs = 4
        model_name = 'Qwen/Qwen3-30B-A3B-Base'
    else:
        raise ValueError(f"Unknown model {args.model}")
    
    pretrain = model.pretrain_recipe(
        name="pwc_nemo_cpt",
        dir=checkpoint_path,
        num_nodes=max([args.num_nodes, 1]),
        num_gpus_per_node=8,
        log_every_n_steps=1,
        tensor_parallelism=tp,
        pipeline_parallelism=pp,
        context_parallelism=cp,
        seq_length=seq_length,
        micro_batch_size=micro_bs,
        global_batch_size=global_bs,
        max_steps=max_steps,
        max_lr=max_lr,
        min_lr=max_lr / 10
    )
    # wandb logging
    pretrain.log = default_log(dir=checkpoint_path, name="pwc_nemo_cpt", wandb_logger=wandb_logger(project='pwc_nemo_cpt', name=args.run_name, entity=wandb_entity))
    # Train in fp8 if relevant, saves memory
    pretrain.trainer.plugins = bf16_mixed() if not args.use_fp8 else bf16_with_fp8_mixed()

    # alternative way to set things:
    # pretrain.optim.lr_scheduler.min_lr = lr / 10
    # pretrain.optim.config.lr = lr
    # pretrain.model.config.rotary_base = lc_new_theta
    # pretrain.trainer.max_steps = max_steps
    # pretrain.trainer.strategy.context_parallel_size = cp
    
    # Set up the data information
    tokenizer = run.Config(get_nmt_tokenizer,
        library='huggingface',
        model_name=model_name
    )
    pretrain.data = run.Config(PreTrainingDataModule, 
        # paths=['100', '/workspace/tok-data/hebdata_hewiki_text_document'], # can include lots of segments with different ratios
        paths=['100', '/workspace/tok-data/low_text_document'], # can include lots of segments with different ratios
        tokenizer=tokenizer, 
        seq_length=seq_length, 
        micro_batch_size=micro_bs,
        global_batch_size=global_bs,
        num_workers=8,
        reset_position_ids=True, # for packing
        reset_attention_mask=True, # for packing
        eod_mask_loss=True, # do we want to predict the eod?
        split="998,1,1" # train/val/test
    )

    pretrain.resume = nemo_resume(model_name)
    pretrain.resume.resume_if_exists = True
    
    if not args.num_nodes:
        run.run(pretrain, executor=run.LocalExecutor())
    else:
        run.run(pretrain, executor=slurm_executor(args.num_nodes))
