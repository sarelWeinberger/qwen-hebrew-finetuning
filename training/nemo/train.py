import argparse
import tempfile
import os

def slurm_executor(nodes: int = 1, container_image: str = 'dockerd://nvcr.io/nvidia/nemo:25.07.nemotron-nano-v2'):
    import nemo_run as run

    # SSH Tunnel
    # ssh_tunnel = run.SSHTunnel(
    #     host="your-slurm-host",
    #     user="your-user",
    #     job_dir="directory-to-store-runs-on-the-slurm-cluster",
    #     identity="optional-path-to-your-key-for-auth",
    # )
    # Local Tunnel to use if you're already on the cluster
    local_tunnel = run.LocalTunnel(job_dir=os.path.join(os.environ['NEMORUN_HOME'], "experiments"))

    # packager = GitArchivePackager(
    #     # This will also be the working directory in your task.
    #     # If empty, the working directory will be toplevel of your git repo
    #     subpath="optional-subpath-from-toplevel-of-your-git-repo"
    # )

    return run.SlurmExecutor(
        # Most of these parameters are specific to slurm
        account="your-account",
        partition="ml.g6e.48xlarge",
        ntasks_per_node=8,
        gpus_per_node=8,
        nodes=nodes,
        tunnel=local_tunnel,
        container_image=container_image,
        time="00:30:00",
        env_vars=dict(
            NCCL_SOCKET_IFNAME='enp137s0',
            NCCL_DEBUG='info',
            WANDB_API_KEY=os.environ['WANDB_API_KEY']
        ),
        container_mounts=[
            '/fsx:/fsx',
            '/opt/slurm:/opt/slurm:ro',
            '/var/run/munge:/var/run/munge:rw',
            '/var/log/aws:/var/log/aws',
            f'{os.environ['WORKSPACE_DIR']}:/workspace',
        ],
        packager=run.Packager(),
    )


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Script to train a Qwen3 model.")
    parser.add_argument("--run_name", required=True, type=str, help="Name of the run for the WandB")
    parser.add_argument("--use_fp8", action='store_true', help="Use the FP8 precision module")

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
    
    checkpoint_path = '/fsx/results/checkpoints'
    seq_length = 4096
    micro_bs, global_bs = 4, 256
    tp, cp, pp, ep = 2, 1, 2, 4
    max_lr = 2e-5
    max_steps = 2000
    wandb_entity = 'llm_train_mafat'
    model_name = 'Qwen/Qwen3-30B-A3B-Base'

    pretrain = llm.qwen3_1p7b.pretrain_recipe(
        name="pwc_qwen3_30b_cpt",
        dir=checkpoint_path,
        num_nodes=2,
        num_gpus_per_node=8,
        log_every_n_steps=1,
        tensor_parallelism=2,
        # pipeline_parallelism=pp,
        # expert_parallelism=ep,
        # context_parallelism=cp,
        # seq_length=seq_length,
        # micro_batch_size=micro_bs,
        # global_batch_size=global_bs,
        max_steps=max_steps,
        max_lr=max_lr,
        min_lr=max_lr / 10
    )
    # wandb logging
    pretrain.log = default_log(dir=checkpoint_path, name="pwc_qwen3_30b_cpt", wandb_logger=wandb_logger(project='pwc_qwen3_30b_cpt', name=args.run_name, entity=wandb_entity))

    # alternative way to set things:
    # pretrain.optim.lr_scheduler.min_lr = lr / 10
    # pretrain.optim.config.lr = lr
    # pretrain.model.config.rotary_base = lc_new_theta
    # pretrain.trainer.max_steps = max_steps
    # pretrain.trainer.strategy.context_parallel_size = cp
    
    # Train in fp8 if relevant, saves memory
    pretrain.trainer.plugins = bf16_mixed() if not args.use_fp8 else bf16_with_fp8_mixed()

    tokenizer = run.Config(get_nmt_tokenizer,
        library='huggingface',
        model_name=model_name
    )

    # Custom data
    pretrain.data = run.Config(PreTrainingDataModule, 
        paths=['100', '/workspace/tok-data/hebdata_hewiki_text_document'], # can include lots of segments with different ratios
        tokenizer=tokenizer, 
        seq_length=seq_length, 
        micro_batch_size=micro_bs,
        global_batch_size=global_bs,
        reset_position_ids=True, # for packing
        reset_attention_mask=True, # for packing
        eod_mask_loss=True, # do we want to predict the eod?
        split="998,1,1" # train/val/test
    )

    # pretrain.resume = nemo_resume(model_name)
    # uncomment the next line when you want to resume a run from a checkpoint in the checkpoint_dir
    # pretrain.resume = default_resume(resume_ignore_no_checkpoint=False)

    run.run(pretrain, executor=slurm_executor(2))
