import argparse
import tempfile

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
    
    checkpoint_path = '/home/ubuntu/nvme/checkpoints'
    seq_length = 2048
    micro_bs, global_bs = 2, 16
    tp, cp, pp = 2, 2, 2
    max_lr = 2e-5
    max_steps = 800000
    wandb_entity = 'llm_train_mafat'
    model_name = 'Qwen/Qwen3-8B-Base'

    pretrain = llm.qwen3_8b.pretrain_recipe(
        name="pwc_qwen3_8b_cpt",
        dir=checkpoint_path,
        num_nodes=1,
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
    # Checkpoint setting
    pretrain.trainer.val_check_interval = 32000

    # wandb logging
    pretrain.log = default_log(dir=checkpoint_path, name="pwc_qwen3_8b_cpt", wandb_logger=wandb_logger(project='pwc_qwen3_8b_cpt', name=args.run_name, entity=wandb_entity))

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
        paths=['100', './tok-data/hebdata_hewiki_text_document'], # can include lots of segments with different ratios
        tokenizer=tokenizer, 
        seq_length=seq_length, 
        micro_batch_size=micro_bs,
        global_batch_size=global_bs,
        reset_position_ids=True, # for packing
        reset_attention_mask=True, # for packing
        eod_mask_loss=True, # do we want to predict the eod?
        split="998,1,1" # train/val/test
    )

    pretrain.resume = nemo_resume(model_name)
    # uncomment the next line when you want to resume a run from a checkpoint in the checkpoint_dir
    # pretrain.resume = default_resume(resume_ignore_no_checkpoint=False)

    run.run(pretrain, executor=run.LocalExecutor())
