#!/usr/bin/env bash
set -x

# Tested with 2 & 4 GPUs

# nproc_per_node=$1
# save_path=$2
nproc_per_node=6
save_path=/local1/lxh/save/offline_grpo/7b_pi1_pmsft

# Timestamp (Pacific time) like 0813-1842
ts=$(TZ=America/Los_Angeles date +"%m%d-%H%M")
exp_name="7b_pi1_pmsft_${ts}"
log_file="7b_pi1_pmsft.log"

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
    -m verl.trainer.fsdp_pmsft_trainer \
    data.train_files=data/pi1/pi1_r128_pm_responses_16000.parquet \
    data.val_files=data/pi1/pi1_r128_pm_responses_16000_valid.parquet \
    data.max_length=1920 \
    data.prompt_key=prompt \
    data.response_key=response \
    data.train_batch_size=12 \
    data.micro_batch_size_per_gpu=2 \
    model.partial_pretrain=$HOME/models/Qwen2.5-Math-7B \
    trainer.default_local_dir=$save_path \
    trainer.project_name=dft \
    trainer.experiment_name=$exp_name \
    trainer.total_epochs=1 \
    trainer.test_freq=40 \
    trainer.save_freq=40 \
    trainer.total_training_steps=200 \
    trainer.logger=['console','wandb'] \
    trainer.default_hdfs_dir=null 2>&1 | tee "$log_file"