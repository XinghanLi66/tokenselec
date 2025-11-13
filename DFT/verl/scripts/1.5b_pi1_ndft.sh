#!/usr/bin/env bash
set -x

# Tested with 2 & 4 GPUs

# nproc_per_node=$1
# save_path=$2
nproc_per_node=8
save_path=/local1/lxh/save/offline_grpo/1.5b_pi1_ndft

# Timestamp (Pacific time) like 0813-1842
ts=$(TZ=America/Los_Angeles date +"%m%d-%H%M")
exp_name="1.5b_pi1_ndft_${ts}"
log_file="1.5b_pi1_ndft.log"

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
    -m verl.trainer.fsdp_ndft_trainer \
    data.train_files=data/pi1/pi1_r128_responses_16000.parquet \
    data.val_files=data/pi1/pi1_r128_responses_16000_valid.parquet \
    data.max_length=1920 \
    data.prompt_key=prompt \
    data.response_key=response \
    data.micro_batch_size_per_gpu=2 \
    model.partial_pretrain=$HOME/models/Qwen2.5-Math-1.5B \
    trainer.default_local_dir=$save_path \
    trainer.project_name=dft \
    trainer.experiment_name=$exp_name \
    trainer.total_epochs=2 \
    trainer.test_freq=5 \
    trainer.save_freq=10 \
    trainer.logger=['console','wandb'] \
    trainer.default_hdfs_dir=null 2>&1 | tee "$log_file"