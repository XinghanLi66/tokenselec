#!/bin/bash
# set -e

source $(conda info --base)/etc/profile.d/conda.sh
conda activate DFT

CUDA_VISIBLE_DEVICES=1,2,3,5,6,7 bash scripts/7b_pi1_pmsft.sh
CUDA_VISIBLE_DEVICES=1,2,3,5,6,7 bash scripts/7b_pi1_ndft.sh
CUDA_VISIBLE_DEVICES=1,2,3,5,6,7 bash scripts/7b_pi1_sft.sh

conda activate eval
cd /homes/gws/lxh22/rl-sft/one-shot-em/Qwen2.5-Eval/evaluation

CUDA_VISIBLE_DEVICES=3,5,6,7 bash sh/run_eval_neo.sh -e "7b_pi1_ofrl" -s "160 120 80 40 200" -v "temp00"
CUDA_VISIBLE_DEVICES=3,5,6,7 bash sh/run_eval_neo.sh -e "7b_pi1_pmsft" -s "160 120 80 40 200" -v "temp00"
CUDA_VISIBLE_DEVICES=3,5,6,7 bash sh/run_eval_neo.sh -e "7b_pi1_pmsft" -s "160 120 80 40 200" -v "temp06"
CUDA_VISIBLE_DEVICES=3,5,6,7 bash sh/run_eval_neo.sh -e "7b_pi1_ndft" -s "160 120 80 40 200" -v "temp00"
CUDA_VISIBLE_DEVICES=3,5,6,7 bash sh/run_eval_neo.sh -e "7b_pi1_ndft" -s "160 120 80 40 200" -v "temp06"
CUDA_VISIBLE_DEVICES=3,5,6,7 bash sh/run_eval_neo.sh -e "7b_pi1_sft" -s "160 120 80 40 200" -v "temp00"
CUDA_VISIBLE_DEVICES=3,5,6,7 bash sh/run_eval_neo.sh -e "7b_pi1_sft" -s "160 120 80 40 200" -v "temp06"

