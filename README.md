# tokenselec

## Multi-GPU Response Generation with 🤗 Accelerate

The script `data-preparation/generate_all_responses.py` already uses `Accelerator` internally, but you MUST launch it with `accelerate launch` (or `torchrun`) for multi-GPU sampling to be enabled. Running `python generate_all_responses.py` will only use a single process/GPU.

### 1. Configure Accelerate (one-time)
Run the interactive config if you haven't yet:

```bash
accelerate config
```

Recommended answers for pure multi-GPU single-node inference:
- Compute environment: "LOCAL_MACHINE"
- Machine rank: 0
- Number of processes: (e.g. 4 for 4 GPUs)
- Distributed training backend: "nccl"
- Mixed precision: bf16 (if your GPUs support it; else fp16)
- DeepSpeed / FSDP: no (not needed for inference sampling)

This generates a config file (usually under `~/.cache/huggingface/accelerate/`).

### 2. Launch the script

Minimal example (adjust paths/GPUs):

```bash
accelerate launch \
	--num_processes 4 \
	data-preparation/generate_all_responses.py \
	--model_path /homes/gws/lxh22/models/Qwen2.5-Math-1.5B \
	--input_data data/pi1_temp_0_2/your_dataset.parquet \
	--output_dir outputs/pi1_gen_multi_gpu \
	--total_samples 16000 \
	--batch_size 512 \
	--precision bf16
```

Notes:
- `--num_processes` should match the number of visible GPUs (`CUDA_VISIBLE_DEVICES`).
- The script automatically splits the requested `--total_samples` across processes per prompt.
- Only the main process writes output files.
- For large generations, consider lowering `--batch_size` if you hit OOM.

### 3. Environment Tips

Set NCCL env vars (optional but helps on some clusters):
```bash
export NCCL_P2P_DISABLE=0
export NCCL_IB_DISABLE=1
export CUDA_VISIBLE_DEVICES=0,1,2,3
```

### 4. Alternative: torchrun
You can also use PyTorch's launcher:
```bash
torchrun --standalone --nproc_per_node=4 data-preparation/generate_all_responses.py \
	--model_path /homes/gws/lxh22/models/Qwen2.5-Math-1.5B \
	--input_data data/pi1_temp_0_2/your_dataset.parquet \
	--output_dir outputs/pi1_gen_multi_gpu
```

### 5. Output Artifacts
The script saves: `correct.parquet/csv`, `incorrect.parquet/csv`, and full aggregates `all.parquet/csv` under `--output_dir`.

### 6. Reproducibility
The script seeds each process differently (offset by process index) for sample diversity. Use `--seed` to control the base seed.

### 7. Troubleshooting
- If you see only one GPU utilized: confirm you used `accelerate launch` and that `CUDA_VISIBLE_DEVICES` lists multiple GPUs.
- Shape mismatch errors during gather: ensure all processes run the same code path; do not modify logic inside loops conditionally per rank.
- OOM: reduce `--batch_size` or `--max_new_tokens`.

---
Feel free to add more dataset or model docs below.