"""Orchestration script generating datasets derived from `pi_merged.csv`.

python data-preparation/generate_diverse_data.py --total_samples=160 2>&1 | tee generate_diverse_data.log 

Behavior (exact sequence):
1. Single-source (pi1 = first row of merged CSV) temperature sweep at T = 0.2, 0.4, 0.6, 0.8, 1.0 (5 runs).
2. Combination runs at T = 1.0 using row prefixes of merged CSV:
    - pi1+pi2 (first 2 rows)
    - pi1+pi2+pi13 (first 3 rows)
    - pi1+pi2+pi13+pi1209 (first 4 rows)
Total: 8 runs.

Assumptions:
- The merged CSV has columns: [data_source, prompt, ability, reward_model, extra_info]
- generate_all_responses.py accepts CSV input and reads 'prompt' and 'reward_model'

You can customize parameters via CLI flags.

Example usage (inside repo root):
        # Single GPU (debug dry-run)
        python data-preparation/generate_diverse_data.py --dry_run

        # Multi-GPU with Accelerate
        python data-preparation/generate_diverse_data.py \
                --model_path /cephfs/lxh/models/qwen2.5-math-1.5b \
                --base_output_dir data \
                --total_samples 16000 \
                --launcher accelerate --num_processes 4

        # Multi-GPU with torchrun
        python data-preparation/generate_diverse_data.py \
                --model_path /cephfs/lxh/models/qwen2.5-math-1.5b \
                --base_output_dir data \
                --total_samples 16000 \
                --launcher torchrun --num_processes 4

After completion you'll have directories like:
    data/merged_temp_0_2/, data/merged_temp_0_4/, ...

Requires: pandas.
Heavy generation dependencies (accelerate, transformers, etc.) must be installed for generation to run.

Multi-GPU note:
`generate_all_responses.py` internally uses `Accelerator`; to actually spawn
multiple processes you must launch it via `accelerate launch` or `torchrun`.
This orchestration script supports a `--launcher` flag so each internal call uses multi-GPU when requested.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import random
from pathlib import Path
try:
    import torch
except Exception:
    torch = None  # allow dry-run without torch installed
import pandas as pd


SINGLE_TEMPS = [0.2, 0.4, 0.6, 0.8, 1.0]
COMBO_STAGES = [2, 3, 4]  # number of rows to include for combination runs (always at T=1.0)
MERGED_CSV = Path("data-preparation/input-data") / "pi_merged.csv"


def build_launch_prefix(args):
    """Return list of command components that launches a Python script with the
    chosen launcher. Empty list means direct python execution.
    """
    if args.launcher == "accelerate":
        parts = ["accelerate", "launch"]
        if args.num_processes:
            parts += ["--num_processes", str(args.num_processes)]
        return parts
    elif args.launcher == "torchrun":
        cuda_count = 0
        if args.num_processes:
            cuda_count = args.num_processes
        elif torch is not None:
            try:
                cuda_count = torch.cuda.device_count()
            except Exception:
                cuda_count = 0
        num = cuda_count or 1
        return ["torchrun", "--standalone", f"--nproc_per_node={num}"]
    else:
        return [sys.executable]


def write_subset(df: pd.DataFrame, n_rows: int, dest: Path) -> Path:
    """Write first n_rows of df to a temp CSV and return the path."""
    subset = df.iloc[:n_rows].copy()
    dest.parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(dest, index=False)
    return dest


def run_generation(input_csv: Path, temperature: float, out_dir: Path, args) -> None:
    prefix = build_launch_prefix(args)
    script = ["data-preparation/generate_all_responses.py"]
    cmd = prefix + script + [
        "--model_path", args.model_path,
        "--input_data", str(input_csv),
        "--output_dir", str(out_dir),
        "--total_samples", str(args.total_samples),
        "--batch_size", str(args.batch_size),
        "--temperature", str(temperature),
        "--precision", args.precision,
    ]
    print(f"[run] {' '.join(cmd)}")
    if not args.dry_run:
        subprocess.run(cmd, check=True)
    else:
        print("[dry-run] skipped generation")

    print(f"[run] python data-preparation/merge_all_csvs.py {out_dir}")
    if not args.dry_run:
        subprocess.run(["python", "data-preparation/merge_all_csvs.py", str(out_dir)], check=True)
    else:
        print("[dry-run] skipped merging")


def get_run_files(dataset_dir: Path):
    """Return paths to correct/incorrect CSV files for a run."""
    return dataset_dir / "correct.csv", dataset_dir / "incorrect.csv"


def load_responses(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Expected CSV not found: {path}")
    df = pd.read_csv(path)
    # Standardize column names for downstream merging
    expected = {"prompt", "response"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"File {path} missing columns: {missing}")
    return df


def merge_and_write(dfs: list[pd.DataFrame], output_dir: Path, filename_stem: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    merged = pd.concat(dfs, ignore_index=True)
    merged = merged.sample(frac=1.0, random_state=42).reset_index(drop=True)
    merged.to_parquet(output_dir / f"{filename_stem}.parquet", index=False)
    merged.to_csv(output_dir / f"{filename_stem}.csv", index=False)
    print(f"[combine] wrote {len(merged)} rows -> {output_dir / (filename_stem + '.{parquet,csv}')}")
    return merged


def orchestrate_runs(args):
    if not MERGED_CSV.exists():
        raise FileNotFoundError(f"Missing merged CSV: {MERGED_CSV}")
    full_df = pd.read_csv(MERGED_CSV)
    if full_df.empty:
        raise ValueError("Merged CSV is empty")
    # Basic sanity: expect at least 4 rows for all combos
    if len(full_df) < 4:
        raise ValueError("Merged CSV must contain at least 4 rows for pi1, pi2, pi13, pi1209.")

    subset_root = Path("data-preparation/input-data/subsets")
    runs = []

    # Phase 1: single-source pi1 temperature sweep
    for temp in SINGLE_TEMPS:
        subset_path = write_subset(full_df, 1, subset_root / "pi1.csv")
        out_dir = Path(args.base_output_dir) / f"pi1_temp_{str(temp).replace('.', '_')}"
        out_dir.mkdir(parents=True, exist_ok=True)
        run_generation(subset_path, temp, out_dir, args)
        runs.append(("pi1", temp, out_dir))

    # Phase 2: combinations at T=1.0
    # for n in COMBO_STAGES:
    #     subset_name = {2: "pi1_pi2", 3: "pi1_pi2_pi13", 4: "pi1_pi2_pi13_pi1209"}[n]
    #     subset_path = write_subset(full_df, n, subset_root / f"{subset_name}.csv")
    #     temp = 1.0
    #     out_dir = Path(args.base_output_dir) / f"{subset_name}_temp_{str(temp).replace('.', '_')}"
    #     out_dir.mkdir(parents=True, exist_ok=True)
    #     run_generation(subset_path, temp, out_dir, args)
    #     runs.append((subset_name, temp, out_dir))

    print(f"[summary] Completed {len(runs)} runs.")
    return runs


## Combinations phase removed: the merged CSV already contains all prompts.


def parse_args():
    p = argparse.ArgumentParser(description="Generate 8 datasets: 5 temperature runs of pi1 + 3 combination runs at T=1.0.")
    p.add_argument("--model_path", type=str, default="/cephfs/shared/Qwen2.5-Math-1.5B")
    p.add_argument("--base_output_dir", type=str, default="data")
    p.add_argument("--total_samples", type=int, default=16000, help="Total samples to generate per run.")
    p.add_argument("--batch_size", type=int, default=1024)
    p.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--launcher", type=str, default="accelerate", choices=["none", "accelerate", "torchrun"],
                   help="Launcher to use for spawning multi-GPU processes for each generation run.")
    p.add_argument("--num_processes", type=int, default=None,
                   help="Number of processes (GPUs) to use when --launcher is accelerate/torchrun. Defaults to torch.cuda.device_count().")
    p.add_argument("--dry_run", action="store_true", help="Print subprocess commands without executing generation.")
    return p.parse_args()


def main():
    args = parse_args()
    Path(args.base_output_dir).mkdir(parents=True, exist_ok=True)

    if args.launcher != "none" and args.num_processes is None:
        # Best-effort default: detect available GPUs
        detected = 0
        if torch is not None:
            try:
                detected = torch.cuda.device_count()
            except Exception:
                detected = 0
        if detected > 1:
            print(f"[auto] num_processes not set; using detected GPU count: {detected}")
            args.num_processes = detected
        else:
            print("[auto] single GPU detected or torch not available; proceeding with 1 process.")

    runs = orchestrate_runs(args)
    print("\nAll done. You can inspect datasets under:", args.base_output_dir)


if __name__ == "__main__":
    main()
