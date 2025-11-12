"""Orchestration script to generate increasingly diverse training data.

Phases:
1. Single-source (pi1) generations across temperature sweep: 0.2,0.4,0.6,0.8,1.0.
2. Combination datasets at temperature=1.0 formed by union of individual
   generations: (pi1+pi2), (pi1+pi2+pi13), (pi1+pi2+pi13+pi1209).

We DO NOT modify `prepare_data.py`; instead we call it repeatedly, then
merge its outputs. Each call to `prepare_data.py` produces a *train*
and *valid* parquet/csv pair; we ensure final merged outputs have
equal train/valid sizes (50/50 split) by re-shuffling and re-splitting.

Assumptions:
- Each input parquet lives under `data-preparation/input-data/{policy}_r128.parquet`.
- `prepare_data.py` reads only the first row; diversity across policies
  therefore comes from generating separate response sets per policy.
- For equal sized splits we set n_train = n_samples // 2 in each run.
- Enough valid samples will be filtered to satisfy n_train; if not,
  we raise with a helpful message.

You can customize parameters via CLI flags.

Example usage (inside repo root):
    python data-preparation/generate_diverse_data.py \
        --model_path /homes/gws/lxh22/models/Qwen2.5-Math-1.5B \
        --base_output_dir generated-data \
        --n_samples 8000 --n_train 4000

After completion you'll have directories:
  generated-data/pi1/temp_0.2/ ... etc
  generated-data/combined/pi1_pi2/ ... etc

Requires: pandas.
Heavy generation dependencies (accelerate, transformers, etc.) must be
installed for prepare_data.py to run.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import random
from pathlib import Path
import pandas as pd


TEMPERATURES = [0.2, 0.4, 0.6, 0.8, 1.0]
POLICIES_SINGLE = ["pi1"]
COMBINATIONS = [
    ["pi1", "pi2"],
    ["pi1", "pi2", "pi13"],
    ["pi1", "pi2", "pi13", "pi1209"],
]


def run_prepare(policy: str, temperature: float, args) -> Path:
    """Invoke prepare_data.py for one policy & temperature.

    Returns path to the dataset directory created under base_output_dir.
    """
    input_parquet = Path("data-preparation/input-data") / f"{policy}_r128.parquet"
    if not input_parquet.exists():
        raise FileNotFoundError(f"Missing input parquet: {input_parquet}")

    # Dataset directory name: e.g., data/pi1_temp_0_2
    dataset_name = f"{policy}_temp_{str(temperature).replace('.', '_')}"
    dataset_dir = Path(args.base_output_dir) / dataset_name
    dataset_dir.mkdir(parents=True, exist_ok=True)

    cmd = [
        sys.executable,
        "data-preparation/prepare_data.py",
        "--model_path", args.model_path,
        "--input_data", str(input_parquet),
        "--output_dir", str(dataset_dir),
        "--n_samples", str(args.n_samples),
        "--n_train", str(args.n_train),
        "--batch_size", str(args.batch_size),
        "--temperature", str(temperature),
        "--precision", args.precision,
        "--split", "none",
    ]

    print(f"[run] {' '.join(cmd)}")
    if not args.dry_run:
        subprocess.run(cmd, check=True)
    else:
        print("[dry-run] skipped generation")

    return dataset_dir


def get_run_files(dataset_dir: Path):
    """Return paths to correct/incorrect parquet files for a run."""
    return dataset_dir / "correct.parquet", dataset_dir / "incorrect.parquet"


def load_responses(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Expected parquet not found: {path}")
    df = pd.read_parquet(path)
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


def generate_single_policy_temperatures(args):
    results = []
    for policy in POLICIES_SINGLE:
        for temp in TEMPERATURES:
            dataset_dir = run_prepare(policy, temp, args)
            c_pq, i_pq = get_run_files(dataset_dir)
            results.append((policy, temp, dataset_dir, c_pq, i_pq))
    return results


def generate_combinations(args):
    """For each combination at temperature=1.0, merge correct and incorrect sets into a single dataset directory under data/."""
    combo_outputs = []
    for combo in COMBINATIONS:
        print(f"[combo] building combination: {'+'.join(combo)}")
        correct_dfs = []
        incorrect_dfs = []
        for policy in combo:
            dataset_name = f"{policy}_temp_{str(1.0).replace('.', '_')}"
            dataset_dir = Path(args.base_output_dir) / dataset_name
            c_pq, i_pq = dataset_dir / "correct.parquet", dataset_dir / "incorrect.parquet"
            if not c_pq.exists() or not i_pq.exists():
                print(f"[combo] missing run for {policy} temp=1.0; generating now.")
                dataset_dir = run_prepare(policy, 1.0, args)
                c_pq, i_pq = get_run_files(dataset_dir)
            correct_dfs.append(load_responses(c_pq))
            incorrect_dfs.append(load_responses(i_pq))

        combo_name = "_".join(combo)
        out_dir = Path(args.base_output_dir) / combo_name
        merge_and_write(correct_dfs, out_dir, filename_stem="correct")
        merge_and_write(incorrect_dfs, out_dir, filename_stem="incorrect")
        combo_outputs.append(str(out_dir))
    return combo_outputs


def parse_args():
    p = argparse.ArgumentParser(description="Generate diverse datasets via prepare_data.py orchestration.")
    p.add_argument("--model_path", type=str, default="/homes/gws/lxh22/models/Qwen2.5-Math-1.5B")
    p.add_argument("--base_output_dir", type=str, default="data")
    p.add_argument("--n_samples", type=int, default=8000, help="Samples to generate per run (pre-filter).")
    p.add_argument("--n_train", type=int, default=4000, help="Train samples to keep; valid will be the remainder from prepare_data then rebalanced later for combos.")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--dry_run", action="store_true", help="Print subprocess commands without executing generation.")
    return p.parse_args()


def main():
    args = parse_args()
    Path(args.base_output_dir).mkdir(parents=True, exist_ok=True)

    print("=== Phase 1: pi1 temperature sweep (no train/valid split) ===")
    single_runs = generate_single_policy_temperatures(args)
    for policy, temp, dataset_dir, c_pq, i_pq in single_runs:
        print(f"[done] {policy} temp={temp} -> {dataset_dir} (correct/incorrect)")

    print("\n=== Phase 2: Combinations at temp=1.0 (merged correct/incorrect) ===")
    combos = generate_combinations(args)
    print(f"[complete] combinations generated: {combos}")

    print("\nAll done. You can inspect datasets under:", args.base_output_dir)


if __name__ == "__main__":
    main()
