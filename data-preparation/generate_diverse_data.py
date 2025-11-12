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

    Returns path to the *train* parquet produced.
    """
    input_parquet = Path("data-preparation/input-data") / f"{policy}_r128.parquet"
    if not input_parquet.exists():
        raise FileNotFoundError(f"Missing input parquet: {input_parquet}")

    out_dir = Path(args.base_output_dir) / policy / f"temp_{temperature}" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    base_name = f"{policy}_temp{temperature}".replace(".", "_")
    output_parquet = out_dir / f"{base_name}.parquet"
    output_csv = out_dir / f"{base_name}.csv"

    cmd = [
        sys.executable,
        "data-preparation/prepare_data.py",
        "--model_path", args.model_path,
        "--input_data", str(input_parquet),
        "--output_parquet", str(output_parquet),
        "--output_csv", str(output_csv),
        "--n_samples", str(args.n_samples),
        "--n_train", str(args.n_train),
        "--batch_size", str(args.batch_size),
        "--temperature", str(temperature),
        "--precision", args.precision,
    ]

    print(f"[run] {' '.join(cmd)}")
    if not args.dry_run:
        subprocess.run(cmd, check=True)
    else:
        print("[dry-run] skipped generation")

    train_parquet = output_parquet  # valid file will have suffix _valid.parquet
    return train_parquet


def collect_split_paths(policy: str, temperature: float, args):
    base_name = f"{policy}_temp{temperature}".replace(".", "_")
    out_dir = Path(args.base_output_dir) / policy / f"temp_{temperature}" / "raw"
    train = out_dir / f"{base_name}.parquet"
    valid = out_dir / f"{base_name}_valid.parquet"
    return train, valid


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


def merge_and_equal_split(dfs: list[pd.DataFrame], output_dir: Path, prefix: str):
    output_dir.mkdir(parents=True, exist_ok=True)
    merged = pd.concat(dfs, ignore_index=True)
    merged = merged.sample(frac=1.0, random_state=42).reset_index(drop=True)
    n_total = len(merged)
    n_half = n_total // 2
    train = merged.iloc[:n_half].copy()
    valid = merged.iloc[n_half: n_half * 2].copy()
    print(f"[combine] total={n_total} train={len(train)} valid={len(valid)}")
    train.to_parquet(output_dir / f"{prefix}_train.parquet", index=False)
    valid.to_parquet(output_dir / f"{prefix}_valid.parquet", index=False)
    train.to_csv(output_dir / f"{prefix}_train.csv", index=False)
    valid.to_csv(output_dir / f"{prefix}_valid.csv", index=False)
    return train, valid


def equalize_and_save_single(policy: str, temperature: float, args):
    """Downsample train/valid to equal sizes and save under a 'final' folder."""
    train_p, valid_p = collect_split_paths(policy, temperature, args)
    train_df = load_responses(train_p)
    valid_df = load_responses(valid_p)
    equal_n = min(len(train_df), len(valid_df), args.n_train)
    if equal_n == 0:
        raise ValueError(f"No data to equalize for {policy} temp={temperature}.")
    train_eq = train_df.sample(n=equal_n, random_state=42).reset_index(drop=True)
    valid_eq = valid_df.sample(n=equal_n, random_state=43).reset_index(drop=True)
    out_dir = Path(args.base_output_dir) / policy / f"temp_{temperature}" / "final"
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = f"{policy}_temp{temperature}".replace(".", "_")
    train_eq.to_parquet(out_dir / f"{prefix}_train.parquet", index=False)
    valid_eq.to_parquet(out_dir / f"{prefix}_valid.parquet", index=False)
    train_eq.to_csv(out_dir / f"{prefix}_train.csv", index=False)
    valid_eq.to_csv(out_dir / f"{prefix}_valid.csv", index=False)
    return out_dir


def generate_single_policy_temperatures(args):
    results = []
    for policy in POLICIES_SINGLE:
        for temp in TEMPERATURES:
            run_prepare(policy, temp, args)
            final_dir = equalize_and_save_single(policy, temp, args)
            train_p, valid_p = collect_split_paths(policy, temp, args)
            results.append((policy, temp, train_p, valid_p, final_dir))
    return results


def generate_combinations(args):
    """For each combination run per-policy generation at temperature=1.0 if not already done, then merge outputs."""
    combo_outputs = []
    for combo in COMBINATIONS:
        print(f"[combo] building combination: {'+'.join(combo)}")
        per_policy_train_dfs = []
        per_policy_valid_dfs = []
        for policy in combo:
            # Ensure single-policy temp=1.0 run exists (generate if missing)
            train_p, valid_p = collect_split_paths(policy, 1.0, args)
            if not train_p.exists() or not valid_p.exists():
                print(f"[combo] missing generation for {policy} temp=1.0; generating now.")
                run_prepare(policy, 1.0, args)
            per_policy_train_dfs.append(load_responses(train_p))
            per_policy_valid_dfs.append(load_responses(valid_p))

        merged_train_valid = per_policy_train_dfs + per_policy_valid_dfs
        combo_name = "_".join(combo)
        out_dir = Path(args.base_output_dir) / "combined" / combo_name
        merge_and_equal_split(merged_train_valid, out_dir, prefix=combo_name)
        combo_outputs.append(combo_name)
    return combo_outputs


def parse_args():
    p = argparse.ArgumentParser(description="Generate diverse datasets via prepare_data.py orchestration.")
    p.add_argument("--model_path", type=str, default="/homes/gws/lxh22/models/Qwen2.5-Math-1.5B")
    p.add_argument("--base_output_dir", type=str, default="generated-data")
    p.add_argument("--n_samples", type=int, default=8000, help="Samples to generate per run (pre-filter).")
    p.add_argument("--n_train", type=int, default=4000, help="Train samples to keep; valid will be the remainder from prepare_data then rebalanced later for combos.")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--dry_run", action="store_true", help="Print subprocess commands without executing generation.")
    return p.parse_args()


def main():
    args = parse_args()
    Path(args.base_output_dir).mkdir(parents=True, exist_ok=True)

    print("=== Phase 1: pi1 temperature sweep ===")
    single_runs = generate_single_policy_temperatures(args)
    for policy, temp, train_p, valid_p in single_runs:
        print(f"[done] {policy} temp={temp} train={train_p} valid={valid_p}")

    print("\n=== Phase 2: Combinations at temp=1.0 ===")
    combos = generate_combinations(args)
    print(f"[complete] combinations generated: {combos}")

    print("\nAll done. You can inspect generated parquet/csv under", args.base_output_dir)


if __name__ == "__main__":
    main()
