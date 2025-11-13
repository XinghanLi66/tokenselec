"""Orchestration script to generate increasingly diverse training data.

Phases:
1. Single-source (pi1) generations across temperature sweep: 0.2,0.4,0.6,0.8,1.0.
2. Combination datasets at temperature=1.0 formed by concatenating their
    input dataframes first, then running one generation over the union:
    (pi1+pi2), (pi1+pi2+pi13), (pi1+pi2+pi13+pi1209).

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
        --base_output_dir data \
        --total_samples 16000

After completion you'll have directories:
  data/pi1/temp_0.2/ ... etc
  data/combined/pi1_pi2/ ... etc

Requires: pandas.
Heavy generation dependencies (accelerate, transformers, etc.) must be
installed for generation to run.
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


def run_single(policy: str, temperature: float, args) -> Path:
    """Invoke generate_all_responses.py for one policy & temperature.

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
        "data-preparation/generate_all_responses.py",
        "--model_path", args.model_path,
        "--input_data", str(input_parquet),
        "--output_dir", str(dataset_dir),
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
            dataset_dir = run_single(policy, temp, args)
            c_pq, i_pq = get_run_files(dataset_dir)
            results.append((policy, temp, dataset_dir, c_pq, i_pq))
            print(f"[info] done for {policy} temp={temp} -> {dataset_dir} (correct/incorrect)")
    return results


def generate_combinations(args):
    """For each combination at temperature=1.0, concatenate input dataframes first, then run one generation.

    Concatenated parquet is stored under data-preparation/input-data/<combo>.parquet.
    If it already exists, it is reused without re-writing.
    """
    combo_outputs = []
    input_dir = Path("data-preparation/input-data")
    input_dir.mkdir(parents=True, exist_ok=True)
    for combo in COMBINATIONS:
        print(f"[combo] building combination: {'+'.join(combo)}")
        # Load input parquets and concatenate
        dfs = []
        for policy in combo:
            p = input_dir / f"{policy}_r128.parquet"
            if not p.exists():
                raise FileNotFoundError(f"Missing input parquet: {p}")
            dfs.append(pd.read_parquet(p))
        concat_df = pd.concat(dfs, ignore_index=True)
        combo_name = "_".join(combo)
        concat_path = input_dir / f"{combo_name}.parquet"
        if not concat_path.exists():
            concat_df.to_parquet(concat_path, index=False)
            print(f"[combo] wrote concatenated input -> {concat_path}")
        else:
            print(f"[combo] using existing concatenated input -> {concat_path}")

        # Output dataset dir
        out_dir = Path(args.base_output_dir) / combo_name
        out_dir.mkdir(parents=True, exist_ok=True)

        # Run one generation over the concatenated inputs at temp=1.0
        cmd = [
            sys.executable,
            "data-preparation/generate_all_responses.py",
            "--model_path", args.model_path,
            "--input_data", str(concat_path),
            "--output_dir", str(out_dir),
            "--total_samples", str(args.total_samples),
            "--batch_size", str(args.batch_size),
            "--temperature", "1.0",
            "--precision", args.precision,
        ]
        print(f"[run combo] {'+'.join(combo)} -> {' '.join(cmd)}")
        if not args.dry_run:
            subprocess.run(cmd, check=True)
        combo_outputs.append(str(out_dir))
        print(f"[info] complete combination: {'+'.join(combo)} -> {out_dir}")
    return combo_outputs


def parse_args():
    p = argparse.ArgumentParser(description="Generate diverse datasets by orchestrating single and concatenated runs.")
    p.add_argument("--model_path", type=str, default="/homes/gws/lxh22/models/Qwen2.5-Math-1.5B")
    p.add_argument("--base_output_dir", type=str, default="data")
    p.add_argument("--total_samples", type=int, default=16000, help="Total samples to generate per run.")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--precision", type=str, default="bf16", choices=["bf16", "fp16", "fp32"])
    p.add_argument("--dry_run", action="store_true", help="Print subprocess commands without executing generation.")
    return p.parse_args()


def main():
    args = parse_args()
    Path(args.base_output_dir).mkdir(parents=True, exist_ok=True)

    print("=== Phase 1: pi1 temperature sweep (no train/valid split) ===")
    single_runs = generate_single_policy_temperatures(args)

    print("\n=== Phase 2: Combinations at temp=1.0 (concatenate inputs, then generate) ===")
    combos = generate_combinations(args)

    print("\nAll done. You can inspect datasets under:", args.base_output_dir)


if __name__ == "__main__":
    main()
