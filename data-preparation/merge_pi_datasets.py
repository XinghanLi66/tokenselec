#!/usr/bin/env python3
"""
Merge pi1, pi2, pi13, pi1209 datasets into a single CSV following the
schema and formatting of `pi1_r128.csv`, ensuring each prompt appears once.

Output: data-preparation/input-data/pi_merged_r128.csv

Deduplication key: 'prompt' (assumes identical question text across sources).
"""
from __future__ import annotations

import sys
from pathlib import Path
import pandas as pd


def main() -> int:
    base = Path(__file__).resolve().parent / "input-data"
    sources = [
        "pi1_r128.parquet",
        "pi2_r128.parquet",
        "pi13_r128.parquet",
        "pi1209_r128.parquet",
    ]

    target_columns = [
        "data_source",
        "prompt",
        "ability",
        "reward_model",
        "extra_info",
    ]

    dfs = []
    for name in sources:
        p = base / name
        if not p.exists():
            print(f"[WARN] Missing source file: {p}")
            continue
        df = pd.read_parquet(p)

        # Normalize column order and check schema
        missing = [c for c in target_columns if c not in df.columns]
        if missing:
            raise ValueError(
                f"File {name} missing columns {missing}. Columns found: {list(df.columns)}"
            )

        # Only keep target columns in the defined order
        df = df[target_columns].copy()

        # Ensure types are strings for stable CSV formatting
        for col in ["data_source", "prompt", "ability", "reward_model", "extra_info"]:
            # Avoid converting non-null dicts/lists to strings implicitly unless needed
            if col in df.columns:
                df[col] = df[col].astype(str)

        dfs.append(df)
        print(f"Loaded {name}: {len(df):,} rows")

    if not dfs:
        print("No input dataframes loaded; nothing to merge.")
        return 1

    merged = pd.concat(dfs, ignore_index=True)
    before = len(merged)

    # Drop duplicates by prompt to ensure each appears once
    merged = merged.drop_duplicates(subset=["prompt"])  # keep='first' by default
    after = len(merged)

    out_path = base / "pi_merged_r128.csv"
    # Write CSV matching pi1 CSV conventions: include header, no index
    merged.to_csv(out_path, index=False)

    print(
        f"Merged {len(dfs)} files: {before:,} -> {after:,} unique prompts.\nSaved: {out_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
