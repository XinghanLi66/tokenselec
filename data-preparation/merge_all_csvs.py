## subprocess.run(["python", "merge_all_csvs.py", out_dir], check=True)
## python data-preparation/merge_all_csvs.py data/pi1_temp_0_2/

import sys
from pathlib import Path
import pandas as pd

out_dir = Path(sys.argv[1])
all_dfs = []
for path in sorted(out_dir.glob("all_*.csv")):
    all_dfs.append(pd.read_csv(path))

out_df = pd.concat(all_dfs, ignore_index=True)
correct_df = out_df[out_df.correct == 1].copy()
incorrect_df = out_df[out_df.correct == 0].copy()

correct_df.to_csv(out_dir / "correct.csv", index=False)
incorrect_df.to_csv(out_dir / "incorrect.csv", index=False)
out_df.to_csv(out_dir / "all.csv", index=False)

print(
    f"Saved CSV files: correct={len(correct_df)} "
    f"incorrect={len(incorrect_df)} total={len(out_df)} in {out_dir}"
)