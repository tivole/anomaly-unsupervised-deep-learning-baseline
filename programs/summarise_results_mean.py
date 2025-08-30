"""
Summarize a result CSV by appending one MEAN row.

- Reads the given CSV
- Removes any existing summary rows (MEAN/STD, case-insensitive)
- Computes mean for all present numeric metric columns
- Appends a single MEAN row
- Writes to --out (if provided) or overwrites input in-place

Examples
--------
python summarize_results_mean.py results.csv
python summarize_results_mean.py results.csv --out results_mean.csv
"""

import argparse
import pandas as pd
import numpy as np
from pathlib import Path

CANDIDATE_NUM_COLS = [
    "threshold", "tau_eer",
    "tn", "fp", "fn", "tp",
    "accuracy", "precision", "recall", "f1",
    "frr", "far", "eer"
]

SUMMARY_TOKENS = {"mean", "avg", "average", "std", "stdev", "st.dev", "st_dev"}

def is_summary_row(row) -> bool:
    """Detect summary rows by looking at base_user or (fallback) first column."""
    if "base_user" in row.index:
        val = str(row["base_user"]).strip().lower()
        return val in SUMMARY_TOKENS
    else:
        first_col = row.index[0]
        val = str(row[first_col]).strip().lower()
        return val in SUMMARY_TOKENS

def main():
    ap = argparse.ArgumentParser(description="Append a MEAN row to a results CSV.")
    ap.add_argument("csv", help="Input CSV path")
    ap.add_argument("--out", help="Output CSV path (default: overwrite input)")
    args = ap.parse_args()

    in_path = Path(args.csv)
    out_path = Path(args.out) if args.out else in_path

    if not in_path.exists():
        raise FileNotFoundError(f"No such file: {in_path}")

    df = pd.read_csv(in_path)

    if len(df) > 0:
        mask_keep = ~df.apply(is_summary_row, axis=1)
        df = df[mask_keep].copy()

    num_cols = [c for c in CANDIDATE_NUM_COLS if c in df.columns]

    SKIP_EXTRA = {"k"}
    for c in df.columns:
        if c in num_cols or c in SKIP_EXTRA:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            num_cols.append(c)

    means = df[num_cols].mean(numeric_only=True)
    mean_row = {col: "" for col in df.columns}

    if "base_user" in df.columns:
        mean_row["base_user"] = "MEAN"
    else:
        mean_row[df.columns[0]] = "MEAN"

    for c in num_cols:
        mean_row[c] = float(means[c]) if c in means and pd.notna(means[c]) else ""

    df_out = pd.concat([df, pd.DataFrame([mean_row])], ignore_index=True)
    df_out.to_csv(out_path, index=False)

    print(f"[OK] Wrote {out_path}")
    if num_cols:
        shown = ", ".join(num_cols)
        print(f"[MEAN] computed over columns: {shown}")

if __name__ == "__main__":
    main()
