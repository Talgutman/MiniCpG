#!/usr/bin/env python3
"""Build outer CV folds (stratified group)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build outer CV folds.")
    parser.add_argument("--manifest", required=True, help="Pilot manifest TSV.")
    parser.add_argument("--out", required=True, help="Output folds TSV.")
    parser.add_argument("--n-splits", type=int, default=3, help="Number of folds.")
    parser.add_argument("--n-repeats", type=int, default=1, help="Number of repeats.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--sample-col", default="sample_id", help="Sample ID column.")
    parser.add_argument("--project-col", default="project_id", help="Class label column.")
    parser.add_argument("--group-col", default="patient_id", help="Grouping column.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    df = pd.read_csv(manifest_path, sep="\t", dtype=str)
    for col in [args.sample_col, args.project_col, args.group_col]:
        if col not in df.columns:
            print(f"ERROR: missing column {col}", file=sys.stderr)
            return 1

    y = df[args.project_col].astype(str)
    groups = df[args.group_col].astype(str)

    out_frames = []
    for repeat in range(args.n_repeats):
        sgkf = StratifiedGroupKFold(
            n_splits=args.n_splits,
            shuffle=True,
            random_state=args.seed + repeat,
        )
        folds = pd.Series(index=df.index, dtype=int)
        for fold, (_, test_idx) in enumerate(sgkf.split(df, y, groups)):
            folds.iloc[test_idx] = fold
        out_frames.append(
            pd.DataFrame(
                {
                    "sample_id": df[args.sample_col].astype(str),
                    "outer_fold": folds.astype(int),
                    "repeat": repeat,
                }
            )
        )

    out_df = pd.concat(out_frames, ignore_index=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, sep="\t", index=False)

    print(f"Wrote folds: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
