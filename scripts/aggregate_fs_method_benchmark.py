#!/usr/bin/env python3
"""Aggregate feature-selection method benchmark outputs."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate FS benchmark outputs.")
    parser.add_argument("--run-dir", required=True, help="Run directory with repeat*/fold*/method outputs.")
    parser.add_argument("--metrics-out", default=None, help="Summary metrics output TSV.")
    parser.add_argument("--timing-out", default=None, help="Summary timing output TSV.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise SystemExit(f"Run dir not found: {run_dir}")

    metrics_files = list(run_dir.glob("repeat*/fold*/*/metrics.tsv"))
    if not metrics_files:
        raise SystemExit("No metrics.tsv files found")

    metrics = pd.concat([pd.read_csv(p, sep="\t") for p in metrics_files], ignore_index=True)
    metrics_cols = [
        c
        for c in metrics.columns
        if c
        not in {
            "method",
            "repeat",
            "fold",
            "k",
            "n_train",
            "n_test",
            "n_classes",
            "best_c",
            "best_inner_macro_f1",
        }
    ]

    agg = {"n_folds": ("accuracy", "size")}
    for col in metrics_cols:
        agg[f"{col}_mean"] = (col, "mean")
        agg[f"{col}_median"] = (col, "median")
        agg[f"{col}_std"] = (col, "std")

    metrics_summary = metrics.groupby(["method", "k"]).agg(**agg).reset_index().sort_values(["method", "k"])
    metrics_out = Path(args.metrics_out) if args.metrics_out else run_dir / "metrics_summary_by_method.tsv"
    metrics_summary.to_csv(metrics_out, sep="\t", index=False)
    print(f"Wrote metrics summary: {metrics_out}")

    timing_files = list(run_dir.glob("repeat*/fold*/*/timing.tsv"))
    if timing_files:
        timing = pd.concat([pd.read_csv(p, sep="\t") for p in timing_files], ignore_index=True)
        timing_cols = [c for c in timing.columns if c not in {"method", "repeat", "fold", "k"}]
        timing[timing_cols] = timing[timing_cols].apply(pd.to_numeric, errors="coerce")
        timing_summary = (
            timing.groupby(["method", "k"])[timing_cols].agg(["mean", "median"]).reset_index()
        )
        timing_summary.columns = [
            "_".join([x for x in col if x]) for col in timing_summary.columns.to_flat_index()
        ]
        timing_summary = timing_summary.rename(columns={"method_": "method", "k_": "k"}).sort_values(["method", "k"])
        timing_out = Path(args.timing_out) if args.timing_out else run_dir / "timing_summary_by_method.tsv"
        timing_summary.to_csv(timing_out, sep="\t", index=False)
        print(f"Wrote timing summary: {timing_out}")

    auc_files = list(run_dir.glob("repeat*/fold*/*/auc_by_class_K*.tsv"))
    if auc_files:
        auc = pd.concat([pd.read_csv(p, sep="\t") for p in auc_files], ignore_index=True)
        auc["auc"] = pd.to_numeric(auc["auc"], errors="coerce")
        auc_summary = (
            auc.groupby(["method", "k", "class"])
            .agg(
                n_folds=("auc", "size"),
                auc_mean=("auc", "mean"),
                auc_median=("auc", "median"),
                auc_std=("auc", "std"),
            )
            .reset_index()
            .sort_values(["method", "k", "class"])
        )
        auc_out = run_dir / "auroc_by_class_summary.tsv"
        auc_summary.to_csv(auc_out, sep="\t", index=False)
        print(f"Wrote per-class AUROC summary: {auc_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
