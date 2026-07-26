#!/usr/bin/env python3
"""Aggregate model benchmark outputs from repeat*/fold*/model directories."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _first_nonempty(series: pd.Series) -> str:
    values = [str(v).strip() for v in series.tolist() if str(v).strip() not in {"", "nan", "None"}]
    return values[0] if values else ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate model benchmark outputs.")
    parser.add_argument("--run-dir", required=True, help="Run directory with repeat*/fold*/model outputs.")
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
    optional_group_cols = [c for c in ["class_weight_mode", "tune_metric"] if c in metrics.columns]
    group_cols = ["model", "selection_method", "k", *optional_group_cols]
    metrics_cols = [
        c
        for c in metrics.columns
        if c
        not in {
            *group_cols,
            "repeat",
            "fold",
            "n_train",
            "n_test",
            "n_classes",
        }
    ]
    numeric_cols: list[str] = []
    nonnumeric_cols: list[str] = []
    for col in metrics_cols:
        parsed = pd.to_numeric(metrics[col], errors="coerce")
        if bool(parsed.notna().any()):
            metrics[col] = parsed
            numeric_cols.append(col)
        else:
            nonnumeric_cols.append(col)

    agg = {"n_folds": ("model", "size")}
    for col in numeric_cols:
        agg[f"{col}_mean"] = (col, "mean")
        agg[f"{col}_median"] = (col, "median")
        agg[f"{col}_std"] = (col, "std")
    for col in nonnumeric_cols:
        agg[col] = (col, _first_nonempty)

    metrics_summary = metrics.groupby(group_cols).agg(**agg).reset_index().sort_values(group_cols)
    metrics_out = Path(args.metrics_out) if args.metrics_out else run_dir / "metrics_summary_by_model.tsv"
    metrics_summary.to_csv(metrics_out, sep="\t", index=False)
    print(f"Wrote metrics summary: {metrics_out}")

    timing_files = list(run_dir.glob("repeat*/fold*/*/timing.tsv"))
    if timing_files:
        timing = pd.concat([pd.read_csv(p, sep="\t") for p in timing_files], ignore_index=True)
        optional_timing_group_cols = [c for c in ["class_weight_mode", "tune_metric"] if c in timing.columns]
        timing_group_cols = ["model", "selection_method", "k", *optional_timing_group_cols]
        timing_cols = [c for c in timing.columns if c not in {*timing_group_cols, "repeat", "fold"}]
        timing[timing_cols] = timing[timing_cols].apply(pd.to_numeric, errors="coerce")
        timing_summary = timing.groupby(timing_group_cols)[timing_cols].agg(["mean", "median"]).reset_index()
        timing_summary.columns = [
            "_".join([x for x in col if x]) for col in timing_summary.columns.to_flat_index()
        ]
        rename_map = {"model_": "model", "selection_method_": "selection_method", "k_": "k"}
        if "class_weight_mode_" in timing_summary.columns:
            rename_map["class_weight_mode_"] = "class_weight_mode"
        if "tune_metric_" in timing_summary.columns:
            rename_map["tune_metric_"] = "tune_metric"
        timing_summary = timing_summary.rename(columns=rename_map).sort_values(timing_group_cols)
        timing_out = Path(args.timing_out) if args.timing_out else run_dir / "timing_summary_by_model.tsv"
        timing_summary.to_csv(timing_out, sep="\t", index=False)
        print(f"Wrote timing summary: {timing_out}")

    auc_files = list(run_dir.glob("repeat*/fold*/*/auc_by_class_K*.tsv"))
    if auc_files:
        auc = pd.concat([pd.read_csv(p, sep="\t") for p in auc_files], ignore_index=True)
        auc["auc"] = pd.to_numeric(auc["auc"], errors="coerce")
        optional_auc_group_cols = [c for c in ["class_weight_mode", "tune_metric"] if c in auc.columns]
        auc_group_cols = ["model", "selection_method", "k", *optional_auc_group_cols, "class"]
        auc_summary = (
            auc.groupby(auc_group_cols)
            .agg(
                n_folds=("auc", "size"),
                auc_mean=("auc", "mean"),
                auc_median=("auc", "median"),
                auc_std=("auc", "std"),
            )
            .reset_index()
            .sort_values(auc_group_cols)
        )
        auc_out = run_dir / "auroc_by_class_summary.tsv"
        auc_summary.to_csv(auc_out, sep="\t", index=False)
        print(f"Wrote per-class AUROC summary: {auc_out}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
