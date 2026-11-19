#!/usr/bin/env python3
"""Aggregate paper FS benchmark outputs candidate-by-candidate."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from build_stable_panel_v1 import (
    build_presence_matrix,
    frequency_histogram,
    load_fold_selections,
    pairwise_jaccard,
    summarize_probes,
)
from run_paper_k_pipeline import read_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate FS outputs for paper candidates.")
    p.add_argument("--execution-plan", required=True, help="execution_plan.json from run_paper_k_pipeline.py")
    p.add_argument("--stage", default="stage1_select", help="Stage name in execution plan.")
    p.add_argument("--out-tsv", default="", help="Override summary TSV path.")
    return p.parse_args()


def write_rankrule_outputs(run_dir: Path, method: str, panel_k: int) -> dict[str, Any]:
    out_dir = run_dir / "stable_panel_rankrule"
    out_dir.mkdir(parents=True, exist_ok=True)
    folds = load_fold_selections(run_dir=run_dir, method=method, rank_k=panel_k, panel_k=panel_k)
    stats_df = summarize_probes(
        folds=folds,
        rank_k=panel_k,
        panel_k=panel_k,
        freq_weight=0.7,
        rank_weight=0.3,
    )
    panel_rule_df = stats_df.head(panel_k).copy()
    presence_df = build_presence_matrix(folds=folds, panel_probes=panel_rule_df["probe_id"].astype(str).tolist())
    jacc_long_df, jacc_summary_df, jacc_matrix_df = pairwise_jaccard(folds=folds)
    hist_df = frequency_histogram(stats_df=stats_df, n_folds=len(folds))

    stats_path = out_dir / f"probe_stability_stats_top{panel_k}.tsv"
    panel_path = out_dir / f"stable_panel_rankrule_top{panel_k}.tsv"
    presence_path = out_dir / f"panel_top{panel_k}_presence_by_fold.tsv"
    jacc_long_path = out_dir / f"jaccard_pairs_top{panel_k}.tsv"
    jacc_summary_path = out_dir / f"jaccard_summary_top{panel_k}.tsv"
    jacc_matrix_path = out_dir / f"jaccard_matrix_top{panel_k}.tsv"
    hist_path = out_dir / f"frequency_histogram_top{panel_k}.tsv"
    meta_path = out_dir / "stable_panel_rankrule_meta.json"

    stats_df.to_csv(stats_path, sep="\t", index=False)
    panel_rule_df.to_csv(panel_path, sep="\t", index=False)
    presence_df.to_csv(presence_path, sep="\t", index=False)
    jacc_long_df.to_csv(jacc_long_path, sep="\t", index=False)
    jacc_summary_df.to_csv(jacc_summary_path, sep="\t", index=False)
    jacc_matrix_df.to_csv(jacc_matrix_path, sep="\t")
    hist_df.to_csv(hist_path, sep="\t", index=False)
    meta_path.write_text(
        json.dumps(
            {
                "selected_run_dir": str(run_dir),
                "method": method,
                "panel_k": int(panel_k),
                "n_folds": len(folds),
                "ranking_rule": "rankrule_only",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    jacc_row = jacc_summary_df.iloc[0].to_dict() if not jacc_summary_df.empty else {}
    return {
        "panel_tsv": str(panel_path),
        "stability_stats_tsv": str(stats_path),
        "jaccard_summary_tsv": str(jacc_summary_path),
        "jaccard_mean": float(pd.to_numeric(jacc_row.get("jaccard_mean", float("nan")), errors="coerce")),
        "jaccard_median": float(pd.to_numeric(jacc_row.get("jaccard_median", float("nan")), errors="coerce")),
        "jaccard_std": float(pd.to_numeric(jacc_row.get("jaccard_std", float("nan")), errors="coerce")),
    }


def first_row(path: Path) -> dict[str, Any]:
    df = pd.read_csv(path, sep="\t")
    if df.empty:
        raise SystemExit(f"Summary file is empty: {path}")
    return df.iloc[0].to_dict()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    plan = read_json(Path(args.execution_plan))
    k = int(plan["k"])
    stage = dict(plan["stages"][str(args.stage)])
    selection_manifest = Path(stage["candidate_manifests"]["selection_candidate_manifest_tsv"])
    summary_dir = Path(stage["summary_dir"])
    out_tsv = Path(args.out_tsv) if args.out_tsv else Path(stage["outputs"]["fs_candidate_summary_tsv"])
    candidates = pd.read_csv(selection_manifest, sep="\t")

    rows: list[dict[str, Any]] = []
    for row in candidates.to_dict(orient="records"):
        run_dir = Path(stage["fs_run_base_dir"]) / str(row["selection_candidate_id"])
        metrics_path = run_dir / "metrics_summary_by_method.tsv"
        timing_path = run_dir / "timing_summary_by_method.tsv"
        cmd = [sys.executable, "scripts/aggregate_fs_method_benchmark.py", "--run-dir", str(run_dir)]
        proc = subprocess.run(cmd, cwd=str(repo_root), check=False)
        if proc.returncode != 0:
            raise SystemExit(f"FS aggregate failed for {run_dir}")
        if not metrics_path.exists():
            raise SystemExit(f"Missing FS metrics summary: {metrics_path}")
        metrics_row = first_row(metrics_path)
        stability = write_rankrule_outputs(run_dir, str(row["selection_method"]), k)
        merged = {
            **row,
            **metrics_row,
            "run_dir": str(run_dir),
            "metrics_summary_tsv": str(metrics_path),
            "timing_summary_tsv": str(timing_path),
            **stability,
        }
        rows.append(merged)

    pd.DataFrame(rows).to_csv(out_tsv, sep="\t", index=False)
    print(f"Wrote: {out_tsv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
