#!/usr/bin/env python3
"""Aggregate paper model benchmark outputs candidate-by-candidate."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from run_paper_k_pipeline import read_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Aggregate model outputs for paper joint candidates.")
    p.add_argument("--execution-plan", required=True, help="execution_plan.json from run_paper_k_pipeline.py")
    p.add_argument("--stage", default="stage1_select", help="Stage name in execution plan.")
    p.add_argument("--out-tsv", default="", help="Override summary TSV path.")
    return p.parse_args()


def first_row(path: Path) -> dict[str, Any]:
    df = pd.read_csv(path, sep="\t")
    if df.empty:
        raise SystemExit(f"Summary file is empty: {path}")
    return df.iloc[0].to_dict()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    plan = read_json(Path(args.execution_plan))
    stage = dict(plan["stages"][str(args.stage)])
    joint_manifest = Path(stage["candidate_manifests"]["joint_candidate_manifest_tsv"])
    summary_dir = Path(stage["summary_dir"])
    out_tsv = Path(args.out_tsv) if args.out_tsv else Path(stage["outputs"]["model_candidate_summary_tsv"])
    candidates = pd.read_csv(joint_manifest, sep="\t")

    rows: list[dict[str, Any]] = []
    for row in candidates.to_dict(orient="records"):
        run_dir = Path(stage["model_run_base_dir"]) / str(row["joint_candidate_id"])
        metrics_path = run_dir / "metrics_summary_by_model.tsv"
        timing_path = run_dir / "timing_summary_by_model.tsv"
        cmd = [sys.executable, "scripts/aggregate_model_benchmark.py", "--run-dir", str(run_dir)]
        proc = subprocess.run(cmd, cwd=str(repo_root), check=False)
        if proc.returncode != 0:
            raise SystemExit(f"Model aggregate failed for {run_dir}")
        if not metrics_path.exists():
            raise SystemExit(f"Missing model metrics summary: {metrics_path}")
        metrics_row = first_row(metrics_path)
        merged = {
            **row,
            **metrics_row,
            "run_dir": str(run_dir),
            "metrics_summary_tsv": str(metrics_path),
            "timing_summary_tsv": str(timing_path),
        }
        rows.append(merged)

    pd.DataFrame(rows).to_csv(out_tsv, sep="\t", index=False)
    print(f"Wrote: {out_tsv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
