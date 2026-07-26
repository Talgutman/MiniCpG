#!/usr/bin/env python3
"""Run one paper FS benchmark task from a manifest row."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from run_paper_k_pipeline import load_config, read_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run one FS task from fs_fold_task_manifest.tsv")
    p.add_argument("--execution-plan", required=True, help="execution_plan.json from run_paper_k_pipeline.py")
    p.add_argument("--task-manifest", required=True, help="fs_fold_task_manifest.tsv")
    p.add_argument(
        "--task-index",
        type=int,
        default=None,
        help="0-based task index. Defaults to $SLURM_ARRAY_TASK_ID if present.",
    )
    p.add_argument("--compute-auroc", action="store_true", help="Enable AUROC computation for the FS classifier.")
    p.add_argument("--n-jobs", type=int, default=1, help="Parallel workers for compatible selectors.")
    return p.parse_args()


def resolve_task_index(raw: int | None) -> int:
    if raw is not None:
        return int(raw)
    env = os.environ.get("SLURM_ARRAY_TASK_ID", "").strip()
    if env:
        return int(env)
    raise SystemExit("task index not provided; use --task-index or SLURM_ARRAY_TASK_ID")


def load_task_row(path: Path, task_index: int) -> dict[str, Any]:
    df = pd.read_csv(path, sep="\t")
    if df.empty:
        raise SystemExit(f"Task manifest is empty: {path}")
    if task_index < 0 or task_index >= len(df):
        raise SystemExit(f"task_index={task_index} out of range for {path} (n={len(df)})")
    return df.iloc[int(task_index)].to_dict()


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    plan_path = Path(args.execution_plan)
    task_manifest_path = Path(args.task_manifest)
    task_index = resolve_task_index(args.task_index)

    plan = read_json(plan_path)
    cfg = load_config(Path(str(plan["config"])))
    row = load_task_row(task_manifest_path, task_index)

    cmd = [
        sys.executable,
        "scripts/run_fs_method_benchmark.py",
        "--probe-matrix",
        str(plan["inputs"]["probe_matrix"]),
        "--manifest",
        str(row.get("manifest_tsv") or plan["manifest"]["dev_manifest_tsv"]),
        "--folds",
        str(row.get("folds_tsv") or plan["manifest"].get("folds_tsv", "")),
        "--out-dir",
        str(row["out_dir"]),
        "--repeat",
        str(int(row["repeat"])),
        "--fold",
        str(int(row["fold"])),
        "--k",
        str(int(plan["k"])),
        "--prefilter-method",
        str(row["prefilter_method"]),
        "--prefilter-k",
        str(int(row["prefilter_k"])),
        "--candidate-spec-json",
        str(row["candidate_spec_json_path"]),
        "--methods",
        str(row["selection_method"]),
        "--inner-folds",
        str(int(cfg["cv"]["inner_folds"])),
        "--seed",
        str(int(cfg["cv"]["seed"])),
        "--blacklist-chen",
        str(plan["inputs"]["blacklist_chen"]),
        "--blacklist-snp",
        str(plan["inputs"]["blacklist_snp"]),
        "--probe-metadata",
        str(plan["inputs"]["probe_metadata"]),
        "--n-jobs",
        str(int(args.n_jobs)),
    ]
    if args.compute_auroc:
        cmd.append("--compute-auroc")

    proc = subprocess.run(cmd, cwd=str(repo_root), check=False)
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
