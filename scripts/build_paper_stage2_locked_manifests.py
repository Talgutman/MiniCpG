#!/usr/bin/env python3
"""Build Stage 2 locked-pipeline manifests from the Stage 1 paper winner."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from run_paper_k_pipeline import (
    build_fs_task_manifest,
    build_model_task_manifest,
    read_json,
    write_candidate_specs,
    write_joint_model_candidate_specs,
    write_tsv,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build stage2_confirm manifests from stage1 winner.")
    p.add_argument("--execution-plan", required=True)
    p.add_argument("--winner-json", default="", help="Override Stage 1 winner JSON path.")
    return p.parse_args()


def _infer_prefilter_profile(selection_candidate_id: str, model: str) -> str:
    if "__pf_logistic" in selection_candidate_id:
        return "logistic"
    if "__pf_default" in selection_candidate_id:
        return "default"
    return "logistic" if model == "elastic_net_multinomial_logistic" else "default"


def _row_from_winner(winner: dict[str, Any]) -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    selection_candidate_id = str(winner["selection_candidate_id"])
    model = str(winner["model"])
    prefilter_profile = _infer_prefilter_profile(selection_candidate_id, model)
    selection_candidate_spec_json = str(winner["selection_candidate_spec_json"])
    model_candidate_spec_json = str(winner["model_candidate_spec_json"])

    selection_row = {
        "selection_candidate_id": selection_candidate_id,
        "fs_candidate_id": str(winner.get("fs_candidate_id", selection_candidate_id)).split("__pf_")[0],
        "selection_method": winner["selection_method"],
        "prefilter_profile": prefilter_profile,
        "prefilter_method": winner["prefilter_method"],
        "prefilter_k": int(winner["prefilter_k"]),
        "candidate_spec_json": selection_candidate_spec_json,
    }
    model_row = {
        "model_candidate_id": winner["model_candidate_id"],
        "model": model,
        "prefilter_profile": prefilter_profile,
        "prefilter_k": int(winner["prefilter_k"]),
        "class_weight_mode": winner["class_weight_mode"],
        "tune_metric": winner["tune_metric"],
        "candidate_spec_json": model_candidate_spec_json,
    }
    joint_row = {
        "joint_candidate_id": winner["joint_candidate_id"],
        "selection_candidate_id": selection_candidate_id,
        "model_candidate_id": winner["model_candidate_id"],
        "selection_method": winner["selection_method"],
        "model": model,
        "prefilter_method": winner["prefilter_method"],
        "prefilter_k": int(winner["prefilter_k"]),
        "class_weight_mode": winner["class_weight_mode"],
        "tune_metric": winner["tune_metric"],
        "selection_candidate_spec_json": selection_candidate_spec_json,
        "model_candidate_spec_json": model_candidate_spec_json,
    }
    return selection_row, model_row, joint_row


def main() -> int:
    args = parse_args()
    plan_path = Path(args.execution_plan)
    plan = read_json(plan_path)
    stage2 = dict(plan["stages"]["stage2_confirm"])
    winner_json = Path(args.winner_json) if args.winner_json else Path(plan["stages"]["stage1_select"]["outputs"]["winner_json"])
    if not winner_json.exists():
        raise SystemExit(f"Missing Stage 1 winner JSON: {winner_json}")

    winner_payload = read_json(winner_json)
    winner = dict(winner_payload["winner"])
    selection_row, model_row, joint_row = _row_from_winner(winner)

    stage2_summary_dir = Path(stage2["summary_dir"])
    stage2_fs_dir = Path(stage2["fs_run_base_dir"])
    stage2_model_dir = Path(stage2["model_run_base_dir"])
    stage2_summary_dir.mkdir(parents=True, exist_ok=True)
    stage2_fs_dir.mkdir(parents=True, exist_ok=True)
    stage2_model_dir.mkdir(parents=True, exist_ok=True)

    write_candidate_specs([selection_row], stage2_fs_dir, "selection_candidate_id", "candidate_spec_json")
    write_candidate_specs([model_row], stage2_model_dir, "model_candidate_id", "candidate_spec_json")
    write_joint_model_candidate_specs([joint_row], stage2_model_dir)

    selection_manifest = Path(stage2["candidate_manifests"]["selection_candidate_manifest_tsv"])
    model_manifest = Path(stage2["candidate_manifests"]["model_candidate_manifest_tsv"])
    joint_manifest = Path(stage2["candidate_manifests"]["joint_candidate_manifest_tsv"])
    fs_task_manifest = Path(stage2["candidate_manifests"]["fs_fold_task_manifest_tsv"])
    model_task_manifest = Path(stage2["candidate_manifests"]["model_fold_task_manifest_tsv"])

    write_tsv(selection_manifest, [selection_row])
    write_tsv(model_manifest, [model_row])
    write_tsv(joint_manifest, [joint_row])

    folds_tsv = Path(stage2["manifest"]["folds_tsv"])
    folds_df = pd.read_csv(folds_tsv, sep="\t", dtype=str)
    dev_manifest_tsv = Path(plan["manifest"]["dev_manifest_tsv"])

    fs_tasks = build_fs_task_manifest(
        selection_candidates=[selection_row],
        folds_df=folds_df,
        fs_run_base_dir=stage2_fs_dir,
        dev_manifest_tsv=dev_manifest_tsv,
        folds_tsv=folds_tsv,
    )
    model_tasks = build_model_task_manifest(
        joint_candidates=[joint_row],
        folds_df=folds_df,
        fs_run_base_dir=stage2_fs_dir,
        model_run_base_dir=stage2_model_dir,
        dev_manifest_tsv=dev_manifest_tsv,
        folds_tsv=folds_tsv,
    )
    write_tsv(fs_task_manifest, fs_tasks)
    write_tsv(model_task_manifest, model_tasks)

    locked_payload = {
        "source_stage1_winner_json": str(winner_json.resolve()),
        "winner": winner,
        "stage2_selection_candidate_manifest_tsv": str(selection_manifest.resolve()),
        "stage2_model_candidate_manifest_tsv": str(model_manifest.resolve()),
        "stage2_joint_candidate_manifest_tsv": str(joint_manifest.resolve()),
        "stage2_fs_fold_task_manifest_tsv": str(fs_task_manifest.resolve()),
        "stage2_model_fold_task_manifest_tsv": str(model_task_manifest.resolve()),
        "n_outer_folds": len(folds_df[["repeat", "outer_fold"]].drop_duplicates()),
        "n_fs_tasks": len(fs_tasks),
        "n_model_tasks": len(model_tasks),
    }
    locked_json = Path(stage2["outputs"]["locked_pipeline_json"])
    locked_json.write_text(json.dumps(locked_payload, indent=2) + "\n", encoding="utf-8")

    plan["stages"]["stage2_confirm"]["locked_manifests_ready"] = True
    plan["stages"]["stage2_confirm"]["counts"]["n_selection_candidates"] = 1
    plan["stages"]["stage2_confirm"]["counts"]["n_model_candidates"] = 1
    plan["stages"]["stage2_confirm"]["counts"]["n_joint_candidates"] = 1
    plan["stages"]["stage2_confirm"]["counts"]["n_fs_tasks"] = len(fs_tasks)
    plan["stages"]["stage2_confirm"]["counts"]["n_model_tasks"] = len(model_tasks)
    plan_path.write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote: {locked_json}")
    print(f"Wrote: {fs_task_manifest}")
    print(f"Wrote: {model_task_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
