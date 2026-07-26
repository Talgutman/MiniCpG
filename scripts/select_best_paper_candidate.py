#!/usr/bin/env python3
"""Select the best dev-only paper candidate for one K."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from run_paper_k_pipeline import load_config, read_json


MODEL_SIMPLICITY = {
    "elastic_net_multinomial_logistic": 0,
    "linear_svm": 1,
    "random_forest": 2,
    "extra_trees": 3,
}

SELECTION_SIMPLICITY = {
    "mi_rank_baseline": 0,
    "pairwise_hard_weighted_auc_stable": 1,
    "rare_class_rank_shift": 2,
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Select best paper candidate from aggregated dev summaries.")
    p.add_argument("--execution-plan", required=True, help="execution_plan.json from run_paper_k_pipeline.py")
    p.add_argument("--stage", default="stage1_select", help="Stage name in execution plan.")
    p.add_argument("--out-json", default="", help="Override winning-candidate JSON path.")
    p.add_argument("--ranking-tsv", default="", help="Override candidate ranking TSV path.")
    return p.parse_args()


def numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def main() -> int:
    args = parse_args()
    plan = read_json(Path(args.execution_plan))
    cfg = load_config(Path(str(plan["config"])))
    run_root = Path(plan["run_root"])
    stage = dict(plan["stages"][str(args.stage)])
    summary_dir = Path(stage["summary_dir"])
    selection_dir = Path(stage.get("selection_dir") or (run_root / "selection"))
    selection_dir.mkdir(parents=True, exist_ok=True)

    fs_summary_path = Path(stage["outputs"]["fs_candidate_summary_tsv"])
    model_summary_path = Path(stage["outputs"]["model_candidate_summary_tsv"])
    if not fs_summary_path.exists():
        raise SystemExit(f"Missing FS candidate summary: {fs_summary_path}")
    if not model_summary_path.exists():
        raise SystemExit(f"Missing model candidate summary: {model_summary_path}")

    fs_df = pd.read_csv(fs_summary_path, sep="\t")
    model_df = pd.read_csv(model_summary_path, sep="\t")
    merged = model_df.merge(
        fs_df[
            [
                "selection_candidate_id",
                "panel_tsv",
                "stability_stats_tsv",
                "jaccard_summary_tsv",
                "jaccard_mean",
                "jaccard_median",
                "jaccard_std",
            ]
        ],
        on="selection_candidate_id",
        how="left",
        validate="many_to_one",
    )

    selection_cfg = dict(cfg.get("selection") or {})
    primary_metric = str(selection_cfg.get("primary_metric", "macro_f1"))
    tie_break_1 = str(selection_cfg.get("tie_break_1", "weighted_f1"))

    merged["primary_score"] = numeric(merged.get(f"{primary_metric}_mean"))
    merged["tie_break_1_score"] = numeric(merged.get(f"{tie_break_1}_mean"))
    merged["panel_stability_score"] = numeric(merged.get("jaccard_mean"))
    merged["model_simplicity_rank"] = merged["model"].map(MODEL_SIMPLICITY).fillna(99).astype(int)
    merged["selection_simplicity_rank"] = merged["selection_method"].map(SELECTION_SIMPLICITY).fillna(99).astype(int)

    ranking = merged.sort_values(
        by=[
            "primary_score",
            "tie_break_1_score",
            "panel_stability_score",
            "model_simplicity_rank",
            "selection_simplicity_rank",
            "joint_candidate_id",
        ],
        ascending=[False, False, False, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    ranking.insert(0, "rank", range(1, len(ranking) + 1))

    winner = ranking.iloc[0].to_dict()
    out_json = Path(args.out_json) if args.out_json else Path(stage["outputs"].get("winner_json") or (selection_dir / "winning_candidate.json"))
    ranking_tsv = Path(args.ranking_tsv) if args.ranking_tsv else Path(stage["outputs"].get("ranking_tsv") or (selection_dir / "candidate_ranking.tsv"))

    winner_payload: dict[str, Any] = {
        "k": int(plan["k"]),
        "run_root": str(run_root),
        "selection_rule": {
            "primary_metric": primary_metric,
            "tie_break_1": tie_break_1,
            "tie_break_2": "panel_stability",
            "tie_break_3": "model_simplicity",
        },
        "winner": winner,
        "paths": {
            "selection_candidate_run_dir": str(Path(stage["fs_run_base_dir"]) / str(winner["selection_candidate_id"])),
            "joint_candidate_run_dir": str(Path(stage["model_run_base_dir"]) / str(winner["joint_candidate_id"])),
            "panel_tsv": str(winner["panel_tsv"]),
            "fs_summary_tsv": str(fs_summary_path),
            "model_summary_tsv": str(model_summary_path),
            "ranking_tsv": str(ranking_tsv),
            "dev_manifest_tsv": str(plan["manifest"]["dev_manifest_tsv"]),
            "lockbox_manifest_tsv": str(plan["manifest"]["lockbox_manifest_tsv"]),
        },
    }
    out_json.write_text(json.dumps(winner_payload, indent=2) + "\n", encoding="utf-8")
    ranking.to_csv(ranking_tsv, sep="\t", index=False)
    print(f"Wrote: {out_json}")
    print(f"Wrote: {ranking_tsv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
