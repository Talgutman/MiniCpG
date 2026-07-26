#!/usr/bin/env python3
"""Fold-array wrapper for specificity-constrained cfDNA two-stage analysis."""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.run_cfdna_specificity_constrained_twostage import (  # noqa: E402
    build_configs,
    inner_evaluate_config,
    load_data,
    rank_binary_effect,
    rank_subtype_features,
    summarize_by_stage,
    summarize_per_class,
    train_predict_config,
    workflow_metrics,
    write_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["fold", "merge"], required=True)
    parser.add_argument("--fold-id", type=int)
    parser.add_argument("--dataset", choices=["gse186007", "prjna628686"], required=True)
    parser.add_argument("--matrix", required=True, type=Path)
    parser.add_argument("--sample-metadata", required=True, type=Path)
    parser.add_argument("--stage-metadata", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--budgets", default="10:40,15:35,20:30,25:25,30:20,35:15,40:10")
    parser.add_argument("--max-union", type=int, default=50)
    parser.add_argument("--value-mode", choices=["methylation", "coverage"], default="methylation")
    parser.add_argument("--model-families", default="extra_trees,logistic_l2")
    parser.add_argument("--subtype-selector", choices=["anova", "ovr_union", "pairwise_confusion"], default="anova")
    parser.add_argument(
        "--config-subtype-metric",
        choices=["subtype_macro_f1_all_cancers", "subtype_macro_f1_gate_positive_cancers"],
        default="subtype_macro_f1_all_cancers",
    )
    parser.add_argument("--pairwise-confusion-splits", type=int, default=3)
    parser.add_argument("--pairwise-confusion-weight-scale", type=float, default=4.0)
    parser.add_argument("--pairwise-confusion-model", choices=["extra_trees", "logistic_l2"], default="logistic_l2")
    parser.add_argument("--prjna-labels", default="CRC,LIHC,Normal,PAAD")
    parser.add_argument("--prjna-normal-label", default="Normal")
    parser.add_argument("--target-specificity", type=float, default=0.80)
    parser.add_argument("--outer-splits", type=int, default=5)
    parser.add_argument("--outer-repeats", type=int, default=5)
    parser.add_argument("--inner-splits", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260616)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--n-jobs", type=int, default=4)
    parser.add_argument("--block-size", type=int, default=50000)
    parser.add_argument("--min-train-present-fraction", type=float, default=0.05)
    parser.add_argument("--min-train-present-per-class", type=int, default=5)
    return parser.parse_args()


def split_list(y: np.ndarray, args: argparse.Namespace) -> list[tuple[np.ndarray, np.ndarray]]:
    splitter = RepeatedStratifiedKFold(n_splits=args.outer_splits, n_repeats=args.outer_repeats, random_state=args.seed)
    return [(train.astype(np.int64), test.astype(np.int64)) for train, test in splitter.split(np.zeros(len(y)), y)]


def run_fold(args: argparse.Namespace) -> int:
    if args.fold_id is None:
        raise SystemExit("--fold-id is required in fold mode")
    started = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fold_dir = args.out_dir / "folds"
    fold_dir.mkdir(exist_ok=True)
    matrix, meta, labels, normal_label = load_data(args)
    configs = build_configs(args)
    max_gate_k = max(c.gate_k for c in configs)
    max_subtype_k = max(c.subtype_k for c in configs)
    y = meta["label"].to_numpy(str)
    splits = split_list(y, args)
    if args.fold_id < 1 or args.fold_id > len(splits):
        raise SystemExit(f"fold-id must be 1..{len(splits)}")
    fold_id = args.fold_id
    train_rows, test_rows = splits[fold_id - 1]
    y_train = y[train_rows]
    y_test = y[test_rows]

    inner_rows = []
    config_results = []
    for config in configs:
        result = inner_evaluate_config(matrix, train_rows, y_train, labels, normal_label, config, args, fold_id)
        config_results.append(result)
        inner_rows.append({"fold_id": fold_id, **result})
    config_df = pd.DataFrame(config_results)
    config_df["_feasible_rank"] = config_df["threshold_feasible"].astype(int)
    best_row = config_df.sort_values(
        ["_feasible_rank", "inner_threshold_sensitivity", args.config_subtype_metric, "macro_f1", "accuracy"],
        ascending=False,
    ).iloc[0]
    best = next(config for config in configs if config.config_id == best_row["config_id"])
    gate_rank = rank_binary_effect(
        matrix,
        train_rows,
        y_train != normal_label,
        max_gate_k,
        block_size=args.block_size,
        min_fraction=args.min_train_present_fraction,
        min_per_class=args.min_train_present_per_class,
        value_mode=args.value_mode,
    )
    subtype_rank = rank_subtype_features(
        matrix,
        train_rows[y_train != normal_label],
        y_train[y_train != normal_label],
        max_subtype_k,
        args,
    )
    final_pred, gate_probs, subtype_pred, info = train_predict_config(
        matrix,
        train_rows,
        test_rows,
        y_train,
        y_test,
        labels,
        normal_label,
        best,
        threshold=float(best_row["threshold"]),
        args=args,
        seed_offset=100000 * fold_id,
        gate_rank=gate_rank,
        subtype_rank=subtype_rank,
    )
    metrics = workflow_metrics(y_test, final_pred, labels, normal_label)
    try:
        gate_auc = roc_auc_score(y_test != normal_label, gate_probs)
    except ValueError:
        gate_auc = np.nan
    fold_df = pd.DataFrame(
        [
            {
                "fold_id": fold_id,
                "config_id": best.config_id,
                "threshold": float(best_row["threshold"]),
                "inner_threshold_feasible": bool(best_row["threshold_feasible"]),
                "inner_threshold_specificity": float(best_row["inner_threshold_specificity"]),
                "inner_threshold_sensitivity": float(best_row["inner_threshold_sensitivity"]),
                "gate_k": best.gate_k,
                "subtype_k": best.subtype_k,
                "gate_model": best.gate_model,
                "subtype_model": best.subtype_model,
                "union_size": info["union_size"],
                "n_test": int(len(test_rows)),
                "gate_auc": float(gate_auc) if np.isfinite(gate_auc) else np.nan,
                **metrics,
            }
        ]
    )
    pred_rows = []
    for local_i, row_i in enumerate(test_rows):
        pred_rows.append(
            {
                "fold_id": fold_id,
                "sample_id": meta.loc[row_i, "sample_id"],
                "true_label": y_test[local_i],
                "final_pred": final_pred[local_i],
                "gate_prob_cancer": float(gate_probs[local_i]),
                "gate_pred": "Cancer" if gate_probs[local_i] >= float(best_row["threshold"]) else "Normal",
                "subtype_pred": subtype_pred[local_i],
                "stage_raw": meta.loc[row_i, "stage_raw"],
                "stage_group": meta.loc[row_i, "stage_group"],
                "config_id": best.config_id,
                "threshold": float(best_row["threshold"]),
            }
        )
    selected_rows = []
    for feature_type, features in [("gate", info["gate_features"]), ("subtype", info["subtype_features"])]:
        for rank, feature_index in enumerate(features, start=1):
            selected_rows.append({"fold_id": fold_id, "feature_type": feature_type, "rank": rank, "feature_index": int(feature_index)})
    prefix = fold_dir / f"fold_{fold_id:03d}"
    fold_df.to_csv(f"{prefix}_outer_fold_metrics.tsv", sep="\t", index=False)
    pd.DataFrame(pred_rows).to_csv(f"{prefix}_outer_predictions.tsv", sep="\t", index=False)
    pd.DataFrame(selected_rows).to_csv(f"{prefix}_selected_features.tsv", sep="\t", index=False)
    pd.DataFrame(inner_rows).to_csv(f"{prefix}_inner_config_results.tsv", sep="\t", index=False)
    (fold_dir / f"fold_{fold_id:03d}_manifest.json").write_text(
        json.dumps({"status": "complete", "fold_id": fold_id, "elapsed_sec": round(time.time() - started, 1)}, indent=2) + "\n"
    )
    print(f"completed fold {fold_id}: {best.config_id}")
    return 0


def merge(args: argparse.Namespace) -> int:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    fold_dir = args.out_dir / "folds"
    if not fold_dir.exists():
        raise SystemExit(f"Missing fold directory: {fold_dir}")
    matrix, meta, labels, normal_label = load_data(args)
    y = meta["label"].to_numpy(str)
    n_folds = len(split_list(y, args))
    metric_paths = [fold_dir / f"fold_{i:03d}_outer_fold_metrics.tsv" for i in range(1, n_folds + 1)]
    missing = [str(path) for path in metric_paths if not path.exists()]
    if missing:
        raise SystemExit(f"Missing fold outputs: {missing[:5]}")
    fold_df = pd.concat([pd.read_csv(path, sep="\t") for path in metric_paths], ignore_index=True)
    pred_df = pd.concat([pd.read_csv(fold_dir / f"fold_{i:03d}_outer_predictions.tsv", sep="\t") for i in range(1, n_folds + 1)], ignore_index=True)
    selected_df = pd.concat([pd.read_csv(fold_dir / f"fold_{i:03d}_selected_features.tsv", sep="\t") for i in range(1, n_folds + 1)], ignore_index=True)
    inner_df = pd.concat([pd.read_csv(fold_dir / f"fold_{i:03d}_inner_config_results.tsv", sep="\t") for i in range(1, n_folds + 1)], ignore_index=True)
    aggregate = (
        fold_df.drop(columns=["fold_id", "config_id", "gate_model", "subtype_model"], errors="ignore")
        .agg(["mean", "std"])
        .T.reset_index()
        .rename(columns={"index": "metric"})
    )
    stage = summarize_by_stage(pred_df, labels, normal_label)
    per_class = summarize_per_class(pred_df, labels, normal_label)
    fold_df.to_csv(args.out_dir / "outer_fold_metrics.tsv", sep="\t", index=False)
    pred_df.to_csv(args.out_dir / "outer_predictions.tsv", sep="\t", index=False)
    selected_df.to_csv(args.out_dir / "selected_features_by_fold.tsv", sep="\t", index=False)
    inner_df.to_csv(args.out_dir / "inner_config_results.tsv", sep="\t", index=False)
    aggregate.to_csv(args.out_dir / "aggregate_metrics.tsv", sep="\t", index=False)
    stage.to_csv(args.out_dir / "stage_group_metrics.tsv", sep="\t", index=False)
    per_class.to_csv(args.out_dir / "per_class_metrics.tsv", sep="\t", index=False)
    write_report(args.out_dir, aggregate, stage, per_class)
    manifest = {
        "status": "complete",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "dataset": args.dataset,
        "n_samples": int(len(meta)),
        "n_features": int(matrix.shape[1]),
        "n_folds": n_folds,
        "target_specificity": args.target_specificity,
        "max_union": args.max_union,
        "value_mode": args.value_mode,
        "subtype_selector": args.subtype_selector,
        "config_subtype_metric": args.config_subtype_metric,
        "pairwise_confusion_splits": args.pairwise_confusion_splits,
        "pairwise_confusion_weight_scale": args.pairwise_confusion_weight_scale,
        "pairwise_confusion_model": args.pairwise_confusion_model,
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {args.out_dir}")
    return 0


def main() -> int:
    args = parse_args()
    if args.mode == "fold":
        return run_fold(args)
    return merge(args)


if __name__ == "__main__":
    raise SystemExit(main())
