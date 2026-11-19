#!/usr/bin/env python3
"""Train on full dev using a fixed panel, then evaluate once on lockbox."""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from split_integrity import validate_partition, validate_locked_development

from run_model_benchmark_from_selected import (
    build_oof_raw_proba,
    canonical_model_name,
    fit_decision_policy,
    evaluate_model,
    parse_list_floats,
    parse_list_ints,
    parse_list_ints_or_none,
    parse_list_mixed_max_features,
    parse_decision_policy_modes,
    read_selected,
    tune_extra_trees,
    tune_random_forest,
    tune_lr_multinomial,
    tune_svm_linear,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="One-shot lockbox evaluation from a fixed panel.")
    parser.add_argument("--probe-matrix", required=True, help="Probe matrix parquet path.")
    parser.add_argument(
        "--dev-manifest", required=True, help="Dev manifest TSV (sample_id, project_id)."
    )
    parser.add_argument(
        "--lockbox-manifest", required=True, help="Lockbox manifest TSV (sample_id, project_id)."
    )
    parser.add_argument("--panel-tsv", required=True, help="Panel TSV with probe_id column.")
    parser.add_argument("--k", type=int, default=30, help="Probe count to use from panel.")
    parser.add_argument("--out-dir", required=True, help="Output directory.")
    parser.add_argument(
        "--model",
        default="extra_trees",
        choices=[
            "lr_multinomial",
            "svm_linear",
            "extra_trees",
            "random_forest",
            "linear_svm",
            "elastic_net_multinomial_logistic",
        ],
        help="Model family to tune on full dev then evaluate on lockbox.",
    )
    parser.add_argument(
        "--class-weight-mode",
        default="balanced",
        choices=["none", "balanced"],
        help="Training class-weight mode for models that support it.",
    )
    parser.add_argument(
        "--tune-metric",
        default="macro_f1",
        choices=["macro_f1", "weighted_f1", "accuracy", "balanced_accuracy"],
        help="Metric optimized in inner CV tuning.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument(
        "--inner-folds", type=int, default=5, help="Inner CV folds for tuning on dev."
    )
    parser.add_argument("--lr-c-grid", default="0.01,0.03,0.1,0.3,1,3,10,30", help="LR C grid.")
    parser.add_argument(
        "--lr-l1-grid", default="0.1,0.5,0.9", help="LR l1_ratio grid (elastic-net)."
    )
    parser.add_argument(
        "--svm-c-grid", default="0.01,0.03,0.1,0.3,1,3,10,30", help="Linear SVM C grid."
    )
    parser.add_argument(
        "--et-n-estimators-grid", default="200,400", help="ExtraTrees n_estimators grid."
    )
    parser.add_argument(
        "--et-max-features-grid", default="sqrt,0.2,0.35", help="ExtraTrees max_features grid."
    )
    parser.add_argument(
        "--et-max-depth-grid", default="0,10,20", help="ExtraTrees max_depth grid (use 0 for None)."
    )
    parser.add_argument(
        "--et-min-samples-leaf-grid", default="1,3,5", help="ExtraTrees min_samples_leaf grid."
    )
    parser.add_argument(
        "--rf-n-estimators-grid", default="200,400", help="RandomForest n_estimators grid."
    )
    parser.add_argument(
        "--rf-max-features-grid", default="sqrt,0.2,0.35", help="RandomForest max_features grid."
    )
    parser.add_argument(
        "--rf-max-depth-grid",
        default="0,10,20",
        help="RandomForest max_depth grid (use 0 for None).",
    )
    parser.add_argument(
        "--rf-min-samples-leaf-grid", default="1,3,5", help="RandomForest min_samples_leaf grid."
    )
    parser.add_argument("--n-jobs", type=int, default=8, help="Parallel workers for tree fitting.")
    parser.add_argument(
        "--compute-auroc", action="store_true", help="Compute macro and per-class AUROC."
    )
    parser.add_argument(
        "--fit-decision-policy",
        action="store_true",
        help="Fit a post-probability decision policy from dev OOF probabilities before lockbox evaluation.",
    )
    parser.add_argument(
        "--decision-policy-metric",
        default="",
        choices=["", "macro_f1", "weighted_f1", "accuracy", "balanced_accuracy"],
        help="Metric optimized for the decision policy. Defaults to --tune-metric.",
    )
    parser.add_argument(
        "--decision-policy-modes",
        default="argmax,temp,offset,temp_offset",
        help="Comma-separated decision policy modes to consider.",
    )
    parser.add_argument(
        "--decision-policy-min-gain",
        type=float,
        default=0.002,
        help="Minimum OOF gain over raw argmax required to keep a non-argmax decision policy.",
    )
    return parser.parse_args()


def _read_manifest(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str)
    req = {"sample_id", "project_id"}
    if not req.issubset(df.columns):
        raise SystemExit(f"Manifest missing required columns {req}: {path}")
    return df


def _read_panel(panel_path: Path, k: int) -> list[str]:
    panel = pd.read_csv(panel_path, sep="\t", dtype=str)
    if "probe_id" not in panel.columns:
        raise SystemExit(f"Panel missing probe_id column: {panel_path}")
    if "panel_rank" in panel.columns:
        panel["panel_rank"] = pd.to_numeric(panel["panel_rank"], errors="coerce")
        panel = panel.sort_values(["panel_rank", "probe_id"], kind="mergesort")
    probes = panel["probe_id"].astype(str).tolist()
    probes = [p for p in probes if p]
    probes = probes[:k]
    if len(probes) < k:
        raise SystemExit(f"Panel has only {len(probes)} probes (< k={k}): {panel_path}")
    return probes


def main() -> int:
    args = parse_args()
    probe_matrix = Path(args.probe_matrix)
    dev_manifest_path = Path(args.dev_manifest)
    lock_manifest_path = Path(args.lockbox_manifest)
    panel_path = Path(args.panel_tsv)
    out_dir = Path(args.out_dir)
    for p in [probe_matrix, dev_manifest_path, lock_manifest_path, panel_path]:
        if not p.exists():
            raise SystemExit(f"Missing input: {p}")
    print("[lockbox] loading manifests/panel", flush=True)
    dev = _read_manifest(dev_manifest_path)
    lock = _read_manifest(lock_manifest_path)
    split_fingerprint = validate_partition(dev.to_dict("records"), lock.to_dict("records"))
    validate_locked_development(dev.to_dict("records"), dev_manifest_path, lock.to_dict("records"))
    out_dir.mkdir(parents=True, exist_ok=True)
    probes = _read_panel(panel_path, args.k)
    dev_ids = dev["sample_id"].astype(str).tolist()
    lock_ids = lock["sample_id"].astype(str).tolist()
    y_dev = dev.set_index("sample_id").loc[dev_ids, "project_id"].astype(str).to_numpy()
    y_lock = lock.set_index("sample_id").loc[lock_ids, "project_id"].astype(str).to_numpy()
    all_classes = sorted(
        set(dev["project_id"].astype(str).tolist()) | set(lock["project_id"].astype(str).tolist())
    )
    print(
        f"[lockbox] dev_n={len(dev_ids)} lockbox_n={len(lock_ids)} classes={len(all_classes)} probes={len(probes)}",
        flush=True,
    )
    t_read_start = time.perf_counter()
    pf = pq.ParquetFile(probe_matrix)
    print("[lockbox] reading selected probe matrix", flush=True)
    X_dev = read_selected(pf, dev_ids, probes)
    X_lock = read_selected(pf, lock_ids, probes)
    t_read = time.perf_counter() - t_read_start
    print(f"[lockbox] read done in {t_read:.2f}s", flush=True)
    t_tune_start = time.perf_counter()
    requested_model = str(args.model).strip()
    model_name = canonical_model_name(requested_model)
    class_weight_mode = (
        "balanced"
        if requested_model in {"linear_svm", "elastic_net_multinomial_logistic"}
        else args.class_weight_mode
    )
    print(
        f"[lockbox] tuning model={requested_model} cw={class_weight_mode} tune_metric={args.tune_metric} inner_folds={args.inner_folds}",
        flush=True,
    )
    et_n_estimators_grid = parse_list_ints(args.et_n_estimators_grid)
    et_max_features_grid = parse_list_mixed_max_features(args.et_max_features_grid)
    et_max_depth_grid = parse_list_ints_or_none(args.et_max_depth_grid)
    et_min_samples_leaf_grid = parse_list_ints(args.et_min_samples_leaf_grid)
    rf_n_estimators_grid = parse_list_ints(args.rf_n_estimators_grid)
    rf_max_features_grid = parse_list_mixed_max_features(args.rf_max_features_grid)
    rf_max_depth_grid = parse_list_ints_or_none(args.rf_max_depth_grid)
    rf_min_samples_leaf_grid = parse_list_ints(args.rf_min_samples_leaf_grid)
    if model_name == "lr_multinomial":
        (model, inner_df, best_params) = tune_lr_multinomial(
            X_train=X_dev,
            y_train=y_dev,
            all_classes=all_classes,
            c_grid=parse_list_floats(args.lr_c_grid),
            l1_grid=parse_list_floats(args.lr_l1_grid),
            inner_folds=args.inner_folds,
            seed=args.seed,
            tune_metric=args.tune_metric,
            class_weight_mode=class_weight_mode,
            penalties=(
                ["elasticnet"]
                if requested_model == "elastic_net_multinomial_logistic"
                else ["l2", "elasticnet"]
            ),
        )
    elif model_name == "svm_linear":
        (model, inner_df, best_params) = tune_svm_linear(
            X_train=X_dev,
            y_train=y_dev,
            all_classes=all_classes,
            c_grid=parse_list_floats(args.svm_c_grid),
            inner_folds=args.inner_folds,
            seed=args.seed,
            tune_metric=args.tune_metric,
            class_weight_mode=class_weight_mode,
        )
    elif model_name == "extra_trees":
        (model, inner_df, best_params) = tune_extra_trees(
            X_train=X_dev,
            y_train=y_dev,
            all_classes=all_classes,
            n_estimators_grid=et_n_estimators_grid,
            max_features_grid=et_max_features_grid,
            max_depth_grid=et_max_depth_grid,
            min_samples_leaf_grid=et_min_samples_leaf_grid,
            inner_folds=args.inner_folds,
            seed=args.seed,
            n_jobs=args.n_jobs,
            tune_metric=args.tune_metric,
            class_weight="balanced" if class_weight_mode == "balanced" else None,
        )
    elif model_name == "random_forest":
        (model, inner_df, best_params) = tune_random_forest(
            X_train=X_dev,
            y_train=y_dev,
            all_classes=all_classes,
            n_estimators_grid=rf_n_estimators_grid,
            max_features_grid=rf_max_features_grid,
            max_depth_grid=rf_max_depth_grid,
            min_samples_leaf_grid=rf_min_samples_leaf_grid,
            inner_folds=args.inner_folds,
            seed=args.seed,
            n_jobs=args.n_jobs,
            tune_metric=args.tune_metric,
            class_weight="balanced" if class_weight_mode == "balanced" else None,
        )
    else:
        raise SystemExit(f"Unsupported model: {requested_model}")
    t_tune = time.perf_counter() - t_tune_start
    print(f"[lockbox] tuning done in {t_tune:.2f}s", flush=True)
    decision_policy = None
    if args.fit_decision_policy:
        print("[lockbox] fitting decision policy from dev OOF probabilities", flush=True)
        policy_metric = str(args.decision_policy_metric).strip() or str(args.tune_metric).strip()
        policy_modes = parse_decision_policy_modes(args.decision_policy_modes)
        raw_proba_oof = build_oof_raw_proba(
            model_name=model_name,
            model=model,
            X_train=X_dev,
            y_train=y_dev,
            all_classes=all_classes,
            class_weight_mode=class_weight_mode,
            inner_folds=args.inner_folds,
            seed=args.seed,
        )
        decision_policy = fit_decision_policy(
            y_true=y_dev,
            raw_proba_oof=raw_proba_oof,
            all_classes=all_classes,
            metric=policy_metric,
            modes=policy_modes,
            min_gain=float(args.decision_policy_min_gain),
        )
        with (out_dir / "decision_policy.json").open("w", encoding="utf-8") as fh:
            json.dump(decision_policy, fh, indent=2)
        print(
            f"[lockbox] decision policy type={decision_policy.get('type', '')} metric={decision_policy.get('metric', '')}",
            flush=True,
        )
    t_eval_start = time.perf_counter()
    print("[lockbox] evaluating on lockbox", flush=True)
    (metrics, cm_df, auc_df, fitted_model) = evaluate_model(
        model_name=model_name,
        model=model,
        X_train=X_dev,
        y_train=y_dev,
        X_test=X_lock,
        y_test=y_lock,
        all_classes=all_classes,
        compute_auroc=args.compute_auroc,
        class_weight_mode=class_weight_mode,
        decision_policy=decision_policy,
    )
    t_eval = time.perf_counter() - t_eval_start
    print(f"[lockbox] eval done in {t_eval:.2f}s", flush=True)
    inner_df.to_csv(out_dir / "innercv_results.tsv", sep="\t", index=False)
    with (out_dir / "best_params.json").open("w", encoding="utf-8") as fh:
        json.dump(best_params, fh, indent=2)
    pd.DataFrame({"probe_id": probes}).to_csv(
        out_dir / f"selected_probes_K{args.k}.tsv", sep="\t", index=False
    )
    cm_df.to_csv(out_dir / f"confusion_matrix_K{args.k}.tsv", sep="\t")
    metrics_row = {
        "model": args.model,
        "panel_file": str(panel_path),
        "k": args.k,
        "n_dev": len(dev_ids),
        "n_lockbox": len(lock_ids),
        "n_classes": len(all_classes),
        "class_weight_mode": class_weight_mode,
        "tune_metric": args.tune_metric,
        **metrics,
    }
    pd.DataFrame([metrics_row]).to_csv(out_dir / "metrics_lockbox.tsv", sep="\t", index=False)
    if args.compute_auroc and (not auc_df.empty):
        auc_df.to_csv(out_dir / f"auc_by_class_K{args.k}.tsv", sep="\t", index=False)
    timing = pd.DataFrame(
        [{"t_read_selected_matrix": t_read, "t_tune_model": t_tune, "t_eval_model": t_eval}]
    )
    timing.to_csv(out_dir / "timing.tsv", sep="\t", index=False)
    artifact = {
        "model": fitted_model,
        "model_name": args.model,
        "probe_ids": probes,
        "class_order": all_classes,
        "best_params": best_params,
        "decision_policy": decision_policy,
        "k": int(args.k),
        "seed": int(args.seed),
    }
    artifact_path = out_dir / "model_bundle.pkl"
    with artifact_path.open("wb") as fh:
        pickle.dump(artifact, fh, protocol=pickle.HIGHEST_PROTOCOL)
    artifact_meta = {
        "split_membership_at_fit": split_fingerprint,
        "model_name": args.model,
        "artifact_file": str(artifact_path.name),
        "panel_file": str(panel_path),
        "probe_matrix": str(probe_matrix),
        "dev_manifest": str(dev_manifest_path),
        "lockbox_manifest": str(lock_manifest_path),
        "probe_count": len(probes),
        "class_count": len(all_classes),
        "class_order": all_classes,
        "best_params": best_params,
        "decision_policy_type": str((decision_policy or {}).get("type", "")).strip(),
        "decision_policy_metric": str((decision_policy or {}).get("metric", "")).strip(),
        "decision_policy": decision_policy,
        "class_weight_mode": class_weight_mode,
        "tune_metric": args.tune_metric,
        "seed": int(args.seed),
    }
    with (out_dir / "model_bundle_meta.json").open("w", encoding="utf-8") as fh:
        json.dump(artifact_meta, fh, indent=2)
    print("[lockbox] writing outputs complete", flush=True)
    print(f"Wrote: {out_dir / 'metrics_lockbox.tsv'}")
    print(f"Wrote: {artifact_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
