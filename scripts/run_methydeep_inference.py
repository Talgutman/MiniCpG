#!/usr/bin/env python3
"""Run released MethyDeep model on an external panel matrix.

Implements the published external preprocessing rule:
- keep rows with n_missing <= max_missing_probes (default 3)
- impute remaining missing values by per-sample mean across panel probes
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run MethyDeep SavedModel inference on external panel matrix.")
    p.add_argument("--panel-matrix", required=True, help="TSV with sample rows and probe columns.")
    p.add_argument("--feature-csv", required=True, help="MethyDeep RF_feature45.csv")
    p.add_argument("--class-csv", required=True, help="MethyDeep targer_names.csv")
    p.add_argument("--model-dir", required=True, help="Path to MethyDeep saved model directory.")
    p.add_argument("--out-dir", required=True, help="Output directory")
    p.add_argument("--k", type=int, default=30, help="Use top-K features from feature CSV (default 30)")
    p.add_argument("--max-missing-probes", type=int, default=3, help="Eligibility threshold (default 3)")
    p.add_argument(
        "--eligible-status-tsv",
        default="",
        help=(
            "Optional sample-status TSV describing the exact current eligible set. "
            "If provided, rows are filtered to keys with extraction_status in {ok_full, ok_partial} "
            "and n_panel_missing <= max-missing-probes before inference."
        ),
    )
    p.add_argument(
        "--target-bucket",
        default="",
        help="Optional target_bucket restriction to apply when filtering eligible-status TSV.",
    )
    return p.parse_args()


def _first_col_values(path: Path) -> list[str]:
    d = pd.read_csv(path, dtype=str)
    if d.empty:
        return []
    col0 = d.columns[0]
    vals = [str(x).strip() for x in d[col0].tolist() if str(x).strip()]
    return vals


def _compute_auc_by_class(y_true: np.ndarray, proba: np.ndarray, classes: list[str]) -> pd.DataFrame:
    rows = []
    for j, c in enumerate(classes):
        y_bin = (y_true == c).astype(int)
        n_pos = int(y_bin.sum())
        n_neg = int(len(y_bin) - n_pos)
        auc = np.nan
        if n_pos > 0 and n_neg > 0:
            auc = float(roc_auc_score(y_bin, proba[:, j]))
        rows.append({"class": c, "auc": auc, "n_pos": n_pos, "n_neg": n_neg})
    return pd.DataFrame(rows)


def _to_key(df: pd.DataFrame) -> pd.Series:
    return df[["gse_id", "sample_id"]].astype(str).agg("||".join, axis=1)


def main() -> int:
    args = parse_args()
    try:
        from tensorflow.keras.models import load_model
    except ImportError as exc:
        raise SystemExit('MethyDeep inference requires its separate TensorFlow/Keras environment; see docs/workflows.md.') from exc
    panel_path = Path(args.panel_matrix)
    feature_path = Path(args.feature_csv)
    class_path = Path(args.class_csv)
    model_dir = Path(args.model_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for p in [panel_path, feature_path, class_path, model_dir]:
        if not p.exists():
            raise SystemExit(f"Missing input: {p}")

    d = pd.read_csv(panel_path, sep="\t", dtype=str).fillna("")
    req = {"gse_id", "sample_id", "tcga_label"}
    miss = req - set(d.columns)
    if miss:
        raise SystemExit(f"Panel matrix missing required columns: {sorted(miss)}")
    if "target_bucket" not in d.columns:
        d["target_bucket"] = "main"

    eligible_status_path = Path(args.eligible_status_tsv) if str(args.eligible_status_tsv).strip() else None
    filtered_by_status = 0
    if eligible_status_path is not None:
        if not eligible_status_path.exists():
            raise SystemExit(f"Missing eligible status TSV: {eligible_status_path}")
        st = pd.read_csv(eligible_status_path, sep="\t", dtype=str).fillna("")
        req_st = {"gse_id", "sample_id"}
        miss_st = req_st - set(st.columns)
        if miss_st:
            raise SystemExit(f"Eligible status TSV missing required columns: {sorted(miss_st)}")
        target_bucket = str(args.target_bucket).strip()
        if target_bucket and "target_bucket" in st.columns:
            st = st[st["target_bucket"].astype(str) == target_bucket].copy()
        if "extraction_status" in st.columns:
            st = st[st["extraction_status"].astype(str).isin({"ok_full", "ok_partial"})].copy()
        if "n_panel_missing" in st.columns:
            st = st[pd.to_numeric(st["n_panel_missing"], errors="coerce") <= int(args.max_missing_probes)].copy()
        eligible_keys = set(_to_key(st).tolist())
        d["__key"] = _to_key(d)
        before = len(d)
        d = d[d["__key"].isin(eligible_keys)].copy()
        filtered_by_status = int(before - len(d))
        d = d.drop(columns="__key")

    features = _first_col_values(feature_path)[: int(args.k)]
    classes = _first_col_values(class_path)
    if len(features) < int(args.k):
        raise SystemExit(f"Feature CSV has only {len(features)} features (<k={args.k})")
    if len(classes) == 0:
        raise SystemExit("Class CSV has no labels.")

    missing_feature_cols = [c for c in features if c not in d.columns]
    if missing_feature_cols:
        raise SystemExit(f"Panel matrix missing required MethyDeep probes: {missing_feature_cols[:8]}")

    X = d[features].apply(pd.to_numeric, errors="coerce")
    n_missing = X.isna().sum(axis=1).astype(int)
    keep_mask = n_missing <= int(args.max_missing_probes)
    dropped_missing_gt = int((~keep_mask).sum())
    d_keep = d.loc[keep_mask].copy()
    X_keep = X.loc[keep_mask].copy()
    n_missing_keep = n_missing.loc[keep_mask]

    # Ning external rule: impute using each sample's mean over available sites.
    row_means = X_keep.mean(axis=1, skipna=True)
    X_imp = X_keep.T.fillna(row_means).T
    all_nan_mask = X_imp.isna().all(axis=1)
    dropped_all_nan = int(all_nan_mask.sum())
    if dropped_all_nan > 0:
        d_keep = d_keep.loc[~all_nan_mask].copy()
        X_imp = X_imp.loc[~all_nan_mask].copy()
        n_missing_keep = n_missing_keep.loc[~all_nan_mask]

    model = load_model(model_dir)
    proba = model.predict(X_imp.to_numpy(dtype=np.float32), verbose=0)
    if proba.shape[1] != len(classes):
        raise SystemExit(
            f"Model output dim {proba.shape[1]} != class labels {len(classes)}. "
            f"Check model/class mapping."
        )
    pred_idx = np.argmax(proba, axis=1).astype(int)
    y_pred = np.array([classes[i] for i in pred_idx], dtype=object)
    y_true = d_keep["tcga_label"].astype(str).to_numpy()

    # Save predictions.
    pred = d_keep[["gse_id", "sample_id", "tcga_label", "target_bucket"]].copy()
    pred["n_missing_features"] = n_missing_keep.to_numpy()
    pred["y_pred"] = y_pred
    pred["is_correct"] = (pred["tcga_label"].astype(str) == pred["y_pred"].astype(str)).astype(int)
    for j, c in enumerate(classes):
        pred[f"proba__{c}"] = proba[:, j]
    pred.to_csv(out_dir / "methydeep_predictions.tsv", sep="\t", index=False)

    # Metrics: union-label and model-label variants.
    labels_union = sorted(set(y_true.tolist()) | set(classes))
    macro_f1_union = float(f1_score(y_true, y_pred, average="macro", labels=labels_union, zero_division=0))
    weighted_f1_union = float(f1_score(y_true, y_pred, average="weighted", labels=labels_union, zero_division=0))
    macro_f1_model = float(f1_score(y_true, y_pred, average="macro", labels=classes, zero_division=0))
    weighted_f1_model = float(f1_score(y_true, y_pred, average="weighted", labels=classes, zero_division=0))
    bal = float(balanced_accuracy_score(y_true, y_pred))
    acc = float(accuracy_score(y_true, y_pred))

    auc_df = _compute_auc_by_class(y_true, proba, classes)
    auc_vals = pd.to_numeric(auc_df["auc"], errors="coerce").dropna().to_numpy()
    macro_auc = float(np.mean(auc_vals)) if len(auc_vals) else float("nan")
    auc_df.to_csv(out_dir / "methydeep_auroc_by_class.tsv", sep="\t", index=False)

    metrics_row = {
        "n_input_rows": int(len(d)),
        "n_kept_missing_rule": int(len(d_keep)),
        "n_filtered_out_by_status_gate": int(filtered_by_status),
        "n_dropped_missing_gt_threshold": int(dropped_missing_gt),
        "n_dropped_all_nan_after_impute": int(dropped_all_nan),
        "missing_threshold": int(args.max_missing_probes),
        "n_truth_classes": int(len(pd.unique(y_true))),
        "n_model_classes": int(len(classes)),
        "n_truth_labels_not_in_model": int(sum(1 for c in pd.unique(y_true) if c not in set(classes))),
        "macro_f1_union_labels": macro_f1_union,
        "weighted_f1_union_labels": weighted_f1_union,
        "macro_f1_model_labels": macro_f1_model,
        "weighted_f1_model_labels": weighted_f1_model,
        "balanced_accuracy": bal,
        "accuracy": acc,
        "auroc_macro_ovr_model_labels": macro_auc,
    }
    pd.DataFrame([metrics_row]).to_csv(out_dir / "methydeep_metrics_overall.tsv", sep="\t", index=False)

    # Confusion outputs.
    cm_union = confusion_matrix(y_true, y_pred, labels=labels_union)
    pd.DataFrame(cm_union, index=labels_union, columns=labels_union).to_csv(
        out_dir / "methydeep_confusion_matrix_union.tsv", sep="\t"
    )
    cm_model = confusion_matrix(y_true, y_pred, labels=classes)
    pd.DataFrame(cm_model, index=classes, columns=classes).to_csv(
        out_dir / "methydeep_confusion_matrix_model_labels.tsv", sep="\t"
    )

    # Convenience per-class supports in truth for audit.
    truth_support = (
        pd.DataFrame({"tcga_label": y_true})
        .value_counts("tcga_label")
        .rename("n_samples")
        .reset_index()
        .sort_values("n_samples", ascending=False, kind="mergesort")
    )
    truth_support.to_csv(out_dir / "methydeep_truth_support.tsv", sep="\t", index=False)

    summary = {
        "inputs": {
            "panel_matrix": str(panel_path),
            "feature_csv": str(feature_path),
            "class_csv": str(class_path),
            "model_dir": str(model_dir),
        },
        "metrics": metrics_row,
    }
    (out_dir / "methydeep_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Wrote: {out_dir / 'methydeep_metrics_overall.tsv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
