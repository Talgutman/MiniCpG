#!/usr/bin/env python3
"""Run external inference/evaluation from a fitted model bundle and panel matrix."""

from __future__ import annotations

import argparse
import json
import pickle
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

from run_model_benchmark_from_selected import apply_decision_policy_proba, argmax_labels_from_proba


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate external panel matrix using fitted model bundle.")
    p.add_argument("--panel-matrix", required=True, help="external_panel_matrix_allow_missing1.tsv")
    p.add_argument("--model-bundle", required=True, help="model_bundle.pkl from lockbox one-shot training")
    p.add_argument("--out-dir", required=True, help="Output directory")
    p.add_argument(
        "--pre-impute",
        choices=["none", "sample_mean"],
        default="none",
        help="Optional pre-imputation on X before model.predict. 'sample_mean' fills each row NaN by that row mean.",
    )
    return p.parse_args()


def _align_proba(proba: np.ndarray, model_classes: list[str], target_classes: list[str]) -> np.ndarray:
    out = np.zeros((proba.shape[0], len(target_classes)), dtype=float)
    col_map = {c: i for i, c in enumerate(model_classes)}
    for j, c in enumerate(target_classes):
        if c in col_map:
            out[:, j] = proba[:, col_map[c]]
    return out


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


def _metrics_row(y_true: np.ndarray, y_pred: np.ndarray, proba: np.ndarray, classes: list[str]) -> dict[str, float]:
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", labels=classes))
    weighted_f1 = float(f1_score(y_true, y_pred, average="weighted", labels=classes))
    bal = float(balanced_accuracy_score(y_true, y_pred))
    acc = float(accuracy_score(y_true, y_pred))
    auc_df = _compute_auc_by_class(y_true, proba, classes)
    auc_vals = pd.to_numeric(auc_df["auc"], errors="coerce").dropna().to_numpy()
    macro_auc = float(np.mean(auc_vals)) if len(auc_vals) else float("nan")
    return {
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "balanced_accuracy": bal,
        "accuracy": acc,
        "auroc": macro_auc,
    }


def _confusion_df(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes_true: list[str],
    classes_pred: list[str],
    *,
    drop_zero_true_rows: bool = True,
) -> pd.DataFrame:
    true_cat = pd.Categorical(y_true, categories=classes_true, ordered=True)
    pred_cat = pd.Categorical(y_pred, categories=classes_pred, ordered=True)
    out = pd.crosstab(true_cat, pred_cat, dropna=False)
    # Keep stable axes even when pandas drops all-empty categorical levels.
    out = out.reindex(index=classes_true, columns=classes_pred, fill_value=0)
    if drop_zero_true_rows:
        row_sum = out.sum(axis=1)
        keep_rows = row_sum[row_sum > 0].index.tolist()
        out = out.loc[keep_rows, :]
    return out


def main() -> int:
    args = parse_args()
    panel_matrix_path = Path(args.panel_matrix)
    model_bundle_path = Path(args.model_bundle)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for p in [panel_matrix_path, model_bundle_path]:
        if not p.exists():
            raise SystemExit(f"Missing input: {p}")

    d = pd.read_csv(panel_matrix_path, sep="\t", dtype=str).fillna("")
    if d.empty:
        raise SystemExit("Panel matrix is empty.")
    req_cols = {"sample_id", "gse_id", "tcga_label", "target_bucket"}
    miss = req_cols - set(d.columns)
    if miss:
        raise SystemExit(f"Panel matrix missing required columns: {sorted(miss)}")

    with model_bundle_path.open("rb") as fh:
        artifact = pickle.load(fh)

    model = artifact["model"]
    probe_ids = [str(x) for x in artifact["probe_ids"]]
    class_order = [str(x) for x in artifact["class_order"]]
    decision_policy = artifact.get("decision_policy")

    miss_probe_cols = [p for p in probe_ids if p not in d.columns]
    if miss_probe_cols:
        raise SystemExit(f"Panel matrix missing probe columns: {miss_probe_cols[:5]} ... (n={len(miss_probe_cols)})")

    X = d[probe_ids].apply(pd.to_numeric, errors="coerce")
    n_missing_features = X.isna().sum(axis=1).astype(int)
    if args.pre_impute == "sample_mean":
        row_means = X.mean(axis=1, skipna=True)
        X = X.T.fillna(row_means).T
    y_true = d["tcga_label"].astype(str).to_numpy()

    proba_raw = model.predict_proba(X)
    model_classes_raw = list(model.named_steps["clf"].classes_)
    if np.issubdtype(np.array(model_classes_raw).dtype, np.number):
        model_classes = [class_order[int(c)] for c in model_classes_raw]
    else:
        model_classes = [str(c) for c in model_classes_raw]
    proba = _align_proba(np.asarray(proba_raw), model_classes, class_order)
    y_pred_raw = argmax_labels_from_proba(proba, class_order)
    y_pred = argmax_labels_from_proba(apply_decision_policy_proba(proba, decision_policy), class_order)

    # Predictions table.
    pred = d[["gse_id", "sample_id", "tcga_label", "target_bucket"]].copy()
    pred["y_pred"] = y_pred
    pred["y_pred_raw_argmax"] = y_pred_raw
    pred["is_correct"] = (pred["tcga_label"].astype(str) == pred["y_pred"].astype(str)).astype(int)
    pred["n_missing_features"] = n_missing_features
    pred["all_features_missing"] = (pred["n_missing_features"] == len(probe_ids)).astype(int)
    for j, c in enumerate(class_order):
        pred[f"proba__{c}"] = proba[:, j]
    pred.to_csv(out_dir / "external_predictions_allow_missing1.tsv", sep="\t", index=False)

    # Overall metrics.
    overall = _metrics_row(y_true, y_pred, proba, class_order)
    overall_row = {
        "n_samples": int(len(d)),
        "n_classes_in_truth": int(len(set(y_true))),
        "n_all_features_missing": int((pred["all_features_missing"] == 1).sum()),
        **overall,
    }
    pd.DataFrame([overall_row]).to_csv(out_dir / "external_metrics_overall.tsv", sep="\t", index=False)

    # By bucket.
    bucket_rows = []
    auc_by_bucket_parts = []
    cm_long_parts = []
    for bucket, g in pred.groupby("target_bucket", dropna=False):
        idx = g.index.to_numpy()
        yb = y_true[idx]
        yp = y_pred[idx]
        pb = proba[idx, :]
        m = _metrics_row(yb, yp, pb, class_order)
        bucket_rows.append(
            {
                "target_bucket": bucket,
                "n_samples": int(len(g)),
                "n_classes_in_truth": int(len(set(yb))),
                "n_all_features_missing": int((g["all_features_missing"] == 1).sum()),
                **m,
            }
        )
        auc_df = _compute_auc_by_class(yb, pb, class_order)
        auc_df.insert(0, "target_bucket", bucket)
        auc_by_bucket_parts.append(auc_df)

        cm_df = _confusion_df(yb, yp, class_order, class_order, drop_zero_true_rows=True)
        cm_long = cm_df.stack().reset_index()
        cm_long.columns = ["true_label", "pred_label", "n_samples"]
        cm_long.insert(0, "target_bucket", bucket)
        cm_long_parts.append(cm_long)

    pd.DataFrame(bucket_rows).sort_values("target_bucket").to_csv(
        out_dir / "external_metrics_by_bucket.tsv", sep="\t", index=False
    )

    # AUROC by class overall and by bucket.
    auc_overall = _compute_auc_by_class(y_true, proba, class_order)
    auc_overall.to_csv(out_dir / "external_auroc_by_class_overall.tsv", sep="\t", index=False)
    if auc_by_bucket_parts:
        pd.concat(auc_by_bucket_parts, ignore_index=True).to_csv(
            out_dir / "external_auroc_by_class_by_bucket.tsv", sep="\t", index=False
        )

    # Confusion matrices.
    cm_overall_df = _confusion_df(y_true, y_pred, class_order, class_order, drop_zero_true_rows=True)
    cm_overall_df.to_csv(out_dir / "external_confusion_matrix_overall.tsv", sep="\t")
    if cm_long_parts:
        pd.concat(cm_long_parts, ignore_index=True).to_csv(
            out_dir / "external_confusion_matrix_by_bucket_long.tsv", sep="\t", index=False
        )

    summary = {
        "n_samples": int(len(d)),
        "n_buckets": int(pred["target_bucket"].nunique()),
        "n_classes_in_truth": int(len(set(y_true))),
        "n_classes_model": int(len(class_order)),
        "pre_impute": args.pre_impute,
        "decision_policy_type": str((decision_policy or {}).get("type", "")).strip(),
        "decision_policy_metric": str((decision_policy or {}).get("metric", "")).strip(),
        "overall_metrics": overall_row,
    }
    (out_dir / "external_eval_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Wrote: {out_dir / 'external_metrics_overall.tsv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
