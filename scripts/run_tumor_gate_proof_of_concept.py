#!/usr/bin/env python3
"""Run a tumor-vs-normal-adjacent gate proof of concept on locked MiniCpG probes."""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedShuffleSplit


DEFAULT_OUT = "outputs/paper_protocolb_v1/post_presentation_tumor_gate_20260531"
DEFAULT_DEV_TUMOR = (
    "outputs/paper_protocolb_v1/k_030/biological_interpretation_current/"
    "tcga_tumor_panel_matrices_current/tcga_dev_panel_matrix.tsv"
)
DEFAULT_TEST_TUMOR = (
    "outputs/paper_protocolb_v1/k_030/biological_interpretation_current/"
    "tcga_tumor_panel_matrices_current/tcga_lockbox_panel_matrix.tsv"
)
DEFAULT_NORMAL = (
    "outputs/paper_protocolb_v1/post_presentation_normal_biology_20260527/"
    "module6_locked_minicpg_on_normals/locked_minicpg_normal_forced_predictions.tsv"
)
DEFAULT_MODEL = "outputs/paper_protocolb_v1/k_030/lockbox_eval_final/model_bundle.pkl"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--active-manifest", default="config/active_locked_run.json")
    p.add_argument("--dev-tumor", default=DEFAULT_DEV_TUMOR)
    p.add_argument("--test-tumor", default=DEFAULT_TEST_TUMOR)
    p.add_argument("--normal", default=DEFAULT_NORMAL)
    p.add_argument("--model-bundle", default=DEFAULT_MODEL)
    p.add_argument("--out-dir", default=DEFAULT_OUT)
    p.add_argument("--seed", type=int, default=20260531)
    p.add_argument("--n-threshold-repeats", type=int, default=200)
    p.add_argument("--threshold-calibration-size", type=float, default=0.25)
    p.add_argument("--target-sensitivity", type=float, default=0.95)
    return p.parse_args()


def validate_active_run(manifest: str) -> dict[str, object]:
    from minicpg.analysis_io import AnalysisInputs
    bound = AnalysisInputs(manifest)
    return {"status": "ok", "locked_run_name": bound.config.get("locked_run_name", "user_bound_locked_model")}


def read_tsv(path: str | Path, **kwargs) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"Missing input: {path}")
    return pd.read_csv(path, sep="\t", **kwargs)


def require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SystemExit(f"{label} missing columns: {missing}")


def load_locked_model(path: str | Path) -> tuple[object, list[str], list[str]]:
    with Path(path).open("rb") as fh:
        bundle = pickle.load(fh)
    return bundle["model"], [str(x) for x in bundle["probe_ids"]], [str(x) for x in bundle["class_order"]]


def align_proba(model: object, proba_raw: np.ndarray, class_order: list[str]) -> np.ndarray:
    model_classes_raw = list(model.named_steps["clf"].classes_)
    if np.issubdtype(np.array(model_classes_raw).dtype, np.number):
        model_classes = [class_order[int(c)] for c in model_classes_raw]
    else:
        model_classes = [str(c) for c in model_classes_raw]
    out = np.zeros((proba_raw.shape[0], len(class_order)), dtype=float)
    class_to_i = {c: i for i, c in enumerate(model_classes)}
    for j, c in enumerate(class_order):
        if c in class_to_i:
            out[:, j] = proba_raw[:, class_to_i[c]]
    return out


def entropy(proba: np.ndarray) -> np.ndarray:
    clipped = np.clip(proba, 1e-12, 1.0)
    return -(clipped * np.log2(clipped)).sum(axis=1)


def minicpg_confidence_rows(
    frame: pd.DataFrame,
    model: object,
    probes: list[str],
    class_order: list[str],
    sample_kind: str,
) -> pd.DataFrame:
    require_columns(frame, {"sample_id", "project_id", *probes}, f"{sample_kind} matrix")
    x = frame[probes].apply(pd.to_numeric, errors="coerce")
    proba = align_proba(model, np.asarray(model.predict_proba(x), dtype=float), class_order)
    top2 = np.sort(proba, axis=1)[:, -2:]
    pred_idx = np.argmax(proba, axis=1)
    rows = pd.DataFrame(
        {
            "sample_id": frame["sample_id"].astype(str).to_numpy(),
            "project_id": frame["project_id"].astype(str).to_numpy(),
            "sample_kind": sample_kind,
            "minicpg_predicted_class": [class_order[i] for i in pred_idx],
            "minicpg_max_probability": proba[np.arange(len(frame)), pred_idx],
            "minicpg_margin": top2[:, 1] - top2[:, 0],
            "minicpg_entropy": entropy(proba),
        }
    )
    native_probs = []
    native_ranks = []
    for i, project in enumerate(rows["project_id"]):
        if project in class_order:
            cls_i = class_order.index(project)
            order = np.argsort(-proba[i])
            native_probs.append(float(proba[i, cls_i]))
            native_ranks.append(int(np.where(order == cls_i)[0][0] + 1))
        else:
            native_probs.append(np.nan)
            native_ranks.append(np.nan)
    rows["native_class_probability"] = native_probs
    rows["native_class_rank"] = native_ranks
    rows["minicpg_correct_tumor_type"] = rows["minicpg_predicted_class"].eq(rows["project_id"]).astype(int)
    return rows


def normal_confidence_rows(normal: pd.DataFrame, sample_kind: str) -> pd.DataFrame:
    required = {
        "sample_id",
        "project_id",
        "forced_predicted_tumor_class",
        "forced_prediction_probability",
        "native_tissue_class_probability",
        "native_tissue_class_rank",
        "forced_prediction_matches_native_tissue",
    }
    require_columns(normal, required, "normal confidence")
    proba_cols = [c for c in normal.columns if c.startswith("proba__")]
    probs = normal[proba_cols].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    top2 = np.sort(probs, axis=1)[:, -2:] if probs.shape[1] >= 2 else np.c_[np.zeros(len(normal)), probs[:, 0]]
    out = pd.DataFrame(
        {
            "sample_id": normal["sample_id"].astype(str),
            "project_id": normal["project_id"].astype(str),
            "sample_kind": sample_kind,
            "minicpg_predicted_class": normal["forced_predicted_tumor_class"].astype(str),
            "minicpg_max_probability": pd.to_numeric(normal["forced_prediction_probability"], errors="coerce"),
            "minicpg_margin": top2[:, 1] - top2[:, 0],
            "minicpg_entropy": entropy(probs),
            "native_class_probability": pd.to_numeric(normal["native_tissue_class_probability"], errors="coerce"),
            "native_class_rank": pd.to_numeric(normal["native_tissue_class_rank"], errors="coerce"),
            "minicpg_correct_tumor_type": pd.to_numeric(
                normal["forced_prediction_matches_native_tissue"], errors="coerce"
            ).fillna(0).astype(int),
        }
    )
    return out


def make_binary_frame(frame: pd.DataFrame, probes: list[str], label: int, split: str) -> pd.DataFrame:
    cols = ["sample_id", "project_id", *probes]
    require_columns(frame, set(cols), split)
    out = frame[cols].copy()
    out["binary_label"] = int(label)
    out["split"] = split
    out["sample_kind"] = "tumor" if label == 1 else "normal_adjacent"
    return out


def fit_centroid_score(train_x: pd.DataFrame, train_y: np.ndarray, test_x: pd.DataFrame) -> np.ndarray:
    imputer = SimpleImputer(strategy="median")
    train_values = imputer.fit_transform(train_x)
    test_values = imputer.transform(test_x)
    tumor_centroid = train_values[train_y == 1].mean(axis=0)
    normal_centroid = train_values[train_y == 0].mean(axis=0)
    values = np.asarray(test_values, dtype=float)
    d_tumor = np.linalg.norm(values - tumor_centroid, axis=1)
    d_normal = np.linalg.norm(values - normal_centroid, axis=1)
    return d_normal - d_tumor


def train_threshold_for_sensitivity(y: np.ndarray, score: np.ndarray, target_sensitivity: float) -> float:
    positives = score[y == 1]
    if len(positives) == 0:
        return float(np.nan)
    return float(np.quantile(positives, 1.0 - target_sensitivity, method="lower"))


def threshold_metrics(y: np.ndarray, score: np.ndarray, threshold: float) -> dict[str, float | int]:
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else np.nan
    specificity = tn / (tn + fp) if tn + fp else np.nan
    return {
        "threshold": float(threshold),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
    }


def metric_summary(y: np.ndarray, score: np.ndarray, model_name: str) -> dict[str, object]:
    roc_auc = roc_auc_score(y, score) if len(np.unique(y)) == 2 else np.nan
    auprc = average_precision_score(y, score) if len(np.unique(y)) == 2 else np.nan
    fpr, tpr, roc_thresholds = roc_curve(y, score)
    precision, recall, pr_thresholds = precision_recall_curve(y, score)
    return {
        "model": model_name,
        "auroc": float(roc_auc),
        "auprc": float(auprc),
        "roc_curve": pd.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": roc_thresholds}),
        "pr_curve": pd.DataFrame(
            {
                "precision": precision,
                "recall": recall,
                "threshold": np.r_[pr_thresholds, np.nan],
            }
        ),
    }


def evaluate_gate_models(train: pd.DataFrame, test: pd.DataFrame, probes: list[str], seed: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_x = train[probes].apply(pd.to_numeric, errors="coerce")
    test_x = test[probes].apply(pd.to_numeric, errors="coerce")
    train_y = train["binary_label"].to_numpy(dtype=int)
    test_y = test["binary_label"].to_numpy(dtype=int)

    models = {
        "logistic_l2_balanced": make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=5000, class_weight="balanced", random_state=seed),
        ),
        "extratrees_balanced": make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesClassifier(
                n_estimators=600,
                class_weight="balanced",
                random_state=seed,
                n_jobs=-1,
                min_samples_leaf=2,
            ),
        ),
    }

    prediction_rows: list[pd.DataFrame] = []
    metric_rows: list[dict[str, object]] = []
    roc_rows: list[pd.DataFrame] = []
    pr_rows: list[pd.DataFrame] = []
    threshold_rows: list[dict[str, object]] = []

    for model_name, model in models.items():
        model.fit(train_x, train_y)
        train_score = np.asarray(model.predict_proba(train_x), dtype=float)[:, 1]
        test_score = np.asarray(model.predict_proba(test_x), dtype=float)[:, 1]
        prediction_rows.append(
            test[["sample_id", "project_id", "sample_kind", "binary_label"]].assign(
                model=model_name,
                tumor_likeness_score=test_score,
            )
        )
        summary = metric_summary(test_y, test_score, model_name)
        metric_rows.append({"model": model_name, "auroc": summary["auroc"], "auprc": summary["auprc"]})
        roc_rows.append(summary["roc_curve"].assign(model=model_name))
        pr_rows.append(summary["pr_curve"].assign(model=model_name))
        for target_sens in [0.95, 0.90]:
            threshold = train_threshold_for_sensitivity(train_y, train_score, target_sens)
            threshold_rows.append(
                {
                    "model": model_name,
                    "threshold_rule": f"train_sensitivity_{target_sens:.2f}",
                    **threshold_metrics(test_y, test_score, threshold),
                }
            )

    train_centroid = fit_centroid_score(train_x, train_y, train_x)
    test_centroid = fit_centroid_score(train_x, train_y, test_x)
    prediction_rows.append(
        test[["sample_id", "project_id", "sample_kind", "binary_label"]].assign(
            model="centroid_distance",
            tumor_likeness_score=test_centroid,
        )
    )
    summary = metric_summary(test_y, test_centroid, "centroid_distance")
    metric_rows.append({"model": "centroid_distance", "auroc": summary["auroc"], "auprc": summary["auprc"]})
    roc_rows.append(summary["roc_curve"].assign(model="centroid_distance"))
    pr_rows.append(summary["pr_curve"].assign(model="centroid_distance"))
    for target_sens in [0.95, 0.90]:
        threshold = train_threshold_for_sensitivity(train_y, train_centroid, target_sens)
        threshold_rows.append(
            {
                "model": "centroid_distance",
                "threshold_rule": f"train_sensitivity_{target_sens:.2f}",
                **threshold_metrics(test_y, test_centroid, threshold),
            }
        )

    return (
        pd.DataFrame(metric_rows),
        pd.concat(prediction_rows, ignore_index=True),
        pd.concat(roc_rows, ignore_index=True),
        pd.concat(pr_rows, ignore_index=True),
        pd.DataFrame(threshold_rows),
    )


def make_gate_model(model_name: str, seed: int, n_estimators: int = 600) -> object:
    if model_name == "logistic_l2_balanced":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            LogisticRegression(max_iter=5000, class_weight="balanced", random_state=seed),
        )
    if model_name == "extratrees_balanced":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesClassifier(
                n_estimators=n_estimators,
                class_weight="balanced",
                random_state=seed,
                n_jobs=-1,
                min_samples_leaf=2,
            ),
        )
    raise ValueError(f"Unknown gate model: {model_name}")


def repeated_threshold_calibration(
    train: pd.DataFrame,
    test: pd.DataFrame,
    probes: list[str],
    seed: int,
    n_repeats: int,
    calibration_size: float,
    target_sensitivity: float,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_x = train[probes].apply(pd.to_numeric, errors="coerce")
    test_x = test[probes].apply(pd.to_numeric, errors="coerce")
    train_y = train["binary_label"].to_numpy(dtype=int)
    test_y = test["binary_label"].to_numpy(dtype=int)
    models = ["logistic_l2_balanced", "extratrees_balanced", "centroid_distance"]
    splitter = StratifiedShuffleSplit(
        n_splits=n_repeats,
        test_size=calibration_size,
        random_state=seed,
    )
    threshold_rows: list[dict[str, object]] = []
    selection_rule = f"repeated_calibration_median_target_sensitivity_{target_sensitivity:.2f}"
    for repeat_i, (fit_idx, calibration_idx) in enumerate(splitter.split(train_x, train_y), start=1):
        fit_x = train_x.iloc[fit_idx]
        fit_y = train_y[fit_idx]
        calibration_x = train_x.iloc[calibration_idx]
        calibration_y = train_y[calibration_idx]
        for model_name in models:
            if model_name == "centroid_distance":
                calibration_score = fit_centroid_score(fit_x, fit_y, calibration_x)
            else:
                model = make_gate_model(model_name, seed=seed + repeat_i, n_estimators=300)
                model.fit(fit_x, fit_y)
                calibration_score = np.asarray(model.predict_proba(calibration_x), dtype=float)[:, 1]
            threshold = train_threshold_for_sensitivity(calibration_y, calibration_score, target_sensitivity)
            calibration_metrics = threshold_metrics(calibration_y, calibration_score, threshold)
            threshold_rows.append(
                {
                    "model": model_name,
                    "repeat": repeat_i,
                    "target_sensitivity": target_sensitivity,
                    "calibration_size": calibration_size,
                    "threshold": threshold,
                    "calibration_sensitivity": calibration_metrics["sensitivity"],
                    "calibration_specificity": calibration_metrics["specificity"],
                    "calibration_balanced_accuracy": calibration_metrics["balanced_accuracy"],
                    "n_fit": int(len(fit_idx)),
                    "n_calibration": int(len(calibration_idx)),
                    "n_calibration_tumor": int(calibration_y.sum()),
                    "n_calibration_normal": int((1 - calibration_y).sum()),
                }
            )
    thresholds = pd.DataFrame(threshold_rows)
    summary_rows: list[dict[str, object]] = []
    test_rows: list[dict[str, object]] = []
    prediction_rows: list[pd.DataFrame] = []
    for model_name in models:
        model_thresholds = pd.to_numeric(thresholds.loc[thresholds["model"].eq(model_name), "threshold"], errors="coerce")
        chosen_threshold = float(model_thresholds.median())
        lower = float(model_thresholds.quantile(0.025))
        upper = float(model_thresholds.quantile(0.975))
        summary_rows.append(
            {
                "model": model_name,
                "threshold_selection_rule": f"median_of_{n_repeats}_calibration_thresholds_target_sensitivity_{target_sensitivity:.2f}",
                "chosen_threshold": chosen_threshold,
                "threshold_mean": float(model_thresholds.mean()),
                "threshold_sd": float(model_thresholds.std(ddof=1)),
                "threshold_q025": lower,
                "threshold_q25": float(model_thresholds.quantile(0.25)),
                "threshold_median": chosen_threshold,
                "threshold_q75": float(model_thresholds.quantile(0.75)),
                "threshold_q975": upper,
                "n_repeats": int(model_thresholds.notna().sum()),
            }
        )
        if model_name == "centroid_distance":
            test_score = fit_centroid_score(train_x, train_y, test_x)
        else:
            model = make_gate_model(model_name, seed=seed)
            model.fit(train_x, train_y)
            test_score = np.asarray(model.predict_proba(test_x), dtype=float)[:, 1]
        prediction_rows.append(
            test[["sample_id", "project_id", "sample_kind", "binary_label"]].assign(
                model=model_name,
                threshold_selection_rule=selection_rule,
                chosen_threshold=chosen_threshold,
                tumor_likeness_score=test_score,
                gate_positive=(test_score >= chosen_threshold).astype(int),
            )
        )
        test_rows.append(
            {
                "model": model_name,
                "threshold_selection_rule": selection_rule,
                **threshold_metrics(test_y, test_score, chosen_threshold),
            }
        )
    calibrated_predictions = pd.concat(prediction_rows, ignore_index=True)
    per_class_rows = []
    for (model_name, project, kind), group in calibrated_predictions.groupby(["model", "project_id", "sample_kind"], sort=True):
        per_class_rows.append(
            {
                "model": model_name,
                "threshold_selection_rule": selection_rule,
                "project_id": project,
                "sample_kind": kind,
                "n_samples": int(len(group)),
                "gate_positive_count": int(group["gate_positive"].sum()),
                "gate_positive_fraction": float(group["gate_positive"].mean()),
            }
        )
    return thresholds, pd.DataFrame(summary_rows), pd.DataFrame(test_rows), calibrated_predictions, pd.DataFrame(per_class_rows)


def evaluate_confidence_rejection(confidence: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y = confidence["binary_label"].to_numpy(dtype=int)
    score_specs = {
        "minicpg_max_probability": confidence["minicpg_max_probability"].to_numpy(dtype=float),
        "minicpg_margin": confidence["minicpg_margin"].to_numpy(dtype=float),
        "negative_entropy": -confidence["minicpg_entropy"].to_numpy(dtype=float),
    }
    metrics = []
    roc = []
    pr = []
    for name, score in score_specs.items():
        summary = metric_summary(y, score, name)
        metrics.append({"model": name, "auroc": summary["auroc"], "auprc": summary["auprc"]})
        roc.append(summary["roc_curve"].assign(model=name))
        pr.append(summary["pr_curve"].assign(model=name))
    return pd.DataFrame(metrics), pd.concat(roc, ignore_index=True), pd.concat(pr, ignore_index=True)


def per_class_table(pred: pd.DataFrame, thresholds: pd.DataFrame, threshold_rule: str) -> pd.DataFrame:
    rows = []
    for model_name, model_pred in pred.groupby("model", sort=False):
        threshold_row = thresholds.loc[
            thresholds["model"].eq(model_name) & thresholds["threshold_rule"].eq(threshold_rule)
        ]
        if threshold_row.empty:
            continue
        threshold = float(threshold_row.iloc[0]["threshold"])
        tmp = model_pred.copy()
        tmp["gate_positive"] = (tmp["tumor_likeness_score"] >= threshold).astype(int)
        for (project, kind), group in tmp.groupby(["project_id", "sample_kind"], sort=True):
            n = len(group)
            positive = int(group["gate_positive"].sum())
            rows.append(
                {
                    "model": model_name,
                    "threshold_rule": threshold_rule,
                    "project_id": project,
                    "sample_kind": kind,
                    "n_samples": n,
                    "gate_positive_count": positive,
                    "gate_positive_fraction": positive / n if n else np.nan,
                }
            )
    return pd.DataFrame(rows)


def two_step_table(
    gate_pred: pd.DataFrame,
    thresholds: pd.DataFrame,
    tumor_confidence: pd.DataFrame,
    threshold_rule: str,
) -> pd.DataFrame:
    tumor_correct = tumor_confidence[["sample_id", "minicpg_correct_tumor_type"]].copy()
    rows = []
    for model_name, model_pred in gate_pred.groupby("model", sort=False):
        threshold_row = thresholds.loc[
            thresholds["model"].eq(model_name) & thresholds["threshold_rule"].eq(threshold_rule)
        ]
        if threshold_row.empty:
            continue
        threshold = float(threshold_row.iloc[0]["threshold"])
        tmp = model_pred.copy()
        tmp["gate_positive"] = (tmp["tumor_likeness_score"] >= threshold).astype(int)
        tumor = tmp.loc[tmp["binary_label"].eq(1)].merge(tumor_correct, on="sample_id", how="left")
        normal = tmp.loc[tmp["binary_label"].eq(0)].copy()
        retained_tumors = tumor.loc[tumor["gate_positive"].eq(1)]
        rows.append(
            {
                "model": model_name,
                "threshold_rule": threshold_rule,
                "test_tumors_total": int(len(tumor)),
                "test_tumors_gate_positive": int(tumor["gate_positive"].sum()),
                "test_tumor_sensitivity": float(tumor["gate_positive"].mean()) if len(tumor) else np.nan,
                "retained_tumors_minicpg_correct": int(retained_tumors["minicpg_correct_tumor_type"].sum()),
                "retained_tumors_minicpg_accuracy": float(retained_tumors["minicpg_correct_tumor_type"].mean())
                if len(retained_tumors)
                else np.nan,
                "all_tumors_gate_and_minicpg_correct": int(
                    (tumor["gate_positive"] * tumor["minicpg_correct_tumor_type"].fillna(0)).sum()
                ),
                "all_tumors_gate_and_minicpg_correct_fraction": float(
                    (tumor["gate_positive"] * tumor["minicpg_correct_tumor_type"].fillna(0)).mean()
                )
                if len(tumor)
                else np.nan,
                "normal_total": int(len(normal)),
                "normal_gate_positive": int(normal["gate_positive"].sum()),
                "normal_false_positive_fraction": float(normal["gate_positive"].mean()) if len(normal) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    active = validate_active_run(args.active_manifest)
    from minicpg.analysis_io import AnalysisInputs
    bound = AnalysisInputs(args.active_manifest)
    for key, value in {"dev_tumor": args.dev_tumor, "test_tumor": args.test_tumor,
                       "normal": args.normal, "model_bundle": args.model_bundle}.items():
        if Path(value).resolve() != bound.files[key]:
            raise ValueError(f"Argument differs from bound input: {key}")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    locked_model, probes, class_order = load_locked_model(args.model_bundle)
    dev_tumor_all = read_tsv(args.dev_tumor)
    test_tumor_all = read_tsv(args.test_tumor)
    normal_all = read_tsv(args.normal, dtype=str)
    normal_all["locked_model_normal_primary_subset"] = normal_all["locked_model_normal_primary_subset"].astype(str)
    normal_all["has_development_tumor"] = normal_all["has_development_tumor"].astype(str)
    normal_all["project_id"] = normal_all["project_id"].astype(str)

    train_normal = normal_all.loc[normal_all["has_development_tumor"].eq("1")].copy()
    test_normal = normal_all.loc[normal_all["locked_model_normal_primary_subset"].eq("1")].copy()
    represented_classes = sorted(set(test_normal["project_id"]) & set(test_tumor_all["project_id"]))
    train_classes = sorted(set(represented_classes) & set(train_normal["project_id"]) & set(dev_tumor_all["project_id"]))
    if not train_classes:
        raise SystemExit("No represented classes available for tumor-gate training")

    dev_tumor = dev_tumor_all.loc[dev_tumor_all["project_id"].isin(train_classes)].copy()
    test_tumor = test_tumor_all.loc[test_tumor_all["project_id"].isin(train_classes)].copy()
    train_normal = train_normal.loc[train_normal["project_id"].isin(train_classes)].copy()
    test_normal = test_normal.loc[test_normal["project_id"].isin(train_classes)].copy()

    train = pd.concat(
        [
            make_binary_frame(dev_tumor, probes, label=1, split="train_dev_tumor"),
            make_binary_frame(train_normal, probes, label=0, split="train_normal_adjacent_development_matched"),
        ],
        ignore_index=True,
    )
    test = pd.concat(
        [
            make_binary_frame(test_tumor, probes, label=1, split="test_heldout_tumor"),
            make_binary_frame(test_normal, probes, label=0, split="test_normal_adjacent_primary"),
        ],
        ignore_index=True,
    )

    gate_metrics, gate_predictions, gate_roc, gate_pr, gate_thresholds = evaluate_gate_models(
        train, test, probes, seed=args.seed
    )
    (
        calibrated_thresholds,
        calibrated_threshold_summary,
        calibrated_test_metrics,
        calibrated_predictions,
        calibrated_per_class,
    ) = repeated_threshold_calibration(
        train,
        test,
        probes,
        seed=args.seed,
        n_repeats=args.n_threshold_repeats,
        calibration_size=args.threshold_calibration_size,
        target_sensitivity=args.target_sensitivity,
    )

    tumor_conf = minicpg_confidence_rows(test_tumor, locked_model, probes, class_order, "heldout_tumor")
    normal_conf = normal_confidence_rows(test_normal, "normal_adjacent_primary")
    confidence = pd.concat([tumor_conf, normal_conf], ignore_index=True)
    confidence["binary_label"] = confidence["sample_kind"].eq("heldout_tumor").astype(int)
    confidence_metrics, confidence_roc, confidence_pr = evaluate_confidence_rejection(confidence)

    per_class = per_class_table(gate_predictions, gate_thresholds, "train_sensitivity_0.95")
    two_step = two_step_table(gate_predictions, gate_thresholds, tumor_conf, "train_sensitivity_0.95")

    train_counts = train.groupby(["sample_kind", "project_id"], as_index=False).size().rename(columns={"size": "n"})
    test_counts = test.groupby(["sample_kind", "project_id"], as_index=False).size().rename(columns={"size": "n"})

    outputs = {
        "train_manifest": out_dir / "tumor_gate_train_samples.tsv",
        "test_manifest": out_dir / "tumor_gate_test_samples.tsv",
        "gate_metrics": out_dir / "tumor_gate_binary_model_metrics.tsv",
        "gate_predictions": out_dir / "tumor_gate_binary_model_predictions.tsv",
        "gate_thresholds": out_dir / "tumor_gate_threshold_metrics.tsv",
        "gate_roc": out_dir / "tumor_gate_roc_curve.tsv",
        "gate_pr": out_dir / "tumor_gate_precision_recall_curve.tsv",
        "confidence_metrics": out_dir / "minicpg_confidence_rejection_metrics.tsv",
        "confidence_scores": out_dir / "minicpg_confidence_rejection_scores.tsv",
        "confidence_roc": out_dir / "minicpg_confidence_rejection_roc_curve.tsv",
        "confidence_pr": out_dir / "minicpg_confidence_rejection_precision_recall_curve.tsv",
        "per_class": out_dir / "tumor_gate_per_class_threshold_0.95_train_sensitivity.tsv",
        "two_step": out_dir / "tumor_gate_two_step_workflow_summary.tsv",
        "calibrated_thresholds": out_dir / f"tumor_gate_repeated_threshold_calibration_{args.target_sensitivity:.2f}_sensitivity.tsv",
        "calibrated_threshold_summary": out_dir / "tumor_gate_repeated_threshold_calibration_summary.tsv",
        "calibrated_test_metrics": out_dir / "tumor_gate_repeated_threshold_calibrated_test_metrics.tsv",
        "calibrated_predictions": out_dir / "tumor_gate_repeated_threshold_calibrated_predictions.tsv",
        "calibrated_per_class": out_dir / "tumor_gate_repeated_threshold_calibrated_per_class.tsv",
        "summary": out_dir / "tumor_gate_summary.json",
        "readme": out_dir / "tumor_gate_summary.md",
    }

    train.to_csv(outputs["train_manifest"], sep="\t", index=False)
    test.to_csv(outputs["test_manifest"], sep="\t", index=False)
    gate_metrics.to_csv(outputs["gate_metrics"], sep="\t", index=False)
    gate_predictions.to_csv(outputs["gate_predictions"], sep="\t", index=False)
    gate_thresholds.to_csv(outputs["gate_thresholds"], sep="\t", index=False)
    gate_roc.to_csv(outputs["gate_roc"], sep="\t", index=False)
    gate_pr.to_csv(outputs["gate_pr"], sep="\t", index=False)
    confidence_metrics.to_csv(outputs["confidence_metrics"], sep="\t", index=False)
    confidence.to_csv(outputs["confidence_scores"], sep="\t", index=False)
    confidence_roc.to_csv(outputs["confidence_roc"], sep="\t", index=False)
    confidence_pr.to_csv(outputs["confidence_pr"], sep="\t", index=False)
    per_class.to_csv(outputs["per_class"], sep="\t", index=False)
    two_step.to_csv(outputs["two_step"], sep="\t", index=False)
    calibrated_thresholds.to_csv(outputs["calibrated_thresholds"], sep="\t", index=False)
    calibrated_threshold_summary.to_csv(outputs["calibrated_threshold_summary"], sep="\t", index=False)
    calibrated_test_metrics.to_csv(outputs["calibrated_test_metrics"], sep="\t", index=False)
    calibrated_predictions.to_csv(outputs["calibrated_predictions"], sep="\t", index=False)
    calibrated_per_class.to_csv(outputs["calibrated_per_class"], sep="\t", index=False)

    best_gate = gate_metrics.sort_values(["auroc", "auprc"], ascending=False).iloc[0].to_dict()
    best_conf = confidence_metrics.sort_values(["auroc", "auprc"], ascending=False).iloc[0].to_dict()
    summary = {
        "active_locked_run_status": active.get("status"),
        "locked_run_name": active.get("locked_run_name"),
        "n_locked_probes": len(probes),
        "represented_classes": train_classes,
        "n_represented_classes": len(train_classes),
        "train_counts": train_counts.to_dict(orient="records"),
        "test_counts": test_counts.to_dict(orient="records"),
        "binary_gate_metrics": gate_metrics.to_dict(orient="records"),
        "confidence_rejection_metrics": confidence_metrics.to_dict(orient="records"),
        "best_binary_gate_by_auroc": best_gate,
        "best_minicpg_confidence_rejection_by_auroc": best_conf,
        "two_step_threshold_rule": "train_sensitivity_0.95",
        "two_step_summary": two_step.to_dict(orient="records"),
        "repeated_threshold_calibration": {
            "n_repeats": args.n_threshold_repeats,
            "calibration_size": args.threshold_calibration_size,
            "target_sensitivity": args.target_sensitivity,
            "threshold_summary": calibrated_threshold_summary.to_dict(orient="records"),
            "heldout_test_metrics": calibrated_test_metrics.to_dict(orient="records"),
        },
        "outputs": {k: str(v) for k, v in outputs.items() if k not in {"summary", "readme"}},
        "interpretation_scope": (
            "Tumor-vs-normal-adjacent proof of concept only; not healthy no-cancer screening "
            "or benign-disease specificity."
        ),
    }
    outputs["summary"].write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# Tumor-Gate Proof-of-Concept Summary",
        "",
        f"- Locked run validation: `{summary['active_locked_run_status']}`",
        f"- Locked probes: `{len(probes)}`",
        f"- Represented classes: `{len(train_classes)}`",
        f"- Training samples: `{len(train)}` (`{int(train['binary_label'].sum())}` tumors, `{int((1 - train['binary_label']).sum())}` normal-adjacent)",
        f"- Test samples: `{len(test)}` (`{int(test['binary_label'].sum())}` tumors, `{int((1 - test['binary_label']).sum())}` normal-adjacent)",
        "",
        "## Binary Gate Models",
        "",
    ]
    for row in gate_metrics.sort_values("auroc", ascending=False).itertuples(index=False):
        lines.append(f"- `{row.model}`: AUROC `{row.auroc:.4f}`, AUPRC `{row.auprc:.4f}`")
    lines.extend(["", "## Locked MiniCpG Confidence Rejection", ""])
    for row in confidence_metrics.sort_values("auroc", ascending=False).itertuples(index=False):
        lines.append(f"- `{row.model}`: AUROC `{row.auroc:.4f}`, AUPRC `{row.auprc:.4f}`")
    lines.extend(["", "## Repeated 95% Sensitivity Threshold Calibration", ""])
    for row in calibrated_test_metrics.sort_values("balanced_accuracy", ascending=False).itertuples(index=False):
        lines.append(
            f"- `{row.model}`: threshold `{row.threshold:.4f}`, held-out sensitivity `{row.sensitivity:.4f}`, "
            f"specificity `{row.specificity:.4f}`, balanced accuracy `{row.balanced_accuracy:.4f}`"
        )
    lines.extend(["", "## Scope Caveat", "", summary["interpretation_scope"]])
    outputs["readme"].write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
