#!/usr/bin/env python3
"""Compare classifiers on pre-selected fold-specific probes (no re-selection)."""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from itertools import product
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from minicpg.release_registry import (
    MODEL_ARG_ALIASES,
    RegistryEntryNotFound as PluginNotFound,
    call_with_supported_kwargs,
    load_model_plugin,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark classifiers using pre-selected probes for one outer fold."
    )
    parser.add_argument("--probe-matrix", required=True, help="Unfiltered probe matrix parquet.")
    parser.add_argument("--manifest", required=True, help="Classification manifest TSV (min100).")
    parser.add_argument("--folds", required=True, help="Outer folds TSV.")
    parser.add_argument("--selected-run-dir", required=True, help="Run dir containing selected_probes files.")
    parser.add_argument("--selection-method", default="mrmr", help="Selection method folder name.")
    parser.add_argument("--k", type=int, default=30, help="Probe count used for selected probes file.")
    parser.add_argument("--out-dir", required=True, help="Output directory for model benchmark.")
    parser.add_argument("--repeat", type=int, required=True, help="Repeat index.")
    parser.add_argument("--fold", type=int, required=True, help="Outer fold index.")
    parser.add_argument(
        "--candidate-spec-json",
        default="",
        help="Optional candidate spec JSON path for freeform model args.",
    )
    parser.add_argument(
        "--models",
        default="lr_multinomial,svm_linear,xgb_shallow,mlp_shallow",
        help=(
            "Comma-separated: "
            "lr_multinomial,svm_linear,xgb_shallow,mlp_shallow,"
            "extra_trees,random_forest,hist_gradient_boosting,gaussian_nb,"
            "linear_svm,elastic_net_multinomial_logistic"
        ),
    )
    parser.add_argument(
        "--class-weight-mode",
        default="none",
        choices=["none", "balanced"],
        help="Training class-weight mode for models that support it.",
    )
    parser.add_argument(
        "--tune-metric",
        default="weighted_f1",
        choices=["macro_f1", "weighted_f1", "accuracy", "balanced_accuracy"],
        help="Metric optimized in inner CV tuning.",
    )
    parser.add_argument("--inner-folds", type=int, default=3, help="Inner CV folds for model tuning.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument(
        "--lr-c-grid",
        default="0.01,0.03,0.1,0.3,1,3,10,30",
        help="C grid for LR models.",
    )
    parser.add_argument("--lr-l1-grid", default="0.1,0.5,0.9", help="l1_ratio grid for LR elastic-net.")
    parser.add_argument("--svm-c-grid", default="0.01,0.03,0.1,0.3,1,3,10,30", help="C grid for linear SVM.")
    parser.add_argument("--xgb-max-depth-grid", default="2,3,4", help="XGBoost max_depth grid.")
    parser.add_argument("--xgb-n-estimators-grid", default="120,240", help="XGBoost n_estimators grid.")
    parser.add_argument("--xgb-learning-rate-grid", default="0.05,0.1", help="XGBoost learning_rate grid.")
    parser.add_argument("--xgb-subsample", type=float, default=0.8, help="XGBoost subsample.")
    parser.add_argument(
        "--xgb-colsample-bytree", type=float, default=0.8, help="XGBoost colsample_bytree."
    )
    parser.add_argument("--xgb-min-child-weight", type=float, default=1.0, help="XGBoost min_child_weight.")
    parser.add_argument("--xgb-reg-lambda", type=float, default=1.0, help="XGBoost reg_lambda.")
    parser.add_argument("--et-n-estimators-grid", default="200,400", help="ExtraTrees n_estimators grid.")
    parser.add_argument(
        "--et-max-features-grid",
        default="sqrt,0.2,0.35",
        help="ExtraTrees max_features grid.",
    )
    parser.add_argument(
        "--et-max-depth-grid",
        default="0,10,20",
        help="ExtraTrees max_depth grid (use 0 for None).",
    )
    parser.add_argument(
        "--et-min-samples-leaf-grid",
        default="1,3,5",
        help="ExtraTrees min_samples_leaf grid.",
    )
    parser.add_argument("--rf-n-estimators-grid", default="200,400", help="RandomForest n_estimators grid.")
    parser.add_argument(
        "--rf-max-features-grid",
        default="sqrt,0.2,0.35",
        help="RandomForest max_features grid.",
    )
    parser.add_argument(
        "--rf-max-depth-grid",
        default="0,10,20",
        help="RandomForest max_depth grid (use 0 for None).",
    )
    parser.add_argument(
        "--rf-min-samples-leaf-grid",
        default="1,3,5",
        help="RandomForest min_samples_leaf grid.",
    )
    parser.add_argument(
        "--hgb-learning-rate-grid",
        default="0.03,0.1",
        help="HistGradientBoosting learning_rate grid.",
    )
    parser.add_argument(
        "--hgb-max-iter-grid",
        default="100,200",
        help="HistGradientBoosting max_iter grid.",
    )
    parser.add_argument(
        "--hgb-max-depth-grid",
        default="0,3,6",
        help="HistGradientBoosting max_depth grid (use 0 for None).",
    )
    parser.add_argument(
        "--hgb-max-leaf-nodes-grid",
        default="15,31",
        help="HistGradientBoosting max_leaf_nodes grid.",
    )
    parser.add_argument(
        "--hgb-min-samples-leaf-grid",
        default="20",
        help="HistGradientBoosting min_samples_leaf grid.",
    )
    parser.add_argument(
        "--gnb-var-smoothing-grid",
        default="1e-9,1e-8,1e-7",
        help="GaussianNB var_smoothing grid.",
    )
    parser.add_argument(
        "--mlp-hidden-layer-grid",
        default="64,128",
        help="MLP hidden layer sizes (single-layer width values).",
    )
    parser.add_argument("--mlp-alpha-grid", default="1e-5,1e-4,1e-3", help="MLP alpha grid.")
    parser.add_argument("--mlp-learning-rate-init-grid", default="1e-3,3e-4", help="MLP learning_rate_init grid.")
    parser.add_argument("--mlp-max-iter", type=int, default=400, help="MLP max_iter.")
    parser.add_argument("--mlp-early-stopping", action="store_true", help="Enable MLP early stopping.")
    parser.add_argument("--n-jobs", type=int, default=1, help="Parallel workers for XGBoost/ExtraTrees.")
    parser.add_argument("--compute-auroc", action="store_true", help="Compute macro/per-class AUROC.")
    parser.add_argument(
        "--fit-decision-policy",
        action="store_true",
        help="Fit a post-probability decision policy from dev OOF probabilities before test evaluation.",
    )
    parser.add_argument(
        "--decision-policy-metric",
        default="",
        choices=["", "macro_f1", "weighted_f1", "accuracy", "balanced_accuracy"],
        help="Metric optimized when fitting the decision policy. Defaults to --tune-metric.",
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
    parser.add_argument("--run-id", default="", help="Run identifier for heartbeat tracking.")
    parser.add_argument(
        "--heartbeat-path",
        default="",
        help="Path to JSONL heartbeat file for cluster monitoring.",
    )
    parser.add_argument(
        "--heartbeat-interval",
        type=float,
        default=30.0,
        help="Heartbeat emission interval in seconds.",
    )
    return parser.parse_args()


def _safe_heartbeat_emit(
    *,
    run_id: str,
    heartbeat_path: str,
    heartbeat_interval: float,
) -> callable:
    run_id = str(run_id).strip()
    heartbeat_path = str(heartbeat_path).strip()
    if not run_id or not heartbeat_path:
        def no_emit(*_: object, **__: object) -> None:
            return

        return no_emit

    interval = max(1.0, float(heartbeat_interval))
    path = Path(heartbeat_path)
    timer = {"last_ts": 0.0}

    def emit(*, event: str, status: str, force: bool = False, **extra: object) -> None:
        now = time.time()
        if not force and now - timer["last_ts"] < interval:
            return
        timer["last_ts"] = now
        payload = {
            "time": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now)),
            "ts": now,
            "run_id": run_id,
            "status": status,
            "event": event,
            "extra": extra,
        }
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload) + "\n")
        except Exception:
            return

    return emit


def parse_list_floats(csv_text: str) -> List[float]:
    return [float(x.strip()) for x in re.split(r"[,\s:]+", str(csv_text)) if x.strip()]


def parse_list_ints(csv_text: str) -> List[int]:
    return [int(x.strip()) for x in re.split(r"[,\s:]+", str(csv_text)) if x.strip()]


def parse_list_ints_or_none(csv_text: str) -> List[int | None]:
    vals: List[int | None] = []
    for token in re.split(r"[,\s:]+", str(csv_text)):
        t = token.strip().lower()
        if not t:
            continue
        if t in {"0", "none", "null", "na"}:
            vals.append(None)
        else:
            vals.append(int(t))
    return vals


def parse_list_mixed_max_features(csv_text: str) -> List[str | float]:
    vals: List[str | float] = []
    for token in re.split(r"[,\s:]+", str(csv_text)):
        t = token.strip()
        if not t:
            continue
        lower = t.lower()
        if lower in {"sqrt", "log2", "auto"}:
            vals.append(lower)
            continue
        vals.append(float(t))
    return vals


def load_candidate_spec(path_text: str) -> Dict[str, object]:
    raw = str(path_text).strip()
    if not raw:
        return {}
    path = Path(raw)
    if not path.exists():
        raise FileNotFoundError(f"candidate_spec_json not found: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("candidate_spec_json must contain a JSON object")
    return value


def canonical_model_name(model_name: str) -> str:
    name = str(model_name).strip()
    aliases = {
        "linear_svm": "svm_linear",
        "elastic_net_multinomial_logistic": "lr_multinomial",
    }
    return aliases.get(name, name)


def merge_model_args(spec: Dict[str, object], model_name: str) -> Dict[str, object]:
    root = spec.get("model_args", {})
    if not isinstance(root, dict):
        return {}
    requested_name = str(model_name).strip()
    canonical_name = canonical_model_name(requested_name)
    known_models = {
        "lr_multinomial",
        "svm_linear",
        "xgb_shallow",
        "mlp_shallow",
        "extra_trees",
        "random_forest",
        "hist_gradient_boosting",
        "gaussian_nb",
        "linear_svm",
        "elastic_net_multinomial_logistic",
        "global",
    }
    merged = {
        str(k): v
        for k, v in root.items()
        if str(k) not in known_models and str(k) not in {requested_name, canonical_name}
    }
    global_args = root.get("global", {})
    if isinstance(global_args, dict):
        merged.update({str(k): v for k, v in global_args.items()})
    for key in [canonical_name, requested_name]:
        local_args = root.get(key, {})
        if isinstance(local_args, dict):
            merged.update({str(k): v for k, v in local_args.items()})
    if requested_name == "elastic_net_multinomial_logistic":
        merged.setdefault("penalties", ["elasticnet"])
    if requested_name == "linear_svm":
        merged.setdefault("class_weight", "balanced")
    return merged


def tune_plugin_model(
    model_name: str,
    *,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    all_classes: Sequence[str],
    tune_metric: str,
    class_weight_mode: str,
    inner_folds: int,
    seed: int,
    n_jobs: int,
    model_args: Dict[str, object],
) -> Tuple[object, pd.DataFrame, Dict[str, object]]:
    plugin = load_model_plugin(model_name)
    result = call_with_supported_kwargs(
        plugin,
        {
            "model_name": model_name,
            "X_train": X_train.copy(),
            "y_train": np.array(y_train, copy=True),
            "all_classes": list(all_classes),
            "tune_metric": str(tune_metric),
            "class_weight_mode": str(class_weight_mode),
            "inner_folds": int(inner_folds),
            "seed": int(seed),
            "n_jobs": int(n_jobs),
            "model_args": dict(model_args),
        },
        aliases=MODEL_ARG_ALIASES,
    )
    if not isinstance(result, dict):
        raise ValueError(f"Model plugin {model_name} must return a dict")
    model = result.get("model", result.get("estimator"))
    inner_df = result.get("inner_df", result.get("cv_results", pd.DataFrame()))
    if isinstance(inner_df, dict):
        inner_df = pd.DataFrame(inner_df)
    elif isinstance(inner_df, list):
        inner_df = pd.DataFrame(inner_df)
    best_params = result.get("best_params", result.get("params", {}))
    if model is None:
        raise ValueError(f"Model plugin {model_name} returned no fitted model")
    if not isinstance(inner_df, pd.DataFrame):
        raise ValueError(f"Model plugin {model_name} returned invalid inner_df")
    if not isinstance(best_params, dict):
        best_params = {}
    return model, inner_df, best_params


def score_metric(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    all_classes: Sequence[str],
    metric: str,
) -> float:
    if metric == "macro_f1":
        return float(f1_score(y_true, y_pred, average="macro", labels=list(all_classes), zero_division=0))
    if metric == "weighted_f1":
        return float(f1_score(y_true, y_pred, average="weighted", labels=list(all_classes), zero_division=0))
    if metric == "accuracy":
        return float(accuracy_score(y_true, y_pred))
    if metric == "balanced_accuracy":
        return float(balanced_accuracy_score(y_true, y_pred))
    raise ValueError(f"Unsupported tune metric: {metric}")


def balanced_sample_weights(y: np.ndarray) -> np.ndarray:
    classes, counts = np.unique(y, return_counts=True)
    n = float(len(y))
    n_classes = float(len(classes))
    per_class = {cls: n / (n_classes * float(cnt)) for cls, cnt in zip(classes, counts)}
    return np.array([per_class[v] for v in y], dtype=np.float64)


def read_matrix_rows(
    pf: pq.ParquetFile,
    sample_ids: Sequence[str],
    probe_set: set[str],
) -> Iterable[pd.DataFrame]:
    cols = list(sample_ids) + ["probe_id"]
    for rg in range(pf.num_row_groups):
        table = pf.read_row_group(rg, columns=cols)
        df = table.to_pandas()
        sub = df[df["probe_id"].isin(probe_set)]
        if not sub.empty:
            yield sub


def read_selected(pf: pq.ParquetFile, sample_ids: Sequence[str], probes: Sequence[str]) -> pd.DataFrame:
    probe_set = set(probes)
    frames = []
    for df in read_matrix_rows(pf, sample_ids, probe_set):
        frames.append(df)
    if not frames:
        raise ValueError("No selected probes found in matrix")
    data = pd.concat(frames, ignore_index=True)
    data = data.set_index("probe_id")
    data = data.loc[list(probes)]
    X = data[list(sample_ids)].T
    X.index.name = "sample_id"
    return X


def align_proba(proba: np.ndarray, model_classes: Sequence[str], all_classes: Sequence[str]) -> np.ndarray:
    idx_map = {str(c): i for i, c in enumerate(model_classes)}
    out = np.zeros((proba.shape[0], len(all_classes)), dtype=np.float64)
    for j, cls in enumerate(all_classes):
        if cls in idx_map:
            out[:, j] = proba[:, idx_map[cls]]
    return out


def _softmax(z: np.ndarray) -> np.ndarray:
    zmax = np.max(z, axis=1, keepdims=True)
    e = np.exp(z - zmax)
    return e / np.clip(np.sum(e, axis=1, keepdims=True), 1e-12, None)


def _proba_to_logits(p: np.ndarray) -> np.ndarray:
    return np.log(np.clip(p, 1e-12, 1.0))


def apply_temp_policy(p: np.ndarray, temperature: float) -> np.ndarray:
    return _softmax(_proba_to_logits(p) / float(temperature))


def apply_offset_policy(p: np.ndarray, offset: np.ndarray, alpha: float) -> np.ndarray:
    return _softmax(_proba_to_logits(p) + float(alpha) * offset.reshape(1, -1))


def apply_temp_offset_policy(p: np.ndarray, temperature: float, offset: np.ndarray, alpha: float) -> np.ndarray:
    logits = (_proba_to_logits(p) / float(temperature)) + float(alpha) * offset.reshape(1, -1)
    return _softmax(logits)


def argmax_labels_from_proba(proba: np.ndarray, classes: Sequence[str]) -> np.ndarray:
    idx = np.argmax(proba, axis=1)
    return np.array([str(classes[int(i)]) for i in idx], dtype=object)


def parse_decision_policy_modes(text: str) -> list[str]:
    allowed = {"argmax", "temp", "offset", "temp_offset"}
    modes = [str(x).strip() for x in re.split(r"[,\s:]+", str(text)) if str(x).strip()]
    if not modes:
        return ["argmax"]
    out: list[str] = []
    seen: set[str] = set()
    for mode in modes:
        if mode not in allowed:
            raise ValueError(f"Unsupported decision policy mode: {mode}")
        if mode in seen:
            continue
        seen.add(mode)
        out.append(mode)
    if "argmax" not in seen:
        out.insert(0, "argmax")
    return out


def _extract_model_classes(fitted_model: object) -> list[str]:
    classes = None
    named_steps = getattr(fitted_model, "named_steps", None)
    if isinstance(named_steps, dict) and "clf" in named_steps:
        classes = getattr(named_steps["clf"], "classes_", None)
    if classes is None:
        classes = getattr(fitted_model, "classes_", None)
    if classes is None:
        raise ValueError("Fitted model does not expose classes_ needed for probability alignment")
    return [str(c) for c in classes]


def fit_model_and_predict_raw_proba(
    *,
    model_name: str,
    model: object,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_pred: pd.DataFrame,
    all_classes: Sequence[str],
    class_weight_mode: str = "none",
) -> tuple[object, np.ndarray]:
    fitted_model = clone(model)
    if model_name == "xgb_shallow":
        class_to_idx = {c: i for i, c in enumerate(all_classes)}
        y_train_fit = np.array([class_to_idx[c] for c in y_train], dtype=np.int32)
        if class_weight_mode == "balanced":
            fitted_model.fit(X_train, y_train_fit, clf__sample_weight=balanced_sample_weights(y_train_fit))
        else:
            fitted_model.fit(X_train, y_train_fit)
        proba_raw = np.asarray(fitted_model.predict_proba(X_pred), dtype=np.float64)
        clf_classes = getattr(fitted_model.named_steps["clf"], "classes_", [])
        model_classes = [str(all_classes[int(c)]) for c in clf_classes]
        proba_aligned = align_proba(proba_raw, model_classes, all_classes)
        return fitted_model, proba_aligned

    fitted_model.fit(X_train, y_train)
    proba_raw = np.asarray(fitted_model.predict_proba(X_pred), dtype=np.float64)
    model_classes = _extract_model_classes(fitted_model)
    proba_aligned = align_proba(proba_raw, model_classes, all_classes)
    return fitted_model, proba_aligned


def build_oof_raw_proba(
    *,
    model_name: str,
    model: object,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    all_classes: Sequence[str],
    class_weight_mode: str,
    inner_folds: int,
    seed: int,
) -> np.ndarray:
    splitter = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    out = np.zeros((X_train.shape[0], len(all_classes)), dtype=np.float64)
    for tr_idx, va_idx in splitter.split(X_train, y_train):
        _, p_va = fit_model_and_predict_raw_proba(
            model_name=model_name,
            model=model,
            X_train=X_train.iloc[tr_idx],
            y_train=y_train[tr_idx],
            X_pred=X_train.iloc[va_idx],
            all_classes=all_classes,
            class_weight_mode=class_weight_mode,
        )
        out[va_idx, :] = p_va
    return out


def apply_decision_policy_proba(
    raw_proba: np.ndarray,
    decision_policy: Dict[str, object] | None,
) -> np.ndarray:
    if not decision_policy:
        return np.asarray(raw_proba, dtype=np.float64)
    mode = str(decision_policy.get("type", "argmax")).strip() or "argmax"
    params = decision_policy.get("params", {})
    if not isinstance(params, dict):
        params = {}
    param_source: Dict[str, object] = dict(params)
    for key, value in decision_policy.items():
        if key == "params":
            continue
        param_source.setdefault(key, value)
    if mode == "argmax":
        return np.asarray(raw_proba, dtype=np.float64)
    if mode == "temp":
        return apply_temp_policy(raw_proba, float(param_source.get("temperature", 1.0)))
    if mode == "offset":
        offset = np.asarray(list(param_source.get("offset", []) or []), dtype=np.float64)
        if offset.size != raw_proba.shape[1]:
            raise ValueError("Decision policy offset length does not match class count")
        return apply_offset_policy(raw_proba, offset, float(param_source.get("alpha", 0.0)))
    if mode == "temp_offset":
        offset = np.asarray(list(param_source.get("offset", []) or []), dtype=np.float64)
        if offset.size != raw_proba.shape[1]:
            raise ValueError("Decision policy offset length does not match class count")
        return apply_temp_offset_policy(
            raw_proba,
            float(param_source.get("temperature", 1.0)),
            offset,
            float(param_source.get("alpha", 0.0)),
        )
    raise ValueError(f"Unsupported decision policy type: {mode}")


def fit_decision_policy(
    *,
    y_true: np.ndarray,
    raw_proba_oof: np.ndarray,
    all_classes: Sequence[str],
    metric: str,
    modes: Sequence[str],
    min_gain: float,
) -> Dict[str, object]:
    classes = list(all_classes)

    def _score(policy: Dict[str, object]) -> tuple[float, float, float, float]:
        adjusted = apply_decision_policy_proba(raw_proba_oof, policy)
        y_pred = argmax_labels_from_proba(adjusted, classes)
        primary = score_metric(y_true, y_pred, classes, metric)
        macro = score_metric(y_true, y_pred, classes, "macro_f1")
        weighted = score_metric(y_true, y_pred, classes, "weighted_f1")
        acc = score_metric(y_true, y_pred, classes, "accuracy")
        return primary, macro, weighted, acc

    prior = np.bincount(np.array([classes.index(v) for v in y_true], dtype=np.int32), minlength=len(classes)).astype(float)
    prior = prior / np.clip(prior.sum(), 1.0, None)
    mean_p = np.mean(np.clip(raw_proba_oof, 1e-12, 1.0), axis=0)
    base_offset = np.log(np.clip(prior, 1e-12, 1.0)) - np.log(np.clip(mean_p, 1e-12, 1.0))

    candidates: list[Dict[str, object]] = [{"type": "argmax"}]
    mode_set = set(modes)
    if "temp" in mode_set:
        for temp in np.arange(0.5, 2.51, 0.1):
            candidates.append({"type": "temp", "temperature": float(round(float(temp), 10))})
    if "offset" in mode_set:
        for alpha in np.arange(0.0, 2.01, 0.05):
            candidates.append(
                {
                    "type": "offset",
                    "offset": base_offset.astype(float).tolist(),
                    "alpha": float(round(float(alpha), 10)),
                }
            )
    if "temp_offset" in mode_set:
        for temp in np.arange(0.5, 2.51, 0.1):
            for alpha in np.arange(0.0, 2.01, 0.1):
                candidates.append(
                    {
                        "type": "temp_offset",
                        "temperature": float(round(float(temp), 10)),
                        "offset": base_offset.astype(float).tolist(),
                        "alpha": float(round(float(alpha), 10)),
                    }
                )

    raw_argmax_policy = {"type": "argmax"}
    best_policy = dict(raw_argmax_policy)
    best_scores = _score(best_policy)
    complexity_rank = {"argmax": 0, "temp": 1, "offset": 2, "temp_offset": 3}
    for policy in candidates[1:]:
        scores = _score(policy)
        if scores > best_scores:
            best_policy = policy
            best_scores = scores
            continue
        if scores == best_scores and complexity_rank.get(str(policy["type"]), 99) < complexity_rank.get(str(best_policy["type"]), 99):
            best_policy = policy
            best_scores = scores

    raw_argmax_scores = _score(raw_argmax_policy)
    candidate_policy = dict(best_policy)
    candidate_scores = best_scores
    selected_policy = dict(candidate_policy)
    selected_scores = candidate_scores
    selected_gain = float(candidate_scores[0] - raw_argmax_scores[0])
    guard_triggered = False
    if str(candidate_policy.get("type", "argmax")).strip() != "argmax" and selected_gain < float(min_gain):
        selected_policy = dict(raw_argmax_policy)
        selected_scores = raw_argmax_scores
        guard_triggered = True
        selected_gain = 0.0
    return {
        "type": str(selected_policy["type"]),
        "metric": metric,
        "params": selected_policy,
        "oof_metric_value": float(selected_scores[0]),
        "oof_macro_f1": float(selected_scores[1]),
        "oof_weighted_f1": float(selected_scores[2]),
        "oof_accuracy": float(selected_scores[3]),
        "oof_gain_vs_raw_argmax_metric": float(selected_gain),
        "raw_argmax_oof_metric_value": float(raw_argmax_scores[0]),
        "raw_argmax_oof_macro_f1": float(raw_argmax_scores[1]),
        "raw_argmax_oof_weighted_f1": float(raw_argmax_scores[2]),
        "raw_argmax_oof_accuracy": float(raw_argmax_scores[3]),
        "decision_policy_min_gain": float(min_gain),
        "decision_policy_guard_triggered": bool(guard_triggered),
        "candidate_type_before_guard": str(candidate_policy["type"]),
        "candidate_oof_metric_value_before_guard": float(candidate_scores[0]),
        "candidate_oof_macro_f1_before_guard": float(candidate_scores[1]),
        "candidate_oof_weighted_f1_before_guard": float(candidate_scores[2]),
        "candidate_oof_accuracy_before_guard": float(candidate_scores[3]),
        "candidate_gain_vs_raw_argmax_metric_before_guard": float(candidate_scores[0] - raw_argmax_scores[0]),
    }


def compute_auc_by_class(
    y_true: np.ndarray,
    proba_aligned: np.ndarray,
    all_classes: Sequence[str],
) -> pd.DataFrame:
    rows = []
    for j, cls in enumerate(all_classes):
        y_bin = (y_true == cls).astype(int)
        n_pos = int(y_bin.sum())
        n_neg = int(len(y_bin) - n_pos)
        if n_pos == 0 or n_neg == 0:
            auc = float("nan")
        else:
            auc = float(roc_auc_score(y_bin, proba_aligned[:, j]))
        rows.append({"class": str(cls), "auc": auc, "n_pos": n_pos, "n_neg": n_neg})
    return pd.DataFrame(rows)


def tune_lr_multinomial(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    all_classes: Sequence[str],
    c_grid: Sequence[float],
    l1_grid: Sequence[float],
    inner_folds: int,
    seed: int,
    tune_metric: str,
    class_weight_mode: str,
    penalties: Sequence[str] | None = None,
) -> Tuple[Pipeline, pd.DataFrame, Dict[str, float]]:
    param_rows = []
    inner = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    best_score = -np.inf
    best_params: Dict[str, float | str] = {}

    grid: List[Dict[str, float | str]] = []
    penalty_set = {str(v).strip().lower() for v in (penalties or ["l2", "elasticnet"]) if str(v).strip()}
    if "l2" in penalty_set:
        for c in c_grid:
            grid.append({"penalty": "l2", "C": float(c)})
    if "elasticnet" in penalty_set:
        for c, l1 in product(c_grid, l1_grid):
            grid.append({"penalty": "elasticnet", "C": float(c), "l1_ratio": float(l1)})
    if not grid:
        raise ValueError("No logistic-regression penalties selected for tuning")

    for params in grid:
        fold_scores = []
        for tr_idx, va_idx in inner.split(X_train, y_train):
            X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
            y_tr, y_va = y_train[tr_idx], y_train[va_idx]
            clf_args = dict(
                max_iter=1500,
                solver="saga",
                multi_class="multinomial",
                penalty=str(params["penalty"]),
                C=float(params["C"]),
                n_jobs=1,
            )
            if class_weight_mode == "balanced":
                clf_args["class_weight"] = "balanced"
            if params["penalty"] == "elasticnet":
                clf_args["l1_ratio"] = float(params["l1_ratio"])
            model = Pipeline([("imputer", SimpleImputer(strategy="median")), ("clf", LogisticRegression(**clf_args))])
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_va)
            fold_scores.append(score_metric(y_va, y_pred, all_classes, tune_metric))

        mean_score = float(np.mean(fold_scores))
        row = {f"mean_{tune_metric}": mean_score, **params, "class_weight_mode": class_weight_mode}
        param_rows.append(row)
        if mean_score > best_score:
            best_score = mean_score
            best_params = dict(params)

    best_clf_args = dict(
        max_iter=1500,
        solver="saga",
        multi_class="multinomial",
        penalty=str(best_params["penalty"]),
        C=float(best_params["C"]),
        n_jobs=1,
    )
    if class_weight_mode == "balanced":
        best_clf_args["class_weight"] = "balanced"
    if best_params["penalty"] == "elasticnet":
        best_clf_args["l1_ratio"] = float(best_params["l1_ratio"])
    best_model = Pipeline(
        [("imputer", SimpleImputer(strategy="median")), ("clf", LogisticRegression(**best_clf_args))]
    )
    out_best = {k: float(v) if isinstance(v, (int, float)) else v for k, v in best_params.items()}
    out_best["class_weight_mode"] = class_weight_mode
    out_best["tune_metric"] = tune_metric
    return best_model, pd.DataFrame(param_rows), out_best


def tune_svm_linear(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    all_classes: Sequence[str],
    c_grid: Sequence[float],
    inner_folds: int,
    seed: int,
    tune_metric: str,
    class_weight_mode: str,
) -> Tuple[Pipeline, pd.DataFrame, Dict[str, float]]:
    param_rows = []
    inner = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    best_score = -np.inf
    best_c = None

    for c in c_grid:
        fold_scores = []
        for tr_idx, va_idx in inner.split(X_train, y_train):
            X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
            y_tr, y_va = y_train[tr_idx], y_train[va_idx]
            model = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "clf",
                        SVC(
                            kernel="linear",
                            C=float(c),
                            probability=True,
                            random_state=seed,
                            class_weight=("balanced" if class_weight_mode == "balanced" else None),
                        ),
                    ),
                ]
            )
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_va)
            fold_scores.append(score_metric(y_va, y_pred, all_classes, tune_metric))
        mean_score = float(np.mean(fold_scores))
        param_rows.append(
            {"C": float(c), f"mean_{tune_metric}": mean_score, "class_weight_mode": class_weight_mode}
        )
        if mean_score > best_score:
            best_score = mean_score
            best_c = float(c)

    best_model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "clf",
                SVC(
                    kernel="linear",
                    C=float(best_c),
                    probability=True,
                    random_state=seed,
                    class_weight=("balanced" if class_weight_mode == "balanced" else None),
                ),
            ),
        ]
    )
    return best_model, pd.DataFrame(param_rows), {
        "C": float(best_c),
        "class_weight_mode": class_weight_mode,
        "tune_metric": tune_metric,
    }


def tune_xgb_shallow(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    all_classes: Sequence[str],
    max_depth_grid: Sequence[int],
    n_estimators_grid: Sequence[int],
    learning_rate_grid: Sequence[float],
    subsample: float,
    colsample_bytree: float,
    min_child_weight: float,
    reg_lambda: float,
    inner_folds: int,
    seed: int,
    n_jobs: int,
    tune_metric: str,
    class_weight_mode: str,
) -> Tuple[Pipeline, pd.DataFrame, Dict[str, float]]:
    try:
        from xgboost import XGBClassifier
    except ModuleNotFoundError as exc:
        raise RuntimeError("xgboost is required for model=xgb_shallow but is not installed in this environment") from exc

    class_to_idx = {c: i for i, c in enumerate(all_classes)}
    y_train_int = np.array([class_to_idx[c] for c in y_train], dtype=np.int32)

    param_rows = []
    inner = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    best_score = -np.inf
    best_params: Dict[str, float | int] = {}

    for max_depth, n_estimators, learning_rate in product(max_depth_grid, n_estimators_grid, learning_rate_grid):
        fold_scores = []
        for tr_idx, va_idx in inner.split(X_train, y_train_int):
            X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
            y_tr, y_va = y_train_int[tr_idx], y_train_int[va_idx]
            model = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "clf",
                        XGBClassifier(
                            objective="multi:softprob",
                            num_class=len(all_classes),
                            max_depth=int(max_depth),
                            n_estimators=int(n_estimators),
                            learning_rate=float(learning_rate),
                            subsample=float(subsample),
                            colsample_bytree=float(colsample_bytree),
                            min_child_weight=float(min_child_weight),
                            reg_lambda=float(reg_lambda),
                            random_state=seed,
                            n_jobs=max(1, int(n_jobs)),
                            tree_method="hist",
                            eval_metric="mlogloss",
                        ),
                    ),
                ]
            )
            if class_weight_mode == "balanced":
                sw = balanced_sample_weights(y_tr)
                model.fit(X_tr, y_tr, clf__sample_weight=sw)
            else:
                model.fit(X_tr, y_tr)
            y_pred_int = model.predict(X_va).astype(int)
            y_pred = np.array([all_classes[i] for i in y_pred_int], dtype=object)
            y_va_lbl = np.array([all_classes[i] for i in y_va], dtype=object)
            fold_scores.append(score_metric(y_va_lbl, y_pred, all_classes, tune_metric))

        mean_score = float(np.mean(fold_scores))
        row = {
            "max_depth": int(max_depth),
            "n_estimators": int(n_estimators),
            "learning_rate": float(learning_rate),
            f"mean_{tune_metric}": mean_score,
            "class_weight_mode": class_weight_mode,
        }
        param_rows.append(row)
        if mean_score > best_score:
            best_score = mean_score
            best_params = {k: v for k, v in row.items() if not k.startswith("mean_")}

    best_model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "clf",
                XGBClassifier(
                    objective="multi:softprob",
                    num_class=len(all_classes),
                    max_depth=int(best_params["max_depth"]),
                    n_estimators=int(best_params["n_estimators"]),
                    learning_rate=float(best_params["learning_rate"]),
                    subsample=float(subsample),
                    colsample_bytree=float(colsample_bytree),
                    min_child_weight=float(min_child_weight),
                    reg_lambda=float(reg_lambda),
                    random_state=seed,
                    n_jobs=max(1, int(n_jobs)),
                    tree_method="hist",
                    eval_metric="mlogloss",
                ),
            ),
        ]
    )
    best_params = {
        "max_depth": int(best_params["max_depth"]),
        "n_estimators": int(best_params["n_estimators"]),
        "learning_rate": float(best_params["learning_rate"]),
        "subsample": float(subsample),
        "colsample_bytree": float(colsample_bytree),
        "min_child_weight": float(min_child_weight),
        "reg_lambda": float(reg_lambda),
        "class_weight_mode": class_weight_mode,
        "tune_metric": tune_metric,
    }
    return best_model, pd.DataFrame(param_rows), best_params


def tune_extra_trees(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    all_classes: Sequence[str],
    n_estimators_grid: Sequence[int],
    max_features_grid: Sequence[str | float],
    max_depth_grid: Sequence[int | None],
    min_samples_leaf_grid: Sequence[int],
    inner_folds: int,
    seed: int,
    n_jobs: int,
    tune_metric: str,
    class_weight: str | None = None,
) -> Tuple[Pipeline, pd.DataFrame, Dict[str, float | int | None | str]]:
    param_rows = []
    inner = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    best_score = -np.inf
    best_params: Dict[str, float | int | None] = {}

    for n_estimators, max_features, max_depth, min_samples_leaf in product(
        n_estimators_grid, max_features_grid, max_depth_grid, min_samples_leaf_grid
    ):
        fold_scores = []
        for tr_idx, va_idx in inner.split(X_train, y_train):
            X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
            y_tr, y_va = y_train[tr_idx], y_train[va_idx]
            model = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "clf",
                        ExtraTreesClassifier(
                            n_estimators=int(n_estimators),
                            max_features=max_features,
                            max_depth=None if max_depth is None else int(max_depth),
                            min_samples_leaf=int(min_samples_leaf),
                            random_state=seed,
                            n_jobs=max(1, int(n_jobs)),
                            class_weight=class_weight,
                        ),
                    ),
                ]
            )
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_va)
            fold_scores.append(score_metric(y_va, y_pred, all_classes, tune_metric))

        mean_score = float(np.mean(fold_scores))
        row = {
            "n_estimators": int(n_estimators),
            "max_features": max_features,
            "max_depth": None if max_depth is None else int(max_depth),
            "min_samples_leaf": int(min_samples_leaf),
            f"mean_{tune_metric}": mean_score,
            "class_weight": class_weight or "",
        }
        param_rows.append(row)
        if mean_score > best_score:
            best_score = mean_score
            best_params = {k: v for k, v in row.items() if not k.startswith("mean_")}

    best_model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "clf",
                ExtraTreesClassifier(
                    n_estimators=int(best_params["n_estimators"]),
                    max_features=best_params["max_features"],
                    max_depth=None if best_params["max_depth"] is None else int(best_params["max_depth"]),
                    min_samples_leaf=int(best_params["min_samples_leaf"]),
                    random_state=seed,
                    n_jobs=max(1, int(n_jobs)),
                    class_weight=class_weight,
                ),
            ),
        ]
    )
    return best_model, pd.DataFrame(param_rows), best_params


def tune_random_forest(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    all_classes: Sequence[str],
    n_estimators_grid: Sequence[int],
    max_features_grid: Sequence[str | float],
    max_depth_grid: Sequence[int | None],
    min_samples_leaf_grid: Sequence[int],
    inner_folds: int,
    seed: int,
    n_jobs: int,
    tune_metric: str,
    class_weight: str | None = None,
) -> Tuple[Pipeline, pd.DataFrame, Dict[str, float | int | None | str]]:
    param_rows = []
    inner = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    best_score = -np.inf
    best_params: Dict[str, float | int | None | str] = {}

    for n_estimators, max_features, max_depth, min_samples_leaf in product(
        n_estimators_grid, max_features_grid, max_depth_grid, min_samples_leaf_grid
    ):
        fold_scores = []
        for tr_idx, va_idx in inner.split(X_train, y_train):
            X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
            y_tr, y_va = y_train[tr_idx], y_train[va_idx]
            model = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "clf",
                        RandomForestClassifier(
                            n_estimators=int(n_estimators),
                            max_features=max_features,
                            max_depth=None if max_depth is None else int(max_depth),
                            min_samples_leaf=int(min_samples_leaf),
                            random_state=seed,
                            n_jobs=max(1, int(n_jobs)),
                            class_weight=class_weight,
                        ),
                    ),
                ]
            )
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_va)
            fold_scores.append(score_metric(y_va, y_pred, all_classes, tune_metric))

        mean_score = float(np.mean(fold_scores))
        row = {
            "n_estimators": int(n_estimators),
            "max_features": max_features,
            "max_depth": None if max_depth is None else int(max_depth),
            "min_samples_leaf": int(min_samples_leaf),
            f"mean_{tune_metric}": mean_score,
            "class_weight": class_weight or "",
        }
        param_rows.append(row)
        if mean_score > best_score:
            best_score = mean_score
            best_params = {k: v for k, v in row.items() if not k.startswith("mean_")}

    best_model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "clf",
                RandomForestClassifier(
                    n_estimators=int(best_params["n_estimators"]),
                    max_features=best_params["max_features"],
                    max_depth=None if best_params["max_depth"] is None else int(best_params["max_depth"]),
                    min_samples_leaf=int(best_params["min_samples_leaf"]),
                    random_state=seed,
                    n_jobs=max(1, int(n_jobs)),
                    class_weight=class_weight,
                ),
            ),
        ]
    )
    return best_model, pd.DataFrame(param_rows), best_params


def tune_hist_gradient_boosting(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    all_classes: Sequence[str],
    learning_rate_grid: Sequence[float],
    max_iter_grid: Sequence[int],
    max_depth_grid: Sequence[int | None],
    max_leaf_nodes_grid: Sequence[int],
    min_samples_leaf_grid: Sequence[int],
    inner_folds: int,
    seed: int,
    tune_metric: str,
) -> Tuple[Pipeline, pd.DataFrame, Dict[str, float | int | None]]:
    param_rows = []
    inner = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    best_score = -np.inf
    best_params: Dict[str, float | int | None] = {}

    for learning_rate, max_iter, max_depth, max_leaf_nodes, min_samples_leaf in product(
        learning_rate_grid,
        max_iter_grid,
        max_depth_grid,
        max_leaf_nodes_grid,
        min_samples_leaf_grid,
    ):
        fold_scores = []
        for tr_idx, va_idx in inner.split(X_train, y_train):
            X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
            y_tr, y_va = y_train[tr_idx], y_train[va_idx]
            model = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "clf",
                        HistGradientBoostingClassifier(
                            learning_rate=float(learning_rate),
                            max_iter=int(max_iter),
                            max_depth=None if max_depth is None else int(max_depth),
                            max_leaf_nodes=int(max_leaf_nodes),
                            min_samples_leaf=int(min_samples_leaf),
                            random_state=seed,
                        ),
                    ),
                ]
            )
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_va)
            fold_scores.append(score_metric(y_va, y_pred, all_classes, tune_metric))

        mean_score = float(np.mean(fold_scores))
        row = {
            "learning_rate": float(learning_rate),
            "max_iter": int(max_iter),
            "max_depth": None if max_depth is None else int(max_depth),
            "max_leaf_nodes": int(max_leaf_nodes),
            "min_samples_leaf": int(min_samples_leaf),
            f"mean_{tune_metric}": mean_score,
        }
        param_rows.append(row)
        if mean_score > best_score:
            best_score = mean_score
            best_params = {k: v for k, v in row.items() if not k.startswith("mean_")}

    best_model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "clf",
                HistGradientBoostingClassifier(
                    learning_rate=float(best_params["learning_rate"]),
                    max_iter=int(best_params["max_iter"]),
                    max_depth=None if best_params["max_depth"] is None else int(best_params["max_depth"]),
                    max_leaf_nodes=int(best_params["max_leaf_nodes"]),
                    min_samples_leaf=int(best_params["min_samples_leaf"]),
                    random_state=seed,
                ),
            ),
        ]
    )
    return best_model, pd.DataFrame(param_rows), best_params


def tune_gaussian_nb(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    all_classes: Sequence[str],
    var_smoothing_grid: Sequence[float],
    inner_folds: int,
    seed: int,
    tune_metric: str,
) -> Tuple[Pipeline, pd.DataFrame, Dict[str, float]]:
    param_rows = []
    inner = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    best_score = -np.inf
    best_var = None

    for var_smoothing in var_smoothing_grid:
        fold_scores = []
        for tr_idx, va_idx in inner.split(X_train, y_train):
            X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
            y_tr, y_va = y_train[tr_idx], y_train[va_idx]
            model = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("clf", GaussianNB(var_smoothing=float(var_smoothing))),
                ]
            )
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_va)
            fold_scores.append(score_metric(y_va, y_pred, all_classes, tune_metric))
        mean_score = float(np.mean(fold_scores))
        param_rows.append({"var_smoothing": float(var_smoothing), f"mean_{tune_metric}": mean_score})
        if mean_score > best_score:
            best_score = mean_score
            best_var = float(var_smoothing)

    best_model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("clf", GaussianNB(var_smoothing=float(best_var))),
        ]
    )
    return best_model, pd.DataFrame(param_rows), {"var_smoothing": float(best_var)}


def tune_mlp_shallow(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    all_classes: Sequence[str],
    hidden_layer_grid: Sequence[int],
    alpha_grid: Sequence[float],
    learning_rate_init_grid: Sequence[float],
    max_iter: int,
    early_stopping: bool,
    inner_folds: int,
    seed: int,
    tune_metric: str,
) -> Tuple[Pipeline, pd.DataFrame, Dict[str, float | int | bool]]:
    param_rows = []
    inner = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    best_score = -np.inf
    best_params: Dict[str, float | int] = {}

    for h, a, lr in product(hidden_layer_grid, alpha_grid, learning_rate_init_grid):
        fold_scores = []
        for tr_idx, va_idx in inner.split(X_train, y_train):
            X_tr, X_va = X_train.iloc[tr_idx], X_train.iloc[va_idx]
            y_tr, y_va = y_train[tr_idx], y_train[va_idx]
            model = Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "clf",
                        MLPClassifier(
                            hidden_layer_sizes=(int(h),),
                            alpha=float(a),
                            learning_rate_init=float(lr),
                            max_iter=int(max_iter),
                            random_state=seed,
                            early_stopping=bool(early_stopping),
                        ),
                    ),
                ]
            )
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_va)
            fold_scores.append(score_metric(y_va, y_pred, all_classes, tune_metric))
        mean_score = float(np.mean(fold_scores))
        row = {
            "hidden_layer_size": int(h),
            "alpha": float(a),
            "learning_rate_init": float(lr),
            f"mean_{tune_metric}": mean_score,
        }
        param_rows.append(row)
        if mean_score > best_score:
            best_score = mean_score
            best_params = {k: v for k, v in row.items() if not k.startswith("mean_")}

    best_model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "clf",
                MLPClassifier(
                    hidden_layer_sizes=(int(best_params["hidden_layer_size"]),),
                    alpha=float(best_params["alpha"]),
                    learning_rate_init=float(best_params["learning_rate_init"]),
                    max_iter=int(max_iter),
                    random_state=seed,
                    early_stopping=bool(early_stopping),
                ),
            ),
        ]
    )
    best_params["max_iter"] = int(max_iter)
    best_params["early_stopping"] = bool(early_stopping)
    best_params["tune_metric"] = tune_metric
    return best_model, pd.DataFrame(param_rows), best_params


def evaluate_model(
    model_name: str,
    model: Pipeline,
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    all_classes: Sequence[str],
    compute_auroc: bool,
    class_weight_mode: str = "none",
    decision_policy: Dict[str, object] | None = None,
) -> Tuple[Dict[str, float], pd.DataFrame, pd.DataFrame, object]:
    fitted_model, raw_proba = fit_model_and_predict_raw_proba(
        model_name=model_name,
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_pred=X_test,
        all_classes=all_classes,
        class_weight_mode=class_weight_mode,
    )
    y_pred_raw = argmax_labels_from_proba(raw_proba, all_classes)
    adjusted_proba = apply_decision_policy_proba(raw_proba, decision_policy)
    y_pred = argmax_labels_from_proba(adjusted_proba, all_classes)

    macro_f1 = float(f1_score(y_test, y_pred, average="macro", labels=list(all_classes), zero_division=0))
    weighted_f1 = float(f1_score(y_test, y_pred, average="weighted", labels=list(all_classes), zero_division=0))
    bal_acc = float(balanced_accuracy_score(y_test, y_pred))
    acc = float(accuracy_score(y_test, y_pred))
    auroc = float("nan")
    auc_df = pd.DataFrame(columns=["class", "auc", "n_pos", "n_neg"])

    if compute_auroc:
        auc_df = compute_auc_by_class(y_test, raw_proba, all_classes)
        auc_vals = pd.to_numeric(auc_df["auc"], errors="coerce").dropna().to_numpy()
        auroc = float(np.mean(auc_vals)) if len(auc_vals) else float("nan")

    cm = confusion_matrix(y_test, y_pred, labels=list(all_classes))
    cm_df = pd.DataFrame(cm, index=list(all_classes), columns=list(all_classes))

    metrics = {
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "balanced_accuracy": bal_acc,
        "accuracy": acc,
        "auroc": auroc,
        "raw_argmax_macro_f1": float(
            f1_score(y_test, y_pred_raw, average="macro", labels=list(all_classes), zero_division=0)
        ),
        "raw_argmax_weighted_f1": float(
            f1_score(y_test, y_pred_raw, average="weighted", labels=list(all_classes), zero_division=0)
        ),
        "raw_argmax_balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred_raw)),
        "raw_argmax_accuracy": float(accuracy_score(y_test, y_pred_raw)),
    }
    if decision_policy:
        policy_params = dict(decision_policy.get("params", {}))
        metrics["decision_policy_type"] = str(decision_policy.get("type", "")).strip()
        metrics["decision_policy_metric"] = str(decision_policy.get("metric", "")).strip()
        metrics["decision_policy_oof_metric_value"] = float(
            pd.to_numeric(decision_policy.get("oof_metric_value", np.nan), errors="coerce")
        )
        metrics["decision_policy_oof_macro_f1"] = float(
            pd.to_numeric(decision_policy.get("oof_macro_f1", np.nan), errors="coerce")
        )
        metrics["decision_policy_oof_weighted_f1"] = float(
            pd.to_numeric(decision_policy.get("oof_weighted_f1", np.nan), errors="coerce")
        )
        metrics["decision_policy_oof_accuracy"] = float(
            pd.to_numeric(decision_policy.get("oof_accuracy", np.nan), errors="coerce")
        )
        metrics["decision_policy_oof_gain_vs_raw_argmax_metric"] = float(
            pd.to_numeric(decision_policy.get("oof_gain_vs_raw_argmax_metric", np.nan), errors="coerce")
        )
        metrics["decision_policy_min_gain"] = float(
            pd.to_numeric(decision_policy.get("decision_policy_min_gain", np.nan), errors="coerce")
        )
        metrics["decision_policy_guard_triggered"] = int(bool(decision_policy.get("decision_policy_guard_triggered", False)))
        metrics["decision_policy_candidate_type_before_guard"] = str(
            decision_policy.get("candidate_type_before_guard", "")
        ).strip()
        metrics["decision_policy_candidate_gain_vs_raw_argmax_metric_before_guard"] = float(
            pd.to_numeric(
                decision_policy.get("candidate_gain_vs_raw_argmax_metric_before_guard", np.nan), errors="coerce"
            )
        )
        metrics["decision_policy_gain_vs_raw_argmax_macro_f1"] = metrics["macro_f1"] - metrics["raw_argmax_macro_f1"]
        metrics["decision_policy_gain_vs_raw_argmax_weighted_f1"] = (
            metrics["weighted_f1"] - metrics["raw_argmax_weighted_f1"]
        )
        metrics["decision_policy_gain_vs_raw_argmax_accuracy"] = metrics["accuracy"] - metrics["raw_argmax_accuracy"]
        if "temperature" in policy_params:
            metrics["decision_policy_temperature"] = float(policy_params["temperature"])
        if "alpha" in policy_params:
            metrics["decision_policy_alpha"] = float(policy_params["alpha"])
    return metrics, cm_df, auc_df, fitted_model


def main() -> int:
    args = parse_args()
    emit_heartbeat = _safe_heartbeat_emit(
        run_id=args.run_id,
        heartbeat_path=args.heartbeat_path,
        heartbeat_interval=args.heartbeat_interval,
    )
    emit_heartbeat(
        event="start",
        status="running",
        force=True,
        stage="run_model_benchmark_from_selected",
        repeat=args.repeat,
        fold=args.fold,
    )

    def _return(code: int, *, error: str = "") -> int:
        emit_heartbeat(
            event="exit",
            status="completed" if code == 0 else "failed",
            force=True,
            exit_code=code,
            error=error,
            stage="run_model_benchmark_from_selected",
        )
        return code

    candidate_spec = load_candidate_spec(args.candidate_spec_json)

    probe_matrix = Path(args.probe_matrix)
    manifest_path = Path(args.manifest)
    folds_path = Path(args.folds)
    selected_run_dir = Path(args.selected_run_dir)
    out_dir = Path(args.out_dir)

    for path in [probe_matrix, manifest_path, folds_path, selected_run_dir]:
        if not path.exists():
            print(f"ERROR: missing input {path}", file=sys.stderr)
            return _return(1, error=f"missing_input:{path}")

    selected_path = (
        selected_run_dir
        / f"repeat{args.repeat}"
        / f"fold{args.fold}"
        / args.selection_method
        / f"selected_probes_K{args.k}.tsv"
    )
    train_manifest_filtered_path = (
        selected_run_dir / f"repeat{args.repeat}" / f"fold{args.fold}" / "qc" / "train_manifest_filtered.tsv"
    )
    if not selected_path.exists():
        print(f"ERROR: missing selected probes file {selected_path}", file=sys.stderr)
        return _return(1, error="missing_selected_probes")
    if not train_manifest_filtered_path.exists():
        print(f"ERROR: missing filtered train manifest {train_manifest_filtered_path}", file=sys.stderr)
        return _return(1, error="missing_train_manifest")

    selected_df = pd.read_csv(selected_path, sep="\t", dtype=str)
    if "probe_id" not in selected_df.columns:
        print(f"ERROR: selected probes file missing probe_id column: {selected_path}", file=sys.stderr)
        return _return(1, error="missing_probe_id_column")
    selected_probes = selected_df["probe_id"].astype(str).tolist()
    if len(selected_probes) < args.k:
        print(f"ERROR: selected probes count {len(selected_probes)} < requested k={args.k}", file=sys.stderr)
        return _return(1, error="too_few_selected_probes")
    selected_probes = selected_probes[: args.k]

    manifest = pd.read_csv(manifest_path, sep="\t", dtype=str)
    folds = pd.read_csv(folds_path, sep="\t", dtype=str)
    if "sample_id" not in manifest.columns or "project_id" not in manifest.columns:
        print("ERROR: manifest missing sample_id/project_id", file=sys.stderr)
        return _return(1, error="missing_manifest_columns")
    if "sample_id" not in folds.columns or "outer_fold" not in folds.columns:
        print("ERROR: folds missing sample_id/outer_fold", file=sys.stderr)
        return _return(1, error="missing_fold_columns")
    if "repeat" not in folds.columns:
        folds["repeat"] = "0"

    folds["outer_fold"] = folds["outer_fold"].astype(int)
    folds["repeat"] = folds["repeat"].astype(int)
    manifest = manifest.merge(folds, on="sample_id", how="inner")

    test_df = manifest[(manifest["repeat"] == args.repeat) & (manifest["outer_fold"] == args.fold)].copy()
    if test_df.empty:
        print("ERROR: no test samples for requested fold/repeat", file=sys.stderr)
        return _return(1, error="empty_test_split")
    test_ids = test_df["sample_id"].astype(str).tolist()

    train_filtered_df = pd.read_csv(train_manifest_filtered_path, sep="\t", dtype=str)
    if "sample_id" not in train_filtered_df.columns or "project_id" not in train_filtered_df.columns:
        print("ERROR: train_manifest_filtered missing sample_id/project_id", file=sys.stderr)
        return _return(1, error="missing_train_manifest_columns")
    train_ids = train_filtered_df["sample_id"].astype(str).tolist()

    y_train = train_filtered_df.set_index("sample_id").loc[train_ids, "project_id"].astype(str).to_numpy()
    y_test = test_df.set_index("sample_id").loc[test_ids, "project_id"].astype(str).to_numpy()
    all_classes = sorted(manifest["project_id"].astype(str).unique().tolist())

    fold_dir = out_dir / f"repeat{args.repeat}" / f"fold{args.fold}"
    fold_dir.mkdir(parents=True, exist_ok=True)
    if candidate_spec:
        (fold_dir / "candidate_spec.json").write_text(json.dumps(candidate_spec, indent=2) + "\n", encoding="utf-8")

    t_read_start = time.perf_counter()
    emit_heartbeat(event="read_selected_matrix", status="running", force=True)
    pf = pq.ParquetFile(probe_matrix)
    X_train = read_selected(pf, train_ids, selected_probes)
    X_test = read_selected(pf, test_ids, selected_probes)
    t_read_selected = time.perf_counter() - t_read_start

    requested_model_names = [m.strip() for m in re.split(r"[,\s:]+", str(args.models)) if m.strip()]
    model_names = [canonical_model_name(name) for name in requested_model_names]
    allowed = {
        "lr_multinomial",
        "svm_linear",
        "xgb_shallow",
        "mlp_shallow",
        "extra_trees",
        "random_forest",
        "hist_gradient_boosting",
        "gaussian_nb",
        "linear_svm",
        "elastic_net_multinomial_logistic",
    }
    bad = [m for m in requested_model_names if canonical_model_name(m) not in allowed]
    plugin_bad = []
    for model_name in bad:
        try:
            load_model_plugin(canonical_model_name(model_name))
        except PluginNotFound:
            plugin_bad.append(model_name)
    if plugin_bad:
        print(f"ERROR: unsupported models requested: {','.join(plugin_bad)}", file=sys.stderr)
        return _return(1, error=f"unsupported_models:{','.join(plugin_bad)}")
    if "xgb_shallow" in model_names:
        try:
            import xgboost  # noqa: F401
        except ModuleNotFoundError:
            print(
                "WARNING: xgboost is not installed in this environment; skipping model=xgb_shallow",
                file=sys.stderr,
            )
            emit_heartbeat(
                event="model_skipped_missing_dependency",
                status="running",
                force=True,
                model="xgb_shallow",
                dependency="xgboost",
            )
            kept_requested = [name for name in requested_model_names if canonical_model_name(name) != "xgb_shallow"]
            requested_model_names = kept_requested
            model_names = [canonical_model_name(name) for name in kept_requested]
    if not model_names:
        print("ERROR: no models requested", file=sys.stderr)
        return _return(1, error="empty_models")

    lr_c_grid = parse_list_floats(args.lr_c_grid)
    lr_l1_grid = parse_list_floats(args.lr_l1_grid)
    svm_c_grid = parse_list_floats(args.svm_c_grid)
    xgb_max_depth_grid = parse_list_ints(args.xgb_max_depth_grid)
    xgb_n_estimators_grid = parse_list_ints(args.xgb_n_estimators_grid)
    xgb_learning_rate_grid = parse_list_floats(args.xgb_learning_rate_grid)
    et_n_estimators_grid = parse_list_ints(args.et_n_estimators_grid)
    et_max_features_grid = parse_list_mixed_max_features(args.et_max_features_grid)
    et_max_depth_grid = parse_list_ints_or_none(args.et_max_depth_grid)
    et_min_samples_leaf_grid = parse_list_ints(args.et_min_samples_leaf_grid)
    rf_n_estimators_grid = parse_list_ints(args.rf_n_estimators_grid)
    rf_max_features_grid = parse_list_mixed_max_features(args.rf_max_features_grid)
    rf_max_depth_grid = parse_list_ints_or_none(args.rf_max_depth_grid)
    rf_min_samples_leaf_grid = parse_list_ints(args.rf_min_samples_leaf_grid)
    hgb_learning_rate_grid = parse_list_floats(args.hgb_learning_rate_grid)
    hgb_max_iter_grid = parse_list_ints(args.hgb_max_iter_grid)
    hgb_max_depth_grid = parse_list_ints_or_none(args.hgb_max_depth_grid)
    hgb_max_leaf_nodes_grid = parse_list_ints(args.hgb_max_leaf_nodes_grid)
    hgb_min_samples_leaf_grid = parse_list_ints(args.hgb_min_samples_leaf_grid)
    gnb_var_smoothing_grid = parse_list_floats(args.gnb_var_smoothing_grid)
    mlp_hidden_layer_grid = parse_list_ints(args.mlp_hidden_layer_grid)
    mlp_alpha_grid = parse_list_floats(args.mlp_alpha_grid)
    mlp_learning_rate_init_grid = parse_list_floats(args.mlp_learning_rate_init_grid)

    for requested_model_name, model_name in zip(requested_model_names, model_names):
        emit_heartbeat(event="model_start", status="running", force=True, model=model_name)
        model_args = merge_model_args(candidate_spec, requested_model_name)
        model_inner_folds = int(model_args.get("inner_folds", args.inner_folds))
        model_seed = int(model_args.get("seed", args.seed))
        model_n_jobs = int(model_args.get("n_jobs", args.n_jobs))
        requested_class_weight = str(model_args.get("class_weight", args.class_weight_mode)).strip()
        if requested_class_weight in {"balanced", "balanced_subsample"}:
            model_class_weight_mode = "balanced"
        elif requested_class_weight in {"", "none"}:
            model_class_weight_mode = "none"
        else:
            model_class_weight_mode = args.class_weight_mode
        logistic_penalties = model_args.get("penalties", ["l2", "elasticnet"])
        if not isinstance(logistic_penalties, list):
            logistic_penalties = [str(logistic_penalties)]
        # Keep fold outputs unique per model + class-weight + tune-metric.
        # Without this, parallel tasks can overwrite each other.
        model_subdir = f"{requested_model_name}__cw_{model_class_weight_mode}__tm_{args.tune_metric}"
        model_dir = fold_dir / model_subdir
        model_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"probe_id": selected_probes}).to_csv(
            model_dir / f"selected_probes_K{args.k}.tsv", sep="\t", index=False
        )

        t_tune_start = time.perf_counter()
        if model_name == "lr_multinomial":
            lr_c_grid_eff = [float(model_args["C"])] if "C" in model_args else lr_c_grid
            lr_l1_grid_eff = [float(model_args["l1_ratio"])] if "l1_ratio" in model_args else lr_l1_grid
            model, inner_df, best_params = tune_lr_multinomial(
                X_train=X_train,
                y_train=y_train,
                all_classes=all_classes,
                c_grid=lr_c_grid_eff,
                l1_grid=lr_l1_grid_eff,
                inner_folds=model_inner_folds,
                seed=model_seed,
                tune_metric=args.tune_metric,
                class_weight_mode=model_class_weight_mode,
                penalties=logistic_penalties,
            )
        elif model_name == "svm_linear":
            svm_c_grid_eff = [float(model_args["C"])] if "C" in model_args else svm_c_grid
            model, inner_df, best_params = tune_svm_linear(
                X_train=X_train,
                y_train=y_train,
                all_classes=all_classes,
                c_grid=svm_c_grid_eff,
                inner_folds=model_inner_folds,
                seed=model_seed,
                tune_metric=args.tune_metric,
                class_weight_mode=model_class_weight_mode,
            )
        elif model_name == "xgb_shallow":
            model, inner_df, best_params = tune_xgb_shallow(
                X_train=X_train,
                y_train=y_train,
                all_classes=all_classes,
                max_depth_grid=xgb_max_depth_grid,
                n_estimators_grid=xgb_n_estimators_grid,
                learning_rate_grid=xgb_learning_rate_grid,
                subsample=args.xgb_subsample,
                colsample_bytree=args.xgb_colsample_bytree,
                min_child_weight=args.xgb_min_child_weight,
                reg_lambda=args.xgb_reg_lambda,
                inner_folds=model_inner_folds,
                seed=model_seed,
                n_jobs=model_n_jobs,
                tune_metric=args.tune_metric,
                class_weight_mode=model_class_weight_mode,
            )
        elif model_name == "mlp_shallow":
            model, inner_df, best_params = tune_mlp_shallow(
                X_train=X_train,
                y_train=y_train,
                all_classes=all_classes,
                hidden_layer_grid=mlp_hidden_layer_grid,
                alpha_grid=mlp_alpha_grid,
                learning_rate_init_grid=mlp_learning_rate_init_grid,
                max_iter=args.mlp_max_iter,
                early_stopping=args.mlp_early_stopping,
                inner_folds=model_inner_folds,
                seed=model_seed,
                tune_metric=args.tune_metric,
            )
        elif model_name == "extra_trees":
            et_n_estimators_grid_eff = [int(model_args["n_estimators"])] if "n_estimators" in model_args else et_n_estimators_grid
            et_max_features_grid_eff = [model_args["max_features"]] if "max_features" in model_args else et_max_features_grid
            et_min_samples_leaf_grid_eff = [int(model_args["min_samples_leaf"])] if "min_samples_leaf" in model_args else et_min_samples_leaf_grid
            et_max_depth_grid_eff = (
                [None if str(model_args["max_depth"]).strip().lower() in {"", "none", "null", "0"} else int(model_args["max_depth"])]
                if "max_depth" in model_args
                else et_max_depth_grid
            )
            model, inner_df, best_params = tune_extra_trees(
                X_train=X_train,
                y_train=y_train,
                all_classes=all_classes,
                n_estimators_grid=et_n_estimators_grid_eff,
                max_features_grid=et_max_features_grid_eff,
                max_depth_grid=et_max_depth_grid_eff,
                min_samples_leaf_grid=et_min_samples_leaf_grid_eff,
                inner_folds=model_inner_folds,
                seed=model_seed,
                n_jobs=model_n_jobs,
                tune_metric=args.tune_metric,
                class_weight=(requested_class_weight or "balanced"),
            )
        elif model_name == "random_forest":
            rf_n_estimators_grid_eff = [int(model_args["n_estimators"])] if "n_estimators" in model_args else rf_n_estimators_grid
            rf_max_features_grid_eff = [model_args["max_features"]] if "max_features" in model_args else rf_max_features_grid
            rf_min_samples_leaf_grid_eff = [int(model_args["min_samples_leaf"])] if "min_samples_leaf" in model_args else rf_min_samples_leaf_grid
            rf_max_depth_grid_eff = (
                [None if str(model_args["max_depth"]).strip().lower() in {"", "none", "null", "0"} else int(model_args["max_depth"])]
                if "max_depth" in model_args
                else rf_max_depth_grid
            )
            model, inner_df, best_params = tune_random_forest(
                X_train=X_train,
                y_train=y_train,
                all_classes=all_classes,
                n_estimators_grid=rf_n_estimators_grid_eff,
                max_features_grid=rf_max_features_grid_eff,
                max_depth_grid=rf_max_depth_grid_eff,
                min_samples_leaf_grid=rf_min_samples_leaf_grid_eff,
                inner_folds=model_inner_folds,
                seed=model_seed,
                n_jobs=model_n_jobs,
                tune_metric=args.tune_metric,
                class_weight=(requested_class_weight or "balanced"),
            )
        elif model_name == "hist_gradient_boosting":
            model, inner_df, best_params = tune_hist_gradient_boosting(
                X_train=X_train,
                y_train=y_train,
                all_classes=all_classes,
                learning_rate_grid=hgb_learning_rate_grid,
                max_iter_grid=hgb_max_iter_grid,
                max_depth_grid=hgb_max_depth_grid,
                max_leaf_nodes_grid=hgb_max_leaf_nodes_grid,
                min_samples_leaf_grid=hgb_min_samples_leaf_grid,
                inner_folds=model_inner_folds,
                seed=model_seed,
                tune_metric=args.tune_metric,
            )
        elif model_name == "gaussian_nb":
            model, inner_df, best_params = tune_gaussian_nb(
                X_train=X_train,
                y_train=y_train,
                all_classes=all_classes,
                var_smoothing_grid=gnb_var_smoothing_grid,
                inner_folds=model_inner_folds,
                seed=model_seed,
                tune_metric=args.tune_metric,
            )
        else:
            try:
                model, inner_df, best_params = tune_plugin_model(
                    model_name,
                    X_train=X_train,
                    y_train=y_train,
                    all_classes=all_classes,
                    tune_metric=args.tune_metric,
                    class_weight_mode=model_class_weight_mode,
                    inner_folds=model_inner_folds,
                    seed=model_seed,
                    n_jobs=model_n_jobs,
                    model_args=model_args,
                )
            except PluginNotFound:
                raise ValueError(f"Unknown model: {model_name}") from None
        t_tune = time.perf_counter() - t_tune_start

        decision_policy = None
        if args.fit_decision_policy:
            t_policy_metric = str(args.decision_policy_metric).strip() or str(args.tune_metric).strip()
            t_policy_modes = parse_decision_policy_modes(args.decision_policy_modes)
            emit_heartbeat(
                event="decision_policy_fit_start",
                status="running",
                force=True,
                model=model_name,
                decision_policy_metric=t_policy_metric,
                decision_policy_modes=",".join(t_policy_modes),
            )
            raw_proba_oof = build_oof_raw_proba(
                model_name=model_name,
                model=model,
                X_train=X_train,
                y_train=y_train,
                all_classes=all_classes,
                class_weight_mode=model_class_weight_mode,
                inner_folds=model_inner_folds,
                seed=model_seed,
            )
            decision_policy = fit_decision_policy(
                y_true=y_train,
                raw_proba_oof=raw_proba_oof,
                all_classes=all_classes,
                metric=t_policy_metric,
                modes=t_policy_modes,
                min_gain=float(args.decision_policy_min_gain),
            )
            (model_dir / "decision_policy.json").write_text(
                json.dumps(json.loads(json.dumps(decision_policy)), indent=2) + "\n",
                encoding="utf-8",
            )
            emit_heartbeat(
                event="decision_policy_fit_done",
                status="running",
                force=True,
                model=model_name,
                decision_policy_type=str(decision_policy.get("type", "")),
            )

        t_eval_start = time.perf_counter()
        metrics, cm_df, auc_df, _ = evaluate_model(
            model_name=model_name,
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            all_classes=all_classes,
            compute_auroc=args.compute_auroc,
            class_weight_mode=model_class_weight_mode,
            decision_policy=decision_policy,
        )
        t_eval = time.perf_counter() - t_eval_start

        inner_df.to_csv(model_dir / "innercv_results.tsv", sep="\t", index=False)
        with (model_dir / "best_params.json").open("w", encoding="utf-8") as handle:
            json.dump(best_params, handle, indent=2)
        cm_df.to_csv(model_dir / f"confusion_matrix_K{args.k}.tsv", sep="\t")

        metrics_row = {
            "model": requested_model_name,
            "selection_method": args.selection_method,
            "repeat": args.repeat,
            "fold": args.fold,
            "k": args.k,
            "n_train": len(train_ids),
            "n_test": len(test_ids),
            "n_classes": len(all_classes),
            "class_weight_mode": model_class_weight_mode,
            "tune_metric": args.tune_metric,
            **metrics,
        }
        pd.DataFrame([metrics_row]).to_csv(model_dir / "metrics.tsv", sep="\t", index=False)

        if args.compute_auroc and not auc_df.empty:
            auc_df.insert(0, "k", args.k)
            auc_df.insert(0, "fold", args.fold)
            auc_df.insert(0, "repeat", args.repeat)
            auc_df.insert(0, "selection_method", args.selection_method)
            auc_df.insert(0, "model", requested_model_name)
            auc_df.to_csv(model_dir / f"auc_by_class_K{args.k}.tsv", sep="\t", index=False)

        timing_row = {
            "model": requested_model_name,
            "selection_method": args.selection_method,
            "repeat": args.repeat,
            "fold": args.fold,
            "k": args.k,
            "class_weight_mode": model_class_weight_mode,
            "tune_metric": args.tune_metric,
            "t_read_selected_matrix": t_read_selected,
            "t_tune_model": t_tune,
            "t_eval_model": t_eval,
        }
        pd.DataFrame([timing_row]).to_csv(model_dir / "timing.tsv", sep="\t", index=False)
        emit_heartbeat(event="model_done", status="running", force=True, model=model_name)

    return _return(0)


if __name__ == "__main__":
    raise SystemExit(main())
