#!/usr/bin/env python3
"""Nested cfDNA two-stage model with a specificity-constrained cancer gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.pipeline import make_pipeline


RANK_CACHE: dict[tuple[object, ...], np.ndarray] = {}
PAIR_WEIGHT_CACHE: dict[tuple[object, ...], dict[tuple[str, str], float]] = {}


@dataclass(frozen=True)
class Config:
    gate_k: int
    subtype_k: int
    gate_model: str
    subtype_model: str

    @property
    def config_id(self) -> str:
        return f"gate{self.gate_model}_k{self.gate_k}__subtype{self.subtype_model}_k{self.subtype_k}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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


def parse_budgets(value: str, max_union: int) -> list[tuple[int, int]]:
    budgets = []
    for item in value.split(","):
        if not item.strip():
            continue
        gate, subtype = item.split(":")
        gate_k = int(gate)
        subtype_k = int(subtype)
        if gate_k + subtype_k > max_union:
            raise SystemExit(f"Budget exceeds max union {max_union}: {item}")
        budgets.append((gate_k, subtype_k))
    if not budgets:
        raise SystemExit("No budgets supplied")
    return budgets


def parse_csv_strings(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_prjna_labels(args: argparse.Namespace) -> tuple[list[str], str]:
    labels = parse_csv_strings(args.prjna_labels)
    if not labels:
        raise SystemExit("--prjna-labels is empty")
    if args.prjna_normal_label not in labels:
        raise SystemExit(f"--prjna-normal-label {args.prjna_normal_label!r} is not in --prjna-labels {labels}")
    return labels, args.prjna_normal_label


def read_tsv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing input: {path}")
    return pd.read_csv(path, sep="\t", dtype=str)


def load_data(args: argparse.Namespace) -> tuple[np.ndarray, pd.DataFrame, list[str], str]:
    if not args.matrix.exists():
        raise SystemExit(f"Missing matrix: {args.matrix}")
    matrix = np.load(args.matrix, mmap_mode="r")
    meta = read_tsv(args.sample_metadata).copy()
    stage = read_tsv(args.stage_metadata).copy()
    if args.dataset == "gse186007":
        meta["sample_id"] = meta["geo_accession"]
        meta["label"] = meta["group"]
        meta = meta.merge(stage[["geo_accession", "stage", "stage_broad"]], on="geo_accession", how="left")
        meta["stage_raw"] = meta["stage"].fillna("")
        meta["stage_group"] = meta["stage_broad"].fillna("missing")
        labels = ["CRC", "HEA", "PDAC"]
        normal_label = "HEA"
    else:
        meta["sample_id"] = meta["Geoid"]
        meta["label"] = meta["analysis_label"]
        stage_cols = ["Geoid", "geo_stage"]
        extra_stage = stage[[col for col in stage_cols if col in stage.columns]].copy()
        meta = meta.merge(extra_stage, on="Geoid", how="left")
        meta["stage_raw"] = meta.get("geo_stage", "").fillna("")
        meta["stage_group"] = [normalize_prjna_stage(stage_value, label) for stage_value, label in zip(meta["stage_raw"], meta["label"], strict=True)]
        labels, normal_label = parse_prjna_labels(args)
    keep = meta["label"].isin(labels)
    meta = meta.loc[keep].reset_index(drop=True)
    row_idx = np.flatnonzero(keep.to_numpy())
    if matrix.shape[0] != len(keep):
        raise SystemExit(f"Matrix rows ({matrix.shape[0]}) do not match metadata rows ({len(keep)})")
    matrix = matrix[row_idx]
    missing = sorted(set(labels) - set(meta["label"]))
    if missing:
        raise SystemExit(f"Missing labels after filtering: {missing}")
    meta["is_cancer"] = meta["label"].ne(normal_label)
    meta.loc[~meta["is_cancer"], "stage_group"] = "Normal"
    return matrix, meta, labels, normal_label


def normalize_prjna_stage(stage: str, label: str) -> str:
    if label == "Normal":
        return "Normal"
    clean = (stage or "").strip()
    if not clean or clean.lower() == "nan":
        return "Cancer stage missing"
    upper = clean.upper()
    if upper in {"I", "IA", "IB", "II", "IIA", "IIB", "III", "IIIA", "IIIB", "IIIC"}:
        return "Cancer early/locoregional"
    if upper in {"IV", "IVA", "IVB"}:
        return "Cancer advanced"
    return "Cancer stage other"


def build_configs(args: argparse.Namespace) -> list[Config]:
    budgets = parse_budgets(args.budgets, args.max_union)
    models = parse_csv_strings(args.model_families)
    for model in models:
        if model not in {"extra_trees", "logistic_l2"}:
            raise SystemExit(f"Unsupported model family: {model}")
    return [Config(gate_k, subtype_k, gate_model, subtype_model) for gate_k, subtype_k in budgets for gate_model in models for subtype_model in models]


def transform_block(raw: np.ndarray, value_mode: str) -> tuple[np.ndarray, np.ndarray]:
    finite = np.isfinite(raw)
    if value_mode == "methylation":
        return np.where(finite, raw, 0.0).astype(np.float32, copy=False), finite
    if value_mode == "coverage":
        return finite.astype(np.float32), np.ones(raw.shape, dtype=bool)
    raise ValueError(value_mode)


def eligibility_from_raw(raw: np.ndarray, masks: list[np.ndarray], n_rows: int, min_fraction: float, min_per_class: int) -> np.ndarray:
    finite = np.isfinite(raw)
    min_total = max(1, math.ceil(n_rows * min_fraction))
    eligible = finite.sum(axis=0) >= min_total
    for mask in masks:
        eligible &= finite[mask].sum(axis=0) >= min_per_class
    return eligible


def group_stats(values: np.ndarray, finite_for_stats: np.ndarray, masks: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    counts = []
    sums = []
    sumsq = []
    for mask in masks:
        subset = values[mask]
        finite = finite_for_stats[mask]
        filled = np.where(finite, subset, 0.0).astype(np.float32, copy=False)
        counts.append(finite.sum(axis=0).astype(np.float32))
        sums.append(filled.sum(axis=0))
        sumsq.append((filled * filled).sum(axis=0))
    return np.vstack(counts), np.vstack(sums), np.vstack(sumsq)


def eligible_mask(counts: np.ndarray, n_rows: int, min_fraction: float, min_per_class: int) -> np.ndarray:
    min_total = max(1, math.ceil(n_rows * min_fraction))
    return (counts.sum(axis=0) >= min_total) & (counts >= min_per_class).all(axis=0)


def rank_binary_effect(
    matrix: np.ndarray,
    rows: np.ndarray,
    y_binary: np.ndarray,
    max_k: int,
    *,
    block_size: int,
    min_fraction: float,
    min_per_class: int,
    value_mode: str,
) -> np.ndarray:
    cached = cached_rank(f"binary:{value_mode}", matrix, rows, y_binary, max_k, block_size, min_fraction, min_per_class)
    if cached is not None:
        return cached
    requested_k = cache_request_k(max_k)
    scores = np.full(matrix.shape[1], -np.inf, dtype=np.float32)
    local_y = y_binary.astype(bool)
    masks = [local_y, ~local_y]
    for start in range(0, matrix.shape[1], block_size):
        end = min(matrix.shape[1], start + block_size)
        raw = np.asarray(matrix[np.ix_(rows, np.arange(start, end, dtype=np.int64))], dtype=np.float32)
        values, finite_for_stats = transform_block(raw, value_mode)
        counts, sums, sumsq = group_stats(values, finite_for_stats, masks)
        eligible = eligibility_from_raw(raw, masks, len(rows), min_fraction, min_per_class)
        with np.errstate(divide="ignore", invalid="ignore"):
            means = sums / counts
            vars_ = (sumsq - counts * means**2) / np.maximum(counts - 1, 1)
            pooled = np.sqrt((vars_[0] + vars_[1]) / 2.0)
            block_scores = np.abs(means[0] - means[1]) / pooled
        scores[start:end] = np.where(np.isfinite(block_scores) & eligible, block_scores, -np.inf).astype(np.float32)
    ranked = top_indices(scores, requested_k)
    store_rank(f"binary:{value_mode}", matrix, rows, y_binary, block_size, min_fraction, min_per_class, ranked)
    return ranked[:max_k]


def rank_multiclass_anova(
    matrix: np.ndarray,
    rows: np.ndarray,
    y: np.ndarray,
    max_k: int,
    *,
    block_size: int,
    min_fraction: float,
    min_per_class: int,
    value_mode: str,
) -> np.ndarray:
    cached = cached_rank(f"multiclass:{value_mode}", matrix, rows, y, max_k, block_size, min_fraction, min_per_class)
    if cached is not None:
        return cached
    requested_k = cache_request_k(max_k)
    scores = np.full(matrix.shape[1], -np.inf, dtype=np.float32)
    classes = np.array(sorted(set(y)))
    masks = [y == label for label in classes]
    for start in range(0, matrix.shape[1], block_size):
        end = min(matrix.shape[1], start + block_size)
        raw = np.asarray(matrix[np.ix_(rows, np.arange(start, end, dtype=np.int64))], dtype=np.float32)
        values, finite_for_stats = transform_block(raw, value_mode)
        counts, sums, sumsq = group_stats(values, finite_for_stats, masks)
        eligible = eligibility_from_raw(raw, masks, len(rows), min_fraction, min_per_class)
        total_n = counts.sum(axis=0)
        with np.errstate(divide="ignore", invalid="ignore"):
            means = sums / counts
            grand = sums.sum(axis=0) / total_n
            between = np.nansum(counts * (means - grand) ** 2, axis=0)
            within = np.nansum(sumsq - counts * means**2, axis=0)
            block_scores = (between / max(len(classes) - 1, 1)) / (within / np.maximum(total_n - len(classes), 1))
        scores[start:end] = np.where(np.isfinite(block_scores) & eligible, block_scores, -np.inf).astype(np.float32)
    ranked = top_indices(scores, requested_k)
    store_rank(f"multiclass:{value_mode}", matrix, rows, y, block_size, min_fraction, min_per_class, ranked)
    return ranked[:max_k]


def rank_multiclass_ovr_union(
    matrix: np.ndarray,
    rows: np.ndarray,
    y: np.ndarray,
    max_k: int,
    *,
    block_size: int,
    min_fraction: float,
    min_per_class: int,
    value_mode: str,
) -> np.ndarray:
    cache_kind = f"multiclass_ovr_union:{value_mode}"
    cached = cached_rank(cache_kind, matrix, rows, y, max_k, block_size, min_fraction, min_per_class)
    if cached is not None:
        return cached
    requested_k = cache_request_k(max_k)
    classes = np.array(sorted(set(y)))
    if len(classes) < 2:
        raise RuntimeError("OVR subtype selector requires at least two cancer classes")
    per_class_k = int(math.ceil(requested_k / len(classes)))
    class_ranks = [
        rank_binary_effect(
            matrix,
            rows,
            y == label,
            per_class_k,
            block_size=block_size,
            min_fraction=min_fraction,
            min_per_class=min_per_class,
            value_mode=value_mode,
        )
        for label in classes
    ]
    selected = []
    seen = set()
    for rank_i in range(per_class_k):
        for ranked in class_ranks:
            if rank_i >= len(ranked):
                continue
            feature_index = int(ranked[rank_i])
            if feature_index in seen:
                continue
            selected.append(feature_index)
            seen.add(feature_index)
            if len(selected) >= requested_k:
                ranked_array = np.asarray(selected, dtype=np.int64)
                store_rank(cache_kind, matrix, rows, y, block_size, min_fraction, min_per_class, ranked_array)
                return ranked_array[:max_k]
    if len(selected) < requested_k:
        fallback_rank = rank_multiclass_anova(
            matrix,
            rows,
            y,
            requested_k,
            block_size=block_size,
            min_fraction=min_fraction,
            min_per_class=min_per_class,
            value_mode=value_mode,
        )
        for feature_index in fallback_rank:
            feature_index = int(feature_index)
            if feature_index in seen:
                continue
            selected.append(feature_index)
            seen.add(feature_index)
            if len(selected) >= requested_k:
                break
    if len(selected) < requested_k:
        raise RuntimeError(f"Only {len(selected)} OVR-union+fallback subtype features for max_k={requested_k}")
    ranked_array = np.asarray(selected, dtype=np.int64)
    store_rank(cache_kind, matrix, rows, y, block_size, min_fraction, min_per_class, ranked_array)
    return ranked_array[:max_k]


def estimate_pairwise_confusion_weights(
    matrix: np.ndarray,
    rows: np.ndarray,
    y: np.ndarray,
    max_k: int,
    args: argparse.Namespace,
) -> dict[tuple[str, str], float]:
    classes = np.array(sorted(set(y)))
    base_weights = {(str(a), str(b)): 1.0 for i, a in enumerate(classes) for b in classes[i + 1 :]}
    if len(classes) < 2:
        return base_weights
    cache_base = rank_cache_base(
        "pairwise_confusion_weights",
        matrix,
        rows,
        y,
        args.block_size,
        args.min_train_present_fraction,
        args.min_train_present_per_class,
    )
    cache_key = (
        *cache_base,
        int(max_k),
        int(args.pairwise_confusion_splits),
        float(args.pairwise_confusion_weight_scale),
        args.pairwise_confusion_model,
        int(args.seed),
    )
    cached = PAIR_WEIGHT_CACHE.get(cache_key)
    if cached is not None:
        return cached

    min_class_count = int(pd.Series(y).value_counts().min())
    n_splits = min(args.pairwise_confusion_splits, min_class_count)
    if n_splits < 2:
        PAIR_WEIGHT_CACHE[cache_key] = base_weights
        return base_weights

    confusion = pd.DataFrame(0, index=classes, columns=classes, dtype=float)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=args.seed + 991)
    for split_i, (train_local, val_local) in enumerate(splitter.split(np.zeros(len(y)), y), start=1):
        train_rows = rows[train_local]
        val_rows = rows[val_local]
        y_train = y[train_local]
        y_val = y[val_local]
        features = rank_multiclass_anova(
            matrix,
            train_rows,
            y_train,
            max(max_k, 1),
            block_size=args.block_size,
            min_fraction=args.min_train_present_fraction,
            min_per_class=args.min_train_present_per_class,
            value_mode=args.value_mode,
        )
        model = make_model(args.pairwise_confusion_model, args.seed + 70000 + split_i, args.n_estimators, args.n_jobs)
        model.fit(extract_values(matrix, train_rows, features, args.value_mode), y_train)
        pred = model.predict(extract_values(matrix, val_rows, features, args.value_mode))
        for true_label, pred_label in zip(y_val, pred, strict=True):
            confusion.loc[true_label, pred_label] += 1

    weights: dict[tuple[str, str], float] = {}
    for i, a in enumerate(classes):
        for b in classes[i + 1 :]:
            support = float(confusion.loc[a].sum() + confusion.loc[b].sum())
            errors = float(confusion.loc[a, b] + confusion.loc[b, a])
            hardness = errors / support if support else 0.0
            weights[(str(a), str(b))] = 1.0 + args.pairwise_confusion_weight_scale * hardness
    PAIR_WEIGHT_CACHE[cache_key] = weights
    return weights


def rank_multiclass_pairwise_confusion(
    matrix: np.ndarray,
    rows: np.ndarray,
    y: np.ndarray,
    max_k: int,
    args: argparse.Namespace,
) -> np.ndarray:
    kind = (
        "multiclass_pairwise_confusion:"
        f"{args.pairwise_confusion_splits}:"
        f"{args.pairwise_confusion_weight_scale}:"
        f"{args.pairwise_confusion_model}"
    )
    cached = cached_rank(
        kind,
        matrix,
        rows,
        y,
        max_k,
        args.block_size,
        args.min_train_present_fraction,
        args.min_train_present_per_class,
    )
    if cached is not None:
        return cached
    requested_k = cache_request_k(max_k)
    classes = np.array(sorted(set(y)))
    masks = [y == label for label in classes]
    pair_weights = estimate_pairwise_confusion_weights(matrix, rows, y, requested_k, args)
    scores = np.full(matrix.shape[1], -np.inf, dtype=np.float32)
    for start in range(0, matrix.shape[1], args.block_size):
        end = min(matrix.shape[1], start + args.block_size)
        block = np.asarray(matrix[np.ix_(rows, np.arange(start, end, dtype=np.int64))], dtype=np.float32)
        counts, sums, sumsq = group_stats(block, masks)
        eligible = eligible_mask(counts, len(rows), args.min_train_present_fraction, args.min_train_present_per_class)
        with np.errstate(divide="ignore", invalid="ignore"):
            means = sums / counts
            vars_ = (sumsq - counts * means**2) / np.maximum(counts - 1, 1)
            block_scores = np.zeros(end - start, dtype=np.float32)
            for i, a in enumerate(classes):
                for j, b in enumerate(classes[i + 1 :], start=i + 1):
                    weight = pair_weights[(str(a), str(b))]
                    pooled = np.sqrt((vars_[i] + vars_[j]) / 2.0)
                    effect = np.abs(means[i] - means[j]) / pooled
                    block_scores += np.where(np.isfinite(effect), weight * effect, 0.0).astype(np.float32)
        scores[start:end] = np.where(np.isfinite(block_scores) & eligible, block_scores, -np.inf).astype(np.float32)
    ranked = top_indices(scores, requested_k)
    store_rank(
        kind,
        matrix,
        rows,
        y,
        args.block_size,
        args.min_train_present_fraction,
        args.min_train_present_per_class,
        ranked,
    )
    return ranked[:max_k]


def rank_subtype_features(
    matrix: np.ndarray,
    rows: np.ndarray,
    y: np.ndarray,
    max_k: int,
    args: argparse.Namespace,
) -> np.ndarray:
    if args.subtype_selector == "anova":
        return rank_multiclass_anova(
            matrix,
            rows,
            y,
            max_k,
            block_size=args.block_size,
            min_fraction=args.min_train_present_fraction,
            min_per_class=args.min_train_present_per_class,
            value_mode=args.value_mode,
        )
    if args.subtype_selector == "ovr_union":
        return rank_multiclass_ovr_union(
            matrix,
            rows,
            y,
            max_k,
            block_size=args.block_size,
            min_fraction=args.min_train_present_fraction,
            min_per_class=args.min_train_present_per_class,
            value_mode=args.value_mode,
        )
    if args.subtype_selector == "pairwise_confusion":
        return rank_multiclass_pairwise_confusion(matrix, rows, y, max_k, args)
    raise ValueError(args.subtype_selector)


def top_indices(scores: np.ndarray, max_k: int) -> np.ndarray:
    finite = np.isfinite(scores)
    if int(finite.sum()) < max_k:
        raise RuntimeError(f"Only {int(finite.sum())} eligible features for max_k={max_k}")
    top = np.argpartition(scores, -max_k)[-max_k:]
    top = top[np.argsort(scores[top])[::-1]]
    return top.astype(np.int64)


def extract_values(matrix: np.ndarray, rows: np.ndarray, features: np.ndarray, value_mode: str) -> np.ndarray:
    raw = np.asarray(matrix[np.ix_(rows, features)], dtype=np.float32)
    values, _ = transform_block(raw, value_mode)
    return values


def cache_request_k(max_k: int) -> int:
    return max(max_k, int(os.environ.get("CFDNA_RANK_CACHE_K", "0") or 0))


def rank_cache_base(
    kind: str,
    matrix: np.ndarray,
    rows: np.ndarray,
    y: np.ndarray,
    block_size: int,
    min_fraction: float,
    min_per_class: int,
) -> tuple[object, ...]:
    digest = hashlib.blake2b(digest_size=16)
    digest.update(np.asarray(rows, dtype=np.int64).tobytes())
    digest.update("\0".join(map(str, y)).encode())
    return (kind, matrix.shape, digest.hexdigest(), block_size, float(min_fraction), int(min_per_class))


def cached_rank(
    kind: str,
    matrix: np.ndarray,
    rows: np.ndarray,
    y: np.ndarray,
    max_k: int,
    block_size: int,
    min_fraction: float,
    min_per_class: int,
) -> np.ndarray | None:
    base = rank_cache_base(kind, matrix, rows, y, block_size, min_fraction, min_per_class)
    for key, ranked in RANK_CACHE.items():
        if key[: len(base)] == base and len(ranked) >= max_k:
            return ranked[:max_k]
    return None


def store_rank(
    kind: str,
    matrix: np.ndarray,
    rows: np.ndarray,
    y: np.ndarray,
    block_size: int,
    min_fraction: float,
    min_per_class: int,
    ranked: np.ndarray,
) -> None:
    base = rank_cache_base(kind, matrix, rows, y, block_size, min_fraction, min_per_class)
    RANK_CACHE[(*base, len(ranked))] = ranked


def make_model(family: str, seed: int, n_estimators: int, n_jobs: int):
    if family == "extra_trees":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            ExtraTreesClassifier(
                n_estimators=n_estimators,
                random_state=seed,
                class_weight="balanced",
                max_features="sqrt",
                min_samples_leaf=1,
                n_jobs=n_jobs,
            ),
        )
    if family == "logistic_l2":
        return make_pipeline(
            SimpleImputer(strategy="median"),
            LogisticRegression(C=1.0, penalty="l2", solver="lbfgs", class_weight="balanced", max_iter=5000, random_state=seed),
        )
    raise ValueError(family)


def positive_probability(model, x: np.ndarray, positive_label: str = "Cancer") -> np.ndarray:
    clf = model[-1]
    classes = list(clf.classes_)
    if positive_label not in classes:
        raise RuntimeError(f"Positive label {positive_label} missing from model classes {classes}")
    return model.predict_proba(x)[:, classes.index(positive_label)]


def choose_threshold(probs: np.ndarray, y_true_binary: np.ndarray, target_specificity: float) -> tuple[float, bool, float, float]:
    y = y_true_binary.astype(bool)
    candidates = np.unique(np.r_[0.0, probs, np.nextafter(probs, 1.0), 1.0])
    best = None
    for threshold in candidates:
        pred = probs >= threshold
        tn = int(np.sum(~y & ~pred))
        fp = int(np.sum(~y & pred))
        tp = int(np.sum(y & pred))
        fn = int(np.sum(y & ~pred))
        specificity = tn / (tn + fp) if (tn + fp) else np.nan
        sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
        feasible = bool(np.isfinite(specificity) and specificity >= target_specificity)
        if feasible:
            key = (
                1,
                sensitivity if np.isfinite(sensitivity) else -1,
                -abs(specificity - target_specificity) if np.isfinite(specificity) else -np.inf,
                specificity if np.isfinite(specificity) else -1,
                -threshold,
            )
        else:
            key = (
                0,
                specificity if np.isfinite(specificity) else -1,
                sensitivity if np.isfinite(sensitivity) else -1,
                -threshold,
            )
        if best is None or key > best[0]:
            best = (key, float(threshold), feasible, float(specificity), float(sensitivity))
    assert best is not None
    return best[1], best[2], best[3], best[4]


def workflow_metrics(y_true: np.ndarray, y_pred: np.ndarray, labels: list[str], normal_label: str) -> dict[str, float]:
    is_cancer_true = y_true != normal_label
    is_cancer_pred = y_pred != normal_label
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)),
        "gate_sensitivity": float(recall_score(is_cancer_true, is_cancer_pred, pos_label=True, zero_division=0)),
        "gate_specificity": float(recall_score(~is_cancer_true, ~is_cancer_pred, pos_label=True, zero_division=0)),
        "subtype_accuracy_all_cancers": np.nan,
        "subtype_macro_f1_all_cancers": np.nan,
        "subtype_accuracy_gate_positive_cancers": np.nan,
        "subtype_macro_f1_gate_positive_cancers": np.nan,
    }
    cancer_labels = [label for label in labels if label != normal_label]
    if np.any(is_cancer_true):
        out["subtype_accuracy_all_cancers"] = float(accuracy_score(y_true[is_cancer_true], y_pred[is_cancer_true]))
        out["subtype_macro_f1_all_cancers"] = float(
            f1_score(y_true[is_cancer_true], y_pred[is_cancer_true], labels=cancer_labels, average="macro", zero_division=0)
        )
    gate_pos_cancers = is_cancer_true & is_cancer_pred
    if np.any(gate_pos_cancers):
        out["subtype_accuracy_gate_positive_cancers"] = float(accuracy_score(y_true[gate_pos_cancers], y_pred[gate_pos_cancers]))
        out["subtype_macro_f1_gate_positive_cancers"] = float(
            f1_score(y_true[gate_pos_cancers], y_pred[gate_pos_cancers], labels=cancer_labels, average="macro", zero_division=0)
        )
    return out


def train_predict_config(
    matrix: np.ndarray,
    train_rows: np.ndarray,
    test_rows: np.ndarray,
    y_train: np.ndarray,
    y_test: np.ndarray,
    labels: list[str],
    normal_label: str,
    config: Config,
    threshold: float,
    args: argparse.Namespace,
    seed_offset: int,
    gate_rank: np.ndarray | None = None,
    subtype_rank: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    if gate_rank is None:
        gate_rank = rank_binary_effect(
            matrix,
            train_rows,
            y_train != normal_label,
            max(config.gate_k, 1),
            block_size=args.block_size,
            min_fraction=args.min_train_present_fraction,
            min_per_class=args.min_train_present_per_class,
            value_mode=args.value_mode,
        )
    cancer_train_mask = y_train != normal_label
    if subtype_rank is None:
        subtype_rank = rank_subtype_features(
            matrix,
            train_rows[cancer_train_mask],
            y_train[cancer_train_mask],
            max(config.subtype_k, 1),
            args,
        )
    gate_features = gate_rank[: config.gate_k]
    subtype_features = subtype_rank[: config.subtype_k]
    union_size = int(len(set(gate_features.tolist()) | set(subtype_features.tolist())))
    if union_size > args.max_union:
        raise RuntimeError(f"Union size exceeded {args.max_union}: {union_size}")

    y_gate_train = np.where(y_train == normal_label, "Normal", "Cancer")
    gate_model = make_model(config.gate_model, args.seed + seed_offset + 11, args.n_estimators, args.n_jobs)
    gate_model.fit(extract_values(matrix, train_rows, gate_features, args.value_mode), y_gate_train)
    gate_probs = positive_probability(gate_model, extract_values(matrix, test_rows, gate_features, args.value_mode))
    gate_pred_cancer = gate_probs >= threshold

    subtype_model = make_model(config.subtype_model, args.seed + seed_offset + 23, args.n_estimators, args.n_jobs)
    subtype_model.fit(
        extract_values(matrix, train_rows[cancer_train_mask], subtype_features, args.value_mode),
        y_train[cancer_train_mask],
    )
    subtype_pred = subtype_model.predict(extract_values(matrix, test_rows, subtype_features, args.value_mode))
    final_pred = np.where(gate_pred_cancer, subtype_pred, normal_label)
    info = {
        "gate_features": gate_features,
        "subtype_features": subtype_features,
        "union_size": union_size,
    }
    return final_pred, gate_probs, subtype_pred, info


def inner_evaluate_config(
    matrix: np.ndarray,
    train_rows: np.ndarray,
    y_train: np.ndarray,
    labels: list[str],
    normal_label: str,
    config: Config,
    args: argparse.Namespace,
    fold_seed: int,
) -> dict[str, object]:
    inner_splits = min(args.inner_splits, int(pd.Series(y_train).value_counts().min()))
    if inner_splits < 2:
        raise RuntimeError("Insufficient inner splits")
    splitter = StratifiedKFold(n_splits=inner_splits, shuffle=True, random_state=args.seed + fold_seed)
    rows = []
    for inner_i, (inner_train_local, inner_val_local) in enumerate(splitter.split(np.zeros(len(y_train)), y_train), start=1):
        inner_train_rows = train_rows[inner_train_local]
        inner_val_rows = train_rows[inner_val_local]
        inner_y_train = y_train[inner_train_local]
        inner_y_val = y_train[inner_val_local]
        max_gate_k = config.gate_k
        max_subtype_k = config.subtype_k
        gate_rank = rank_binary_effect(
            matrix,
            inner_train_rows,
            inner_y_train != normal_label,
            max_gate_k,
            block_size=args.block_size,
            min_fraction=args.min_train_present_fraction,
            min_per_class=args.min_train_present_per_class,
            value_mode=args.value_mode,
        )
        subtype_rank = rank_subtype_features(
            matrix,
            inner_train_rows[inner_y_train != normal_label],
            inner_y_train[inner_y_train != normal_label],
            max_subtype_k,
            args,
        )
        pred, prob, subtype_pred, info = train_predict_config(
            matrix,
            inner_train_rows,
            inner_val_rows,
            inner_y_train,
            inner_y_val,
            labels,
            normal_label,
            config,
            threshold=0.5,
            args=args,
            seed_offset=10000 * fold_seed + inner_i,
            gate_rank=gate_rank,
            subtype_rank=subtype_rank,
        )
        for local_i, row in enumerate(inner_val_rows):
            rows.append(
                {
                    "row": int(row),
                    "true_label": inner_y_val[local_i],
                    "gate_prob_cancer": float(prob[local_i]),
                    "subtype_pred": subtype_pred[local_i],
                    "union_size": info["union_size"],
                }
            )
    inner = pd.DataFrame(rows)
    y_true = inner["true_label"].to_numpy()
    y_binary = y_true != normal_label
    threshold, feasible, spec, sens = choose_threshold(inner["gate_prob_cancer"].to_numpy(float), y_binary, args.target_specificity)
    final_pred = np.where(inner["gate_prob_cancer"].to_numpy(float) >= threshold, inner["subtype_pred"].to_numpy(str), normal_label)
    metrics = workflow_metrics(y_true, final_pred, labels, normal_label)
    return {
        "config_id": config.config_id,
        "threshold": threshold,
        "threshold_feasible": feasible,
        "inner_threshold_specificity": spec,
        "inner_threshold_sensitivity": sens,
        "inner_mean_union_size": float(inner["union_size"].mean()),
        **metrics,
    }


def summarize_by_stage(predictions: pd.DataFrame, labels: list[str], normal_label: str) -> pd.DataFrame:
    rows = []
    for stage_group, group in predictions.groupby("stage_group", sort=True):
        y_true = group["true_label"].to_numpy(str)
        y_pred = group["final_pred"].to_numpy(str)
        rows.append({"stage_group": stage_group, "n_rows": int(len(group)), "n_unique_samples": int(group["sample_id"].nunique()), **workflow_metrics(y_true, y_pred, labels, normal_label)})
    return pd.DataFrame(rows)


def summarize_per_class(predictions: pd.DataFrame, labels: list[str], normal_label: str) -> pd.DataFrame:
    rows = []
    for label in labels:
        group = predictions[predictions["true_label"].eq(label)]
        if len(group) == 0:
            continue
        rows.append(
            {
                "label": label,
                "n_rows": int(len(group)),
                "n_unique_samples": int(group["sample_id"].nunique()),
                "recall": float(recall_score(group["true_label"], group["final_pred"], labels=[label], average="macro", zero_division=0)),
                "precision": float(precision_score(group["true_label"], group["final_pred"], labels=[label], average="macro", zero_division=0)),
                "f1": float(f1_score(group["true_label"], group["final_pred"], labels=[label], average="macro", zero_division=0)),
            }
        )
    return pd.DataFrame(rows)


def write_report(out_dir: Path, aggregate: pd.DataFrame, stage: pd.DataFrame, per_class: pd.DataFrame) -> None:
    def md_table(df: pd.DataFrame) -> str:
        cols = list(df.columns)
        lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
        for _, row in df.iterrows():
            vals = []
            for col in cols:
                val = row[col]
                if isinstance(val, float):
                    vals.append("NA" if pd.isna(val) else f"{val:.4f}")
                else:
                    vals.append("NA" if pd.isna(val) else str(val))
            lines.append("| " + " | ".join(vals) + " |")
        return "\n".join(lines)

    lines = [
        "# Specificity-Constrained cfDNA Two-Stage Readback",
        "",
        "Gate threshold is selected inside inner CV to target at least 80% normal specificity, then locked for each outer held-out fold.",
        "",
        "## Aggregate",
        "",
        md_table(aggregate),
        "",
        "## Stage Groups",
        "",
        md_table(stage),
        "",
        "## Per Class",
        "",
        md_table(per_class),
    ]
    (out_dir / "specificity_constrained_twostage_report.md").write_text("\n".join(lines) + "\n")


def main() -> int:
    args = parse_args()
    started = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    matrix, meta, labels, normal_label = load_data(args)
    configs = build_configs(args)
    max_gate_k = max(c.gate_k for c in configs)
    max_subtype_k = max(c.subtype_k for c in configs)
    y = meta["label"].to_numpy(str)
    splitter = RepeatedStratifiedKFold(n_splits=args.outer_splits, n_repeats=args.outer_repeats, random_state=args.seed)
    fold_rows = []
    prediction_rows = []
    selected_feature_rows = []
    inner_rows = []
    for fold_id, (train_local, test_local) in enumerate(splitter.split(np.zeros(len(y)), y), start=1):
        train_rows = train_local.astype(np.int64)
        test_rows = test_local.astype(np.int64)
        y_train = y[train_local]
        y_test = y[test_local]
        config_results = []
        for config in configs:
            result = inner_evaluate_config(matrix, train_rows, y_train, labels, normal_label, config, args, fold_id)
            config_results.append(result)
            inner_rows.append({"fold_id": fold_id, **result})
        config_df = pd.DataFrame(config_results)
        config_df["_feasible_rank"] = config_df["threshold_feasible"].astype(int)
        best_row = config_df.sort_values(
            [
                "_feasible_rank",
                "inner_threshold_sensitivity",
                args.config_subtype_metric,
                "macro_f1",
                "accuracy",
            ],
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
        fold_rows.append(
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
        )
        for feature_type, features in [("gate", info["gate_features"]), ("subtype", info["subtype_features"])]:
            for rank, feature_index in enumerate(features, start=1):
                selected_feature_rows.append(
                    {
                        "fold_id": fold_id,
                        "feature_type": feature_type,
                        "rank": rank,
                        "feature_index": int(feature_index),
                    }
                )
        for local_i, row_i in enumerate(test_rows):
            prediction_rows.append(
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
        print(f"completed fold {fold_id}/{args.outer_splits * args.outer_repeats}: {best.config_id}", flush=True)

    fold_df = pd.DataFrame(fold_rows)
    pred_df = pd.DataFrame(prediction_rows)
    selected_df = pd.DataFrame(selected_feature_rows)
    inner_df = pd.DataFrame(inner_rows)
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
        "elapsed_sec": round(time.time() - started, 1),
        "dataset": args.dataset,
        "matrix": str(args.matrix),
        "sample_metadata": str(args.sample_metadata),
        "stage_metadata": str(args.stage_metadata),
        "n_samples": int(len(meta)),
        "n_features": int(matrix.shape[1]),
        "labels": labels,
        "normal_label": normal_label,
        "target_specificity": args.target_specificity,
        "max_union": args.max_union,
        "value_mode": args.value_mode,
        "subtype_selector": args.subtype_selector,
        "config_subtype_metric": args.config_subtype_metric,
        "pairwise_confusion_splits": args.pairwise_confusion_splits,
        "pairwise_confusion_weight_scale": args.pairwise_confusion_weight_scale,
        "pairwise_confusion_model": args.pairwise_confusion_model,
        "budgets": args.budgets,
        "model_families": parse_csv_strings(args.model_families),
        "outer_splits": args.outer_splits,
        "outer_repeats": args.outer_repeats,
        "inner_splits": args.inner_splits,
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
