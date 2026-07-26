from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def _as_series(y_train: Any, n_rows: int) -> pd.Series:
    if isinstance(y_train, pd.Series):
        series = y_train.copy()
    else:
        arr = np.asarray(y_train, dtype=object).reshape(-1)
        series = pd.Series(arr)
    if len(series) != n_rows:
        raise ValueError(f"Expected {n_rows} labels but got {len(series)}")
    if not isinstance(series.index, pd.Index) or len(series.index) != n_rows:
        series.index = np.arange(n_rows)
    return series.astype(str)


def _safe_corrcoef(X: np.ndarray) -> np.ndarray:
    if X.shape[1] <= 1:
        return np.eye(X.shape[1], dtype=float)
    X_centered = X - X.mean(axis=0, keepdims=True)
    scale = X_centered.std(axis=0, ddof=0, keepdims=True)
    scale[scale == 0.0] = 1.0
    X_scaled = X_centered / scale
    corr = np.corrcoef(X_scaled, rowvar=False)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    if corr.ndim == 0:
        corr = np.array([[1.0]], dtype=float)
    np.fill_diagonal(corr, 1.0)
    return np.abs(corr)


def _normalize_pair_key(left: str, right: str) -> tuple[str, str]:
    a = str(left).strip()
    b = str(right).strip()
    return tuple(sorted((a, b)))


def _normalize_target_pairs(raw_pairs: Any) -> set[tuple[str, str]]:
    normalized: set[tuple[str, str]] = set()
    for item in list(raw_pairs or []):
        if isinstance(item, dict):
            left = str(item.get("a", "")).strip()
            right = str(item.get("b", "")).strip()
        elif isinstance(item, (list, tuple)) and len(item) >= 2:
            left = str(item[0]).strip()
            right = str(item[1]).strip()
        else:
            continue
        if left and right and left != right:
            normalized.add(_normalize_pair_key(left, right))
    return normalized


def _hard_pair_probe_score(
    class_medians: np.ndarray,
    *,
    pair_quantile: float,
    top_n: int,
    pair_power: float,
) -> tuple[np.ndarray, dict[str, float]]:
    n_classes, n_probes = class_medians.shape
    if n_classes < 2 or top_n <= 0:
        return np.zeros(n_probes, dtype=float), {}

    pair_effects: list[np.ndarray] = []
    pair_gaps: list[float] = []
    for left in range(n_classes - 1):
        for right in range(left + 1, n_classes):
            diff = np.abs(class_medians[left] - class_medians[right])
            diff = np.nan_to_num(diff, nan=0.0, posinf=0.0, neginf=0.0)
            pair_effects.append(diff)
            pair_gaps.append(float(np.quantile(diff, pair_quantile)))

    if not pair_effects:
        return np.zeros(n_probes, dtype=float), {}

    pair_gaps_arr = np.asarray(pair_gaps, dtype=float)
    hardest_idx = np.argsort(pair_gaps_arr, kind="mergesort")[: min(len(pair_gaps_arr), int(top_n))]
    selected_gaps = pair_gaps_arr[hardest_idx]
    selected_effects = np.vstack([pair_effects[idx] for idx in hardest_idx])

    hardness = np.power(np.maximum(1.0 - selected_gaps, 1e-6), max(pair_power, 0.0))
    if not np.isfinite(hardness).all() or not np.any(hardness > 0):
        hardness = np.ones(len(hardest_idx), dtype=float)
    hardness = hardness / np.sum(hardness)

    score = np.average(selected_effects, axis=0, weights=hardness)
    stats = {
        "hard_pair_count": float(len(hardest_idx)),
        "hard_pair_gap_median": float(np.median(selected_gaps)),
        "hard_pair_gap_max": float(np.max(selected_gaps)),
    }
    return score, stats


def _target_pair_probe_score(
    class_medians: np.ndarray,
    class_labels: list[str],
    *,
    target_pairs: set[tuple[str, str]],
    pair_quantile: float,
    pair_power: float,
    coverage_quantile: float,
    coverage_weight: float,
) -> tuple[np.ndarray, dict[str, float]]:
    n_classes, n_probes = class_medians.shape
    if n_classes < 2 or not target_pairs:
        return np.zeros(n_probes, dtype=float), {}

    pair_effects: list[np.ndarray] = []
    pair_gaps: list[float] = []
    for left in range(n_classes - 1):
        for right in range(left + 1, n_classes):
            key = _normalize_pair_key(class_labels[left], class_labels[right])
            if key not in target_pairs:
                continue
            diff = np.abs(class_medians[left] - class_medians[right])
            diff = np.nan_to_num(diff, nan=0.0, posinf=0.0, neginf=0.0)
            pair_effects.append(diff)
            pair_gaps.append(float(np.quantile(diff, pair_quantile)))

    if not pair_effects:
        return np.zeros(n_probes, dtype=float), {}

    pair_effect_matrix = np.vstack(pair_effects)
    pair_gaps_arr = np.asarray(pair_gaps, dtype=float)
    hardness = np.power(np.maximum(1.0 - pair_gaps_arr, 1e-6), max(pair_power, 0.0))
    if not np.isfinite(hardness).all() or not np.any(hardness > 0):
        hardness = np.ones(len(pair_effects), dtype=float)
    hardness = hardness / np.sum(hardness)

    mean_score = np.average(pair_effect_matrix, axis=0, weights=hardness)
    coverage_score = np.quantile(pair_effect_matrix, coverage_quantile, axis=0)
    score = coverage_weight * coverage_score + (1.0 - coverage_weight) * mean_score
    stats = {
        "target_pair_count": float(len(pair_effects)),
        "target_pair_gap_median": float(np.median(pair_gaps_arr)),
        "target_pair_gap_max": float(np.max(pair_gaps_arr)),
        "target_pair_coverage_quantile": coverage_quantile,
        "target_pair_coverage_weight": coverage_weight,
    }
    return score, stats


def run_selector(
    *,
    X_train_pref: pd.DataFrame,
    y_train: Any,
    all_classes: list[str] | tuple[str, ...],
    k: int,
    prefilter_probes: list[str] | tuple[str, ...],
    seed: int,
    selection_args: dict[str, Any] | None = None,
    prefilter_args: dict[str, Any] | None = None,
) -> dict[str, Any]:
    del seed, prefilter_args

    if not isinstance(X_train_pref, pd.DataFrame):
        X_train_pref = pd.DataFrame(X_train_pref, columns=list(prefilter_probes))
    if X_train_pref.empty:
        return {"selected_probes": [], "score_df": pd.DataFrame(columns=["probe_id", "score"]), "extra_json": {}}

    args = dict(selection_args or {})
    coverage_quantile = float(args.get("coverage_quantile", 0.2))
    coverage_weight = float(args.get("coverage_weight", 0.45))
    floor_weight = float(args.get("floor_weight", 0.3))
    rare_class_power = float(args.get("rare_class_power", 0.5))
    redundancy_penalty = float(args.get("redundancy_penalty", 0.18))
    pool_size = int(args.get("pool_size", max(int(k) * 7, int(k))))
    hard_pair_weight = float(args.get("hard_pair_weight", 0.0))
    hard_pair_quantile = float(args.get("hard_pair_quantile", 0.9))
    hard_pair_top_n = int(args.get("hard_pair_top_n", 0))
    hard_pair_power = float(args.get("hard_pair_power", 1.0))
    min_class_n = max(2, int(args.get("min_class_n", 2)))
    rare_classes = {str(item).strip() for item in list(args.get("rare_classes", []) or []) if str(item).strip()}
    rare_class_weight = float(args.get("rare_class_weight", 0.0))
    target_pairs = _normalize_target_pairs(args.get("target_pairs", []))
    target_pair_weight = float(args.get("target_pair_weight", 0.0))
    target_pair_coverage_quantile = float(args.get("target_pair_coverage_quantile", 0.25))
    target_pair_coverage_weight = float(args.get("target_pair_coverage_weight", 0.55))

    coverage_quantile = min(max(coverage_quantile, 0.0), 1.0)
    coverage_weight = min(max(coverage_weight, 0.0), 1.0)
    floor_weight = min(max(floor_weight, 0.0), 1.0 - coverage_weight)
    rare_class_power = max(0.0, rare_class_power)
    pool_size = max(int(k), pool_size)
    hard_pair_weight = min(max(hard_pair_weight, 0.0), 1.0)
    hard_pair_quantile = min(max(hard_pair_quantile, 0.0), 1.0)
    hard_pair_top_n = max(0, hard_pair_top_n)
    hard_pair_power = max(0.0, hard_pair_power)
    rare_class_weight = min(max(rare_class_weight, 0.0), 1.0)
    target_pair_weight = min(max(target_pair_weight, 0.0), 1.0)
    target_pair_coverage_quantile = min(max(target_pair_coverage_quantile, 0.0), 1.0)
    target_pair_coverage_weight = min(max(target_pair_coverage_weight, 0.0), 1.0)

    X = X_train_pref.apply(pd.to_numeric, errors="coerce")
    y = _as_series(y_train, len(X)).reset_index(drop=True)
    X = X.reset_index(drop=True)
    probe_ids = X.columns.astype(str).tolist()

    X_values = X.to_numpy(dtype=float, copy=True)
    col_medians = np.nanmedian(X_values, axis=0)
    col_medians = np.where(np.isfinite(col_medians), col_medians, 0.0)
    nan_mask = np.isnan(X_values)
    if nan_mask.any():
        X_values[nan_mask] = np.take(col_medians, np.where(nan_mask)[1])

    X_rank = pd.DataFrame(X_values, columns=probe_ids).rank(axis=0, method="average", pct=True)
    X_rank_values = X_rank.to_numpy(dtype=float, copy=False)

    observed_classes = set(y.astype(str))
    classes = [str(cls) for cls in all_classes if str(cls) in observed_classes]
    if len(classes) < 2:
        score_df = pd.DataFrame({"probe_id": probe_ids, "score": np.zeros(len(probe_ids), dtype=float)})
        return {
            "selected_probes": probe_ids[: int(k)],
            "score_df": score_df,
            "extra_json": {"n_effective_classes": float(len(classes))},
        }

    class_effects: list[np.ndarray] = []
    class_medians: list[np.ndarray] = []
    class_weights: list[float] = []
    effective_class_labels: list[str] = []
    for cls in classes:
        mask = (y == cls).to_numpy()
        cls_block = X_rank_values[mask]
        rest_block = X_rank_values[~mask]
        if cls_block.shape[0] < min_class_n or rest_block.shape[0] < min_class_n:
            continue
        cls_median = np.median(cls_block, axis=0)
        rest_median = np.median(rest_block, axis=0)
        effect = np.abs(cls_median - rest_median)
        class_effects.append(np.nan_to_num(effect, nan=0.0, posinf=0.0, neginf=0.0))
        class_medians.append(np.nan_to_num(cls_median, nan=0.0, posinf=0.0, neginf=0.0))
        class_weights.append(float(max(cls_block.shape[0], 1)) ** (-rare_class_power))
        effective_class_labels.append(cls)

    if not class_effects:
        score_df = pd.DataFrame({"probe_id": probe_ids, "score": np.zeros(len(probe_ids), dtype=float)})
        return {"selected_probes": probe_ids[: int(k)], "score_df": score_df, "extra_json": {"n_effective_classes": 0.0}}

    effect_matrix = np.vstack(class_effects)
    class_median_matrix = np.vstack(class_medians)
    weights = np.asarray(class_weights, dtype=float)
    if not np.isfinite(weights).all() or not np.any(weights > 0):
        weights = np.ones(effect_matrix.shape[0], dtype=float)
    weights = weights / np.sum(weights)

    floor_score = np.min(effect_matrix, axis=0)
    coverage_score = np.quantile(effect_matrix, coverage_quantile, axis=0)
    mean_score = np.average(effect_matrix, axis=0, weights=weights)
    mean_weight = max(0.0, 1.0 - coverage_weight - floor_weight)
    base_score = floor_weight * floor_score + coverage_weight * coverage_score + mean_weight * mean_score

    hard_pair_score = np.zeros_like(base_score)
    pair_stats: dict[str, float] = {}
    if hard_pair_weight > 0.0 and len(class_medians) >= 2:
        hard_pair_score, pair_stats = _hard_pair_probe_score(
            class_median_matrix,
            pair_quantile=hard_pair_quantile,
            top_n=hard_pair_top_n,
            pair_power=hard_pair_power,
        )

    rare_class_score = np.zeros_like(base_score)
    rare_class_stats: dict[str, float] = {}
    rare_idx = [idx for idx, label in enumerate(effective_class_labels) if label in rare_classes]
    if rare_idx:
        rare_weights = weights[rare_idx]
        rare_weights = rare_weights / np.sum(rare_weights)
        rare_class_score = np.average(effect_matrix[rare_idx], axis=0, weights=rare_weights)
        rare_class_stats = {"target_rare_class_count": float(len(rare_idx))}

    target_pair_score = np.zeros_like(base_score)
    target_pair_stats: dict[str, float] = {}
    if target_pairs:
        target_pair_score, target_pair_stats = _target_pair_probe_score(
            class_median_matrix,
            effective_class_labels,
            target_pairs=target_pairs,
            pair_quantile=hard_pair_quantile,
            pair_power=hard_pair_power,
            coverage_quantile=target_pair_coverage_quantile,
            coverage_weight=target_pair_coverage_weight,
        )

    active_weights = {
        "hard_pair": hard_pair_weight if pair_stats else 0.0,
        "rare_class": rare_class_weight if rare_class_stats else 0.0,
        "target_pair": target_pair_weight if target_pair_stats else 0.0,
    }
    active_total = sum(active_weights.values())
    if active_total > 1.0:
        scale = 1.0 / active_total
        active_weights = {key: value * scale for key, value in active_weights.items()}
        active_total = 1.0
    base_mix_weight = max(0.0, 1.0 - active_total)

    combined_score = (
        base_mix_weight * base_score
        + active_weights["hard_pair"] * hard_pair_score
        + active_weights["rare_class"] * rare_class_score
        + active_weights["target_pair"] * target_pair_score
    )

    score_df = pd.DataFrame(
        {
            "probe_id": probe_ids,
            "score": combined_score,
            "base_score": base_score,
            "floor_score": floor_score,
            "coverage_score": coverage_score,
            "mean_score": mean_score,
            "hard_pair_score": hard_pair_score,
            "rare_class_score": rare_class_score,
            "target_pair_score": target_pair_score,
        }
    ).sort_values(
        [
            "score",
            "base_score",
            "floor_score",
            "coverage_score",
            "mean_score",
            "hard_pair_score",
            "rare_class_score",
            "target_pair_score",
            "probe_id",
        ],
        ascending=[False, False, False, False, False, False, False, False, True],
        kind="mergesort",
    )

    ranked_probes = score_df["probe_id"].astype(str).tolist()
    top_pool = ranked_probes[: min(len(ranked_probes), pool_size)]
    if not top_pool:
        return {
            "selected_probes": [],
            "score_df": score_df.reset_index(drop=True),
            "extra_json": {"n_effective_classes": float(len(class_effects))},
        }

    top_positions = [probe_ids.index(probe) for probe in top_pool]
    corr = _safe_corrcoef(X_rank_values[:, top_positions])
    top_scores = score_df.set_index("probe_id").loc[top_pool, "score"].to_numpy(dtype=float)

    selected_idx: list[int] = []
    remaining = list(range(len(top_pool)))
    while remaining and len(selected_idx) < int(k):
        best_pos = remaining[0]
        best_value = -np.inf
        for pos in remaining:
            penalty = 0.0 if not selected_idx else float(np.max(corr[pos, selected_idx]))
            adjusted = top_scores[pos] - redundancy_penalty * penalty
            if adjusted > best_value:
                best_value = adjusted
                best_pos = pos
        selected_idx.append(best_pos)
        remaining.remove(best_pos)

    selected_probes = [top_pool[pos] for pos in selected_idx]
    if len(selected_probes) < int(k):
        seen = set(selected_probes)
        for probe in ranked_probes:
            if probe in seen:
                continue
            selected_probes.append(probe)
            seen.add(probe)
            if len(selected_probes) >= int(k):
                break

    return {
        "selected_probes": selected_probes[: int(k)],
        "score_df": score_df.reset_index(drop=True),
        "extra_json": {
            "coverage_quantile": coverage_quantile,
            "coverage_weight": coverage_weight,
            "floor_weight": floor_weight,
            "mean_weight": mean_weight,
            "rare_class_power": rare_class_power,
            "redundancy_penalty": redundancy_penalty,
            "pool_size": float(min(len(ranked_probes), pool_size)),
            "min_class_n": float(min_class_n),
            "hard_pair_weight": active_weights["hard_pair"],
            "hard_pair_quantile": hard_pair_quantile,
            "hard_pair_top_n": float(hard_pair_top_n),
            "hard_pair_power": hard_pair_power,
            "rare_class_weight": active_weights["rare_class"],
            "target_pair_weight": active_weights["target_pair"],
            "target_pair_coverage_quantile": target_pair_coverage_quantile,
            "target_pair_coverage_weight": target_pair_coverage_weight,
            "base_mix_weight": base_mix_weight,
            "target_pair_requested_count": float(len(target_pairs)),
            "target_rare_class_requested_count": float(len(rare_classes)),
            "n_effective_classes": float(len(class_effects)),
            **pair_stats,
            **rare_class_stats,
            **target_pair_stats,
        },
    }
