#!/usr/bin/env python3
"""Benchmark feature-selection methods at fixed K on one outer fold."""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from split_integrity import validate_folds, validate_locked_development, validate_cached_training

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.feature_selection import f_classif, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mutual_info_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline

from minicpg.release_registry import (
    PREFILTER_ARG_ALIASES,
    SELECTOR_ARG_ALIASES,
    RegistryEntryNotFound as PluginNotFound,
    call_with_supported_kwargs,
    load_prefilter_plugin,
    load_selector_plugin,
    selector_y_train_payload,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Feature-selection runtime benchmark on one fold.")
    parser.add_argument("--probe-matrix", required=True, help="Unfiltered probe matrix parquet.")
    parser.add_argument("--manifest", required=True, help="Classification manifest TSV.")
    parser.add_argument("--folds", required=True, help="Outer folds TSV (with repeat).")
    parser.add_argument("--out-dir", required=True, help="Run output directory.")
    parser.add_argument("--repeat", type=int, required=True, help="Repeat index.")
    parser.add_argument("--fold", type=int, required=True, help="Outer fold index.")
    parser.add_argument("--k", type=int, default=30, help="Final probe count per method.")
    parser.add_argument(
        "--prefilter-method", default="mi", help="Mutual-information prefilter used in SM2."
    )
    parser.add_argument(
        "--prefilter-k", type=int, default=3500, help="Prefilter size before heavy selectors."
    )
    parser.add_argument(
        "--candidate-spec-json",
        default="",
        help="Optional candidate spec JSON path for freeform selection/prefilter args.",
    )
    parser.add_argument(
        "--methods",
        default="rare_class_rank_shift,mi_rank_baseline,pairwise_hard_weighted_auc_stable",
        help="Comma-separated selectors: rare_class_rank_shift,mi_rank_baseline,pairwise_hard_weighted_auc_stable.",
    )
    parser.add_argument(
        "--chunk-size", type=int, default=5000, help="Probe chunk size for ANOVA scanning."
    )
    parser.add_argument(
        "--inner-folds", type=int, default=3, help="Inner CV folds for classifier C tuning."
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed.")
    parser.add_argument("--c-grid", default="0.1,1,10", help="Classifier C grid.")
    parser.add_argument(
        "--sample-missing-thresh", type=float, default=0.2, help="Sample missingness threshold."
    )
    parser.add_argument(
        "--probe-missing-thresh", type=float, default=0.1, help="Probe missingness threshold."
    )
    parser.add_argument(
        "--blacklist-chen", required=True, help="Chen cross-reactive blacklist CSV."
    )
    parser.add_argument("--blacklist-snp", required=True, help="minfi SNP-affected probes list.")
    parser.add_argument(
        "--probe-metadata", required=True, help="Probe metadata TSV with chromosome column."
    )
    parser.add_argument(
        "--exclude-sex-chr", action="store_true", default=True, help="Exclude chrX/chrY probes."
    )
    parser.add_argument("--no-exclude-sex-chr", dest="exclude_sex_chr", action="store_false")
    parser.add_argument(
        "--pairwise-top-per-pair",
        type=int,
        default=30,
        help="Top probes retained per class-pair for pairwise ANOVA aggregation.",
    )
    parser.add_argument(
        "--hardpair-cv-folds",
        type=int,
        default=3,
        help="CV folds to estimate hard class-pair weights on training data.",
    )
    parser.add_argument(
        "--hardpair-baseline-topk",
        type=int,
        default=500,
        help="Top ANOVA-prefilter probes used by hard-pair baseline confusion model.",
    )
    parser.add_argument(
        "--hardpair-gamma",
        type=float,
        default=2.0,
        help="Amplification factor for hard-pair weighting.",
    )
    parser.add_argument(
        "--stable-n-boot",
        type=int,
        default=12,
        help="Bootstrap/subsample rounds for stability-aware pairwise selector.",
    )
    parser.add_argument(
        "--stable-subsample-frac",
        type=float,
        default=0.8,
        help="Per-class subsample fraction for stability rounds.",
    )
    parser.add_argument(
        "--stable-alpha",
        type=float,
        default=0.5,
        help="Mixing weight between base pairwise score and stability frequency.",
    )
    parser.add_argument("--n-jobs", type=int, default=1, help="Worker count.")
    parser.add_argument(
        "--compute-auroc", action="store_true", help="Compute macro and per-class AUROC."
    )
    parser.add_argument("--run-id", default="", help="Run identifier for heartbeat tracking.")
    parser.add_argument(
        "--heartbeat-path", default="", help="Path to JSONL heartbeat file for cluster monitoring."
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


def load_list(path: Path) -> List[str]:
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


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


def canonical_selection_method(method: str) -> str:
    name = str(method).strip()
    aliases = {
        "mi_rank_baseline": "mi",
    }
    return aliases.get(name, name)


def candidate_section(spec: Dict[str, object], key: str) -> Dict[str, object]:
    value = spec.get(key, {})
    if isinstance(value, dict):
        return dict(value)
    return {}


def override_int(overrides: Dict[str, object], key: str, default: int) -> int:
    if key not in overrides:
        return int(default)
    return int(overrides[key])


def override_float(overrides: Dict[str, object], key: str, default: float) -> float:
    if key not in overrides:
        return float(default)
    return float(overrides[key])


def override_text(overrides: Dict[str, object], key: str, default: str) -> str:
    if key not in overrides:
        return str(default)
    value = overrides[key]
    if isinstance(value, list):
        return ",".join(str(v) for v in value)
    return str(value)


def slugify(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip()).strip("_") or "na"


def plugin_prefilter(
    name: str,
    *,
    pf: pq.ParquetFile,
    train_ids: Sequence[str],
    y_train: np.ndarray,
    probe_set: set[str],
    chunk_size: int,
    seed: int,
    prefilter_k: int,
    prefilter_args: Dict[str, object],
    anova_scores: pd.DataFrame,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    plugin = load_prefilter_plugin(name)
    result = call_with_supported_kwargs(
        plugin,
        {
            "pf": pf,
            "train_ids": train_ids,
            "y_train": y_train,
            "probe_set": probe_set,
            "chunk_size": chunk_size,
            "seed": seed,
            "prefilter_k": prefilter_k,
            "prefilter_args": dict(prefilter_args),
            "anova_scores": anova_scores.copy(),
        },
        aliases=PREFILTER_ARG_ALIASES,
    )
    if isinstance(result, tuple) and len(result) == 2:
        score_df, meta = result
    elif isinstance(result, dict):
        score_df = result.get("score_df", result.get("scores"))
        meta = result.get("meta", result.get("extra_json", {}))
    else:
        raise ValueError(f"Prefilter plugin {name} returned unsupported payload type")
    if isinstance(score_df, pd.Series):
        score_df = score_df.rename_axis("probe_id").reset_index(name="score")
    if not isinstance(score_df, pd.DataFrame) or not {"probe_id", "score"}.issubset(
        score_df.columns
    ):
        raise ValueError(
            f"Prefilter plugin {name} must return score_df with probe_id and score columns"
        )
    if not isinstance(meta, dict):
        meta = {}
    return score_df.sort_values("score", ascending=False), {
        str(k): float(v) if isinstance(v, (int, float, np.integer, np.floating)) else v
        for k, v in meta.items()
    }


def plugin_selector(
    name: str,
    *,
    X_train_pref: pd.DataFrame,
    y_train: np.ndarray,
    all_classes: Sequence[str],
    k: int,
    prefilter_probes: Sequence[str],
    seed: int,
    selection_args: Dict[str, object],
    prefilter_args: Dict[str, object],
) -> Tuple[List[str], pd.DataFrame, Dict[str, object]]:
    plugin = load_selector_plugin(name)
    prefilter_probe_list = [str(v) for v in prefilter_probes]
    result = call_with_supported_kwargs(
        plugin,
        {
            "X_train_pref": X_train_pref.copy(),
            "y_train": selector_y_train_payload(X_train_pref, np.array(y_train, copy=True)),
            "all_classes": list(all_classes),
            "k": int(k),
            "prefilter_probes": prefilter_probe_list,
            "seed": int(seed),
            "selection_args": dict(selection_args),
            "prefilter_args": dict(prefilter_args),
        },
        aliases=SELECTOR_ARG_ALIASES,
    )
    if not isinstance(result, dict):
        raise ValueError(f"Selector plugin {name} must return a dict")
    selected_raw = result.get("selected_probes")
    if selected_raw is None and "selected_indices" in result:
        selected_raw = []
        for value in result.get("selected_indices", []):
            idx = int(value)
            if idx < 0 or idx >= len(prefilter_probe_list):
                raise ValueError(f"Selector plugin {name} returned out-of-range selected_indices")
            selected_raw.append(prefilter_probe_list[idx])
    if isinstance(selected_raw, (pd.Index, np.ndarray, tuple)):
        selected_raw = list(selected_raw)
    if selected_raw is None:
        selected_raw = []
    if not isinstance(selected_raw, list):
        raise ValueError(f"Selector plugin {name} returned invalid selected_probes")
    selected_probes = [str(v) for v in selected_raw if str(v).strip()]
    score_df = result.get(
        "score_df", result.get("scores", pd.DataFrame(columns=["probe_id", "score"]))
    )
    if isinstance(score_df, pd.Series):
        score_df = score_df.rename_axis("probe_id").reset_index(name="score")
    extra_json = result.get("extra_json", result.get("meta", {}))
    if not isinstance(score_df, pd.DataFrame):
        raise ValueError(f"Selector plugin {name} returned invalid score_df")
    if not {"probe_id", "score"}.issubset(score_df.columns):
        raise ValueError(f"Selector plugin {name} returned score_df without probe_id/score columns")
    if not isinstance(extra_json, dict):
        extra_json = {}
    return selected_probes, score_df, extra_json


def canonical_selected_order_df(
    selected_probes: Sequence[str], score_df: pd.DataFrame
) -> pd.DataFrame:
    probe_list = [str(v) for v in selected_probes if str(v).strip()]
    if not probe_list:
        raise ValueError("selected_probes is empty")
    deduped = list(dict.fromkeys(probe_list))
    if len(deduped) != len(probe_list):
        raise ValueError("selected_probes contains duplicates")
    score_map: Dict[str, float] = {}
    if isinstance(score_df, pd.DataFrame) and {"probe_id", "score"}.issubset(score_df.columns):
        score_source = score_df.loc[:, ["probe_id", "score"]].copy()
        score_source["probe_id"] = score_source["probe_id"].astype(str)
        score_source = score_source.dropna(subset=["probe_id"])
        score_source = score_source.drop_duplicates(subset=["probe_id"], keep="first")
        for probe_id, score in score_source.itertuples(index=False):
            try:
                score_map[str(probe_id)] = float(score)
            except (TypeError, ValueError):
                continue
    rows = []
    for idx, probe_id in enumerate(deduped, start=1):
        row = {"rank": idx, "probe_id": probe_id}
        if probe_id in score_map:
            row["score"] = score_map[probe_id]
        rows.append(row)
    return pd.DataFrame(rows)


def load_blacklist_chen(path: Path) -> set[str]:
    probes = set()
    with path.open() as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for row in reader:
            if row:
                probes.add(row[0].strip())
    return probes


def load_blacklist_simple(path: Path) -> set[str]:
    return set(load_list(path))


def load_sex_probes(path: Path) -> set[str]:
    df = pd.read_csv(path, sep="\t", dtype=str)
    if "probe_id" not in df.columns or "chromosome" not in df.columns:
        raise ValueError("probe_metadata must contain probe_id and chromosome columns")
    chrom = df["chromosome"].str.lower()
    return set(df.loc[chrom.isin(["chrx", "chry", "x", "y"]), "probe_id"].astype(str))


def compute_missingness(
    pf: pq.ParquetFile,
    sample_ids: List[str],
) -> Tuple[np.ndarray, List[str], List[int]]:
    sample_missing = np.zeros(len(sample_ids), dtype=np.int64)
    probe_ids: List[str] = []
    probe_missing: List[int] = []

    for rg in range(pf.num_row_groups):
        table = pf.read_row_group(rg, columns=sample_ids + ["probe_id"])
        df = table.to_pandas()
        X = df[sample_ids].to_numpy(dtype=np.float32)
        mask = np.isnan(X)
        sample_missing += mask.sum(axis=0)
        probe_missing.extend(mask.sum(axis=1).tolist())
        probe_ids.extend(df["probe_id"].astype(str).tolist())

    return sample_missing, probe_ids, probe_missing


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


def read_selected(
    pf: pq.ParquetFile,
    sample_ids: Sequence[str],
    probes: Sequence[str],
) -> pd.DataFrame:
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


def score_anova_full(
    pf: pq.ParquetFile,
    train_ids: List[str],
    y_train: np.ndarray,
    probe_set: set[str],
    chunk_size: int,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    scores: List[Tuple[str, float]] = []
    t_read = 0.0
    t_score = 0.0

    for df in read_matrix_rows(pf, train_ids, probe_set):
        for start in range(0, len(df), chunk_size):
            chunk = df.iloc[start : start + chunk_size]

            t0 = time.perf_counter()
            X = chunk[train_ids].to_numpy(dtype=np.float32).T
            t_read += time.perf_counter() - t0

            t1 = time.perf_counter()
            med = np.nanmedian(X, axis=0)
            keep = ~np.isnan(med)
            if not keep.any():
                t_score += time.perf_counter() - t1
                continue

            probe_ids = chunk["probe_id"].astype(str).to_numpy()[keep]
            if not keep.all():
                X = X[:, keep]
                med = med[keep]

            nan_mask = np.isnan(X)
            if nan_mask.any():
                X[nan_mask] = np.take(med, np.where(nan_mask)[1])

            f_vals, _ = f_classif(X, y_train)
            f_vals = np.nan_to_num(f_vals, nan=-np.inf, neginf=-np.inf, posinf=-np.inf)
            scores.extend(zip(probe_ids.tolist(), f_vals.tolist()))
            t_score += time.perf_counter() - t1

    if not scores:
        raise ValueError("No ANOVA scores computed")

    out = pd.DataFrame(scores, columns=["probe_id", "score"])
    out = out.groupby("probe_id", as_index=False)["score"].max()
    out = out.sort_values("score", ascending=False).reset_index(drop=True)
    return out, {"t_read_train_matrix_anova": t_read, "t_anova_scoring": t_score}


def score_mi_full(
    pf: pq.ParquetFile,
    train_ids: List[str],
    y_train: np.ndarray,
    probe_set: set[str],
    chunk_size: int,
    seed: int,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    scores: List[Tuple[str, float]] = []
    y_int, _ = pd.factorize(y_train)
    t_read = 0.0
    t_score = 0.0

    for df in read_matrix_rows(pf, train_ids, probe_set):
        for start in range(0, len(df), chunk_size):
            chunk = df.iloc[start : start + chunk_size]

            t0 = time.perf_counter()
            X = chunk[train_ids].to_numpy(dtype=np.float32).T
            t_read += time.perf_counter() - t0

            t1 = time.perf_counter()
            med = np.nanmedian(X, axis=0)
            keep = ~np.isnan(med)
            if not keep.any():
                t_score += time.perf_counter() - t1
                continue

            probe_ids = chunk["probe_id"].astype(str).to_numpy()[keep]
            if not keep.all():
                X = X[:, keep]
                med = med[keep]

            nan_mask = np.isnan(X)
            if nan_mask.any():
                X[nan_mask] = np.take(med, np.where(nan_mask)[1])

            mi = mutual_info_classif(X, y_int, random_state=seed)
            mi = np.nan_to_num(mi, nan=-np.inf, neginf=-np.inf, posinf=-np.inf)
            scores.extend(zip(probe_ids.tolist(), mi.tolist()))
            t_score += time.perf_counter() - t1

    if not scores:
        raise ValueError("No MI scores computed")

    out = pd.DataFrame(scores, columns=["probe_id", "score"])
    out = out.groupby("probe_id", as_index=False)["score"].max()
    out = out.sort_values("score", ascending=False).reset_index(drop=True)
    return out, {"t_read_train_matrix_mi_prefilter": t_read, "t_mi_prefilter_scoring": t_score}


def compute_auc_by_class(
    y_true: np.ndarray, proba: np.ndarray, classes: np.ndarray
) -> List[Dict[str, float]]:
    rows = []
    for idx, cls in enumerate(classes):
        y_bin = (y_true == cls).astype(int)
        n_pos = int(y_bin.sum())
        n_neg = int(len(y_bin) - n_pos)
        if n_pos == 0 or n_neg == 0:
            auc = float("nan")
        else:
            auc = float(roc_auc_score(y_bin, proba[:, idx]))
        rows.append({"class": str(cls), "auc": auc, "n_pos": n_pos, "n_neg": n_neg})
    return rows


def score_pairwise_anova(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    top_per_pair: int,
    mode: str,
    pair_weights: Optional[Dict[Tuple[str, str], float]] = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X_train).astype(np.float32, copy=False)
    y = np.asarray(y_train)
    n_features = X_imp.shape[1]
    top_per_pair = max(1, min(int(top_per_pair), n_features))

    class_stats: Dict[str, Tuple[int, np.ndarray, np.ndarray]] = {}
    for cls in sorted(pd.unique(y)):
        mask = y == cls
        n_cls = int(mask.sum())
        if n_cls < 2:
            continue
        x_cls = X_imp[mask]
        mean = x_cls.mean(axis=0, dtype=np.float64)
        var = x_cls.var(axis=0, ddof=1, dtype=np.float64)
        class_stats[str(cls)] = (n_cls, mean, var)

    classes = sorted(class_stats)
    if len(classes) < 2:
        raise ValueError("pairwise ANOVA requires >=2 classes with >=2 samples each")

    freq = np.zeros(n_features, dtype=np.float64)
    inv_rank = np.zeros(n_features, dtype=np.float64)
    max_f = np.zeros(n_features, dtype=np.float64)
    sum_f = np.zeros(n_features, dtype=np.float64)
    n_pairs = 0
    weight_total = 0.0

    t0 = time.perf_counter()
    eps = 1e-12
    for i in range(len(classes)):
        cls_i = classes[i]
        n_i, mean_i, var_i = class_stats[cls_i]
        for j in range(i + 1, len(classes)):
            cls_j = classes[j]
            n_j, mean_j, var_j = class_stats[cls_j]
            denom = (var_i / float(n_i)) + (var_j / float(n_j))
            with np.errstate(divide="ignore", invalid="ignore"):
                f_score = ((mean_i - mean_j) ** 2) / (denom + eps)
            f_score = np.nan_to_num(f_score, nan=0.0, neginf=0.0, posinf=0.0)
            weight = 1.0
            if pair_weights is not None:
                weight = float(
                    pair_weights.get((cls_i, cls_j), pair_weights.get((cls_j, cls_i), 1.0))
                )

            top_idx = np.argpartition(f_score, -top_per_pair)[-top_per_pair:]
            order = top_idx[np.argsort(f_score[top_idx])[::-1]]
            ranks = np.arange(1, len(order) + 1, dtype=np.float64)

            freq[order] += weight
            inv_rank[order] += weight * (1.0 / ranks)
            sum_f[order] += weight * f_score[order]
            max_f[order] = np.maximum(max_f[order], weight * f_score[order])
            n_pairs += 1
            weight_total += weight

    t_pair = time.perf_counter() - t0

    if mode == "freq":
        # Primary signal is pairwise support count; tie-break with rank and strength.
        scores = freq + (1e-3 * inv_rank) + (1e-9 * max_f)
    elif mode == "rank":
        # Primary signal is reciprocal-rank accumulation across all class pairs.
        scores = inv_rank + (1e-4 * freq) + (1e-9 * sum_f)
    else:
        raise ValueError(f"Unsupported pairwise ANOVA mode: {mode}")

    meta = {
        "pairwise_mode": mode,
        "pairwise_top_per_pair": float(top_per_pair),
        "pairwise_n_pairs": float(n_pairs),
        "pairwise_weight_sum": float(weight_total),
        "pairwise_weight_mean": float(weight_total / max(1, n_pairs)),
        "pairwise_n_valid_classes": float(len(classes)),
        "t_pairwise_scoring": t_pair,
    }
    return scores, meta


def compute_pairwise_auc_separation(X: np.ndarray, y_bin: np.ndarray) -> np.ndarray:
    n, p = X.shape
    n_pos = int(y_bin.sum())
    n_neg = int(n - n_pos)
    if n_pos == 0 or n_neg == 0:
        return np.zeros(p, dtype=np.float64)

    # Fast column-wise rank-based AUC approximation.
    order = np.argsort(X, axis=0, kind="mergesort")
    ranks = np.empty((n, p), dtype=np.float64)
    col_idx = np.arange(p)
    ranks[order, col_idx] = np.arange(1, n + 1, dtype=np.float64)[:, None]

    sum_pos = ranks[y_bin == 1].sum(axis=0)
    u_stat = sum_pos - (float(n_pos) * float(n_pos + 1) / 2.0)
    auc = u_stat / float(n_pos * n_neg)
    sep = np.abs(auc - 0.5) * 2.0
    return np.clip(np.nan_to_num(sep, nan=0.0, neginf=0.0, posinf=1.0), 0.0, 1.0)


def score_pairwise_auc(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    top_per_pair: int,
    mode: str,
    pair_weights: Optional[Dict[Tuple[str, str], float]] = None,
) -> Tuple[np.ndarray, Dict[str, float]]:
    imputer = SimpleImputer(strategy="median")
    X_imp = imputer.fit_transform(X_train).astype(np.float32, copy=False)
    y = np.asarray(y_train)
    n_features = X_imp.shape[1]
    top_per_pair = max(1, min(int(top_per_pair), n_features))

    classes = sorted([str(c) for c in pd.unique(y)])
    if len(classes) < 2:
        raise ValueError("pairwise AUC requires >=2 classes")

    freq = np.zeros(n_features, dtype=np.float64)
    inv_rank = np.zeros(n_features, dtype=np.float64)
    max_sep = np.zeros(n_features, dtype=np.float64)
    sum_sep = np.zeros(n_features, dtype=np.float64)
    n_pairs = 0
    weight_total = 0.0

    t0 = time.perf_counter()
    for i in range(len(classes)):
        cls_i = classes[i]
        for j in range(i + 1, len(classes)):
            cls_j = classes[j]
            mask = (y == cls_i) | (y == cls_j)
            n_pair = int(mask.sum())
            if n_pair < 3:
                continue
            y_pair = (y[mask] == cls_i).astype(np.int8)
            if y_pair.sum() == 0 or y_pair.sum() == len(y_pair):
                continue
            X_pair = X_imp[mask]
            sep = compute_pairwise_auc_separation(X_pair, y_pair)

            weight = 1.0
            if pair_weights is not None:
                weight = float(
                    pair_weights.get((cls_i, cls_j), pair_weights.get((cls_j, cls_i), 1.0))
                )

            top_idx = np.argpartition(sep, -top_per_pair)[-top_per_pair:]
            order = top_idx[np.argsort(sep[top_idx])[::-1]]
            ranks = np.arange(1, len(order) + 1, dtype=np.float64)

            freq[order] += weight
            inv_rank[order] += weight * (1.0 / ranks)
            sum_sep[order] += weight * sep[order]
            max_sep[order] = np.maximum(max_sep[order], weight * sep[order])
            n_pairs += 1
            weight_total += weight

    t_pair = time.perf_counter() - t0

    if mode == "freq":
        scores = freq + (1e-3 * inv_rank) + (1e-6 * max_sep)
    elif mode == "rank":
        scores = inv_rank + (1e-4 * freq) + (1e-6 * sum_sep)
    else:
        raise ValueError(f"Unsupported pairwise AUC mode: {mode}")

    meta = {
        "pairwise_mode": mode,
        "pairwise_score_type": "auc",
        "pairwise_top_per_pair": float(top_per_pair),
        "pairwise_n_pairs": float(n_pairs),
        "pairwise_weight_sum": float(weight_total),
        "pairwise_weight_mean": float(weight_total / max(1, n_pairs)),
        "pairwise_n_valid_classes": float(len(classes)),
        "t_pairwise_scoring": t_pair,
    }
    return scores, meta


def compute_hard_pair_weights(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    classes: Sequence[str],
    cv_folds: int,
    baseline_topk: int,
    gamma: float,
    seed: int,
) -> Tuple[Dict[Tuple[str, str], float], Dict[str, float]]:
    if cv_folds < 2:
        return {}, {
            "hardpair_cv_folds": float(cv_folds),
            "hardpair_n_pairs": 0.0,
            "hardpair_mean_weight": 1.0,
        }

    labels = list(classes)
    topk = max(2, min(int(baseline_topk), X_train.shape[1]))
    X_small = X_train.iloc[:, :topk]

    inner = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    cm_total = np.zeros((len(labels), len(labels)), dtype=np.float64)

    for tr_idx, va_idx in inner.split(X_small, y_train):
        X_tr, X_va = X_small.iloc[tr_idx], X_small.iloc[va_idx]
        y_tr, y_va = y_train[tr_idx], y_train[va_idx]
        model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=500,
                        solver="lbfgs",
                        penalty="l2",
                        C=1.0,
                        multi_class="multinomial",
                        n_jobs=1,
                    ),
                ),
            ]
        )
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_va)
        cm_total += confusion_matrix(y_va, y_pred, labels=labels)

    hardness: Dict[Tuple[str, str], float] = {}
    hard_vals: List[float] = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            support = float(cm_total[i, :].sum() + cm_total[j, :].sum())
            mis = float(cm_total[i, j] + cm_total[j, i])
            h = mis / support if support > 0 else 0.0
            hardness[(labels[i], labels[j])] = h
            hard_vals.append(h)

    positive = [x for x in hard_vals if x > 0]
    scale = float(np.mean(positive)) if positive else 1.0
    pair_weights: Dict[Tuple[str, str], float] = {}
    for pair, h in hardness.items():
        pair_weights[pair] = 1.0 + float(gamma) * (h / (scale + 1e-12))

    meta = {
        "hardpair_cv_folds": float(cv_folds),
        "hardpair_baseline_topk": float(topk),
        "hardpair_gamma": float(gamma),
        "hardpair_n_pairs": float(len(hardness)),
        "hardpair_mean_hardness": float(np.mean(hard_vals) if hard_vals else 0.0),
        "hardpair_mean_weight": float(
            np.mean(list(pair_weights.values())) if pair_weights else 1.0
        ),
    }
    return pair_weights, meta


def sample_stratified_indices(y: np.ndarray, frac: float, rng: np.random.Generator) -> np.ndarray:
    y = np.asarray(y)
    out: List[int] = []
    for cls in np.unique(y):
        idx = np.where(y == cls)[0]
        n = len(idx)
        if n == 0:
            continue
        n_take = int(round(float(frac) * n))
        if n >= 2:
            n_take = max(2, n_take)
        else:
            n_take = 1
        n_take = min(n, max(1, n_take))
        pick = rng.choice(idx, size=n_take, replace=False)
        out.extend(pick.tolist())
    out = sorted(set(out))
    return np.array(out, dtype=int)


def score_pairwise_hard_weighted_stable(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    k: int,
    top_per_pair: int,
    pair_weights: Dict[Tuple[str, str], float],
    n_boot: int,
    subsample_frac: float,
    alpha: float,
    seed: int,
    score_kind: str = "anova",
) -> Tuple[np.ndarray, Dict[str, float]]:
    if score_kind == "anova":
        base_scores, base_meta = score_pairwise_anova(
            X_train=X_train,
            y_train=y_train,
            top_per_pair=top_per_pair,
            mode="rank",
            pair_weights=pair_weights,
        )
    elif score_kind == "auc":
        base_scores, base_meta = score_pairwise_auc(
            X_train=X_train,
            y_train=y_train,
            top_per_pair=top_per_pair,
            mode="rank",
            pair_weights=pair_weights,
        )
    else:
        raise ValueError(f"Unsupported score_kind: {score_kind}")
    n_features = len(base_scores)
    k = max(1, min(int(k), n_features))
    n_boot = max(1, int(n_boot))
    alpha = float(np.clip(alpha, 0.0, 1.0))
    frac = float(np.clip(subsample_frac, 0.2, 1.0))
    rng = np.random.default_rng(seed)

    freq = np.zeros(n_features, dtype=np.float64)
    used_boot = 0
    for b in range(n_boot):
        idx = sample_stratified_indices(y_train, frac=frac, rng=rng)
        if len(idx) < 2 or len(np.unique(y_train[idx])) < 2:
            continue
        if score_kind == "anova":
            sub_scores, _ = score_pairwise_anova(
                X_train=X_train.iloc[idx],
                y_train=y_train[idx],
                top_per_pair=top_per_pair,
                mode="rank",
                pair_weights=pair_weights,
            )
        else:
            sub_scores, _ = score_pairwise_auc(
                X_train=X_train.iloc[idx],
                y_train=y_train[idx],
                top_per_pair=top_per_pair,
                mode="rank",
                pair_weights=pair_weights,
            )
        top = np.argpartition(sub_scores, -k)[-k:]
        freq[top] += 1.0
        used_boot += 1

    freq_norm = freq / float(max(1, used_boot))
    bmin, bmax = float(np.min(base_scores)), float(np.max(base_scores))
    if bmax > bmin:
        base_norm = (base_scores - bmin) / (bmax - bmin)
    else:
        base_norm = np.zeros_like(base_scores, dtype=np.float64)

    scores = ((1.0 - alpha) * base_norm) + (alpha * freq_norm)
    meta = {
        **base_meta,
        "pairwise_score_type": score_kind,
        "stable_n_boot_requested": float(n_boot),
        "stable_n_boot_used": float(used_boot),
        "stable_subsample_frac": float(frac),
        "stable_alpha": float(alpha),
    }
    return scores, meta


def tune_and_eval_classifier(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    X_test: pd.DataFrame,
    y_test: np.ndarray,
    all_classes: Sequence[str],
    c_grid: List[float],
    inner_folds: int,
    seed: int,
    compute_auroc: bool,
) -> Tuple[Dict[str, float], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    inner = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed)
    inner_rows = []
    best_c = None
    best_score = -np.inf

    t_inner_start = time.perf_counter()
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
                        LogisticRegression(
                            max_iter=500,
                            solver="saga",
                            penalty="l2",
                            C=c,
                            n_jobs=1,
                            multi_class="multinomial",
                        ),
                    ),
                ]
            )
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_va)
            fold_scores.append(f1_score(y_va, y_pred, average="macro", labels=list(all_classes)))
        mean_score = float(np.mean(fold_scores))
        inner_rows.append({"C": c, "mean_macro_f1": mean_score})
        if mean_score > best_score:
            best_score = mean_score
            best_c = c
    t_inner_total = time.perf_counter() - t_inner_start

    t_refit_start = time.perf_counter()
    best_model = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "clf",
                LogisticRegression(
                    max_iter=500,
                    solver="saga",
                    penalty="l2",
                    C=best_c,
                    n_jobs=1,
                    multi_class="multinomial",
                ),
            ),
        ]
    )
    best_model.fit(X_train, y_train)
    t_refit_total = time.perf_counter() - t_refit_start

    t_eval_start = time.perf_counter()
    y_pred = best_model.predict(X_test)
    macro_f1 = float(f1_score(y_test, y_pred, average="macro", labels=list(all_classes)))
    bal_acc = float(balanced_accuracy_score(y_test, y_pred))
    acc = float(accuracy_score(y_test, y_pred))
    auroc = float("nan")
    auc_df = pd.DataFrame(columns=["class", "auc", "n_pos", "n_neg"])
    if compute_auroc:
        proba = best_model.predict_proba(X_test)
        auc_rows = compute_auc_by_class(y_test, proba, best_model.named_steps["clf"].classes_)
        auc_df = pd.DataFrame(auc_rows)
        auc_vals = [row["auc"] for row in auc_rows if not np.isnan(row["auc"])]
        auroc = float(np.mean(auc_vals)) if auc_vals else float("nan")
    t_eval_total = time.perf_counter() - t_eval_start

    cm = confusion_matrix(y_test, y_pred, labels=list(all_classes))
    cm_df = pd.DataFrame(cm, index=list(all_classes), columns=list(all_classes))

    metrics = {
        "macro_f1": macro_f1,
        "balanced_accuracy": bal_acc,
        "accuracy": acc,
        "auroc": auroc,
        "best_c": float(best_c),
        "best_inner_macro_f1": float(best_score),
        "t_inner_cv_total": t_inner_total,
        "t_refit_full_train": t_refit_total,
        "t_eval_metrics": t_eval_total,
    }

    return metrics, pd.DataFrame(inner_rows), cm_df, auc_df


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
        stage="run_fs_method_benchmark",
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
            stage="run_fs_method_benchmark",
        )
        return code

    candidate_spec = load_candidate_spec(args.candidate_spec_json)
    selection_args = candidate_section(candidate_spec, "selection_args")
    prefilter_args = candidate_section(candidate_spec, "prefilter_args")
    probe_matrix = Path(args.probe_matrix)
    manifest_path = Path(args.manifest)
    folds_path = Path(args.folds)
    out_dir = Path(args.out_dir)
    for path in [probe_matrix, manifest_path, folds_path]:
        if not path.exists():
            print(f"ERROR: missing input {path}", file=sys.stderr)
            return _return(1, error=f"missing_input:{path}")
    pf = pq.ParquetFile(probe_matrix)
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
    try:
        validate_locked_development(manifest.to_dict("records"), manifest_path)
        validate_folds(manifest.to_dict("records"), folds.to_dict("records"))
    except ValueError as exc:
        print(f"ERROR: split integrity: {exc}", file=sys.stderr)
        return _return(1, error=f"split_integrity:{exc}")
    manifest = manifest.merge(folds, on="sample_id", how="inner", validate="one_to_many")
    test_df = manifest[
        (manifest["repeat"] == args.repeat) & (manifest["outer_fold"] == args.fold)
    ].copy()
    if test_df.empty:
        print("ERROR: no samples for requested fold/repeat", file=sys.stderr)
        return _return(1, error="empty_test_split")
    train_df = manifest[
        (manifest["repeat"] == args.repeat) & (manifest["outer_fold"] != args.fold)
    ].copy()
    train_ids = train_df["sample_id"].astype(str).tolist()
    test_ids = test_df["sample_id"].astype(str).tolist()
    all_classes = sorted(manifest["project_id"].astype(str).unique())
    fold_dir = out_dir / f"repeat{args.repeat}" / f"fold{args.fold}"
    qc_dir = fold_dir / "qc"
    fold_dir.mkdir(parents=True, exist_ok=True)
    qc_dir.mkdir(parents=True, exist_ok=True)
    if candidate_spec:
        (qc_dir / "candidate_spec.json").write_text(
            json.dumps(candidate_spec, indent=2) + "\n", encoding="utf-8"
        )
    emit_heartbeat(
        event="prepared_output_dirs",
        status="running",
        force=True,
        repeat=args.repeat,
        fold=args.fold,
    )
    prefilter_method = (
        override_text(prefilter_args, "prefilter_method", args.prefilter_method).strip() or "mi"
    )
    prefilter_k = override_int(prefilter_args, "prefilter_k", args.prefilter_k)
    chunk_size = override_int(prefilter_args, "chunk_size", args.chunk_size)
    pairwise_top_per_pair = override_int(
        selection_args, "pairwise_top_per_pair", args.pairwise_top_per_pair
    )
    hardpair_cv_folds = override_int(selection_args, "hardpair_cv_folds", args.hardpair_cv_folds)
    hardpair_baseline_topk = override_int(
        selection_args, "hardpair_baseline_topk", args.hardpair_baseline_topk
    )
    hardpair_gamma = override_float(selection_args, "hardpair_gamma", args.hardpair_gamma)
    stable_n_boot = override_int(selection_args, "stable_n_boot", args.stable_n_boot)
    stable_subsample_frac = override_float(
        selection_args, "stable_subsample_frac", args.stable_subsample_frac
    )
    stable_alpha = override_float(selection_args, "stable_alpha", args.stable_alpha)
    timing_global: Dict[str, float] = {}
    emit_heartbeat(event="compute_missingness_train_all", status="running", force=True)
    t0 = time.perf_counter()
    (sample_missing, probe_ids, _) = compute_missingness(pf, train_ids)
    timing_global["t_missingness_train_all"] = time.perf_counter() - t0
    total_probes = len(probe_ids)
    if total_probes == 0:
        print("ERROR: no probes found in matrix", file=sys.stderr)
        return _return(1, error="empty_probe_matrix")
    sample_missing_frac = sample_missing / float(total_probes)
    sample_keep = sample_missing_frac <= args.sample_missing_thresh
    train_ids_filtered = [sid for (sid, keep) in zip(train_ids, sample_keep) if keep]
    if len(train_ids_filtered) < 2:
        print("ERROR: too few training samples after sample missingness filter", file=sys.stderr)
        return _return(1, error="too_few_train_samples")
    pd.DataFrame(
        {"sample_id": train_ids, "missing_frac": sample_missing_frac, "keep": sample_keep}
    ).to_csv(qc_dir / "qc_samples_dropped.tsv", sep="\t", index=False)
    emit_heartbeat(event="compute_missingness_train_filtered", status="running", force=True)
    t1 = time.perf_counter()
    (_, probe_ids2, probe_missing2) = compute_missingness(pf, train_ids_filtered)
    timing_global["t_missingness_train_filtered"] = time.perf_counter() - t1
    probe_missing_frac = np.array(probe_missing2, dtype=np.float32) / float(len(train_ids_filtered))
    probe_keep_mask = probe_missing_frac <= args.probe_missing_thresh
    probe_df = pd.DataFrame(
        {"probe_id": probe_ids2, "missing_frac": probe_missing_frac, "keep": probe_keep_mask}
    )
    chen = load_blacklist_chen(Path(args.blacklist_chen))
    snp = load_blacklist_simple(Path(args.blacklist_snp))
    sex = load_sex_probes(Path(args.probe_metadata)) if args.exclude_sex_chr else set()
    probe_keep = set(probe_df.loc[probe_df["keep"], "probe_id"].astype(str))
    probe_keep -= chen
    probe_keep -= snp
    probe_keep -= sex
    probe_df["drop_reason"] = ""
    probe_df.loc[~probe_df["keep"], "drop_reason"] = "missingness"
    probe_df.loc[probe_df["probe_id"].isin(chen), "drop_reason"] = "blacklist_chen"
    probe_df.loc[probe_df["probe_id"].isin(snp), "drop_reason"] = "blacklist_snp"
    if sex:
        probe_df.loc[probe_df["probe_id"].isin(sex), "drop_reason"] = "sex_chr"
    probe_df["keep"] = probe_df["probe_id"].isin(probe_keep)
    probe_df.loc[~probe_df["keep"], ["probe_id", "missing_frac", "drop_reason"]].to_csv(
        qc_dir / "qc_probes_dropped.tsv", sep="\t", index=False
    )
    pd.DataFrame({"probe_id": sorted(probe_keep)}).to_csv(
        qc_dir / "probe_keep_list.tsv", sep="\t", index=False
    )
    train_df_filtered = train_df[train_df["sample_id"].isin(train_ids_filtered)].copy()
    train_df_filtered.to_csv(qc_dir / "train_manifest_filtered.tsv", sep="\t", index=False)
    y_train = (
        train_df_filtered.set_index("sample_id")
        .loc[train_ids_filtered, "project_id"]
        .astype(str)
        .to_numpy()
    )
    y_test = test_df.set_index("sample_id").loc[test_ids, "project_id"].astype(str).to_numpy()
    emit_heartbeat(event="anovascore_prefilter", status="running", force=True)
    t2 = time.perf_counter()
    (anova_scores, anova_t) = score_anova_full(
        pf=pf,
        train_ids=train_ids_filtered,
        y_train=y_train,
        probe_set=probe_keep,
        chunk_size=chunk_size,
    )
    timing_global.update(anova_t)
    timing_global["t_anova_total"] = time.perf_counter() - t2
    if prefilter_method == "mi":
        t_pref = time.perf_counter()
        (prefilter_scores, prefilter_meta) = score_mi_full(
            pf=pf,
            train_ids=train_ids_filtered,
            y_train=y_train,
            probe_set=probe_keep,
            chunk_size=chunk_size,
            seed=args.seed,
        )
        timing_global["t_mi_prefilter_total"] = time.perf_counter() - t_pref
    else:
        t_pref = time.perf_counter()
        try:
            (prefilter_scores, prefilter_meta) = plugin_prefilter(
                prefilter_method,
                pf=pf,
                train_ids=train_ids_filtered,
                y_train=y_train,
                probe_set=probe_keep,
                chunk_size=chunk_size,
                seed=args.seed,
                prefilter_k=prefilter_k,
                prefilter_args=prefilter_args,
                anova_scores=anova_scores,
            )
        except PluginNotFound:
            print(
                f"ERROR: unsupported prefilter_method requested: {prefilter_method}",
                file=sys.stderr,
            )
            return _return(1, error=f"unsupported_prefilter:{prefilter_method}")
        timing_global[f"t_plugin_prefilter_{slugify(prefilter_method)}_total"] = (
            time.perf_counter() - t_pref
        )
    if len(prefilter_scores) < max(args.k, prefilter_k):
        print(
            "ERROR: not enough probes after filtering for requested k/prefilter-k", file=sys.stderr
        )
        return _return(1, error="insufficient_prefilter_scores")
    prefilter_probes = prefilter_scores.head(prefilter_k)["probe_id"].astype(str).tolist()
    pd.DataFrame({"probe_id": prefilter_probes}).to_csv(
        qc_dir / f"{slugify(prefilter_method)}_prefilter_topk.tsv", sep="\t", index=False
    )
    (qc_dir / "prefilter_metadata.json").write_text(
        json.dumps(
            {
                "prefilter_method": prefilter_method,
                "prefilter_k": prefilter_k,
                "prefilter_args": prefilter_args,
                **prefilter_meta,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    t3 = time.perf_counter()
    emit_heartbeat(event="read_prefiltered_matrix", status="running", force=True)
    X_train_pref = read_selected(pf, train_ids_filtered, prefilter_probes)
    X_test_pref = read_selected(pf, test_ids, prefilter_probes)
    timing_global["t_read_prefilter_train_test"] = time.perf_counter() - t3
    all_methods = ["mi", "rare_class_rank_shift", "pairwise_hard_weighted_auc_stable"]
    methods = [m.strip() for m in re.split("[,\\s:]+", str(args.methods)) if m.strip()]
    if not methods:
        print("ERROR: no methods requested", file=sys.stderr)
        return _return(1, error="empty_methods")
    c_grid = [float(x.strip()) for x in re.split("[,\\s:]+", str(args.c_grid)) if x.strip()]
    hardpair_methods = {"pairwise_hard_weighted_auc_stable"}
    pair_weights_cache: Optional[Dict[Tuple[str, str], float]] = None
    pair_weights_meta: Dict[str, float] = {}
    for method in methods:
        emit_heartbeat(event="method_start", status="running", force=True, method=method)
        method_dir = fold_dir / method
        method_dir.mkdir(parents=True, exist_ok=True)
        method_timing: Dict[str, float] = {}
        selected_probes: List[str] = []
        score_df = pd.DataFrame(columns=["probe_id", "score"])
        extra_json: Dict[str, float] = {}
        t_sel = time.perf_counter()
        runtime_method = canonical_selection_method(method)
        if runtime_method == "mi":
            imp = SimpleImputer(strategy="median")
            X_imp = imp.fit_transform(X_train_pref)
            (y_int, _) = pd.factorize(y_train)
            scores = mutual_info_classif(X_imp, y_int, random_state=args.seed)
            scores = np.nan_to_num(scores, nan=-np.inf, neginf=-np.inf, posinf=-np.inf)
            score_df = pd.DataFrame({"probe_id": prefilter_probes, "score": scores}).sort_values(
                "score", ascending=False
            )
            selected_probes = score_df.head(args.k)["probe_id"].astype(str).tolist()
        else:
            if runtime_method in hardpair_methods:
                if pair_weights_cache is None:
                    (pair_weights_cache, pair_weights_meta) = compute_hard_pair_weights(
                        X_train=X_train_pref,
                        y_train=y_train,
                        classes=all_classes,
                        cv_folds=hardpair_cv_folds,
                        baseline_topk=hardpair_baseline_topk,
                        gamma=hardpair_gamma,
                        seed=args.seed,
                    )
                extra_json.update(pair_weights_meta)
                uses_auc = "_auc" in method
                score_kind = "auc" if uses_auc else "anova"
                if runtime_method in {
                    "pairwise_hard_weighted_stable",
                    "pairwise_hard_weighted_auc_stable",
                }:
                    (scores, stable_meta) = score_pairwise_hard_weighted_stable(
                        X_train=X_train_pref,
                        y_train=y_train,
                        k=args.k,
                        top_per_pair=pairwise_top_per_pair,
                        pair_weights=pair_weights_cache,
                        n_boot=stable_n_boot,
                        subsample_frac=stable_subsample_frac,
                        alpha=stable_alpha,
                        seed=args.seed,
                        score_kind=score_kind,
                    )
                    score_df = pd.DataFrame(
                        {"probe_id": prefilter_probes, "score": scores}
                    ).sort_values("score", ascending=False)
                    selected_probes = score_df.head(args.k)["probe_id"].astype(str).tolist()
                    extra_json.update(stable_meta)
                else:
                    raise ValueError(f"Unsupported hardpair method: {method}")
            else:
                try:
                    (selected_probes, score_df, extra_json) = plugin_selector(
                        runtime_method,
                        X_train_pref=X_train_pref,
                        y_train=y_train,
                        all_classes=all_classes,
                        k=args.k,
                        prefilter_probes=prefilter_probes,
                        seed=args.seed,
                        selection_args=selection_args,
                        prefilter_args=prefilter_args,
                    )
                except PluginNotFound:
                    raise ValueError(f"Unsupported method: {method}") from None
        method_timing["t_feature_selection"] = time.perf_counter() - t_sel
        if len(selected_probes) < args.k:
            print(
                f"ERROR: method={method} selected only {len(selected_probes)} probes (<{args.k})",
                file=sys.stderr,
            )
            return _return(1, error=f"method_selected_too_few:{method}")
        emit_heartbeat(event="method_done", status="running", force=True, method=method)
        selected_probes = selected_probes[: args.k]
        selected_order_df = canonical_selected_order_df(selected_probes, score_df)
        score_df.to_csv(method_dir / "scores.tsv", sep="\t", index=False)
        selected_order_df.to_csv(
            method_dir / f"selected_order_K{args.k}.tsv", sep="\t", index=False
        )
        pd.DataFrame({"probe_id": selected_probes}).to_csv(
            method_dir / f"selected_probes_K{args.k}.tsv", sep="\t", index=False
        )
        if extra_json:
            with (method_dir / "selection_meta.json").open("w", encoding="utf-8") as handle:
                json.dump(extra_json, handle, indent=2)
        t_clf_data = time.perf_counter()
        X_train_sel = X_train_pref[selected_probes]
        X_test_sel = X_test_pref[selected_probes]
        method_timing["t_prepare_classifier_data"] = time.perf_counter() - t_clf_data
        (metrics, inner_df, cm_df, auc_df) = tune_and_eval_classifier(
            X_train=X_train_sel,
            y_train=y_train,
            X_test=X_test_sel,
            y_test=y_test,
            all_classes=all_classes,
            c_grid=c_grid,
            inner_folds=args.inner_folds,
            seed=args.seed,
            compute_auroc=args.compute_auroc,
        )
        method_timing["t_inner_cv_total"] = metrics.pop("t_inner_cv_total")
        method_timing["t_refit_full_train"] = metrics.pop("t_refit_full_train")
        method_timing["t_eval_metrics"] = metrics.pop("t_eval_metrics")
        metrics_row = {
            "method": method,
            "repeat": args.repeat,
            "fold": args.fold,
            "k": args.k,
            "n_train": len(train_ids_filtered),
            "n_test": len(test_ids),
            "n_classes": len(all_classes),
            **metrics,
        }
        pd.DataFrame([metrics_row]).to_csv(method_dir / "metrics.tsv", sep="\t", index=False)
        inner_df.to_csv(method_dir / "innercv_results.tsv", sep="\t", index=False)
        cm_df.to_csv(method_dir / f"confusion_matrix_K{args.k}.tsv", sep="\t")
        if args.compute_auroc and (not auc_df.empty):
            auc_df.insert(0, "k", args.k)
            auc_df.insert(0, "fold", args.fold)
            auc_df.insert(0, "repeat", args.repeat)
            auc_df.insert(0, "method", method)
            auc_df.to_csv(method_dir / f"auc_by_class_K{args.k}.tsv", sep="\t", index=False)
        timing_row = {
            "method": method,
            "repeat": args.repeat,
            "fold": args.fold,
            "k": args.k,
            **timing_global,
            **method_timing,
        }
        pd.DataFrame([timing_row]).to_csv(method_dir / "timing.tsv", sep="\t", index=False)
    return _return(0)


if __name__ == "__main__":
    raise SystemExit(main())
