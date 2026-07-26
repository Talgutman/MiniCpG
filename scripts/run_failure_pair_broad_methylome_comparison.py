#!/usr/bin/env python3
"""Compare 30-CpG failure-pair distances with a broader TCGA CpG background."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--active-manifest", default="config/active_locked_run.json")
    p.add_argument(
        "--pair-table-30",
        default=(
            "outputs/paper_protocolb_v1/manuscript_reproduction/"
            "failure_pair_confusion_methylation_similarity_30cpg.tsv"
        ),
    )
    p.add_argument(
        "--tumor-broad-matrix",
        default=(
            "outputs/paper_protocolb_v1/k_030/biological_interpretation_current/"
            "composition_calibration_current/tcga_dev_prefilter_union_probe_matrix.tsv"
        ),
    )
    p.add_argument(
        "--normal-broad-matrix",
        default=(
            "outputs/paper_protocolb_v1/k_030/biological_interpretation_current/"
            "lineage_calibration_current/tcga_normals_all24_prefilter_union_gdc_current/"
            "tcga_gdc_panel_matrix.tsv"
        ),
    )
    p.add_argument(
        "--out-dir",
        default="outputs/paper_protocolb_v1/manuscript_reproduction",
    )
    p.add_argument("--n-permutations", type=int, default=20000)
    p.add_argument("--seed", type=int, default=20260712)
    return p.parse_args()


def validate_active_run(manifest: str) -> dict[str, object]:
    cmd = [sys.executable, "scripts/validate_active_locked_run.py", "--manifest", manifest]
    proc = subprocess.run(cmd, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise SystemExit(f"Active locked-run validation failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    payload = json.loads(proc.stdout)
    if payload.get("status") != "ok":
        raise SystemExit(f"Active locked-run validation status is not ok: {payload.get('status')}")
    return payload


def read_header(path: Path) -> list[str]:
    with path.open("r", encoding="utf-8") as f:
        return f.readline().rstrip("\n").split("\t")


def feature_columns(path: Path, metadata_cols: set[str]) -> list[str]:
    return [c for c in read_header(path) if c not in metadata_cols and c.startswith("cg")]


def read_centroids(path: Path, *, label_col: str, features: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    usecols = [label_col, *features]
    df = pd.read_csv(path, sep="\t", usecols=usecols, na_values=["NA", "NaN", ""], low_memory=False)
    counts = df.groupby(label_col).size().rename("n_samples")
    centroids = df.groupby(label_col, sort=True)[features].median()
    centroids = centroids.apply(pd.to_numeric, errors="coerce")
    return centroids, counts


def pair_distances(values_a: pd.Series, values_b: pd.Series) -> tuple[float, float, int]:
    a = values_a.astype(float)
    b = values_b.astype(float)
    keep = ~(a.isna() | b.isna())
    if int(keep.sum()) == 0:
        return math.nan, math.nan, 0
    diff = (a[keep] - b[keep]).to_numpy(dtype=float)
    return float(np.mean(np.abs(diff))), float(np.sqrt(np.sum(diff * diff))), int(keep.sum())


def add_broad_distances(pair_df: pd.DataFrame, centroids: pd.DataFrame, counts: pd.Series, prefix: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    features = list(centroids.columns)
    for row in pair_df.itertuples(index=False):
        a = getattr(row, "class_a")
        b = getattr(row, "class_b")
        has_pair = a in centroids.index and b in centroids.index
        mad = euclidean = math.nan
        n_probes = 0
        if has_pair:
            mad, euclidean, n_probes = pair_distances(centroids.loc[a, features], centroids.loc[b, features])
        rows.append(
            {
                "class_a": a,
                "class_b": b,
                f"{prefix}_broad_mad": mad,
                f"{prefix}_broad_similarity_mad": -mad if not math.isnan(mad) else math.nan,
                f"{prefix}_broad_euclidean": euclidean,
                f"{prefix}_broad_similarity_euclidean": -euclidean if not math.isnan(euclidean) else math.nan,
                f"{prefix}_broad_n_probes": n_probes,
                f"{prefix}_broad_n_a": int(counts.get(a, 0)),
                f"{prefix}_broad_n_b": int(counts.get(b, 0)),
                f"has_{prefix}_broad_pair": bool(has_pair),
            }
        )
    return pair_df.merge(pd.DataFrame(rows), on=["class_a", "class_b"], how="left")


def spearman_perm(x: pd.Series, y: pd.Series, *, rng: np.random.Generator, n_perm: int) -> tuple[float, float, int]:
    df = pd.DataFrame({"x": x, "y": y}).dropna()
    n = len(df)
    if n < 3 or df["x"].nunique() < 2 or df["y"].nunique() < 2:
        return math.nan, math.nan, n
    xr = df["x"].rank(method="average").to_numpy(dtype=float)
    yr = df["y"].rank(method="average").to_numpy(dtype=float)
    rho = float(np.corrcoef(xr, yr)[0, 1])
    if n_perm <= 0:
        return rho, math.nan, n
    extreme = 0
    abs_rho = abs(rho)
    for _ in range(n_perm):
        perm_rho = float(np.corrcoef(xr, rng.permutation(yr))[0, 1])
        if abs(perm_rho) >= abs_rho:
            extreme += 1
    return rho, (extreme + 1) / (n_perm + 1), n


def compute_stats(df: pd.DataFrame, n_perm: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    comparisons = [
        ("confusion_vs_tumor_30_mad_similarity", "tumor_similarity_mad_30", "pair_confusion_rate"),
        ("confusion_vs_tumor_broad_mad_similarity", "tumor_broad_similarity_mad", "pair_confusion_rate"),
        ("confusion_vs_normal_30_mad_similarity", "normal_similarity_mad_30", "pair_confusion_rate"),
        ("confusion_vs_normal_broad_mad_similarity", "normal_broad_similarity_mad", "pair_confusion_rate"),
        ("tumor_30_vs_tumor_broad_mad_distance", "tumor_mad_30", "tumor_broad_mad"),
        ("tumor_30_vs_tumor_broad_euclidean_distance", "tumor_euclidean_30", "tumor_broad_euclidean"),
        ("normal_30_vs_normal_broad_mad_distance", "normal_mad_30", "normal_broad_mad"),
        ("normal_30_vs_normal_broad_euclidean_distance", "normal_euclidean_30", "normal_broad_euclidean"),
        ("tumor_30_vs_normal_30_mad_distance", "tumor_mad_30", "normal_mad_30"),
        ("tumor_broad_vs_normal_broad_mad_distance", "tumor_broad_mad", "normal_broad_mad"),
    ]
    rows: list[dict[str, object]] = []
    for name, x_col, y_col in comparisons:
        rho, p, n = spearman_perm(df[x_col], df[y_col], rng=rng, n_perm=n_perm)
        rows.append(
            {
                "comparison": name,
                "x_col": x_col,
                "y_col": y_col,
                "n_pairs": n,
                "spearman_rho": rho,
                "permutation_p": p,
                "n_permutations": n_perm,
            }
        )
    return pd.DataFrame(rows)


def add_percentiles(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in ["pair_confusion_rate", "tumor_mad_30", "normal_mad_30", "tumor_broad_mad", "normal_broad_mad"]:
        if col not in out:
            continue
        pct_col = col + "_percentile"
        out[pct_col] = out[col].rank(method="average", pct=True)
    for col in ["tumor_mad_30", "normal_mad_30", "tumor_broad_mad", "normal_broad_mad"]:
        pct = col + "_percentile"
        if pct in out:
            out[col.replace("_mad", "_similarity_rank")] = 1.0 - out[pct]
    return out


def flag(value: float, threshold: float) -> bool:
    return bool(pd.notna(value) and value >= threshold)


def pair_context(a: str, b: str) -> str:
    pair = {a.replace("TCGA-", ""), b.replace("TCGA-", "")}
    if pair <= {"KICH", "KIRC", "KIRP"}:
        return "kidney-related"
    if pair <= {"LUAD", "LUSC"}:
        return "lung-related"
    if pair <= {"UCEC", "UCS"}:
        return "uterine-related"
    if pair <= {"SKCM", "UVM"}:
        return "melanocytic-related"
    if pair <= {"CESC", "HNSC", "LUSC"}:
        return "squamous-related"
    if pair <= {"COAD", "ESCA", "PAAD", "STAD"}:
        return "gastrointestinal/upper-GI-related"
    return "no_simple_context_assigned"


def pair_label(row: pd.Series) -> str:
    high_conf = flag(row.get("pair_confusion_rate_percentile"), 0.90)
    tumor30_close = flag(row.get("tumor_mad_30_similarity_rank"), 0.80)
    tumor_broad_close = flag(row.get("tumor_broad_similarity_rank"), 0.80)
    normal30_close = flag(row.get("normal_mad_30_similarity_rank"), 0.80)
    normal_broad_close = flag(row.get("normal_broad_similarity_rank"), 0.80)

    if high_conf and (tumor30_close or tumor_broad_close) and (normal30_close or normal_broad_close):
        return "high_confusion_tumor_and_normal_similarity"
    if high_conf and (tumor30_close or tumor_broad_close):
        return "high_confusion_tumor_similarity_only"
    if high_conf and (normal30_close or normal_broad_close):
        return "high_confusion_normal_similarity_only"
    if high_conf:
        return "high_confusion_not_explained_by_pairwise_similarity"
    if (tumor30_close or tumor_broad_close) and row.get("pair_confusion_rate", 0) == 0:
        return "methylation_close_without_observed_confusion"
    return "background_pair"


def build_pair_review(df: pd.DataFrame) -> pd.DataFrame:
    d = add_percentiles(df)
    d["context_label"] = [pair_context(a, b) for a, b in zip(d["class_a"], d["class_b"], strict=False)]
    d["interpretive_pattern"] = d.apply(pair_label, axis=1)
    d["normal_minus_tumor_mad_30"] = d["normal_mad_30"] - d["tumor_mad_30"]
    d["normal_minus_tumor_broad_mad"] = d["normal_broad_mad"] - d["tumor_broad_mad"]

    interesting = (
        (d["pair_confusion_rate_percentile"] >= 0.90)
        | (d["tumor_mad_30_similarity_rank"] >= 0.95)
        | (d["tumor_broad_similarity_rank"] >= 0.95)
        | (d["normal_mad_30_similarity_rank"] >= 0.90)
        | (d["normal_broad_similarity_rank"] >= 0.90)
        | (d["normal_minus_tumor_mad_30"].abs() >= 0.07)
        | (d["normal_minus_tumor_broad_mad"].abs() >= 0.03)
        | (d["context_label"] != "no_simple_context_assigned")
    )
    cols = [
        "class_a",
        "class_b",
        "context_label",
        "interpretive_pattern",
        "pair_confusion_count",
        "pair_confusion_rate",
        "n_a",
        "n_b",
        "tumor_mad_30",
        "tumor_broad_mad",
        "normal_mad_30",
        "normal_broad_mad",
        "normal_minus_tumor_mad_30",
        "normal_minus_tumor_broad_mad",
        "tumor_mad_30_similarity_rank",
        "tumor_broad_similarity_rank",
        "normal_mad_30_similarity_rank",
        "normal_broad_similarity_rank",
    ]
    return (
        d.loc[interesting, cols]
        .sort_values(["pair_confusion_rate", "pair_confusion_count"], ascending=False)
        .reset_index(drop=True)
    )


def top_pairs_by_category(review: pd.DataFrame, top_n: int = 10) -> dict[str, list[dict[str, object]]]:
    payload: dict[str, list[dict[str, object]]] = {}
    for pattern, g in review.groupby("interpretive_pattern", sort=True):
        payload[pattern] = g.head(top_n).to_dict(orient="records")
    return payload


def clean_json(value: object) -> object:
    if isinstance(value, dict):
        return {k: clean_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [clean_json(v) for v in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return value


def main() -> None:
    args = parse_args()
    active = validate_active_run(args.active_manifest)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    pair_df = pd.read_csv(args.pair_table_30, sep="\t")
    tumor_path = Path(args.tumor_broad_matrix)
    normal_path = Path(args.normal_broad_matrix)

    tumor_meta = {"sample_id", "project_id"}
    normal_meta = {"project_id", "sample_id", "patient_id", "sample_type", "sample_type_code", "file_uuid", "file_name"}
    tumor_features = feature_columns(tumor_path, tumor_meta)
    normal_features = feature_columns(normal_path, normal_meta)
    broad_features = sorted(set(tumor_features) & set(normal_features))
    if len(broad_features) < 100:
        raise SystemExit(f"Too few shared broad CpGs between tumor and normal matrices: {len(broad_features)}")

    tumor_centroids, tumor_counts = read_centroids(tumor_path, label_col="project_id", features=broad_features)
    normal_centroids, normal_counts = read_centroids(normal_path, label_col="project_id", features=broad_features)

    augmented = add_broad_distances(pair_df, tumor_centroids, tumor_counts, "tumor")
    augmented = add_broad_distances(augmented, normal_centroids, normal_counts, "normal")
    augmented = add_percentiles(augmented)
    review = build_pair_review(augmented)
    stats = compute_stats(augmented, args.n_permutations, args.seed)

    augmented_tsv = out_dir / "failure_pair_confusion_methylation_similarity_30cpg_vs_broad.tsv"
    stats_tsv = out_dir / "failure_pair_30cpg_vs_broad_spearman_stats.tsv"
    review_tsv = out_dir / "failure_pair_tumor_normal_discordance_review.tsv"
    summary_json = out_dir / "failure_pair_broad_methylome_summary.json"

    augmented.to_csv(augmented_tsv, sep="\t", index=False)
    stats.to_csv(stats_tsv, sep="\t", index=False)
    review.to_csv(review_tsv, sep="\t", index=False)

    summary = {
        "active_locked_run_status": active.get("status"),
        "locked_run_name": active.get("locked_run_name"),
        "broad_background_definition": (
            "Shared TCGA prefilter-union CpGs available in both local TCGA development tumor "
            "and TCGA normal-adjacent matrices; broader than the locked 30 CpGs but not the full HM450 array."
        ),
        "tumor_broad_matrix": str(tumor_path),
        "normal_broad_matrix": str(normal_path),
        "n_shared_broad_cpgs": len(broad_features),
        "n_tumor_centroid_classes": int(len(tumor_centroids)),
        "n_normal_centroid_classes": int(len(normal_centroids)),
        "n_class_pairs": int(len(augmented)),
        "n_pairs_with_tumor_broad": int(augmented["tumor_broad_mad"].notna().sum()),
        "n_pairs_with_normal_broad": int(augmented["normal_broad_mad"].notna().sum()),
        "spearman": stats.to_dict(orient="records"),
        "top_pairs_by_interpretive_pattern": top_pairs_by_category(review),
        "outputs": {
            "augmented_pair_table": str(augmented_tsv),
            "stats": str(stats_tsv),
            "pair_review": str(review_tsv),
            "summary_json": str(summary_json),
        },
    }
    summary_json.write_text(json.dumps(clean_json(summary), indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(clean_json(summary["outputs"] | {"n_shared_broad_cpgs": len(broad_features)}), indent=2))


if __name__ == "__main__":
    main()
