#!/usr/bin/env python3
"""Relate external pair confusions to external 30-CpG methylation similarity."""

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
from scipy.stats import spearmanr


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--active-manifest", default="config/active_locked_run.json")
    parser.add_argument(
        "--predictions",
        default=(
            "outputs/paper_protocolb_v1/k_030/external_main_beta_sanitized_current/eval/"
            "external_predictions_allow_missing1.tsv"
        ),
    )
    parser.add_argument(
        "--external-panel-matrix",
        default=(
            "outputs/paper_protocolb_v1/k_030/external_main_beta_sanitized_current/inputs/"
            "external_panel_matrix_main_missing_le5_beta_sanitized.tsv"
        ),
    )
    parser.add_argument(
        "--panel-tsv",
        default=(
            "outputs/paper_protocolb_v1/k_030/stage2_confirm/fs_locked/"
            "fs_rare_class_rank_shift_09__pf_default/stable_panel_rankrule/stable_panel_rankrule_top30.tsv"
        ),
    )
    parser.add_argument(
        "--tcga-review-table",
        default=(
            "outputs/paper_protocolb_v1/manuscript_reproduction/"
            "failure_pair_euclidean_primary_review.tsv"
        ),
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/paper_protocolb_v1/manuscript_reproduction",
    )
    parser.add_argument("--min-class-n", type=int, default=10)
    return parser.parse_args()


def validate_active_run(manifest: str) -> dict[str, object]:
    proc = subprocess.run(
        [sys.executable, "scripts/validate_active_locked_run.py", "--manifest", manifest],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if proc.returncode != 0:
        raise SystemExit(f"Active run validation failed:\n{proc.stdout}\n{proc.stderr}")
    payload = json.loads(proc.stdout)
    if payload.get("status") != "ok":
        raise SystemExit(f"Active run validation status is not ok: {payload.get('status')}")
    return payload


def pair_distance(a: pd.Series, b: pd.Series) -> tuple[float, float, int]:
    keep = ~(a.isna() | b.isna())
    if keep.sum() == 0:
        return math.nan, math.nan, 0
    diff = (a[keep].astype(float) - b[keep].astype(float)).to_numpy()
    mad = float(np.mean(np.abs(diff)))
    euclidean = float(np.sqrt(np.square(diff).sum()))
    return mad, euclidean, int(keep.sum())


def spearman_row(df: pd.DataFrame, name: str, x_col: str, y_col: str) -> dict[str, object]:
    valid = df[[x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if len(valid) < 3 or valid[x_col].nunique() < 2 or valid[y_col].nunique() < 2:
        return {
            "comparison": name,
            "x_col": x_col,
            "y_col": y_col,
            "n_pairs": int(len(valid)),
            "spearman_rho": np.nan,
            "spearman_p_scipy": np.nan,
        }
    rho, p_value = spearmanr(valid[x_col], valid[y_col])
    return {
        "comparison": name,
        "x_col": x_col,
        "y_col": y_col,
        "n_pairs": int(len(valid)),
        "spearman_rho": float(rho),
        "spearman_p_scipy": float(p_value),
    }


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    active = validate_active_run(args.active_manifest)

    predictions = pd.read_csv(args.predictions, sep="\t", low_memory=False)
    matrix = pd.read_csv(args.external_panel_matrix, sep="\t", low_memory=False)
    panel = pd.read_csv(args.panel_tsv, sep="\t")
    review = pd.read_csv(args.tcga_review_table, sep="\t")
    probes = panel["probe_id"].astype(str).tolist()

    missing = {"gse_id", "sample_id", "tcga_label"} - set(matrix.columns)
    if missing:
        raise SystemExit(f"External matrix missing columns: {sorted(missing)}")
    missing_probes = sorted(set(probes) - set(matrix.columns))
    if missing_probes:
        raise SystemExit(f"External matrix missing panel probes: {missing_probes[:5]}")

    required_pred = {"gse_id", "sample_id", "tcga_label", "y_pred"}
    if required_pred - set(predictions.columns):
        raise SystemExit(f"Predictions missing columns: {sorted(required_pred - set(predictions.columns))}")

    classes = sorted(predictions["tcga_label"].astype(str).unique())
    class_counts = predictions["tcga_label"].value_counts().to_dict()
    conf = pd.crosstab(predictions["tcga_label"], predictions["y_pred"])
    centroids = matrix.groupby("tcga_label", sort=True)[probes].median()

    rows = []
    for class_a, class_b in combinations(classes, 2):
        n_a = int(class_counts.get(class_a, 0))
        n_b = int(class_counts.get(class_b, 0))
        n_a_pred_b = int(conf.loc[class_a, class_b]) if class_a in conf.index and class_b in conf.columns else 0
        n_b_pred_a = int(conf.loc[class_b, class_a]) if class_b in conf.index and class_a in conf.columns else 0
        mad, euclidean, n_probes = pair_distance(centroids.loc[class_a, probes], centroids.loc[class_b, probes])
        rows.append(
            {
                "class_a": class_a,
                "class_b": class_b,
                "pair_label": f"{class_a} vs {class_b}",
                "n_a": n_a,
                "n_b": n_b,
                "external_min_class_n": min(n_a, n_b),
                "n_a_pred_b": n_a_pred_b,
                "n_b_pred_a": n_b_pred_a,
                "pair_confusion_count": n_a_pred_b + n_b_pred_a,
                "pair_confusion_rate": (n_a_pred_b + n_b_pred_a) / (n_a + n_b) if (n_a + n_b) else math.nan,
                "external_mad_30": mad,
                "external_similarity_mad_30": -mad if not math.isnan(mad) else math.nan,
                "external_euclidean_30": euclidean,
                "external_similarity_euclidean_30": -euclidean if not math.isnan(euclidean) else math.nan,
                "external_n_probes": n_probes,
                "passes_min_class_n": min(n_a, n_b) >= args.min_class_n,
            }
        )
    out = pd.DataFrame(rows)

    merge_cols = [
        "class_a",
        "class_b",
        "tumor_euclidean_30",
        "tumor_30_euclidean_close_percentile",
        "tumor_all_hm450_euclidean",
        "tumor_all_hm450_euclidean_close_percentile",
        "normal_euclidean_30",
        "normal_30_euclidean_close_percentile",
    ]
    out = out.merge(review[merge_cols], on=["class_a", "class_b"], how="left")
    out["external_vs_tcga_30_euclidean_delta"] = out["external_euclidean_30"] - out["tumor_euclidean_30"]
    out["external_30_euclidean_close_rank"] = out["external_euclidean_30"].rank(method="min", ascending=True)
    n_pairs = len(out)
    out["external_30_euclidean_close_percentile"] = 1.0 - (
        (out["external_30_euclidean_close_rank"] - 1.0) / max(n_pairs - 1, 1)
    )
    out["external_vs_tcga_30_close_percentile_delta"] = (
        out["external_30_euclidean_close_percentile"] - out["tumor_30_euclidean_close_percentile"]
    )

    stats_rows = [
        spearman_row(
            out,
            "confusion_vs_external_30_euclidean_similarity_all_pairs",
            "external_similarity_euclidean_30",
            "pair_confusion_rate",
        ),
        spearman_row(
            out[out["passes_min_class_n"]],
            f"confusion_vs_external_30_euclidean_similarity_min_class_n_ge_{args.min_class_n}",
            "external_similarity_euclidean_30",
            "pair_confusion_rate",
        ),
        spearman_row(
            out,
            "confusion_vs_tcga_30_euclidean_distance_same_pairs",
            "tumor_euclidean_30",
            "pair_confusion_rate",
        ),
        spearman_row(
            out,
            "external_30_vs_tcga_30_euclidean_distance",
            "external_euclidean_30",
            "tumor_euclidean_30",
        ),
    ]
    stats = pd.DataFrame(stats_rows)

    tgct = out[(out["class_a"] == "TCGA-TGCT") | (out["class_b"] == "TCGA-TGCT")].copy()
    tgct = tgct.sort_values(["pair_confusion_count", "pair_confusion_rate"], ascending=[False, False])

    out_path = out_dir / "failure_pair_external_30cpg_similarity.tsv"
    stats_path = out_dir / "failure_pair_external_30cpg_similarity_spearman_stats.tsv"
    tgct_path = out_dir / "failure_pair_external_30cpg_similarity_tgct_pairs.tsv"
    summary_path = out_dir / "failure_pair_external_30cpg_similarity_summary.json"
    out.to_csv(out_path, sep="\t", index=False)
    stats.to_csv(stats_path, sep="\t", index=False)
    tgct.to_csv(tgct_path, sep="\t", index=False)

    summary = {
        "active_locked_run_status": active.get("status"),
        "inputs": {
            "predictions": args.predictions,
            "external_panel_matrix": args.external_panel_matrix,
            "panel": args.panel_tsv,
            "tcga_review_table": args.tcga_review_table,
        },
        "n_pairs": int(len(out)),
        "min_class_n": int(args.min_class_n),
        "n_pairs_passing_min_class_n": int(out["passes_min_class_n"].sum()),
        "stats": stats.to_dict(orient="records"),
        "top_tgct_pairs": tgct.head(12).to_dict(orient="records"),
        "outputs": {
            "pair_table": str(out_path),
            "stats": str(stats_path),
            "tgct_pairs": str(tgct_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps({"status": "ok", "out_dir": str(out_dir), **summary}, indent=2))


if __name__ == "__main__":
    main()
