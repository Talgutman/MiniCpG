#!/usr/bin/env python3
"""Relate external confusion rates to 30-CpG tumor and normal methylation similarity."""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile
from itertools import combinations
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "methylation_classification_mpl_cache"),
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PAPER_BG = "#fbfaf7"
INK = "#24313f"
MUTED = "#65717d"
GRID = "#e3dccf"
BLUE = "#2f6f9f"
TEAL = "#4f8a78"
GOLD = "#c7903a"
PURPLE = "#7d6a9b"
ORANGE = "#c46a3a"
WHITE = "#fffdf9"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--active-manifest", default="config/active_locked_run.json")
    p.add_argument(
        "--predictions-tsv",
        default="outputs/paper_protocolb_v1/k_030/external_main_beta_sanitized_current/eval/external_predictions_allow_missing1.tsv",
    )
    p.add_argument(
        "--tcga-dev-panel-matrix",
        default="outputs/paper_protocolb_v1/k_030/biological_interpretation_current/tcga_tumor_panel_matrices_current/tcga_dev_panel_matrix.tsv",
    )
    p.add_argument(
        "--normal-medians-wide",
        default="outputs/paper_protocolb_v1/k_030/lineage_normals_tcga_only_current/normal_reference_tissue_medians_pooled_wide.tsv",
    )
    p.add_argument(
        "--normal-counts",
        default="outputs/paper_protocolb_v1/k_030/lineage_normals_tcga_only_current/normal_reference_counts_by_tissue.tsv",
    )
    p.add_argument(
        "--panel-tsv",
        default="outputs/paper_protocolb_v1/k_030/stage2_confirm/fs_locked/fs_rare_class_rank_shift_09__pf_default/stable_panel_rankrule/stable_panel_rankrule_top30.tsv",
    )
    p.add_argument(
        "--out-dir",
        default="outputs/paper_protocolb_v1/manuscript_reproduction",
    )
    p.add_argument("--n-permutations", type=int, default=20000)
    p.add_argument("--seed", type=int, default=20260712)
    return p.parse_args()


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.facecolor": PAPER_BG,
            "axes.facecolor": PAPER_BG,
            "savefig.facecolor": PAPER_BG,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def validate_active_run(manifest: str) -> dict[str, object]:
    cmd = [sys.executable, "scripts/validate_active_locked_run.py", "--manifest", manifest]
    proc = subprocess.run(cmd, check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if proc.returncode != 0:
        raise SystemExit(f"Active locked-run validation failed:\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    payload = json.loads(proc.stdout)
    if payload.get("status") != "ok":
        raise SystemExit(f"Active locked-run validation status is not ok: {payload.get('status')}")
    return payload


def read_panel_probes(path: Path) -> list[str]:
    df = pd.read_csv(path, sep="\t")
    if "probe_id" not in df.columns:
        raise SystemExit(f"Panel file missing probe_id column: {path}")
    probes = df["probe_id"].astype(str).tolist()
    if len(probes) != len(set(probes)):
        raise SystemExit(f"Panel file contains duplicated probe IDs: {path}")
    return probes


def tumor_centroids(path: Path, probes: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(path, sep="\t")
    missing = {"project_id", *probes} - set(df.columns)
    if missing:
        raise SystemExit(f"TCGA development panel matrix missing columns: {sorted(missing)[:10]}")
    counts = df.groupby("project_id").size().rename("tumor_n")
    centroids = df.groupby("project_id", sort=True)[probes].median()
    return centroids, counts


def normal_centroids(path: Path, counts_path: Path, probes: list[str]) -> tuple[pd.DataFrame, pd.Series]:
    wide = pd.read_csv(path, sep="\t")
    if "probe_id" not in wide.columns:
        raise SystemExit(f"Normal medians file missing probe_id column: {path}")
    wide = wide.copy()
    wide["probe_id_clean"] = wide["probe_id"].astype(str).str.split(" | ", regex=False).str[0]
    wide = wide.set_index("probe_id_clean")
    missing = [p for p in probes if p not in wide.index]
    if missing:
        raise SystemExit(f"Normal medians missing {len(missing)} panel probes, e.g. {missing[:5]}")
    centroids = wide.loc[probes].drop(columns=["probe_id"], errors="ignore").T
    centroids.columns = probes
    centroids = centroids.apply(pd.to_numeric, errors="coerce")

    counts = pd.read_csv(counts_path, sep="\t")
    if {"tissue_label", "n_samples"} - set(counts.columns):
        raise SystemExit(f"Normal counts file missing expected columns: {counts_path}")
    count_series = counts.set_index("tissue_label")["n_samples"]
    return centroids, count_series


def pair_distances(values_a: pd.Series, values_b: pd.Series) -> tuple[float, float, int]:
    a = values_a.astype(float)
    b = values_b.astype(float)
    keep = ~(a.isna() | b.isna())
    if keep.sum() == 0:
        return math.nan, math.nan, 0
    diff = (a[keep] - b[keep]).to_numpy(dtype=float)
    mad = float(np.mean(np.abs(diff)))
    euclidean = float(np.sqrt(np.sum(diff * diff)))
    return mad, euclidean, int(keep.sum())


def build_pair_table(
    predictions: pd.DataFrame,
    probes: list[str],
    tumor: pd.DataFrame,
    tumor_counts: pd.Series,
    normal: pd.DataFrame,
    normal_counts: pd.Series,
) -> pd.DataFrame:
    required = {"tcga_label", "y_pred"}
    if required - set(predictions.columns):
        raise SystemExit(f"Predictions file missing columns: {sorted(required - set(predictions.columns))}")

    classes = sorted(predictions["tcga_label"].astype(str).unique())
    class_counts = predictions["tcga_label"].value_counts().to_dict()
    conf = pd.crosstab(predictions["tcga_label"], predictions["y_pred"])

    rows: list[dict[str, object]] = []
    for a, b in combinations(classes, 2):
        n_a = int(class_counts.get(a, 0))
        n_b = int(class_counts.get(b, 0))
        n_a_pred_b = int(conf.loc[a, b]) if a in conf.index and b in conf.columns else 0
        n_b_pred_a = int(conf.loc[b, a]) if b in conf.index and a in conf.columns else 0
        pair_confusion_rate = (n_a_pred_b + n_b_pred_a) / (n_a + n_b) if (n_a + n_b) else math.nan

        tumor_mad = tumor_euclidean = math.nan
        tumor_n_probes = 0
        has_tumor_pair = a in tumor.index and b in tumor.index
        if has_tumor_pair:
            tumor_mad, tumor_euclidean, tumor_n_probes = pair_distances(tumor.loc[a, probes], tumor.loc[b, probes])

        normal_mad = normal_euclidean = math.nan
        normal_n_probes = 0
        has_normal_pair = a in normal.index and b in normal.index
        if has_normal_pair:
            normal_mad, normal_euclidean, normal_n_probes = pair_distances(normal.loc[a, probes], normal.loc[b, probes])

        rows.append(
            {
                "class_a": a,
                "class_b": b,
                "n_a": n_a,
                "n_b": n_b,
                "n_a_pred_b": n_a_pred_b,
                "n_b_pred_a": n_b_pred_a,
                "rate_a_pred_b": n_a_pred_b / n_a if n_a else math.nan,
                "rate_b_pred_a": n_b_pred_a / n_b if n_b else math.nan,
                "pair_confusion_count": n_a_pred_b + n_b_pred_a,
                "pair_confusion_rate": pair_confusion_rate,
                "tumor_mad_30": tumor_mad,
                "tumor_similarity_mad_30": -tumor_mad if not math.isnan(tumor_mad) else math.nan,
                "tumor_euclidean_30": tumor_euclidean,
                "tumor_similarity_euclidean_30": -tumor_euclidean if not math.isnan(tumor_euclidean) else math.nan,
                "tumor_n_probes": tumor_n_probes,
                "tumor_n_a": int(tumor_counts.get(a, 0)),
                "tumor_n_b": int(tumor_counts.get(b, 0)),
                "has_tumor_pair": bool(has_tumor_pair),
                "normal_mad_30": normal_mad,
                "normal_similarity_mad_30": -normal_mad if not math.isnan(normal_mad) else math.nan,
                "normal_euclidean_30": normal_euclidean,
                "normal_similarity_euclidean_30": -normal_euclidean if not math.isnan(normal_euclidean) else math.nan,
                "normal_n_probes": normal_n_probes,
                "normal_n_a": int(normal_counts.get(a, 0)),
                "normal_n_b": int(normal_counts.get(b, 0)),
                "has_normal_pair": bool(has_normal_pair),
            }
        )
    return pd.DataFrame(rows)


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
        perm = rng.permutation(yr)
        perm_rho = float(np.corrcoef(xr, perm)[0, 1])
        if abs(perm_rho) >= abs_rho:
            extreme += 1
    p = (extreme + 1) / (n_perm + 1)
    return rho, p, n


def compute_stats(pair_df: pd.DataFrame, n_perm: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    analyses = [
        ("tumor", "mad", "tumor_mad_30", "tumor_similarity_mad_30"),
        ("tumor", "euclidean", "tumor_euclidean_30", "tumor_similarity_euclidean_30"),
        ("normal", "mad", "normal_mad_30", "normal_similarity_mad_30"),
        ("normal", "euclidean", "normal_euclidean_30", "normal_similarity_euclidean_30"),
    ]
    for source, metric, distance_col, similarity_col in analyses:
        rho_d, p_d, n_d = spearman_perm(
            pair_df[distance_col], pair_df["pair_confusion_rate"], rng=rng, n_perm=n_perm
        )
        # Similarity is defined as -distance, so Spearman rho flips sign while
        # the two-sided permutation p-value and pair count are identical.
        rho_s, p_s, n_s = -rho_d, p_d, n_d
        rows.append(
            {
                "methylation_source": source,
                "distance_metric": metric,
                "n_pairs": n_d,
                "spearman_rho_distance": rho_d,
                "permutation_p_distance": p_d,
                "spearman_rho_similarity": rho_s,
                "permutation_p_similarity": p_s,
                "n_permutations": n_perm,
            }
        )
        if n_s != n_d:
            raise RuntimeError("Distance and similarity pair counts diverged unexpectedly")
    return pd.DataFrame(rows)


def fmt_p(value: float) -> str:
    if math.isnan(value):
        return "NA"
    if value < 0.001:
        return f"{value:.1e}"
    return f"{value:.3f}"


def add_panel_label(ax: plt.Axes, letter: str) -> None:
    ax.text(-0.12, 1.04, letter, transform=ax.transAxes, ha="left", va="bottom", fontsize=10.5, fontweight="bold")


def plot_scatter(ax: plt.Axes, df: pd.DataFrame, *, x_col: str, title: str, stats_row: pd.Series, color: str) -> None:
    d = df[[x_col, "pair_confusion_rate", "pair_confusion_count", "class_a", "class_b"]].dropna().copy()
    sizes = 12 + np.sqrt(d["pair_confusion_count"].clip(lower=0)) * 10
    ax.scatter(
        d[x_col],
        d["pair_confusion_rate"],
        s=sizes,
        color=color,
        alpha=0.72,
        edgecolor=WHITE,
        linewidth=0.45,
    )
    if len(d) >= 3:
        coef = np.polyfit(d[x_col].to_numpy(dtype=float), d["pair_confusion_rate"].to_numpy(dtype=float), deg=1)
        xs = np.linspace(float(d[x_col].min()), float(d[x_col].max()), 100)
        ax.plot(xs, coef[0] * xs + coef[1], color=INK, linewidth=0.8, alpha=0.75)

    top = d.sort_values(["pair_confusion_rate", "pair_confusion_count"], ascending=False).head(5)
    label_offsets = [(0.000, 0.010), (0.006, 0.006), (-0.006, 0.014), (0.000, 0.011), (0.006, 0.004)]
    for offset, (_, row) in zip(label_offsets, top.iterrows(), strict=False):
        label = f"{str(row['class_a']).replace('TCGA-', '')}-{str(row['class_b']).replace('TCGA-', '')}"
        ax.text(
            row[x_col] + offset[0],
            row["pair_confusion_rate"] + offset[1],
            label,
            fontsize=5.0,
            ha="center",
            va="bottom",
            color=INK,
        )

    rho = stats_row["spearman_rho_similarity"]
    pval = stats_row["permutation_p_similarity"]
    ax.text(
        0.03,
        0.05,
        f"Spearman rho = {rho:.2f}\npermutation p = {fmt_p(pval)}\nn = {int(stats_row['n_pairs'])} pairs",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=6.1,
        color=INK,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": WHITE, "edgecolor": GRID, "linewidth": 0.5},
    )
    ax.set_title(title, fontsize=8.0, fontweight="bold", pad=5)
    ax.set_xlabel("30-CpG methylation similarity\n(-mean absolute beta difference)")
    ax.set_ylabel("Total pair confusion rate")
    ax.grid(True, color=GRID, linewidth=0.4, alpha=0.7)
    ax.spines[["top", "right"]].set_visible(False)


def plot_figure(pair_df: pd.DataFrame, stats: pd.DataFrame, out_png: Path, out_pdf: Path) -> None:
    configure_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.15), sharey=True)
    tumor_stats = stats[(stats["methylation_source"] == "tumor") & (stats["distance_metric"] == "mad")].iloc[0]
    normal_stats = stats[(stats["methylation_source"] == "normal") & (stats["distance_metric"] == "mad")].iloc[0]
    plot_scatter(
        axes[0],
        pair_df,
        x_col="tumor_similarity_mad_30",
        title="TCGA development tumors",
        stats_row=tumor_stats,
        color=BLUE,
    )
    plot_scatter(
        axes[1],
        pair_df,
        x_col="normal_similarity_mad_30",
        title="TCGA normal-adjacent tissues",
        stats_row=normal_stats,
        color=TEAL,
    )
    add_panel_label(axes[0], "A")
    add_panel_label(axes[1], "B")
    axes[1].set_ylabel("")
    fig.tight_layout(w_pad=1.5)
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.03, facecolor=fig.get_facecolor())
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.03, facecolor=fig.get_facecolor())
    plt.close(fig)


def write_summary(
    path: Path,
    *,
    active: dict[str, object],
    predictions: pd.DataFrame,
    pair_df: pd.DataFrame,
    stats: pd.DataFrame,
    probes: list[str],
    outputs: dict[str, str],
) -> None:
    def clean_json_value(value: object) -> object:
        if isinstance(value, dict):
            return {k: clean_json_value(v) for k, v in value.items()}
        if isinstance(value, list):
            return [clean_json_value(v) for v in value]
        if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
            return None
        return value

    stats_records = stats.to_dict(orient="records")
    top_pairs = (
        pair_df.sort_values(["pair_confusion_rate", "pair_confusion_count"], ascending=False)
        .head(10)
        .to_dict(orient="records")
    )
    payload = {
        "active_locked_run_status": active.get("status"),
        "locked_run_name": active.get("locked_run_name"),
        "n_panel_probes": len(probes),
        "n_external_samples": int(len(predictions)),
        "n_external_classes": int(predictions["tcga_label"].nunique()),
        "n_class_pairs": int(len(pair_df)),
        "n_pairs_with_tumor_similarity": int(pair_df["tumor_mad_30"].notna().sum()),
        "n_pairs_with_normal_similarity": int(pair_df["normal_mad_30"].notna().sum()),
        "confusion_rate_definition": "(A predicted as B + B predicted as A) / (true A samples + true B samples)",
        "distance_definition": "centroid distance over the locked 30 MiniCpG probes; MAD is mean absolute beta difference",
        "spearman": stats_records,
        "top_pairs_by_confusion_rate": top_pairs,
        "outputs": outputs,
    }
    path.write_text(json.dumps(clean_json_value(payload), indent=2, allow_nan=False), encoding="utf-8")


def main() -> None:
    args = parse_args()
    active = validate_active_run(args.active_manifest)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    probes = read_panel_probes(Path(args.panel_tsv))
    predictions = pd.read_csv(args.predictions_tsv, sep="\t")
    tumor, tumor_counts = tumor_centroids(Path(args.tcga_dev_panel_matrix), probes)
    normal, normal_counts = normal_centroids(Path(args.normal_medians_wide), Path(args.normal_counts), probes)

    pair_df = build_pair_table(predictions, probes, tumor, tumor_counts, normal, normal_counts)
    stats = compute_stats(pair_df, args.n_permutations, args.seed)

    pair_tsv = out_dir / "failure_pair_confusion_methylation_similarity_30cpg.tsv"
    stats_tsv = out_dir / "failure_pair_similarity_spearman_stats.tsv"
    fig_png = out_dir / "failure_pair_similarity_vs_confusion.png"
    fig_pdf = out_dir / "failure_pair_similarity_vs_confusion.pdf"
    summary_json = out_dir / "failure_pair_similarity_summary.json"

    pair_df.to_csv(pair_tsv, sep="\t", index=False)
    stats.to_csv(stats_tsv, sep="\t", index=False)
    plot_figure(pair_df, stats, fig_png, fig_pdf)
    outputs = {
        "pair_table": str(pair_tsv),
        "stats": str(stats_tsv),
        "figure_png": str(fig_png),
        "figure_pdf": str(fig_pdf),
        "summary_json": str(summary_json),
    }
    write_summary(
        summary_json,
        active=active,
        predictions=predictions,
        pair_df=pair_df,
        stats=stats,
        probes=probes,
        outputs=outputs,
    )
    print(json.dumps({"outputs": outputs, "stats": stats.to_dict(orient="records")}, indent=2))


if __name__ == "__main__":
    main()
