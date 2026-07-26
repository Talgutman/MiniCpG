#!/usr/bin/env python3
"""Plot PRJNA628686 cfDNA-native supplementary panel."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


LABEL_ORDER = ["Normal", "CRC", "ESCA", "GASTRIC", "LIHC", "PAAD"]
CANCER_ORDER = ["CRC", "ESCA", "GASTRIC", "LIHC", "PAAD"]

METRIC_LABELS = {
    "gate_specificity": "Normal specificity",
    "gate_sensitivity": "Cancer sensitivity",
    "subtype_accuracy_gate_positive_cancers": "Subtype accuracy\n(gate-positive)",
    "subtype_macro_f1_gate_positive_cancers": "Subtype macro-F1\n(gate-positive)",
}

COLORS = {
    "Normal": "#7A7A7A",
    "CRC": "#4F8C6B",
    "ESCA": "#C76F42",
    "GASTRIC": "#8C6BB1",
    "LIHC": "#3B6EA8",
    "PAAD": "#C9A227",
}


def read_metric_table(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    if not {"metric", "mean", "std"}.issubset(df.columns):
        raise ValueError(f"Unexpected metric table columns in {path}: {list(df.columns)}")
    return df


def metric_value(metrics: pd.DataFrame, name: str) -> float:
    rows = metrics.loc[metrics["metric"] == name, "mean"]
    if rows.empty:
        raise KeyError(f"Metric not found: {name}")
    return float(rows.iloc[0])


def build_plot_data(run_dir: Path, coverage_dir: Path | None, out_dir: Path) -> dict:
    pred_path = run_dir / "outer_predictions.tsv"
    metrics_path = run_dir / "aggregate_metrics.tsv"
    manifest_path = run_dir / "run_manifest.json"
    if not pred_path.exists():
        raise FileNotFoundError(pred_path)
    if not metrics_path.exists():
        raise FileNotFoundError(metrics_path)
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)

    pred = pd.read_csv(pred_path, sep="\t")
    metrics = read_metric_table(metrics_path)
    manifest = json.loads(manifest_path.read_text())

    unique_samples = (
        pred[["sample_id", "true_label"]]
        .drop_duplicates()
        .groupby("true_label", observed=False)
        .size()
        .reindex(LABEL_ORDER)
        .fillna(0)
        .astype(int)
        .rename("n_samples")
        .reset_index()
        .rename(columns={"true_label": "label"})
    )
    unique_samples["color"] = unique_samples["label"].map(COLORS)

    metric_rows = []
    for key, label in METRIC_LABELS.items():
        metric_rows.append({"metric": key, "label": label, "value": metric_value(metrics, key)})
    headline = pd.DataFrame(metric_rows)

    gp = pred[(pred["true_label"].isin(CANCER_ORDER)) & (pred["gate_pred"] == "Cancer")].copy()
    conf_counts = pd.crosstab(gp["true_label"], gp["subtype_pred"])
    conf_counts = conf_counts.reindex(index=CANCER_ORDER, columns=CANCER_ORDER, fill_value=0)
    conf_pct = conf_counts.div(conf_counts.sum(axis=1).replace(0, np.nan), axis=0) * 100
    conf_pct = conf_pct.fillna(0)

    gate_pass = (
        pred[pred["true_label"].isin(CANCER_ORDER)]
        .assign(gate_positive=lambda x: x["gate_pred"].eq("Cancer"))
        .groupby("true_label", observed=False)
        .agg(
            n_fold_predictions=("sample_id", "size"),
            n_gate_positive=("gate_positive", "sum"),
            n_unique_samples=("sample_id", "nunique"),
        )
        .reindex(CANCER_ORDER)
        .reset_index()
    )
    gate_pass["gate_positive_fraction"] = gate_pass["n_gate_positive"] / gate_pass["n_fold_predictions"]

    out_dir.mkdir(parents=True, exist_ok=True)
    unique_samples.to_csv(out_dir / "prjna_cfdna_cohort_counts.tsv", sep="\t", index=False)
    headline.to_csv(out_dir / "prjna_cfdna_k100_headline_metrics.tsv", sep="\t", index=False)
    conf_counts.to_csv(out_dir / "prjna_cfdna_k100_gate_positive_confusion_counts.tsv", sep="\t")
    conf_pct.to_csv(out_dir / "prjna_cfdna_k100_gate_positive_confusion_row_percent.tsv", sep="\t")
    gate_pass.to_csv(out_dir / "prjna_cfdna_k100_gate_positive_denominators.tsv", sep="\t", index=False)

    coverage_summary = None
    if coverage_dir is not None and (coverage_dir / "aggregate_metrics.tsv").exists():
        cov = read_metric_table(coverage_dir / "aggregate_metrics.tsv")
        coverage_rows = []
        for key, label in METRIC_LABELS.items():
            coverage_rows.append(
                {
                    "metric": key,
                    "label": label,
                    "methylation_value": metric_value(metrics, key),
                    "coverage_value": metric_value(cov, key),
                }
            )
        coverage_summary = pd.DataFrame(coverage_rows)
        coverage_summary.to_csv(out_dir / "prjna_cfdna_k100_coverage_control_metrics.tsv", sep="\t", index=False)

    summary = {
        "source_run_dir": str(run_dir),
        "coverage_run_dir": str(coverage_dir) if coverage_dir else None,
        "n_samples": int(unique_samples["n_samples"].sum()),
        "n_classes": int((unique_samples["n_samples"] > 0).sum()),
        "n_folds": int(manifest["n_folds"]),
        "max_union": int(manifest["max_union"]),
        "target_specificity": float(manifest["target_specificity"]),
        "headline_metrics": {row["metric"]: float(row["value"]) for row in metric_rows},
        "coverage_control_sidecar": coverage_summary is not None,
    }
    (out_dir / "prjna_cfdna_supp_panel_summary.json").write_text(json.dumps(summary, indent=2))

    return {
        "counts": unique_samples,
        "headline": headline,
        "conf_pct": conf_pct,
        "conf_counts": conf_counts,
        "gate_pass": gate_pass,
        "summary": summary,
    }


def style_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)
        spine.set_color("#444444")
    ax.tick_params(axis="both", labelsize=7, width=0.6, length=2.5, color="#444444")


def panel_letter(ax, letter: str) -> None:
    ax.text(-0.12, 1.08, letter, transform=ax.transAxes, fontsize=11, fontweight="bold", va="top")


def draw_workflow(ax, summary: dict) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    boxes = [
        (0.05, 0.58, 0.24, 0.25, "cfDNA\nmethylation"),
        (0.38, 0.58, 0.24, 0.25, "Cancer vs\nnormal"),
        (0.71, 0.58, 0.24, 0.25, "Tumor-type\nassignment"),
    ]
    for x, y, w, h, text in boxes:
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.02,rounding_size=0.025",
            linewidth=0.8,
            edgecolor="#374151",
            facecolor="#F6F4EF",
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8, color="#1F2937")
    for x0, x1 in [(0.29, 0.38), (0.62, 0.71)]:
        ax.add_patch(
            FancyArrowPatch(
                (x0, 0.705),
                (x1, 0.705),
                arrowstyle="-|>",
                mutation_scale=9,
                linewidth=0.9,
                color="#4B5563",
            )
        )

    ax.text(
        0.50,
        0.35,
        f"{summary['max_union']} CpGs | {summary['n_folds']} CV folds",
        ha="center",
        va="center",
        fontsize=7,
        color="#374151",
    )
    ax.text(
        0.50,
        0.22,
        f"normal specificity target: {summary['target_specificity']:.0%}",
        ha="center",
        va="center",
        fontsize=7,
        color="#374151",
    )


def plot_panel(plot_data: dict, out_dir: Path) -> None:
    counts = plot_data["counts"]
    headline = plot_data["headline"]
    conf_pct = plot_data["conf_pct"]
    conf_counts = plot_data["conf_counts"]
    summary = plot_data["summary"]

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig = plt.figure(figsize=(7.0, 5.8), facecolor="white")
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.25], height_ratios=[0.85, 1.15], wspace=0.32, hspace=0.48)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])

    # A: cohort composition.
    y = np.arange(len(counts))
    ax_a.barh(y, counts["n_samples"], color=counts["color"], edgecolor="none", height=0.66)
    ax_a.set_yticks(y)
    ax_a.set_yticklabels(counts["label"], fontsize=7)
    ax_a.invert_yaxis()
    ax_a.set_xlabel("Samples", fontsize=8)
    ax_a.set_xlim(0, max(counts["n_samples"]) * 1.18)
    ax_a.grid(axis="x", linewidth=0.35, color="#D6D3CB", alpha=0.8)
    for yi, val in zip(y, counts["n_samples"]):
        ax_a.text(val + max(counts["n_samples"]) * 0.025, yi, str(int(val)), va="center", fontsize=7, color="#333333")
    style_axes(ax_a)
    panel_letter(ax_a, "A")

    # B: workflow.
    draw_workflow(ax_b, summary)
    panel_letter(ax_b, "B")

    # C: headline metrics.
    metric_order = [
        "Normal specificity",
        "Cancer sensitivity",
        "Subtype accuracy\n(gate-positive)",
        "Subtype macro-F1\n(gate-positive)",
    ]
    metric_df = headline.set_index("label").loc[metric_order].reset_index()
    yy = np.arange(len(metric_df))
    colors = ["#6B7280", "#4F8C6B", "#3B6EA8", "#3B6EA8"]
    ax_c.barh(yy, metric_df["value"] * 100, color=colors, height=0.58)
    ax_c.set_yticks(yy)
    ax_c.set_yticklabels(metric_df["label"], fontsize=7)
    ax_c.invert_yaxis()
    ax_c.set_xlim(0, 100)
    ax_c.set_xlabel("Performance (%)", fontsize=8)
    ax_c.grid(axis="x", linewidth=0.35, color="#D6D3CB", alpha=0.8)
    for yi, val in zip(yy, metric_df["value"] * 100):
        ax_c.text(val + 1.5, yi, f"{val:.1f}", va="center", fontsize=7, color="#333333")
    style_axes(ax_c)
    panel_letter(ax_c, "C")

    # D: gate-positive cancer confusion.
    matrix = conf_pct.loc[CANCER_ORDER, CANCER_ORDER].to_numpy()
    im = ax_d.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=max(80, float(np.nanmax(matrix))))
    ax_d.set_xticks(np.arange(len(CANCER_ORDER)))
    ax_d.set_yticks(np.arange(len(CANCER_ORDER)))
    ax_d.set_xticklabels(CANCER_ORDER, fontsize=7, rotation=35, ha="right")
    ax_d.set_yticklabels(CANCER_ORDER, fontsize=7)
    ax_d.set_xlabel("Subtype prediction", fontsize=8)
    ax_d.set_ylabel("True cancer type", fontsize=8)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            count = int(conf_counts.iloc[i, j])
            if val >= 5 or i == j:
                color = "white" if val >= 45 else "#1F2937"
                ax_d.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=6.5, color=color)
    for spine in ax_d.spines.values():
        spine.set_visible(False)
    ax_d.tick_params(axis="both", length=0)
    cbar = fig.colorbar(im, ax=ax_d, fraction=0.046, pad=0.03)
    cbar.ax.tick_params(labelsize=7, width=0.5, length=2)
    cbar.set_label("Subtype calls (%)", fontsize=7)
    panel_letter(ax_d, "D")

    out_png = out_dir / "prjna_cfdna_supp_panel.png"
    out_pdf = out_dir / "prjna_cfdna_supp_panel.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.03, facecolor=fig.get_facecolor())
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.03, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path(
            "outputs/paper_protocolb_v1/manuscript_reproduction/"
            "prjna628686_fullset_separate_esca_gastric_ovrsubtype_ksweep_methyl_k100_v1"
        ),
    )
    parser.add_argument(
        "--coverage-dir",
        type=Path,
        default=Path(
            "outputs/paper_protocolb_v1/manuscript_reproduction/"
            "prjna628686_fullset_separate_esca_gastric_ovrsubtype_ksweep_coverage_k100_v1"
        ),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "outputs/paper_protocolb_v1/manuscript_current/figures/"
            "supp_prjna_cfdna_panel_current"
        ),
    )
    args = parser.parse_args()
    data = build_plot_data(args.run_dir, args.coverage_dir, args.out_dir)
    plot_panel(data, args.out_dir)
    print(f"Wrote {args.out_dir / 'prjna_cfdna_supp_panel.png'}")


if __name__ == "__main__":
    main()
