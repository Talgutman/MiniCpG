#!/usr/bin/env python3
"""Build a standalone biological evidence map for the locked MiniCpG panel."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "methylation_classification_mpl_cache"),
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


PAPER_BG = "#fbfaf7"
INK = "#24313f"
MUTED = "#66717d"
GRID = "#e6ded3"
MISSING = "#eee9df"

TUMOR = "#c76548"
LINEAGE = "#2f6f9f"
COMPOSITION = "#d39a3f"
EXPRESSION = "#4f8a78"


FULL_EVIDENCE_COLUMNS = [
    {
        "key": "tumor_shift_abs_delta",
        "label": "Tumor shift\nmax |delta beta|",
        "color": TUMOR,
        "value_col": "max_replicated_mean_abs_delta",
        "norm_col": "tumor_shift_abs_delta_norm",
        "display": "continuous",
    },
    {
        "key": "tumor_shift_breadth",
        "label": "Tumor shift\nTCGA projects",
        "color": TUMOR,
        "value_col": "n_tcga_replicated_projects_ge_0_30_sensitivity",
        "norm_col": "tumor_shift_breadth_norm",
        "display": "count",
    },
    {
        "key": "lineage_range",
        "label": "Lineage\nnormal range pct.",
        "color": LINEAGE,
        "value_col": "lineage_dynamic_percentile",
        "norm_col": "lineage_range_norm",
        "display": "continuous",
    },
    {
        "key": "lineage_order",
        "label": "Lineage\ntumor-normal order pct.",
        "color": LINEAGE,
        "value_col": "lineage_order_positive_percentile",
        "norm_col": "lineage_order_norm",
        "display": "continuous",
    },
    {
        "key": "composition",
        "label": "Composition\nscore",
        "color": COMPOSITION,
        "value_col": "composition_score",
        "norm_col": "composition_norm",
        "display": "continuous",
    },
    {
        "key": "expression",
        "label": "RNA support\nstringent rows",
        "color": EXPRESSION,
        "value_col": "expression_null_stringent_rows",
        "norm_col": "expression_norm",
        "display": "count",
    },
]


COMPACT_EVIDENCE_COLUMNS = [
    {
        "key": "tumor_shift_breadth",
        "label": "Tumor shift\nTCGA projects",
        "color": TUMOR,
        "value_col": "n_tcga_replicated_projects_ge_0_30_sensitivity",
        "norm_col": "tumor_shift_breadth_norm",
        "display": "count",
    },
    {
        "key": "lineage_joint",
        "label": "Lineage\njoint score",
        "color": LINEAGE,
        "value_col": "lineage_joint_score",
        "norm_col": "lineage_joint_norm",
        "display": "continuous",
    },
    {
        "key": "composition",
        "label": "Composition\nscore",
        "color": COMPOSITION,
        "value_col": "composition_score",
        "norm_col": "composition_norm",
        "display": "continuous",
    },
    {
        "key": "expression",
        "label": "RNA support\nstringent rows",
        "color": EXPRESSION,
        "value_col": "expression_null_stringent_rows",
        "norm_col": "expression_norm",
        "display": "count",
    },
]


def evidence_columns(view: str) -> list[dict[str, str]]:
    if view == "full":
        return FULL_EVIDENCE_COLUMNS
    if view == "compact":
        return COMPACT_EVIDENCE_COLUMNS
    if view == "compact_no_rna":
        return [col for col in COMPACT_EVIDENCE_COLUMNS if col["key"] != "expression"]
    raise ValueError(f"Unsupported evidence-map view: {view}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--active-manifest", default="config/active_locked_run.json")
    parser.add_argument(
        "--evidence-tsv",
        default=(
            "outputs/paper_protocolb_v1/k_030/biological_interpretation_current/"
            "manuscript_current/biological_threshold_sensitivity_current/"
            "continuous_biological_evidence_scores.tsv"
        ),
    )
    parser.add_argument(
        "--expression-pair-summary-tsv",
        default=(
            "outputs/paper_protocolb_v1/k_030/biological_interpretation_current/"
            "mechanism_expression_screen_current/transcriptome_null_inspection_current/"
            "transcriptome_null_pair_summary.tsv"
        ),
    )
    parser.add_argument(
        "--out-dir",
        default=(
            "outputs/paper_protocolb_v1/k_030/biological_interpretation_current/"
            "manuscript_current/evidence_map_current"
        ),
    )
    parser.add_argument("--out-stem", default="panel_biological_evidence_map")
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"], choices=["png", "pdf"])
    parser.add_argument(
        "--view",
        default="compact",
        choices=["full", "compact", "compact_no_rna"],
        help="Evidence-map column set. compact uses one joint lineage score and tumor-shift breadth.",
    )
    return parser.parse_args()


def validate_active_run(manifest: str) -> dict[str, object]:
    proc = subprocess.run(
        [sys.executable, "scripts/validate_active_locked_run.py", "--manifest", manifest],
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(proc.stdout)
    if payload.get("status") != "ok":
        raise SystemExit(f"Active locked-run validation failed: {payload.get('status')}")
    return payload


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 6.4,
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


def require_columns(frame: pd.DataFrame, required: set[str], label: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise SystemExit(f"{label} missing columns: {missing}")


def normalize(series: pd.Series, high: float | None = None) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    if high is None:
        high = float(values.max(skipna=True)) if values.notna().any() else 1.0
    if high <= 0 or not np.isfinite(high):
        high = 1.0
    return (values / high).clip(0.0, 1.0)


def aggregate_expression_pairs(path: Path) -> pd.DataFrame:
    pair = pd.read_csv(path, sep="\t")
    required = {
        "probe_id",
        "gene",
        "null_stringent_rows",
        "null_interesting_rows",
        "top_1pct_rows",
        "max_abs_rho",
        "median_transcriptome_abs_rho_percentile",
    }
    require_columns(pair, required, "expression pair summary")
    pair["null_stringent_rows"] = pd.to_numeric(pair["null_stringent_rows"], errors="coerce").fillna(0.0)
    pair["null_interesting_rows"] = pd.to_numeric(pair["null_interesting_rows"], errors="coerce").fillna(0.0)
    pair["top_1pct_rows"] = pd.to_numeric(pair["top_1pct_rows"], errors="coerce").fillna(0.0)
    return (
        pair.groupby("probe_id", as_index=False)
        .agg(
            expression_linked_genes=("gene", "nunique"),
            expression_null_stringent_rows=("null_stringent_rows", "sum"),
            expression_null_interesting_rows=("null_interesting_rows", "sum"),
            expression_top_1pct_rows=("top_1pct_rows", "sum"),
            expression_max_abs_rho=("max_abs_rho", "max"),
            expression_best_median_percentile=("median_transcriptome_abs_rho_percentile", "max"),
        )
    )


def short_gene(value: object, max_len: int = 18) -> str:
    if pd.isna(value) or str(value).strip() == "":
        return ""
    gene = str(value)
    if len(gene) <= max_len:
        return gene
    return gene[: max_len - 3] + "..."


def build_plot_data(evidence_path: Path, expression_path: Path) -> pd.DataFrame:
    evidence = pd.read_csv(evidence_path, sep="\t")
    required = {
        "probe_id",
        "panel_rank",
        "gene_list",
        "max_replicated_mean_abs_delta",
        "n_tcga_replicated_projects_ge_0_30_sensitivity",
        "lineage_dynamic_percentile",
        "lineage_order_positive_percentile",
        "composition_score",
        "tumor_shift_sensitivity_group",
        "lineage_sensitivity_group",
        "composition_sensitivity_group",
    }
    require_columns(evidence, required, "continuous evidence")
    expr = aggregate_expression_pairs(expression_path)
    out = evidence.merge(expr, on="probe_id", how="left")
    fill_zero = [
        "expression_linked_genes",
        "expression_null_stringent_rows",
        "expression_null_interesting_rows",
        "expression_top_1pct_rows",
    ]
    out[fill_zero] = out[fill_zero].fillna(0.0)
    out = out.sort_values("panel_rank").reset_index(drop=True)
    def display_label(row: pd.Series) -> str:
        gene = short_gene(row["gene_list"])
        display = gene if gene else str(row["probe_id"])
        return f"{int(row['panel_rank']):02d} {display}"

    out["display_label"] = out.apply(display_label, axis=1)

    out["tumor_shift_abs_delta_norm"] = normalize(out["max_replicated_mean_abs_delta"], high=0.60)
    out["tumor_shift_breadth_norm"] = normalize(out["n_tcga_replicated_projects_ge_0_30_sensitivity"])
    out["lineage_range_norm"] = normalize(out["lineage_dynamic_percentile"], high=1.0)
    out["lineage_order_norm"] = normalize(out["lineage_order_positive_percentile"], high=1.0)
    lineage_range = pd.to_numeric(out["lineage_dynamic_percentile"], errors="coerce").clip(0.0, 1.0)
    lineage_order = pd.to_numeric(out["lineage_order_positive_percentile"], errors="coerce").clip(0.0, 1.0)
    out["lineage_joint_score"] = np.sqrt(lineage_range * lineage_order)
    out["lineage_joint_norm"] = normalize(out["lineage_joint_score"], high=1.0)
    out["composition_norm"] = normalize(out["composition_score"], high=0.55)
    out["expression_norm"] = normalize(out["expression_null_stringent_rows"])
    return out


def dot_size(norm: float) -> float:
    if pd.isna(norm):
        return 18.0
    return 24.0 + 178.0 * float(norm)


def plot_evidence_map(data: pd.DataFrame, columns: list[dict[str, str]], out_paths: list[Path]) -> None:
    configure_matplotlib()
    n_rows = len(data)
    n_cols = len(columns)
    width = 3.9 + 0.65 * n_cols
    fig, ax = plt.subplots(figsize=(width, 7.4))

    ax.set_xlim(-0.55, n_cols - 0.45)
    ax.set_ylim(n_rows - 0.5, -0.5)
    ax.set_xticks(range(n_cols))
    ax.set_xticklabels([col["label"] for col in columns], rotation=35, ha="left", va="bottom")
    ax.xaxis.tick_top()
    ax.tick_params(axis="x", pad=5, length=0)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(data["display_label"].tolist(), fontsize=6.0)
    ax.tick_params(axis="y", length=0)

    for idx in range(n_cols + 1):
        ax.axvline(idx - 0.5, color=GRID, linewidth=0.6, zorder=0)
    for idx in range(n_rows + 1):
        ax.axhline(idx - 0.5, color=GRID, linewidth=0.35, zorder=0)

    for x, col in enumerate(columns):
        values = pd.to_numeric(data[col["value_col"]], errors="coerce")
        norms = pd.to_numeric(data[col["norm_col"]], errors="coerce")
        color = str(col["color"])
        for y, (value, norm) in enumerate(zip(values, norms, strict=True)):
            if pd.isna(value):
                ax.scatter(x, y, s=25, color=MISSING, edgecolor="none", zorder=2)
                continue
            alpha = 0.18 + 0.72 * float(np.clip(norm, 0.0, 1.0))
            ax.scatter(
                x,
                y,
                s=dot_size(float(norm)),
                color=color,
                alpha=alpha,
                edgecolor=PAPER_BG,
                linewidth=0.7,
                zorder=3,
            )
            if col["display"] == "count" and float(value) > 0:
                ax.text(x, y, str(int(round(float(value)))), ha="center", va="center", fontsize=5.7, color=INK, zorder=4)

    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("")
    ax.set_ylabel("")

    size_handles = [
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="#9aa6ad", alpha=0.35, markeredgecolor="none", markersize=4, label="Low"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="#9aa6ad", alpha=0.65, markeredgecolor="none", markersize=7, label="Medium"),
        Line2D([0], [0], marker="o", linestyle="none", markerfacecolor="#9aa6ad", alpha=0.90, markeredgecolor="none", markersize=10, label="High"),
    ]
    if n_cols <= 4:
        ax.legend(
            handles=size_handles,
            loc="lower center",
            bbox_to_anchor=(0.5, -0.085),
            frameon=False,
            ncol=3,
            handletextpad=0.35,
            columnspacing=0.75,
        )
    else:
        present = {str(col["key"]) for col in columns}
        color_specs = [
            ("tumor_shift_breadth", TUMOR, "Tumor shift"),
            ("tumor_shift_abs_delta", TUMOR, "Tumor shift"),
            ("lineage_range", LINEAGE, "Lineage-like"),
            ("lineage_order", LINEAGE, "Lineage-like"),
            ("lineage_joint", LINEAGE, "Lineage-like"),
            ("composition", COMPOSITION, "Composition"),
            ("expression", EXPRESSION, "Methylation-RNA"),
        ]
        color_handles = []
        seen_labels = set()
        for key, color, label in color_specs:
            if key in present and label not in seen_labels:
                color_handles.append(
                    Line2D([0], [0], marker="o", linestyle="none", markerfacecolor=color, markeredgecolor="none", markersize=6, label=label)
                )
                seen_labels.add(label)
        legend1 = ax.legend(
            handles=color_handles,
            loc="lower left",
            bbox_to_anchor=(-0.01, -0.085),
            frameon=False,
            ncol=len(color_handles),
            handletextpad=0.35,
            columnspacing=0.9,
        )
        ax.add_artist(legend1)
        ax.legend(
            handles=size_handles,
            loc="lower right",
            bbox_to_anchor=(1.0, -0.085),
            frameon=False,
            ncol=3,
            handletextpad=0.35,
            columnspacing=0.75,
        )

    for path in out_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() == ".png":
            fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.03, facecolor=fig.get_facecolor())
        else:
            fig.savefig(path, bbox_inches="tight", pad_inches=0.03, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> int:
    args = parse_args()
    active = validate_active_run(args.active_manifest)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    data = build_plot_data(Path(args.evidence_tsv), Path(args.expression_pair_summary_tsv))
    columns = evidence_columns(args.view)
    out_paths = [out_dir / f"{args.out_stem}.{fmt}" for fmt in args.formats]
    plot_evidence_map(data, columns, out_paths)

    data_out = out_dir / f"{args.out_stem}_plot_data.tsv"
    summary_out = out_dir / f"{args.out_stem}_summary.json"
    data.to_csv(data_out, sep="\t", index=False)
    summary = {
        "active_locked_run_status": active.get("status"),
        "locked_run_name": active.get("locked_run_name"),
        "n_panel_probes": int(len(data)),
        "view": args.view,
        "columns": [col["key"] for col in columns],
        "evidence_source": str(args.evidence_tsv),
        "expression_source": str(args.expression_pair_summary_tsv),
        "outputs": [str(path) for path in out_paths] + [str(data_out), str(summary_out)],
        "normalization_notes": {
            "tumor_shift_abs_delta": "max replicated mean absolute tumor-normal beta delta divided by 0.60 and clipped to 1",
            "tumor_shift_breadth": "replicated TCGA projects with abs(delta beta) >= 0.30 divided by the panel maximum",
            "lineage_columns": "empirical percentiles from threshold-sensitivity analysis",
            "lineage_joint": "geometric mean of lineage dynamic-range percentile and tumor-normal order percentile",
            "composition": "composition score divided by 0.55 and clipped to 1",
            "expression": "sum of transcriptome-null stringent probe-gene-cohort rows divided by the panel maximum",
        },
    }
    summary_out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
