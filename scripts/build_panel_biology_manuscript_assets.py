#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "methylation_classification_mpl_cache"),
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ANCHOR_PROBES = ["cg26687072", "cg19817912", "cg02546607", "cg02748089"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build manuscript-ready biology summary markdown and overview figures from integrated panel evidence."
    )
    parser.add_argument("--probe-master-tsv", required=True)
    parser.add_argument("--gene-master-tsv", required=True)
    parser.add_argument("--lineage-inventory-tsv", required=True)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def plot_evidence_heatmap(probe: pd.DataFrame, output_path: Path) -> None:
    mag = probe[
        [
            "combined_max_tumor_abs_delta",
            "max_external_primary_abs_delta",
            "best_centered_immune_abs",
            "best_centered_purity_spearman_abs",
            "pooled_dynamic_range",
        ]
    ].copy()
    mag.columns = ["TCGA/Ext tumor shift", "External shift", "Immune assoc", "Purity assoc", "Normal range"]
    mag.index = probe["probe_label"]
    signed = probe[["pooled_tumor_normal_order_spearman"]].copy()
    signed.columns = ["Tumor-vs-normal order"]
    signed.index = probe["probe_label"]

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(11.5, max(8.0, 0.32 * len(probe) + 1.5)),
        gridspec_kw={"width_ratios": [5.5, 1.4]},
    )
    sns.set_theme(style="white")
    sns.heatmap(mag, cmap="mako", vmin=0.0, vmax=max(0.5, float(mag.max().max())), linewidths=0.4, ax=axes[0], cbar_kws={"shrink": 0.7})
    axes[0].set_title("Panel Biology Evidence Magnitudes")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("")
    sns.heatmap(signed, cmap="coolwarm", center=0.0, vmin=-1.0, vmax=1.0, linewidths=0.4, ax=axes[1], cbar_kws={"shrink": 0.7})
    axes[1].set_title("Lineage Order")
    axes[1].set_xlabel("")
    axes[1].set_ylabel("")
    plt.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_overview(probe: pd.DataFrame, output_path: Path) -> None:
    label_counts = probe["final_biological_label"].value_counts().sort_values(ascending=False)
    lineage_counts = (
        probe["lineage_state"].fillna("missing").astype(str)
        + " / "
        + probe["lineage_strength"].fillna("missing").astype(str)
    ).value_counts().sort_values(ascending=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), gridspec_kw={"width_ratios": [1.1, 1.1, 1.8]})
    sns.set_theme(style="whitegrid")

    sns.barplot(x=label_counts.values, y=label_counts.index, orient="h", color="#35608d", ax=axes[0])
    axes[0].set_title("Final Probe Labels")
    axes[0].set_xlabel("Probe count")
    axes[0].set_ylabel("")

    sns.barplot(x=lineage_counts.values, y=lineage_counts.index, orient="h", color="#5a9367", ax=axes[1])
    axes[1].set_title("Normal-lineage Calls")
    axes[1].set_xlabel("Probe count")
    axes[1].set_ylabel("")

    color_map = {
        "tumor_shift": "#1b9e77",
        "lineage": "#66a61e",
        "composition": "#d95f02",
        "tumor_shift_plus_lineage": "#7570b3",
        "tumor_shift_plus_composition": "#e6ab02",
        "lineage_plus_composition": "#a6761d",
        "tumor_shift_plus_lineage_plus_composition": "#e7298a",
        "unclear": "#8c8c8c",
    }
    point_sizes = 70 + 220 * probe[["best_centered_immune_abs", "best_centered_purity_spearman_abs"]].max(axis=1)
    axes[2].scatter(
        probe["combined_max_tumor_abs_delta"],
        probe["pooled_tumor_normal_order_spearman"],
        s=point_sizes,
        c=probe["final_biological_label"].map(color_map).fillna("#444444"),
        alpha=0.85,
        edgecolor="white",
        linewidth=0.6,
    )
    label_df = probe.loc[
        probe["probe_id"].isin(ANCHOR_PROBES)
        | (probe["combined_max_tumor_abs_delta"] >= probe["combined_max_tumor_abs_delta"].quantile(0.85))
    ]
    for row in label_df.itertuples(index=False):
        axes[2].text(float(row.combined_max_tumor_abs_delta) + 0.005, float(row.pooled_tumor_normal_order_spearman) + 0.015, str(row.probe_id), fontsize=8)
    axes[2].axhline(0.0, color="#bbbbbb", linestyle="--", linewidth=1)
    axes[2].set_title("Tumor Shift vs Lineage Preservation")
    axes[2].set_xlabel("Max tumor-vs-reference abs delta")
    axes[2].set_ylabel("Tumor-vs-normal tissue-order Spearman")
    plt.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def write_summary(
    probe: pd.DataFrame,
    gene: pd.DataFrame,
    lineage_inventory: pd.DataFrame,
    out_path: Path,
) -> None:
    covered_genes = set(gene["gene"].astype(str))
    missing_bridge_genes = sorted(
        {
            g
            for genes in probe["gene_list"].fillna("")
            for g in [x.strip() for x in str(genes).split(";") if x.strip()]
            if g not in covered_genes
        }
    )

    lines: list[str] = []
    lines.append("# Manuscript-ready Biology Summary")
    lines.append("")
    lines.append(
        f"The biological interpretation uses `{len(probe)}` active panel probes, `{len(lineage_inventory)}` normal/reference samples for lineage analysis, TCGA tumor-vs-normal replication, external tumor-vs-reference replication, and panel-wide immune/purity analysis."
    )
    lines.append("")
    lines.append("## Panel-level Conclusion")
    lines.append("")
    label_counts = probe["final_biological_label"].value_counts()
    lines.append(
        "The panel is best interpreted as a mixture of tumor-shifted, lineage-structured, and composition-sensitive loci rather than a purely tumor-intrinsic set."
    )
    for label, count in label_counts.items():
        lines.append(f"- `{label}`: `{int(count)}` probes")
    lines.append("")
    lines.append("## Anchor Loci")
    lines.append("")
    for probe_id in ANCHOR_PROBES:
        row = probe.loc[probe["probe_id"] == probe_id].iloc[0]
        lines.append(
            f"- `{row['probe_id']}` (`{row['gene_list'] or '__NO_GENE__'}`): `{row['final_biological_label']}`. "
            f"Tumor-shift `{row['combined_max_tumor_abs_delta']:.3f}`, lineage `{row['lineage_state']} / {row['lineage_strength']}`, "
            f"tumor-vs-normal order `{row['pooled_tumor_normal_order_spearman']:.3f}`, immune `{row['best_centered_immune_abs']:.3f}`, purity `{row['best_centered_purity_spearman_abs']:.3f}`."
        )
    lines.append("")
    lines.append("## Probe-level Themes")
    lines.append("")
    top_intrinsic = probe.loc[probe["final_biological_label"] == "tumor_shift"].sort_values("combined_max_tumor_abs_delta", ascending=False).head(5)
    if not top_intrinsic.empty:
        lines.append("Tumor-shift only:")
        for row in top_intrinsic.itertuples(index=False):
            lines.append(f"- `{row.probe_id}` (`{row.gene_list or '__NO_GENE__'}`) with max tumor-shift `{row.combined_max_tumor_abs_delta:.3f}`.")
    top_micro = probe.loc[probe["final_biological_label"].str.contains("composition", na=False)].sort_values(["best_centered_immune_abs", "best_centered_purity_spearman_abs"], ascending=[False, False]).head(6)
    if not top_micro.empty:
        lines.append("Composition-sensitive:")
        for row in top_micro.itertuples(index=False):
            lines.append(
                f"- `{row.probe_id}` (`{row.gene_list or '__NO_GENE__'}`) with immune `{row.best_centered_immune_abs:.3f}` and purity `{row.best_centered_purity_spearman_abs:.3f}`."
            )
    strong_lineage = probe.loc[
        probe["lineage_strength"].isin(["strong", "moderate"])
    ].sort_values(["pooled_dynamic_range", "pooled_tumor_normal_order_spearman"], ascending=[False, False]).head(8)
    if not strong_lineage.empty:
        lines.append("Lineage-preserving normals:")
        for row in strong_lineage.itertuples(index=False):
            lines.append(
                f"- `{row.probe_id}` (`{row.gene_list or '__NO_GENE__'}`): normal range `{row.pooled_dynamic_range:.3f}`, tumor-vs-normal order `{row.pooled_tumor_normal_order_spearman:.3f}`."
            )
    lines.append("")
    lines.append("## Gene-level Themes")
    lines.append("")
    supported = gene.loc[gene["gene_level_interpretation"].isin(["supported_by_probe_behavior", "partially_supported"])].copy()
    for row in supported.head(12).itertuples(index=False):
        lines.append(
            f"- `{row.gene}`: `{row.gene_level_interpretation}` from probes `{row.panel_probe_ids}` with literature bucket `{row.literature_bucket}`."
        )
    lines.append("")
    lines.append("## Suggested Manuscript Wording")
    lines.append("")
    lines.append(
        "A defensible summary is: the selected 30-probe panel is not dominated by a single biological mode. "
        "Several loci show strong replicated tumor-vs-reference shifts, a subset combines tumor shift with preserved tissue-of-origin "
        "structure, and composition-sensitive loci are now labeled independently rather than only as tumor-shift add-ons. "
        "Within the HOX anchors, `HOXB1` and `HOXA-AS2/HOXA3` both retain lineage structure while also shifting in tumors. "
        "The two `WWOX` probes are non-redundant: one retains lineage support, while the other is currently tumor-shift only."
    )
    lines.append("")
    lines.append("## Caveats")
    lines.append("")
    tissue_label_col = "tissue_label" if "tissue_label" in lineage_inventory.columns else None
    if tissue_label_col:
        n_tissues = lineage_inventory[tissue_label_col].nunique()
        lines.append(f"- The lineage analysis uses `{n_tissues}` tissue labels and mixes TCGA solid normals with external matched or adjacent normals.")
    else:
        lines.append("- The lineage analysis mixes TCGA solid normals with external matched or adjacent normals.")
    if missing_bridge_genes:
        lines.append(
            f"- The manual gene-literature bridge is not yet refreshed for annotation-updated genes: `{', '.join(missing_bridge_genes)}`. Probe-level interpretation is current, but those genes are not yet represented in the gene master table."
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    probe = pd.read_csv(args.probe_master_tsv, sep="\t").copy()
    gene = pd.read_csv(args.gene_master_tsv, sep="\t").copy()
    lineage_inventory = pd.read_csv(args.lineage_inventory_tsv, sep="\t").copy()

    probe["panel_rank"] = pd.to_numeric(probe["panel_rank"], errors="coerce")
    probe["gene_list"] = probe["gene_list"].fillna("")
    probe = probe.sort_values(["panel_rank", "probe_id"]).reset_index(drop=True)
    probe["probe_label"] = probe["probe_id"] + " | " + probe["gene_list"].fillna("").replace("", "__NO_GENE__")
    probe["best_centered_purity_spearman_abs"] = pd.to_numeric(probe["best_centered_purity_spearman"], errors="coerce").abs().fillna(0.0)
    numeric_cols = [
        "combined_max_tumor_abs_delta",
        "max_external_primary_abs_delta",
        "best_centered_immune_abs",
        "pooled_dynamic_range",
        "pooled_tumor_normal_order_spearman",
    ]
    for col in numeric_cols:
        probe[col] = pd.to_numeric(probe[col], errors="coerce").fillna(0.0)

    plot_evidence_heatmap(probe, out_dir / "figures" / "panel_biology_evidence_heatmap.png")
    plot_overview(probe, out_dir / "figures" / "panel_biology_summary_overview.png")
    write_summary(probe, gene, lineage_inventory, out_dir / "panel_biology_manuscript_summary.md")

    key = {
        "n_probes": int(len(probe)),
        "final_biological_label_counts": probe["final_biological_label"].value_counts().to_dict(),
        "lineage_state_counts": probe["lineage_state"].value_counts().to_dict(),
        "lineage_strength_counts": probe["lineage_strength"].value_counts().to_dict(),
    }
    (out_dir / "panel_biology_manuscript_key_findings.json").write_text(json.dumps(key, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
