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
import numpy as np
import pandas as pd
import seaborn as sns


FOCUS_PROBES = ["cg26687072", "cg19817912", "cg02546607", "cg02748089"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze panel probe behavior across normal/reference tissues to assess lineage support. "
            "Panel-agnostic and reusable across future panels."
        )
    )
    parser.add_argument("--panel-annotations-tsv", required=True)
    parser.add_argument("--probes-tsv", required=True)
    parser.add_argument("--tcga-normal-matrix-tsv", required=True)
    parser.add_argument("--external-panel-matrix-tsv", default="")
    parser.add_argument("--external-labels-tsv", default="")
    parser.add_argument("--tcga-dev-matrix-tsv", required=True)
    parser.add_argument("--tcga-lockbox-matrix-tsv", default="")
    parser.add_argument("--tcga-only-lineage", action="store_true")
    parser.add_argument("--min-normal-samples", type=int, default=10)
    parser.add_argument("--min-tumor-samples", type=int, default=30)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def read_probe_list(path: Path) -> list[str]:
    df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    if df.empty:
        raise SystemExit(f"Empty probe file: {path}")
    col = "probe_id" if "probe_id" in df.columns else df.columns[0]
    probes = [str(v).strip() for v in df[col].tolist() if str(v).strip()]
    if not probes:
        raise SystemExit(f"No probe IDs found in: {path}")
    return probes


def load_probe_meta(path: Path, probe_order: list[str]) -> pd.DataFrame:
    ann = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    ann = ann.loc[ann["probe_id"].isin(probe_order), ["probe_id", "gene_list", "panel_rank", "context_group"]].copy()
    ann["gene_label"] = ann["gene_list"].replace("", "Intergenic")
    ann = ann.set_index("probe_id")
    missing = [probe for probe in probe_order if probe not in ann.index]
    if missing:
        filler = pd.DataFrame(
            {
                "gene_list": [""] * len(missing),
                "panel_rank": [""] * len(missing),
                "context_group": [""] * len(missing),
                "gene_label": ["Intergenic"] * len(missing),
            },
            index=missing,
        )
        ann = pd.concat([ann, filler], axis=0)
    ann.index.name = "probe_id"
    return ann.loc[probe_order].reset_index()


def _safe_spearman(a: pd.Series, b: pd.Series) -> float:
    joined = pd.concat([a, b], axis=1).dropna()
    if len(joined) < 3:
        return np.nan
    if joined.iloc[:, 0].nunique() < 2 or joined.iloc[:, 1].nunique() < 2:
        return np.nan
    return float(joined.iloc[:, 0].corr(joined.iloc[:, 1], method="spearman"))


def classify_lineage_support(row: pd.Series) -> str:
    n_tissues = int(row["n_pooled_tissues"])
    dynamic_range = float(row["pooled_dynamic_range"]) if pd.notna(row["pooled_dynamic_range"]) else 0.0
    corr = float(row["pooled_tumor_normal_order_spearman"]) if pd.notna(row["pooled_tumor_normal_order_spearman"]) else np.nan

    if n_tissues >= 5 and dynamic_range >= 0.20 and pd.notna(corr) and corr >= 0.60:
        return "strong_lineage_support"
    if n_tissues >= 4 and dynamic_range >= 0.15 and pd.notna(corr) and corr >= 0.40:
        return "moderate_lineage_support"
    if n_tissues >= 4 and dynamic_range >= 0.15 and pd.notna(corr) and corr <= 0.00:
        return "tumor_shifted_away_from_normal_baseline"
    if n_tissues >= 3 and dynamic_range >= 0.10:
        return "weak_lineage_support"
    return "limited_lineage_support"


def load_tcga_normals(path: Path, probe_order: list[str]) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    required = {"project_id", "sample_id"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"TCGA normal matrix missing required columns: {sorted(missing)}")
    missing_probes = [probe for probe in probe_order if probe not in df.columns]
    if missing_probes:
        raise SystemExit(f"TCGA normal matrix missing probes: {missing_probes[:8]}")
    base_cols = [col for col in ["project_id", "sample_id", "patient_id", "sample_type", "sample_type_code"] if col in df.columns]
    out = df[base_cols + probe_order].copy()
    out["source_group"] = "tcga_normal"
    out["source_id"] = out["project_id"]
    out["tissue_label"] = out["project_id"]
    out["reference_category"] = "tcga_solid_normal"
    return out


def load_external_normals(matrix_path: Path, labels_path: Path, probe_order: list[str]) -> pd.DataFrame:
    matrix = pd.read_csv(matrix_path, sep="\t")
    labels = pd.read_csv(labels_path, sep="\t", dtype=str).fillna("")
    normals = labels.loc[labels["canonical_sample_role"] == "normal", ["gse_id", "sample_id", "tcga_label", "reference_category", "sample_reference_label", "comparison_role", "comparison_ready"]].copy()
    normals["comparison_ready"] = pd.to_numeric(normals["comparison_ready"], errors="coerce").fillna(0).astype(int)
    normals = normals.loc[normals["comparison_ready"] == 1].copy()
    missing_probes = [probe for probe in probe_order if probe not in matrix.columns]
    if missing_probes:
        raise SystemExit(f"External panel matrix missing probes: {missing_probes[:8]}")
    merged = normals.merge(matrix[["gse_id", "sample_id"] + probe_order], on=["gse_id", "sample_id"], how="inner")
    merged["source_group"] = "external_reference"
    merged["source_id"] = merged["gse_id"]
    merged["tissue_label"] = merged["tcga_label"]
    merged["reference_category"] = merged["reference_category"].replace("", "external_reference")
    return merged


def empty_external_normals(probe_order: list[str]) -> pd.DataFrame:
    cols = [
        "gse_id",
        "sample_id",
        "tcga_label",
        "reference_category",
        "sample_reference_label",
        "comparison_role",
        "comparison_ready",
        "source_group",
        "source_id",
        "tissue_label",
    ] + probe_order
    return pd.DataFrame(columns=cols)


def load_tcga_tumors(dev_path: Path, lockbox_path: Path | None, probe_order: list[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for source, path in [("dev", dev_path), ("lockbox", lockbox_path)]:
        if path is None:
            continue
        df = pd.read_csv(path, sep="\t")
        required = {"project_id", "sample_id"}
        missing = required - set(df.columns)
        if missing:
            raise SystemExit(f"TCGA {source} tumor matrix missing required columns: {sorted(missing)}")
        missing_probes = [probe for probe in probe_order if probe not in df.columns]
        if missing_probes:
            raise SystemExit(f"TCGA {source} tumor matrix missing probes: {missing_probes[:8]}")
        out = df[["project_id", "sample_id"] + probe_order].copy()
        out["lineage_tumor_source"] = source
        frames.append(out)
    if not frames:
        raise SystemExit("No TCGA tumor matrices supplied.")
    return pd.concat(frames, ignore_index=True, sort=False)


def melt_samples(df: pd.DataFrame, probe_meta: pd.DataFrame, probe_order: list[str]) -> pd.DataFrame:
    id_cols = [col for col in ["source_group", "source_id", "tissue_label", "reference_category", "project_id", "gse_id", "sample_id"] if col in df.columns]
    long_df = df.melt(id_vars=id_cols, value_vars=probe_order, var_name="probe_id", value_name="beta_value")
    long_df = long_df.merge(probe_meta, on="probe_id", how="left")
    long_df["probe_label"] = long_df["probe_id"] + " | " + long_df["gene_label"]
    return long_df


def plot_heatmap(wide: pd.DataFrame, output_path: Path) -> None:
    plt.figure(figsize=(max(7.0, 1.2 * wide.shape[1]), max(7.0, 0.33 * wide.shape[0] + 1.5)))
    sns.set_theme(style="white")
    ax = sns.heatmap(wide, cmap="viridis", vmin=0.0, vmax=1.0, linewidths=0.4)
    ax.set_title("Normal / Reference Tissue Median Beta Values")
    ax.set_xlabel("Tissue label")
    ax.set_ylabel("Probe")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_lineage_scatter(summary: pd.DataFrame, output_path: Path) -> None:
    color_map = {
        "strong_lineage_support": "#1b9e77",
        "moderate_lineage_support": "#66a61e",
        "weak_lineage_support": "#7570b3",
        "tumor_shifted_away_from_normal_baseline": "#d95f02",
        "limited_lineage_support": "#8c8c8c",
    }
    plot_df = summary.copy()
    plt.figure(figsize=(8.2, 6.2))
    sns.set_theme(style="whitegrid")
    ax = plt.gca()
    for call, sub in plot_df.groupby("lineage_support_call", sort=False):
        ax.scatter(
            sub["pooled_dynamic_range"],
            sub["pooled_tumor_normal_order_spearman"],
            s=55,
            alpha=0.9,
            label=call,
            color=color_map.get(call, "#444444"),
        )
    label_df = plot_df.loc[
        plot_df["probe_id"].isin(FOCUS_PROBES)
        | (plot_df["pooled_dynamic_range"] >= plot_df["pooled_dynamic_range"].quantile(0.85))
    ].copy()
    for row in label_df.itertuples(index=False):
        ax.text(
            float(row.pooled_dynamic_range) + 0.004,
            float(row.pooled_tumor_normal_order_spearman) + 0.01 if pd.notna(row.pooled_tumor_normal_order_spearman) else 0.01,
            str(row.probe_id),
            fontsize=8,
        )
    ax.axhline(0.4, color="#bbbbbb", linestyle="--", linewidth=1)
    ax.axvline(0.15, color="#bbbbbb", linestyle="--", linewidth=1)
    ax.set_xlabel("Pooled normal/reference tissue median range")
    ax.set_ylabel("Tumor vs normal tissue-order Spearman")
    ax.set_title("Probe-level lineage support from normal tissues")
    ax.legend(frameon=True, fontsize=8, loc="lower right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def write_summary_md(
    summary: pd.DataFrame,
    counts: pd.DataFrame,
    output_path: Path,
) -> None:
    lines: list[str] = []
    lines.append("# Normal-only Lineage Analysis Summary")
    lines.append("")
    total_n = int(counts["n_samples"].sum())
    lines.append(f"Normal/reference samples analyzed: `{total_n}` across `{int(counts['tissue_label'].nunique())}` tissue labels.")
    lines.append("")
    lines.append("## Sample Counts")
    lines.append("")
    for row in counts.itertuples(index=False):
        lines.append(f"- `{row.tissue_label}` / `{row.source_group}` / `{row.reference_category}`: `{int(row.n_samples)}`")
    lines.append("")
    lines.append("## Lineage-support Call Counts")
    lines.append("")
    for call, n in summary["lineage_support_call"].value_counts().sort_index().items():
        lines.append(f"- `{call}`: `{int(n)}`")
    lines.append("")
    lines.append("## Strongest Lineage-support Probes")
    lines.append("")
    strong = summary.sort_values(
        ["lineage_support_call", "pooled_dynamic_range", "pooled_tumor_normal_order_spearman"],
        ascending=[True, False, False],
    )
    for row in strong.loc[strong["lineage_support_call"].isin(["strong_lineage_support", "moderate_lineage_support"])].head(10).itertuples(index=False):
        lines.append(
            f"- `{row.probe_id}` (`{row.gene_label}`): `{row.lineage_support_call}` with pooled normal range `{row.pooled_dynamic_range:.3f}` and tumor-normal tissue-order Spearman `{row.pooled_tumor_normal_order_spearman:.3f}` across `{int(row.n_pooled_tissues)}` tissues."
        )
    lines.append("")
    lines.append("## HOX / WWOX Anchors")
    lines.append("")
    for probe_id in FOCUS_PROBES:
        sub = summary.loc[summary["probe_id"] == probe_id]
        if sub.empty:
            continue
        row = sub.iloc[0]
        lines.append(
            f"- `{row['probe_id']}` (`{row['gene_label']}`): `{row['lineage_support_call']}`; pooled normal range `{row['pooled_dynamic_range']:.3f}`, pooled tumor-normal tissue-order Spearman `{row['pooled_tumor_normal_order_spearman']:.3f}`, strongest pooled-normal tissues `{row['top_pooled_normal_tissue_high']}` vs `{row['top_pooled_normal_tissue_low']}`."
        )
    lines.append("")
    lines.append("## Output Files")
    lines.append("")
    lines.append("- `normal_reference_sample_inventory.tsv`")
    lines.append("- `normal_reference_counts_by_tissue.tsv`")
    lines.append("- `normal_reference_tissue_medians_pooled.tsv`")
    lines.append("- `normal_reference_tissue_medians_by_source.tsv`")
    lines.append("- `probe_lineage_summary.tsv`")
    lines.append("- `figures/normal_reference_tissue_heatmap.png`")
    lines.append("- `figures/probe_lineage_support_scatter.png`")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    probe_order = read_probe_list(Path(args.probes_tsv))
    probe_meta = load_probe_meta(Path(args.panel_annotations_tsv), probe_order)
    probe_meta_lookup = probe_meta.set_index("probe_id")

    tcga_normals = load_tcga_normals(Path(args.tcga_normal_matrix_tsv), probe_order)
    if args.external_panel_matrix_tsv and args.external_labels_tsv and not args.tcga_only_lineage:
        external_normals = load_external_normals(Path(args.external_panel_matrix_tsv), Path(args.external_labels_tsv), probe_order)
    else:
        external_normals = empty_external_normals(probe_order)

    tumor = load_tcga_tumors(
        Path(args.tcga_dev_matrix_tsv),
        Path(args.tcga_lockbox_matrix_tsv) if args.tcga_lockbox_matrix_tsv else None,
        probe_order,
    )
    normal_counts = tcga_normals.groupby("project_id").size().rename("n_normal_samples")
    tumor_counts = tumor.groupby("project_id").size().rename("n_tumor_samples")
    project_counts = (
        pd.concat([normal_counts, tumor_counts], axis=1)
        .fillna(0)
        .astype(int)
        .reset_index()
        .rename(columns={"index": "project_id"})
    )
    project_counts["eligible_for_lineage"] = (
        (project_counts["n_normal_samples"] >= args.min_normal_samples)
        & (project_counts["n_tumor_samples"] >= args.min_tumor_samples)
    )
    project_counts.to_csv(out_dir / "tcga_lineage_project_eligibility.tsv", sep="\t", index=False)
    eligible_projects = project_counts.loc[project_counts["eligible_for_lineage"], "project_id"].astype(str).tolist()
    if len(eligible_projects) < 3:
        raise SystemExit(
            "Fewer than 3 TCGA projects pass lineage eligibility "
            f"(min normals={args.min_normal_samples}, min tumors={args.min_tumor_samples})."
        )

    tcga_normals = tcga_normals.loc[tcga_normals["project_id"].isin(eligible_projects)].copy()
    tumor = tumor.loc[tumor["project_id"].isin(eligible_projects)].copy()
    if args.tcga_only_lineage:
        combined_normals = tcga_normals.copy()
    else:
        external_normals = external_normals.loc[external_normals["tissue_label"].isin(eligible_projects)].copy()
        combined_normals = pd.concat([tcga_normals, external_normals], ignore_index=True, sort=False)
    combined_normals.to_csv(out_dir / "normal_reference_sample_inventory.tsv", sep="\t", index=False)

    counts = (
        combined_normals.groupby(["tissue_label", "source_group", "reference_category"], dropna=False)
        .size()
        .reset_index(name="n_samples")
        .sort_values(["tissue_label", "source_group", "reference_category"])
    )
    counts.to_csv(out_dir / "normal_reference_counts_by_tissue.tsv", sep="\t", index=False)

    long_normals = melt_samples(combined_normals, probe_meta, probe_order)

    pooled_medians = (
        long_normals.groupby(["tissue_label", "probe_id", "probe_label", "gene_label"], dropna=False)["beta_value"]
        .median()
        .reset_index(name="median_beta")
    )
    pooled_medians.to_csv(out_dir / "normal_reference_tissue_medians_pooled.tsv", sep="\t", index=False)

    source_medians = (
        long_normals.groupby(["source_group", "tissue_label", "probe_id", "probe_label", "gene_label"], dropna=False)["beta_value"]
        .median()
        .reset_index(name="median_beta")
    )
    source_medians.to_csv(out_dir / "normal_reference_tissue_medians_by_source.tsv", sep="\t", index=False)

    tumor_medians = tumor.groupby("project_id")[probe_order].median().T

    tcga_normals_long = melt_samples(tcga_normals, probe_meta, probe_order)
    tcga_medians = (
        tcga_normals_long.groupby(["tissue_label", "probe_id"], dropna=False)["beta_value"]
        .median()
        .reset_index(name="median_beta")
    )
    ext_normals_long = melt_samples(external_normals, probe_meta, probe_order)
    ext_medians = (
        ext_normals_long.groupby(["tissue_label", "probe_id"], dropna=False)["beta_value"]
        .median()
        .reset_index(name="median_beta")
    )

    summary_rows: list[dict[str, object]] = []
    for probe_id in probe_order:
        tcga_series = tcga_medians.loc[tcga_medians["probe_id"] == probe_id].set_index("tissue_label")["median_beta"]
        ext_series = ext_medians.loc[ext_medians["probe_id"] == probe_id].set_index("tissue_label")["median_beta"]
        pooled_series = pooled_medians.loc[pooled_medians["probe_id"] == probe_id].set_index("tissue_label")["median_beta"]
        tumor_series = tumor_medians.loc[probe_id].copy()

        overlap_tcga = tcga_series.index.intersection(tumor_series.index)
        overlap_ext = ext_series.index.intersection(tumor_series.index)
        overlap_pooled = pooled_series.index.intersection(tumor_series.index)

        top_high = str(pooled_series.sort_values(ascending=False).index[0]) if not pooled_series.empty else ""
        top_low = str(pooled_series.sort_values(ascending=True).index[0]) if not pooled_series.empty else ""

        row = {
            "probe_id": probe_id,
            "gene_label": probe_meta_lookup.at[probe_id, "gene_label"],
            "lineage_source": "tcga_only" if args.tcga_only_lineage else "tcga_plus_external_reference",
            "lineage_tumor_sources": "+".join(sorted(tumor["lineage_tumor_source"].dropna().astype(str).unique())),
            "lineage_min_normal_samples": int(args.min_normal_samples),
            "lineage_min_tumor_samples": int(args.min_tumor_samples),
            "n_lineage_projects": int(len(overlap_pooled)),
            "lineage_eligible_projects": ";".join(overlap_pooled.astype(str).tolist()),
            "n_tcga_normal_tissues": int(tcga_series.index.nunique()),
            "n_external_reference_tissues": int(ext_series.index.nunique()),
            "n_pooled_tissues": int(pooled_series.index.nunique()),
            "tcga_normal_dynamic_range": float(tcga_series.max() - tcga_series.min()) if len(tcga_series) >= 2 else np.nan,
            "external_reference_dynamic_range": float(ext_series.max() - ext_series.min()) if len(ext_series) >= 2 else np.nan,
            "pooled_dynamic_range": float(pooled_series.max() - pooled_series.min()) if len(pooled_series) >= 2 else np.nan,
            "lineage_normal_dynamic_range": float(pooled_series.max() - pooled_series.min()) if len(pooled_series) >= 2 else np.nan,
            "tcga_tumor_normal_order_spearman": _safe_spearman(tcga_series.reindex(overlap_tcga), tumor_series.reindex(overlap_tcga)),
            "external_tumor_normal_order_spearman": _safe_spearman(ext_series.reindex(overlap_ext), tumor_series.reindex(overlap_ext)),
            "pooled_tumor_normal_order_spearman": _safe_spearman(pooled_series.reindex(overlap_pooled), tumor_series.reindex(overlap_pooled)),
            "lineage_tumor_normal_order_spearman": _safe_spearman(pooled_series.reindex(overlap_pooled), tumor_series.reindex(overlap_pooled)),
            "tcga_external_normal_order_spearman": _safe_spearman(tcga_series, ext_series),
            "top_pooled_normal_tissue_high": top_high,
            "top_pooled_normal_tissue_low": top_low,
        }
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary["lineage_support_call"] = summary.apply(classify_lineage_support, axis=1)
    summary.to_csv(out_dir / "probe_lineage_summary.tsv", sep="\t", index=False)

    heatmap_df = pooled_medians.pivot(index="probe_label", columns="tissue_label", values="median_beta")
    heatmap_df = heatmap_df.reindex(probe_meta["probe_id"].map(lambda p: f"{p} | {probe_meta_lookup.at[p, 'gene_label']}"))
    heatmap_df.to_csv(out_dir / "normal_reference_tissue_medians_pooled_wide.tsv", sep="\t")

    plot_heatmap(heatmap_df, out_dir / "figures" / "normal_reference_tissue_heatmap.png")
    plot_lineage_scatter(summary, out_dir / "figures" / "probe_lineage_support_scatter.png")
    write_summary_md(summary, counts, out_dir / "lineage_normal_summary.md")

    key = {
        "n_normal_reference_samples": int(len(combined_normals)),
        "n_tissue_labels": int(combined_normals["tissue_label"].nunique()),
        "lineage_source": "tcga_only" if args.tcga_only_lineage else "tcga_plus_external_reference",
        "lineage_tumor_sources": sorted(tumor["lineage_tumor_source"].dropna().astype(str).unique().tolist()),
        "min_normal_samples": int(args.min_normal_samples),
        "min_tumor_samples": int(args.min_tumor_samples),
        "n_eligible_tcga_projects": int(len(eligible_projects)),
        "eligible_tcga_projects": eligible_projects,
        "lineage_support_call_counts": summary["lineage_support_call"].value_counts().to_dict(),
    }
    (out_dir / "lineage_normal_key_findings.json").write_text(json.dumps(key, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
