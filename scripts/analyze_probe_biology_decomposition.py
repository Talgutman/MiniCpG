#!/usr/bin/env python3
"""Modules 2-5: tumor-side probe biology decomposition for MiniCpG probes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd




KEY_COLS = {"sample_id", "project_id", "patient_id", "sample_type", "sample_type_code", "source_split"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--normal-lineage", required=True)
    p.add_argument("--normal-audit", required=True)
    p.add_argument("--biology-master", required=True)
    p.add_argument("--tcga-dev", required=True)
    p.add_argument("--tcga-lockbox", required=True)
    p.add_argument("--tumor-normal-dev", required=True)
    p.add_argument("--tumor-normal-lockbox", required=True)
    p.add_argument("--matched-dev", required=True)
    p.add_argument("--matched-lockbox", required=True)
    p.add_argument("--composition", required=True)
    p.add_argument("--within-composition", required=True)
    p.add_argument("--composition-breadth-rho", type=float, default=0.30)
    p.add_argument("--composition-min-samples", type=int, default=30)
    return p.parse_args()


def read_tsv(path: str | Path, **kwargs) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"Missing input: {path}")
    return pd.read_csv(path, sep="\t", **kwargs)


def probe_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if str(c).startswith("cg")]


def eta_squared(values: pd.Series, groups: pd.Series) -> float:
    y = pd.to_numeric(values, errors="coerce")
    valid = y.notna() & groups.notna()
    y = y.loc[valid]
    g = groups.loc[valid]
    if len(y) < 2:
        return np.nan
    grand = float(y.mean())
    ss_total = float(((y - grand) ** 2).sum())
    if ss_total <= 0:
        return 0.0
    ss_between = 0.0
    for _, vals in y.groupby(g):
        ss_between += len(vals) * (float(vals.mean()) - grand) ** 2
    return ss_between / ss_total


def percentile_score(df: pd.DataFrame, cols: list[str], out_col: str) -> pd.DataFrame:
    ranks = []
    for col in cols:
        ranks.append(df[col].rank(ascending=True, pct=True))
    df[out_col] = pd.concat(ranks, axis=1).mean(axis=1)
    return df


def spearman_no_scipy(x: pd.Series, y: pd.Series) -> float:
    xr = pd.to_numeric(x, errors="coerce").rank(method="average")
    yr = pd.to_numeric(y, errors="coerce").rank(method="average")
    valid = xr.notna() & yr.notna()
    if int(valid.sum()) < 3:
        return np.nan
    return float(xr.loc[valid].corr(yr.loc[valid], method="pearson"))


def lineage_preservation(normal_audit_path: str, dev_path: str, lockbox_path: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    normal = read_tsv(normal_audit_path)
    dev = read_tsv(dev_path)
    lockbox = read_tsv(lockbox_path)
    dev["source_split"] = "development"
    lockbox["source_split"] = "heldout_tcga"
    probes = probe_columns(dev)
    tumor = pd.concat([dev[["sample_id", "project_id", "source_split"] + probes], lockbox[["sample_id", "project_id", "source_split"] + probes]], ignore_index=True)
    represented = sorted(set(normal["project_id"].astype(str)).intersection(tumor["project_id"].astype(str)))

    rows = []
    for split_name, split_df in [("normal_adjacent", normal), ("development", dev), ("heldout_tcga", lockbox), ("combined_tcga", tumor)]:
        groups = split_df["project_id"].astype(str)
        for probe in probes:
            vals = pd.to_numeric(split_df[probe], errors="coerce")
            med = vals.groupby(groups).median()
            rows.append(
                {
                    "split": split_name,
                    "probe_id": probe,
                    "n_samples": int(vals.notna().sum()),
                    "n_tumor_classes": int(med.notna().sum()),
                    "tumor_class_dynamic_range": float(med.max() - med.min()),
                    "tumor_class_median_sd": float(med.std(ddof=0)),
                    "tumor_eta_squared": float(eta_squared(vals, groups)),
                    "top_tumor_class_high": str(med.idxmax()),
                    "top_tumor_class_low": str(med.idxmin()),
                    "top_tumor_class_high_median": float(med.max()),
                    "top_tumor_class_low_median": float(med.min()),
                }
            )
    long_df = pd.DataFrame(rows)
    normal_groups = normal["project_id"].astype(str)
    tumor_groups = tumor["project_id"].astype(str)
    summary_rows = []
    for probe in probes:
        normal_vals = pd.to_numeric(normal[probe], errors="coerce")
        tumor_vals = pd.to_numeric(tumor[probe], errors="coerce")
        normal_med = normal_vals.groupby(normal_groups).median().reindex(represented)
        tumor_med = tumor_vals.groupby(tumor_groups).median().reindex(represented)
        valid = normal_med.notna() & tumor_med.notna()
        rho = spearman_no_scipy(normal_med.loc[valid], tumor_med.loc[valid]) if int(valid.sum()) >= 3 else np.nan
        rho_positive = max(float(rho), 0.0) if pd.notna(rho) else np.nan
        normal_dynamic_range = float(normal_med.max() - normal_med.min())
        tumor_dynamic_range = float(tumor_med.max() - tumor_med.min())
        summary_rows.append(
            {
                "probe_id": probe,
                "n_normal_tissues": int(normal_med.notna().sum()),
                "n_tumor_classes_for_preservation": int(valid.sum()),
                "normal_lineage_strength_dynamic_range": normal_dynamic_range,
                "normal_tissue_median_sd": float(normal_med.std(ddof=0)),
                "normal_eta_squared": float(eta_squared(normal_vals, normal_groups)),
                "tumor_lineage_dynamic_range_represented_classes": tumor_dynamic_range,
                "normal_tumor_lineage_spearman": float(rho) if pd.notna(rho) else np.nan,
                "normal_tumor_lineage_preservation_positive": rho_positive,
                "top_normal_tissue_high": str(normal_med.idxmax()),
                "top_normal_tissue_low": str(normal_med.idxmin()),
                "top_normal_tissue_high_median": float(normal_med.max()),
                "top_normal_tissue_low_median": float(normal_med.min()),
                "top_tumor_class_high": str(tumor_med.idxmax()),
                "top_tumor_class_low": str(tumor_med.idxmin()),
                "top_tumor_class_high_median": float(tumor_med.max()),
                "top_tumor_class_low_median": float(tumor_med.min()),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary["lineage_strength_percentile"] = summary["normal_lineage_strength_dynamic_range"].rank(ascending=True, pct=True)
    summary["lineage_preservation_percentile"] = summary["normal_tumor_lineage_preservation_positive"].rank(ascending=True, pct=True)
    summary["lineage_score"] = summary[["lineage_strength_percentile", "lineage_preservation_percentile"]].mean(axis=1)
    return long_df, summary.sort_values("lineage_score", ascending=False)


def summarize_tumor_normal(unpaired_path: str, matched_path: str, split: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    unpaired = read_tsv(unpaired_path)
    matched = read_tsv(matched_path)
    unpaired["split"] = split
    matched["split"] = split
    return unpaired, matched


def tumor_normal_shift(dev_path: str, lockbox_path: str, matched_dev_path: str, matched_lockbox_path: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dev, matched_dev = summarize_tumor_normal(dev_path, matched_dev_path, "development")
    lockbox, matched_lockbox = summarize_tumor_normal(lockbox_path, matched_lockbox_path, "heldout_tcga")
    unpaired = pd.concat([dev, lockbox], ignore_index=True)
    matched = pd.concat([matched_dev, matched_lockbox], ignore_index=True)

    rows = []
    for probe, sub in unpaired.groupby("probe_id", sort=False):
        dev_sub = sub.loc[sub["split"] == "development"].copy()
        lock_sub = sub.loc[sub["split"] == "heldout_tcga"].copy()
        dev_d = dev_sub.set_index("project_id")["delta_median_beta"]
        lock_d = lock_sub.set_index("project_id")["delta_median_beta"]
        common = sorted(set(dev_d.index).intersection(lock_d.index))
        same_direction = []
        min_abs_rep = []
        for project in common:
            d1 = float(dev_d.loc[project])
            d2 = float(lock_d.loc[project])
            if d1 == 0 or d2 == 0:
                same = False
            else:
                same = np.sign(d1) == np.sign(d2)
            if same:
                same_direction.append(project)
                min_abs_rep.append(min(abs(d1), abs(d2)))
        all_abs = sub["delta_median_beta"].abs()
        matched_sub = matched.loc[matched["probe_id"] == probe].copy()
        matched_abs = matched_sub["matched_delta_median_beta"].abs() if not matched_sub.empty else pd.Series(dtype=float)
        rows.append(
            {
                "probe_id": probe,
                "n_unpaired_project_split_rows": int(len(sub)),
                "n_projects_development": int(dev_sub["project_id"].nunique()),
                "n_projects_heldout_tcga": int(lock_sub["project_id"].nunique()),
                "n_projects_replicated_direction": int(len(same_direction)),
                "replicated_direction_projects": ";".join(same_direction),
                "replicated_max_min_abs_delta": float(max(min_abs_rep)) if min_abs_rep else 0.0,
                "replicated_median_min_abs_delta": float(np.median(min_abs_rep)) if min_abs_rep else 0.0,
                "replicated_n_projects_min_abs_ge_0_20": int(sum(v >= 0.20 for v in min_abs_rep)),
                "replicated_n_projects_min_abs_ge_0_30": int(sum(v >= 0.30 for v in min_abs_rep)),
                "unpaired_max_abs_delta": float(all_abs.max()) if len(all_abs) else np.nan,
                "unpaired_median_abs_delta": float(all_abs.median()) if len(all_abs) else np.nan,
                "matched_max_abs_delta": float(matched_abs.max()) if len(matched_abs) else np.nan,
                "matched_median_abs_delta": float(matched_abs.median()) if len(matched_abs) else np.nan,
                "matched_n_project_split_rows": int(len(matched_sub)),
                "matched_n_project_split_rows_abs_ge_0_20": int((matched_abs >= 0.20).sum()) if len(matched_abs) else 0,
                "matched_n_project_split_rows_abs_ge_0_30": int((matched_abs >= 0.30).sum()) if len(matched_abs) else 0,
            }
        )
    summary = pd.DataFrame(rows)
    summary["tumor_shift_strength_percentile"] = summary["replicated_max_min_abs_delta"].rank(ascending=True, pct=True)
    summary["tumor_shift_breadth_percentile"] = summary["replicated_n_projects_min_abs_ge_0_20"].rank(ascending=True, pct=True)
    summary["tumor_normal_shift_score"] = summary[["tumor_shift_strength_percentile", "tumor_shift_breadth_percentile"]].mean(axis=1)
    summary = summary.sort_values("tumor_normal_shift_score", ascending=False)
    return unpaired, matched, summary


def tumor_composition(path: str, within_path: str, breadth_rho: float, min_samples: int) -> pd.DataFrame:
    comp = read_tsv(path)
    within = read_tsv(within_path)
    within["n_samples"] = pd.to_numeric(within["n_samples"], errors="coerce")
    within["abs_spearman"] = pd.to_numeric(within["spearman"], errors="coerce").abs()
    within_eligible = within.loc[within["n_samples"] >= min_samples].copy()
    rows = []
    for row in comp.itertuples(index=False):
        centered_purity = abs(float(getattr(row, "best_centered_purity_spearman")))
        centered_immune = abs(float(getattr(row, "best_centered_immune_abs")))
        within_abs = abs(float(getattr(row, "max_within_class_abs_spearman")))
        probe_within = within_eligible.loc[within_eligible["probe_id"] == row.probe_id].copy()
        broad_hits = probe_within.loc[probe_within["abs_spearman"] >= breadth_rho].copy()
        n_broad_project_covariate_hits = int(len(broad_hits))
        n_broad_projects = int(broad_hits["project_id"].nunique()) if not broad_hits.empty else 0
        n_broad_covariates = int(broad_hits["covariate"].nunique()) if not broad_hits.empty else 0
        median_abs_hit = float(broad_hits["abs_spearman"].median()) if not broad_hits.empty else 0.0
        rows.append(
            {
                "probe_id": row.probe_id,
                "best_centered_purity_spearman": float(getattr(row, "best_centered_purity_spearman")),
                "best_centered_immune_covariate": getattr(row, "best_centered_immune_covariate"),
                "best_centered_immune_spearman": float(getattr(row, "best_centered_immune_spearman")),
                "best_centered_immune_abs": centered_immune,
                "pan_tumor_centered_composition_abs": max(centered_purity, centered_immune),
                "max_within_class_project_id": getattr(row, "max_within_class_project_id"),
                "max_within_class_covariate": getattr(row, "max_within_class_covariate"),
                "max_within_class_spearman": float(getattr(row, "max_within_class_spearman")),
                "max_within_class_abs_spearman": within_abs,
                "tumor_composition_max_strength": within_abs,
                "tumor_composition_breadth_project_count": n_broad_projects,
                "tumor_composition_breadth_project_covariate_count": n_broad_project_covariate_hits,
                "tumor_composition_breadth_covariate_count": n_broad_covariates,
                "tumor_composition_breadth_median_abs_spearman_ge_threshold": median_abs_hit,
                "tumor_composition_breadth_rho_threshold": breadth_rho,
                "tumor_composition_breadth_min_samples": min_samples,
                "tumor_composition_primary_call": getattr(row, "primary_call"),
            }
        )
    out = pd.DataFrame(rows)
    out["tumor_composition_strength_percentile"] = out["tumor_composition_max_strength"].rank(ascending=True, pct=True)
    out["tumor_composition_breadth_percentile"] = out["tumor_composition_breadth_project_count"].rank(ascending=True, pct=True)
    out["tumor_composition_score"] = out[["tumor_composition_strength_percentile", "tumor_composition_breadth_percentile"]].mean(axis=1)
    return out.sort_values("tumor_composition_score", ascending=False)


def evidence_tags(row: pd.Series) -> str:
    tags: list[str] = []
    if float(row.get("lineage_score", 0)) >= 0.75:
        tags.append("lineage")
    if float(row.get("tumor_normal_shift_score", 0)) >= 0.75:
        tags.append("tumor_normal_shift")
    if float(row.get("tumor_composition_score", 0)) >= 0.75:
        tags.append("tumor_composition")
    if not tags:
        tags.append("mixed_or_lower_evidence")
    return ";".join(tags)


def integrate(
    normal_path: str,
    biology_path: str,
    lineage_summary: pd.DataFrame,
    tumor_normal_summary: pd.DataFrame,
    comp_summary: pd.DataFrame,
) -> pd.DataFrame:
    normal = read_tsv(normal_path)
    biology = read_tsv(biology_path)
    meta_cols = [
        c
        for c in [
            "probe_id",
            "panel_rank",
            "gene_label",
            "gene_list",
            "context_group",
            "stability_selection_rate",
            "final_biological_label",
            "label_rationale",
        ]
        if c in biology.columns
    ]
    base = biology[meta_cols].drop_duplicates("probe_id").copy()
    out = base.merge(normal, on="probe_id", how="left")
    out = out.merge(lineage_summary, on="probe_id", how="left", suffixes=("", "_lineage"))
    out = out.merge(tumor_normal_summary, on="probe_id", how="left")
    out = out.merge(comp_summary, on="probe_id", how="left")
    out["normal_composition_status"] = out.get("normal_composition_status", "not_used")
    out["normal_composition_score"] = np.nan
    out["evidence_tags"] = out.apply(evidence_tags, axis=1)
    score_cols = [
        "lineage_score",
        "tumor_normal_shift_score",
        "tumor_composition_score",
    ]
    out["dominant_axis"] = out[score_cols].idxmax(axis=1).str.replace("_score", "", regex=False)
    out["dominant_axis_score"] = out[score_cols].max(axis=1)
    return out.sort_values(["panel_rank", "probe_id"])


def write_summary(out_dir: Path, master: pd.DataFrame, lineage: pd.DataFrame, tumor_shift: pd.DataFrame, comp: pd.DataFrame) -> None:
    tag_counts = master["evidence_tags"].str.get_dummies(sep=";").sum().sort_values(ascending=False)
    lines = [
        "# Probe Biology Decomposition",
        "",
        "Modules 2-5 output for the healthy/normal tissue probe biology decomposition plan.",
        "",
        "## Scope",
        f"- Probes: `{master['probe_id'].nunique()}`",
        "- Normal composition: `not_used` by user decision.",
        "- Lineage score: mean percentile of normal tissue dynamic range and normal-to-tumor rank preservation across the 14 normal-represented classes.",
        "- Tumor-shift score: mean percentile of replicated tumor-normal shift strength and breadth.",
        "- Composition score: mean percentile of within-class purity/leukocyte association strength and breadth.",
        "",
        "## Top Lineage Probes",
    ]
    for row in lineage.head(8).itertuples(index=False):
        lines.append(
            f"- `{row.probe_id}`: score `{row.lineage_score:.3f}`, "
            f"normal range `{row.normal_lineage_strength_dynamic_range:.3f}`, "
            f"normal-tumor Spearman `{row.normal_tumor_lineage_spearman:.3f}`, "
            f"normal high `{row.top_normal_tissue_high}`, normal low `{row.top_normal_tissue_low}`"
        )
    lines.extend(["", "## Top Tumor-Normal Shift Probes"])
    for row in tumor_shift.head(8).itertuples(index=False):
        lines.append(
            f"- `{row.probe_id}`: score `{row.tumor_normal_shift_score:.3f}`, "
            f"replicated max-min abs delta `{row.replicated_max_min_abs_delta:.3f}`, "
            f"replicated projects >=0.20 `{row.replicated_n_projects_min_abs_ge_0_20}`"
        )
    lines.extend(["", "## Top Tumor-Composition Probes"])
    for row in comp.head(8).itertuples(index=False):
        lines.append(
            f"- `{row.probe_id}`: score `{row.tumor_composition_score:.3f}`, "
            f"max within-class abs rho `{row.tumor_composition_max_strength:.3f}`, "
            f"breadth projects `{row.tumor_composition_breadth_project_count}`, "
            f"best within-class `{row.max_within_class_project_id}/{row.max_within_class_covariate}`"
        )
    lines.extend(["", "## Evidence Tag Counts"])
    for tag, count in tag_counts.items():
        lines.append(f"- `{tag}`: `{int(count)}`")
    lines.extend(
        [
            "",
            "## Outputs",
            "- `module2_lineage_long.tsv`",
            "- `probe_lineage_evidence.tsv`",
            "- `module3_tumor_normal_unpaired_project_split.tsv`",
            "- `module3_tumor_normal_matched_project_split.tsv`",
            "- `probe_tumor_normal_shift_evidence.tsv`",
            "- `probe_tumor_composition_evidence.tsv`",
            "- `probe_biology_decomposition_master.tsv`",
            "- `probe_biology_decomposition_compact.tsv`",
            "- `figures/probe_biology_decomposition_score_heatmap.png`",
        ]
    )
    (out_dir / "probe_biology_decomposition_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_compact_table(out_dir: Path, master: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "probe_id",
        "panel_rank",
        "gene_label",
        "lineage_score",
        "lineage_strength_percentile",
        "lineage_preservation_percentile",
        "normal_lineage_strength_dynamic_range",
        "normal_tumor_lineage_spearman",
        "tumor_normal_shift_score",
        "tumor_shift_strength_percentile",
        "tumor_shift_breadth_percentile",
        "tumor_composition_score",
        "tumor_composition_strength_percentile",
        "tumor_composition_breadth_percentile",
        "normal_composition_status",
        "evidence_tags",
        "dominant_axis",
        "dominant_axis_score",
        "top_normal_tissue_high",
        "top_normal_tissue_low",
        "top_tumor_class_high",
        "top_tumor_class_low",
        "replicated_n_projects_min_abs_ge_0_20",
        "max_within_class_project_id",
        "max_within_class_covariate",
        "final_biological_label",
    ]
    compact = master[[c for c in cols if c in master.columns]].copy()
    compact.to_csv(out_dir / "probe_biology_decomposition_compact.tsv", sep="\t", index=False)
    return compact


def plot_score_heatmap(out_dir: Path, compact: pd.DataFrame) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_df = compact.copy().sort_values("panel_rank")
    score_cols = [
        "lineage_score",
        "tumor_normal_shift_score",
        "tumor_composition_score",
    ]
    labels = ["Lineage", "Tumor shift", "Composition"]
    mat = plot_df[score_cols].to_numpy(dtype=float)
    ylabels = [
        f"{int(row.panel_rank):02d}  {row.probe_id}  {row.gene_label if str(row.gene_label) else 'Intergenic'}"
        for row in plot_df.itertuples(index=False)
    ]
    fig_h = max(7.2, 0.28 * len(plot_df) + 1.8)
    fig, ax = plt.subplots(figsize=(8.2, fig_h))
    im = ax.imshow(mat, aspect="auto", vmin=0, vmax=1, cmap="viridis")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.set_yticks(np.arange(len(ylabels)))
    ax.set_yticklabels(ylabels, fontsize=7)
    ax.tick_params(axis="both", length=0)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=6.5, color="white" if mat[i, j] > 0.55 else "black")
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Evidence score", rotation=90)
    ax.set_frame_on(False)
    fig.tight_layout()
    fig.savefig(fig_dir / "probe_biology_decomposition_score_heatmap.png", dpi=300)
    fig.savefig(fig_dir / "probe_biology_decomposition_score_heatmap.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    lineage_long, lineage_summary = lineage_preservation(args.normal_audit, args.tcga_dev, args.tcga_lockbox)
    unpaired, matched, shift_summary = tumor_normal_shift(
        args.tumor_normal_dev,
        args.tumor_normal_lockbox,
        args.matched_dev,
        args.matched_lockbox,
    )
    comp_summary = tumor_composition(args.composition, args.within_composition, args.composition_breadth_rho, args.composition_min_samples)
    master = integrate(args.normal_lineage, args.biology_master, lineage_summary, shift_summary, comp_summary)

    lineage_long.to_csv(out_dir / "module2_lineage_long.tsv", sep="\t", index=False)
    lineage_summary.to_csv(out_dir / "probe_lineage_evidence.tsv", sep="\t", index=False)
    unpaired.to_csv(out_dir / "module3_tumor_normal_unpaired_project_split.tsv", sep="\t", index=False)
    matched.to_csv(out_dir / "module3_tumor_normal_matched_project_split.tsv", sep="\t", index=False)
    shift_summary.to_csv(out_dir / "probe_tumor_normal_shift_evidence.tsv", sep="\t", index=False)
    comp_summary.to_csv(out_dir / "probe_tumor_composition_evidence.tsv", sep="\t", index=False)
    master.to_csv(out_dir / "probe_biology_decomposition_master.tsv", sep="\t", index=False)
    compact = write_compact_table(out_dir, master)
    plot_score_heatmap(out_dir, compact)
    write_summary(out_dir, master, lineage_summary, shift_summary, comp_summary)

    manifest = {
        "out_dir": str(out_dir),
        "n_probes": int(master["probe_id"].nunique()),
        "top_lineage_probes": lineage_summary.head(8)["probe_id"].tolist(),
        "top_tumor_normal_shift_probes": shift_summary.head(8)["probe_id"].tolist(),
        "top_tumor_composition_probes": comp_summary.head(8)["probe_id"].tolist(),
        "normal_composition_status": "not_used",
    }
    (out_dir / "probe_biology_decomposition_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
