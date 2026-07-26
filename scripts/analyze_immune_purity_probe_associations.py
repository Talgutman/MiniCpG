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


LEGACY_IMMUNE_PROBES = ["cg08940169", "cg02753354", "cg02075410", "cg27147556", "cg02357751"]
LEGACY_CONTROL_PROBES = ["cg26687072", "cg19817912", "cg02546607", "cg02748089"]
LEGACY_FOCUS_PROBES = LEGACY_IMMUNE_PROBES + LEGACY_CONTROL_PROBES

LYMPHOID_COLS = [
    "B.cells.naive",
    "B.cells.memory",
    "Plasma.cells",
    "T.cells.CD8",
    "T.cells.CD4.naive",
    "T.cells.CD4.memory.resting",
    "T.cells.CD4.memory.activated",
    "T.cells.follicular.helper",
    "T.cells.regulatory..Tregs.",
    "T.cells.gamma.delta",
    "NK.cells.resting",
    "NK.cells.activated",
]
MYELOID_COLS = [
    "Monocytes",
    "Macrophages.M0",
    "Macrophages.M1",
    "Macrophages.M2",
    "Dendritic.cells.resting",
    "Dendritic.cells.activated",
    "Mast.cells.resting",
    "Mast.cells.activated",
    "Eosinophils",
    "Neutrophils",
]
COVARIATE_ORDER = ["purity", "leukocyte_fraction", "lymphoid_fraction", "myeloid_fraction"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assess whether selected methylation probes track purity or immune covariates.")
    parser.add_argument("--probe-matrix-tsv", required=True)
    parser.add_argument("--panel-annotations-tsv", required=True)
    parser.add_argument("--probes-tsv", default="", help="Optional ordered probe list TSV. Defaults to annotation-ranked probes found in the matrix.")
    parser.add_argument(
        "--focus-probes-tsv",
        default="",
        help="Optional subset probe TSV to highlight in the markdown summary. Defaults to legacy immune/HOX/WWOX probes when present.",
    )
    parser.add_argument("--dev-manifest-tsv", required=True)
    parser.add_argument("--purity-tsv", required=True)
    parser.add_argument("--leukocyte-tsv", required=True)
    parser.add_argument("--cibersort-tsv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--min-within-class-n", type=int, default=20)
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


def derive_probe_order(matrix: pd.DataFrame, ann_path: Path, probes_tsv: str) -> list[str]:
    matrix_probes = [col for col in matrix.columns if col.startswith("cg")]
    matrix_probe_set = set(matrix_probes)
    if probes_tsv:
        ordered = [probe for probe in read_probe_list(Path(probes_tsv)) if probe in matrix_probe_set]
        if not ordered:
            raise SystemExit("None of the requested probes from --probes-tsv were found in the probe matrix.")
        return ordered

    ann = pd.read_csv(ann_path, sep="\t", dtype=str).fillna("")
    ann = ann.loc[ann["probe_id"].isin(matrix_probe_set), ["probe_id", "panel_rank"]].copy()
    if ann.empty:
        return matrix_probes
    ann["panel_rank_num"] = pd.to_numeric(ann["panel_rank"], errors="coerce")
    ann = ann.sort_values(["panel_rank_num", "probe_id"], na_position="last")
    ordered = ann["probe_id"].tolist()
    seen = set(ordered)
    ordered.extend([probe for probe in matrix_probes if probe not in seen])
    return ordered


def derive_focus_probes(probe_order: list[str], focus_probes_tsv: str) -> list[str]:
    if focus_probes_tsv:
        focus = [probe for probe in read_probe_list(Path(focus_probes_tsv)) if probe in set(probe_order)]
        if not focus:
            raise SystemExit("None of the requested probes from --focus-probes-tsv were found in the active probe order.")
        return focus
    return [probe for probe in LEGACY_FOCUS_PROBES if probe in set(probe_order)]


def _safe_corr(x: pd.Series, y: pd.Series, method: str) -> float:
    mask = x.notna() & y.notna()
    if mask.sum() < 8:
        return np.nan
    x2 = x.loc[mask]
    y2 = y.loc[mask]
    if x2.nunique() < 2 or y2.nunique() < 2:
        return np.nan
    return float(x2.corr(y2, method=method))


def _class_centered_corr(df: pd.DataFrame, x_col: str, y_col: str, class_col: str, method: str) -> tuple[float, int]:
    sub = df[[class_col, x_col, y_col]].dropna().copy()
    if sub.empty:
        return np.nan, 0
    sub[x_col] = sub[x_col] - sub.groupby(class_col)[x_col].transform("mean")
    sub[y_col] = sub[y_col] - sub.groupby(class_col)[y_col].transform("mean")
    mask = (sub.groupby(class_col)[x_col].transform("nunique") > 1) & (sub.groupby(class_col)[y_col].transform("nunique") > 1)
    sub = sub.loc[mask]
    if len(sub) < 8 or sub[x_col].nunique() < 2 or sub[y_col].nunique() < 2:
        return np.nan, int(len(sub))
    return float(sub[x_col].corr(sub[y_col], method=method)), int(len(sub))


def load_probe_metadata(path: Path, probe_order: list[str], focus_probe_set: set[str]) -> pd.DataFrame:
    ann = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    ann = ann.loc[ann["probe_id"].isin(probe_order), ["probe_id", "gene_list", "context_group", "panel_rank"]].copy()
    ann["gene_label"] = ann["gene_list"].replace("", "Intergenic")
    ann["probe_group"] = np.where(ann["probe_id"].isin(focus_probe_set), "focus_probe", "panel_probe")
    ann = ann.set_index("probe_id")
    missing = [probe for probe in probe_order if probe not in ann.index]
    if missing:
        filler = pd.DataFrame(
            {
                "gene_list": [""] * len(missing),
                "context_group": [""] * len(missing),
                "panel_rank": [""] * len(missing),
                "gene_label": ["Intergenic"] * len(missing),
                "probe_group": ["focus_probe" if probe in focus_probe_set else "panel_probe" for probe in missing],
            },
            index=missing,
        )
        ann = pd.concat([ann, filler], axis=0)
    ann.index.name = "probe_id"
    return ann.loc[probe_order].reset_index()


def load_covariates(dev_manifest_tsv: Path, purity_tsv: Path, leukocyte_tsv: Path, cibersort_tsv: Path) -> pd.DataFrame:
    dev = pd.read_csv(dev_manifest_tsv, sep="\t", dtype=str).fillna("")
    required_cols = ["sample_id", "project_id"]
    missing_required = [col for col in required_cols if col not in dev.columns]
    if missing_required:
        raise SystemExit(f"Dev manifest missing required columns: {missing_required}")
    dev = dev[required_cols].copy()
    dev = dev.drop_duplicates(subset=["sample_id"], keep="first").copy()
    dev["patient_id"] = dev["sample_id"].str.slice(0, 12)

    purity = pd.read_csv(purity_tsv, sep="\t")
    purity = purity.rename(columns={"sample": "purity_sample_barcode"})
    purity["patient_id"] = purity["purity_sample_barcode"].astype(str).str.slice(0, 12)
    purity["purity"] = pd.to_numeric(purity["purity"], errors="coerce")
    purity = purity.sort_values(["patient_id", "purity"], ascending=[True, False]).drop_duplicates("patient_id")
    purity = purity[["patient_id", "purity", "purity_sample_barcode"]]

    leuk = pd.read_csv(leukocyte_tsv, sep="\t", header=None, names=["cancer_type_short", "leuk_sample_barcode", "leukocyte_fraction"])
    leuk["patient_id"] = leuk["leuk_sample_barcode"].astype(str).str.slice(0, 12)
    leuk["leukocyte_fraction"] = pd.to_numeric(leuk["leukocyte_fraction"], errors="coerce")
    leuk = leuk.sort_values(["patient_id", "leukocyte_fraction"], ascending=[True, False]).drop_duplicates("patient_id")
    leuk = leuk[["patient_id", "cancer_type_short", "leuk_sample_barcode", "leukocyte_fraction"]]

    cib = pd.read_csv(cibersort_tsv, sep="\t")
    cib["patient_id"] = cib["SampleID"].astype(str).str.replace(".", "-", regex=False).str.slice(0, 12)
    for col in LYMPHOID_COLS + MYELOID_COLS:
        cib[col] = pd.to_numeric(cib[col], errors="coerce")
    cib["lymphoid_fraction"] = cib[LYMPHOID_COLS].sum(axis=1)
    cib["myeloid_fraction"] = cib[MYELOID_COLS].sum(axis=1)
    cib = cib.sort_values(["patient_id", "P.value"], ascending=[True, True]).drop_duplicates("patient_id")
    cib = cib[["patient_id", "SampleID", "P.value", "Correlation", "RMSE", "lymphoid_fraction", "myeloid_fraction"]]

    joined = dev.merge(purity, on="patient_id", how="left")
    joined = joined.merge(leuk, on="patient_id", how="left")
    joined = joined.merge(cib, on="patient_id", how="left")
    return joined


def classify_probe(summary_row: pd.Series) -> str:
    purity_abs = abs(float(summary_row["best_centered_purity_spearman"])) if pd.notna(summary_row["best_centered_purity_spearman"]) else 0.0
    immune_abs = abs(float(summary_row["best_centered_immune_spearman"])) if pd.notna(summary_row["best_centered_immune_spearman"]) else 0.0
    if purity_abs >= 0.25 and immune_abs < 0.20:
        return "purity_associated"
    if immune_abs >= 0.25 and purity_abs < 0.20:
        return "immune_associated"
    if purity_abs >= 0.20 and immune_abs >= 0.20:
        return "mixed"
    if pd.notna(summary_row["max_within_class_abs_spearman"]) and float(summary_row["max_within_class_abs_spearman"]) >= 0.5:
        return "cancer_specific_or_heterogeneous"
    return "weak_or_unclear"


def plot_heatmap(summary: pd.DataFrame, output_path: Path) -> None:
    heat = summary.set_index("probe_label")[[f"centered_spearman__{c}" for c in COVARIATE_ORDER]]
    heat.columns = ["Purity", "Leukocyte", "Lymphoid", "Myeloid"]
    fig_h = max(5.0, 0.33 * len(summary) + 1.5)
    plt.figure(figsize=(8.8, fig_h))
    sns.set_theme(style="white")
    ax = sns.heatmap(heat, cmap="coolwarm", center=0.0, vmin=-0.6, vmax=0.6, annot=True, fmt=".2f", linewidths=0.4)
    ax.set_title("Class-centered Spearman correlations")
    ax.set_xlabel("Covariate")
    ax.set_ylabel("Probe")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def plot_strength(summary: pd.DataFrame, output_path: Path) -> None:
    plot_df = summary[["probe_label", "best_centered_purity_spearman", "best_centered_immune_spearman"]].copy()
    plot_df["sort_key"] = plot_df[["best_centered_purity_spearman", "best_centered_immune_spearman"]].abs().max(axis=1)
    plot_df = plot_df.sort_values("sort_key", ascending=False)
    plot_df = plot_df.melt(id_vars="probe_label", var_name="association", value_name="correlation")
    plot_df["association"] = plot_df["association"].map(
        {
            "best_centered_purity_spearman": "Purity",
            "best_centered_immune_spearman": "Immune",
        }
    )
    fig_w = max(10.0, 0.3 * summary.shape[0] + 4.0)
    plt.figure(figsize=(fig_w, 5.8))
    sns.set_theme(style="whitegrid")
    ax = sns.barplot(data=plot_df, x="probe_label", y="correlation", hue="association", palette=["#6b8e23", "#1f77b4"])
    ax.axhline(0.0, color="#444444", linewidth=1)
    ax.set_title("Strongest class-centered association by probe")
    ax.set_xlabel("Probe")
    ax.set_ylabel("Spearman r")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close()


def write_summary_md(summary: pd.DataFrame, joined: pd.DataFrame, focus_probes: list[str], output_path: Path) -> None:
    lines: list[str] = []
    lines.append("# Immune / Purity Probe Association Summary")
    lines.append("")
    lines.append(
        f"Joined cohort size: `{len(joined)}` TCGA-dev samples across `{joined['project_id'].nunique()}` projects. Covariate availability: purity `{joined['purity'].notna().sum()}`, leukocyte `{joined['leukocyte_fraction'].notna().sum()}`, CIBERSORT `{joined['lymphoid_fraction'].notna().sum()}`."
    )
    lines.append("")
    lines.append("## Call Counts")
    lines.append("")
    for call, n in summary["primary_call"].value_counts().sort_index().items():
        lines.append(f"- `{call}`: `{int(n)}`")
    lines.append("")
    lines.append("## Main Calls")
    lines.append("")
    focus_df = summary.loc[summary["probe_id"].isin(focus_probes)].copy()
    if focus_df.empty:
        focus_df = summary.sort_values(
            ["best_centered_immune_abs", "max_within_class_abs_spearman"],
            ascending=[False, False],
        ).head(min(10, len(summary)))
    for row in focus_df.itertuples(index=False):
        lines.append(
            f"- `{row.probe_id}` (`{row.gene_label}`): `{row.primary_call}`. Best immune-centered signal: `{row.best_centered_immune_covariate}` (`r = {row.best_centered_immune_spearman:.3f}`); purity-centered `r = {row.best_centered_purity_spearman:.3f}`; strongest within-class hit: `{row.max_within_class_project_id}` / `{row.max_within_class_covariate}` (`r = {row.max_within_class_spearman:.3f}`)."
        )
    lines.append("")
    lines.append("## Strongest Immune-linked Candidates")
    lines.append("")
    immune_sorted = summary.sort_values("best_centered_immune_abs", ascending=False)
    for row in immune_sorted.head(min(10, len(immune_sorted))).itertuples(index=False):
        lines.append(
            f"- `{row.probe_id}` (`{row.gene_label}`): strongest with `{row.best_centered_immune_covariate}` at class-centered Spearman `r = {row.best_centered_immune_spearman:.3f}`."
        )
    lines.append("")
    lines.append("## Strongest Purity-linked Candidates")
    lines.append("")
    purity_sorted = summary.reindex(summary["best_centered_purity_spearman"].abs().sort_values(ascending=False).index)
    for row in purity_sorted.head(min(10, len(purity_sorted))).itertuples(index=False):
        lines.append(
            f"- `{row.probe_id}` (`{row.gene_label}`): class-centered purity Spearman `r = {row.best_centered_purity_spearman:.3f}`."
        )
    lines.append("")
    lines.append("## Cancer-specific Signals")
    lines.append("")
    hetero = summary.loc[summary["primary_call"] == "cancer_specific_or_heterogeneous"].sort_values(
        "max_within_class_abs_spearman", ascending=False
    )
    if hetero.empty:
        lines.append("- No probe met the current threshold for strong cancer-specific within-class association.")
    else:
        for row in hetero.itertuples(index=False):
            lines.append(
                f"- `{row.probe_id}` (`{row.gene_label}`): strongest within `{row.max_within_class_project_id}` against `{row.max_within_class_covariate}` at Spearman `r = {row.max_within_class_spearman:.3f}`."
            )
    lines.append("")
    lines.append("## Output Files")
    lines.append("")
    lines.append("- `sample_joined_probe_covariates.tsv`")
    lines.append("- `probe_covariate_correlations.tsv`")
    lines.append("- `within_class_probe_covariate_correlations.tsv`")
    lines.append("- `probe_association_summary.tsv`")
    lines.append("- `figures/probe_covariate_centered_spearman_heatmap.png`")
    lines.append("- `figures/probe_purity_vs_immune_strength.png`")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(parents=True, exist_ok=True)

    matrix = pd.read_csv(args.probe_matrix_tsv, sep="\t")
    probe_order = derive_probe_order(matrix, Path(args.panel_annotations_tsv), args.probes_tsv)
    focus_probes = derive_focus_probes(probe_order, args.focus_probes_tsv)
    probe_meta = load_probe_metadata(Path(args.panel_annotations_tsv), probe_order, set(focus_probes))
    joined_cov = load_covariates(
        Path(args.dev_manifest_tsv),
        Path(args.purity_tsv),
        Path(args.leukocyte_tsv),
        Path(args.cibersort_tsv),
    )
    joined = matrix.merge(joined_cov, on=["sample_id", "project_id"], how="left")

    long_df = joined.melt(
        id_vars=["sample_id", "project_id", "patient_id", "purity", "leukocyte_fraction", "lymphoid_fraction", "myeloid_fraction"],
        value_vars=probe_order,
        var_name="probe_id",
        value_name="beta_value",
    )
    long_df = long_df.merge(probe_meta, on="probe_id", how="left")
    long_df["probe_label"] = long_df["probe_id"] + " | " + long_df["gene_label"]

    corr_rows: list[dict[str, object]] = []
    within_rows: list[dict[str, object]] = []

    for probe_id in probe_order:
        probe_df = long_df.loc[long_df["probe_id"] == probe_id].copy()
        for cov in COVARIATE_ORDER:
            pearson = _safe_corr(probe_df["beta_value"], probe_df[cov], "pearson")
            spearman = _safe_corr(probe_df["beta_value"], probe_df[cov], "spearman")
            centered_pearson, centered_n = _class_centered_corr(probe_df, "beta_value", cov, "project_id", "pearson")
            centered_spearman, _ = _class_centered_corr(probe_df, "beta_value", cov, "project_id", "spearman")
            corr_rows.append(
                {
                    "probe_id": probe_id,
                    "covariate": cov,
                    "n_samples": int(probe_df[["beta_value", cov]].dropna().shape[0]),
                    "pearson": pearson,
                    "spearman": spearman,
                    "centered_pearson": centered_pearson,
                    "centered_spearman": centered_spearman,
                    "centered_n": centered_n,
                }
            )

            for project_id, cls_df in probe_df.groupby("project_id"):
                n = cls_df[["beta_value", cov]].dropna().shape[0]
                if n < args.min_within_class_n:
                    continue
                within_rows.append(
                    {
                        "probe_id": probe_id,
                        "project_id": project_id,
                        "covariate": cov,
                        "n_samples": int(n),
                        "pearson": _safe_corr(cls_df["beta_value"], cls_df[cov], "pearson"),
                        "spearman": _safe_corr(cls_df["beta_value"], cls_df[cov], "spearman"),
                    }
                )

    corr_df = pd.DataFrame(corr_rows)
    within_df = pd.DataFrame(within_rows)

    summary_rows: list[dict[str, object]] = []
    probe_meta_lookup = probe_meta.set_index("probe_id")
    for probe_id in probe_order:
        probe_corr = corr_df.loc[corr_df["probe_id"] == probe_id].set_index("covariate")
        probe_within = within_df.loc[within_df["probe_id"] == probe_id].copy()
        immune_centered = probe_corr.loc[["leukocyte_fraction", "lymphoid_fraction", "myeloid_fraction"], "centered_spearman"]
        immune_abs = immune_centered.abs()
        if immune_abs.notna().any():
            best_immune_cov = str(immune_abs.idxmax())
            best_immune_val = float(immune_centered.loc[best_immune_cov])
        else:
            best_immune_cov = "none"
            best_immune_val = np.nan
        purity_raw = probe_corr.loc["purity", "centered_spearman"]
        purity_val = float(purity_raw) if pd.notna(purity_raw) else np.nan
        if not probe_within.empty and probe_within["spearman"].notna().any():
            probe_within["abs_spearman"] = probe_within["spearman"].abs()
            top_within = probe_within.sort_values("abs_spearman", ascending=False).iloc[0]
            max_within_project = str(top_within["project_id"])
            max_within_cov = str(top_within["covariate"])
            max_within_val = float(top_within["spearman"])
            max_within_abs = float(top_within["abs_spearman"])
        else:
            max_within_project = "none"
            max_within_cov = "none"
            max_within_val = np.nan
            max_within_abs = np.nan
        row = {
            "probe_id": probe_id,
            "gene_label": probe_meta_lookup.at[probe_id, "gene_label"],
            "probe_group": probe_meta_lookup.at[probe_id, "probe_group"],
            "probe_label": f"{probe_id} | {probe_meta_lookup.at[probe_id, 'gene_label']}",
            "best_centered_purity_spearman": purity_val,
            "best_centered_immune_covariate": best_immune_cov,
            "best_centered_immune_spearman": best_immune_val,
            "best_centered_immune_abs": abs(best_immune_val),
            "max_within_class_project_id": max_within_project,
            "max_within_class_covariate": max_within_cov,
            "max_within_class_spearman": max_within_val,
            "max_within_class_abs_spearman": max_within_abs,
        }
        for cov in COVARIATE_ORDER:
            row[f"pearson__{cov}"] = float(probe_corr.loc[cov, "pearson"]) if pd.notna(probe_corr.loc[cov, "pearson"]) else np.nan
            row[f"spearman__{cov}"] = float(probe_corr.loc[cov, "spearman"]) if pd.notna(probe_corr.loc[cov, "spearman"]) else np.nan
            row[f"centered_pearson__{cov}"] = float(probe_corr.loc[cov, "centered_pearson"]) if pd.notna(probe_corr.loc[cov, "centered_pearson"]) else np.nan
            row[f"centered_spearman__{cov}"] = float(probe_corr.loc[cov, "centered_spearman"]) if pd.notna(probe_corr.loc[cov, "centered_spearman"]) else np.nan
        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows)
    summary["primary_call"] = summary.apply(classify_probe, axis=1)

    joined.to_csv(out_dir / "sample_joined_probe_covariates.tsv", sep="\t", index=False)
    corr_df.to_csv(out_dir / "probe_covariate_correlations.tsv", sep="\t", index=False)
    within_df.to_csv(out_dir / "within_class_probe_covariate_correlations.tsv", sep="\t", index=False)
    summary.to_csv(out_dir / "probe_association_summary.tsv", sep="\t", index=False)

    key = {
        "n_samples_joined": int(len(joined)),
        "n_projects_joined": int(joined["project_id"].nunique()),
        "n_with_purity": int(joined["purity"].notna().sum()),
        "n_with_leukocyte_fraction": int(joined["leukocyte_fraction"].notna().sum()),
        "n_with_cibersort": int(joined["lymphoid_fraction"].notna().sum()),
        "probe_order": probe_order,
        "focus_probes": focus_probes,
        "probe_calls": summary[["probe_id", "gene_label", "primary_call", "best_centered_immune_covariate", "best_centered_immune_spearman", "best_centered_purity_spearman"]].to_dict("records"),
    }
    (out_dir / "immune_purity_key_findings.json").write_text(json.dumps(key, indent=2) + "\n", encoding="utf-8")

    plot_heatmap(summary, out_dir / "figures" / "probe_covariate_centered_spearman_heatmap.png")
    plot_strength(summary, out_dir / "figures" / "probe_purity_vs_immune_strength.png")
    write_summary_md(summary, joined, focus_probes, out_dir / "immune_purity_summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
