#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


TUMOR_CUTOFFS = [0.20, 0.25, 0.30, 0.35, 0.40]
LINEAGE_TOP_PCTS = [10, 20, 30, 40]
COMPOSITION_TOP_PCTS = [10, 20, 30]
COMPOSITION_SCORE_CUTOFFS = [0.20, 0.30, 0.40]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run threshold sensitivity analyses for tumor-shift, lineage, and composition "
            "evidence classes in the locked MiniCpG panel."
        )
    )
    parser.add_argument("--probe-master-tsv", required=True)
    parser.add_argument("--broad-lineage-tsv", required=True)
    parser.add_argument("--broad-composition-tsv", required=True)
    parser.add_argument("--tumor-normal-dev-tsv", required=True)
    parser.add_argument("--tumor-normal-lockbox-tsv", required=True)
    parser.add_argument("--external-primary-summary-tsv", default="")
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def percentile_rank(value: float, background: pd.Series) -> float:
    if pd.isna(value):
        return np.nan
    clean = pd.to_numeric(background, errors="coerce").dropna().astype(float)
    if clean.empty:
        return np.nan
    return float((clean <= float(value)).mean())


def top_threshold(background: pd.Series, top_pct: int) -> float:
    clean = pd.to_numeric(background, errors="coerce").dropna().astype(float)
    if clean.empty:
        return np.nan
    return float(clean.quantile(1.0 - top_pct / 100.0))


def markdown_table(df: pd.DataFrame, float_digits: int = 3) -> str:
    if df.empty:
        return "_No rows._"
    headers = list(df.columns)
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for _, row in df.iterrows():
        values = []
        for col in headers:
            value = row[col]
            if pd.isna(value):
                values.append("")
            elif isinstance(value, (int, np.integer)):
                values.append(str(int(value)))
            elif isinstance(value, float):
                if float(value).is_integer():
                    values.append(str(int(value)))
                else:
                    values.append(f"{value:.{float_digits}f}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def semicolon(values: Iterable[str]) -> str:
    clean = [str(v) for v in values if str(v)]
    return ";".join(clean)


def aggregate_replicated_tumor_shift(dev: pd.DataFrame, lockbox: pd.DataFrame) -> pd.DataFrame:
    merged = dev.merge(
        lockbox,
        on=["project_id", "probe_id"],
        suffixes=("_dev", "_lockbox"),
        how="inner",
    )
    merged["delta_median_beta_dev"] = pd.to_numeric(merged["delta_median_beta_dev"], errors="coerce")
    merged["delta_median_beta_lockbox"] = pd.to_numeric(merged["delta_median_beta_lockbox"], errors="coerce")
    merged["same_direction"] = (merged["delta_median_beta_dev"] * merged["delta_median_beta_lockbox"]) > 0
    merged["replicated_mean_abs_delta"] = (
        merged["delta_median_beta_dev"].abs() + merged["delta_median_beta_lockbox"].abs()
    ) / 2.0
    return merged


def tumor_sensitivity(
    panel: pd.DataFrame,
    dev: pd.DataFrame,
    lockbox: pd.DataFrame,
    external: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    replicated = aggregate_replicated_tumor_shift(dev, lockbox)
    ext = external.copy() if not external.empty else pd.DataFrame(columns=["probe_id", "delta_median_beta"])
    if "delta_median_beta" in ext.columns:
        ext["external_abs_delta"] = pd.to_numeric(ext["delta_median_beta"], errors="coerce").abs()
    else:
        ext["external_abs_delta"] = np.nan

    rows = []
    summary_rows = []
    panel_probe_ids = panel["probe_id"].astype(str).tolist()
    for cutoff in TUMOR_CUTOFFS:
        for probe_id in panel_probe_ids:
            sub = replicated.loc[(replicated["probe_id"] == probe_id) & replicated["same_direction"]].copy()
            ext_sub = ext.loc[ext["probe_id"] == probe_id].copy()
            n_tcga = int((sub["replicated_mean_abs_delta"] >= cutoff).sum())
            max_tcga = float(sub["replicated_mean_abs_delta"].max()) if not sub.empty else 0.0
            max_external = float(ext_sub["external_abs_delta"].max()) if not ext_sub.empty else 0.0
            tcga_pass = max_tcga >= cutoff
            external_pass = max_external >= cutoff
            combined_pass = tcga_pass or external_pass
            if n_tcga >= 3:
                breadth = "broad"
            elif n_tcga >= 1 or external_pass:
                breadth = "focal"
            else:
                breadth = "limited"
            rows.append(
                {
                    "probe_id": probe_id,
                    "cutoff_abs_delta": cutoff,
                    "n_tcga_replicated_projects_ge_cutoff": n_tcga,
                    "max_tcga_replicated_abs_delta": max_tcga,
                    "max_external_primary_abs_delta": max_external,
                    "tcga_pass": bool(tcga_pass),
                    "external_pass": bool(external_pass),
                    "combined_pass": bool(combined_pass),
                    "support_breadth_at_cutoff": breadth,
                }
            )
        cutoff_rows = [row for row in rows if row["cutoff_abs_delta"] == cutoff]
        cutoff_df = pd.DataFrame(cutoff_rows)
        summary_rows.append(
            {
                "cutoff_abs_delta": cutoff,
                "n_combined_pass": int(cutoff_df["combined_pass"].sum()),
                "n_tcga_pass": int(cutoff_df["tcga_pass"].sum()),
                "n_external_pass": int(cutoff_df["external_pass"].sum()),
                "n_broad": int(((cutoff_df["combined_pass"]) & (cutoff_df["support_breadth_at_cutoff"] == "broad")).sum()),
                "n_focal": int(((cutoff_df["combined_pass"]) & (cutoff_df["support_breadth_at_cutoff"] == "focal")).sum()),
                "n_limited_or_no_pass": int((~cutoff_df["combined_pass"]).sum()),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(summary_rows)


def lineage_sensitivity(
    panel: pd.DataFrame,
    broad_lineage: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    broad = broad_lineage.copy()
    broad["dynamic_range"] = pd.to_numeric(broad["dynamic_range"], errors="coerce")
    broad["tumor_normal_order_spearman"] = pd.to_numeric(broad["tumor_normal_order_spearman"], errors="coerce")
    positive_order = broad.loc[broad["tumor_normal_order_spearman"] > 0, "tumor_normal_order_spearman"]

    panel_lineage = broad.loc[broad["probe_id"].isin(panel["probe_id"])].copy()
    panel_lineage["lineage_dynamic_percentile"] = panel_lineage["dynamic_range"].apply(
        lambda x: percentile_rank(float(x), broad["dynamic_range"])
    )
    panel_lineage["lineage_order_positive_percentile"] = panel_lineage["tumor_normal_order_spearman"].apply(
        lambda x: percentile_rank(float(x), positive_order) if pd.notna(x) and float(x) > 0 else 0.0
    )

    dynamic_thresholds = {pct: top_threshold(broad["dynamic_range"], pct) for pct in LINEAGE_TOP_PCTS}
    order_thresholds = {pct: top_threshold(positive_order, pct) for pct in LINEAGE_TOP_PCTS}

    calls = []
    summary = []
    for dyn_pct, dyn_threshold in dynamic_thresholds.items():
        for order_pct, order_threshold in order_thresholds.items():
            for row in panel_lineage.itertuples(index=False):
                dynamic_pass = float(row.dynamic_range) >= dyn_threshold
                order_pass = float(row.tumor_normal_order_spearman) >= order_threshold
                joint_pass = dynamic_pass and order_pass
                calls.append(
                    {
                        "probe_id": row.probe_id,
                        "dynamic_top_pct": dyn_pct,
                        "order_top_pct": order_pct,
                        "dynamic_threshold": dyn_threshold,
                        "order_threshold": order_threshold,
                        "dynamic_range": float(row.dynamic_range),
                        "tumor_normal_order_spearman": float(row.tumor_normal_order_spearman),
                        "dynamic_pass": bool(dynamic_pass),
                        "order_pass": bool(order_pass),
                        "joint_lineage_pass": bool(joint_pass),
                    }
                )
            combo = pd.DataFrame([c for c in calls if c["dynamic_top_pct"] == dyn_pct and c["order_top_pct"] == order_pct])
            summary.append(
                {
                    "dynamic_top_pct": dyn_pct,
                    "order_top_pct": order_pct,
                    "dynamic_threshold": dyn_threshold,
                    "order_threshold": order_threshold,
                    "n_joint_pass": int(combo["joint_lineage_pass"].sum()),
                    "n_dynamic_only": int(((combo["dynamic_pass"]) & (~combo["order_pass"])).sum()),
                    "n_order_only": int(((~combo["dynamic_pass"]) & (combo["order_pass"])).sum()),
                    "n_neither": int(((~combo["dynamic_pass"]) & (~combo["order_pass"])).sum()),
                    "joint_pass_probes": semicolon(combo.loc[combo["joint_lineage_pass"], "probe_id"].tolist()),
                }
            )

    calls_df = pd.DataFrame(calls)
    pass_fraction = calls_df.groupby("probe_id")["joint_lineage_pass"].mean().rename("lineage_grid_pass_fraction")
    panel_lineage = panel_lineage.merge(pass_fraction, on="probe_id", how="left")
    return calls_df, pd.DataFrame(summary), panel_lineage


def composition_sensitivity(
    panel: pd.DataFrame,
    broad_composition: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    broad = broad_composition.copy()
    broad["composition_score"] = pd.to_numeric(broad["composition_score"], errors="coerce")
    panel_comp = broad.loc[broad["probe_id"].isin(panel["probe_id"])].copy()
    panel_comp["composition_percentile"] = panel_comp["composition_score"].apply(
        lambda x: percentile_rank(float(x), broad["composition_score"])
    )
    percentile_thresholds = {pct: top_threshold(broad["composition_score"], pct) for pct in COMPOSITION_TOP_PCTS}

    calls = []
    summary = []
    for top_pct, pct_threshold in percentile_thresholds.items():
        for score_cutoff in COMPOSITION_SCORE_CUTOFFS:
            for row in panel_comp.itertuples(index=False):
                percentile_pass = float(row.composition_score) >= pct_threshold
                score_pass = float(row.composition_score) >= score_cutoff
                joint_pass = percentile_pass and score_pass
                calls.append(
                    {
                        "probe_id": row.probe_id,
                        "composition_top_pct": top_pct,
                        "composition_percentile_threshold": pct_threshold,
                        "composition_score_cutoff": score_cutoff,
                        "composition_score": float(row.composition_score),
                        "composition_percentile": float(row.composition_percentile),
                        "percentile_pass": bool(percentile_pass),
                        "score_pass": bool(score_pass),
                        "joint_composition_pass": bool(joint_pass),
                    }
                )
            combo = pd.DataFrame(
                [
                    c
                    for c in calls
                    if c["composition_top_pct"] == top_pct and c["composition_score_cutoff"] == score_cutoff
                ]
            )
            summary.append(
                {
                    "composition_top_pct": top_pct,
                    "composition_percentile_threshold": pct_threshold,
                    "composition_score_cutoff": score_cutoff,
                    "n_joint_pass": int(combo["joint_composition_pass"].sum()),
                    "joint_pass_probes": semicolon(combo.loc[combo["joint_composition_pass"], "probe_id"].tolist()),
                }
            )
    calls_df = pd.DataFrame(calls)
    pass_fraction = calls_df.groupby("probe_id")["joint_composition_pass"].mean().rename(
        "composition_grid_pass_fraction"
    )
    panel_comp = panel_comp.merge(pass_fraction, on="probe_id", how="left")
    return calls_df, pd.DataFrame(summary), panel_comp


def add_groups(evidence: pd.DataFrame) -> pd.DataFrame:
    out = evidence.copy()
    delta = pd.to_numeric(out["combined_max_tumor_abs_delta"], errors="coerce").fillna(0.0)
    n_tcga_030 = pd.to_numeric(out["n_tcga_replicated_projects_ge_0_30_sensitivity"], errors="coerce").fillna(0)
    tumor_group = []
    for value, n_tcga in zip(delta, n_tcga_030):
        if value >= 0.30 and n_tcga >= 3:
            tumor_group.append("large_broad_shift")
        elif value >= 0.30:
            tumor_group.append("large_focal_shift")
        elif value >= 0.20:
            tumor_group.append("moderate_shift")
        else:
            tumor_group.append("below_moderate_shift")
    out["tumor_shift_sensitivity_group"] = tumor_group

    dyn = pd.to_numeric(out["lineage_dynamic_percentile"], errors="coerce").fillna(0.0)
    order = pd.to_numeric(out["lineage_order_positive_percentile"], errors="coerce").fillna(0.0)
    lineage_group = []
    for d, o in zip(dyn, order):
        if d >= 0.80 and o >= 0.80:
            lineage_group.append("robust_top20_joint")
        elif d >= 0.70 and o >= 0.70:
            lineage_group.append("supportive_top30_joint")
        elif d >= 0.60 and o >= 0.60:
            lineage_group.append("permissive_top40_joint")
        elif d >= 0.80 or o >= 0.80:
            lineage_group.append("one_axis_upper_tail")
        else:
            lineage_group.append("not_lineage_by_grid")
    out["lineage_sensitivity_group"] = lineage_group

    comp_score = pd.to_numeric(out["composition_score"], errors="coerce").fillna(0.0)
    comp_pct = pd.to_numeric(out["composition_percentile"], errors="coerce").fillna(0.0)
    comp_group = []
    for score, pct in zip(comp_score, comp_pct):
        if score >= 0.40 and pct >= 0.90:
            comp_group.append("robust_composition")
        elif score >= 0.30 and pct >= 0.80:
            comp_group.append("moderate_composition")
        elif score >= 0.20 and pct >= 0.70:
            comp_group.append("weak_composition")
        else:
            comp_group.append("not_composition_by_grid")
    out["composition_sensitivity_group"] = comp_group
    return out


def write_report(
    out_path: Path,
    evidence: pd.DataFrame,
    tumor_summary: pd.DataFrame,
    lineage_summary: pd.DataFrame,
    comp_summary: pd.DataFrame,
) -> None:
    lines: list[str] = []
    lines.append("# Biological Threshold Sensitivity Analysis")
    lines.append("")
    lines.append("Purpose: test whether biological evidence conclusions survive reasonable threshold movement.")
    lines.append("")
    lines.append("## Principle")
    lines.append("")
    lines.append("- Thresholds are operational summaries, not biological truth.")
    lines.append("- Continuous evidence remains the primary result.")
    lines.append("- A categorical claim is safer when the same probe is retained across nearby threshold choices.")
    lines.append("")

    lines.append("## Tumor-Shift Sensitivity")
    lines.append("")
    lines.append(markdown_table(tumor_summary))
    lines.append("")
    tumor_groups = evidence["tumor_shift_sensitivity_group"].value_counts().sort_index().reset_index()
    tumor_groups.columns = ["tumor_shift_sensitivity_group", "n_probes"]
    lines.append("Tumor-shift group counts:")
    lines.append("")
    lines.append(markdown_table(tumor_groups, float_digits=0))
    lines.append("")

    lines.append("## Lineage Sensitivity")
    lines.append("")
    lineage_pivot = lineage_summary.pivot(
        index="dynamic_top_pct",
        columns="order_top_pct",
        values="n_joint_pass",
    ).reset_index()
    lineage_pivot.columns = [str(c) for c in lineage_pivot.columns]
    lines.append("Number of probes passing both dynamic-range and tumor-normal-order thresholds:")
    lines.append("")
    lines.append(markdown_table(lineage_pivot, float_digits=0))
    lines.append("")
    lineage_groups = evidence["lineage_sensitivity_group"].value_counts().sort_index().reset_index()
    lineage_groups.columns = ["lineage_sensitivity_group", "n_probes"]
    lines.append("Lineage sensitivity group counts:")
    lines.append("")
    lines.append(markdown_table(lineage_groups, float_digits=0))
    lines.append("")
    top_lineage = evidence.sort_values(
        ["lineage_grid_pass_fraction", "lineage_dynamic_percentile", "lineage_order_positive_percentile"],
        ascending=False,
    ).head(10)
    lines.append("Top lineage candidates by grid stability:")
    lines.append("")
    lines.append(
        markdown_table(
            top_lineage[
                [
                    "probe_id",
                    "gene_list",
                    "lineage_sensitivity_group",
                    "lineage_dynamic_range",
                    "lineage_order_spearman",
                    "lineage_dynamic_percentile",
                    "lineage_order_positive_percentile",
                    "lineage_grid_pass_fraction",
                ]
            ]
        )
    )
    lines.append("")

    lines.append("## Composition Sensitivity")
    lines.append("")
    comp_pivot = comp_summary.pivot(
        index="composition_top_pct",
        columns="composition_score_cutoff",
        values="n_joint_pass",
    ).reset_index()
    comp_pivot.columns = [str(c) for c in comp_pivot.columns]
    lines.append("Number of probes passing both composition percentile and score cutoffs:")
    lines.append("")
    lines.append(markdown_table(comp_pivot, float_digits=0))
    lines.append("")
    comp_groups = evidence["composition_sensitivity_group"].value_counts().sort_index().reset_index()
    comp_groups.columns = ["composition_sensitivity_group", "n_probes"]
    lines.append("Composition sensitivity group counts:")
    lines.append("")
    lines.append(markdown_table(comp_groups, float_digits=0))
    lines.append("")
    top_comp = evidence.sort_values(
        ["composition_grid_pass_fraction", "composition_score", "composition_percentile"],
        ascending=False,
    ).head(10)
    lines.append("Top composition candidates by grid stability:")
    lines.append("")
    lines.append(
        markdown_table(
            top_comp[
                [
                    "probe_id",
                    "gene_list",
                    "composition_sensitivity_group",
                    "composition_score",
                    "composition_percentile",
                    "composition_grid_pass_fraction",
                ]
            ]
        )
    )
    lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        "- Tumor-shift evidence is least sensitive because many panel probes have beta shifts above several tested cutoffs."
    )
    lines.append(
        "- Lineage evidence is sensitive to whether we require both axes. Probes with one strong axis but the other below the grid should be described as continuous/supportive evidence, not categorical lineage calls."
    )
    lines.append(
        "- Composition evidence should combine effect size and empirical rarity. A probe with a top-tail percentile but small correlation should not be overinterpreted."
    )
    lines.append("")
    lines.append("## Output Files")
    lines.append("")
    lines.append("- `continuous_biological_evidence_scores.tsv`")
    lines.append("- `tumor_shift_sensitivity_by_cutoff.tsv`")
    lines.append("- `tumor_shift_probe_calls_by_cutoff.tsv`")
    lines.append("- `lineage_sensitivity_grid.tsv`")
    lines.append("- `lineage_probe_calls_by_grid.tsv`")
    lines.append("- `composition_sensitivity_grid.tsv`")
    lines.append("- `composition_probe_calls_by_grid.tsv`")
    lines.append("- `biological_threshold_sensitivity_summary.json`")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    panel = pd.read_csv(args.probe_master_tsv, sep="\t")
    broad_lineage = pd.read_csv(args.broad_lineage_tsv, sep="\t")
    broad_composition = pd.read_csv(args.broad_composition_tsv, sep="\t")
    tumor_dev = pd.read_csv(args.tumor_normal_dev_tsv, sep="\t")
    tumor_lockbox = pd.read_csv(args.tumor_normal_lockbox_tsv, sep="\t")
    external = pd.read_csv(args.external_primary_summary_tsv, sep="\t") if args.external_primary_summary_tsv else pd.DataFrame()

    tumor_calls, tumor_summary = tumor_sensitivity(panel, tumor_dev, tumor_lockbox, external)
    lineage_calls, lineage_summary, panel_lineage = lineage_sensitivity(panel, broad_lineage)
    comp_calls, comp_summary, panel_comp = composition_sensitivity(panel, broad_composition)

    evidence_cols = [
        "probe_id",
        "panel_rank",
        "gene_list",
        "final_biological_label",
        "combined_max_tumor_abs_delta",
        "max_replicated_mean_abs_delta",
        "max_external_primary_abs_delta",
        "stability_selection_rate",
    ]
    evidence = panel[[c for c in evidence_cols if c in panel.columns]].copy()
    tumor_030 = tumor_calls.loc[tumor_calls["cutoff_abs_delta"] == 0.30][
        ["probe_id", "n_tcga_replicated_projects_ge_cutoff", "support_breadth_at_cutoff"]
    ].rename(
        columns={
            "n_tcga_replicated_projects_ge_cutoff": "n_tcga_replicated_projects_ge_0_30_sensitivity",
            "support_breadth_at_cutoff": "support_breadth_at_0_30_sensitivity",
        }
    )
    evidence = evidence.merge(tumor_030, on="probe_id", how="left")
    evidence = evidence.merge(
        panel_lineage[
            [
                "probe_id",
                "dynamic_range",
                "tumor_normal_order_spearman",
                "lineage_dynamic_percentile",
                "lineage_order_positive_percentile",
                "lineage_grid_pass_fraction",
            ]
        ].rename(
            columns={
                "dynamic_range": "lineage_dynamic_range",
                "tumor_normal_order_spearman": "lineage_order_spearman",
            }
        ),
        on="probe_id",
        how="left",
    )
    evidence = evidence.merge(
        panel_comp[
            [
                "probe_id",
                "best_centered_immune_abs",
                "best_centered_purity_abs",
                "composition_score",
                "composition_percentile",
                "composition_grid_pass_fraction",
            ]
        ],
        on="probe_id",
        how="left",
    )
    evidence = add_groups(evidence)

    tumor_calls.to_csv(out_dir / "tumor_shift_probe_calls_by_cutoff.tsv", sep="\t", index=False)
    tumor_summary.to_csv(out_dir / "tumor_shift_sensitivity_by_cutoff.tsv", sep="\t", index=False)
    lineage_calls.to_csv(out_dir / "lineage_probe_calls_by_grid.tsv", sep="\t", index=False)
    lineage_summary.to_csv(out_dir / "lineage_sensitivity_grid.tsv", sep="\t", index=False)
    comp_calls.to_csv(out_dir / "composition_probe_calls_by_grid.tsv", sep="\t", index=False)
    comp_summary.to_csv(out_dir / "composition_sensitivity_grid.tsv", sep="\t", index=False)
    evidence.to_csv(out_dir / "continuous_biological_evidence_scores.tsv", sep="\t", index=False)

    summary = {
        "n_probes": int(len(panel)),
        "tumor_cutoffs": TUMOR_CUTOFFS,
        "lineage_top_percentiles": LINEAGE_TOP_PCTS,
        "composition_top_percentiles": COMPOSITION_TOP_PCTS,
        "composition_score_cutoffs": COMPOSITION_SCORE_CUTOFFS,
        "tumor_shift_group_counts": {
            str(k): int(v) for k, v in evidence["tumor_shift_sensitivity_group"].value_counts().sort_index().items()
        },
        "lineage_group_counts": {
            str(k): int(v) for k, v in evidence["lineage_sensitivity_group"].value_counts().sort_index().items()
        },
        "composition_group_counts": {
            str(k): int(v) for k, v in evidence["composition_sensitivity_group"].value_counts().sort_index().items()
        },
    }
    (out_dir / "biological_threshold_sensitivity_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    write_report(
        out_dir / "biological_threshold_sensitivity_report.md",
        evidence,
        tumor_summary,
        lineage_summary,
        comp_summary,
    )
    print(f"Wrote biological threshold sensitivity outputs to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
