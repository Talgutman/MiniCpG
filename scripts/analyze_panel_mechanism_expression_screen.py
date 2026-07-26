#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd


TUMOR_GROUP = "primary_tumor"
NORMAL_GROUP = "normal"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Screen panel-linked genes for methylation-expression support and RNA tumor-normal shifts. "
            "This is a targeted all-panel-gene scan, intended to run on the cluster expression files."
        )
    )
    parser.add_argument("--probe-biology-tsv", required=True)
    parser.add_argument("--expression-metadata-tsv", required=True)
    parser.add_argument("--tcga-dev-matrix-tsv", required=True)
    parser.add_argument("--tcga-lockbox-matrix-tsv", required=True)
    parser.add_argument("--tumor-normal-dev-tsv", required=True)
    parser.add_argument("--tumor-normal-lockbox-tsv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--expression-field", default="tpm_unstranded")
    parser.add_argument("--min-project-samples", type=int, default=20)
    parser.add_argument("--min-expression-normal-samples", type=int, default=5)
    parser.add_argument("--focus-probes", default="")
    parser.add_argument("--max-expression-files", type=int, default=0)
    return parser.parse_args()


def split_semicolon(value: object) -> list[str]:
    if not isinstance(value, str):
        return []
    out: list[str] = []
    for item in value.split(";"):
        item = item.strip()
        if not item or item.lower() == "nan" or item == "__NO_GENE__":
            continue
        if item not in out:
            out.append(item)
    return out


def parse_distances(value: object) -> list[float]:
    vals: list[float] = []
    if not isinstance(value, str):
        return vals
    for item in value.split(";"):
        item = item.strip()
        if not item:
            continue
        try:
            vals.append(float(item))
        except ValueError:
            continue
    return vals


def spearman(x: pd.Series, y: pd.Series) -> float:
    tmp = pd.DataFrame({"x": x, "y": y}).replace([np.inf, -np.inf], np.nan).dropna()
    if len(tmp) < 3:
        return float("nan")
    if tmp["x"].nunique() < 2 or tmp["y"].nunique() < 2:
        return float("nan")
    return float(tmp["x"].rank(method="average").corr(tmp["y"].rank(method="average"), method="pearson"))


def strength_consistent_direction(delta: float) -> int:
    if delta > 0:
        return 1
    if delta < 0:
        return -1
    return 0


def build_probe_gene_pairs(probe_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for row in probe_df.sort_values("panel_rank").itertuples(index=False):
        genes = split_semicolon(getattr(row, "gene_list", ""))
        distances = parse_distances(getattr(row, "distToTSS", ""))
        min_abs_tss = min((abs(x) for x in distances), default=float("nan"))
        promoter_like = bool(math.isfinite(min_abs_tss) and min_abs_tss <= 1500)
        cgi_pos = str(getattr(row, "CGIposition", "") or "")
        context_group = str(getattr(row, "context_group", "") or "")
        for gene in genes:
            rows.append(
                {
                    "probe_id": getattr(row, "probe_id"),
                    "panel_rank": int(getattr(row, "panel_rank")),
                    "gene": gene,
                    "gene_list": getattr(row, "gene_list", ""),
                    "context_group": context_group,
                    "CGIposition": cgi_pos,
                    "min_abs_dist_to_tss": min_abs_tss,
                    "promoter_like": promoter_like,
                    "tumor_shift_strength": getattr(row, "tumor_shift_strength", ""),
                    "tumor_shift_breadth": getattr(row, "tumor_shift_breadth", ""),
                    "lineage_strength": getattr(row, "lineage_strength", ""),
                    "composition_strength": getattr(row, "composition_strength", ""),
                    "final_biological_label": getattr(row, "final_biological_label", ""),
                    "stability_selection_rate": getattr(row, "stability_selection_rate", np.nan),
                    "combined_gene_theme": getattr(row, "combined_gene_theme", ""),
                    "gene_literature_note": getattr(row, "gene_literature_note", ""),
                }
            )
    return pd.DataFrame(rows)


def target_genes_from_pairs(pairs: pd.DataFrame) -> set[str]:
    return {str(g) for g in pairs["gene"].dropna().unique() if str(g)}


def load_target_expression(metadata: pd.DataFrame, genes: set[str], expression_field: str, max_files: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    use = metadata.loc[metadata["sample_group"].isin([TUMOR_GROUP, NORMAL_GROUP])].copy()
    use = use.loc[use["expression_file_path"].astype(str).str.len() > 0].copy()
    if max_files:
        use = use.head(max_files).copy()

    for idx, row in enumerate(use.itertuples(index=False), start=1):
        path = Path(str(row.expression_file_path))
        if not path.exists():
            continue
        try:
            with path.open("r", encoding="utf-8") as handle:
                header: list[str] | None = None
                gene_name_idx = -1
                field_idx = -1
                gene_id_idx = -1
                gene_type_idx = -1
                for line in handle:
                    if not line.strip():
                        continue
                    if line.startswith("#"):
                        continue
                    parts = line.rstrip("\n").split("\t")
                    if header is None:
                        header = parts
                        gene_name_idx = header.index("gene_name")
                        field_idx = header.index(expression_field)
                        gene_id_idx = header.index("gene_id") if "gene_id" in header else -1
                        gene_type_idx = header.index("gene_type") if "gene_type" in header else -1
                        continue
                    gene_name = parts[gene_name_idx] if gene_name_idx < len(parts) else ""
                    if gene_name not in genes:
                        continue
                    try:
                        tpm = float(parts[field_idx])
                    except (ValueError, IndexError):
                        continue
                    rows.append(
                        {
                            "file_uuid": row.file_uuid,
                            "case_submitter_id": row.case_submitter_id,
                            "project_id": row.project_id,
                            "sample_submitter_id": row.sample_submitter_id,
                            "sample_type": row.sample_type,
                            "sample_type_code": row.sample_type_code,
                            "sample_group": row.sample_group,
                            "gene": gene_name,
                            "gene_id": parts[gene_id_idx] if 0 <= gene_id_idx < len(parts) else "",
                            "gene_type": parts[gene_type_idx] if 0 <= gene_type_idx < len(parts) else "",
                            "tpm": tpm,
                            "log2_tpm_plus1": math.log2(tpm + 1.0),
                        }
                    )
        except OSError:
            continue
        if idx % 500 == 0:
            print(f"Scanned expression files: {idx}/{len(use)}", flush=True)
    return pd.DataFrame(rows)


def primary_file_priority(sample_submitter_id: str, file_uuid: str) -> tuple[int, str, str]:
    sid = str(sample_submitter_id or "")
    suffix = sid.split("-")[3] if len(sid.split("-")) > 3 else ""
    vial = suffix[2:3] if len(suffix) >= 3 else "Z"
    return (0 if suffix.startswith("01") else 1, vial, str(file_uuid))


def deduplicate_expression(expr: pd.DataFrame) -> pd.DataFrame:
    if expr.empty:
        return expr
    rows = []
    for _, sub in expr.groupby(["case_submitter_id", "project_id", "sample_group", "gene"], sort=False):
        sub = sub.copy()
        sub["_priority"] = [
            primary_file_priority(sid, fid)
            for sid, fid in zip(sub["sample_submitter_id"].astype(str), sub["file_uuid"].astype(str))
        ]
        sub = sub.sort_values("_priority")
        chosen = sub.iloc[0].drop(labels=["_priority"]).to_dict()
        chosen["n_files_for_case_gene_group"] = int(len(sub))
        rows.append(chosen)
    return pd.DataFrame(rows)


def load_methylation(dev_path: Path, lock_path: Path) -> pd.DataFrame:
    frames = []
    for split, path in [("dev", dev_path), ("lockbox", lock_path)]:
        df = pd.read_csv(path, sep="\t")
        df["split"] = split
        frames.append(df)
    meth = pd.concat(frames, ignore_index=True)
    meth["case_submitter_id"] = meth["sample_id"].astype(str).str.slice(0, 12)
    return meth.drop_duplicates(["case_submitter_id", "project_id"], keep="first")


def expression_tumor_normal_summary(expr: pd.DataFrame, min_normals: int) -> pd.DataFrame:
    columns = [
        "project_id",
        "gene",
        "n_expression_tumor",
        "n_expression_normal",
        "tumor_median_log2_tpm_plus1",
        "normal_median_log2_tpm_plus1",
        "delta_tumor_minus_normal_log2_tpm_plus1",
    ]
    rows: list[dict[str, object]] = []
    if expr.empty:
        return pd.DataFrame(rows, columns=columns)
    for (project, gene), sub in expr.groupby(["project_id", "gene"], sort=True):
        tumor = sub.loc[sub["sample_group"] == TUMOR_GROUP, "log2_tpm_plus1"].dropna()
        normal = sub.loc[sub["sample_group"] == NORMAL_GROUP, "log2_tpm_plus1"].dropna()
        if tumor.empty or len(normal) < min_normals:
            continue
        rows.append(
            {
                "project_id": project,
                "gene": gene,
                "n_expression_tumor": int(len(tumor)),
                "n_expression_normal": int(len(normal)),
                "tumor_median_log2_tpm_plus1": float(tumor.median()),
                "normal_median_log2_tpm_plus1": float(normal.median()),
                "delta_tumor_minus_normal_log2_tpm_plus1": float(tumor.median() - normal.median()),
            }
        )
    return pd.DataFrame(rows, columns=columns)


def summarize_methylation_shift(dev: pd.DataFrame, lock: pd.DataFrame) -> pd.DataFrame:
    merged = dev.merge(
        lock,
        on=["project_id", "probe_id"],
        suffixes=("_dev", "_lockbox"),
        how="inner",
    )
    merged["same_direction"] = (merged["delta_median_beta_dev"] * merged["delta_median_beta_lockbox"]) > 0
    merged["mean_delta_median_beta"] = (merged["delta_median_beta_dev"] + merged["delta_median_beta_lockbox"]) / 2.0
    merged["mean_abs_delta_median_beta"] = (
        merged["delta_median_beta_dev"].abs() + merged["delta_median_beta_lockbox"].abs()
    ) / 2.0
    return merged


def correlation_summaries(pairs: pd.DataFrame, meth: pd.DataFrame, expr: pd.DataFrame, min_project_samples: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    tumor_expr = expr.loc[expr["sample_group"] == TUMOR_GROUP].copy()
    merged = meth.merge(tumor_expr, on=["case_submitter_id", "project_id"], how="inner")
    pair_rows: list[dict[str, object]] = []
    project_rows: list[dict[str, object]] = []
    for pair in pairs.itertuples(index=False):
        if pair.probe_id not in merged.columns:
            continue
        sub = merged.loc[merged["gene"] == pair.gene].copy()
        if sub.empty:
            continue
        sub["beta"] = pd.to_numeric(sub[pair.probe_id], errors="coerce")
        raw_rho = spearman(sub["beta"], sub["log2_tpm_plus1"])
        sub["beta_centered"] = sub["beta"] - sub.groupby("project_id")["beta"].transform("median")
        sub["expr_centered"] = sub["log2_tpm_plus1"] - sub.groupby("project_id")["log2_tpm_plus1"].transform("median")
        centered_rho = spearman(sub["beta_centered"], sub["expr_centered"])
        project_rhos: list[float] = []
        n_project_ge = 0
        n_project_negative = 0
        n_project_abs_ge_02 = 0
        for project, psub in sub.groupby("project_id", sort=True):
            n = int(psub[["beta", "log2_tpm_plus1"]].dropna().shape[0])
            if n < min_project_samples:
                continue
            rho = spearman(psub["beta"], psub["log2_tpm_plus1"])
            if math.isfinite(rho):
                project_rhos.append(rho)
                n_project_ge += 1
                n_project_negative += int(rho < 0)
                n_project_abs_ge_02 += int(abs(rho) >= 0.20)
            project_rows.append(
                {
                    "probe_id": pair.probe_id,
                    "gene": pair.gene,
                    "project_id": project,
                    "n_matched": n,
                    "spearman_beta_expr": rho,
                }
            )
        pair_rows.append(
            {
                "probe_id": pair.probe_id,
                "gene": pair.gene,
                "n_matched_pan_tcga": int(sub[["beta", "log2_tpm_plus1"]].dropna().shape[0]),
                "n_projects_with_min_samples": n_project_ge,
                "pan_tcga_raw_spearman_beta_expr": raw_rho,
                "pan_tcga_project_centered_spearman_beta_expr": centered_rho,
                "median_within_project_spearman_beta_expr": float(np.nanmedian(project_rhos)) if project_rhos else float("nan"),
                "n_projects_negative_spearman": n_project_negative,
                "n_projects_abs_spearman_ge_0_20": n_project_abs_ge_02,
            }
        )
    return pd.DataFrame(pair_rows), pd.DataFrame(project_rows)


def mechanism_call(row: pd.Series) -> str:
    rho = row.get("pan_tcga_project_centered_spearman_beta_expr", np.nan)
    expr_delta = row.get("top_methylation_project_expr_delta", np.nan)
    meth_delta = row.get("top_methylation_project_mean_delta", np.nan)
    promoter_like = bool(row.get("promoter_like", False))
    if promoter_like and pd.notna(rho) and float(rho) <= -0.20:
        if pd.notna(expr_delta) and pd.notna(meth_delta):
            direction_ok = (float(meth_delta) > 0 and float(expr_delta) < 0) or (
                float(meth_delta) < 0 and float(expr_delta) > 0
            )
            if direction_ok:
                return "promoter_repression_supported"
        return "promoter_inverse_expression_supported"
    if pd.notna(rho) and abs(float(rho)) >= 0.20:
        return "expression_associated_noncanonical_or_marker"
    return "no_clear_expression_support"


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    probe_df = pd.read_csv(args.probe_biology_tsv, sep="\t")
    pairs = build_probe_gene_pairs(probe_df)
    pairs.to_csv(out_dir / "panel_probe_gene_pairs.tsv", sep="\t", index=False)
    genes = target_genes_from_pairs(pairs)

    metadata = pd.read_csv(args.expression_metadata_tsv, sep="\t")
    expr_raw = load_target_expression(metadata, genes, args.expression_field, args.max_expression_files)
    expr = deduplicate_expression(expr_raw)
    expr.to_csv(out_dir / "target_gene_expression_values.tsv", sep="\t", index=False)

    meth = load_methylation(Path(args.tcga_dev_matrix_tsv), Path(args.tcga_lockbox_matrix_tsv))
    corr_summary, project_corr = correlation_summaries(pairs, meth, expr, args.min_project_samples)
    corr_summary.to_csv(out_dir / "probe_gene_expression_correlation_summary.tsv", sep="\t", index=False)
    project_corr.to_csv(out_dir / "probe_gene_project_correlations.tsv", sep="\t", index=False)

    expr_tn = expression_tumor_normal_summary(expr, args.min_expression_normal_samples)
    expr_tn.to_csv(out_dir / "gene_expression_tumor_normal_summary.tsv", sep="\t", index=False)

    tn_dev = pd.read_csv(args.tumor_normal_dev_tsv, sep="\t")
    tn_lock = pd.read_csv(args.tumor_normal_lockbox_tsv, sep="\t")
    methyl_shift = summarize_methylation_shift(tn_dev, tn_lock)
    methyl_shift.to_csv(out_dir / "probe_project_replicated_methylation_shift.tsv", sep="\t", index=False)

    top_methyl = (
        methyl_shift.loc[methyl_shift["same_direction"]]
        .sort_values(["probe_id", "mean_abs_delta_median_beta"], ascending=[True, False])
        .drop_duplicates("probe_id")
    )
    top_methyl = top_methyl[
        ["probe_id", "project_id", "mean_delta_median_beta", "mean_abs_delta_median_beta"]
    ].rename(
        columns={
            "project_id": "top_methylation_project_id",
            "mean_delta_median_beta": "top_methylation_project_mean_delta",
            "mean_abs_delta_median_beta": "top_methylation_project_mean_abs_delta",
        }
    )
    expr_top = expr_tn.rename(columns={"project_id": "top_methylation_project_id"})
    screen = (
        pairs.merge(corr_summary, on=["probe_id", "gene"], how="left")
        .merge(top_methyl, on="probe_id", how="left")
        .merge(expr_top, on=["top_methylation_project_id", "gene"], how="left")
        .rename(columns={"delta_tumor_minus_normal_log2_tpm_plus1": "top_methylation_project_expr_delta"})
    )
    screen["mechanism_expression_call"] = screen.apply(mechanism_call, axis=1)
    screen = screen.sort_values(["panel_rank", "probe_id", "gene"])
    screen.to_csv(out_dir / "panel_probe_gene_mechanism_screen.tsv", sep="\t", index=False)

    focus = [x.strip() for x in args.focus_probes.split(",") if x.strip()]
    if focus:
        screen.loc[screen["probe_id"].isin(focus)].to_csv(
            out_dir / "candidate_probe_mechanism_screen.tsv", sep="\t", index=False
        )

    summary = {
        "n_panel_probes": int(probe_df["probe_id"].nunique()),
        "n_probe_gene_pairs": int(len(pairs)),
        "n_target_genes": int(len(genes)),
        "n_expression_values": int(len(expr)),
        "n_expression_tumor_values": int((expr["sample_group"] == TUMOR_GROUP).sum()) if not expr.empty else 0,
        "n_expression_normal_values": int((expr["sample_group"] == NORMAL_GROUP).sum()) if not expr.empty else 0,
        "n_mechanism_rows": int(len(screen)),
        "mechanism_call_counts": screen["mechanism_expression_call"].value_counts(dropna=False).to_dict(),
        "outputs": {
            "screen": str(out_dir / "panel_probe_gene_mechanism_screen.tsv"),
            "candidate_screen": str(out_dir / "candidate_probe_mechanism_screen.tsv") if focus else "",
            "expression_values": str(out_dir / "target_gene_expression_values.tsv"),
        },
    }
    with (out_dir / "mechanism_expression_screen_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    with (out_dir / "mechanism_expression_screen_summary.md").open("w", encoding="utf-8") as handle:
        handle.write("# Mechanism Expression Screen Summary\n\n")
        for key, value in summary.items():
            if key == "outputs":
                continue
            handle.write(f"- `{key}`: `{value}`\n")
        handle.write("\n## Outputs\n")
        for key, value in summary["outputs"].items():
            handle.write(f"- `{key}`: `{value}`\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
