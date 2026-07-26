#!/usr/bin/env python3
"""Audit UCS failure modes in held-out TCGA and external primary validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist


def find_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    for col in candidates:
        if col in df.columns:
            return col
    return None


def load_matrix(path: Path) -> tuple[pd.DataFrame, str, list[str]]:
    df = pd.read_csv(path, sep="\t", low_memory=False)
    label_col = find_col(
        df,
        ["project_id", "tcga_project", "tcga_label", "true_class", "class", "label"],
    )
    if label_col is None:
        raise ValueError(f"No label column found in {path}")
    probes = [col for col in df.columns if col.startswith("cg")]
    if not probes:
        raise ValueError(f"No probe columns found in {path}")
    return df, label_col, probes


def class_dispersion(path: Path, dataset: str) -> pd.DataFrame:
    df, label_col, probes = load_matrix(path)
    rows = []
    for cls, group in df.groupby(label_col):
        if len(group) < 2:
            continue
        x = group[probes].astype(float).to_numpy()
        centroid = np.nanmean(x, axis=0)
        centroid_dist = np.sqrt(np.nansum((x - centroid) ** 2, axis=1))
        rows.append(
            {
                "dataset": dataset,
                "class": cls,
                "n": int(len(group)),
                "median_centroid_distance": float(np.nanmedian(centroid_dist)),
                "mean_centroid_distance": float(np.nanmean(centroid_dist)),
                "pairwise_median_distance": float(np.nanmedian(pdist(x, metric="euclidean"))),
            }
        )
    out = pd.DataFrame(rows)
    out["rank_median_distance_desc"] = (
        out["median_centroid_distance"].rank(ascending=False, method="min").astype(int)
    )
    return out.sort_values(["dataset", "rank_median_distance_desc"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--failure-dir",
        default="outputs/paper_protocolb_v1/k_030/failure_mode_analysis_current",
    )
    parser.add_argument(
        "--manifest-dir",
        default="outputs/paper_protocolb_v1/k_030/manifest",
    )
    parser.add_argument(
        "--tcga-matrix-dir",
        default="outputs/paper_protocolb_v1/k_030/biological_interpretation_current/tcga_tumor_panel_matrices_current",
    )
    parser.add_argument(
        "--external-matrix",
        default="outputs/paper_protocolb_v1/k_030/external_main_beta_sanitized_current/inputs/external_panel_matrix_main_missing_le5_beta_sanitized.tsv",
    )
    parser.add_argument("--external-master", default="resources/external_samples_master.tsv")
    parser.add_argument("--external-inventory", default="resources/external_data_inventory.tsv")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    failure_dir = Path(args.failure_dir)
    sample_summary = pd.read_csv(failure_dir / "failure_mode_sample_summary.tsv", sep="\t")
    class_summary = pd.read_csv(failure_dir / "failure_mode_class_summary.tsv", sep="\t")

    ucs_class_summary = class_summary[class_summary["class"].eq("TCGA-UCS")].copy()
    ucs_class_summary.to_csv(out_dir / "ucs_class_summary.tsv", sep="\t", index=False)

    rows = []
    for dataset in ["lockbox", "external_main"]:
        subset = sample_summary[
            sample_summary["dataset"].eq(dataset)
            & sample_summary["true_class"].eq("TCGA-UCS")
        ]
        counts = subset["y_pred"].value_counts()
        for pred, count in counts.items():
            rows.append(
                {
                    "dataset": dataset,
                    "true_class": "TCGA-UCS",
                    "predicted_class": pred,
                    "n": int(count),
                    "fraction_of_ucs": float(count / len(subset)) if len(subset) else np.nan,
                }
            )
    pd.DataFrame(rows).to_csv(out_dir / "ucs_prediction_destinations.tsv", sep="\t", index=False)

    manifest_rows = []
    for split in ["dev", "lockbox"]:
        manifest = pd.read_csv(Path(args.manifest_dir) / f"paper_protocolb_k030_{split}.tsv", sep="\t")
        label_col = find_col(manifest, ["project_id", "tcga_project", "tcga_label", "class"])
        if label_col is None:
            raise ValueError(f"No label column found in {split} manifest")
        manifest_rows.append(
            {
                "split": split,
                "ucs_n": int(manifest[label_col].eq("TCGA-UCS").sum()),
                "total_n": int(len(manifest)),
            }
        )
    pd.DataFrame(manifest_rows).to_csv(out_dir / "ucs_tcga_sample_support.tsv", sep="\t", index=False)

    dispersion_parts = []
    tcga_matrix_dir = Path(args.tcga_matrix_dir)
    for dataset, path in [
        ("tcga_dev", tcga_matrix_dir / "tcga_dev_panel_matrix.tsv"),
        ("tcga_lockbox", tcga_matrix_dir / "tcga_lockbox_panel_matrix.tsv"),
        ("external_main", Path(args.external_matrix)),
    ]:
        dispersion_parts.append(class_dispersion(path, dataset))
    dispersion = pd.concat(dispersion_parts, ignore_index=True)
    dispersion.to_csv(out_dir / "class_30cpg_dispersion.tsv", sep="\t", index=False)
    dispersion[dispersion["class"].eq("TCGA-UCS")].to_csv(
        out_dir / "ucs_30cpg_dispersion.tsv", sep="\t", index=False
    )

    master = pd.read_csv(args.external_master, sep="\t", low_memory=False)
    inventory = pd.read_csv(args.external_inventory, sep="\t", low_memory=False)
    external_matrix, _, probes = load_matrix(Path(args.external_matrix))

    ucs_external = master[
        master["gse_id"].eq("GSE136790")
        & master["tcga_label"].eq("TCGA-UCS")
        & master["target_bucket"].eq("main")
        & master["include"].eq(1)
    ][["gse_id", "sample_id", "tcga_label", "target_bucket", "include", "signal_text"]].copy()

    inv_cols = [
        "gse_id",
        "sample_id",
        "mapped_column_index",
        "mapped_column_name",
        "protocolb_panel_extraction_status",
        "protocolb_panel_value_source",
        "protocolb_panel_n_found",
        "protocolb_panel_n_missing",
    ]
    pred_cols = ["sample_id", "y_pred", "is_correct", "n_missing_features"]
    matrix_subset = external_matrix[
        external_matrix["gse_id"].eq("GSE136790")
        & external_matrix["tcga_label"].eq("TCGA-UCS")
    ][["sample_id"] + probes].copy()
    matrix_subset["mean_30cpg"] = matrix_subset[probes].mean(axis=1, skipna=True)
    matrix_subset["max_30cpg"] = matrix_subset[probes].max(axis=1, skipna=True)
    matrix_subset["n_zeroish_30cpg"] = (matrix_subset[probes] < 1e-6).sum(axis=1)

    external_audit = (
        ucs_external.merge(inventory[inv_cols], on=["gse_id", "sample_id"], how="left")
        .merge(
            sample_summary[
                sample_summary["dataset"].eq("external_main")
                & sample_summary["true_class"].eq("TCGA-UCS")
            ][pred_cols],
            on="sample_id",
            how="left",
        )
        .merge(matrix_subset[["sample_id", "mean_30cpg", "max_30cpg", "n_zeroish_30cpg"]], on="sample_id", how="left")
    )
    external_audit["mapped_value_kind"] = np.where(
        external_audit["mapped_column_name"].str.contains("Detection_Pval", na=False),
        "Detection_Pval",
        np.where(external_audit["mapped_column_name"].str.contains("_Beta", na=False), "Beta", "Other"),
    )
    external_audit.to_csv(out_dir / "gse136790_ucs_mapping_prediction_audit.tsv", sep="\t", index=False)

    mapping_summary = (
        external_audit.groupby(["mapped_value_kind", "y_pred"], dropna=False)
        .agg(
            n=("sample_id", "size"),
            n_correct=("is_correct", "sum"),
            mean_30cpg=("mean_30cpg", "mean"),
            median_30cpg=("mean_30cpg", "median"),
            median_zeroish_30cpg=("n_zeroish_30cpg", "median"),
        )
        .reset_index()
    )
    mapping_summary.to_csv(out_dir / "gse136790_ucs_mapping_prediction_summary.tsv", sep="\t", index=False)

    summary = {
        "ucs_lockbox": {
            "n": int(ucs_class_summary.loc[ucs_class_summary["dataset"].eq("lockbox"), "n_samples"].iloc[0]),
            "n_correct": int(ucs_class_summary.loc[ucs_class_summary["dataset"].eq("lockbox"), "n_correct"].iloc[0]),
            "recall": float(ucs_class_summary.loc[ucs_class_summary["dataset"].eq("lockbox"), "recall"].iloc[0]),
        },
        "ucs_external": {
            "n": int(ucs_class_summary.loc[ucs_class_summary["dataset"].eq("external_main"), "n_samples"].iloc[0]),
            "n_correct": int(ucs_class_summary.loc[ucs_class_summary["dataset"].eq("external_main"), "n_correct"].iloc[0]),
            "recall": float(ucs_class_summary.loc[ucs_class_summary["dataset"].eq("external_main"), "recall"].iloc[0]),
        },
        "gse136790_main_ucs_mapping_kind_counts": external_audit["mapped_value_kind"].value_counts().to_dict(),
        "gse136790_detection_pval_predictions": external_audit[
            external_audit["mapped_value_kind"].eq("Detection_Pval")
        ]["y_pred"].value_counts().to_dict(),
    }
    (out_dir / "ucs_failure_audit_summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
