#!/usr/bin/env python3
"""Audit TCGA normal sample metadata and composition covariate availability."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd




def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--normal-inventory", required=True)
    p.add_argument("--normal-gdc-inventory", required=True)
    p.add_argument("--tumor-dev", required=True)
    p.add_argument("--tumor-lockbox", required=True)
    p.add_argument("--absolute", required=True)
    p.add_argument("--leukocyte", required=True)
    p.add_argument("--cibersort", required=True)
    p.add_argument("--quality", required=True)
    return p.parse_args()


def read_tsv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing input: {path}")
    return pd.read_csv(path, sep="\t", **kwargs)


def patient_from_barcode(value: object) -> str:
    text = str(value)
    return text[:12] if text and text != "nan" else ""


def sample16_from_barcode(value: object) -> str:
    text = str(value).replace(".", "-")
    return text[:16] if text and text != "nan" else ""


def sample15_from_barcode(value: object) -> str:
    text = str(value).replace(".", "-")
    return text[:15] if text and text != "nan" else ""


def load_tumor_split(dev_path: Path, lockbox_path: Path) -> pd.DataFrame:
    parts = []
    for split, path in [("development", dev_path), ("heldout_tcga", lockbox_path)]:
        df = read_tsv(path, usecols=["sample_id", "project_id"], dtype=str)
        df["patient_id"] = df["sample_id"].map(patient_from_barcode)
        df["tumor_split"] = split
        parts.append(df)
    tumor = pd.concat(parts, ignore_index=True)
    return tumor.drop_duplicates(["patient_id", "tumor_split", "project_id"])


def covariate_keys(args: argparse.Namespace) -> dict[str, set[str]]:
    keys: dict[str, set[str]] = {}
    abs_df = read_tsv(Path(args.absolute), dtype=str)
    keys["absolute_patient"] = set(abs_df["sample"].dropna().map(patient_from_barcode))
    keys["absolute_sample15"] = set(abs_df["array"].dropna().map(sample15_from_barcode)) | set(abs_df["sample"].dropna().map(sample15_from_barcode))
    keys["absolute_sample16"] = set(abs_df["sample"].dropna().map(sample16_from_barcode))

    leuk = pd.read_csv(Path(args.leukocyte), sep="\t", header=None, names=["project_short", "barcode", "leukocyte_fraction"], dtype=str)
    keys["leukocyte_patient"] = set(leuk["barcode"].dropna().map(patient_from_barcode))
    keys["leukocyte_sample15"] = set(leuk["barcode"].dropna().map(sample15_from_barcode))
    keys["leukocyte_sample16"] = set(leuk["barcode"].dropna().map(sample16_from_barcode))

    cib = read_tsv(Path(args.cibersort), dtype=str)
    keys["cibersort_patient"] = set(cib["SampleID"].dropna().map(patient_from_barcode))
    keys["cibersort_sample15"] = set(cib["SampleID"].dropna().map(sample15_from_barcode))
    keys["cibersort_sample16"] = set(cib["SampleID"].dropna().map(sample16_from_barcode))

    qual = read_tsv(Path(args.quality), dtype=str)
    keys["quality_patient"] = set(qual["patient_barcode"].dropna().map(patient_from_barcode))
    keys["quality_sample15"] = set(qual["aliquot_barcode"].dropna().map(sample15_from_barcode))
    keys["quality_sample16"] = set(qual["aliquot_barcode"].dropna().map(sample16_from_barcode))
    return keys


def add_covariate_flags(df: pd.DataFrame, keys: dict[str, set[str]]) -> pd.DataFrame:
    out = df.copy()
    out["sample15"] = out["sample_id"].map(sample15_from_barcode)
    out["sample16"] = out["sample_id"].map(sample16_from_barcode)
    for source in ["absolute", "leukocyte", "cibersort", "quality"]:
        out[f"{source}_patient_match"] = out["patient_id"].isin(keys[f"{source}_patient"]).astype(int)
        out[f"{source}_sample15_match"] = out["sample15"].isin(keys[f"{source}_sample15"]).astype(int)
        out[f"{source}_sample16_match"] = out["sample16"].isin(keys[f"{source}_sample16"]).astype(int)
        out[f"{source}_any_match"] = (
            (out[f"{source}_patient_match"] == 1)
            | (out[f"{source}_sample15_match"] == 1)
            | (out[f"{source}_sample16_match"] == 1)
        ).astype(int)
    return out


def summarize_counts(audit: pd.DataFrame) -> pd.DataFrame:
    group_cols = [
        "project_id",
        "sample_type",
        "sample_type_code",
        "tissue_type",
        "reference_category",
        "matched_tumor_split_status",
    ]
    return (
        audit.groupby(group_cols, dropna=False)
        .agg(
            n_samples=("sample_id", "nunique"),
            n_patients=("patient_id", "nunique"),
            n_with_development_matched_tumor=("has_development_tumor", "sum"),
            n_with_heldout_matched_tumor=("has_heldout_tumor", "sum"),
        )
        .reset_index()
        .sort_values(["project_id", "sample_type_code", "matched_tumor_split_status"])
    )


def summarize_covariates(audit: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for source in ["absolute", "leukocyte", "cibersort", "quality"]:
        for level in ["patient", "sample15", "sample16", "any"]:
            col = f"{source}_{level}_match"
            rows.append(
                {
                    "covariate_source": source,
                    "match_level": level,
                    "n_normal_samples_matched": int(audit[col].sum()),
                    "fraction_normal_samples_matched": float(audit[col].mean()) if len(audit) else 0.0,
                    "n_projects_with_match": int(audit.loc[audit[col] == 1, "project_id"].nunique()),
                }
            )
    return pd.DataFrame(rows)


def write_summary(out_dir: Path, audit: pd.DataFrame, counts: pd.DataFrame, cov: pd.DataFrame) -> None:
    lines = [
        "# Normal Metadata And Composition Availability Audit",
        "",
        "Exploratory Module 0 output for the healthy/normal tissue probe biology decomposition plan.",
        "",
        "## Normal Sample Scope",
        f"- Normal samples: `{audit['sample_id'].nunique()}`",
        f"- Normal patients: `{audit['patient_id'].nunique()}`",
        f"- TCGA projects: `{audit['project_id'].nunique()}`",
        "- Sample types: "
        + "; ".join(
            f"{row.sample_type_code} / {row.sample_type}: {int(row.n_samples)}"
            for row in audit.groupby(["sample_type_code", "sample_type"], dropna=False)
            .agg(n_samples=("sample_id", "nunique"))
            .reset_index()
            .itertuples()
        ),
        "",
        "## Matched Tumor Split Status",
    ]
    for row in audit.groupby("matched_tumor_split_status").agg(n_samples=("sample_id", "nunique"), n_patients=("patient_id", "nunique")).reset_index().itertuples():
        lines.append(f"- `{row.matched_tumor_split_status}`: `{int(row.n_samples)}` samples, `{int(row.n_patients)}` patients")
    lines.extend(["", "## Composition / Purity Covariate Availability"])
    for row in cov.loc[cov["match_level"] == "sample16"].itertuples():
        lines.append(
            f"- `{row.covariate_source}` exact sample-level (`sample16`) matches: "
            f"`{int(row.n_normal_samples_matched)}` samples across `{int(row.n_projects_with_match)}` projects"
        )
    lines.extend(
        [
            "",
            "Patient-level matches are reported only as an availability warning because tumor-derived covariates may exist for the same patient while the normal aliquot itself lacks a valid normal-sample composition estimate.",
            "",
            "## Outputs",
            "- `normal_sample_metadata_audit.tsv`",
            "- `normal_sample_counts_by_project_type_split.tsv`",
            "- `normal_composition_covariate_availability.tsv`",
            "- `normal_metadata_audit_manifest.json`",
        ]
    )
    (out_dir / "normal_metadata_audit_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    normal = read_tsv(Path(args.normal_inventory), dtype=str)
    gdc = read_tsv(Path(args.normal_gdc_inventory), dtype=str)
    gdc_cols = [
        "sample_id",
        "case_id",
        "case_submitter_id",
        "tissue_type",
        "file_uuid",
        "file_name",
        "platform",
    ]
    gdc_meta = gdc[[c for c in gdc_cols if c in gdc.columns]].drop_duplicates("sample_id")
    audit = normal.merge(gdc_meta, on="sample_id", how="left", suffixes=("", "_gdc"))

    tumor = load_tumor_split(Path(args.tumor_dev), Path(args.tumor_lockbox))
    dev_patients = set(tumor.loc[tumor["tumor_split"] == "development", "patient_id"])
    held_patients = set(tumor.loc[tumor["tumor_split"] == "heldout_tcga", "patient_id"])
    audit["has_development_tumor"] = audit["patient_id"].isin(dev_patients).astype(int)
    audit["has_heldout_tumor"] = audit["patient_id"].isin(held_patients).astype(int)
    audit["matched_tumor_split_status"] = "no_matched_modeled_tumor"
    audit.loc[(audit["has_development_tumor"] == 1) & (audit["has_heldout_tumor"] == 0), "matched_tumor_split_status"] = "development_matched"
    audit.loc[(audit["has_development_tumor"] == 0) & (audit["has_heldout_tumor"] == 1), "matched_tumor_split_status"] = "heldout_matched"
    audit.loc[(audit["has_development_tumor"] == 1) & (audit["has_heldout_tumor"] == 1), "matched_tumor_split_status"] = "both_dev_and_heldout_patient"
    heldout_clean = (audit["has_development_tumor"] == 0) & (audit["has_heldout_tumor"] == 1)
    no_modeled_tumor = (audit["has_development_tumor"] == 0) & (audit["has_heldout_tumor"] == 0)
    audit["locked_model_normal_primary_subset"] = (heldout_clean | no_modeled_tumor).astype(int)
    audit["locked_model_normal_subset_reason"] = "development_paired_or_mixed"
    audit.loc[heldout_clean, "locked_model_normal_subset_reason"] = "heldout_matched_no_dev_tumor"
    audit.loc[no_modeled_tumor, "locked_model_normal_subset_reason"] = "no_matched_modeled_tumor"

    keys = covariate_keys(args)
    audit = add_covariate_flags(audit, keys)

    front_cols = [
        "project_id",
        "sample_id",
        "patient_id",
        "sample_type",
        "sample_type_code",
        "tissue_type",
        "reference_category",
        "source_group",
        "source_id",
        "case_id",
        "case_submitter_id",
        "file_uuid",
        "file_name",
        "platform",
        "has_development_tumor",
        "has_heldout_tumor",
        "matched_tumor_split_status",
        "locked_model_normal_primary_subset",
        "locked_model_normal_subset_reason",
    ]
    cov_cols = [c for c in audit.columns if c.endswith("_match")]
    probe_cols = [c for c in audit.columns if c.startswith("cg")]
    other_cols = [c for c in audit.columns if c not in set(front_cols + cov_cols + probe_cols)]
    audit_out = audit[[c for c in front_cols if c in audit.columns] + cov_cols + other_cols + probe_cols]
    counts = summarize_counts(audit)
    cov = summarize_covariates(audit)

    audit_out.to_csv(out_dir / "normal_sample_metadata_audit.tsv", sep="\t", index=False)
    counts.to_csv(out_dir / "normal_sample_counts_by_project_type_split.tsv", sep="\t", index=False)
    cov.to_csv(out_dir / "normal_composition_covariate_availability.tsv", sep="\t", index=False)
    write_summary(out_dir, audit, counts, cov)

    manifest = {
        "out_dir": str(out_dir),
        "n_normal_samples": int(audit["sample_id"].nunique()),
        "n_normal_patients": int(audit["patient_id"].nunique()),
        "n_projects": int(audit["project_id"].nunique()),
        "matched_tumor_split_counts": audit["matched_tumor_split_status"].value_counts().to_dict(),
        "locked_model_normal_primary_subset_n": int(audit["locked_model_normal_primary_subset"].sum()),
        "sample16_covariate_matches": {
            source: int(audit[f"{source}_sample16_match"].sum())
            for source in ["absolute", "leukocyte", "cibersort", "quality"]
        },
    }
    (out_dir / "normal_metadata_audit_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
