#!/usr/bin/env python3
"""Build a classification-ready manifest aligned to HM450 probe matrix columns."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List

import pandas as pd
import pyarrow.parquet as pq


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build classification manifest for HM450 probe matrix.")
    parser.add_argument("--probe-matrix", required=True, help="Aggregated HM450 probe matrix parquet.")
    parser.add_argument("--manifest", required=True, help="Filtered manifest TSV (HM450).")
    parser.add_argument(
        "--out",
        default="outputs/manifests/classification_manifest_hm450.tsv",
        help="Output manifest TSV.",
    )
    parser.add_argument(
        "--counts-out",
        default="outputs/manifests/classification_counts_hm450.tsv",
        help="Output counts TSV.",
    )
    parser.add_argument(
        "--sample-col",
        default="sample_id",
        help="Manifest column that matches probe matrix sample IDs.",
    )
    parser.add_argument(
        "--patient-col",
        default="patient_id",
        help="Manifest patient ID column.",
    )
    parser.add_argument(
        "--project-col",
        default="project_id",
        help="Manifest project (class label) column.",
    )
    parser.add_argument(
        "--sample-type-col",
        default="sample_type",
        help="Manifest sample type column.",
    )
    parser.add_argument(
        "--primary-only",
        action="store_true",
        default=True,
        help="Keep only Primary samples (default: true).",
    )
    parser.add_argument(
        "--no-primary-only",
        dest="primary_only",
        action="store_false",
        help="Disable Primary-only filtering.",
    )
    parser.add_argument(
        "--source",
        default="methylation_survival_probe_matrix_hm450",
        help="Source label to attach to output rows.",
    )
    return parser.parse_args()


def require_columns(df: pd.DataFrame, cols: List[str]) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def load_matrix_sample_ids(path: Path) -> set[str]:
    pf = pq.ParquetFile(path)
    cols = pf.schema.names
    return {c for c in cols if str(c).startswith("TCGA-")}


def main() -> int:
    args = parse_args()
    probe_matrix = Path(args.probe_matrix)
    manifest_path = Path(args.manifest)

    if not probe_matrix.exists():
        print(f"ERROR: probe matrix not found: {probe_matrix}", file=sys.stderr)
        return 1
    if not manifest_path.exists():
        print(f"ERROR: manifest not found: {manifest_path}", file=sys.stderr)
        return 1

    manifest = pd.read_csv(manifest_path, sep="\t")

    try:
        require_columns(manifest, [args.sample_col, args.project_col, args.patient_col])
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if args.primary_only:
        if args.sample_type_col not in manifest.columns:
            print(f"ERROR: sample_type column {args.sample_type_col} missing for Primary-only filter", file=sys.stderr)
            return 1
        sample_type = manifest[args.sample_type_col].astype(str)
        manifest = manifest[sample_type.str.contains("primary", case=False, na=False)].copy()

    before_dedup = len(manifest)
    sort_cols = [args.patient_col, args.sample_col]
    if "file_UUID" in manifest.columns:
        sort_cols.append("file_UUID")
    manifest = manifest.sort_values(sort_cols).drop_duplicates(subset=[args.patient_col], keep="first")

    sample_ids = load_matrix_sample_ids(probe_matrix)
    if not sample_ids:
        print("ERROR: no TCGA sample columns found in probe matrix", file=sys.stderr)
        return 1

    manifest = manifest[manifest[args.sample_col].astype(str).isin(sample_ids)].copy()

    manifest["sample_id"] = manifest[args.sample_col].astype(str)
    manifest["source"] = args.source

    preferred_cols = [
        "sample_id",
        args.project_col,
        args.patient_col,
        args.sample_type_col,
        "source",
        args.sample_col,
        "case_id",
        "file_UUID",
        "file_uuid",
        "cancer_acronym",
        "clinical_cancer_type",
    ]
    seen: set[str] = set()
    out_cols: list[str] = []
    for col in preferred_cols:
        if col in manifest.columns and col not in seen:
            out_cols.append(col)
            seen.add(col)
    out_df = manifest[out_cols].copy()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, sep="\t", index=False)

    counts = (
        out_df.groupby(args.project_col)
        .agg(n_samples=("sample_id", "size"), n_patients=(args.patient_col, "nunique"))
        .reset_index()
        .sort_values("n_samples", ascending=False)
    )
    counts_path = Path(args.counts_out)
    counts_path.parent.mkdir(parents=True, exist_ok=True)
    counts.to_csv(counts_path, sep="\t", index=False)

    print(f"Input rows: {before_dedup}")
    print(f"After patient dedup: {len(manifest)}")
    print(f"Wrote manifest: {out_path}")
    print(f"Wrote counts: {counts_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
