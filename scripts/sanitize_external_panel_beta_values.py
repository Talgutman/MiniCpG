#!/usr/bin/env python3
"""Mask non-beta-scale values in an external panel matrix and recompute eligibility."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Set external panel probe values outside [0, 1] to missing, recompute panel "
            "missingness, and emit beta-scale QC tables."
        )
    )
    parser.add_argument("--panel-matrix", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--max-missing-probes", type=int, default=5)
    parser.add_argument("--matrix-name", default="external_panel_matrix_main_missing_le5_beta_sanitized.tsv")
    parser.add_argument("--all-matrix-name", default="external_panel_matrix_all_beta_sanitized.tsv")
    return parser.parse_args()


def _probe_columns(df: pd.DataFrame) -> list[str]:
    probes = [column for column in df.columns if column.startswith("cg")]
    if not probes:
        raise SystemExit("No probe columns found; expected Illumina CpG IDs beginning with 'cg'.")
    return probes


def _missing_like(series: pd.Series) -> pd.Series:
    text = series.fillna("").astype(str).str.strip()
    return text.eq("") | text.str.lower().isin({"nan", "na", "none", "null"})


def main() -> int:
    args = parse_args()
    matrix_path = Path(args.panel_matrix)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not matrix_path.exists():
        raise SystemExit(f"Missing input matrix: {matrix_path}")

    df = pd.read_csv(matrix_path, sep="\t", dtype=str).fillna("")
    probes = _probe_columns(df)
    if "sample_id" not in df.columns or "gse_id" not in df.columns:
        raise SystemExit("Input matrix must contain gse_id and sample_id.")
    label_col = "tcga_label" if "tcga_label" in df.columns else ""

    sanitized = df.copy()
    bad_parts: list[pd.DataFrame] = []
    for probe in probes:
        values = pd.to_numeric(df[probe], errors="coerce")
        missing = _missing_like(df[probe])
        bad = values.notna() & ~missing & ((values < 0.0) | (values > 1.0))
        if not bad.any():
            continue
        cols = [c for c in ["gse_id", "sample_id", "tcga_label", "target_bucket"] if c in df.columns]
        part = df.loc[bad, cols].copy()
        part["probe_id"] = probe
        part["invalid_value"] = values.loc[bad].to_numpy(dtype=float)
        bad_parts.append(part)
        sanitized.loc[bad, probe] = ""

    if bad_parts:
        bad_cells = pd.concat(bad_parts, ignore_index=True)
    else:
        bad_cells = pd.DataFrame(
            columns=[c for c in ["gse_id", "sample_id", "tcga_label", "target_bucket"] if c in df.columns]
            + ["probe_id", "invalid_value"]
        )

    valid_value_frame = pd.DataFrame(index=sanitized.index)
    for probe in probes:
        values = pd.to_numeric(sanitized[probe], errors="coerce")
        valid_value_frame[probe] = values.notna() & values.between(0.0, 1.0, inclusive="both")
    n_found = valid_value_frame.sum(axis=1).astype(int)
    n_missing = (len(probes) - n_found).astype(int)
    sanitized["n_panel_found"] = n_found.astype(str)
    sanitized["n_panel_missing"] = n_missing.astype(str)
    sanitized["n_beta_out_of_range_masked"] = "0"
    if not bad_cells.empty:
        per_sample_bad = bad_cells.groupby(["gse_id", "sample_id"]).size()
        sample_keys = sanitized["gse_id"].astype(str).str.cat(sanitized["sample_id"].astype(str), sep="||")
        bad_by_key = {f"{gse}||{sample}": int(count) for (gse, sample), count in per_sample_bad.items()}
        sanitized["n_beta_out_of_range_masked"] = sample_keys.map(bad_by_key).fillna(0).astype(int).astype(str)

    if "extraction_status" in sanitized.columns:
        sanitized["extraction_status"] = np.where(
            n_found.eq(0),
            "no_probe_values",
            np.where(n_missing.eq(0), "ok_full", "ok_partial"),
        )

    eligible = sanitized.loc[n_missing <= int(args.max_missing_probes)].copy()

    bad_cells_path = out_dir / "beta_scale_qc_bad_cells.tsv"
    by_gse_label_path = out_dir / "beta_scale_qc_by_gse_label.tsv"
    by_probe_path = out_dir / "beta_scale_qc_by_probe.tsv"
    all_matrix_path = out_dir / args.all_matrix_name
    eligible_matrix_path = out_dir / args.matrix_name
    summary_path = out_dir / "beta_scale_sanitization_summary.json"

    bad_cells.to_csv(bad_cells_path, sep="\t", index=False, quoting=csv.QUOTE_MINIMAL)
    group_cols = [c for c in ["gse_id", "tcga_label"] if c in bad_cells.columns]
    if group_cols and not bad_cells.empty:
        (
            bad_cells.groupby(group_cols, dropna=False)
            .agg(
                n_bad_cells=("probe_id", "size"),
                n_bad_samples=("sample_id", "nunique"),
                n_bad_probes=("probe_id", "nunique"),
                probes=("probe_id", lambda x: ";".join(sorted(set(map(str, x))))),
            )
            .reset_index()
            .sort_values(["n_bad_cells"] + group_cols, ascending=[False] + [True] * len(group_cols), kind="mergesort")
            .to_csv(by_gse_label_path, sep="\t", index=False)
        )
    else:
        pd.DataFrame(columns=group_cols + ["n_bad_cells", "n_bad_samples", "n_bad_probes", "probes"]).to_csv(
            by_gse_label_path, sep="\t", index=False
        )
    if not bad_cells.empty:
        (
            bad_cells.groupby("probe_id", dropna=False)
            .agg(
                n_bad_cells=("invalid_value", "size"),
                n_bad_samples=("sample_id", "nunique"),
                min_invalid=("invalid_value", "min"),
                median_invalid=("invalid_value", "median"),
                max_invalid=("invalid_value", "max"),
            )
            .reset_index()
            .sort_values(["n_bad_cells", "probe_id"], ascending=[False, True], kind="mergesort")
            .to_csv(by_probe_path, sep="\t", index=False)
        )
    else:
        pd.DataFrame(
            columns=["probe_id", "n_bad_cells", "n_bad_samples", "min_invalid", "median_invalid", "max_invalid"]
        ).to_csv(by_probe_path, sep="\t", index=False)

    sanitized.to_csv(all_matrix_path, sep="\t", index=False, quoting=csv.QUOTE_MINIMAL)
    eligible.to_csv(eligible_matrix_path, sep="\t", index=False, quoting=csv.QUOTE_MINIMAL)

    summary = {
        "input_matrix": str(matrix_path),
        "n_input_samples": int(len(df)),
        "n_probe_columns": int(len(probes)),
        "max_missing_probes": int(args.max_missing_probes),
        "n_bad_cells_masked": int(len(bad_cells)),
        "n_samples_with_bad_cells": int(bad_cells[["gse_id", "sample_id"]].drop_duplicates().shape[0])
        if not bad_cells.empty
        else 0,
        "n_output_samples_all": int(len(sanitized)),
        "n_output_samples_eligible": int(len(eligible)),
        "n_samples_removed_by_sanitized_missingness": int(len(sanitized) - len(eligible)),
        "eligible_matrix_tsv": str(eligible_matrix_path),
        "all_sanitized_matrix_tsv": str(all_matrix_path),
        "bad_cells_tsv": str(bad_cells_path),
        "qc_by_gse_label_tsv": str(by_gse_label_path),
        "qc_by_probe_tsv": str(by_probe_path),
    }
    if label_col:
        summary["eligible_counts_by_label"] = {
            str(k): int(v) for k, v in eligible[label_col].value_counts(dropna=False).sort_index().items()
        }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
