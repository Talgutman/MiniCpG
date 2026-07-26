#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


KEY_COLS = {
    "sample_id",
    "project_id",
    "patient_id",
    "sample_type",
    "sample_type_code",
    "file_uuid",
    "file_name",
}

PROJECT_SUMMARY_COLUMNS = [
    "project_id",
    "probe_id",
    "comparator_type",
    "evidence_tier",
    "n_tumor",
    "n_normal",
    "n_matched_pairs_available",
    "tumor_median_beta",
    "normal_median_beta",
    "delta_median_beta",
    "tumor_mean_beta",
    "normal_mean_beta",
    "delta_mean_beta",
]

MATCHED_SUMMARY_COLUMNS = [
    "project_id",
    "probe_id",
    "comparator_type",
    "evidence_tier",
    "n_pairs",
    "matched_delta_median_beta",
    "matched_delta_mean_beta",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Summarize tumor-vs-normal methylation behavior for a selected probe panel. "
            "This utility is panel-agnostic and can be reused for future panels."
        )
    )
    parser.add_argument("--tumor-matrix-tsv", required=True, help="Tumor panel matrix TSV.")
    parser.add_argument("--normal-matrix-tsv", required=True, help="Normal/reference panel matrix TSV.")
    parser.add_argument("--out-dir", required=True, help="Output directory.")
    parser.add_argument(
        "--project-ids",
        default="",
        help="Optional comma-separated project IDs to keep.",
    )
    return parser.parse_args()


def load_matrix(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    required = {"sample_id", "project_id"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns in {path}: {sorted(missing)}")
    if "patient_id" not in df.columns:
        df["patient_id"] = df["sample_id"].astype(str).str.slice(0, 12)
    else:
        patient_id = df["patient_id"].astype(str)
        df["patient_id"] = patient_id.where(patient_id != "", df["sample_id"].astype(str).str.slice(0, 12))
    return df


def probe_columns(df: pd.DataFrame) -> list[str]:
    return [col for col in df.columns if col not in KEY_COLS]


def to_numeric(df: pd.DataFrame, probes: list[str]) -> pd.DataFrame:
    out = df.copy()
    for probe in probes:
        out[probe] = pd.to_numeric(out[probe], errors="coerce")
    return out


def evidence_tier(n_reference: int) -> str:
    if n_reference >= 20:
        return "high"
    if n_reference >= 5:
        return "medium"
    return "low"


def matched_pair_counts(tumor: pd.DataFrame, normal: pd.DataFrame) -> dict[str, int]:
    tumor_match = tumor.loc[tumor["patient_id"] != "", ["project_id", "patient_id"]].drop_duplicates()
    normal_match = normal.loc[normal["patient_id"] != "", ["project_id", "patient_id"]].drop_duplicates()
    merged = tumor_match.merge(normal_match, on=["project_id", "patient_id"], how="inner")
    if merged.empty:
        return {}
    return merged.groupby("project_id").size().astype(int).to_dict()


def project_probe_summary(tumor: pd.DataFrame, normal: pd.DataFrame, probes: list[str]) -> pd.DataFrame:
    rows: list[dict] = []
    projects = sorted(set(tumor["project_id"]).intersection(set(normal["project_id"])))
    matched_counts = matched_pair_counts(tumor, normal)
    for project_id in projects:
        tumor_sub = tumor.loc[tumor["project_id"] == project_id]
        normal_sub = normal.loc[normal["project_id"] == project_id]
        n_pairs = int(matched_counts.get(project_id, 0))
        for probe in probes:
            tumor_vals = tumor_sub[probe].dropna()
            normal_vals = normal_sub[probe].dropna()
            if tumor_vals.empty or normal_vals.empty:
                continue
            rows.append(
                {
                    "project_id": project_id,
                    "probe_id": probe,
                    "comparator_type": "same_tissue_tcga_solid_normal",
                    "evidence_tier": evidence_tier(int(normal_vals.shape[0])),
                    "n_tumor": int(tumor_vals.shape[0]),
                    "n_normal": int(normal_vals.shape[0]),
                    "n_matched_pairs_available": n_pairs,
                    "tumor_median_beta": float(tumor_vals.median()),
                    "normal_median_beta": float(normal_vals.median()),
                    "delta_median_beta": float(tumor_vals.median() - normal_vals.median()),
                    "tumor_mean_beta": float(tumor_vals.mean()),
                    "normal_mean_beta": float(normal_vals.mean()),
                    "delta_mean_beta": float(tumor_vals.mean() - normal_vals.mean()),
                }
            )
    return pd.DataFrame(rows, columns=PROJECT_SUMMARY_COLUMNS)


def matched_pair_summary(tumor: pd.DataFrame, normal: pd.DataFrame, probes: list[str]) -> pd.DataFrame:
    if "patient_id" not in tumor.columns or "patient_id" not in normal.columns:
        return pd.DataFrame(columns=MATCHED_SUMMARY_COLUMNS)

    tumor_match = tumor.loc[tumor["patient_id"] != ""].copy()
    normal_match = normal.loc[normal["patient_id"] != ""].copy()
    tumor_match = tumor_match.drop_duplicates(subset=["project_id", "patient_id"], keep="first")
    normal_match = normal_match.drop_duplicates(subset=["project_id", "patient_id"], keep="first")
    merged = tumor_match.merge(
        normal_match,
        on=["project_id", "patient_id"],
        suffixes=("_tumor", "_normal"),
        how="inner",
    )
    if merged.empty:
        return pd.DataFrame(columns=MATCHED_SUMMARY_COLUMNS)

    rows: list[dict] = []
    for project_id, sub in merged.groupby("project_id", sort=True):
        for probe in probes:
            tumor_col = f"{probe}_tumor"
            normal_col = f"{probe}_normal"
            if tumor_col not in sub.columns or normal_col not in sub.columns:
                continue
            paired = sub[[tumor_col, normal_col]].dropna()
            if paired.empty:
                continue
            delta = paired[tumor_col] - paired[normal_col]
            rows.append(
                {
                    "project_id": project_id,
                    "probe_id": probe,
                    "comparator_type": "matched_normal",
                    "evidence_tier": evidence_tier(int(paired.shape[0])),
                    "n_pairs": int(paired.shape[0]),
                    "matched_delta_median_beta": float(delta.median()),
                    "matched_delta_mean_beta": float(delta.mean()),
                }
            )
    return pd.DataFrame(rows, columns=MATCHED_SUMMARY_COLUMNS)


def write_summary(
    out_path: Path,
    tumor_path: Path,
    normal_path: Path,
    project_summary: pd.DataFrame,
    matched_summary: pd.DataFrame,
) -> None:
    lines: list[str] = []
    lines.append("# Panel Tumor-vs-Normal Summary")
    lines.append("")
    lines.append(f"- Tumor matrix: `{tumor_path}`")
    lines.append(f"- Normal matrix: `{normal_path}`")
    lines.append(f"- Projects with both tumor and normal data: `{project_summary['project_id'].nunique() if not project_summary.empty else 0}`")
    lines.append(f"- Probe-project rows: `{len(project_summary)}`")
    lines.append(f"- Matched-pair probe-project rows: `{len(matched_summary)}`")
    if not project_summary.empty:
        project_rollup = (
            project_summary[["project_id", "n_normal", "n_matched_pairs_available", "evidence_tier"]]
            .drop_duplicates()
            .sort_values(["n_normal", "project_id"], ascending=[False, True])
        )
        lines.append("")
        lines.append("Project-level normal/reference coverage:")
        lines.append("")
        for row in project_rollup.itertuples(index=False):
            lines.append(
                f"- `{row.project_id}`: `n_normal={row.n_normal}`, `n_matched_pairs_available={row.n_matched_pairs_available}`, `evidence_tier={row.evidence_tier}`"
            )
    lines.append("")
    if not project_summary.empty:
        top_abs = (
            project_summary.assign(abs_delta=project_summary["delta_median_beta"].abs())
            .sort_values(["abs_delta", "project_id", "probe_id"], ascending=[False, True, True])
            .head(12)
        )
        lines.append("Top absolute tumor-vs-normal median shifts:")
        lines.append("")
        for row in top_abs.itertuples(index=False):
            lines.append(
                f"- `{row.project_id}` `{row.probe_id}`: "
                f"`delta_median_beta={row.delta_median_beta:.3f}` "
                f"(tumor `{row.tumor_median_beta:.3f}` vs normal `{row.normal_median_beta:.3f}`; "
                f"`n_tumor={row.n_tumor}`, `n_normal={row.n_normal}`)"
            )
        lines.append("")
    if not matched_summary.empty:
        top_matched = (
            matched_summary.assign(abs_delta=matched_summary["matched_delta_median_beta"].abs())
            .sort_values(["abs_delta", "project_id", "probe_id"], ascending=[False, True, True])
            .head(12)
        )
        lines.append("Top absolute matched-pair median shifts:")
        lines.append("")
        for row in top_matched.itertuples(index=False):
            lines.append(
                f"- `{row.project_id}` `{row.probe_id}`: "
                f"`matched_delta_median_beta={row.matched_delta_median_beta:.3f}` "
                f"with `n_pairs={row.n_pairs}`"
            )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    args = parse_args()
    tumor_path = Path(args.tumor_matrix_tsv)
    normal_path = Path(args.normal_matrix_tsv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    tumor = load_matrix(tumor_path)
    normal = load_matrix(normal_path)

    if args.project_ids:
        keep = {x.strip() for x in args.project_ids.split(",") if x.strip()}
        tumor = tumor.loc[tumor["project_id"].isin(keep)].copy()
        normal = normal.loc[normal["project_id"].isin(keep)].copy()

    probes = [probe for probe in probe_columns(tumor) if probe in probe_columns(normal)]
    if not probes:
        raise SystemExit("No shared probe columns found between tumor and normal matrices.")

    tumor = to_numeric(tumor, probes)
    normal = to_numeric(normal, probes)

    project_summary = project_probe_summary(tumor, normal, probes)
    matched_summary = matched_pair_summary(tumor, normal, probes)

    project_summary.to_csv(out_dir / "tumor_normal_probe_summary.tsv", sep="\t", index=False)
    matched_summary.to_csv(out_dir / "matched_pair_probe_summary.tsv", sep="\t", index=False)
    write_summary(out_dir / "tumor_normal_summary.md", tumor_path, normal_path, project_summary, matched_summary)

    print(f"Wrote tumor-vs-normal summary to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
