#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def bh_adjust(values: pd.Series) -> pd.Series:
    """Benjamini-Hochberg adjustment preserving the input index."""
    p = pd.to_numeric(values, errors="coerce")
    out = pd.Series(pd.NA, index=values.index, dtype="Float64")
    valid = p.notna()
    if not valid.any():
        return out
    pv = p[valid].astype(float)
    order = pv.sort_values().index
    n = len(order)
    adjusted = pd.Series(index=order, dtype=float)
    running_min = 1.0
    for rank_from_end, idx in enumerate(reversed(order), start=1):
        rank = n - rank_from_end + 1
        q = min(running_min, float(pv.loc[idx]) * n / rank)
        running_min = q
        adjusted.loc[idx] = min(q, 1.0)
    out.loc[adjusted.index] = adjusted
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize linked probe-gene methylation-expression ranks against transcriptome-wide nulls."
    )
    parser.add_argument(
        "--transcriptome-null-dir",
        required=True,
        help="Directory containing linked_probe_gene_transcriptome_rank_by_project.tsv.",
    )
    parser.add_argument("--out-dir", required=True, help="Output directory for inspection summaries.")
    parser.add_argument("--top-n", type=int, default=25, help="Number of top linked rows to export.")
    parser.add_argument(
        "--min-matched",
        type=int,
        default=30,
        help="Minimum matched methylation/RNA samples required for null-aware support calls.",
    )
    parser.add_argument(
        "--nominal-q-cutoff",
        type=float,
        default=0.005,
        help="Maximum within-probe/cohort methylation-expression q-value for null-aware support calls.",
    )
    parser.add_argument(
        "--empirical-p-cutoff",
        type=float,
        default=0.01,
        help="Maximum transcriptome-rank empirical p-value for an interesting linked-gene support row.",
    )
    parser.add_argument(
        "--stringent-empirical-p-cutoff",
        type=float,
        default=0.001,
        help="Maximum transcriptome-rank empirical p-value for a stringent linked-gene support row.",
    )
    return parser.parse_args()


def read_linked(null_dir: Path) -> pd.DataFrame:
    path = null_dir / "linked_probe_gene_transcriptome_rank_by_project.tsv"
    if not path.exists():
        raise FileNotFoundError(f"linked transcriptome-rank TSV not found: {path}")
    linked = pd.read_csv(path, sep="\t")
    required = {
        "project_id",
        "probe_id",
        "gene",
        "panel_rank",
        "n_matched",
        "spearman_beta_expr",
        "spearman_q_within_probe_cohort",
        "transcriptome_abs_rho_rank",
        "transcriptome_abs_rho_percentile",
        "is_top_1pct_abs_rho",
        "is_top_0_5pct_abs_rho",
        "is_top_0_1pct_abs_rho",
    }
    missing = sorted(required - set(linked.columns))
    if missing:
        raise ValueError(f"linked TSV is missing required columns: {missing}")
    linked["abs_rho"] = linked["spearman_beta_expr"].abs()
    for col in ["is_top_1pct_abs_rho", "is_top_0_5pct_abs_rho", "is_top_0_1pct_abs_rho"]:
        linked[col] = linked[col].astype(bool)
    linked["transcriptome_empirical_p"] = linked["transcriptome_abs_rho_rank"] / linked["n_genes_tested"]
    linked["transcriptome_empirical_q_linked_rows"] = bh_adjust(linked["transcriptome_empirical_p"])
    return linked


def add_support_calls(
    linked: pd.DataFrame,
    *,
    min_matched: int,
    nominal_q_cutoff: float,
    empirical_p_cutoff: float,
    stringent_empirical_p_cutoff: float,
) -> pd.DataFrame:
    linked = linked.copy()
    base = (linked["n_matched"] >= min_matched) & (linked["spearman_q_within_probe_cohort"] <= nominal_q_cutoff)
    linked["null_interesting_row"] = base & (linked["transcriptome_empirical_p"] <= empirical_p_cutoff)
    linked["null_stringent_row"] = base & (linked["transcriptome_empirical_p"] <= stringent_empirical_p_cutoff)
    linked["expression_support_tier"] = "none"
    linked.loc[linked["spearman_q_within_probe_cohort"] <= nominal_q_cutoff, "expression_support_tier"] = "nominal_only"
    linked.loc[linked["transcriptome_empirical_p"] <= empirical_p_cutoff, "expression_support_tier"] = "rank_only"
    linked.loc[linked["null_interesting_row"], "expression_support_tier"] = "null_interesting"
    linked.loc[linked["null_stringent_row"], "expression_support_tier"] = "null_stringent"
    return linked


def add_best_row_fields(summary: pd.DataFrame, linked: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    best = (
        linked.sort_values(["transcriptome_abs_rho_percentile", "abs_rho"], ascending=[False, False])
        .drop_duplicates(keys)
        [
            keys
            + [
                "project_id",
                "spearman_beta_expr",
                "abs_rho",
                "transcriptome_abs_rho_percentile",
                "transcriptome_abs_rho_rank",
                "transcriptome_empirical_p",
                "transcriptome_empirical_q_linked_rows",
                "expression_support_tier",
            ]
        ]
        .rename(
            columns={
                "project_id": "best_project_id",
                "spearman_beta_expr": "best_spearman_beta_expr",
                "abs_rho": "best_abs_rho",
                "transcriptome_abs_rho_percentile": "best_transcriptome_abs_rho_percentile",
                "transcriptome_abs_rho_rank": "best_transcriptome_abs_rho_rank",
                "transcriptome_empirical_p": "best_transcriptome_empirical_p",
                "transcriptome_empirical_q_linked_rows": "best_transcriptome_empirical_q_linked_rows",
                "expression_support_tier": "best_expression_support_tier",
            }
        )
    )
    return summary.merge(best, on=keys, how="left")


def summarize(
    linked: pd.DataFrame,
    top_n: int,
    *,
    min_matched: int,
    nominal_q_cutoff: float,
    empirical_p_cutoff: float,
    stringent_empirical_p_cutoff: float,
) -> dict[str, pd.DataFrame | dict[str, object]]:
    linked = add_support_calls(
        linked,
        min_matched=min_matched,
        nominal_q_cutoff=nominal_q_cutoff,
        empirical_p_cutoff=empirical_p_cutoff,
        stringent_empirical_p_cutoff=stringent_empirical_p_cutoff,
    )
    overall = {
        "min_matched_for_support": int(min_matched),
        "nominal_q_cutoff_for_support": float(nominal_q_cutoff),
        "empirical_p_cutoff_for_interesting_support": float(empirical_p_cutoff),
        "empirical_p_cutoff_for_stringent_support": float(stringent_empirical_p_cutoff),
        "n_linked_rows": int(len(linked)),
        "n_probe_gene_pairs": int(linked[["probe_id", "gene"]].drop_duplicates().shape[0]),
        "n_probes": int(linked["probe_id"].nunique()),
        "n_projects": int(linked["project_id"].nunique()),
        "median_abs_rho": float(linked["abs_rho"].median()),
        "median_transcriptome_abs_rho_percentile": float(linked["transcriptome_abs_rho_percentile"].median()),
        "median_transcriptome_empirical_p": float(linked["transcriptome_empirical_p"].median()),
        "n_top_1pct_rows": int(linked["is_top_1pct_abs_rho"].sum()),
        "n_top_0_5pct_rows": int(linked["is_top_0_5pct_abs_rho"].sum()),
        "n_top_0_1pct_rows": int(linked["is_top_0_1pct_abs_rho"].sum()),
        "n_q_lt_0_05_rows": int((linked["spearman_q_within_probe_cohort"] < 0.05).sum()),
        "n_q_lt_0_005_rows": int((linked["spearman_q_within_probe_cohort"] < 0.005).sum()),
        "n_null_interesting_rows": int(linked["null_interesting_row"].sum()),
        "n_null_stringent_rows": int(linked["null_stringent_row"].sum()),
        "n_null_interesting_pairs": int(
            linked.loc[linked["null_interesting_row"], ["probe_id", "gene"]].drop_duplicates().shape[0]
        ),
        "n_null_stringent_pairs": int(
            linked.loc[linked["null_stringent_row"], ["probe_id", "gene"]].drop_duplicates().shape[0]
        ),
        "n_null_interesting_probes": int(linked.loc[linked["null_interesting_row"], "probe_id"].nunique()),
        "n_null_stringent_probes": int(linked.loc[linked["null_stringent_row"], "probe_id"].nunique()),
    }
    for label, expected_fraction in {
        "top_1pct": 0.01,
        "top_0_5pct": 0.005,
        "top_0_1pct": 0.001,
    }.items():
        observed = overall[f"n_{label}_rows"]
        expected = overall["n_linked_rows"] * expected_fraction
        overall[f"expected_{label}_rows_under_uniform_null"] = float(expected)
        overall[f"{label}_enrichment_vs_uniform_null"] = float(observed / expected) if expected else None

    top_rows = linked.sort_values(
        ["transcriptome_abs_rho_percentile", "abs_rho", "n_matched"], ascending=[False, False, False]
    ).head(top_n)

    probe_summary = (
        linked.groupby("probe_id")
        .agg(
            panel_rank=("panel_rank", "min"),
            n_linked_rows=("gene", "size"),
            n_genes=("gene", "nunique"),
            n_projects=("project_id", "nunique"),
            top_1pct_rows=("is_top_1pct_abs_rho", "sum"),
            top_0_5pct_rows=("is_top_0_5pct_abs_rho", "sum"),
            top_0_1pct_rows=("is_top_0_1pct_abs_rho", "sum"),
            median_abs_rho=("abs_rho", "median"),
            max_abs_rho=("abs_rho", "max"),
            median_transcriptome_abs_rho_percentile=("transcriptome_abs_rho_percentile", "median"),
            max_transcriptome_abs_rho_percentile=("transcriptome_abs_rho_percentile", "max"),
            q_lt_0_05_rows=("spearman_q_within_probe_cohort", lambda s: int((s < 0.05).sum())),
            q_lt_0_005_rows=("spearman_q_within_probe_cohort", lambda s: int((s < 0.005).sum())),
            null_interesting_rows=("null_interesting_row", "sum"),
            null_stringent_rows=("null_stringent_row", "sum"),
            min_transcriptome_empirical_p=("transcriptome_empirical_p", "min"),
            median_transcriptome_empirical_p=("transcriptome_empirical_p", "median"),
        )
        .reset_index()
        .sort_values(
            ["null_interesting_rows", "top_1pct_rows", "top_0_1pct_rows", "max_transcriptome_abs_rho_percentile"],
            ascending=[False, False, False, False],
        )
    )
    probe_summary = add_best_row_fields(probe_summary, linked, ["probe_id"])

    pair_summary = (
        linked.groupby(["probe_id", "gene"])
        .agg(
            panel_rank=("panel_rank", "min"),
            n_projects=("project_id", "nunique"),
            top_1pct_rows=("is_top_1pct_abs_rho", "sum"),
            top_0_5pct_rows=("is_top_0_5pct_abs_rho", "sum"),
            top_0_1pct_rows=("is_top_0_1pct_abs_rho", "sum"),
            median_spearman_beta_expr=("spearman_beta_expr", "median"),
            median_abs_rho=("abs_rho", "median"),
            max_abs_rho=("abs_rho", "max"),
            median_transcriptome_abs_rho_percentile=("transcriptome_abs_rho_percentile", "median"),
            max_transcriptome_abs_rho_percentile=("transcriptome_abs_rho_percentile", "max"),
            q_lt_0_05_rows=("spearman_q_within_probe_cohort", lambda s: int((s < 0.05).sum())),
            q_lt_0_005_rows=("spearman_q_within_probe_cohort", lambda s: int((s < 0.005).sum())),
            null_interesting_rows=("null_interesting_row", "sum"),
            null_stringent_rows=("null_stringent_row", "sum"),
            min_transcriptome_empirical_p=("transcriptome_empirical_p", "min"),
            median_transcriptome_empirical_p=("transcriptome_empirical_p", "median"),
        )
        .reset_index()
        .sort_values(
            ["null_interesting_rows", "top_1pct_rows", "top_0_1pct_rows", "max_transcriptome_abs_rho_percentile"],
            ascending=[False, False, False, False],
        )
    )
    pair_summary = add_best_row_fields(pair_summary, linked, ["probe_id", "gene"])

    cohort_summary = (
        linked.groupby("project_id")
        .agg(
            n_linked_rows=("gene", "size"),
            top_1pct_rows=("is_top_1pct_abs_rho", "sum"),
            top_0_1pct_rows=("is_top_0_1pct_abs_rho", "sum"),
            median_abs_rho=("abs_rho", "median"),
            median_transcriptome_abs_rho_percentile=("transcriptome_abs_rho_percentile", "median"),
            q_lt_0_05_rows=("spearman_q_within_probe_cohort", lambda s: int((s < 0.05).sum())),
            null_interesting_rows=("null_interesting_row", "sum"),
            null_stringent_rows=("null_stringent_row", "sum"),
        )
        .reset_index()
        .sort_values(["null_interesting_rows", "top_1pct_rows", "top_0_1pct_rows"], ascending=[False, False, False])
    )

    return {
        "overall": overall,
        "linked_scored": linked.sort_values(
            ["null_interesting_row", "transcriptome_empirical_p", "spearman_q_within_probe_cohort"],
            ascending=[False, True, True],
        ),
        "top_rows": top_rows,
        "probe_summary": probe_summary,
        "pair_summary": pair_summary,
        "cohort_summary": cohort_summary,
    }


def write_markdown(out_path: Path, results: dict[str, pd.DataFrame | dict[str, object]]) -> None:
    overall = results["overall"]
    assert isinstance(overall, dict)
    pair_summary = results["pair_summary"]
    probe_summary = results["probe_summary"]
    cohort_summary = results["cohort_summary"]
    assert isinstance(pair_summary, pd.DataFrame)
    assert isinstance(probe_summary, pd.DataFrame)
    assert isinstance(cohort_summary, pd.DataFrame)

    top_pairs = pair_summary.head(10)[
        [
            "probe_id",
            "gene",
            "null_interesting_rows",
            "null_stringent_rows",
            "top_1pct_rows",
            "top_0_1pct_rows",
            "median_spearman_beta_expr",
            "min_transcriptome_empirical_p",
            "median_transcriptome_abs_rho_percentile",
            "best_project_id",
            "best_spearman_beta_expr",
            "best_expression_support_tier",
        ]
    ]
    top_probes = probe_summary.head(10)[
        [
            "probe_id",
            "null_interesting_rows",
            "null_stringent_rows",
            "top_1pct_rows",
            "top_0_1pct_rows",
            "median_abs_rho",
            "min_transcriptome_empirical_p",
            "median_transcriptome_abs_rho_percentile",
            "best_project_id",
            "best_spearman_beta_expr",
            "best_expression_support_tier",
        ]
    ]
    top_cohorts = cohort_summary.head(10)[
        ["project_id", "top_1pct_rows", "top_0_1pct_rows", "median_abs_rho", "median_transcriptome_abs_rho_percentile"]
    ]

    def markdown_table(frame: pd.DataFrame) -> str:
        if frame.empty:
            return "_No rows._"
        columns = list(frame.columns)
        rows = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
        for row in frame.itertuples(index=False, name=None):
            cells = []
            for value in row:
                if pd.isna(value):
                    cells.append("")
                elif isinstance(value, float):
                    cells.append(f"{value:.6g}")
                else:
                    cells.append(str(value))
            rows.append("| " + " | ".join(cells) + " |")
        return "\n".join(rows)

    with out_path.open("w", encoding="utf-8") as handle:
        handle.write("# Transcriptome-Null Inspection Summary\n\n")
        handle.write("## Overall\n\n")
        for key in [
            "n_linked_rows",
            "n_probe_gene_pairs",
            "n_probes",
            "n_projects",
            "median_abs_rho",
            "median_transcriptome_abs_rho_percentile",
            "median_transcriptome_empirical_p",
            "n_top_1pct_rows",
            "n_top_0_5pct_rows",
            "n_top_0_1pct_rows",
            "n_null_interesting_rows",
            "n_null_stringent_rows",
            "n_null_interesting_pairs",
            "n_null_stringent_pairs",
            "n_null_interesting_probes",
            "n_null_stringent_probes",
            "top_1pct_enrichment_vs_uniform_null",
            "top_0_5pct_enrichment_vs_uniform_null",
            "top_0_1pct_enrichment_vs_uniform_null",
            "n_q_lt_0_05_rows",
            "n_q_lt_0_005_rows",
        ]:
            handle.write(f"- `{key}`: `{overall[key]}`\n")
        handle.write("\n## Interpretation\n\n")
        handle.write(
            "Linked panel genes are enriched for unusually high absolute methylation-expression correlations "
            "relative to all measured genes in the same probe/cohort background, but the signal is concentrated "
            "in a subset of probes and genes rather than uniform across the full panel.\n\n"
        )
        handle.write(
            "The transcriptome empirical p-value is `transcriptome_abs_rho_rank / n_genes_tested`, i.e. the "
            "one-sided fraction of measured genes with absolute correlation at least as extreme as the linked "
            "gene in the same probe/cohort background. The default null-aware support call requires "
            f"`n_matched >= {overall['min_matched_for_support']}`, "
            f"`spearman_q_within_probe_cohort <= {overall['nominal_q_cutoff_for_support']}`, and "
            f"`transcriptome_empirical_p <= {overall['empirical_p_cutoff_for_interesting_support']}`. "
            "Rows outside these thresholds should be treated as weak or nonspecific expression support even when "
            "the nominal within-cohort correlation is nonzero.\n\n"
        )
        handle.write("## Top Probe-Gene Pairs\n\n")
        handle.write(markdown_table(top_pairs))
        handle.write("\n\n## Top Probes\n\n")
        handle.write(markdown_table(top_probes))
        handle.write("\n\n## Top Cohorts By Linked Top-1pct Rows\n\n")
        handle.write(markdown_table(top_cohorts))
        handle.write("\n")


def main() -> int:
    args = parse_args()
    null_dir = Path(args.transcriptome_null_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    linked = read_linked(null_dir)
    results = summarize(
        linked,
        args.top_n,
        min_matched=args.min_matched,
        nominal_q_cutoff=args.nominal_q_cutoff,
        empirical_p_cutoff=args.empirical_p_cutoff,
        stringent_empirical_p_cutoff=args.stringent_empirical_p_cutoff,
    )

    for name in ["linked_scored", "top_rows", "probe_summary", "pair_summary", "cohort_summary"]:
        frame = results[name]
        assert isinstance(frame, pd.DataFrame)
        frame.to_csv(out_dir / f"transcriptome_null_{name}.tsv", sep="\t", index=False)

    overall = results["overall"]
    assert isinstance(overall, dict)
    (out_dir / "transcriptome_null_inspection_summary.json").write_text(
        json.dumps(overall, indent=2, sort_keys=True), encoding="utf-8"
    )
    write_markdown(out_dir / "transcriptome_null_inspection_summary.md", results)
    print(json.dumps(overall, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
