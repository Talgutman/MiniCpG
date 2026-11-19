#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats


TUMOR_GROUP = "primary_tumor"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "For each TCGA cohort and each panel CpG, correlate tumor methylation against all "
            "available RNA-seq genes and rank the annotated linked genes against that "
            "transcriptome-wide background."
        )
    )
    parser.add_argument("--probe-gene-pairs-tsv", required=True)
    parser.add_argument("--expression-metadata-tsv", required=True)
    parser.add_argument("--tcga-dev-matrix-tsv", required=True)
    parser.add_argument("--tcga-lockbox-matrix-tsv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--expression-field", default="tpm_unstranded")
    parser.add_argument("--min-matched", type=int, default=30)
    parser.add_argument("--min-gene-nonzero-fraction", type=float, default=0.05)
    parser.add_argument("--max-projects", type=int, default=0, help="Debug limit after sorting projects.")
    parser.add_argument("--focus-projects", default="", help="Optional comma-separated TCGA project IDs.")
    parser.add_argument("--project-list-out", default="", help="Write eligible projects and exit.")
    parser.add_argument(
        "--shard-by-project",
        action="store_true",
        help="Write per-project output shards under out-dir instead of combined final outputs.",
    )
    parser.add_argument(
        "--save-full-correlations-parquet",
        action="store_true",
        help="Write all probe-gene-cohort correlations to compressed Parquet.",
    )
    parser.add_argument(
        "--top-n-per-probe-project",
        type=int,
        default=100,
        help="Number of top |rho| transcriptome hits to save per probe/cohort.",
    )
    return parser.parse_args()


def split_semicolon(value: object) -> list[str]:
    if not isinstance(value, str):
        return []
    out: list[str] = []
    for item in value.split(";"):
        item = item.strip()
        if item and item.lower() != "nan" and item not in out:
            out.append(item)
    return out


def bh_fdr(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=float)
    q = np.full(p.shape, np.nan, dtype=float)
    valid = np.where(np.isfinite(p))[0]
    if valid.size == 0:
        return q
    order = valid[np.argsort(p[valid])]
    ranked = p[order]
    n = ranked.size
    adjusted = ranked * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    q[order] = np.clip(adjusted, 0.0, 1.0)
    return q


def primary_file_priority(sample_submitter_id: str, file_uuid: str) -> tuple[int, str, str]:
    sid = str(sample_submitter_id or "")
    suffix = sid.split("-")[3] if len(sid.split("-")) > 3 else ""
    vial = suffix[2:3] if len(suffix) >= 3 else "Z"
    return (0 if suffix.startswith("01") else 1, vial, str(file_uuid))


def load_methylation(dev_path: Path, lockbox_path: Path) -> pd.DataFrame:
    frames = []
    for split, path in [("dev", dev_path), ("lockbox", lockbox_path)]:
        df = pd.read_csv(path, sep="\t")
        df["split"] = split
        frames.append(df)
    meth = pd.concat(frames, ignore_index=True)
    meth["case_submitter_id"] = meth["sample_id"].astype(str).str.slice(0, 12)
    return meth.drop_duplicates(["case_submitter_id", "project_id"], keep="first")


def select_expression_files(metadata: pd.DataFrame, project: str, cases: set[str]) -> pd.DataFrame:
    use = metadata.loc[
        (metadata["project_id"].astype(str) == project)
        & (metadata["sample_group"].astype(str) == TUMOR_GROUP)
        & (metadata["case_submitter_id"].astype(str).isin(cases))
        & (metadata["expression_file_path"].astype(str).str.len() > 0)
    ].copy()
    if use.empty:
        return use
    use["_priority"] = [
        primary_file_priority(sid, fid)
        for sid, fid in zip(use["sample_submitter_id"].astype(str), use["file_uuid"].astype(str))
    ]
    use = use.sort_values(["case_submitter_id", "_priority"])
    return use.drop_duplicates("case_submitter_id", keep="first").drop(columns=["_priority"])


def read_expression_file(path: Path, expression_field: str) -> tuple[list[str], list[str], np.ndarray]:
    genes: list[str] = []
    gene_types: list[str] = []
    values: list[float] = []
    with path.open("r", encoding="utf-8") as handle:
        header: list[str] | None = None
        gene_name_idx = -1
        gene_type_idx = -1
        value_idx = -1
        for line in handle:
            if not line.strip() or line.startswith("#"):
                continue
            parts = line.rstrip("\n").split("\t")
            if header is None:
                header = parts
                gene_name_idx = header.index("gene_name")
                gene_type_idx = header.index("gene_type") if "gene_type" in header else -1
                value_idx = header.index(expression_field)
                continue
            gene = parts[gene_name_idx] if gene_name_idx < len(parts) else ""
            if not gene or gene.startswith("__"):
                continue
            try:
                val = float(parts[value_idx])
            except (ValueError, IndexError):
                continue
            genes.append(gene)
            gene_types.append(parts[gene_type_idx] if 0 <= gene_type_idx < len(parts) else "")
            values.append(math.log2(val + 1.0))
    return genes, gene_types, np.asarray(values, dtype=np.float32)


def load_project_expression_matrix(files: pd.DataFrame, expression_field: str) -> tuple[list[str], list[str], np.ndarray, list[str]]:
    genes: list[str] = []
    gene_types: list[str] = []
    columns: list[np.ndarray] = []
    case_ids: list[str] = []
    gene_to_index: dict[str, int] | None = None
    for row in files.itertuples(index=False):
        path = Path(str(row.expression_file_path))
        if not path.exists():
            continue
        cur_genes, cur_types, cur_values = read_expression_file(path, expression_field)
        if not columns:
            genes = cur_genes
            gene_types = cur_types
            gene_to_index = {gene: idx for idx, gene in enumerate(genes)}
            columns.append(cur_values.astype(np.float32))
            case_ids.append(str(row.case_submitter_id))
            continue
        if cur_genes == genes:
            columns.append(cur_values.astype(np.float32))
        else:
            assert gene_to_index is not None
            aligned = np.full(len(genes), np.nan, dtype=np.float32)
            for gene, val in zip(cur_genes, cur_values):
                idx = gene_to_index.get(gene)
                if idx is not None:
                    aligned[idx] = val
            columns.append(aligned)
        case_ids.append(str(row.case_submitter_id))
    if not columns:
        return [], [], np.empty((0, 0), dtype=np.float32), []
    matrix = np.column_stack(columns).astype(np.float32)
    return genes, gene_types, matrix, case_ids


def correlation_against_all_genes(beta: np.ndarray, expr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid_beta = np.isfinite(beta)
    if valid_beta.sum() < 3:
        n_genes = expr.shape[0]
        return np.full(n_genes, np.nan), np.full(n_genes, np.nan)
    expr_use = expr[:, valid_beta]
    beta_use = beta[valid_beta]
    beta_rank = stats.rankdata(beta_use).astype(np.float32)
    beta_centered = beta_rank - beta_rank.mean()
    beta_ss = float(np.sum(beta_centered * beta_centered))
    if beta_ss <= 0:
        n_genes = expr.shape[0]
        return np.full(n_genes, np.nan), np.full(n_genes, np.nan)
    expr_rank = stats.rankdata(expr_use, axis=1).astype(np.float32)
    expr_rank -= expr_rank.mean(axis=1, keepdims=True)
    expr_ss = np.sum(expr_rank * expr_rank, axis=1)
    denom = np.sqrt(expr_ss * beta_ss)
    rho = np.divide(
        expr_rank @ beta_centered,
        denom,
        out=np.full(expr_rank.shape[0], np.nan, dtype=np.float32),
        where=denom > 0,
    ).astype(float)
    n = beta_use.size
    p = np.full(rho.shape, np.nan, dtype=float)
    finite = np.isfinite(rho) & (np.abs(rho) < 1)
    t_stat = rho[finite] * np.sqrt((n - 2) / np.maximum(1 - rho[finite] ** 2, np.finfo(float).tiny))
    p[finite] = 2.0 * stats.t.sf(np.abs(t_stat), df=n - 2)
    p[np.isfinite(rho) & (np.abs(rho) >= 1)] = 0.0
    return rho, p


def ranked_expression_cache(expr: np.ndarray, valid_beta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    expr_use = expr[:, valid_beta]
    expr_rank = stats.rankdata(expr_use, axis=1).astype(np.float32)
    expr_rank -= expr_rank.mean(axis=1, keepdims=True)
    expr_ss = np.sum(expr_rank * expr_rank, axis=1)
    return expr_rank, expr_ss


def correlation_against_ranked_genes(
    beta: np.ndarray,
    valid_beta: np.ndarray,
    expr_rank: np.ndarray,
    expr_ss: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    n_genes = expr_rank.shape[0]
    n = int(valid_beta.sum())
    if n < 3:
        return np.full(n_genes, np.nan), np.full(n_genes, np.nan)
    beta_use = beta[valid_beta]
    beta_rank = stats.rankdata(beta_use).astype(np.float32)
    beta_centered = beta_rank - beta_rank.mean()
    beta_ss = float(np.sum(beta_centered * beta_centered))
    if beta_ss <= 0:
        return np.full(n_genes, np.nan), np.full(n_genes, np.nan)
    denom = np.sqrt(expr_ss * beta_ss)
    rho = np.divide(
        expr_rank @ beta_centered,
        denom,
        out=np.full(n_genes, np.nan, dtype=np.float32),
        where=denom > 0,
    ).astype(float)
    p = np.full(rho.shape, np.nan, dtype=float)
    finite = np.isfinite(rho) & (np.abs(rho) < 1)
    t_stat = rho[finite] * np.sqrt((n - 2) / np.maximum(1 - rho[finite] ** 2, np.finfo(float).tiny))
    p[finite] = 2.0 * stats.t.sf(np.abs(t_stat), df=n - 2)
    p[np.isfinite(rho) & (np.abs(rho) >= 1)] = 0.0
    return rho, p


def build_full_project_frame(
    project: str,
    probe: str,
    panel_rank: object,
    n_matched: int,
    genes: np.ndarray,
    gene_types: np.ndarray,
    rho: np.ndarray,
    pvalue: np.ndarray,
    qvalue: np.ndarray,
    rank: np.ndarray,
    percentile: np.ndarray,
    linked_genes: set[str],
    n_matched_project: int | None = None,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "project_id": project,
            "probe_id": probe,
            "panel_rank": panel_rank,
            "gene": genes,
            "gene_type": gene_types,
            "n_matched": np.int32(n_matched),
            "n_used_for_correlation": np.int32(n_matched),
            "n_matched_project": np.int32(n_matched if n_matched_project is None else n_matched_project),
            "spearman_beta_expr": rho.astype(np.float32),
            "spearman_p_value": pvalue.astype(np.float32),
            "spearman_q_within_probe_cohort": qvalue.astype(np.float32),
            "transcriptome_abs_rho_rank": rank.astype(np.int32),
            "transcriptome_abs_rho_percentile": percentile.astype(np.float32),
            "is_linked_probe_gene": np.fromiter((str(g) in linked_genes for g in genes), dtype=bool, count=len(genes)),
        }
    )


def plot_rank_summary(rows: pd.DataFrame, out_png: Path) -> None:
    sig = rows.loc[rows["transcriptome_abs_rho_percentile"].notna()].copy()
    if sig.empty:
        return
    sig["cohort"] = sig["project_id"].astype(str).str.replace("TCGA-", "", regex=False)
    sig["row_label"] = (
        sig["panel_rank"].astype("Int64").astype(str)
        + ". "
        + sig["probe_id"].astype(str)
        + " / "
        + sig["gene"].astype(str)
    )
    row_order = sig.sort_values(["panel_rank", "probe_id", "gene"])["row_label"].drop_duplicates().tolist()
    col_order = sorted(sig["cohort"].drop_duplicates())
    sig["x"] = sig["cohort"].map({c: i for i, c in enumerate(col_order)})
    sig["y"] = sig["row_label"].map({r: i for i, r in enumerate(row_order)})
    sig["size"] = 16 + 220 * sig["transcriptome_abs_rho_percentile"].clip(0, 1) ** 4

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.facecolor": "#fbf7ef",
            "figure.facecolor": "#fbf7ef",
            "savefig.facecolor": "#fbf7ef",
        }
    )
    fig, ax = plt.subplots(figsize=(15, max(8, 0.28 * len(row_order) + 2)), constrained_layout=True)
    sc = ax.scatter(
        sig["x"],
        sig["y"],
        c=sig["spearman_beta_expr"],
        s=sig["size"],
        cmap="BrBG",
        vmin=-0.8,
        vmax=0.8,
        edgecolor="#302923",
        linewidth=0.35,
        alpha=0.9,
    )
    cbar = fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.01)
    cbar.set_label("Linked-gene Spearman rho")
    ax.set_xticks(range(len(col_order)))
    ax.set_xticklabels(col_order, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(row_order)))
    ax.set_yticklabels(row_order, fontsize=7)
    ax.invert_yaxis()
    ax.grid(True, color="#dfd5c3", linewidth=0.6, alpha=0.9)
    ax.set_title(
        "Linked gene rank in transcriptome-wide methylation-expression background",
        loc="left",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xlabel("TCGA cohort")
    ax.set_ylabel("Probe / linked gene")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def eligible_projects(meth: pd.DataFrame, metadata: pd.DataFrame, min_matched: int) -> pd.DataFrame:
    rows = []
    for project, meth_project in meth.groupby("project_id", sort=True):
        files = select_expression_files(metadata, project, set(meth_project["case_submitter_id"].astype(str)))
        rows.append(
            {
                "project_id": project,
                "n_methylation_cases": int(meth_project["case_submitter_id"].nunique()),
                "n_matched_expression_files": int(len(files)),
                "eligible": int(len(files) >= min_matched),
            }
        )
    return pd.DataFrame(rows).sort_values("project_id")


def write_outputs(
    out_dir: Path,
    linked: pd.DataFrame,
    project_summaries: list[dict[str, object]],
    full_frames: list[pd.DataFrame],
    top_frames: list[pd.DataFrame],
    args: argparse.Namespace,
    shard_label: str = "",
) -> None:
    if args.shard_by_project:
        label = shard_label or "unknown"
        shard_dir = out_dir / "project_shards" / label
        if shard_dir.exists():
            shutil.rmtree(shard_dir)
        shard_dir.mkdir(parents=True, exist_ok=True)
        linked.to_csv(shard_dir / "linked_probe_gene_transcriptome_rank_by_project.tsv", sep="\t", index=False)
        pd.DataFrame(project_summaries).to_csv(shard_dir / "transcriptome_null_project_summary.tsv", sep="\t", index=False)
        if full_frames:
            pd.concat(full_frames, ignore_index=True).to_parquet(
                shard_dir / "all_probe_gene_transcriptome_correlations.parquet",
                index=False,
                compression="zstd",
            )
        if top_frames:
            pd.concat(top_frames, ignore_index=True).to_csv(
                shard_dir / "top_transcriptome_hits_by_probe_project.tsv", sep="\t", index=False
            )
        if not linked.empty:
            linked.loc[linked["is_top_1pct_abs_rho"]].to_csv(
                shard_dir / "linked_probe_gene_transcriptome_top1pct.tsv", sep="\t", index=False
            )
        summary = build_summary(linked, project_summaries, args, shard_dir)
        return

    linked.to_csv(out_dir / "linked_probe_gene_transcriptome_rank_by_project.tsv", sep="\t", index=False)
    pd.DataFrame(project_summaries).to_csv(out_dir / "transcriptome_null_project_summary.tsv", sep="\t", index=False)
    if full_frames:
        full = pd.concat(full_frames, ignore_index=True)
        full.to_parquet(out_dir / "all_probe_gene_transcriptome_correlations.parquet", index=False, compression="zstd")
    if top_frames:
        top_hits = pd.concat(top_frames, ignore_index=True)
        top_hits.to_csv(out_dir / "top_transcriptome_hits_by_probe_project.tsv", sep="\t", index=False)
    build_summary(linked, project_summaries, args, out_dir)
    plot_rank_summary(linked, out_dir / "linked_gene_transcriptome_rank_dotplot.png")


def build_summary(
    linked: pd.DataFrame,
    project_summaries: list[dict[str, object]],
    args: argparse.Namespace,
    out_dir: Path,
) -> dict[str, object]:
    summary = {
        "min_matched": args.min_matched,
        "min_gene_nonzero_fraction": args.min_gene_nonzero_fraction,
        "n_projects_requested": int(len(project_summaries)),
        "n_projects_ok": int(sum(1 for x in project_summaries if x.get("status") == "ok")),
        "n_linked_rows": int(len(linked)),
        "n_top_1pct_rows": int(linked["is_top_1pct_abs_rho"].sum()) if not linked.empty else 0,
        "n_top_0_5pct_rows": int(linked["is_top_0_5pct_abs_rho"].sum()) if not linked.empty else 0,
        "n_top_0_1pct_rows": int(linked["is_top_0_1pct_abs_rho"].sum()) if not linked.empty else 0,
        "median_transcriptome_abs_rho_percentile": float(linked["transcriptome_abs_rho_percentile"].median()) if not linked.empty else None,
        "full_correlations_parquet": str(out_dir / "all_probe_gene_transcriptome_correlations.parquet")
        if args.save_full_correlations_parquet
        else "",
        "top_hits_tsv": str(out_dir / "top_transcriptome_hits_by_probe_project.tsv")
        if args.top_n_per_probe_project > 0
        else "",
    }
    (out_dir / "transcriptome_null_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    with (out_dir / "transcriptome_null_summary.md").open("w", encoding="utf-8") as handle:
        handle.write("# Transcriptome-Wide Methylation-Expression Null Summary\n\n")
        for key, value in summary.items():
            handle.write(f"- `{key}`: `{value}`\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs = pd.read_csv(args.probe_gene_pairs_tsv, sep="\t")
    metadata = pd.read_csv(args.expression_metadata_tsv, sep="\t")
    meth = load_methylation(Path(args.tcga_dev_matrix_tsv), Path(args.tcga_lockbox_matrix_tsv))
    if args.project_list_out:
        projects_df = eligible_projects(meth, metadata, args.min_matched)
        project_list_out = Path(args.project_list_out)
        project_list_out.parent.mkdir(parents=True, exist_ok=True)
        projects_df.to_csv(project_list_out, sep="\t", index=False)
        print(f"Wrote {project_list_out}")
        return 0
    probe_ids = [c for c in meth.columns if str(c).startswith("cg")]
    linked_by_probe = {
        probe: split_semicolon(";".join(sub["gene"].astype(str)))
        for probe, sub in pairs.groupby("probe_id", sort=False)
    }
    pair_meta = pairs.drop_duplicates(["probe_id", "gene"]).set_index(["probe_id", "gene"])

    projects = sorted(meth["project_id"].dropna().unique())
    focus = [x.strip() for x in args.focus_projects.split(",") if x.strip()]
    if focus:
        projects = [p for p in projects if p in set(focus)]
    if args.max_projects:
        projects = projects[: args.max_projects]

    rows: list[dict[str, object]] = []
    full_frames: list[pd.DataFrame] = []
    top_frames: list[pd.DataFrame] = []
    project_summaries: list[dict[str, object]] = []
    for project in projects:
        meth_project = meth.loc[meth["project_id"] == project].copy()
        files = select_expression_files(metadata, project, set(meth_project["case_submitter_id"].astype(str)))
        if len(files) < args.min_matched:
            project_summaries.append({"project_id": project, "status": "too_few_matched_expression_files", "n_files": len(files)})
            continue
        genes, gene_types, expr_matrix, case_ids = load_project_expression_matrix(files, args.expression_field)
        if expr_matrix.size == 0:
            project_summaries.append({"project_id": project, "status": "no_expression_matrix", "n_files": len(files)})
            continue
        case_order = pd.Series(range(len(case_ids)), index=case_ids)
        meth_project = meth_project.loc[meth_project["case_submitter_id"].isin(case_order.index)].copy()
        meth_project["_order"] = meth_project["case_submitter_id"].map(case_order)
        meth_project = meth_project.sort_values("_order")
        expr_matrix = expr_matrix[:, meth_project["_order"].to_numpy(dtype=int)]
        nonzero_fraction = np.mean(expr_matrix > 0, axis=1)
        variable = np.nanstd(expr_matrix, axis=1) > 0
        gene_keep = (nonzero_fraction >= args.min_gene_nonzero_fraction) & variable
        genes_arr = np.asarray(genes, dtype=object)[gene_keep]
        gene_types_arr = np.asarray(gene_types, dtype=object)[gene_keep]
        expr_keep = expr_matrix[gene_keep, :]
        gene_to_idx = {str(gene): idx for idx, gene in enumerate(genes_arr)}
        n_genes = len(genes_arr)
        if n_genes == 0:
            project_summaries.append({"project_id": project, "status": "no_genes_after_filter", "n_files": len(files)})
            continue
        expression_rank_cache: dict[bytes, tuple[np.ndarray, np.ndarray]] = {}
        for probe in probe_ids:
            beta = pd.to_numeric(meth_project[probe], errors="coerce").to_numpy(dtype=float)
            valid_beta = np.isfinite(beta)
            if int(valid_beta.sum()) < 3:
                rho = np.full(n_genes, np.nan)
                pvalue = np.full(n_genes, np.nan)
            else:
                cache_key = valid_beta.tobytes()
                if cache_key not in expression_rank_cache:
                    expression_rank_cache[cache_key] = ranked_expression_cache(expr_keep, valid_beta)
                expr_rank, expr_ss = expression_rank_cache[cache_key]
                rho, pvalue = correlation_against_ranked_genes(beta, valid_beta, expr_rank, expr_ss)
            qvalue = bh_fdr(pvalue)
            abs_rho = np.abs(rho)
            finite = np.isfinite(abs_rho)
            if not finite.any():
                continue
            order_desc = np.argsort(-np.where(finite, abs_rho, -np.inf))
            rank = np.empty(n_genes, dtype=float)
            rank[:] = np.nan
            rank[order_desc] = np.arange(1, n_genes + 1)
            percentile = 1.0 - (rank - 1.0) / max(n_genes - 1, 1)
            rank_int = rank.astype(np.int32)
            linked_genes = set(linked_by_probe.get(probe, []))
            panel_rank = pairs.loc[pairs["probe_id"] == probe, "panel_rank"].dropna()
            panel_rank_value = int(panel_rank.iloc[0]) if not panel_rank.empty else np.nan
            if args.save_full_correlations_parquet:
                full_frames.append(
                    build_full_project_frame(
                        project,
                        probe,
                        panel_rank_value,
                        int(valid_beta.sum()),
                        genes_arr,
                        gene_types_arr,
                        rho,
                        pvalue,
                        qvalue,
                        rank_int,
                        percentile,
                        linked_genes,
                        n_matched_project=len(meth_project),
                    )
                )
            if args.top_n_per_probe_project > 0:
                top_idx = order_desc[: min(args.top_n_per_probe_project, len(order_desc))]
                top_frames.append(
                    build_full_project_frame(
                        project,
                        probe,
                        panel_rank_value,
                        int(valid_beta.sum()),
                        genes_arr[top_idx],
                        gene_types_arr[top_idx],
                        rho[top_idx],
                        pvalue[top_idx],
                        qvalue[top_idx],
                        rank_int[top_idx],
                        percentile[top_idx],
                        linked_genes,
                        n_matched_project=len(meth_project),
                    )
                )
            for gene in linked_by_probe.get(probe, []):
                idx = gene_to_idx.get(gene)
                if idx is None:
                    continue
                meta = pair_meta.loc[(probe, gene)] if (probe, gene) in pair_meta.index else pd.Series(dtype=object)
                rows.append(
                    {
                        "project_id": project,
                        "probe_id": probe,
                        "gene": gene,
                        "panel_rank": meta.get("panel_rank", np.nan),
                        # Correlations omit missing beta values. Retain the project
                        # overlap separately so it cannot be mistaken for test N.
                        "n_matched": int(valid_beta.sum()),
                        "n_used_for_correlation": int(valid_beta.sum()),
                        "n_matched_project": int(len(meth_project)),
                        "n_genes_tested": int(n_genes),
                        "gene_type": gene_types_arr[idx],
                        "spearman_beta_expr": float(rho[idx]),
                        "spearman_p_value": float(pvalue[idx]),
                        "spearman_q_within_probe_cohort": float(qvalue[idx]),
                        "transcriptome_abs_rho_rank": int(rank[idx]),
                        "transcriptome_abs_rho_percentile": float(percentile[idx]),
                        "is_top_1pct_abs_rho": bool(percentile[idx] >= 0.99),
                        "is_top_0_5pct_abs_rho": bool(percentile[idx] >= 0.995),
                        "is_top_0_1pct_abs_rho": bool(percentile[idx] >= 0.999),
                    }
                )
        project_summaries.append(
            {
                "project_id": project,
                "status": "ok",
                "n_files": int(len(files)),
                "n_genes_tested": int(n_genes),
            }
        )
        print(f"Finished {project}: n={len(files)}, genes={n_genes}", flush=True)

    linked = pd.DataFrame(rows)
    if not linked.empty:
        linked = linked.sort_values(["panel_rank", "probe_id", "gene", "project_id"])
    if not linked.empty:
        top = linked.loc[linked["is_top_1pct_abs_rho"]].copy()
        if args.shard_by_project:
            pass
        else:
            top.to_csv(out_dir / "linked_probe_gene_transcriptome_top1pct.tsv", sep="\t", index=False)
    shard_label = projects[0] if len(projects) == 1 else ""
    write_outputs(out_dir, linked, project_summaries, full_frames, top_frames, args, shard_label=shard_label)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
