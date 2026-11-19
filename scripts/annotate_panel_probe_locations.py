#!/usr/bin/env python3
"""Annotate a fixed probe panel with gene and CpG-island context metadata."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import List

import pandas as pd

NA_TOKENS = {"", "na", "nan", "none", "."}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Annotate panel probes using HM450 manifest.")
    parser.add_argument("--panel-tsv", required=True, help="Panel TSV with probe_id column (optionally panel_rank).")
    parser.add_argument("--annotation-tsv", required=True, help="HM450 annotation TSV path.")
    parser.add_argument("--out-dir", required=True, help="Output directory.")
    parser.add_argument("--k", type=int, default=30, help="Number of panel probes to keep (after rank ordering).")
    return parser.parse_args()


def _is_na(x: object) -> bool:
    if x is None:
        return True
    s = str(x).strip().lower()
    return s in NA_TOKENS


def _split_items(raw: object) -> List[str]:
    if _is_na(raw):
        return []
    parts = [p.strip() for p in re.split(r"[;,|]", str(raw))]
    out = []
    for p in parts:
        if p and p.lower() not in NA_TOKENS:
            out.append(p)
    return out


def _context_group(cgiposition: object, cgi: object) -> tuple[str, str]:
    raw = "" if _is_na(cgiposition) else str(cgiposition).strip()
    if raw:
        low = raw.lower()
        if "shore" in low:
            return raw, "Shore"
        if "shelf" in low:
            return raw, "Shelf"
        if "island" in low:
            return raw, "Island"
        return raw, "Other"

    cgi_raw = "" if _is_na(cgi) else str(cgi).strip()
    if cgi_raw:
        return cgi_raw, "Island"
    return "OpenSea", "OpenSea"


def main() -> int:
    args = parse_args()
    panel_path = Path(args.panel_tsv)
    ann_path = Path(args.annotation_tsv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.k < 1:
        raise SystemExit("--k must be >= 1")
    if not panel_path.exists():
        raise SystemExit(f"panel-tsv not found: {panel_path}")
    if not ann_path.exists():
        raise SystemExit(f"annotation-tsv not found: {ann_path}")

    panel = pd.read_csv(panel_path, sep="\t", dtype=str)
    if "probe_id" not in panel.columns:
        raise SystemExit(f"panel-tsv missing probe_id column: {panel_path}")
    if "panel_rank" in panel.columns:
        panel["panel_rank"] = pd.to_numeric(panel["panel_rank"], errors="coerce")
        panel = panel.sort_values(["panel_rank", "probe_id"], kind="mergesort")
    panel = panel.copy()
    panel["probe_id"] = panel["probe_id"].astype(str)
    panel = panel.dropna(subset=["probe_id"])
    panel = panel[panel["probe_id"] != ""].head(args.k).copy()
    panel["panel_rank"] = range(1, len(panel) + 1)
    probe_set = set(panel["probe_id"].tolist())

    usecols = [
        "probeID",
        "CpG_chrm",
        "CpG_beg",
        "CpG_end",
        "genesUniq",
        "geneNames",
        "distToTSS",
        "CGI",
        "CGIposition",
    ]
    ann_parts: list[pd.DataFrame] = []
    found: set[str] = set()
    for chunk in pd.read_csv(ann_path, sep="\t", usecols=usecols, dtype=str, chunksize=100_000):
        chunk = chunk.rename(columns={"probeID": "probe_id"})
        sub = chunk[chunk["probe_id"].isin(probe_set)].copy()
        if sub.empty:
            continue
        ann_parts.append(sub)
        found.update(sub["probe_id"].astype(str).tolist())
        if found >= probe_set:
            break
    ann = pd.concat(ann_parts, ignore_index=True) if ann_parts else pd.DataFrame(columns=["probe_id", *usecols[1:]])

    merged = panel.merge(ann, on="probe_id", how="left")
    merged = merged.sort_values("panel_rank").reset_index(drop=True)

    gene_lists = []
    raw_ctx = []
    grp_ctx = []
    for row in merged.itertuples(index=False):
        genes = _split_items(getattr(row, "genesUniq", None))
        if not genes:
            genes = _split_items(getattr(row, "geneNames", None))
        gene_lists.append(genes)
        raw, grp = _context_group(getattr(row, "CGIposition", None), getattr(row, "CGI", None))
        raw_ctx.append(raw)
        grp_ctx.append(grp)

    merged["gene_list"] = [";".join(g) for g in gene_lists]
    merged["n_genes"] = [len(g) for g in gene_lists]
    merged["context_raw"] = raw_ctx
    merged["context_group"] = grp_ctx

    # Exploded probe-gene table
    gene_rows = []
    for row in merged.itertuples(index=False):
        genes = _split_items(getattr(row, "gene_list", None))
        if not genes:
            genes = ["__NO_GENE__"]
        for g in genes:
            gene_rows.append(
                {
                    "panel_rank": int(row.panel_rank),
                    "probe_id": str(row.probe_id),
                    "gene": g,
                }
            )
    gene_long = pd.DataFrame(gene_rows).sort_values(["panel_rank", "gene"]).reset_index(drop=True)
    gene_summary = (
        gene_long[gene_long["gene"] != "__NO_GENE__"]
        .groupby("gene")
        .agg(n_probes=("probe_id", "nunique"))
        .reset_index()
        .sort_values(["n_probes", "gene"], ascending=[False, True])
        .reset_index(drop=True)
    )

    context_summary = (
        merged.groupby("context_group")
        .agg(n_probes=("probe_id", "nunique"))
        .reset_index()
        .sort_values(["n_probes", "context_group"], ascending=[False, True])
        .reset_index(drop=True)
    )
    context_summary["fraction"] = context_summary["n_probes"] / float(len(merged))

    context_raw_summary = (
        merged.groupby(["context_group", "context_raw"])
        .agg(n_probes=("probe_id", "nunique"))
        .reset_index()
        .sort_values(["n_probes", "context_group", "context_raw"], ascending=[False, True, True])
        .reset_index(drop=True)
    )
    context_raw_summary["fraction"] = context_raw_summary["n_probes"] / float(len(merged))

    chrom_summary = (
        merged.groupby("CpG_chrm")
        .agg(n_probes=("probe_id", "nunique"))
        .reset_index()
        .sort_values(["n_probes", "CpG_chrm"], ascending=[False, True])
        .reset_index(drop=True)
    )
    chrom_summary["fraction"] = chrom_summary["n_probes"] / float(len(merged))

    missing = sorted(probe_set - set(ann["probe_id"].astype(str).tolist()))

    summary = {
        "panel_tsv": str(panel_path),
        "annotation_tsv": str(ann_path),
        "k_requested": int(args.k),
        "n_panel_probes": int(len(merged)),
        "n_annotated_probes": int(len(merged) - len(missing)),
        "n_missing_probes": int(len(missing)),
        "missing_probes": missing,
        "n_unique_genes": int(gene_summary["gene"].nunique() if not gene_summary.empty else 0),
        "n_no_gene_probes": int((merged["n_genes"] == 0).sum()),
        "context_counts": {
            str(r.context_group): int(r.n_probes) for r in context_summary.itertuples(index=False)
        },
    }

    merged.to_csv(out_dir / "panel_probe_annotations.tsv", sep="\t", index=False)
    gene_long.to_csv(out_dir / "panel_probe_gene_long.tsv", sep="\t", index=False)
    gene_summary.to_csv(out_dir / "panel_gene_summary.tsv", sep="\t", index=False)
    context_summary.to_csv(out_dir / "panel_context_group_summary.tsv", sep="\t", index=False)
    context_raw_summary.to_csv(out_dir / "panel_context_raw_summary.tsv", sep="\t", index=False)
    chrom_summary.to_csv(out_dir / "panel_chromosome_summary.tsv", sep="\t", index=False)
    (out_dir / "panel_annotation_summary.json").write_text(json.dumps(summary, indent=2))

    print(f"Wrote panel annotations to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
