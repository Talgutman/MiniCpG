#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from analyze_panel_meth_expr_transcriptome_null import plot_rank_summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate per-cohort transcriptome-wide methylation-expression null shards."
    )
    parser.add_argument("--out-dir", required=True, help="Transcriptome-null root output directory.")
    parser.add_argument("--shard-root", default="", help="Defaults to out-dir/project_shards.")
    return parser.parse_args()


def read_tsv_if_present(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        return pd.DataFrame()
    try:
        return pd.read_csv(path, sep="\t")
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def parquet_num_rows(path: Path) -> int | None:
    if not path.exists():
        return None
    try:
        import pyarrow.parquet as pq

        return int(pq.ParquetFile(path).metadata.num_rows)
    except Exception:
        return None


def concat_frames(frames: list[pd.DataFrame]) -> pd.DataFrame:
    nonempty = [frame for frame in frames if not frame.empty]
    if not nonempty:
        return pd.DataFrame()
    return pd.concat(nonempty, ignore_index=True)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    shard_root = Path(args.shard_root) if args.shard_root else out_dir / "project_shards"
    out_dir.mkdir(parents=True, exist_ok=True)

    shard_dirs = sorted(path for path in shard_root.iterdir() if path.is_dir()) if shard_root.exists() else []
    if not shard_dirs:
        raise SystemExit(f"No shard directories found under {shard_root}")

    linked_frames: list[pd.DataFrame] = []
    summary_frames: list[pd.DataFrame] = []
    top_frames: list[pd.DataFrame] = []
    parquet_rows: list[dict[str, object]] = []

    for shard_dir in shard_dirs:
        project_id = shard_dir.name
        linked_frames.append(read_tsv_if_present(shard_dir / "linked_probe_gene_transcriptome_rank_by_project.tsv"))
        summary_frames.append(read_tsv_if_present(shard_dir / "transcriptome_null_project_summary.tsv"))
        top_frames.append(read_tsv_if_present(shard_dir / "top_transcriptome_hits_by_probe_project.tsv"))

        parquet_path = shard_dir / "all_probe_gene_transcriptome_correlations.parquet"
        parquet_rows.append(
            {
                "project_id": project_id,
                "parquet_path": str(parquet_path),
                "exists": int(parquet_path.exists()),
                "file_size_bytes": int(parquet_path.stat().st_size) if parquet_path.exists() else 0,
                "n_rows": parquet_num_rows(parquet_path),
            }
        )

    linked = concat_frames(linked_frames)
    if not linked.empty:
        linked = linked.sort_values(["panel_rank", "probe_id", "gene", "project_id"])
    linked.to_csv(out_dir / "linked_probe_gene_transcriptome_rank_by_project.tsv", sep="\t", index=False)
    if not linked.empty:
        linked.loc[linked["is_top_1pct_abs_rho"]].to_csv(
            out_dir / "linked_probe_gene_transcriptome_top1pct.tsv", sep="\t", index=False
        )
        plot_rank_summary(linked, out_dir / "linked_gene_transcriptome_rank_dotplot.png")

    project_summary = concat_frames(summary_frames)
    if not project_summary.empty:
        project_summary = project_summary.sort_values("project_id")
    project_summary.to_csv(out_dir / "transcriptome_null_project_summary.tsv", sep="\t", index=False)

    top_hits = concat_frames(top_frames)
    if not top_hits.empty:
        top_hits = top_hits.sort_values(["project_id", "probe_id", "transcriptome_abs_rho_rank"])
    top_hits.to_csv(out_dir / "top_transcriptome_hits_by_probe_project.tsv", sep="\t", index=False)

    parquet_manifest = pd.DataFrame(parquet_rows).sort_values("project_id")
    parquet_manifest.to_csv(out_dir / "all_probe_gene_transcriptome_correlations_manifest.tsv", sep="\t", index=False)

    summary = {
        "n_project_shards": int(len(shard_dirs)),
        "n_projects_ok": int((project_summary["status"] == "ok").sum()) if "status" in project_summary.columns else 0,
        "n_linked_rows": int(len(linked)),
        "n_top_1pct_rows": int(linked["is_top_1pct_abs_rho"].sum()) if not linked.empty else 0,
        "n_top_0_5pct_rows": int(linked["is_top_0_5pct_abs_rho"].sum()) if not linked.empty else 0,
        "n_top_0_1pct_rows": int(linked["is_top_0_1pct_abs_rho"].sum()) if not linked.empty else 0,
        "median_transcriptome_abs_rho_percentile": float(linked["transcriptome_abs_rho_percentile"].median())
        if not linked.empty
        else None,
        "n_full_correlation_rows": int(parquet_manifest["n_rows"].fillna(0).sum()),
        "full_correlations_storage": "per-project parquet shards under project_shards/<PROJECT>/",
    }
    (out_dir / "transcriptome_null_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    with (out_dir / "transcriptome_null_summary.md").open("w", encoding="utf-8") as handle:
        handle.write("# Transcriptome-Wide Methylation-Expression Null Summary\n\n")
        for key, value in summary.items():
            handle.write(f"- `{key}`: `{value}`\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
