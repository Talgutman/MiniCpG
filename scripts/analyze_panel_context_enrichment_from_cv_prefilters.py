#!/usr/bin/env python3
"""Compare panel CpG-context composition against full-array and CV-prefilter backgrounds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd
from scipy.stats import binomtest, fisher_exact

CONTEXTS = ["Island", "Shore", "Shelf", "OpenSea", "Other"]
NA_TOKENS = {"", "na", "nan", "none", "."}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze panel CpG-island context enrichment using per-fold prefilter lists."
    )
    parser.add_argument("--annotation-tsv", required=True, help="HM450 annotation manifest TSV.")
    parser.add_argument("--panel-annotations-tsv", required=True, help="Annotated panel TSV with context_group.")
    parser.add_argument(
        "--prefilter-glob",
        required=True,
        help="Glob for per-fold prefilter TSV files (mi_prefilter_topk.tsv).",
    )
    parser.add_argument("--out-dir", required=True, help="Output directory.")
    return parser.parse_args()


def _is_na(x: object) -> bool:
    if x is None:
        return True
    return str(x).strip().lower() in NA_TOKENS


def _context_group(cgiposition: object, cgi: object) -> str:
    raw = "" if _is_na(cgiposition) else str(cgiposition).strip()
    if raw:
        low = raw.lower()
        if "shore" in low:
            return "Shore"
        if "shelf" in low:
            return "Shelf"
        if "island" in low:
            return "Island"
        return "Other"
    cgi_raw = "" if _is_na(cgi) else str(cgi).strip()
    if cgi_raw:
        return "Island"
    return "OpenSea"


def _series_counts(series: pd.Series) -> dict[str, int]:
    counts = {ctx: 0 for ctx in CONTEXTS}
    for key, value in series.value_counts().items():
        ctx = str(key)
        counts[ctx] = int(value)
    return counts


def _load_annotation_contexts(path: Path) -> pd.DataFrame:
    ann = pd.read_csv(
        path,
        sep="\t",
        usecols=["probeID", "CGI", "CGIposition"],
        dtype=str,
    ).rename(columns={"probeID": "probe_id"})
    ann["context_group"] = [
        _context_group(cgiposition, cgi)
        for cgiposition, cgi in zip(ann["CGIposition"], ann["CGI"], strict=False)
    ]
    ann = ann[["probe_id", "context_group"]].drop_duplicates("probe_id", keep="first")
    return ann


def _fisher(panel_count: int, panel_total: int, bg_count: int, bg_total: int) -> float:
    table = [
        [int(panel_count), int(panel_total - panel_count)],
        [int(bg_count), int(bg_total - bg_count)],
    ]
    return float(fisher_exact(table, alternative="two-sided").pvalue)


def main() -> int:
    args = parse_args()
    ann_path = Path(args.annotation_tsv)
    panel_path = Path(args.panel_annotations_tsv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not ann_path.exists():
        raise SystemExit(f"annotation-tsv not found: {ann_path}")
    if not panel_path.exists():
        raise SystemExit(f"panel-annotations-tsv not found: {panel_path}")

    prefilter_files = sorted(Path().glob(args.prefilter_glob))
    if not prefilter_files:
        raise SystemExit(f"No prefilter files matched: {args.prefilter_glob}")

    ann = _load_annotation_contexts(ann_path)
    ann_context = ann.set_index("probe_id")["context_group"]

    full_counts = _series_counts(ann["context_group"])
    full_total = int(len(ann))

    panel = pd.read_csv(panel_path, sep="\t", dtype=str)
    if "context_group" not in panel.columns:
        raise SystemExit(f"panel annotations missing context_group: {panel_path}")
    panel_counts = _series_counts(panel["context_group"])
    panel_total = int(len(panel))

    fold_rows: list[dict[str, object]] = []
    for path in prefilter_files:
        df = pd.read_csv(path, sep="\t", dtype=str)
        if "probe_id" not in df.columns:
            raise SystemExit(f"prefilter file missing probe_id column: {path}")
        ctx = df["probe_id"].astype(str).map(ann_context).fillna("Other")
        counts = _series_counts(ctx)
        total = int(len(df))
        rel = path.relative_to(prefilter_files[0].anchor) if path.is_absolute() else path
        row: dict[str, object] = {"prefilter_file": str(rel), "n_probes": total}
        for context in CONTEXTS:
            row[f"{context}_count"] = counts[context]
            row[f"{context}_fraction"] = counts[context] / float(total) if total else 0.0
        fold_rows.append(row)

    by_fold = pd.DataFrame(fold_rows).sort_values("prefilter_file").reset_index(drop=True)
    by_fold.to_csv(out_dir / "prefilter_context_by_fold.tsv", sep="\t", index=False)

    summary_rows: list[dict[str, object]] = []
    result_rows: list[dict[str, object]] = []
    for context in CONTEXTS:
        mean_frac = float(by_fold[f"{context}_fraction"].mean())
        std_frac = float(by_fold[f"{context}_fraction"].std(ddof=1))
        mean_count = float(by_fold[f"{context}_count"].mean())
        std_count = float(by_fold[f"{context}_count"].std(ddof=1))
        summary_rows.append(
            {
                "context_group": context,
                "n_folds": int(len(by_fold)),
                "prefilter_mean_count": mean_count,
                "prefilter_sd_count": std_count,
                "prefilter_mean_fraction": mean_frac,
                "prefilter_sd_fraction": std_frac,
                "full_count": int(full_counts[context]),
                "full_fraction": float(full_counts[context] / float(full_total)),
                "panel_count": int(panel_counts[context]),
                "panel_fraction": float(panel_counts[context] / float(panel_total)),
            }
        )
        result_rows.append(
            {
                "context_group": context,
                "full_fraction": float(full_counts[context] / float(full_total)),
                "prefilter_mean_fraction": mean_frac,
                "prefilter_sd_fraction": std_frac,
                "panel_fraction": float(panel_counts[context] / float(panel_total)),
                "full_count": int(full_counts[context]),
                "prefilter_mean_count": mean_count,
                "panel_count": int(panel_counts[context]),
                "panel_vs_full_enrichment": (
                    (panel_counts[context] / float(panel_total))
                    / (full_counts[context] / float(full_total))
                    if full_counts[context] > 0
                    else float("nan")
                ),
                "panel_vs_prefilter_mean_enrichment": (
                    (panel_counts[context] / float(panel_total)) / mean_frac if mean_frac > 0 else float("nan")
                ),
                "panel_vs_full_fisher_p": _fisher(panel_counts[context], panel_total, full_counts[context], full_total),
                "panel_vs_prefilter_mean_binom_p": float(
                    binomtest(k=int(panel_counts[context]), n=panel_total, p=mean_frac).pvalue
                ),
            }
        )

    pd.DataFrame(summary_rows).to_csv(out_dir / "prefilter_context_summary.tsv", sep="\t", index=False)
    pd.DataFrame(result_rows).to_csv(out_dir / "context_enrichment_cv_prefilter.tsv", sep="\t", index=False)

    meta = {
        "annotation_tsv": str(ann_path),
        "panel_annotations_tsv": str(panel_path),
        "prefilter_glob": args.prefilter_glob,
        "n_prefilter_files": int(len(prefilter_files)),
        "panel_total": panel_total,
        "full_total": full_total,
        "contexts": CONTEXTS,
        "notes": {
            "full_background_test": "two-sided Fisher exact on full HM450 manifest counts",
            "cv_prefilter_background_test": "two-sided exact binomial test using mean Stage-2 prefilter context fraction over 100 folds",
        },
    }
    (out_dir / "context_enrichment_cv_prefilter_meta.json").write_text(
        json.dumps(meta, indent=2),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
