#!/usr/bin/env python3
"""Build a paper-protocol TCGA manifest from the current Protocol B source manifest.

This initial version keeps the current merged source manifest as authority and
filters to classes meeting the paper's minimum cohort-size rule.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames is None:
            raise SystemExit(f"Missing header: {path}")
        return list(reader.fieldnames), [dict(r) for r in reader]


def write_tsv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build paper-protocol TCGA manifest from current Protocol B manifest.")
    p.add_argument("--source-manifest", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--prefix", default="paper_protocolb")
    p.add_argument("--label-col", default="project_id")
    p.add_argument("--min-class-n", type=int, default=100)
    return p.parse_args()


def main() -> int:
    args = parse_args()
    source_manifest = Path(args.source_manifest)
    if not source_manifest.exists():
        raise SystemExit(f"Source manifest not found: {source_manifest}")

    fieldnames, rows = read_tsv(source_manifest)
    if args.label_col not in fieldnames:
        raise SystemExit(f"Label column not found: {args.label_col}")

    class_counts = Counter(str(r[args.label_col]).strip() for r in rows)
    keep_labels = {lab for lab, n in class_counts.items() if n >= args.min_class_n}
    kept_rows = [r for r in rows if str(r[args.label_col]).strip() in keep_labels]

    out_dir = Path(args.out_dir)
    manifest_path = out_dir / f"{args.prefix}_manifest.tsv"
    counts_path = out_dir / f"{args.prefix}_class_counts.tsv"
    summary_path = out_dir / f"{args.prefix}_summary.json"

    write_tsv(manifest_path, fieldnames, kept_rows)
    count_rows = [
        {args.label_col: lab, "n_samples": int(class_counts[lab]), "kept": 1 if lab in keep_labels else 0}
        for lab in sorted(class_counts)
    ]
    write_tsv(counts_path, [args.label_col, "n_samples", "kept"], count_rows)

    summary = {
        "source_manifest": str(source_manifest),
        "min_class_n": int(args.min_class_n),
        "n_rows_in": len(rows),
        "n_rows_out": len(kept_rows),
        "n_classes_in": len(class_counts),
        "n_classes_out": len(keep_labels),
        "kept_labels": sorted(keep_labels),
        "outputs": {
            "manifest_tsv": str(manifest_path),
            "counts_tsv": str(counts_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
