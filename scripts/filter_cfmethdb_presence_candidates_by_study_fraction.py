#!/usr/bin/env python3
"""Filter merged cfMethDB presence atlas by one study's label-blind coverage."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter
from datetime import date
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--merged-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--study-id", required=True)
    parser.add_argument("--study-denominator", type=int, required=True)
    parser.add_argument("--min-study-fraction", type=float, default=0.80)
    parser.add_argument("--expected-shards", type=int, default=128)
    return parser.parse_args()


def feature_id(chrom: str, start: str, end: str) -> str:
    return f"{chrom}:{start}-{end}"


def main() -> int:
    args = parse_args()
    started = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    paths = sorted(args.merged_dir.glob("presence_counts_merged_shard_*.tsv"))
    if args.expected_shards and len(paths) != args.expected_shards:
        raise SystemExit(f"Expected {args.expected_shards} merged shards, found {len(paths)}")
    if not paths:
        raise SystemExit(f"No merged shard TSVs found under {args.merged_dir}")

    min_study_count = int(math.ceil(args.study_denominator * args.min_study_fraction))
    study_col = f"study__{args.study_id}"
    out_path = args.out_dir / (
        f"candidate_coordinates_{args.study_id}_study_fraction_ge{args.min_study_fraction:.2f}.tsv"
    )
    scanned = 0
    kept = 0
    reject_counts: Counter[str] = Counter()
    header_ref: list[str] | None = None
    study_idx: int | None = None
    label_cols: list[str] = []
    study_label_cols: list[str] = []

    with out_path.open("w", newline="") as out_handle:
        writer: csv.DictWriter[str] | None = None
        for path in paths:
            with path.open(newline="") as handle:
                reader = csv.reader(handle, delimiter="\t")
                header = next(reader)
                if header_ref is None:
                    header_ref = header
                    if study_col not in header:
                        raise SystemExit(f"Merged atlas missing {study_col}")
                    study_idx = header.index(study_col)
                    label_indices = [i for i, name in enumerate(header) if name.startswith("label__")]
                    study_label_indices = [
                        i for i, name in enumerate(header) if name.startswith(f"study_label__") and args.study_id in name
                    ]
                    label_cols = [header[i] for i in label_indices]
                    study_label_cols = [header[i] for i in study_label_indices]
                    fields = [
                        "feature_id",
                        "chrom",
                        "start",
                        "end",
                        "study_id",
                        "study_denominator",
                        "study_observed_count",
                        "study_observed_fraction",
                        "min_study_fraction",
                        "min_study_count",
                        "n_samples_observed",
                    ] + label_cols + study_label_cols
                    writer = csv.DictWriter(out_handle, delimiter="\t", fieldnames=fields, lineterminator="\n")
                    writer.writeheader()
                elif header != header_ref:
                    raise SystemExit(f"Header mismatch in {path}")
                assert writer is not None
                assert study_idx is not None
                for row in reader:
                    scanned += 1
                    study_count = int(row[study_idx])
                    if study_count < min_study_count:
                        reject_counts["min_study_count"] += 1
                        continue
                    out_row = {
                        "feature_id": feature_id(row[0], row[1], row[2]),
                        "chrom": row[0],
                        "start": row[1],
                        "end": row[2],
                        "study_id": args.study_id,
                        "study_denominator": args.study_denominator,
                        "study_observed_count": study_count,
                        "study_observed_fraction": f"{study_count / args.study_denominator:.8g}",
                        "min_study_fraction": args.min_study_fraction,
                        "min_study_count": min_study_count,
                        "n_samples_observed": row[3],
                    }
                    for col in label_cols:
                        out_row[col] = row[header.index(col)]
                    for col in study_label_cols:
                        out_row[col] = row[header.index(col)]
                    writer.writerow(out_row)
                    kept += 1

    manifest = {
        "status": "study_fraction_candidate_filter_complete",
        "date": date.today().isoformat(),
        "merged_dir": str(args.merged_dir),
        "out_dir": str(args.out_dir),
        "study_id": args.study_id,
        "study_denominator": args.study_denominator,
        "min_study_fraction": args.min_study_fraction,
        "min_study_count": min_study_count,
        "expected_shards": args.expected_shards,
        "n_scanned_coordinates": scanned,
        "n_candidate_coordinates": kept,
        "reject_counts": dict(reject_counts),
        "output_path": str(out_path),
        "elapsed_sec": round(time.time() - started, 3),
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
