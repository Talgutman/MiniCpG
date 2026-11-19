#!/usr/bin/env python3
"""Combine cfMethDB NumPy beta-matrix chunks."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import time
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chunks-dir", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--candidate-coordinates", required=True, type=Path)
    parser.add_argument("--expected-chunks", type=int, required=True)
    return parser.parse_args()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def count_candidate_rows(path: Path) -> int:
    with path.open() as handle:
        return max(sum(1 for _ in handle) - 1, 0)


def main() -> int:
    args = parse_args()
    started = time.time()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    chunks = [args.chunks_dir / f"chunk_{idx:05d}" for idx in range(1, args.expected_chunks + 1)]
    missing = [str(path) for path in chunks if not (path / "beta_values.npy").exists()]
    if missing:
        raise SystemExit(f"Missing beta_values.npy for {len(missing)} chunks, first={missing[:5]}")

    arrays = [np.load(path / "beta_values.npy", mmap_mode="r") for path in chunks]
    n_features = arrays[0].shape[1]
    bad_shapes = [str(path) for path, arr in zip(chunks, arrays) if arr.ndim != 2 or arr.shape[1] != n_features]
    if bad_shapes:
        raise SystemExit(f"Chunk shape mismatch: {bad_shapes[:5]}")
    n_candidate_rows = count_candidate_rows(args.candidate_coordinates)
    if n_candidate_rows != n_features:
        raise SystemExit(f"Candidate count {n_candidate_rows} != matrix feature count {n_features}")

    sample_rows: list[dict[str, object]] = []
    for chunk in chunks:
        sample_rows.extend(read_tsv(chunk / "sample_metadata.tsv"))
    n_samples = len(sample_rows)

    matrix_path = args.out_dir / "beta_matrix.npy"
    matrix = np.lib.format.open_memmap(matrix_path, mode="w+", dtype=np.float32, shape=(n_samples, n_features))
    offset = 0
    for arr in arrays:
        next_offset = offset + arr.shape[0]
        matrix[offset:next_offset, :] = arr
        offset = next_offset
    matrix.flush()

    sample_fields = list(sample_rows[0]) if sample_rows else []
    write_tsv(args.out_dir / "sample_metadata.tsv", sample_rows, sample_fields)

    feature_path = args.out_dir / "candidate_coordinates.tsv"
    if not feature_path.exists():
        shutil.copyfile(args.candidate_coordinates, feature_path)

    status_counts = Counter(str(row.get("status", "")) for row in sample_rows)
    label_counts = Counter(str(row.get("analysis_label", "")) for row in sample_rows)
    ok_rows = [row for row in sample_rows if row.get("status") == "ok"]
    presence_values = [
        float(row["feature_presence_fraction"])
        for row in ok_rows
        if str(row.get("feature_presence_fraction", "")) not in {"", "nan"}
    ]
    manifest = {
        "status": "np_matrix_combined",
        "date": date.today().isoformat(),
        "chunks_dir": str(args.chunks_dir),
        "out_dir": str(args.out_dir),
        "candidate_coordinates": str(args.candidate_coordinates),
        "matrix_path": str(matrix_path),
        "sample_metadata": str(args.out_dir / "sample_metadata.tsv"),
        "feature_metadata": str(feature_path),
        "expected_chunks": args.expected_chunks,
        "n_samples": n_samples,
        "n_features": n_features,
        "status_counts": dict(sorted(status_counts.items())),
        "label_counts": dict(sorted(label_counts.items())),
        "ok_presence_fraction_min": min(presence_values) if presence_values else None,
        "ok_presence_fraction_median": float(np.median(presence_values)) if presence_values else None,
        "ok_presence_fraction_max": max(presence_values) if presence_values else None,
        "elapsed_sec": round(time.time() - started, 3),
    }
    (args.out_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
