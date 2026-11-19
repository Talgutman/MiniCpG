#!/usr/bin/env python3
"""Extract cfMethDB BigWig values into compact NumPy chunks."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import time
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path

import numpy as np


BIGWIG_TEMPLATE = "https://cfmethdb.hzau.edu.cn/cfbwdata/{geoid}.CG.dm.txt_v3.bw.cg"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-manifest", required=True, type=Path)
    parser.add_argument("--candidate-coordinates", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--cache-dir", required=True, type=Path)
    parser.add_argument("--task-index", required=True, type=int, help="1-based chunk index.")
    parser.add_argument("--chunk-size", type=int, default=5)
    parser.add_argument("--download-timeout-sec", type=int, default=900)
    parser.add_argument("--delete-bigwigs", action="store_true")
    parser.add_argument("--progress-log", type=Path)
    return parser.parse_args()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_manifest(path: Path) -> list[dict[str, str]]:
    rows = read_tsv(path)
    if not rows:
        raise SystemExit(f"No rows in manifest: {path}")
    required = {"Geoid", "analysis_label", "studyid", "bigwig_url"}
    missing = required - set(rows[0])
    if missing:
        raise SystemExit(f"Manifest missing required columns: {sorted(missing)}")
    return rows


def load_candidates(path: Path) -> tuple[int, dict[str, dict[str, np.ndarray]]]:
    by_chrom: dict[str, dict[str, list[int]]] = defaultdict(lambda: {"index": [], "start": [], "end": []})
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"feature_id", "chrom", "start", "end"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise SystemExit(f"Candidate file missing columns: {sorted(missing)}")
        n_features = 0
        for idx, row in enumerate(reader):
            chrom = row["chrom"]
            by_chrom[chrom]["index"].append(idx)
            by_chrom[chrom]["start"].append(int(row["start"]))
            by_chrom[chrom]["end"].append(int(row["end"]))
            n_features += 1
    if n_features == 0:
        raise SystemExit(f"No candidate coordinates in {path}")

    packed: dict[str, dict[str, np.ndarray]] = {}
    for chrom, data in by_chrom.items():
        starts = np.asarray(data["start"], dtype=np.int32)
        order = np.argsort(starts, kind="mergesort")
        packed[chrom] = {
            "index": np.asarray(data["index"], dtype=np.int32)[order],
            "start": starts[order],
            "end": np.asarray(data["end"], dtype=np.int32)[order],
        }
    return n_features, packed


def download_bigwig(row: dict[str, str], cache_dir: Path, timeout_sec: int) -> Path | None:
    geoid = row["Geoid"]
    url = row.get("bigwig_url") or BIGWIG_TEMPLATE.format(geoid=geoid)
    cache_dir.mkdir(parents=True, exist_ok=True)
    dest = cache_dir / f"{geoid}.CG.dm.txt_v3.bw.cg"
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    tmp = dest.with_suffix(dest.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    try:
        subprocess.run(
            [
                "curl",
                "-L",
                "--fail",
                "--connect-timeout",
                "30",
                "--max-time",
                str(timeout_sec),
                "-o",
                str(tmp),
                url,
            ],
            check=True,
        )
    except subprocess.CalledProcessError:
        if tmp.exists():
            tmp.unlink()
        return None
    tmp.replace(dest)
    return dest


def finite_signal(value: object) -> bool:
    if value is None:
        return False
    try:
        number = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(number) and -1.0005 <= number <= 1.0005


def extract_values(bw_path: Path, by_chrom: dict[str, dict[str, np.ndarray]], n_features: int) -> np.ndarray:
    try:
        import pyBigWig  # type: ignore
    except ImportError as exc:
        raise SystemExit("pyBigWig is required. Add the project-local pyBigWig target to PYTHONPATH.") from exc

    values = np.full(n_features, np.nan, dtype=np.float32)
    bw = pyBigWig.open(str(bw_path))
    if bw is None:
        raise RuntimeError(f"pyBigWig.open returned None for {bw_path}")
    try:
        chroms = bw.chroms()
        for chrom, data in by_chrom.items():
            if chrom not in chroms:
                continue
            intervals = bw.intervals(chrom)
            if not intervals:
                continue
            starts = data["start"]
            ends = data["end"]
            out_idx = data["index"]
            interval_i = 0
            n_intervals = len(intervals)
            for cand_i, (start, end) in enumerate(zip(starts, ends)):
                while interval_i < n_intervals and intervals[interval_i][1] <= start:
                    interval_i += 1
                scan_i = interval_i
                total = 0.0
                n = 0
                while scan_i < n_intervals and intervals[scan_i][0] < end:
                    int_start, int_end, value = intervals[scan_i]
                    if int_end > start and finite_signal(value):
                        total += float(value)
                        n += 1
                    scan_i += 1
                if n:
                    values[out_idx[cand_i]] = total / n
    finally:
        bw.close()
    return values


def write_tsv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def append_progress(path: Path | None, row: dict[str, object]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, delimiter="\t", fieldnames=list(row), lineterminator="\n")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def main() -> int:
    args = parse_args()
    if args.task_index < 1:
        raise SystemExit("--task-index must be 1-based")
    started = time.time()
    rows = read_manifest(args.sample_manifest)
    n_features, by_chrom = load_candidates(args.candidate_coordinates)

    start = (args.task_index - 1) * args.chunk_size
    end = min(len(rows), start + args.chunk_size)
    chunk_rows = rows[start:end]
    if not chunk_rows:
        raise SystemExit(f"Task {args.task_index} has no rows for manifest size {len(rows)}")

    chunk_dir = args.out_dir / f"chunk_{args.task_index:05d}"
    chunk_dir.mkdir(parents=True, exist_ok=True)
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    matrix = np.full((len(chunk_rows), n_features), np.nan, dtype=np.float32)
    sample_rows: list[dict[str, object]] = []
    status_counts: Counter[str] = Counter()

    for sample_i, row in enumerate(chunk_rows):
        sample_started = time.time()
        path = download_bigwig(row, args.cache_dir, args.download_timeout_sec)
        if path is None:
            status = "download_failed"
            observed = 0
        else:
            try:
                matrix[sample_i, :] = extract_values(path, by_chrom, n_features)
                observed = int(np.isfinite(matrix[sample_i, :]).sum())
                status = "ok"
            except Exception as exc:  # noqa: BLE001 - keep sample-level failure and continue chunk.
                status = f"error:{type(exc).__name__}"
                observed = 0
            if args.delete_bigwigs and path.exists():
                path.unlink()
        status_counts[status] += 1
        elapsed = time.time() - sample_started
        base = {
            "manifest_index": row.get("manifest_index", ""),
            "Geoid": row["Geoid"],
            "analysis_label": row["analysis_label"],
            "studyid": row["studyid"],
            "cfmethdb_label": row.get("cfmethdb_label", ""),
            "selection_role": row.get("selection_role", ""),
            "source": row.get("source", ""),
            "bases": row.get("bases", ""),
            "status": status,
            "n_features_observed": observed if status == "ok" else "",
            "feature_presence_fraction": f"{observed / n_features:.8g}" if status == "ok" else "",
            "elapsed_sec": f"{elapsed:.3f}",
        }
        sample_rows.append(base)
        append_progress(
            args.progress_log,
            {
                "task_index": args.task_index,
                "sample_in_chunk": sample_i + 1,
                "Geoid": row["Geoid"],
                "status": status,
                "n_features_observed": observed,
                "feature_presence_fraction": f"{observed / n_features:.8g}" if status == "ok" else "",
                "elapsed_sec": f"{elapsed:.3f}",
                "chunk_elapsed_sec": f"{time.time() - started:.3f}",
            },
        )
        print(f"{row['Geoid']} {status} observed={observed} elapsed_sec={elapsed:.1f}", flush=True)

    np.save(chunk_dir / "beta_values.npy", matrix, allow_pickle=False)
    fields = [
        "manifest_index",
        "Geoid",
        "analysis_label",
        "studyid",
        "cfmethdb_label",
        "selection_role",
        "source",
        "bases",
        "status",
        "n_features_observed",
        "feature_presence_fraction",
        "elapsed_sec",
    ]
    write_tsv(chunk_dir / "sample_metadata.tsv", sample_rows, fields)

    manifest = {
        "status": "np_chunk_complete",
        "date": date.today().isoformat(),
        "sample_manifest": str(args.sample_manifest),
        "candidate_coordinates": str(args.candidate_coordinates),
        "out_dir": str(args.out_dir),
        "cache_dir": str(args.cache_dir),
        "task_index": args.task_index,
        "chunk_size": args.chunk_size,
        "n_manifest_rows": len(rows),
        "n_chunk_rows": len(chunk_rows),
        "n_features": n_features,
        "status_counts": dict(status_counts),
        "elapsed_sec": round(time.time() - started, 3),
    }
    (chunk_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
