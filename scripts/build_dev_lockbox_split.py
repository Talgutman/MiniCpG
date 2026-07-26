#!/usr/bin/env python3
"""Build a reproducible stratified dev/lockbox split manifest.

Split is performed at the group level (patient_id by default) to avoid leakage.
Outputs:
- split assignment table
- dev manifest
- lockbox manifest
- per-class counts table
- summary JSON
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create stratified dev/lockbox split from a classification manifest.")
    parser.add_argument(
        "--manifest",
        default="outputs/manifests/classification_manifest_hm450_min100.tsv",
        help="Input manifest TSV.",
    )
    parser.add_argument("--label-col", default="project_id", help="Class label column (default: project_id).")
    parser.add_argument("--group-col", default="patient_id", help="Grouping column to avoid leakage (default: patient_id).")
    parser.add_argument("--sample-col", default="sample_id", help="Sample ID column (default: sample_id).")
    parser.add_argument("--lockbox-fraction", type=float, default=0.2, help="Fraction for lockbox split (default: 0.2).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument(
        "--out-dir",
        default="outputs/manifests",
        help="Output directory for generated split files.",
    )
    parser.add_argument("--prefix", default="classification_manifest_hm450_min100", help="Output file prefix.")
    return parser.parse_args()


def read_tsv(path: Path) -> Tuple[List[str], List[Dict[str, str]]]:
    with path.open() as f:
        reader = csv.DictReader(f, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError(f"{path} is missing header.")
        rows = [dict(r) for r in reader]
        return list(reader.fieldnames), rows


def write_tsv(path: Path, fieldnames: List[str], rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def stratified_group_split(
    rows: List[Dict[str, str]],
    label_col: str,
    group_col: str,
    lockbox_fraction: float,
    seed: int,
) -> Dict[str, str]:
    groups: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for r in rows:
        groups[r[group_col]].append(r)

    # Validate each group maps to a single class label.
    group_to_label: Dict[str, str] = {}
    for gid, grows in groups.items():
        labs = {g[label_col] for g in grows}
        if len(labs) != 1:
            raise ValueError(f"Group {gid} has multiple labels: {sorted(labs)}")
        group_to_label[gid] = next(iter(labs))

    groups_by_label: Dict[str, List[str]] = defaultdict(list)
    for gid, lab in group_to_label.items():
        groups_by_label[lab].append(gid)

    rng = random.Random(seed)
    split_by_group: Dict[str, str] = {}
    for lab in sorted(groups_by_label):
        gids = list(groups_by_label[lab])
        rng.shuffle(gids)

        n = len(gids)
        if n < 2:
            n_lockbox = 0
        else:
            n_lockbox = int(math.floor(n * lockbox_fraction))
            n_lockbox = max(1, min(n_lockbox, n - 1))

        lockbox = set(gids[:n_lockbox])
        for gid in gids:
            split_by_group[gid] = "lockbox" if gid in lockbox else "dev"

    return split_by_group


def main() -> int:
    args = parse_args()
    if not (0.0 < args.lockbox_fraction < 1.0):
        raise SystemExit("--lockbox-fraction must be in (0,1)")

    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found: {manifest_path}")

    fieldnames, rows = read_tsv(manifest_path)
    for col in [args.label_col, args.group_col, args.sample_col]:
        if col not in fieldnames:
            raise SystemExit(f"Column not found in manifest: {col}")

    split_by_group = stratified_group_split(
        rows=rows,
        label_col=args.label_col,
        group_col=args.group_col,
        lockbox_fraction=args.lockbox_fraction,
        seed=args.seed,
    )

    split_rows: List[Dict[str, object]] = []
    dev_rows: List[Dict[str, str]] = []
    lockbox_rows: List[Dict[str, str]] = []
    for r in rows:
        split = split_by_group[r[args.group_col]]
        out = dict(r)
        out["split"] = split
        split_rows.append(out)
        if split == "dev":
            dev_rows.append(r)
        else:
            lockbox_rows.append(r)

    out_dir = Path(args.out_dir)
    split_path = out_dir / f"{args.prefix}_split_dev_lockbox.tsv"
    dev_path = out_dir / f"{args.prefix}_dev.tsv"
    lockbox_path = out_dir / f"{args.prefix}_lockbox.tsv"
    counts_path = out_dir / f"{args.prefix}_dev_lockbox_counts.tsv"
    summary_path = out_dir / f"{args.prefix}_dev_lockbox_summary.json"

    write_tsv(split_path, fieldnames + ["split"], split_rows)
    write_tsv(dev_path, fieldnames, dev_rows)
    write_tsv(lockbox_path, fieldnames, lockbox_rows)

    c_dev = Counter(r[args.label_col] for r in dev_rows)
    c_lock = Counter(r[args.label_col] for r in lockbox_rows)
    labels = sorted(set(c_dev) | set(c_lock))
    count_rows: List[Dict[str, object]] = []
    for lab in labels:
        n_dev = int(c_dev.get(lab, 0))
        n_lock = int(c_lock.get(lab, 0))
        n_total = n_dev + n_lock
        frac = (n_lock / n_total) if n_total else 0.0
        count_rows.append(
            {
                args.label_col: lab,
                "n_dev": n_dev,
                "n_lockbox": n_lock,
                "n_total": n_total,
                "lockbox_fraction": f"{frac:.6f}",
            }
        )
    write_tsv(counts_path, [args.label_col, "n_dev", "n_lockbox", "n_total", "lockbox_fraction"], count_rows)

    summary = {
        "input_manifest": str(manifest_path),
        "n_rows": len(rows),
        "n_dev": len(dev_rows),
        "n_lockbox": len(lockbox_rows),
        "n_classes": len(labels),
        "seed": args.seed,
        "lockbox_fraction_requested": args.lockbox_fraction,
        "lockbox_fraction_observed": (len(lockbox_rows) / len(rows)) if rows else 0.0,
        "group_col": args.group_col,
        "label_col": args.label_col,
        "sample_col": args.sample_col,
        "all_classes_have_dev": all(c_dev.get(l, 0) > 0 for l in labels),
        "all_classes_have_lockbox": all(c_lock.get(l, 0) > 0 for l in labels),
        "min_class_dev": min(c_dev.get(l, 0) for l in labels) if labels else 0,
        "min_class_lockbox": min(c_lock.get(l, 0) for l in labels) if labels else 0,
        "outputs": {
            "split": str(split_path),
            "dev_manifest": str(dev_path),
            "lockbox_manifest": str(lockbox_path),
            "counts": str(counts_path),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2))

    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

