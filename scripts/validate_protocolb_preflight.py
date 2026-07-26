#!/usr/bin/env python3
"""Preflight validation for protocol-B model-stage submissions.

This script fails fast if required inputs are missing, selected-probe files are
incomplete, or patched scripts are not in place.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


def split_tokens(text: str) -> List[str]:
    return [x.strip() for x in re.split(r"[,\s:]+", str(text)) if x.strip()]


def parse_csv_list(text: str) -> List[str]:
    return split_tokens(text)


def parse_int_list(text: str) -> List[int]:
    return [int(x) for x in split_tokens(text)]


def ensure_file(path: Path, name: str, errors: List[str]) -> None:
    if not path.exists():
        errors.append(f"missing_{name}: {path}")
    elif not path.is_file():
        errors.append(f"not_a_file_{name}: {path}")


def ensure_dir(path: Path, name: str, errors: List[str]) -> None:
    if not path.exists():
        errors.append(f"missing_{name}: {path}")
    elif not path.is_dir():
        errors.append(f"not_a_dir_{name}: {path}")


def check_script_markers(project_dir: Path, errors: List[str]) -> None:
    model_script = project_dir / "scripts" / "run_model_benchmark_from_selected.py"
    agg_script = (
        project_dir
        / "slurm"
        / "aggregate_model_benchmark_protocolB_dev_5x3_jointk_min50merge_v2_sys39.slurm"
    )
    ensure_file(model_script, "run_model_script", errors)
    ensure_file(agg_script, "aggregate_script", errors)
    if errors:
        return

    text_model = model_script.read_text()
    if 'model_subdir = f"{model_name}__cw_{args.class_weight_mode}__tm_{args.tune_metric}"' not in text_model:
        errors.append("run_model_script_missing_overwrite_fix_marker")

    text_agg = agg_script.read_text()
    for marker in ['METHODS_STR="${METHODS[*]}"', 'KS_STR="${KS[*]}"', 'methods = "$METHODS_STR".split()']:
        if marker not in text_agg:
            errors.append(f"aggregate_script_missing_marker: {marker}")


def check_folds(path: Path, n_repeats: int, n_splits: int, errors: List[str]) -> Dict[str, int]:
    out: Dict[str, int] = {}
    if not path.exists():
        return out
    df = pd.read_csv(path, sep="\t", dtype=str)
    for col in ["sample_id", "outer_fold"]:
        if col not in df.columns:
            errors.append(f"{path.name}_missing_col:{col}")
    if "repeat" not in df.columns:
        df["repeat"] = "0"
    if errors:
        return out

    try:
        df["outer_fold"] = df["outer_fold"].astype(int)
        df["repeat"] = df["repeat"].astype(int)
    except Exception:
        errors.append(f"{path.name}_repeat_or_fold_not_int")
        return out

    expected_pairs = {(r, f) for r in range(n_repeats) for f in range(n_splits)}
    seen_pairs = set(zip(df["repeat"], df["outer_fold"]))
    missing_pairs = sorted(expected_pairs - seen_pairs)
    if missing_pairs:
        errors.append(f"{path.name}_missing_repeat_fold_pairs:{missing_pairs[:8]}")

    out["n_rows"] = int(len(df))
    out["n_unique_samples"] = int(df["sample_id"].nunique())
    out["n_unique_pairs"] = int(len(seen_pairs))
    return out


def check_selected_files(
    selected_run_dir: Path,
    methods: List[str],
    ks: List[int],
    n_repeats: int,
    n_splits: int,
    errors: List[str],
) -> Dict[str, int]:
    n_expected = len(methods) * len(ks) * n_repeats * n_splits
    n_found = 0
    n_bad_count = 0
    for method in methods:
        for k in ks:
            for r in range(n_repeats):
                for f in range(n_splits):
                    p = (
                        selected_run_dir
                        / f"repeat{r}"
                        / f"fold{f}"
                        / method
                        / f"selected_probes_K{k}.tsv"
                    )
                    if not p.exists():
                        errors.append(f"missing_selected:{p}")
                        continue
                    n_found += 1
                    try:
                        df = pd.read_csv(p, sep="\t")
                    except Exception:
                        errors.append(f"unreadable_selected:{p}")
                        continue
                    if "probe_id" not in df.columns:
                        errors.append(f"selected_missing_probe_id_col:{p}")
                    if len(df) < k:
                        n_bad_count += 1
                        errors.append(f"selected_short_for_k:{p} rows={len(df)} k={k}")
    return {
        "n_expected_selected_files": n_expected,
        "n_found_selected_files": n_found,
        "n_bad_selected_count": n_bad_count,
    }


def is_dir_empty(path: Path) -> bool:
    return not any(path.iterdir())


def main() -> int:
    p = argparse.ArgumentParser(description="Validate inputs and script integrity before protocol-B submission.")
    p.add_argument("--project-dir", required=True)
    p.add_argument("--probe-matrix", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--folds-5x3", required=True)
    p.add_argument("--folds-5x20", required=True)
    p.add_argument("--selected-run-dir", required=True)
    p.add_argument("--methods", default="anova,mrmr,sparse_lr")
    p.add_argument("--ks", default="5,10,20,30,50,100")
    p.add_argument("--n-repeats-5x3", type=int, default=3)
    p.add_argument("--n-splits-5x3", type=int, default=5)
    p.add_argument("--run-root", required=True, help="New tournament run root to be created.")
    p.add_argument("--locked-root", required=True, help="New selected/lockbox run root to be created.")
    p.add_argument("--allow-nonempty-roots", action="store_true")
    p.add_argument("--report-json", default=None)
    args = p.parse_args()

    project_dir = Path(args.project_dir)
    probe_matrix = Path(args.probe_matrix)
    manifest = Path(args.manifest)
    folds_5x3 = Path(args.folds_5x3)
    folds_5x20 = Path(args.folds_5x20)
    selected_run_dir = Path(args.selected_run_dir)
    run_root = Path(args.run_root)
    locked_root = Path(args.locked_root)

    methods = parse_csv_list(args.methods)
    ks = parse_int_list(args.ks)
    errors: List[str] = []
    warnings: List[str] = []
    report: Dict[str, object] = {}

    ensure_dir(project_dir, "project_dir", errors)
    ensure_file(probe_matrix, "probe_matrix", errors)
    ensure_file(manifest, "manifest", errors)
    ensure_file(folds_5x3, "folds_5x3", errors)
    ensure_file(folds_5x20, "folds_5x20", errors)
    ensure_dir(selected_run_dir, "selected_run_dir", errors)

    check_script_markers(project_dir, errors)
    report["folds_5x3"] = check_folds(folds_5x3, args.n_repeats_5x3, args.n_splits_5x3, errors)
    report["folds_5x20"] = check_folds(folds_5x20, 20, 5, errors)
    report["selected_files"] = check_selected_files(
        selected_run_dir,
        methods,
        ks,
        args.n_repeats_5x3,
        args.n_splits_5x3,
        errors,
    )

    for root_name, root_path in [("run_root", run_root), ("locked_root", locked_root)]:
        if root_path.exists() and root_path.is_dir() and not is_dir_empty(root_path):
            if args.allow_nonempty_roots:
                warnings.append(f"nonempty_{root_name}_allowed:{root_path}")
            else:
                errors.append(f"nonempty_{root_name}:{root_path}")

    report["methods"] = methods
    report["ks"] = ks
    report["errors"] = errors
    report["warnings"] = warnings
    report["ok"] = len(errors) == 0

    text = json.dumps(report, indent=2)
    if args.report_json:
        out = Path(args.report_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n")

    print(text)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
