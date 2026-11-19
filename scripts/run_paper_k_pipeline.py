#!/usr/bin/env python3
"""Prepare one-K paper protocol run directory and stage manifests."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from split_integrity import read_rows, validate_partition, validate_folds, id_hash

import pandas as pd
import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare one-K paper protocol run.")
    p.add_argument("--config", required=True)
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--out-dir", default="")
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f) or {}


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_dirs(root: Path) -> dict[str, Any]:
    top_level = {
        "manifest": root / "manifest",
        "search_space": root / "search_space",
        "lockbox_eval": root / "lockbox_eval",
        "external_main": root / "external_main",
        "external_stress_ffpe": root / "external_stress_ffpe",
        "external_stress_epic": root / "external_stress_epic",
        "external_stress_metastatic": root / "external_stress_metastatic",
        "summary": root / "summary",
    }
    for path in top_level.values():
        path.mkdir(parents=True, exist_ok=True)

    stage1 = {
        "root": root / "stage1_select",
        "fs_benchmark": root / "stage1_select" / "fs_benchmark",
        "model_benchmark": root / "stage1_select" / "model_benchmark",
        "selection": root / "stage1_select" / "selection",
        "summary": root / "stage1_select" / "summary",
    }
    stage2 = {
        "root": root / "stage2_confirm",
        "fs_locked": root / "stage2_confirm" / "fs_locked",
        "model_locked": root / "stage2_confirm" / "model_locked",
        "stable_panel": root / "stage2_confirm" / "stable_panel",
        "summary": root / "stage2_confirm" / "summary",
    }
    for group in [stage1, stage2]:
        for path in group.values():
            if isinstance(path, Path):
                path.mkdir(parents=True, exist_ok=True)

    return {
        **top_level,
        "stage1_select": stage1,
        "stage2_confirm": stage2,
    }


def run_cmd(cmd: list[str], cwd: Path) -> None:
    proc = subprocess.run(cmd, cwd=str(cwd), check=False)
    if proc.returncode != 0:
        raise SystemExit(f"Command failed ({proc.returncode}): {' '.join(cmd)}")


def maybe_copy(src_text: str, dst: Path) -> bool:
    src_raw = str(src_text).strip()
    if not src_raw:
        return False
    src = Path(src_raw)
    if not src.is_file():
        raise FileNotFoundError(f"Configured reuse source is unavailable; refusing regeneration: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def build_selection_candidates(
    *,
    fs_specs: dict[str, Any],
    default_prefilter_k: int,
    logistic_prefilter_k: int,
    prefilter_method: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for method, specs in fs_specs.items():
        for spec in list(specs or []):
            fs_candidate_id = str(spec["candidate_id"])
            selection_method = str(spec["selection_method"])
            base_candidate_spec = dict(spec.get("candidate_spec") or {})
            for prefilter_profile, prefilter_k in [
                ("default", int(default_prefilter_k)),
                ("logistic", int(logistic_prefilter_k)),
            ]:
                candidate_spec = json.loads(json.dumps(base_candidate_spec))
                prefilter_args = dict(candidate_spec.get("prefilter_args") or {})
                prefilter_args["prefilter_method"] = str(prefilter_method)
                if prefilter_profile == "logistic" or "prefilter_k" not in prefilter_args:
                    prefilter_args["prefilter_k"] = int(prefilter_k)
                candidate_spec["prefilter_args"] = prefilter_args
                rows.append(
                    {
                        "selection_candidate_id": f"{fs_candidate_id}__pf_{prefilter_profile}",
                        "fs_candidate_id": fs_candidate_id,
                        "selection_method": selection_method,
                        "prefilter_profile": prefilter_profile,
                        "prefilter_method": str(prefilter_method),
                        "prefilter_k": int(prefilter_args["prefilter_k"]),
                        "candidate_spec_json": json.dumps(candidate_spec, sort_keys=True),
                    }
                )
    return rows


def build_model_candidates(
    *,
    model_specs: dict[str, Any],
    default_prefilter_k: int,
    logistic_prefilter_k: int,
    class_weight_mode: str,
    tune_metric: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for model_name, specs in model_specs.items():
        prefilter_profile = "logistic" if model_name == "elastic_net_multinomial_logistic" else "default"
        prefilter_k = logistic_prefilter_k if prefilter_profile == "logistic" else default_prefilter_k
        for spec in list(specs or []):
            rows.append(
                {
                    "model_candidate_id": str(spec["candidate_id"]),
                    "model": str(spec["model"]),
                    "prefilter_profile": prefilter_profile,
                    "prefilter_k": int(prefilter_k),
                    "class_weight_mode": str(spec.get("class_weight_mode") or class_weight_mode),
                    "tune_metric": tune_metric,
                    "candidate_spec_json": json.dumps(spec.get("candidate_spec") or {}, sort_keys=True),
                }
            )
    return rows


def build_joint_candidates(
    *,
    selection_candidates: list[dict[str, object]],
    model_candidates: list[dict[str, object]],
    k: int,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for selection_row in selection_candidates:
        for model_row in model_candidates:
            if str(selection_row["prefilter_profile"]) != str(model_row["prefilter_profile"]):
                continue
            rows.append(
                {
                    "joint_candidate_id": (
                        f"{selection_row['selection_candidate_id']}__{model_row['model_candidate_id']}__k{int(k):03d}"
                    ),
                    "selection_candidate_id": selection_row["selection_candidate_id"],
                    "model_candidate_id": model_row["model_candidate_id"],
                    "selection_method": selection_row["selection_method"],
                    "model": model_row["model"],
                    "prefilter_method": selection_row["prefilter_method"],
                    "prefilter_k": int(selection_row["prefilter_k"]),
                    "class_weight_mode": model_row["class_weight_mode"],
                    "tune_metric": model_row["tune_metric"],
                    "selection_candidate_spec_json": selection_row["candidate_spec_json"],
                    "model_candidate_spec_json": model_row["candidate_spec_json"],
                }
            )
    return rows


def build_fs_task_manifest(
    *,
    selection_candidates: list[dict[str, object]],
    folds_df: pd.DataFrame,
    fs_run_base_dir: Path,
    dev_manifest_tsv: Path,
    folds_tsv: Path,
) -> list[dict[str, object]]:
    unique_folds = (
        folds_df[["repeat", "outer_fold"]]
        .drop_duplicates()
        .astype({"repeat": int, "outer_fold": int})
        .sort_values(["repeat", "outer_fold"], kind="mergesort")
    )
    rows: list[dict[str, object]] = []
    for fold_row in unique_folds.itertuples(index=False):
        for selection_row in selection_candidates:
            selection_candidate_id = str(selection_row["selection_candidate_id"])
            candidate_dir = fs_run_base_dir / selection_candidate_id
            spec_path = candidate_dir / "candidate_spec.json"
            rows.append(
                {
                    "selection_candidate_id": selection_candidate_id,
                    "selection_method": selection_row["selection_method"],
                    "repeat": int(fold_row.repeat),
                    "fold": int(fold_row.outer_fold),
                    "manifest_tsv": str(dev_manifest_tsv),
                    "folds_tsv": str(folds_tsv),
                    "prefilter_method": selection_row["prefilter_method"],
                    "prefilter_k": int(selection_row["prefilter_k"]),
                    "candidate_spec_json_path": str(spec_path),
                    "out_dir": str(candidate_dir),
                }
            )
    return rows


def build_model_task_manifest(
    *,
    joint_candidates: list[dict[str, object]],
    folds_df: pd.DataFrame,
    fs_run_base_dir: Path,
    model_run_base_dir: Path,
    dev_manifest_tsv: Path,
    folds_tsv: Path,
) -> list[dict[str, object]]:
    unique_folds = (
        folds_df[["repeat", "outer_fold"]]
        .drop_duplicates()
        .astype({"repeat": int, "outer_fold": int})
        .sort_values(["repeat", "outer_fold"], kind="mergesort")
    )
    rows: list[dict[str, object]] = []
    for fold_row in unique_folds.itertuples(index=False):
        for joint_row in joint_candidates:
            selection_candidate_id = str(joint_row["selection_candidate_id"])
            joint_candidate_id = str(joint_row["joint_candidate_id"])
            model_dir = model_run_base_dir / joint_candidate_id
            rows.append(
                {
                    "joint_candidate_id": joint_candidate_id,
                    "selection_candidate_id": selection_candidate_id,
                    "model_candidate_id": str(joint_row["model_candidate_id"]),
                    "selection_method": joint_row["selection_method"],
                    "model": joint_row["model"],
                    "repeat": int(fold_row.repeat),
                    "fold": int(fold_row.outer_fold),
                    "manifest_tsv": str(dev_manifest_tsv),
                    "folds_tsv": str(folds_tsv),
                    "selected_run_dir": str(fs_run_base_dir / selection_candidate_id),
                    "candidate_spec_json_path": str(model_dir / "candidate_spec.json"),
                    "out_dir": str(model_dir),
                    "class_weight_mode": joint_row["class_weight_mode"],
                    "tune_metric": joint_row["tune_metric"],
                }
            )
    return rows


def ensure_candidate_spec_dirs(rows: list[dict[str, object]], base_dir_key: str) -> None:
    for row in rows:
        candidate_dir = Path(str(row[base_dir_key]))
        candidate_dir.mkdir(parents=True, exist_ok=True)


def write_candidate_specs(
    rows: list[dict[str, object]],
    run_base_dir: Path,
    id_key: str,
    spec_key: str,
) -> None:
    for row in rows:
        candidate_dir = run_base_dir / str(row[id_key])
        candidate_dir.mkdir(parents=True, exist_ok=True)
        (candidate_dir / "candidate_spec.json").write_text(
            json.dumps(json.loads(str(row[spec_key])), indent=2) + "\n",
            encoding="utf-8",
        )


def write_joint_model_candidate_specs(
    joint_rows: list[dict[str, object]],
    model_run_base_dir: Path,
) -> None:
    for row in joint_rows:
        candidate_dir = model_run_base_dir / str(row["joint_candidate_id"])
        candidate_dir.mkdir(parents=True, exist_ok=True)
        (candidate_dir / "candidate_spec.json").write_text(
            json.dumps(json.loads(str(row["model_candidate_spec_json"])), indent=2) + "\n",
            encoding="utf-8",
        )


def prepare_stage_folds(
    *,
    repo_root: Path,
    cfg: dict[str, Any],
    stage_cfg: dict[str, Any],
    dev_manifest: Path,
    out_path: Path,
) -> Path:
    existing_folds = str(stage_cfg.get("existing_outer_folds", "")).strip()
    if existing_folds:
        validate_folds(read_rows(dev_manifest), read_rows(existing_folds), int(stage_cfg["outer_repeats"]), int(stage_cfg["outer_splits"]))
    if maybe_copy(existing_folds, out_path):
        print(f"Reused existing folds: {out_path}")
        return out_path

    build_folds_cmd = [
        sys.executable,
        "scripts/build_outer_folds.py",
        "--manifest",
        str(dev_manifest),
        "--out",
        str(out_path),
        "--n-splits",
        str(stage_cfg["outer_splits"]),
        "--n-repeats",
        str(stage_cfg["outer_repeats"]),
        "--seed",
        str(cfg["cv"]["seed"]),
        "--sample-col",
        str(cfg["cohort"]["sample_id_col"]),
        "--project-col",
        str(cfg["cohort"]["label_col"]),
        "--group-col",
        str(cfg["cohort"]["patient_id_col"]),
    ]
    run_cmd(build_folds_cmd, repo_root)
    validate_folds(read_rows(dev_manifest), read_rows(out_path), int(stage_cfg["outer_repeats"]), int(stage_cfg["outer_splits"]))
    return out_path


def main() -> int:
    args = parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)

    split_cfg = cfg.get("split") or {}
    split_mode = split_cfg.get("mode")
    if split_mode not in {"reuse", "create"}:
        raise ValueError("split.mode must explicitly be reuse or create")
    existing_dev = str(split_cfg.get("existing_dev_manifest", "")).strip()
    existing_lock = str(split_cfg.get("existing_lockbox_manifest", "")).strip()
    if split_mode == "reuse":
        for source in (existing_dev, existing_lock):
            if not source or not Path(source).is_file():
                raise FileNotFoundError(f"Locked split source unavailable; refusing regeneration: {source}")
        partition = validate_partition(read_rows(existing_dev), read_rows(existing_lock))
        expected_path = split_cfg.get("membership_lock")
        if expected_path:
            expected = read_json(Path(expected_path))
            for key, value in partition.items():
                if key.endswith("sha256") and expected.get(key) != value:
                    raise ValueError(f"Configured sources differ from locked split: {key}")
    elif existing_dev or existing_lock:
        raise ValueError("split.mode=create cannot also request existing split sources")
    for stage in (cfg["selection_stage"], cfg["confirm_stage"]):
        source = str(stage.get("existing_outer_folds", "")).strip()
        if source and not Path(source).is_file():
            raise FileNotFoundError(f"Configured fold source unavailable: {source}")

    local_output_root = Path(str(cfg["project"]["local_output_root"]))
    root = Path(args.out_dir) if args.out_dir else (local_output_root / f"k_{int(args.k):03d}")
    dirs = ensure_dirs(root)

    manifest_prefix = f"paper_protocolb_k{int(args.k):03d}"

    build_manifest_cmd = [
        sys.executable,
        "scripts/build_protocolb_paper_manifest.py",
        "--source-manifest",
        str(cfg["cohort"]["source_manifest"]),
        "--out-dir",
        str(dirs["manifest"]),
        "--prefix",
        manifest_prefix,
        "--label-col",
        str(cfg["cohort"]["label_col"]),
        "--min-class-n",
        str(cfg["cohort"]["min_class_n"]),
    ]
    run_cmd(build_manifest_cmd, repo_root)

    paper_manifest = dirs["manifest"] / f"{manifest_prefix}_manifest.tsv"
    dev_manifest = dirs["manifest"] / f"{manifest_prefix}_dev.tsv"
    lockbox_manifest = dirs["manifest"] / f"{manifest_prefix}_lockbox.tsv"
    reused_split = split_mode == "reuse"
    if reused_split:
        # Read/validate both inputs before replacing either destination. Roll back
        # both on a write failure; never mix a new DEV with an old held-out file.
        dev_source, lock_source = Path(existing_dev), Path(existing_lock)
        validate_partition(read_rows(dev_source), read_rows(lock_source), read_rows(paper_manifest))
        sources = [dev_source.read_bytes(), lock_source.read_bytes()]
        targets = [dev_manifest, lockbox_manifest]
        previous = [p.read_bytes() if p.exists() else None for p in targets]
        try:
            for p, data in zip(targets, sources):
                p.with_suffix(p.suffix + ".staged").write_bytes(data)
            for p in targets:
                p.with_suffix(p.suffix + ".staged").replace(p)
        except Exception:
            for p, data in zip(targets, previous):
                if data is None:
                    p.unlink(missing_ok=True)
                else:
                    p.write_bytes(data)
                p.with_suffix(p.suffix + ".staged").unlink(missing_ok=True)
            raise
    if reused_split:
        split_path = dirs["manifest"] / f"{manifest_prefix}_split_dev_lockbox.tsv"
        counts_path = dirs["manifest"] / f"{manifest_prefix}_dev_lockbox_counts.tsv"
        dev_df = pd.read_csv(dev_manifest, sep="\t", dtype=str)
        lock_df = pd.read_csv(lockbox_manifest, sep="\t", dtype=str)
        split_df = pd.concat(
            [dev_df.assign(split="dev"), lock_df.assign(split="lockbox")],
            ignore_index=True,
        )
        split_df.to_csv(split_path, sep="\t", index=False)
        counts_df = (
            split_df.groupby(["split", str(cfg["cohort"]["label_col"])], dropna=False)
            .size()
            .reset_index(name="n_samples")
            .sort_values(["split", str(cfg["cohort"]["label_col"])], kind="mergesort")
        )
        counts_df.to_csv(counts_path, sep="\t", index=False)
    else:
        split_cmd = [
            sys.executable,
            "scripts/build_dev_lockbox_split.py",
            "--manifest",
            str(paper_manifest),
            "--label-col",
            str(cfg["cohort"]["label_col"]),
            "--group-col",
            str(cfg["cohort"]["patient_id_col"]),
            "--sample-col",
            str(cfg["cohort"]["sample_id_col"]),
            "--lockbox-fraction",
            str(cfg["split"]["lockbox_fraction"]),
            "--seed",
            str(cfg["split"]["seed"]),
            "--out-dir",
            str(dirs["manifest"]),
            "--prefix",
            manifest_prefix,
        ]
        run_cmd(split_cmd, repo_root)

    partition = validate_partition(read_rows(dev_manifest), read_rows(lockbox_manifest), read_rows(paper_manifest))
    (dirs["manifest"] / "split_membership_lock.json").write_text(json.dumps({
        "split_seed": cfg["split"]["seed"], "cv_model_seed": cfg["cv"]["seed"], **partition
    }, indent=2) + "\n")

    stage1_folds_path = prepare_stage_folds(
        repo_root=repo_root,
        cfg=cfg,
        stage_cfg=dict(cfg["selection_stage"]),
        dev_manifest=dev_manifest,
        out_path=dirs["manifest"] / "folds_stage1_select_5x3.tsv",
    )
    stage2_folds_path = prepare_stage_folds(
        repo_root=repo_root,
        cfg=cfg,
        stage_cfg=dict(cfg["confirm_stage"]),
        dev_manifest=dev_manifest,
        out_path=dirs["manifest"] / "folds_stage2_confirm_5x20.tsv",
    )

    sample_search_cmd = [
        sys.executable,
        "scripts/sample_paper_search_space.py",
        "--config",
        str(cfg_path),
        "--k",
        str(int(args.k)),
        "--out-dir",
        str(dirs["search_space"]),
        "--seed",
        str(args.seed),
    ]
    run_cmd(sample_search_cmd, repo_root)

    fs_specs = read_json(dirs["search_space"] / "fs_search_specs.json")
    model_specs = read_json(dirs["search_space"] / "model_search_specs.json")
    stage1_folds_df = pd.read_csv(stage1_folds_path, sep="\t", dtype=str)

    selection_candidates = build_selection_candidates(
        fs_specs=fs_specs,
        default_prefilter_k=int(cfg["prefilter"]["default_k"]),
        logistic_prefilter_k=int(cfg["prefilter"]["logistic_k"]),
        prefilter_method=str(cfg["prefilter"]["method"]),
    )
    model_candidates = build_model_candidates(
        model_specs=model_specs,
        default_prefilter_k=int(cfg["prefilter"]["default_k"]),
        logistic_prefilter_k=int(cfg["prefilter"]["logistic_k"]),
        class_weight_mode=str(cfg["training"]["class_weight_mode"]),
        tune_metric=str(cfg["training"]["tune_metric"]),
    )
    joint_candidates = build_joint_candidates(
        selection_candidates=selection_candidates,
        model_candidates=model_candidates,
        k=int(args.k),
    )

    stage1_dirs = dirs["stage1_select"]
    stage2_dirs = dirs["stage2_confirm"]

    write_candidate_specs(
        selection_candidates,
        stage1_dirs["fs_benchmark"],
        "selection_candidate_id",
        "candidate_spec_json",
    )
    write_candidate_specs(
        model_candidates,
        stage1_dirs["model_benchmark"],
        "model_candidate_id",
        "candidate_spec_json",
    )
    write_joint_model_candidate_specs(
        joint_candidates,
        stage1_dirs["model_benchmark"],
    )

    stage1_selection_manifest_path = stage1_dirs["summary"] / "selection_candidate_manifest.tsv"
    stage1_model_manifest_path = stage1_dirs["summary"] / "model_candidate_manifest.tsv"
    stage1_joint_manifest_path = stage1_dirs["summary"] / "joint_candidate_manifest.tsv"
    stage1_fs_task_manifest_path = stage1_dirs["summary"] / "fs_fold_task_manifest.tsv"
    stage1_model_task_manifest_path = stage1_dirs["summary"] / "model_fold_task_manifest.tsv"

    write_tsv(stage1_selection_manifest_path, selection_candidates)
    write_tsv(stage1_model_manifest_path, model_candidates)
    write_tsv(stage1_joint_manifest_path, joint_candidates)

    stage1_fs_tasks = build_fs_task_manifest(
        selection_candidates=selection_candidates,
        folds_df=stage1_folds_df,
        fs_run_base_dir=stage1_dirs["fs_benchmark"],
        dev_manifest_tsv=dev_manifest,
        folds_tsv=stage1_folds_path,
    )
    stage1_model_tasks = build_model_task_manifest(
        joint_candidates=joint_candidates,
        folds_df=stage1_folds_df,
        fs_run_base_dir=stage1_dirs["fs_benchmark"],
        model_run_base_dir=stage1_dirs["model_benchmark"],
        dev_manifest_tsv=dev_manifest,
        folds_tsv=stage1_folds_path,
    )
    write_tsv(stage1_fs_task_manifest_path, stage1_fs_tasks)
    write_tsv(stage1_model_task_manifest_path, stage1_model_tasks)

    execution_plan = {
        "config": str(cfg_path.resolve()),
        "k": int(args.k),
        "run_root": str(root.resolve()),
        "manifest": {
            "paper_manifest_tsv": str(paper_manifest.resolve()),
            "dev_manifest_tsv": str(dev_manifest.resolve()),
            "lockbox_manifest_tsv": str(lockbox_manifest.resolve()),
        },
        "inputs": {
            "probe_matrix": str(cfg["inputs"]["probe_matrix"]),
            "blacklist_chen": str(cfg["inputs"]["blacklist_chen"]),
            "blacklist_snp": str(cfg["inputs"]["blacklist_snp"]),
            "probe_metadata": str(cfg["inputs"]["probe_metadata"]),
        },
        "search_space": {
            "fs_search_specs_json": str((dirs["search_space"] / "fs_search_specs.json").resolve()),
            "model_search_specs_json": str((dirs["search_space"] / "model_search_specs.json").resolve()),
        },
        "stable_panel_policy": {
            "method": str(cfg["stable_panel"]["method"]),
            "variant_mode": str(cfg["stable_panel"]["variant_mode"]),
        },
        "selection_rule": dict(cfg.get("selection") or {}),
        "stages": {
            "stage1_select": {
                "stage_name": "stage1_select",
                "root": str(stage1_dirs["root"].resolve()),
                "fs_run_base_dir": str(stage1_dirs["fs_benchmark"].resolve()),
                "model_run_base_dir": str(stage1_dirs["model_benchmark"].resolve()),
                "selection_dir": str(stage1_dirs["selection"].resolve()),
                "summary_dir": str(stage1_dirs["summary"].resolve()),
                "manifest": {
                    "folds_tsv": str(stage1_folds_path.resolve()),
                },
                "candidate_manifests": {
                    "selection_candidate_manifest_tsv": str(stage1_selection_manifest_path.resolve()),
                    "model_candidate_manifest_tsv": str(stage1_model_manifest_path.resolve()),
                    "joint_candidate_manifest_tsv": str(stage1_joint_manifest_path.resolve()),
                    "fs_fold_task_manifest_tsv": str(stage1_fs_task_manifest_path.resolve()),
                    "model_fold_task_manifest_tsv": str(stage1_model_task_manifest_path.resolve()),
                },
                "outputs": {
                    "fs_candidate_summary_tsv": str((stage1_dirs["summary"] / "fs_candidate_summary.tsv").resolve()),
                    "model_candidate_summary_tsv": str((stage1_dirs["summary"] / "model_candidate_summary.tsv").resolve()),
                    "winner_json": str((stage1_dirs["selection"] / "stage1_winning_candidate.json").resolve()),
                    "ranking_tsv": str((stage1_dirs["selection"] / "stage1_candidate_ranking.tsv").resolve()),
                },
                "counts": {
                    "n_selection_candidates": len(selection_candidates),
                    "n_model_candidates": len(model_candidates),
                    "n_joint_candidates": len(joint_candidates),
                    "n_outer_folds": int(cfg["selection_stage"]["outer_splits"]) * int(cfg["selection_stage"]["outer_repeats"]),
                    "n_fs_tasks": len(stage1_fs_tasks),
                    "n_model_tasks": len(stage1_model_tasks),
                },
                "pending_steps": [
                    "run_stage1_fs_benchmark",
                    "aggregate_stage1_fs_candidates",
                    "run_stage1_model_benchmark",
                    "aggregate_stage1_model_candidates",
                    "select_stage1_winner",
                ],
            },
            "stage2_confirm": {
                "stage_name": "stage2_confirm",
                "root": str(stage2_dirs["root"].resolve()),
                "fs_run_base_dir": str(stage2_dirs["fs_locked"].resolve()),
                "model_run_base_dir": str(stage2_dirs["model_locked"].resolve()),
                "stable_panel_dir": str(stage2_dirs["stable_panel"].resolve()),
                "summary_dir": str(stage2_dirs["summary"].resolve()),
                "manifest": {
                    "folds_tsv": str(stage2_folds_path.resolve()),
                },
                "candidate_manifests": {
                    "selection_candidate_manifest_tsv": str((stage2_dirs["summary"] / "selection_candidate_manifest.tsv").resolve()),
                    "model_candidate_manifest_tsv": str((stage2_dirs["summary"] / "model_candidate_manifest.tsv").resolve()),
                    "joint_candidate_manifest_tsv": str((stage2_dirs["summary"] / "joint_candidate_manifest.tsv").resolve()),
                    "fs_fold_task_manifest_tsv": str((stage2_dirs["summary"] / "fs_fold_task_manifest.tsv").resolve()),
                    "model_fold_task_manifest_tsv": str((stage2_dirs["summary"] / "model_fold_task_manifest.tsv").resolve()),
                },
                "outputs": {
                    "fs_candidate_summary_tsv": str((stage2_dirs["summary"] / "fs_candidate_summary.tsv").resolve()),
                    "model_candidate_summary_tsv": str((stage2_dirs["summary"] / "model_candidate_summary.tsv").resolve()),
                    "locked_pipeline_json": str((stage2_dirs["summary"] / "locked_pipeline_from_stage1.json").resolve()),
                    "lockbox_eval_dir": str(dirs["lockbox_eval"].resolve()),
                },
                "counts": {
                    "n_outer_folds": int(cfg["confirm_stage"]["outer_splits"]) * int(cfg["confirm_stage"]["outer_repeats"]),
                },
                "locked_manifests_ready": False,
                "pending_steps": [
                    "build_stage2_locked_manifests_from_stage1_winner",
                    "run_stage2_locked_fs",
                    "aggregate_stage2_locked_fs_and_build_rankrule_panel",
                    "run_stage2_locked_model",
                    "aggregate_stage2_locked_model",
                    "evaluate_lockbox",
                    "evaluate_external_main",
                    "evaluate_external_stress_ffpe",
                    "evaluate_external_stress_epic",
                    "evaluate_external_stress_metastatic",
                ],
            },
        },
    }
    (dirs["summary"] / "execution_plan.json").write_text(json.dumps(execution_plan, indent=2) + "\n", encoding="utf-8")

    next_steps = f"""#!/usr/bin/env bash
set -euo pipefail

# Prepared by scripts/run_paper_k_pipeline.py
# K={int(args.k)}
# Run root: {root}

echo "Stage 1 selection run root: {stage1_dirs['root']}"
echo "Stage 1 selection manifest: {stage1_selection_manifest_path}"
echo "Stage 1 model manifest: {stage1_model_manifest_path}"
echo "Stage 1 FS task manifest: {stage1_fs_task_manifest_path}"
echo "Stage 1 model task manifest: {stage1_model_task_manifest_path}"
echo
echo "Run one Stage 1 FS task:"
echo "  ./.venv/bin/python scripts/run_paper_fs_task.py --execution-plan {dirs['summary'] / 'execution_plan.json'} --task-manifest {stage1_fs_task_manifest_path} --task-index 0"
echo
echo "Run one Stage 1 model task:"
echo "  ./.venv/bin/python scripts/run_paper_model_task.py --execution-plan {dirs['summary'] / 'execution_plan.json'} --task-manifest {stage1_model_task_manifest_path} --task-index 0"
echo
echo "Aggregate Stage 1 FS candidates:"
echo "  ./.venv/bin/python scripts/aggregate_paper_fs_candidates.py --execution-plan {dirs['summary'] / 'execution_plan.json'} --stage stage1_select"
echo
echo "Aggregate Stage 1 model candidates:"
echo "  ./.venv/bin/python scripts/aggregate_paper_model_candidates.py --execution-plan {dirs['summary'] / 'execution_plan.json'} --stage stage1_select"
echo
echo "Select Stage 1 winner:"
echo "  ./.venv/bin/python scripts/select_best_paper_candidate.py --execution-plan {dirs['summary'] / 'execution_plan.json'} --stage stage1_select"
echo
echo "Build Stage 2 locked manifests from Stage 1 winner:"
echo "  ./.venv/bin/python scripts/build_paper_stage2_locked_manifests.py --execution-plan {dirs['summary'] / 'execution_plan.json'}"
"""
    (dirs["summary"] / "next_steps.sh").write_text(next_steps, encoding="utf-8")
    print(
        json.dumps(
            {
                "run_root": str(root),
                "execution_plan_json": str((dirs["summary"] / "execution_plan.json")),
                "stage1_selection_candidate_manifest_tsv": str(stage1_selection_manifest_path),
                "stage1_model_candidate_manifest_tsv": str(stage1_model_manifest_path),
                "stage1_joint_candidate_manifest_tsv": str(stage1_joint_manifest_path),
                "stage1_fs_fold_task_manifest_tsv": str(stage1_fs_task_manifest_path),
                "stage1_model_fold_task_manifest_tsv": str(stage1_model_task_manifest_path),
                "stage2_folds_tsv": str(stage2_folds_path),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
