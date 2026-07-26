#!/usr/bin/env python3
"""Validate that model/panel/bucket paths are aligned to the active locked version."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Validate active locked run manifest consistency.")
    p.add_argument(
        "--manifest",
        default="config/active_locked_run.json",
        help="Path to active locked run JSON manifest.",
    )
    p.add_argument(
        "--out-json",
        default="",
        help="Optional output JSON path for validation report.",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Exit non-zero on warnings as well as errors.",
    )
    return p.parse_args()


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_tsv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _to_int(x: object, default: int = 0) -> int:
    try:
        if x is None:
            return default
        s = str(x).strip()
        if s == "":
            return default
        return int(float(s))
    except Exception:
        return default


def _panel_probe_ids(panel_rows: list[dict[str, str]]) -> list[str]:
    if not panel_rows:
        return []
    # Expected column is probe_id.
    keys = list(panel_rows[0].keys())
    if "probe_id" in keys:
        col = "probe_id"
        return [str(r.get(col, "")).strip() for r in panel_rows if str(r.get(col, "")).strip()]
    # Fallback for unexpected schema.
    for candidate in keys:
        if "probe" in candidate.lower() and "id" in candidate.lower():
            return [str(r.get(candidate, "")).strip() for r in panel_rows if str(r.get(candidate, "")).strip()]
    return []


def _sha256_lines(lines: list[str]) -> str:
    payload = "\n".join(lines).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest)
    if not manifest_path.exists():
        raise SystemExit(f"Manifest not found: {manifest_path}")

    m = _read_json(manifest_path)

    required_keys = [
        "locked_run_name",
        "cluster_locked_run_dir",
        "selected_pipeline_json",
        "stable_variant_json",
        "panel_tsv",
        "model_bundle_pkl",
        "lockbox_metrics_tsv",
        "external_current_panel_status_tsv",
        "external_ours_eval_main_protocolb_eligible_dir",
    ]
    missing_manifest_keys = [k for k in required_keys if k not in m]

    errors: list[str] = []
    warnings: list[str] = []

    if missing_manifest_keys:
        errors.append(f"missing_manifest_keys:{','.join(missing_manifest_keys)}")

    path_keys = [
        "cluster_locked_run_dir",
        "selected_pipeline_json",
        "stable_variant_json",
        "panel_tsv",
        "model_bundle_pkl",
        "lockbox_metrics_tsv",
        "external_current_panel_status_tsv",
    ]
    path_exists: dict[str, bool] = {}
    for k in path_keys:
        p = Path(str(m.get(k, "")))
        ok = p.exists()
        path_exists[k] = ok
        if not ok:
            errors.append(f"missing_path:{k}:{p}")

    report: dict[str, Any] = {
        "manifest_path": str(manifest_path.resolve()),
        "version": m.get("version", {}),
        "locked_run_name": m.get("locked_run_name"),
        "path_exists": path_exists,
    }

    if not errors:
        selected = _read_json(Path(m["selected_pipeline_json"]))
        stable = _read_json(Path(m["stable_variant_json"]))
        lockbox_rows = _read_tsv_rows(Path(m["lockbox_metrics_tsv"]))
        panel_rows = _read_tsv_rows(Path(m["panel_tsv"]))
        status_rows = _read_tsv_rows(Path(m["external_current_panel_status_tsv"]))

        probes = _panel_probe_ids(panel_rows)
        probes_sorted_unique = sorted(set(probes))
        panel_hash = _sha256_lines(probes_sorted_unique)

        selected_k = _to_int(selected.get("k", 0), 0)
        panel_n_unique = len(probes_sorted_unique)
        panel_n_rows = len(probes)
        if panel_n_unique != selected_k:
            errors.append(f"panel_unique_count_mismatch:selected_k={selected_k},panel_unique={panel_n_unique}")
        if panel_n_rows < selected_k:
            errors.append(f"panel_row_count_too_small:selected_k={selected_k},panel_rows={panel_n_rows}")

        stable_panel_path = str(stable.get("panel_tsv", ""))
        manifest_panel_path = str(m.get("panel_tsv", ""))
        if stable_panel_path and stable_panel_path != manifest_panel_path:
            errors.append("panel_path_mismatch:stable_variant_json.panel_tsv != manifest.panel_tsv")

        lockbox = lockbox_rows[0] if lockbox_rows else {}
        if not lockbox:
            errors.append("empty_lockbox_metrics_tsv")
        else:
            lockbox_model = str(lockbox.get("model", ""))
            lockbox_k = _to_int(lockbox.get("k", 0), 0)
            if lockbox_model != str(selected.get("model", "")):
                errors.append(
                    f"model_mismatch:selected={selected.get('model','')},lockbox={lockbox_model}"
                )
            if lockbox_k != selected_k:
                errors.append(f"k_mismatch:selected={selected_k},lockbox={lockbox_k}")

        # External current-panel status counts.
        external_missing_threshold_max = _to_int(m.get("external_missing_threshold_max", 1), default=1)
        n_total = len(status_rows)
        n_mapped = 0
        n_extract_ok = 0
        n_pass_threshold = 0
        pass_by_cohort: dict[str, int] = {}
        for r in status_rows:
            mapped = str(r.get("mapping_found", "0")) == "1"
            if mapped:
                n_mapped += 1
            ok = str(r.get("extraction_status", "")) in {"ok_full", "ok_partial"}
            if ok:
                n_extract_ok += 1
            if ok and _to_int(r.get("n_panel_missing", ""), default=999) <= external_missing_threshold_max:
                n_pass_threshold += 1
                c = str(r.get("tcga_label", "")).strip() or "__UNMAPPED__"
                pass_by_cohort[c] = pass_by_cohort.get(c, 0) + 1

        external_eval_dir = Path(str(m.get("external_ours_eval_main_protocolb_eligible_dir", "")))
        external_summary_path = external_eval_dir / "external_eval_summary.json"
        external_eval_n = None
        if external_summary_path.exists():
            ext = _read_json(external_summary_path)
            external_eval_n = _to_int(ext.get("n_samples", None), default=-1)
            if external_eval_n != n_pass_threshold:
                errors.append(
                    f"external_eval_n_mismatch:summary_n={external_eval_n},status_pass_threshold={n_pass_threshold},threshold={external_missing_threshold_max}"
                )
        else:
            warnings.append(f"missing_external_eval_summary:{external_summary_path}")

        report.update(
            {
                "selected_pipeline": selected,
                "stable_variant": stable,
                "panel_probe_count_rows": panel_n_rows,
                "panel_probe_count_unique": panel_n_unique,
                "panel_probe_sha256_unique_sorted": panel_hash,
                "lockbox_metrics_row": lockbox,
                "external_status_counts": {
                    "n_rows": n_total,
                    "n_mapped": n_mapped,
                    "n_extract_ok": n_extract_ok,
                    "external_missing_threshold_max": external_missing_threshold_max,
                    "n_pass_missing_threshold": n_pass_threshold,
                    "example_pass_by_cohort_top10": sorted(
                        pass_by_cohort.items(), key=lambda kv: (-kv[1], kv[0])
                    )[:10],
                },
                "external_eval_summary_n_samples": external_eval_n,
            }
        )

    report["errors"] = errors
    report["warnings"] = warnings
    report["status"] = "ok" if not errors and (not warnings or not args.strict) else "failed"

    txt = json.dumps(report, indent=2, sort_keys=False)
    print(txt)
    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(txt + "\n", encoding="utf-8")
        print(f"Wrote: {out}")

    if errors:
        return 2
    if warnings and args.strict:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
