#!/usr/bin/env python3
"""Extract fixed-panel matrix for external samples from mapped source files.

Supports two value paths:
1) Direct beta/value columns.
2) Signal-derived beta from paired methylated/unmethylated columns:
   beta = M / (M + U + 100).

The output includes per-sample flags so signal-derived samples are traceable.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd
from external_path_utils import resolve_external_data_source_path


BEGIN_MARKERS = {
    "!series_matrix_table_begin": "!series_matrix_table_end",
    "!dataset_table_begin": "!dataset_table_end",
}

MEASURE_PRIORITY = {
    "beta": 0,
    "value": 1,
    "methylated_signal": 2,
    "unmethylated_signal": 3,
    "pval": 9,
}


def _parse_mapped_column_index(value: object) -> int | None:
    """Accept integer-like mapping indices, including float-formatted strings."""
    raw = str(value).strip()
    if not raw or raw.lower() in {"nan", "none"}:
        return None
    try:
        return int(raw)
    except Exception:  # noqa: BLE001
        try:
            parsed = float(raw)
        except Exception:  # noqa: BLE001
            return None
        if not parsed.is_integer():
            return None
        return int(parsed)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build external panel matrix from source files + sample-column mapping.")
    p.add_argument("--mapping-tsv", required=True, help="external_sample_column_mapping.tsv")
    p.add_argument("--panel-tsv", required=True, help="Panel TSV with probe_id column")
    p.add_argument("--k", type=int, default=30, help="Use first K probes from panel")
    p.add_argument("--out-dir", required=True, help="Output directory")
    p.add_argument(
        "--keep-all-mapping-rows",
        action="store_true",
        help="Do not auto-filter the mapping TSV to include==1 rows. Useful for biology/reference extraction.",
    )
    return p.parse_args()


def _open_text(path: Path):
    opener = gzip.open if path.suffix == ".gz" else open
    return opener(path, "rt", encoding="utf-8", errors="replace")
def _strip_cell(x: str) -> str:
    return x.strip().strip('"').strip("'")


def _norm(x: str) -> str:
    return re.sub(r"\s+", " ", _strip_cell(x).lower()).strip()


def _split_line(line: str, delim: str) -> list[str]:
    return [_strip_cell(x) for x in line.rstrip("\n").split(delim)]


def _read_panel(panel_path: Path, k: int) -> list[str]:
    panel = pd.read_csv(panel_path, sep="\t", dtype=str)
    if "probe_id" not in panel.columns:
        raise SystemExit(f"Panel missing probe_id column: {panel_path}")
    if "panel_rank" in panel.columns:
        panel["panel_rank"] = pd.to_numeric(panel["panel_rank"], errors="coerce")
        panel = panel.sort_values(["panel_rank", "probe_id"], kind="mergesort")
    probes = [str(x).strip() for x in panel["probe_id"].tolist() if str(x).strip()]
    probes = probes[:k]
    if len(probes) < k:
        raise SystemExit(f"Panel has only {len(probes)} probes (<k={k})")
    return probes


def _has_probe_id_header(header_fields: list[str]) -> bool:
    if not header_fields:
        return False
    first = _norm(header_fields[0])
    return (
        first in {"", "id_ref", "id", "probe", "probes", "probeid", "probe_id", "cgid", "targetid", "target id"}
        or first.startswith("id_ref")
        or first.startswith("targetid")
        or bool(re.match(r"^cg\d{5,}$", first))
    )


def _looks_like_missing_probe_header(header_fields: list[str], next_fields: list[str]) -> bool:
    return (
        bool(header_fields)
        and bool(next_fields)
        and not _has_probe_id_header(header_fields)
        and re.match(r"^cg\d{5,}$", _norm(next_fields[0])) is not None
        and len(next_fields) == len(header_fields) + 1
    )


def _find_header_for_free_table(path: Path, max_lines: int = 2000) -> tuple[list[str], str, int]:
    with _open_text(path) as fh:
        for i, raw in enumerate(fh):
            if i >= max_lines:
                break
            line = raw.rstrip("\n")
            if not line:
                continue
            if line.startswith(("!", "^", "#")):
                continue
            tab_count = line.count("\t")
            comma_count = line.count(",")
            if max(tab_count, comma_count) < 1:
                continue
            delim = "\t" if tab_count >= comma_count else ","
            parts = _split_line(line, delim)
            if len(parts) < 2:
                continue
            next_raw = fh.readline()
            if next_raw:
                next_parts = _split_line(next_raw.rstrip("\n"), delim)
                if _looks_like_missing_probe_header(parts, next_parts):
                    parts = ["ID_REF"] + parts
            return parts, delim, i
    return [], "", -1


def _resolve_header(path: Path) -> tuple[list[str], str, str]:
    """Return (header, delimiter, mode)."""
    with _open_text(path) as fh:
        in_table = False
        end_marker = ""
        for raw in fh:
            line = raw.rstrip("\n")
            if not in_table:
                for begin, end in BEGIN_MARKERS.items():
                    if line.startswith(begin):
                        in_table = True
                        end_marker = end
                        break
                if in_table:
                    continue
                break
            if end_marker and line.startswith(end_marker):
                break
            if not line:
                continue
            hdr = _split_line(line, "\t")
            if len(hdr) >= 2:
                return hdr, "\t", "geo_marked_table"

    hdr, delim, _ = _find_header_for_free_table(path)
    if hdr:
        return hdr, delim, "free_table"
    return [], "", "no_header"


def _header_sample_start(header_fields: list[str]) -> int:
    if not header_fields:
        return 1
    if _has_probe_id_header(header_fields):
        return 1
    return 0


def _normalize_header_label(label: str) -> tuple[str, str, str]:
    raw_norm = _norm(label)
    key = raw_norm
    measure = "value"

    if re.search(r"(?:\bdetection\s*pval\b|\bpval\b)$", key):
        measure = "pval"
        key = re.sub(r"(?:\bdetection\s*pval\b|\bpval\b)$", "", key).strip()
    elif re.search(r"\bbeta\b$", key):
        measure = "beta"
        key = re.sub(r"\bbeta\b$", "", key).strip()
    elif re.search(r"\bunmethylated\s+signal\b$", key):
        measure = "unmethylated_signal"
        key = re.sub(r"\bunmethylated\s+signal\b$", "", key).strip()
    elif re.search(r"\bmethylated\s+signal\b$", key):
        measure = "methylated_signal"
        key = re.sub(r"\bmethylated\s+signal\b$", "", key).strip()

    key = re.sub(r"[\s_\-:;.,]+$", "", key).strip()
    return key, measure, raw_norm


def _parse_header_columns(header_fields: list[str]) -> list[dict[str, object]]:
    cols: list[dict[str, object]] = []
    start = _header_sample_start(header_fields)
    for idx in range(start, len(header_fields)):
        raw = str(header_fields[idx])
        key, measure, raw_norm = _normalize_header_label(raw)
        cols.append(
            {
                "index": idx,
                "raw": raw,
                "raw_norm": raw_norm,
                "key": key,
                "measure": measure,
            }
        )
    return cols


def _safe_get(parts: list[str], idx: int) -> str:
    if 0 <= idx < len(parts):
        return str(parts[idx]).strip()
    return ""


def _to_float(x: str) -> float | None:
    s = str(x).strip()
    if not s:
        return None
    try:
        return float(s)
    except Exception:  # noqa: BLE001
        return None


def _beta_from_signals(m_raw: str, u_raw: str) -> str:
    m = _to_float(m_raw)
    u = _to_float(u_raw)
    if m is None or u is None:
        return ""
    den = m + u + 100.0
    if den <= 0:
        return ""
    beta = m / den
    if beta < 0:
        beta = 0.0
    elif beta > 1:
        beta = 1.0
    return f"{beta:.8g}"


def _beta_from_m_value(m_raw: str) -> str:
    m = _to_float(m_raw)
    if m is None:
        return ""
    try:
        beta = 1.0 / (1.0 + math.pow(2.0, -m))
    except OverflowError:
        beta = 0.0 if m < 0 else 1.0
    if beta < 0:
        beta = 0.0
    elif beta > 1:
        beta = 1.0
    return f"{beta:.8g}"


def _pick_nearest(idxs: list[int], ref_idx: int) -> int | None:
    if not idxs:
        return None
    return sorted(idxs, key=lambda x: (abs(x - ref_idx), x))[0]


def _measure_rank(measure: str) -> int:
    return MEASURE_PRIORITY.get(measure, 5)


def _build_sample_specs(
    mapped_rows: pd.DataFrame,
    parsed_cols: list[dict[str, object]],
    header_len: int,
) -> tuple[dict[str, dict[str, object]], dict[str, dict[str, object]]]:
    """Return (sample_specs, sample_flags).

    sample_specs contain executable extraction plan.
    sample_flags contain metadata for status outputs.
    """
    by_idx = {int(c["index"]): c for c in parsed_cols}
    by_key_measure: dict[tuple[str, str], list[int]] = defaultdict(list)
    for c in parsed_cols:
        key = str(c["key"])
        meas = str(c["measure"])
        by_key_measure[(key, meas)].append(int(c["index"]))

    sample_specs: dict[str, dict[str, object]] = {}
    sample_flags: dict[str, dict[str, object]] = {}

    for _, r in mapped_rows.iterrows():
        sid = str(r.get("sample_id", ""))
        idx_raw = str(r.get("mapped_column_index", "")).strip()
        if not idx_raw:
            sample_flags[sid] = {
                "value_source": "mapped_index_missing",
                "signal_derived": 0,
                "signal_pair_available": 0,
                "mapped_measure": "",
            }
            continue
        idx = _parse_mapped_column_index(idx_raw)
        if idx is None:
            sample_flags[sid] = {
                "value_source": "mapped_index_invalid",
                "signal_derived": 0,
                "signal_pair_available": 0,
                "mapped_measure": "",
            }
            continue
        if idx < 0 or idx >= header_len:
            sample_flags[sid] = {
                "value_source": "mapped_index_out_of_range",
                "signal_derived": 0,
                "signal_pair_available": 0,
                "mapped_measure": "",
            }
            continue

        col = by_idx.get(idx)
        if col is None:
            sample_flags[sid] = {
                "value_source": "mapped_index_not_in_sample_cols",
                "signal_derived": 0,
                "signal_pair_available": 0,
                "mapped_measure": "",
            }
            continue

        key = str(col["key"])
        measure = str(col["measure"])

        value_transform = str(r.get("value_transform", "")).strip().lower()
        mapping_mode = str(r.get("mapping_mode", "")).strip().lower()
        inventory_status = str(r.get("inventory_status", "")).strip().lower()
        use_m_to_beta = (
            value_transform in {"m_value_to_beta", "m_to_beta"}
            or "m_value_to_beta" in mapping_mode
            or "m_value_to_beta" in inventory_status
        )

        # Direct value path.
        if measure in {"beta", "value"}:
            if use_m_to_beta:
                sample_specs[sid] = {"mode": "direct_m_to_beta", "value_idx": idx}
                value_source = f"direct_{measure}_m_to_beta"
            else:
                sample_specs[sid] = {"mode": "direct", "value_idx": idx}
                value_source = f"direct_{measure}"
            sample_flags[sid] = {
                "value_source": value_source,
                "signal_derived": 0,
                "signal_pair_available": 0,
                "mapped_measure": measure,
            }
            continue

        # pval path: remap to beta/value if available.
        if measure == "pval":
            beta_idxs = by_key_measure.get((key, "beta"), [])
            value_idxs = by_key_measure.get((key, "value"), [])
            cand = beta_idxs if beta_idxs else value_idxs
            if cand:
                use_idx = _pick_nearest(cand, idx)
                if use_idx is not None:
                    chosen_meas = "beta" if beta_idxs else "value"
                    sample_specs[sid] = {"mode": "direct", "value_idx": use_idx}
                    sample_flags[sid] = {
                        "value_source": f"pval_remap_{chosen_meas}",
                        "signal_derived": 0,
                        "signal_pair_available": 0,
                        "mapped_measure": measure,
                    }
                    continue
            # else try signal-pair remap.
            m_idx = _pick_nearest(by_key_measure.get((key, "methylated_signal"), []), idx)
            u_idx = _pick_nearest(by_key_measure.get((key, "unmethylated_signal"), []), idx)
            if m_idx is not None and u_idx is not None:
                sample_specs[sid] = {"mode": "signal_beta", "m_idx": m_idx, "u_idx": u_idx}
                sample_flags[sid] = {
                    "value_source": "pval_remap_signal_beta",
                    "signal_derived": 1,
                    "signal_pair_available": 1,
                    "mapped_measure": measure,
                }
                continue
            sample_flags[sid] = {
                "value_source": "pval_unresolved",
                "signal_derived": 0,
                "signal_pair_available": 0,
                "mapped_measure": measure,
            }
            continue

        # signal path -> derive beta.
        if measure in {"methylated_signal", "unmethylated_signal"}:
            if measure == "methylated_signal":
                m_idx = idx
                u_idx = _pick_nearest(by_key_measure.get((key, "unmethylated_signal"), []), idx)
            else:
                u_idx = idx
                m_idx = _pick_nearest(by_key_measure.get((key, "methylated_signal"), []), idx)
            if m_idx is not None and u_idx is not None:
                sample_specs[sid] = {"mode": "signal_beta", "m_idx": m_idx, "u_idx": u_idx}
                sample_flags[sid] = {
                    "value_source": "signal_derived_beta",
                    "signal_derived": 1,
                    "signal_pair_available": 1,
                    "mapped_measure": measure,
                }
            else:
                sample_flags[sid] = {
                    "value_source": "signal_pair_missing",
                    "signal_derived": 0,
                    "signal_pair_available": 0,
                    "mapped_measure": measure,
                }
            continue

        # Fallback.
        sample_flags[sid] = {
            "value_source": "unsupported_measure",
            "signal_derived": 0,
            "signal_pair_available": 0,
            "mapped_measure": measure,
        }

    return sample_specs, sample_flags


def _extract_values(
    path: Path,
    probe_set: set[str],
    sample_specs: dict[str, dict[str, object]],
) -> tuple[dict[str, dict[str, str]], set[str], int, str]:
    """Return (sample->probe->value, probes_found, rows_scanned, parse_mode)."""
    values: dict[str, dict[str, str]] = defaultdict(dict)
    probes_found: set[str] = set()
    rows_scanned = 0
    parse_mode = "unknown"

    hdr, delim, mode = _resolve_header(path)
    parse_mode = mode
    if not hdr:
        return values, probes_found, rows_scanned, parse_mode
    if not sample_specs:
        return values, probes_found, rows_scanned, parse_mode

    # Filter specs to valid header bounds.
    valid_specs: dict[str, dict[str, object]] = {}
    for sid, spec in sample_specs.items():
        if spec.get("mode") == "direct":
            idx = int(spec["value_idx"])
            if 0 <= idx < len(hdr):
                valid_specs[sid] = spec
        elif spec.get("mode") == "direct_m_to_beta":
            idx = int(spec["value_idx"])
            if 0 <= idx < len(hdr):
                valid_specs[sid] = spec
        elif spec.get("mode") == "signal_beta":
            m_idx = int(spec["m_idx"])
            u_idx = int(spec["u_idx"])
            if 0 <= m_idx < len(hdr) and 0 <= u_idx < len(hdr):
                valid_specs[sid] = spec
    if not valid_specs:
        return values, probes_found, rows_scanned, parse_mode

    direct_items = [(sid, int(sp["value_idx"])) for sid, sp in valid_specs.items() if sp.get("mode") == "direct"]
    m_to_beta_items = [
        (sid, int(sp["value_idx"])) for sid, sp in valid_specs.items() if sp.get("mode") == "direct_m_to_beta"
    ]
    signal_items = [
        (sid, int(sp["m_idx"]), int(sp["u_idx"])) for sid, sp in valid_specs.items() if sp.get("mode") == "signal_beta"
    ]

    # Parse data rows.
    try:
        with _open_text(path) as fh:
            if mode == "geo_marked_table":
                in_table = False
                end_marker = ""
                header_consumed = False
                for raw in fh:
                    line = raw.rstrip("\n")
                    if not in_table:
                        for begin, end in BEGIN_MARKERS.items():
                            if line.startswith(begin):
                                in_table = True
                                end_marker = end
                                break
                        continue
                    if end_marker and line.startswith(end_marker):
                        break
                    if not line:
                        continue
                    parts = _split_line(line, "\t")
                    if len(parts) < 2:
                        continue
                    if not header_consumed:
                        header_consumed = True
                        continue
                    probe_id = parts[0]
                    rows_scanned += 1
                    if probe_id not in probe_set:
                        continue
                    for sid, idx in direct_items:
                        values[sid][probe_id] = _safe_get(parts, idx)
                    for sid, idx in m_to_beta_items:
                        values[sid][probe_id] = _beta_from_m_value(_safe_get(parts, idx))
                    for sid, m_idx, u_idx in signal_items:
                        values[sid][probe_id] = _beta_from_signals(_safe_get(parts, m_idx), _safe_get(parts, u_idx))
                    probes_found.add(probe_id)
                    if len(probes_found) == len(probe_set):
                        break
            else:
                # free table: skip up to detected header line, then parse rows.
                _, _, header_line_idx = _find_header_for_free_table(path)
                if header_line_idx < 0:
                    return values, probes_found, rows_scanned, parse_mode
                for i, raw in enumerate(_open_text(path)):
                    if i <= header_line_idx:
                        continue
                    line = raw.rstrip("\n")
                    if not line:
                        continue
                    parts = _split_line(line, delim)
                    if len(parts) < 2:
                        continue
                    probe_id = parts[0]
                    rows_scanned += 1
                    if probe_id not in probe_set:
                        continue
                    for sid, idx in direct_items:
                        values[sid][probe_id] = _safe_get(parts, idx)
                    for sid, idx in m_to_beta_items:
                        values[sid][probe_id] = _beta_from_m_value(_safe_get(parts, idx))
                    for sid, m_idx, u_idx in signal_items:
                        values[sid][probe_id] = _beta_from_signals(_safe_get(parts, m_idx), _safe_get(parts, u_idx))
                    probes_found.add(probe_id)
                    if len(probes_found) == len(probe_set):
                        break
    except EOFError:
        parse_mode = f"{parse_mode}_truncated"

    return values, probes_found, rows_scanned, parse_mode


def main() -> int:
    args = parse_args()
    mapping_tsv = Path(args.mapping_tsv)
    panel_tsv = Path(args.panel_tsv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not mapping_tsv.exists():
        raise SystemExit(f"Missing mapping TSV: {mapping_tsv}")
    if not panel_tsv.exists():
        raise SystemExit(f"Missing panel TSV: {panel_tsv}")

    probes = _read_panel(panel_tsv, args.k)
    probe_set = set(probes)

    m = pd.read_csv(mapping_tsv, sep="\t", dtype=str).fillna("")
    if (not args.keep_all_mapping_rows) and "include" in m.columns:
        m = m[m["include"] == "1"].copy()
    req = {"gse_id", "sample_id", "data_source_path", "mapping_found", "mapped_column_index"}
    miss = req - set(m.columns)
    if miss:
        raise SystemExit(f"Mapping TSV missing required columns: {sorted(miss)}")

    status_rows: list[dict[str, str]] = []
    matrix_rows: list[dict[str, str]] = []

    grouped = m.groupby(["gse_id", "data_source_path"], dropna=False)
    for (gse_id, source_path), gdf in grouped:
        p = resolve_external_data_source_path(str(source_path))
        if not source_path or not p.exists():
            for _, r in gdf.iterrows():
                rr = dict(r)
                rr["rows_scanned"] = "0"
                rr["parse_mode"] = "missing_source_file"
                rr["panel_value_source"] = "missing_source_file"
                rr["signal_derived"] = "0"
                rr["signal_pair_available"] = "0"
                rr["mapped_measure"] = ""
                rr["extraction_status"] = "missing_source_file"
                rr["n_panel_found"] = "0"
                rr["n_panel_missing"] = str(len(probes))
                rr["missing_probes"] = ";".join(probes)
                rr["all_panel_missing"] = "1"
                status_rows.append(rr)
            continue

        hdr, _, header_mode = _resolve_header(p)
        parsed_cols = _parse_header_columns(hdr) if hdr else []

        mapped = gdf[gdf["mapping_found"] == "1"].copy()
        sample_specs: dict[str, dict[str, object]] = {}
        sample_flags: dict[str, dict[str, object]] = {}
        if not mapped.empty and hdr:
            sample_specs, sample_flags = _build_sample_specs(mapped, parsed_cols, len(hdr))

        sample_probe_vals, _, rows_scanned, parse_mode = _extract_values(p, probe_set, sample_specs)

        for _, r in gdf.iterrows():
            sid = str(r["sample_id"])
            rr = dict(r)
            rr["rows_scanned"] = str(rows_scanned)
            rr["parse_mode"] = parse_mode if parse_mode != "unknown" else header_mode

            if str(r["mapping_found"]) != "1":
                rr["panel_value_source"] = "column_unmapped"
                rr["signal_derived"] = "0"
                rr["signal_pair_available"] = "0"
                rr["mapped_measure"] = ""
                rr["extraction_status"] = "column_unmapped"
                rr["n_panel_found"] = "0"
                rr["n_panel_missing"] = str(len(probes))
                rr["missing_probes"] = ";".join(probes)
                rr["all_panel_missing"] = "1"
                status_rows.append(rr)
                continue

            flags = sample_flags.get(
                sid,
                {
                    "value_source": "mapped_unresolved",
                    "signal_derived": 0,
                    "signal_pair_available": 0,
                    "mapped_measure": "",
                },
            )
            rr["panel_value_source"] = str(flags.get("value_source", "mapped_unresolved"))
            rr["signal_derived"] = str(int(flags.get("signal_derived", 0)))
            rr["signal_pair_available"] = str(int(flags.get("signal_pair_available", 0)))
            rr["mapped_measure"] = str(flags.get("mapped_measure", ""))

            vals = sample_probe_vals.get(sid, {})
            present = [pr for pr in probes if pr in vals and str(vals[pr]).strip() != ""]
            missing = [pr for pr in probes if pr not in vals or str(vals.get(pr, "")).strip() == ""]
            rr["n_panel_found"] = str(len(present))
            rr["n_panel_missing"] = str(len(missing))
            rr["missing_probes"] = ";".join(missing)
            rr["all_panel_missing"] = "1" if len(present) == 0 else "0"
            if len(present) == len(probes):
                rr["extraction_status"] = "ok_full"
            elif len(present) > 0:
                rr["extraction_status"] = "ok_partial"
            else:
                if rr["panel_value_source"] in {"signal_pair_missing", "pval_unresolved", "mapped_unresolved"}:
                    rr["extraction_status"] = "mapped_unusable"
                else:
                    rr["extraction_status"] = "no_probe_values"
            status_rows.append(rr)

            mx = {
                "gse_id": str(r.get("gse_id", "")),
                "sample_id": sid,
                "tcga_label": str(r.get("tcga_label", "")),
                "target_bucket": str(r.get("target_bucket", "")),
                "data_source_type": str(r.get("data_source_type", "")),
                "data_source_path": str(r.get("data_source_path", "")),
                "mapping_mode": str(r.get("mapping_mode", "")),
                "mapped_measure": rr["mapped_measure"],
                "panel_value_source": rr["panel_value_source"],
                "signal_derived": rr["signal_derived"],
                "signal_pair_available": rr["signal_pair_available"],
                "extraction_status": rr["extraction_status"],
                "n_panel_found": rr["n_panel_found"],
            }
            for pr in probes:
                mx[pr] = vals.get(pr, "")
            matrix_rows.append(mx)

    status_df = pd.DataFrame(status_rows)
    status_df.to_csv(
        out_dir / "external_panel_extraction_sample_status.tsv",
        sep="\t",
        index=False,
        quoting=csv.QUOTE_MINIMAL,
    )

    matrix_df = pd.DataFrame(matrix_rows)
    if not matrix_df.empty:
        sort_cols = [c for c in ["target_bucket", "tcga_label", "gse_id", "sample_id"] if c in matrix_df.columns]
        matrix_df = matrix_df.sort_values(sort_cols, kind="mergesort").reset_index(drop=True)
    matrix_df.to_csv(out_dir / "external_panel_matrix_allow_missing1.tsv", sep="\t", index=False, quoting=csv.QUOTE_MINIMAL)
    if not matrix_df.empty:
        matrix_df.to_parquet(out_dir / "external_panel_matrix_allow_missing1.parquet", index=False)

    if not status_df.empty:
        by_bucket = (
            status_df.groupby(["target_bucket", "extraction_status"], dropna=False)
            .size()
            .reset_index(name="n_samples")
            .sort_values(["target_bucket", "extraction_status"], kind="mergesort")
        )
        by_bucket.to_csv(out_dir / "external_panel_extraction_by_bucket.tsv", sep="\t", index=False)

        by_source = (
            status_df.groupby(["target_bucket", "panel_value_source", "signal_derived"], dropna=False)
            .size()
            .reset_index(name="n_samples")
            .sort_values(["target_bucket", "panel_value_source", "signal_derived"], kind="mergesort")
        )
        by_source.to_csv(out_dir / "external_panel_value_source_by_bucket.tsv", sep="\t", index=False)

    summary = {
        "n_samples_input": int(len(m)),
        "n_samples_matrix_rows": int(len(matrix_df)),
        "n_samples_ok_full": int((status_df.get("extraction_status", pd.Series(dtype=str)) == "ok_full").sum()),
        "n_samples_ok_partial": int((status_df.get("extraction_status", pd.Series(dtype=str)) == "ok_partial").sum()),
        "n_samples_no_probe_values": int((status_df.get("extraction_status", pd.Series(dtype=str)) == "no_probe_values").sum()),
        "n_samples_mapped_unusable": int((status_df.get("extraction_status", pd.Series(dtype=str)) == "mapped_unusable").sum()),
        "n_samples_column_unmapped": int((status_df.get("extraction_status", pd.Series(dtype=str)) == "column_unmapped").sum()),
        "n_samples_missing_source_file": int((status_df.get("extraction_status", pd.Series(dtype=str)) == "missing_source_file").sum()),
        "n_samples_signal_derived": int((status_df.get("signal_derived", pd.Series(dtype=str)) == "1").sum()),
        "panel_value_source_counts": dict(Counter(status_df.get("panel_value_source", pd.Series(dtype=str)).astype(str).tolist())),
        "extraction_status_counts": dict(Counter(status_df.get("extraction_status", pd.Series(dtype=str)).astype(str).tolist())),
        "signal_formula": "beta = M / (M + U + 100)",
        "m_value_transform_formula": "beta = 2^M / (1 + 2^M)",
    }
    (out_dir / "external_panel_extraction_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))
    print(f"Wrote: {out_dir / 'external_panel_matrix_allow_missing1.tsv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
