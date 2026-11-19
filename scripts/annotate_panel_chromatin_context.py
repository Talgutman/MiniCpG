#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


CLASS_PRIORITY = {
    "PLS": 0,
    "pELS": 1,
    "dELS": 2,
    "CA-H3K4me3": 3,
    "CA-CTCF": 4,
    "CA-TF": 5,
    "CA": 6,
    "TF": 7,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Annotate panel CpGs with ENCODE SCREEN GRCh38 cCRE regulatory classes."
    )
    parser.add_argument("--panel-biology-tsv", required=True)
    parser.add_argument("--ccre-dir", required=True)
    parser.add_argument("--mechanism-screen-tsv", default="")
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def load_probes(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    required = ["probe_id", "panel_rank", "CpG_chrm", "CpG_beg", "CpG_end", "gene_list", "final_biological_label"]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"missing required columns in {path}: {missing}")
    probes = df[required].copy()
    # The manifest already uses a zero-based start and an exclusive end.
    # Thus [CpG_beg, CpG_end) spans the two bases of a CpG; do not subtract one.
    probes["bed_start"] = pd.to_numeric(probes["CpG_beg"], errors="raise").astype(int)
    probes["bed_end"] = pd.to_numeric(probes["CpG_end"], errors="raise").astype(int)
    invalid = probes["bed_end"] <= probes["bed_start"]
    if invalid.any():
        invalid_ids = ", ".join(probes.loc[invalid, "probe_id"].astype(str))
        raise ValueError(f"CpG_end must exceed CpG_beg in {path}: {invalid_ids}")
    return probes


def ccre_files(ccre_dir: Path) -> list[Path]:
    files = sorted(ccre_dir.glob("GRCh38-cCREs.*.bed"))
    if not files:
        raise FileNotFoundError(f"no GRCh38-cCREs.*.bed files found in {ccre_dir}")
    return files


def source_class_from_file(path: Path) -> str:
    name = path.name
    prefix = "GRCh38-cCREs."
    suffix = ".bed"
    if name.startswith(prefix) and name.endswith(suffix):
        return name[len(prefix) : -len(suffix)]
    return path.stem


def annotate_ccre(probes: pd.DataFrame, files: list[Path]) -> pd.DataFrame:
    by_chr: dict[str, list[dict]] = {}
    for row in probes.to_dict("records"):
        by_chr.setdefault(str(row["CpG_chrm"]), []).append(row)

    rows: list[dict] = []
    for bed_path in files:
        source_class = source_class_from_file(bed_path)
        with bed_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip() or line.startswith("#"):
                    continue
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 6:
                    continue
                chrom = parts[0]
                if chrom not in by_chr:
                    continue
                start = int(parts[1])
                end = int(parts[2])
                for probe in by_chr[chrom]:
                    if int(probe["bed_start"]) < end and int(probe["bed_end"]) > start:
                        rows.append(
                            {
                                "probe_id": probe["probe_id"],
                                "panel_rank": probe["panel_rank"],
                                "ccre_class_source_file": source_class,
                                "ccre_class": parts[5],
                                "ccre_accession_d": parts[3],
                                "ccre_accession_e": parts[4],
                                "ccre_chrom": chrom,
                                "ccre_start": start,
                                "ccre_end": end,
                                "ccre_overlap_bp": min(int(probe["bed_end"]), end)
                                - max(int(probe["bed_start"]), start),
                            }
                        )
    return pd.DataFrame(rows)


def collapse_classes(sub: pd.DataFrame) -> str:
    if sub.empty:
        return "none"
    classes = sorted(set(sub["ccre_class"].astype(str)), key=lambda x: CLASS_PRIORITY.get(x, 99))
    return ";".join(classes)


def best_class(sub: pd.DataFrame) -> str:
    if sub.empty:
        return "none"
    return sorted(set(sub["ccre_class"].astype(str)), key=lambda x: CLASS_PRIORITY.get(x, 99))[0]


def interpretation_label(classes: str) -> str:
    class_set = set(classes.split(";")) if classes and classes != "none" else set()
    if "PLS" in class_set:
        return "promoter_like_ccre"
    if "pELS" in class_set or "dELS" in class_set:
        return "enhancer_like_ccre"
    if "CA-H3K4me3" in class_set:
        return "accessible_h3k4me3_non_pls"
    if "CA-CTCF" in class_set:
        return "ctcf_accessible_ccre"
    if class_set:
        return "other_accessible_or_tf_ccre"
    return "no_screen_ccre_overlap"


def build_summary(probes: pd.DataFrame, overlaps: pd.DataFrame, mechanism: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for probe in probes.sort_values("panel_rank").itertuples(index=False):
        sub = overlaps.loc[overlaps["probe_id"] == probe.probe_id] if not overlaps.empty else pd.DataFrame()
        classes = collapse_classes(sub)
        row = {
            "panel_rank": probe.panel_rank,
            "probe_id": probe.probe_id,
            "gene_list": probe.gene_list,
            "CpG_chrm": probe.CpG_chrm,
            "CpG_beg": probe.CpG_beg,
            "CpG_end": probe.CpG_end,
            "final_biological_label": probe.final_biological_label,
            "n_ccre_overlaps": int(len(sub)),
            "ccre_classes": classes,
            "best_ccre_class": best_class(sub),
            "chromatin_context_call": interpretation_label(classes),
        }
        if not mechanism.empty:
            mech = mechanism.loc[mechanism["probe_id"] == probe.probe_id].copy()
            if not mech.empty:
                row["mechanism_expression_calls"] = ";".join(sorted(set(mech["mechanism_expression_call"].astype(str))))
                row["mechanism_genes"] = ";".join(sorted(set(mech["gene"].astype(str))))
            else:
                row["mechanism_expression_calls"] = ""
                row["mechanism_genes"] = ""
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    probes = load_probes(Path(args.panel_biology_tsv))
    files = ccre_files(Path(args.ccre_dir))
    overlaps = annotate_ccre(probes, files)
    overlaps = overlaps.sort_values(["panel_rank", "probe_id", "ccre_class", "ccre_start"])
    overlaps.to_csv(out_dir / "panel_probe_encode_screen_ccre_overlaps.tsv", sep="\t", index=False)

    mechanism = pd.DataFrame()
    if args.mechanism_screen_tsv:
        mechanism_path = Path(args.mechanism_screen_tsv)
        if mechanism_path.exists():
            mechanism = pd.read_csv(mechanism_path, sep="\t")

    summary = build_summary(probes, overlaps, mechanism)
    summary.to_csv(out_dir / "panel_probe_chromatin_context_summary.tsv", sep="\t", index=False)
    supported = summary.loc[
        summary["mechanism_expression_calls"].astype(str).str.contains("supported|associated", na=False)
    ].copy() if "mechanism_expression_calls" in summary.columns else summary.iloc[0:0].copy()
    supported.to_csv(out_dir / "expression_supported_probe_chromatin_context.tsv", sep="\t", index=False)

    meta = {
        "source": "ENCODE SCREEN Registry V4 Human GRCh38/hg38 cCREs by class",
        "source_url": "https://screen.wenglab.org/downloads",
        "ccre_dir": str(Path(args.ccre_dir)),
        "n_panel_probes": int(probes["probe_id"].nunique()),
        "n_ccre_files": len(files),
        "ccre_files": [str(path.name) for path in files],
        "n_overlap_rows": int(len(overlaps)),
        "n_probes_with_ccre_overlap": int(overlaps["probe_id"].nunique()) if not overlaps.empty else 0,
        "chromatin_context_counts": summary["chromatin_context_call"].value_counts(dropna=False).to_dict(),
        "out_dir": str(out_dir),
    }
    with (out_dir / "chromatin_context_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(meta, handle, indent=2, sort_keys=True)
    with (out_dir / "chromatin_context_summary.md").open("w", encoding="utf-8") as handle:
        handle.write("# Chromatin Context Summary\n\n")
        handle.write(f"- source: `{meta['source']}`\n")
        handle.write(f"- source_url: `{meta['source_url']}`\n")
        handle.write(f"- n_panel_probes: `{meta['n_panel_probes']}`\n")
        handle.write(f"- n_ccre_files: `{meta['n_ccre_files']}`\n")
        handle.write(f"- n_overlap_rows: `{meta['n_overlap_rows']}`\n")
        handle.write(f"- n_probes_with_ccre_overlap: `{meta['n_probes_with_ccre_overlap']}`\n")
        handle.write(f"- chromatin_context_counts: `{meta['chromatin_context_counts']}`\n")
    print(json.dumps(meta, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
