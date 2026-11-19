#!/usr/bin/env python3
"""Build a transparent frequency+rank based stable probe panel from CV fold selections.

This implements the "no-relationship" stable-panel variant:
- Primary ranking rule: frequency in top-K desc, then Borda rank score desc.
- Exports probe-level stability statistics, panel candidates, and overlap diagnostics.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


@dataclass
class FoldSelection:
    fold_id: str
    rank_map: Dict[str, int]
    top_panel_set: set[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build stable panel v1 from fold-level selected probe files.")
    parser.add_argument("--selected-run-dir", required=True, help="Run dir containing repeat*/fold*/<method>/ files.")
    parser.add_argument("--method", default="rare_class_rank_shift", help="Selection method folder under each fold (default: rare_class_rank_shift).")
    parser.add_argument("--rank-k", type=int, default=200, help="Max ranked probes to load from selected_order_K<k>.tsv.")
    parser.add_argument("--panel-k", type=int, default=30, help="Target stable panel size and top-K frequency definition.")
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Output directory. Default: <selected-run-dir>/stable_panel_v1",
    )
    parser.add_argument(
        "--freq-weight",
        type=float,
        default=0.7,
        help="Weight for top-K frequency in composite score (default: 0.7).",
    )
    parser.add_argument(
        "--rank-weight",
        type=float,
        default=0.3,
        help="Weight for normalized Borda rank score in composite score (default: 0.3).",
    )
    return parser.parse_args()


def make_fold_id(run_dir: Path, method_dir: Path) -> str:
    rel = method_dir.relative_to(run_dir)
    parts = rel.parts
    if len(parts) >= 3:
        return f"{parts[0]}/{parts[1]}"
    return str(rel)


def load_fold_selections(run_dir: Path, method: str, rank_k: int, panel_k: int) -> List[FoldSelection]:
    fold_dirs = sorted(run_dir.glob(f"repeat*/fold*/{method}"))
    if not fold_dirs:
        raise FileNotFoundError(f"No fold directories found under {run_dir} for method={method}")

    out: List[FoldSelection] = []
    for method_dir in fold_dirs:
        order_path = method_dir / f"selected_order_K{rank_k}.tsv"
        probes_path = method_dir / f"selected_probes_K{panel_k}.tsv"
        fold_id = make_fold_id(run_dir, method_dir)

        if order_path.exists():
            order_df = pd.read_csv(order_path, sep="\t")
            if "probe_id" not in order_df.columns:
                raise ValueError(f"Missing probe_id column in {order_path}")
            if "rank" not in order_df.columns:
                order_df = order_df.reset_index().rename(columns={"index": "rank"})
                order_df["rank"] = order_df["rank"].astype(int) + 1
            order_df["rank"] = pd.to_numeric(order_df["rank"], errors="coerce").astype("Int64")
            order_df = order_df.dropna(subset=["rank"])
            order_df["rank"] = order_df["rank"].astype(int)
            order_df = order_df[order_df["rank"] <= rank_k].copy()
            order_df = order_df.sort_values("rank")
            rank_map = dict(zip(order_df["probe_id"].astype(str), order_df["rank"].astype(int)))
            top_panel_set = {p for p, r in rank_map.items() if r <= panel_k}
        elif probes_path.exists():
            probes_df = pd.read_csv(probes_path, sep="\t")
            if "probe_id" not in probes_df.columns:
                raise ValueError(f"Missing probe_id column in {probes_path}")
            probes = probes_df["probe_id"].astype(str).tolist()
            rank_map = {p: i + 1 for i, p in enumerate(probes[:rank_k])}
            top_panel_set = set(probes[:panel_k])
        else:
            raise FileNotFoundError(f"Missing both {order_path.name} and {probes_path.name} in {method_dir}")

        out.append(FoldSelection(fold_id=fold_id, rank_map=rank_map, top_panel_set=top_panel_set))

    return out


def summarize_probes(
    folds: List[FoldSelection],
    rank_k: int,
    panel_k: int,
    freq_weight: float,
    rank_weight: float,
) -> pd.DataFrame:
    n_folds = len(folds)
    probes = sorted({probe for fold in folds for probe in fold.rank_map.keys()})
    rows: List[Dict[str, float | int | str]] = []

    max_borda = float(n_folds * rank_k)
    for probe in probes:
        ranks = [fold.rank_map.get(probe) for fold in folds]
        selected_ranks = [r for r in ranks if r is not None]

        freq_top_panel = sum(1 for r in selected_ranks if r <= panel_k)
        freq_top_rankk = len(selected_ranks)
        sel_rate_panel = float(freq_top_panel) / float(n_folds)
        sel_rate_rankk = float(freq_top_rankk) / float(n_folds)

        borda_score = float(sum((rank_k - r + 1) for r in selected_ranks))
        borda_norm = borda_score / max_borda if max_borda > 0 else 0.0

        arr = np.array(selected_ranks, dtype=float) if selected_ranks else np.array([], dtype=float)
        mean_rank = float(arr.mean()) if arr.size else np.nan
        median_rank = float(np.median(arr)) if arr.size else np.nan
        rank_sd = float(arr.std(ddof=1)) if arr.size > 1 else 0.0

        composite_score = float(freq_weight * sel_rate_panel + rank_weight * borda_norm)
        rows.append(
            {
                "probe_id": probe,
                "n_folds_top_panel": int(freq_top_panel),
                "n_folds_top_rankk": int(freq_top_rankk),
                "selection_rate_top_panel": sel_rate_panel,
                "selection_rate_top_rankk": sel_rate_rankk,
                "borda_score": borda_score,
                "borda_norm": borda_norm,
                "mean_rank_if_selected": mean_rank,
                "median_rank_if_selected": median_rank,
                "rank_sd_if_selected": rank_sd,
                "composite_score": composite_score,
            }
        )

    df = pd.DataFrame(rows)
    df = df.sort_values(
        by=["n_folds_top_panel", "borda_score", "mean_rank_if_selected", "probe_id"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)
    df["panel_rank_rule"] = np.arange(1, len(df) + 1)

    df_comp = df.sort_values(
        by=["composite_score", "n_folds_top_panel", "borda_score", "mean_rank_if_selected", "probe_id"],
        ascending=[False, False, False, True, True],
    ).reset_index(drop=True)[["probe_id"]]
    df = df.merge(df_comp.reset_index().rename(columns={"index": "panel_rank_composite"}), on="probe_id")
    df["panel_rank_composite"] = df["panel_rank_composite"].astype(int) + 1
    return df


def build_presence_matrix(folds: List[FoldSelection], panel_probes: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, int | str]] = []
    for fold in folds:
        row: Dict[str, int | str] = {"fold_id": fold.fold_id}
        for probe in panel_probes:
            row[probe] = 1 if probe in fold.top_panel_set else 0
        row["n_selected_in_panel"] = int(sum(row[p] for p in panel_probes))
        rows.append(row)
    return pd.DataFrame(rows).sort_values("fold_id").reset_index(drop=True)


def pairwise_jaccard(folds: List[FoldSelection]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    long_rows: List[Dict[str, float | str | int]] = []
    n = len(folds)
    matrix = np.ones((n, n), dtype=float)

    for i in range(n):
        for j in range(i + 1, n):
            a = folds[i].top_panel_set
            b = folds[j].top_panel_set
            inter = len(a & b)
            union = len(a | b)
            jac = float(inter) / float(union) if union > 0 else np.nan
            matrix[i, j] = jac
            matrix[j, i] = jac
            long_rows.append(
                {
                    "fold_i": folds[i].fold_id,
                    "fold_j": folds[j].fold_id,
                    "intersection": inter,
                    "union": union,
                    "jaccard": jac,
                }
            )

    long_df = pd.DataFrame(long_rows).sort_values(["fold_i", "fold_j"]).reset_index(drop=True)
    tri = long_df["jaccard"].to_numpy(dtype=float)
    summary = pd.DataFrame(
        [
            {
                "n_pairs": int(len(tri)),
                "jaccard_mean": float(np.mean(tri)) if tri.size else np.nan,
                "jaccard_median": float(np.median(tri)) if tri.size else np.nan,
                "jaccard_std": float(np.std(tri, ddof=1)) if tri.size > 1 else 0.0,
                "jaccard_min": float(np.min(tri)) if tri.size else np.nan,
                "jaccard_q25": float(np.quantile(tri, 0.25)) if tri.size else np.nan,
                "jaccard_q75": float(np.quantile(tri, 0.75)) if tri.size else np.nan,
                "jaccard_max": float(np.max(tri)) if tri.size else np.nan,
            }
        ]
    )
    mat_df = pd.DataFrame(matrix, index=[f.fold_id for f in folds], columns=[f.fold_id for f in folds])
    mat_df.index.name = "fold_id"
    return long_df, summary, mat_df


def frequency_histogram(stats_df: pd.DataFrame, n_folds: int) -> pd.DataFrame:
    counts = stats_df["n_folds_top_panel"].value_counts().to_dict()
    rows = [{"n_folds_selected": i, "n_probes": int(counts.get(i, 0))} for i in range(0, n_folds + 1)]
    return pd.DataFrame(rows).sort_values("n_folds_selected").reset_index(drop=True)


def main() -> int:
    args = parse_args()
    if args.freq_weight < 0 or args.rank_weight < 0:
        raise SystemExit("freq-weight and rank-weight must be non-negative")
    if (args.freq_weight + args.rank_weight) == 0:
        raise SystemExit("freq-weight + rank-weight must be > 0")

    run_dir = Path(args.selected_run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir / "stable_panel_v1"
    out_dir.mkdir(parents=True, exist_ok=True)

    folds = load_fold_selections(run_dir=run_dir, method=args.method, rank_k=args.rank_k, panel_k=args.panel_k)
    n_folds = len(folds)

    stats_df = summarize_probes(
        folds=folds,
        rank_k=args.rank_k,
        panel_k=args.panel_k,
        freq_weight=args.freq_weight,
        rank_weight=args.rank_weight,
    )

    panel_rule_df = stats_df.head(args.panel_k).copy()
    panel_comp_df = stats_df.sort_values("panel_rank_composite").head(args.panel_k).copy()
    panel_rule_probes = panel_rule_df["probe_id"].astype(str).tolist()

    presence_df = build_presence_matrix(folds=folds, panel_probes=panel_rule_probes)
    jacc_long_df, jacc_summary_df, jacc_matrix_df = pairwise_jaccard(folds=folds)
    freq_hist_df = frequency_histogram(stats_df=stats_df, n_folds=n_folds)

    stats_path = out_dir / f"probe_stability_stats_top{args.rank_k}.tsv"
    panel_rule_path = out_dir / f"stable_panel_v1_rankrule_top{args.panel_k}.tsv"
    panel_comp_path = out_dir / f"stable_panel_v1_composite_top{args.panel_k}.tsv"
    presence_path = out_dir / f"panel_top{args.panel_k}_presence_by_fold.tsv"
    jacc_long_path = out_dir / f"jaccard_pairs_top{args.panel_k}.tsv"
    jacc_summary_path = out_dir / f"jaccard_summary_top{args.panel_k}.tsv"
    jacc_matrix_path = out_dir / f"jaccard_matrix_top{args.panel_k}.tsv"
    hist_path = out_dir / f"frequency_histogram_top{args.panel_k}.tsv"
    meta_path = out_dir / "stable_panel_v1_meta.json"

    stats_df.to_csv(stats_path, sep="\t", index=False)
    panel_rule_df.to_csv(panel_rule_path, sep="\t", index=False)
    panel_comp_df.to_csv(panel_comp_path, sep="\t", index=False)
    presence_df.to_csv(presence_path, sep="\t", index=False)
    jacc_long_df.to_csv(jacc_long_path, sep="\t", index=False)
    jacc_summary_df.to_csv(jacc_summary_path, sep="\t", index=False)
    jacc_matrix_df.to_csv(jacc_matrix_path, sep="\t")
    freq_hist_df.to_csv(hist_path, sep="\t", index=False)

    meta = {
        "selected_run_dir": str(run_dir),
        "method": args.method,
        "rank_k": args.rank_k,
        "panel_k": args.panel_k,
        "n_folds": n_folds,
        "freq_weight": args.freq_weight,
        "rank_weight": args.rank_weight,
        "primary_ranking_rule": "n_folds_top_panel desc, borda_score desc, mean_rank_if_selected asc, probe_id asc",
        "composite_ranking_rule": "freq_weight*selection_rate_top_panel + rank_weight*borda_norm",
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"Folds loaded: {n_folds}")
    print(f"Wrote: {stats_path}")
    print(f"Wrote: {panel_rule_path}")
    print(f"Wrote: {panel_comp_path}")
    print(f"Wrote: {presence_path}")
    print(f"Wrote: {jacc_summary_path}")
    print(f"Wrote: {hist_path}")
    print(f"Wrote: {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
