#!/usr/bin/env python3
"""Run locked MiniCpG on TCGA solid-tissue normal samples as forced tumor predictions."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd




def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["compute", "plot", "all"], default="all")
    p.add_argument("--model-bundle", required=True)
    p.add_argument("--normal-audit", required=True)
    p.add_argument("--tcga-dev", required=True)
    p.add_argument("--decomposition", required=True)
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


def read_tsv(path: str | Path, **kwargs) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"Missing input: {path}")
    return pd.read_csv(path, sep="\t", **kwargs)


def load_model(model_path: Path) -> tuple[object, list[str], list[str]]:
    with model_path.open("rb") as fh:
        bundle = pickle.load(fh)
    return bundle["model"], [str(x) for x in bundle["probe_ids"]], [str(x) for x in bundle["class_order"]]


def align_proba(model: object, proba_raw: np.ndarray, class_order: list[str]) -> np.ndarray:
    model_classes_raw = list(model.named_steps["clf"].classes_)
    if np.issubdtype(np.array(model_classes_raw).dtype, np.number):
        model_classes = [class_order[int(c)] for c in model_classes_raw]
    else:
        model_classes = [str(c) for c in model_classes_raw]
    out = np.zeros((proba_raw.shape[0], len(class_order)), dtype=float)
    class_to_i = {c: i for i, c in enumerate(model_classes)}
    for j, c in enumerate(class_order):
        if c in class_to_i:
            out[:, j] = proba_raw[:, class_to_i[c]]
    return out


def add_prediction_columns(df: pd.DataFrame, proba: np.ndarray, class_order: list[str]) -> pd.DataFrame:
    pred_i = np.argmax(proba, axis=1)
    order = np.argsort(-proba, axis=1)
    out = df.copy()
    out["forced_predicted_tumor_class"] = [class_order[i] for i in pred_i]
    out["forced_prediction_probability"] = proba[np.arange(len(out)), pred_i]
    native_ranks = []
    native_probs = []
    native_top3 = []
    native_top5 = []
    for i, native in enumerate(out["project_id"].astype(str).to_numpy()):
        if native in class_order:
            cls_i = class_order.index(native)
            rank = int(np.where(order[i] == cls_i)[0][0] + 1)
            native_ranks.append(rank)
            native_probs.append(float(proba[i, cls_i]))
            native_top3.append(int(rank <= 3))
            native_top5.append(int(rank <= 5))
        else:
            native_ranks.append(np.nan)
            native_probs.append(np.nan)
            native_top3.append(0)
            native_top5.append(0)
    out["native_tissue_class_probability"] = native_probs
    out["native_tissue_class_rank"] = native_ranks
    out["native_tissue_in_top3"] = native_top3
    out["native_tissue_in_top5"] = native_top5
    out["forced_prediction_matches_native_tissue"] = (
        out["forced_predicted_tumor_class"].astype(str) == out["project_id"].astype(str)
    ).astype(int)
    for j, c in enumerate(class_order):
        out[f"proba__{c}"] = proba[:, j]
    return out


def summarize_subset(pred: pd.DataFrame, subset_name: str, mask: pd.Series) -> dict[str, float | str | int]:
    sub = pred.loc[mask].copy()
    if sub.empty:
        return {"subset": subset_name, "n_samples": 0}
    return {
        "subset": subset_name,
        "n_samples": int(len(sub)),
        "n_native_tissues": int(sub["project_id"].nunique()),
        "n_predicted_tumor_classes": int(sub["forced_predicted_tumor_class"].nunique()),
        "forced_same_tissue_count": int(sub["forced_prediction_matches_native_tissue"].sum()),
        "forced_same_tissue_fraction": float(sub["forced_prediction_matches_native_tissue"].mean()),
        "native_tissue_top3_fraction": float(sub["native_tissue_in_top3"].mean()),
        "native_tissue_top5_fraction": float(sub["native_tissue_in_top5"].mean()),
        "median_native_tissue_rank": float(pd.to_numeric(sub["native_tissue_class_rank"], errors="coerce").median()),
        "median_forced_prediction_probability": float(pd.to_numeric(sub["forced_prediction_probability"], errors="coerce").median()),
        "median_native_tissue_probability": float(pd.to_numeric(sub["native_tissue_class_probability"], errors="coerce").median()),
    }


def class_summaries(pred: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    for subset, mask in {
        "all_normals": pd.Series(True, index=pred.index),
        "primary_leakage_safer": pred["locked_model_normal_primary_subset"].astype(str) == "1",
        "development_paired": pred["locked_model_normal_primary_subset"].astype(str) != "1",
    }.items():
        sub = pred.loc[mask].copy()
        for tissue, g in sub.groupby("project_id", sort=True):
            rows.append(
                {
                    "subset": subset,
                    "native_tissue": tissue,
                    "n_samples": int(len(g)),
                    "forced_same_tissue_count": int(g["forced_prediction_matches_native_tissue"].sum()),
                    "forced_same_tissue_fraction": float(g["forced_prediction_matches_native_tissue"].mean()),
                    "median_native_tissue_rank": float(pd.to_numeric(g["native_tissue_class_rank"], errors="coerce").median()),
                    "median_native_tissue_probability": float(pd.to_numeric(g["native_tissue_class_probability"], errors="coerce").median()),
                    "top_forced_prediction": str(g["forced_predicted_tumor_class"].value_counts().idxmax()),
                    "top_forced_prediction_count": int(g["forced_predicted_tumor_class"].value_counts().max()),
                }
            )
    by_tissue = pd.DataFrame(rows)
    pred_rows = []
    for subset, mask in {
        "all_normals": pd.Series(True, index=pred.index),
        "primary_leakage_safer": pred["locked_model_normal_primary_subset"].astype(str) == "1",
        "development_paired": pred["locked_model_normal_primary_subset"].astype(str) != "1",
    }.items():
        sub = pred.loc[mask].copy()
        vc = sub["forced_predicted_tumor_class"].value_counts()
        for cls, count in vc.items():
            pred_rows.append(
                {
                    "subset": subset,
                    "predicted_tumor_class": cls,
                    "n_forced_predictions": int(count),
                    "fraction": float(count / len(sub)) if len(sub) else np.nan,
                    "n_native_same_tissue": int(((sub["forced_predicted_tumor_class"] == cls) & (sub["project_id"] == cls)).sum()),
                }
            )
    by_pred = pd.DataFrame(pred_rows)
    return by_tissue, by_pred


def centroid_and_axis_link(pred: pd.DataFrame, dev_path: str, probes: list[str], decomp_path: str) -> pd.DataFrame:
    dev = read_tsv(dev_path)
    decomp = read_tsv(decomp_path)
    score_cols = [
        "normal_lineage_score_with_classifier",
        "tumor_lineage_score",
        "tumor_normal_shift_score",
        "tumor_composition_score",
    ]
    weights = decomp.set_index("probe_id")[score_cols].reindex(probes).apply(pd.to_numeric, errors="coerce").fillna(0.0)
    weights = weights.div(weights.sum(axis=0).replace(0, np.nan), axis=1).fillna(0.0)
    centroids = dev.groupby("project_id")[probes].median(numeric_only=False).apply(pd.to_numeric, errors="coerce")
    x = pred[probes].apply(pd.to_numeric, errors="coerce")
    rows = []
    for i, row in pred.iterrows():
        native = str(row["project_id"])
        pred_cls = str(row["forced_predicted_tumor_class"])
        native_centroid = centroids.loc[native] if native in centroids.index else pd.Series(np.nan, index=probes)
        pred_centroid = centroids.loc[pred_cls] if pred_cls in centroids.index else pd.Series(np.nan, index=probes)
        vals = x.loc[i, probes]
        native_abs = (vals - native_centroid).abs()
        pred_abs = (vals - pred_centroid).abs()
        rec = {
            "sample_id": row["sample_id"],
            "project_id": native,
            "forced_predicted_tumor_class": pred_cls,
            "locked_model_normal_primary_subset": row["locked_model_normal_primary_subset"],
            "forced_prediction_matches_native_tissue": row["forced_prediction_matches_native_tissue"],
            "mean_abs_distance_to_native_tumor_centroid": float(native_abs.mean()),
            "mean_abs_distance_to_forced_predicted_tumor_centroid": float(pred_abs.mean()),
            "predicted_minus_native_centroid_distance": float(pred_abs.mean() - native_abs.mean()),
        }
        for col in score_cols:
            axis = col.replace("_score", "").replace("_with_classifier", "")
            rec[f"{axis}_weighted_abs_distance_to_native_centroid"] = float((native_abs * weights[col]).sum())
            rec[f"{axis}_weighted_abs_distance_to_predicted_centroid"] = float((pred_abs * weights[col]).sum())
        rows.append(rec)
    return pd.DataFrame(rows)


def write_outputs(pred: pd.DataFrame, centroid_link: pd.DataFrame, out_dir: Path) -> None:
    subset_rows = [
        summarize_subset(pred, "all_normals", pd.Series(True, index=pred.index)),
        summarize_subset(pred, "primary_leakage_safer", pred["locked_model_normal_primary_subset"].astype(str) == "1"),
        summarize_subset(pred, "development_paired", pred["locked_model_normal_primary_subset"].astype(str) != "1"),
    ]
    subset_summary = pd.DataFrame(subset_rows)
    by_tissue, by_pred = class_summaries(pred)
    pred.to_csv(out_dir / "locked_minicpg_normal_forced_predictions.tsv", sep="\t", index=False)
    subset_summary.to_csv(out_dir / "locked_minicpg_normal_subset_summary.tsv", sep="\t", index=False)
    by_tissue.to_csv(out_dir / "locked_minicpg_normal_by_native_tissue_summary.tsv", sep="\t", index=False)
    by_pred.to_csv(out_dir / "locked_minicpg_normal_by_forced_prediction_summary.tsv", sep="\t", index=False)
    centroid_link.to_csv(out_dir / "locked_minicpg_normal_centroid_axis_link.tsv", sep="\t", index=False)
    for subset, mask in {
        "all_normals": pd.Series(True, index=pred.index),
        "primary_leakage_safer": pred["locked_model_normal_primary_subset"].astype(str) == "1",
    }.items():
        sub = pred.loc[mask]
        cross = pd.crosstab(sub["project_id"], sub["forced_predicted_tumor_class"], dropna=False)
        cross.to_csv(out_dir / f"locked_minicpg_normal_{subset}_native_by_forced_prediction_crosstab.tsv", sep="\t")

    primary = subset_summary.loc[subset_summary["subset"] == "primary_leakage_safer"].iloc[0]
    all_row = subset_summary.loc[subset_summary["subset"] == "all_normals"].iloc[0]
    primary_top_pred = by_pred.loc[by_pred["subset"] == "primary_leakage_safer"].head(8)
    primary_tissue = by_tissue.loc[by_tissue["subset"] == "primary_leakage_safer"].sort_values(
        ["forced_same_tissue_fraction", "n_samples"], ascending=[False, False]
    )
    centroid_primary = centroid_link.loc[centroid_link["locked_model_normal_primary_subset"].astype(str) == "1"]
    dist_by_correct = centroid_primary.groupby("forced_prediction_matches_native_tissue", as_index=False).agg(
        n_samples=("sample_id", "size"),
        median_distance_to_native_centroid=("mean_abs_distance_to_native_tumor_centroid", "median"),
        median_distance_to_predicted_centroid=("mean_abs_distance_to_forced_predicted_tumor_centroid", "median"),
        median_predicted_minus_native_distance=("predicted_minus_native_centroid_distance", "median"),
    )

    lines = [
        "# Locked MiniCpG On TCGA Solid-Tissue Normals",
        "",
        "The locked tumor-type MiniCpG model has no normal class. These outputs therefore describe forced tumor-class prediction behavior, not diagnostic normal/tumor accuracy.",
        "",
        "## Scope",
        f"- All normal samples: `{int(all_row.n_samples)}`",
        f"- Primary leakage-safer normal samples: `{int(primary.n_samples)}`",
        f"- Primary native tissues: `{int(primary.n_native_tissues)}`",
        "",
        "## Primary Leakage-Safer Subset",
        f"- Forced same-tissue predictions: `{int(primary.forced_same_tissue_count)}/{int(primary.n_samples)}` (`{float(primary.forced_same_tissue_fraction):.3f}`)",
        f"- Native tissue in top 3: `{float(primary.native_tissue_top3_fraction):.3f}`",
        f"- Native tissue in top 5: `{float(primary.native_tissue_top5_fraction):.3f}`",
        f"- Median native tissue rank: `{float(primary.median_native_tissue_rank):.2f}`",
        f"- Median forced-prediction probability: `{float(primary.median_forced_prediction_probability):.3f}`",
        f"- Median native-tissue probability: `{float(primary.median_native_tissue_probability):.3f}`",
        "",
        "## Top Forced Tumor Predictions In Primary Subset",
    ]
    for row in primary_top_pred.itertuples(index=False):
        lines.append(f"- `{row.predicted_tumor_class}`: `{row.n_forced_predictions}` samples (`{row.fraction:.3f}`), native same-tissue `{row.n_native_same_tissue}`")
    lines.extend(["", "## Native Tissue Summaries In Primary Subset"])
    for row in primary_tissue.itertuples(index=False):
        lines.append(
            f"- `{row.native_tissue}`: n=`{row.n_samples}`, same-tissue `{row.forced_same_tissue_count}/{row.n_samples}`, "
            f"median native rank `{row.median_native_tissue_rank:.1f}`, top forced `{row.top_forced_prediction}` (`{row.top_forced_prediction_count}`)"
        )
    lines.extend(["", "## Centroid Distance Link In Primary Subset"])
    for row in dist_by_correct.itertuples(index=False):
        status = "same_tissue_forced" if int(row.forced_prediction_matches_native_tissue) == 1 else "not_same_tissue_forced"
        lines.append(
            f"- `{status}`: n=`{row.n_samples}`, median native-centroid distance `{row.median_distance_to_native_centroid:.3f}`, "
            f"median forced-predicted-centroid distance `{row.median_distance_to_predicted_centroid:.3f}`"
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "- `locked_minicpg_normal_forced_predictions.tsv`",
            "- `locked_minicpg_normal_subset_summary.tsv`",
            "- `locked_minicpg_normal_by_native_tissue_summary.tsv`",
            "- `locked_minicpg_normal_by_forced_prediction_summary.tsv`",
            "- `locked_minicpg_normal_centroid_axis_link.tsv`",
            "- `locked_minicpg_normal_primary_leakage_safer_native_by_forced_prediction_crosstab.tsv`",
        ]
    )
    (out_dir / "locked_minicpg_normal_forced_prediction_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_summary(out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    subset = read_tsv(out_dir / "locked_minicpg_normal_subset_summary.tsv")
    by_tissue = read_tsv(out_dir / "locked_minicpg_normal_by_native_tissue_summary.tsv")
    by_pred = read_tsv(out_dir / "locked_minicpg_normal_by_forced_prediction_summary.tsv")
    centroid = read_tsv(out_dir / "locked_minicpg_normal_centroid_axis_link.tsv")

    primary_tissue = by_tissue.loc[by_tissue["subset"] == "primary_leakage_safer"].copy()
    primary_tissue = primary_tissue.sort_values("forced_same_tissue_fraction", ascending=True)
    primary_pred = by_pred.loc[by_pred["subset"] == "primary_leakage_safer"].copy().head(10)
    primary_centroid = centroid.loc[centroid["locked_model_normal_primary_subset"].astype(str) == "1"].copy()
    primary_centroid["forced_prediction_matches_native_tissue"] = pd.to_numeric(
        primary_centroid["forced_prediction_matches_native_tissue"], errors="coerce"
    ).fillna(0).astype(int)

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.4), gridspec_kw={"height_ratios": [1.0, 1.35]})
    ax = axes[0, 0]
    primary = subset.loc[subset["subset"] == "primary_leakage_safer"].iloc[0]
    metrics = [
        ("Same tissue", float(primary["forced_same_tissue_fraction"])),
        ("Native top 3", float(primary["native_tissue_top3_fraction"])),
        ("Native top 5", float(primary["native_tissue_top5_fraction"])),
    ]
    ax.bar([m[0] for m in metrics], [m[1] for m in metrics], color=["#3f7f93", "#73a57a", "#a5bd6b"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction")
    ax.set_title("Primary normal subset")
    for i, (_, v) in enumerate(metrics):
        ax.text(i, v + 0.025, f"{v:.2f}", ha="center", va="bottom", fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[0, 1]
    pred_plot = primary_pred.sort_values("n_forced_predictions", ascending=True)
    colors = ["#3f7f93" if row.n_native_same_tissue > 0 else "#8c8c8c" for row in pred_plot.itertuples()]
    ax.barh(pred_plot["predicted_tumor_class"], pred_plot["n_forced_predictions"], color=colors)
    ax.set_xlabel("Normal samples forced to class")
    ax.set_title("Most frequent forced tumor classes")
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1, 0]
    y = np.arange(len(primary_tissue))
    ax.barh(y, primary_tissue["forced_same_tissue_fraction"], color="#587da7")
    ax.set_yticks(y)
    ax.set_yticklabels(primary_tissue["native_tissue"], fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Forced same-tissue fraction")
    ax.set_title("By native normal tissue")
    for yi, row in enumerate(primary_tissue.itertuples(index=False)):
        ax.text(min(row.forced_same_tissue_fraction + 0.02, 0.98), yi, f"n={row.n_samples}", va="center", fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1, 1]
    groups = [
        ("Not same tissue", primary_centroid.loc[primary_centroid["forced_prediction_matches_native_tissue"] == 0]),
        ("Same tissue", primary_centroid.loc[primary_centroid["forced_prediction_matches_native_tissue"] == 1]),
    ]
    positions = [0, 1, 3, 4]
    values = [
        groups[0][1]["mean_abs_distance_to_native_tumor_centroid"].dropna(),
        groups[0][1]["mean_abs_distance_to_forced_predicted_tumor_centroid"].dropna(),
        groups[1][1]["mean_abs_distance_to_native_tumor_centroid"].dropna(),
        groups[1][1]["mean_abs_distance_to_forced_predicted_tumor_centroid"].dropna(),
    ]
    bp = ax.boxplot(values, positions=positions, widths=0.65, patch_artist=True, showfliers=False)
    for patch, color in zip(bp["boxes"], ["#a8c7d0", "#d8a870", "#a8c7d0", "#d8a870"]):
        patch.set_facecolor(color)
        patch.set_edgecolor("#333333")
    for part in ["whiskers", "caps", "medians"]:
        for artist in bp[part]:
            artist.set_color("#333333")
    ax.set_xticks([0.5, 3.5])
    ax.set_xticklabels(["Not same tissue", "Same tissue"])
    ax.set_ylabel("Mean absolute beta distance")
    ax.set_title("Distance to tumor centroids")
    ax.legend([bp["boxes"][0], bp["boxes"][1]], ["Native tissue centroid", "Forced class centroid"], frameon=False, loc="upper right")
    ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle("Locked MiniCpG Forced Predictions On TCGA Normal-Adjacent Tissue", fontsize=14, y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_dir / "locked_minicpg_normal_forced_prediction_summary.png", dpi=300)
    fig.savefig(fig_dir / "locked_minicpg_normal_forced_prediction_summary.pdf")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.mode in {"compute", "all"}:
        model, probes, class_order = load_model(Path(args.model_bundle))
        normals = read_tsv(args.normal_audit, dtype=str)
        missing = [p for p in probes if p not in normals.columns]
        if missing:
            raise SystemExit(f"Normal audit missing model probes: {missing}")
        x = normals[probes].apply(pd.to_numeric, errors="coerce")
        proba = align_proba(model, np.asarray(model.predict_proba(x), dtype=float), class_order)
        pred = add_prediction_columns(normals, proba, class_order)
        centroid_link = centroid_and_axis_link(pred, args.tcga_dev, probes, args.decomposition)
        write_outputs(pred, centroid_link, out_dir)
    if args.mode in {"plot", "all"}:
        plot_summary(out_dir)
    if args.mode == "plot":
        manifest_path = out_dir / "locked_minicpg_normal_manifest.json"
        if manifest_path.exists():
            print(manifest_path.read_text(encoding="utf-8").strip())
        else:
            print(json.dumps({"out_dir": str(out_dir), "mode": "plot"}, indent=2))
        return
    manifest = {
        "out_dir": str(out_dir),
        "n_all_normals": int(len(pred)),
        "n_primary_leakage_safer": int((pred["locked_model_normal_primary_subset"].astype(str) == "1").sum()),
        "n_model_probes": int(len(probes)),
        "n_model_classes": int(len(class_order)),
    }
    (out_dir / "locked_minicpg_normal_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
