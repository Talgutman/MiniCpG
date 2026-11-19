#!/usr/bin/env python3
"""Module 1: normal tissue identity signal at MiniCpG probes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_AUDIT = "outputs/paper_protocolb_v1/post_presentation_normal_biology_20260527/module0_metadata_audit/normal_sample_metadata_audit.tsv"
DEFAULT_OUT = "outputs/paper_protocolb_v1/post_presentation_normal_biology_20260527/module1_normal_tissue_identity"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["compute", "plot", "all"], default="all")
    p.add_argument("--normal-audit", default=DEFAULT_AUDIT)
    p.add_argument("--out-dir", default=DEFAULT_OUT)
    p.add_argument("--random-state", type=int, default=17)
    p.add_argument("--n-estimators", type=int, default=500)
    return p.parse_args()


def read_tsv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing input: {path}")
    return pd.read_csv(path, sep="\t", **kwargs)


def probe_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if str(c).startswith("cg")]


def pca_scores(x: pd.DataFrame, meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    arr = x.to_numpy(dtype=float)
    arr = arr - arr.mean(axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(arr, full_matrices=False)
    scores = u[:, :5] * s[:5]
    denom = np.sum(s**2)
    explained = (s**2) / denom if denom else np.zeros_like(s)
    coord = meta.copy()
    for i in range(scores.shape[1]):
        coord[f"PC{i + 1}"] = scores[:, i]
    expl = pd.DataFrame(
        {
            "pc": [f"PC{i + 1}" for i in range(min(10, len(explained)))],
            "variance_explained": explained[: min(10, len(explained))],
        }
    )
    return coord, expl


def eta_squared(values: pd.Series, groups: pd.Series) -> float:
    y = pd.to_numeric(values, errors="coerce")
    valid = y.notna() & groups.notna()
    y = y.loc[valid]
    g = groups.loc[valid]
    if len(y) < 2:
        return np.nan
    grand = float(y.mean())
    ss_total = float(((y - grand) ** 2).sum())
    if ss_total <= 0:
        return 0.0
    ss_between = 0.0
    for _, vals in y.groupby(g):
        ss_between += len(vals) * (float(vals.mean()) - grand) ** 2
    return ss_between / ss_total


def normal_lineage_evidence(df: pd.DataFrame, probes: list[str]) -> pd.DataFrame:
    rows = []
    groups = df["project_id"].astype(str)
    for probe in probes:
        vals = pd.to_numeric(df[probe], errors="coerce")
        med = vals.groupby(groups).median()
        mean = vals.groupby(groups).mean()
        dynamic_range = float(med.max() - med.min())
        rows.append(
            {
                "probe_id": probe,
                "n_samples": int(vals.notna().sum()),
                "n_tissues": int(med.notna().sum()),
                "normal_tissue_dynamic_range": dynamic_range,
                "normal_tissue_median_sd": float(med.std(ddof=0)),
                "normal_tissue_mean_sd": float(mean.std(ddof=0)),
                "normal_eta_squared": float(eta_squared(vals, groups)),
                "top_normal_tissue_high": str(med.idxmax()),
                "top_normal_tissue_low": str(med.idxmin()),
                "top_normal_tissue_high_median": float(med.max()),
                "top_normal_tissue_low_median": float(med.min()),
            }
        )
    out = pd.DataFrame(rows)
    for col in ["normal_tissue_dynamic_range", "normal_tissue_median_sd", "normal_eta_squared"]:
        rank = out[col].rank(ascending=True, pct=True)
        out[f"{col}_percentile"] = rank
    out["normal_lineage_score"] = out[
        [
            "normal_tissue_dynamic_range_percentile",
            "normal_tissue_median_sd_percentile",
            "normal_eta_squared_percentile",
        ]
    ].mean(axis=1)
    out["normal_composition_status"] = "not_used"
    return out.sort_values("normal_lineage_score", ascending=False)


def normal_classifier(df: pd.DataFrame, probes: list[str], random_state: int, n_estimators: int) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
    from sklearn.model_selection import StratifiedKFold

    x = df[probes].apply(pd.to_numeric, errors="coerce")
    med = x.median(axis=0, skipna=True)
    x = x.fillna(med)
    y = df["project_id"].astype(str).to_numpy()
    classes = np.array(sorted(pd.unique(y)))
    min_count = int(pd.Series(y).value_counts().min())
    n_splits = min(5, min_count)
    if n_splits < 2:
        raise SystemExit("Need at least two normal samples per class for CV.")
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    pred = np.empty(len(y), dtype=object)
    proba = np.zeros((len(y), len(classes)), dtype=float)
    importances = []
    fold_rows = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(x, y), start=1):
        clf = ExtraTreesClassifier(
            n_estimators=n_estimators,
            random_state=random_state + fold,
            class_weight="balanced",
            max_features=0.5,
            n_jobs=-1,
        )
        clf.fit(x.iloc[train_idx], y[train_idx])
        fold_pred = clf.predict(x.iloc[test_idx])
        fold_proba_raw = clf.predict_proba(x.iloc[test_idx])
        pred[test_idx] = fold_pred
        class_to_i = {c: i for i, c in enumerate(clf.classes_)}
        for j, c in enumerate(classes):
            if c in class_to_i:
                proba[test_idx, j] = fold_proba_raw[:, class_to_i[c]]
        importances.append(pd.Series(clf.feature_importances_, index=probes, name=f"fold_{fold}"))
        fold_rows.append(
            {
                "fold": fold,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
                "accuracy": float(accuracy_score(y[test_idx], fold_pred)),
                "macro_f1": float(f1_score(y[test_idx], fold_pred, average="macro", zero_division=0)),
                "balanced_accuracy": float(balanced_accuracy_score(y[test_idx], fold_pred)),
            }
        )
    pred_df = df[["project_id", "sample_id", "patient_id", "sample_type", "sample_type_code", "tissue_type", "matched_tumor_split_status"]].copy()
    pred_df["y_true"] = y
    pred_df["y_pred"] = pred
    pred_df["is_correct"] = (pred_df["y_true"] == pred_df["y_pred"]).astype(int)
    pred_df["max_probability"] = proba.max(axis=1)
    order = np.argsort(-proba, axis=1)
    pred_df["true_class_rank"] = [int(np.where(order[i] == np.where(classes == y[i])[0][0])[0][0] + 1) for i in range(len(y))]
    for j, c in enumerate(classes):
        pred_df[f"proba__{c}"] = proba[:, j]
    metrics = pd.DataFrame(fold_rows)
    aggregate = {
        "fold": "aggregate",
        "n_train": np.nan,
        "n_test": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "n_splits": n_splits,
        "n_classes": int(len(classes)),
        "n_samples": int(len(y)),
    }
    metrics = pd.concat([metrics, pd.DataFrame([aggregate])], ignore_index=True)
    imp = pd.concat(importances, axis=1)
    imp_summary = pd.DataFrame(
        {
            "probe_id": probes,
            "normal_classifier_importance_mean": imp.mean(axis=1).to_numpy(),
            "normal_classifier_importance_sd": imp.std(axis=1, ddof=0).to_numpy(),
        }
    ).sort_values("normal_classifier_importance_mean", ascending=False)
    return pred_df, metrics, imp_summary


def compute(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = read_tsv(Path(args.normal_audit), dtype=str)
    probes = probe_columns(df)
    x = df[probes].apply(pd.to_numeric, errors="coerce")
    x = x.fillna(x.median(axis=0, skipna=True))
    meta_cols = ["project_id", "sample_id", "patient_id", "sample_type", "sample_type_code", "tissue_type", "matched_tumor_split_status", "locked_model_normal_primary_subset"]
    meta = df[[c for c in meta_cols if c in df.columns]].copy()
    pca, expl = pca_scores(x, meta)
    evidence = normal_lineage_evidence(df, probes)
    pred, metrics, importance = normal_classifier(df, probes, args.random_state, args.n_estimators)
    evidence = evidence.merge(importance, on="probe_id", how="left")
    evidence["normal_lineage_score_with_classifier"] = evidence[
        ["normal_lineage_score", "normal_classifier_importance_mean"]
    ].rank(pct=True).mean(axis=1)
    evidence = evidence.sort_values("normal_lineage_score_with_classifier", ascending=False)

    counts = df.groupby("project_id", as_index=False).agg(n_samples=("sample_id", "nunique"), n_patients=("patient_id", "nunique")).sort_values("n_samples", ascending=False)
    pca.to_csv(out_dir / "normal_tissue_pca_30cpg.tsv", sep="\t", index=False)
    expl.to_csv(out_dir / "normal_tissue_pca_variance.tsv", sep="\t", index=False)
    evidence.to_csv(out_dir / "probe_normal_lineage_evidence.tsv", sep="\t", index=False)
    pred.to_csv(out_dir / "normal_tissue_classifier_predictions.tsv", sep="\t", index=False)
    metrics.to_csv(out_dir / "normal_tissue_separability_cv_metrics.tsv", sep="\t", index=False)
    importance.to_csv(out_dir / "normal_tissue_separability_feature_importance.tsv", sep="\t", index=False)
    counts.to_csv(out_dir / "normal_tissue_counts.tsv", sep="\t", index=False)

    agg = metrics.loc[metrics["fold"].astype(str) == "aggregate"].iloc[0].to_dict()
    top_probes = evidence.head(8)
    lines = [
        "# Normal Tissue Identity Signal",
        "",
        "Module 1 output for the healthy/normal tissue probe biology decomposition plan.",
        "",
        "## Scope",
        f"- Normal samples: `{len(df)}`",
        f"- TCGA normal tissue labels/projects: `{df['project_id'].nunique()}`",
        f"- Probes: `{len(probes)}`",
        "- Normal composition was not analyzed in this module by user decision.",
        "",
        "## Normal Tissue Separability",
        f"- CV folds: `{int(agg['n_splits'])}`",
        f"- Accuracy: `{float(agg['accuracy']):.4f}`",
        f"- Macro-F1: `{float(agg['macro_f1']):.4f}`",
        f"- Balanced accuracy: `{float(agg['balanced_accuracy']):.4f}`",
        "",
        "## Top Normal-Lineage Evidence Probes",
    ]
    for row in top_probes.itertuples():
        lines.append(
            f"- `{row.probe_id}`: score `{row.normal_lineage_score_with_classifier:.3f}`, "
            f"range `{row.normal_tissue_dynamic_range:.3f}`, eta2 `{row.normal_eta_squared:.3f}`, "
            f"high `{row.top_normal_tissue_high}`, low `{row.top_normal_tissue_low}`"
        )
    lines.extend(
        [
            "",
            "## Outputs",
            "- `probe_normal_lineage_evidence.tsv`",
            "- `normal_tissue_pca_30cpg.tsv`",
            "- `normal_tissue_pca_variance.tsv`",
            "- `normal_tissue_classifier_predictions.tsv`",
            "- `normal_tissue_separability_cv_metrics.tsv`",
            "- `normal_tissue_separability_feature_importance.tsv`",
        ]
    )
    (out_dir / "normal_tissue_identity_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    manifest = {
        "out_dir": str(out_dir),
        "n_samples": int(len(df)),
        "n_classes": int(df["project_id"].nunique()),
        "n_probes": int(len(probes)),
        "accuracy": float(agg["accuracy"]),
        "macro_f1": float(agg["macro_f1"]),
        "balanced_accuracy": float(agg["balanced_accuracy"]),
        "top_normal_lineage_probes": top_probes["probe_id"].tolist(),
    }
    (out_dir / "normal_tissue_identity_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def plot(args: argparse.Namespace) -> None:
    import matplotlib as mpl

    mpl.use("Agg")
    import matplotlib.pyplot as plt

    out_dir = Path(args.out_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    pca = read_tsv(out_dir / "normal_tissue_pca_30cpg.tsv")
    expl = read_tsv(out_dir / "normal_tissue_pca_variance.tsv")
    evidence = read_tsv(out_dir / "probe_normal_lineage_evidence.tsv")
    metrics = read_tsv(out_dir / "normal_tissue_separability_cv_metrics.tsv")
    agg = metrics.loc[metrics["fold"].astype(str) == "aggregate"].iloc[0]

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.7,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6.3,
        }
    )
    fig = plt.figure(figsize=(7.1, 5.2), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.15, 1.0])
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])

    classes = sorted(pca["project_id"].unique())
    cmap = plt.get_cmap("tab20")
    colors = {c: cmap(i % 20) for i, c in enumerate(classes)}
    for c in classes:
        g = pca.loc[pca["project_id"] == c]
        ax_a.scatter(g["PC1"], g["PC2"], s=14, alpha=0.75, color=colors[c], linewidth=0, label=c.replace("TCGA-", ""))
    ax_a.set_xlabel(f"PC1 ({float(expl.loc[expl['pc'] == 'PC1', 'variance_explained'].iloc[0]) * 100:.1f}%)")
    ax_a.set_ylabel(f"PC2 ({float(expl.loc[expl['pc'] == 'PC2', 'variance_explained'].iloc[0]) * 100:.1f}%)")
    ax_a.legend(frameon=False, ncol=7, loc="upper center", bbox_to_anchor=(0.5, -0.13), handletextpad=0.2, columnspacing=0.8)

    metric_names = ["accuracy", "macro_f1", "balanced_accuracy"]
    vals = [float(agg[m]) for m in metric_names]
    labels = ["Accuracy", "Macro-F1", "Balanced\naccuracy"]
    ax_b.bar(np.arange(len(vals)), vals, color="#4c78a8", linewidth=0)
    ax_b.set_xticks(np.arange(len(vals)))
    ax_b.set_xticklabels(labels)
    ax_b.set_ylim(0, 1.0)
    ax_b.set_ylabel("Cross-validated score")
    for i, v in enumerate(vals):
        ax_b.text(i, v + 0.025, f"{v:.2f}", ha="center", va="bottom", fontsize=7)

    top = evidence.sort_values("normal_lineage_score_with_classifier", ascending=True).tail(12)
    y = np.arange(len(top))
    ax_c.barh(y, top["normal_lineage_score_with_classifier"], color="#72b7b2", linewidth=0)
    ax_c.set_yticks(y)
    ax_c.set_yticklabels(top["probe_id"])
    ax_c.set_xlabel("Normal-lineage evidence score")
    ax_c.set_xlim(0, 1.0)

    for ax, label in zip([ax_a, ax_b, ax_c], "ABC", strict=True):
        ax.text(-0.055, 1.04, label, transform=ax.transAxes, fontsize=11, fontweight="bold", va="bottom")
        ax.tick_params(width=0.6, length=2.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    out_png = fig_dir / "normal_tissue_identity_module1.png"
    out_pdf = fig_dir / "normal_tissue_identity_module1.pdf"
    fig.savefig(out_png, dpi=300, bbox_inches="tight", pad_inches=0.03, facecolor=fig.get_facecolor())
    fig.savefig(out_pdf, bbox_inches="tight", pad_inches=0.03, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(json.dumps({"figure_png": str(out_png), "figure_pdf": str(out_pdf)}, indent=2))


def main() -> None:
    args = parse_args()
    if args.mode in {"compute", "all"}:
        compute(args)
    if args.mode in {"plot", "all"}:
        plot(args)


if __name__ == "__main__":
    main()
