#!/usr/bin/env python3
"""Build Results Fig. 4: domain-shift validation."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "methylation_classification_mpl_cache"),
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


PAPER_BG = "#fbfaf7"
INK = "#24313f"
MUTED = "#66717d"
GRID = "#e5ded3"
BLUE = "#2f6f9f"
TEAL = "#4f8a78"
GOLD = "#d39a3f"
ORANGE = "#c46a3a"
PURPLE = "#8b6f9f"
GREY = "#8b949e"

METRICS = ["macro_f1", "weighted_f1", "accuracy", "auroc"]
METRIC_LABELS = {
    "macro_f1": "Present-class macro-F1",
    "weighted_f1": "Weighted-F1",
    "accuracy": "Accuracy",
    "auroc": "AUROC",
}
METRIC_COLORS = {
    "macro_f1": BLUE,
    "weighted_f1": TEAL,
    "accuracy": GOLD,
    "auroc": PURPLE,
}
BUCKET_ORDER = ["main", "stress_metastatic", "stress_ffpe", "stress_epic"]
BUCKET_LABELS = {
    "main": "Primary external",
    "stress_metastatic": "Metastatic",
    "stress_ffpe": "FFPE",
    "stress_epic": "EPIC",
}
BUCKET_CONTEXT = {
    "main": "Primary tumors",
    "stress_metastatic": "Disease-context shift",
    "stress_ffpe": "Preservation shift",
    "stress_epic": "Platform shift",
}
BUCKET_COLORS = {
    "stress_epic": TEAL,
    "stress_ffpe": GOLD,
    "stress_metastatic": ORANGE,
}
DISPLAY_CLASS_LABELS = {
    "TCGA-GBM": "GBM/LGG",
    "TCGA-COAD": "COAD/READ",
    "TCGA-ESCA": "ESCA/STAD",
}
CONFUSION_PATHS = {
    "main": "external_main_beta_sanitized_current/eval/external_confusion_matrix_overall.tsv",
    "stress_epic": "external_stress_epic_beta_sanitized_current/eval/external_confusion_matrix_overall.tsv",
    "stress_ffpe": "external_stress_ffpe_beta_sanitized_current/eval/external_confusion_matrix_overall.tsv",
    "stress_metastatic": "external_stress_metastatic_beta_sanitized_current/eval/external_confusion_matrix_overall.tsv",
}
PREDICTION_PATHS = {
    "main": "external_main_beta_sanitized_current/eval/external_predictions_allow_missing1.tsv",
    "stress_epic": "external_stress_epic_beta_sanitized_current/eval/external_predictions_allow_missing1.tsv",
    "stress_ffpe": "external_stress_ffpe_beta_sanitized_current/eval/external_predictions_allow_missing1.tsv",
    "stress_metastatic": "external_stress_metastatic_beta_sanitized_current/eval/external_predictions_allow_missing1.tsv",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--active-manifest", default="config/active_locked_run.json")
    parser.add_argument("--run-root", default="outputs/paper_protocolb_v1/k_030")
    parser.add_argument(
        "--metrics-tsv",
        default=(
            "outputs/paper_protocolb_v1/k_030/sanitized_external_metrics_current/"
            "sanitized_external_metrics_long_raw.tsv"
        ),
    )
    parser.add_argument(
        "--out-dir",
        default="outputs/paper_protocolb_v1/manuscript_current/figures/fig4_domain_shift_validation_current",
    )
    parser.add_argument("--out-stem", default="fig4_domain_shift_validation")
    parser.add_argument(
        "--variant",
        default="all",
        choices=["all", "lollipop", "heatmap"],
        help="Metric-delta panel style to render. 'all' writes both comparison variants.",
    )
    parser.add_argument(
        "--class-panel",
        default="heatmap",
        choices=["heatmap", "waterfall"],
        help="Class-level stress-test panel style.",
    )
    parser.add_argument("--formats", nargs="+", default=["png", "pdf"], choices=["png", "pdf"])
    return parser.parse_args()


def validate_active_run(manifest: str) -> dict[str, object]:
    proc = subprocess.run(
        [sys.executable, "scripts/validate_active_locked_run.py", "--manifest", manifest],
        check=True,
        text=True,
        capture_output=True,
    )
    payload = json.loads(proc.stdout)
    if payload.get("status") != "ok":
        raise SystemExit(f"Active locked-run validation failed: {payload.get('status')}")
    return payload


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.facecolor": PAPER_BG,
            "axes.facecolor": PAPER_BG,
            "savefig.facecolor": PAPER_BG,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def add_panel_letter(ax: plt.Axes, letter: str, x: float = -0.10, y: float = 1.05) -> None:
    ax.text(
        x,
        y,
        letter,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=11,
        fontweight="bold",
        color=INK,
    )


def _present_class_macro_f1_from_predictions(path: Path) -> tuple[float, int, int]:
    if not path.exists():
        raise SystemExit(f"Missing prediction table for present-class macro-F1: {path}")
    df = pd.read_csv(path, sep="\t", dtype=str)
    required = {"tcga_label", "y_pred"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Prediction table missing columns {sorted(missing)}: {path}")
    y_true = df["tcga_label"].astype(str)
    y_pred = df["y_pred"].astype(str)
    present_classes = sorted(y_true.unique())
    score = _macro_f1_for_labels(y_true, y_pred, present_classes)
    return float(score), int(len(df)), int(len(present_classes))


def _macro_f1_for_labels(y_true: pd.Series, y_pred: pd.Series, labels: list[str]) -> float:
    scores: list[float] = []
    for label in labels:
        true_is_label = y_true.eq(label)
        pred_is_label = y_pred.eq(label)
        tp = int((true_is_label & pred_is_label).sum())
        fp = int((~true_is_label & pred_is_label).sum())
        fn = int((true_is_label & ~pred_is_label).sum())
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        scores.append((2.0 * precision * recall / (precision + recall)) if (precision + recall) else 0.0)
    return float(np.mean(scores)) if scores else float("nan")


def load_full_metrics(path: Path, run_root: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    required = {"bucket", "analysis", "model", "n_samples", "n_classes_in_truth", *METRICS}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Metrics table missing columns: {sorted(missing)}")
    out = df.loc[(df["analysis"].eq("full_bucket")) & (df["model"].eq("ours"))].copy()
    present = set(out["bucket"].astype(str))
    missing_buckets = [b for b in BUCKET_ORDER if b not in present]
    if missing_buckets:
        raise SystemExit(f"Metrics table missing full-bucket rows: {missing_buckets}")
    out["bucket"] = pd.Categorical(out["bucket"], categories=BUCKET_ORDER, ordered=True)
    out = out.sort_values("bucket").reset_index(drop=True)
    for metric in METRICS:
        out[metric] = pd.to_numeric(out[metric], errors="raise")
    out["n_samples"] = pd.to_numeric(out["n_samples"], errors="raise").astype(int)
    out["n_classes_in_truth"] = pd.to_numeric(out["n_classes_in_truth"], errors="raise").astype(int)
    if "macro_f1_fixed_label_axis" in out.columns:
        out["macro_f1_fixed_label_axis"] = pd.to_numeric(out["macro_f1_fixed_label_axis"], errors="raise")
    else:
        out["macro_f1_fixed_label_axis"] = out["macro_f1"].astype(float)
    for idx, row in out.iterrows():
        bucket = str(row["bucket"])
        present_macro, n_samples, n_classes = _present_class_macro_f1_from_predictions(
            run_root / PREDICTION_PATHS[bucket]
        )
        if n_samples != int(row["n_samples"]):
            raise SystemExit(
                f"Prediction/metric sample count mismatch for {bucket}: "
                f"{n_samples} predictions vs {int(row['n_samples'])} metrics"
            )
        if n_classes != int(row["n_classes_in_truth"]):
            raise SystemExit(
                f"Prediction/metric truth-class count mismatch for {bucket}: "
                f"{n_classes} prediction classes vs {int(row['n_classes_in_truth'])} metrics"
            )
        out.at[idx, "macro_f1_present_classes"] = present_macro
        out.at[idx, "macro_f1"] = present_macro
    return out


def build_delta_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    main = metrics.loc[metrics["bucket"].astype(str).eq("main")].iloc[0]
    rows = []
    for _, row in metrics.iterrows():
        bucket = str(row["bucket"])
        for metric in METRICS:
            rows.append(
                {
                    "bucket": bucket,
                    "bucket_label": BUCKET_LABELS[bucket],
                    "context": BUCKET_CONTEXT[bucket],
                    "metric": metric,
                    "metric_label": METRIC_LABELS[metric],
                    "value": float(row[metric]),
                    "main_value": float(main[metric]),
                    "delta_vs_main": float(row[metric] - main[metric]),
                    "n_samples": int(row["n_samples"]),
                    "n_classes_in_truth": int(row["n_classes_in_truth"]),
                }
            )
    return pd.DataFrame(rows)


def build_methydeep_delta_metrics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t")
    required = {"bucket", "analysis", "model", "n_samples", "n_classes_in_truth", *METRICS}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Metrics table missing columns for MethyDeep comparison: {sorted(missing)}")
    buckets = ["stress_metastatic", "stress_ffpe", "stress_epic"]
    h2h = df.loc[df["analysis"].astype(str).eq("head_to_head") & df["bucket"].isin(buckets)].copy()
    rows = []
    for bucket in buckets:
        sub = h2h.loc[h2h["bucket"].astype(str).eq(bucket)]
        if set(sub["model"].astype(str)) != {"ours", "methydeep"}:
            raise SystemExit(f"Missing head-to-head model rows for {bucket}")
        ours = sub.loc[sub["model"].astype(str).eq("ours")].iloc[0]
        meth = sub.loc[sub["model"].astype(str).eq("methydeep")].iloc[0]
        for metric in METRICS:
            ours_value = float(pd.to_numeric(ours[metric], errors="raise"))
            meth_value = float(pd.to_numeric(meth[metric], errors="raise"))
            rows.append(
                {
                    "bucket": bucket,
                    "bucket_label": BUCKET_LABELS[bucket],
                    "context": BUCKET_CONTEXT[bucket],
                    "n_samples": int(pd.to_numeric(ours["n_samples"], errors="raise")),
                    "n_classes_in_truth": int(pd.to_numeric(ours["n_classes_in_truth"], errors="raise")),
                    "metric": metric,
                    "metric_label": METRIC_LABELS[metric],
                    "ours_value": ours_value,
                    "methydeep_value": meth_value,
                    "delta_ours_minus_methydeep": ours_value - meth_value,
                    "winner": "MiniCpG"
                    if ours_value > meth_value
                    else "MethyDeep"
                    if meth_value > ours_value
                    else "Tie",
                }
            )
    return pd.DataFrame(rows)


def _short_class(label: str) -> str:
    label = str(label)
    return DISPLAY_CLASS_LABELS.get(label, label.replace("TCGA-", ""))


def _read_recall_by_class(path: Path, bucket: str) -> pd.DataFrame:
    if not path.exists():
        raise SystemExit(f"Missing confusion matrix: {path}")
    counts = pd.read_csv(path, sep="\t", index_col=0)
    if counts.empty:
        raise SystemExit(f"Empty confusion matrix: {path}")
    counts = counts.apply(pd.to_numeric, errors="coerce").fillna(0)
    rows = []
    for klass, values in counts.iterrows():
        support = float(values.sum())
        if support <= 0:
            continue
        correct = float(values.get(klass, 0.0))
        rows.append(
            {
                "bucket": bucket,
                "class": str(klass),
                "class_short": _short_class(str(klass)),
                "support": int(support),
                "recall": correct / support,
            }
        )
    return pd.DataFrame(rows)


def build_class_recall_delta(run_root: Path) -> pd.DataFrame:
    parts = [
        _read_recall_by_class(run_root / rel, bucket)
        for bucket, rel in CONFUSION_PATHS.items()
    ]
    recall = pd.concat(parts, ignore_index=True)
    main = recall.loc[recall["bucket"].eq("main"), ["class", "recall"]].rename(columns={"recall": "main_recall"})
    out = recall.loc[~recall["bucket"].eq("main")].merge(main, on="class", how="left")
    out["recall_delta_vs_main"] = out["recall"] - out["main_recall"]
    out["bucket_label"] = out["bucket"].map(BUCKET_LABELS)
    out["context"] = out["bucket"].map(BUCKET_CONTEXT)
    return out


def build_class_recall_matrix_plot_data(class_delta: pd.DataFrame) -> pd.DataFrame:
    stress_buckets = ["stress_metastatic", "stress_ffpe", "stress_epic"]
    present = class_delta.loc[class_delta["bucket"].isin(stress_buckets)].copy()
    if present.empty:
        raise SystemExit("No stress-bucket class recall rows available.")
    class_summary = (
        present.groupby(["class", "class_short"], as_index=False)
        .agg(
            n_stress_buckets=("bucket", "nunique"),
            total_support=("support", "sum"),
            min_recall_delta=("recall_delta_vs_main", "min"),
        )
        .sort_values(
            ["n_stress_buckets", "min_recall_delta", "total_support", "class_short"],
            ascending=[False, True, False, True],
        )
    )
    rows = []
    for _, class_row in class_summary.iterrows():
        klass = str(class_row["class"])
        for bucket in stress_buckets:
            hit = present.loc[(present["class"].eq(klass)) & (present["bucket"].eq(bucket))]
            if hit.empty:
                rows.append(
                    {
                        "class": klass,
                        "class_short": str(class_row["class_short"]),
                        "bucket": bucket,
                        "bucket_label": BUCKET_LABELS[bucket],
                        "context": BUCKET_CONTEXT[bucket],
                        "present": False,
                        "support": 0,
                        "recall": np.nan,
                        "main_recall": np.nan,
                        "recall_delta_vs_main": np.nan,
                        "n_stress_buckets": int(class_row["n_stress_buckets"]),
                        "total_support": int(class_row["total_support"]),
                    }
                )
            else:
                rec = hit.iloc[0]
                rows.append(
                    {
                        "class": klass,
                        "class_short": str(class_row["class_short"]),
                        "bucket": bucket,
                        "bucket_label": BUCKET_LABELS[bucket],
                        "context": BUCKET_CONTEXT[bucket],
                        "present": True,
                        "support": int(rec["support"]),
                        "recall": float(rec["recall"]),
                        "main_recall": float(rec["main_recall"]),
                        "recall_delta_vs_main": float(rec["recall_delta_vs_main"]),
                        "n_stress_buckets": int(class_row["n_stress_buckets"]),
                        "total_support": int(class_row["total_support"]),
                    }
                )
    return pd.DataFrame(rows)


def build_class_recall_waterfall_plot_data(class_delta: pd.DataFrame) -> pd.DataFrame:
    buckets = ["stress_metastatic", "stress_ffpe", "stress_epic"]
    rows = []
    plot_df = class_delta.loc[class_delta["bucket"].isin(buckets) & class_delta["main_recall"].notna()].copy()
    for bucket in buckets:
        sub = (
            plot_df.loc[plot_df["bucket"].eq(bucket)]
            .sort_values(["recall_delta_vs_main", "support", "class_short"], ascending=[True, False, True])
            .reset_index(drop=True)
        )
        sub["waterfall_rank"] = np.arange(1, len(sub) + 1)
        rows.append(sub)
    if not rows:
        raise SystemExit("No class recall rows available for waterfall plot.")
    return pd.concat(rows, ignore_index=True)


def plot_metric_values(ax: plt.Axes, metrics: pd.DataFrame) -> None:
    y_base = np.arange(len(metrics))[::-1]
    offsets = np.linspace(-0.18, 0.18, len(METRICS))
    for offset, metric in zip(offsets, METRICS, strict=True):
        ax.scatter(
            metrics[metric],
            y_base + offset,
            s=34,
            color=METRIC_COLORS[metric],
            edgecolor=PAPER_BG,
            linewidth=0.7,
            zorder=3,
            label=METRIC_LABELS[metric],
        )
    labels = [
        f"{BUCKET_LABELS[str(row.bucket)]}\nn={row.n_samples}, c={row.n_classes_in_truth}"
        for row in metrics.itertuples(index=False)
    ]
    ax.set_yticks(y_base)
    ax.set_yticklabels(labels, fontsize=6.8)
    ax.set_xlim(0, 1.03)
    ax.set_xlabel("Metric value")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.tick_params(axis="y", length=0)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=4, handletextpad=0.3, columnspacing=0.75)


def plot_metric_deltas(ax: plt.Axes, delta: pd.DataFrame, compact: bool = False) -> None:
    plot_df = delta.loc[~delta["bucket"].eq("main")].copy()
    buckets = ["stress_metastatic", "stress_ffpe", "stress_epic"]
    y_base = np.arange(len(buckets))[::-1]
    offsets = np.linspace(-0.18, 0.18, len(METRICS))
    marker_size = 28 if compact else 34
    for yi, bucket in zip(y_base, buckets, strict=True):
        sub = plot_df.loc[plot_df["bucket"].eq(bucket)].set_index("metric")
        for offset, metric in zip(offsets, METRICS, strict=True):
            value = float(sub.loc[metric, "delta_vs_main"])
            ax.plot([0, value], [yi + offset, yi + offset], color=GRID, linewidth=1.0, zorder=1)
            ax.scatter(
                [value],
                [yi + offset],
                s=marker_size,
                color=METRIC_COLORS[metric],
                edgecolor=PAPER_BG,
                linewidth=0.7,
                zorder=3,
            )
    labels = [f"{BUCKET_LABELS[b]}\n{BUCKET_CONTEXT[b]}" for b in buckets]
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.set_yticks(y_base)
    ax.set_yticklabels(labels, fontsize=6.3 if compact else 6.8)
    ax.set_xlim(-0.55, 0.08)
    ax.set_xlabel("Delta vs primary external")
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.tick_params(axis="y", length=0)
    if compact:
        ax.tick_params(axis="x", labelsize=6.4)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(GRID)


def plot_metric_delta_heatmap(ax: plt.Axes, delta: pd.DataFrame) -> None:
    buckets = ["stress_metastatic", "stress_ffpe", "stress_epic"]
    plot_df = delta.loc[delta["bucket"].isin(buckets)].copy()
    matrix = (
        plot_df.pivot(index="bucket", columns="metric", values="delta_vs_main")
        .reindex(index=buckets, columns=METRICS)
    )
    values = matrix.to_numpy(dtype=float)
    norm = TwoSlopeNorm(vmin=-0.55, vcenter=0.0, vmax=0.20)
    image = ax.imshow(values, cmap="RdBu", norm=norm, aspect="auto")
    ax.set_xticks(np.arange(len(METRICS)))
    ax.set_xticklabels([METRIC_LABELS[m] for m in METRICS], rotation=0)
    ax.set_yticks(np.arange(len(buckets)))
    ax.set_yticklabels([f"{BUCKET_LABELS[b]}\n{BUCKET_CONTEXT[b]}" for b in buckets], fontsize=6.8)
    ax.tick_params(axis="both", length=0)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            text_color = "white" if abs(value) >= 0.32 else INK
            ax.text(j, i, f"{value:+.2f}", ha="center", va="center", fontsize=7, color=text_color)
    ax.set_xticks(np.arange(-0.5, len(METRICS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(buckets), 1), minor=True)
    ax.grid(which="minor", color=PAPER_BG, linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = ax.figure.colorbar(image, ax=ax, fraction=0.050, pad=0.035)
    cbar.set_label("Delta vs primary external", fontsize=7)
    cbar.ax.tick_params(labelsize=6.5, length=2, colors=MUTED)
    cbar.outline.set_edgecolor(GRID)


def plot_class_recall_delta_heatmap(ax: plt.Axes, class_matrix: pd.DataFrame) -> None:
    buckets = ["stress_metastatic", "stress_ffpe", "stress_epic"]
    classes = class_matrix[["class", "class_short"]].drop_duplicates()["class"].tolist()
    short_labels = class_matrix[["class", "class_short"]].drop_duplicates()["class_short"].tolist()
    matrix = (
        class_matrix.pivot(index="class", columns="bucket", values="recall_delta_vs_main")
        .reindex(index=classes, columns=buckets)
    )
    support = (
        class_matrix.pivot(index="class", columns="bucket", values="support")
        .reindex(index=classes, columns=buckets)
    )
    values = matrix.to_numpy(dtype=float)
    masked = np.ma.masked_invalid(values)
    cmap = plt.get_cmap("RdBu").copy()
    cmap.set_bad("#eee9df")
    norm = TwoSlopeNorm(vmin=-1.0, vcenter=0.0, vmax=0.35)
    image = ax.imshow(masked, cmap=cmap, norm=norm, aspect="auto")
    bucket_labels = []
    for bucket in buckets:
        n_classes = int(class_matrix.loc[class_matrix["bucket"].eq(bucket) & class_matrix["present"], "class"].nunique())
        bucket_labels.append(f"{BUCKET_LABELS[bucket]}\n{n_classes} classes")
    ax.set_xticks(np.arange(len(buckets)))
    ax.set_xticklabels(bucket_labels)
    ax.set_yticks(np.arange(len(classes)))
    ax.set_yticklabels(short_labels, fontsize=6.4)
    ax.set_xlabel("Domain-shifted cohort (class n)")
    ax.tick_params(axis="both", length=0)
    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            if np.isnan(values[i, j]):
                continue
            n = int(support.iat[i, j])
            text_color = "white" if abs(values[i, j]) >= 0.48 else INK
            ax.text(j, i, f"{n}", ha="center", va="center", fontsize=5.9, color=text_color)
    ax.set_xticks(np.arange(-0.5, len(buckets), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(classes), 1), minor=True)
    ax.grid(which="minor", color=PAPER_BG, linewidth=0.9)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    cbar = ax.figure.colorbar(image, ax=ax, fraction=0.030, pad=0.020)
    cbar.set_label("Recall delta vs primary external", fontsize=7)
    cbar.ax.tick_params(labelsize=6.5, length=2, colors=MUTED)
    cbar.outline.set_edgecolor(GRID)


def plot_methydeep_delta_heatmap(ax: plt.Axes, plot_data: pd.DataFrame, compact: bool = False) -> None:
    buckets = ["stress_metastatic", "stress_ffpe", "stress_epic"]
    matrix = (
        plot_data.pivot(index="bucket", columns="metric", values="delta_ours_minus_methydeep")
        .reindex(index=buckets, columns=METRICS)
    )
    values = matrix.to_numpy(dtype=float)
    max_abs = max(0.10, float(np.nanmax(np.abs(values))) * 1.05)
    norm = TwoSlopeNorm(vmin=-max_abs, vcenter=0.0, vmax=max_abs)
    cmap = LinearSegmentedColormap.from_list(
        "methydeep_to_minicpg",
        ["#5f4778", "#f1ece6", "#2f6f58"],
    )
    image = ax.imshow(values, cmap=cmap, norm=norm, aspect="auto")

    y_labels = []
    winner_labels = []
    for bucket in buckets:
        row = plot_data.loc[plot_data["bucket"].eq(bucket)].iloc[0]
        n_wins = int((plot_data.loc[plot_data["bucket"].eq(bucket), "winner"] == "MiniCpG").sum())
        y_labels.append(
            f"{BUCKET_LABELS[bucket]}\n{BUCKET_CONTEXT[bucket]}\nn={int(row['n_samples'])}, c={int(row['n_classes_in_truth'])}"
        )
        winner_labels.append(f"{n_wins}/4")

    ax.set_xticks(np.arange(len(METRICS)))
    if compact:
        ax.set_xticklabels(["Macro-F1\n(present)", "Weighted-F1", "Accuracy", "AUROC"])
    else:
        ax.set_xticklabels([METRIC_LABELS[m] for m in METRICS])
    ax.set_yticks(np.arange(len(buckets)))
    ax.set_yticklabels(y_labels, fontsize=5.9 if compact else 6.5)
    ax.tick_params(axis="both", length=0)
    if compact:
        ax.tick_params(axis="x", labelsize=5.9)

    for i in range(values.shape[0]):
        for j in range(values.shape[1]):
            value = values[i, j]
            text_color = "white" if abs(value) >= max_abs * 0.55 else INK
            ax.text(j, i, f"{value:+.3f}", ha="center", va="center", fontsize=6.1 if compact else 6.8, color=text_color)
        ax.text(
            len(METRICS) + 0.18,
            i,
            winner_labels[i],
            ha="center",
            va="center",
            fontsize=6.6 if compact else 7.2,
            color=INK,
        )

    ax.text(
        len(METRICS) + 0.18,
        -0.64,
        "MiniCpG\nhigher",
        ha="center",
        va="bottom",
        fontsize=5.9 if compact else 6.4,
        color=MUTED,
    )
    ax.set_xticks(np.arange(-0.5, len(METRICS), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(buckets), 1), minor=True)
    ax.grid(which="minor", color=PAPER_BG, linewidth=1.2)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.set_xlim(-0.5, len(METRICS) + 0.55)
    for spine in ax.spines.values():
        spine.set_visible(False)

    cbar = ax.figure.colorbar(image, ax=ax, fraction=0.035, pad=0.035)
    cbar.set_label("Metric difference\n(MiniCpG - MethyDeep)", fontsize=7)
    cbar.set_ticks([-max_abs, 0.0, max_abs])
    cbar.set_ticklabels([f"{-max_abs:+.3f}", "0", f"{max_abs:+.3f}"])
    cbar.ax.tick_params(labelsize=6.1 if compact else 6.5, length=2, colors=MUTED)
    cbar.outline.set_edgecolor(GRID)


def plot_class_recall_delta_waterfalls(axes: list[plt.Axes], class_waterfall: pd.DataFrame) -> None:
    buckets = ["stress_metastatic", "stress_ffpe", "stress_epic"]
    xlim = (-1.0, 0.42)
    xticks = [-1.0, -0.5, 0.0]
    for ax, bucket in zip(axes, buckets, strict=True):
        sub = class_waterfall.loc[class_waterfall["bucket"].eq(bucket)].copy()
        y = np.arange(len(sub))
        values = sub["recall_delta_vs_main"].to_numpy(dtype=float)
        colors = np.where(values < -0.05, ORANGE, np.where(values > 0.05, BLUE, GREY))
        ax.barh(y, values, color=colors, alpha=0.88, height=0.66, edgecolor=PAPER_BG, linewidth=0.35)
        ax.axvline(0, color=MUTED, linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(sub["class_short"].tolist(), fontsize=6.0)
        ax.invert_yaxis()
        ax.set_xlim(*xlim)
        ax.set_xticks(xticks)
        ax.tick_params(axis="x", labelsize=6.2)
        ax.tick_params(axis="y", length=0, pad=1.5)
        ax.grid(axis="x", color=GRID, linewidth=0.7)
        n_classes = int(sub["class"].nunique())
        ax.set_title(f"{BUCKET_LABELS[bucket]}\n{n_classes} classes", fontsize=8, pad=5)
        for yi, support in zip(y, sub["support"], strict=True):
            ax.text(
                xlim[1] - 0.02,
                yi,
                f"{int(support)}",
                ha="right",
                va="center",
                fontsize=5.7,
                color=MUTED,
            )
        for spine in ["top", "right", "left"]:
            ax.spines[spine].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
    axes[0].set_ylabel("Class", labelpad=2)
    axes[1].set_xlabel("Recall delta vs primary external")
    axes[-1].text(
        xlim[1] - 0.02,
        -0.93,
        "n",
        ha="right",
        va="bottom",
        fontsize=5.8,
        color=MUTED,
        transform=axes[-1].transData,
    )


def plot_figure(
    delta: pd.DataFrame,
    class_matrix: pd.DataFrame,
    class_waterfall: pd.DataFrame,
    methydeep_delta: pd.DataFrame,
    metric_panel: str,
    class_panel: str,
    out_paths: list[Path],
) -> None:
    configure_matplotlib()
    fig = plt.figure(figsize=(7.2, 6.85))
    gs = fig.add_gridspec(2, 1, height_ratios=[1.05, 2.35], hspace=0.42)
    top_gs = gs[0, 0].subgridspec(1, 2, width_ratios=[0.95, 1.25], wspace=0.50)
    ax_a = fig.add_subplot(top_gs[0, 0])
    ax_b = fig.add_subplot(top_gs[0, 1])

    if metric_panel == "lollipop":
        plot_metric_deltas(ax_a, delta, compact=True)
        add_panel_letter(ax_a, "A", x=-0.20, y=1.16)
        handles = [
            Line2D([0], [0], marker="o", color="none", markerfacecolor=METRIC_COLORS[m], markeredgecolor=PAPER_BG, markersize=5.2)
            for m in METRICS
        ]
        ax_a.legend(
            handles,
            [METRIC_LABELS[m] for m in METRICS],
            frameon=False,
            loc="lower left",
            bbox_to_anchor=(-0.02, 1.00),
            ncol=2,
            handletextpad=0.25,
            columnspacing=0.75,
            fontsize=6.4,
        )
    elif metric_panel == "heatmap":
        plot_metric_delta_heatmap(ax_a, delta)
        add_panel_letter(ax_a, "A", x=-0.20, y=1.14)
    else:
        raise ValueError(metric_panel)

    plot_methydeep_delta_heatmap(ax_b, methydeep_delta, compact=True)
    add_panel_letter(ax_b, "B", x=-0.18, y=1.14)

    if class_panel == "heatmap":
        ax_c = fig.add_subplot(gs[1, 0])
        plot_class_recall_delta_heatmap(ax_c, class_matrix)
        add_panel_letter(ax_c, "C", x=-0.08, y=1.04)
    elif class_panel == "waterfall":
        sub_gs = gs[1, 0].subgridspec(1, 3, wspace=0.36)
        axes_b = [fig.add_subplot(sub_gs[0, idx]) for idx in range(3)]
        plot_class_recall_delta_waterfalls(axes_b, class_waterfall)
        add_panel_letter(axes_b[0], "C", x=-0.38, y=1.05)
    else:
        raise ValueError(class_panel)

    for path in out_paths:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix.lower() == ".png":
            fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.03, facecolor=fig.get_facecolor())
        else:
            fig.savefig(path, bbox_inches="tight", pad_inches=0.03, facecolor=fig.get_facecolor())
    plt.close(fig)


def main() -> int:
    args = parse_args()
    active = validate_active_run(args.active_manifest)
    metrics_path = Path(args.metrics_tsv)
    run_root = Path(args.run_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = load_full_metrics(metrics_path, run_root)
    delta = build_delta_metrics(metrics)
    methydeep_delta = build_methydeep_delta_metrics(metrics_path)
    class_delta = build_class_recall_delta(run_root)
    class_matrix = build_class_recall_matrix_plot_data(class_delta)
    class_waterfall = build_class_recall_waterfall_plot_data(class_delta)

    variants = ["lollipop", "heatmap"] if args.variant == "all" else [args.variant]
    figure_outputs = []
    for variant in variants:
        class_suffix = "" if args.class_panel == "heatmap" else f"_class_{args.class_panel}"
        out_paths = [out_dir / f"{args.out_stem}_metric_delta_{variant}{class_suffix}.{fmt}" for fmt in args.formats]
        plot_figure(delta, class_matrix, class_waterfall, methydeep_delta, variant, args.class_panel, out_paths)
        figure_outputs.extend(out_paths)

    metrics_out = out_dir / f"{args.out_stem}_full_bucket_metric_plot_data.tsv"
    delta_out = out_dir / f"{args.out_stem}_delta_metric_plot_data.tsv"
    class_out = out_dir / f"{args.out_stem}_class_recall_delta_plot_data.tsv"
    class_matrix_out = out_dir / f"{args.out_stem}_class_recall_delta_matrix_plot_data.tsv"
    class_waterfall_out = out_dir / f"{args.out_stem}_class_recall_delta_waterfall_plot_data.tsv"
    methydeep_out = out_dir / f"{args.out_stem}_methydeep_delta_plot_data.tsv"
    summary_out = out_dir / f"{args.out_stem}_summary.json"
    metrics.to_csv(metrics_out, sep="\t", index=False)
    delta.to_csv(delta_out, sep="\t", index=False)
    class_delta.to_csv(class_out, sep="\t", index=False)
    class_matrix.to_csv(class_matrix_out, sep="\t", index=False)
    class_waterfall.to_csv(class_waterfall_out, sep="\t", index=False)
    methydeep_delta.to_csv(methydeep_out, sep="\t", index=False)
    n_minicpg_higher = int(methydeep_delta["winner"].eq("MiniCpG").sum())
    n_methydeep_higher = int(methydeep_delta["winner"].eq("MethyDeep").sum())
    summary = {
        "active_locked_run_status": active.get("status"),
        "locked_run_name": active.get("locked_run_name"),
        "metrics_source": str(metrics_path),
        "macro_f1_definition": "macro-F1 over truth classes present in each evaluated cohort; fixed-label-axis macro-F1 retained in metric sidecar as macro_f1_fixed_label_axis",
        "n_buckets": int(metrics["bucket"].nunique()),
        "buckets": [str(x) for x in metrics["bucket"].tolist()],
        "metric_panel_variants": variants,
        "class_panel": args.class_panel,
        "stress_class_counts": {
            BUCKET_LABELS[bucket]: int(
                class_matrix.loc[class_matrix["bucket"].eq(bucket) & class_matrix["present"], "class"].nunique()
            )
            for bucket in ["stress_metastatic", "stress_ffpe", "stress_epic"]
        },
        "class_heatmap_rows": int(class_matrix["class"].nunique()),
        "class_heatmap_cell_text": "domain-shifted cohort class support",
        "class_waterfall_cell_text": "right-side row label is domain-shifted cohort class support",
        "panel_order": [
            "A: aggregate metric deltas vs primary external",
            "B: MiniCpG minus MethyDeep metric deltas on shifted matched subsets",
            "C: class-level recall deltas vs primary external",
        ],
        "methydeep_comparison_cell_value": "MiniCpG metric minus MethyDeep metric on matched head-to-head subset",
        "methydeep_comparison_minicpg_higher_cells": n_minicpg_higher,
        "methydeep_comparison_methydeep_higher_cells": n_methydeep_higher,
        "methydeep_comparison_max_positive_delta": float(methydeep_delta["delta_ours_minus_methydeep"].max()),
        "methydeep_comparison_max_negative_delta": float(methydeep_delta["delta_ours_minus_methydeep"].min()),
        "outputs": [str(p) for p in figure_outputs]
        + [
            str(metrics_out),
            str(delta_out),
            str(class_out),
            str(class_matrix_out),
            str(class_waterfall_out),
            str(methydeep_out),
            str(summary_out),
        ],
    }
    summary_out.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
