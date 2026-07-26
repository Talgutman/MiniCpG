#!/usr/bin/env python3
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
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, Normalize


PAPER_BG = "#fbfaf7"
PANEL_BG = "#fbfaf7"
INK = "#24313f"
MUTED = "#6b7680"
GRID = "#e6ded1"
BLUE = "#315f8c"
LIGHT_BLUE = "#8aa8bf"
ORANGE = "#c76548"
GOLD = "#d39a3f"
GREEN = "#6f7f75"
PALE = "#efe8dc"
TGCT_BLUE = "#0072b2"
ESCA_ORANGE = "#d55e00"


FLAG_ORDER = [
    "external_drop",
    "low_recall_internal",
    "specific_confusion",
    "diffuse_confusion",
    "low_confidence",
    "overconfident_wrong",
    "missingness_risk",
    "low_sample_support",
]

FLAG_LABELS = {
    "external_drop": "External drop",
    "low_recall_internal": "Weak in lockbox",
    "specific_confusion": "Specific\nconfusion",
    "diffuse_confusion": "Diffuse\nconfusion",
    "low_confidence": "Low\nconfidence",
    "overconfident_wrong": "High-conf.\nwrong",
    "missingness_risk": "Missingness\nrisk",
    "low_sample_support": "Low sample\nsupport",
}

FLAG_COLORS = {
    "external_drop": "#7d8790",
    "low_recall_internal": "#8f4d3f",
    "specific_confusion": GOLD,
    "diffuse_confusion": "#b9915b",
    "low_confidence": "#6f7f75",
    "overconfident_wrong": "#9b5d63",
    "missingness_risk": "#738aa6",
    "low_sample_support": "#8e8a7a",
}

EDGE_BIOLOGY_LABELS = {
    "shared_lineage_site": "Shared lineage/site",
    "epithelial_regional": "Epithelial/regional",
    "no_simple_match": "No simple match",
    "low_support_uncertain": "Limited target support",
}

DASHED = (3.0, 2.0)
DASH_DOT = (4.0, 2.0, 1.2, 2.0)
DOTTED = (1.0, 2.0)

EDGE_BIOLOGY_STYLES = {
    "shared_lineage_site": {"color": GREEN, "dashes": None, "alpha": 0.66},
    "epithelial_regional": {"color": GOLD, "dashes": DASHED, "alpha": 0.62},
    "no_simple_match": {"color": ORANGE, "dashes": None, "alpha": 0.52},
    "low_support_uncertain": {"color": MUTED, "dashes": DOTTED, "alpha": 0.35},
}

EDGE_SUPPORT_FLOOR = 30

GENERIC_LINEAGE_TAGS = {"carcinoma", "epithelial"}

CLASS_CONTEXT = {
    "TCGA-ACC": {"site_group": "adrenal", "lineage_tags": "endocrine", "is_epithelial": True},
    "TCGA-BLCA": {"site_group": "urothelial", "lineage_tags": "urothelial;carcinoma", "is_epithelial": True},
    "TCGA-BRCA": {"site_group": "breast", "lineage_tags": "carcinoma", "is_epithelial": True},
    "TCGA-CESC": {"site_group": "gynecologic_lower", "lineage_tags": "squamous;carcinoma", "is_epithelial": True},
    "TCGA-COAD": {"site_group": "lower_gastrointestinal", "lineage_tags": "colorectal;carcinoma", "is_epithelial": True},
    "TCGA-ESCA": {"site_group": "upper_gastrointestinal", "lineage_tags": "aerodigestive;carcinoma", "is_epithelial": True},
    "TCGA-GBM": {"site_group": "central_nervous_system", "lineage_tags": "glial", "is_epithelial": False},
    "TCGA-HNSC": {"site_group": "head_neck", "lineage_tags": "squamous;carcinoma", "is_epithelial": True},
    "TCGA-KICH": {"site_group": "kidney", "lineage_tags": "renal;carcinoma", "is_epithelial": True},
    "TCGA-KIRC": {"site_group": "kidney", "lineage_tags": "renal;carcinoma", "is_epithelial": True},
    "TCGA-KIRP": {"site_group": "kidney", "lineage_tags": "renal;carcinoma", "is_epithelial": True},
    "TCGA-LAML": {"site_group": "blood_marrow", "lineage_tags": "myeloid;hematopoietic", "is_epithelial": False},
    "TCGA-LIHC": {"site_group": "hepatobiliary", "lineage_tags": "hepatocellular;carcinoma", "is_epithelial": True},
    "TCGA-LUAD": {"site_group": "lung", "lineage_tags": "lung;adenocarcinoma;carcinoma", "is_epithelial": True},
    "TCGA-LUSC": {"site_group": "lung", "lineage_tags": "squamous;lung;carcinoma", "is_epithelial": True},
    "TCGA-MESO": {"site_group": "serosal_thoracic", "lineage_tags": "mesothelial;serosal", "is_epithelial": False},
    "TCGA-PAAD": {"site_group": "pancreatobiliary", "lineage_tags": "pancreatobiliary;carcinoma", "is_epithelial": True},
    "TCGA-PCPG": {"site_group": "paraganglia_adrenal", "lineage_tags": "neuroendocrine", "is_epithelial": False},
    "TCGA-PRAD": {"site_group": "prostate", "lineage_tags": "prostate;adenocarcinoma;carcinoma", "is_epithelial": True},
    "TCGA-SARC": {"site_group": "soft_tissue", "lineage_tags": "mesenchymal", "is_epithelial": False},
    "TCGA-SKCM": {"site_group": "skin", "lineage_tags": "melanocytic", "is_epithelial": False},
    "TCGA-TGCT": {"site_group": "testis", "lineage_tags": "germ_cell", "is_epithelial": False},
    "TCGA-THCA": {"site_group": "thyroid", "lineage_tags": "thyroid;endocrine;carcinoma", "is_epithelial": True},
    "TCGA-THYM": {"site_group": "thymus", "lineage_tags": "thymic;immune_adjacent", "is_epithelial": True},
    "TCGA-UCEC": {"site_group": "uterus", "lineage_tags": "uterine;endometrial;carcinoma", "is_epithelial": True},
    "TCGA-UCS": {"site_group": "uterus", "lineage_tags": "uterine;mixed_epithelial_mesenchymal", "is_epithelial": True},
    "TCGA-UVM": {"site_group": "uvea", "lineage_tags": "melanocytic", "is_epithelial": False},
}

EDGE_CONTEXT_OVERRIDES = {
    ("TCGA-MESO", "TCGA-SARC"): (
        "shared_lineage_site",
        "Serosal/mesenchymal-adjacent context; keep as a broad lineage/site note, not an exact histology claim.",
    ),
    ("TCGA-COAD", "TCGA-ESCA"): ("epithelial_regional", "Gastrointestinal epithelial axis."),
    ("TCGA-PAAD", "TCGA-ESCA"): ("epithelial_regional", "Upper-GI/pancreatobiliary epithelial axis."),
    ("TCGA-LUAD", "TCGA-ESCA"): ("no_simple_match", "Broad epithelial context only; no simple shared tissue or lineage match."),
    ("TCGA-BRCA", "TCGA-ESCA"): ("no_simple_match", "Broad epithelial context only; no simple shared tissue or lineage match."),
    ("TCGA-CESC", "TCGA-BLCA"): (
        "epithelial_regional",
        "Pelvic/urogenital epithelial context, with target-class support caveat captured separately.",
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Main manuscript-style failure-mode figures for active K=30.")
    parser.add_argument("--active-manifest", default="config/active_locked_run.json")
    parser.add_argument(
        "--failure-dir",
        default="outputs/paper_protocolb_v1/k_030/failure_mode_analysis_current",
    )
    parser.add_argument(
        "--hub-audit-dir",
        default="outputs/paper_protocolb_v1/manuscript_reproduction",
    )
    parser.add_argument(
        "--tgct-perturbation-dir",
        default="outputs/paper_protocolb_v1/manuscript_reproduction",
    )
    parser.add_argument(
        "--tcga-dev-matrix",
        default=(
            "outputs/paper_protocolb_v1/k_030/biological_interpretation_current/"
            "tcga_tumor_panel_matrices_current/tcga_dev_panel_matrix.tsv"
        ),
    )
    parser.add_argument("--external-drop-threshold", type=float, default=-0.15)
    parser.add_argument("--low-support-threshold", type=int, default=EDGE_SUPPORT_FLOOR)
    parser.add_argument("--formats", nargs="+", default=["png"], choices=["png", "pdf"])
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


def set_theme() -> None:
    sns.set_theme(
        context="paper",
        style="white",
        rc={
            "font.family": "DejaVu Sans",
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "figure.facecolor": PAPER_BG,
            "axes.facecolor": PANEL_BG,
            "savefig.facecolor": PAPER_BG,
        },
    )


def strip_tcga(label: str) -> str:
    return str(label).replace("TCGA-", "")


def split_flags(value: object) -> set[str]:
    if pd.isna(value) or not str(value).strip():
        return set()
    return {flag for flag in str(value).split(";") if flag}


def split_tags(value: object) -> set[str]:
    if pd.isna(value) or not str(value).strip():
        return set()
    return {tag.strip() for tag in str(value).split(";") if tag.strip()}


def specific_lineage_tags(context: dict[str, object]) -> set[str]:
    return split_tags(context.get("lineage_tags", "")) - GENERIC_LINEAGE_TAGS


def class_context(label: str) -> dict[str, object]:
    context = CLASS_CONTEXT.get(label, {})
    return {
        "site_group": str(context.get("site_group", "unmapped")),
        "lineage_tags": str(context.get("lineage_tags", "unmapped")),
        "is_epithelial": bool(context.get("is_epithelial", False)),
    }


def load_inputs(failure_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    class_summary = pd.read_csv(failure_dir / "failure_mode_class_summary.tsv", sep="\t")
    conf = pd.read_csv(failure_dir / "failure_mode_top_confusions.tsv", sep="\t")
    network_nodes = pd.read_csv(failure_dir / "failure_mode_confusion_network_nodes.tsv", sep="\t")
    network_edges = pd.read_csv(failure_dir / "failure_mode_confusion_network_edges.tsv", sep="\t")
    return class_summary, conf, network_nodes, network_edges


def build_class_plot_table(class_summary: pd.DataFrame) -> pd.DataFrame:
    ext = class_summary.loc[class_summary["dataset"].eq("external_main")].copy()
    lock = class_summary.loc[class_summary["dataset"].eq("lockbox")].copy()
    keep_lock = ["class", "recall", "failure_flags"]
    table = ext.merge(
        lock[keep_lock].rename(columns={"recall": "lockbox_recall", "failure_flags": "lockbox_failure_flags"}),
        on="class",
        how="left",
    )
    table = table.rename(columns={"recall": "external_recall"})
    table["external_recall_delta"] = table["external_recall"] - table["lockbox_recall"]
    table["class_short"] = table["class"].map(strip_tcga)
    table["sort_key"] = table["external_recall_delta"].fillna(0)
    table = table.sort_values(["sort_key", "external_recall"], ascending=[True, True]).reset_index(drop=True)

    flag_rows = []
    for row in table.to_dict(orient="records"):
        flags = split_flags(row.get("failure_flags"))
        flags |= split_flags(row.get("lockbox_failure_flags"))
        for flag in FLAG_ORDER:
            flag_rows.append(
                {
                    "class": row["class"],
                    "class_short": row["class_short"],
                    "flag": flag,
                    "flag_label": FLAG_LABELS[flag],
                    "flag_present": flag in flags,
                    "external_recall": row["external_recall"],
                    "lockbox_recall": row["lockbox_recall"],
                    "external_recall_delta": row["external_recall_delta"],
                    "n_samples": row["n_samples"],
                    "failure_flags": row.get("failure_flags", ""),
                    "lockbox_failure_flags": row.get("lockbox_failure_flags", ""),
                }
            )
    return table, pd.DataFrame(flag_rows)


def annotate_network_edges(edges: pd.DataFrame, nodes: pd.DataFrame, low_support_threshold: int) -> pd.DataFrame:
    edges = edges.copy()
    target_support = nodes.set_index("class")["n_samples"].to_dict()
    incoming_source_counts = edges.groupby("y_pred")["true_class"].nunique().to_dict()
    categories = []
    labels = []
    notes = []
    rules = []
    true_site_groups = []
    pred_site_groups = []
    true_lineage_tags = []
    pred_lineage_tags = []
    true_is_epithelial = []
    pred_is_epithelial = []
    pred_true_class_support = []
    target_incoming_source_counts = []
    for row in edges.to_dict(orient="records"):
        key = (row["true_class"], row["y_pred"])
        true_context = class_context(str(row["true_class"]))
        pred_context = class_context(str(row["y_pred"]))
        shared_lineage = sorted(specific_lineage_tags(true_context) & specific_lineage_tags(pred_context))
        true_support = int(row.get("true_class_n", 0) or 0)
        pred_support = int(target_support.get(row["y_pred"], 0) or 0)
        incoming_sources = int(incoming_source_counts.get(row["y_pred"], 0) or 0)

        if true_support < low_support_threshold or pred_support < low_support_threshold:
            category = "low_support_uncertain"
            rule = "low_support"
            if true_support < low_support_threshold and pred_support < low_support_threshold:
                note = f"True and predicted classes have external support below {low_support_threshold}."
            elif true_support < low_support_threshold:
                note = f"True class external support is below {low_support_threshold}."
            else:
                note = f"Predicted target class external support is below {low_support_threshold}."
        elif key in EDGE_CONTEXT_OVERRIDES:
            category, note = EDGE_CONTEXT_OVERRIDES[key]
            rule = "pre_specified_pair_context"
        elif shared_lineage:
            category = "shared_lineage_site"
            rule = "shared_specific_lineage_tag"
            note = "Shared specific lineage tag(s): " + ", ".join(shared_lineage) + "."
        elif true_context["site_group"] == pred_context["site_group"] and true_context["site_group"] != "unmapped":
            category = "shared_lineage_site"
            rule = "shared_site_group"
            note = f"Shared class-map site group: {true_context['site_group']}."
        elif true_context["is_epithelial"] and pred_context["is_epithelial"]:
            category = "epithelial_regional"
            rule = "both_epithelial_broad"
            note = "Both classes are epithelial, but no more specific pre-specified site/lineage match was assigned."
        elif incoming_sources >= 3:
            category = "no_simple_match"
            rule = "recurrent_predicted_target"
            note = (
                f"Predicted target has incoming errors from {incoming_sources} source classes in this graph, "
                "without a simple class-map site/lineage match."
            )
        else:
            category = "no_simple_match"
            rule = "no_simple_pair_context"
            note = "No simple class-map site/lineage explanation was assigned."
        categories.append(category)
        labels.append(EDGE_BIOLOGY_LABELS[category])
        notes.append(note)
        rules.append(rule)
        true_site_groups.append(true_context["site_group"])
        pred_site_groups.append(pred_context["site_group"])
        true_lineage_tags.append(true_context["lineage_tags"])
        pred_lineage_tags.append(pred_context["lineage_tags"])
        true_is_epithelial.append(true_context["is_epithelial"])
        pred_is_epithelial.append(pred_context["is_epithelial"])
        pred_true_class_support.append(pred_support)
        target_incoming_source_counts.append(incoming_sources)
    edges["biology_edge_category"] = categories
    edges["biology_edge_label"] = labels
    edges["biology_edge_note"] = notes
    edges["biology_edge_rule"] = rules
    edges["true_site_group"] = true_site_groups
    edges["pred_site_group"] = pred_site_groups
    edges["true_lineage_tags"] = true_lineage_tags
    edges["pred_lineage_tags"] = pred_lineage_tags
    edges["true_is_epithelial"] = true_is_epithelial
    edges["pred_is_epithelial"] = pred_is_epithelial
    edges["pred_true_class_support"] = pred_true_class_support
    edges["target_incoming_source_classes"] = target_incoming_source_counts
    return edges


def build_class_context_table(nodes: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for label in sorted(nodes["class"].astype(str).unique()):
        context = class_context(label)
        rows.append(
            {
                "class": label,
                "site_group": context["site_group"],
                "lineage_tags": context["lineage_tags"],
                "is_epithelial": context["is_epithelial"],
            }
        )
    return pd.DataFrame(rows)


def write_network_annotation_rules(out_path: Path, low_support_threshold: int) -> None:
    lines = [
        "# Failure-Mode Confusion Network Edge Annotation Rules",
        "",
        "This sidecar documents the categorical edge labels used in the failure-mode confusion network.",
        "The labels are descriptive context for directed external-main errors, not causal explanations.",
        "",
        "## Rules",
        "",
        f"- `Limited target support`: true or predicted target class has external support below `{low_support_threshold}`.",
        "- `Shared lineage/site`: classes share a non-generic lineage tag or a site group in the class context map.",
        "- `Epithelial/regional`: classes are epithelial and either have a pre-specified regional/broad epithelial context or no more specific shared lineage/site match.",
        "- `No simple match`: no simple site, lineage, epithelial/regional, or low-support rule applies. Some no-simple-match targets are recurrent hubs, but recurrence is recorded in the audit table rather than used as a figure label.",
        "",
        "Line styles are defined once in the plotting script and reused for both graph arrows and legend handles.",
        "",
        "The full per-edge audit is written to `failure_mode_confusion_network_edge_annotation_audit.tsv`.",
    ]
    out_path.write_text("\n".join(lines) + "\n")


def plot_recall_slope(table: pd.DataFrame, ax: plt.Axes, drop_threshold: float, title: str) -> None:
    plot_df = table.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(plot_df))
    for i, row in plot_df.iterrows():
        ax.plot([row["lockbox_recall"], row["external_recall"]], [i, i], color=GRID, linewidth=1.2, alpha=0.95)
    ax.scatter(plot_df["lockbox_recall"], y, s=32, color=BLUE, edgecolor=PAPER_BG, linewidth=0.7, label="Held-out TCGA test", zorder=3)
    ax.scatter(plot_df["external_recall"], y, s=34, color=ORANGE, edgecolor=PAPER_BG, linewidth=0.7, label="Primary external", zorder=3)
    for i, row in plot_df.iterrows():
        if pd.notna(row["external_recall_delta"]) and row["external_recall_delta"] <= drop_threshold:
            ax.text(
                min(row["lockbox_recall"], row["external_recall"]) - 0.025,
                i,
                f"{row['external_recall_delta']:.2f}",
                ha="right",
                va="center",
                fontsize=6.5,
                color=MUTED,
            )
    ax.set_yticks(y)
    ax.set_yticklabels(plot_df["class_short"], fontsize=7)
    ax.set_xlim(-0.08, 1.04)
    ax.set_xlabel("Recall")
    if title:
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=7)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.legend(frameon=False, loc="lower left", fontsize=7)
    sns.despine(ax=ax, left=True, bottom=True)


def plot_flag_matrix(flag_df: pd.DataFrame, class_table: pd.DataFrame, ax: plt.Axes, title: str) -> None:
    row_order = list(class_table["class"].iloc[::-1])
    row_pos = {klass: idx for idx, klass in enumerate(row_order)}
    col_pos = {flag: idx for idx, flag in enumerate(FLAG_ORDER)}
    present = flag_df.loc[flag_df["flag_present"]].copy()
    absent = flag_df.loc[~flag_df["flag_present"]].copy()
    ax.scatter(
        absent["flag"].map(col_pos),
        absent["class"].map(row_pos),
        s=35,
        marker="s",
        color=PALE,
        edgecolor=PAPER_BG,
        linewidth=0.5,
    )
    for flag in FLAG_ORDER:
        sub = present.loc[present["flag"].eq(flag)]
        ax.scatter(
            sub["flag"].map(col_pos),
            sub["class"].map(row_pos),
            s=46,
            marker="s",
            color=FLAG_COLORS[flag],
            edgecolor=PAPER_BG,
            linewidth=0.5,
        )
    ax.set_xticks(range(len(FLAG_ORDER)))
    ax.set_xticklabels([FLAG_LABELS[f] for f in FLAG_ORDER], rotation=35, ha="right", fontsize=7)
    ax.set_yticks(range(len(row_order)))
    ax.set_yticklabels([strip_tcga(k) for k in row_order], fontsize=7)
    ax.set_xlim(-0.6, len(FLAG_ORDER) + 2.9)
    ax.set_ylim(-0.8, len(row_order) - 0.2)
    if title:
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=7)
    ax.grid(False)
    ax.tick_params(axis="both", length=0)
    x0 = len(FLAG_ORDER) - 0.1
    ax.text(x0 + 0.5, len(row_order) - 0.1, "Ext.\nrecall", ha="center", va="bottom", fontsize=7, color=MUTED)
    ax.text(x0 + 1.35, len(row_order) - 0.1, "Delta", ha="center", va="bottom", fontsize=7, color=MUTED)
    ax.text(x0 + 2.1, len(row_order) - 0.1, "n", ha="center", va="bottom", fontsize=7, color=MUTED)
    table_lookup = class_table.set_index("class")
    for klass, y in row_pos.items():
        row = table_lookup.loc[klass]
        delta = row["external_recall_delta"]
        ax.text(x0 + 0.5, y, f"{row['external_recall']:.2f}", ha="center", va="center", fontsize=7, color=INK)
        ax.text(x0 + 1.35, y, "" if pd.isna(delta) else f"{delta:.2f}", ha="center", va="center", fontsize=7, color=MUTED)
        ax.text(x0 + 2.1, y, f"{int(row['n_samples'])}", ha="center", va="center", fontsize=7, color=MUTED)
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_beta_shift(class_summary: pd.DataFrame, ax: plt.Axes, title: str, size_scale: float = 1.0) -> None:
    ext = class_summary.loc[class_summary["dataset"].eq("external_main")].copy()
    ext = ext.loc[ext["median_abs_beta_shift"].notna()].copy()
    sizes = size_scale * (32 + 180 * (ext["n_samples"] / ext["n_samples"].max()))
    ax.scatter(
        ext["median_abs_beta_shift"],
        ext["recall_delta_external_minus_lockbox"],
        s=sizes,
        color=BLUE,
        alpha=0.72,
        edgecolor=PAPER_BG,
        linewidth=0.8,
    )
    ax.axhline(0, color=GRID, linewidth=1.0)
    ax.axhline(-0.15, color=ORANGE, linewidth=1.0, linestyle=(0, DASHED))
    label_rows = ext.sort_values("recall_delta_external_minus_lockbox").head(8).copy()
    label_specs: list[dict[str, float | str]] = []
    previous_y: float | None = None
    previous_x: float | None = None
    for _, row in label_rows.sort_values("recall_delta_external_minus_lockbox").iterrows():
        point_x = float(row["median_abs_beta_shift"])
        point_y = float(row["recall_delta_external_minus_lockbox"])
        label_y = point_y
        if previous_y is not None and previous_x is not None and abs(point_x - previous_x) < 0.06:
            label_y = max(label_y, previous_y + 0.045)
        label_specs.append(
            {
                "class": str(row["class"]),
                "point_x": point_x,
                "point_y": point_y,
                "label_y": label_y,
            }
        )
        previous_y = label_y
        previous_x = point_x
    for label in label_specs:
        ax.text(
            float(label["point_x"]) + 0.006,
            float(label["label_y"]),
            strip_tcga(str(label["class"])),
            fontsize=7,
            color=INK,
            va="center",
        )
        if abs(float(label["label_y"]) - float(label["point_y"])) > 0.012:
            ax.plot(
                [float(label["point_x"]) + 0.003, float(label["point_x"]) + 0.006],
                [float(label["point_y"]), float(label["label_y"])],
                color=MUTED,
                linewidth=0.45,
                alpha=0.55,
            )
    ax.set_xlabel("Median absolute beta shift")
    ax.set_ylabel("External recall - TCGA test recall")
    if title:
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=7)
    ax.grid(color=GRID, linewidth=0.8)
    sns.despine(ax=ax)


def centroid_distance_long(centrality: pd.DataFrame, dev_matrix_path: Path) -> pd.DataFrame:
    if not dev_matrix_path.exists():
        raise SystemExit(f"Missing TCGA development matrix for centroid panel: {dev_matrix_path}")
    dev = pd.read_csv(dev_matrix_path, sep="\t")
    probes = [c for c in dev.columns if str(c).startswith("cg")]
    if "project_id" not in dev.columns or not probes:
        raise SystemExit(f"TCGA development matrix missing project_id or probe columns: {dev_matrix_path}")
    x = dev[probes].apply(pd.to_numeric, errors="coerce")
    med = x.median(axis=0, skipna=True)
    med = med.fillna(float(np.nanmedian(x.to_numpy(dtype=float))))
    xi = x.fillna(med)
    centroids = xi.assign(class_label=dev["project_id"].astype(str).to_numpy()).groupby("class_label")[probes].median()
    labels = centroids.index.astype(str).to_numpy()
    arr = centroids.to_numpy(dtype=float)
    dmat = np.mean(np.abs(arr[:, None, :] - arr[None, :, :]), axis=2)
    rows = []
    for i, source in enumerate(labels):
        for j, target in enumerate(labels):
            if i == j:
                continue
            rows.append({"source_class": source, "target_class": target, "distance": float(dmat[i, j])})
    out = pd.DataFrame(rows)
    rank = centrality.copy()
    rank["centrality_rank"] = rank["mean_distance_from_other_centroids"].rank(method="first", ascending=True).astype(int)
    return out.merge(rank[["class_label", "centrality_rank"]], left_on="source_class", right_on="class_label", how="left")


def plot_centroid_centrality(
    centrality: pd.DataFrame,
    distance_long: pd.DataFrame,
    ax: plt.Axes,
    title: str,
) -> None:
    cent = centrality.copy()
    cent["centrality_rank"] = cent["mean_distance_from_other_centroids"].rank(method="first", ascending=True).astype(int)
    order = cent.sort_values("mean_distance_from_other_centroids")["class_label"].tolist()
    data = [distance_long.loc[distance_long["source_class"].eq(cls), "distance"].to_numpy() for cls in order]
    positions = np.arange(len(order), dtype=float)
    parts = ax.violinplot(data, positions=positions, widths=0.82, showmedians=False, showextrema=False)
    for body, cls in zip(parts["bodies"], order, strict=True):
        color = ESCA_ORANGE if cls == "TCGA-ESCA" else TGCT_BLUE if cls == "TCGA-TGCT" else "#c7c7c7"
        body.set_facecolor(color)
        body.set_edgecolor("none")
        body.set_alpha(0.70 if cls in {"TCGA-ESCA", "TCGA-TGCT"} else 0.36)
    rng = np.random.default_rng(19)
    means = cent.set_index("class_label").loc[order, "mean_distance_from_other_centroids"].to_numpy(dtype=float)
    for i, cls in enumerate(order):
        vals = data[i]
        jitter = rng.normal(0, 0.035, size=len(vals))
        color = ESCA_ORANGE if cls == "TCGA-ESCA" else TGCT_BLUE if cls == "TCGA-TGCT" else "#8e9397"
        alpha = 0.55 if cls in {"TCGA-ESCA", "TCGA-TGCT"} else 0.18
        ax.scatter(np.full(len(vals), positions[i]) + jitter, vals, s=4.6, color=color, alpha=alpha, linewidth=0, zorder=2)
        med = float(np.median(vals))
        ax.plot([positions[i] - 0.20, positions[i] + 0.20], [med, med], color=INK, linewidth=0.55, zorder=3)
    ax.plot(positions, means, color=INK, linewidth=0.8, alpha=0.72, zorder=4)
    ax.scatter(
        positions,
        means,
        s=[28 if cls in {"TCGA-ESCA", "TCGA-TGCT"} else 10 for cls in order],
        color=[ESCA_ORANGE if cls == "TCGA-ESCA" else TGCT_BLUE if cls == "TCGA-TGCT" else INK for cls in order],
        edgecolor=PAPER_BG,
        linewidth=0.35,
        zorder=5,
    )
    labels = [strip_tcga(cls) for cls in order]
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=58, ha="right", fontsize=5.8)
    ax.set_ylabel("Distance to other class centroids")
    ax.set_xlabel("Classes sorted by centroid centrality")
    if title:
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=7)
    esca_pos = order.index("TCGA-ESCA")
    tgct_pos = order.index("TCGA-TGCT")
    esca_rank = int(cent.loc[cent["class_label"].eq("TCGA-ESCA"), "centrality_rank"].iloc[0])
    tgct_rank = int(cent.loc[cent["class_label"].eq("TCGA-TGCT"), "centrality_rank"].iloc[0])
    y_top = max(float(np.nanmax(distance_long["distance"])) + 0.010, float(np.nanmax(means)) + 0.028)
    ax.text(esca_pos, y_top, f"ESCA rank {esca_rank}", color=ESCA_ORANGE, fontsize=6.6, fontweight="bold", ha="center", va="bottom")
    ax.text(tgct_pos, y_top, f"TGCT rank {tgct_rank}", color=TGCT_BLUE, fontsize=6.6, fontweight="bold", ha="center", va="bottom")
    ax.set_ylim(top=y_top + 0.035)
    ax.grid(axis="y", color=GRID, linewidth=0.6, alpha=0.75)
    sns.despine(ax=ax)


def plot_tgct_perturbation(perturbation: pd.DataFrame, ax: plt.Axes, title: str) -> pd.DataFrame:
    pert = perturbation.copy()
    pert["n_true_tgct_classified"] = pd.to_numeric(pert["n_predicted_tgct"], errors="coerce") - pd.to_numeric(
        pert["n_incoming_tgct_errors"], errors="coerce"
    )
    cum = pert.loc[
        pert["perturbation"].astype(str).str.startswith("cumulative_top") | pert["perturbation"].eq("baseline")
    ].copy()
    cum["n_masked_probes"] = pd.to_numeric(cum["n_masked_probes"], errors="coerce")
    cum = cum.sort_values("n_masked_probes")
    true_total = int(pert.loc[pert["perturbation"].eq("baseline"), "n_true_tgct"].iloc[0])
    baseline = pert.loc[pert["perturbation"].eq("baseline")].iloc[0]
    top5 = pert.loc[pert["perturbation"].eq("cumulative_top5")].iloc[0]
    baseline_true = int(baseline["n_true_tgct_classified"])
    top5_true = int(top5["n_true_tgct_classified"])

    ax.plot(
        cum["n_masked_probes"],
        cum["n_incoming_tgct_errors"],
        marker="o",
        linewidth=1.4,
        markersize=4.2,
        color=TGCT_BLUE,
    )
    ax.fill_between(cum["n_masked_probes"], 0, cum["n_incoming_tgct_errors"], color=TGCT_BLUE, alpha=0.08)
    ax.axhline(0, color=INK, linewidth=0.6)
    ax.set_xlabel("Top TGCT-attracting probes neutralized")
    ax.set_ylabel("Incoming off-target TGCT predictions")
    if title:
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=7)
    ax.set_xticks(cum["n_masked_probes"].tolist())
    ax.set_ylim(-10, max(cum["n_incoming_tgct_errors"]) * 1.15)
    baseline_y = float(baseline["n_incoming_tgct_errors"])
    top5_y = float(top5["n_incoming_tgct_errors"])
    ax.annotate(
        f"{baseline_true}/{true_total} true TGCT\nsamples classified",
        xy=(0, baseline_y),
        xytext=(1.1, baseline_y - 44),
        fontsize=7,
        color=INK,
        ha="left",
        va="top",
        arrowprops={"arrowstyle": "-", "linewidth": 0.6, "color": MUTED},
    )
    ax.annotate(
        f"{top5_true}/{true_total} true TGCT\nsamples classified",
        xy=(5, top5_y),
        xytext=(5.65, 78),
        fontsize=7,
        color=INK,
        ha="left",
        va="center",
        arrowprops={"arrowstyle": "-", "linewidth": 0.6, "color": MUTED},
    )
    ax.grid(color=GRID, linewidth=0.7, alpha=0.85)
    sns.despine(ax=ax)
    return cum


def network_positions(nodes: pd.DataFrame, edges: pd.DataFrame) -> dict[str, np.ndarray]:
    try:
        import networkx as nx

        graph = nx.DiGraph()
        for node in nodes["class"].astype(str):
            graph.add_node(node)
        max_n = max(float(edges["n"].max()), 1.0)
        for edge in edges.itertuples(index=False):
            graph.add_edge(edge.true_class, edge.y_pred, weight=0.4 + float(edge.n) / max_n)
        pos = nx.spring_layout(graph, seed=19, k=1.35, weight="weight", iterations=800)
    except Exception:
        labels = list(nodes["class"].astype(str))
        angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False)
        pos = {label: np.array([np.cos(angle), np.sin(angle)]) for label, angle in zip(labels, angles)}

    arr = np.vstack([np.asarray(xy, dtype=float) for xy in pos.values()])
    center = arr.mean(axis=0)
    span = np.maximum(arr.max(axis=0) - arr.min(axis=0), 1e-9)
    return {label: (np.asarray(xy, dtype=float) - center) / span.max() * 2.0 for label, xy in pos.items()}


def repel_network_positions(
    pos: dict[str, np.ndarray],
    min_distance: float = 0.235,
    iterations: int = 140,
) -> dict[str, np.ndarray]:
    labels = list(pos)
    coords = np.vstack([np.asarray(pos[label], dtype=float) for label in labels])
    fixed_center = coords.mean(axis=0, keepdims=True)
    for _ in range(iterations):
        delta_total = np.zeros_like(coords)
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                delta = coords[i] - coords[j]
                dist = float(np.linalg.norm(delta))
                if dist >= min_distance:
                    continue
                if dist < 1e-8:
                    direction = np.array([1.0, 0.0])
                else:
                    direction = delta / dist
                push = (min_distance - dist) * 0.5 * direction
                delta_total[i] += push
                delta_total[j] -= push
        coords += delta_total * 0.55
        coords -= coords.mean(axis=0, keepdims=True) - fixed_center
    span = np.maximum(coords.max(axis=0) - coords.min(axis=0), 1e-9)
    coords = (coords - coords.mean(axis=0, keepdims=True)) / span.max() * 2.0
    return {label: coords[i] for i, label in enumerate(labels)}


def circular_network_positions(nodes: pd.DataFrame) -> dict[str, np.ndarray]:
    preferred_order = [
        "TCGA-TGCT",
        "TCGA-LAML",
        "TCGA-CESC",
        "TCGA-HNSC",
        "TCGA-LUSC",
        "TCGA-BLCA",
        "TCGA-THYM",
        "TCGA-GBM",
        "TCGA-UVM",
        "TCGA-SKCM",
        "TCGA-MESO",
        "TCGA-SARC",
        "TCGA-COAD",
        "TCGA-PAAD",
        "TCGA-ESCA",
        "TCGA-BRCA",
        "TCGA-LUAD",
        "TCGA-UCEC",
        "TCGA-UCS",
    ]
    labels = nodes["class"].astype(str).tolist()
    ordered = [label for label in preferred_order if label in labels]
    ordered.extend([label for label in labels if label not in ordered])
    n = len(ordered)
    start = np.deg2rad(92.0)
    angles = start - np.arange(n, dtype=float) * (2.0 * np.pi / n)
    radius = 1.0
    return {
        label: np.array([radius * np.cos(angle), radius * np.sin(angle)], dtype=float)
        for label, angle in zip(ordered, angles, strict=True)
    }


def edge_curvature(edge: object) -> float:
    pair = tuple(sorted([str(edge.true_class), str(edge.y_pred)]))
    reciprocal = str(edge.true_class) > str(edge.y_pred)
    base = 0.14 if reciprocal else -0.14
    if pair in {("TCGA-CESC", "TCGA-HNSC"), ("TCGA-HNSC", "TCGA-LUSC")}:
        return base * 2.25
    if pair in {("TCGA-COAD", "TCGA-ESCA"), ("TCGA-LUAD", "TCGA-LUSC"), ("TCGA-UCEC", "TCGA-UCS")}:
        return base * 1.8
    return base


def plot_edge_label(category: str) -> str:
    return EDGE_BIOLOGY_LABELS[category]


def edge_linestyle(category: str) -> str | tuple[int, tuple[float, ...]]:
    dashes = EDGE_BIOLOGY_STYLES[category].get("dashes")
    if not dashes:
        return "solid"
    return (0, tuple(float(x) for x in dashes))


def plot_network(
    nodes: pd.DataFrame,
    edges: pd.DataFrame,
    ax: plt.Axes,
    title: str,
    colorbar: bool = True,
    node_scale: float = 1.0,
    edge_scale: float = 1.0,
    node_label_fontsize: float = 7.0,
    edge_label_fontsize: float = 6.5,
    show_edge_labels: bool = True,
    edge_shrink: float = 15.0,
    repel_positions: bool = False,
    layout: str = "spring",
) -> None:
    if layout == "circle":
        pos = circular_network_positions(nodes)
    else:
        pos = network_positions(nodes, edges)
    if repel_positions:
        pos = repel_network_positions(pos)
    cmap = LinearSegmentedColormap.from_list("recall", [ORANGE, "#f6f2ea", BLUE])
    norm = Normalize(vmin=0, vmax=1)
    max_fraction = max(float(edges["fraction_of_true_class"].max()), 1e-9)
    for edge in edges.sort_values("fraction_of_true_class").itertuples(index=False):
        start = pos[edge.true_class]
        end = pos[edge.y_pred]
        width = edge_scale * (0.6 + 3.4 * float(edge.fraction_of_true_class) / max_fraction)
        category = getattr(edge, "biology_edge_category", "no_simple_match")
        style = EDGE_BIOLOGY_STYLES.get(category, EDGE_BIOLOGY_STYLES["no_simple_match"])
        rad = edge_curvature(edge)
        ax.annotate(
            "",
            xy=end,
            xytext=start,
            arrowprops={
                "arrowstyle": "-|>",
                "color": style["color"],
                "lw": width,
                "alpha": style["alpha"],
                "linestyle": edge_linestyle(category),
                "shrinkA": edge_shrink,
                "shrinkB": edge_shrink,
                "mutation_scale": 11 * edge_scale,
                "connectionstyle": f"arc3,rad={rad}",
                "zorder": 3.2,
            },
        )
    max_incoming = max(1, int(nodes["wrong_incoming_n"].max()))
    for row in nodes.to_dict(orient="records"):
        node_class = row["class"]
        xy = pos[node_class]
        size = node_scale * (170 + 620 * row["wrong_incoming_n"] / max_incoming)
        ax.scatter(
            [xy[0]],
            [xy[1]],
            s=size,
            c=[cmap(norm(row["recall"]))],
            edgecolors=PAPER_BG,
            linewidths=0.9,
            zorder=2.5,
        )
        ax.text(
            xy[0],
            xy[1],
            strip_tcga(node_class),
            ha="center",
            va="center",
            fontsize=node_label_fontsize,
            color=INK,
            fontweight="bold",
            zorder=4,
        )
    if show_edge_labels:
        for edge in edges.loc[edges["n"].ge(40)].sort_values(["fraction_of_true_class", "n"], ascending=False).itertuples(index=False):
            start = pos[edge.true_class]
            end = pos[edge.y_pred]
            vec = end - start
            norm_vec = np.linalg.norm(vec)
            if norm_vec > 0:
                perp = np.array([-vec[1], vec[0]]) / norm_vec
            else:
                perp = np.array([0.0, 0.0])
            mid = (start + end) / 2 + perp * edge_curvature(edge) * 0.24
            ax.text(
                mid[0],
                mid[1],
                f"n={edge.n}",
                fontsize=edge_label_fontsize,
                color=INK,
                ha="center",
                va="center",
                bbox={"boxstyle": "round,pad=0.13", "facecolor": PAPER_BG, "edgecolor": GRID, "linewidth": 0.5},
                zorder=5,
            )
    if title:
        ax.set_title(title, loc="left", fontsize=12, fontweight="bold", pad=7)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    ax.set_frame_on(False)
    legend_handles = []
    present_categories = [c for c in EDGE_BIOLOGY_LABELS if c in set(edges.get("biology_edge_category", []))]
    for category in present_categories:
        style = EDGE_BIOLOGY_STYLES[category]
        handle = plt.Line2D(
            [0],
            [0],
            color=style["color"],
            linewidth=2.2,
            linestyle=edge_linestyle(category),
            alpha=max(style["alpha"], 0.7),
            label=plot_edge_label(category),
        )
        legend_handles.append(handle)
    if legend_handles:
        ax.legend(
            handles=legend_handles,
            frameon=False,
            loc="lower left",
            bbox_to_anchor=(0.0, -0.02),
            fontsize=6.6,
            title="Edge interpretation",
            title_fontsize=7,
            handlelength=3.4,
            handletextpad=0.75,
        )
    ax.margins(0.18)
    if colorbar:
        sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, fraction=0.044, pad=0.01)
        cbar.set_label("External recall", fontsize=8)
        cbar.ax.tick_params(labelsize=7)


def add_panel_letters(axes: list[plt.Axes], labels: list[str]) -> None:
    for ax, label in zip(axes, labels, strict=True):
        ax.text(
            -0.08,
            1.035,
            label,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=13,
            fontweight="bold",
            color=INK,
            clip_on=False,
        )


def save_figure(fig: plt.Figure, out_dir: Path, stem: str, formats: list[str]) -> list[str]:
    paths = []
    for fmt in formats:
        path = out_dir / f"{stem}.{fmt}"
        fig.savefig(
            path,
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.03,
            facecolor=fig.get_facecolor(),
        )
        paths.append(str(path))
    plt.close(fig)
    return paths


def main() -> None:
    args = parse_args()
    validation = validate_active_run(args.active_manifest)
    failure_dir = Path(args.failure_dir)
    hub_audit_dir = Path(args.hub_audit_dir)
    tgct_perturbation_dir = Path(args.tgct_perturbation_dir)
    fig_dir = failure_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    set_theme()

    class_summary, conf, network_nodes, network_edges = load_inputs(failure_dir)
    hub_centrality = pd.read_csv(hub_audit_dir / "hub_centroid_centrality.tsv", sep="\t")
    tgct_perturbation = pd.read_csv(tgct_perturbation_dir / "tgct_probe_perturbation_summary.tsv", sep="\t")
    hub_distance_long = centroid_distance_long(hub_centrality, Path(args.tcga_dev_matrix))
    network_edges = annotate_network_edges(network_edges, network_nodes, args.low_support_threshold)
    network_class_context = build_class_context_table(network_nodes)
    class_table, flag_df = build_class_plot_table(class_summary)

    class_table_path = failure_dir / "failure_mode_main_class_plot_data.tsv"
    flag_matrix_path = failure_dir / "failure_mode_flag_matrix_plot_data.tsv"
    bio_edges_path = failure_dir / "failure_mode_confusion_network_edges_biology.tsv"
    edge_audit_path = failure_dir / "failure_mode_confusion_network_edge_annotation_audit.tsv"
    class_context_path = failure_dir / "failure_mode_confusion_network_class_context.tsv"
    annotation_rules_path = failure_dir / "failure_mode_confusion_network_annotation_rules.md"
    class_table.to_csv(class_table_path, sep="\t", index=False)
    flag_df.to_csv(flag_matrix_path, sep="\t", index=False)
    hub_distance_path = failure_dir / "failure_mode_hub_centroid_distance_long.tsv"
    tgct_cumulative_path = failure_dir / "failure_mode_tgct_cumulative_probe_perturbation.tsv"
    hub_distance_long.to_csv(hub_distance_path, sep="\t", index=False)
    network_edges.to_csv(bio_edges_path, sep="\t", index=False)
    network_edges.to_csv(edge_audit_path, sep="\t", index=False)
    network_class_context.to_csv(class_context_path, sep="\t", index=False)
    write_network_annotation_rules(annotation_rules_path, args.low_support_threshold)

    figure_paths: list[str] = []

    fig, ax = plt.subplots(figsize=(6.1, max(5.2, 0.27 * len(class_table) + 1.2)))
    plot_recall_slope(class_table, ax, args.external_drop_threshold, "External degradation by class")
    figure_paths.extend(save_figure(fig, fig_dir, "failure_mode_recall_slope", args.formats))

    fig, ax = plt.subplots(figsize=(8.0, max(5.4, 0.27 * len(class_table) + 1.3)))
    plot_flag_matrix(flag_df, class_table, ax, "Failure-mode flags by class")
    figure_paths.extend(save_figure(fig, fig_dir, "failure_mode_flag_matrix", args.formats))

    fig, ax = plt.subplots(figsize=(8.6, 7.0))
    plot_network(network_nodes, network_edges, ax, "Directed external confusions", colorbar=True)
    fig.tight_layout()
    figure_paths.extend(save_figure(fig, fig_dir, "failure_mode_confusion_network_biology", args.formats))

    fig, axes = plt.subplots(2, 2, figsize=(15.5, 12.3), gridspec_kw={"width_ratios": [0.88, 1.12]})
    plot_recall_slope(class_table, axes[0, 0], args.external_drop_threshold, "A. External degradation by class")
    plot_network(network_nodes, network_edges, axes[0, 1], "B. Directed external confusions", colorbar=True)
    plot_flag_matrix(flag_df, class_table, axes[1, 0], "C. Failure-mode flags")
    plot_beta_shift(class_summary, axes[1, 1], "D. Recall drop vs methylation shift")
    fig.suptitle(
        "Failure modes of the 30-CpG classifier",
        x=0.02,
        y=0.995,
        ha="left",
        fontsize=17,
        fontweight="bold",
        color=INK,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    figure_paths.extend(save_figure(fig, fig_dir, "failure_mode_main_panel", args.formats))

    fig = plt.figure(figsize=(15.5, 17.6), constrained_layout=False)
    gs = fig.add_gridspec(
        3,
        2,
        width_ratios=[0.70, 1.30],
        height_ratios=[1.20, 1.08, 0.82],
        hspace=0.31,
        wspace=0.24,
    )
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 0])
    ax_d = fig.add_subplot(gs[1, 1])
    ax_e = fig.add_subplot(gs[2, 0])
    ax_f = fig.add_subplot(gs[2, 1])
    plot_recall_slope(class_table, ax_a, args.external_drop_threshold, "")
    plot_network(
        network_nodes,
        network_edges,
        ax_b,
        "",
        colorbar=True,
        node_scale=0.76,
        edge_scale=0.78,
        node_label_fontsize=6.8,
        show_edge_labels=False,
        edge_shrink=11.5,
        layout="circle",
    )
    plot_flag_matrix(flag_df, class_table, ax_c, "")
    plot_beta_shift(class_summary, ax_d, "", size_scale=0.92)
    plot_centroid_centrality(hub_centrality, hub_distance_long, ax_e, "")
    tgct_cumulative = plot_tgct_perturbation(tgct_perturbation, ax_f, "")
    tgct_cumulative.to_csv(tgct_cumulative_path, sep="\t", index=False)
    add_panel_letters([ax_a, ax_b, ax_c, ax_d, ax_e, ax_f], list("ABCDEF"))
    fig.subplots_adjust(left=0.06, right=0.975, top=0.965, bottom=0.075, hspace=0.34, wspace=0.22)
    figure_paths.extend(save_figure(fig, fig_dir, "failure_mode_main_panel_hub_extended", args.formats))

    summary = {
        "active_manifest": args.active_manifest,
        "active_validation_status": validation.get("status"),
        "active_locked_run_name": validation.get("locked_run_name"),
        "active_version": validation.get("version"),
        "inputs": {
            "failure_dir": str(failure_dir),
            "hub_audit_dir": str(hub_audit_dir),
            "tgct_perturbation_dir": str(tgct_perturbation_dir),
            "tcga_dev_matrix": str(args.tcga_dev_matrix),
            "class_summary_tsv": str(failure_dir / "failure_mode_class_summary.tsv"),
            "network_nodes_tsv": str(failure_dir / "failure_mode_confusion_network_nodes.tsv"),
            "network_edges_tsv": str(failure_dir / "failure_mode_confusion_network_edges.tsv"),
            "hub_centroid_centrality_tsv": str(hub_audit_dir / "hub_centroid_centrality.tsv"),
            "tgct_probe_perturbation_summary_tsv": str(tgct_perturbation_dir / "tgct_probe_perturbation_summary.tsv"),
        },
        "outputs": {
            "class_plot_data_tsv": str(class_table_path),
            "flag_matrix_plot_data_tsv": str(flag_matrix_path),
            "hub_centroid_distance_long_tsv": str(hub_distance_path),
            "tgct_cumulative_probe_perturbation_tsv": str(tgct_cumulative_path),
            "network_edges_biology_tsv": str(bio_edges_path),
            "network_edge_annotation_audit_tsv": str(edge_audit_path),
            "network_class_context_tsv": str(class_context_path),
            "network_annotation_rules_md": str(annotation_rules_path),
            "figures": figure_paths,
        },
        "network_edge_category_counts": network_edges["biology_edge_label"].value_counts().to_dict(),
        "network_edge_rule_counts": network_edges["biology_edge_rule"].value_counts().to_dict(),
        "n_classes_external": int(len(class_table)),
        "n_flagged_external_classes": int(
            class_table["failure_flags"].fillna("").astype(str).str.len().gt(0).sum()
        ),
        "n_external_drop_classes": int((class_table["external_recall_delta"] <= args.external_drop_threshold).sum()),
        "external_drop_threshold": args.external_drop_threshold,
        "network_low_support_threshold": args.low_support_threshold,
        "esca_centroid_centrality_rank": int(
            hub_centrality["mean_distance_from_other_centroids"].rank(method="first", ascending=True)[
                hub_centrality["class_label"].eq("TCGA-ESCA")
            ].iloc[0]
        ),
        "tgct_centroid_centrality_rank": int(
            hub_centrality["mean_distance_from_other_centroids"].rank(method="first", ascending=True)[
                hub_centrality["class_label"].eq("TCGA-TGCT")
            ].iloc[0]
        ),
        "tgct_baseline_incoming_offtarget": int(
            tgct_perturbation.loc[tgct_perturbation["perturbation"].eq("baseline"), "n_incoming_tgct_errors"].iloc[0]
        ),
        "tgct_top5_incoming_offtarget": int(
            tgct_perturbation.loc[tgct_perturbation["perturbation"].eq("cumulative_top5"), "n_incoming_tgct_errors"].iloc[0]
        ),
        "classes_sorted_by_delta": class_table[
            ["class", "external_recall", "lockbox_recall", "external_recall_delta", "n_samples", "failure_flags"]
        ].to_dict(orient="records"),
    }
    summary_path = failure_dir / "failure_mode_main_figures_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
