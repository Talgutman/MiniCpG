#!/usr/bin/env python3
"""Build a study-design figure for MiniCpG."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "methylation_classification_mpl_cache"),
)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


PAPER_BG = "#fbfaf7"
INK = "#24313f"
MUTED = "#65717d"
GRID = "#e3dccf"
BLUE = "#2f6f9f"
BLUE_LIGHT = "#d9e7f2"
TEAL = "#4f8a78"
TEAL_LIGHT = "#dcebe4"
GOLD = "#c7903a"
GOLD_LIGHT = "#f2e4c7"
ORANGE = "#c46a3a"
ORANGE_LIGHT = "#f0d9cb"
PURPLE = "#7d6a9b"
PURPLE_LIGHT = "#e4deed"
WHITE = "#fffdf9"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--active-manifest", default="config/active_locked_run.json")
    p.add_argument(
        "--cohort-counts-tsv",
        default=(
            "outputs/paper_protocolb_v1/manuscript_current/figures/"
            "methods_anchor_figures_current/methods_fig_m2_tcga_cohort_counts_plot_data.tsv"
        ),
    )
    p.add_argument(
        "--cohort-harmonization-summary-json",
        default="outputs/manifests/classification_manifest_hm450_min50_merged_summary.json",
    )
    p.add_argument(
        "--selected-pipeline-json",
        default="outputs/paper_protocolb_v1/k_030/selected_pipeline.json",
    )
    p.add_argument(
        "--external-primary-summary-json",
        default="outputs/paper_protocolb_v1/k_030/external_main_beta_sanitized_current/eval/external_eval_summary.json",
    )
    p.add_argument(
        "--external-metastatic-summary-json",
        default=(
            "outputs/paper_protocolb_v1/k_030/"
            "external_stress_metastatic_beta_sanitized_current/eval/external_eval_summary.json"
        ),
    )
    p.add_argument(
        "--external-ffpe-summary-json",
        default="outputs/paper_protocolb_v1/k_030/external_stress_ffpe_beta_sanitized_current/eval/external_eval_summary.json",
    )
    p.add_argument(
        "--external-epic-summary-json",
        default="outputs/paper_protocolb_v1/k_030/external_stress_epic_beta_sanitized_current/eval/external_eval_summary.json",
    )
    p.add_argument(
        "--candidate-metrics-tsv",
        default=(
            "outputs/paper_protocolb_v1/manuscript_current/figures/"
            "fig1_tcga_model_selection_current/fig1_tcga_model_selection_dev_stage1_candidate_metrics.tsv"
        ),
    )
    p.add_argument(
        "--external-design-summary-json",
        default=(
            "outputs/paper_protocolb_v1/manuscript_current/figures/"
            "methods_anchor_figures_current/methods_fig_m3_external_validation_design_summary.json"
        ),
    )
    p.add_argument(
        "--out-dir",
        default="outputs/paper_protocolb_v1/manuscript_reproduction",
    )
    p.add_argument("--out-stem", default="fig1_study_design_revision_candidate")
    p.add_argument("--formats", nargs="+", default=["png", "pdf"], choices=["png", "pdf"])
    return p.parse_args()


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


def read_tsv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    if not rows:
        raise SystemExit(f"Input table is empty: {path}")
    return rows


def load_tcga_summary(path: Path) -> dict[str, int]:
    rows = read_tsv_rows(path)
    required = {"tcga_label", "n_dev", "n_lockbox", "n_total"}
    missing = required - set(rows[0])
    if missing:
        raise SystemExit(f"TCGA cohort counts table missing columns: {sorted(missing)}")
    return {
        "n_classes": len(rows),
        "n_samples": sum(int(r["n_total"]) for r in rows),
        "n_development": sum(int(r["n_dev"]) for r in rows),
        "n_heldout": sum(int(r["n_lockbox"]) for r in rows),
    }


def load_harmonization_summary(path: Path) -> dict[str, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "n_rows_input": int(payload["n_rows_input"]),
        "n_classes_pre": int(payload["n_classes_pre"]),
        "n_rows_output": int(payload["n_rows_output"]),
        "n_classes_post_filter": int(payload["n_classes_post_filter"]),
    }


def load_selected_pipeline(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "k": int(payload["k"]),
        "model": str(payload["model"]),
        "selection_method": str(payload["selection_method"]),
        "prefilter_method": str(payload["prefilter_method"]),
        "prefilter_k": int(payload["prefilter_k"]),
    }


def load_candidate_screen_summary(path: Path) -> dict[str, int]:
    rows = read_tsv_rows(path)
    required = {"joint_candidate_id", "k", "n_folds"}
    missing = required - set(rows[0])
    if missing:
        raise SystemExit(f"Candidate metrics table missing columns: {sorted(missing)}")
    k_values = sorted({int(r["k"]) for r in rows})
    return {
        "n_candidates": len({r["joint_candidate_id"] for r in rows}),
        "k_min": min(k_values),
        "k_max": max(k_values),
        "n_k_values": len(k_values),
        "stage1_outer_folds": max(int(r["n_folds"]) for r in rows),
    }


def load_external_design(path: Path) -> dict[str, dict[str, int]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, dict[str, int]] = {}
    for bucket in payload["buckets"]:
        result[str(bucket["bucket"])] = {
            "n_samples": int(bucket["n_samples"]),
            "n_classes": int(bucket["n_classes"]),
            "n_datasets": int(bucket["n_datasets"]),
        }
    return result


def load_external_summary(path: Path) -> dict[str, int]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {
        "n_samples": int(payload["n_samples"]),
        "n_classes": int(payload["n_classes_in_truth"]),
    }


def fmt_int(value: int) -> str:
    return f"{value:,}"


def wrap(text: str, width: int) -> str:
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False))


def draw_round_rect(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    facecolor: str,
    edgecolor: str,
    linewidth: float = 0.9,
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.014,rounding_size=0.018",
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=linewidth,
        )
    )


def draw_card(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    step: str,
    title: str,
    lines: list[str],
    facecolor: str,
    edgecolor: str,
    title_width: int = 26,
) -> None:
    draw_round_rect(ax, x, y, w, h, facecolor=facecolor, edgecolor=edgecolor)
    ax.text(
        x + 0.025,
        y + h - 0.075,
        step,
        ha="left",
        va="center",
        fontsize=6.2,
        fontweight="bold",
        color=edgecolor,
    )
    ax.text(
        x + 0.025,
        y + h - 0.135,
        wrap(title, title_width),
        ha="left",
        va="top",
        fontsize=7.2,
        fontweight="bold",
        color=INK,
        linespacing=1.04,
    )
    wrapped_lines = []
    for line in lines:
        wrapped_lines.extend(textwrap.wrap(line, width=32, break_long_words=False))
    ax.text(
        x + 0.025,
        y + h - 0.245,
        "\n".join(wrapped_lines),
        ha="left",
        va="top",
        fontsize=5.9,
        color=INK,
        linespacing=1.25,
    )


def draw_arrow(ax: plt.Axes, start: tuple[float, float], end: tuple[float, float]) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=10,
            linewidth=0.9,
            color=MUTED,
            shrinkA=5,
            shrinkB=5,
            connectionstyle="arc3,rad=0.0",
        )
    )


def draw_validation_band(
    ax: plt.Axes,
    *,
    x: float,
    y: float,
    w: float,
    h: float,
    tcga: dict[str, int],
    primary: dict[str, int],
    metastatic: dict[str, int],
    ffpe: dict[str, int],
    epic: dict[str, int],
) -> None:
    draw_round_rect(ax, x, y, w, h, facecolor=WHITE, edgecolor=TEAL, linewidth=0.9)
    ax.text(
        x + 0.025,
        y + h - 0.07,
        "LOCKED VALIDATION",
        ha="left",
        va="center",
        fontsize=6.2,
        fontweight="bold",
        color=TEAL,
    )
    ax.text(
        x + 0.025,
        y + h - 0.125,
        "Final classifier evaluated without further tuning",
        ha="left",
        va="top",
        fontsize=7.2,
        fontweight="bold",
        color=INK,
    )

    items = [
        ("Held-out TCGA", f"{fmt_int(tcga['n_heldout'])} samples\n{tcga['n_classes']} classes", TEAL_LIGHT, TEAL),
        ("Primary external", f"{fmt_int(primary['n_samples'])} samples\n{primary['n_classes']} classes", BLUE_LIGHT, BLUE),
        ("Metastatic", f"{fmt_int(metastatic['n_samples'])} samples\n{metastatic['n_classes']} classes", ORANGE_LIGHT, ORANGE),
        ("FFPE", f"{fmt_int(ffpe['n_samples'])} samples\n{ffpe['n_classes']} classes", GOLD_LIGHT, GOLD),
        ("EPIC", f"{fmt_int(epic['n_samples'])} samples\n{epic['n_classes']} classes", TEAL_LIGHT, TEAL),
    ]
    gap = 0.012
    sw = (w - 0.05 - gap * (len(items) - 1)) / len(items)
    sy = y + 0.045
    sh = 0.155
    for i, (label, value, face, edge) in enumerate(items):
        sx = x + 0.025 + i * (sw + gap)
        draw_round_rect(ax, sx, sy, sw, sh, facecolor=face, edgecolor=edge, linewidth=0.65)
        ax.text(sx + sw / 2, sy + sh - 0.055, label, ha="center", va="center", fontsize=6.1, fontweight="bold")
        ax.text(sx + sw / 2, sy + 0.045, value, ha="center", va="bottom", fontsize=5.9, linespacing=1.18)


def draw_band(
    ax: plt.Axes,
    *,
    x: float,
    y: float,
    w: float,
    h: float,
    step: str,
    title: str,
    facecolor: str,
    edgecolor: str,
) -> None:
    draw_round_rect(ax, x, y, w, h, facecolor=facecolor, edgecolor=edgecolor, linewidth=0.95)
    ax.text(
        x + 0.025,
        y + h - 0.055,
        step,
        ha="left",
        va="center",
        fontsize=5.9,
        fontweight="bold",
        color=edgecolor,
    )
    if title:
        ax.text(
            x + 0.025,
            y + h - 0.095,
            wrap(title, 58),
            ha="left",
            va="top",
            fontsize=7.0,
            fontweight="bold",
            color=INK,
            linespacing=1.02,
        )


def draw_tile(
    ax: plt.Axes,
    *,
    x: float,
    y: float,
    w: float,
    h: float,
    label: str,
    value: str,
    facecolor: str,
    edgecolor: str,
) -> None:
    draw_round_rect(ax, x, y, w, h, facecolor=facecolor, edgecolor=edgecolor, linewidth=0.65)
    ax.text(
        x + w / 2,
        y + h / 2,
        f"{wrap(label, 18)}\n{value}",
        ha="center",
        va="center",
        fontsize=5.4,
        color=INK,
        linespacing=1.18,
    )


def draw_tile_row(
    ax: plt.Axes,
    *,
    x: float,
    y: float,
    w: float,
    h: float,
    items: list[tuple[str, str, str, str]],
) -> None:
    gap = 0.012
    tile_w = (w - gap * (len(items) - 1)) / len(items)
    for i, (label, value, face, edge) in enumerate(items):
        draw_tile(
            ax,
            x=x + i * (tile_w + gap),
            y=y,
            w=tile_w,
            h=h,
            label=label,
            value=value,
            facecolor=face,
            edgecolor=edge,
        )


def draw_rect_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    text: str,
    facecolor: str,
    edgecolor: str = "#4c4c4c",
    letter: str | None = None,
    fontsize: float = 6.2,
    weight: str = "normal",
    wrap_width: int = 24,
    linestyle: str = "-",
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.006,rounding_size=0.006",
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=0.85,
            linestyle=linestyle,
        )
    )
    if letter:
        ax.text(
            x + 0.008,
            y + h / 2,
            f"{letter}.",
            ha="left",
            va="center",
            fontsize=6.0,
            fontweight="bold",
            color=edgecolor,
        )
        text_x = x + w / 2 + 0.012
    else:
        text_x = x + w / 2
    ax.text(
        text_x,
        y + h / 2,
        wrap(text, wrap_width),
        ha="center",
        va="center",
        fontsize=fontsize,
        fontweight=weight,
        color=INK,
        linespacing=1.12,
    )


def draw_step_box(
    ax: plt.Axes,
    x: float,
    y: float,
    w: float,
    h: float,
    *,
    letter: str,
    title: str | list[str],
    details: list[str],
    facecolor: str,
    edgecolor: str,
    title_size: float = 5.3,
    detail_size: float = 4.85,
) -> None:
    title_lines = [title] if isinstance(title, str) else title
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.006,rounding_size=0.006",
            facecolor=facecolor,
            edgecolor=edgecolor,
            linewidth=0.85,
        )
    )
    ax.text(
        x + 0.014,
        y + h / 2,
        f"{letter}.",
        ha="left",
        va="center",
        fontsize=6.0,
        fontweight="bold",
        color=edgecolor,
    )
    text_x = x + w / 2 + 0.012
    items: list[tuple[str, float, str, float]] = []
    for line in title_lines:
        items.append((line, title_size, "bold", 1.10))
    if title_lines and details:
        items.append(("", detail_size, "normal", 0.28))
    for line in details:
        items.append((line, detail_size, "normal", 1.00))

    top_pad = min(0.012, h * 0.18)
    bottom_pad = min(0.010, h * 0.16)
    available = h - top_pad - bottom_pad
    total_units = sum(unit for _, _, _, unit in items)
    y_cursor = y + h - top_pad
    for line, size, weight, unit in items:
        segment = available * unit / total_units
        y_center = y_cursor - segment / 2
        y_cursor -= segment
        if not line:
            continue
        ax.text(
            text_x,
            y_center,
            line,
            ha="center",
            va="center",
            fontsize=size,
            fontweight=weight,
            color=INK,
        )


def draw_line_arrow(
    ax: plt.Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    linestyle: str = "-",
    color: str = "#4f5b66",
) -> None:
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=9,
            linewidth=0.85,
            color=color,
            linestyle=linestyle,
            shrinkA=3,
            shrinkB=3,
        )
    )


def draw_panel_label(ax: plt.Axes, letter: str, x: float, y: float) -> None:
    ax.text(x, y, letter, ha="left", va="top", fontsize=10.5, fontweight="bold", color=INK)


def draw_technology_panel(ax: plt.Axes) -> None:
    draw_panel_label(ax, "A", 0.025, 0.975)
    box_y = 0.798
    box_h = 0.115
    box_w = 0.170
    title_y = box_y + box_h + 0.010

    chip_x, chip_y, chip_w, chip_h = 0.055, box_y, box_w, box_h
    ax.add_patch(
        FancyBboxPatch(
            (chip_x, chip_y),
            chip_w,
            chip_h,
            boxstyle="round,pad=0.006,rounding_size=0.01",
            facecolor="#e7eef6",
            edgecolor=BLUE,
            linewidth=1.0,
        )
    )
    ax.text(chip_x + chip_w / 2, title_y, "Sample methylation profile", ha="center", va="bottom", fontsize=6.0, fontweight="bold", color=INK)
    grid_cols = 9
    grid_rows = 4
    dx = 0.015
    dy = 0.017
    grid_w = (grid_cols - 1) * dx
    grid_h = (grid_rows - 1) * dy
    start_x = chip_x + chip_w / 2 - grid_w / 2
    start_y = chip_y + chip_h / 2 - grid_h / 2
    for r in range(4):
        for c in range(9):
            color = BLUE if (r + c) % 3 == 0 else GOLD if (r + c) % 3 == 1 else TEAL
            ax.plot(start_x + c * dx, start_y + r * dy, "o", ms=2.2, color=color, alpha=0.85)

    draw_line_arrow(ax, (0.238, 0.855), (0.282, 0.855))

    strip_x, strip_y, strip_w, strip_h = 0.295, box_y, box_w, box_h
    ax.add_patch(
        FancyBboxPatch(
            (strip_x, strip_y),
            strip_w,
            strip_h,
            boxstyle="round,pad=0.006,rounding_size=0.008",
            facecolor=GOLD_LIGHT,
            edgecolor=GOLD,
            linewidth=1.0,
        )
    )
    ax.text(strip_x + strip_w / 2, title_y, "30 selected CpGs", ha="center", va="bottom", fontsize=6.0, fontweight="bold", color=INK)
    strip_inner_w = 0.086
    strip_start_x = strip_x + strip_w / 2 - strip_inner_w / 2
    for i in range(10):
        x0 = strip_start_x + i * 0.008
        height = 0.026 + (i % 5) * 0.011
        color = plt.cm.YlGnBu(0.25 + 0.055 * i)
        ax.add_patch(
            FancyBboxPatch(
                (x0, strip_y + 0.020),
                0.006,
                height,
                boxstyle="round,pad=0.001,rounding_size=0.002",
                facecolor=color,
                edgecolor="none",
            )
        )

    draw_line_arrow(ax, (0.478, 0.855), (0.512, 0.855))

    model_x, model_y, model_w, model_h = 0.525, box_y, box_w, box_h
    ax.add_patch(
        FancyBboxPatch(
            (model_x, model_y),
            model_w,
            model_h,
            boxstyle="round,pad=0.006,rounding_size=0.008",
            facecolor=TEAL_LIGHT,
            edgecolor=TEAL,
            linewidth=1.0,
        )
    )
    ax.text(model_x + model_w / 2, title_y, "Locked MiniCpG classifier", ha="center", va="bottom", fontsize=6.0, fontweight="bold", color=INK)
    tree_xs = [model_x + 0.042, model_x + 0.085, model_x + 0.128]
    for tx in tree_xs:
        ty = model_y + 0.058
        branch_dx = 0.015
        ax.plot([tx, tx], [ty - 0.030, ty], color=TEAL, linewidth=0.9)
        ax.plot([tx, tx - branch_dx], [ty - 0.006, ty - 0.025], color=TEAL, linewidth=0.9)
        ax.plot([tx, tx + branch_dx], [ty - 0.006, ty - 0.025], color=TEAL, linewidth=0.9)
        ax.plot(tx, ty, "o", ms=3.2, color=TEAL)
        ax.plot([tx - branch_dx, tx + branch_dx], [ty - 0.025, ty - 0.025], "o", ms=2.6, color=TEAL)

    draw_line_arrow(ax, (0.708, 0.855), (0.752, 0.855))

    out_x, out_y, out_w, out_h = 0.765, box_y, box_w, box_h
    ax.add_patch(
        FancyBboxPatch(
            (out_x, out_y),
            out_w,
            out_h,
            boxstyle="round,pad=0.006,rounding_size=0.008",
            facecolor=WHITE,
            edgecolor=PURPLE,
            linewidth=1.0,
        )
    )
    ax.text(out_x + out_w / 2, title_y, "Cancer-type prediction", ha="center", va="bottom", fontsize=6.0, fontweight="bold", color=INK)
    labels = ["BRCA", "LUAD", "COAD", "GBM", "PRAD", "KIRC", "OV", "LUSC"]
    colors = [PURPLE, BLUE, TEAL, GOLD, ORANGE, "#58b6b1", "#86a9d6", "#d7a85b"]
    tile_w = 0.029
    tile_h = 0.019
    gap_x = 0.007
    x0 = out_x + 0.015
    y0 = out_y + 0.075
    for i, (label, color) in enumerate(zip(labels, colors, strict=True)):
        col = i % 4
        row = i // 4
        selected = i == 0
        w = tile_w + (0.010 if selected else 0.0)
        h = tile_h + (0.006 if selected else 0.0)
        xx = x0 + col * (tile_w + gap_x)
        if i > 0:
            xx += 0.008
        if selected:
            xx -= 0.004
        yy = y0 - row * 0.038 - (0.003 if selected else 0.0)
        ax.add_patch(
            FancyBboxPatch(
                (xx, yy),
                w,
                h,
                boxstyle="round,pad=0.002,rounding_size=0.004",
                facecolor=PURPLE_LIGHT if selected else "#fbf8f1",
                edgecolor=color,
                linewidth=0.8 if selected else 0.45,
            )
        )
        ax.text(
            xx + w / 2,
            yy + h / 2,
            label,
            ha="center",
            va="center",
            fontsize=4.9 if selected else 3.9,
            fontweight="bold" if selected else "normal",
            color=INK if selected else MUTED,
        )


def plot_figure(
    *,
    tcga: dict[str, int],
    harmonization: dict[str, int],
    pipeline: dict[str, object],
    primary: dict[str, int],
    metastatic: dict[str, int],
    ffpe: dict[str, int],
    epic: dict[str, int],
    screen: dict[str, int],
    external_design: dict[str, dict[str, int]],
    stage2_folds: int,
    out_paths: list[Path],
) -> None:
    configure_matplotlib()
    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    draw_technology_panel(ax)
    draw_panel_label(ax, "B", 0.025, 0.755)

    split_dev_pct = round(tcga["n_development"] / tcga["n_samples"] * 100)
    split_hold_pct = 100 - split_dev_pct

    block_specs = [
        (0.045, 0.315, 0.43, 0.38, "Model development and locking", BLUE, "#f7fbff"),
        (0.525, 0.315, 0.43, 0.38, "Validation predictions", TEAL, "#f6fbf9"),
        (0.20, 0.08, 0.60, 0.17, "Interpretation", PURPLE, "#f7f3fb"),
    ]
    for x0, y0, w0, h0, title, edge, face in block_specs:
        ax.add_patch(
            FancyBboxPatch(
                (x0, y0),
                w0,
                h0,
                boxstyle="round,pad=0.01,rounding_size=0.006",
                facecolor=face,
                edgecolor=edge,
                linewidth=0.95,
                linestyle="--",
            )
        )
        ax.text(x0 + 0.018, y0 + h0 - 0.018, title, ha="left", va="top", fontsize=6.4, fontweight="bold", color=edge)

    left_boxes = [
        (
            "A",
            "Dataset curation",
            ["TCGA primary tumors", "HM450 methylation data", f"(n = {fmt_int(harmonization['n_rows_input'])}, {harmonization['n_classes_pre']} cohorts)"],
            0.065,
            0.575,
            0.18,
            0.068,
        ),
        (
            "B",
            "Cohort harmonization",
            ["label merging", "cohort filtering", f"(n = {fmt_int(harmonization['n_rows_output'])}, {harmonization['n_classes_post_filter']} cohorts)"],
            0.275,
            0.575,
            0.18,
            0.068,
        ),
        (
            "C",
            ["Stratified", f"{split_dev_pct}/{split_hold_pct} split"],
            [f"DEV (n = {fmt_int(tcga['n_development'])})", f"Held-out (n = {fmt_int(tcga['n_heldout'])})"],
            0.065,
            0.488,
            0.18,
            0.068,
        ),
        (
            "D",
            "Pipeline selection",
            [f"K = {screen['k_min']}-{screen['k_max']}", "feature selector", "model + HPs", f"{screen['stage1_outer_folds']} outer CVs", f"{fmt_int(screen['n_candidates'])} candidates"],
            0.275,
            0.463,
            0.18,
            0.093,
        ),
        (
            "E",
            ["Minimal panel", "selection"],
            [f"{stage2_folds} CVs on locked pipeline", f"most frequent {pipeline['k']} CpGs"],
            0.065,
            0.375,
            0.18,
            0.070,
        ),
        (
            "F",
            ["Final model", "training"],
            ["locked panel + model", "full DEV set"],
            0.275,
            0.375,
            0.18,
            0.070,
        ),
    ]
    for letter, title, details, x0, y0, w0, h0 in left_boxes:
        draw_step_box(
            ax,
            x0,
            y0,
            w0,
            h0,
            letter=letter,
            title=title,
            details=details,
            facecolor=BLUE_LIGHT,
            edgecolor=BLUE,
            title_size=4.8,
            detail_size=4.15,
        )

    validation_boxes = [
        ("G", "Held-out TCGA", [f"(n = {fmt_int(tcga['n_heldout'])})"], 0.55, 0.585, 0.17, 0.052),
        ("H", "Primary external", [f"(n = {fmt_int(primary['n_samples'])})"], 0.755, 0.585, 0.17, 0.052),
        ("I", ["Metastatic", "external"], [f"(n = {fmt_int(metastatic['n_samples'])})"], 0.55, 0.505, 0.17, 0.052),
        ("J", "FFPE external", [f"(n = {fmt_int(ffpe['n_samples'])})"], 0.755, 0.505, 0.17, 0.052),
        ("K", "EPIC external", [f"(n = {fmt_int(epic['n_samples'])})"], 0.55, 0.425, 0.17, 0.052),
        ("L", ["Comparator", "benchmark"], ["MethyDeep"], 0.755, 0.425, 0.17, 0.052),
    ]
    for letter, title, details, x0, y0, w0, h0 in validation_boxes:
        draw_step_box(
            ax,
            x0,
            y0,
            w0,
            h0,
            letter=letter,
            title=title,
            details=details,
            facecolor=TEAL_LIGHT,
            edgecolor=TEAL,
            title_size=4.75,
            detail_size=4.2,
        )

    interpretation_boxes = [
        ("M", ["Failure pattern", "analysis"], ["error structure"], 0.245, 0.125, 0.21, 0.055),
        ("N", ["Biological context", "of probe panel"], [], 0.545, 0.125, 0.21, 0.055),
    ]
    for letter, title, details, x0, y0, w0, h0 in interpretation_boxes:
        draw_step_box(
            ax,
            x0,
            y0,
            w0,
            h0,
            letter=letter,
            title=title,
            details=details,
            facecolor=PURPLE_LIGHT,
            edgecolor=PURPLE,
            title_size=4.85,
            detail_size=4.25,
        )

    # Minimal routing: stage-to-stage arrows only; box letters carry the within-stage order.
    draw_line_arrow(ax, (0.475, 0.505), (0.525, 0.505))
    draw_line_arrow(ax, (0.74, 0.315), (0.50, 0.25))

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
    tcga = load_tcga_summary(Path(args.cohort_counts_tsv))
    harmonization = load_harmonization_summary(Path(args.cohort_harmonization_summary_json))
    pipeline = load_selected_pipeline(Path(args.selected_pipeline_json))
    screen = load_candidate_screen_summary(Path(args.candidate_metrics_tsv))
    external_design = load_external_design(Path(args.external_design_summary_json))
    primary = load_external_summary(Path(args.external_primary_summary_json))
    metastatic = load_external_summary(Path(args.external_metastatic_summary_json))
    ffpe = load_external_summary(Path(args.external_ffpe_summary_json))
    epic = load_external_summary(Path(args.external_epic_summary_json))
    stage2_folds = int(
        active.get("selected_pipeline", {})
        .get("stage2_model_metrics", {})
        .get("n_folds", 100)
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_paths = [out_dir / f"{args.out_stem}.{fmt}" for fmt in args.formats]
    plot_figure(
        tcga=tcga,
        harmonization=harmonization,
        pipeline=pipeline,
        primary=primary,
        metastatic=metastatic,
        ffpe=ffpe,
        epic=epic,
        screen=screen,
        external_design=external_design,
        stage2_folds=stage2_folds,
        out_paths=out_paths,
    )

    plot_data_path = out_dir / f"{args.out_stem}_plot_data.json"
    summary = {
        "active_locked_run_status": active.get("status"),
        "locked_run_name": active.get("locked_run_name"),
        "tcga": tcga,
        "cohort_harmonization": harmonization,
        "selected_pipeline": pipeline,
        "candidate_screen": screen,
        "stage2_folds": stage2_folds,
        "external_primary": primary,
        "external_stress_metastatic": metastatic,
        "external_stress_ffpe": ffpe,
        "external_stress_epic": epic,
        "external_design": external_design,
        "outputs": [str(p) for p in out_paths] + [str(plot_data_path)],
    }
    plot_data_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
