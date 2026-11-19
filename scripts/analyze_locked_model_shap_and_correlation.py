#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", tempfile.mkdtemp(prefix="mplconfig_"))
from shap import TreeExplainer


LOCKBOX_ID_COLS = ["sample_id", "project_id"]
EXTERNAL_ID_COLS = ["gse_id", "sample_id", "tcga_label", "target_bucket"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-bundle", required=True)
    parser.add_argument("--lockbox-panel-tsv", required=True)
    parser.add_argument("--external-panel-tsv", required=True)
    parser.add_argument("--external-predictions-tsv", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--top-n-pairs", type=int, default=20)
    parser.add_argument("--top-n-classes", type=int, default=3)
    parser.add_argument("--approximate-shap", dest="approximate_shap", action="store_true")
    parser.add_argument("--exact-shap", dest="approximate_shap", action="store_false")
    parser.set_defaults(approximate_shap=True)
    return parser.parse_args()


def _ensure_numeric(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = df.copy()
    for column in columns:
        out[column] = pd.to_numeric(out[column], errors="coerce")
    return out


def _class_center(df: pd.DataFrame, feature_cols: list[str], label_col: str) -> pd.DataFrame:
    centered = df[feature_cols].copy()
    grouped = df[[label_col] + feature_cols].groupby(label_col, dropna=False)
    for feature in feature_cols:
        centered[feature] = df[feature] - grouped[feature].transform("mean")
    return centered


def _upper_triangle_long(corr_df: pd.DataFrame) -> pd.DataFrame:
    pairs: list[dict[str, object]] = []
    cols = corr_df.columns.tolist()
    values = corr_df.to_numpy(dtype=float)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            corr = values[i, j]
            pairs.append(
                {
                    "probe_a": cols[i],
                    "probe_b": cols[j],
                    "spearman_r": corr,
                    "abs_spearman_r": abs(corr),
                }
            )
    return pd.DataFrame(pairs).sort_values(
        ["abs_spearman_r", "probe_a", "probe_b"], ascending=[False, True, True]
    )


def _correlation_outputs(
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    dataset_name: str,
    out_dir: Path,
    top_n_pairs: int,
) -> dict[str, object]:
    raw_corr = df[feature_cols].corr(method="spearman")
    raw_corr.to_csv(out_dir / f"{dataset_name}_probe_spearman_matrix.tsv", sep="\t")

    raw_pairs = _upper_triangle_long(raw_corr)
    raw_pairs.to_csv(
        out_dir / f"{dataset_name}_probe_spearman_pairs.tsv", sep="\t", index=False
    )

    centered = _class_center(df, feature_cols, label_col)
    centered_corr = centered.corr(method="spearman")
    centered_corr.to_csv(
        out_dir / f"{dataset_name}_probe_centered_spearman_matrix.tsv", sep="\t"
    )

    centered_pairs = _upper_triangle_long(centered_corr)
    centered_pairs.to_csv(
        out_dir / f"{dataset_name}_probe_centered_spearman_pairs.tsv",
        sep="\t",
        index=False,
    )

    raw_top = raw_pairs.head(top_n_pairs).copy()
    centered_top = centered_pairs.head(top_n_pairs).copy()
    raw_top.to_csv(
        out_dir / f"{dataset_name}_top_probe_spearman_pairs.tsv", sep="\t", index=False
    )
    centered_top.to_csv(
        out_dir / f"{dataset_name}_top_probe_centered_spearman_pairs.tsv",
        sep="\t",
        index=False,
    )

    return {
        "dataset": dataset_name,
        "n_samples": int(df.shape[0]),
        "n_probes": int(len(feature_cols)),
        "max_abs_raw_spearman": float(raw_pairs["abs_spearman_r"].max()),
        "max_abs_centered_spearman": float(centered_pairs["abs_spearman_r"].max()),
        "n_raw_pairs_abs_ge_0_70": int((raw_pairs["abs_spearman_r"] >= 0.70).sum()),
        "n_raw_pairs_abs_ge_0_50": int((raw_pairs["abs_spearman_r"] >= 0.50).sum()),
        "n_centered_pairs_abs_ge_0_70": int(
            (centered_pairs["abs_spearman_r"] >= 0.70).sum()
        ),
        "n_centered_pairs_abs_ge_0_50": int(
            (centered_pairs["abs_spearman_r"] >= 0.50).sum()
        ),
        "top_raw_pairs": raw_top.to_dict(orient="records"),
        "top_centered_pairs": centered_top.to_dict(orient="records"),
    }


def _class_aligned_shap(
    shap_values: np.ndarray,
    class_to_idx: dict[str, int],
    labels: pd.Series,
) -> np.ndarray:
    idx = labels.map(class_to_idx)
    if idx.isna().any():
        missing = sorted(labels[idx.isna()].dropna().unique().tolist())
        raise ValueError(f"Missing class indices for labels: {missing}")
    idx_array = idx.to_numpy(dtype=int)
    sample_axis = np.arange(shap_values.shape[0], dtype=int)
    return shap_values[sample_axis, :, idx_array]


def _summarize_shap_dataset(
    dataset_name: str,
    df: pd.DataFrame,
    feature_cols: list[str],
    label_col: str,
    model,
    class_order: list[str],
    out_dir: Path,
    top_n_classes: int,
    approximate_shap: bool,
) -> dict[str, object]:
    X = df[feature_cols]
    y_true = df[label_col].astype(str)
    y_pred = pd.Series(model.predict(X), index=df.index, dtype=str)
    is_correct = y_true.eq(y_pred)

    X_imputed = model.named_steps["imputer"].transform(X)
    clf = model.named_steps["clf"]
    explainer = TreeExplainer(clf)
    shap_values = np.asarray(
        explainer.shap_values(
            X_imputed,
            approximate=approximate_shap,
            check_additivity=False,
        ),
        dtype=float,
    )
    class_to_idx = {str(c): i for i, c in enumerate(class_order)}
    aligned = _class_aligned_shap(shap_values, class_to_idx, y_true)

    aligned_abs = np.abs(aligned)
    correct_mask = is_correct.to_numpy(dtype=bool)
    aligned_correct = aligned[correct_mask]
    aligned_abs_correct = aligned_abs[correct_mask]

    probe_summary = pd.DataFrame(
        {
            "probe_id": feature_cols,
            "dataset": dataset_name,
            "mean_abs_trueclass_shap_all": aligned_abs.mean(axis=0),
            "mean_signed_trueclass_shap_all": aligned.mean(axis=0),
            "mean_abs_trueclass_shap_correct": (
                aligned_abs_correct.mean(axis=0) if correct_mask.any() else np.nan
            ),
            "mean_signed_trueclass_shap_correct": (
                aligned_correct.mean(axis=0) if correct_mask.any() else np.nan
            ),
        }
    ).sort_values(
        ["mean_abs_trueclass_shap_correct", "mean_abs_trueclass_shap_all", "probe_id"],
        ascending=[False, False, True],
        na_position="last",
    )

    class_rows: list[dict[str, object]] = []
    for cls in class_order:
        cls_mask = y_true.eq(cls).to_numpy()
        if not cls_mask.any():
            continue
        cls_correct_mask = cls_mask & correct_mask
        cls_idx = class_to_idx[cls]
        cls_signed = shap_values[cls_mask, :, cls_idx]
        cls_abs = np.abs(cls_signed)
        if cls_correct_mask.any():
            cls_correct_signed = shap_values[cls_correct_mask, :, cls_idx]
            cls_correct_abs = np.abs(cls_correct_signed)
        else:
            cls_correct_signed = None
            cls_correct_abs = None
        for feat_idx, probe in enumerate(feature_cols):
            class_rows.append(
                {
                    "dataset": dataset_name,
                    "true_class": cls,
                    "probe_id": probe,
                    "n_true_class_samples": int(cls_mask.sum()),
                    "n_correct_samples": int(cls_correct_mask.sum()),
                    "mean_abs_trueclass_shap_all": float(cls_abs[:, feat_idx].mean()),
                    "mean_signed_trueclass_shap_all": float(
                        cls_signed[:, feat_idx].mean()
                    ),
                    "mean_abs_trueclass_shap_correct": float(
                        cls_correct_abs[:, feat_idx].mean()
                    )
                    if cls_correct_abs is not None and cls_correct_abs.shape[0] > 0
                    else np.nan,
                    "mean_signed_trueclass_shap_correct": float(
                        cls_correct_signed[:, feat_idx].mean()
                    )
                    if cls_correct_signed is not None and cls_correct_signed.shape[0] > 0
                    else np.nan,
                }
            )

    class_df = pd.DataFrame(class_rows)
    class_df.to_csv(out_dir / f"{dataset_name}_probe_trueclass_shap_usage.tsv", sep="\t", index=False)

    top_class_rows: list[dict[str, object]] = []
    for probe in feature_cols:
        subset = class_df[class_df["probe_id"] == probe].copy()
        subset = subset.sort_values(
            ["mean_abs_trueclass_shap_correct", "mean_abs_trueclass_shap_all", "true_class"],
            ascending=[False, False, True],
            na_position="last",
        ).head(top_n_classes)
        probe_row = {
            "probe_id": probe,
            "dataset": dataset_name,
        }
        for rank, (_, row) in enumerate(subset.iterrows(), start=1):
            probe_row[f"top_true_class_{rank}"] = row["true_class"]
            probe_row[f"top_true_class_{rank}_mean_abs_shap_correct"] = row[
                "mean_abs_trueclass_shap_correct"
            ]
            probe_row[f"top_true_class_{rank}_mean_signed_shap_correct"] = row[
                "mean_signed_trueclass_shap_correct"
            ]
        top_class_rows.append(probe_row)

    top_class_df = pd.DataFrame(top_class_rows)
    probe_summary = probe_summary.merge(top_class_df, on=["probe_id", "dataset"], how="left")
    probe_summary.to_csv(out_dir / f"{dataset_name}_probe_shap_summary.tsv", sep="\t", index=False)

    return {
        "dataset": dataset_name,
        "n_samples": int(df.shape[0]),
        "n_correct": int(is_correct.sum()),
        "accuracy": float(is_correct.mean()),
        "approximate_shap": bool(approximate_shap),
        "top_probes": probe_summary.head(10).to_dict(orient="records"),
    }


def _load_external_dataset(
    panel_tsv: str,
    predictions_tsv: str,
    probe_cols: list[str],
) -> pd.DataFrame:
    panel = pd.read_csv(panel_tsv, sep="\t")
    panel = _ensure_numeric(panel, probe_cols)
    pred = pd.read_csv(predictions_tsv, sep="\t")
    keep_pred_cols = ["gse_id", "sample_id", "y_pred", "is_correct"]
    pred = pred[keep_pred_cols].drop_duplicates()
    merged = panel.merge(pred, on=["gse_id", "sample_id"], how="inner", validate="one_to_one")
    return merged


def _write_summary(
    out_path: Path,
    corr_summaries: list[dict[str, object]],
    shap_summaries: list[dict[str, object]],
) -> None:
    lines: list[str] = []
    lines.append("# Locked Model Correlation + SHAP Summary")
    lines.append("")
    lines.append("## Correlation Sanity Check")
    lines.append("")
    for summary in corr_summaries:
        lines.append(f"### {summary['dataset']}")
        lines.append(f"- Samples: `{summary['n_samples']}`")
        lines.append(f"- Probes: `{summary['n_probes']}`")
        lines.append(f"- Max abs raw Spearman: `{summary['max_abs_raw_spearman']:.3f}`")
        lines.append(
            f"- Max abs class-centered Spearman: `{summary['max_abs_centered_spearman']:.3f}`"
        )
        lines.append(f"- Raw pairs with abs rho >= 0.70: `{summary['n_raw_pairs_abs_ge_0_70']}`")
        lines.append(
            f"- Centered pairs with abs rho >= 0.70: `{summary['n_centered_pairs_abs_ge_0_70']}`"
        )
        top_raw = summary["top_raw_pairs"][0] if summary["top_raw_pairs"] else None
        top_centered = summary["top_centered_pairs"][0] if summary["top_centered_pairs"] else None
        if top_raw:
            lines.append(
                f"- Top raw pair: `{top_raw['probe_a']}` vs `{top_raw['probe_b']}` (`rho={top_raw['spearman_r']:.3f}`)"
            )
        if top_centered:
            lines.append(
                f"- Top centered pair: `{top_centered['probe_a']}` vs `{top_centered['probe_b']}` (`rho={top_centered['spearman_r']:.3f}`)"
            )
        lines.append("")
    lines.append("## SHAP Usage Summary")
    lines.append("")
    for summary in shap_summaries:
        lines.append(f"### {summary['dataset']}")
        lines.append(f"- Samples: `{summary['n_samples']}`")
        lines.append(f"- Accuracy: `{summary['accuracy']:.3f}`")
        lines.append(f"- Correct predictions: `{summary['n_correct']}`")
        lines.append(f"- Approximate SHAP: `{summary['approximate_shap']}`")
        top = summary["top_probes"][:5]
        if top:
            lines.append("- Top probes by mean abs true-class SHAP on correct samples:")
            for row in top:
                lines.append(
                    f"  - `{row['probe_id']}`: `{row['mean_abs_trueclass_shap_correct']:.4f}`; "
                    f"top class `{row.get('top_true_class_1', 'NA')}`"
                )
        lines.append("")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    bundle = joblib.load(args.model_bundle)
    model = bundle["model"]
    probe_cols = [str(x) for x in bundle["probe_ids"]]
    class_order = [str(x) for x in bundle["class_order"]]

    lockbox = pd.read_csv(args.lockbox_panel_tsv, sep="\t")
    lockbox = _ensure_numeric(lockbox, probe_cols)

    external = _load_external_dataset(
        panel_tsv=args.external_panel_tsv,
        predictions_tsv=args.external_predictions_tsv,
        probe_cols=probe_cols,
    )

    corr_summaries = [
        _correlation_outputs(
            df=lockbox,
            feature_cols=probe_cols,
            label_col="project_id",
            dataset_name="lockbox",
            out_dir=out_dir,
            top_n_pairs=args.top_n_pairs,
        ),
        _correlation_outputs(
            df=external,
            feature_cols=probe_cols,
            label_col="tcga_label",
            dataset_name="external_main",
            out_dir=out_dir,
            top_n_pairs=args.top_n_pairs,
        ),
    ]

    shap_summaries = [
        _summarize_shap_dataset(
            dataset_name="lockbox",
            df=lockbox,
            feature_cols=probe_cols,
            label_col="project_id",
            model=model,
            class_order=class_order,
            out_dir=out_dir,
            top_n_classes=args.top_n_classes,
            approximate_shap=args.approximate_shap,
        ),
        _summarize_shap_dataset(
            dataset_name="external_main",
            df=external,
            feature_cols=probe_cols,
            label_col="tcga_label",
            model=model,
            class_order=class_order,
            out_dir=out_dir,
            top_n_classes=args.top_n_classes,
            approximate_shap=args.approximate_shap,
        ),
    ]

    merged_probe = pd.read_csv(out_dir / "lockbox_probe_shap_summary.tsv", sep="\t").merge(
        pd.read_csv(out_dir / "external_main_probe_shap_summary.tsv", sep="\t"),
        on="probe_id",
        how="outer",
        suffixes=("_lockbox", "_external_main"),
    )
    merged_probe.to_csv(out_dir / "panel_probe_shap_usage.tsv", sep="\t", index=False)

    summary_json = {
        "correlation": corr_summaries,
        "shap": shap_summaries,
    }
    (out_dir / "locked_model_shap_and_correlation_summary.json").write_text(
        json.dumps(summary_json, indent=2),
        encoding="utf-8",
    )
    _write_summary(
        out_path=out_dir / "locked_model_shap_and_correlation_summary.md",
        corr_summaries=corr_summaries,
        shap_summaries=shap_summaries,
    )


if __name__ == "__main__":
    main()
