#!/usr/bin/env python3
"""Run same-patient normal-adjacent substitution through the locked MiniCpG model."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-bundle", required=True, type=Path)
    parser.add_argument("--lockbox-tumor-matrix", required=True, type=Path)
    parser.add_argument("--lockbox-manifest", required=True, type=Path)
    parser.add_argument("--normal-matrix", required=True, type=Path)
    parser.add_argument("--out-dir", required=True, type=Path)
    return parser.parse_args()


def predict_frame(model, class_order: list[str], x: pd.DataFrame) -> pd.DataFrame:
    pred = model.predict(x)
    out = pd.DataFrame({"pred_label": pred})
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)
        out["pred_prob"] = np.max(proba, axis=1)
        for idx, label in enumerate(class_order):
            out[f"prob_{label}"] = proba[:, idx]
    return out


def summarize_predictions(df: pd.DataFrame, true_col: str, pred_col: str) -> dict[str, float | int]:
    y_true = df[true_col].to_numpy()
    y_pred = df[pred_col].to_numpy()
    present_labels = sorted(pd.Series(y_true).unique())
    return {
        "n": int(len(df)),
        "n_classes": int(pd.Series(y_true).nunique()),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1_present_labels": float(
            f1_score(
                y_true,
                y_pred,
                labels=present_labels,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_f1_true_pred_union": float(
            f1_score(y_true, y_pred, average="macro", zero_division=0)
        ),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    with args.model_bundle.open("rb") as handle:
        bundle = pickle.load(handle)
    model = bundle["model"]
    probe_ids = list(bundle["probe_ids"])
    class_order = list(bundle["class_order"])

    tumor = pd.read_csv(args.lockbox_tumor_matrix, sep="\t")
    manifest = pd.read_csv(args.lockbox_manifest, sep="\t")
    normals = pd.read_csv(args.normal_matrix, sep="\t")

    missing_tumor = sorted(set(probe_ids) - set(tumor.columns))
    missing_normal = sorted(set(probe_ids) - set(normals.columns))
    if missing_tumor or missing_normal:
        raise SystemExit(
            "Missing locked probes. "
            f"tumor_missing={missing_tumor}; normal_missing={missing_normal}"
        )

    tumor_meta = tumor.merge(
        manifest[["sample_id", "patient_id", "project_id"]],
        on=["sample_id", "project_id"],
        how="left",
        validate="one_to_one",
    )
    tumor_meta["patient_id_from_sample"] = tumor_meta["sample_id"].astype(str).str[:12]
    tumor_meta["patient_id"] = tumor_meta["patient_id"].fillna(
        tumor_meta["patient_id_from_sample"]
    )
    if tumor_meta["patient_id"].isna().any():
        raise SystemExit("Some lockbox tumor rows could not be assigned a patient_id.")

    normals = normals[normals["sample_type_code"].astype(str).eq("11")].copy()
    normals = normals.sort_values(["patient_id", "sample_id"]).drop_duplicates(
        subset=["patient_id"], keep="first"
    )

    keep_cols = [
        "sample_id",
        "patient_id",
        "project_id",
        *probe_ids,
    ]
    paired = tumor_meta[keep_cols].merge(
        normals[["sample_id", "patient_id", "project_id", *probe_ids]],
        on="patient_id",
        how="inner",
        suffixes=("_tumor", "_normal"),
        validate="one_to_one",
    )
    paired = paired[paired["project_id_tumor"].eq(paired["project_id_normal"])].copy()
    if paired.empty:
        raise SystemExit("No matched same-project lockbox tumor/normal-adjacent patient pairs found.")

    x_tumor = paired[[f"{probe}_tumor" for probe in probe_ids]].copy()
    x_tumor.columns = probe_ids
    x_normal_sub = paired[[f"{probe}_normal" for probe in probe_ids]].copy()
    x_normal_sub.columns = probe_ids

    tumor_pred = predict_frame(model, class_order, x_tumor).add_prefix("tumor_")
    normal_pred = predict_frame(model, class_order, x_normal_sub).add_prefix("normal_sub_")

    result = pd.concat(
        [
            paired[
                [
                    "patient_id",
                    "sample_id_tumor",
                    "sample_id_normal",
                    "project_id_tumor",
                    "project_id_normal",
                ]
            ]
            .rename(columns={"project_id_tumor": "true_label"})
            .reset_index(drop=True),
            tumor_pred.reset_index(drop=True),
            normal_pred.reset_index(drop=True),
        ],
        axis=1,
    )
    result["tumor_correct"] = result["tumor_pred_label"].eq(result["true_label"])
    result["normal_sub_correct"] = result["normal_sub_pred_label"].eq(result["true_label"])
    result["prediction_changed"] = ~result["tumor_pred_label"].eq(result["normal_sub_pred_label"])

    sample_path = args.out_dir / "matched_normal_substitution_predictions.tsv"
    result.to_csv(sample_path, sep="\t", index=False)

    summary_rows = []
    for condition, pred_col in [
        ("original_heldout_tumor", "tumor_pred_label"),
        ("same_patient_normal_substitution", "normal_sub_pred_label"),
    ]:
        row = {"condition": condition}
        row.update(summarize_predictions(result, "true_label", pred_col))
        summary_rows.append(row)
    summary = pd.DataFrame(summary_rows)
    summary_path = args.out_dir / "matched_normal_substitution_metric_summary.tsv"
    summary.to_csv(summary_path, sep="\t", index=False)

    by_class = []
    for label, group in result.groupby("true_label", sort=True):
        row = {
            "true_label": label,
            "n_pairs": int(len(group)),
            "tumor_correct": int(group["tumor_correct"].sum()),
            "tumor_accuracy": float(group["tumor_correct"].mean()),
            "normal_sub_correct": int(group["normal_sub_correct"].sum()),
            "normal_sub_accuracy": float(group["normal_sub_correct"].mean()),
            "prediction_changed": int(group["prediction_changed"].sum()),
            "prediction_changed_fraction": float(group["prediction_changed"].mean()),
        }
        by_class.append(row)
    by_class_df = pd.DataFrame(by_class)
    by_class_path = args.out_dir / "matched_normal_substitution_by_class.tsv"
    by_class_df.to_csv(by_class_path, sep="\t", index=False)

    confusion = pd.crosstab(
        result["true_label"],
        result["normal_sub_pred_label"],
        rownames=["true_label"],
        colnames=["normal_sub_pred_label"],
        dropna=False,
    )
    confusion_path = args.out_dir / "matched_normal_substitution_confusion_matrix.tsv"
    confusion.to_csv(confusion_path, sep="\t")

    metadata = {
        "model_bundle": str(args.model_bundle),
        "lockbox_tumor_matrix": str(args.lockbox_tumor_matrix),
        "lockbox_manifest": str(args.lockbox_manifest),
        "normal_matrix": str(args.normal_matrix),
        "probe_count": len(probe_ids),
        "n_matched_pairs": int(len(result)),
        "n_classes": int(result["true_label"].nunique()),
        "outputs": {
            "sample_predictions": str(sample_path),
            "metric_summary": str(summary_path),
            "by_class": str(by_class_path),
            "confusion_matrix": str(confusion_path),
        },
    }
    (args.out_dir / "matched_normal_substitution_manifest.json").write_text(
        json.dumps(metadata, indent=2) + "\n"
    )

    tumor_acc = summary.loc[
        summary["condition"].eq("original_heldout_tumor"), "accuracy"
    ].iloc[0]
    normal_acc = summary.loc[
        summary["condition"].eq("same_patient_normal_substitution"), "accuracy"
    ].iloc[0]
    md = [
        "# Matched Normal-Adjacent Substitution Diagnostic",
        "",
        f"- Matched same-patient pairs: `{len(result)}` across `{result['true_label'].nunique()}` TCGA classes.",
        f"- Original held-out tumor accuracy on the paired subset: `{tumor_acc:.4f}`.",
        f"- Same-patient normal-substituted accuracy on the paired subset: `{normal_acc:.4f}`.",
        f"- Predictions changed after substitution in `{int(result['prediction_changed'].sum())}/{len(result)}` pairs (`{result['prediction_changed'].mean():.4f}`).",
        "",
        "Interpretation: the normal-substitution result estimates how much same-tissue lineage signal remains when tumor beta values are replaced by patient-matched normal-adjacent beta values at the locked 30 CpGs.",
        "",
        "## Outputs",
        f"- `{sample_path}`",
        f"- `{summary_path}`",
        f"- `{by_class_path}`",
        f"- `{confusion_path}`",
    ]
    (args.out_dir / "matched_normal_substitution_summary.md").write_text("\n".join(md) + "\n")

    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
