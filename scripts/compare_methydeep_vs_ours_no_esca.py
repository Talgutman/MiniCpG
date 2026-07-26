#!/usr/bin/env python3
"""Compare locked model vs MethyDeep without ESCA truth-penalty.

Default behavior:
- filter to `target_bucket == main`
- drop truth rows with `tcga_label == TCGA-ESCA`
- evaluate on intersection of sample keys (gse_id, sample_id)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare locked model vs MethyDeep on no-ESCA subset.")
    p.add_argument("--ours-pred-tsv", required=True)
    p.add_argument("--methydeep-pred-tsv", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--target-bucket", default="main")
    p.add_argument("--drop-truth-label", default="TCGA-ESCA")
    p.add_argument(
        "--truth-remap",
        default="",
        help=(
            "Comma-separated truth label remaps old:new applied before drop-truth-label, "
            "e.g. 'TCGA-LGG:TCGA-GBM,TCGA-STAD:TCGA-ESCA,TCGA-READ:TCGA-COAD'"
        ),
    )
    p.add_argument(
        "--ours-remap-pred-label-to-runnerup",
        default="",
        help="If set (e.g. TCGA-ESCA), remap our predictions of this label to full-model runner-up class.",
    )
    p.add_argument(
        "--fixed-classes-tsv",
        default="",
        help="Optional TSV with column tcga_label to force a fixed evaluation class axis.",
    )
    p.add_argument(
        "--exclude-fixed-classes",
        default="",
        help="Comma-separated labels to exclude from the fixed class axis (e.g. TCGA-ESCA).",
    )
    p.add_argument(
        "--remap-out-of-axis-to-runnerup",
        action="store_true",
        help="Remap any prediction not in the evaluation class axis to runner-up among allowed classes.",
    )
    p.add_argument(
        "--force-pred-axis-to-classes",
        action="store_true",
        help="Force confusion-matrix prediction axis to exactly the evaluation class axis.",
    )
    p.add_argument(
        "--matched-only",
        dest="matched_only",
        action="store_true",
        help="Evaluate only on intersection of sample keys (default).",
    )
    p.add_argument(
        "--allow-unmatched",
        dest="matched_only",
        action="store_false",
        help="Disable key intersection and evaluate each table as-is.",
    )
    p.add_argument(
        "--key-cols",
        nargs="+",
        default=["gse_id", "sample_id"],
        help="Columns used to match samples across prediction files.",
    )
    p.add_argument(
        "--require-methydeep-eligible-tsv",
        default="",
        help=(
            "Optional TSV describing the exact current MethyDeep-eligible sample set. "
            "If provided, the MethyDeep prediction table must match this key set exactly "
            "after target-bucket filtering."
        ),
    )
    p.add_argument(
        "--methydeep-max-missing-probes",
        type=int,
        default=3,
        help="Eligibility threshold used when deriving expected MethyDeep keys from a status TSV.",
    )
    p.set_defaults(matched_only=True)
    return p.parse_args()


def _read_pred(path: Path) -> pd.DataFrame:
    d = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    req = {"tcga_label", "y_pred", "target_bucket"}
    miss = req - set(d.columns)
    if miss:
        raise SystemExit(f"Missing required columns in {path}: {sorted(miss)}")
    return d


def _to_key(df: pd.DataFrame, key_cols: list[str]) -> pd.Series:
    return df[key_cols].astype(str).agg("||".join, axis=1)


def _parse_truth_remap(spec: str) -> dict[str, str]:
    out: dict[str, str] = {}
    s = str(spec or "").strip()
    if not s:
        return out
    for tok in s.split(","):
        tok = tok.strip()
        if not tok:
            continue
        if ":" not in tok:
            raise SystemExit(f"Invalid --truth-remap token (expected old:new): {tok}")
        old, new = tok.split(":", 1)
        old = old.strip()
        new = new.strip()
        if not old or not new:
            raise SystemExit(f"Invalid --truth-remap token (empty old/new): {tok}")
        out[old] = new
    return out


def _parse_csv_labels(spec: str) -> set[str]:
    out: set[str] = set()
    for tok in str(spec or "").split(","):
        t = tok.strip()
        if t:
            out.add(t)
    return out


def _read_fixed_classes(path: str, exclude: set[str]) -> list[str]:
    p = Path(path)
    if not p.exists():
        raise SystemExit(f"Missing --fixed-classes-tsv: {p}")
    d = pd.read_csv(p, sep="\t", dtype=str).fillna("")
    if "tcga_label" not in d.columns:
        raise SystemExit(f"--fixed-classes-tsv missing tcga_label column: {p}")
    classes = []
    for x in d["tcga_label"].astype(str).tolist():
        s = x.strip()
        if s and s not in exclude:
            classes.append(s)
    # preserve file order while removing duplicates
    seen: set[str] = set()
    out = []
    for c in classes:
        if c in seen:
            continue
        seen.add(c)
        out.append(c)
    if not out:
        raise SystemExit("Fixed class list is empty after exclusions.")
    return out


def _assert_unique_keys(df: pd.DataFrame, key_cols: list[str], table_name: str) -> None:
    dup_mask = df.duplicated(subset=key_cols, keep=False)
    if not bool(dup_mask.any()):
        return
    dup_rows = df.loc[dup_mask, key_cols].astype(str).copy()
    dup_rows["__key"] = dup_rows[key_cols].agg("||".join, axis=1)
    top = dup_rows["__key"].value_counts().head(5)
    raise SystemExit(
        f"Duplicate sample keys in {table_name} for key_cols={key_cols}. "
        f"Top duplicates: {top.to_dict()}"
    )


def _expected_methydeep_keys(
    path: Path,
    target_bucket: str,
    key_cols: list[str],
    max_missing_probes: int,
    truth_remap: dict[str, str],
    drop_truth_label: str,
) -> set[str]:
    d = pd.read_csv(path, sep="\t", dtype=str).fillna("")
    for c in key_cols:
        if c not in d.columns:
            raise SystemExit(f"Expected-eligibility TSV missing key column {c}: {path}")

    if "target_bucket" in d.columns:
        d = d[d["target_bucket"].astype(str) == target_bucket].copy()
    if "tcga_label" in d.columns and truth_remap:
        d["tcga_label"] = d["tcga_label"].astype(str).replace(truth_remap)
    if "tcga_label" in d.columns and drop_truth_label:
        d = d[d["tcga_label"].astype(str) != drop_truth_label].copy()

    # If this is a sample-status table, derive exact eligibility from the status fields.
    if "extraction_status" in d.columns:
        d = d[d["extraction_status"].astype(str).isin({"ok_full", "ok_partial"})].copy()
    if "n_panel_missing" in d.columns:
        miss = pd.to_numeric(d["n_panel_missing"], errors="coerce")
        d = d[miss <= int(max_missing_probes)].copy()
    elif "n_missing_features" in d.columns:
        miss = pd.to_numeric(d["n_missing_features"], errors="coerce")
        d = d[miss <= int(max_missing_probes)].copy()

    return set(_to_key(d, key_cols).tolist())


def _macro_auc_by_class(df: pd.DataFrame, classes_eval: list[str]) -> tuple[float, pd.DataFrame]:
    y_true = df["tcga_label"].astype(str).to_numpy()
    proba_cols = [c for c in df.columns if c.startswith("proba__")]
    proba = df[proba_cols].apply(pd.to_numeric, errors="coerce")
    rows: list[dict[str, object]] = []
    for c in classes_eval:
        col = f"proba__{c}"
        n_pos = int((y_true == c).sum())
        n_neg = int(len(y_true) - n_pos)
        auc = np.nan
        if col in proba.columns and n_pos > 0 and n_neg > 0:
            auc = float(roc_auc_score((y_true == c).astype(int), proba[col].to_numpy()))
        rows.append({"class": c, "auc": auc, "n_pos": n_pos, "n_neg": n_neg, "has_proba_col": int(col in proba.columns)})
    auc_df = pd.DataFrame(rows)
    macro_auc = float(pd.to_numeric(auc_df["auc"], errors="coerce").dropna().mean())
    return macro_auc, auc_df


def _eval(df: pd.DataFrame, classes_eval: list[str], model_name: str) -> tuple[dict[str, object], pd.DataFrame]:
    y_true = df["tcga_label"].astype(str).to_numpy()
    y_pred = df["y_pred"].astype(str).to_numpy()
    macro_auc, auc_df = _macro_auc_by_class(df, classes_eval)
    row = {
        "model": model_name,
        "n_samples": int(len(df)),
        "n_classes_in_truth": int(len(pd.unique(y_true))),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", labels=classes_eval, zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", labels=classes_eval, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "auroc_ovr_macro": macro_auc,
    }
    return row, auc_df


def _write_confusion(df: pd.DataFrame, classes_true: list[str], classes_pred: list[str], out_tsv: Path) -> None:
    y_true = df["tcga_label"].astype(str).to_numpy()
    y_pred = df["y_pred"].astype(str).to_numpy()
    true_cat = pd.Categorical(y_true, categories=classes_true, ordered=True)
    pred_cat = pd.Categorical(y_pred, categories=classes_pred, ordered=True)
    out = pd.crosstab(true_cat, pred_cat, dropna=False)
    out.index = classes_true
    out.columns = classes_pred
    row_sum = out.sum(axis=1)
    keep_rows = row_sum[row_sum > 0].index.tolist()
    out = out.loc[keep_rows, :]
    out.to_csv(out_tsv, sep="\t")


def _remap_pred_label_to_runnerup(df: pd.DataFrame, label: str) -> tuple[pd.DataFrame, int]:
    """Remap rows predicted as `label` to highest-probability non-`label` class.

    Uses full `proba__*` columns, so runner-up can be any model class.
    """
    if not label:
        return df, 0
    out = df.copy()
    pred = out["y_pred"].astype(str)
    mask = pred == label
    n_target = int(mask.sum())
    if n_target == 0:
        return out, 0

    proba_cols = [c for c in out.columns if c.startswith("proba__")]
    if not proba_cols:
        return out, 0
    classes_all = [c.replace("proba__", "", 1) for c in proba_cols]
    if label not in classes_all:
        return out, 0

    cls_to_idx = {c: i for i, c in enumerate(classes_all)}
    label_idx = cls_to_idx[label]
    probs = (
        out.loc[mask, proba_cols]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .to_numpy(dtype=float)
        .copy()
    )
    probs[:, label_idx] = -np.inf
    second_idx = np.argmax(probs, axis=1)
    second = np.array(classes_all, dtype=object)[second_idx]
    out.loc[mask, "y_pred"] = second
    return out, n_target


def _remap_out_of_axis_to_runnerup(df: pd.DataFrame, classes_allowed: list[str]) -> tuple[pd.DataFrame, int]:
    out = df.copy()
    pred = out["y_pred"].astype(str)
    allowed = set(classes_allowed)
    mask = ~pred.isin(allowed)
    n_target = int(mask.sum())
    if n_target == 0:
        return out, 0

    cols = [f"proba__{c}" for c in classes_allowed if f"proba__{c}" in out.columns]
    if not cols:
        raise SystemExit("Cannot remap out-of-axis predictions: no probability columns for allowed classes.")
    if len(cols) != len(classes_allowed):
        miss = [c for c in classes_allowed if f"proba__{c}" not in out.columns]
        raise SystemExit(f"Cannot remap out-of-axis predictions: missing proba columns for allowed classes: {miss[:10]}")

    probs = out.loc[mask, cols].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=float)
    idx = np.argmax(probs, axis=1)
    cls = np.array(classes_allowed, dtype=object)
    out.loc[mask, "y_pred"] = cls[idx]
    return out, n_target


def main() -> int:
    args = parse_args()
    ours_path = Path(args.ours_pred_tsv)
    meth_path = Path(args.methydeep_pred_tsv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not ours_path.exists():
        raise SystemExit(f"Missing ours predictions: {ours_path}")
    if not meth_path.exists():
        raise SystemExit(f"Missing methydeep predictions: {meth_path}")

    ours = _read_pred(ours_path)
    meth = _read_pred(meth_path)
    truth_remap = _parse_truth_remap(args.truth_remap)
    exclude_fixed = _parse_csv_labels(args.exclude_fixed_classes)

    for c in args.key_cols:
        if c not in ours.columns:
            raise SystemExit(f"Missing key column {c} in ours file.")
        if c not in meth.columns:
            raise SystemExit(f"Missing key column {c} in methydeep file.")

    ours = ours[ours["target_bucket"].astype(str) == args.target_bucket].copy()
    meth = meth[meth["target_bucket"].astype(str) == args.target_bucket].copy()

    if truth_remap:
        ours["tcga_label"] = ours["tcga_label"].astype(str).replace(truth_remap)
        meth["tcga_label"] = meth["tcga_label"].astype(str).replace(truth_remap)

    if args.drop_truth_label:
        ours = ours[ours["tcga_label"].astype(str) != args.drop_truth_label].copy()
        meth = meth[meth["tcga_label"].astype(str) != args.drop_truth_label].copy()

    ours["__key"] = _to_key(ours, args.key_cols)
    meth["__key"] = _to_key(meth, args.key_cols)
    _assert_unique_keys(ours, args.key_cols, "ours")
    _assert_unique_keys(meth, args.key_cols, "methydeep")

    eligibility_summary: dict[str, object] = {}
    if args.require_methydeep_eligible_tsv:
        expected_keys = _expected_methydeep_keys(
            Path(args.require_methydeep_eligible_tsv),
            args.target_bucket,
            args.key_cols,
            args.methydeep_max_missing_probes,
            truth_remap,
            args.drop_truth_label,
        )
        actual_keys = set(meth["__key"].tolist())
        missing_keys = sorted(expected_keys - actual_keys)
        extra_keys = sorted(actual_keys - expected_keys)
        eligibility_summary = {
            "require_methydeep_eligible_tsv": args.require_methydeep_eligible_tsv,
            "expected_methydeep_keys": int(len(expected_keys)),
            "actual_methydeep_keys": int(len(actual_keys)),
            "missing_vs_expected": int(len(missing_keys)),
            "extra_vs_expected": int(len(extra_keys)),
        }
        if missing_keys or extra_keys:
            raise SystemExit(
                "MethyDeep prediction table does not match the current eligible sample set. "
                f"expected={len(expected_keys)} actual={len(actual_keys)} "
                f"missing={len(missing_keys)} extra={len(extra_keys)} "
                f"sample_missing={missing_keys[:5]} sample_extra={extra_keys[:5]}"
            )

    ours_keys = set(ours["__key"].tolist())
    meth_keys = set(meth["__key"].tolist())
    inter_keys = ours_keys & meth_keys

    if args.matched_only:
        ours = ours[ours["__key"].isin(inter_keys)].copy()
        meth = meth[meth["__key"].isin(inter_keys)].copy()
        if ours.empty or meth.empty:
            raise SystemExit("No matched rows after key intersection.")

    remapped_n = 0
    if args.ours_remap_pred_label_to_runnerup:
        ours, remapped_n = _remap_pred_label_to_runnerup(ours, args.ours_remap_pred_label_to_runnerup)

    ours = ours.sort_values(args.key_cols, kind="mergesort").reset_index(drop=True)
    meth = meth.sort_values(args.key_cols, kind="mergesort").reset_index(drop=True)
    if args.matched_only:
        ours_key_arr = ours[args.key_cols].astype(str).agg("||".join, axis=1).to_numpy()
        meth_key_arr = meth[args.key_cols].astype(str).agg("||".join, axis=1).to_numpy()
        if len(ours_key_arr) != len(meth_key_arr) or not np.array_equal(ours_key_arr, meth_key_arr):
            raise SystemExit("Matched-only mode requires identical key ordering after sort, but keys differ.")

    label_mismatch = int((ours["tcga_label"].astype(str).to_numpy() != meth["tcga_label"].astype(str).to_numpy()).sum())
    if args.matched_only and label_mismatch > 0:
        raise SystemExit(
            f"Truth-label mismatch on {label_mismatch} matched rows between ours and methydeep tables. "
            "Refusing to score mismatched pairs."
        )

    if args.fixed_classes_tsv:
        classes_eval = _read_fixed_classes(args.fixed_classes_tsv, exclude_fixed)
    else:
        classes_eval = sorted(pd.unique(ours["tcga_label"].astype(str)))

    n_out_axis_ours = 0
    n_out_axis_meth = 0
    if args.remap_out_of_axis_to_runnerup:
        ours, n_out_axis_ours = _remap_out_of_axis_to_runnerup(ours, classes_eval)
        meth, n_out_axis_meth = _remap_out_of_axis_to_runnerup(meth, classes_eval)

    if args.force_pred_axis_to_classes:
        pred_axis = list(classes_eval)
    else:
        pred_axis = sorted(set(ours["y_pred"].astype(str).tolist()) | set(meth["y_pred"].astype(str).tolist()) | set(classes_eval))

    if args.force_pred_axis_to_classes:
        outside_ours = sorted(set(ours["y_pred"].astype(str)) - set(pred_axis))
        outside_meth = sorted(set(meth["y_pred"].astype(str)) - set(pred_axis))
        if outside_ours or outside_meth:
            raise SystemExit(
                "Predictions outside forced prediction axis remain. "
                f"ours_outside={outside_ours[:10]} meth_outside={outside_meth[:10]}"
            )
    row_ours, auc_ours = _eval(ours, classes_eval, "ours_locked_protocolB")
    row_meth, auc_meth = _eval(meth, classes_eval, "methydeep")

    comp = pd.DataFrame([row_ours, row_meth])
    comp.to_csv(out_dir / "comparison_metrics.tsv", sep="\t", index=False)

    auc_cmp = auc_ours.rename(columns={"auc": "auc_ours"}).merge(
        auc_meth.rename(columns={"auc": "auc_methydeep", "has_proba_col": "methydeep_has_proba_col"}),
        on=["class", "n_pos", "n_neg"],
        how="outer",
    )
    auc_cmp.to_csv(out_dir / "comparison_auc_by_class.tsv", sep="\t", index=False)
    _write_confusion(ours, classes_eval, pred_axis, out_dir / "confusion_matrix_ours.tsv")
    _write_confusion(meth, classes_eval, pred_axis, out_dir / "confusion_matrix_methydeep.tsv")

    summary = {
        "target_bucket": args.target_bucket,
        "drop_truth_label": args.drop_truth_label,
        "truth_remap": truth_remap,
        "ours_remap_pred_label_to_runnerup": args.ours_remap_pred_label_to_runnerup,
        "ours_n_predictions_remapped": int(remapped_n),
        "ours_n_out_of_axis_remapped": int(n_out_axis_ours),
        "methydeep_n_out_of_axis_remapped": int(n_out_axis_meth),
        "matched_only": bool(args.matched_only),
        "key_cols": args.key_cols,
        "n_ours_before_match": int(len(ours_keys)),
        "n_methydeep_before_match": int(len(meth_keys)),
        "n_intersection": int(len(inter_keys)),
        "label_mismatch_on_matched_rows": label_mismatch,
        "classes_eval": classes_eval,
        "confusion_pred_axis": pred_axis,
        "fixed_classes_tsv": args.fixed_classes_tsv,
        "methydeep_eligibility_check": eligibility_summary,
    }
    (out_dir / "comparison_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(comp.to_csv(sep="\t", index=False))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
