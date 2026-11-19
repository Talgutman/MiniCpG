"""Inference with the locked tissue model and its external coverage policy."""
from __future__ import annotations

import csv
import gzip
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import sklearn

MODEL_SKLEARN_VERSION = "1.7.2"
MAX_MISSING_PROBES = 5


def load_model_bundle(path: str | Path) -> dict[str, Any]:
    """Load a trusted local pickle in the model's supported sklearn environment."""
    if sklearn.__version__ != MODEL_SKLEARN_VERSION:
        raise RuntimeError(
            f"The released model requires scikit-learn {MODEL_SKLEARN_VERSION}; "
            f"found {sklearn.__version__}. Run uv sync --locked."
        )
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rb") as handle:
        bundle = pickle.load(handle)
    _validate_bundle(bundle)
    return bundle


def _validate_bundle(bundle: dict[str, Any]) -> None:
    required = {"model", "probe_ids", "class_order", "decision_policy"}
    if not isinstance(bundle, dict) or required.difference(bundle):
        raise ValueError(f"Model bundle must contain {sorted(required)}")
    probes = list(bundle["probe_ids"])
    if len(probes) != 30 or len(set(probes)) != len(probes):
        raise ValueError("Expected the locked panel of 30 distinct probes")
    if bundle["decision_policy"] is not None:
        raise ValueError("The locked tissue model must use raw argmax (null decision_policy)")
    model = bundle["model"]
    if hasattr(model, "feature_names_in_") and list(model.feature_names_in_) != probes:
        raise ValueError("Stored feature names do not match the bundle probe order")
    classes = [str(v) for v in model.classes_]
    if len(classes) != 27 or set(classes) != set(map(str, bundle["class_order"])):
        raise ValueError("Model classes do not match the locked 27-class label space")


def read_beta_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    sep = "," if path.suffix.lower() == ".csv" else "\t"
    with path.open(encoding="utf-8-sig", newline="") as handle:
        header = next(csv.reader(handle, delimiter=sep), [])
    if not header or len(set(header)) != len(header):
        raise ValueError("Input must have a nonempty header with unique column names")
    # Preserve IDs such as 'NA' and '001'; convert only probe measurements below.
    table = pd.read_csv(path, sep=sep, dtype=str, keep_default_na=False)
    if table.empty:
        raise ValueError("Input beta table is empty")
    if "sample_id" in table.columns:
        table = table.set_index("sample_id")
    elif str(table.columns[0]).lower() in {"sample", "sample_name", "id"}:
        table = table.set_index(table.columns[0])
    else:
        table.index = pd.Index([f"row_{i + 1}" for i in range(len(table))])
    table.index = table.index.astype(str)
    return table


def predict_beta_table(
    beta_table: pd.DataFrame,
    bundle: dict[str, Any],
    *,
    include_probabilities: bool = False,
) -> pd.DataFrame:
    """Retain every input row; emit no prediction below 25 valid panel probes.

    Absent columns, nonnumeric/nonfinite values and values outside [0, 1]
    are missing. Eligible rows use only the saved training-median imputer.
    """
    _validate_bundle(bundle)
    if len(beta_table) == 0 or not beta_table.columns.is_unique:
        raise ValueError("Input must be nonempty with unique columns")
    ids = pd.Index(beta_table.index.astype(str))
    if not ids.is_unique or any(not x.strip() for x in ids):
        raise ValueError("Sample identifiers must be unique and nonempty")
    probes = list(map(str, bundle["probe_ids"]))
    raw = beta_table.reindex(columns=probes)
    X = raw.apply(pd.to_numeric, errors="coerce").astype(float)
    valid = np.isfinite(X) & X.ge(0) & X.le(1)
    X = X.where(valid)
    measured = valid.sum(axis=1).to_numpy(dtype=int)
    eligible = measured >= len(probes) - MAX_MISSING_PROBES
    result = pd.DataFrame({
        "sample_id": ids,
        "n_measured_probes": measured,
        "n_missing_probes": len(probes) - measured,
        "status": np.where(eligible, "predicted", "insufficient_coverage"),
        "predicted_label": pd.Series([None] * len(X), dtype=object),
        "max_probability": np.nan,
    })
    model = bundle["model"]
    classes = np.asarray(model.classes_, dtype=str)
    if include_probabilities:
        probability_columns = [f"prob_{label.removeprefix('TCGA-')}" for label in classes]
        for column in probability_columns:
            result[column] = np.nan
    if eligible.any():
        model_input = X.iloc[np.flatnonzero(eligible)]
        if not hasattr(model, "feature_names_in_"):
            model_input = model_input.to_numpy()
        proba = model.predict_proba(model_input)
        result.loc[eligible, "predicted_label"] = classes[np.argmax(proba, axis=1)]
        result.loc[eligible, "max_probability"] = proba.max(axis=1)
        if include_probabilities:
            result.loc[eligible, probability_columns] = proba
    return result


def predict_file(
    input_path: str | Path,
    model_path: str | Path,
    *,
    include_probabilities: bool = False,
) -> pd.DataFrame:
    return predict_beta_table(
        read_beta_table(input_path), load_model_bundle(model_path),
        include_probabilities=include_probabilities,
    )
