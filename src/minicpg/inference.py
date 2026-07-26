"""Inference helpers for the locked MiniCpG K=30 model."""

from __future__ import annotations

import gzip
import pickle
from pathlib import Path
from typing import Any

import pandas as pd


def load_model_bundle(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rb') as handle:
        bundle = pickle.load(handle)
    required = {'model', 'probe_ids', 'class_order'}
    missing = required.difference(bundle)
    if missing:
        raise ValueError(f'Model bundle is missing required keys: {sorted(missing)}')
    return bundle


def read_beta_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    sep = ',' if path.suffix.lower() == '.csv' else '\t'
    table = pd.read_csv(path, sep=sep)
    if table.empty:
        raise ValueError('Input beta table is empty')
    if 'sample_id' in table.columns:
        table = table.set_index('sample_id')
    else:
        first = table.columns[0]
        if str(first).lower() in {'sample', 'sample_name', 'id'}:
            table = table.set_index(first)
    table.index = table.index.astype(str)
    return table


def predict_beta_table(
    beta_table: pd.DataFrame,
    bundle: dict[str, Any],
    *,
    include_probabilities: bool = False,
) -> pd.DataFrame:
    probe_ids = [str(v) for v in bundle['probe_ids']]
    missing = [probe for probe in probe_ids if probe not in beta_table.columns]
    if missing:
        raise ValueError(f'Input is missing {len(missing)} required MiniCpG probes: {missing}')
    X = beta_table.loc[:, probe_ids].apply(pd.to_numeric, errors='coerce')
    model = bundle['model']
    labels = model.predict(X)
    result = pd.DataFrame({'sample_id': X.index.astype(str), 'predicted_label': labels})
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(X)
        classes = [str(v) for v in getattr(model, 'classes_', bundle.get('class_order', []))]
        result['max_probability'] = proba.max(axis=1)
        if include_probabilities:
            prob_df = pd.DataFrame(
                proba,
                columns=[f'prob_{label.replace("TCGA-", "")}' for label in classes],
            )
            result = pd.concat([result, prob_df], axis=1)
    return result


def predict_file(
    input_path: str | Path,
    model_path: str | Path,
    *,
    include_probabilities: bool = False,
) -> pd.DataFrame:
    bundle = load_model_bundle(model_path)
    beta_table = read_beta_table(input_path)
    return predict_beta_table(beta_table, bundle, include_probabilities=include_probabilities)
