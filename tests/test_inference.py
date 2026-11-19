"""Synthetic inputs only; no study specimen data are needed."""
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest

from minicpg.inference import load_model_bundle, predict_beta_table, read_beta_table

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def bundle():
    return load_model_bundle(ROOT / "model/minicpg_k30_model_bundle.pkl.gz")


def inputs(bundle, n=3):
    rng = np.random.default_rng(42)
    return pd.DataFrame(rng.uniform(0, 1, (n, 30)), columns=bundle["probe_ids"],
                        index=[f"synthetic_{i}" for i in range(n)])


def test_complete_inputs_preserve_locked_predictions_and_probabilities(bundle):
    X = inputs(bundle)
    expected = bundle["model"].predict_proba(X)
    actual = predict_beta_table(X, bundle, include_probabilities=True)
    np.testing.assert_allclose(actual.filter(like="prob_").to_numpy(), expected, rtol=0, atol=0)
    assert actual.predicted_label.tolist() == list(bundle["model"].predict(X))
    assert actual.status.eq("predicted").all()
    # Input order, extra columns and non-default row indices must not alter results.
    reordered = X[X.columns[::-1]].assign(unused="ignored")
    pd.testing.assert_frame_equal(actual, predict_beta_table(reordered, bundle, include_probabilities=True))


def test_coverage_boundary_and_training_median_imputation(bundle):
    X = inputs(bundle)
    X.iloc[0, :5] = np.nan
    X.iloc[1, :6] = np.nan
    X.iloc[2, :] = np.nan
    actual = predict_beta_table(X, bundle, include_probabilities=True)
    assert actual.n_measured_probes.tolist() == [25, 24, 0]
    assert actual.status.tolist() == ["predicted", "insufficient_coverage", "insufficient_coverage"]
    assert actual.loc[1:, "predicted_label"].isna().all()
    assert actual.loc[1:, actual.columns.str.startswith("prob_")].isna().all().all()
    model = bundle["model"]
    filled = X.iloc[[0]].to_numpy().copy()
    filled[0, :5] = model.named_steps["imputer"].statistics_[:5]
    expected = model.named_steps["clf"].predict_proba(filled)
    np.testing.assert_array_equal(actual.filter(like="prob_").iloc[[0]].to_numpy(), expected)


def test_absent_columns_equal_missing_values(bundle):
    X = inputs(bundle)
    omitted = X.drop(columns=X.columns[:5])
    X.iloc[:, :5] = np.nan
    pd.testing.assert_frame_equal(predict_beta_table(X, bundle), predict_beta_table(omitted, bundle))
    omitted = omitted.drop(columns=omitted.columns[:1])
    assert predict_beta_table(omitted, bundle).status.eq("insufficient_coverage").all()


def test_invalid_values_count_as_missing(bundle):
    X = inputs(bundle, 1).astype(object)
    X.iloc[0, :6] = ["NA", "malformed", np.inf, -np.inf, -0.01, 1.01]
    actual = predict_beta_table(X, bundle)
    assert actual.n_measured_probes.iloc[0] == 24
    assert actual.predicted_label.isna().all()
    X.iloc[0, :2] = [0, 1]
    assert predict_beta_table(X, bundle).n_measured_probes.iloc[0] == 26


def test_reject_duplicate_or_blank_identifiers(bundle):
    X = inputs(bundle, 2)
    for names in [["same", "same"], ["ok", " "]]:
        X.index = names
        with pytest.raises(ValueError, match="identifiers"):
            predict_beta_table(X, bundle)


def test_input_ids_and_duplicate_headers(tmp_path):
    path = tmp_path / "input.tsv"
    path.write_text("sample_id\tcg1\n001\tNA\nNA\t0.5\n")
    assert read_beta_table(path).index.tolist() == ["001", "NA"]
    path.write_text("sample_id\tcg1\tcg1\na\t0.1\t0.2\n")
    with pytest.raises(ValueError, match="unique"):
        read_beta_table(path)


def test_unsupported_model_environment(monkeypatch):
    monkeypatch.setattr("minicpg.inference.sklearn.__version__", "0.0")
    with pytest.raises(RuntimeError, match="1.7.2"):
        load_model_bundle("unused")


def test_cli_keeps_rejected_rows(bundle, tmp_path):
    X = inputs(bundle, 2)
    X.iloc[1, :6] = np.nan
    source, output = tmp_path / "input.tsv", tmp_path / "predictions.tsv"
    X.to_csv(source, sep="\t", index_label="sample_id")
    result = subprocess.run([sys.executable, str(ROOT / "scripts/predict_minicpg.py"),
        "--input", str(source), "--output", str(output), "--model",
        str(ROOT / "model/minicpg_k30_model_bundle.pkl.gz"), "--include-probabilities"],
        check=True, capture_output=True, text=True)
    assert "1 predicted, 1 below coverage threshold" in result.stdout
    table = pd.read_csv(output, sep="\t")
    assert len(table) == 2 and pd.isna(table.predicted_label.iloc[1])


def test_no_probe_columns_retains_rows_without_predictions(bundle):
    X = pd.DataFrame(index=["synthetic_a", "synthetic_b"])
    actual = predict_beta_table(X, bundle)
    assert len(actual) == 2
    assert actual.n_measured_probes.eq(0).all()
    assert actual.predicted_label.isna().all()


def test_development_decision_policy_is_not_silently_ignored(bundle):
    changed = dict(bundle, decision_policy={"kind": "offset"})
    with pytest.raises(ValueError, match="raw argmax"):
        predict_beta_table(inputs(bundle), changed)
