# MiniCpG

MiniCpG assigns tumor types from a compact panel of 30 DNA methylation probes. The saved tissue model predicts 27 modeled classes using an ExtraTrees classifier and training-set median imputation.

**Revision branch — not a published release.** This branch is being aligned with the revised CSBJ manuscript. The final-paper analysis workflows, trained model, examples and revised aggregate tables are packaged here; see [revision status](docs/revision_status.md) for validation scope and remaining provenance limits. The published `v0.1.0` tag remains the original July 2026 snapshot.

## Predict with the saved model

From the repository root, using [uv](https://docs.astral.sh/uv/):

```bash
uv sync --locked
uv run --locked minicpg-predict \
  --input examples/example_input_30cpg.tsv \
  --output outputs/example_predictions.tsv \
  --include-probabilities
```

The lockfile specifies the tested environment, including scikit-learn 1.7.2, which matches the saved model. Python 3.11 is the default; the package supports Python 3.11–3.12. The model is distributed in this repository, separately from the Python wheel. Use `--model /path/to/minicpg_k30_model_bundle.pkl.gz` when running outside the repository root. Load only trusted pickle files.

Input is a TSV or CSV with one specimen per row, an optional unique `sample_id`, and probe columns matching `model/feature_order.tsv`. Without IDs, the reader assigns `row_1`, `row_2`, etc. Extra columns are ignored. The example is synthetic and does not represent a study specimen.

- At least 25 of the 30 panel probes must have finite beta values in [0, 1].
- Absent probe columns, blank/nonnumeric cells, infinities and out-of-range values count as missing.
- Eligible rows use the stored training-set medians for missing values. Nothing is fitted on the input specimens.
- Every row is retained. Rows below the coverage threshold have `status=insufficient_coverage` and blank prediction/probability fields.
- Predictions use the original class probabilities and raw argmax. ESCA/STAD predictions are retained; no class is removed or reassigned.

Output adds measured/missing probe counts, status, predicted label and maximum probability; `--include-probabilities` adds all modeled class probabilities. Internal TCGA identifiers retain the historical merge mapping; see `model/class_labels.tsv`.

## Reproduction and development

```bash
uv sync --locked --extra analysis
uv run --locked --extra analysis pytest
uv run --locked python examples/run_synthetic_evaluation.py --out-dir outputs/synthetic_evaluation
```

[Reproduction workflow](docs/workflows.md) provides the current analysis entry points, exact input contracts and examples. [Analysis policies](docs/analysis_policies.md) records the cohort, missingness, label-space and statistical conventions to preserve. Optional MethyDeep execution requires its separately obtained model and TensorFlow environment; it is not part of the inference install.

[Revised supplementary results](results/revision_20261119/README.md) contain all 25 tables as 71 aggregate TSV blocks. Superseded scripts and historical results have been removed from this branch; the original `v0.1.0` tag remains available. No raw methylation data, specimen-level curation or predictions are distributed. See [data access](docs/data_access.md).

## Citation and license

Citation metadata are in `CITATION.cff`. Academic and non-commercial research use is governed by [LICENSE](LICENSE). Commercial use, clinical diagnostic use and paid-service use require separate written permission. The research model is not a clinical diagnostic device.
