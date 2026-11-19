# Examples

These examples use synthetic data. Run them from the repository root after following the [installation instructions](../README.md#quick-start).

## Predict with complete panel measurements

This example runs the saved MiniCpG model on specimens with all panel measurements:

```bash
uv run --locked minicpg-predict \
  --input examples/example_input_30cpg.tsv \
  --output outputs/example_predictions.tsv \
  --include-probabilities
```

Open `outputs/example_predictions.tsv` to see each specimen's prediction, coverage status and class probabilities. Use the same command with your own input file to predict new specimens.

## Predict with missing or invalid measurements

```bash
uv run --locked minicpg-predict \
  --input examples/example_missingness.tsv \
  --output outputs/example_missingness_predictions.tsv \
  --include-probabilities
```

The input contains four rows: a complete panel, 25 measured probes, 24 measured probes, and a panel with invalid cells. The first two rows receive predictions. The last two are retained with `status=insufficient_coverage` and blank prediction fields. Missing values in eligible rows are filled using the model's saved training medians.

## Try the evaluation workflow

This example is for readers reproducing the paper's performance analyses. It creates a small synthetic prediction table with toy class labels, records its input-file hashes, and calculates performance metrics and patient-group bootstrap uncertainty:

```bash
uv run --locked python examples/run_synthetic_evaluation.py \
  --out-dir outputs/synthetic_evaluation
```

Inspect `outputs/synthetic_evaluation/results/metrics.tsv` and `uncertainty.tsv`. The same directory contains class-level results, confusion counts, bootstrap draws and input/output provenance. This example demonstrates the evaluation interface; it does not run the trained MiniCpG model or reproduce a study result. Choose a new or empty output directory when repeating it.

For study data, continue with the [reproduction guide](../docs/reproduction_workflow.md).
