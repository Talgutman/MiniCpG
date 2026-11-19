# Synthetic examples

Run from the repository root after `uv sync --locked`:

```bash
uv run --locked minicpg-predict --input examples/example_input_30cpg.tsv \
  --output outputs/example_predictions.tsv --include-probabilities
uv run --locked minicpg-predict --input examples/example_missingness.tsv \
  --output outputs/example_missingness_predictions.tsv --include-probabilities
uv run --locked python examples/run_synthetic_evaluation.py \
  --out-dir outputs/synthetic_evaluation
```

The first input demonstrates complete panel inference. The missingness example includes complete, 25-measurement and 24-measurement rows, plus invalid cells. The first two rows receive predictions; the last two remain in the output with `insufficient_coverage`. Missing/invalid cells use the saved training medians only after the coverage check.

The evaluation example builds a small synthetic dataset with toy class labels, hashes its inputs and runs clustered uncertainty calculations. It illustrates the analysis interface and does not evaluate the trained model or reproduce scientific results. None of these examples contains study specimens.
