# MiniCpG K=30 Trained Model

This directory contains the locked MiniCpG K=30 inference artifact used for the manuscript analyses.

## Files

- `minicpg_k30_model_bundle.pkl.gz`: gzip-compressed Python pickle containing the fitted sklearn pipeline.
- `feature_order.tsv`: required CpG order for inference.
- `class_labels.tsv`: modeled TCGA class labels and full names.

## Bundle Contents

The model bundle contains a fitted median-imputation step, an ExtraTrees classifier, the 30-CpG feature order, the modeled class order, selected hyperparameters, and the random seed. It does not contain raw methylation matrices or sample-level metadata.

## Held-Out TCGA Readback

- `n_dev`: `6780`
- `n_lockbox`: `1681`
- `n_classes`: `27`
- `accuracy`: `0.9250446162998215`
- `weighted_f1`: `0.9248173282194696`
- `macro_f1`: `0.9206644714720833`
- `auroc`: `0.9972515884053619`

## Input Format

Provide a TSV or CSV with one row per sample and columns named by Illumina CpG probe IDs. A `sample_id` column is recommended. Extra columns are ignored; all 30 MiniCpG probes must be present. Missing beta values are allowed and are imputed by the fitted training-set medians stored in the model bundle.

## Example

```bash
PYTHONPATH=src python scripts/predict_minicpg.py \
  --model model/minicpg_k30_model_bundle.pkl.gz \
  --input examples/example_input_30cpg.tsv \
  --output examples/example_predictions.tsv \
  --include-probabilities
```

## License And Intended Use

The model is provided for academic and non-commercial research use under the MiniCpG Academic and Non-commercial Research Use License. It is not intended for commercial use or clinical diagnostic use. Clinical translation would require assay-specific calibration, prospective validation, regulatory review, and separate permission.
