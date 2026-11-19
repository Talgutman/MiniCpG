# Trained MiniCpG model

The saved MiniCpG tissue classifier uses the 30-CpG consensus panel and predicts 27 tumor classes. It combines a fitted median-imputation step with an ExtraTrees classifier. Predictions use the class with the highest probability across the full output space.

## Files and contents

| File | Purpose |
|---|---|
| [minicpg_k30_model_bundle.pkl.gz](minicpg_k30_model_bundle.pkl.gz) | Compressed Python bundle containing the fitted pipeline, probe/class order, selected hyperparameters and seed |
| [feature_order.tsv](feature_order.tsv) | The ordered CpG identifiers used by the model |
| [class_labels.tsv](class_labels.tsv) | Internal TCGA identifiers and displayed tumor names, including combined classes |
| [artifact_manifest.json](artifact_manifest.json) | Model, panel and class-metadata hashes |

The bundle contains no raw methylation matrices or specimen-level metadata. Its fitted model is unchanged from the original `v0.1.0` release. Held-out TCGA performance is reported in [Supplementary Table S1](../results/revision_20261119/Table_S01_01.tsv); use the [results index](../results/revision_20261119/README.md) for the complete table mapping.

## Run the model

Use the repository lockfile, which specifies scikit-learn 1.7.2. From the repository root after installation:

```bash
uv run --locked minicpg-predict \
  --input examples/example_input_30cpg.tsv \
  --output outputs/example_predictions.tsv \
  --include-probabilities
```

Input is a TSV or CSV with one specimen per row and columns named by CpG identifier. At least 25 panel probes must have finite beta values in [0, 1]. Missing or invalid measurements count toward the coverage threshold. Eligible specimens use only the saved training-set medians; specimens below the threshold remain in the output with an explicit status and no prediction.

See [input and output](../README.md#input-and-output) for the full format and [examples](../examples/README.md) for missing-data behavior. When running outside the repository root, supply `--model /path/to/minicpg_k30_model_bundle.pkl.gz`. The model is distributed with the repository/source archive and separately from the Python wheel.

## Scope and intended use

This is the tissue tumor-type classifier. The paper's cell-free DNA and normal-adjacent detection analyses use separate fitted workflows; they are not additional modes of this saved model.

The model is provided for academic and non-commercial research under [LICENSE](../LICENSE). It is not a clinical diagnostic device; commercial or clinical diagnostic use requires separate written permission.
