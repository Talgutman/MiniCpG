# MiniCpG

MiniCpG classifies tumor tissue using a compact panel of 30 DNA methylation probes. The trained model predicts 27 tumor classes using an ExtraTrees classifier and imputation fitted on the development data.

This repository accompanies “MiniCpG: Broad validation and characterization of a 30-CpG cancer classifier” and provides the trained model, analysis code and aggregate supplementary results. The current code version is `0.2.0.dev0`.

| I want to… | Start here |
|---|---|
| Predict tumor type from my methylation data | [Quick start](#quick-start) and [input requirements](#input-and-output) |
| Try complete or partially measured inputs | [Prediction examples](examples/README.md) |
| Reproduce an analysis from the paper | [Reproduction guide](docs/reproduction_workflow.md) |
| Find supplementary numerical results | [Results index](results/revision_20261119/README.md) |
| Inspect the model or validation evidence | [Model description](model/model_card.md) · [Reproduction requirements](docs/reproduction_requirements.md) |

## Quick start

Using [uv](https://docs.astral.sh/uv/), clone the repository and run the synthetic prediction example:

```bash
git clone --branch main https://github.com/Talgutman/MiniCpG.git
cd MiniCpG
uv sync --locked
uv run --locked minicpg-predict \
  --input examples/example_input_30cpg.tsv \
  --output outputs/example_predictions.tsv \
  --include-probabilities
```

If you already have the repository checked out, start at `uv sync --locked`. All commands in the documentation assume the repository root as the working directory. The example contains synthetic measurements; its predictions are for demonstrating usage.

To analyze your own data, replace the input and output paths. Python 3.11 is the default; Python 3.11–3.12 are supported. The lockfile includes scikit-learn 1.7.2, matching the saved model.

## Input and output

Input is a TSV or CSV with one specimen per row and probe columns named as in [feature_order.tsv](model/feature_order.tsv). An optional `sample_id` column must contain unique, nonempty identifiers. If it is omitted, rows receive identifiers such as `row_1`. Extra columns are ignored.

- A specimen needs at least **25 of the 30 probes** measured as finite beta values between 0 and 1.
- Absent probe columns, blank or nonnumeric cells, infinities and out-of-range values count as missing.
- Eligible specimens have missing values filled with the saved training-set medians. No preprocessing is fitted on the new specimens.
- The prediction is the class with the highest model probability across all 27 outputs.

The output contains one row for every input specimen: its identifier, measured/missing probe counts, status, predicted class and maximum probability. `--include-probabilities` adds one probability column per modeled class. Specimens below the coverage threshold have `status=insufficient_coverage` and blank prediction/probability fields.

Predictions use The Cancer Genome Atlas (TCGA) identifiers. [class_labels.tsv](model/class_labels.tsv) maps them to tumor names, including the combined COAD/READ, ESCA/STAD and GBM/LGG classes. See the [model description](model/model_card.md) for bundle contents and compatibility.

The model is included in the repository and source distribution. A Python-wheel installation requires a separate model path, supplied as `--model /path/to/minicpg_k30_model_bundle.pkl.gz`. Load only trusted model bundles.

## Paper analyses and results

The [reproduction guide](docs/reproduction_workflow.md) explains how to choose an analysis, prepare its inputs and run it. The [workflow recipes](docs/workflows.md) contain commands and configuration requirements; the [command reference](docs/cli_reference.md) lists script arguments.

[Supplementary results](results/revision_20261119/README.md) cover Tables S1–S25 as 71 aggregate TSV blocks. Study-level measurements, clinical metadata and specimen-level predictions are obtained separately; see [data access](docs/data_access.md). The [reproduction requirements](docs/reproduction_requirements.md) explain the data and software needed for these workflows.

## Citation and license

Citation metadata are in [CITATION.cff](CITATION.cff). Academic and non-commercial research use is governed by [LICENSE](LICENSE). Commercial use, clinical diagnostic use and paid-service use require separate written permission. MiniCpG is a research model, not a clinical diagnostic device.
