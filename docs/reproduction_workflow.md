# Reproducing the paper analyses

This guide is for reproducing the analyses in the manuscript. To classify new specimens with the saved MiniCpG model, use the [quick start](../README.md#quick-start).

## Prepare the environment

Run from the repository root:

```bash
uv sync --locked --extra analysis
```

This installs the Python analysis dependencies. MethyDeep requires a separate compatible TensorFlow/Keras environment, and upstream SeSAMe processing requires its R environment. Those environments are described where relevant in the [workflow recipes](workflows.md).

## Choose an analysis and prepare its inputs

1. Find the analysis in the [workflow index](workflows.md#workflow-index). Each recipe identifies the script, configuration and required inputs.
2. Obtain the relevant measurements and curated metadata using the [data-access guide](data_access.md). Aggregate result tables cannot replace specimen-level inputs.
3. Prepare the [input tables](workflows.md#common-input-tables) and [record their paths and hashes](workflows.md#record-input-files-and-hashes) where the runner requires a bound configuration.
4. Run the chosen recipe in a fresh output directory. Preserve the model, cohort definitions and statistical settings described in [analysis conventions](analysis_policies.md).

The [synthetic evaluation example](../examples/README.md#try-the-evaluation-workflow) demonstrates the configuration and output format without study data. The [supplementary results index](../results/revision_20261119/README.md) identifies the numerical sources reported in the manuscript.

## Check the installation

```bash
uv run --locked --extra analysis pytest
uv run --locked python scripts/verify_model_artifacts.py
```

The tests exercise synthetic inputs, implementation checks and the saved model. They do not rerun the full scientific analyses. [Reproduction requirements](reproduction_requirements.md) describes the data and software needed to run the study workflows.

`uv.lock` specifies the tested package environment. Reproducing the reported results also depends on the source-data versions, upstream processing and manual cohort curation documented for each analysis.
