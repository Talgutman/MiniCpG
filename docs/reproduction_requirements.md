# Reproduction requirements

The repository contains the trained 30-CpG MiniCpG classifier, the analysis workflows described in the paper, synthetic usage examples and aggregate supplementary tables. The [workflow index](workflows.md#workflow-index) maps each analysis to its methods section and reported outputs.

## Data and software

Exact reproduction requires the evaluated measurements, curated specimen membership, patient groups, development/test split and fold assignments. Aggregate supplementary tables cannot substitute for these inputs. Obtain source data under their applicable access terms and follow the [data-access guide](data_access.md).

`uv.lock` specifies the supported Python environment. Upstream SeSAMe processing requires R and its associated packages; MethyDeep requires a separate TensorFlow/Keras environment compatible with its SavedModel. The cfDNA workflow starts from cfMethDB processed profiles; exact upstream software versions are unavailable for individual source files, as described in SM7.

The saved tissue model is included. The 20-CpG sensitivity model and the separately fitted normal-adjacent and cfDNA models require their corresponding fitting workflows. Their settings are supplied in the configurations and recipes. Compute-intensive fitting and expression scans should run on suitable compute resources.

## Check an installation

```bash
uv sync --locked --extra analysis
uv run --locked pytest
uv run --locked python scripts/verify_model_artifacts.py
```

Tests cover inference, input validation, statistical calculations and synthetic regression cases. They do not rerun the full study. `release_manifest.json` records distributed-file checksums; the model and aggregate results have their own manifests.

## Scope of the analyses

Clinical interpretation remains bounded by the populations and processing conditions evaluated in the paper. The tissue classifier assigns among its modeled tumor classes. The normal-adjacent gate and cfDNA models are separate experiments. Cost and turnaround outputs are protocol-based scenarios, not measurements of an implemented clinical assay.
