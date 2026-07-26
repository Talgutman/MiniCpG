# MiniCpG Code Release

This directory contains a public-code snapshot for the MiniCpG manuscript: a compact 30-CpG DNA methylation classifier for cancer-type assignment.

This release is provided for academic and non-commercial research use under the MiniCpG Academic and Non-commercial Research Use License (see `LICENSE`). Commercial use, clinical diagnostic use, and paid-service use require separate written permission.

## Contents

- `scripts/`: pipeline, validation, benchmark, analysis, and figure-generation entry points.
- `src/minicpg/`: release-native selector and registry used by the final locked pipeline.
- `model/`: compressed trained MiniCpG K=30 model bundle and inference metadata.
- `config/`: sanitized templates with placeholders for local data locations.
- `resources/`: final panel, class maps, public accession summaries, and cfDNA summary inputs.
- `examples/`: minimal 30-CpG input schema example for direct inference.
- `results/`: small aggregate tables used as manuscript or supplement source data.
- `docs/`: data-access notes, reproduction overview, manifest, and safety scan.

Raw methylation matrices, sample-level curation files, sample-level predictions, local manuscript packages, and local development-only files are intentionally not included.

## Minimal Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Direct Model Use

```bash
PYTHONPATH=src python scripts/predict_minicpg.py \
  --model model/minicpg_k30_model_bundle.pkl.gz \
  --input examples/example_input_30cpg.tsv \
  --output examples/example_predictions.tsv
```

Populate the placeholders in `config/paths.template.yaml` and `config/paper_protocolb_v1.release.yaml` after downloading the permitted public inputs.

## Citation

Please cite the associated MiniCpG manuscript and this code release. Citation metadata are provided in `CITATION.cff`; replace the placeholder repository URL after public deposition.

## Intended Use

The included trained model is a research-use artifact for reproducibility and method evaluation. It is not a clinical diagnostic device or a substitute for validated pathology workflows.
