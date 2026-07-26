# Reproduction Workflow Overview

1. Download permitted public methylation inputs and probe annotation resources.
2. Fill the path placeholders in `config/paths.template.yaml` and `config/paper_protocolb_v1.release.yaml`.
3. Build or provide a sample-by-CpG beta matrix compatible with the script inputs.
4. Run the TCGA cohort/split, feature-selection, model benchmark, stable-panel, held-out validation, external validation, MethyDeep comparison, failure-analysis, biological-interpretation, and cfDNA supplement scripts as needed.
5. Use the aggregate files in `results/figure_source_data/` and `results/supplementary_tables/` to verify the manuscript-facing reported values.

The final manuscript model is the `K=30` panel in `resources/panels/stable_panel_rankrule_top30.tsv` with the `rare_class_rank_shift` selector and ExtraTrees classifier. For direct inference with the already fitted model, use `scripts/predict_minicpg.py` with `model/minicpg_k30_model_bundle.pkl.gz`.
