# Paper analysis workflows

Run commands from the repository root after `uv sync --locked --extra analysis`. Each section below corresponds to an analysis in the main paper or Supplementary Methods (SM1–SM8). The [command reference](cli_reference.md) lists arguments; [scientific conventions](analysis_policies.md) explains cohort, prediction and statistical rules.

The saved model is sufficient for prediction. Reproducing analyses requires the measurements and curated metadata described in [data access](data_access.md). Training, expression scans and sequencing-matrix extraction require suitable compute resources.

## Workflow index

| Paper analysis | Methods and reported outputs | Recipe |
|---|---|---|
| Development, consensus selection and 20-CpG sensitivity | SM2; Fig. 2; Fig. S1; Tables S1, S4 | [Development](#development-and-reduced-panel-sensitivity) |
| Primary external validation and MethyDeep | SM1–SM3; Fig. 3; Tables S2–S3, S12 | [External evaluation](#primary-external-validation-and-methydeep) |
| Consensus versus single-split panels | SM2–SM3; Fig. S5; Table S5 | [Panel comparison](#consensus-versus-single-split-panels) |
| Methylation shift, confusion, purity and study-specific errors | SM5; Fig. 4; Figs. S6–S7; Tables S6–S11 | [Error analyses](#error-patterns-geometry-and-purity) |
| Metastatic, FFPE and EPIC cohorts; paired clinical/processing comparisons | SM1, SM3–SM4; Fig. 5; Tables S3, S12–S15 | [Additional cohorts](#expanded-domains-and-paired-conditions) |
| Genomic context, biological scores, normal-adjacent analyses, expression and attribution | SM6; Fig. 6; Figs. S8–S11; Tables S16–S22 | [Biological context](#genomic-and-biological-context) |
| Exploratory cfDNA-specific models and coverage control | SM7; Fig. 7; Table S23 | [Cell-free DNA](#native-cell-free-dna) |
| Cost and turnaround assumptions | SM8; Table S24 | [Assay estimate](#assay-cost-and-turnaround) |
| Age representation | SM1; Table S25 | [Age summaries](#age-representation) |
| Sample and probe principal components | Figs. S2–S3 | [PCA](#sample-and-probe-pca) |

## Synthetic evaluation example

To run MiniCpG with the saved classifier, start with the [prediction examples](../examples/README.md). The example below demonstrates the separate evaluation workflow: it creates synthetic predictions and calculates metrics and clustered uncertainty. It does not run the trained classifier or reproduce study findings.

```bash
uv run --locked python examples/run_synthetic_evaluation.py --out-dir outputs/synthetic_evaluation
```

## Record input files and hashes

Evaluation commands use a JSON configuration whose `files` object maps an input name to a path and SHA256 digest. Relative paths resolve against the configuration file, not the current directory. Templates in `config/workflows/` deliberately contain placeholders. Bind every listed input to your local file; for example:

```bash
uv run --locked python scripts/bind_analysis_inputs.py \
  --template config/workflows/study_errors.json \
  --output work/study_errors.json \
  --file predictions=local_data/predictions.tsv \
  --file canonical_master=local_data/master.tsv \
  --file canonical_inventory=local_data/inventory.tsv
uv run --locked python scripts/validate_analysis_inputs.py --config work/study_errors.json
uv run --locked python scripts/summarize_study_errors.py \
  --config work/study_errors.json --out-dir outputs/study_errors
```

For configurations with many files, generate the same `files` dictionary in Python using `minicpg.analysis_io.sha256`. Do not remove required inputs to bypass a mismatch. The bound runners reject changed inputs and nonempty output directories and write `provenance.json`. Other CLI scripts use their own hash checks or manifests; use explicit arguments and fresh output directories for those scripts too. Keep specimen-level outputs local.

## Common input tables

All tables below are tab-delimited unless noted. Preserve specimen order when reproducing a frozen bootstrap stream.

| Input | Required structure |
|---|---|
| External matrix | Unique `gse_id`, `sample_id`; `tcga_label`; CpG beta columns named by probe identifier |
| Canonical master | Unique `gse_id`, `sample_id`; `tcga_label`; eligibility fields used in the configuration, including `include` and `target_bucket` for primary tumors |
| Measurement inventory | Unique `gse_id`, `sample_id`; `tcga_label`; source/availability annotations; inference and geometry also check configured eligibility fields |
| Prediction table | Unique `gse_id`, `sample_id`; `tcga_label`, `y_pred`; probabilities named `proba__TCGA-…` for available model outputs |
| Patient units | Unique `gse_id`, `sample_id`; `resampling_unit`, explicitly namespaced by study and patient; unresolved identities remain separate specimen units |
| Coverage/status | Unique specimen keys; named cohort and eligibility columns referenced by each evaluation job; flags serialized as `1` or `0` |
| Class axis | One `tcga_label` per row, in the declared modeled-class order |
| TCGA panel matrix | `sample_id`, `project_id`, then panel CpGs; development and held-out manifests must retain their locked patient separation |

Use the evaluated curation snapshot together with its measurement inventory. The code validates supplied identities; it cannot reconstruct manual diagnosis, tissue, source-availability or patient-identity adjudication from an accession list. Supplementary Table S12 supplies aggregate study information, not a specimen-membership substitute.

## Development and reduced-panel sensitivity

`config/paper_protocolb_v1.release.yaml` preserves the executed search ranges, split seed 42, inner-CV seed 7, initial 5×3 outer folds and confirmation 5×20 outer folds. The initial comparison used K = 5, 10, 20, 30, 50 and 100; confirmation used the selected K = 30 configuration. Replace its input placeholders. The template explicitly requests a **new** split; to reproduce an existing run, set `split.mode: reuse`, supply `split.existing_dev_manifest`/`existing_lockbox_manifest`, and set the stages' `existing_outer_folds` to the frozen assignments. Never silently regenerate a missing frozen assignment.

```bash
uv run --locked --extra analysis python scripts/run_paper_k_pipeline.py \
  --config work/development.yaml --k 30 --out-dir outputs/development_k30 --seed 7
```

This prepares manifests and task lists for one panel size. For the initial comparison, repeat it with each screened `--k` value and a separate output directory. Execute their feature-selection tasks with `run_paper_fs_task.py`, aggregate with `aggregate_paper_fs_candidates.py`, execute model tasks with `run_paper_model_task.py`, aggregate with `aggregate_paper_model_candidates.py`, and construct the locked confirmation manifests with `build_paper_stage2_locked_manifests.py`. Use the emitted task manifests and CLI reference for the task indices. `build_stable_panel_v1.py` constructs the selection-frequency consensus panel. Fold preprocessing is fitted within training folds. The selector registry and benchmark scripts implement the feature-selection and classifier families described in SM2. Patient split, fold and cached-training membership checks are included.

After selecting the pipeline and constructing its consensus panel, `evaluate_lockbox_from_panel.py` fits the final model on development data and evaluates the held-out set once. Its input Parquet matrix has `probe_id` rows and sample-ID columns; both manifests must include `sample_id`, `patient_id`, `project_id`. For the adopted model, use ExtraTrees, balanced class weights, macro-F1 tuning, five inner folds and seed 7. The executed final estimator is recorded in `config/panel_estimator.json`. Do not enable `--fit-decision-policy` for the final model. For example, a fixed-parameter fit after selection is:

```bash
uv run --locked --extra analysis python scripts/evaluate_lockbox_from_panel.py \
  --probe-matrix local_data/probe_matrix.parquet \
  --dev-manifest local_data/dev.tsv --lockbox-manifest local_data/heldout.tsv \
  --panel-tsv resources/panels/stable_panel_rankrule_top30.tsv --k 30 \
  --out-dir outputs/final_fit --model extra_trees --class-weight-mode balanced \
  --tune-metric macro_f1 --inner-folds 5 --seed 7 \
  --et-n-estimators-grid 400 --et-max-features-grid 0.35 \
  --et-max-depth-grid 0 --et-min-samples-leaf-grid 1 --compute-auroc
```

This fixed-parameter command reproduces the fitting step given the selected settings; it does not repeat the earlier search. Use the source search grids for the full development procedure. For reduced-panel sensitivity, `evaluate_external_from_bundle.py` accepts its uncompressed fitted bundle and an already sanitized, coverage-filtered panel matrix (including `target_bucket`). Keep `--pre-impute none`; its emitted predictions feed the current paired evaluator. Its filename `external_predictions_allow_missing1.tsv` is retained for source compatibility and does not define the supplied cohort's coverage threshold.

The input methylation matrix must already be processed and annotated. This Python environment does not reconstruct the upstream TCGA SeSAMe/R environment or download TCGA measurements. Record those versions and input hashes separately. The YAML defines the search; exact reproduction also requires the evaluated split and fold assignments.

`config/k20_reproduction.json` records the K20 selector settings, estimator parameters and executed model hash. Use the same development workflow with `--k 20` and its configuration for this sensitivity; its fitted binary is not redistributed here. Evaluate supplied K20 predictions through `evaluate_external.py` with the same primary cohort/axis conventions.

## Primary external validation and MethyDeep

Extract panel measurements with `build_external_panel_matrix.py`, then sanitize out-of-range beta values with `sanitize_external_panel_beta_values.py`. Canonical curation and source inventory determine membership before inference. `predict_external_cohort.py` with `config/workflows/inference.json` uses the released model and emits eligible predictions in the schema above, plus coverage status for every input specimen.

Bind `config/workflows/primary.json` and run:

```bash
uv run --locked python scripts/evaluate_external.py \
  --config work/primary.json --out-dir outputs/primary_evaluation
```

The template preserves original predictions, the fixed full/shared macro-F1 axes, paired truth-stratified patient-cluster draws, specimen and patient-balanced estimates, and a separately named ESCA/STAD runner-up sensitivity. Primary AUROC averages evaluable truth classes. Missing probability columns and class axes are recorded. Do not renormalize predictions or remove an output class in the principal comparison. `float_precision: null`, unsorted selected rows and `first_seen` stratum ordering preserve the adopted primary bootstrap stream.

MethyDeep execution uses `run_methydeep_inference.py` with its separately obtained `RF_feature45.csv`, `targer_names.csv` and SavedModel. Use a compatible TensorFlow/Keras SavedModel environment; TensorFlow is intentionally not in the tissue-inference lock. The runner's `--help` is available without TensorFlow. It uses the top 30 probes, requires 27 measurements, and imputes using the specimen's measured-panel mean. Pass its original specimen-level predictions to `evaluate_external.py` to calculate the matched comparison with the paper’s class axis and resampling rules.

## Expanded domains and paired conditions

Bind `config/workflows/expanded.json` and use the same evaluator. Each metastatic, FFPE and EPIC membership flag is independent; a specimen can belong to multiple cohorts. This template uses present-truth class axes, full-axis AUROC availability checks, sorted strata and the recorded domain/subset seeds.

`evaluate_paired_conditions.py` uses `config/workflows/paired.json` for the adopted same-patient primary/regional-node and same-specimen processing contrasts. Input names in that template are literal file keys. Prediction files use the common schema; clinical membership additionally has `patient_namespace_and_id` and `sample_role` (`primary`/`metastatic`); IDAT membership has specimen keys; beta-difference inputs have `gse_id`, `model`, `difference`; `PROTOCOL.txt` describes and binds the comparison. `canonical_master_scope.tsv` supplies export-comparison membership. The two export comparisons remain study-specific (GSE144487 and GSE234379).

```bash
uv run --locked python scripts/evaluate_paired_conditions.py \
  --config work/paired.json --out-dir outputs/paired_evaluation
```

The source retains patient aggregation, common four-way eligibility and 20,000 paired resamples. Same-patient contrasts are not propensity-matched independent patients. IDAT contrasts compare complete processing workflows; upstream SeSAMe processing must be supplied separately. The runner produces specimen-level audit outputs locally as well as aggregate results.

## Consensus versus single-split panels

`config/single_split_panels.json` contains the adopted consensus plus single-split probe definitions, ten classifier seeds and maximum missingness. `config/panel_estimator.json` binds the fixed ExtraTrees parameters and canonical seed.

`train_panel_comparison.py --config work/panel_training.json --out-dir outputs/panel_task_0 --task 0` fits one of ten task shards, using full development data only. Repeat tasks 0–9 on appropriate compute. The `panel_training.json` template names the required union matrices, locked model and reference predictions. Bind `panel_plan` to `config/single_split_panels.json` and `estimator_plan` to `config/panel_estimator.json`. Union matrices are pandas pickles with `sample_id`, `project_id` and all required CpGs; external matrices use the common schema. Pickles must come from trusted sources. The model input for this runner is an **uncompressed** bundle. For example, `gzip -dc model/minicpg_k30_model_bundle.pkl.gz > work/model_bundle.pkl` preserves its bytes; verify against `model/artifact_manifest.json`.

Only specimens satisfying every compared panel's coverage criterion enter a common matrix. Do not impute first and then decide coverage. Reference predictions are used to check the consensus model at its canonical seed.

Bind `config/workflows/panel_summary.json`, including one input key `{panel_id}_{seed}_main` per prediction file, then run:

```bash
uv run --locked python scripts/summarize_panel_comparison.py \
  --config work/panel_summary.json --out-dir outputs/panel_comparison --bucket main
```

The plan order begins with consensus. All predictions must have identical specimen order, truth and original argmax. `units` uses the patient-unit schema. The runner averages across classifier seeds, computes ranks and selection-frequency associations, and uses paired clustered uncertainty. Output fields named `alternative_*` refer to the single-split panels.

## Error patterns, geometry and purity

`analyze_class_geometry.py` uses `config/workflows/geometry.json`. Its required inputs are the model, external matrix, predictions, canonical master and inventory. Optional `heldout` plus `recalls` add methylation-shift analysis; recalls has `tcga_label`, `heldout_recall`, `external_recall`. Optional `development` adds nearest-neighbor geometry. Remove both optional keys when a component is not requested, before binding the template.

```bash
uv run --locked python scripts/analyze_class_geometry.py \
  --config work/geometry.json --out-dir outputs/class_geometry
```

All class profiles are arithmetic means across the locked probes after the saved training-median imputation. Pair confusion uses both directions and both class denominators, retaining zero-confusion pairs. The primary test permutes whole class labels jointly across rows and columns; it does not treat class pairs as independent observations. Methylation-shift Spearman p-values remain nominal.

`analyze_heldout_purity.py --config work/purity.json --out-dir outputs/purity` accepts a held-out manifest (`sample_id`, `file_uuid`), predictions (`sample_id`, `tcga_label`, `y_pred`), specimen metadata (`file_uuid`, `sample_submitter_id`, `sample_type`), and ABSOLUTE estimates (`sample`, `purity`). It prioritizes exact specimen/vial matching, keeps relaxed matches as sensitivities, fits class-conditional logistic regression and performs the within-class permutation check. Run `check_heldout_purity_sensitivity.py --analysis-dir outputs/purity` afterward. Sample-level matches stay local.

`summarize_study_errors.py` reports study/class recall and confusion. Optional `annotations` and `annotation_fields` aggregate curated source characteristics; these annotations do not change eligibility or imply causality. The geometry runner also exports class dispersion and UCS/UCEC study distances. `analyze_uvm_skcm_diagnostics.py` with `config/workflows/uvm_skcm.json` retains the adopted UVM/SKCM study, probe-missingness, observed-dimension and comparator diagnostics. It additionally requires both TCGA split matrices/manifests, held-out predictions, MethyDeep predictions/matrix and the coverage table. The exported tables preserve the adopted subtype and dispersion findings. Automated summary code does not replace source-paper review of specific subtypes, nor does it reconstruct those manual annotations.

## Genomic and biological context

Supply the input and output paths listed by each command. The sequence below follows the biological analyses in SM6; see the command reference for complete argument names.

| Component | Entry points and input contracts |
|---|---|
| Genomic annotation | `annotate_panel_probe_locations.py`: selected-panel and HM450 annotation tables; `annotate_panel_chromatin_context.py`: panel biology with `probe_id`, `panel_rank`, `CpG_chrm`, `CpG_beg`, `CpG_end`, `gene_list`, `final_biological_label`, plus GRCh38 SCREEN BED files. Coordinates are already zero-based, half-open. |
| Context enrichment | `analyze_panel_context_enrichment_from_cv_prefilters.py`: annotation, annotated panel and per-fold MI-prefilter TSV glob; background is the development prefilters. |
| Tumor–normal / lineage | `analyze_panel_tumor_normal.py`, `analyze_panel_lineage_normals.py`, `audit_normal_metadata_composition.py`, `analyze_normal_tissue_identity.py`: separately curated normal matrices, tumor split matrices and matched metadata. Normal eligibility must distinguish tissues with development tumors. |
| Composition | `analyze_immune_purity_probe_associations.py`: panel matrices and specimen-linked covariates; its outputs feed `analyze_probe_biology_decomposition.py`. |
| Evidence decomposition | `analyze_probe_biology_decomposition.py`: normal-lineage/audit, biology master, development/held-out tumor matrices, unpaired and matched tumor–normal summaries, and pooled/within-class composition associations. |
| Same-patient substitution | `run_matched_normal_substitution_minicpg.py`: uncompressed locked model, held-out tumor matrix/manifest and normal matrix with `patient_id`, `sample_id`, `sample_type_code` and probe values. It requires same-patient normal-adjacent specimens, not external healthy controls. |
| Separate normal gate | `run_tumor_gate_proof_of_concept.py`: explicit `--dev-tumor`, `--test-tumor`, `--normal`, `--model-bundle`, `--out-dir`, and `--active-manifest work/normal_gate.json` bound from the supplied template. The normal audit supplies `has_development_tumor` and `locked_model_normal_primary_subset`. This is a separate fitted detection experiment, not a rejection rule added to the released tumor classifier. |

The normal-gate configuration uses seed 20260531. Keep development and held-out patients separate. `analyze_locked_minicpg_on_normals.py` prepares the normal-sample predictions used for the gate’s fixed-classifier confidence comparators; it requires the normal metadata audit, development matrix and biological decomposition outputs.

For expression, run `analyze_panel_meth_expr_transcriptome_null.py` with `--probe-gene-pairs-tsv`, `--expression-metadata-tsv`, `--tcga-dev-matrix-tsv`, `--tcga-lockbox-matrix-tsv`, and `--out-dir`. It supports `--focus-projects` and `--shard-by-project` for jobs; aggregate using `aggregate_panel_meth_expr_transcriptome_null_shards.py`, then summarize with `summarize_panel_meth_expr_transcriptome_null.py`. Inputs are annotated probe/gene pairs and project-resolved RNA-seq metadata pointing to local expression files. The finite-beta correlation N is distinct from the broader expression overlap N. Transcriptome backgrounds and linked-gene support are calculated within project.

For model usage, `analyze_locked_model_shap_and_correlation.py` takes an uncompressed `--model-bundle`, `--lockbox-panel-tsv`, `--external-panel-tsv`, `--external-predictions-tsv`, and `--out-dir`. The adopted convention uses true-class SHAP and the original approximation option; it is descriptive model usage rather than evidence of causality. Bind external predictions to the same adopted primary membership before running.

## Native cell-free DNA

This is a separate nested workflow and panel, not inference with the tissue model. `config/cfdna_executed_runs.json` records the executed K100 methylation/coverage-control metadata. Start from the cfMethDB processed sequencing intermediate, not array betas.

For data preparation, `filter_cfmethdb_presence_candidates_by_study_fraction.py` applies the label-blind 80% study-presence filter to merged presence-atlas shards. Supply the actual study denominator. `extract_cfmethdb_candidate_matrix_np_chunk.py` reads candidate coordinates (`feature_id`, `chrom`, `start`, `end`) and a sample manifest (`Geoid`, `analysis_label`, `studyid`, `bigwig_url`), downloading BigWigs into an explicit cache directory. This extraction optionally requires `uv sync --locked --extra analysis --extra cfdna-data` and curl. `combine_cfmethdb_candidate_matrix_np_chunks.py` creates the NumPy matrix and aligned metadata. The atlas itself and source annotation need to be obtained/constructed from the permitted upstream source. Exact upstream per-file software versions were not recoverable.

The analysis input matrix is a specimen-by-feature `.npy` array; metadata rows must align exactly and contain `Geoid`, `analysis_label`. Stage metadata contains `Geoid`, `geo_stage`. The defaults follow the reported 100-CpG analysis. The complete command is:

```bash
uv run --locked --extra analysis python scripts/run_cfdna_specificity_constrained_twostage_array.py \
  --mode fold --fold-id 1 --dataset prjna628686 \
  --matrix local_data/cfdna/beta_matrix.npy \
  --sample-metadata local_data/cfdna/sample_metadata.tsv \
  --stage-metadata local_data/cfdna/stage_metadata.tsv \
  --out-dir outputs/cfdna_methylation \
  --prjna-labels CRC,ESCA,GASTRIC,LIHC,Normal,PAAD --prjna-normal-label Normal \
  --budgets 20:80,30:70,50:50,70:30,80:20 --max-union 100 \
  --value-mode methylation --model-families extra_trees,logistic_l2 \
  --subtype-selector ovr_union --config-subtype-metric subtype_macro_f1_gate_positive_cancers \
  --target-specificity 0.80 --outer-splits 5 --outer-repeats 5 --inner-splits 4 --seed 20260616
```

Run fold IDs 1–25, then the same command with `--mode merge` (omit `--fold-id`). Repeat with `--value-mode coverage` and a separate output directory for the coverage-indicator control. Report repeated-CV means/SDs as descriptive summaries, not independent-fold confidence intervals. Each workflow's subtype result is conditional on its own gate-positive samples.

## Age representation

`summarize_age_representation.py` with `config/workflows/age.json` uses explicit membership (`cohort`, specimen keys, `tcga_label`) and source-curated `age_category` (`adult`, `paediatric`, `unknown`). Paediatric means <18; adult means ≥18. Unresolved/crossing bands remain unknown, including missing annotations. Independent cohort memberships can overlap. This produces representation counts; it is not a powered age-specific validation analysis. Table S25 reports the cohort-specific counts.

## Assay cost and turnaround

`calculate_assay_cost.py` derives batch and sensitivity totals from `config/assay_cost.json`; references and price/overhead assumptions are included. Table S24 also contains the proposed turnaround schedule. That schedule is an assumption-based protocol scenario, not an empirical runtime benchmark.

```bash
uv run --locked python scripts/calculate_assay_cost.py \
  --config config/assay_cost.json --out-dir outputs/assay_cost
```

The outputs summarize costs under the supplied assumptions; they are not a validated assay quote.

## Sample and probe PCA

For sample PCA, create a fresh run directory with `inputs/` containing `panel.tsv`, `split_lock.json`, `dev_manifest.tsv`, `heldout_manifest.tsv`, `dev_matrix.tsv`, `heldout_matrix.tsv`, and `input_hashes.json` mapping each filename to its digest. Panel rows include `probe_id`, `panel_rank_rule`; matrix columns are `sample_id`, `project_id`, then probes in panel rank order. The split lock contains `dev_samples`, `heldout_samples`, `dev_sample_sha256`, `heldout_sample_sha256`; identifier hashes are SHA256 of sorted unique IDs joined by newline, without a trailing newline.

```bash
uv run --locked python scripts/analyze_sample_pca_20261106.py --run-dir work/sample_pca
uv run --locked python scripts/analyze_probe_pca_20261106.py --run-dir work/probe_pca
```

Probe PCA additionally needs `fold_presence.tsv` (`fold_id` plus binary probe columns), panel `n_folds_top_panel`/`selection_rate_top_panel`, and `annotations.tsv` with probe/rank/frequency/context fields. Use separate directories for sample and probe PCA. Both scripts verify hashes/membership. Sample PCA fits imputation and centering on development data and projects held-out specimens. Probe PCA uses development class means and includes the within-probe-centered sensitivity. These descriptive projections are conditional on supervised feature selection.

## Aggregate results

The [results index](../results/revision_20261119/README.md) links Tables S1–S25 to machine-readable TSV files. `tables.json` records their titles and blocks; `manifest.tsv` records file hashes. The tables contain aggregate results, with specimen-level inputs and outputs kept separately.

See [reproduction requirements](reproduction_requirements.md) for software and data prerequisites. The example data demonstrate usage; they do not reproduce the paper’s numerical findings.
