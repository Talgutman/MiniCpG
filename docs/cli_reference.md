# Command reference

Run these commands from the repository root after `uv sync --locked --extra analysis`. Start with the [workflow guide](workflows.md) to select the analysis and prepare its inputs. Expand a command below for its arguments.

- [Prediction and input validation](#prediction-and-input-validation)
- [Development and consensus panel](#development-and-consensus-panel)
- [External evaluation and single-split panels](#external-evaluation-and-single-split-panels)
- [Error analyses](#error-analyses)
- [Biological context](#biological-context)
- [Cell-free DNA](#cell-free-dna)
- [PCA, age and assay estimates](#pca-age-and-assay-estimates)

## Prediction and input validation

<details>
<summary><code>predict_minicpg.py</code></summary>

```text
usage: predict_minicpg.py [-h] --input INPUT --output OUTPUT [--model MODEL]
                          [--include-probabilities]

Command-line prediction with the locked MiniCpG tissue model.

options:
  -h, --help            show this help message and exit
  --input INPUT         Sample-by-probe TSV or CSV.
  --output OUTPUT       Prediction and coverage-status TSV.
  --model MODEL         Trusted local model pickle.
  --include-probabilities
```

</details>

<details>
<summary><code>verify_model_artifacts.py</code></summary>

```text
Model, panel and label artifact hashes verified
```

</details>

<details>
<summary><code>bind_analysis_inputs.py</code></summary>

```text
usage: bind_analysis_inputs.py [-h] --template TEMPLATE --output OUTPUT
                               [--file KEY=PATH]

Bind local permitted files to a JSON analysis template using SHA256 digests.

options:
  -h, --help           show this help message and exit
  --template TEMPLATE
  --output OUTPUT
  --file KEY=PATH
```

</details>

<details>
<summary><code>validate_analysis_inputs.py</code></summary>

```text
usage: validate_analysis_inputs.py [-h] --config CONFIG

Validate a bound input manifest before an analysis or cluster submission.

options:
  -h, --help       show this help message and exit
  --config CONFIG
```

</details>


## Development and consensus panel

<details>
<summary><code>build_classification_manifest_hm450.py</code></summary>

```text
usage: build_classification_manifest_hm450.py [-h] --probe-matrix PROBE_MATRIX
                                              --manifest MANIFEST [--out OUT]
                                              [--counts-out COUNTS_OUT]
                                              [--sample-col SAMPLE_COL]
                                              [--patient-col PATIENT_COL]
                                              [--project-col PROJECT_COL]
                                              [--sample-type-col SAMPLE_TYPE_COL]
                                              [--primary-only]
                                              [--no-primary-only]
                                              [--source SOURCE]

Build classification manifest for HM450 probe matrix.

options:
  -h, --help            show this help message and exit
  --probe-matrix PROBE_MATRIX
                        Aggregated HM450 probe matrix parquet.
  --manifest MANIFEST   Filtered manifest TSV (HM450).
  --out OUT             Output manifest TSV.
  --counts-out COUNTS_OUT
                        Output counts TSV.
  --sample-col SAMPLE_COL
                        Manifest column that matches probe matrix sample IDs.
  --patient-col PATIENT_COL
                        Manifest patient ID column.
  --project-col PROJECT_COL
                        Manifest project (class label) column.
  --sample-type-col SAMPLE_TYPE_COL
                        Manifest sample type column.
  --primary-only        Keep only Primary samples (default: true).
  --no-primary-only     Disable Primary-only filtering.
  --source SOURCE       Source label to attach to output rows.
```

</details>

<details>
<summary><code>build_dev_lockbox_split.py</code></summary>

```text
usage: build_dev_lockbox_split.py [-h] [--manifest MANIFEST]
                                  [--label-col LABEL_COL]
                                  [--group-col GROUP_COL]
                                  [--sample-col SAMPLE_COL]
                                  [--lockbox-fraction LOCKBOX_FRACTION]
                                  [--seed SEED] [--out-dir OUT_DIR]
                                  [--prefix PREFIX]

Create stratified dev/lockbox split from a classification manifest.

options:
  -h, --help            show this help message and exit
  --manifest MANIFEST   Input manifest TSV.
  --label-col LABEL_COL
                        Class label column (default: project_id).
  --group-col GROUP_COL
                        Grouping column to avoid leakage (default:
                        patient_id).
  --sample-col SAMPLE_COL
                        Sample ID column (default: sample_id).
  --lockbox-fraction LOCKBOX_FRACTION
                        Fraction for lockbox split (default: 0.2).
  --seed SEED           Random seed (default: 42).
  --out-dir OUT_DIR     Output directory for generated split files.
  --prefix PREFIX       Output file prefix.
```

</details>

<details>
<summary><code>build_outer_folds.py</code></summary>

```text
usage: build_outer_folds.py [-h] --manifest MANIFEST --out OUT
                            [--n-splits N_SPLITS] [--n-repeats N_REPEATS]
                            [--seed SEED] [--sample-col SAMPLE_COL]
                            [--project-col PROJECT_COL]
                            [--group-col GROUP_COL]

Build outer CV folds.

options:
  -h, --help            show this help message and exit
  --manifest MANIFEST   Pilot manifest TSV.
  --out OUT             Output folds TSV.
  --n-splits N_SPLITS   Number of folds.
  --n-repeats N_REPEATS
                        Number of repeats.
  --seed SEED           Random seed.
  --sample-col SAMPLE_COL
                        Sample ID column.
  --project-col PROJECT_COL
                        Class label column.
  --group-col GROUP_COL
                        Grouping column.
```

</details>

<details>
<summary><code>run_paper_k_pipeline.py</code></summary>

```text
usage: run_paper_k_pipeline.py [-h] --config CONFIG --k K [--out-dir OUT_DIR]
                               [--seed SEED]

Prepare one-K paper protocol run.

options:
  -h, --help         show this help message and exit
  --config CONFIG
  --k K
  --out-dir OUT_DIR
  --seed SEED
```

</details>

<details>
<summary><code>sample_paper_search_space.py</code></summary>

```text
usage: sample_paper_search_space.py [-h] --config CONFIG --k K --out-dir
                                    OUT_DIR [--seed SEED]

Sample FS/model random-search specs for the paper pipeline.

options:
  -h, --help         show this help message and exit
  --config CONFIG
  --k K
  --out-dir OUT_DIR
  --seed SEED
```

</details>

<details>
<summary><code>build_protocolb_paper_manifest.py</code></summary>

```text
usage: build_protocolb_paper_manifest.py [-h] --source-manifest
                                         SOURCE_MANIFEST --out-dir OUT_DIR
                                         [--prefix PREFIX]
                                         [--label-col LABEL_COL]
                                         [--min-class-n MIN_CLASS_N]

Build paper-protocol TCGA manifest from current Protocol B manifest.

options:
  -h, --help            show this help message and exit
  --source-manifest SOURCE_MANIFEST
  --out-dir OUT_DIR
  --prefix PREFIX
  --label-col LABEL_COL
  --min-class-n MIN_CLASS_N
```

</details>

<details>
<summary><code>build_paper_stage2_locked_manifests.py</code></summary>

```text
usage: build_paper_stage2_locked_manifests.py [-h] --execution-plan
                                              EXECUTION_PLAN
                                              [--winner-json WINNER_JSON]

Build stage2_confirm manifests from stage1 winner.

options:
  -h, --help            show this help message and exit
  --execution-plan EXECUTION_PLAN
  --winner-json WINNER_JSON
                        Override Stage 1 winner JSON path.
```

</details>

<details>
<summary><code>run_paper_fs_task.py</code></summary>

```text
usage: run_paper_fs_task.py [-h] --execution-plan EXECUTION_PLAN
                            --task-manifest TASK_MANIFEST
                            [--task-index TASK_INDEX] [--compute-auroc]
                            [--n-jobs N_JOBS]

Run one FS task from fs_fold_task_manifest.tsv

options:
  -h, --help            show this help message and exit
  --execution-plan EXECUTION_PLAN
                        execution_plan.json from run_paper_k_pipeline.py
  --task-manifest TASK_MANIFEST
                        fs_fold_task_manifest.tsv
  --task-index TASK_INDEX
                        0-based task index. Defaults to $SLURM_ARRAY_TASK_ID
                        if present.
  --compute-auroc       Enable AUC computation for the FS classifier.
  --n-jobs N_JOBS       Parallel workers for compatible selectors.
```

</details>

<details>
<summary><code>run_fs_method_benchmark.py</code></summary>

```text
usage: run_fs_method_benchmark.py [-h] --probe-matrix PROBE_MATRIX --manifest
                                  MANIFEST --folds FOLDS --out-dir OUT_DIR
                                  --repeat REPEAT --fold FOLD [--k K]
                                  [--prefilter-method PREFILTER_METHOD]
                                  [--prefilter-k PREFILTER_K]
                                  [--candidate-spec-json CANDIDATE_SPEC_JSON]
                                  [--methods METHODS]
                                  [--chunk-size CHUNK_SIZE]
                                  [--inner-folds INNER_FOLDS] [--seed SEED]
                                  [--c-grid C_GRID]
                                  [--sample-missing-thresh SAMPLE_MISSING_THRESH]
                                  [--probe-missing-thresh PROBE_MISSING_THRESH]
                                  --blacklist-chen BLACKLIST_CHEN
                                  --blacklist-snp BLACKLIST_SNP
                                  --probe-metadata PROBE_METADATA
                                  [--exclude-sex-chr] [--no-exclude-sex-chr]
                                  [--pairwise-top-per-pair PAIRWISE_TOP_PER_PAIR]
                                  [--hardpair-cv-folds HARDPAIR_CV_FOLDS]
                                  [--hardpair-baseline-topk HARDPAIR_BASELINE_TOPK]
                                  [--hardpair-gamma HARDPAIR_GAMMA]
                                  [--stable-n-boot STABLE_N_BOOT]
                                  [--stable-subsample-frac STABLE_SUBSAMPLE_FRAC]
                                  [--stable-alpha STABLE_ALPHA]
                                  [--n-jobs N_JOBS] [--compute-auroc]
                                  [--run-id RUN_ID]
                                  [--heartbeat-path HEARTBEAT_PATH]
                                  [--heartbeat-interval HEARTBEAT_INTERVAL]

Feature-selection runtime benchmark on one fold.

options:
  -h, --help            show this help message and exit
  --probe-matrix PROBE_MATRIX
                        Unfiltered probe matrix parquet.
  --manifest MANIFEST   Classification manifest TSV.
  --folds FOLDS         Outer folds TSV (with repeat).
  --out-dir OUT_DIR     Run output directory.
  --repeat REPEAT       Repeat index.
  --fold FOLD           Outer fold index.
  --k K                 Final probe count per method.
  --prefilter-method PREFILTER_METHOD
                        Mutual-information prefilter used in SM2.
  --prefilter-k PREFILTER_K
                        Prefilter size before heavy selectors.
  --candidate-spec-json CANDIDATE_SPEC_JSON
                        Optional candidate spec JSON path for freeform
                        selection/prefilter args.
  --methods METHODS     Comma-separated selectors: rare_class_rank_shift,mi_ra
                        nk_baseline,pairwise_hard_weighted_auc_stable.
  --chunk-size CHUNK_SIZE
                        Probe chunk size for ANOVA scanning.
  --inner-folds INNER_FOLDS
                        Inner CV folds for classifier C tuning.
  --seed SEED           Random seed.
  --c-grid C_GRID       Classifier C grid.
  --sample-missing-thresh SAMPLE_MISSING_THRESH
                        Sample missingness threshold.
  --probe-missing-thresh PROBE_MISSING_THRESH
                        Probe missingness threshold.
  --blacklist-chen BLACKLIST_CHEN
                        Chen cross-reactive blacklist CSV.
  --blacklist-snp BLACKLIST_SNP
                        minfi SNP-affected probes list.
  --probe-metadata PROBE_METADATA
                        Probe metadata TSV with chromosome column.
  --exclude-sex-chr     Exclude chrX/chrY probes.
  --no-exclude-sex-chr
  --pairwise-top-per-pair PAIRWISE_TOP_PER_PAIR
                        Top probes retained per class-pair for pairwise ANOVA
                        aggregation.
  --hardpair-cv-folds HARDPAIR_CV_FOLDS
                        CV folds to estimate hard class-pair weights on
                        training data.
  --hardpair-baseline-topk HARDPAIR_BASELINE_TOPK
                        Top ANOVA-prefilter probes used by hard-pair baseline
                        confusion model.
  --hardpair-gamma HARDPAIR_GAMMA
                        Amplification factor for hard-pair weighting.
  --stable-n-boot STABLE_N_BOOT
                        Bootstrap/subsample rounds for stability-aware
                        pairwise selector.
  --stable-subsample-frac STABLE_SUBSAMPLE_FRAC
                        Per-class subsample fraction for stability rounds.
  --stable-alpha STABLE_ALPHA
                        Mixing weight between base pairwise score and
                        stability frequency.
  --n-jobs N_JOBS       Worker count.
  --compute-auroc       Compute macro and per-class AUC.
  --run-id RUN_ID       Run identifier for heartbeat tracking.
  --heartbeat-path HEARTBEAT_PATH
                        Path to JSONL heartbeat file for cluster monitoring.
  --heartbeat-interval HEARTBEAT_INTERVAL
                        Heartbeat emission interval in seconds.
```

</details>

<details>
<summary><code>aggregate_fs_method_benchmark.py</code></summary>

```text
usage: aggregate_fs_method_benchmark.py [-h] --run-dir RUN_DIR
                                        [--metrics-out METRICS_OUT]
                                        [--timing-out TIMING_OUT]

Aggregate FS benchmark outputs.

options:
  -h, --help            show this help message and exit
  --run-dir RUN_DIR     Run directory with repeat*/fold*/method outputs.
  --metrics-out METRICS_OUT
                        Summary metrics output TSV.
  --timing-out TIMING_OUT
                        Summary timing output TSV.
```

</details>

<details>
<summary><code>aggregate_paper_fs_candidates.py</code></summary>

```text
usage: aggregate_paper_fs_candidates.py [-h] --execution-plan EXECUTION_PLAN
                                        [--stage STAGE] [--out-tsv OUT_TSV]

Aggregate FS outputs for paper candidates.

options:
  -h, --help            show this help message and exit
  --execution-plan EXECUTION_PLAN
                        execution_plan.json from run_paper_k_pipeline.py
  --stage STAGE         Stage name in execution plan.
  --out-tsv OUT_TSV     Override summary TSV path.
```

</details>

<details>
<summary><code>run_paper_model_task.py</code></summary>

```text
usage: run_paper_model_task.py [-h] --execution-plan EXECUTION_PLAN
                               --task-manifest TASK_MANIFEST
                               [--task-index TASK_INDEX] [--compute-auroc]
                               [--n-jobs N_JOBS] [--fit-decision-policy]
                               [--no-fit-decision-policy]

Run one model task from model_fold_task_manifest.tsv

options:
  -h, --help            show this help message and exit
  --execution-plan EXECUTION_PLAN
                        execution_plan.json from run_paper_k_pipeline.py
  --task-manifest TASK_MANIFEST
                        model_fold_task_manifest.tsv
  --task-index TASK_INDEX
                        0-based task index. Defaults to $SLURM_ARRAY_TASK_ID
                        if present.
  --compute-auroc       Enable AUC computation.
  --n-jobs N_JOBS       Parallel workers for compatible models.
  --fit-decision-policy
                        Fit decision policy during model benchmarking
                        (default: enabled).
  --no-fit-decision-policy
                        Disable decision-policy fitting.
```

</details>

<details>
<summary><code>run_model_benchmark_from_selected.py</code></summary>

```text
usage: run_model_benchmark_from_selected.py [-h] --probe-matrix PROBE_MATRIX
                                            --manifest MANIFEST --folds FOLDS
                                            --selected-run-dir
                                            SELECTED_RUN_DIR
                                            [--selection-method SELECTION_METHOD]
                                            [--k K] --out-dir OUT_DIR --repeat
                                            REPEAT --fold FOLD
                                            [--candidate-spec-json CANDIDATE_SPEC_JSON]
                                            [--models MODELS]
                                            [--class-weight-mode {none,balanced}]
                                            [--tune-metric {macro_f1,weighted_f1,accuracy,balanced_accuracy}]
                                            [--inner-folds INNER_FOLDS]
                                            [--seed SEED]
                                            [--lr-c-grid LR_C_GRID]
                                            [--lr-l1-grid LR_L1_GRID]
                                            [--svm-c-grid SVM_C_GRID]
                                            [--et-n-estimators-grid ET_N_ESTIMATORS_GRID]
                                            [--et-max-features-grid ET_MAX_FEATURES_GRID]
                                            [--et-max-depth-grid ET_MAX_DEPTH_GRID]
                                            [--et-min-samples-leaf-grid ET_MIN_SAMPLES_LEAF_GRID]
                                            [--rf-n-estimators-grid RF_N_ESTIMATORS_GRID]
                                            [--rf-max-features-grid RF_MAX_FEATURES_GRID]
                                            [--rf-max-depth-grid RF_MAX_DEPTH_GRID]
                                            [--rf-min-samples-leaf-grid RF_MIN_SAMPLES_LEAF_GRID]
                                            [--n-jobs N_JOBS]
                                            [--compute-auroc]
                                            [--fit-decision-policy]
                                            [--decision-policy-metric {,macro_f1,weighted_f1,accuracy,balanced_accuracy}]
                                            [--decision-policy-modes DECISION_POLICY_MODES]
                                            [--decision-policy-min-gain DECISION_POLICY_MIN_GAIN]
                                            [--run-id RUN_ID]
                                            [--heartbeat-path HEARTBEAT_PATH]
                                            [--heartbeat-interval HEARTBEAT_INTERVAL]

Benchmark classifiers using pre-selected probes for one outer fold.

options:
  -h, --help            show this help message and exit
  --probe-matrix PROBE_MATRIX
                        Unfiltered probe matrix parquet.
  --manifest MANIFEST   Classification manifest TSV.
  --folds FOLDS         Outer folds TSV.
  --selected-run-dir SELECTED_RUN_DIR
                        Run dir containing selected_probes files.
  --selection-method SELECTION_METHOD
                        Selection method folder name.
  --k K                 Probe count used for selected probes file.
  --out-dir OUT_DIR     Output directory for model benchmark.
  --repeat REPEAT       Repeat index.
  --fold FOLD           Outer fold index.
  --candidate-spec-json CANDIDATE_SPEC_JSON
                        Optional candidate spec JSON path for freeform model
                        args.
  --models MODELS       Comma-separated classifier families: extra_trees,rando
                        m_forest,linear_svm,elastic_net_multinomial_logistic.
  --class-weight-mode {none,balanced}
                        Training class-weight mode for models that support it.
  --tune-metric {macro_f1,weighted_f1,accuracy,balanced_accuracy}
                        Metric optimized in inner CV tuning.
  --inner-folds INNER_FOLDS
                        Inner CV folds for model tuning.
  --seed SEED           Random seed.
  --lr-c-grid LR_C_GRID
                        C grid for LR models.
  --lr-l1-grid LR_L1_GRID
                        l1_ratio grid for LR elastic-net.
  --svm-c-grid SVM_C_GRID
                        C grid for linear SVM.
  --et-n-estimators-grid ET_N_ESTIMATORS_GRID
                        ExtraTrees n_estimators grid.
  --et-max-features-grid ET_MAX_FEATURES_GRID
                        ExtraTrees max_features grid.
  --et-max-depth-grid ET_MAX_DEPTH_GRID
                        ExtraTrees max_depth grid (use 0 for None).
  --et-min-samples-leaf-grid ET_MIN_SAMPLES_LEAF_GRID
                        ExtraTrees min_samples_leaf grid.
  --rf-n-estimators-grid RF_N_ESTIMATORS_GRID
                        RandomForest n_estimators grid.
  --rf-max-features-grid RF_MAX_FEATURES_GRID
                        RandomForest max_features grid.
  --rf-max-depth-grid RF_MAX_DEPTH_GRID
                        RandomForest max_depth grid (use 0 for None).
  --rf-min-samples-leaf-grid RF_MIN_SAMPLES_LEAF_GRID
                        RandomForest min_samples_leaf grid.
  --n-jobs N_JOBS       Parallel workers for tree fitting.
  --compute-auroc       Compute macro/per-class AUC.
  --fit-decision-policy
                        Fit a post-probability decision policy from dev OOF
                        probabilities before test evaluation.
  --decision-policy-metric {,macro_f1,weighted_f1,accuracy,balanced_accuracy}
                        Metric optimized when fitting the decision policy.
                        Defaults to --tune-metric.
  --decision-policy-modes DECISION_POLICY_MODES
                        Comma-separated decision policy modes to consider.
  --decision-policy-min-gain DECISION_POLICY_MIN_GAIN
                        Minimum OOF gain over raw argmax required to keep a
                        non-argmax decision policy.
  --run-id RUN_ID       Run identifier for heartbeat tracking.
  --heartbeat-path HEARTBEAT_PATH
                        Path to JSONL heartbeat file for cluster monitoring.
  --heartbeat-interval HEARTBEAT_INTERVAL
                        Heartbeat emission interval in seconds.
```

</details>

<details>
<summary><code>aggregate_model_benchmark.py</code></summary>

```text
usage: aggregate_model_benchmark.py [-h] --run-dir RUN_DIR
                                    [--metrics-out METRICS_OUT]
                                    [--timing-out TIMING_OUT]

Aggregate model benchmark outputs.

options:
  -h, --help            show this help message and exit
  --run-dir RUN_DIR     Run directory with repeat*/fold*/model outputs.
  --metrics-out METRICS_OUT
                        Summary metrics output TSV.
  --timing-out TIMING_OUT
                        Summary timing output TSV.
```

</details>

<details>
<summary><code>aggregate_paper_model_candidates.py</code></summary>

```text
usage: aggregate_paper_model_candidates.py [-h] --execution-plan
                                           EXECUTION_PLAN [--stage STAGE]
                                           [--out-tsv OUT_TSV]

Aggregate model outputs for paper joint candidates.

options:
  -h, --help            show this help message and exit
  --execution-plan EXECUTION_PLAN
                        execution_plan.json from run_paper_k_pipeline.py
  --stage STAGE         Stage name in execution plan.
  --out-tsv OUT_TSV     Override summary TSV path.
```

</details>

<details>
<summary><code>select_best_paper_candidate.py</code></summary>

```text
usage: select_best_paper_candidate.py [-h] --execution-plan EXECUTION_PLAN
                                      [--stage STAGE] [--out-json OUT_JSON]
                                      [--ranking-tsv RANKING_TSV]

Select best paper candidate from aggregated dev summaries.

options:
  -h, --help            show this help message and exit
  --execution-plan EXECUTION_PLAN
                        execution_plan.json from run_paper_k_pipeline.py
  --stage STAGE         Stage name in execution plan.
  --out-json OUT_JSON   Override winning-candidate JSON path.
  --ranking-tsv RANKING_TSV
                        Override candidate ranking TSV path.
```

</details>

<details>
<summary><code>build_stable_panel_v1.py</code></summary>

```text
usage: build_stable_panel_v1.py [-h] --selected-run-dir SELECTED_RUN_DIR
                                [--method METHOD] [--rank-k RANK_K]
                                [--panel-k PANEL_K] [--out-dir OUT_DIR]
                                [--freq-weight FREQ_WEIGHT]
                                [--rank-weight RANK_WEIGHT]

Build stable panel v1 from fold-level selected probe files.

options:
  -h, --help            show this help message and exit
  --selected-run-dir SELECTED_RUN_DIR
                        Run dir containing repeat*/fold*/<method>/ files.
  --method METHOD       Selection method folder under each fold (default:
                        rare_class_rank_shift).
  --rank-k RANK_K       Max ranked probes to load from
                        selected_order_K<k>.tsv.
  --panel-k PANEL_K     Target stable panel size and top-K frequency
                        definition.
  --out-dir OUT_DIR     Output directory. Default: <selected-run-
                        dir>/stable_panel_v1
  --freq-weight FREQ_WEIGHT
                        Weight for top-K frequency in composite score
                        (default: 0.7).
  --rank-weight RANK_WEIGHT
                        Weight for normalized Borda rank score in composite
                        score (default: 0.3).
```

</details>

<details>
<summary><code>evaluate_lockbox_from_panel.py</code></summary>

```text
usage: evaluate_lockbox_from_panel.py [-h] --probe-matrix PROBE_MATRIX
                                      --dev-manifest DEV_MANIFEST
                                      --lockbox-manifest LOCKBOX_MANIFEST
                                      --panel-tsv PANEL_TSV [--k K] --out-dir
                                      OUT_DIR
                                      [--model {lr_multinomial,svm_linear,extra_trees,random_forest,linear_svm,elastic_net_multinomial_logistic}]
                                      [--class-weight-mode {none,balanced}]
                                      [--tune-metric {macro_f1,weighted_f1,accuracy,balanced_accuracy}]
                                      [--seed SEED]
                                      [--inner-folds INNER_FOLDS]
                                      [--lr-c-grid LR_C_GRID]
                                      [--lr-l1-grid LR_L1_GRID]
                                      [--svm-c-grid SVM_C_GRID]
                                      [--et-n-estimators-grid ET_N_ESTIMATORS_GRID]
                                      [--et-max-features-grid ET_MAX_FEATURES_GRID]
                                      [--et-max-depth-grid ET_MAX_DEPTH_GRID]
                                      [--et-min-samples-leaf-grid ET_MIN_SAMPLES_LEAF_GRID]
                                      [--rf-n-estimators-grid RF_N_ESTIMATORS_GRID]
                                      [--rf-max-features-grid RF_MAX_FEATURES_GRID]
                                      [--rf-max-depth-grid RF_MAX_DEPTH_GRID]
                                      [--rf-min-samples-leaf-grid RF_MIN_SAMPLES_LEAF_GRID]
                                      [--n-jobs N_JOBS] [--compute-auroc]
                                      [--fit-decision-policy]
                                      [--decision-policy-metric {,macro_f1,weighted_f1,accuracy,balanced_accuracy}]
                                      [--decision-policy-modes DECISION_POLICY_MODES]
                                      [--decision-policy-min-gain DECISION_POLICY_MIN_GAIN]

One-shot lockbox evaluation from a fixed panel.

options:
  -h, --help            show this help message and exit
  --probe-matrix PROBE_MATRIX
                        Probe matrix parquet path.
  --dev-manifest DEV_MANIFEST
                        Dev manifest TSV (sample_id, project_id).
  --lockbox-manifest LOCKBOX_MANIFEST
                        Lockbox manifest TSV (sample_id, project_id).
  --panel-tsv PANEL_TSV
                        Panel TSV with probe_id column.
  --k K                 Probe count to use from panel.
  --out-dir OUT_DIR     Output directory.
  --model {lr_multinomial,svm_linear,extra_trees,random_forest,linear_svm,elastic_net_multinomial_logistic}
                        Model family to tune on full dev then evaluate on
                        lockbox.
  --class-weight-mode {none,balanced}
                        Training class-weight mode for models that support it.
  --tune-metric {macro_f1,weighted_f1,accuracy,balanced_accuracy}
                        Metric optimized in inner CV tuning.
  --seed SEED           Random seed.
  --inner-folds INNER_FOLDS
                        Inner CV folds for tuning on dev.
  --lr-c-grid LR_C_GRID
                        LR C grid.
  --lr-l1-grid LR_L1_GRID
                        LR l1_ratio grid (elastic-net).
  --svm-c-grid SVM_C_GRID
                        Linear SVM C grid.
  --et-n-estimators-grid ET_N_ESTIMATORS_GRID
                        ExtraTrees n_estimators grid.
  --et-max-features-grid ET_MAX_FEATURES_GRID
                        ExtraTrees max_features grid.
  --et-max-depth-grid ET_MAX_DEPTH_GRID
                        ExtraTrees max_depth grid (use 0 for None).
  --et-min-samples-leaf-grid ET_MIN_SAMPLES_LEAF_GRID
                        ExtraTrees min_samples_leaf grid.
  --rf-n-estimators-grid RF_N_ESTIMATORS_GRID
                        RandomForest n_estimators grid.
  --rf-max-features-grid RF_MAX_FEATURES_GRID
                        RandomForest max_features grid.
  --rf-max-depth-grid RF_MAX_DEPTH_GRID
                        RandomForest max_depth grid (use 0 for None).
  --rf-min-samples-leaf-grid RF_MIN_SAMPLES_LEAF_GRID
                        RandomForest min_samples_leaf grid.
  --n-jobs N_JOBS       Parallel workers for tree fitting.
  --compute-auroc       Compute macro and per-class AUC.
  --fit-decision-policy
                        Fit a post-probability decision policy from dev OOF
                        probabilities before lockbox evaluation.
  --decision-policy-metric {,macro_f1,weighted_f1,accuracy,balanced_accuracy}
                        Metric optimized for the decision policy. Defaults to
                        --tune-metric.
  --decision-policy-modes DECISION_POLICY_MODES
                        Comma-separated decision policy modes to consider.
  --decision-policy-min-gain DECISION_POLICY_MIN_GAIN
                        Minimum OOF gain over raw argmax required to keep a
                        non-argmax decision policy.
```

</details>


## External evaluation and single-split panels

<details>
<summary><code>build_external_panel_matrix.py</code></summary>

```text
usage: build_external_panel_matrix.py [-h] --mapping-tsv MAPPING_TSV
                                      --panel-tsv PANEL_TSV [--k K] --out-dir
                                      OUT_DIR [--keep-all-mapping-rows]

Build external panel matrix from source files + sample-column mapping.

options:
  -h, --help            show this help message and exit
  --mapping-tsv MAPPING_TSV
                        external_sample_column_mapping.tsv
  --panel-tsv PANEL_TSV
                        Panel TSV with probe_id column
  --k K                 Use first K probes from panel
  --out-dir OUT_DIR     Output directory
  --keep-all-mapping-rows
                        Do not auto-filter the mapping TSV to include==1 rows.
                        Useful for biology/reference extraction.
```

</details>

<details>
<summary><code>sanitize_external_panel_beta_values.py</code></summary>

```text
usage: sanitize_external_panel_beta_values.py [-h] --panel-matrix PANEL_MATRIX
                                              --out-dir OUT_DIR
                                              [--max-missing-probes MAX_MISSING_PROBES]
                                              [--matrix-name MATRIX_NAME]
                                              [--all-matrix-name ALL_MATRIX_NAME]

Set external panel probe values outside [0, 1] to missing, recompute panel
missingness, and emit beta-scale QC tables.

options:
  -h, --help            show this help message and exit
  --panel-matrix PANEL_MATRIX
  --out-dir OUT_DIR
  --max-missing-probes MAX_MISSING_PROBES
  --matrix-name MATRIX_NAME
  --all-matrix-name ALL_MATRIX_NAME
```

</details>

<details>
<summary><code>predict_external_cohort.py</code></summary>

```text
usage: predict_external_cohort.py [-h] --config CONFIG --out-dir OUT_DIR

Infer canonical cohort rows and emit the evaluation runner's prediction
schema.

options:
  -h, --help         show this help message and exit
  --config CONFIG
  --out-dir OUT_DIR
```

</details>

<details>
<summary><code>run_methydeep_inference.py</code></summary>

```text
usage: run_methydeep_inference.py [-h] --panel-matrix PANEL_MATRIX
                                  --feature-csv FEATURE_CSV --class-csv
                                  CLASS_CSV --model-dir MODEL_DIR --out-dir
                                  OUT_DIR [--k K]
                                  [--max-missing-probes MAX_MISSING_PROBES]
                                  [--eligible-status-tsv ELIGIBLE_STATUS_TSV]
                                  [--target-bucket TARGET_BUCKET]

Run MethyDeep SavedModel inference on external panel matrix.

options:
  -h, --help            show this help message and exit
  --panel-matrix PANEL_MATRIX
                        TSV with sample rows and probe columns.
  --feature-csv FEATURE_CSV
                        MethyDeep RF_feature45.csv
  --class-csv CLASS_CSV
                        MethyDeep targer_names.csv
  --model-dir MODEL_DIR
                        Path to MethyDeep saved model directory.
  --out-dir OUT_DIR     Output directory
  --k K                 Use top-K features from feature CSV (default 30)
  --max-missing-probes MAX_MISSING_PROBES
                        Eligibility threshold (default 3)
  --eligible-status-tsv ELIGIBLE_STATUS_TSV
                        Optional sample-status TSV describing the exact
                        current eligible set. If provided, rows are filtered
                        to keys with extraction_status in {ok_full,
                        ok_partial} and n_panel_missing <= max-missing-probes
                        before inference.
  --target-bucket TARGET_BUCKET
                        Optional target_bucket restriction to apply when
                        filtering eligible-status TSV.
```

</details>

<details>
<summary><code>evaluate_external.py</code></summary>

```text
usage: evaluate_external.py [-h] --config CONFIG --out-dir OUT_DIR

Fixed-class metrics and paired, truth-stratified patient-cluster resampling.
Ported from the adopted primary and overlapping-domain evaluations. Class
axes, probability availability, specimen ordering and stratum ordering remain
explicit.

options:
  -h, --help         show this help message and exit
  --config CONFIG
  --out-dir OUT_DIR
```

</details>

<details>
<summary><code>evaluate_external_from_bundle.py</code></summary>

```text
usage: evaluate_external_from_bundle.py [-h] --panel-matrix PANEL_MATRIX
                                        --model-bundle MODEL_BUNDLE --out-dir
                                        OUT_DIR
                                        [--pre-impute {none,sample_mean}]

Evaluate external panel matrix using fitted model bundle.

options:
  -h, --help            show this help message and exit
  --panel-matrix PANEL_MATRIX
                        external_panel_matrix_allow_missing1.tsv
  --model-bundle MODEL_BUNDLE
                        model_bundle.pkl from lockbox one-shot training
  --out-dir OUT_DIR     Output directory
  --pre-impute {none,sample_mean}
                        Optional pre-imputation on X before model.predict.
                        'sample_mean' fills each row NaN by that row mean.
```

</details>

<details>
<summary><code>evaluate_paired_conditions.py</code></summary>

```text
usage: evaluate_paired_conditions.py [-h] --config CONFIG --out-dir OUT_DIR

Paired clinical and preprocessing contrasts using the recorded
patient/specimen design

options:
  -h, --help         show this help message and exit
  --config CONFIG
  --out-dir OUT_DIR
```

</details>

<details>
<summary><code>train_panel_comparison.py</code></summary>

```text
usage: train_panel_comparison.py [-h] --config CONFIG --out-dir OUT_DIR --task
                                 {0,1,2,3,4,5,6,7,8,9}

Train fixed consensus/single-split panels on development data, one of ten task
shards

options:
  -h, --help            show this help message and exit
  --config CONFIG
  --out-dir OUT_DIR
  --task {0,1,2,3,4,5,6,7,8,9}
```

</details>

<details>
<summary><code>summarize_panel_comparison.py</code></summary>

```text
usage: summarize_panel_comparison.py [-h] --config CONFIG --out-dir OUT_DIR
                                     [--bucket {main}]

Consensus versus single-split panels on a common scoreable subset

options:
  -h, --help         show this help message and exit
  --config CONFIG
  --out-dir OUT_DIR
  --bucket {main}
```

</details>


## Error analyses

<details>
<summary><code>analyze_class_geometry.py</code></summary>

```text
usage: analyze_class_geometry.py [-h] --config CONFIG --out-dir OUT_DIR

Arithmetic class profiles, methylation shift and class-label permutation.

options:
  -h, --help         show this help message and exit
  --config CONFIG
  --out-dir OUT_DIR
```

</details>

<details>
<summary><code>summarize_study_errors.py</code></summary>

```text
usage: summarize_study_errors.py [-h] --config CONFIG --out-dir OUT_DIR

Aggregate study/class errors and supplied source annotations without changing
eligibility.

options:
  -h, --help         show this help message and exit
  --config CONFIG
  --out-dir OUT_DIR
```

</details>

<details>
<summary><code>analyze_uvm_skcm_diagnostics.py</code></summary>

```text
usage: analyze_uvm_skcm_diagnostics.py [-h] --config CONFIG --out-dir OUT_DIR

Recorded UVM/SKCM study geometry, missing-probe and comparator diagnostics.

options:
  -h, --help         show this help message and exit
  --config CONFIG
  --out-dir OUT_DIR
```

</details>

<details>
<summary><code>analyze_heldout_purity.py</code></summary>

```text
usage: analyze_heldout_purity.py [-h] --config CONFIG --out-dir OUT_DIR

Post hoc association of specimen-matched ABSOLUTE purity with locked TCGA
errors.

options:
  -h, --help         show this help message and exit
  --config CONFIG
  --out-dir OUT_DIR
```

</details>

<details>
<summary><code>check_heldout_purity_sensitivity.py</code></summary>

```text
usage: check_heldout_purity_sensitivity.py [-h] --analysis-dir ANALYSIS_DIR

Leave-one-class-out purity sensitivity; outputs remain local

options:
  -h, --help            show this help message and exit
  --analysis-dir ANALYSIS_DIR
```

</details>


## Biological context

<details>
<summary><code>annotate_panel_probe_locations.py</code></summary>

```text
usage: annotate_panel_probe_locations.py [-h] --panel-tsv PANEL_TSV
                                         --annotation-tsv ANNOTATION_TSV
                                         --out-dir OUT_DIR [--k K]

Annotate panel probes using HM450 manifest.

options:
  -h, --help            show this help message and exit
  --panel-tsv PANEL_TSV
                        Panel TSV with probe_id column (optionally
                        panel_rank).
  --annotation-tsv ANNOTATION_TSV
                        HM450 annotation TSV path.
  --out-dir OUT_DIR     Output directory.
  --k K                 Number of panel probes to keep (after rank ordering).
```

</details>

<details>
<summary><code>annotate_panel_chromatin_context.py</code></summary>

```text
usage: annotate_panel_chromatin_context.py [-h] --panel-biology-tsv
                                           PANEL_BIOLOGY_TSV --ccre-dir
                                           CCRE_DIR
                                           [--mechanism-screen-tsv MECHANISM_SCREEN_TSV]
                                           --out-dir OUT_DIR

Annotate panel CpGs with ENCODE SCREEN GRCh38 cCRE regulatory classes.

options:
  -h, --help            show this help message and exit
  --panel-biology-tsv PANEL_BIOLOGY_TSV
  --ccre-dir CCRE_DIR
  --mechanism-screen-tsv MECHANISM_SCREEN_TSV
  --out-dir OUT_DIR
```

</details>

<details>
<summary><code>analyze_panel_context_enrichment_from_cv_prefilters.py</code></summary>

```text
usage: analyze_panel_context_enrichment_from_cv_prefilters.py
       [-h] --annotation-tsv ANNOTATION_TSV --panel-annotations-tsv
       PANEL_ANNOTATIONS_TSV --prefilter-glob PREFILTER_GLOB --out-dir OUT_DIR

Analyze panel CpG-island context enrichment using per-fold prefilter lists.

options:
  -h, --help            show this help message and exit
  --annotation-tsv ANNOTATION_TSV
                        HM450 annotation manifest TSV.
  --panel-annotations-tsv PANEL_ANNOTATIONS_TSV
                        Annotated panel TSV with context_group.
  --prefilter-glob PREFILTER_GLOB
                        Glob for per-fold prefilter TSV files
                        (mi_prefilter_topk.tsv).
  --out-dir OUT_DIR     Output directory.
```

</details>

<details>
<summary><code>analyze_panel_tumor_normal.py</code></summary>

```text
usage: analyze_panel_tumor_normal.py [-h] --tumor-matrix-tsv TUMOR_MATRIX_TSV
                                     --normal-matrix-tsv NORMAL_MATRIX_TSV
                                     --out-dir OUT_DIR
                                     [--project-ids PROJECT_IDS]

Summarize tumor-vs-normal methylation behavior for a selected probe panel.
This utility is panel-agnostic and can be reused for future panels.

options:
  -h, --help            show this help message and exit
  --tumor-matrix-tsv TUMOR_MATRIX_TSV
                        Tumor panel matrix TSV.
  --normal-matrix-tsv NORMAL_MATRIX_TSV
                        Normal/reference panel matrix TSV.
  --out-dir OUT_DIR     Output directory.
  --project-ids PROJECT_IDS
                        Optional comma-separated project IDs to keep.
```

</details>

<details>
<summary><code>analyze_panel_lineage_normals.py</code></summary>

```text
usage: analyze_panel_lineage_normals.py [-h] --panel-annotations-tsv
                                        PANEL_ANNOTATIONS_TSV --probes-tsv
                                        PROBES_TSV --tcga-normal-matrix-tsv
                                        TCGA_NORMAL_MATRIX_TSV
                                        [--external-panel-matrix-tsv EXTERNAL_PANEL_MATRIX_TSV]
                                        [--external-labels-tsv EXTERNAL_LABELS_TSV]
                                        --tcga-dev-matrix-tsv
                                        TCGA_DEV_MATRIX_TSV
                                        [--tcga-lockbox-matrix-tsv TCGA_LOCKBOX_MATRIX_TSV]
                                        [--tcga-only-lineage]
                                        [--min-normal-samples MIN_NORMAL_SAMPLES]
                                        [--min-tumor-samples MIN_TUMOR_SAMPLES]
                                        --out-dir OUT_DIR

Analyze panel probe behavior across normal/reference tissues to assess lineage
support. Panel-agnostic and reusable across future panels.

options:
  -h, --help            show this help message and exit
  --panel-annotations-tsv PANEL_ANNOTATIONS_TSV
  --probes-tsv PROBES_TSV
  --tcga-normal-matrix-tsv TCGA_NORMAL_MATRIX_TSV
  --external-panel-matrix-tsv EXTERNAL_PANEL_MATRIX_TSV
  --external-labels-tsv EXTERNAL_LABELS_TSV
  --tcga-dev-matrix-tsv TCGA_DEV_MATRIX_TSV
  --tcga-lockbox-matrix-tsv TCGA_LOCKBOX_MATRIX_TSV
  --tcga-only-lineage
  --min-normal-samples MIN_NORMAL_SAMPLES
  --min-tumor-samples MIN_TUMOR_SAMPLES
  --out-dir OUT_DIR
```

</details>

<details>
<summary><code>audit_normal_metadata_composition.py</code></summary>

```text
usage: audit_normal_metadata_composition.py [-h] --out-dir OUT_DIR
                                            --normal-inventory
                                            NORMAL_INVENTORY
                                            --normal-gdc-inventory
                                            NORMAL_GDC_INVENTORY --tumor-dev
                                            TUMOR_DEV --tumor-lockbox
                                            TUMOR_LOCKBOX --absolute ABSOLUTE
                                            --leukocyte LEUKOCYTE --cibersort
                                            CIBERSORT --quality QUALITY

Audit TCGA normal sample metadata and composition covariate availability.

options:
  -h, --help            show this help message and exit
  --out-dir OUT_DIR
  --normal-inventory NORMAL_INVENTORY
  --normal-gdc-inventory NORMAL_GDC_INVENTORY
  --tumor-dev TUMOR_DEV
  --tumor-lockbox TUMOR_LOCKBOX
  --absolute ABSOLUTE
  --leukocyte LEUKOCYTE
  --cibersort CIBERSORT
  --quality QUALITY
```

</details>

<details>
<summary><code>analyze_normal_tissue_identity.py</code></summary>

```text
usage: analyze_normal_tissue_identity.py [-h] [--mode {compute,plot,all}]
                                         --normal-audit NORMAL_AUDIT --out-dir
                                         OUT_DIR [--random-state RANDOM_STATE]
                                         [--n-estimators N_ESTIMATORS]

Module 1: normal tissue identity signal at MiniCpG probes.

options:
  -h, --help            show this help message and exit
  --mode {compute,plot,all}
  --normal-audit NORMAL_AUDIT
  --out-dir OUT_DIR
  --random-state RANDOM_STATE
  --n-estimators N_ESTIMATORS
```

</details>

<details>
<summary><code>analyze_immune_purity_probe_associations.py</code></summary>

```text
usage: analyze_immune_purity_probe_associations.py [-h] --probe-matrix-tsv
                                                   PROBE_MATRIX_TSV
                                                   --panel-annotations-tsv
                                                   PANEL_ANNOTATIONS_TSV
                                                   [--probes-tsv PROBES_TSV]
                                                   [--focus-probes-tsv FOCUS_PROBES_TSV]
                                                   --dev-manifest-tsv
                                                   DEV_MANIFEST_TSV
                                                   --purity-tsv PURITY_TSV
                                                   --leukocyte-tsv
                                                   LEUKOCYTE_TSV
                                                   --cibersort-tsv
                                                   CIBERSORT_TSV --out-dir
                                                   OUT_DIR
                                                   [--min-within-class-n MIN_WITHIN_CLASS_N]

Assess whether selected methylation probes track purity or immune covariates.

options:
  -h, --help            show this help message and exit
  --probe-matrix-tsv PROBE_MATRIX_TSV
  --panel-annotations-tsv PANEL_ANNOTATIONS_TSV
  --probes-tsv PROBES_TSV
                        Optional ordered probe list TSV. Defaults to
                        annotation-ranked probes found in the matrix.
  --focus-probes-tsv FOCUS_PROBES_TSV
                        Optional subset probe TSV to highlight in the markdown
                        summary. Defaults to all panel probes.
  --dev-manifest-tsv DEV_MANIFEST_TSV
  --purity-tsv PURITY_TSV
  --leukocyte-tsv LEUKOCYTE_TSV
  --cibersort-tsv CIBERSORT_TSV
  --out-dir OUT_DIR
  --min-within-class-n MIN_WITHIN_CLASS_N
```

</details>

<details>
<summary><code>analyze_probe_biology_decomposition.py</code></summary>

```text
usage: analyze_probe_biology_decomposition.py [-h] --out-dir OUT_DIR
                                              --normal-lineage NORMAL_LINEAGE
                                              --normal-audit NORMAL_AUDIT
                                              --biology-master BIOLOGY_MASTER
                                              --tcga-dev TCGA_DEV
                                              --tcga-lockbox TCGA_LOCKBOX
                                              --tumor-normal-dev
                                              TUMOR_NORMAL_DEV
                                              --tumor-normal-lockbox
                                              TUMOR_NORMAL_LOCKBOX
                                              --matched-dev MATCHED_DEV
                                              --matched-lockbox
                                              MATCHED_LOCKBOX --composition
                                              COMPOSITION --within-composition
                                              WITHIN_COMPOSITION
                                              [--composition-breadth-rho COMPOSITION_BREADTH_RHO]
                                              [--composition-min-samples COMPOSITION_MIN_SAMPLES]

Modules 2-5: tumor-side probe biology decomposition for MiniCpG probes.

options:
  -h, --help            show this help message and exit
  --out-dir OUT_DIR
  --normal-lineage NORMAL_LINEAGE
  --normal-audit NORMAL_AUDIT
  --biology-master BIOLOGY_MASTER
  --tcga-dev TCGA_DEV
  --tcga-lockbox TCGA_LOCKBOX
  --tumor-normal-dev TUMOR_NORMAL_DEV
  --tumor-normal-lockbox TUMOR_NORMAL_LOCKBOX
  --matched-dev MATCHED_DEV
  --matched-lockbox MATCHED_LOCKBOX
  --composition COMPOSITION
  --within-composition WITHIN_COMPOSITION
  --composition-breadth-rho COMPOSITION_BREADTH_RHO
  --composition-min-samples COMPOSITION_MIN_SAMPLES
```

</details>

<details>
<summary><code>analyze_locked_minicpg_on_normals.py</code></summary>

```text
usage: analyze_locked_minicpg_on_normals.py [-h] [--mode {compute,plot,all}]
                                            --model-bundle MODEL_BUNDLE
                                            --normal-audit NORMAL_AUDIT
                                            --tcga-dev TCGA_DEV
                                            --decomposition DECOMPOSITION
                                            --out-dir OUT_DIR

Run locked MiniCpG on TCGA solid-tissue normal samples as forced tumor
predictions.

options:
  -h, --help            show this help message and exit
  --mode {compute,plot,all}
  --model-bundle MODEL_BUNDLE
  --normal-audit NORMAL_AUDIT
  --tcga-dev TCGA_DEV
  --decomposition DECOMPOSITION
  --out-dir OUT_DIR
```

</details>

<details>
<summary><code>run_matched_normal_substitution_minicpg.py</code></summary>

```text
usage: run_matched_normal_substitution_minicpg.py [-h] --model-bundle
                                                  MODEL_BUNDLE
                                                  --lockbox-tumor-matrix
                                                  LOCKBOX_TUMOR_MATRIX
                                                  --lockbox-manifest
                                                  LOCKBOX_MANIFEST
                                                  --normal-matrix
                                                  NORMAL_MATRIX --out-dir
                                                  OUT_DIR

Run same-patient normal-adjacent substitution through the locked MiniCpG
model.

options:
  -h, --help            show this help message and exit
  --model-bundle MODEL_BUNDLE
  --lockbox-tumor-matrix LOCKBOX_TUMOR_MATRIX
  --lockbox-manifest LOCKBOX_MANIFEST
  --normal-matrix NORMAL_MATRIX
  --out-dir OUT_DIR
```

</details>

<details>
<summary><code>run_tumor_gate_proof_of_concept.py</code></summary>

```text
usage: run_tumor_gate_proof_of_concept.py [-h]
                                          [--active-manifest ACTIVE_MANIFEST]
                                          --dev-tumor DEV_TUMOR --test-tumor
                                          TEST_TUMOR --normal NORMAL
                                          --model-bundle MODEL_BUNDLE
                                          --out-dir OUT_DIR [--seed SEED]
                                          [--n-threshold-repeats N_THRESHOLD_REPEATS]
                                          [--threshold-calibration-size THRESHOLD_CALIBRATION_SIZE]
                                          [--target-sensitivity TARGET_SENSITIVITY]

Run a tumor-vs-normal-adjacent gate proof of concept on locked MiniCpG probes.

options:
  -h, --help            show this help message and exit
  --active-manifest ACTIVE_MANIFEST
  --dev-tumor DEV_TUMOR
  --test-tumor TEST_TUMOR
  --normal NORMAL
  --model-bundle MODEL_BUNDLE
  --out-dir OUT_DIR
  --seed SEED
  --n-threshold-repeats N_THRESHOLD_REPEATS
  --threshold-calibration-size THRESHOLD_CALIBRATION_SIZE
  --target-sensitivity TARGET_SENSITIVITY
```

</details>

<details>
<summary><code>analyze_panel_meth_expr_transcriptome_null.py</code></summary>

```text
usage: analyze_panel_meth_expr_transcriptome_null.py [-h]
                                                     --probe-gene-pairs-tsv
                                                     PROBE_GENE_PAIRS_TSV
                                                     --expression-metadata-tsv
                                                     EXPRESSION_METADATA_TSV
                                                     --tcga-dev-matrix-tsv
                                                     TCGA_DEV_MATRIX_TSV
                                                     --tcga-lockbox-matrix-tsv
                                                     TCGA_LOCKBOX_MATRIX_TSV
                                                     --out-dir OUT_DIR
                                                     [--expression-field EXPRESSION_FIELD]
                                                     [--min-matched MIN_MATCHED]
                                                     [--min-gene-nonzero-fraction MIN_GENE_NONZERO_FRACTION]
                                                     [--max-projects MAX_PROJECTS]
                                                     [--focus-projects FOCUS_PROJECTS]
                                                     [--project-list-out PROJECT_LIST_OUT]
                                                     [--shard-by-project]
                                                     [--save-full-correlations-parquet]
                                                     [--top-n-per-probe-project TOP_N_PER_PROBE_PROJECT]

For each TCGA cohort and each panel CpG, correlate tumor methylation against
all available RNA-seq genes and rank the annotated linked genes against that
transcriptome-wide background.

options:
  -h, --help            show this help message and exit
  --probe-gene-pairs-tsv PROBE_GENE_PAIRS_TSV
  --expression-metadata-tsv EXPRESSION_METADATA_TSV
  --tcga-dev-matrix-tsv TCGA_DEV_MATRIX_TSV
  --tcga-lockbox-matrix-tsv TCGA_LOCKBOX_MATRIX_TSV
  --out-dir OUT_DIR
  --expression-field EXPRESSION_FIELD
  --min-matched MIN_MATCHED
  --min-gene-nonzero-fraction MIN_GENE_NONZERO_FRACTION
  --max-projects MAX_PROJECTS
                        Debug limit after sorting projects.
  --focus-projects FOCUS_PROJECTS
                        Optional comma-separated TCGA project IDs.
  --project-list-out PROJECT_LIST_OUT
                        Write eligible projects and exit.
  --shard-by-project    Write per-project output shards under out-dir instead
                        of combined final outputs.
  --save-full-correlations-parquet
                        Write all probe-gene-cohort correlations to compressed
                        Parquet.
  --top-n-per-probe-project TOP_N_PER_PROBE_PROJECT
                        Number of top |rho| transcriptome hits to save per
                        probe/cohort.
```

</details>

<details>
<summary><code>aggregate_panel_meth_expr_transcriptome_null_shards.py</code></summary>

```text
usage: aggregate_panel_meth_expr_transcriptome_null_shards.py
       [-h] --out-dir OUT_DIR [--shard-root SHARD_ROOT]

Aggregate per-cohort transcriptome-wide methylation-expression null shards.

options:
  -h, --help            show this help message and exit
  --out-dir OUT_DIR     Transcriptome-null root output directory.
  --shard-root SHARD_ROOT
                        Defaults to out-dir/project_shards.
```

</details>

<details>
<summary><code>summarize_panel_meth_expr_transcriptome_null.py</code></summary>

```text
usage: summarize_panel_meth_expr_transcriptome_null.py [-h]
                                                       --transcriptome-null-dir
                                                       TRANSCRIPTOME_NULL_DIR
                                                       --out-dir OUT_DIR
                                                       [--top-n TOP_N]
                                                       [--min-matched MIN_MATCHED]
                                                       [--nominal-q-cutoff NOMINAL_Q_CUTOFF]
                                                       [--empirical-p-cutoff EMPIRICAL_P_CUTOFF]
                                                       [--stringent-empirical-p-cutoff STRINGENT_EMPIRICAL_P_CUTOFF]

Summarize linked probe-gene methylation-expression ranks against
transcriptome-wide nulls.

options:
  -h, --help            show this help message and exit
  --transcriptome-null-dir TRANSCRIPTOME_NULL_DIR
                        Directory containing
                        linked_probe_gene_transcriptome_rank_by_project.tsv.
  --out-dir OUT_DIR     Output directory for inspection summaries.
  --top-n TOP_N         Number of top linked rows to export.
  --min-matched MIN_MATCHED
                        Minimum matched methylation/RNA samples required for
                        null-aware support calls.
  --nominal-q-cutoff NOMINAL_Q_CUTOFF
                        Maximum within-probe/cohort methylation-expression
                        q-value for null-aware support calls.
  --empirical-p-cutoff EMPIRICAL_P_CUTOFF
                        Maximum transcriptome-rank empirical p-value for an
                        interesting linked-gene support row.
  --stringent-empirical-p-cutoff STRINGENT_EMPIRICAL_P_CUTOFF
                        Maximum transcriptome-rank empirical p-value for a
                        stringent linked-gene support row.
```

</details>

<details>
<summary><code>analyze_locked_model_shap_and_correlation.py</code></summary>

```text
usage: analyze_locked_model_shap_and_correlation.py [-h] --model-bundle
                                                    MODEL_BUNDLE
                                                    --lockbox-panel-tsv
                                                    LOCKBOX_PANEL_TSV
                                                    --external-panel-tsv
                                                    EXTERNAL_PANEL_TSV
                                                    --external-predictions-tsv
                                                    EXTERNAL_PREDICTIONS_TSV
                                                    --out-dir OUT_DIR
                                                    [--top-n-pairs TOP_N_PAIRS]
                                                    [--top-n-classes TOP_N_CLASSES]
                                                    [--approximate-shap]
                                                    [--exact-shap]

options:
  -h, --help            show this help message and exit
  --model-bundle MODEL_BUNDLE
  --lockbox-panel-tsv LOCKBOX_PANEL_TSV
  --external-panel-tsv EXTERNAL_PANEL_TSV
  --external-predictions-tsv EXTERNAL_PREDICTIONS_TSV
  --out-dir OUT_DIR
  --top-n-pairs TOP_N_PAIRS
  --top-n-classes TOP_N_CLASSES
  --approximate-shap
  --exact-shap
```

</details>


## Cell-free DNA

<details>
<summary><code>filter_cfmethdb_presence_candidates_by_study_fraction.py</code></summary>

```text
usage: filter_cfmethdb_presence_candidates_by_study_fraction.py
       [-h] --merged-dir MERGED_DIR --out-dir OUT_DIR --study-id STUDY_ID
       --study-denominator STUDY_DENOMINATOR
       [--min-study-fraction MIN_STUDY_FRACTION]
       [--expected-shards EXPECTED_SHARDS]

Filter merged cfMethDB presence atlas by one study's label-blind coverage.

options:
  -h, --help            show this help message and exit
  --merged-dir MERGED_DIR
  --out-dir OUT_DIR
  --study-id STUDY_ID
  --study-denominator STUDY_DENOMINATOR
  --min-study-fraction MIN_STUDY_FRACTION
  --expected-shards EXPECTED_SHARDS
```

</details>

<details>
<summary><code>extract_cfmethdb_candidate_matrix_np_chunk.py</code></summary>

```text
usage: extract_cfmethdb_candidate_matrix_np_chunk.py [-h] --sample-manifest
                                                     SAMPLE_MANIFEST
                                                     --candidate-coordinates
                                                     CANDIDATE_COORDINATES
                                                     --out-dir OUT_DIR
                                                     --cache-dir CACHE_DIR
                                                     --task-index TASK_INDEX
                                                     [--chunk-size CHUNK_SIZE]
                                                     [--download-timeout-sec DOWNLOAD_TIMEOUT_SEC]
                                                     [--delete-bigwigs]
                                                     [--progress-log PROGRESS_LOG]

Extract cfMethDB BigWig values into compact NumPy chunks.

options:
  -h, --help            show this help message and exit
  --sample-manifest SAMPLE_MANIFEST
  --candidate-coordinates CANDIDATE_COORDINATES
  --out-dir OUT_DIR
  --cache-dir CACHE_DIR
  --task-index TASK_INDEX
                        1-based chunk index.
  --chunk-size CHUNK_SIZE
  --download-timeout-sec DOWNLOAD_TIMEOUT_SEC
  --delete-bigwigs
  --progress-log PROGRESS_LOG
```

</details>

<details>
<summary><code>combine_cfmethdb_candidate_matrix_np_chunks.py</code></summary>

```text
usage: combine_cfmethdb_candidate_matrix_np_chunks.py [-h] --chunks-dir
                                                      CHUNKS_DIR --out-dir
                                                      OUT_DIR
                                                      --candidate-coordinates
                                                      CANDIDATE_COORDINATES
                                                      --expected-chunks
                                                      EXPECTED_CHUNKS

Combine cfMethDB NumPy beta-matrix chunks.

options:
  -h, --help            show this help message and exit
  --chunks-dir CHUNKS_DIR
  --out-dir OUT_DIR
  --candidate-coordinates CANDIDATE_COORDINATES
  --expected-chunks EXPECTED_CHUNKS
```

</details>

<details>
<summary><code>run_cfdna_specificity_constrained_twostage.py</code></summary>

```text
usage: run_cfdna_specificity_constrained_twostage.py [-h] --dataset
                                                     {prjna628686} --matrix
                                                     MATRIX --sample-metadata
                                                     SAMPLE_METADATA
                                                     --stage-metadata
                                                     STAGE_METADATA --out-dir
                                                     OUT_DIR
                                                     [--budgets BUDGETS]
                                                     [--max-union MAX_UNION]
                                                     [--value-mode {methylation,coverage}]
                                                     [--model-families MODEL_FAMILIES]
                                                     [--subtype-selector {ovr_union}]
                                                     [--config-subtype-metric {subtype_macro_f1_gate_positive_cancers}]
                                                     [--prjna-labels PRJNA_LABELS]
                                                     [--prjna-normal-label PRJNA_NORMAL_LABEL]
                                                     [--target-specificity TARGET_SPECIFICITY]
                                                     [--outer-splits OUTER_SPLITS]
                                                     [--outer-repeats OUTER_REPEATS]
                                                     [--inner-splits INNER_SPLITS]
                                                     [--seed SEED]
                                                     [--n-estimators N_ESTIMATORS]
                                                     [--n-jobs N_JOBS]
                                                     [--block-size BLOCK_SIZE]
                                                     [--min-train-present-fraction MIN_TRAIN_PRESENT_FRACTION]
                                                     [--min-train-present-per-class MIN_TRAIN_PRESENT_PER_CLASS]

Nested cfDNA two-stage model with a specificity-constrained cancer gate.

options:
  -h, --help            show this help message and exit
  --dataset {prjna628686}
  --matrix MATRIX
  --sample-metadata SAMPLE_METADATA
  --stage-metadata STAGE_METADATA
  --out-dir OUT_DIR
  --budgets BUDGETS
  --max-union MAX_UNION
  --value-mode {methylation,coverage}
  --model-families MODEL_FAMILIES
  --subtype-selector {ovr_union}
  --config-subtype-metric {subtype_macro_f1_gate_positive_cancers}
  --prjna-labels PRJNA_LABELS
  --prjna-normal-label PRJNA_NORMAL_LABEL
  --target-specificity TARGET_SPECIFICITY
  --outer-splits OUTER_SPLITS
  --outer-repeats OUTER_REPEATS
  --inner-splits INNER_SPLITS
  --seed SEED
  --n-estimators N_ESTIMATORS
  --n-jobs N_JOBS
  --block-size BLOCK_SIZE
  --min-train-present-fraction MIN_TRAIN_PRESENT_FRACTION
  --min-train-present-per-class MIN_TRAIN_PRESENT_PER_CLASS
```

</details>

<details>
<summary><code>run_cfdna_specificity_constrained_twostage_array.py</code></summary>

```text
usage: run_cfdna_specificity_constrained_twostage_array.py [-h] --mode
                                                           {fold,merge}
                                                           [--fold-id FOLD_ID]
                                                           --dataset
                                                           {prjna628686}
                                                           --matrix MATRIX
                                                           --sample-metadata
                                                           SAMPLE_METADATA
                                                           --stage-metadata
                                                           STAGE_METADATA
                                                           --out-dir OUT_DIR
                                                           [--budgets BUDGETS]
                                                           [--max-union MAX_UNION]
                                                           [--value-mode {methylation,coverage}]
                                                           [--model-families MODEL_FAMILIES]
                                                           [--subtype-selector {ovr_union}]
                                                           [--config-subtype-metric {subtype_macro_f1_gate_positive_cancers}]
                                                           [--prjna-labels PRJNA_LABELS]
                                                           [--prjna-normal-label PRJNA_NORMAL_LABEL]
                                                           [--target-specificity TARGET_SPECIFICITY]
                                                           [--outer-splits OUTER_SPLITS]
                                                           [--outer-repeats OUTER_REPEATS]
                                                           [--inner-splits INNER_SPLITS]
                                                           [--seed SEED]
                                                           [--n-estimators N_ESTIMATORS]
                                                           [--n-jobs N_JOBS]
                                                           [--block-size BLOCK_SIZE]
                                                           [--min-train-present-fraction MIN_TRAIN_PRESENT_FRACTION]
                                                           [--min-train-present-per-class MIN_TRAIN_PRESENT_PER_CLASS]

Fold-array wrapper for specificity-constrained cfDNA two-stage analysis.

options:
  -h, --help            show this help message and exit
  --mode {fold,merge}
  --fold-id FOLD_ID
  --dataset {prjna628686}
  --matrix MATRIX
  --sample-metadata SAMPLE_METADATA
  --stage-metadata STAGE_METADATA
  --out-dir OUT_DIR
  --budgets BUDGETS
  --max-union MAX_UNION
  --value-mode {methylation,coverage}
  --model-families MODEL_FAMILIES
  --subtype-selector {ovr_union}
  --config-subtype-metric {subtype_macro_f1_gate_positive_cancers}
  --prjna-labels PRJNA_LABELS
  --prjna-normal-label PRJNA_NORMAL_LABEL
  --target-specificity TARGET_SPECIFICITY
  --outer-splits OUTER_SPLITS
  --outer-repeats OUTER_REPEATS
  --inner-splits INNER_SPLITS
  --seed SEED
  --n-estimators N_ESTIMATORS
  --n-jobs N_JOBS
  --block-size BLOCK_SIZE
  --min-train-present-fraction MIN_TRAIN_PRESENT_FRACTION
  --min-train-present-per-class MIN_TRAIN_PRESENT_PER_CLASS
```

</details>


## PCA, age and assay estimates

<details>
<summary><code>analyze_sample_pca_20261106.py</code></summary>

```text
usage: analyze_sample_pca_20261106.py [-h] --run-dir RUN_DIR

options:
  -h, --help         show this help message and exit
  --run-dir RUN_DIR
```

</details>

<details>
<summary><code>analyze_probe_pca_20261106.py</code></summary>

```text
usage: analyze_probe_pca_20261106.py [-h] --run-dir RUN_DIR

options:
  -h, --help         show this help message and exit
  --run-dir RUN_DIR
```

</details>

<details>
<summary><code>summarize_age_representation.py</code></summary>

```text
usage: summarize_age_representation.py [-h] --config CONFIG --out-dir OUT_DIR

Aggregate curated age categories on explicitly supplied cohort memberships.

options:
  -h, --help         show this help message and exit
  --config CONFIG
  --out-dir OUT_DIR
```

</details>

<details>
<summary><code>calculate_assay_cost.py</code></summary>

```text
usage: calculate_assay_cost.py [-h] [--config CONFIG] --out-dir OUT_DIR

Recalculate the adopted Lam-based research operating budget in SM8/Table S27.

options:
  -h, --help         show this help message and exit
  --config CONFIG
  --out-dir OUT_DIR
```

</details>
