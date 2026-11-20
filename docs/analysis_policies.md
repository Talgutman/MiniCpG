# Scientific conventions

These conventions define the analyses reported in the paper. Use them with the [workflow recipes](workflows.md); the [reproduction requirements](reproduction_requirements.md) list the necessary data and software.

## Tissue classifier and matched comparison

The saved MiniCpG model uses the 30-probe consensus panel, stored training-set medians and the highest-probability class across its original 27 outputs. External specimens need at least 25 valid panel measurements. The two 30-probe panels do not overlap. MethyDeep requires at least 27 measured probes and fills missing values with the specimen's mean across its measured panel probes.

The matched comparison uses identical scoreable specimens. GBM/LGG and COAD/READ are combined classes in both models. ESCA/STAD reference specimens are excluded because MethyDeep lacks that class; MiniCpG ESCA/STAD predictions are retained and counted as errors. Reassigning those predictions to a remaining class is a separate sensitivity analysis.

During development, prediction-policy selection occurs within the development procedure. The final saved model uses no additional decision policy. The consensus panel is ranked by selection frequency, with ties resolved by Borda score, mean rank and probe identifier.

## Cohort definitions

The primary external cohort consists of HM450 primary tumors without documented formalin-fixed paraffin-embedded (FFPE) preservation. This criterion does not establish confirmed frozen preservation for every specimen. Its membership and predictions stay fixed during the additional domain analyses.

Metastatic, FFPE and EPIC-array evaluations use independent membership flags. A specimen may belong to multiple cohorts but appears once within each; pending entries are not scored. Each cohort is evaluated separately.

Primary–regional-lymph-node comparisons use samples from the same patients. Author-export and IDAT preprocessing comparisons estimate differences between complete processing workflows. They do not isolate normalization effects or reconstruct the historical TCGA processing environment exactly.

## Metrics and uncertainty

Primary macro-F1 uses the fixed modeled class axis, including absent-reference classes with zero F1. Matched comparisons use their specified shared axis. Expanded-domain metrics use the classes represented in the original evaluated subset. All prediction columns are retained, including outputs without reference specimens. AUROC availability is reported when a reference class lacks probability support.

Patient-balanced point estimates weight each specimen by the inverse of its patient's specimen count within the subset. Bootstrap uncertainty instead resamples patient groups; paired model comparisons use the same draws. The configurations preserve each analysis's resampling units, strata, seeds and replicate counts.

## Error patterns and biological context

Class profiles are arithmetic mean beta values over the saved panel after training-median imputation. Class similarity uses Euclidean distances, and pair-confusion inference permutes whole class labels. Methylation shift is the mean absolute difference between external and held-out TCGA class profiles.

Candidate cis-regulatory element (cCRE) intervals use the annotation manifest's zero-based, half-open coordinates. Expression-correlation sample counts exclude nonfinite beta values and therefore differ from the broader project-level expression overlap counts. The biological workflows preserve their specified multiple-testing families, normal-reference reuse and composition-score definitions. SHAP summaries use the contribution to the reference tumor class.

Normal-adjacent substitution uses the saved tissue classifier. Normal-adjacent detection uses a separate fitted gate; it does not add a rejection rule to that classifier.

## Native cell-free DNA

The exploratory cell-free DNA analysis uses separate detection and subtype models, nested repeated validation and a total panel budget of 100 probes. Confusion matrices pool held-out prediction occurrences across repeats, and fold standard deviations are descriptive summaries rather than confidence intervals. Cancer-type performance is conditional on cancers passing the detection gate.

## Names used in the code

| Identifier | Meaning in this repository |
|---|---|
| `dev` / `DEV` | TCGA development set |
| `lockbox` | TCGA held-out test set |
| `stable_panel` in saved filenames | The consensus panel; names are retained for artifact compatibility |
| `alternative_*` in panel-comparison outputs | Single-split panels |
| `main` in external input tables | Primary external cohort |
