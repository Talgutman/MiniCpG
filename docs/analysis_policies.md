# Revised analysis policies

These are scientific contracts for the repository migration. They do not imply that every corresponding workflow has been packaged or rerun.

## Tissue model and comparison

The locked model uses the 30-probe consensus panel, a stored training-median imputer and raw argmax across its original 27 classes. External specimens need at least 25 valid panel probes. MethyDeep requires at least 27 measured probes and uses the specimen's mean across its measured panel probes for missing values. Comparisons use the identical scoreable specimens for both models. The merged GBM/LGG and COAD/READ labels are shared. ESCA/STAD reference specimens are excluded from the matched comparison because MethyDeep lacks that class; original MiniCpG ESCA/STAD predictions remain and count as errors. Runner-up reassignment is a separate sensitivity analysis.

Development prediction-policy selection occurs within the development procedure. It must not be applied retrospectively to the locked model. Consensus-panel ties use selection frequency, Borda score, mean rank, then probe identifier. Reader-facing descriptions use “single-split panels”; historical filenames and model identifiers remain unchanged.

## Cohorts and evaluation

Primary external eligibility is HM450 primary tumors without documented FFPE preservation. This does not establish confirmed frozen preservation for every specimen. Its adopted membership and predictions remain fixed during expanded-domain analyses.

Metastatic, FFPE and EPIC evaluations use independent membership flags. A specimen may belong to multiple cohorts, but appears once within each. Pending entries are not scored. Do not define these cohorts through the old mutually exclusive bucket field or report pooled-union performance.

Primary macro-F1 uses the fixed modeled class axis, including absent-reference classes with zero F1. Matched comparisons preserve their specified shared axis. Expanded-domain metrics use the original subset's present-reference class axis. Prediction columns remain intact even where a class has no reference specimens. AUROC availability must be reported explicitly when a reference class lacks probability support.

Patient-balanced point estimates use inverse within-subset specimen multiplicity. Patient-group bootstrap uncertainty is a separate operation; paired comparisons share draws. Preserve the recorded resampling unit, stratification, seeds and replicate counts for each analysis. Do not substitute specimen bootstrap for a patient-group analysis.

Controlled primary–regional-lymph-node contrasts use samples from the same patients. Author-export and IDAT preprocessing comparisons estimate workflow differences, not isolated normalization effects or an exact reconstruction of historical TCGA preprocessing.

## Error and biological analyses

Class geometry uses arithmetic mean beta profiles over the locked panel with stored training-median imputation, and Euclidean distances. Pair-confusion inference permutes whole class labels. Methylation shift is the mean absolute difference between external and held-out TCGA class mean profiles. Do not restore earlier median-profile analyses.

cCRE intervals use the manifest's zero-based, half-open coordinates. Expression correlation sample counts exclude nonfinite beta values and are distinct from project-level expression overlap counts. Preserve the specified multiple-testing families, normal-reference reuse, composition-component sums and true-class SHAP convention.

## Separate cell-free DNA workflow and limitations

The exploratory cell-free DNA analysis uses separate detection and subtype models, nested repeated validation and a total panel budget of 100 probes. The coverage-only control uses observed/missing indicators. Confusion matrices pool held-out prediction occurrences across repeats; reported fold standard deviations are descriptive, not confidence intervals. Exact upstream per-file cfMethDB software versions are not recoverable from the available provenance. The historical normal-adjacent gate seed is reconstructed from commands/defaults, not an explicit run receipt. Neither limitation should be hidden by inventing a configuration.
