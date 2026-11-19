# Revision status

Working version: `0.2.0.dev0`. This is an unpublished candidate based on `v0.1.0` (commit `07ad8f804a9558387d565688916757c10f36bef2`).

## Implemented in this candidate

- Direct tissue inference applies coverage and beta validity rules, preserves rejected rows and uses the unchanged saved model.
- Class metadata describes the merged COAD/READ, ESCA/STAD and GBM/LGG classes without changing internal identifiers.
- A dependency lock and synthetic regression tests cover the inference entry point.
- Current extractor, cCRE coordinate and expression sample-count source fixes are included.
- Documentation distinguishes historical aggregates from revised analyses and records the scientific conventions below.

## Migration checklist

“Historical” means code exists but its complete revised workflow has not been verified. “Pending” means additional current code/configuration must be ported and checked. No scientific analysis needs to be rerun merely to migrate the code.

| Analysis | Repository status | Remaining work |
|---|---|---|
| TCGA construction, splits and preprocessing | Historical scripts | Verify dependencies, input schemas and recorded fold configuration |
| Pipeline selection and consensus panel | Historical scripts | Bind exact selector/search and inner decision-policy settings; verify registry coverage |
| K20 sensitivity | Pending | Bind recovered K20 artifact/configuration; preserve the K100 decision |
| Primary external and MethyDeep comparison | Extractor fixed; historical evaluation | Port adopted curation, original-prediction matching and grouped uncertainty |
| Consensus versus single-split panels | Pending | Package panel/seed comparison and correlation workflows |
| Methylation shift and recall | Pending | Package current mean-profile analysis |
| Purity and classification errors | Pending | Package matching, conditional regression and sensitivity analysis |
| Class similarity and pair confusion | Pending | Package mean geometry and whole-class-label permutation |
| Study-specific errors and subtypes | Historical diagnostics | Reconcile current aggregate summaries and evidence |
| Expanded domains and paired contrasts | Pending | Port independent membership flags and paired/patient-group workflows |
| Genomic/context annotation | cCRE fix included | Bind manifests, enrichment background and annotation sources |
| Tumor/lineage/composition scores | Historical partial code | Package current score decomposition and component definitions |
| Normal-adjacent substitution | Historical script | Verify current inputs, pairs and uncertainty |
| Separate normal-adjacent detection gate | Pending | Package learner/calibration configuration with seed caveat |
| Expression and transcriptome background | Current exporter included | Complete sharding/aggregation recipe and input schema |
| True-class SHAP | Pending | Package current convention and source bindings |
| Native cell-free DNA and coverage control | Historical scripts | Bind the executed nested configuration and data-layer provenance |
| Cost and turnaround scenarios | Pending | Package assumptions and machine-readable model |
| Age subgroup evaluation | Pending | Package aggregate-only reporting |
| PCA | Pending | Package fitting/projection and panel-data provenance |

Current figure and supplementary table numbers differ from the historical release. Export revised aggregate sources only after confirming the final manuscript mapping. Keep raw matrices, specimen identifiers, linked clinical metadata, predictions, internal audits and backups outside the public tree.

Before publication: complete this checklist, test documented commands from a clean environment, audit the explicit release file list, regenerate the release manifest, reconcile the manuscript Code Availability statement, and create a new immutable tag. Do not move or replace `v0.1.0`.
