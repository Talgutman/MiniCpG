# Revision status

Working version: `0.2.0.dev0`, an unpublished local candidate. The saved tissue model and consensus panel are unchanged. This tree contains the final-paper analysis code and aggregate results; superseded scripts and July result snapshots were removed from this branch. The original `v0.1.0` tag remains unchanged.

## Analysis coverage and evidence

“Source checked” means the current calculation code and required helpers are packaged with input/command documentation; it does not mean training or the complete analysis was rerun. “Parity checked” means a stated numerical comparison against the adopted recorded outputs passed. All workflows require the original permitted inputs; manual source curation is not inferred from aggregate tables.

| Analysis | Packaged entry point/configuration | Validation scope |
|---|---|---|
| TCGA construction, splits, fold preprocessing | Development pipeline, manifest/fold builders, split-integrity helpers | Current source and dependency closure; synthetic patient-leakage tests |
| Selection, classifiers, consensus panel | Search YAML, benchmark/task/aggregation scripts, static rare-class selector | Actual search ranges preserved; selector source identical; synthetic final-fit/bundle test; no scientific development refit |
| K20 sensitivity | `k20_reproduction.json`, same development/evaluation workflow | Executed settings/model hash recovered; binary and retraining not included |
| Primary external / MethyDeep | Extraction, sanitation, canonical inference, `evaluate_external.py`, primary template | Saved-model tests; first 32 recorded bootstrap draws agree for full/shared and both weighting schemes |
| Consensus versus single-split panels | Ten-seed definitions, training and summary runners | Adopted source calculations retained; input contracts/CLI checked; no 1,010-model refit |
| Methylation shift | `analyze_class_geometry.py` | All observed class shifts and profiles agree with recorded values |
| Purity and errors | `analyze_heldout_purity.py`, sensitivity runner | Matching/regression/permutation source retained; CLI checked; no full regression rerun |
| Similarity / pair confusion | `analyze_class_geometry.py` | Observed association parity; permutation algorithm checked against independent SciPy calculations |
| Study errors, dispersion and subtype context | Study aggregator, geometry diagnostics, `analyze_uvm_skcm_diagnostics.py` | Synthetic aggregation tests; recorded dispersion/UCS/UVM diagnostic tables compared; manual source annotations remain inputs |
| Expanded domains / paired conditions | Expanded template, `evaluate_paired_conditions.py` | First 32 recorded domain bootstrap draws and cluster multiplicities agree; full paired aggregate outputs agree |
| Genomic/context annotation | Location/cCRE annotation and prefilter-background enrichment | Source checked; cCRE coordinate regression tests |
| Tumor, lineage and composition evidence | Decomposition, normal metadata/tissue identity, threshold sensitivity | Current source/CLI checked; no biological scan rerun |
| Normal-adjacent substitution | Same-patient substitution runner | Source/CLI checked; original model retained |
| Separate normal gate | Gate runner and bound-input template | Source checked; seed provenance limitation below |
| Expression/transcriptome background | Scan, shard aggregator, summary | Current exporter source; finite-beta N regression test; no transcriptome rescan |
| True-class SHAP | Locked-model SHAP/correlation runner | Current source and CLI checked; no complete SHAP rerun |
| Native cell-free DNA / coverage control | Separate nested workflow, preparation helpers, executed configuration | Source/CLI checked; no nested retraining; upstream provenance limitation below |
| Cost / turnaround | Cost calculator/configuration; Table S24 schedule | All batch/sensitivity costs agree numerically; turnaround is a proposed scenario |
| Age representation | Cohort-bound aggregation; Table S25 | Synthetic overlap/unknown handling tests; approved aggregate export |
| Sample/probe PCA | Hash-checked PCA runners | Current source and CLI checked; development-only fitting/projection rules retained |

All 25 supplementary tables are exported as 71 aggregate TSV blocks with numeric source precision and hashes. They are data sources, not reformatted manuscript documents. The trained model, feature order, class metadata and synthetic inference/evaluation examples are included.

## Remaining limitations and publication steps

- The normal-gate seed was reconstructed from commands/defaults rather than an explicit run receipt.
- Exact upstream cfMethDB per-file software versions and some coverage-control inner-search receipts were not recovered. The processed data layer and separate native workflow are explicit.
- Raw matrices, individual-level clinical metadata, curation, predictions and audit outputs are not redistributed. Reproduction requires those local inputs; this package does not claim a one-command download-to-paper rebuild.
- The lockfile is the tested candidate environment, not a reconstruction of every historical Python/R/TensorFlow environment. TensorFlow/Keras for MethyDeep and upstream SeSAMe are separate.
- Final publication still requires reconciling the manuscript Code Availability URL/version with the approved repository release and creating a new immutable tag. No public push, tag or manuscript edit is part of this local update.

See [workflows](workflows.md), [command reference](cli_reference.md), and [source provenance](source_provenance.tsv).
