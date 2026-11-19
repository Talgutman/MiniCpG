# Reproduction workflow

## Saved-model inference

Follow the repository README. `uv.lock` is the candidate's tested environment, not a reconstruction of every historical analysis environment. The trained model and probe order are unchanged from v0.1.0. `model/artifact_manifest.json` records their hashes and the uncompressed model's identity against the locked analysis artifact.

The prediction tests use synthetic inputs. They check complete-input probabilities against the saved pipeline, training-median imputation, the 25-probe coverage boundary, absent columns, malformed measurements and row identity. They do not rerun model development or establish clinical performance.

## Analysis inputs

Obtain source measurements and metadata from their original repositories under applicable terms. Do not treat the historical accession summaries as the revised cohort membership list. Supply explicit paths to analysis scripts; `config/paths.template.yaml` is a documentation template, not a universal configuration loader. Different entry points accept their own arguments.

Current source fixes included here:

```bash
uv run --locked --extra analysis python scripts/build_external_panel_matrix.py --help
uv run --locked --extra analysis python scripts/annotate_panel_chromatin_context.py --help
uv run --locked --extra analysis python scripts/analyze_panel_meth_expr_transcriptome_null.py --help
```

The extractor normalizes literal missing/nonfinite cells before coverage counting. Follow extraction with `sanitize_external_panel_beta_values.py` to remove beta values outside [0, 1]. The cCRE annotator uses zero-based, half-open manifest coordinates without subtracting one. The expression exporter distinguishes the finite-beta correlation sample count from the project-level matched expression count. Source digests are recorded in `docs/source_provenance.tsv`.

These entry points require the tabular schemas named in their arguments; their inclusion does not supply the restricted or specimen-level inputs. CLI/import checks are not full scientific reproduction tests.

## Revised manuscript workflows

Use `docs/revision_status.md` as the migration checklist. The original development scripts and aggregate files remain available for historical reference, but are not a complete recipe for the revised manuscript. In particular, revised primary curation, overlapping metastatic/FFPE/EPIC cohorts, grouped uncertainty and mean-profile error analyses must not be replaced with the older disjoint-bucket or median-profile workflows.

The revised external cohorts require two distinct sources of truth: specimen annotation and measurement/source inventory. Resolve membership and measurement availability before inference, and bind predictions to the exact adopted source snapshot, panel and model. Sensitive specimen-level inputs must be rebuilt locally rather than published with this code.

Keep final inference distinct from model development: the released tissue model uses a null decision policy and raw argmax, whereas development pipelines can select inner-validation prediction policies. The cell-free DNA analysis uses a separate fitted workflow and panel, not the released tissue model.

No single command is claimed to reproduce the entire revised paper at this stage. A future release requires complete per-analysis input schemas, commands, versioned configurations and source-to-result checks.
