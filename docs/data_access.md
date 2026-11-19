# Data required for reproduction

Running the trained MiniCpG model requires a specimen-by-probe beta-value table in the [documented input format](../README.md#input-and-output). The synthetic examples and trained model are included in the repository.

Reproducing the paper's analyses requires the study measurements and curated metadata listed below. These individual-level inputs are not distributed here; the repository supplies [aggregate results](../results/revision_20261119/README.md).

| Analysis | Inputs to obtain or prepare |
|---|---|
| TCGA development and held-out evaluation | Processed methylation values, specimen/patient manifests, probe annotations and exclusion lists |
| External tissue evaluation | GEO/ArrayExpress measurements, verified tumor labels, specimen characteristics, source-availability records and patient identifiers where available |
| Purity and biological context | Specimen-linked ABSOLUTE purity/covariates, normal-tissue references, HM450 annotation, SCREEN regulatory intervals and RNA-seq inputs as required by the chosen analysis |
| MethyDeep comparison | Published MethyDeep model, feature/class files and measurements for its probe panel |
| Native cell-free DNA | cfMethDB processed sequencing measurements, feature coordinates, specimen labels and stage annotations |

## Identify the study sources

[Supplementary Table S12](../results/revision_20261119/README.md#table-index) provides the external study/accession information. Obtain measurements and metadata from their original sources under the applicable terms. The accession table describes studies; it does not contain the specimen membership needed to reproduce an evaluation.

For each dataset, retain the source versions and file hashes, verify the diagnoses and specimen identities, and apply the paper's eligibility rules before inference. Patient identities must preserve development/test separation and the resampling groups. The [workflow recipes](workflows.md) specify the table columns and configurations for each analysis.

## Preserve the evaluated data version

Exact reproduction depends on the processed data and curation used for that analysis. A newly downloaded file can differ from the evaluated snapshot even if its accession is unchanged. Keep specimen annotation separate from measurement availability, and use them together to determine which rows are scored.

Upstream SeSAMe processing, MethyDeep's TensorFlow/Keras environment and cfMethDB processing have requirements beyond the Python lockfile. The [reproduction requirements](reproduction_requirements.md#data-and-software) describes the additional software and source-data requirements.

Analysis runners may write specimen-level audit files. Keep those local outputs separate from the aggregate files intended for sharing. The repository license does not grant rights to third-party data; see [LICENSE](../LICENSE).
