External-cohort reproducibility manifest

Scope: 9,402 distinct study-qualified specimens from 131 public studies. This is the union of frozen primary-eligible, expanded-domain-eligible and expanded-domain-pending records; it is NOT the complete 20,033-record screening flow. Outside-scope exclusions are not represented.

Files: external_samples.tsv contains public identifiers and project-derived class/cohort/model dispositions. source_studies.tsv provides official source links and publication identifiers present in archived metadata. data_dictionary.json defines fields and values. Empty class means pending-only eligibility, not a normal/control class.

Frozen subsets (cohorts overlap; do not sum rows across cohorts):
cohort	eligible	pending	MiniCpG evaluated	MethyDeep scored	shared benchmark
primary	3723	0	3683	3563	3496
metastatic	1002	290	985	848	848
ffpe	3466	227	3397	3299	3276
epic	3560	197	3443	3253	3253

Reconstruction: filter the relevant cohort eligibility column for eligible records. MiniCpG status=evaluated reconstructs its full evaluated subset. shared_benchmark=1 reconstructs the exact paired comparison. For primary MethyDeep, evaluated denotes the shared comparison and scored_not_in_shared_benchmark denotes additional scored specimens outside the shared truth scope; their union reconstructs its prediction set. For expanded domains, MethyDeep evaluated reconstructs its full model-specific set. In FFPE, some MethyDeep evaluated specimens are outside the shared truth scope. Model evaluation and shared-benchmark membership are therefore deliberately separate.

Provenance: primary eligibility and analysis membership bind the adopted November 10 snapshot; expanded-domain decisions and membership bind November 13. Canonical labels, eligibility and input mappings were checked against these snapshots; later inventory coverage annotations do not supersede frozen evaluated memberships. Eligibility reasons summarize existing decisions with controlled codes; they do not constitute a new pathology review. Harmonized classes are the adopted analytical assignments based on public source annotations and project curation, with their existing scientific limitations. Pending-only provisional classes are omitted.

Identifiers: 148 internal GSM-plus-alias composites were reduced to verified public GSM accessions. Normalization is one-to-one within study, with zero collisions or specimen merges. ArrayExpress identifiers are exact names found in public SDRFs, not newly exposed internal identifiers. Seven generic ArrayExpress sample names recur between two studies; these remain distinct study-qualified records, with no inference that name equality proves specimen identity. No GEO GSM is reused across the scoped studies. Study-qualified keys remain authoritative.

Public source scope: GEO and ArrayExpress/BioStudies. Previously assessed source families were supplemented by archived-public-metadata checks for GSE161651, GSE222933, GSE97529. No study-specific restriction was identified in those checked metadata; this is not an exhaustive legal determination. Public availability alone is not a blanket redistribution license. NCBI does not itself restrict use/distribution but warns about third-party rights; EMBL-EBI adds no redistribution restrictions beyond original owners. Current CC0 treatment of new BioStudies deposits was not assumed retroactively for legacy ArrayExpress studies. Repository records and original papers should be cited.
https://www.ncbi.nlm.nih.gov/home/about/policies/
https://www.ebi.ac.uk/about/terms-of-use/

Excluded fields: raw measurements, source clinical prose, exact ages/dates/outcomes, patient crosswalks, private aliases and internal file paths. This is a reproducibility manifest, not a redistribution of source clinical tables or an independent certification of cohort correctness. Internal evidence and QA are retained separately and are not part of the public bundle.

Release: included with MiniCpG v0.2.0. This release documents the frozen analysis memberships; it does not change the model, predictions, eligibility decisions or source measurements.
