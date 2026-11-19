# Data access

The repository includes the fitted MiniCpG model, its consensus panel/class metadata, synthetic examples and aggregate numerical sources for the final paper. It does not redistribute individual methylation profiles, IDAT files, linked clinical metadata, sample-level curation or predictions.

Download or obtain permitted TCGA, GEO, ArrayExpress and sequencing inputs from their original sources. Supplementary Table S12 in `results/revision_20261119/` supplies the current aggregate accession/study information. It is not a sample-membership manifest. Combine source measurements with independently verified diagnosis, tissue, processing and measurement-availability annotations, and record exact file hashes. Preserve the same patient separation and curation decisions when reproducing the paper.

The [workflow guide](workflows.md) describes required input schemas and source prerequisites, including ABSOLUTE purity, HM450 annotations, SCREEN cCRE intervals, RNA-seq and the cfMethDB processed data layer. MethyDeep's model must be obtained separately and run in a compatible TensorFlow/Keras environment. Public source availability does not itself establish equivalence to the adopted processed snapshot.

The model and code are distributed under [LICENSE](../LICENSE). Local reproduction outputs can include specimen-level data; keep them outside the public file list.
