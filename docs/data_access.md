# Data Access And Redistribution Policy

This release does not redistribute raw methylation beta matrices, IDAT files, sample-level metadata, or sample-level prediction outputs.

The included external-validation files are aggregate public-source summaries derived from accession-level inventories and class/bucket counts. Users should download raw TCGA, GEO, ArrayExpress, or sequencing data from the original repositories under the applicable data-use terms, then point the config templates to those local files.

The final 30-CpG panel, aggregate manuscript source tables, and fitted model bundle are included because they do not expose individual-level methylation profiles. The model bundle is data-derived and should be treated as a research-use artifact under the MiniCpG Academic and Non-commercial Research Use License; clinical translation would require assay-specific calibration, prospective validation, regulatory review, and separate permission.
