# Changelog

## 0.2.0 — 2026-11-23

- Align controlled comparisons with the MiniCpG-only manuscript scope; add the audited external accession manifest and approved supplementary updates.

- Correct the assay-cost calculator to the adopted Lam-based SM8/Table S27 model; preserve published table values and add table-based regression checks.

- Enforce at least 25 valid panel measurements for tissue prediction; absent columns and invalid beta values count as missing. Retain ineligible rows with explicit status and blank predictions.
- Correct merged-class display labels for COAD/READ, ESCA/STAD and GBM/LGG while retaining internal identifiers.
- Preserve fitted training-median imputation, raw argmax and the original model artifact.
- Add an installed `minicpg-predict` command, supported dependency lock and synthetic regression tests.
- Handle nonfinite extracted beta values, use zero-based cCRE coordinates and report valid methylation–expression pair counts.
- Package final-paper analysis workflows, explicit input contracts, executed configurations and a synthetic clustered-evaluation example.
- Export all 27 supplementary tables as 71 aggregate TSV blocks with source precision and hashes.

## 0.1.0 — 2026-07-26

Original public model, code and aggregate-result snapshot. The tag remains unchanged.
