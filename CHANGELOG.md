# Changelog

## 0.2.0.dev0 — unreleased

- Enforce at least 25 valid panel measurements for tissue prediction; absent columns and invalid beta values count as missing. Retain ineligible rows with explicit status and blank predictions.
- Correct merged-class display labels for COAD/READ, ESCA/STAD and GBM/LGG while retaining internal identifiers.
- Preserve fitted training-median imputation, raw argmax and the original model artifact.
- Add an installed `minicpg-predict` command, supported dependency lock and synthetic regression tests.
- Handle nonfinite extracted beta values, use zero-based cCRE coordinates and report valid methylation–expression pair counts.
- Package final-paper analysis workflows, explicit input contracts, executed configurations and a synthetic clustered-evaluation example.
- Export all 25 supplementary tables as 71 aggregate TSV blocks with source precision and hashes.

## 0.1.0 — 2026-07-26

Original public model, code and aggregate-result snapshot. The tag remains unchanged.

### Input-processing analysis scope

- Restrict alternative-export and IDAT preprocessing contrasts to MiniCpG; align S15 aggregate exports and input requirements. Clinical contrasts and the main external MethyDeep benchmark are unchanged.
