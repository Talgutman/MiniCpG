# Changelog

## 0.2.0.dev0 — unreleased

- Enforce at least 25 valid panel measurements for tissue prediction; absent columns and invalid beta values count as missing. Retain ineligible rows with explicit status and blank predictions.
- Correct merged-class display labels for COAD/READ, ESCA/STAD and GBM/LGG while retaining internal identifiers.
- Preserve fitted training-median imputation, raw argmax and the original model artifact.
- Add an installed `minicpg-predict` command, supported dependency lock and synthetic regression tests.
- Include the literal-NA/nonfinite extractor fix, corrected cCRE coordinates and actual expression-correlation sample counts.
- Package final-paper analysis workflows, explicit input contracts, executed configurations and a synthetic clustered-evaluation example.
- Export all 25 supplementary tables as 71 aggregate TSV blocks with source precision and hashes.
- Include current development helpers and patient split guards; correct the portable search configuration.
- Remove superseded analysis scripts, old aggregate results and historical documentation from the working tree; preserve v0.1.0 history.
- Check numerical agreement for external/paired evaluation, class geometry and cost scenarios; document workflows not retrained during packaging.

## 0.1.0 — 2026-07-26

Original public model, code and aggregate-result snapshot. The tag remains unchanged.
