# Changelog

## 0.2.0.dev0 — unreleased

- Enforce at least 25 valid panel measurements for tissue prediction; absent columns and invalid beta values count as missing. Retain ineligible rows with explicit status and blank predictions.
- Correct merged-class display labels for COAD/READ, ESCA/STAD and GBM/LGG while retaining internal identifiers.
- Preserve fitted training-median imputation, raw argmax and the original model artifact.
- Add an installed `minicpg-predict` command, supported dependency lock and synthetic regression tests.
- Include the literal-NA/nonfinite extractor fix, corrected cCRE coordinates and actual expression-correlation sample counts.
- Distinguish historical results and scripts from revised workflows awaiting migration.

## 0.1.0 — 2026-07-26

Original public model, code and aggregate-result snapshot. The tag remains unchanged.
