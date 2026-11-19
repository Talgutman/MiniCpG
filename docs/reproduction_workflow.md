# Reproduction workflow

Start with the [README](../README.md) to run the saved model. Use [revised analysis workflows](workflows.md) for the complete input contracts, configurations, command sequences and known provenance limits. The [command reference](cli_reference.md) lists packaged CLI arguments; [revision status](revision_status.md) distinguishes numerical parity checks from source-only checks.

The tissue model, probe order and raw-argmax prediction policy are unchanged. `model/artifact_manifest.json` records their hashes. The synthetic examples and regression tests run without study data. Analysis workflows require permitted local measurements and curated metadata; sample-level inputs and outputs are not public assets.

This branch contains final-paper workflows and current aggregate tables. Use the unchanged `v0.1.0` tag for the older code/results; obsolete scripts and snapshots are not retained in this working tree. `uv.lock` describes the tested candidate environment, not every historical analysis environment.
