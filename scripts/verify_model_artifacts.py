#!/usr/bin/env python3
"""Verify distributed model/panel/label bytes without unpickling the model."""
import gzip
import hashlib
import json
from pathlib import Path


def digest(handle):
    return hashlib.file_digest(handle, "sha256").hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "model/artifact_manifest.json").read_text())
    for relative, expected in manifest["files"].items():
        path = root / relative
        with path.open("rb") as handle:
            if digest(handle) != expected:
                raise SystemExit(f"Artifact hash mismatch: {relative}")
    with gzip.open(root / "model/minicpg_k30_model_bundle.pkl.gz", "rb") as handle:
        if digest(handle) != manifest["uncompressed_model_sha256"]:
            raise SystemExit("Uncompressed model hash mismatch")
    print("Model, panel and label artifact hashes verified")


if __name__ == "__main__":
    main()
