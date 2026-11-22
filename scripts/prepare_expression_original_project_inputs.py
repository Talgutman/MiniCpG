"""Restore original TCGA projects in isolated expression-analysis inputs.

Classifier matrices deliberately use merged classes. Do not overwrite them:
this adapter requires an original-project manifest and writes separate inputs.
"""
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path


def read_tsv(path):
    with Path(path).open() as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return reader.fieldnames, list(reader)


def original_project_map(rows):
    mapping = {}
    for row in rows:
        sample = row["sample_id"]
        original = row["project_id_original"]
        if not original:
            raise ValueError(f"Missing original project for {sample}")
        value = (row["project_id"], original)
        if sample in mapping and mapping[sample] != value:
            raise ValueError(f"Conflicting project mapping for {sample}")
        mapping[sample] = value
    return mapping


def restore(rows, mapping):
    result = []
    for row in rows:
        sample = row["sample_id"]
        if sample not in mapping:
            raise ValueError(f"No original project mapping for {sample}")
        merged, original = mapping[sample]
        if row["project_id"] != merged:
            raise ValueError(f"Classifier label differs from manifest for {sample}")
        result.append(dict(row, classifier_class_id=merged, project_id=original))
    return result


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--dev", required=True)
    parser.add_argument("--heldout", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    out = Path(args.out_dir)
    if out.exists() and any(out.iterdir()):
        raise ValueError(f"Output directory must be empty: {out}")
    _, manifest = read_tsv(args.manifest)
    mapping = original_project_map(manifest)
    prepared = []
    for split, path in [("dev", args.dev), ("lockbox", args.heldout)]:
        fields, rows = read_tsv(path)
        restored = restore(rows, mapping)
        prepared.append((split, path, fields, rows, restored))
    out.mkdir(parents=True, exist_ok=True)
    audit = {"manifest": str(args.manifest), "manifest_sha256": sha256(args.manifest), "splits": {}}
    for split, path, fields, rows, restored in prepared:
        dest = out / f"tcga_{split}_panel_matrix.tsv"
        with dest.open("w") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields + ["classifier_class_id"], delimiter="\t", lineterminator="\n")
            writer.writeheader()
            writer.writerows(restored)
        changes = collections.Counter((a["project_id"], b["project_id"]) for a, b in zip(rows, restored) if a["project_id"] != b["project_id"])
        # All non-label fields must remain exact strings, including missing betas.
        assert all(all(a[k] == b[k] for k in fields if k != "project_id") for a, b in zip(rows, restored))
        audit["splits"][split] = {"source": str(path), "source_sha256": sha256(path), "output": str(dest), "output_sha256": sha256(dest), "rows": len(rows), "changes": [{"classifier_class": a, "original_project": b, "n": n} for (a, b), n in sorted(changes.items())], "non_label_fields_unchanged": True}
    (out / "INPUT_AUDIT.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
