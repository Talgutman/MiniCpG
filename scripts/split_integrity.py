"""Sample-membership checks shared by preparation, training and reporting.

These checks inspect identifiers, not model weights; they do not claim to
reconstruct historical model fit membership.
"""
from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def read_rows(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def id_hash(ids):
    return hashlib.sha256("\n".join(sorted(set(ids))).encode()).hexdigest()


def unique_ids(rows, column="sample_id", context="table"):
    ids = [str(row.get(column, "")).strip() for row in rows]
    if not ids or any(x in {"", "nan", "None"} for x in ids):
        raise ValueError(f"{context}: empty/missing {column}")
    if len(set(ids)) != len(ids):
        raise ValueError(f"{context}: duplicate {column}")
    return set(ids)


def require_same_ids(left, right, context="membership"):
    a, b = unique_ids(left, context=context + " left"), unique_ids(right, context=context + " right")
    if a != b:
        raise ValueError(f"{context}: sample mismatch: left_only={len(a-b)}, right_only={len(b-a)}")


def validate_partition(dev, heldout, cohort=None, patient_col="patient_id"):
    d, h = unique_ids(dev, context="DEV"), unique_ids(heldout, context="held-out")
    if d & h:
        raise ValueError("DEV/held-out sample overlap")
    patients = []
    for label, rows in [("DEV", dev), ("held-out", heldout)]:
        values = [str(r.get(patient_col, "")).strip() for r in rows]
        if any(x in {"", "nan", "None"} for x in values):
            raise ValueError(f"{label}: missing {patient_col}")
        patients.append(set(values))
    if patients[0] & patients[1]:
        raise ValueError("DEV/held-out patient overlap")
    if cohort is not None and unique_ids(cohort, context="cohort") != d | h:
        raise ValueError("DEV/held-out union differs from cohort")
    return {"dev_samples": len(d), "heldout_samples": len(h),
            "dev_sample_sha256": id_hash(d), "heldout_sample_sha256": id_hash(h),
            "dev_patient_sha256": id_hash(patients[0]), "heldout_patient_sha256": id_hash(patients[1])}


def validate_folds(dev, folds, expected_repeats=None, expected_splits=None):
    ids = unique_ids(dev, context="fold DEV")
    patients = {r["sample_id"]: r.get("patient_id", r["sample_id"]) for r in dev}
    by_repeat = defaultdict(list)
    for row in folds:
        by_repeat[int(row.get("repeat", 0))].append(row)
    if not by_repeat:
        raise ValueError("Empty folds")
    if expected_repeats is not None and set(by_repeat) != set(range(expected_repeats)):
        raise ValueError("Unexpected repeat coverage")
    for repeat, rows in by_repeat.items():
        if unique_ids(rows, context=f"repeat {repeat}") != ids:
            raise ValueError(f"repeat {repeat}: fold IDs differ from DEV")
        fs = {int(r["outer_fold"]) for r in rows}
        if expected_splits is not None and fs != set(range(expected_splits)):
            raise ValueError(f"repeat {repeat}: unexpected fold coverage")
        patient_folds = defaultdict(set)
        for row in rows:
            patient_folds[patients[row["sample_id"]]].add(int(row["outer_fold"]))
        if any(len(x) != 1 for x in patient_folds.values()):
            raise ValueError(f"repeat {repeat}: patient occurs in train and test")
    return {"repeats": len(by_repeat), "rows": len(folds)}


def validate_cached_training(cached, expected_train, test):
    ids = unique_ids(cached, context="cached training")
    allowed = unique_ids(expected_train, context="expected training")
    test_ids = unique_ids(test, context="test")
    if ids - allowed or ids & test_ids:
        raise ValueError("Cached selector training membership disagrees with current fold")
    labels = {r["sample_id"]: r.get("project_id") for r in expected_train}
    if any(r.get("project_id") != labels[r["sample_id"]] for r in cached):
        raise ValueError("Cached selector training labels disagree with current manifest")


def validate_locked_development(rows, manifest_path, heldout=None):
    """Check an optional run-local lock before a training job starts."""
    lock_path = Path(manifest_path).parent / "split_membership_lock.json"
    if lock_path.exists():
        lock = json.loads(lock_path.read_text())
        if id_hash(unique_ids(rows, context="locked DEV")) != lock["dev_sample_sha256"]:
            raise ValueError(f"DEV membership differs from locked assignment: {lock_path}")
        if heldout is not None and id_hash(unique_ids(heldout, context="locked held-out")) != lock["heldout_sample_sha256"]:
            raise ValueError(f"Held-out membership differs from locked assignment: {lock_path}")


def validate_lock_contract(path):
    """Validate all active preparations and saved matrix/prediction memberships."""
    contract = json.loads(Path(path).read_text())
    result = {"runs": [], "artifacts": []}
    for run in contract["runs"]:
        dev, heldout = read_rows(run["dev"]), read_rows(run["heldout"])
        got = validate_partition(dev, heldout, read_rows(run["cohort"]))
        for key in ("dev_sample_sha256", "heldout_sample_sha256", "dev_patient_sha256", "heldout_patient_sha256"):
            if got[key] != contract[key]:
                raise ValueError(f"{run['dev']}: locked {key} mismatch")
        split = read_rows(run["split"])
        require_same_ids(split, dev + heldout, "combined split")
        for label, truth in [("dev", dev), ("lockbox", heldout)]:
            require_same_ids([r for r in split if r["split"] == label], truth, label)
        for fold in run["folds"]:
            validate_folds(dev, read_rows(fold["path"]), fold["repeats"], fold["splits"])
        result["runs"].append({"dev": run["dev"], **got})
    for item in contract["artifacts"]:
        rows = read_rows(item["path"])
        ids = unique_ids(rows, context=item["path"])
        if id_hash(ids) != contract[item["split"] + "_sample_sha256"]:
            raise ValueError(f"Saved artifact membership differs from locked split: {item['path']}")
        result["artifacts"].append({"path": item["path"], "n_samples": len(ids)})
    for item in contract.get("protected_artifacts", []):
        actual = hashlib.sha256(Path(item["path"]).read_bytes()).hexdigest()
        if actual != item["sha256"]:
            raise ValueError(f"Protected model/panel artifact hash changed: {item['path']}")
    return result
