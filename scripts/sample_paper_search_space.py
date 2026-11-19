#!/usr/bin/env python3
"""Sample paper-protocol FS/model search spaces from config."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from pathlib import Path
from typing import Any

import yaml


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Sample FS/model random-search specs for the paper pipeline.")
    p.add_argument("--config", required=True)
    p.add_argument("--k", type=int, required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--seed", type=int, default=7)
    return p.parse_args()


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f) or {}


def _round_if_float(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, 6)
    return value


def sample_param(spec: dict[str, Any], rng: random.Random) -> Any:
    kind = str(spec.get("kind", "choice")).strip()
    if kind == "constant":
        return spec.get("value")
    if kind == "choice":
        values = list(spec.get("values", []))
        if not values:
            raise ValueError("choice spec requires non-empty values")
        return rng.choice(values)
    if kind == "int":
        return int(rng.randint(int(spec["min"]), int(spec["max"])))
    if kind == "float":
        return _round_if_float(rng.uniform(float(spec["min"]), float(spec["max"])))
    if kind == "log_float":
        lo = math.log10(float(spec["min"]))
        hi = math.log10(float(spec["max"]))
        return _round_if_float(10 ** rng.uniform(lo, hi))
    raise ValueError(f"Unknown parameter kind: {kind}")


def sample_draws(space: dict[str, Any], n_draws: int, rng: random.Random) -> list[dict[str, Any]]:
    draws: list[dict[str, Any]] = []
    for _ in range(n_draws):
        draw = {name: sample_param(dict(spec), rng) for name, spec in space.items()}
        draws.append(draw)
    return draws


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", str(text).strip()).strip("_").lower()


def main() -> int:
    args = parse_args()
    cfg = load_config(Path(args.config))
    search_cfg = dict(cfg.get("search") or {})

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fs_draws = int(search_cfg.get("fs_random_draws", 10))
    model_draws = int(search_cfg.get("model_random_draws", 10))

    fs_spaces = dict(search_cfg.get("fs_spaces") or {})
    model_spaces = dict(search_cfg.get("model_spaces") or {})
    prefilter_cfg = dict(cfg.get("prefilter") or {})
    default_prefilter_search = dict(prefilter_cfg.get("default_search") or {})

    rng = random.Random(int(args.seed) + int(args.k))

    fs_specs: dict[str, Any] = {}
    for method in cfg.get("fs_methods", []):
        method = str(method)
        if method == "mi_rank_baseline":
            fs_specs[method] = [
                {
                    "candidate_id": f"fs_{_slug(method)}_01",
                    "selection_method": method,
                    "candidate_spec": {"selection_args": {}},
                }
            ]
            continue
        space = dict(fs_spaces.get(method) or {})
        if not space:
            raise SystemExit(f"Missing FS search space for method: {method}")
        fs_specs[method] = []
        for idx, draw in enumerate(sample_draws(space, fs_draws, rng), start=1):
            candidate_spec = {"selection_args": draw}
            if default_prefilter_search:
                candidate_spec["prefilter_args"] = {
                    "prefilter_method": str(prefilter_cfg.get("method", "mi")),
                    "prefilter_k": int(sample_param(default_prefilter_search, rng)),
                }
            fs_specs[method].append(
                {
                    "candidate_id": f"fs_{_slug(method)}_{idx:02d}",
                    "selection_method": method,
                    "candidate_spec": candidate_spec,
                }
            )

    model_specs: dict[str, Any] = {}
    for model in cfg.get("models", []):
        model = str(model)
        space = dict(model_spaces.get(model) or {})
        if not space:
            raise SystemExit(f"Missing model search space for model: {model}")
        model_specs[model] = []
        for idx, draw in enumerate(sample_draws(space, model_draws, rng), start=1):
            model_specs[model].append(
                {
                    "candidate_id": f"model_{_slug(model)}_{idx:02d}",
                    "model": model,
                    "class_weight_mode": "balanced" if str(draw.get("class_weight", "")).strip() else "none",
                    "candidate_spec": {"model_args": {model: draw}},
                }
            )

    meta = {
        "config": str(Path(args.config).resolve()),
        "k": int(args.k),
        "seed": int(args.seed),
        "fs_random_draws": fs_draws,
        "model_random_draws": model_draws,
    }

    (out_dir / "fs_search_specs.json").write_text(json.dumps(fs_specs, indent=2))
    (out_dir / "model_search_specs.json").write_text(json.dumps(model_specs, indent=2))
    (out_dir / "search_space_meta.json").write_text(json.dumps(meta, indent=2))

    print(json.dumps({
        "fs_specs_json": str((out_dir / "fs_search_specs.json").resolve()),
        "model_specs_json": str((out_dir / "model_search_specs.json").resolve()),
        "meta_json": str((out_dir / "search_space_meta.json").resolve()),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
