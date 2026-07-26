#!/usr/bin/env python3
"""Helpers for resolving canonical external data-source paths."""

from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent


def resolve_external_data_source_path(path_str: str, *, project_root: Path | None = None) -> Path:
    p = Path(str(path_str).strip())
    if not str(path_str).strip():
        return p
    if p.is_absolute():
        return p
    if p.exists():
        return p
    root = (project_root or PROJECT_ROOT).resolve()
    rooted = (root / p).resolve()
    if rooted.exists():
        return rooted
    return p


def external_data_source_exists(path_str: str, *, project_root: Path | None = None) -> bool:
    return bool(str(path_str).strip()) and resolve_external_data_source_path(path_str, project_root=project_root).exists()
