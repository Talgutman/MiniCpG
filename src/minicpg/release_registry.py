"""Small static registry used by the MiniCpG public release scripts."""

from __future__ import annotations

import inspect
from typing import Any, Callable

from minicpg.selection import run_selector as run_rare_class_rank_shift_selector


class RegistryEntryNotFound(LookupError):
    """Raised when a release script requests an unavailable extension."""


PREFILTER_ARG_ALIASES = {
    "y_train": ("y",),
    "probe_set": ("probes",),
    "prefilter_k": ("k",),
    "prefilter_args": ("args",),
    "anova_scores": ("scores",),
}

SELECTOR_ARG_ALIASES = {
    "X_train_pref": ("X_train", "X"),
    "y_train": ("y",),
    "all_classes": ("classes",),
    "prefilter_probes": ("probes", "probe_ids"),
    "selection_args": ("selector_args", "args"),
    "prefilter_args": ("filter_args",),
}

MODEL_ARG_ALIASES = {
    "X_train": ("X",),
    "y_train": ("y",),
    "all_classes": ("classes",),
    "tune_metric": ("metric",),
    "model_args": ("args",),
}


def _slug(name: str) -> str:
    return str(name).strip().lower().replace("-", "_")


def load_prefilter_plugin(name: str) -> Callable[..., Any]:
    raise RegistryEntryNotFound(f"Unavailable prefilter extension: {name}")


def load_selector_plugin(name: str) -> Callable[..., Any]:
    slug = _slug(name)
    if slug in {"rare_class_rank_shift", "rare_aware_rank_shift"}:
        return run_rare_class_rank_shift_selector
    raise RegistryEntryNotFound(f"Unavailable selector extension: {name}")


def load_model_plugin(name: str) -> Callable[..., Any]:
    raise RegistryEntryNotFound(f"Unavailable model extension: {name}")


def call_with_supported_kwargs(
    fn: Callable[..., Any],
    kwargs: dict[str, Any],
    *,
    aliases: dict[str, tuple[str, ...]] | None = None,
) -> Any:
    sig = inspect.signature(fn)
    params = sig.parameters
    accepts_var_kwargs = any(
        param.kind == inspect.Parameter.VAR_KEYWORD for param in params.values()
    )
    keyword_names = {
        name
        for name, param in params.items()
        if param.kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY)
    }
    call_kwargs: dict[str, Any] = {}
    if accepts_var_kwargs:
        call_kwargs.update(kwargs)
    else:
        for name, value in kwargs.items():
            if name in keyword_names:
                call_kwargs[name] = value
    for canonical, alias_names in (aliases or {}).items():
        if canonical not in kwargs:
            continue
        value = kwargs[canonical]
        for alias_name in alias_names:
            if accepts_var_kwargs or alias_name in keyword_names:
                call_kwargs.setdefault(alias_name, value)
    return fn(**call_kwargs)


def selector_y_train_payload(X_train_pref: Any, y_train: Any) -> Any:
    try:
        import numpy as np
        import pandas as pd
    except ImportError:
        return y_train
    if not isinstance(X_train_pref, pd.DataFrame):
        return y_train
    if isinstance(y_train, pd.Series):
        series = y_train.copy()
    else:
        values = np.asarray(y_train, dtype=object)
        if values.ndim != 1:
            values = values.reshape(-1)
        if len(values) != len(X_train_pref.index):
            return y_train
        series = pd.Series(values, index=X_train_pref.index)
    if not series.index.equals(X_train_pref.index):
        values = np.asarray(series, dtype=object)
        if len(values) != len(X_train_pref.index):
            return y_train
        series = pd.Series(values, index=X_train_pref.index)
    return series
