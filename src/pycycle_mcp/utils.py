"""Utility helpers for the pyCycle MCP server."""

from __future__ import annotations

import importlib
import logging
from collections.abc import Callable, Iterable, Sequence
from typing import cast

from .errors import error_response

LOGGER = logging.getLogger(__name__)


def load_callable(path: str) -> Callable[[], object]:
    """Load a callable from a fully-qualified import path."""

    module_path, _, attr = path.rpartition(".")
    if not module_path:
        raise ImportError(f"Invalid import path: {path}")
    module = importlib.import_module(module_path)
    target = getattr(module, attr)
    if not callable(target):
        raise TypeError(f"Target at {path} is not callable")
    return cast(Callable[[], object], target)


def error_on_missing_session(session_id: str, exc: Exception) -> dict[str, object]:
    """Standardize unknown session errors."""

    return error_response(
        error_type=exc.__class__.__name__,
        message=f"Session '{session_id}' not found",
        details=str(exc),
    )


def select_interesting_variables(variables: list[tuple[str, dict[str, object]]], keywords: Iterable[str]) -> list[str]:
    """Select promoted names containing any keyword in a case-insensitive fashion."""

    selected: list[str] = []
    for name, _ in variables:
        lowered = name.lower()
        if any(keyword.lower() in lowered for keyword in keywords):
            selected.append(name)
    return selected


def jsonable_variable_value(metadata: dict[str, object]) -> object:
    """Return an OpenMDAO variable's value as a JSON-serializable Python object.

    Two traps this exists to avoid, both hit in practice:

    1. **Never use ``metadata.get("val") or metadata.get("value")``.** OpenMDAO
       values are numpy arrays, and ``or`` evaluates ``bool(array)``, which
       raises ``ValueError: The truth value of an array with more than one
       element is ambiguous``. That is exactly how ``get_cycle_summary`` came to
       fail on EVERY call (6/6 in the historical run logs), returning empty
       ``options``/``key_inputs``/``key_outputs`` — the propulsion discipline's
       summary tool was totally broken. Use an explicit key check.
    2. Raw numpy types are not JSON-serializable, so they must be converted.
    """
    import numpy as np

    value = metadata.get("value") if "value" in metadata else metadata.get("val")
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.complexfloating)):
        return float(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.bool_):
        return bool(value)
    return value


def render_variable_entry(name: str, metadata: dict[str, object], io: str) -> dict[str, object]:
    """Format an OpenMDAO variable metadata entry."""
    value = jsonable_variable_value(metadata)
    return {
        "name": name,
        "io": io,
        "promoted": metadata.get("promoted_name", name) == name,
        "units": metadata.get("units"),
        "shape": _normalize_shape(metadata.get("shape")),
        "desc": metadata.get("desc"),
        "value": value,
    }


def _normalize_shape(shape: object | None) -> object | None:
    if shape is None:
        return None
    if isinstance(shape, Sequence) and not isinstance(shape, (str, bytes)):
        return list(shape)
    return shape


def ordered_cartesian_product(values: Sequence[Sequence[object]]) -> list[list[object]]:
    """Return nested-loop ordering for a Cartesian product."""

    if not values:
        return []
    results: list[list[object]] = [[]]
    for group in values:
        next_results: list[list[object]] = []
        for prefix in results:
            for item in group:
                next_results.append(prefix + [item])
        results = next_results
    return results
