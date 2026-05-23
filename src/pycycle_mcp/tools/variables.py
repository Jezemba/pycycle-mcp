"""Variable management tools for pyCycle models."""

from __future__ import annotations

from ..canonical_inputs import get_design_inputs_for_cycle
from ..errors import error_response, to_error
from ..session_manager import session_manager
from ..utils import error_on_missing_session, render_variable_entry


def list_variables(payload: dict[str, object]) -> dict[str, object]:
    """List variables in the cycle model."""

    session_id = payload.get("session_id")
    if not session_id:
        return error_response("ValidationError", "session_id is required")

    kind = payload.get("kind", "both")
    promoted_only = payload.get("promoted_only", True)
    name_filter = payload.get("name_filter")
    max_variables_raw = payload.get("max_variables", 200)
    max_variables = int(max_variables_raw) if isinstance(max_variables_raw, (int, str)) else 200

    try:
        problem, _ = session_manager.get(str(session_id))
        results: list[dict[str, object]] = []

        if kind in ("inputs", "both"):
            for name, metadata in problem.model.list_inputs(prom_name=True, out_stream=None):
                if promoted_only and metadata.get("promoted_name") not in (None, name):
                    continue
                if name_filter and str(name_filter).lower() not in name.lower():
                    continue
                results.append(render_variable_entry(name, metadata, "input"))

        if kind in ("outputs", "both"):
            for name, metadata in problem.model.list_outputs(prom_name=True, out_stream=None):
                if promoted_only and metadata.get("promoted_name") not in (None, name):
                    continue
                if name_filter and str(name_filter).lower() not in name.lower():
                    continue
                results.append(render_variable_entry(name, metadata, "output"))

        return {"variables": results[:max_variables]}
    except KeyError as exc:
        return error_on_missing_session(str(session_id), exc)
    except Exception as exc:  # pragma: no cover
        return to_error(exc)


def set_inputs(payload: dict[str, object]) -> dict[str, object]:
    """Set one or more input variables in the cycle model."""

    session_id = payload.get("session_id")
    values: dict[str, object] = payload.get("values", {})  # type: ignore[assignment]
    allow_missing = bool(payload.get("allow_missing", False))

    if not session_id:
        return error_response("ValidationError", "session_id is required")
    if not values:
        return error_response("ValidationError", "values must contain at least one entry")

    try:
        problem, _ = session_manager.get(str(session_id))
        updated: list[str] = []
        skipped: list[dict[str, str]] = []

        for name, value in values.items():
            try:
                problem.set_val(name, value)
                updated.append(name)
            except Exception as exc:
                if allow_missing:
                    skipped.append({"name": name, "reason": str(exc)})
                else:
                    return to_error(exc)

        return {"updated": updated, "skipped": skipped}
    except KeyError as exc:
        return error_on_missing_session(str(session_id), exc)
    except Exception as exc:  # pragma: no cover
        return to_error(exc)


def get_design_inputs(payload: dict[str, object]) -> dict[str, object]:
    """Return the curated list of design-point input variables for the
    session's cycle type.

    Each entry has ``name`` (the exact path to pass to ``set_inputs``),
    ``units``, ``default``, ``description``, and ``category``. The list
    is intentionally short — these are the design dials, not the
    full 900+ promoted variables that ``list_variables`` returns.
    """

    session_id = payload.get("session_id")
    if not session_id:
        return error_response("ValidationError", "session_id is required")

    try:
        problem, meta = session_manager.get(str(session_id))
        cycle_type = str(meta.get("cycle_type", ""))
        inputs = get_design_inputs_for_cycle(cycle_type)
        # Augment each entry with the current value from the live model
        # so the agent can see what's already set and only override
        # what it actually needs to.
        for entry in inputs:
            try:
                current = problem.get_val(entry["name"])
                entry["current_value"] = (
                    current.item() if hasattr(current, "item") else current
                )
            except Exception:
                entry["current_value"] = entry.get("default")
        return {
            "cycle_type": cycle_type,
            "design_inputs": inputs,
            "note": (
                "These are the canonical design-point inputs for this "
                "cycle. Pass any subset of these names verbatim to "
                "set_inputs — they are guaranteed to resolve. The "
                "full promoted-variable list from list_variables "
                "contains 900+ internal entries that are NOT safe to "
                "set directly."
            )
            if inputs
            else (
                "No curated design-input list available for this cycle "
                "type. Fall back to list_variables to discover input "
                "paths."
            ),
        }
    except KeyError as exc:
        return error_on_missing_session(str(session_id), exc)
    except Exception as exc:  # pragma: no cover
        return to_error(exc)


def get_outputs(payload: dict[str, object]) -> dict[str, object]:
    """Fetch values for one or more outputs after a run."""

    session_id = payload.get("session_id")
    names: list[str] = payload.get("names", [])  # type: ignore[assignment]
    allow_missing = bool(payload.get("allow_missing", False))

    if not session_id:
        return error_response("ValidationError", "session_id is required")
    if not names:
        return error_response("ValidationError", "names must contain at least one entry")

    try:
        problem, _ = session_manager.get(str(session_id))
        values: dict[str, object] = {}
        missing: list[str] = []

        for name in names:
            try:
                current = problem.get_val(name)
                values[name] = current.item() if hasattr(current, "item") else current
            except Exception:
                if allow_missing:
                    missing.append(name)
                else:
                    return error_response("LookupError", f"Unknown output: {name}")

        return {"values": values, "missing": missing}
    except KeyError as exc:
        return error_on_missing_session(str(session_id), exc)
    except Exception as exc:  # pragma: no cover
        return to_error(exc)
