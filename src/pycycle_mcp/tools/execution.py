"""Execution tools for running cycle models."""

from __future__ import annotations

import logging

import numpy as np

from ..errors import error_response, to_error
from ..session_manager import session_manager
from ..utils import error_on_missing_session

LOGGER = logging.getLogger(__name__)

DEFAULT_OUTPUTS = [
    "perf.Fn",
    "perf.TSFC",
    "perf.OPR",
    "fc.Fl_O:stat:MN",
    "fc.alt",
    "splitter.BPR",
]


def _to_serializable(val: object) -> object:
    """Convert numpy types to native Python for JSON serialization."""
    if isinstance(val, np.ndarray):
        flat = val.flatten()
        if flat.size == 1:
            return float(flat[0])
        return [float(v) for v in flat]
    if isinstance(val, (np.floating, np.complexfloating)):
        return float(val)
    if isinstance(val, np.integer):
        return int(val)
    return val


def run_cycle(payload: dict[str, object]) -> dict[str, object]:
    """Run the cycle model and return selected outputs."""

    session_id = payload.get("session_id")
    outputs_of_interest: list[str] = payload.get("outputs_of_interest") or DEFAULT_OUTPUTS  # type: ignore[assignment]
    use_driver = bool(payload.get("use_driver", False))

    if not session_id:
        return error_response("ValidationError", "session_id is required")

    try:
        problem, _ = session_manager.get(str(session_id))
        messages: list[str] = []
        try:
            problem.set_solver_print(level=-1)
            if use_driver:
                messages.append("Ran driver")
                problem.run_driver()
            else:
                messages.append("Ran model")
                problem.run_model()
        except Exception as exc:
            LOGGER.error("Run failed: %s", exc)
            return to_error(exc)

        outputs: dict[str, object | None] = {}
        missing: list[str] = []
        for name in outputs_of_interest:
            try:
                outputs[name] = _to_serializable(problem.get_val(name))
            except Exception as exc:
                outputs[name] = None
                missing.append(name)
                messages.append(f"Missing output {name}: {exc}")

        # `success: True` used to be returned unconditionally, so a call that
        # resolved NOTHING still reported success. Observed live:
        #
        #   {"success": true, "outputs": {"SFC": null, "Fn": null}, ...}
        #
        # The names were wrong (they are `perf.TSFC` / `perf.Fn`), and OpenMDAO
        # said so usefully inside `messages` -- but nothing downstream inspects
        # prose, and `success: true` is precisely the flag a consumer would
        # filter on. Null propagates as a real result. `model_ran` keeps the
        # distinction that the cycle itself did execute.
        return {
            "success": not missing,
            "model_ran": True,
            "outputs": outputs,
            "missing_outputs": missing,
            "messages": messages,
        }
    except KeyError as exc:
        return error_on_missing_session(str(session_id), exc)
    except Exception as exc:  # pragma: no cover
        return to_error(exc)
