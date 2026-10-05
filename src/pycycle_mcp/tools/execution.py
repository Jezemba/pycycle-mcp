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


# A design-point cycle is solved when the balance hit its thrust target and the low-pressure
# turbine ratio is physical. run_model uses Newton with err_on_non_converge=False, so a failed solve
# still returns, with numbers (measured: perf.Fn = 1.46e24 lbf reported as success). The audit's
# validity rule: |perf.Fn - target| small and balance.lpt_PR < 8.
FN_REL_TOLERANCE = 1e-3
LPT_PR_MAX = 8.0
_THRUST_TARGETS = ("Fn_DES", "balance.Fn_target")


def _scalar(problem: object, name: str) -> float | None:
    try:
        value = _to_serializable(problem.get_val(name))  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001 - absent on this archetype
        return None
    return float(value) if isinstance(value, (int, float)) else None


def convergence_check(problem: object) -> dict[str, object]:
    """Whether the design point is actually solved, with the numbers that say so."""
    fn = _scalar(problem, "perf.Fn")
    target_name, target = next(((n, v) for n in _THRUST_TARGETS
                                if (v := _scalar(problem, n)) is not None), (None, None))
    lpt_pr = _scalar(problem, "balance.lpt_PR")
    check: dict[str, object] = {"perf.Fn": fn, "thrust_target": target, "thrust_target_name": target_name,
                                "balance.lpt_PR": lpt_pr}
    reasons: list[str] = []
    values = [v for v in (fn, target, lpt_pr) if v is not None]
    if any(not np.isfinite(v) for v in values):
        reasons.append("a solved value is not finite")
    if fn is None or target is None:
        reasons.append("thrust or its target could not be read")
    else:
        rel = abs(fn - target) / max(abs(target), 1e-9)
        check["thrust_rel_error"] = rel
        if not np.isfinite(rel) or rel > FN_REL_TOLERANCE:
            reasons.append(f"perf.Fn misses its target by {rel:.3g} (> {FN_REL_TOLERANCE})")
    if lpt_pr is not None and np.isfinite(lpt_pr) and lpt_pr >= LPT_PR_MAX:
        reasons.append(f"balance.lpt_PR {lpt_pr:.3g} >= {LPT_PR_MAX} (not physical)")
    check["converged"] = not reasons
    check["reasons"] = reasons
    return check


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
                # B114: OpenMDAO answers an unknown name with a similarity search over every
                # internal variable, and its guesses are wrong -- 'SFC' was answered with
                # ['Fn_DES', 'perf...'] in geoauth_all8_7960 while the name wanted was
                # 'perf.TSFC'. Say the name is wrong and send the caller to the list that is
                # right, rather than passing a bad guess through.
                messages.append(
                    f"'{name}' is not an output of this cycle model. Do not guess another name: "
                    "call list_variables on this session for the exact output names and use one "
                    "of those."
                )

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
        # `success` says the names resolved; `converged` says the cycle is solved.
        check = convergence_check(problem)
        if not check["converged"]:
            messages.append("The cycle did NOT converge: " + "; ".join(check["reasons"]) +
                            ". Do not use these outputs as a design result.")
        return {
            "success": not missing,
            "model_ran": True,
            "converged": check["converged"],
            "convergence_check": check,
            "outputs": outputs,
            "missing_outputs": missing,
            "messages": messages,
        }
    except KeyError as exc:
        return error_on_missing_session(str(session_id), exc)
    except Exception as exc:  # pragma: no cover
        return to_error(exc)
