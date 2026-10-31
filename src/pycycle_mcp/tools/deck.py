"""Engine deck tool: the session's engine over the flight envelope, for the mission (Avion B3)."""

from __future__ import annotations

import json
import logging
import os

from .. import engine_deck as ed
from ..errors import error_response, to_error
from ..session_manager import session_manager
from ..utils import error_on_missing_session

LOGGER = logging.getLogger(__name__)

# The mission envelope a deck must cover with at least two throttles per cell (aviary-mcp's default
# phase_info: climb from M0.18 at sea level, cruise up to the mission Mach + 0.02 below a 38,000 ft
# ceiling, descent to M0.34). Cells outside it (static at 39,000 ft, ...) may fail without harm.
ENVELOPE_MACH = (0.2, 0.85)
ENVELOPE_ALT_FT = (0.0, 38000.0)

# Same validity rule as run_cycle's convergence report.
_FN_REL_TOL = 1e-3
_LPT_PR_MAX = 8.0


def deck_dir() -> str:
    """Where decks are cached: ``PYCYCLE_DECK_DIR`` or ``./engine_decks`` (the server's directory)."""
    return os.path.abspath(os.environ.get("PYCYCLE_DECK_DIR") or "engine_decks")


def envelope_holes(sidecar: dict) -> list[dict[str, float]]:
    """Grid cells inside the mission envelope that the written deck does not have."""
    present = {(c["mach"], c["alt_ft"]) for c in sidecar.get("cells", [])}
    grid = sidecar["grid"]
    return [{"mach": m, "alt_ft": a}
            for m in grid["mach"] if ENVELOPE_MACH[0] <= m <= ENVELOPE_MACH[1]
            for a in grid["altitude_ft"] if ENVELOPE_ALT_FT[0] <= a <= ENVELOPE_ALT_FT[1]
            if (m, a) not in present]


def _summary(sidecar: dict, cached: bool) -> dict[str, object]:
    d = sidecar["design"]
    return {
        "success": True,
        "cached": cached,
        "deck_file": sidecar["deck_path"],
        "design_key": sidecar["design_key"],
        "reference_fn_des_lbf": sidecar["reference_fn_des_lbf"],
        "size_normalised": sidecar["size_normalised"],
        "design_tsfc": d["TSFC"],
        "design_bpr": d["BPR"],
        "sls_net_thrust_lbf": sidecar["sls_net_thrust_lbf"],
        "n_points": sidecar["n_points"],
        "n_failed": sidecar["n_failed"],
        "envelope_holes": envelope_holes(sidecar),
        "wall_seconds": sidecar["wall_seconds"],
        "messages": [
            "Fly this deck in aviary with set_engine_deck(deck_file=...). SCALE_FACTOR = design thrust / "
            f"{sidecar['reference_fn_des_lbf']:.1f} lbf gives the engine at its sized thrust."
        ],
    }


def generate_engine_deck(payload: dict[str, object]) -> dict[str, object]:
    """Thrust and fuel flow of the session's engine over Mach x altitude x throttle, as an Aviary deck.

    The deck depends on the cycle only (size-invariant cycles are generated at a reference thrust);
    a deck already made for the same cycle is returned from the cache without solving again.
    """
    session_id = payload.get("session_id")
    if not session_id:
        return error_response("ValidationError", "session_id is required")
    try:
        problem, _ = session_manager.get(str(session_id))
    except KeyError as exc:
        return error_on_missing_session(str(session_id), exc)
    try:
        inputs, normalised = ed.deck_inputs(ed.session_design_inputs(problem))
        grid = {"mach": sorted(ed.DEFAULT_MACHS), "altitude_ft": sorted(ed.DEFAULT_ALTITUDES_FT),
                "throttle": sorted(ed.DEFAULT_THROTTLES, reverse=True)}
        key = ed.design_key(inputs, grid)
        path = os.path.join(deck_dir(), f"deck_{key}.deck")
        side_path = os.path.splitext(path)[0] + ".json"
        if os.path.isfile(path) and os.path.isfile(side_path):
            with open(side_path, encoding="utf-8") as fh:
                return _summary(json.load(fh), cached=True)

        design = ed.solve_design_point(inputs)
        rel = abs(design["Fn_lbf"] - design["Fn_DES"]) / max(abs(design["Fn_DES"]), 1e-9)
        if not (rel <= _FN_REL_TOL and design["lpt_PR"] < _LPT_PR_MAX):
            return error_response(
                "DesignNotConverged",
                f"The design point of this cycle does not solve (thrust misses its target by {rel:.3g}, "
                f"LPT PR {design['lpt_PR']:.3g}); no deck can be made from it. Change the cycle inputs "
                "and run_cycle until it converges.",
                details={"design": design},
            )
        workers = payload.get("workers")
        result = ed.generate_deck(inputs, workers=int(workers) if workers else None)
        sidecar = ed.write_deck(result, path, key)
        sidecar["reference_fn_des_lbf"] = float(inputs["Fn_DES"]["value"])  # type: ignore[arg-type]
        sidecar["size_normalised"] = normalised
        with open(side_path, "w", encoding="utf-8") as fh:
            json.dump(sidecar, fh, indent=1, sort_keys=True)
        LOGGER.info("engine deck %s: %d points, %d failed, %.0f s", key, sidecar["n_points"],
                    sidecar["n_failed"], sidecar["wall_seconds"])
        return _summary(sidecar, cached=False)
    except Exception as exc:  # pragma: no cover - surfaced as a structured error
        LOGGER.exception("generate_engine_deck failed")
        return to_error(exc)
