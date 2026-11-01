"""Off-design engine deck from a sized design-point cycle (Avion B3).

A pycycle-mcp session holds a design-point HBTF only: ``run_cycle`` sizes the engine to ``Fn_DES``
at one flight condition and returns one TSFC. A mission flies the engine everywhere else -- climb
from sea level, part-power cruise, idle descent -- so the mission needs the engine's thrust and
fuel flow over Mach x altitude x throttle. This module builds that table from the session's own
cycle: a multipoint model whose DESIGN point carries every independent input of the session (so it
reproduces the session's design point exactly), plus two off-design points (full power at the
design T4, and part power as a fraction of it), marched over the grid with continuation.

The table is written in Aviary's engine-deck format (same columns as Aviary's bundled
``turbofan_28k.deck``: Mach, altitude, throttle, gross thrust, ram drag, fuel flow) so aviary-mcp
can fly it through ``Aircraft.Engine.DATA_FILE``.

Ratings (Avion B3, 2026-11-01): maximum power holds the turbine inlet temperature at the take-off rating
(``T4_TAKEOFF``) low down, the climb rating (``T4_CLIMB``) in the climb and the design ``T4_MAX`` (cruise) at
and above the design altitude -- see ``rating_t4``. Part power is a
fraction of that maximum net thrust.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import time
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from multiprocessing import get_context

LOGGER = logging.getLogger(__name__)

# Grid default: covers the Avion mission (climb from M0.2 at sea level, cruise M0.78 at 32-34 kft
# within a 38 kft ceiling, descent to M0.36 at 500 ft) with margin, on the bench deck's breakpoints.
DEFAULT_MACHS: tuple[float, ...] = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9)
DEFAULT_ALTITUDES_FT: tuple[float, ...] = (0.0, 2000.0, 5000.0, 10000.0, 15000.0, 20000.0, 25000.0,
                                           30000.0, 33000.0, 35000.0, 37000.0, 39000.0, 41000.0)
DEFAULT_THROTTLES: tuple[float, ...] = (1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3)

# pyCycle's flight conditions are singular at M = 0; its own example runs "static" at M 0.001.
STATIC_MACH = 0.001

# The off-design inlet-flow balance in HBTF is bounded to [10, 1000] lbm/s by default, which a large
# engine exceeds at sea level. The deck model bounds it relative to the design flow instead.
OD_W_UPPER_PER_DESIGN_W = 10.0
# Off-design bypass ratio rises at part power: a BPR-8 design reached 15.6 at 30% thrust (measured), and
# HBTF's fixed bound of 10 failed 6 of 10 points on that engine vs 0 of 48 with 20. The bound follows the design.
OD_BPR_UPPER_PER_DESIGN_BPR = 3.0
OD_BPR_UPPER_MIN = 10.0

# Converged off-design solves took <= 8 Newton iterations (measured); a solve still unconverged at
# this cap is failing, and the HBTF default of 50 only made failures cost ~70 s each.
OD_NEWTON_MAXITER = 20

# An off-design point is solved when every residual is small and the part-power thrust matches its
# target. Newton runs with err_on_non_converge=False, so a failed solve still returns numbers.
RESIDUAL_TOL = 1e-6
THRUST_REL_TOL = 1e-4

# Continuation: grid points are reached in steps of at most this much, halving a step that fails.
MACH_STEP = 0.1
ALT_STEP_FT = 8000.0
MAX_HALVINGS = 4
MAX_THROTTLE_HALVINGS = 1
MACH_CHUNK = 2
DEFAULT_DECK_WORKERS = 12

# Thrust a size-invariant cycle's deck is generated at (pyCycle's canonical Fn_DES); see deck_inputs.
REFERENCE_FN_DES_LBF = 5900.0

DECK_COLUMNS = ("Mach_Number (unitless)", "Altitude (ft)", "Throttle (unitless)",
                "Gross_Thrust (lbf)", "Ram_Drag (lbf)", "Fuel_Flow (lb/h)")

# Session inputs that describe the design flight condition or the sizing targets. They are copied
# to the DESIGN point only; the off-design points get their own flight condition and throttle.
_DESIGN_ONLY_PREFIXES = ("fc.",)
_DESIGN_ONLY_NAMES = {"Fn_DES", "T4_MAX", "PC", "Fn_max", "LP_Nmech", "HP_Nmech", "splitter.BPR",
                      "T4_TAKEOFF", "T4_CLIMB"}


@dataclass
class DeckPoint:
    mach: float
    alt_ft: float
    throttle: float
    gross_thrust_lbf: float
    ram_drag_lbf: float
    fuel_flow_lbh: float
    net_thrust_lbf: float
    t4_degR: float
    bpr: float = float("nan")


@dataclass
class DeckResult:
    points: list[DeckPoint]
    failed: list[dict[str, float | str]]
    design: dict[str, float]
    design_inputs: dict[str, dict[str, object]]
    grid: dict[str, list[float]]
    wall_seconds: float
    meta: dict[str, object] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Pure helpers (no pyCycle): deck text, reading it back, grid ordering, keys
# ---------------------------------------------------------------------------

# Bump when a change to the builder changes the deck for the same inputs (solver bounds, continuation,
# point acceptance). Part of the key, so a new builder never overwrites or reuses an old deck
# (2026-10-31: the BPR-bound fix regenerated a deck under the same key and overwrote the old one).
DECK_BUILDER_VERSION = 4   # 3: take-off / climb ratings; 4: canonical input values (2026-11-01)


def design_key(design_inputs: dict[str, dict[str, object]], grid: dict[str, Sequence[float]]) -> str:
    """Stable hash of everything the deck depends on -- same inputs and builder give the same key."""
    payload = json.dumps({"inputs": design_inputs, "grid": {k: list(v) for k, v in grid.items()},
                          "builder": DECK_BUILDER_VERSION}, sort_keys=True, default=float)
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def format_deck(points: Iterable[DeckPoint], header_lines: Sequence[str] = ()) -> str:
    """Aviary engine-deck text: comment header, column line, one row per point (sorted)."""
    rows = sorted(points, key=lambda p: (p.mach, p.alt_ft, p.throttle))
    out = [f"# {line}" if line else "#" for line in header_lines]
    out.append("")
    out.append(", ".join(DECK_COLUMNS))
    for p in rows:
        out.append(f"{p.mach:.4f}, {p.alt_ft:.1f}, {p.throttle:.4f}, {p.gross_thrust_lbf:.4f}, "
                   f"{p.ram_drag_lbf:.4f}, {p.fuel_flow_lbh:.4f}")
    return "\n".join(out) + "\n"


MIN_THROTTLES_PER_CELL = 2


def usable_points(points: Iterable[DeckPoint]) -> tuple[list[DeckPoint], list[dict[str, float]]]:
    """Points Aviary can use, and the (Mach, altitude) cells dropped.

    Aviary's flight-idle generation extrapolates from the two lowest throttle points of every
    (Mach, altitude) cell; a cell with one point makes its column arrays different lengths and the
    EngineDeck fails to build (measured 2026-10-31: a T4 deck whose M0 / 37,000 ft cell solved only
    full power -> "inhomogeneous shape" in EngineDeck._sort_data). Such cells are left out.
    """
    cells: dict[tuple[float, float], list[DeckPoint]] = {}
    for p in points:
        cells.setdefault((p.mach, p.alt_ft), []).append(p)
    kept: list[DeckPoint] = []
    dropped: list[dict[str, float]] = []
    for (mach, alt), pts in sorted(cells.items()):
        if len({p.throttle for p in pts}) >= MIN_THROTTLES_PER_CELL:
            kept.extend(pts)
        else:
            dropped.append({"mach": mach, "alt_ft": alt, "points": len(pts)})
    return kept, dropped


def read_deck(text: str) -> list[dict[str, float]]:
    """Parse deck text written by :func:`format_deck` (comments and blank lines skipped)."""
    rows: list[dict[str, float]] = []
    header: list[str] | None = None
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        cells = [c.strip() for c in line.split(",")]
        if header is None:
            header = [c.split(" (")[0].lower() for c in cells]
            continue
        rows.append({name: float(value) for name, value in zip(header, cells, strict=True)})
    return rows


def sls_net_thrust(rows: Sequence[dict[str, float]]) -> float | None:
    """Maximum sea-level static net thrust (the reference Aviary scales engine mass with)."""
    static = [r["gross_thrust"] - r["ram_drag"] for r in rows
              if abs(r["mach_number"]) < 1e-6 and abs(r["altitude"]) < 1e-6]
    return max(static) if static else None


# Ratings (Avion B3, Jessica 2026-11-01: "a hotter take-off/climb setting, not a constant turbine temperature").
# Maximum power holds T4 at the take-off rating up to the thrust-reduction altitude, at the climb rating up to
# the design altitude, and at the design T4 (the cruise rating, which sizes the engine) from there up. Short
# linear ramps join them so the off-design continuation can step across a rating change; at the deck's
# altitudes the values are exact (0 ft take-off; 2,000 ft up to design - 2,000 ft climb; design alt and above cruise).
RATING_TAKEOFF_ALT_FT = 1500.0
RATING_TAKEOFF_RAMP_FT = 500.0
RATING_CRUISE_RAMP_FT = 2000.0


def rating_t4(alt_ft: float, design_inputs: dict[str, dict[str, object]], design_alt_ft: float) -> float:
    """Turbine inlet temperature (degR) at maximum power at this altitude (take-off / climb / cruise rating)."""
    t4_cruise = float(design_inputs["T4_MAX"]["value"])  # type: ignore[arg-type]
    t4_to = float(design_inputs.get("T4_TAKEOFF", {"value": t4_cruise})["value"])  # type: ignore[arg-type]
    t4_clb = float(design_inputs.get("T4_CLIMB", {"value": t4_cruise})["value"])  # type: ignore[arg-type]
    xs = [RATING_TAKEOFF_ALT_FT, RATING_TAKEOFF_ALT_FT + RATING_TAKEOFF_RAMP_FT,
          design_alt_ft - RATING_CRUISE_RAMP_FT, design_alt_ft]
    ys = [t4_to, t4_clb, t4_clb, t4_cruise]
    if xs[2] < xs[1]:          # design altitude below the take-off band: climb rating has no room
        xs, ys = xs[:2] + [max(design_alt_ft, xs[1])], ys[:2] + [t4_cruise]
    if alt_ft <= xs[0]:
        return ys[0]
    for (x0, y0), (x1, y1) in zip(zip(xs, ys), zip(xs[1:], ys[1:])):
        if alt_ft <= x1:
            return y0 + (y1 - y0) * (alt_ft - x0) / max(x1 - x0, 1e-9)
    return ys[-1]


def chunk_jobs(machs: Sequence[float], altitudes: Sequence[float], design_mach: float,
               chunk: int = 3) -> list[tuple[float, tuple[float, ...]]]:
    """Split the grid into (altitude, Mach run) jobs; each run is ordered from the design Mach outward.

    A job walks from the design condition to its first Mach and then steps along its run, each
    point starting from the previous solution, so no step jumps far.
    """
    ordered = sorted(float(m) for m in machs)
    runs = [ordered[i:i + chunk] for i in range(0, len(ordered), chunk)]
    jobs = []
    for alt in sorted(float(a) for a in altitudes):
        for run in runs:
            if abs(run[-1] - design_mach) < abs(run[0] - design_mach):
                run = run[::-1]
            jobs.append((alt, tuple(run)))
    return jobs


def size_invariant(design_inputs: dict[str, dict[str, object]]) -> bool:
    """True when no input is an absolute power or flow, so thrust and fuel flow scale exactly with size.

    Measured 2026-10-31 on the session cycle (HPX 0, all bleeds fractional): tripling Fn_DES
    multiplied every off-design thrust, ram drag and fuel flow by 3.00000000 +/- 1e-8. Shaft power
    extraction (``*.HPX``) is the absolute term that would break it.
    """
    for name, entry in design_inputs.items():
        if name.endswith("HPX") and float(entry["value"]) != 0.0:  # type: ignore[arg-type]
            return False
    return True


def deck_inputs(design_inputs: dict[str, dict[str, object]]) -> tuple[dict[str, dict[str, object]], bool]:
    """The inputs the deck is generated from, and whether it is size-normalised.

    A size-invariant cycle is generated at ``REFERENCE_FN_DES_LBF`` so the deck depends on the cycle
    only (BPR, pressure ratios, T4, efficiencies, design condition) and not on how the framework sized
    it; the mission then scales it by Fn_DES / REFERENCE_FN_DES_LBF (Aviary's SCALE_FACTOR, exact here).
    """
    if not size_invariant(design_inputs):
        return design_inputs, False
    normalised = dict(design_inputs)
    normalised["Fn_DES"] = {"value": REFERENCE_FN_DES_LBF, "units": "lbf"}
    return normalised, True


# ---------------------------------------------------------------------------
# Session -> design inputs
# ---------------------------------------------------------------------------

# A session's unset inputs drift by round-off while run_cycle solves: measured 2026-11-01, 28 of them (bleed fractions
# 0 -> ~1e-22, nozzle Cv 1.0 -> 0.9999999999999996, duct11.MN 0.3063 -> 0.30630000000000007), so two sessions of
# the same cycle gave different deck keys and every geometry change rebuilt a 45-minute deck. Values are taken to
# 10 significant figures and |x| < 1e-12 counts as 0, for the key AND the deck, so the deck depends on the cycle only.
CANONICAL_SIG_FIGS = 10
CANONICAL_ZERO = 1e-12


def canonical(x: float) -> float:
    """A design input as the deck uses it: round-off removed (see CANONICAL_SIG_FIGS)."""
    x = float(x)
    if abs(x) < CANONICAL_ZERO:
        return 0.0
    return float(f"{x:.{CANONICAL_SIG_FIGS}g}")


def session_design_inputs(problem: object) -> dict[str, dict[str, object]]:
    """Every independent input of a design-point session, with its value and units.

    "Independent" means fed by OpenMDAO's automatic IVC -- the values a caller can set. Copying all
    of them (not only the curated design list) makes the deck's DESIGN point the session's cycle,
    including anything an agent set through ``set_inputs`` beyond the canonical list.
    """
    import numpy as np

    model = problem.model  # type: ignore[attr-defined]
    items = model.list_inputs(prom_name=True, units=True, is_indep_var=True, out_stream=None)
    seen: dict[str, dict[str, object]] = {}
    for _abs_name, meta in items:
        prom = meta["prom_name"]
        if prom in seen:
            continue
        units = meta.get("units")
        value = problem.get_val(prom, units=units) if units else problem.get_val(prom)  # type: ignore[attr-defined]
        arr = np.asarray(value, dtype=float).ravel()
        seen[prom] = {"value": canonical(arr[0]) if arr.size == 1 else [canonical(v) for v in arr],
                      "units": units}
    return dict(sorted(seen.items()))


# ---------------------------------------------------------------------------
# The multipoint model (pyCycle imported lazily: workers and tests import this module cheaply)
# ---------------------------------------------------------------------------

def _build_problem(design_inputs: dict[str, dict[str, object]], w_upper: float, bpr_upper: float = OD_BPR_UPPER_MIN):
    import openmdao.api as om
    import pycycle.api as pyc

    from .cycles.high_bypass_turbofan import HBTF

    class DeckCycle(pyc.MPCycle):
        def setup(self):
            self.pyc_add_pnt("DESIGN", HBTF(thermo_method="CEA"))
            self.pyc_add_pnt("OD_full_pwr", HBTF(design=False, thermo_method="CEA", throttle_mode="T4",
                                                 od_W_upper=w_upper, od_BPR_upper=bpr_upper))
            self.pyc_add_pnt("OD_part_pwr", HBTF(design=False, thermo_method="CEA",
                                                 throttle_mode="percent_thrust", od_W_upper=w_upper, od_BPR_upper=bpr_upper))
            self.connect("OD_full_pwr.perf.Fn", "OD_part_pwr.Fn_max")
            for pt in ("OD_full_pwr", "OD_part_pwr"):
                self.set_input_defaults(f"{pt}.fc.MN", 0.8)
                self.set_input_defaults(f"{pt}.fc.alt", 35000.0, units="ft")
                self.set_input_defaults(f"{pt}.fc.dTs", 0.0, units="degR")
            for name in ("LP_Nmech", "HP_Nmech"):
                entry = design_inputs.get(name)
                if entry is not None:
                    self.set_input_defaults(f"DESIGN.{name}", entry["value"], units=entry["units"])
            self.pyc_use_default_des_od_conns()
            self.pyc_connect_des_od("core_nozz.Throat:stat:area", "balance.rhs:W")
            self.pyc_connect_des_od("byp_nozz.Throat:stat:area", "balance.rhs:BPR")
            super().setup()

    prob = om.Problem(reports=False)
    prob.model = DeckCycle()
    prob.setup(check=False)
    for pt in ("OD_full_pwr", "OD_part_pwr"):
        getattr(prob.model, pt).nonlinear_solver.options["maxiter"] = OD_NEWTON_MAXITER
    return prob


def _set(prob, name: str, entry: dict[str, object]) -> bool:
    try:
        if entry.get("units"):
            prob.set_val(name, entry["value"], units=entry["units"])
        else:
            prob.set_val(name, entry["value"])
        return True
    except Exception:  # noqa: BLE001 - name absent on this point
        return False


def _apply_inputs(prob, design_inputs: dict[str, dict[str, object]]) -> None:
    indep = {meta["prom_name"] for _n, meta in
             prob.model.list_inputs(prom_name=True, is_indep_var=True, out_stream=None)}
    for name, entry in design_inputs.items():
        if f"DESIGN.{name}" in indep:
            _set(prob, f"DESIGN.{name}", entry)
        if name.startswith(_DESIGN_ONLY_PREFIXES) or name in _DESIGN_ONLY_NAMES:
            continue
        for pt in ("OD_full_pwr", "OD_part_pwr"):
            if f"{pt}.{name}" in indep:
                _set(prob, f"{pt}.{name}", entry)
    t4 = design_inputs.get("T4_MAX")
    if t4 is not None:
        _set(prob, "OD_full_pwr.T4_MAX", t4)


def _val(prob, name: str, units: str | None = None) -> float:
    import numpy as np

    v = prob.get_val(name, units=units) if units else prob.get_val(name)
    return float(np.asarray(v, dtype=float).ravel()[0])


def solve_design_point(design_inputs: dict[str, dict[str, object]]) -> dict[str, float]:
    """Solve the design point alone, in a design-only HBTF built the way a session builds it."""
    import openmdao.api as om

    from .cycles.high_bypass_turbofan import HBTF

    prob = om.Problem(reports=False)
    prob.model = HBTF(thermo_method="CEA")
    prob.setup(check=False)
    for name, entry in design_inputs.items():
        _set(prob, name, entry)
    # Newton initial guesses from create_model._apply_design_defaults (the session's own)
    prob["balance.FAR"] = 0.025
    prob["balance.W"] = 100.0
    prob["balance.lpt_PR"] = 4.0
    prob["balance.hpt_PR"] = 3.0
    prob["fc.balance.Pt"] = 5.2
    prob["fc.balance.Tt"] = 440.0
    prob.set_solver_print(level=-1)
    prob.run_model()
    return {
        "Fn_lbf": _val(prob, "perf.Fn", "lbf"),
        "TSFC": _val(prob, "perf.TSFC"),
        "W_lbms": _val(prob, "balance.W", "lbm/s"),
        "FAR": _val(prob, "balance.FAR"),
        "BPR": _val(prob, "splitter.BPR"),
        "LP_Nmech": _val(prob, "LP_Nmech", "rpm"),
        "HP_Nmech": _val(prob, "HP_Nmech", "rpm"),
        "hpt_PR": _val(prob, "balance.hpt_PR"),
        "lpt_PR": _val(prob, "balance.lpt_PR"),
        "Fn_DES": _val(prob, "Fn_DES", "lbf"),
        "alt_ft": _val(prob, "fc.alt", "ft"),
        "MN": _val(prob, "fc.MN"),
        "T4_degR": _val(prob, "burner.Fl_O:tot:T", "degR"),
        "fc_Pt": _val(prob, "fc.balance.Pt"),
        "fc_Tt": _val(prob, "fc.balance.Tt"),
        "residual_norm": float(prob.model._residuals.get_norm()),
    }


def _seed(prob, design: dict[str, float]) -> None:
    """Every point's Newton guesses = the design solution; off-design starts AT the design condition."""
    prob["DESIGN.balance.FAR"] = design["FAR"]
    prob["DESIGN.balance.W"] = design["W_lbms"]
    prob["DESIGN.balance.lpt_PR"] = design["lpt_PR"]
    prob["DESIGN.balance.hpt_PR"] = design["hpt_PR"]
    for pt in ("DESIGN", "OD_full_pwr", "OD_part_pwr"):
        prob[f"{pt}.fc.balance.Pt"] = design["fc_Pt"]
        prob[f"{pt}.fc.balance.Tt"] = design["fc_Tt"]
    for pt in ("OD_full_pwr", "OD_part_pwr"):
        prob[f"{pt}.balance.FAR"] = design["FAR"]
        prob[f"{pt}.balance.W"] = design["W_lbms"]
        prob[f"{pt}.balance.BPR"] = design["BPR"]
        prob[f"{pt}.balance.lp_Nmech"] = design["LP_Nmech"]
        prob[f"{pt}.balance.hp_Nmech"] = design["HP_Nmech"]
        prob[f"{pt}.hpt.PR"] = design["hpt_PR"]
        prob[f"{pt}.lpt.PR"] = design["lpt_PR"]
        for comp in ("fan", "lpc", "hpc"):
            prob[f"{pt}.{comp}.map.RlineMap"] = 2.0


def _set_condition(prob, mach: float, alt_ft: float) -> None:
    m = max(mach, STATIC_MACH)
    for pt in ("OD_full_pwr", "OD_part_pwr"):
        prob.set_val(f"{pt}.fc.MN", m)
        prob.set_val(f"{pt}.fc.alt", alt_ft, units="ft")


def _solved(prob, throttle: float) -> tuple[bool, str]:
    res_full = float(prob.model.OD_full_pwr._residuals.get_norm())
    res_part = float(prob.model.OD_part_pwr._residuals.get_norm())
    fn_full = _val(prob, "OD_full_pwr.perf.Fn", "lbf")
    fn_part = _val(prob, "OD_part_pwr.perf.Fn", "lbf")
    wf = _val(prob, "OD_part_pwr.perf.Wfuel", "lbm/h")
    values = (res_full, res_part, fn_full, fn_part, wf)
    if not all(math.isfinite(v) for v in values):
        return False, "non-finite value"
    if res_full > RESIDUAL_TOL or res_part > RESIDUAL_TOL:
        return False, f"residual norm {max(res_full, res_part):.3g} > {RESIDUAL_TOL}"
    if fn_full <= 0 or wf <= 0:
        return False, f"non-physical (Fn_max {fn_full:.4g} lbf, Wfuel {wf:.4g} lbm/h)"
    rel = abs(fn_part - throttle * fn_full) / max(abs(fn_full), 1e-9)
    if rel > THRUST_REL_TOL:
        return False, f"part-power thrust misses target by {rel:.3g}"
    return True, ""


def _snapshot(prob):
    return prob.model._outputs.asarray().copy()


def _restore(prob, saved) -> None:
    prob.model._outputs.set_val(saved)


def _continue(prob, run: Callable[[], None], apply: Callable[[tuple[float, ...]], None],
              check: Callable[[tuple[float, ...]], tuple[bool, str]],
              start: tuple[float, ...], target: tuple[float, ...], steps: int,
              max_halvings: int = MAX_HALVINGS) -> tuple[bool, str]:
    """Move a solved model from ``start`` to ``target`` in ``steps`` equal steps (continuation).

    A step that does not solve is undone (outputs restored to the last solved state) and replaced by
    its midpoint, at most ``max_halvings`` times in the whole walk. Deterministic: the same inputs
    take the same path.
    """
    path = [tuple(s + (t - s) * k / steps for s, t in zip(start, target, strict=True))
            for k in range(1, steps + 1)]
    good, current, halvings = _snapshot(prob), start, 0
    while path:
        nxt = path[0]
        apply(nxt)
        run()
        ok, why = check(nxt)
        if ok:
            good, current = _snapshot(prob), nxt
            path.pop(0)
            continue
        _restore(prob, good)
        apply(current)
        if halvings >= max_halvings:
            return False, why
        path.insert(0, tuple((c + n) / 2.0 for c, n in zip(current, nxt, strict=True)))
        halvings += 1
    return True, ""


def _steps(d_mach: float, d_alt: float) -> int:
    return max(1, math.ceil(max(abs(d_mach) / MACH_STEP, abs(d_alt) / ALT_STEP_FT)))


def _run_chunk(args: tuple) -> dict[str, object]:
    """Worker: one altitude and a contiguous run of Mach numbers (own process, own model)."""
    design_inputs, design, alt_ft, machs, throttles, w_upper, bpr_upper = args
    t_setup = time.time()
    prob = _build_problem(design_inputs, w_upper, bpr_upper)
    _apply_inputs(prob, design_inputs)
    _seed(prob, design)
    prob.set_solver_print(level=-1)
    t_setup = time.time() - t_setup
    run_times: list[float] = []

    raised: list[str] = []

    def run() -> None:
        # pyCycle can raise inside a solve that wanders off (log of zero, NaN, singular Jacobian);
        # measured 2026-10-31: one such point killed four whole deck runs. It is a failed point.
        t = time.time()
        raised.clear()
        try:
            prob.run_model()
        except Exception as exc:  # noqa: BLE001
            raised.append(f"{type(exc).__name__}: {str(exc).splitlines()[0][:160]}")
        run_times.append(time.time() - t)

    def checked(throttle: float) -> tuple[bool, str]:
        return (False, f"solver raised {raised[0]}") if raised else _solved(prob, throttle)

    def at_full_power(cond: tuple[float, ...]) -> None:
        prob.set_val("OD_part_pwr.PC", 1.0)
        _set_condition(prob, cond[0], cond[1])
        prob.set_val("OD_full_pwr.T4_MAX", rating_t4(cond[1], design_inputs, design["alt_ft"]), units="degR")

    def set_throttle(pc: tuple[float, ...]) -> None:
        prob.set_val("OD_part_pwr.PC", pc[0])

    full_check = lambda _cond: checked(1.0)  # noqa: E731
    part_check = lambda pc: checked(pc[0])  # noqa: E731
    points: list[dict[str, float]] = []
    failed: list[dict[str, object]] = []

    def result() -> dict[str, object]:
        timing = {"setup_s": round(t_setup, 2), "runs": len(run_times),
                  "run_s_total": round(sum(run_times), 2), "run_s_max": round(max(run_times, default=0.0), 2)}
        return {"alt_ft": alt_ft, "machs": list(machs), "points": points, "failed": failed, "timing": timing}

    # the off-design points start AT the design condition, where the design solution is their answer
    here = (design["MN"], design["alt_ft"])
    at_full_power(here)
    run()
    ok, why = full_check(here)
    if not ok:
        failed.extend({"mach": m, "alt_ft": alt_ft, "throttle": None,
                       "reason": f"off-design solve at the design condition failed: {why}"} for m in machs)
        return result()

    for mach in machs:
        target = (float(mach), float(alt_ft))
        ok, why = _continue(prob, run, at_full_power, full_check, here, target,
                            _steps(target[0] - here[0], target[1] - here[1]))
        if not ok:
            failed.append({"mach": mach, "alt_ft": alt_ft, "throttle": 1.0, "reason": f"reaching condition: {why}"})
            continue  # the model is back at `here`; the next Mach walks from there
        here = target
        full_power = _snapshot(prob)
        previous = 1.0
        for throttle in throttles:
            if throttle != 1.0:
                ok, why = _continue(prob, run, set_throttle, part_check, (previous,), (throttle,), 1,
                                    max_halvings=MAX_THROTTLE_HALVINGS)
                if not ok:
                    failed.append({"mach": mach, "alt_ft": alt_ft, "throttle": throttle, "reason": why})
                    break  # lower throttles would continue from an unsolved state
                previous = throttle
            points.append({
                "mach": mach, "alt_ft": alt_ft, "throttle": throttle,
                "gross_thrust_lbf": _val(prob, "OD_part_pwr.perf.Fg", "lbf"),
                "ram_drag_lbf": _val(prob, "OD_part_pwr.inlet.F_ram", "lbf"),
                "fuel_flow_lbh": _val(prob, "OD_part_pwr.perf.Wfuel", "lbm/h"),
                "net_thrust_lbf": _val(prob, "OD_part_pwr.perf.Fn", "lbf"),
                "t4_degR": _val(prob, "OD_part_pwr.burner.Fl_O:tot:T", "degR"),
                "bpr": _val(prob, "OD_part_pwr.balance.BPR"),
            })
        _restore(prob, full_power)
        at_full_power(here)
    return result()


def _lowest_priority() -> None:
    """Deck workers yield the CPU to everything else on a shared machine (nice 19)."""
    try:
        os.nice(19 - os.nice(0))
    except OSError:  # pragma: no cover - not permitted here; run at the inherited priority
        pass


def default_workers(n_jobs: int) -> int:
    """Worker processes: ``PYCYCLE_DECK_WORKERS`` if set, else ``DEFAULT_DECK_WORKERS``.

    ws8 is shared: 65 deck workers at once loaded it to ~90 and Jessica asked to keep it down (2026-10-31).
    12 workers make a full deck in 32-51 min (measured, nice 19, other jobs running).
    """
    env = os.environ.get("PYCYCLE_DECK_WORKERS")
    if env:
        return max(1, min(n_jobs, int(env)))
    return max(1, min(n_jobs, DEFAULT_DECK_WORKERS))


def generate_deck(design_inputs: dict[str, dict[str, object]],
                  machs: Sequence[float] = DEFAULT_MACHS,
                  altitudes_ft: Sequence[float] = DEFAULT_ALTITUDES_FT,
                  throttles: Sequence[float] = DEFAULT_THROTTLES,
                  workers: int | None = None, chunk: int = MACH_CHUNK) -> DeckResult:
    """Run the off-design grid for a design (parallel over altitude x Mach chunks) and collect the deck."""
    throttles = sorted(set(float(t) for t in throttles), reverse=True)
    if not throttles or throttles[0] != 1.0:
        raise ValueError("throttles must include 1.0 (maximum power at the design T4)")
    grid = {"mach": sorted(float(m) for m in machs), "altitude_ft": sorted(float(a) for a in altitudes_ft),
            "throttle": throttles}
    start = time.time()

    # the design solution seeds every worker and sets the off-design flow bound
    design = solve_design_point(design_inputs)
    w_upper = max(1000.0, OD_W_UPPER_PER_DESIGN_W * design["W_lbms"])
    bpr_upper = max(OD_BPR_UPPER_MIN, OD_BPR_UPPER_PER_DESIGN_BPR * design["BPR"])

    jobs = [(design_inputs, design, alt, ms, throttles, w_upper, bpr_upper)
            for alt, ms in chunk_jobs(grid["mach"], grid["altitude_ft"], design["MN"], chunk)]
    n = workers or default_workers(len(jobs))
    if n <= 1:
        results = [_run_chunk(j) for j in jobs]
    else:
        with ProcessPoolExecutor(max_workers=n, mp_context=get_context("spawn"),
                                 initializer=_lowest_priority) as pool:
            results = list(pool.map(_run_chunk, jobs))

    points = [DeckPoint(**p) for r in results for p in r["points"]]  # type: ignore[arg-type]
    failed = [f for r in results for f in r["failed"]]  # type: ignore[union-attr]
    timings = [r["timing"] for r in results]
    return DeckResult(points=points, failed=failed, design=design, design_inputs=design_inputs,
                      grid=grid, wall_seconds=time.time() - start,
                      meta={"w_upper_lbms": w_upper, "bpr_upper": bpr_upper, "workers": n, "jobs": len(jobs),
                            "cpu_seconds": round(sum(t["run_s_total"] + t["setup_s"] for t in timings), 1),
                            "max_job_seconds": round(max(t["run_s_total"] + t["setup_s"] for t in timings), 1),
                            "solves": sum(t["runs"] for t in timings)})



def write_deck(result: DeckResult, path: str, key: str) -> dict[str, object]:
    """Write the deck file and a JSON sidecar describing how it was made."""
    kept, dropped = usable_points(result.points)
    d = result.design
    header = [
        "pyCycle-generated engine deck (pycycle-mcp generate_engine_deck, Avion B3)",
        f"design key {key}: Fn_DES {d['Fn_DES']:.1f} lbf at M{d['MN']:.3f} / {d['alt_ft']:.0f} ft, "
        f"BPR {d['BPR']:.4f}, T4 {d['T4_degR']:.1f} degR, design TSFC {d['TSFC']:.5f} lbm/h/lbf",
        f"max power: T4 {rating_t4(0.0, result.design_inputs, d['alt_ft']):.1f} degR take-off (<= {RATING_TAKEOFF_ALT_FT:.0f} ft), "
        f"{rating_t4(d['alt_ft'] - RATING_CRUISE_RAMP_FT, result.design_inputs, d['alt_ft']):.1f} degR climb, "
        f"{d['T4_degR']:.1f} degR cruise (>= design altitude); throttle = fraction of max net thrust there",
        f"Mach 0 rows were solved at M{STATIC_MACH} (pyCycle flight conditions are singular at M 0)",
    ]
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(format_deck(kept, header))
    rows = read_deck(open(path, encoding="utf-8").read())
    sidecar = {
        "deck_path": os.path.abspath(path),
        "design_key": key,
        "design": result.design,
        "grid": result.grid,
        "n_points": len(kept),
        "dropped_cells": dropped,
        "cells": [{"mach": m, "alt_ft": a} for m, a in sorted({(p.mach, p.alt_ft) for p in kept})],
        "n_failed": len(result.failed),
        "failed": result.failed,
        "sls_net_thrust_lbf": sls_net_thrust(rows),
        "wall_seconds": round(result.wall_seconds, 2),
        "meta": result.meta,
        "design_inputs": result.design_inputs,
    }
    with open(os.path.splitext(path)[0] + ".json", "w", encoding="utf-8") as fh:
        json.dump(sidecar, fh, indent=1, sort_keys=True)
    return sidecar
