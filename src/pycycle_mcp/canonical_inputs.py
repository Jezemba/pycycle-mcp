"""Curated lists of the design-point input variables for each built-in cycle.

When an agent calls ``create_cycle_model``, ``list_variables`` returns the
full 900+ promoted-variable list from the underlying OpenMDAO model. That's
useful for inspection but not for *driving* the cycle — most of those
entries are flow-station internals, residuals, balance-state variables,
etc. There is no signal in the response telling the agent which handful
are the actual design dials.

``get_design_inputs`` returns the small curated list that's safe to
pass to ``set_inputs``. The paths match exactly the ones that the
internal ``_apply_design_defaults`` already exercises in
``tools/create_model.py`` for each cycle type, so they're guaranteed
to be settable. Each entry carries enough metadata for the agent to
pick sensible values without guessing.

Avoids the failure mode observed in pipeline Run #17 (2026-05-23):
the propulsion agent guessed ``fan.map.design.BPR`` and several
similar made-up names, set_inputs returned ``Variable not found``
warnings, key inputs (most importantly BPR) were never set, and
the Newton solver ran on an under-constrained model and emitted
non-physical thrust (-2.6e+19 lbf) and TSFC (-0.0029).
"""

from __future__ import annotations

from typing import Any

# Each entry shape:
#   name        — exact path to pass to set_inputs (settable)
#   units       — units the value is interpreted in (None = unitless)
#   default     — current default the model already has set
#   description — what the variable controls, short
#   category    — grouping hint for the agent
CanonicalInput = dict[str, Any]

HBTF_DESIGN_INPUTS: list[CanonicalInput] = [
    # ── Flight condition ────────────────────────────────────────────
    {
        "name": "fc.alt",
        "units": "ft",
        "default": 35000.0,
        "description": "Cruise altitude (flight condition).",
        "category": "flight_condition",
    },
    {
        "name": "fc.MN",
        "units": None,
        "default": 0.8,
        "description": "Cruise Mach number (flight condition).",
        "category": "flight_condition",
    },
    # ── Thrust / temperature targets ────────────────────────────────
    {
        "name": "Fn_DES",
        "units": "lbf",
        "default": 5900.0,
        "description": (
            "Design thrust target. Used by the balance component to "
            "solve for inlet mass flow."
        ),
        "category": "performance_target",
    },
    {
        "name": "T4_MAX",
        "units": "degR",
        "default": 2857.0,
        "description": (
            "Combustor exit temperature (turbine inlet) target. Used "
            "by the balance component to solve for fuel-air ratio."
        ),
        "category": "performance_target",
    },
    # ── Bypass ratio (the one Run #17 missed) ───────────────────────
    {
        "name": "splitter.BPR",
        "units": None,
        "default": 5.105,
        "description": (
            "Bypass ratio (bypass mass flow / core mass flow). Set on "
            "the splitter component — there is NO top-level 'BPR' "
            "input on this cycle."
        ),
        "category": "bypass",
    },
    # ── Component pressure ratios ───────────────────────────────────
    {
        "name": "fan.PR",
        "units": None,
        "default": 1.685,
        "description": "Fan pressure ratio.",
        "category": "compressor",
    },
    {
        "name": "lpc.PR",
        "units": None,
        "default": 1.935,
        "description": "Low-pressure compressor pressure ratio.",
        "category": "compressor",
    },
    {
        "name": "hpc.PR",
        "units": None,
        "default": 9.369,
        "description": (
            "High-pressure compressor pressure ratio. Overall "
            "pressure ratio OPR ≈ fan.PR * lpc.PR * hpc.PR — set the "
            "three ratios individually, NOT a single 'OPR' input."
        ),
        "category": "compressor",
    },
    # ── Component polytropic efficiencies ───────────────────────────
    {
        "name": "fan.eff",
        "units": None,
        "default": 0.8948,
        "description": "Fan polytropic efficiency.",
        "category": "efficiency",
    },
    {
        "name": "lpc.eff",
        "units": None,
        "default": 0.9243,
        "description": "Low-pressure compressor polytropic efficiency.",
        "category": "efficiency",
    },
    {
        "name": "hpc.eff",
        "units": None,
        "default": 0.8707,
        "description": "High-pressure compressor polytropic efficiency.",
        "category": "efficiency",
    },
    {
        "name": "hpt.eff",
        "units": None,
        "default": 0.8888,
        "description": "High-pressure turbine polytropic efficiency.",
        "category": "efficiency",
    },
    {
        "name": "lpt.eff",
        "units": None,
        "default": 0.8996,
        "description": "Low-pressure turbine polytropic efficiency.",
        "category": "efficiency",
    },
]


TURBOJET_DESIGN_INPUTS: list[CanonicalInput] = [
    {
        "name": "fc.alt",
        "units": "ft",
        "default": 0.0,
        "description": "Flight altitude.",
        "category": "flight_condition",
    },
    {
        "name": "fc.MN",
        "units": None,
        "default": 0.000001,
        "description": "Flight Mach number.",
        "category": "flight_condition",
    },
    {
        "name": "balance.Fn_target",
        "units": "lbf",
        "default": 11800.0,
        "description": "Design thrust target.",
        "category": "performance_target",
    },
    {
        "name": "balance.T4_target",
        "units": "degR",
        "default": 2370.0,
        "description": "Turbine inlet temperature target.",
        "category": "performance_target",
    },
    {
        "name": "comp.PR",
        "units": None,
        "default": 13.5,
        "description": "Compressor pressure ratio.",
        "category": "compressor",
    },
    {
        "name": "comp.eff",
        "units": None,
        "default": 0.83,
        "description": "Compressor polytropic efficiency.",
        "category": "efficiency",
    },
    {
        "name": "turb.eff",
        "units": None,
        "default": 0.86,
        "description": "Turbine polytropic efficiency.",
        "category": "efficiency",
    },
]


# cycle_type (as stored in session meta) → canonical input list
DESIGN_INPUTS_BY_CYCLE: dict[str, list[CanonicalInput]] = {
    "turbofan": HBTF_DESIGN_INPUTS,
    "turbojet": TURBOJET_DESIGN_INPUTS,
}


def get_design_inputs_for_cycle(cycle_type: str) -> list[CanonicalInput]:
    """Return the curated canonical-input list for a built-in cycle type.

    Returns an empty list for unrecognized cycle types (e.g. ``"custom"``)
    — the agent should fall back to ``list_variables`` for those.

    Each call returns a deep copy so the live tool can safely mutate
    entries (adding ``current_value`` per session) without corrupting
    the module-level constants.
    """
    import copy

    return copy.deepcopy(DESIGN_INPUTS_BY_CYCLE.get(cycle_type, []))
