"""Tests for the canonical-input discovery layer (Phase I).

Pipeline Run #17 (2026-05-23) had the propulsion agent guess
``fan.map.design.BPR`` and ``fan.BPR`` for the bypass ratio — neither
exists. The Newton solver ran with BPR unset, produced ``perf.Fn =
-2.6e+19 lbf``, and propulsion cascaded ``UPSTREAM_ERROR``. The fix
is a discovery tool, ``get_design_inputs``, that hands the agent the
exact short list of settable design dials for the session's cycle
type — no more guessing against 936 promoted variables.
"""

from __future__ import annotations

from typing import cast

import pytest

from pycycle_mcp.canonical_inputs import (
    HBTF_DESIGN_INPUTS,
    TURBOJET_DESIGN_INPUTS,
    get_design_inputs_for_cycle,
)
from pycycle_mcp.session_manager import session_manager
from pycycle_mcp.tools import variables
from pycycle_mcp.types import CycleProblem

from .conftest import DummyProblem


# ---------------------------------------------------------------------------
# Pure-data sanity checks (no model required)
# ---------------------------------------------------------------------------


class TestCanonicalInputsData:
    def test_hbtf_includes_bypass_ratio(self) -> None:
        """splitter.BPR must be in the HBTF design-input list — this is
        the variable Run #17's agent kept missing (guessed ``fan.BPR``
        and ``fan.map.design.BPR`` instead)."""
        names = {entry["name"] for entry in HBTF_DESIGN_INPUTS}
        assert "splitter.BPR" in names, (
            "HBTF canonical list is missing splitter.BPR — the very "
            "variable Phase I exists to expose."
        )

    def test_hbtf_includes_flight_condition_and_thrust_target(self) -> None:
        names = {entry["name"] for entry in HBTF_DESIGN_INPUTS}
        for required in ("fc.alt", "fc.MN", "Fn_DES", "T4_MAX"):
            assert required in names, f"HBTF list missing {required}"

    def test_hbtf_includes_three_compressor_pressure_ratios(self) -> None:
        """OPR is the product fan.PR * lpc.PR * hpc.PR — there is no
        single ``OPR`` input. Make sure all three are exposed so the
        agent doesn't try to set a top-level OPR that doesn't exist."""
        names = {entry["name"] for entry in HBTF_DESIGN_INPUTS}
        for required in ("fan.PR", "lpc.PR", "hpc.PR"):
            assert required in names

    def test_every_hbtf_entry_has_required_fields(self) -> None:
        required_keys = {"name", "units", "default", "description", "category"}
        for entry in HBTF_DESIGN_INPUTS:
            missing = required_keys - set(entry.keys())
            assert not missing, f"{entry.get('name')} missing fields: {missing}"

    def test_lookup_by_cycle_type(self) -> None:
        assert get_design_inputs_for_cycle("turbofan") == HBTF_DESIGN_INPUTS
        assert get_design_inputs_for_cycle("turbojet") == TURBOJET_DESIGN_INPUTS
        # Unknown cycle type returns an empty list so the agent can
        # fall back to list_variables (no exception thrown).
        assert get_design_inputs_for_cycle("custom") == []
        assert get_design_inputs_for_cycle("unknown") == []

    def test_lookup_returns_independent_copy(self) -> None:
        """Successive calls must not share mutable state — the live
        tool mutates entries to add ``current_value`` per session."""
        a = get_design_inputs_for_cycle("turbofan")
        a[0]["sentinel"] = "modified"
        b = get_design_inputs_for_cycle("turbofan")
        assert "sentinel" not in b[0]


# ---------------------------------------------------------------------------
# Tool dispatch (uses DummyProblem fixture so no real pycycle needed)
# ---------------------------------------------------------------------------


def _make_session(cycle_type: str) -> str:
    problem = DummyProblem()
    return session_manager.create_session(
        problem=cast(CycleProblem, problem),
        meta={"mode": "design", "options": {}, "cycle_type": cycle_type},
    )


class TestGetDesignInputsTool:
    def test_returns_curated_list_for_turbofan_session(self) -> None:
        session_id = _make_session("turbofan")
        resp = variables.get_design_inputs({"session_id": session_id})

        assert "error" not in resp, f"Unexpected error: {resp.get('error')}"
        assert resp["cycle_type"] == "turbofan"
        inputs = resp["design_inputs"]
        names = {entry["name"] for entry in inputs}
        assert "splitter.BPR" in names
        assert "fc.alt" in names
        assert "T4_MAX" in names
        # Each entry must have current_value populated (from the live
        # model, with the default as a fallback).
        for entry in inputs:
            assert "current_value" in entry

    def test_returns_empty_for_unknown_cycle_type(self) -> None:
        session_id = _make_session("custom")
        resp = variables.get_design_inputs({"session_id": session_id})
        assert resp["cycle_type"] == "custom"
        assert resp["design_inputs"] == []
        assert "note" in resp

    def test_returns_not_found_for_missing_session(self) -> None:
        resp = variables.get_design_inputs({"session_id": "does-not-exist"})
        # error_on_missing_session uses exc.__class__.__name__ — for a
        # missing session key that's "KeyError".
        assert resp["error"]["type"] == "KeyError"
        assert "not found" in resp["error"]["message"].lower()

    def test_returns_validation_error_when_session_id_omitted(self) -> None:
        resp = variables.get_design_inputs({})
        assert resp["error"]["type"] == "ValidationError"


# ---------------------------------------------------------------------------
# Regression: every canonical HBTF name must be settable on a real HBTF
# Skipped if pycycle isn't importable so the unit suite stays portable.
# ---------------------------------------------------------------------------


def test_every_canonical_hbtf_path_is_settable_on_real_model() -> None:
    """End-to-end: build a real HBTF, then for every name in the
    canonical list, prove ``problem.set_val(name, default)`` succeeds.
    Catches drift between this file and the real model's promoted
    paths — if pycycle renames ``splitter.BPR`` to something else
    we want to know here, not on the next live pipeline run.
    """
    try:
        from pycycle_mcp.tools.create_model import (
            _build_problem,
            _resolve_builtin_cycle,
        )
    except ImportError:
        pytest.skip("pycycle not installed")

    try:
        _, builder = _resolve_builtin_cycle("turbofan")
    except ImportError:
        pytest.skip("pycycle not importable in this environment")

    problem, _ = _build_problem(builder=builder, mode="design", options={})

    failures: list[tuple[str, str]] = []
    for entry in HBTF_DESIGN_INPUTS:
        try:
            problem.set_val(entry["name"], entry["default"])
        except Exception as exc:
            failures.append((entry["name"], str(exc)))

    assert not failures, (
        "Some canonical-input paths no longer exist on the real HBTF "
        f"model — drift detected: {failures}"
    )
