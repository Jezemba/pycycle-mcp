"""run_cycle reports whether the design point is solved, not only whether names resolved (Avion B3 audit).

Newton runs with err_on_non_converge=False, so a failed solve returns numbers and used to be reported as
success (measured: perf.Fn = 1.46e24 lbf). The validity rule: perf.Fn hits its target and
balance.lpt_PR < 8.
"""

import math

import pytest

from pycycle_mcp.tools.execution import convergence_check


class _Problem:
    def __init__(self, **vals):
        self.vals = vals

    def get_val(self, name):
        if name not in self.vals:
            raise KeyError(name)
        return self.vals[name]


def test_a_solved_cycle_is_converged():
    c = convergence_check(_Problem(**{"perf.Fn": 5900.0, "Fn_DES": 5900.0, "balance.lpt_PR": 3.03}))
    assert c["converged"] is True and c["reasons"] == []


def test_missing_the_thrust_target_is_not_converged():
    c = convergence_check(_Problem(**{"perf.Fn": 5952.0, "Fn_DES": 5900.0, "balance.lpt_PR": 3.0}))
    assert c["converged"] is False and "misses its target" in c["reasons"][0]


def test_an_unphysical_lpt_ratio_is_not_converged():
    c = convergence_check(_Problem(**{"perf.Fn": 5900.0, "Fn_DES": 5900.0, "balance.lpt_PR": 8.0}))
    assert c["converged"] is False and "lpt_PR" in c["reasons"][0]


def test_a_runaway_solution_is_not_converged():
    c = convergence_check(_Problem(**{"perf.Fn": 1.46e24, "Fn_DES": 5900.0, "balance.lpt_PR": 3.0}))
    assert c["converged"] is False


def test_a_non_finite_value_is_not_converged():
    c = convergence_check(_Problem(**{"perf.Fn": math.nan, "Fn_DES": 5900.0}))
    assert c["converged"] is False and any("not finite" in r for r in c["reasons"])


def test_a_turbojet_uses_its_own_thrust_target():
    c = convergence_check(_Problem(**{"perf.Fn": 11800.0, "balance.Fn_target": 11800.0}))
    assert c["converged"] is True and c["thrust_target_name"] == "balance.Fn_target"


def test_live_default_converges_and_bpr_12_does_not():
    pytest.importorskip("pycycle")
    from pycycle_mcp.tools import create_model, execution, variables

    sid = create_model.create_cycle_model({"cycle_type": "turbofan", "mode": "design"})["session_id"]
    good = execution.run_cycle({"session_id": sid, "outputs_of_interest": ["perf.TSFC"]})
    assert good["success"] is True and good["converged"] is True

    sid = create_model.create_cycle_model({"cycle_type": "turbofan", "mode": "design"})["session_id"]
    variables.set_inputs({"session_id": sid, "values": {"splitter.BPR": 12.0}})
    bad = execution.run_cycle({"session_id": sid, "outputs_of_interest": ["perf.TSFC"]})
    assert bad["success"] is True and bad["converged"] is False      # names resolved, cycle not solved
    assert any("did NOT converge" in m for m in bad["messages"])
