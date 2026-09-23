"""B114: an unknown variable name must not be answered with a wrong guess.

geoauth_all8_7960 networked_iterative_feedback L1: the agent called get_design_inputs four times,
was shown `splitter.BPR`, asked for `fan.BPR`, and OpenMDAO answered "Perhaps you meant one of the
following variables: ['balance.hpt_PR', 'balance.lpt_PR', 'perf.OPR']" -- all three wrong. The
cycle was never run and the link failed. graph_routed L1 lost 11 run_cycle calls the same way on
output names.
"""
from pycycle_mcp.tools.variables import _unknown_variable_message, set_inputs


class _Problem:
    def set_val(self, name, value):
        raise KeyError(f"': Variable \\'{name}\\' not found. Perhaps you meant one of the "
                       f"following variables: [\\'balance.hpt_PR\\', \\'perf.OPR\\'].'")


def test_the_message_names_the_bad_variable_and_the_listing_tool():
    msg = _unknown_variable_message("fan.BPR", KeyError("Variable 'fan.BPR' not found."), "s1")
    assert "'fan.BPR' is not a variable" in msg
    assert "get_design_inputs(session_id='s1')" in msg


def test_openmdao_guesses_are_not_passed_through():
    msg = _unknown_variable_message(
        "fan.BPR",
        KeyError("Variable 'fan.BPR' not found. Perhaps you meant one of the following "
                 "variables: ['balance.hpt_PR', 'balance.lpt_PR', 'perf.OPR']."),
        "s1")
    for wrong in ("balance.hpt_PR", "balance.lpt_PR", "perf.OPR", "Perhaps you meant"):
        assert wrong not in msg


def test_an_unrelated_failure_keeps_its_own_message():
    assert _unknown_variable_message("fan.PR", ValueError("shape mismatch (3,) vs (1,)"), "s1") is None


def test_set_inputs_returns_the_clean_error(monkeypatch):
    import pycycle_mcp.tools.variables as v

    monkeypatch.setattr(v.session_manager, "get", lambda sid: (_Problem(), None))
    out = set_inputs({"session_id": "s1", "values": {"fan.BPR": 5.1}})
    assert out["error"]["type"] == "UnknownVariable"
    assert "get_design_inputs" in out["error"]["message"]
    assert "Perhaps you meant" not in out["error"]["message"]


def test_allow_missing_still_skips_with_the_clean_reason(monkeypatch):
    import pycycle_mcp.tools.variables as v

    monkeypatch.setattr(v.session_manager, "get", lambda sid: (_Problem(), None))
    out = set_inputs({"session_id": "s1", "values": {"fan.BPR": 5.1}, "allow_missing": True})
    assert out["skipped"][0]["name"] == "fan.BPR"
    assert "get_design_inputs" in out["skipped"][0]["reason"]
