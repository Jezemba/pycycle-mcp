from __future__ import annotations

from typing import cast

from pycycle_mcp.session_manager import session_manager
from pycycle_mcp.tools import execution, variables
from pycycle_mcp.types import CycleProblem

from .conftest import DummyProblem


def setup_dummy_session() -> str:
    problem = DummyProblem()
    problem.model.inputs = [
        (
            "Mach",
            {"promoted_name": "Mach", "units": "", "desc": "Mach number", "val": 0.0},
        )
    ]
    problem.model.outputs = [("Fn", {"promoted_name": "Fn", "units": "lbf", "desc": "Thrust", "val": 0.0})]
    return session_manager.create_session(problem=cast(CycleProblem, problem), meta={"mode": "design", "options": {}})


def test_set_and_get_inputs() -> None:
    session_id = setup_dummy_session()

    set_response = variables.set_inputs({"session_id": session_id, "values": {"Mach": 0.75}})
    assert set_response["updated"] == ["Mach"]

    list_response = variables.list_variables({"session_id": session_id, "kind": "inputs"})
    variables_payload = cast(list[dict[str, object]], list_response["variables"])
    assert variables_payload[0]["name"] == "Mach"


def test_run_cycle_and_get_outputs() -> None:
    session_id = setup_dummy_session()

    # `Fn` must actually resolve for this to be a success case. Previously the
    # dummy raised KeyError for it and run_cycle returned success:True anyway,
    # so the assertion below passed while the output was null -- the test
    # encoded the bug it was meant to guard.
    variables.set_inputs({"session_id": session_id, "values": {"Fn": 6336.0}})

    run_response = execution.run_cycle({"session_id": session_id, "outputs_of_interest": ["Fn"]})
    assert run_response["success"] is True
    assert run_response["outputs"]["Fn"] == 6336.0
    assert run_response["missing_outputs"] == []


def test_run_cycle_reports_unresolved_outputs_as_failure() -> None:
    """Observed live: {"success": true, "outputs": {"SFC": null, "Fn": null}}.

    The requested names were wrong (they are `perf.TSFC` / `perf.Fn`) and
    OpenMDAO said so inside `messages`, but nothing downstream reads prose --
    `success` is the flag a consumer filters on, so nulls propagated as results.
    """
    session_id = setup_dummy_session()

    response = execution.run_cycle(
        {"session_id": session_id, "outputs_of_interest": ["SFC", "Fn"]}
    )
    assert response["success"] is False
    assert response["model_ran"] is True, "the cycle itself did run; keep that distinct"
    assert sorted(response["missing_outputs"]) == ["Fn", "SFC"]

    # populate output for retrieval
    problem, _ = session_manager.get(session_id)
    assert isinstance(problem, DummyProblem)
    problem.values["Fn"] = 123.4

    output_response = variables.get_outputs({"session_id": session_id, "names": ["Fn"]})
    values = cast(dict[str, object], output_response["values"])
    assert values["Fn"] == 123.4


def test_get_outputs_missing_allowed() -> None:
    session_id = setup_dummy_session()
    response = variables.get_outputs({"session_id": session_id, "names": ["Missing"], "allow_missing": True})
    assert response["missing"] == ["Missing"]
