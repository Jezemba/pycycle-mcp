from __future__ import annotations

from collections.abc import Callable
from typing import cast

import numpy as np
import pytest

from pycycle_mcp.session_manager import session_manager
from pycycle_mcp.tools import create_model, derivatives, execution, sweep, variables
from pycycle_mcp.types import CycleProblem

from .conftest import DummyModel, DummyProblem


class _SetValFailsProblem(DummyProblem):
    def set_val(self, name: str | None = None, value: object | None = None, **kwargs: object) -> None:
        if name == "bad":
            raise RuntimeError("cannot set variable")
        super().set_val(name=name, value=value, **kwargs)


class _RunFailsProblem(DummyProblem):
    def set_solver_print(self, level: int = 0) -> None:
        del level
        raise RuntimeError("solver setup failed")


class _SweepSetValFailsProblem(DummyProblem):
    def set_val(self, name: str | None = None, value: object | None = None, **kwargs: object) -> None:
        del name, value, kwargs
        raise RuntimeError("sweep set failed")


def test_create_cycle_model_custom_requires_module_path() -> None:
    response = create_model.create_cycle_model({"cycle_type": "custom", "mode": "design"})

    assert response["error"]["type"] == "ValueError"


def test_create_cycle_model_with_builtin_type_uses_resolver(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_builder() -> DummyModel:
        return DummyModel(inputs=[("Mach", {"units": ""})], outputs=[("Fn", {"units": "lbf"})])

    def fake_build_problem(
        builder: Callable[[], DummyModel],
        mode: str,
        options: dict[str, object],
    ) -> tuple[DummyProblem, str]:
        del mode, options
        problem = DummyProblem()
        problem.model = builder()
        return problem, "builtin-dummy"

    monkeypatch.setattr(create_model, "_resolve_builtin_cycle", lambda cycle_type: (cycle_type, fake_builder))
    monkeypatch.setattr(create_model, "_build_problem", fake_build_problem)

    response = create_model.create_cycle_model({"cycle_type": "turbofan", "mode": "design", "options": {}})

    assert response["model_name"] == "builtin-dummy"
    assert "session_id" in response


def test_create_cycle_model_returns_structured_error_on_build_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(create_model, "load_callable", lambda _: lambda: DummyModel())
    monkeypatch.setattr(
        create_model,
        "_build_problem",
        lambda builder, mode, options: (_ for _ in ()).throw(RuntimeError("build failed")),
    )

    response = create_model.create_cycle_model(
        {
            "cycle_type": "custom",
            "cycle_module_path": "unused.path",
            "mode": "design",
            "options": {},
        }
    )

    assert response["error"]["type"] == "RuntimeError"


def test_close_cycle_model_validation_and_error_paths(monkeypatch: pytest.MonkeyPatch) -> None:
    assert create_model.close_cycle_model({})["error"]["type"] == "ValidationError"

    monkeypatch.setattr(create_model.session_manager, "close", lambda _: (_ for _ in ()).throw(RuntimeError("x")))
    response = create_model.close_cycle_model({"session_id": "abc"})
    assert response["error"]["type"] == "RuntimeError"


def test_get_cycle_summary_validation_and_missing_session_paths() -> None:
    assert create_model.get_cycle_summary({})["error"]["type"] == "ValidationError"

    response = create_model.get_cycle_summary({"session_id": "missing-id"})
    assert response["error"]["type"] == "KeyError"


def test_list_variables_validation_and_missing_session_paths() -> None:
    assert variables.list_variables({})["error"]["type"] == "ValidationError"

    response = variables.list_variables({"session_id": "missing-id"})
    assert response["error"]["type"] == "KeyError"


def test_list_variables_filters_and_limit_paths() -> None:
    problem = DummyProblem()
    problem.model.inputs = [
        ("keep_input", {"promoted_name": "keep_input", "units": "ft"}),
        ("skip_promoted_input", {"promoted_name": "different_name"}),
    ]
    problem.model.outputs = [
        ("keep_output", {"promoted_name": "keep_output", "units": "lbf"}),
        ("skip_promoted_output", {"promoted_name": "different_output"}),
        ("other_output", {"promoted_name": "other_output"}),
    ]
    session_id = session_manager.create_session(problem=cast(CycleProblem, problem), meta={})

    response = variables.list_variables(
        {
            "session_id": session_id,
            "kind": "both",
            "promoted_only": True,
            "name_filter": "keep",
            "max_variables": "1",
        }
    )
    assert len(response["variables"]) == 1
    assert response["variables"][0]["name"] in {"keep_input", "keep_output"}


def test_set_inputs_validation_and_error_paths() -> None:
    assert variables.set_inputs({})["error"]["type"] == "ValidationError"
    assert variables.set_inputs({"session_id": "x", "values": {}})["error"]["type"] == "ValidationError"
    assert variables.set_inputs({"session_id": "missing-id", "values": {"x": 1.0}})["error"]["type"] == "KeyError"

    problem = _SetValFailsProblem()
    session_id = session_manager.create_session(problem=cast(CycleProblem, problem), meta={})

    fail_response = variables.set_inputs({"session_id": session_id, "values": {"bad": 1.0}})
    assert fail_response["error"]["type"] == "RuntimeError"

    allow_response = variables.set_inputs(
        {"session_id": session_id, "values": {"bad": 1.0, "ok": 2.0}, "allow_missing": True}
    )
    assert allow_response["updated"] == ["ok"]
    assert allow_response["skipped"][0]["name"] == "bad"


def test_get_outputs_validation_and_lookup_paths() -> None:
    assert variables.get_outputs({})["error"]["type"] == "ValidationError"
    assert variables.get_outputs({"session_id": "x", "names": []})["error"]["type"] == "ValidationError"
    assert variables.get_outputs({"session_id": "missing-id", "names": ["Fn"]})["error"]["type"] == "KeyError"

    problem = DummyProblem()
    problem.values["Fn"] = 12.3
    session_id = session_manager.create_session(problem=cast(CycleProblem, problem), meta={})

    response = variables.get_outputs({"session_id": session_id, "names": ["missing"]})
    assert response["error"]["type"] == "LookupError"


def test_run_cycle_error_paths_and_driver_branch() -> None:
    assert execution.run_cycle({})["error"]["type"] == "ValidationError"
    assert execution.run_cycle({"session_id": "missing-id"})["error"]["type"] == "KeyError"

    run_fail_session = session_manager.create_session(problem=cast(CycleProblem, _RunFailsProblem()), meta={})
    run_fail_response = execution.run_cycle({"session_id": run_fail_session})
    assert run_fail_response["error"]["type"] == "RuntimeError"

    driver_problem = DummyProblem()
    driver_problem.values["Fn"] = 44.0
    driver_session = session_manager.create_session(problem=cast(CycleProblem, driver_problem), meta={})
    driver_response = execution.run_cycle(
        {
            "session_id": driver_session,
            "outputs_of_interest": ["Fn", "missing_output"],
            "use_driver": True,
        }
    )
    # One name resolves, one does not -- so this is NOT a success. It used to
    # report success:True with "missing_output": null sitting in the payload.
    assert driver_response["success"] is False
    assert driver_response["model_ran"] is True
    assert driver_response["outputs"]["Fn"] == 44.0
    assert driver_response["missing_outputs"] == ["missing_output"]
    assert driver_response["messages"][0] == "Ran driver"
    assert driver_response["outputs"]["missing_output"] is None


def test_compute_totals_validation_dense_and_missing_session_paths() -> None:
    assert derivatives.compute_totals({})["error"]["type"] == "ValidationError"
    assert derivatives.compute_totals({"session_id": "x", "of": [], "wrt": []})["error"]["type"] == "ValidationError"

    problem = DummyProblem()
    session_id = session_manager.create_session(problem=cast(CycleProblem, problem), meta={})
    dense_response = derivatives.compute_totals(
        {"session_id": session_id, "of": ["Fn"], "wrt": ["Mach"], "return_format": "dense"}
    )
    assert dense_response["jacobian"] == {"of": ["Fn"], "wrt": ["Mach"], "data": [[[[1.0]]]]}

    missing_response = derivatives.compute_totals({"session_id": "missing", "of": ["Fn"], "wrt": ["Mach"]})
    assert missing_response["error"]["type"] == "KeyError"


def test_sweep_inputs_validation_and_failure_paths() -> None:
    assert sweep.sweep_inputs({})["error"]["type"] == "ValidationError"
    assert sweep.sweep_inputs({"session_id": "x", "sweep": []})["error"]["type"] == "ValidationError"
    assert (
        sweep.sweep_inputs({"session_id": "missing-id", "sweep": [{"name": "x", "values": [1]}]})["error"]["type"]
        == "KeyError"
    )

    failing_problem = _SweepSetValFailsProblem()
    session_id = session_manager.create_session(problem=cast(CycleProblem, failing_problem), meta={})

    skip_response = sweep.sweep_inputs(
        {
            "session_id": session_id,
            "sweep": [{"name": "Mach", "values": [0.7]}],
            "skip_on_failure": True,
        }
    )
    assert skip_response["results"][0]["success"] is False

    fail_fast_response = sweep.sweep_inputs(
        {
            "session_id": session_id,
            "sweep": [{"name": "Mach", "values": [0.7]}],
            "skip_on_failure": False,
        }
    )
    assert fail_fast_response["error"]["type"] == "RuntimeError"


@pytest.mark.parametrize(
    "value,expected",
    [
        (42, 42),
        (3.5, 3.5),
        (np.array([4.2]), 4.2),
        (np.array([1.0, 2.0]), [1.0, 2.0]),
        (np.int64(7), 7),
        (np.float64(8.5), 8.5),
    ],
)
def test_to_serializable_passthrough_scalars(value: object, expected: object) -> None:
    assert execution._to_serializable(value) == expected
