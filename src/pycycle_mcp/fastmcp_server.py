"""Factory for the FastMCP server exposing pyCycle tools."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any, Literal, TypedDict, TypeVar, Unpack, cast

from fastmcp.server import FastMCP
from fastmcp.tools.tool import FunctionTool
from mcp.types import ToolAnnotations
from pydantic import BaseModel

from . import tools
from .schemas import (
    CloseCycleModelResponse,
    ComputeTotalsRequest,
    ComputeTotalsResponse,
    CreateCycleModelRequest,
    CreateCycleModelResponse,
    CycleSummaryResponse,
    GetDesignInputsRequest,
    GetDesignInputsResponse,
    GetOutputsRequest,
    GetOutputsResponse,
    ListVariablesRequest,
    ListVariablesResponse,
    RunCycleRequest,
    RunCycleResponse,
    SetInputsRequest,
    SetInputsResponse,
    SweepInputsRequest,
    SweepInputsResponse,
    SweepVariable,
)
from .tools import PingResponse

__all__ = ["build_server", "tools"]

LOGGER = logging.getLogger(__name__)

ResponseModel = TypeVar("ResponseModel", bound=BaseModel)
ToolFunc = Callable[..., dict[str, Any]]


class ToolKwargs(TypedDict, total=False):
    """Subset of FastMCP tool configuration parameters used by this module."""

    name: str
    title: str | None
    description: str | None
    tags: set[str]
    output_schema: dict[str, Any] | None
    annotations: ToolAnnotations | dict[str, Any] | None


def _tool(server: FastMCP, **kwargs: Unpack[ToolKwargs]) -> Callable[[ToolFunc], ToolFunc]:
    """Typed wrapper around :meth:`FastMCP.tool` to satisfy type checkers."""

    return cast(Callable[[ToolFunc], ToolFunc], server.tool(**kwargs))


def _validated_response(response: dict[str, Any], response_model: type[ResponseModel]) -> dict[str, Any]:
    """Validate and normalize a tool response using a Pydantic model."""

    validated = response_model.model_validate(response)
    return validated.model_dump(exclude_none=True)


def _register_tools(server: FastMCP) -> None:
    """Attach the pyCycle tool implementations to a FastMCP instance."""

    server.add_tool(
        FunctionTool.from_function(
            tools.ping,
            name="ping",
            description="Simple healthcheck for the pyCycle MCP server.",
            annotations=ToolAnnotations(title="Ping", readOnlyHint=True),
            output_schema=PingResponse.model_json_schema(),
        )
    )

    @_tool(
        server,
        name="create_cycle_model",
        description="Instantiate a pyCycle/OpenMDAO Problem for a specified engine cycle.",
        tags={"pycycle", "model"},
        output_schema=CreateCycleModelResponse.model_json_schema(),
        annotations=ToolAnnotations(title="Create cycle model"),
    )
    def create_cycle_model_tool(
        cycle_type: str,
        mode: str,
        options: dict[str, Any] | None = None,
        cycle_module_path: str | None = None,
    ) -> dict[str, Any]:
        request = CreateCycleModelRequest(
            cycle_type=cycle_type,
            mode=mode,
            options=options or {},
            cycle_module_path=cycle_module_path,
        )
        response = tools.create_model.create_cycle_model(request.model_dump(exclude_none=True))
        return _validated_response(response, CreateCycleModelResponse)

    @_tool(
        server,
        name="close_cycle_model",
        description="Close a pyCycle session and free resources.",
        tags={"pycycle", "session"},
        output_schema=CloseCycleModelResponse.model_json_schema(),
        annotations=ToolAnnotations(title="Close cycle model", destructiveHint=True),
    )
    def close_cycle_model_tool(session_id: str) -> dict[str, Any]:
        response = tools.create_model.close_cycle_model({"session_id": session_id})
        return _validated_response(response, CloseCycleModelResponse)

    @_tool(
        server,
        name="get_cycle_summary",
        description="Return a succinct summary of the current cycle model.",
        tags={"pycycle", "summary"},
        output_schema=CycleSummaryResponse.model_json_schema(),
        annotations=ToolAnnotations(title="Get cycle summary", readOnlyHint=True),
    )
    def get_cycle_summary_tool(session_id: str) -> dict[str, Any]:
        response = tools.create_model.get_cycle_summary({"session_id": session_id})
        return _validated_response(response, CycleSummaryResponse)

    @_tool(
        server,
        name="list_variables",
        description="List variables in the cycle model.",
        tags={"pycycle", "variables"},
        output_schema=ListVariablesResponse.model_json_schema(),
        annotations=ToolAnnotations(title="List variables", readOnlyHint=True),
    )
    def list_variables_tool(
        session_id: str,
        kind: Literal["inputs", "outputs", "both"] = "both",
        promoted_only: bool = True,
        name_filter: str | None = None,
        max_variables: int = 200,
    ) -> dict[str, Any]:
        request = ListVariablesRequest(
            session_id=session_id,
            kind=kind,
            promoted_only=promoted_only,
            name_filter=name_filter,
            max_variables=max_variables,
        )
        response = tools.variables.list_variables(request.model_dump(exclude_none=True))
        return _validated_response(response, ListVariablesResponse)

    @_tool(
        server,
        name="get_design_inputs",
        description=(
            "Return the curated list of canonical design-point input "
            "variables for the session's cycle type. Each entry "
            "includes the exact path to pass to set_inputs, units, "
            "default value, current value, and a short description. "
            "Use this BEFORE set_inputs to discover the design dials "
            "for this cycle — the full list_variables output contains "
            "900+ internal entries (residuals, flow stations, balance "
            "states) that are NOT safe to set directly."
        ),
        tags={"pycycle", "variables"},
        output_schema=GetDesignInputsResponse.model_json_schema(),
        annotations=ToolAnnotations(title="Get design inputs", readOnlyHint=True),
    )
    def get_design_inputs_tool(session_id: str) -> dict[str, Any]:
        request = GetDesignInputsRequest(session_id=session_id)
        response = tools.variables.get_design_inputs(request.model_dump())
        return _validated_response(response, GetDesignInputsResponse)

    @_tool(
        server,
        name="set_inputs",
        description="Set one or more input variables in the cycle model.",
        tags={"pycycle", "variables"},
        output_schema=SetInputsResponse.model_json_schema(),
        annotations=ToolAnnotations(title="Set inputs"),
    )
    def set_inputs_tool(
        session_id: str,
        values: dict[str, Any],
        allow_missing: bool = False,
    ) -> dict[str, Any]:
        request = SetInputsRequest(session_id=session_id, values=values, allow_missing=allow_missing)
        response = tools.variables.set_inputs(request.model_dump())
        return _validated_response(response, SetInputsResponse)

    @_tool(
        server,
        name="get_outputs",
        description="Fetch values for one or more outputs after a run.",
        tags={"pycycle", "variables"},
        output_schema=GetOutputsResponse.model_json_schema(),
        annotations=ToolAnnotations(title="Get outputs", readOnlyHint=True),
    )
    def get_outputs_tool(
        session_id: str,
        names: list[str],
        allow_missing: bool = False,
    ) -> dict[str, Any]:
        request = GetOutputsRequest(session_id=session_id, names=names, allow_missing=allow_missing)
        response = tools.variables.get_outputs(request.model_dump())
        return _validated_response(response, GetOutputsResponse)

    @_tool(
        server,
        name="run_cycle",
        description="Run the cycle model and return selected outputs.",
        tags={"pycycle", "execution"},
        output_schema=RunCycleResponse.model_json_schema(),
        annotations=ToolAnnotations(title="Run cycle"),
    )
    def run_cycle_tool(
        session_id: str,
        outputs_of_interest: list[str] | None = None,
        use_driver: bool = False,
    ) -> dict[str, Any]:
        request = RunCycleRequest(
            session_id=session_id,
            outputs_of_interest=outputs_of_interest or [],
            use_driver=use_driver,
        )
        response = tools.execution.run_cycle(request.model_dump())
        return _validated_response(response, RunCycleResponse)

    @_tool(
        server,
        name="sweep_inputs",
        description="Perform a parametric sweep over input variables.",
        tags={"pycycle", "sweep"},
        output_schema=SweepInputsResponse.model_json_schema(),
        annotations=ToolAnnotations(title="Sweep inputs"),
    )
    def sweep_inputs_tool(
        session_id: str,
        sweep: list[SweepVariable],
        outputs_of_interest: list[str] | None = None,
        use_driver: bool = False,
        skip_on_failure: bool = True,
    ) -> dict[str, Any]:
        request = SweepInputsRequest(
            session_id=session_id,
            sweep=sweep,
            outputs_of_interest=outputs_of_interest or [],
            use_driver=use_driver,
            skip_on_failure=skip_on_failure,
        )
        response = tools.sweep.sweep_inputs(request.model_dump())
        return _validated_response(response, SweepInputsResponse)

    @_tool(
        server,
        name="compute_totals",
        description="Compute total derivatives using OpenMDAO.",
        tags={"pycycle", "derivatives"},
        output_schema=ComputeTotalsResponse.model_json_schema(),
        annotations=ToolAnnotations(title="Compute totals", readOnlyHint=True),
    )
    def compute_totals_tool(
        session_id: str,
        of: list[str],
        wrt: list[str],
        return_format: Literal["by_pair", "dense"] = "by_pair",
    ) -> dict[str, Any]:
        request = ComputeTotalsRequest(
            session_id=session_id,
            of=of,
            wrt=wrt,
            return_format=return_format,
        )
        response = tools.derivatives.compute_totals(request.model_dump())
        return _validated_response(response, ComputeTotalsResponse)


def build_server() -> FastMCP:
    """Construct a FastMCP server with all pyCycle tools registered."""

    server = FastMCP(
        name="pycycle-mcp",
        instructions=("Expose pyCycle/OpenMDAO utilities for creating and running engine cycle models."),
        strict_input_validation=True,
    )
    _register_tools(server)
    LOGGER.debug("FastMCP server configured")
    return server
