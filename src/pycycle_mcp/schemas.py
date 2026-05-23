"""Typed request and response payloads for FastMCP tool bindings.

The models defined here mirror the existing dictionary-based tool contracts
but provide explicit JSON Schema generation for FastMCP consumers.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class ToolError(BaseModel):
    """Structured error envelope shared by all tool responses."""

    type: str
    message: str
    details: Any | None = None

    model_config = ConfigDict(extra="ignore")


class BaseResponse(BaseModel):
    """Base response wrapper that carries optional error information."""

    error: ToolError | None = None

    model_config = ConfigDict(extra="ignore")


class NamedVariable(BaseModel):
    """Common variable metadata representation."""

    name: str
    units: str | None = None
    desc: str | None = None
    current_value: Any | None = None
    io: str | None = None
    promoted: bool | None = None
    shape: Any | None = None
    value: Any | None = None

    model_config = ConfigDict(extra="ignore")


class CreateCycleModelRequest(BaseModel):
    """Payload for creating a new pyCycle/OpenMDAO model."""

    cycle_type: str
    mode: str
    options: dict[str, Any] = Field(default_factory=dict)
    cycle_module_path: str | None = None

    model_config = ConfigDict(extra="forbid")


class CreateCycleModelResponse(BaseResponse):
    """Response returned after creating a cycle model."""

    session_id: str | None = None
    model_name: str | None = None
    top_promoted_inputs: list[NamedVariable] = Field(default_factory=list)
    top_promoted_outputs: list[NamedVariable] = Field(default_factory=list)


class CloseCycleModelResponse(BaseResponse):
    """Response from closing a cycle model session."""

    success: bool | None = None


class CycleSummaryResponse(BaseResponse):
    """Summary of a model's configuration and key variables."""

    model_name: str | None = None
    mode: str | None = None
    options: dict[str, Any] = Field(default_factory=dict)
    key_inputs: list[NamedVariable] = Field(default_factory=list)
    key_outputs: list[NamedVariable] = Field(default_factory=list)


class ListVariablesRequest(BaseModel):
    """Payload for listing variables."""

    session_id: str
    kind: Literal["inputs", "outputs", "both"] = "both"
    promoted_only: bool = True
    name_filter: str | None = None
    max_variables: int = 200

    model_config = ConfigDict(extra="forbid")


class ListVariablesResponse(BaseResponse):
    """Response containing variable metadata entries."""

    variables: list[NamedVariable] = Field(default_factory=list)


class GetDesignInputsRequest(BaseModel):
    """Payload for retrieving the curated design-input list for a session."""

    session_id: str

    model_config = ConfigDict(extra="forbid")


class CanonicalInputEntry(BaseModel):
    """Curated design-point input descriptor."""

    name: str
    units: str | None = None
    default: Any | None = None
    description: str | None = None
    category: str | None = None
    current_value: Any | None = None


class GetDesignInputsResponse(BaseResponse):
    """Response from get_design_inputs — the curated design-dial list."""

    cycle_type: str | None = None
    design_inputs: list[CanonicalInputEntry] = Field(default_factory=list)
    note: str | None = None


class SetInputsRequest(BaseModel):
    """Payload for setting one or more input variables."""

    session_id: str
    values: dict[str, Any]
    allow_missing: bool = False

    model_config = ConfigDict(extra="forbid")


class SetInputsResponse(BaseResponse):
    """Response from setting inputs."""

    updated: list[str] = Field(default_factory=list)
    skipped: list[dict[str, Any]] = Field(default_factory=list)


class GetOutputsRequest(BaseModel):
    """Payload for retrieving one or more outputs."""

    session_id: str
    names: list[str]
    allow_missing: bool = False

    model_config = ConfigDict(extra="forbid")


class GetOutputsResponse(BaseResponse):
    """Response carrying requested outputs."""

    values: dict[str, Any] = Field(default_factory=dict)
    missing: list[str] = Field(default_factory=list)


class RunCycleRequest(BaseModel):
    """Payload for executing a cycle run."""

    session_id: str
    outputs_of_interest: list[str] = Field(default_factory=list)
    use_driver: bool = False

    model_config = ConfigDict(extra="forbid")


class RunCycleResponse(BaseResponse):
    """Response from running a cycle model."""

    success: bool | None = None
    converged: Any | None = None
    iterations: int | None = None
    outputs: dict[str, Any] = Field(default_factory=dict)
    residual_norm: Any | None = None
    messages: list[str] = Field(default_factory=list)


class SweepVariable(BaseModel):
    """Single sweep variable specification."""

    name: str
    values: list[Any]

    model_config = ConfigDict(extra="forbid")


class SweepInputsRequest(BaseModel):
    """Payload describing a sweep run."""

    session_id: str
    sweep: list[SweepVariable]
    outputs_of_interest: list[str] = Field(default_factory=list)
    use_driver: bool = False
    skip_on_failure: bool = True

    model_config = ConfigDict(extra="forbid")


class SweepResult(BaseModel):
    """Result for a single sweep iteration."""

    inputs: dict[str, Any] = Field(default_factory=dict)
    success: bool
    outputs: dict[str, Any] = Field(default_factory=dict)
    error_message: str | None = None

    model_config = ConfigDict(extra="ignore")


class SweepInputsResponse(BaseResponse):
    """Response for sweep execution."""

    results: list[SweepResult] = Field(default_factory=list)


class ComputeTotalsRequest(BaseModel):
    """Payload for computing total derivatives."""

    session_id: str
    of: list[str]
    wrt: list[str]
    return_format: Literal["by_pair", "dense"] = "by_pair"

    model_config = ConfigDict(extra="forbid")


class ComputeTotalsResponse(BaseResponse):
    """Response carrying jacobian data."""

    jacobian: dict[str, Any] = Field(default_factory=dict)
    messages: list[str] = Field(default_factory=list)
