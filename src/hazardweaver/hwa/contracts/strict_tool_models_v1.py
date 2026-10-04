"""Single-source Pydantic models for Agent-Strict v2 commit → run_capability → submit."""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Set, Tuple

from pydantic import BaseModel, ConfigDict, Field

# Ultimate Method §3.1 — authoritative tool parameter models.
STRICT_TOOL_MODELS: Dict[str, type[BaseModel]] = {}


class _StrictToolBase(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ControllerCommitRouteParams(_StrictToolBase):
    route_id: Optional[str] = Field(
        default=None,
        description="Committed route id (e.g. route:cap:pfdf_volume_gorr_v2).",
    )
    record_id: Optional[str] = Field(
        default=None,
        description="PFDF record id — with scenario_id/split enables SCC (commit+execute in one step).",
    )
    scenario_id: Optional[str] = Field(default=None, description="Scenario id for atlas/parametric caps.")
    split: Optional[str] = Field(default=None, description="Split for FL-2 caps.")
    handles: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Explicit handles for SCC chained execute after commit.",
    )
    execute: bool = Field(
        default=True,
        description="Run execution immediately after commit",
    )
    fixture: str = Field(
        default="wf_hard",
        description="HCG fixture name for initial handles",
    )


class RunCapabilityStrictParams(_StrictToolBase):
    capability_id: str
    controller_token: Optional[str] = Field(
        default=None,
        description="From controller_commit_route.execution_token (same value).",
    )
    route_id: Optional[str] = Field(
        default=None,
        description="Must match committed route_id (e.g. route:cap:pfdf_volume_gorr_v2).",
    )
    record_id: Optional[str] = None
    scenario_id: Optional[str] = None
    sample_id: Optional[str] = Field(default=None, description="Alias of record_id")
    anchor_id: Optional[str] = None
    oracle: bool = False
    record: Optional[Dict[str, Any]] = None
    handles: Optional[Dict[str, Any]] = None
    artifact_refs: Optional[List[str]] = None
    burn_summary: Optional[Dict[str, Any]] = None


class SubmitSolutionParams(_StrictToolBase):
    route_id: str
    execution_id: str
    final_artifact_id: str
    rationale: Optional[str] = None
    model_id_used: Optional[str] = None


STRICT_TOOL_MODELS.update(
    {
        "controller_commit_route": ControllerCommitRouteParams,
        "run_capability": RunCapabilityStrictParams,
        "submit_solution": SubmitSolutionParams,
    }
)


def pydantic_parameters_schema(model: type[BaseModel]) -> Dict[str, Any]:
    """OpenAI function.parameters JSON schema from a Pydantic model."""
    schema = model.model_json_schema()
    props = dict(schema.get("properties") or {})
    required = list(schema.get("required") or [])
    out: Dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "properties": props,
    }
    if required:
        out["required"] = required
    return out


def strict_tool_parameters(tool_name: str) -> Dict[str, Any]:
    model = STRICT_TOOL_MODELS.get(tool_name)
    if model is None:
        raise KeyError(tool_name)
    return strict_tool_parameters_for_llm(tool_name)


def strict_tool_parameters_for_llm(tool_name: str) -> Dict[str, Any]:
    """Minimal OpenAI parameters — omit nullable fields that invite JSON null literals."""
    if tool_name == "run_capability":
        return {
            "type": "object",
            "additionalProperties": False,
            "required": ["capability_id"],
            "properties": {
                "capability_id": {
                    "type": "string",
                    "description": "HWA capability id (e.g. pfdf_volume_gorr_v2).",
                },
                "record_id": {
                    "type": "string",
                    "description": "PFDF record id when required by task handles.",
                },
                "scenario_id": {
                    "type": "string",
                    "description": "Scenario id for atlas/parametric tasks.",
                },
                "handles": {
                    "type": "object",
                    "description": (
                        "Optional explicit handles. After controller_commit_route, "
                        "pass controller_token (= execution_token) and route_id "
                        "from the commit tool result — never JSON null."
                    ),
                },
            },
        }
    if tool_name == "controller_commit_route":
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "route_id": {
                    "type": "string",
                    "description": (
                        "Admissible route id (e.g. route:cap:pfdf_volume_gorr_v2). "
                        "Server validates admissibility; separate enumerate/propose optional."
                    ),
                },
                "record_id": {
                    "type": "string",
                    "description": "PFDF record id when required — enables chained commit+execute.",
                },
                "scenario_id": {
                    "type": "string",
                    "description": "Scenario id for atlas/parametric tasks.",
                },
                "split": {
                    "type": "string",
                    "description": "Split for FL-2 capability family.",
                },
                "execute": {
                    "type": "boolean",
                    "default": True,
                    "description": "Under strict v2: chained execute only when handles are present.",
                },
            },
        }
    if tool_name == "submit_solution":
        return {
            "type": "object",
            "additionalProperties": False,
            "required": ["route_id", "execution_id", "final_artifact_id"],
            "properties": {
                "route_id": {"type": "string"},
                "execution_id": {"type": "string"},
                "final_artifact_id": {"type": "string"},
                "rationale": {"type": "string"},
                "model_id_used": {"type": "string"},
            },
        }
    return pydantic_parameters_schema(STRICT_TOOL_MODELS[tool_name])


def _schema_keys(params: Mapping[str, Any]) -> Tuple[Set[str], Set[str]]:
    props = set((params.get("properties") or {}).keys())
    required = set(params.get("required") or [])
    return props, required


def validate_tool_parameters_against_model(
    tool_name: str,
    parameters: Mapping[str, Any],
) -> List[str]:
    """Return mismatch messages; empty list means contract OK."""
    model = STRICT_TOOL_MODELS.get(tool_name)
    if model is None:
        return [f"unknown_strict_tool:{tool_name}"]
    expected = strict_tool_parameters_for_llm(tool_name)
    exp_props, exp_req = _schema_keys(expected)
    act_props, act_req = _schema_keys(parameters)
    errors: List[str] = []
    missing_props = exp_props - act_props
    extra_props = act_props - exp_props
    if missing_props:
        errors.append(f"{tool_name}:missing_properties:{sorted(missing_props)}")
    if extra_props:
        errors.append(f"{tool_name}:extra_properties:{sorted(extra_props)}")
    if exp_req != act_req:
        errors.append(
            f"{tool_name}:required_mismatch:expected={sorted(exp_req)} actual={sorted(act_req)}"
        )
    return errors
