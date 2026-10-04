"""Agent bench task schema helpers (HWB → HW)."""

from __future__ import annotations

import json
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

SCHEMA_PATH = Path(__file__).resolve().parent / "schema" / "agent_bench_task.schema.json"
EXAMPLES_DIR = Path(__file__).resolve().parent / "examples"

GOLD_KEY = "gold"
HCG_ORACLE_KEY = "hcg_oracle"
SOLVER_FORBIDDEN_KEYS = frozenset({GOLD_KEY, HCG_ORACLE_KEY})
SCHEMA_VERSION = "agent_bench_task/v1"
SCHEMA_VERSION_V1_1 = "agent_bench_task/v1.1"
V1_1_SCHEMA_PATH = (
    Path(__file__).resolve().parents[3]
    / "docs/engineering/wildfire_dual_system_mvp/contracts/schemas/agent_bench_task.v1_1.schema.json"
)
V1_1_REQUIRED_TOP = (
    "schema_version",
    "task_id",
    "task_layer",
    "domain",
    "task_family",
    "difficulty",
    "user_facing_goal",
    "solver_visible",
    "success_criteria",
    "provenance",
)


class SampleRef(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_id: str
    sample_id: str
    split: Optional[str] = None
    notes: Optional[str] = None


class SolverInputs(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sample_refs: List[SampleRef] = Field(min_length=1)
    constraints: List[str] = Field(default_factory=list)
    hints: List[str] = Field(default_factory=list)


class AllowedInventory(BaseModel):
    model_config = ConfigDict(extra="forbid")

    dataset_ids: List[str]
    model_ids: List[str]
    tool_ids: List[str]


class SolverVisible(BaseModel):
    model_config = ConfigDict(extra="forbid")

    inputs: SolverInputs
    allowed_inventory: AllowedInventory
    pack_refs: List[str] = Field(default_factory=list)


class SuccessCriteria(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer_type: Literal[
        "class_id",
        "class_label",
        "vector",
        "raster_summary",
        "text_report",
        "abstain_or_explain",
        "log_volume",
    ]
    description: str
    metric: Optional[str] = None


class Gold(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: Any = None
    label: Any = None
    preferred_model_id: Optional[str] = None
    gold_type: Optional[
        Literal[
            "observed_label",
            "oracle_tool_result",
            "abstain_reference",
            "counterfactual_oracle",
        ]
    ] = None
    eval_notes: Optional[str] = None


class Provenance(BaseModel):
    model_config = ConfigDict(extra="forbid")

    generator_id: str
    generator_version: str
    created_utc: datetime
    seed: Optional[int] = None
    generation_mode: Optional[
        Literal[
            "template_reverse_qa",
            "llm_back_instruct",
            "handwrite",
            "pfdf_factual_compose",
            "pfdf_counterfactual",
        ]
    ] = None
    llm_provider: Optional[str] = None
    llm_model_id: Optional[str] = None
    bootstrap_note: Optional[str] = None


class AgentBenchTask(BaseModel):
    """Full HWB task (may include gold)."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal["agent_bench_task/v1"]
    task_id: str
    domain: Literal["wildfire", "pfdf"]
    task_family: Literal[
        "danger",
        "spread",
        "multi_step_compose",
        "factual_compose",
        "counterfactual_compose",
    ]
    diversity_bucket: Literal[
        "A_single_danger",
        "B_single_spread",
        "C_model_choice",
        "D_constrained",
        "E_multi_step",
        "direct_inference",
        "asset_discovery",
        "schema_adapter",
        "model_choice",
        "multi_step_compose",
        "failure_recovery",
        "compare_interpret",
        "clarification",
        "factual_compose",
        "factual_compose_tool",
        "factual_asset_discovery",
        "counterfactual_burn",
        "counterfactual_rainfall",
        "counterfactual_oracle_burn",
    ]
    user_facing_goal: str = Field(min_length=20)
    solver_visible: SolverVisible
    success_criteria: SuccessCriteria
    provenance: Provenance
    gold: Optional[Gold] = None
    theory_or_card_refs: List[str] = Field(default_factory=list)

    @field_validator("task_id")
    @classmethod
    def _task_id_shape(cls, value: str) -> str:
        import re

        if not re.fullmatch(r"[a-z0-9][a-z0-9_\-]{2,127}", value):
            raise ValueError("task_id must match ^[a-z0-9][a-z0-9_\\-]{2,127}$")
        return value


def load_schema() -> Dict[str, Any]:
    return json.loads(SCHEMA_PATH.read_text(encoding="utf-8"))


def load_task(path: Path | str) -> Dict[str, Any]:
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8"))


def solver_view(task: Dict[str, Any]) -> Dict[str, Any]:
    """Strip emit-only fields (gold, hcg_oracle) for HWA agent consumption (GOLD_VISIBILITY)."""
    view = deepcopy(task)
    for key in SOLVER_FORBIDDEN_KEYS:
        view.pop(key, None)
    return view


def validate_task(task: Dict[str, Any], *, require_gold: bool = False) -> AgentBenchTask:
    """Validate legacy v1 task dict via Pydantic (JSON Schema file remains the human/doc contract)."""
    parsed = AgentBenchTask.model_validate(task)
    if require_gold and parsed.gold is None:
        raise ValueError("full HWB task requires 'gold' for external eval")
    return parsed


def validate_task_v1_1(
    task: Dict[str, Any],
    *,
    require_gold: bool = False,
    require_hcg_oracle: bool = False,
) -> Dict[str, Any]:
    """Lightweight v1.1 required-field check (no jsonschema dependency)."""
    if task.get("schema_version") != SCHEMA_VERSION_V1_1:
        raise ValueError(
            f"schema_version must be {SCHEMA_VERSION_V1_1!r}, got {task.get('schema_version')!r}"
        )
    missing = [k for k in V1_1_REQUIRED_TOP if k not in task]
    if missing:
        raise ValueError(f"v1.1 task missing required keys: {missing}")
    if task["task_layer"] not in {"data_task", "scenario_task"}:
        raise ValueError(f"invalid task_layer: {task['task_layer']!r}")
    if task["task_layer"] == "scenario_task" and not task.get("parent_data_task_id"):
        raise ValueError("scenario_task requires parent_data_task_id")
    if not isinstance(task.get("user_facing_goal"), str) or len(task["user_facing_goal"]) < 20:
        raise ValueError("user_facing_goal must be a string of length >= 20")
    import re

    tid = str(task.get("task_id") or "")
    if not re.fullmatch(r"[a-z0-9][a-z0-9_\-]{2,127}", tid):
        raise ValueError(f"invalid task_id shape: {tid!r}")
    sv = task.get("solver_visible") or {}
    inputs = sv.get("inputs") or {}
    if not inputs.get("sample_refs"):
        raise ValueError("solver_visible.inputs.sample_refs required")
    inv = sv.get("allowed_inventory") or {}
    for key in ("dataset_ids", "model_ids", "tool_ids"):
        if key not in inv:
            raise ValueError(f"allowed_inventory.{key} required")
    sc = task.get("success_criteria") or {}
    if "answer_type" not in sc or "description" not in sc:
        raise ValueError("success_criteria requires answer_type and description")
    prov = task.get("provenance") or {}
    for key in ("generator_id", "generator_version", "created_utc"):
        if key not in prov:
            raise ValueError(f"provenance.{key} required")
    if require_gold and "gold" not in task:
        raise ValueError("full HWB task requires 'gold'")
    if require_hcg_oracle and "hcg_oracle" not in task:
        raise ValueError("full emit task requires 'hcg_oracle'")
    return task


def _gold_answer_substrings(task: Dict[str, Any]) -> List[str]:
    gold = task.get("gold") or {}
    answer = gold.get("answer")
    out: List[str] = []
    if answer is None:
        return out
    if isinstance(answer, dict):
        for v in answer.values():
            if isinstance(v, (int, float)):
                out.append(str(v))
                out.append(f"{float(v):.4f}")
            elif isinstance(v, str) and len(v) >= 4:
                out.append(v)
    elif isinstance(answer, (int, float)):
        out.append(str(answer))
    return out


def assert_no_gold_in_solver_view(view: Dict[str, Any], *, full: Optional[Dict[str, Any]] = None) -> None:
    leaked = SOLVER_FORBIDDEN_KEYS.intersection(view.keys())
    if leaked:
        raise AssertionError(f"solver view leaked hidden keys: {sorted(leaked)}")
    visible_blob = json.dumps(view.get("solver_visible") or {}, ensure_ascii=False)
    if "gold_edges" in visible_blob:
        raise AssertionError("solver_visible must not contain gold_edges")
    goal = str(view.get("user_facing_goal") or "")
    hints = (view.get("solver_visible") or {}).get("inputs", {}).get("hints") or []
    text = goal + "\n" + "\n".join(str(h) for h in hints)
    source = full if full is not None else view
    for tok in _gold_answer_substrings(source):
        if tok and tok in text:
            raise AssertionError(f"solver-visible text leaks gold answer token: {tok!r}")
    edges = ((source.get("gold") or {}).get("gold_edges")) or []
    for edge in edges:
        if edge and edge in text:
            raise AssertionError(f"solver-visible text leaks gold edge id: {edge!r}")
