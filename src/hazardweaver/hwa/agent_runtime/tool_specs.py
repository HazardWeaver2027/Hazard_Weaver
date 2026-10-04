"""Load pack tools.json and emit OpenAI-compatible tool specs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from hazardweaver.hwa.wildfire.data_store import PACK_ROOT

from hazardweaver.hwa.scientific_controller.tool_policy import (
    CONTROLLER_EXEC_TOOLS,
    CONTROLLER_READONLY_TOOLS,
    CONTROLLER_SEMANTIC_TOOLS,
)

OPTIONAL_PLAN_TOOL_NAMES = frozenset({"suggest_plan", "validate_plan"})

CORE_TOOL_NAMES = frozenset(
    {
        "list_inventory",
        "read_card",
        "load_sample",
        "run_predictor",
        "run_capability",
        "inspect_artifact",
        "get_execution_result",
        "tool_hcg_find_paths",
        "tool_hcg_run_path",
        "tool_hcg_explain_edge",
        "ask_user",
        "submit_answer",
        "submit",  # alias → submit_answer (G1 / adversarial schema)
        "submit_solution",
        "submit_clarification",
        "submit_abstention",
    }
)


def load_pack_tools(pack_root: Path | None = None) -> Dict[str, Any]:
    root = Path(pack_root or PACK_ROOT)
    return json.loads((root / "tools.json").read_text(encoding="utf-8"))


def _openai_function(entry: Mapping[str, Any]) -> Dict[str, Any]:
    name = str(entry["name"])
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": entry.get("description", ""),
            "parameters": _parameters_for_tool(name, entry),
        },
    }


def _parameters_for_tool(name: str, entry: Mapping[str, Any]) -> Dict[str, Any]:
    from hazardweaver.hwa.experiments.agent_strict_v2 import agent_strict_v2_enabled

    if agent_strict_v2_enabled():
        from hazardweaver.hwa.contracts.strict_tool_models_v1 import (
            STRICT_TOOL_MODELS,
            strict_tool_parameters,
        )

        if name in STRICT_TOOL_MODELS:
            return strict_tool_parameters(name)
    return entry.get("parameters") or {"type": "object", "properties": {}}


def openai_tool_specs(
    pack_root: Path | None = None,
    *,
    include_optional: bool = True,
    allowed_tool_ids: Optional[Sequence[str]] = None,
    controller_mode: bool = False,
) -> List[Dict[str, Any]]:
    """Convert pack tool entries to OpenAI `tools` array.

    Optional plan tools are exposed when ``include_optional`` even if the task
    inventory omits them (W3-4: helpers, not fixed controller).

    When ``controller_mode=True``, execution tools are stripped and controller
    semantic tools are injected (ICLR §4 Deterministic Scientific Controller).
    """
    from hazardweaver.hwa.scientific_controller.tools import CONTROLLER_TOOL_SPECS

    doc = load_pack_tools(pack_root)
    allowed = set(allowed_tool_ids) if allowed_tool_ids is not None else None
    from hazardweaver.hwa.experiments.agent_strict_v2 import agent_strict_v2_allows_run_capability

    specs: List[Dict[str, Any]] = []
    for entry in doc.get("tools", []):
        name = entry["name"]
        if controller_mode and name in CONTROLLER_EXEC_TOOLS:
            if (
                name == "run_capability"
                and agent_strict_v2_allows_run_capability()
                and (allowed is None or name in allowed)
            ):
                specs.append(_openai_function(entry))
            continue
        if name in OPTIONAL_PLAN_TOOL_NAMES:
            if include_optional and not controller_mode:
                specs.append(_openai_function(entry))
            continue
        if allowed is not None and name not in allowed:
            continue
        specs.append(_openai_function(entry))
    if controller_mode:
        existing = {s["function"]["name"] for s in specs}
        for spec in CONTROLLER_TOOL_SPECS:
            if spec["function"]["name"] not in existing:
                specs.append(spec)
    return specs


def all_tool_names(pack_root: Path | None = None) -> List[str]:
    doc = load_pack_tools(pack_root)
    return [t["name"] for t in doc.get("tools", [])]
