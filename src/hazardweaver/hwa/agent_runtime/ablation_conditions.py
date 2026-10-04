"""Ablation inventory transforms.

Legacy Strength R3 arms: ``full`` / ``no_hcg`` / ``no_compose_macro`` (unchanged
semantics; kept as permanent aliases — see ``hwa/tests/test_r3_hcg_forced.py``).

Phase 3 W2 Hard-pack arms (``docs/engineering/hcg/ABLATION_HARD_G6_v1.yaml``,
pack ``g6_hard_v1``; semantics aligned with
``docs/engineering/hcg/ABLATION_MAIN_TABLE_G5.md``):

  - ``S0``: Theory ON (main arm; alias of ``full`` — theory_retrieve, when a
    pack exposes it, stays in the allowed inventory)
  - ``S1``: Theory OFF (strips a ``theory_retrieve`` / ``tool_theory_*`` tool
    id if present; inert no-op on packs that do not expose one yet)
  - ``S2``: adapter OFF (strips ``compose_*`` / ``adapter_*`` / ``tool_adapter_*``
    tool ids; direct capability calls via ``run_capability`` remain reachable)
  - ``S4``: select-one vs MultiDAG (strips ``tool_hcg_find_paths`` so the agent
    cannot enumerate/select among alternate routes; ``tool_hcg_run_path``
    stays for a single pre-selected path)

S3 (executable-only vs +proposed edges), S5 (weak single-shot baseline) and S6
(ScriptedLLM, CI-only) are not runtime tool-surface transforms and are
intentionally out of scope for this module.

Mutates a deep copy of the solver_view task only — never writes back to disk.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Mapping, Sequence

# Legacy Strength R3 conditions (unchanged semantics; permanent aliases).
_LEGACY_CONDITIONS = frozenset({"full", "no_hcg", "no_compose_macro"})

# Phase 3 W2 Hard-pack arms (docs/engineering/hcg/ABLATION_HARD_G6_v1.yaml).
_W2_CONDITIONS = frozenset({"S0", "S1", "S2", "S4"})

VALID_CONDITIONS = frozenset(_LEGACY_CONDITIONS | _W2_CONDITIONS)

# Conditions that behave as a pass-through (no tool_ids filtering) surface.
_PASSTHROUGH_CONDITIONS = frozenset({"full", "S0"})

# Injected when stripping compose macros so PFDF Hard/compose family can still
# use HCG (DS2 PFDF solver_view inventories historically omit tool_hcg_*).
_HCG_CORE = ("tool_hcg_find_paths", "tool_hcg_run_path")

# Forward-compatible tool-id family for the not-yet-wired-into-agent_runtime
# theory_retrieve capability (experiments/hkc/theory_retrieve_v1.py;
# see docs/engineering/ULTIMATE_PLAN_HCG_HWB_H0H8.md H7). Stripping is a no-op
# until a pack actually lists such a tool id in allowed_inventory.
_THEORY_TOOL_IDS = frozenset({"theory_retrieve", "tool_theory_retrieve"})
_THEORY_PREFIX = "tool_theory_"

# Adapter/compose family stripped for S2 (adapter OFF). Superset of the legacy
# no_compose_macro strip (adds explicit adapter_ / tool_adapter_ ids).
_ADAPTER_PREFIXES = ("compose_", "adapter_", "tool_adapter_")


def _is_theory_tool(tool_id: str) -> bool:
    t = str(tool_id)
    return t in _THEORY_TOOL_IDS or t.startswith(_THEORY_PREFIX)


def _is_adapter_tool(tool_id: str) -> bool:
    t = str(tool_id)
    return any(t.startswith(p) for p in _ADAPTER_PREFIXES)


def apply_ablation_condition(
    task: Mapping[str, Any],
    condition: str,
    *,
    ensure_hcg_on_no_compose: bool = True,
) -> Dict[str, Any]:
    """Return a deep-copied task with tool_ids filtered for ``condition``.

    Legacy: full | no_hcg | no_compose_macro (Strength R3; unchanged).
    W2 Hard arms: S0 | S1 | S2 | S4 (see module docstring).
    """
    cond = str(condition or "full").strip()
    if cond not in VALID_CONDITIONS:
        raise ValueError(
            f"invalid ablation condition {cond!r}; expected one of {sorted(VALID_CONDITIONS)}"
        )
    out = copy.deepcopy(dict(task))

    if cond in _PASSTHROUGH_CONDITIONS:
        out["_ablation_condition"] = cond
        out["_ablation_flags"] = {"theory": "on", "adapter": "on", "multi_dag": True}
        return out

    sv = dict(out.get("solver_visible") or {})
    inv = dict(sv.get("allowed_inventory") or {})
    tools: List[str] = [str(t) for t in (inv.get("tool_ids") or [])]
    flags: Dict[str, Any] = {"theory": "on", "adapter": "on", "multi_dag": True}

    if cond == "no_hcg":
        tools = [t for t in tools if not t.startswith("tool_hcg_")]
        flags["multi_dag"] = False
    elif cond == "no_compose_macro":
        tools = [t for t in tools if not t.startswith("compose_")]
        if ensure_hcg_on_no_compose:
            tools = _ensure_tools(tools, _HCG_CORE)
    elif cond == "S1":
        tools = [t for t in tools if not _is_theory_tool(t)]
        flags["theory"] = "off"
    elif cond == "S2":
        tools = [t for t in tools if not _is_adapter_tool(t)]
        if ensure_hcg_on_no_compose:
            tools = _ensure_tools(tools, _HCG_CORE)
        flags["adapter"] = "off"
    elif cond == "S4":
        tools = [t for t in tools if t != "tool_hcg_find_paths"]
        flags["multi_dag"] = False

    # Keep submit/list reachable for ablations that strip most of the surface.
    tools = _ensure_tools(tools, ("list_inventory", "submit_answer"))
    inv["tool_ids"] = tools
    sv["allowed_inventory"] = inv
    out["solver_visible"] = sv
    out["_ablation_condition"] = cond
    out["_ablation_flags"] = flags
    return out


def _ensure_tools(tools: Sequence[str], required: Sequence[str]) -> List[str]:
    out = list(tools)
    for t in required:
        if t not in out:
            out.append(t)
    return out


def tool_ids_of(task: Mapping[str, Any]) -> List[str]:
    inv = (task.get("solver_visible") or {}).get("allowed_inventory") or {}
    return [str(t) for t in (inv.get("tool_ids") or [])]
