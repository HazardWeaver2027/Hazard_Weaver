"""HCG blocker taxonomy → HWA Minimal Repair R0–R6 + AgentDebug class ()."""

from __future__ import annotations

from typing import Dict, Literal

from hazardweaver.hcg.contracts.blocker_taxonomy import BlockerCode, blocker_to_repair_level

AgentDebugClass = Literal["System", "Planning", "Memory", "Reflection", "Action"]

# 
_BLOCKER_TO_AGENT_DEBUG: Dict[BlockerCode, AgentDebugClass] = {
    BlockerCode.CHECKPOINT_MISSING: "System",
    BlockerCode.DEPENDENCY_UNAVAILABLE: "System",
    BlockerCode.RESOURCE_UNAVAILABLE: "System",
    BlockerCode.PLATFORM_UNAVAILABLE: "System",
    BlockerCode.LICENSE_UNAVAILABLE: "System",
    BlockerCode.CAPABILITY_INVALIDATED: "System",
    BlockerCode.MISSING_ARTIFACT: "Planning",
    BlockerCode.MISSING_CAPABILITY: "System",
    BlockerCode.ADAPTER_UNAVAILABLE: "System",
    BlockerCode.IMPLEMENTATION_MISSING: "System",
    BlockerCode.INTERFACE_VARIABLE_MISMATCH: "Planning",
    BlockerCode.UNIT_MISMATCH: "Planning",
    BlockerCode.CRS_MISMATCH: "Planning",
    BlockerCode.SPATIAL_SUPPORT_MISMATCH: "Planning",
    BlockerCode.TEMPORAL_SUPPORT_MISMATCH: "Planning",
    BlockerCode.RESOLUTION_MISMATCH: "Planning",
    BlockerCode.VALIDATION_POLICY_UNMET: "Planning",
}

ALL_BLOCKER_CODES = frozenset(c.value for c in BlockerCode)


def repair_level_for_blocker(code: str) -> str:
    """Return R0–R6 or pre_execution_cert_bug."""
    return blocker_to_repair_level(code)


def agent_debug_class_for_blocker(code: str) -> AgentDebugClass:
    try:
        bc = BlockerCode(code)
    except ValueError:
        return "System"
    return _BLOCKER_TO_AGENT_DEBUG.get(bc, "System")


def should_recheck_a_cap(agent_class: AgentDebugClass) -> bool:
    return agent_class in {"System", "Action"}


def should_recheck_a_sci(agent_class: AgentDebugClass) -> bool:
    return agent_class in {"Planning", "Reflection"}
