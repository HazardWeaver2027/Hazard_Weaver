"""W3/MW4 SWE-style HWA agent runtime (wildfire + PFDF packs; no Phase-04 fixed graph)."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hazardweaver.hwa.agent_runtime.loop import AgentRunResult, SWEAgentLoop

__all__ = ["SWEAgentLoop", "AgentRunResult"]


def __getattr__(name: str):
    if name == "SWEAgentLoop":
        from hazardweaver.hwa.agent_runtime.loop import SWEAgentLoop

        return SWEAgentLoop
    if name == "AgentRunResult":
        from hazardweaver.hwa.agent_runtime.loop import AgentRunResult

        return AgentRunResult
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
