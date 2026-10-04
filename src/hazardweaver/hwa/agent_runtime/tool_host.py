"""Resolve WildfireAgentTools → bound tool env for lease/controller/abstention."""

from __future__ import annotations

from typing import Any


def resolve_agent_host(host: Any) -> Any:
    """WildfireAgentTools → bound WildfireToolEnv; env-like hosts pass through."""
    env = getattr(host, "_agent_env", None)
    if env is not None:
        return env
    if hasattr(host, "tools") and hasattr(host, "workdir"):
        return host
    return host
