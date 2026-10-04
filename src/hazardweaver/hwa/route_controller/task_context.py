"""Formal task context q from solver_view (HWA_AGENT_STATE_v2)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional


@dataclass
class TaskContext:
    """Solver-visible task context q."""

    task_id: str
    domain: str = ""
    task_family: str = ""
    route_family_id: str = ""
    goal_artifacts: List[str] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)
    region: str = ""
    raw: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_task(cls, task: Mapping[str, Any]) -> "TaskContext":
        solver = task.get("solver_visible") or {}
        inputs = solver.get("inputs") or {}
        return cls(
            task_id=str(task.get("task_id") or ""),
            domain=str(task.get("domain") or ""),
            task_family=str(task.get("task_family") or ""),
            route_family_id=str(
                solver.get("route_family_id")
                or task.get("route_family_id")
                or task.get("task_family")
                or ""
            ),
            goal_artifacts=[str(g) for g in (inputs.get("goal_artifacts") or [])],
            constraints=[str(c) for c in (inputs.get("constraints") or [])],
            region=str(inputs.get("region") or ""),
            raw=dict(task),
        )

    def to_state_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "domain": self.domain,
            "task_family": self.task_family,
            "route_family_id": self.route_family_id,
            "goal_artifacts": self.goal_artifacts,
            "constraints": self.constraints,
            "region": self.region,
        }
