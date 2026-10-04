"""CARE stopping rule — replaces fixed max_paths enumeration."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence


def care_should_stop(
    routes: Sequence[Mapping[str, Any]],
    *,
    n_steps: int,
    max_steps: int = 8,
    min_admissible: int = 1,
) -> Dict[str, Any]:
    """Deterministic CARE stop: enough admissible frontier or step budget."""
    admissible = [r for r in routes if r.get("admissible")]
    if len(admissible) >= min_admissible and n_steps >= 1:
        return {"stop": True, "reason": "min_admissible_met", "n_admissible": len(admissible)}
    if n_steps >= max_steps:
        return {"stop": True, "reason": "max_steps", "n_admissible": len(admissible)}
    return {"stop": False, "reason": "continue", "n_admissible": len(admissible)}
