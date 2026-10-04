"""Per-backbone agent budgets for unified @143 reruns (DL-199)."""

from __future__ import annotations

from typing import Any, Dict, Mapping

# Per-cell SLURM ``timeout`` cap (outer wall). Must exceed inner ``max_wall_s`` or cells die at limit_wall.
UNIFIED_CELL_OUTER_WALL_S = 300
OUTER_WALL_BUFFER_S = 120

# Empirical: Mixtral/Olmo need more steps/wall on E12 cells; outer must cover inner budget.
BACKBONE_AGENT_LIMITS: Dict[str, Dict[str, int]] = {
    "llama70b": {"max_steps": 30, "max_wall_s": 600, "outer_base_s": UNIFIED_CELL_OUTER_WALL_S},
    "gemma31b": {"max_steps": 30, "max_wall_s": 600, "outer_base_s": UNIFIED_CELL_OUTER_WALL_S},
    "deepseek": {"max_steps": 30, "max_wall_s": 600, "outer_base_s": UNIFIED_CELL_OUTER_WALL_S},
    "mixtral8x22": {"max_steps": 50, "max_wall_s": 1200, "outer_base_s": UNIFIED_CELL_OUTER_WALL_S},
    "olmo32b": {"max_steps": 50, "max_wall_s": 1200, "outer_base_s": UNIFIED_CELL_OUTER_WALL_S},
}

INCOMPLETE_BOOST: Dict[str, int] = {
    "max_steps": 45,
    "max_wall_s": 1200,
    "outer_base_s": UNIFIED_CELL_OUTER_WALL_S,
}


def _coherent_outer_wall(lim: Dict[str, int]) -> Dict[str, int]:
    out = dict(lim)
    inner = int(out.get("max_wall_s") or 0)
    floor = int(out.get("outer_base_s") or UNIFIED_CELL_OUTER_WALL_S)
    out["outer_base_s"] = max(floor, inner + OUTER_WALL_BUFFER_S)
    return out


def agent_limits_for_backbone(
    backbone: str,
    *,
    incomplete_boost: bool = False,
) -> Dict[str, int]:
    base = dict(BACKBONE_AGENT_LIMITS.get(backbone, BACKBONE_AGENT_LIMITS["llama70b"]))
    if incomplete_boost:
        for key, val in INCOMPLETE_BOOST.items():
            base[key] = max(int(base.get(key, 0)), int(val))
    return _coherent_outer_wall(base)


def apply_limits_to_env(backbone: str, env: Mapping[str, str], *, incomplete_boost: bool = False) -> Dict[str, str]:
    """Return env dict with HWA_AGENT_* exports for submit scripts."""
    lim = agent_limits_for_backbone(backbone, incomplete_boost=incomplete_boost)
    out = dict(env)
    out["HWA_AGENT_MAX_STEPS"] = str(lim["max_steps"])
    out["HWA_AGENT_MAX_WALL_S"] = str(lim["max_wall_s"])
    out["HWA_CELL_OUTER_WALL_S"] = str(lim["outer_base_s"])
    return out
