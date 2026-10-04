"""SayCan / Geo-OLM tool surface policy under Π_adm."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from hazardweaver.hwa.agent_runtime.tool_specs import openai_tool_specs


def tool_specs_for_controller(
    pack_root: Path,
    *,
    include_optional: bool = True,
    allowed_tool_ids: Optional[Sequence[str]] = None,
    admissible_routes: Optional[Sequence[Mapping[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Expose controller semantic tools; strip exec tools when Π_adm is active."""
    specs = openai_tool_specs(
        pack_root,
        include_optional=include_optional,
        allowed_tool_ids=list(allowed_tool_ids) if allowed_tool_ids else None,
        controller_mode=True,
    )
    if admissible_routes is not None and not any(r.get("admissible") for r in admissible_routes):
        # No admissible routes — keep clarify/abstain terminals only via controller policy
        return specs
    return specs
