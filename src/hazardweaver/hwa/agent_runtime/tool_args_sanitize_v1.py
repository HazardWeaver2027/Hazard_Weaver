"""Sanitize LLM tool-call arguments before dispatch (strict v2 hygiene)."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Tuple

from hazardweaver.hwa.agent_runtime.hcg_tools import coerce_artifact_id_sequence

PLACEHOLDER_STRINGS = frozenset(
    {"null", "none", "undefined", "nil", "n/a", "na", ""}
)

# Tool args that must be list[str] but small LLMs often stringify as "['a', 'b']".
_LIST_ARG_KEYS: Dict[str, Tuple[str, ...]] = {
    "controller_enumerate_routes": ("sources",),
    "tool_hcg_find_paths": ("sources",),
    "tool_hcg_run_path": ("sources",),
    "tool_hcg_search_composition": ("sources",),
}


def is_placeholder_value(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str) and value.strip().lower() in PLACEHOLDER_STRINGS:
        return True
    return False


def sanitize_tool_arguments(
    tool_name: str,
    args: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Drop null/placeholder keys so optional fields behave as omitted."""
    if not args:
        return {}
    out: Dict[str, Any] = {}
    for key, val in dict(args).items():
        if is_placeholder_value(val):
            continue
        if isinstance(val, dict):
            nested = {
                k: v
                for k, v in val.items()
                if not is_placeholder_value(v)
            }
            if nested:
                out[key] = nested
            continue
        if key in _LIST_ARG_KEYS.get(tool_name, ()):
            out[key] = coerce_artifact_id_sequence(val)
            continue
        out[key] = val
    return out


def coerce_optional_str(value: Any) -> Optional[str]:
    if is_placeholder_value(value):
        return None
    return str(value).strip() or None
