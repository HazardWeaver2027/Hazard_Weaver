"""Single-source typed contract for commit → run_capability → submit_solution."""

from __future__ import annotations

from typing import Any, Mapping, NewType, Optional, Tuple, TypedDict

from hazardweaver.hwa.agent_runtime.tool_args_sanitize_v1 import is_placeholder_value

RouteId = NewType("RouteId", str)
ExecutionToken = NewType("ExecutionToken", str)

ROUTE_ID_PREFIX = "route:"


class CommitRouteResponse(TypedDict, total=False):
    ok: bool
    route_id: str
    execution_token: str
    lease_id: str
    allowed_capability_ids: list


class RunCapabilityRequest(TypedDict, total=False):
    capability_id: str
    controller_token: str
    route_id: str
    handles: dict


class SubmitSolutionRequest(TypedDict, total=False):
    route_id: str
    execution_id: str
    final_artifact_id: str


def make_route_id_for_capability(capability_id: str) -> RouteId:
    cid = str(capability_id or "").strip()
    return RouteId(f"route:cap:{cid}")


def is_committed_route_id(route_id: str) -> bool:
    return str(route_id or "").strip().startswith(ROUTE_ID_PREFIX)


def route_id_matches_capability(route_id: str, capability_id: str) -> bool:
    rid = str(route_id or "").strip()
    cid = str(capability_id or "").strip()
    if not rid or not cid:
        return False
    if rid == make_route_id_for_capability(cid):
        return True
    return rid.startswith(ROUTE_ID_PREFIX) and cid in rid


def normalize_controller_route_id(
    route_id: Optional[str],
    *,
    capability_id: Optional[str] = None,
) -> Optional[str]:
    """Map bare ``CAP-*`` / ``cap:*`` LLM aliases onto canonical ``route:cap:*`` ids."""
    rid = str(route_id or "").strip()
    if not rid or is_placeholder_value(rid):
        return None
    if rid.startswith(ROUTE_ID_PREFIX):
        return rid
    if rid.startswith("CAP-"):
        return str(make_route_id_for_capability(rid))
    cid = str(capability_id or "").strip()
    if cid.startswith("CAP-") and (rid == cid or rid == f"cap:{cid}"):
        return str(make_route_id_for_capability(cid))
    return rid


def normalize_controller_token(
    *,
    controller_token: Optional[str] = None,
    handles: Optional[Mapping[str, Any]] = None,
    kwargs: Optional[Mapping[str, Any]] = None,
) -> Optional[str]:
    """Map commit ``execution_token`` onto run_capability ``controller_token``."""
    if controller_token and not is_placeholder_value(controller_token):
        return str(controller_token).strip()
    h = handles or {}
    for key in ("controller_token", "execution_token"):
        val = h.get(key)
        if val is not None and not is_placeholder_value(val):
            return str(val).strip()
    kw = kwargs or {}
    for key in ("controller_token", "execution_token"):
        val = kw.get(key)
        if val is not None and not is_placeholder_value(val):
            return str(val).strip()
    return None


def resolve_route_id(
    *,
    route_id: Optional[str] = None,
    handles: Optional[Mapping[str, Any]] = None,
    capability_id: Optional[str] = None,
    active_route_id: Optional[str] = None,
    strict_v2: bool = False,
) -> Tuple[Optional[str], Optional[str]]:
    """Return ``(route_id, error_code)``; error_code set when strict requires explicit route."""
    rid = str(route_id or "").strip()
    if is_placeholder_value(rid):
        rid = ""
    if not rid and handles:
        h_rid = str(handles.get("route_id") or "").strip()
        if h_rid and not is_placeholder_value(h_rid):
            rid = h_rid
    if not rid and active_route_id:
        rid = str(active_route_id).strip()
    if not rid and not strict_v2 and capability_id:
        rid = f"cap:{capability_id}"
    if not rid and strict_v2:
        return None, "route_id_required"
    norm = normalize_controller_route_id(rid, capability_id=capability_id)
    if norm:
        rid = norm
    if rid and capability_id and strict_v2 and rid == f"cap:{capability_id}":
        return None, "route_id_format_mismatch"
    return (rid or None), None
