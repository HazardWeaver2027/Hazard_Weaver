"""Scoped Execution Lease minting and validation ()."""

from __future__ import annotations

import secrets
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Mapping, Optional, Sequence

from hazardweaver.hcg.contracts.execution_event import ScopedExecutionLease


def mint_lease(
    *,
    route_id: str,
    allowed_capability_ids: Sequence[str],
    state_version: str = "s0",
    certificate_version: str = "",
    segment_id: str = "",
    ttl_s: int = 600,
    lease_id: Optional[str] = None,
) -> ScopedExecutionLease:
    lid = lease_id or f"L_{secrets.token_hex(8)}"
    expires = datetime.now(timezone.utc) + timedelta(seconds=int(ttl_s))
    return ScopedExecutionLease(
        lease_id=lid,
        route_id=str(route_id),
        segment_id=str(segment_id or ""),
        certificate_version=str(certificate_version or ""),
        state_version=str(state_version or "s0"),
        allowed_capability_ids=[str(c) for c in allowed_capability_ids],
        expires_at=expires,
    )


def lease_to_dict(lease: ScopedExecutionLease) -> Dict[str, Any]:
    return lease.model_dump(mode="json")


def lease_from_dict(data: Mapping[str, Any]) -> ScopedExecutionLease:
    return ScopedExecutionLease.model_validate(dict(data))


def store_lease_on_host(host: Any, lease: ScopedExecutionLease) -> None:
    if hasattr(host, "session_state"):
        host.session_state.active_lease = lease_to_dict(lease)
    elif hasattr(host, "state"):
        host.state.active_lease = lease_to_dict(lease)
    else:
        host._active_lease = lease_to_dict(lease)  # type: ignore[attr-defined]


def require_lease(host: Any) -> Optional[ScopedExecutionLease]:
    raw = None
    if hasattr(host, "session_state") and getattr(host.session_state, "active_lease", None):
        raw = host.session_state.active_lease
    elif hasattr(host, "state") and getattr(host.state, "active_lease", None):
        raw = host.state.active_lease
    elif hasattr(host, "_active_lease"):
        raw = getattr(host, "_active_lease", None)
    if not raw:
        return None
    return lease_from_dict(raw)


def bind_controller_commit(
    controller: Any,
    route_id: str,
    allowed_capability_ids: Sequence[str],
) -> ScopedExecutionLease:
    """Mint lease after controller commit; attach to existing execution token."""
    state = getattr(controller, "state", None)
    sv = getattr(state, "state_version", "s0") if state else "s0"
    cv = getattr(state, "certificate_version", "") if state else ""
    seg = getattr(state, "segment_id", "") if state else ""
    lease = mint_lease(
        route_id=route_id,
        allowed_capability_ids=allowed_capability_ids,
        state_version=sv,
        certificate_version=cv,
        segment_id=seg,
    )
    tokens = getattr(controller, "_execution_tokens", None) or {}
    for token, rec in reversed(list(tokens.items())):
        if rec.get("route_id") == route_id and not rec.get("used"):
            rec["lease_id"] = lease.lease_id
            break
    env = getattr(controller, "env", None)
    if env is not None:
        store_lease_on_host(env, lease)
    return lease
