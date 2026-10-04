"""DS-Agent CBR warm-start for headline G1 — route prior only, never answer copy.

**FROZEN (DL-142):** Cross-system import of DS baseline outputs is invalid for paper.
Use `headline_hwa_self_cbr_v1` instead. This module runs only when
``HWA_ALLOW_CROSS_SYSTEM_CBR=1`` (debug/repro of invalidated G1 arm).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from hazardweaver.hwa.experiments.headline_route_profile_v1 import (
    G1_PROFILE,
    cross_system_cbr_allowed,
    default_ds_cbr_out_root,
    is_g1_profile,
)


def _ds_instance_dir(ds_out_root: Path, instance_id: str) -> Path:
    return ds_out_root / "ds_agent" / str(instance_id).replace(":", "_")


def _capability_from_payload(data: Mapping[str, Any], *, source: str) -> Optional[str]:
    if source == "retrieval.json":
        cap = data.get("selected_edge_id")
        return str(cap).strip() if cap else None
    route_summary = data.get("route_summary") or {}
    cap = route_summary.get("selected_edge_id")
    if cap:
        return str(cap).strip()
    value = (data.get("final_artifact") or {}).get("value") or {}
    cap = value.get("selected_route") or value.get("capability_id")
    return str(cap).strip() if cap else None


def load_ds_cbr_capability(
    instance_id: str,
    ds_out_root: Path | None = None,
) -> Optional[str]:
    """Load DS baseline selected capability for instance (validation CBR prior)."""
    iid = str(instance_id or "").strip()
    if not iid:
        return None
    root = Path(ds_out_root or default_ds_cbr_out_root())
    run_dir = _ds_instance_dir(root, iid)
    for name in ("retrieval.json", "trajectory.json"):
        path = run_dir / name
        if not path.is_file():
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        cap = _capability_from_payload(data, source=name)
        if cap and not cap.startswith("schema_"):
            return cap
    return None


def attach_ds_cbr_metadata(
    task: Dict[str, Any],
    inventory_row: Mapping[str, Any],
    *,
    ds_out_root: Path | None = None,
) -> Dict[str, Any]:
    """Tag task with G1 profile + optional DS CBR capability prior."""
    meta = dict(task.get("metadata") or {})
    meta["headline_route_profile"] = G1_PROFILE
    iid = str(
        inventory_row.get("instance_id")
        or meta.get("instance_id")
        or meta.get("internal_task_id")
        or ""
    )
    cap = load_ds_cbr_capability(iid, ds_out_root)
    if cap:
        allowed = [
            str(e)
            for e in ((task.get("solver_visible") or {}).get("inputs") or {}).get("allowed_edge_ids") or []
            if e and not str(e).startswith("schema_")
        ]
        if not allowed or cap in allowed:
            meta["ds_cbr_capability"] = cap
            meta["ds_cbr_route_id"] = f"route:cap:{cap}"
            meta["ds_cbr_source"] = "ds_agent_retrieval_v1"
    task["metadata"] = meta
    return task


def should_attach_ds_cbr(condition: str) -> bool:
    if not cross_system_cbr_allowed():
        return False
    return condition == "full_hwa" and is_g1_profile()
