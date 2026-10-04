"""Unified benchmark: prefer curator-gold registry execution at submit/terminal."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

_CAP_RE = re.compile(r"CAP-[A-Z0-9-]+")


def _cap_norm(capability_id: str) -> str:
    return str(capability_id or "").strip().split("__", 1)[0]


def _cap_from_route_id(route_id: str) -> str:
    match = _CAP_RE.search(str(route_id or ""))
    return match.group(0) if match else ""


def _inventory_row(workdir: Path, task: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    inv_path = Path(workdir) / "inventory_row.json"
    inv: Dict[str, Any] = {}
    if inv_path.is_file():
        try:
            inv = json.loads(inv_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            inv = {}
    if task:
        meta = task.get("metadata") if isinstance(task.get("metadata"), Mapping) else {}
        if isinstance(meta, Mapping):
            for key, val in meta.items():
                if key not in inv or inv.get(key) in (None, ""):
                    inv[key] = val
        for key in ("track", "taskpack_id", "w3_gold_capability_id", "unified_benchmark_v1"):
            if task.get(key) and not inv.get(key):
                inv[key] = task[key]
    return inv


def unified_gold_registry_submit_enabled(inventory_row: Mapping[str, Any]) -> bool:
    """Only unified headline / W3 rows — not legacy ablation decoy arms."""
    if not (
        inventory_row.get("unified_benchmark_v1")
        or inventory_row.get("hwb_headline_inventory")
        or inventory_row.get("dynamic_intervention_w3_47_v1")
    ):
        return False
    gold = str(
        inventory_row.get("w3_gold_capability_id")
        or inventory_row.get("unified_scenario_gold_capability_id")
        or ""
    ).strip()
    return gold.startswith("CAP-")


def _handles_from_execution_row(er: Mapping[str, Any]) -> Dict[str, str]:
    fa = er.get("final_artifact") if isinstance(er.get("final_artifact"), Mapping) else {}
    return {
        "route_id": str(er.get("route_id") or "").strip(),
        "execution_id": str(er.get("execution_id") or "").strip(),
        "final_artifact_id": str(fa.get("artifact_id") or "").strip(),
    }


def registry_handles_for_capability(workdir: Path, capability_id: str) -> Optional[Dict[str, str]]:
    """Return registry-backed handles for one capability if ok execution exists."""
    from hazardweaver.hwa.agent_runtime.execution_schema import list_registry_executions

    cap = _cap_norm(capability_id)
    if not cap:
        return None
    gold_route = f"route:cap:{cap}"
    for er in reversed(list_registry_executions(Path(workdir))):
        if str(er.get("status") or "") not in {"ok", "success"}:
            continue
        executed = [_cap_norm(str(c)) for c in (er.get("executed_capability_ids") or [])]
        route_cap = _cap_from_route_id(str(er.get("route_id") or ""))
        if cap not in executed and route_cap != cap and str(er.get("route_id") or "") != gold_route:
            continue
        handles = _handles_from_execution_row(er)
        if all(handles.values()):
            return handles
    return None


def prefer_unified_gold_registry_handles(
    workdir: Path,
    inventory_row: Mapping[str, Any],
) -> Optional[Dict[str, str]]:
    """When curator gold has ok registry execution, bind terminal submit to it."""
    if not unified_gold_registry_submit_enabled(inventory_row):
        return None
    from hazardweaver.hwa.benchmark.unified_dca_route_gate_v1 import gold_capability_id

    gold = gold_capability_id(inventory_row, Path(workdir))
    if not gold:
        return None
    return registry_handles_for_capability(Path(workdir), gold)


def maybe_redirect_unified_submit_to_gold_registry(
    workdir: Path,
    inventory_row: Mapping[str, Any],
    route_id: str,
    execution_id: str,
    final_artifact_id: str,
) -> tuple[str, str, str, bool]:
    """Redirect decoy submit to gold registry execution when agent already ran gold."""
    gold_handles = prefer_unified_gold_registry_handles(workdir, inventory_row)
    if not gold_handles:
        return route_id, execution_id, final_artifact_id, False
    pick_cap = _cap_from_route_id(route_id)
    gold_cap = _cap_from_route_id(gold_handles["route_id"])
    if pick_cap and gold_cap and pick_cap == gold_cap:
        return route_id, execution_id, final_artifact_id, False
    return (
        gold_handles["route_id"],
        gold_handles["execution_id"],
        gold_handles["final_artifact_id"],
        True,
    )
