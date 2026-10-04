"""Native step-by-step HWA_TRAJECTORY_v2 ledger (Task 5 — not post-hoc only)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

LEDGER_FILENAME = "hwa_trajectory_v2_ledger.jsonl"


def ledger_path(workdir: Path) -> Path:
    return Path(workdir) / LEDGER_FILENAME


def append_ledger_step(workdir: Path, step: Mapping[str, Any]) -> Dict[str, Any]:
    """Append one native trajectory step at event time."""
    workdir = Path(workdir)
    workdir.mkdir(parents=True, exist_ok=True)
    row: Dict[str, Any] = {
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        **dict(step),
    }
    if "step_index" not in row:
        row["step_index"] = len(load_ledger_steps(workdir))
    path = ledger_path(workdir)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    return row


def load_ledger_steps(workdir: Path) -> List[Dict[str, Any]]:
    path = ledger_path(workdir)
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def record_controller_decision(workdir: Path, decision: Mapping[str, Any]) -> Dict[str, Any]:
    return append_ledger_step(
        workdir,
        {
            "kind": "controller_decision",
            "action": decision.get("action"),
            "route_id": decision.get("route_id"),
            "checkpoint": decision.get("checkpoint"),
            "A_sci": decision.get("A_sci"),
            "A_cap": decision.get("A_cap"),
            "reachability_certificate": decision.get("reachability_certificate"),
            "execution_id": decision.get("execution_id"),
            "lease_id": (decision.get("extra") or {}).get("lease_id")
            if isinstance(decision.get("extra"), dict)
            else decision.get("lease_id"),
        },
    )


def record_tool_execution(
    workdir: Path,
    *,
    tool: str,
    capability_id: str = "",
    ok: bool = True,
    execution_id: str = "",
    lease_id: str = "",
    route_id: str = "",
    execution_certificate: Optional[Mapping[str, Any]] = None,
    reachability_certificate: Optional[Mapping[str, Any]] = None,
    execution_event: Optional[Mapping[str, Any]] = None,
    error: str = "",
    parameter_provenance: Optional[Mapping[str, Any]] = None,
    inference_backend: str = "",
    hcg_role: Optional[str] = None,
) -> Dict[str, Any]:
    return append_ledger_step(
        workdir,
        {
            "kind": "tool_execution",
            "tool": tool,
            "ok": ok,
            "capability_id": capability_id,
            "execution_id": execution_id,
            "lease_id": lease_id,
            "route_id": route_id,
            "execution_certificate": dict(execution_certificate) if execution_certificate else None,
            "reachability_certificate": dict(reachability_certificate) if reachability_certificate else None,
            "execution_event": dict(execution_event) if execution_event else None,
            "error": error or None,
            "parameter_provenance": dict(parameter_provenance) if parameter_provenance else None,
            "inference_backend": inference_backend or None,
            "hcg_role": hcg_role,
        },
    )
