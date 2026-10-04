"""Structured controller decision logging (controller_decisions.jsonl)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional


def decision_log_path(workdir: Path) -> Path:
    return Path(workdir) / "controller_decisions.jsonl"


def append_decision(
    workdir: Path,
    *,
    checkpoint: int,
    action: str,
    route_id: Optional[str] = None,
    capability_ids: Optional[list[str]] = None,
    a_sci: Optional[Mapping[str, Any]] = None,
    a_cap: Optional[Mapping[str, Any]] = None,
    theory_arm: Optional[str] = None,
    execution_id: Optional[str] = None,
    notes: Optional[str] = None,
    extra: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    row: Dict[str, Any] = {
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "checkpoint": checkpoint,
        "action": action,
    }
    if route_id is not None:
        row["route_id"] = route_id
    if capability_ids is not None:
        row["capability_ids"] = list(capability_ids)
    if a_sci is not None:
        row["A_sci"] = dict(a_sci)
    if a_cap is not None:
        row["A_cap"] = dict(a_cap)
    reach_cert = (extra or {}).get("reachability_certificate") if extra else None
    if reach_cert is None and a_cap is not None and isinstance(a_cap, Mapping):
        reach_cert = a_cap.get("reachability_certificate")
    if reach_cert is not None:
        row["reachability_certificate"] = dict(reach_cert) if isinstance(reach_cert, Mapping) else reach_cert
    if theory_arm is not None:
        row["theory_arm"] = theory_arm
    if execution_id is not None:
        row["execution_id"] = execution_id
    if notes is not None:
        row["notes"] = notes
    if extra:
        row.update(dict(extra))

    path = decision_log_path(workdir)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
    try:
        from hazardweaver.hwa.runtime.trajectory_ledger import record_controller_decision

        record_controller_decision(workdir, row)
    except Exception:  # noqa: BLE001 — ledger must not break controller
        pass
    return row


def load_decisions(workdir: Path) -> list[Dict[str, Any]]:
    path = decision_log_path(workdir)
    if not path.is_file():
        return []
    rows: list[Dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows
