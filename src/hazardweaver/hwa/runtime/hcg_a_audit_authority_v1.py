"""Read-only HCG-A audit authority for HWA seven-track tier / eligibility (DL-110b)."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

DEFAULT_HCG_A_AUDIT_ROOT = Path("runs/carp/audit")
DEFAULT_HCG_A_READINESS = Path("runs/carp/hcg_7track_scientific_readiness_v1.json")
HCGA_ABSTAIN_TIER = "hcg_a_abstain"


def load_hcg_a_audit_record(
    track: str,
    capability_id: str,
    *,
    audit_root: Path | str = DEFAULT_HCG_A_AUDIT_ROOT,
) -> Optional[Dict[str, Any]]:
    track_u = str(track).upper()
    cap = str(capability_id).upper()
    path = Path(audit_root) / track_u / cap / "audit_record.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def load_hcg_a_readiness() -> Dict[str, Any]:
    path = DEFAULT_HCG_A_READINESS
    if not path.is_file():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def hcg_a_completion_tier(
    track: str,
    capability_id: str,
    manifest_row: Optional[Mapping[str, Any]] = None,
) -> str:
    audit = load_hcg_a_audit_record(track, capability_id)
    if audit and audit.get("completion_tier"):
        return str(audit["completion_tier"]).upper()
    row = dict(manifest_row or {})
    return str(row.get("completion_tier") or "").upper()


def hcg_a_is_l2_headline_cap(
    track: str,
    capability_id: str,
    manifest_row: Optional[Mapping[str, Any]] = None,
) -> bool:
    return hcg_a_completion_tier(track, capability_id, manifest_row) == "L2"


def abstain_tier_from_audit(audit: Mapping[str, Any]) -> str:
    verdict = str(audit.get("verdict") or "")
    tier = str(audit.get("completion_tier") or "").upper()
    if tier == "L0" or verdict == "PI_WAIVER":
        return "L1_placeholder"
    if verdict == "RUNTIME_BLOCKED" or "BLOCKED" in verdict:
        return "L1_blocked_replay"
    if "EXTERNAL" in verdict:
        return HCGA_ABSTAIN_TIER
    return HCGA_ABSTAIN_TIER


def abstain_fields_from_audit(audit: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "trajectory_tier": abstain_tier_from_audit(audit),
        "hwb_headline_eligible": False,
        "hcg_a_abstain": True,
        "hcg_a_verdict": audit.get("verdict"),
        "hcg_a_blockers": list(audit.get("blockers_remaining") or []),
        "completion_tier": audit.get("completion_tier"),
    }
