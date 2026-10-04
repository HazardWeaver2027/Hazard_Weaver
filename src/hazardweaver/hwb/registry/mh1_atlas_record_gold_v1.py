"""Per-fire PFDF record gold for MH-1 atlas headline instances (Gorr 2026 WEST)."""

from __future__ import annotations

import re
from typing import Any, Dict, Optional

_MH1_RECORD_SCENARIO_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*_[A-Za-z0-9]+_\d+$")


def is_mh1_pfdf_record_scenario(scenario_id: str) -> bool:
    """True when ``scenario_id`` is a USGS PFDF ``record_id`` (not CRC variant key)."""
    sid = str(scenario_id or "").strip()
    if not sid or sid.startswith("mh1_") or sid.startswith("CAP-"):
        return False
    return bool(_MH1_RECORD_SCENARIO_RE.match(sid))


def compute_mh1_record_log_volume_v1(record_id: str) -> float:
    """Gorr WEST ``log_volume`` (stored as ``log_volume_v1`` metric) for one inventory row."""
    from models.pfdf_volume_adapter.gorr_west import GorrWestPFDFVolumeAdapter
    from hazardweaver.hwa.pfdf_agent.data_access import PfdfDataAccess

    rec = PfdfDataAccess().get_record(str(record_id))
    out = GorrWestPFDFVolumeAdapter().predict(dict(rec))
    log_vol = out.get("log_volume_v1")
    if log_vol is None:
        log_vol = out.get("log_volume")
    if log_vol is None or not out.get("valid", True):
        raise ValueError(f"mh1_record_gold_unavailable:{record_id}")
    return float(log_vol)


def mh1_record_scenario_ref(record_id: str) -> Dict[str, Any]:
    """Sidecar-shaped gold entry for one atlas ``record_id`` scenario."""
    rid = str(record_id).strip()
    log_vol = compute_mh1_record_log_volume_v1(rid)
    return {
        "record_id": rid,
        "scenario_id": rid,
        "outputs": {
            "metric_name": "log_volume_v1",
            "reference_score": log_vol,
            "log_volume_v1": log_vol,
            "scenario_id": rid,
        },
        "tolerance": {
            "metric": "log_volume_v1",
            "max_abs_error": 1.0,
        },
        "provenance": {
            "claim_tier": "mh1_atlas_headline_record",
            "gold_source": "gorr_2026_west_formula",
            "record_id": rid,
        },
    }


def try_mh1_record_scenario_ref(scenario_id: str) -> Optional[Dict[str, Any]]:
    if not is_mh1_pfdf_record_scenario(scenario_id):
        return None
    try:
        return mh1_record_scenario_ref(scenario_id)
    except (KeyError, ValueError, TypeError):
        return None
