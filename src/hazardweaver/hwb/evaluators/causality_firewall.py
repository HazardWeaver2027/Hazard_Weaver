"""Submission-time decision-time causality firewall."""

from __future__ import annotations

from datetime import datetime
from typing import Any, List, Mapping, Optional


def _parse_ts(value: Any) -> Optional[datetime]:
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    if len(s) == 7 and s[4] == "-":  # YYYY-MM issue month
        s = f"{s}-01T00:00:00Z"
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None


def check_v_causal(
    steps: List[Mapping[str, Any]],
    *,
    cutoff_manifest: Mapping[str, Any],
) -> List[str]:
    """
    Fail if any step references artifacts with t_available > decision_time.

    cutoff_manifest keys: decision_time, issue_yyyymm (DR-OUT), forecast_init (TC).
    """
    errors: List[str] = []
    decision = _parse_ts(cutoff_manifest.get("decision_time"))
    if decision is None:
        decision = _parse_ts(cutoff_manifest.get("issue_yyyymm"))
    if decision is None:
        return errors  # no cutoff declared — skip enforcement

    for step in steps:
        prov = step.get("provenance") or {}
        t_avail = _parse_ts(prov.get("t_available") or prov.get("available_at"))
        if t_avail and t_avail > decision:
            errors.append(f"causal:future_data:{step.get('step_index')}")
        artifact_refs = step.get("artifact_refs") or []
        for ref in artifact_refs:
            if isinstance(ref, Mapping):
                t_ref = _parse_ts(ref.get("t_available") or ref.get("available_at"))
                if t_ref and t_ref > decision:
                    errors.append(f"causal:future_artifact_ref:{step.get('step_index')}")
    return errors


def build_cutoff_manifest(
    track: str,
    *,
    decision_time: Optional[str] = None,
    issue_yyyymm: Optional[str] = None,
    scenario_id: Optional[str] = None,
) -> dict:
    """Build cutoff manifest for pilot tracks FL-2, DR-OUT, TC-TRK."""
    manifest: dict = {"track": track, "scenario_id": scenario_id}
    if decision_time:
        manifest["decision_time"] = decision_time
    if issue_yyyymm:
        manifest["issue_yyyymm"] = issue_yyyymm
        if not decision_time:
            manifest["decision_time"] = f"{issue_yyyymm}-01T00:00:00Z"
    return manifest
