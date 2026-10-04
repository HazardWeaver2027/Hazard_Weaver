"""Benchmark-side Π_adm^B — independent of HWA/HKC runtime certificates."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Set

ROOT = Path(__file__).resolve().parents[3]
POLICY_PATH = ROOT / "hwb" / "registry" / "pi_adm_policy_v1.json"


@dataclass
class AdmissibilityResult:
    admissible: bool
    reason_code: str  # ADMIT | REJECT | UNKNOWN
    blocked_capabilities: List[str]
    messages: List[str]

    def pi_adm_empty(self) -> bool:
        return not self.admissible


def _load_policy() -> Dict[str, Any]:
    if POLICY_PATH.is_file():
        return json.loads(POLICY_PATH.read_text(encoding="utf-8"))
    return {
        "schema_version": "pi_adm_policy_v1",
        "forbidden_when_abstain_required": ["direct_submit_without_inspection"],
        "asci_unknown_triggers": ["missing_label_contract", "ambiguous_goal"],
    }


def compute_pi_adm_b(
    taskpack: Mapping[str, Any],
    submission: Mapping[str, Any],
    *,
    policy: Optional[Mapping[str, Any]] = None,
) -> AdmissibilityResult:
    """
    Π_adm^B(q, s_k) — benchmark-side admissibility.

    Does NOT read HWA/HKC solver certificates (Kickoff #5).
    Uses frozen capability manifest + hidden A_sci policy JSON only.
    """
    pol = dict(policy or _load_policy())
    ref = taskpack.get("reference_view") or {}
    expected = str(ref.get("expected_action") or "solve")
    messages: List[str] = []
    blocked: List[str] = []

    terminal = str(submission.get("terminal_action") or "")
    meta = submission.get("abstention_metadata") or {}
    hwa_verdict = str(meta.get("asci_verdict") or submission.get("asci_verdict") or "")

    if expected == "abstain":
        cert = ref.get("no_path_certificate") or {}
        reason = str(cert.get("reason") or "missing_label_contract")
        if terminal == "abstain":
            return AdmissibilityResult(True, "ADMIT", [], ["correct_abstention"])
        if terminal == "clarify" or "UNKNOWN" in hwa_verdict.upper():
            return AdmissibilityResult(False, "UNKNOWN", blocked, [f"clarify_unknown:{reason}"])
        if reason in pol.get("asci_unknown_triggers", []):
            return AdmissibilityResult(False, "UNKNOWN", blocked, [f"should_abstain:{reason}"])
        return AdmissibilityResult(False, "REJECT", blocked, [f"should_abstain:{reason}"])

    if expected == "clarify":
        if terminal == "clarify":
            return AdmissibilityResult(True, "ADMIT", [], ["correct_clarification"])
        if "UNKNOWN" in hwa_verdict.upper():
            return AdmissibilityResult(False, "UNKNOWN", blocked, ["clarify_unknown"])
        return AdmissibilityResult(False, "REJECT", [], ["should_clarify"])

    allowed: Set[str] = {str(e) for e in (taskpack.get("solver_view") or {}).get("allowed_edge_ids") or []}
    steps = submission.get("steps") or []
    for step in steps:
        cid = step.get("capability_id")
        if cid and str(cid) not in allowed:
            blocked.append(str(cid))
    if blocked:
        return AdmissibilityResult(False, "REJECT", blocked, ["capability_not_admissible"])
    return AdmissibilityResult(True, "ADMIT", [], [])
