"""E_q ∧ V_q dual gate — Valid Task Completion with abstention/clarify/unknown outcomes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional

from hazardweaver.hwb.evaluators.abstention_outcomes import CompletionOutcome, classify_completion
from hazardweaver.hwb.evaluators.admissibility_benchmark import AdmissibilityResult, compute_pi_adm_b
from hazardweaver.hwb.evaluators.artifact_checker import ArtifactEqResult, check_artifact_eq
from hazardweaver.hwb.evaluators.trajectory_checker import TrajectoryVqResult, check_trajectory_vq


@dataclass
class DualGateResult:
    valid: bool
    outcome: str  # solve | abstain | clarify | unknown | invalid
    eq: Optional[ArtifactEqResult]
    vq: Optional[TrajectoryVqResult]
    adm: Optional[AdmissibilityResult]
    taskpack_id: str
    reporting_bucket: str = "invalid"
    reason_code: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "valid": self.valid,
            "outcome": self.outcome,
            "reporting_bucket": self.reporting_bucket,
            "reason_code": self.reason_code,
            "taskpack_id": self.taskpack_id,
            "E_q": None if self.eq is None else self.eq.eq,
            "V_q": None if self.vq is None else self.vq.vq,
            "Pi_adm_B_empty": None if self.adm is None else self.adm.pi_adm_empty(),
            "adm_reason_code": None if self.adm is None else self.adm.reason_code,
            "metric_name": None if self.eq is None else self.eq.metric_name,
            "score": None if self.eq is None else self.eq.score,
            "matched_witness_id": None if self.vq is None else self.vq.matched_witness_id,
            "eq_errors": [] if self.eq is None else self.eq.errors,
            "vq_errors": [] if self.vq is None else self.vq.errors,
        }


def evaluate_submission(
    taskpack: Mapping[str, Any],
    trajectory: Mapping[str, Any],
    *,
    require_both: bool = True,
    reference_view: Optional[Mapping[str, Any]] = None,
    vq_mode: str = "property",
    scenario: Optional[Mapping[str, Any]] = None,
) -> DualGateResult:
    """
    Completion outcomes (§4.10):
    - solve → E_q(y) ∧ V_q(τ)
    - abstain → correct terminal abstention
    - clarify → correct clarification
    - unknown → information-insufficient (excluded from AbsRec@K / over-abstention)
    """
    taskpack_id = str(taskpack.get("taskpack_id") or "")
    expected = str((taskpack.get("reference_view") or {}).get("expected_action") or "solve")
    adm = compute_pi_adm_b(taskpack, trajectory)

    if adm.reason_code == "UNKNOWN" and expected in ("abstain", "clarify"):
        classification = classify_completion(
            taskpack, trajectory, adm_reason_code="UNKNOWN"
        )
        return DualGateResult(
            valid=classification.valid,
            outcome=classification.outcome.value,
            eq=None,
            vq=None,
            adm=adm,
            taskpack_id=taskpack_id,
            reporting_bucket=classification.reporting_bucket,
            reason_code="UNKNOWN",
        )

    if expected == "abstain":
        classification = classify_completion(taskpack, trajectory, adm_reason_code=adm.reason_code)
        return DualGateResult(
            valid=classification.valid,
            outcome=classification.outcome.value,
            eq=None,
            vq=None,
            adm=adm,
            taskpack_id=taskpack_id,
            reporting_bucket=classification.reporting_bucket,
            reason_code=adm.reason_code,
        )

    if expected == "clarify":
        classification = classify_completion(taskpack, trajectory, adm_reason_code=adm.reason_code)
        return DualGateResult(
            valid=classification.valid,
            outcome=classification.outcome.value,
            eq=None,
            vq=None,
            adm=adm,
            taskpack_id=taskpack_id,
            reporting_bucket=classification.reporting_bucket,
            reason_code=adm.reason_code,
        )

    final_artifact = trajectory.get("final_artifact") or {}
    eq = check_artifact_eq(taskpack, final_artifact, reference_view=reference_view)
    vq = check_trajectory_vq(taskpack, trajectory, vq_mode=vq_mode, scenario=scenario)
    valid = eq.eq and vq.vq if require_both else (eq.eq or vq.vq)
    classification = classify_completion(
        taskpack,
        trajectory,
        adm_reason_code=adm.reason_code,
        eq_valid=eq.eq,
        vq_valid=vq.vq,
    )
    return DualGateResult(
        valid=valid,
        outcome=classification.outcome.value,
        eq=eq,
        vq=vq,
        adm=adm,
        taskpack_id=taskpack_id,
        reporting_bucket=classification.reporting_bucket,
        reason_code=adm.reason_code,
    )
