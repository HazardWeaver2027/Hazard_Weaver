"""Abstention outcome taxonomy — §4.10 UNKNOWN as third reporting column."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Mapping, Optional


class CompletionOutcome(str, Enum):
    SOLVE = "solve"
    ABSTAIN = "abstain"
    CLARIFY = "clarify"
    UNKNOWN = "unknown"  # §4.10 — neither correct nor incorrect abstain
    INVALID = "invalid"


@dataclass
class OutcomeClassification:
    outcome: CompletionOutcome
    valid: bool
    reason_code: str  # ADMIT | REJECT | UNKNOWN
    reporting_bucket: str  # solve | abstain | clarify | unknown | invalid


def classify_completion(
    taskpack: Mapping[str, Any],
    trajectory: Mapping[str, Any],
    *,
    adm_reason_code: str,
    eq_valid: Optional[bool] = None,
    vq_valid: Optional[bool] = None,
) -> OutcomeClassification:
    """
    Map dual_gate inputs to §4.10 reporting buckets.

    UNKNOWN: information-insufficient (clarify trigger) — excluded from AbsRec@K
    numerator/denominator and from over-abstention rate.
    """
    expected = str((taskpack.get("reference_view") or {}).get("expected_action") or "solve")
    terminal = str(trajectory.get("terminal_action") or "")
    meta = trajectory.get("abstention_metadata") or {}
    hwa_verdict = str(meta.get("asci_verdict") or trajectory.get("asci_verdict") or "")
    reason = str(meta.get("reason_code") or adm_reason_code or "")

    if reason == "UNKNOWN" or "UNKNOWN" in hwa_verdict.upper():
        return OutcomeClassification(
            outcome=CompletionOutcome.UNKNOWN,
            valid=terminal == "clarify",
            reason_code="UNKNOWN",
            reporting_bucket="unknown",
        )

    if expected == "abstain":
        if terminal == "abstain":
            return OutcomeClassification(
                outcome=CompletionOutcome.ABSTAIN,
                valid=True,
                reason_code="ADMIT",
                reporting_bucket="abstain",
            )
        if terminal == "clarify":
            return OutcomeClassification(
                outcome=CompletionOutcome.UNKNOWN,
                valid=True,
                reason_code="UNKNOWN",
                reporting_bucket="unknown",
            )
        return OutcomeClassification(
            outcome=CompletionOutcome.INVALID,
            valid=False,
            reason_code="REJECT",
            reporting_bucket="invalid",
        )

    if expected == "clarify":
        if terminal == "clarify":
            return OutcomeClassification(
                outcome=CompletionOutcome.CLARIFY,
                valid=True,
                reason_code="ADMIT",
                reporting_bucket="clarify",
            )
        return OutcomeClassification(
            outcome=CompletionOutcome.INVALID,
            valid=False,
            reason_code="REJECT",
            reporting_bucket="invalid",
        )

    solve_valid = bool(eq_valid and vq_valid)
    return OutcomeClassification(
        outcome=CompletionOutcome.SOLVE if solve_valid else CompletionOutcome.INVALID,
        valid=solve_valid,
        reason_code="ADMIT" if solve_valid else "REJECT",
        reporting_bucket="solve" if solve_valid else "invalid",
    )


def absrec_counts(records: list[Dict[str, Any]]) -> Dict[str, Any]:
    """AbsRec@K + over-abstention with UNKNOWN excluded from abstain tallies (§4.10)."""
    eligible = [r for r in records if r.get("reporting_bucket") != "unknown"]
    expected_abstain = [r.get("expected_action") == "abstain" for r in eligible]
    predicted_abstain = [r.get("terminal_action") == "abstain" for r in eligible]
    tp = sum(1 for e, p in zip(expected_abstain, predicted_abstain) if e and p)
    fn = sum(1 for e, p in zip(expected_abstain, predicted_abstain) if e and not p)
    fp = sum(1 for e, p in zip(expected_abstain, predicted_abstain) if not e and p)
    unknown_n = sum(1 for r in records if r.get("reporting_bucket") == "unknown")
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    over_rate = fp / len(eligible) if eligible else 0.0
    return {
        "absrec_at_k": recall,
        "over_abstention_rate": over_rate,
        "unknown_clarify_count": unknown_n,
        "n_eligible": len(eligible),
        "n_unknown_excluded": unknown_n,
    }
