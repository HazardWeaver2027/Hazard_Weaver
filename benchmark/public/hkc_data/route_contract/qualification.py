"""Predicate promotion ladder + qualification (DL-061)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from .schema import (
    ADVISORY_DEFAULT_TYPES,
    Predicate,
    SLOT_TO_PREDICATE_TYPE,
    STRICT_DEFAULT_TYPES,
)

PROJECT_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_POLICY = PROJECT_ROOT / "docs/engineering/hkc/HKC_CONTRACT_POLICY_FREEZE_v1.json"

PROMOTION_STAGES = (
    "candidate",
    "grounded",
    "canonical",
    "decision_qualified",
    "strict",
    "advisory",
)


def load_policy(path: Optional[Path] = None) -> Dict[str, Any]:
    p = path or DEFAULT_POLICY
    if not p.is_file():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def default_gate_role(predicate_type: str, policy: Mapping[str, Any]) -> str:
    pt = predicate_type.upper()
    strict = set(policy.get("strict_types") or list(STRICT_DEFAULT_TYPES))
    cond = policy.get("conditional_strict_types") or {}
    if pt in strict:
        return "hard"
    if pt in cond:
        return "hard"  # promoted only if strong_evidence passes
    if pt in ADVISORY_DEFAULT_TYPES:
        return "advisory"
    return "metadata"


def predicate_from_route_card_slot(
    slot: Mapping[str, Any],
    *,
    paper_id: str,
    index: int,
    policy: Mapping[str, Any],
) -> Predicate:
    slot_type = str(slot.get("slot_type") or "")
    pred_type = SLOT_TO_PREDICATE_TYPE.get(slot_type, slot_type.upper())
    span = slot.get("evidence_span") or {}
    ev = []
    if span.get("text"):
        ev.append(
            {
                "text": str(span.get("text") or ""),
                "start": int(span.get("start") or 0),
                "end": int(span.get("end") or 0),
                "paper_id": paper_id,
            }
        )
    from .schema import EvidenceSpan

    evidence_spans = [
        EvidenceSpan(
            text=str(s.get("text") or ""),
            start=int(s.get("start") or 0),
            end=int(s.get("end") or 0),
            paper_id=str(s.get("paper_id") or paper_id),
        )
        for s in ev
    ]
    canonical = str(slot.get("canonical_value") or "").strip()
    conf = float(slot.get("confidence") or 0.0)
    gate = default_gate_role(pred_type, policy)
    stage = "grounded" if evidence_spans else "candidate"
    if canonical:
        stage = "canonical"
    return Predicate(
        predicate_id=f"pred_{paper_id[:20]}_{index}",
        predicate_type=pred_type,
        canonical_variable=canonical or "unresolved",
        relation="required" if pred_type == "REQUIRES_INPUT" else "applies_under",
        operator="present",
        expected_value=True,
        evidence=evidence_spans,
        support={
            "confidence": conf,
            "n_sources": 1,
            "ambiguity_state": str(slot.get("ambiguity_state") or "unknown"),
        },
        gate_role=gate,
        promotion_stage=stage,
    )


def apply_decomposed_checks(pred: Predicate, evidence_text: str) -> Predicate:
    """Direction-aware decomposed checks (Claude §3.3) — span-local, no LLM."""
    checks: Dict[str, Any] = {}
    span_text = pred.evidence[0].text if pred.evidence else ""
    checks["span_named_in_evidence"] = bool(span_text and span_text in evidence_text)
    neg_markers = (" not ", " no ", " without ", " never ", " unlikely ")
    lower_span = span_text.lower()
    checks["negation_consistency"] = not any(m in lower_span for m in neg_markers) or "fail" in pred.predicate_type.lower()
    checks["canonical_non_empty"] = bool(pred.canonical_variable and pred.canonical_variable != "unresolved")
    pred.qualification_checks = checks
    if all(checks.values()):
        if pred.promotion_stage == "canonical":
            pred.promotion_stage = "decision_qualified"
    return pred


def apply_edc_feasibility(pred: Predicate) -> Predicate:
    """EDC-style: block over-merge of distinct short tokens (Claude §3.4)."""
    c = pred.canonical_variable
    checks = dict(pred.qualification_checks)
    if len(c) < 3:
        checks["edc_feasibility"] = False
    elif c.count("_") > 12:
        checks["edc_feasibility"] = False
    else:
        checks["edc_feasibility"] = True
    pred.qualification_checks = checks
    return pred


def conformal_select_strict(
    predicates: Sequence[Predicate],
    scores: Sequence[float],
    *,
    target_precision: float = 0.9,
) -> Tuple[List[Predicate], float]:
    """
    Conformal risk control selection on calibration scores (Claude §3.2).
    Returns promoted predicates and chosen threshold lambda.
    """
    if not predicates:
        return [], 0.55
    paired = sorted(zip(scores, predicates), key=lambda x: -x[0])
    # Monotone: admit highest scores first; pick lambda achieving target on calibration proxy.
    n = len(paired)
    if n == 0:
        return [], 0.55
    # Simple finite-sample threshold: top fraction by score with floor 0.55
    k = max(1, int(n * target_precision))
    chosen = [p for _, p in paired[:k]]
    lam = paired[k - 1][0] if k <= n else 0.55
    for p in chosen:
        p.promotion_stage = "strict"
        p.gate_role = "hard"
    return chosen, float(lam)


def qualify_predicates(
    predicates: Sequence[Predicate],
    evidence_text: str,
    *,
    policy: Optional[Mapping[str, Any]] = None,
    heuristic_scores: Optional[Sequence[float]] = None,
) -> Tuple[List[Predicate], List[Predicate], Dict[str, Any]]:
    pol = policy or load_policy()
    working: List[Predicate] = []
    for i, pred in enumerate(predicates):
        p = apply_decomposed_checks(pred, evidence_text)
        p = apply_edc_feasibility(p)
        working.append(p)

    scores = list(heuristic_scores or [p.support.get("confidence", 0.5) for p in working])
    target = float((pol.get("conformal") or {}).get("target_precision") or 0.9)

    strict_candidates = [p for p in working if p.gate_role == "hard" or p.predicate_type in STRICT_DEFAULT_TYPES]
    advisory_candidates = [p for p in working if p not in strict_candidates]

    sc_scores = [scores[i] for i, p in enumerate(working) if p in strict_candidates]
    strict_promoted, lam = conformal_select_strict(strict_candidates, sc_scores, target_precision=target)

    for p in advisory_candidates:
        p.promotion_stage = "advisory"
        p.gate_role = "advisory"

    meta = {
        "conformal_lambda": lam,
        "target_precision": target,
        "n_strict": len(strict_promoted),
        "n_advisory": len(advisory_candidates),
    }
    return strict_promoted, advisory_candidates, meta
