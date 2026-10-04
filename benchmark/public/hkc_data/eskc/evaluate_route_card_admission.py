"""Paper-disjoint Route Card admission evaluation (DL-052 frozen gates)."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .route_card_schema import (
    ADMISSION_PRECISION_STRETCH,
    ADMISSION_PRECISION_THRESHOLD,
    EVIDENCE_RECALL_THRESHOLD,
    PROVENANCE_CORRECTNESS_THRESHOLD,
    REQUIRED_SLOT_RECALL_THRESHOLD,
    RouteCard,
)
from .slot_extractor import HeuristicSlotExtractor, SklearnSlotExtractor
from .slot_grounding import ground_slot_assertions
from .slot_schema import SLOT_FAMILIES, normalize_slot_type
from .slot_verifier import verify_slot_assertion
from .stage0_retrieval import retrieve_candidate_graph


def _gold_slots(row: Mapping[str, Any]) -> set[str]:
    return {
        normalize_slot_type(str(s))
        for s in (row.get("multi_label_slots") or [])
        if normalize_slot_type(str(s)) in SLOT_FAMILIES
    }


def _provenance_ok(evidence: str, span_text: str, start: int, end: int) -> bool:
    if not span_text or span_text not in evidence:
        return False
    expected_start = evidence.index(span_text)
    expected_end = expected_start + len(span_text)
    return int(start) == expected_start and int(end) == expected_end


def evaluate_route_card_admission(
    rows: Sequence[Mapping[str, Any]],
    cards: Sequence[RouteCard],
    *,
    slot_model: Optional[SklearnSlotExtractor] = None,
    admission_threshold: float = 0.55,
    ablation: Optional[str] = None,
    v1_scores: Optional[Mapping[str, Mapping[str, Any]]] = None,
    require_v1: bool = False,
    slot_denylist: Optional[frozenset[str]] = None,
) -> Dict[str, Any]:
    deny = slot_denylist or frozenset()
    extractor = slot_model or HeuristicSlotExtractor()
    card_by_item = {c.item_id: c for c in cards}

    raw_tp = raw_fp = raw_fn = 0
    admit_tp = admit_fp = 0
    gold_total = 0
    evidence_hits = 0
    provenance_ok = 0
    provenance_total = 0
    items_with_gold = 0
    items_with_admitted_correct = 0
    per_slot_admit_tp: Counter = Counter()
    per_slot_admit_fp: Counter = Counter()
    item_rows: List[Dict[str, Any]] = []

    for row in rows:
        evidence = str(row.get("evidence_text") or "")
        gold = _gold_slots(row)
        if not gold or not evidence.strip():
            continue
        items_with_gold += 1
        gold_total += len(gold)
        item_id = str(row.get("item_id") or "")

        bundle = retrieve_candidate_graph(evidence)
        if bundle.n_clusters > 0:
            evidence_hits += 1

        assertions = extractor.extract(evidence)
        pred_slots = {a.slot_type for a in assertions}
        raw_tp += len(gold & pred_slots)
        raw_fp += len(pred_slots - gold)
        raw_fn += len(gold - pred_slots)

        grounded = ground_slot_assertions(bundle.graph, evidence, assertions)
        admitted_slots: List[str] = []
        for assertion in grounded:
            score_key = f"{item_id}::{assertion.slot_type}"
            v1_score = (v1_scores or {}).get(score_key) if v1_scores else None
            if require_v1 and not v1_score and ablation != "heuristic_v0_only":
                continue
            vr = verify_slot_assertion(
                assertion,
                evidence,
                admission_threshold=admission_threshold,
                v1_score=v1_score,
                ablation=ablation,
            )
            in_gold = assertion.slot_type in gold
            if assertion.slot_type in deny:
                continue
            if vr.admitted:
                admitted_slots.append(assertion.slot_type)
                if in_gold:
                    admit_tp += 1
                    per_slot_admit_tp[assertion.slot_type] += 1
                else:
                    admit_fp += 1
                    per_slot_admit_fp[assertion.slot_type] += 1

        admitted_set = set(admitted_slots)
        if gold & admitted_set:
            items_with_admitted_correct += 1

        card = card_by_item.get(item_id)
        if card:
            for slot_rec in card.all_admitted_slots():
                provenance_total += 1
                span = slot_rec.evidence_span
                if _provenance_ok(evidence, span.text, span.start, span.end):
                    provenance_ok += 1

        item_rows.append(
            {
                "item_id": item_id,
                "slot_id": row.get("slot_id"),
                "paper_id": row.get("paper_id"),
                "gold_slots": sorted(gold),
                "raw_pred_slots": sorted(pred_slots),
                "admitted_slots": sorted(admitted_set),
                "route_card_id": card.route_card_id if card else None,
                "n_route_card_admitted": len(card.all_admitted_slots()) if card else 0,
            }
        )

    raw_prec = raw_tp / (raw_tp + raw_fp) if (raw_tp + raw_fp) else 0.0
    raw_rec = raw_tp / (raw_tp + raw_fn) if (raw_tp + raw_fn) else 0.0
    admitted_prec = admit_tp / (admit_tp + admit_fp) if (admit_tp + admit_fp) else 0.0
    admitted_cov = admit_tp / gold_total if gold_total else 0.0
    item_cov = items_with_admitted_correct / items_with_gold if items_with_gold else 0.0
    evidence_recall = evidence_hits / items_with_gold if items_with_gold else 0.0
    provenance_corr = provenance_ok / provenance_total if provenance_total else 1.0

    gates = {
        "evidence_recall_pass": evidence_recall >= EVIDENCE_RECALL_THRESHOLD,
        "required_slot_recall_pass": raw_rec >= REQUIRED_SLOT_RECALL_THRESHOLD,
        "admitted_precision_pass": admitted_prec >= ADMISSION_PRECISION_THRESHOLD,
        "admitted_precision_stretch_pass": admitted_prec >= ADMISSION_PRECISION_STRETCH,
        "provenance_correctness_pass": provenance_corr >= PROVENANCE_CORRECTNESS_THRESHOLD,
    }
    headline_pass = all(
        [
            gates["evidence_recall_pass"],
            gates["required_slot_recall_pass"],
            gates["admitted_precision_pass"],
            gates["provenance_correctness_pass"],
        ]
    )

    return {
        "metric_contract": "route_card_admission_eval_v1",
        "evaluation_mode": "paper_disjoint_slot_gold",
        "verifier_mode": "v1_scores" if v1_scores and not ablation else (ablation or "default"),
        "admission_threshold": admission_threshold,
        "slot_denylist": sorted(deny),
        "population": {"n_items": items_with_gold, "gold_slot_total": gold_total},
        "metrics": {
            "evidence_recall": evidence_recall,
            "required_slot_recall": raw_rec,
            "required_slot_precision": raw_prec,
            "admitted_slot_precision": admitted_prec,
            "admitted_slot_coverage": admitted_cov,
            "item_admitted_coverage": item_cov,
            "provenance_correctness": provenance_corr,
        },
        "thresholds": {
            "evidence_recall": EVIDENCE_RECALL_THRESHOLD,
            "required_slot_recall": REQUIRED_SLOT_RECALL_THRESHOLD,
            "admitted_slot_precision": ADMISSION_PRECISION_THRESHOLD,
            "admitted_slot_precision_stretch": ADMISSION_PRECISION_STRETCH,
            "provenance_correctness": PROVENANCE_CORRECTNESS_THRESHOLD,
        },
        "gates": gates,
        "headline_pass": headline_pass,
        "per_slot_admit_tp": dict(per_slot_admit_tp),
        "per_slot_admit_fp": dict(per_slot_admit_fp),
        "items": item_rows,
    }
