"""HWA bridge: Scientific Route Cards → A_sci(π, q; K) inputs."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from hazardweaver.hwa.scientific_controller.admissibility import evaluate_A_sci, is_admissible
from hazardweaver.hwa.scientific_controller.reason_codes import ASciVerdict


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@dataclass
class RouteCardIndex:
    """In-memory index of compiled Route Cards."""

    by_family_item: Dict[str, Dict[str, Dict[str, Any]]] = field(default_factory=dict)
    by_family_paper: Dict[str, Dict[str, List[Dict[str, Any]]]] = field(default_factory=dict)

    @classmethod
    def load(cls, path: Path) -> "RouteCardIndex":
        idx = cls()
        paths = [path]
        if path.is_dir():
            paths = sorted(path.glob("*.jsonl"))
        for p in paths:
            idx._ingest_file(p)
        return idx

    @classmethod
    def load_many(cls, paths: Sequence[Path]) -> "RouteCardIndex":
        idx = cls()
        for p in paths:
            if p.is_file():
                idx._ingest_file(p)
        return idx

    def _ingest_file(self, path: Path) -> None:
        for row in _read_jsonl(path):
            binding = row.get("route_binding") or {}
            family_id = str(binding.get("family_id") or binding.get("route_id") or "")
            item_id = str(row.get("item_id") or "")
            paper_id = str(row.get("paper_id") or "")
            if not family_id:
                continue
            self.by_family_item.setdefault(family_id, {})[item_id] = row
            self.by_family_paper.setdefault(family_id, {}).setdefault(paper_id, []).append(row)

    def get(self, family_id: str, item_id: str) -> Optional[Dict[str, Any]]:
        return self.by_family_item.get(family_id, {}).get(item_id)

    def for_paper(self, family_id: str, paper_id: str) -> List[Dict[str, Any]]:
        return list(self.by_family_paper.get(family_id, {}).get(paper_id, []))


def _slot_lists(card: Mapping[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    keys = (
        "required_inputs",
        "assumptions",
        "applicability_conditions",
        "valid_scales",
        "mechanisms",
        "uses_equations",
        "calibration_thresholds",
        "uncertainty",
        "failure_conditions",
        "provenance_slots",
    )
    return {k: list(card.get(k) or []) for k in keys}


def route_card_to_theory_overlay(
    route_card: Mapping[str, Any],
    *,
    family_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Project admitted Route Card slots into HWA route-candidate theory fields."""
    binding = route_card.get("route_binding") or {}
    fid = family_id or str(binding.get("family_id") or "")
    slots = _slot_lists(route_card)
    refs: List[str] = []
    constraints: List[str] = []
    blocked_regions: List[str] = []

    for bucket in slots.values():
        for slot in bucket:
            prov = slot.get("provenance") or {}
            span = slot.get("evidence_span") or {}
            paper = str(prov.get("paper_id") or route_card.get("paper_id") or "")
            text = str(span.get("text") or "")[:120]
            slot_type = str(slot.get("slot_type") or "")
            if paper and text:
                refs.append(f"hkc:{paper}:{slot_type}:{text[:60]}")
            canonical = str(slot.get("canonical_value") or "").strip()
            if canonical and slot_type in {"RequiresInput", "Assumption"}:
                constraints.append(f"require:{canonical}")

    for fc in slots.get("failure_conditions") or []:
        canonical = str(fc.get("canonical_value") or "").lower()
        if canonical:
            blocked_regions.append(canonical)

    return {
        "route_card_id": route_card.get("route_card_id"),
        "family_id": fid,
        "theory_alignment_refs": refs,
        "theory_constraints": sorted(set(constraints)),
        "blocked_regions": sorted(set(blocked_regions)),
        "hkc_provenance": dict(route_card.get("provenance") or {}),
        "requires_theory": bool(refs or constraints),
        "route_card_admitted_slots": {
            k: len(v) for k, v in slots.items() if v
        },
    }


def evaluate_A_sci_with_route_card(
    route: Mapping[str, Any],
    task: Mapping[str, Any],
    route_card: Optional[Mapping[str, Any]],
    *,
    theory_arm: str = "verified",
) -> Dict[str, Any]:
    """A_sci using compiled Route Card knowledge when available."""
    if theory_arm == "off":
        return evaluate_A_sci(route, task, theory_arm=theory_arm)

    if route_card is None:
        return evaluate_A_sci(route, task, theory_arm=theory_arm)

    overlay = route_card_to_theory_overlay(route_card)
    enriched = dict(route)
    enriched.update(
        {
            "theory_alignment_refs": overlay.get("theory_alignment_refs") or [],
            "theory_constraints": overlay.get("theory_constraints") or [],
            "blocked_regions": list(route.get("blocked_regions") or []) + overlay.get("blocked_regions") or [],
            "requires_theory": True,
            "route_card_id": overlay.get("route_card_id"),
            "hkc_family_id": overlay.get("family_id"),
        }
    )

    slots = _slot_lists(route_card)
    admitted_total = sum(len(v) for v in slots.values())
    if admitted_total == 0:
        return {
            "verdict": ASciVerdict.UNKNOWN_PENDING_THEORY.value,
            "codes": ["route_card_empty"],
            "refs": [],
            "route_card_id": overlay.get("route_card_id"),
        }

    base = evaluate_A_sci(enriched, task, theory_arm=theory_arm)
    base["refs"] = overlay.get("theory_alignment_refs") or base.get("refs") or []
    base["route_card_id"] = overlay.get("route_card_id")
    base["hkc_admitted_slot_counts"] = overlay.get("route_card_admitted_slots")
    return base


def annotate_route_with_route_card(
    route: Mapping[str, Any],
    task: Mapping[str, Any],
    state: Any,
    *,
    route_card_index: Optional[RouteCardIndex] = None,
    family_id: Optional[str] = None,
    item_id: Optional[str] = None,
    graph: Any = None,
) -> Dict[str, Any]:
    """Annotate route with A_sci from Route Card + standard A_cap."""
    from hazardweaver.hwa.scientific_controller.admissibility import annotate_route

    fid = family_id or str(route.get("family_id") or route.get("hkc_family_id") or "")
    card = None
    if route_card_index and fid and item_id:
        card = route_card_index.get(fid, item_id)

    out = annotate_route(route, task, state, graph=graph)
    if card is not None:
        out["A_sci"] = evaluate_A_sci_with_route_card(
            out,
            task,
            card,
            theory_arm=getattr(state, "theory_arm", "verified"),
        )
        out["admissible"] = is_admissible(out["A_sci"], out["A_cap"])
        out["route_card_id"] = card.get("route_card_id")
    return out


def load_route_card_index(path: Path) -> RouteCardIndex:
    return RouteCardIndex.load(path)


def _fold_constraint(text: str) -> str:
    folded = str(text or "").strip().lower().replace(" ", "_")
    return folded[:80] if folded else "unresolved"


def grounded_assertions_to_raw_overlay(
    assertions: Sequence[Any],
    *,
    paper_id: str,
    item_id: str,
    family_id: str,
) -> Dict[str, Any]:
    """Project high-recall grounded slots into HWA fields (no admission / no G3)."""
    refs: List[str] = []
    constraints: List[str] = []
    blocked_regions: List[str] = []

    for assertion in assertions:
        if hasattr(assertion, "slot_type"):
            slot_type = str(assertion.slot_type or "")
            span_text = str(assertion.evidence_span or "")
        else:
            slot_type = str((assertion or {}).get("slot_type") or "")
            span_text = str((assertion or {}).get("evidence_span") or "")

        if paper_id and span_text:
            refs.append(f"hkc:raw:{paper_id}:{slot_type}:{span_text[:60]}")
        if span_text and slot_type in {"RequiresInput", "Assumption"}:
            constraints.append(f"require:{_fold_constraint(span_text)}")
        if span_text and slot_type == "FailureCondition":
            blocked_regions.append(_fold_constraint(span_text))

    return {
        "raw_retrieval_id": f"raw_{family_id}_{item_id}",
        "family_id": family_id,
        "item_id": item_id,
        "paper_id": paper_id,
        "theory_alignment_refs": refs,
        "theory_constraints": sorted(set(constraints)),
        "blocked_regions": sorted(set(blocked_regions)),
        "requires_theory": bool(refs or constraints),
        "n_grounded_slots": len(assertions),
    }


def _import_eskc_raw_pipeline() -> tuple[Any, Any, Any]:
    import sys

    eskc_root = Path(__file__).resolve().parents[3] / "benchmark/public/hkc_data"
    eskc_root_str = str(eskc_root)
    if eskc_root_str not in sys.path:
        sys.path.insert(0, eskc_root_str)
    from eskc.route_card_compiler import load_frozen_slot_extractor
    from eskc.slot_grounding import ground_slot_assertions
    from eskc.stage0_retrieval import retrieve_candidate_graph

    return retrieve_candidate_graph, load_frozen_slot_extractor, ground_slot_assertions


def build_raw_retrieval_overlay_from_evidence(
    evidence: str,
    *,
    paper_id: str,
    item_id: str,
    family_id: str,
) -> Dict[str, Any]:
    """Frozen Stage-0 retrieval + slot_extractor_frozen_v1 + grounding (no admission)."""
    retrieve_candidate_graph, load_frozen_slot_extractor, ground_slot_assertions = _import_eskc_raw_pipeline()
    bundle = retrieve_candidate_graph(evidence)
    extractor = load_frozen_slot_extractor()
    assertions = extractor.extract(evidence, route_context={"route_id": family_id})
    grounded = ground_slot_assertions(bundle.graph, evidence, assertions)
    return grounded_assertions_to_raw_overlay(
        grounded,
        paper_id=paper_id,
        item_id=item_id,
        family_id=family_id,
    )


def build_proxy_raw_overlay_from_gold_row(
    row: Mapping[str, Any],
    *,
    family_id: str,
) -> Dict[str, Any]:
    """CPU-only raw overlay from human slot labels (login-safe eval proxy)."""
    evidence = str(row.get("evidence_text") or "")
    paper_id = str(row.get("paper_id") or "")
    item_id = str(row.get("item_id") or "")
    assertions: List[Dict[str, Any]] = []
    for slot_type in row.get("multi_label_slots") or []:
        assertions.append(
            {
                "slot_type": str(slot_type),
                "evidence_span": evidence[:120] if evidence else str(slot_type),
            }
        )
    if not assertions and evidence:
        assertions.append({"slot_type": "RequiresInput", "evidence_span": evidence[:120]})
    return grounded_assertions_to_raw_overlay(
        assertions,
        paper_id=paper_id,
        item_id=item_id,
        family_id=family_id,
    )


@dataclass
class RawRetrievalIndex:
    """Per-item raw retrieval overlays (high-recall proposer, no admission)."""

    by_family_item: Dict[str, Dict[str, Dict[str, Any]]] = field(default_factory=dict)

    @classmethod
    def build_from_gold(
        cls,
        gold_rows: Sequence[Mapping[str, Any]],
        *,
        family_id: str,
        proxy: bool = False,
    ) -> "RawRetrievalIndex":
        idx = cls()
        seen: set[str] = set()
        for row in gold_rows:
            item_id = str(row.get("item_id") or "")
            if not item_id or item_id in seen:
                continue
            seen.add(item_id)
            evidence = str(row.get("evidence_text") or "")
            paper_id = str(row.get("paper_id") or "")
            if not evidence.strip():
                continue
            if proxy or not row.get("multi_label_slots"):
                overlay = build_proxy_raw_overlay_from_gold_row(row, family_id=family_id)
            else:
                try:
                    overlay = build_raw_retrieval_overlay_from_evidence(
                        evidence,
                        paper_id=paper_id,
                        item_id=item_id,
                        family_id=family_id,
                    )
                except Exception:
                    overlay = build_proxy_raw_overlay_from_gold_row(row, family_id=family_id)
            idx.by_family_item.setdefault(family_id, {})[item_id] = overlay
        return idx

    @classmethod
    def build_from_gold_proxy(
        cls,
        gold_rows: Sequence[Mapping[str, Any]],
        *,
        family_id: str,
    ) -> "RawRetrievalIndex":
        return cls.build_from_gold(gold_rows, family_id=family_id, proxy=True)

    def get(self, family_id: str, item_id: str) -> Optional[Dict[str, Any]]:
        return self.by_family_item.get(family_id, {}).get(item_id)


def evaluate_A_sci_with_raw_retrieval(
    route: Mapping[str, Any],
    task: Mapping[str, Any],
    raw_overlay: Optional[Mapping[str, Any]],
    *,
    theory_arm: str = "raw",
) -> Dict[str, Any]:
    """A_sci using raw retrieval overlay (no Route Card admission)."""
    if theory_arm == "off" or raw_overlay is None:
        return evaluate_A_sci(route, task, theory_arm=theory_arm if theory_arm != "off" else "off")

    enriched = dict(route)
    enriched.update(
        {
            "theory_alignment_refs": list(raw_overlay.get("theory_alignment_refs") or []),
            "theory_constraints": list(raw_overlay.get("theory_constraints") or []),
            "blocked_regions": list(route.get("blocked_regions") or []) + list(raw_overlay.get("blocked_regions") or []),
            "requires_theory": True,
            "raw_retrieval_id": raw_overlay.get("raw_retrieval_id"),
            "hkc_family_id": raw_overlay.get("family_id"),
        }
    )
    if not enriched.get("theory_alignment_refs"):
        return {
            "verdict": ASciVerdict.UNKNOWN_PENDING_THEORY.value,
            "codes": ["raw_retrieval_empty"],
            "refs": [],
            "raw_retrieval_id": raw_overlay.get("raw_retrieval_id"),
        }

    base = evaluate_A_sci(enriched, task, theory_arm=theory_arm)
    base["refs"] = enriched.get("theory_alignment_refs") or []
    base["raw_retrieval_id"] = raw_overlay.get("raw_retrieval_id")
    base["n_grounded_slots"] = raw_overlay.get("n_grounded_slots")
    return base
