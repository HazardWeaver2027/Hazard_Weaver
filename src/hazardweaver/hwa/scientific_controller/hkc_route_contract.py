"""HWA bridge: Scientific Route Contract → A_sci single source (DL-061)."""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

# route_contract lives under benchmark/public/hkc_data
_HKC_ROOT = Path(__file__).resolve().parents[3] / "benchmark/public/hkc_data"
if str(_HKC_ROOT) not in sys.path:
    sys.path.insert(0, str(_HKC_ROOT))

from route_contract.admission import certificate_to_hwa_verdict, evaluate_contract
from route_contract.converter import compile_contract_from_route_card
from route_contract.schema import ScientificRouteContract

from hazardweaver.hwa.scientific_controller.admissibility import evaluate_A_sci, is_admissible
from hazardweaver.hwa.scientific_controller.hkc_route_card import RouteCardIndex, evaluate_A_sci_with_route_card


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


@dataclass
class RouteContractIndex:
    """Index compiled Scientific Route Contracts by (family_id, item_id) and paper_id."""

    by_family_item: Dict[str, Dict[str, Dict[str, Any]]] = field(default_factory=dict)
    by_family_paper: Dict[str, Dict[str, Dict[str, Any]]] = field(default_factory=dict)
    _headline_family_rep: Dict[str, Dict[str, Any]] = field(default_factory=dict, repr=False)

    @classmethod
    def load(cls, path: Path) -> "RouteContractIndex":
        idx = cls()
        for row in _read_jsonl(path):
            binding = row.get("route_binding") or {}
            family_id = str(binding.get("family_id") or "")
            meta = row.get("compiler_metadata") or {}
            src = str(meta.get("source_route_card_id") or row.get("contract_id") or "")
            item_id = str(row.get("item_id") or "")
            if not item_id and "_bulk_" in src:
                item_id = src.split("_")[-1] if src else ""
            if not family_id:
                continue
            if not item_id:
                item_id = str((row.get("provenance") or {}).get("evidence_item_id") or row.get("contract_id"))
            paper_id = str(row.get("paper_id") or "")
            idx.by_family_item.setdefault(family_id, {})[item_id] = row
            if paper_id:
                idx.by_family_paper.setdefault(family_id, {})[paper_id] = row
        return idx

    def get(self, family_id: str, item_id: str) -> Optional[Dict[str, Any]]:
        return self.by_family_item.get(family_id, {}).get(item_id)

    def get_by_paper(self, family_id: str, paper_id: str) -> Optional[Dict[str, Any]]:
        return self.by_family_paper.get(family_id, {}).get(paper_id)

    def headline_family_contract(self, family_id: str) -> Optional[Dict[str, Any]]:
        """Representative frozen contract for headline g6 routes (no item_id/paper_id on route)."""
        fid = str(family_id or "").strip()
        if not fid:
            return None
        if fid in self._headline_family_rep:
            return self._headline_family_rep[fid]
        items = self.by_family_item.get(fid) or {}
        if not items:
            return None
        rep = min(items.values(), key=lambda r: len(r.get("strict_predicates") or []))
        self._headline_family_rep[fid] = rep
        return rep


def load_route_contract_index(path: Path) -> RouteContractIndex:
    return RouteContractIndex.load(path)


def contract_dict_to_model(row: Mapping[str, Any]) -> ScientificRouteContract:
    from route_contract.schema import Predicate, EvidenceSpan

    strict = []
    for p in row.get("strict_predicates") or []:
        ev = [
            EvidenceSpan(
                text=str(e.get("text") or ""),
                start=int(e.get("start") or 0),
                end=int(e.get("end") or 0),
                paper_id=str(e.get("paper_id") or ""),
            )
            for e in p.get("evidence") or []
        ]
        strict.append(
            Predicate(
                predicate_id=str(p.get("predicate_id") or ""),
                predicate_type=str(p.get("predicate_type") or ""),
                canonical_variable=str(p.get("canonical_variable") or ""),
                relation=str(p.get("relation") or ""),
                operator=str(p.get("operator") or ""),
                evidence=ev,
                support=dict(p.get("support") or {}),
                gate_role=str(p.get("gate_role") or "hard"),
                promotion_stage=str(p.get("promotion_stage") or ""),
            )
        )
    advisory = []
    for p in row.get("advisory_predicates") or []:
        advisory.append(
            Predicate(
                predicate_id=str(p.get("predicate_id") or ""),
                predicate_type=str(p.get("predicate_type") or ""),
                canonical_variable=str(p.get("canonical_variable") or ""),
                gate_role="advisory",
            )
        )
    return ScientificRouteContract(
        contract_id=str(row.get("contract_id") or ""),
        route_binding=dict(row.get("route_binding") or {}),
        strict_predicates=strict,
        advisory_predicates=advisory,
        unresolved=list(row.get("unresolved") or []),
        provenance=dict(row.get("provenance") or {}),
        compiler_metadata=dict(row.get("compiler_metadata") or {}),
    )


def evaluate_A_sci_with_contract(
    route: Mapping[str, Any],
    task: Mapping[str, Any],
    contract_row: Optional[Mapping[str, Any]],
    *,
    theory_arm: str = "verified",
) -> Dict[str, Any]:
    """A_sci from Scientific Route Contract + certificate (single source)."""
    if theory_arm == "off":
        return evaluate_A_sci(route, task, theory_arm=theory_arm)
    if contract_row is None:
        return evaluate_A_sci(route, task, theory_arm=theory_arm)

    contract = contract_dict_to_model(contract_row)
    cert = evaluate_contract(contract, task)
    out = certificate_to_hwa_verdict(cert)
    out["family_id"] = cert.family_id
    out["route_id"] = cert.route_id
    return out


def evaluate_A_sci_with_src_pipeline(
    route: Mapping[str, Any],
    task: Mapping[str, Any],
    route_card: Optional[Mapping[str, Any]],
    *,
    evidence_text: str = "",
    theory_arm: str = "src_conformal_hkc",
) -> Dict[str, Any]:
    """Fourth RQ2 arm: RouteCard → Contract → Certificate (G-HKC-1)."""
    if theory_arm == "off" or route_card is None:
        return evaluate_A_sci(route, task, theory_arm="off")
    contract = compile_contract_from_route_card(route_card, evidence_text)
    cert = evaluate_contract(contract, task)
    return certificate_to_hwa_verdict(cert)


def annotate_route_with_contract(
    route: Mapping[str, Any],
    task: Mapping[str, Any],
    state: Any,
    *,
    contract_index: Optional[RouteContractIndex] = None,
    route_card_index: Optional[RouteCardIndex] = None,
    family_id: Optional[str] = None,
    item_id: Optional[str] = None,
    graph: Any = None,
) -> Dict[str, Any]:
    """Prefer contract certificate; fallback to legacy route card overlay."""
    from hazardweaver.hwa.scientific_controller.admissibility import annotate_route

    fid = family_id or str(route.get("family_id") or "")
    out = annotate_route(route, task, state, graph=graph)

    contract_row = None
    if contract_index and fid and item_id:
        contract_row = contract_index.get(fid, item_id)

    if contract_row is not None:
        out["A_sci"] = evaluate_A_sci_with_contract(
            out,
            task,
            contract_row,
            theory_arm=getattr(state, "theory_arm", "verified"),
        )
    elif route_card_index and fid and item_id:
        card = route_card_index.get(fid, item_id)
        out["A_sci"] = evaluate_A_sci_with_route_card(
            out,
            task,
            card,
            theory_arm=getattr(state, "theory_arm", "verified"),
        )

    out["admissible"] = is_admissible(out.get("A_sci") or {}, out.get("A_cap") or {})
    return out
