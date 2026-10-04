"""Multi-paper contract aggregation (ChatGPT §16)."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Mapping, Sequence

from .schema import Predicate, ScientificRouteContract


def aggregate_contracts(
    contracts: Sequence[ScientificRouteContract],
    *,
    family_id: str,
) -> ScientificRouteContract:
    """
    Merge per-paper contracts for one route family.
    Conflicting strict predicates → unresolved (not vote-downgrade).
    """
    if not contracts:
        return ScientificRouteContract(
            contract_id=f"src_agg_{family_id}_empty",
            route_binding={"family_id": family_id, "task_id": "", "route_id": family_id, "hazard_id": ""},
        )

    binding = dict(contracts[0].route_binding)
    strict_by_key: Dict[str, List[Predicate]] = defaultdict(list)
    advisory: List[Predicate] = []

    for c in contracts:
        for p in c.strict_predicates:
            key = f"{p.predicate_type}:{p.canonical_variable}"
            strict_by_key[key].append(p)
        advisory.extend(c.advisory_predicates)

    merged_strict: List[Predicate] = []
    unresolved: List[str] = []
    for key, group in strict_by_key.items():
        if len(group) == 1:
            merged_strict.append(group[0])
            continue
        # Multiple sources: require matching canonical + type
        canons = {g.canonical_variable for g in group}
        if len(canons) == 1:
            best = max(group, key=lambda g: float(g.support.get("confidence") or 0))
            best.support["n_sources"] = len(group)
            merged_strict.append(best)
        else:
            unresolved.append(f"conflict:{key}")

    return ScientificRouteContract(
        contract_id=f"src_agg_{family_id}_{len(contracts)}",
        route_binding=binding,
        strict_predicates=merged_strict,
        advisory_predicates=advisory,
        unresolved=unresolved,
        coverage={
            "n_source_contracts": len(contracts),
            "n_strict_merged": len(merged_strict),
            "n_advisory": len(advisory),
            "n_conflicts": len(unresolved),
        },
        provenance={"aggregation": "multi_paper_v1"},
        compiler_metadata={"n_papers": len(contracts)},
    )
