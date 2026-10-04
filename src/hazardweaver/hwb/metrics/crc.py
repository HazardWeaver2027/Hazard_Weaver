"""Counterfactual Route Consistency (CRC) — """

from __future__ import annotations

from typing import Dict, List, Mapping, Sequence


def counterfactual_route_consistency(
    paired_records: Sequence[Mapping[str, object]],
    *,
    route_key: str = "route_family",
) -> Dict[str, float]:
    """
    CRC = fraction of paired counterfactual variants where route family is preserved
    under scientifically equivalent perturbations.
    """
    if not paired_records:
        return {"crc": 0.0, "n_pairs": 0}

    groups: Dict[str, List[str]] = {}
    for rec in paired_records:
        pair_id = str(rec.get("pair_id") or rec.get("scenario_id") or "")
        route = str(rec.get(route_key) or rec.get("route_id") or "")
        groups.setdefault(pair_id, []).append(route)

    consistent = 0
    for routes in groups.values():
        if len(routes) >= 2 and len(set(routes)) == 1:
            consistent += 1
    n_pairs = len(groups)
    return {"crc": consistent / n_pairs if n_pairs else 0.0, "n_pairs": n_pairs}
