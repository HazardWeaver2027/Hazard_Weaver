"""HKC route_contract — conflict detection for multi-paper aggregation."""

from __future__ import annotations

from typing import List, Sequence

from .aggregation import aggregate_contracts
from .schema import ScientificRouteContract


def detect_strict_conflicts(contracts: Sequence[ScientificRouteContract]) -> List[str]:
    """Return unresolved conflict keys without building merged contract."""
    if not contracts:
        return []
    agg = aggregate_contracts(list(contracts), family_id="probe")
    return list(agg.unresolved or [])


def merge_family_contracts(
    contracts: Sequence[ScientificRouteContract],
    *,
    family_id: str,
) -> ScientificRouteContract:
    """Public merge entry — delegates to aggregate_contracts."""
    return aggregate_contracts(list(contracts), family_id=family_id)
