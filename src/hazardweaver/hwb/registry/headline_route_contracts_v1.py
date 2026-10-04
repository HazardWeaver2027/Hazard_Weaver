"""Solver-visible route contracts for headline baseline materialization (no answer keys)."""

from __future__ import annotations

from typing import Any, Dict

DR_OUT_ROUTE_CONTRACTS: Dict[str, Dict[str, Any]] = {
    "CAP-DROUT-01": {
        "metric": "sdo_skill",
        "modality": "expert_operational_synthesis",
        "description": (
            "CPC operational drought outlook synthesis via vendor_fetch "
            "(RF-EXPERT-OPERATIONAL-SYNTHESIS family)."
        ),
        "summary": "Expert operational CPC drought outlook synthesis",
    },
    "CAP-DROUT-02": {
        "metric": "sdo_skill",
        "modality": "persistence_baseline",
        "description": (
            "Persistence baseline from issue-time USDM category "
            "(RF-PERSISTENCE-BASELINE family)."
        ),
        "summary": "USDM persistence baseline forecast",
    },
    "CAP-DROUT-03": {
        "metric": "sdo_skill",
        "modality": "objective_statistical_tendency",
        "description": (
            "CPC objective drought tendency GeoTIFF statistical forecast "
            "(RF-OBJECTIVE-STATISTICAL-TENDEN family)."
        ),
        "summary": "Objective CPC statistical drought tendency",
    },
}

ROUTE_CONTRACTS_BY_PREFIX: Dict[str, Dict[str, Dict[str, Any]]] = {
    "CAP-DROUT-": DR_OUT_ROUTE_CONTRACTS,
}


def lookup_route_contract(edge_id: str) -> Dict[str, Any]:
    for prefix, table in ROUTE_CONTRACTS_BY_PREFIX.items():
        if edge_id.startswith(prefix) and edge_id in table:
            return dict(table[edge_id])
    return {}
