"""Solver-visible HCG capability contracts for baseline route materialization (no answer keys)."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CAPABILITIES_PATH = ROOT / "docs/final_four/HCG/manifests/capabilities.jsonl"

# Never expose benchmark scores, gold routing, or evaluator thresholds to the solver.
_FORBIDDEN_CONTRACT_KEYS = frozenset(
    {
        "reported_performance",
        "reference_score",
        "gold",
        "oracle",
        "tolerance",
        "min_score",
        "max_abs_error",
        "outputs",
        "witness",
        "checkpoint_test_leakage_risk",
        "evidence_ids",
    }
)

_SOLVER_VISIBLE_MANIFEST_KEYS = (
    "capability_id",
    "name",
    "track_id",
    "taskpack_id",
    "canonical_output_schema_id",
    "anchor_id",
    "route_family_id",
    "route_family",
    "scientific_assumptions",
    "inputs",
    "output",
    "tier",
    "modality",
    "acquisition_priority",
    "artifact_or_repo",
    "hazardweaver_compatibility",
    "license",
    "inference_compute",
    "training_compute",
    "training_required",
    "data_footprint",
)


def _metric_for_capability(capability_id: str) -> str:
    try:
        from hazardweaver.hwa.route_controller.route_metric_contract_v1 import capability_tolerance_metric

        return str(capability_tolerance_metric(capability_id) or "")
    except ImportError:
        return ""


@lru_cache(maxsize=1)
def load_hcg_capability_index(
    capabilities_path: Optional[str] = None,
) -> Dict[str, Dict[str, Any]]:
    path = Path(capabilities_path) if capabilities_path else DEFAULT_CAPABILITIES_PATH
    if not path.is_file():
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        row = json.loads(line)
        cap_id = str(row.get("capability_id") or "").strip()
        if cap_id:
            out[cap_id] = dict(row)
    return out


def solver_visible_manifest_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    """Strip HCG manifest row to solver-visible capability description (no scores)."""
    cap_id = str(row.get("capability_id") or "")
    contract: Dict[str, Any] = {}
    for key in _SOLVER_VISIBLE_MANIFEST_KEYS:
        val = row.get(key)
        if val is None or val == "" or val == []:
            continue
        contract[key] = val
    metric = _metric_for_capability(cap_id)
    if metric:
        contract["metric"] = metric
    name = str(row.get("name") or "").strip()
    family = str(row.get("route_family") or row.get("route_family_id") or "").strip()
    parts = [p for p in (name, family) if p]
    contract["summary"] = " — ".join(parts) if parts else cap_id
    assumptions = str(row.get("scientific_assumptions") or "").strip()
    inputs = str(row.get("inputs") or "").strip()
    output = str(row.get("output") or "").strip()
    desc_bits = [b for b in (assumptions, f"inputs: {inputs}" if inputs else "", f"output: {output}" if output else "") if b]
    if desc_bits:
        contract["description"] = "; ".join(desc_bits)
    return contract


def lookup_hcg_solver_route_contract(
    edge_id: str,
    *,
    capabilities_path: Optional[str] = None,
) -> Dict[str, Any]:
    row = load_hcg_capability_index(capabilities_path).get(str(edge_id))
    if not row:
        return {}
    return solver_visible_manifest_row(row)


def merge_solver_route_contracts(*sources: Mapping[str, Any]) -> Dict[str, Any]:
    """Merge contract dicts left→right; later sources override earlier non-empty fields."""
    out: Dict[str, Any] = {}
    for src in sources:
        if not isinstance(src, Mapping):
            continue
        for key, val in src.items():
            if key in _FORBIDDEN_CONTRACT_KEYS:
                continue
            if val is None or val == "" or val == []:
                continue
            out[key] = val
    return out


def route_contract_is_sparse(contract: Mapping[str, Any], edge_id: str) -> bool:
    """True when contract carries no meaningful capability description beyond the id."""
    if not contract:
        return True
    summary = str(contract.get("summary") or "").strip()
    if summary and summary != edge_id:
        return False
    for key in ("description", "route_family", "name", "scientific_assumptions", "inputs", "output"):
        if str(contract.get(key) or "").strip():
            return False
    return True


def format_route_catalog_line(edge_id: str, contract: Mapping[str, Any]) -> str:
    """Single-line route description for LLM prompts."""
    if not contract:
        return f"- {edge_id}"
    parts = [
        str(contract.get("name") or contract.get("summary") or edge_id),
        str(contract.get("route_family") or contract.get("route_family_id") or ""),
        str(contract.get("metric") or contract.get("metric_name") or ""),
        str(contract.get("scientific_assumptions") or ""),
        str(contract.get("inputs") or ""),
        str(contract.get("output") or ""),
    ]
    body = " | ".join(p.strip() for p in parts if p and str(p).strip())
    return f"- {edge_id}: {body}" if body else f"- {edge_id}"


def build_allowed_capability_catalog(
    allowed_edge_ids: list[str],
    *,
    contracts_by_edge: Optional[Mapping[str, Mapping[str, Any]]] = None,
    capabilities_path: Optional[str] = None,
) -> list[Dict[str, Any]]:
    """Catalog rows for allowed capabilities only (written beside routes.json)."""
    contracts_by_edge = contracts_by_edge or {}
    rows: list[Dict[str, Any]] = []
    for edge_id in allowed_edge_ids:
        merged = merge_solver_route_contracts(
            lookup_hcg_solver_route_contract(edge_id, capabilities_path=capabilities_path),
            contracts_by_edge.get(edge_id) or {},
        )
        if merged:
            rows.append({"edge_id": edge_id, "contract": merged})
        else:
            rows.append({"edge_id": edge_id, "contract": {"summary": edge_id}})
    return rows
