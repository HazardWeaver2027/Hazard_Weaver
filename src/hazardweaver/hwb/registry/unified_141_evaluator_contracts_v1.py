"""Build sealed unified @141 evaluator contracts from inventory (login-safe).

Does not scan ``runs/hwa``, does not run ``eval_*_cap``, and does not fold
``track=PFDF`` into MH-1 (DL-220). Gold CAP is resolved from inventory
fields only (no workdir, no TaskPack resolve).
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INVENTORY = ROOT / "benchmark/public/manifest.jsonl"
STRATUM_M_INVENTORY = ROOT / "benchmark/public/stratum_M.jsonl"

HEADLINE_N = 141
STRATUM_M_DENOM_141 = 96
PFDF_TRACK = "PFDF"

TRACK_ORDER: Tuple[str, ...] = (
    "DR-OUT",
    "E1-E3",
    "FL-2",
    "HW-MED",
    "L2",
    "MH-1",
    "MH-2",
    "MH-3",
    "MH-4",
    "TC-TRK",
    "WF-3",
)

EXPECTED_TRACK_N: Dict[str, int] = {
    "DR-OUT": 14,
    "E1-E3": 14,
    "L2": 14,
    "HW-MED": 17,
    "WF-3": 13,
    "TC-TRK": 6,
    "MH-1": 16,
    "MH-2": 12,
    "MH-3": 11,
    "MH-4": 12,
    "FL-2": 12,
}

FIXTURE_TASKPACK_IDS = frozenset(
    {
        "hwb_wf3_spread_fixture_v1",
        "hwb_mh3_compound_fixture_v1",
        "hwb_fl2_fixture_v1",
    }
)

ALTERNATIVE_ROUTE_POLICY = (
    "Unregistered routes that are not the inventory gold CAP fail headline DCA, "
    "even if the artifact is numerically close to gold. Multi-route acceptance "
    "is limited to the sealed eligible set."
)

PASS_RULE = (
    "PASS iff (no gold route defined and dca_result.valid) or "
    "(picked_capability == gold_capability and dca_result.valid)"
)

TOLERANCE_CONTRACT = "unified_benchmark_rescore_v2 / calibrate_tolerance"


def load_submit_inventory(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    inv = path or DEFAULT_INVENTORY
    return [json.loads(line) for line in inv.read_text(encoding="utf-8").splitlines() if line.strip()]


def is_headline_row(row: Mapping[str, Any]) -> bool:
    return str(row.get("track") or "").strip().upper() != PFDF_TRACK


def headline_rows(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return [dict(r) for r in rows if is_headline_row(r)]


def load_stratum_m_inventory(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    inv = path or STRATUM_M_INVENTORY
    return [json.loads(line) for line in inv.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_stratum_m_141(path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Stratum M rows aligned with headline @141 (DL-220 excludes ``track=PFDF``)."""
    return headline_rows(load_stratum_m_inventory(path))


def _cap_prefix(sid: str) -> str:
    text = str(sid or "").strip()
    if not text.startswith("CAP-") or text == "CAP-PLACEHOLDER":
        return ""
    return text.split("__", 1)[0].strip()


def _hcg_scenario_is_decoy(row: Mapping[str, Any], scenario_cap: str) -> bool:
    cap = str(scenario_cap or "").strip()
    if not cap:
        return False
    decoy = str(row.get("ablation_hcg_decoy_capability_id") or row.get("ablation_decoy_capability_id") or "").strip()
    return bool(decoy) and decoy == cap


def inventory_gold_capability_id(row: Mapping[str, Any]) -> str:
    """Gold CAP from inventory fields only (no workdir / TaskPack I/O)."""
    w3 = str(row.get("w3_gold_capability_id") or "").strip()
    if w3:
        return w3
    track = str(row.get("track") or "").upper()
    elig = str(row.get("route_eligibility") or "").strip()
    sid = str(row.get("scenario_id") or "").strip()
    if track == "E1-E3" and elig == "multi_eligible":
        scenario = _cap_prefix(sid) if sid.startswith("CAP-E1E3-") else ""
        if scenario:
            if _hcg_scenario_is_decoy(row, scenario):
                witness = str(row.get("ablation_witness_capability_id") or "").strip()
                return witness or scenario
            return scenario
    if elig == "multi_eligible":
        cap = _cap_prefix(sid)
        if cap:
            return cap
    if elig == "single_route":
        cap = _cap_prefix(sid)
        if cap:
            return cap
        allowed = [str(e).strip() for e in (row.get("allowed_edge_ids") or []) if str(e).strip()]
        if len(allowed) == 1:
            return allowed[0]
    return str(row.get("ablation_witness_capability_id") or "").strip()


def _tolerance_dict(row: Mapping[str, Any]) -> Dict[str, Any]:
    raw = row.get("unified_dca_tolerance")
    if isinstance(raw, Mapping) and (raw.get("metric") or "min_score" in raw or "max_abs_error" in raw):
        return dict(raw)
    return {}


def _metric_name(row: Mapping[str, Any], tol: Mapping[str, Any]) -> str:
    return str(tol.get("metric") or row.get("unified_canonical_metric") or "").strip()


def _reference_score(row: Mapping[str, Any], tol: Mapping[str, Any]) -> Optional[float]:
    for blob in (tol, row):
        if not isinstance(blob, Mapping):
            continue
        for key in ("reference_score", "unified_reference_score"):
            if blob.get(key) is not None:
                try:
                    return float(blob[key])
                except (TypeError, ValueError):
                    pass
    outputs = row.get("unified_dca_outputs")
    if isinstance(outputs, Mapping) and outputs.get("reference_score") is not None:
        try:
            return float(outputs["reference_score"])
        except (TypeError, ValueError):
            pass
    return None


def _threshold_fields(tol: Mapping[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"min_score": None, "max_abs_error": None}
    if "min_score" in tol:
        try:
            out["min_score"] = float(tol["min_score"])
        except (TypeError, ValueError):
            pass
    if "max_abs_error" in tol:
        try:
            out["max_abs_error"] = float(tol["max_abs_error"])
        except (TypeError, ValueError):
            pass
    return out


def gold_source_class(row: Mapping[str, Any], *, metric: str, has_threshold: bool) -> str:
    if not metric or not has_threshold:
        return "unknown"
    taskpack = str(row.get("taskpack_id") or "")
    if taskpack in FIXTURE_TASKPACK_IDS or "fixture" in taskpack.lower():
        return "fixture_or_proxy"
    return "hcg_scientific_replay"


def gold_dataset_note(row: Mapping[str, Any], source: str) -> str:
    track = str(row.get("track") or "")
    if source == "fixture_or_proxy":
        return "fixture_or_scalar_proxy"
    if source == "unknown":
        return "not_specified"
    if track == "FL-2":
        return "floodcast_official_product_via_hcg_replay"
    return "hcg_capability_scientific_replay"


def contract_for_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    tol = _tolerance_dict(row)
    metric = _metric_name(row, tol)
    thr = _threshold_fields(tol)
    has_threshold = thr["min_score"] is not None or thr["max_abs_error"] is not None
    gold_cap = inventory_gold_capability_id(row)
    source = gold_source_class(row, metric=metric, has_threshold=has_threshold)
    witnesses: List[str] = []
    w = str(row.get("ablation_witness_capability_id") or "").strip()
    if w:
        witnesses.append(w)
    decoy = str(row.get("ablation_decoy_capability_id") or "").strip()
    hcg_decoy = str(row.get("ablation_hcg_decoy_capability_id") or "").strip()
    return {
        "instance_id": str(row.get("instance_id") or ""),
        "track": str(row.get("track") or ""),
        "stratum": str(row.get("stratum") or ""),
        "scenario_id": str(row.get("scenario_id") or ""),
        "taskpack_id": str(row.get("taskpack_id") or ""),
        "route_eligibility": str(row.get("route_eligibility") or ""),
        "metric": metric or None,
        "tolerance_contract": str(tol.get("dca_tolerance_contract") or TOLERANCE_CONTRACT),
        "min_score": thr["min_score"],
        "max_abs_error": thr["max_abs_error"],
        "reference_score": _reference_score(row, tol),
        "gold_capability_id": gold_cap or None,
        "route_gold_required": bool(gold_cap),
        "accepted_witness_capability_ids": witnesses,
        "ablation_decoy_capability_id": decoy or None,
        "ablation_hcg_decoy_capability_id": hcg_decoy or None,
        "gold_source_class": source,
        "gold_dataset_note": gold_dataset_note(row, source),
        "spatial_tolerance": "not_specified",
        "temporal_tolerance": "not_specified",
        "alternative_route_policy": ALTERNATIVE_ROUTE_POLICY,
        "pass_rule": PASS_RULE,
        "headline_denominator": HEADLINE_N,
    }


def build_contracts(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    return [contract_for_row(r) for r in headline_rows(rows)]


def summarize_contracts(contracts: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    grouped: Dict[str, List[Mapping[str, Any]]] = defaultdict(list)
    for c in contracts:
        grouped[str(c.get("track") or "unknown")].append(c)
    by_track: Dict[str, Dict[str, Any]] = {}
    ordered = list(TRACK_ORDER) + sorted(t for t in grouped if t not in TRACK_ORDER)
    for track in ordered:
        cells = grouped.get(track) or []
        if not cells and track not in TRACK_ORDER:
            continue
        sources = Counter(str(c.get("gold_source_class") or "unknown") for c in cells)
        by_track[track] = {
            "n": len(cells),
            "expected_n": EXPECTED_TRACK_N.get(track),
            "metrics": dict(Counter(str(c.get("metric") or "missing") for c in cells)),
            "gold_source_class": dict(sources),
            "n_unknown": int(sources.get("unknown", 0)),
            "n_route_gold_defined": sum(1 for c in cells if c.get("route_gold_required")),
        }
    n_unknown = sum(1 for c in contracts if c.get("gold_source_class") == "unknown")
    missing_threshold = [
        str(c.get("instance_id"))
        for c in contracts
        if c.get("min_score") is None and c.get("max_abs_error") is None
    ]
    return {
        "n": len(contracts),
        "headline_n": HEADLINE_N,
        "n_unknown": n_unknown,
        "unknown_instance_ids": [
            str(c.get("instance_id")) for c in contracts if c.get("gold_source_class") == "unknown"
        ],
        "missing_threshold_instance_ids": missing_threshold,
        "by_track": by_track,
        "gold_source_class": dict(
            Counter(str(c.get("gold_source_class") or "unknown") for c in contracts)
        ),
        "metrics": dict(Counter(str(c.get("metric") or "missing") for c in contracts)),
        "pass_rule": PASS_RULE,
        "alternative_route_policy": ALTERNATIVE_ROUTE_POLICY,
        "independence_claim": (
            "solver_evaluator_separation_and_sealed_inventory_only; "
            "not_independent_domain_expert_labels"
        ),
    }


def build_from_inventory_path(path: Optional[Path] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    raw = load_submit_inventory(path)
    n_pfdf = sum(1 for r in raw if str(r.get("track") or "").upper() == PFDF_TRACK)
    contracts = build_contracts(raw)
    summary = summarize_contracts(contracts)
    summary["n_submit_rows"] = len(raw)
    summary["n_pfdf_excluded"] = n_pfdf
    return contracts, summary


CSV_FIELDS: Tuple[str, ...] = (
    "instance_id",
    "track",
    "stratum",
    "scenario_id",
    "taskpack_id",
    "metric",
    "min_score",
    "max_abs_error",
    "reference_score",
    "gold_capability_id",
    "route_gold_required",
    "gold_source_class",
    "gold_dataset_note",
)


def _write_csv(contracts: Iterable[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(CSV_FIELDS), extrasaction="ignore")
        writer.writeheader()
        for row in contracts:
            writer.writerow({k: row.get(k) for k in CSV_FIELDS})


def write_contract_artifacts(
    *,
    inventory_path: Optional[Path] = None,
    jsonl_path: Path,
    summary_path: Path,
    csv_path: Optional[Path] = None,
) -> Dict[str, Any]:
    contracts, summary = build_from_inventory_path(inventory_path)
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in contracts:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if csv_path is not None:
        _write_csv(contracts, csv_path)
        summary["csv_path"] = str(csv_path)
    summary["jsonl_path"] = str(jsonl_path)
    summary["summary_path"] = str(summary_path)
    return summary
