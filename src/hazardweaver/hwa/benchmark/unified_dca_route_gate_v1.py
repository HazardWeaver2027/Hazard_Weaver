"""Route-aware DCA validity for unified benchmark v1 (141 cells post DL-220).

Headline DCC must not treat wrong-route or ablation-decoy false-passes as solve.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from hazardweaver.hwb.registry.unified_benchmark_tolerance_v1 import is_unified_benchmark_row

ROUTE_CONTRACT_VIOLATION = "route_contract_violation"
ABLATION_DECOY_FALSE_PASS = "ablation_decoy_false_pass"
ABLATION_WITNESS_ON_OFF_ARM = "ablation_witness_on_off_arm"


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _cap_from_route_id(route_id: str) -> str:
    rid = str(route_id or "").strip()
    if rid.startswith("route:cap:"):
        return rid.split("route:cap:", 1)[-1]
    return rid


def _episode_artifacts_stale_for_pick(workdir: Path) -> bool:
    """Ignore pre-rerun or incomplete episodes when inferring picked capability."""
    meta = _read_json(workdir / "run_meta.json")
    if not meta:
        return not (workdir / "answer.json").is_file()
    exit_reason = str(meta.get("exit_reason") or "")
    if not meta.get("submitted") and exit_reason in {
        "limit_steps",
        "limit_wall",
        "parse_loop",
        "failed_clarify",
        "env_null_no_user",
        "exception",
        "timeout",
        "max_wall_s",
        "ablation_integrity_fail",
    }:
        return True
    try:
        from hazardweaver.hwa.benchmark.dca_episode_eligibility_v1 import is_dca_eligible

        if (workdir / "answer.json").is_file() and not is_dca_eligible(workdir):
            return True
    except Exception:  # noqa: BLE001
        pass
    return False


def picked_capability_id(workdir: Path) -> str:
    if _episode_artifacts_stale_for_pick(workdir):
        return ""
    meta = _read_json(workdir / "run_meta.json")
    cap = _cap_from_route_id(str(meta.get("route_id") or ""))
    if cap:
        return cap
    ans = _read_json(workdir / "answer.json")
    body = ans.get("answer") or {}
    cap = _cap_from_route_id(str(body.get("route_id") or ""))
    if cap:
        return cap
    cap = str(body.get("capability_id") or body.get("picked_capability_id") or "").strip()
    return cap


def expected_decoy_capability_id(row: Mapping[str, Any]) -> str:
    mech = str(row.get("ablation_mechanism") or "")
    if mech == "dual" and row.get("ablation_hcg_decoy_capability_id"):
        return str(row["ablation_hcg_decoy_capability_id"])
    return str(row.get("ablation_decoy_capability_id") or "")


def witness_capability_id(row: Mapping[str, Any]) -> str:
    """Gold witness route for ablation arms only.

    Headline inventory rows (``ablation_mechanism`` unset / ``none``) omit
    ``ablation_witness_capability_id``; routing is not part of their DCA contract.
    Never fall back to ``scenario_id`` (fire-year, case id, etc.) — that caused
    false ``route_contract_violation`` on correct CAP picks (DL-194).
    """
    return str(row.get("ablation_witness_capability_id") or "").strip()


def _gold_from_single_route_scenario_id(inventory_row: Mapping[str, Any]) -> str:
    """S-stratum single_route rows: scenario_id often encodes the gold CAP (before ``__``)."""
    if str(inventory_row.get("route_eligibility") or "") != "single_route":
        return ""
    sid = str(inventory_row.get("scenario_id") or "").strip()
    if not sid.startswith("CAP-") or sid == "CAP-PLACEHOLDER":
        return ""
    return sid.split("__", 1)[0].strip()


def e1e3_multi_scenario_capability_id(inventory_row: Mapping[str, Any]) -> str:
    """E1-E3 multi_eligible: scientific gold binds to scenario CAP, not witness lazy route."""
    if str(inventory_row.get("track") or "").upper() != "E1-E3":
        return ""
    if str(inventory_row.get("route_eligibility") or "") != "multi_eligible":
        return ""
    sid = str(inventory_row.get("scenario_id") or "").strip()
    if not sid.startswith("CAP-E1E3-") or sid == "CAP-PLACEHOLDER":
        return ""
    return sid.split("__", 1)[0].strip()


def unified_scenario_gold_capability_id(
    inventory_row: Mapping[str, Any],
    *,
    workdir: Optional[Path] = None,
) -> str:
    """Authoritative scenario gold for unified benchmark route discrimination.

    - multi_eligible + CAP-* scenario_id: gold is the scenario capability (M stratum).
    - single_route: gold is the sole solver-visible route (S stratum; case_N / CAP-*__tier).
    - E1-E3 retains dedicated helper for backwards-compatible tests.
    - W3-47 curator gold (``w3_gold_capability_id``) overrides ``scenario_id`` decoy variants.
    """
    w3_gold = str(inventory_row.get("w3_gold_capability_id") or "").strip()
    if w3_gold:
        return w3_gold
    scenario_gold = e1e3_multi_scenario_capability_id(inventory_row)
    if scenario_gold:
        return scenario_gold
    elig = str(inventory_row.get("route_eligibility") or "").strip()
    sid = str(inventory_row.get("scenario_id") or "").strip()
    if elig == "multi_eligible" and sid.startswith("CAP-") and sid != "CAP-PLACEHOLDER":
        return sid.split("__", 1)[0].strip()
    single_cap = _gold_from_single_route_scenario_id(inventory_row)
    if single_cap:
        return single_cap
    if str(inventory_row.get("track") or "").upper() == "MH-1" and elig == "single_route":
        from hazardweaver.hwa.scientific_controller.mh_unified_hkc_bindings_v1 import mh1_witness_gold_capability_id

        mh1_gold = mh1_witness_gold_capability_id(inventory_row)
        if mh1_gold:
            return mh1_gold
    if elig != "single_route":
        return ""
    taskpack: Dict[str, Any] = {}
    if workdir is not None:
        taskpack = _read_json(workdir / "resolved_taskpack.json")
    if not taskpack:
        try:
            from hazardweaver.hwb.run.eval_dca_submission_v1 import resolve_taskpack_for_inventory_row

            taskpack = resolve_taskpack_for_inventory_row(dict(inventory_row))
        except Exception:  # noqa: BLE001
            taskpack = {}
    if not taskpack:
        return ""
    inv_scenario = str(inventory_row.get("scenario_id") or "").strip()
    if inv_scenario:
        taskpack = dict(taskpack)
        sv = dict(taskpack.get("solver_view") or {})
        params = dict(sv.get("parameters") or {})
        params["scenario_id"] = inv_scenario
        sv["parameters"] = params
        # Materialized workdirs may leak full multi-route whitelists; gold is single-route.
        sv["allowed_edge_ids"] = []
        taskpack["solver_view"] = sv
    from hazardweaver.hwb.registry.solver_allowed_edges_v1 import resolve_solver_allowed_edge_ids

    edges = resolve_solver_allowed_edge_ids(taskpack, inventory_row=inventory_row)
    if len(edges) == 1:
        return str(edges[0]).strip()
    return ""


def gold_capability_id(
    inventory_row: Mapping[str, Any],
    workdir: Path,
) -> str:
    """Authoritative gold capability for unified DCA (witness or taskpack witness edge)."""
    if str(inventory_row.get("track") or "").upper() == "E1-E3":
        from hazardweaver.hwa.scientific_controller.e1e3_hkc_bindings_v1 import e1e3_unified_gold_capability_id

        scenario_gold = e1e3_unified_gold_capability_id(inventory_row)
        if scenario_gold:
            return scenario_gold
    scenario_gold = unified_scenario_gold_capability_id(inventory_row, workdir=workdir)
    if scenario_gold:
        return scenario_gold
    witness = witness_capability_id(inventory_row)
    if witness:
        return witness
    tp = _read_json(workdir / "resolved_taskpack.json")
    witnesses = (tp.get("reference_view") or {}).get("accepted_witnesses") or []
    if witnesses:
        edges = witnesses[0].get("edges") or []
        if edges:
            return str(edges[0]).strip()
        caps = witnesses[0].get("capability_ids") or []
        if caps:
            return str(caps[0]).strip()
    return _gold_from_single_route_scenario_id(inventory_row)


def picked_baseline_capability_id(workdir: Path) -> str:
    """Final route pick from faithful baseline workdirs."""
    sel = _read_json(workdir / "route_selection.json")
    edge = str(sel.get("selected_edge_id") or sel.get("selected") or "").strip()
    if edge:
        return edge
    rounds = sel.get("rounds") or []
    if rounds:
        edge = str(rounds[-1].get("selected_edge_id") or "").strip()
        if edge:
            return edge
    tr = _read_json(workdir / "trajectory.json")
    rs = tr.get("route_summary") or {}
    edge = str(rs.get("selected_edge_id") or "").strip()
    if edge:
        return edge
    fa = (tr.get("final_artifact") or {}).get("value") or {}
    return str(fa.get("capability_id") or fa.get("selected_route") or "").strip()


def unified_dca_valid(
    inventory_row: Mapping[str, Any],
    workdir: Path,
    *,
    picked_capability: str,
) -> Dict[str, Any]:
    """Unified DCA v2: gold route match (when defined) + dca_result.valid.

    Headline DCC must not treat wrong-route or ablation-decoy false-passes as solve.
    Denominator: unified inventory N=141 (post DL-220 PFDF removal).
    """
    dca = _read_json(workdir / "dca_result.json")
    raw_valid = bool(dca.get("valid"))
    gold = gold_capability_id(inventory_row, workdir)
    picked = str(picked_capability or "").strip()
    route_ok: Optional[bool]
    if gold:
        route_ok = bool(picked) and picked == gold
        unified = bool(route_ok and raw_valid)
    else:
        route_ok = None
        unified = raw_valid
    return {
        "unified_dca_valid": unified,
        "dca_valid_raw": raw_valid,
        "route_correct": route_ok,
        "gold_capability_id": gold or None,
        "picked_capability_id": picked or None,
    }


def baseline_unified_dca_valid(
    inventory_row: Mapping[str, Any],
    workdir: Path,
) -> Dict[str, Any]:
    """Unified DCA v2 for faithful baseline workdirs."""
    return unified_dca_valid(
        inventory_row,
        workdir,
        picked_capability=picked_baseline_capability_id(workdir),
    )


def hwa_unified_dca_valid(
    inventory_row: Mapping[str, Any],
    workdir: Path,
) -> Dict[str, Any]:
    """Unified DCA v2 for full_hwa workdirs (same contract as baselines)."""
    return unified_dca_valid(
        inventory_row,
        workdir,
        picked_capability=picked_capability_id(workdir),
    )


def route_gate_violation(
    *,
    inventory_row: Mapping[str, Any],
    workdir: Path,
    condition: str,
) -> Optional[str]:
    """Return violation code if route/condition pairing invalidates DCA, else None."""
    if not is_unified_benchmark_row(inventory_row):
        return None
    picked = picked_capability_id(workdir)
    if not picked:
        return None
    witness = witness_capability_id(inventory_row)
    decoy = expected_decoy_capability_id(inventory_row)
    cond = str(condition or "full_hwa").strip()
    if cond == "full_hwa":
        gold = gold_capability_id(inventory_row, workdir)
        if gold:
            if picked != gold:
                return ROUTE_CONTRACT_VIOLATION
            return None
        if witness and picked != witness:
            return ROUTE_CONTRACT_VIOLATION
        return None
    if cond == "hkc_off":
        if decoy and picked == decoy:
            return ABLATION_DECOY_FALSE_PASS
        if witness and picked == witness:
            return ABLATION_WITNESS_ON_OFF_ARM
        return None
    if cond == "hcg_untyped":
        hcg_decoy = str(inventory_row.get("ablation_hcg_decoy_capability_id") or decoy)
        if hcg_decoy and picked == hcg_decoy:
            return ABLATION_DECOY_FALSE_PASS
        if witness and picked == witness:
            return ABLATION_WITNESS_ON_OFF_ARM
        return None
    return None


def infer_dca_route_condition(
    workdir: Path,
    *,
    out_root_hint: str = "",
) -> str:
    """Infer ablation route arm for DCA gating (full_hwa | hkc_off | hcg_untyped)."""
    hint = str(out_root_hint or "").lower()
    if "hcg_untyped" in hint:
        return "hcg_untyped"
    if "hkc_off" in hint:
        return "hkc_off"
    meta = _read_json(workdir / "run_meta.json")
    if str(meta.get("dca_route_condition") or "") in {
        "full_hwa",
        "hkc_off",
        "hcg_untyped",
    }:
        return str(meta["dca_route_condition"])
    try:
        from hazardweaver.hwa.experiments.headline_ablation_modes_v1 import hcg_untyped

        if hcg_untyped():
            return "hcg_untyped"
    except ImportError:
        pass
    if str(meta.get("hkc_mode") or "").lower() == "off":
        return "hkc_off"
    return "full_hwa"


def apply_baseline_unified_route_gate_to_dca(
    dca_dict: Mapping[str, Any],
    workdir: Path,
    *,
    inventory_row: Mapping[str, Any],
) -> Dict[str, Any]:
    """Invalidate baseline DCA when picked route != unified scenario gold."""
    if not is_unified_benchmark_row(inventory_row):
        return dict(dca_dict)
    gate = baseline_unified_dca_valid(inventory_row, workdir)
    gold = str(gate.get("gold_capability_id") or "").strip()
    if not gold:
        return dict(dca_dict)
    out = dict(dca_dict)
    out["gold_capability_id"] = gold
    out["picked_capability_id"] = gate.get("picked_capability_id")
    out["route_correct"] = gate.get("route_correct")
    out["dca_valid_raw"] = bool(dca_dict.get("valid"))
    if gate.get("route_correct") is False:
        out["valid"] = False
        out["contract_violated"] = True
        out["counted"] = False
        out["dca_score"] = 0.0
        out["outcome"] = "invalid"
        out["reason_code"] = ROUTE_CONTRACT_VIOLATION
        out["route_gate_violation"] = ROUTE_CONTRACT_VIOLATION
        return out
    out.pop("route_gate_violation", None)
    return out


def apply_unified_route_gate_to_dca(
    dca_dict: Mapping[str, Any],
    workdir: Path,
    *,
    inventory_row: Mapping[str, Any],
    condition: str = "full_hwa",
) -> Dict[str, Any]:
    """Invalidate DCA when route/condition contract is broken."""
    if dca_dict.get("skipped") or dca_dict.get("ok") is False:
        return dict(dca_dict)
    try:
        from hazardweaver.hwa.benchmark.dca_episode_eligibility_v1 import is_dca_eligible

        if not is_dca_eligible(workdir):
            meta = _read_json(workdir / "run_meta.json")
            out = dict(dca_dict)
            out["valid"] = False
            out["contract_violated"] = False
            out["counted"] = False
            out["dca_score"] = 0.0
            out["outcome"] = "unknown"
            out["reason_code"] = str(meta.get("exit_reason") or "incomplete_episode")
            out["episode_ineligible"] = True
            out.pop("route_gate_violation", None)
            return out
    except ImportError:
        pass
    violation = route_gate_violation(
        inventory_row=inventory_row,
        workdir=workdir,
        condition=condition,
    )
    if not violation:
        out = dict(dca_dict)
        out.pop("route_gate_violation", None)
        return out
    out = dict(dca_dict)
    out["valid"] = False
    out["contract_violated"] = True
    out["counted"] = False
    out["dca_score"] = 0.0
    out["outcome"] = "invalid"
    out["reason_code"] = violation
    out["route_gate_violation"] = violation
    out["picked_capability_id"] = picked_capability_id(workdir)
    return out
