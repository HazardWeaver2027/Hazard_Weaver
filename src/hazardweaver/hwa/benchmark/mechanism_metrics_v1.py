"""Round2 mechanism metrics — per-cell extraction from HWA workdirs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from hazardweaver.hwa.experiments.headline_inventory_runner_v1 import inventory_workdir_key
from hazardweaver.hwa.scientific_controller.reason_codes import ACapVerdict, ASciVerdict

_INAPPLICABLE_ASCI = {
    ASciVerdict.INAPPLICABLE_REGION.value,
    ASciVerdict.INAPPLICABLE_TEMPORAL_SCALE.value,
    ASciVerdict.INAPPLICABLE_HAZARD_CLASS.value,
    ASciVerdict.INAPPLICABLE_THEORY_MISMATCH.value,
    "REJECT",
}
_REACHABLE_ACAP = {
    ACapVerdict.REACHABLE.value,
    "REACHABLE",
    "OK",
}


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _committed_route(decisions: List[Mapping[str, Any]]) -> Optional[Dict[str, Any]]:
    commit_row: Optional[Dict[str, Any]] = None
    for row in decisions:
        if str(row.get("action") or "") == "commit":
            commit_row = dict(row)
    return commit_row


def _submitted_solve(workdir: Path) -> bool:
    meta = _read_json(workdir / "run_meta.json") or {}
    if meta.get("submitted"):
        return True
    if str(meta.get("exit_reason") or "") == "submitted":
        return True
    traj = _read_json(workdir / "trajectory.json") or {}
    if str(traj.get("terminal_action") or "") == "solve":
        return True
    exit_reason = str(meta.get("exit_reason") or "")
    if exit_reason in {"limit_steps", "exception", "max_wall_s", "timeout"}:
        return False
    ans = _read_json(workdir / "answer.json") or {}
    if str((ans.get("answer") or {}).get("action") or "") != "solve":
        return False
    dca = _read_json(workdir / "dca_result.json") or {}
    return bool(dca.get("valid")) and str(dca.get("outcome") or "") == "solve"


def _hkc_binding_source(commit: Optional[Mapping[str, Any]]) -> str:
    if not commit:
        return ""
    a_sci = commit.get("A_sci") or {}
    return str(commit.get("hkc_binding_source") or a_sci.get("source") or "")


def _hcg_binding_role(
    decisions: List[Mapping[str, Any]],
    commit: Optional[Mapping[str, Any]],
) -> str:
    """Last explicit HCG bind/commit role (pi_valid_recovery vs pi_near_miss)."""
    role = ""
    for row in decisions:
        if str(row.get("action") or "") != "bind":
            continue
        extra = row.get("extra") or {}
        role = str(row.get("hcg_binding_role") or extra.get("hcg_binding_role") or role)
    if commit:
        role = str(commit.get("hcg_binding_role") or role)
    return role


def _commit_type_compat_violation(decisions: List[Mapping[str, Any]]) -> bool:
    for row in decisions:
        if str(row.get("action") or "") != "commit":
            continue
        codes = [str(c) for c in (row.get("A_cap") or {}).get("codes") or []]
        if "hcg_untyped_skip_semantics" in codes:
            return True
        if any("compat" in c.lower() or "interface" in c.lower() for c in codes):
            return True
        rc = row.get("reachability_certificate") or {}
        for chk in rc.get("compatibility_checks") or []:
            if isinstance(chk, Mapping) and str(chk.get("status") or "").upper() in {
                "FAIL",
                "VIOLATION",
            }:
                return True
    return False


def _static_sibling_workdir(workdir: Path, inv: Mapping[str, Any]) -> Optional[Path]:
    """SRC dynamic pair: locate static-control sibling workdir."""
    variant = str(inv.get("src_dynamic_variant") or "")
    rq4_cell = str(inv.get("rq4_cell_id") or "")
    if variant != "route_change" or not rq4_cell.startswith("src_dynamic:"):
        return None
    static_cell = "src_static:" + rq4_cell.split(":", 1)[1]
    static_key = static_cell.replace(":", "__").replace("/", "_")
    sibling = workdir.parent / static_key
    return sibling if sibling.is_dir() else None


def extract_mechanism_metrics(workdir: Path) -> Dict[str, Any]:
    """Per-cell Round2 mechanism metrics."""
    decisions = _read_jsonl(workdir / "controller_decisions.jsonl")
    dca = _read_json(workdir / "dca_result.json") or {}
    inv = _read_json(workdir / "inventory_row.json") or {}
    commit = _committed_route(decisions)
    has_commit_binding = commit is not None
    submitted = _submitted_solve(workdir)

    a_sci = (commit or {}).get("A_sci") or {}
    a_cap = (commit or {}).get("A_cap") or {}
    verdict = str(a_sci.get("verdict") or "")
    binding = _hkc_binding_source(commit)

    override = inv.get("scientific_condition_override") or {}
    force_inapplicable = isinstance(override, Mapping) and bool(override.get("force_inapplicable"))

    invalid_route = bool(
        submitted
        and commit
        and (
            verdict in _INAPPLICABLE_ASCI
            or (
                force_inapplicable
                and verdict not in _INAPPLICABLE_ASCI
            )
        )
    )

    near_miss_cell = inv.get("hcg_near_miss_role") == "pi_near_miss"
    bind_role = _hcg_binding_role(decisions, commit)
    cap_codes = [str(c) for c in a_cap.get("codes") or []]

    near_miss_commit = bool(near_miss_cell and submitted and commit)

    valid_recovery_binding = bool(
        submitted and commit and near_miss_cell and bind_role == "pi_valid_recovery"
    )
    invalid_near_miss_binding = bool(
        submitted
        and commit
        and near_miss_cell
        and (
            bind_role == "pi_near_miss"
            or "hcg_untyped_skip_semantics" in cap_codes
        )
    )

    type_violation = False
    if near_miss_cell and submitted and commit:
        if bind_role == "pi_valid_recovery":
            type_violation = False
        elif bind_role == "pi_near_miss" or "hcg_untyped_skip_semantics" in cap_codes:
            type_violation = True
        elif _commit_type_compat_violation(decisions):
            type_violation = True

    false_reach = bool(
        submitted
        and str(a_cap.get("verdict") or "") in _REACHABLE_ACAP
        and bool(dca.get("contract_violated"))
    )

    correct_route_update: Optional[bool] = None
    if str(inv.get("src_dynamic_variant") or "") == "route_change":
        static_wd = _static_sibling_workdir(workdir, inv)
        if static_wd and commit:
            static_commit = _committed_route(_read_jsonl(static_wd / "controller_decisions.jsonl"))
            if static_commit:
                correct_route_update = str(commit.get("route_id") or "") != str(
                    static_commit.get("route_id") or ""
                )

    provenance_valid = not bool(dca.get("provenance_missing")) and not bool(
        (dca.get("codes") or []) and "provenance" in " ".join(str(c) for c in dca.get("codes") or []).lower()
    )

    return {
        "workdir": str(workdir),
        "instance_id": inv.get("instance_id"),
        "has_commit_binding": has_commit_binding,
        "submitted": submitted,
        "hkc_binding_source": binding or None,
        "scientifically_invalid_route_commitment": invalid_route,
        "type_violation": type_violation,
        "hcg_binding_role": bind_role or None,
        "valid_recovery_binding": valid_recovery_binding,
        "invalid_near_miss_binding": invalid_near_miss_binding,
        "near_miss_commit": near_miss_commit,
        "false_reachability": false_reach,
        "correct_route_update": correct_route_update,
        "provenance_valid": provenance_valid,
        "a_sci_verdict": verdict or None,
        "a_cap_verdict": str(a_cap.get("verdict") or "") or None,
        "contract_violated": bool(dca.get("contract_violated")),
        "dca_valid": bool(dca.get("valid")) if dca.get("valid") is not None else bool(dca.get("counted")),
        "inventory_workdir_key": inventory_workdir_key(inv) if inv else None,
    }


def aggregate_mechanism_metrics(cells: List[Mapping[str, Any]]) -> Dict[str, Any]:
    n = len(cells)
    if n == 0:
        return {"n": 0}

    def rate(key: str) -> float:
        vals = [c.get(key) for c in cells if c.get(key) is not None]
        if not vals:
            return 0.0
        return sum(1 for v in vals if v) / len(vals)

    commit_cells = [c for c in cells if c.get("has_commit_binding") or c.get("hkc_binding_source")]
    frozen_k = sum(1 for c in commit_cells if c.get("hkc_binding_source") == "frozen_k_hkc_v1")
    return {
        "n": n,
        "n_with_commit_binding": len(commit_cells),
        "scientifically_invalid_route_commitment_rate": rate("scientifically_invalid_route_commitment"),
        "type_violation_rate": rate("type_violation"),
        "valid_recovery_binding_rate": rate("valid_recovery_binding"),
        "invalid_near_miss_binding_rate": rate("invalid_near_miss_binding"),
        "near_miss_commit_rate": rate("near_miss_commit"),
        "false_reachability_rate": rate("false_reachability"),
        "correct_route_update_rate": rate("correct_route_update"),
        "provenance_valid_rate": rate("provenance_valid"),
        "frozen_k_binding_rate": frozen_k / len(commit_cells) if commit_cells else 0.0,
        "dca_valid_rate": rate("dca_valid"),
    }
