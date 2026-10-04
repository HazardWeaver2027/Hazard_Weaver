"""Aligned E_q / V_q / unified DCA layers for HWA unified benchmark workdirs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

RESCORE_CONTRACT = "hwa_unified_dual_gate_v1"


def _read_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def eval_report_has_aligned_dual_gate(eval_report: Mapping[str, Any]) -> bool:
    """True when eval_report was written by the unified dual-gate sidecar rescore."""
    if str(eval_report.get("rescore_contract") or "") != RESCORE_CONTRACT:
        return False
    dg = eval_report.get("dual_gate") or {}
    return dg.get("E_q") is not None and dg.get("V_q") is not None


def layers_from_eval_report(eval_report: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    if not eval_report_has_aligned_dual_gate(eval_report):
        return None
    dg = dict(eval_report.get("dual_gate") or {})
    eq_block = eval_report.get("E_q") if isinstance(eval_report.get("E_q"), Mapping) else {}
    return {
        "eq_pass": bool(dg.get("E_q")),
        "vq_pass": bool(dg.get("V_q")),
        "dual_gate_valid": bool(dg.get("valid")),
        "dual_gate_outcome": dg.get("outcome"),
        "dual_gate": dg,
        "E_q_block": dict(eq_block),
        "source": "eval_report",
        "error": None,
    }


def evaluate_hwa_workdir_dual_gate_layers(
    workdir: Path,
    *,
    inventory_row: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Replay bound dual-gate with the same binding path as headline DCA rescore."""
    if inventory_row is None:
        inv = _read_json(workdir / "inventory_row.json")
    else:
        inv = dict(inventory_row)
    out: Dict[str, Any] = {
        "eq_pass": None,
        "vq_pass": None,
        "dual_gate_valid": None,
        "dual_gate_outcome": None,
        "dual_gate": None,
        "E_q_block": None,
        "source": "replay",
        "error": None,
    }
    if not (workdir / "answer.json").is_file():
        out["error"] = "missing_answer"
        return out
    try:
        from hazardweaver.hwb.bridge.hwa_workdir import load_hwa_submission
        from hazardweaver.hwb.evaluators.artifact_checker import check_artifact_eq
        from hazardweaver.hwb.evaluators.dca_scorer import evaluate_bound_dual_gate_submission
        from hazardweaver.hwb.run.eval_dca_submission_v1 import resolve_taskpack_for_inventory_row

        sub = load_hwa_submission(workdir)
        tp = resolve_taskpack_for_inventory_row(inv)
        dg = evaluate_bound_dual_gate_submission(
            tp,
            sub,
            inventory_row=inv,
            workdir=workdir,
        )
        eq = check_artifact_eq(tp, sub.get("final_artifact") or {})
        out["eq_pass"] = bool(dg.eq.eq) if dg.eq is not None else False
        out["vq_pass"] = bool(dg.vq.vq) if dg.vq is not None else False
        out["dual_gate_valid"] = bool(dg.valid)
        out["dual_gate_outcome"] = dg.outcome
        out["dual_gate"] = dg.to_dict()
        out["E_q_block"] = {
            "eq": eq.eq,
            "metric_name": eq.metric_name,
            "score": eq.score,
            "threshold_met": eq.threshold_met,
            "errors": eq.errors,
        }
    except Exception as exc:  # noqa: BLE001
        out["error"] = f"{type(exc).__name__}:{exc}"
    return out


def load_hwa_workdir_dual_gate_layers(
    workdir: Path,
    *,
    inventory_row: Optional[Mapping[str, Any]] = None,
    replay_if_missing: bool = True,
) -> Dict[str, Any]:
    """Prefer aligned eval_report sidecar; optionally replay bound dual-gate."""
    eval_rep = _read_json(workdir / "eval_report.json")
    cached = layers_from_eval_report(eval_rep)
    if cached is not None:
        return cached
    if not replay_if_missing:
        return {
            "eq_pass": None,
            "vq_pass": None,
            "dual_gate_valid": None,
            "dual_gate_outcome": None,
            "dual_gate": None,
            "E_q_block": None,
            "source": "missing",
            "error": "no_aligned_eval_report",
        }
    return evaluate_hwa_workdir_dual_gate_layers(workdir, inventory_row=inventory_row)


def build_aligned_eval_report(
    workdir: Path,
    *,
    inventory_row: Optional[Mapping[str, Any]] = None,
    out_root_hint: str = "",
) -> Dict[str, Any]:
    """Build eval_report.json payload for unified @141 dual-gate sidecar rescore."""
    if inventory_row is None:
        inv = _read_json(workdir / "inventory_row.json")
    else:
        inv = dict(inventory_row)
    layers = evaluate_hwa_workdir_dual_gate_layers(workdir, inventory_row=inv)
    dca = _read_json(workdir / "dca_result.json")
    return {
        "taskpack_id": inv.get("taskpack_id") or dca.get("taskpack_id"),
        "instance_id": inv.get("instance_id"),
        "workdir": str(workdir),
        "out_root_hint": out_root_hint,
        "rescore_contract": RESCORE_CONTRACT,
        "dual_gate": layers.get("dual_gate") or {},
        "E_q": layers.get("E_q_block") or {},
        "V_q_errors": (
            [] if not layers.get("dual_gate") else (layers["dual_gate"].get("vq_errors") or [])
        ),
        "replay_error": layers.get("error"),
        "dca_result_valid": dca.get("valid"),
    }


def write_aligned_eval_report(
    workdir: Path,
    *,
    inventory_row: Optional[Mapping[str, Any]] = None,
    out_root_hint: str = "",
) -> Dict[str, Any]:
    """Persist aligned eval_report.json; return payload + whether file changed."""
    report = build_aligned_eval_report(
        workdir,
        inventory_row=inventory_row,
        out_root_hint=out_root_hint,
    )
    path = workdir / "eval_report.json"
    old = _read_json(path) if path.is_file() else {}
    changed = report != old
    if changed:
        path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report["changed"] = changed
    return report


def route_gate_pass(
    inventory_row: Mapping[str, Any],
    workdir: Path,
) -> Optional[bool]:
    from hazardweaver.hwa.benchmark.unified_dca_route_gate_v1 import (
        gold_capability_id,
        picked_capability_id,
    )

    gold = gold_capability_id(inventory_row, workdir) or ""
    pick = picked_capability_id(workdir) or ""
    if not gold:
        return True
    return bool(pick) and pick == gold


def unified_layers_for_cell(
    inventory_row: Mapping[str, Any],
    workdir: Path,
    *,
    replay_if_missing: bool = True,
) -> Dict[str, Any]:
    """Per-cell headline layers: route gate + E_q + V_q + unified DCA (replay-aligned)."""
    from hazardweaver.hwa.benchmark.unified_dca_route_gate_v1 import hwa_unified_dca_valid

    dg = load_hwa_workdir_dual_gate_layers(
        workdir,
        inventory_row=inventory_row,
        replay_if_missing=replay_if_missing,
    )
    sealed = hwa_unified_dca_valid(inventory_row, workdir)
    eq_pass = dg.get("eq_pass")
    vq_pass = dg.get("vq_pass")
    route_ok = route_gate_pass(inventory_row, workdir)
    dual_solve = (
        bool(eq_pass and vq_pass)
        if eq_pass is not None and vq_pass is not None
        else None
    )
    unified_replay: Optional[bool] = None
    if route_ok is not None and eq_pass is not None and vq_pass is not None:
        unified_replay = bool(route_ok and eq_pass and vq_pass)
    return {
        **dg,
        "route_gate": route_ok,
        "unified_dca_valid": bool(sealed.get("unified_dca_valid")),
        "unified_dca_replay": unified_replay,
        "unified_dca_sealed": bool(sealed.get("unified_dca_valid")),
        "dca_valid_raw": sealed.get("dca_valid_raw"),
        "route_correct": sealed.get("route_correct"),
        "gold_capability_id": sealed.get("gold_capability_id"),
        "picked_capability_id": sealed.get("picked_capability_id"),
        "dual_gate_solve": dual_solve,
        "E_q": eq_pass,
        "V_q": vq_pass,
    }
