"""Headline decomposed metrics — decision / execution / numerical layers (D1 primary).

Primary headline table uses Pred@τ + Submittable + Task Usability@τ.
DCA is reported only as DCC (diagnostic). E_q and V_q are split for transparency.
"""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[3]


def load_json(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def finite_float(val: Any) -> Optional[float]:
    if val is None:
        return None
    try:
        out = float(val)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def answer_body(workdir: Path) -> Dict[str, Any]:
    data = load_json(workdir / "answer.json")
    body = data.get("answer")
    return dict(body) if isinstance(body, Mapping) else {}


def reference_and_tolerance(workdir: Path) -> Tuple[Optional[float], Dict[str, Any], str]:
    tp = load_json(workdir / "resolved_taskpack.json")
    inv = load_json(workdir / "inventory_row.json")
    ref_view = tp.get("reference_view") or {}
    scenario_id = str(inv.get("scenario_id") or "")
    taskpack_id = str(tp.get("taskpack_id") or inv.get("taskpack_id") or "")
    if scenario_id and taskpack_id:
        try:
            if taskpack_id.startswith("hwb_fl2"):
                from hazardweaver.hwb.registry.fl2_parametric_resolver import resolve_reference_view

                ref_view = resolve_reference_view(tp, scenario_id=scenario_id)
            elif "parametric" in taskpack_id or str(inv.get("source") or "").startswith(
                ("param:", "variant:")
            ):
                from hazardweaver.hwb.registry.track_parametric_resolver import resolve_reference_view

                ref_view = resolve_reference_view(tp, scenario_id=scenario_id)
        except Exception:  # noqa: BLE001
            pass
    outputs = ref_view.get("outputs") or {}
    tolerance = dict(ref_view.get("tolerance") or outputs.get("tolerance") or {})
    metric_name = str(outputs.get("metric_name") or tolerance.get("metric_name") or "")
    ref = outputs.get("reference_score")
    if ref is None:
        ref = outputs.get("metric_value")
    try:
        ref_f = float(ref) if ref is not None else None
    except (TypeError, ValueError):
        ref_f = None
    return ref_f, tolerance, metric_name


def pred_value(workdir: Path, dca: Optional[Mapping[str, Any]] = None) -> Optional[float]:
    dca = dca or load_json(workdir / "dca_result.json")
    pred = finite_float(dca.get("raw_score"))
    if pred is not None:
        return pred
    try:
        from hazardweaver.hwb.bridge.hwa_workdir import load_hwa_submission

        sub = load_hwa_submission(workdir)
        fa = sub.get("final_artifact") or {}
        value = fa.get("value") or {}
        for key in ("reference_score", "metric_value", "rmse_depth", "log_volume_v1"):
            pred = finite_float(value.get(key))
            if pred is not None:
                return pred
    except Exception:  # noqa: BLE001
        pass
    # baseline trajectory without HWA bridge
    traj = load_json(workdir / "trajectory.json")
    fa = traj.get("final_artifact") or {}
    value = fa.get("value") or {}
    for key in ("reference_score", "metric_value", "rmse_depth", "log_volume_v1"):
        pred = finite_float(value.get(key))
        if pred is not None:
            return pred
    eval_rep = load_json(workdir / "eval_report.json")
    eq = eval_rep.get("E_q") if isinstance(eval_rep.get("E_q"), Mapping) else {}
    if isinstance(eq, Mapping):
        pred = finite_float(eq.get("score"))
        if pred is not None:
            return pred
    dg = eval_rep.get("dual_gate") or {}
    pred = finite_float(dg.get("score"))
    return pred


def within_tolerance(
    pred: Optional[float],
    ref: Optional[float],
    tolerance: Mapping[str, Any],
    *,
    multiplier: float,
) -> bool:
    if pred is None:
        return False
    if tolerance.get("label_match"):
        return pred >= 0.5
    if ref is not None and "max_abs_error" in tolerance:
        cap = float(tolerance["max_abs_error"]) * multiplier
        return abs(float(pred) - float(ref)) <= cap
    if ref is not None and "min_score" in tolerance:
        floor = float(tolerance["min_score"])
        return float(pred) >= floor
    return False


def _task_usable(
    *,
    expected_action: str,
    action: str,
    pred_ok: bool,
    submittable: bool,
) -> bool:
    exp = str(expected_action or "solve")
    act = str(action or "")
    if exp == "solve":
        return pred_ok
    if exp == "abstain":
        return act == "abstain"
    if exp == "clarify":
        return act == "clarify"
    return submittable and pred_ok


def _dual_gate_layers(workdir: Path) -> Dict[str, Any]:
    inv = load_json(workdir / "inventory_row.json")
    dca = load_json(workdir / "dca_result.json")
    out: Dict[str, Any] = {
        "eq_pass": None,
        "vq_pass": None,
        "dual_gate_valid": bool(dca.get("valid")) if dca else None,
        "dual_gate_outcome": None,
        "contract_violated": bool(dca.get("contract_violated")) if dca else None,
    }
    try:
        from hazardweaver.hwa.benchmark.hwa_workdir_dual_gate_v1 import load_hwa_workdir_dual_gate_layers

        layers = load_hwa_workdir_dual_gate_layers(
            workdir,
            inventory_row=inv,
            replay_if_missing=True,
        )
        if layers.get("eq_pass") is not None:
            out["eq_pass"] = layers.get("eq_pass")
        if layers.get("vq_pass") is not None:
            out["vq_pass"] = layers.get("vq_pass")
        if layers.get("dual_gate_valid") is not None:
            out["dual_gate_valid"] = layers.get("dual_gate_valid")
        out["dual_gate_outcome"] = layers.get("dual_gate_outcome")
    except Exception:  # noqa: BLE001
        pass
    return out


def _commit_executed(workdir: Path) -> Optional[bool]:
    ctrl = workdir / "controller_decisions.jsonl"
    if not ctrl.is_file():
        return None
    had_commit = had_execute = False
    for line in ctrl.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        act = str(row.get("action") or "")
        if act == "commit":
            had_commit = True
        if act == "execute":
            had_execute = True
    if not had_commit:
        return None
    return had_execute


def _route_at_k(workdir: Path, *, k: int) -> Optional[bool]:
    routes_doc = load_json(workdir / "routes.json")
    allowed = {str(x) for x in (routes_doc.get("allowed_edge_ids") or []) if x}
    if len(allowed) < 2:
        return None
    selected = ""
    traj = load_json(workdir / "trajectory.json")
    fa = traj.get("final_artifact") or {}
    value = fa.get("value") or {}
    if isinstance(value, Mapping):
        selected = str(value.get("selected_route") or value.get("capability_id") or "")
    retrieval = load_json(workdir / "retrieval.json")
    ranked: List[str] = []
    for row in retrieval.get("retrieved") or []:
        if not isinstance(row, Mapping):
            continue
        cid = str(row.get("capability_id") or row.get("case_id") or "")
        if cid in allowed and cid not in ranked:
            ranked.append(cid)
    if not ranked or not selected:
        return None
    return selected in ranked[:k]


def analyze_cell(workdir: Path) -> Dict[str, Any]:
    """Per-cell decomposed metrics (HWA workdir or baseline run_dir)."""
    meta = load_json(workdir / "run_meta.json")
    inv = load_json(workdir / "inventory_row.json")
    dca = load_json(workdir / "dca_result.json")
    body = answer_body(workdir)
    action = str(body.get("action") or "")
    if not action and (workdir / "trajectory.json").is_file():
        traj = load_json(workdir / "trajectory.json")
        action = str(traj.get("terminal_action") or "")
        if not action and traj.get("final_artifact"):
            action = "solve"
    expected = str(inv.get("expected_action") or "solve")
    ref, tolerance, metric_name = reference_and_tolerance(workdir)
    pred = pred_value(workdir, dca)
    has_score = pred is not None
    submittable = action == "solve" and has_score
    pred1 = within_tolerance(pred, ref, tolerance, multiplier=1.0)
    pred2 = within_tolerance(pred, ref, tolerance, multiplier=2.0)
    pred5 = within_tolerance(pred, ref, tolerance, multiplier=5.0)
    usable = _task_usable(
        expected_action=expected,
        action=action,
        pred_ok=pred1,
        submittable=submittable,
    )
    dg = _dual_gate_layers(workdir)
    delta = abs(float(pred) - float(ref)) if pred is not None and ref is not None else None
    decision_correct = (
        (expected == "solve" and action == "solve")
        or (expected == "abstain" and action == "abstain")
        or (expected == "clarify" and action == "clarify")
    )
    traj = load_json(workdir / "HWA_TRAJECTORY_v2.json") or load_json(workdir / "trajectory.json")
    return {
        "workdir": str(workdir),
        "instance_id": str(inv.get("instance_id") or ""),
        "condition": str(meta.get("same_llm_condition") or "?"),
        "track": str(meta.get("track") or inv.get("track") or "?"),
        "tier": str(meta.get("difficulty_tier") or inv.get("difficulty_tier") or "?"),
        "action": action,
        "expected_action": expected,
        "decision_action_correct": decision_correct,
        "submittable": submittable,
        "has_score": has_score,
        "pred_at_1x": pred1,
        "pred_at_2x": pred2,
        "pred_at_5x": pred5,
        "task_usability_at_tau": usable,
        "eq_pass": dg.get("eq_pass"),
        "vq_pass": dg.get("vq_pass"),
        "dual_gate_valid": dg.get("dual_gate_valid"),
        "dual_gate_outcome": dg.get("dual_gate_outcome"),
        "dcc": bool(dca.get("valid")) if dca else dg.get("dual_gate_valid"),
        "contract_violated": dg.get("contract_violated"),
        "vtc": bool(traj.get("final_artifact")),
        "raw_score": pred,
        "reference_score": ref,
        "abs_delta": delta,
        "metric_name": metric_name or dca.get("metric_name"),
        "commit_executed": _commit_executed(workdir),
        "route_at_1": _route_at_k(workdir, k=1),
        "route_at_5": _route_at_k(workdir, k=5),
    }


def _rate(num: int, den: int) -> float:
    return float(num) / float(den) if den else 0.0


def _mean(vals: Sequence[float]) -> Optional[float]:
    if not vals:
        return None
    return sum(vals) / float(len(vals))


def aggregate_layers(rows: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    solve = [r for r in rows if str(r.get("action") or "") == "solve"]
    scored = [r for r in rows if r.get("has_score")]
    submittable = [r for r in rows if r.get("submittable")]
    eq_known = [r for r in rows if r.get("eq_pass") is not None]
    vq_known = [r for r in rows if r.get("vq_pass") is not None]
    route1_known = [r for r in rows if r.get("route_at_1") is not None]

    def _bool_rate(key: str, pool: Sequence[Mapping[str, Any]] = rows) -> float:
        pn = len(pool)
        return _rate(sum(1 for r in pool if r.get(key)), pn) if pn else 0.0

    deltas = [float(r["abs_delta"]) for r in scored if r.get("abs_delta") is not None]

    return {
        "n_cells": n,
        "decision": {
            "solve_rate": _rate(len(solve), n),
            "action_accuracy": _bool_rate("decision_action_correct"),
            "abstain_rate": _rate(sum(1 for r in rows if r.get("action") == "abstain"), n),
            "clarify_rate": _rate(sum(1 for r in rows if r.get("action") == "clarify"), n),
            "action_counts": dict(Counter(str(r.get("action") or "?") for r in rows)),
        },
        "execution": {
            "submittable_rate": _bool_rate("submittable"),
            "has_score_rate": _bool_rate("has_score"),
            "eq_pass_rate": _rate(sum(1 for r in eq_known if r.get("eq_pass")), len(eq_known)),
            "vq_pass_rate": _rate(sum(1 for r in vq_known if r.get("vq_pass")), len(vq_known)),
            "dual_gate_valid_rate": _bool_rate("dual_gate_valid"),
            "dcc_rate": _bool_rate("dcc"),
            "contract_violation_rate": _rate(
                sum(1 for r in rows if r.get("contract_violated") is True), n
            ),
            "commit_executed_rate": _rate(
                sum(1 for r in rows if r.get("commit_executed") is True),
                sum(1 for r in rows if r.get("commit_executed") is not None),
            ),
            "n_eq_evaluated": len(eq_known),
            "n_vq_evaluated": len(vq_known),
        },
        "numerical": {
            "pred_at_1x_all": _bool_rate("pred_at_1x"),
            "pred_at_2x_all": _bool_rate("pred_at_2x"),
            "pred_at_5x_all": _bool_rate("pred_at_5x"),
            "pred_at_1x_on_scored": _bool_rate("pred_at_1x", scored),
            "pred_at_1x_on_submittable": _bool_rate("pred_at_1x", submittable),
            "pred_at_5x_on_scored": _bool_rate("pred_at_5x", scored),
            "task_usability_at_tau": _bool_rate("task_usability_at_tau"),
            "n_scored": len(scored),
            "n_submittable": len(submittable),
            "mean_abs_delta_on_scored": _mean(deltas),
        },
        "routing": {
            "route_at_1": _rate(sum(1 for r in route1_known if r.get("route_at_1")), len(route1_known)),
            "route_at_5": _rate(
                sum(1 for r in rows if r.get("route_at_5")),
                sum(1 for r in rows if r.get("route_at_5") is not None),
            ),
            "n_route_evaluated": len(route1_known),
        },
    }


def aggregate_by_dimension(
    rows: Sequence[Mapping[str, Any]], dim: str
) -> Dict[str, Dict[str, Any]]:
    groups: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row.get(dim) or "?")].append(dict(row))
    return {key: aggregate_layers(grp) for key, grp in groups.items()}


def build_decomposed_main_table(
    rows: Sequence[Mapping[str, Any]], *, agent_id: str = "hwa"
) -> Dict[str, Any]:
    """Primary headline table — Pred@τ layers; DCC diagnostic only."""
    layers = aggregate_layers(rows)
    return {
        "schema_version": "HEADLINE_DECOMPOSED_MAIN_TABLE_v1",
        "primary_metrics": [
            "numerical.pred_at_1x_all",
            "numerical.pred_at_5x_all",
            "execution.submittable_rate",
            "numerical.task_usability_at_tau",
            "execution.eq_pass_rate",
            "execution.vq_pass_rate",
        ],
        "diagnostic_metrics": ["execution.dcc_rate", "execution.contract_violation_rate"],
        "agent_id": agent_id,
        "layers": layers,
        "by_tier": aggregate_by_dimension(rows, "tier"),
        "by_track": aggregate_by_dimension(rows, "track"),
        "by_condition": aggregate_by_dimension(rows, "condition"),
    }


def render_markdown_table(layers: Mapping[str, Any], *, title: str) -> str:
    d = layers.get("decision") or {}
    e = layers.get("execution") or {}
    n = layers.get("numerical") or {}
    r = layers.get("routing") or {}
    lines = [
        f"## {title}",
        "",
        f"n_cells: **{layers.get('n_cells', 0)}**",
        "",
        "### 决策层",
        f"- Solve rate: **{d.get('solve_rate', 0):.1%}** · Action accuracy: **{d.get('action_accuracy', 0):.1%}**",
        f"- Actions: `{d.get('action_counts', {})}`",
        "",
        "### 执行层",
        f"- Submittable: **{e.get('submittable_rate', 0):.1%}** · Has score: **{e.get('has_score_rate', 0):.1%}**",
        f"- E_q pass: **{e.get('eq_pass_rate', 0):.1%}** (n={e.get('n_eq_evaluated', 0)}) · "
        f"V_q pass: **{e.get('vq_pass_rate', 0):.1%}** (n={e.get('n_vq_evaluated', 0)})",
        f"- DCC (diagnostic): **{e.get('dcc_rate', 0):.1%}** · Contract violation: **{e.get('contract_violation_rate', 0):.1%}**",
        "",
        "### 数值层（与 DCC 解耦）",
        f"- Pred@1× (all): **{n.get('pred_at_1x_all', 0):.1%}** · Pred@5× (all): **{n.get('pred_at_5x_all', 0):.1%}**",
        f"- Pred@1× on scored (n={n.get('n_scored', 0)}): **{n.get('pred_at_1x_on_scored', 0):.1%}**",
        f"- Pred@1× on submittable (n={n.get('n_submittable', 0)}): **{n.get('pred_at_1x_on_submittable', 0):.1%}**",
        f"- Task Usability@τ: **{n.get('task_usability_at_tau', 0):.1%}**",
    ]
    if r.get("n_route_evaluated"):
        lines.extend(
            [
                "",
                "### 路由层（多路线子集）",
                f"- Route@1: **{r.get('route_at_1', 0):.1%}** · Route@5: **{r.get('route_at_5', 0):.1%}** "
                f"(n={r.get('n_route_evaluated', 0)})",
            ]
        )
    lines.append("")
    return "\n".join(lines)
