"""Decision-Contract Compliance (DCC) scorer — headline diagnostic metric.

Public name: **Decision-Contract Compliance (DCC)**. Legacy internal name DCA
(Decision-Constrained Accuracy) retained in field names for compatibility.
Authority: docs/core_hwb/HWB_BENCHMARK_STRATEGY_v2_REVISION.md
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

from hazardweaver.hwb.bridge.taskpack_eval import enrich_taskpack_for_evaluation
from hazardweaver.hwb.evaluators.dual_gate import DualGateResult, evaluate_submission
from hazardweaver.hwb.evaluators.baseline_fairness_enforcer import verify_baseline_run_manifest
from hazardweaver.hwb.metrics.cluster_bootstrap import cluster_bootstrap_ci


@dataclass
class DCAResult:
    taskpack_id: str
    agent_id: str
    dca_score: float
    counted: bool
    contract_violated: bool
    budget_within_limit: bool
    metric_name: str
    raw_score: Optional[float]
    outcome: str
    reason_code: str = ""
    difficulty_tier: str = "L1"
    valid: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "taskpack_id": self.taskpack_id,
            "agent_id": self.agent_id,
            "dca_score": self.dca_score,
            "counted": self.counted,
            "contract_violated": self.contract_violated,
            "budget_within_limit": self.budget_within_limit,
            "metric_name": self.metric_name,
            "raw_score": self.raw_score,
            "outcome": self.outcome,
            "reason_code": self.reason_code,
            "difficulty_tier": self.difficulty_tier,
            "valid": self.valid,
        }


def _normalize_metric_score(
    raw: Optional[float],
    *,
    metric_name: str,
    tolerance: Mapping[str, Any],
    reference_score: Optional[float],
) -> float:
    """Map native metric to [0,1] where higher is better."""
    if raw is None:
        return 0.0
    val = float(raw)
    if tolerance.get("label_match"):
        return 1.0 if val >= 0.5 else 0.0
    if "min_score" in tolerance:
        ref = float(reference_score if reference_score is not None else tolerance["min_score"])
        floor = float(tolerance["min_score"])
        if ref <= floor:
            return 1.0 if val >= floor else max(0.0, val / floor) if floor else 0.0
        return max(0.0, min(1.0, (val - floor) / max(ref - floor, 1e-9)))
    if "max_abs_error" in tolerance and reference_score is not None:
        err = abs(val - float(reference_score))
        cap = float(tolerance["max_abs_error"])
        return max(0.0, 1.0 - err / max(cap, 1e-9))
    # lower-is-better metrics (rmse, mae, brier treated as distance from ref)
    if metric_name in {"rmse", "rmse_depth", "mae", "mse"} and reference_score is not None:
        err = abs(val - float(reference_score))
        cap = float(tolerance.get("max_abs_error") or abs(float(reference_score)) or 1.0)
        return max(0.0, 1.0 - err / max(cap, 1e-9))
    return max(0.0, min(1.0, val))


def _contract_violated(dg: DualGateResult) -> bool:
    if dg.adm and dg.adm.reason_code == "REJECT":
        return True
    if dg.vq and not dg.vq.vq:
        return True
    if dg.outcome == "invalid":
        return True
    if dg.eq and not dg.eq.eq and dg.outcome == "solve":
        return True
    return False


def compute_dca_from_dual_gate(
    *,
    taskpack: Mapping[str, Any],
    dg: DualGateResult,
    agent_id: str,
    budget_manifest: Optional[Mapping[str, Any]] = None,
    difficulty_tier: str = "L1",
) -> DCAResult:
    """Compute DCA for one submission given an evaluated dual_gate result."""
    taskpack_id = str(taskpack.get("taskpack_id") or dg.taskpack_id)
    ref = taskpack.get("reference_view") or {}
    expected = str(ref.get("expected_action") or "solve")
    tol = ref.get("tolerance") or {}
    metric_name = str(tol.get("metric") or (dg.eq.metric_name if dg.eq else "") or "unknown")
    ref_out = ref.get("outputs") or {}
    reference_score = ref_out.get(metric_name) or ref_out.get("reference_score")

    fairness = verify_baseline_run_manifest(budget_manifest or {})
    budget_ok = fairness.passed if budget_manifest else True

    violated = _contract_violated(dg)
    raw_score = dg.eq.score if dg.eq else None

    if expected in {"abstain", "clarify"}:
        dca = 1.0 if dg.valid and dg.outcome == expected else 0.0
        return DCAResult(
            taskpack_id=taskpack_id,
            agent_id=agent_id,
            dca_score=dca,
            counted=budget_ok,
            contract_violated=violated,
            budget_within_limit=budget_ok,
            metric_name="abstention_correct",
            raw_score=dca,
            outcome=dg.outcome,
            reason_code=dg.reason_code,
            difficulty_tier=difficulty_tier,
            valid=dg.valid,
        )

    if violated or not budget_ok:
        return DCAResult(
            taskpack_id=taskpack_id,
            agent_id=agent_id,
            dca_score=0.0,
            counted=False,
            contract_violated=violated,
            budget_within_limit=budget_ok,
            metric_name=metric_name,
            raw_score=raw_score,
            outcome=dg.outcome,
            reason_code="contract_violation" if violated else "budget_exceeded",
            difficulty_tier=difficulty_tier,
            valid=False,
        )

    normalized = _normalize_metric_score(
        raw_score,
        metric_name=metric_name,
        tolerance=tol,
        reference_score=reference_score,
    )
    dca_valid = bool(dg.valid)
    return DCAResult(
        taskpack_id=taskpack_id,
        agent_id=agent_id,
        dca_score=normalized if dg.valid else 0.0,
        counted=True,
        contract_violated=False,
        budget_within_limit=True,
        metric_name=metric_name,
        raw_score=raw_score,
        outcome=dg.outcome,
        reason_code=dg.reason_code,
        difficulty_tier=difficulty_tier,
        valid=dca_valid,
    )


def prepare_taskpack_for_unified_dca_eval(
    taskpack: Mapping[str, Any],
    trajectory: Mapping[str, Any],
    *,
    inventory_row: Optional[Mapping[str, Any]] = None,
    workdir: Optional[Any] = None,
) -> Dict[str, Any]:
    """Bind taskpack for unified DCA / bound dual-gate (same path as ``evaluate_dca_submission``)."""
    from pathlib import Path

    from hazardweaver.hwb.bridge.taskpack_eval import executed_capability_id_from_trajectory
    from hazardweaver.hwb.registry.track_parametric_resolver import align_reference_view_to_executed_capability

    tp = enrich_taskpack_for_evaluation(
        taskpack,
        inventory_row=inventory_row,
        final_artifact=trajectory.get("final_artifact"),
    )
    executed = executed_capability_id_from_trajectory(trajectory)
    scenario_gold = ""
    if inventory_row:
        from hazardweaver.hwa.benchmark.unified_dca_route_gate_v1 import unified_scenario_gold_capability_id

        wd = Path(workdir) if workdir is not None else None
        scenario_gold = unified_scenario_gold_capability_id(inventory_row, workdir=wd)
    track_u = str((inventory_row or {}).get("track") or "").upper()
    fl2_replay_bind = False
    if track_u == "FL-2" and executed and inventory_row:
        from hazardweaver.hwb.registry.fl2_parametric_agent_replay_reference_v1 import (
            fl2_parametric_agent_replay_eval_authority,
        )
        from hazardweaver.hwb.registry.fl2_solver_agent_replay_reference_v1 import (
            fl2_solver_agent_replay_eval_authority,
        )

        if fl2_solver_agent_replay_eval_authority(inventory_row) or fl2_parametric_agent_replay_eval_authority(
            inventory_row
        ):
            fl2_replay_bind = True
        elif not scenario_gold:
            tp = align_reference_view_to_executed_capability(tp, executed_capability_id=executed)
    elif executed and not scenario_gold:
        tp = align_reference_view_to_executed_capability(tp, executed_capability_id=executed)
    if inventory_row and (inventory_row.get("unified_benchmark_v1") or inventory_row.get("unified_dca_tolerance")):
        from hazardweaver.hwb.registry.unified_benchmark_tolerance_v1 import apply_unified_benchmark_eval_binding

        metric_bind_cap = executed if (fl2_replay_bind or executed) else scenario_gold
        tp = apply_unified_benchmark_eval_binding(
            tp,
            inventory_row=inventory_row,
            executed_capability_id=metric_bind_cap or None,
        )
    return tp


def evaluate_bound_dual_gate_submission(
    taskpack: Mapping[str, Any],
    trajectory: Mapping[str, Any],
    *,
    inventory_row: Optional[Mapping[str, Any]] = None,
    workdir: Optional[Any] = None,
) -> DualGateResult:
    """Dual gate with unified DCA metric binding (diagnostic E_q / V_q aligned to headline scorer)."""
    tp = prepare_taskpack_for_unified_dca_eval(
        taskpack,
        trajectory,
        inventory_row=inventory_row,
        workdir=workdir,
    )
    return evaluate_submission(tp, trajectory)


def evaluate_dca_submission(
    taskpack: Mapping[str, Any],
    trajectory: Mapping[str, Any],
    *,
    agent_id: str,
    budget_manifest: Optional[Mapping[str, Any]] = None,
    difficulty_tier: str = "L1",
    inventory_row: Optional[Mapping[str, Any]] = None,
    workdir: Optional[Any] = None,
) -> DCAResult:
    """End-to-end DCA for one agent submission."""
    tp = prepare_taskpack_for_unified_dca_eval(
        taskpack,
        trajectory,
        inventory_row=inventory_row,
        workdir=workdir,
    )
    dg = evaluate_submission(tp, trajectory)
    return compute_dca_from_dual_gate(
        taskpack=tp,
        dg=dg,
        agent_id=agent_id,
        budget_manifest=budget_manifest,
        difficulty_tier=difficulty_tier,
    )


def aggregate_dca_by_agent(
    results: Sequence[DCAResult],
) -> Dict[str, Dict[str, Any]]:
    """Per-agent DCA summary with contract violation rate."""
    by_agent: Dict[str, List[DCAResult]] = {}
    for r in results:
        by_agent.setdefault(r.agent_id, []).append(r)

    summary: Dict[str, Dict[str, Any]] = {}
    for agent_id, rows in sorted(by_agent.items()):
        counted = [r for r in rows if r.counted]
        violations = sum(1 for r in rows if r.contract_violated)
        dca_mean = sum(r.dca_score for r in counted) / len(counted) if counted else 0.0
        summary[agent_id] = {
            "dca_mean": dca_mean,
            "n_total": len(rows),
            "n_counted": len(counted),
            "contract_violation_rate": violations / len(rows) if rows else 0.0,
            "budget_within_rate": sum(1 for r in rows if r.budget_within_limit) / len(rows) if rows else 0.0,
        }
    return summary


def learning_curve_by_tier(
    results: Sequence[DCAResult],
    *,
    agent_id: Optional[str] = None,
) -> Dict[str, float]:
    """Mean DCA per difficulty tier (L1–L4) for learning-curve figures."""
    tiers = ("L1", "L2", "L3", "L4")
    filtered = [r for r in results if r.counted and (agent_id is None or r.agent_id == agent_id)]
    out: Dict[str, float] = {}
    for tier in tiers:
        tier_rows = [r for r in filtered if r.difficulty_tier.upper() == tier]
        out[tier] = sum(r.dca_score for r in tier_rows) / len(tier_rows) if tier_rows else float("nan")
    return out


def build_dca_main_table(
    results: Sequence[DCAResult],
    *,
    cluster_key: str = "taskpack_id",
) -> Dict[str, Any]:
    """Build headline main table JSON (v2 §3 structure)."""
    agent_summary = aggregate_dca_by_agent(results)
    rows = []
    for agent_id, stats in agent_summary.items():
        recs = [
            {
                cluster_key: r.taskpack_id,
                "valid": r.dca_score,
                "difficulty_tier": r.difficulty_tier,
            }
            for r in results
            if r.agent_id == agent_id and r.counted
        ]
        point, lo, hi = cluster_bootstrap_ci(recs, cluster_key=cluster_key, value_key="valid")
        rows.append(
            {
                "agent": agent_id,
                "dca_mean": stats["dca_mean"],
                "dca_ci_low": lo,
                "dca_ci_high": hi,
                "contract_violation_rate": stats["contract_violation_rate"],
                "budget_within_rate": stats["budget_within_rate"],
                "n_counted": stats["n_counted"],
                "n_total": stats["n_total"],
            }
        )
    return {
        "schema_version": "HWB_DCA_MAIN_TABLE_v1",
        "metric": "decision_constrained_accuracy",
        "metric_alias": "decision_contract_compliance",
        "metric_public_name": "DCC",
        "primary_table_authority": "hwb/metrics/headline_decomposed_v1.py · scripts/benchmark/report_headline_decomposed_metrics_v1.py",
        "note": "DCC is diagnostic only; primary headline metrics are Pred@τ + Submittable + E_q/V_q layers.",
        "agents": rows,
        "learning_curves": {
            agent: learning_curve_by_tier(results, agent_id=agent) for agent in agent_summary
        },
    }
