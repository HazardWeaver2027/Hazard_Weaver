"""Decision-layer headline gate (13th gate) — HWB ICLR benchmark v1/v2/v3.

Routes capability-fitting tasks to Atlas; admits decision-layer tasks to headline DCA pool.
Criteria align with docs/core_hwb/HWB_BENCHMARK_TASK_AND_BASELINE_STRATEGY_v1.md §2.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Set

from hazardweaver.hwb.registry.g6_hard_anchor_meta import TABULAR_SUPERVISED_ANCHORS

CAPABILITY_FITTING_GENERATION_MODES: Set[str] = {
    "g6_hard",
    "tabular_supervised",
    "fixture",
}

HEADLINE_PREFERRED_GENERATION_MODES: Set[str] = {
    "pfdf_factual_compose",
    "data_then_scenario_wrap",
    "data_then_scenario_llm_r2",
    "llm_back_instruct",
    "template_reverse_qa",
    "pfdf_counterfactual",
    "parametric",
}

ABSTENTION_FLOOR_MAX_FRACTION = 0.05


@dataclass
class DecisionLayerCriterion:
    name: str
    passed: bool
    reason: str = ""


@dataclass
class DecisionLayerGateReport:
    taskpack_id: str
    headline_eligible: bool
    capability_fitting_only: bool
    difficulty_tier: str
    criteria: List[DecisionLayerCriterion] = field(default_factory=list)
    routing_tier: str = "atlas"  # headline | atlas | abstention_floor

    def to_dict(self) -> Dict[str, Any]:
        return {
            "taskpack_id": self.taskpack_id,
            "headline_eligible": self.headline_eligible,
            "capability_fitting_only": self.capability_fitting_only,
            "difficulty_tier": self.difficulty_tier,
            "routing_tier": self.routing_tier,
            "criteria": [
                {"name": c.name, "passed": c.passed, "reason": c.reason} for c in self.criteria
            ],
        }


def _generation_mode(taskpack: Mapping[str, Any]) -> str:
    meta = taskpack.get("metadata") or {}
    prov = taskpack.get("provenance") or {}
    src = taskpack.get("source") or {}
    return str(
        meta.get("generation_mode")
        or prov.get("generation_mode")
        or src.get("generation_mode")
        or taskpack.get("generation_mode")
        or ""
    ).lower()


def _difficulty_tier(taskpack: Mapping[str, Any]) -> str:
    meta = taskpack.get("metadata") or {}
    return str(meta.get("difficulty_tier") or taskpack.get("difficulty_tier") or "L1").upper()


def _anchor_from_source(src: Mapping[str, Any]) -> str:
    anchor = str(src.get("anchor_id") or "")
    if anchor:
        return anchor
    g6_id = str(src.get("g6_task_id") or "")
    if g6_id.startswith("H_"):
        base = g6_id.split("_COREEXEC")[0].split("_ABSTAIN")[0].split("_CLARIFY")[0]
        return base[2:] if len(base) > 2 else ""
    return ""


def _is_capability_fitting(taskpack: Mapping[str, Any]) -> bool:
    meta = taskpack.get("metadata") or {}
    if meta.get("capability_fitting_only"):
        return True
    if meta.get("decision_layer_headline"):
        return False

    mode = _generation_mode(taskpack)
    ref = taskpack.get("reference_view") or {}
    expected = str(ref.get("expected_action") or "solve")
    src = taskpack.get("source") or {}

    if mode in {"pfdf_factual_compose", "llm_back_instruct", "data_then_scenario_wrap"}:
        return False

    if expected in {"abstain", "clarify"}:
        return False

    if mode in {"capability_native_v1"} or meta.get("table_a_instance_id"):
        return True

    track = str(taskpack.get("headline_target") or "")
    if track.startswith("MH-") and not meta.get("capability_fitting_only"):
        return False

    if mode == "g6_hard" or src.get("pack_id") == "g6_hard_v1":
        stratum = str(src.get("stratum") or "")
        if stratum == "CoreExec":
            anchor = _anchor_from_source(src)
            if anchor in TABULAR_SUPERVISED_ANCHORS:
                return True

    if mode in CAPABILITY_FITTING_GENERATION_MODES and not track.startswith("MH-"):
        witnesses = (ref.get("accepted_witnesses") or [])
        if len(witnesses) >= 2 and mode == "g6_hard":
            return True

    if _difficulty_tier(taskpack) == "L0":
        return True

    return False


def _criterion_multi_route(taskpack: Mapping[str, Any]) -> DecisionLayerCriterion:
    ref = taskpack.get("reference_view") or {}
    witnesses = list(ref.get("accepted_witnesses") or [])
    routes = list(taskpack.get("candidate_routes") or [])
    reject_routes = [r for r in routes if str(r.get("admissibility_label") or "").upper() == "REJECT"]
    if len(witnesses) >= 2 or len(routes) >= 2:
        if reject_routes or len(witnesses) >= 2:
            return DecisionLayerCriterion(
                "multi_route_admissibility",
                True,
                f"witnesses={len(witnesses)} reject_routes={len(reject_routes)}",
            )
    constraints = (ref.get("trajectory_constraints") or {})
    if constraints.get("forbidden_shortcuts"):
        return DecisionLayerCriterion(
            "multi_route_admissibility",
            True,
            "forbidden_shortcuts imply route discrimination",
        )
    return DecisionLayerCriterion("multi_route_admissibility", False, "single-route or no reject trap")


def _criterion_abstention(taskpack: Mapping[str, Any]) -> DecisionLayerCriterion:
    ref = taskpack.get("reference_view") or {}
    expected = str(ref.get("expected_action") or "solve")
    if expected in {"abstain", "clarify"}:
        return DecisionLayerCriterion("abstention_scoring", True, f"expected_action={expected}")
    return DecisionLayerCriterion("abstention_scoring", False, "solve-only task")


def _criterion_coupling(taskpack: Mapping[str, Any]) -> DecisionLayerCriterion:
    meta = taskpack.get("metadata") or {}
    if meta.get("capability_fitting_only") and not meta.get("mh_coupling"):
        return DecisionLayerCriterion("multi_hazard_coupling", False, "table_a single-cap fitting")
    track = str(taskpack.get("headline_target") or "")
    hazard_set = taskpack.get("hazard_set") or (taskpack.get("metadata") or {}).get("hazard_set") or []
    coupling = taskpack.get("coupling_link") or (taskpack.get("metadata") or {}).get("coupling_link")
    if track.startswith("MH-") or len(hazard_set) >= 2 or coupling:
        return DecisionLayerCriterion(
            "multi_hazard_coupling",
            True,
            f"track={track} hazards={len(hazard_set)}",
        )
    meta = taskpack.get("metadata") or {}
    if meta.get("mh_coupling") or meta.get("mh1_finish_line_c"):
        return DecisionLayerCriterion("multi_hazard_coupling", True, "mh metadata flag")
    return DecisionLayerCriterion("multi_hazard_coupling", False, "single-hazard task")


def _criterion_causality(taskpack: Mapping[str, Any]) -> DecisionLayerCriterion:
    meta = taskpack.get("metadata") or {}
    cutoff = taskpack.get("cutoff_manifest") or meta.get("cutoff_manifest") or {}
    if taskpack.get("decision_time") or cutoff.get("decision_time") or cutoff.get("issue_yyyymm"):
        return DecisionLayerCriterion("decision_time_causality", True, "decision_time/cutoff present")
    if meta.get("leaked_future_info") or taskpack.get("leaked_future_info"):
        return DecisionLayerCriterion("decision_time_causality", True, "leaked_future_info trap")
    if _difficulty_tier(taskpack) in {"L2", "L3", "L4"}:
        return DecisionLayerCriterion("decision_time_causality", True, f"tier={_difficulty_tier(taskpack)}")
    return DecisionLayerCriterion("decision_time_causality", False, "no temporal boundary")


def evaluate_decision_layer_gate(taskpack: Mapping[str, Any]) -> DecisionLayerGateReport:
    """Evaluate whether a TaskPack belongs in headline DCA pool vs Atlas."""
    taskpack_id = str(taskpack.get("taskpack_id") or "unknown")
    fitting = _is_capability_fitting(taskpack)
    criteria = [
        _criterion_multi_route(taskpack),
        _criterion_abstention(taskpack),
        _criterion_coupling(taskpack),
        _criterion_causality(taskpack),
    ]
    passed_count = sum(1 for c in criteria if c.passed)
    tier = _difficulty_tier(taskpack)
    expected = str((taskpack.get("reference_view") or {}).get("expected_action") or "solve")

    if fitting:
        routing = "atlas"
        headline = False
    elif expected in {"abstain", "clarify"}:
        routing = "abstention_floor"
        headline = True
    elif passed_count >= 1 or tier in {"L3", "L4"}:
        routing = "headline"
        headline = True
    else:
        routing = "atlas"
        headline = False

    return DecisionLayerGateReport(
        taskpack_id=taskpack_id,
        headline_eligible=headline,
        capability_fitting_only=fitting,
        difficulty_tier=tier,
        criteria=criteria,
        routing_tier=routing,
    )


def filter_headline_pool(
    taskpacks: Sequence[Mapping[str, Any]],
    *,
    max_abstention_fraction: float = ABSTENTION_FLOOR_MAX_FRACTION,
) -> List[DecisionLayerGateReport]:
    """Filter taskpacks; enforce abstention floor cap when selecting headline set."""
    reports = [evaluate_decision_layer_gate(tp) for tp in taskpacks]
    headline = [r for r in reports if r.routing_tier == "headline"]
    abstain = [r for r in reports if r.routing_tier == "abstention_floor"]
    max_abstain = max(1, int(len(headline) * max_abstention_fraction))
    if len(abstain) > max_abstain and headline:
        # Keep abstention floor tasks but flag overflow in routing for audit
        for r in abstain[max_abstain:]:
            r.routing_tier = "atlas"
            r.headline_eligible = False
    return reports
