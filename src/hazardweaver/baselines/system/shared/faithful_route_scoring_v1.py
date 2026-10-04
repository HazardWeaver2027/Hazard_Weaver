"""Deterministic route-ranking signals for faithful v4.1 baselines (no hidden evaluator).

Each policy uses a *different* public signal so Llama T≈0 cannot collapse Table 1
rows to one CAP. Gold capability ids are never read.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

_METRIC_KEYS = ("metric", "metric_name", "primary_metric")
_NARRATIVE_KEYS = ("summary", "description", "method", "hypothesis", "scientific_assumptions")
_SCHEMA_KEYS = ("modality", "inputs", "input_schema", "outputs", "output_schema", "units")
_MECHANISM_TERMS = {
    "model",
    "physics",
    "hydrolog",
    "atmospher",
    "statist",
    "bayes",
    "neural",
    "forecast",
    "process",
    "empirical",
    "segment",
    "detect",
    "track",
    "spread",
    "inundat",
}


def _route_row(routes_payload: Mapping[str, Any], edge_id: str) -> Dict[str, Any]:
    for row in routes_payload.get("routes") or []:
        if isinstance(row, Mapping) and str(row.get("edge_id")) == edge_id:
            return dict(row)
    return {}


def _route_contract(routes_payload: Mapping[str, Any], edge_id: str) -> Dict[str, Any]:
    row = _route_row(routes_payload, edge_id)
    contract = dict(row.get("contract") or {})
    if row.get("summary") and not contract.get("summary"):
        contract["summary"] = row.get("summary")
    return contract


def _join_fields(contract: Mapping[str, Any], keys: Sequence[str]) -> str:
    parts: List[str] = []
    for key in keys:
        val = contract.get(key)
        if isinstance(val, (list, tuple)):
            parts.extend(str(x) for x in val)
        elif val:
            parts.append(str(val))
    return " ".join(parts)


def success_metric_hint(taskpack: Mapping[str, Any]) -> str:
    sc = taskpack.get("success_criteria") or {}
    for key in ("metric", "metric_name", "primary_metric"):
        val = str(sc.get(key) or "").strip()
        if val:
            return val
    ref = taskpack.get("reference_view") or {}
    outs = ref.get("outputs") or {}
    if isinstance(outs, Mapping):
        for key in ("metric", "metric_name", "name"):
            val = str(outs.get(key) or "").strip()
            if val:
                return val
    return ""


def enrich_selection_goal(
    goal: str,
    *,
    taskpack: Mapping[str, Any],
    inventory_row: Mapping[str, Any],
) -> str:
    """Public task context for baselines — not oracle gold CAP."""
    track = str(inventory_row.get("track") or "")
    elig = str(inventory_row.get("route_eligibility") or "")
    metric = success_metric_hint(taskpack)
    lines = [goal.strip()]
    if track:
        lines.append(f"Track: {track}")
    if elig:
        lines.append(f"Route eligibility: {elig}")
    if metric:
        lines.append(f"Target evaluation metric (task contract): {metric}")
    return "\n".join(lines)


def tokenize(text: str) -> set[str]:
    return {t for t in re.split(r"[^a-z0-9_]+", str(text).lower()) if len(t) > 2}


def contract_token_overlap(text_a: str, text_b: str) -> float:
    ta, tb = tokenize(text_a), tokenize(text_b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / max(len(ta), len(tb))


def _primary_metric_align(contract: Mapping[str, Any], primary: str) -> float:
    if not primary:
        return 0.0
    metric_blob = _join_fields(contract, _METRIC_KEYS)
    if not metric_blob:
        return 0.0
    p = primary.strip().lower()
    m = metric_blob.strip().lower()
    if p == m or p in m or m in p:
        return 1.0
    return contract_token_overlap(primary, metric_blob)


def metric_program_fit_score(
    edge_id: str,
    metric_program: Mapping[str, Any],
    routes_payload: Mapping[str, Any],
) -> float:
    """MLE-STAR: primary metric dominates; secondary cannot beat a primary hit."""
    contract = _route_contract(routes_payload, edge_id)
    primary = str(metric_program.get("primary_metric") or "")
    secondary = " ".join(str(x) for x in (metric_program.get("secondary_metrics") or []))
    primary_align = _primary_metric_align(contract, primary)
    secondary_blob = _join_fields(contract, _METRIC_KEYS)
    secondary_align = contract_token_overlap(secondary, secondary_blob) if secondary else 0.0
    # Secondary is a tie-break / refinement term only.
    score = primary_align + 0.12 * secondary_align * (1.0 if primary_align > 0.0 else 0.2)
    obj = str(metric_program.get("optimization_objective") or "").lower()
    blob = _join_fields(contract, _METRIC_KEYS + _NARRATIVE_KEYS)
    if obj and obj in blob.lower():
        score += 0.02
    return min(1.0, score)


def infer_task_schema(goal: str, taskpack: Mapping[str, Any]) -> Dict[str, Any]:
    """AutoML-Agent: I/O + modality from public goal text — not metric names."""
    g = f"{goal} {success_metric_hint(taskpack)}".lower()
    if any(k in g for k in ("track", "trajectory", "storm", "cyclone", "ibtracs")):
        modality = "track"
    elif any(k in g for k in ("table", "tabular", "spreadsheet", "csi", "spread")):
        modality = "tabular"
    elif any(k in g for k in ("raster", "grid", "pixel", "depth", "image", "burn", "flood", "seg")):
        modality = "raster"
    else:
        modality = ""
    return {
        "required_inputs": [],
        "expected_outputs": [],
        "modality": modality,
        "granularity": "",
        "pipeline_notes": "inferred_from_goal_modality_only",
    }


def task_schema_fit_score(
    edge_id: str,
    task_schema: Mapping[str, Any],
    routes_payload: Mapping[str, Any],
) -> float:
    """AutoML-Agent: contract I/O / modality. Metric fields are excluded on purpose."""
    contract = _route_contract(routes_payload, edge_id)
    req_inputs = " ".join(str(x) for x in (task_schema.get("required_inputs") or []))
    exp_outputs = " ".join(str(x) for x in (task_schema.get("expected_outputs") or []))
    modality = str(task_schema.get("modality") or "")
    schema_blob = _join_fields(contract, _SCHEMA_KEYS)
    score = 0.0
    route_mod = str(contract.get("modality") or "").lower()
    if modality and route_mod:
        score += 0.7 if modality.lower() == route_mod else 0.15 * contract_token_overlap(modality, route_mod)
    elif modality:
        score += 0.35 * contract_token_overlap(modality, schema_blob)
    if req_inputs:
        score += 0.15 * contract_token_overlap(req_inputs, schema_blob)
    if exp_outputs:
        # Do not reward metric-name leakage through expected_outputs.
        if tokenize(exp_outputs) & tokenize(_join_fields(contract, _METRIC_KEYS)):
            score += 0.0
        else:
            score += 0.15 * contract_token_overlap(exp_outputs, schema_blob)
    return min(1.0, score)


def idea_space_fit_score(
    edge_id: str,
    *,
    goal: str,
    routes_payload: Mapping[str, Any],
) -> float:
    """AIDE: scientific idea / method narrative, not the evaluation metric name."""
    contract = _route_contract(routes_payload, edge_id)
    narrative = _join_fields(contract, _NARRATIVE_KEYS)
    metric_tokens = tokenize(_join_fields(contract, _METRIC_KEYS))
    goal_tokens = tokenize(goal) - metric_tokens
    narr_tokens = tokenize(narrative) - metric_tokens
    if not narr_tokens or not goal_tokens:
        return 0.0
    return len(goal_tokens & narr_tokens) / max(len(goal_tokens), len(narr_tokens))


def hypothesis_plausibility_score(
    edge_id: str,
    *,
    goal: str,
    routes_payload: Mapping[str, Any],
) -> float:
    """RD-Agent: mechanism / research-hypothesis fit."""
    contract = _route_contract(routes_payload, edge_id)
    blob = _join_fields(contract, _NARRATIVE_KEYS + ("modality",))
    goal_l = goal.lower()
    mech = sum(1 for term in _MECHANISM_TERMS if term in blob.lower() or term in goal_l)
    narr = idea_space_fit_score(edge_id, goal=goal, routes_payload=routes_payload)
    return min(1.0, 0.60 * narr + 0.40 * min(1.0, mech / 3.0))


def typed_tool_constraint_score(
    edge_id: str,
    *,
    goal: str,
    routes_payload: Mapping[str, Any],
    taskpack: Mapping[str, Any],
) -> float:
    """DisasterBench-style: typed tool-interface completeness + constraint fit."""
    contract = _route_contract(routes_payload, edge_id)
    filled = 0
    for key in ("modality", "inputs", "input_schema", "outputs", "output_schema", "units"):
        if contract.get(key):
            filled += 1
    completeness = filled / 6.0
    schema = infer_task_schema(goal, taskpack)
    mod = task_schema_fit_score(edge_id, schema, routes_payload)
    return min(1.0, 0.55 * completeness + 0.45 * mod)


def heuristic_tot_value(
    edge_id: str,
    *,
    goal: str,
    routes_payload: Mapping[str, Any],
    taskpack: Mapping[str, Any],
) -> float:
    return typed_tool_constraint_score(
        edge_id, goal=goal, routes_payload=routes_payload, taskpack=taskpack
    )


def plan_first_step_score(
    edge_id: str,
    *,
    allowed: Sequence[str],
    routes_payload: Mapping[str, Any],
) -> float:
    """Plan-and-Execute: inventory order is the frozen plan; first listed step first."""
    try:
        idx = list(allowed).index(edge_id)
    except ValueError:
        idx = len(allowed)
    n = max(len(allowed), 1)
    completeness = typed_tool_constraint_score(
        edge_id, goal="", routes_payload=routes_payload, taskpack={}
    )
    return (1.0 - idx / n) * 0.75 + 0.25 * completeness


def blend_ranked_order(
    allowed: Sequence[str],
    llm_order: Sequence[str],
    det_scores: Mapping[str, float],
    *,
    llm_weight: float,
) -> List[str]:
    llm_weight = max(0.0, min(1.0, llm_weight))
    det_weight = 1.0 - llm_weight
    llm_rank = {eid: i for i, eid in enumerate(llm_order) if eid in allowed}
    n = max(len(allowed), 1)

    def combined(eid: str) -> Tuple[float, float, str]:
        lr = llm_rank.get(eid, n)
        llm_score = 1.0 - (lr / n)
        det = float(det_scores.get(eid) or 0.0)
        return (llm_weight * llm_score + det_weight * det, det, eid)

    return [eid for _, _, eid in sorted((combined(e) for e in allowed), reverse=True)]


def argmax_score(
    allowed: Sequence[str],
    scores: Mapping[str, float],
    *,
    tie_mode: str = "edge_id_asc",
) -> str:
    if not allowed:
        raise ValueError("empty_allowed_edge_ids")
    if tie_mode == "edge_id_desc":
        return sorted(allowed, key=lambda e: (float(scores.get(e) or 0.0), e))[-1]
    return sorted(allowed, key=lambda e: (-float(scores.get(e) or 0.0), e))[0]


def ranked_by_score(allowed: Sequence[str], scores: Mapping[str, float]) -> List[str]:
    return sorted(allowed, key=lambda e: (-float(scores.get(e) or 0.0), e))


def mle_metric_program_from_public(goal: str, taskpack: Mapping[str, Any]) -> Dict[str, Any]:
    hint = success_metric_hint(taskpack)
    obj = "minimize" if any(w in goal.lower() for w in ("minimiz", "rmse", "mae", "error")) else "maximize"
    return {
        "primary_metric": hint or "",
        "secondary_metrics": [],
        "optimization_objective": obj,
        "constraints": [],
    }


def policy_fallback_pick(
    policy: str,
    *,
    allowed: Sequence[str],
    routes_payload: Mapping[str, Any],
    goal: str,
    taskpack: Mapping[str, Any],
    banned: Optional[Sequence[str]] = None,
) -> str:
    """No-LLM / LLM-error pick that still implements the paper's selection philosophy."""
    allowed = list(allowed)
    banned_set = {str(x) for x in (banned or []) if x}
    candidates = [e for e in allowed if e not in banned_set] or allowed
    program = mle_metric_program_from_public(goal, taskpack)
    metric_s = {e: metric_program_fit_score(e, program, routes_payload) for e in allowed}
    schema = infer_task_schema(goal, taskpack)
    schema_s = {e: task_schema_fit_score(e, schema, routes_payload) for e in allowed}
    idea_s = {e: idea_space_fit_score(e, goal=goal, routes_payload=routes_payload) for e in allowed}
    hypo_s = {e: hypothesis_plausibility_score(e, goal=goal, routes_payload=routes_payload) for e in allowed}
    tot_s = {
        e: typed_tool_constraint_score(e, goal=goal, routes_payload=routes_payload, taskpack=taskpack)
        for e in allowed
    }
    metric_rank = ranked_by_score(allowed, metric_s)
    greedy = metric_rank[0]

    if policy in {"mle_star", "mle_star_faithful_v4"}:
        return argmax_score(candidates, metric_s, tie_mode="edge_id_asc")
    if policy in {"automl_agent", "automl_agent_faithful_v4"}:
        return argmax_score(candidates, schema_s, tie_mode="edge_id_desc")
    if policy in {"aide", "aide_faithful_v3"}:
        # AIDE without trials: expand one node away from the metric-greedy idea.
        alts = [e for e in metric_rank if e != greedy and e in candidates]
        return alts[0] if alts else greedy
    if policy in {"rd_agent", "rd_agent_faithful_v3"}:
        if any(hypo_s[e] > 0 for e in candidates):
            return argmax_score(candidates, hypo_s, tie_mode="edge_id_asc")
        # Novel research hypothesis: opposite end of the metric order.
        rev = [e for e in reversed(metric_rank) if e in candidates]
        return rev[0] if rev else candidates[0]
    if policy in {"ds_agent", "ds_agent_cbr_faithful_v3"}:
        return candidates[0]
    if policy in {"plan_execute_faithful_v1", "plan_execute"}:
        plan_s = {e: plan_first_step_score(e, allowed=allowed, routes_payload=routes_payload) for e in candidates}
        return argmax_score(candidates, plan_s, tie_mode="edge_id_asc")
    if policy in {"self_consistency_faithful_v1"}:
        votes = [greedy, argmax_score(candidates, schema_s), argmax_score(candidates, tot_s)]
        counts: Dict[str, int] = {}
        for v in votes:
            counts[v] = counts.get(v, 0) + 1
        return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]
    if policy in {"disasterbench_tot_faithful_v1"}:
        return argmax_score(candidates, tot_s, tie_mode="edge_id_asc")
    if policy in {"reflexion_faithful_v1"}:
        if any(idea_s[e] > 0 for e in candidates):
            return argmax_score(candidates, idea_s, tie_mode="edge_id_asc")
        alts = [e for e in metric_rank if e in candidates]
        return alts[min(1, len(alts) - 1)]
    if policy in {"react_same_tool", "react_same_tool_v1"}:
        pair = metric_rank[:2] if len(metric_rank) >= 2 else metric_rank
        pair = [e for e in pair if e in candidates] or candidates[:2]
        if len(pair) >= 2:
            return pair[1]
        return pair[0]
    return candidates[0]


def apply_primary_metric_refinement(
    order: Sequence[str],
    *,
    metric_program: Mapping[str, Any],
    routes_payload: Mapping[str, Any],
) -> List[str]:
    """MLE-STAR targeted refinement: a primary-metric hit outranks a primary miss."""
    if not order:
        return []
    scores = {eid: metric_program_fit_score(eid, metric_program, routes_payload) for eid in order}
    primary_hits = [eid for eid in order if _primary_metric_align(_route_contract(routes_payload, eid), str(metric_program.get("primary_metric") or "")) > 0.0]
    if not primary_hits:
        return list(order)
    winner = primary_hits[0]
    rest = [eid for eid in order if eid != winner]
    return [winner] + rest
