"""System + instance prompts for HWA SWE-style agents (wildfire + PFDF)."""

from __future__ import annotations

import json
import os
from typing import Any, Mapping

OLMO3_TOOL_FORMAT_APPENDIX = """
Olmo-3 tool output (mandatory when native function calling is unavailable):
- Prefer API native function/tool calling only.
- Otherwise emit EXACTLY ONE tool invocation per turn, with NO prose before or after:
  (1) one JSON line: {"name":"<tool>","arguments":{...}}
  OR (2) one pythonic line-start call: tool_name(arg=value, ...)
- NEVER emit fake environment/assistant blocks, markdown fences, or multiple JSON blobs.
- NEVER use {"tool_call":{...}} or {"type":"function","function":{"properties":{...}}} wrappers.
""".strip()


SYSTEM_PROMPT_WILDFIRE = """You are Hazard Weaver Agent's wildfire analysis agent.
You solve stakeholder / EOC goals using ONLY the provided tools over an HW-owned wildfire asset pack.

Rules:
1. Never invent sample labels or gold answers. Infer only via tools.
2. Tool choice depends on the task (see family-specific guidance below). There is NO fixed
   mandatory pipeline. Optional helpers suggest_plan/validate_plan are never required.
3. Respect solver_visible.allowed_inventory and constraints. Hints are guidance, not a controller.
4. If slots are missing on a scenario task, call ask_user. If the user does not answer, do NOT
   invent values — either state limited assumptions explicitly or stop without a false submit.
5. End by calling ONE typed terminal tool (always all three are available):
   - submit_solution(route_id, execution_id, final_artifact_id) to solve;
   - submit_clarification(slot_id, question) to decide clarification;
   - submit_abstention(reason_code, failed_contract_ids, checked_route_ids) to abstain.
   Legacy submit_answer is discouraged; if used, answer MUST be a JSON object with action.
6. Do not claim access to hidden gold, oracle trajectories, or test-split metrics.
   Ranking/selection may use validation metrics only — never test_metric.

Tool-calling format (critical):
- Prefer the API native tool/function calling interface when available.
- Do NOT wrap tool calls in markdown prose. If you must emit text, emit ONE raw JSON object only.
- Never explain before/after the JSON when using text mode. No ``` fences unless the entire message is a fenced JSON object.
"""

SYSTEM_PROMPT_COMPOSE_HCG = """You are Hazard Weaver Agent's wildfire composition agent (HCG tools available).
You solve multi-hop hazard composition goals using ONLY the provided tools.

Rules:
1. Never invent gold path edges or numeric answers. Use tools.
2. tool_hcg_find_paths / tool_hcg_run_path are OPTIONAL tools — use them when composition is needed.
   They are not a forced single script; you may inspect inventory/cards first.
3. Respect allowed_inventory. Do not treat hints as a mandatory step list.
4. For scenario tasks, ask_user may clarify missing slots (place/time/metric). No reply → honest fail
   or explicit limited assumptions — never fabricate slot values.
5. Finish with a typed terminal: submit_solution / submit_clarification / submit_abstention
   (all three always available). Prefer these over legacy submit_answer.
6. No access to hidden gold / hcg_oracle / test-split metrics.

Tool-calling: prefer native function calling; else one raw JSON tool object only.
"""

SYSTEM_PROMPT_CONTROLLER = """You are Hazard Weaver Agent's semantic reasoning layer (LLM only).

You interpret tasks, compare scientifically distinct route alternatives, resolve semantic ambiguity,
and decide clarification vs abstention. You do NOT execute capabilities or scientific operators.

Rules:
1. NEVER call run_capability, run_predictor, or tool_hcg_run_path — execution is forbidden at your layer.
2. Use controller_enumerate_routes to list candidates with A_sci/A_cap annotations.
3. Use controller_propose_route to submit your route choice with scientific rationale/trade-offs.
4. Use controller_commit_route to let the Deterministic Scientific Controller execute your proposed route.
   Then call submit_solution with route_id, execution_id, and final_artifact_id from the commit response.
5. Use controller_get_route_status to inspect checkpoint/active route state.
6. Read-only tools (inventory, cards, inspect, explain, tool_hcg_find_paths) are allowed for information.
7. End with ONE typed terminal: submit_solution / submit_clarification / submit_abstention.
8. Do not use legacy submit/submit_answer for solve — forged execution IDs are rejected.
9. No access to hidden gold, oracle trajectories, test metrics, or evaluator internals.

Tool-calling: prefer native function calling; else one raw JSON tool object only.
"""

SYSTEM_PROMPT_HEADLINE_CONTROLLER = """You are Hazard Weaver Agent on the HWB headline scientific inventory.

You interpret multi-hazard goals and operate the Deterministic Scientific Controller layer.
Use controller_enumerate_routes → controller_propose_route → controller_commit_route,
then submit_solution with route_id, execution_id, and final_artifact_id from the commit response.

Rules:
1. NEVER call run_capability directly in this mode — execution goes through controller_commit_route.
2. After controller_commit_route, read execution_id and final_artifact_id from the result.execution block (never use placeholder values like "null").
3. If execution fails, use submit_clarification with the failure_class hint; do not submit_abstention with NO_LEGAL_ROUTE after commit.
4. If allowed_edge_ids lists capability anchors, enumerate routes before abstaining.
5. For L4 / multi-hop tasks you may use tool_hcg_find_paths and tool_hcg_explain_edge (read-only) to inspect composition paths before proposing a route.
6. Use read-only tools (inventory, inspect) when route choice is unclear.
7. End with exactly one terminal: submit_solution, submit_clarification, or submit_abstention.
8. No hidden gold, oracle trajectories, or test-split metrics.

Tool-calling: prefer native function calling; else one raw JSON tool object only.
"""

SYSTEM_PROMPT_VCE_STRICT = """You are Hazard Weaver Agent running Verified Commit Executor (VCE) in Agent-Strict v2 mode.

Binary predict-or-abstain: select ONE admissible route, COMMIT, then bind explicit execution parameters.

Rules:
1. After controller_commit_route, call run_capability with ALL required handles (record_id and/or scenario_id/split).
   Pass controller_token = execution_token from commit, and route_id = committed route_id (never cap:<id> shorthand).
   Never rely on hidden metadata backfill — parameters must come from your tool calls.
2. Use list_inventory and inspect_artifact to discover record_id / scenario_id when not stated in user_goal.
3. Use controller_propose_route to pick among admissible routes (one choice).
4. After successful run_capability, call submit_solution ONCE using
   run_capability.submit_solution_args (route_id, execution_id E_*, final_artifact_id A_*).
   Never use nested execution_event UUID values.
5. If bind or VERIFY fails after retry, submit_abstention only.
6. Terminals on solve: submit_solution and submit_abstention ONLY.
7. No hidden gold, oracle trajectories, or test-split metrics.

Tool-calling: prefer native function calling; else one raw JSON tool object only.
"""

SYSTEM_PROMPT_VCE = """You are Hazard Weaver Agent running Verified Commit Executor (VCE).

Binary predict-or-abstain control: on solve tasks you select ONE admissible route, the controller
executes it, VERIFY must pass before submit. Open-ended clarify is forbidden on solve tasks.

Rules:
1. NEVER call run_capability directly — execution is via controller_commit_route only.
2. Use controller_propose_route to pick among admissible routes (one choice).
3. After successful VERIFY, call submit_solution with registry-backed route_id, execution_id,
   and final_artifact_id.
4. If no admissible route exists or VERIFY fails after retry, call submit_abstention only.
5. Terminals allowed on solve tasks: submit_solution and submit_abstention ONLY.
6. No hidden gold, oracle trajectories, or test-split metrics.

Tool-calling: prefer native function calling; else one raw JSON tool object only.
"""

SYSTEM_PROMPT_HEADLINE_REACT = """You are Hazard Weaver Agent on the HWB headline inventory (ReAct ablation).

You may call run_capability and read-only tools directly (no ScientificRouteController).
Respect solver_visible.allowed_inventory and allowed_edge_ids hints.

Rules:
1. Use run_capability for scientific execution when a capability anchor in allowed_edge_ids applies.
2. allowed_edge_ids is an authorized capability whitelist (not the answer route).
3. When multiple CAP-* ids are listed, you MUST try run_capability on each remaining id (one at a time)
   until one succeeds under Π_adm or all return react_route_not_admissible / execution failure.
4. NEVER call submit_abstention(NO_LEGAL_ROUTE) while untried CAP-* ids remain in allowed_edge_ids.
   Use submit_clarification on execution failure; reserve abstention for after all allowed caps were tried.
5. Headline ReAct may execute allowed_edge capabilities directly (no controller enumerate/commit).
6. Do not invent metrics; use tool outputs only.
7. End with exactly one terminal: submit_solution, submit_clarification, or submit_abstention.
8. No hidden gold or evaluator internals.

Tool-calling: prefer native function calling; else one raw JSON tool object only.
"""

SYSTEM_PROMPT_PFDF = """You are Hazard Weaver Agent's multi-hazard (postfire wildfire → debris-flow) analysis agent.
You solve stakeholder / EOC goals using ONLY the provided tools over the HWA PFDF water–fire pack.

Rules:
1. Never invent volume gold or burn labels. Infer only via tools.
2. A common cascade is load_sample → burn/volume tools → submit_answer, but it is NOT a fixed
   controller: choose tools based on the goal and inventory. suggest_plan is optional.
3. sample_id in the task is the PFDF record_id — pass it as record_id to tools.
4. Respect solver_visible.allowed_inventory, constraints, and any stated counterfactual intervention.
5. ask_user for missing scenario slots; without a reply, fail honestly or document assumptions.
6. End by calling a typed terminal (submit_solution / submit_clarification / submit_abstention).
7. Do not claim access to hidden gold, oracle trajectories, or test-split metrics.

Tool-calling format (critical):
- Prefer native tool/function calling when available.
- If text mode: ONE raw JSON object only.
"""

# Backward-compatible alias used by older imports / wildfire tests
SYSTEM_PROMPT = SYSTEM_PROMPT_WILDFIRE


def _uses_olmo3_tool_profile() -> bool:
    blob = " ".join(
        [
            os.environ.get("HW_LLM_PROFILE", ""),
            os.environ.get("ICLR_MODEL_ARM", ""),
            os.environ.get("VLLM_MODEL_ID", ""),
        ]
    ).lower()
    return "olmo3" in blob or "olmo-3" in blob


def finalize_system_prompt(base: str) -> str:
    if _uses_olmo3_tool_profile():
        return f"{base.rstrip()}\n\n{OLMO3_TOOL_FORMAT_APPENDIX}"
    return base


def system_prompt_for_task(
    task: Mapping[str, Any],
    *,
    controller_mode: bool = False,
    vce_mode: bool = False,
) -> str:
    meta = task.get("metadata") or {}
    if vce_mode:
        try:
            from hazardweaver.hwa.experiments.agent_strict_v2 import agent_strict_v2_enabled

            if agent_strict_v2_enabled():
                return finalize_system_prompt(SYSTEM_PROMPT_VCE_STRICT)
        except ImportError:
            pass
        return finalize_system_prompt(SYSTEM_PROMPT_VCE)
    if meta.get("hwb_headline_inventory"):
        return finalize_system_prompt(
            SYSTEM_PROMPT_HEADLINE_CONTROLLER
            if controller_mode
            else SYSTEM_PROMPT_HEADLINE_REACT
        )
    if controller_mode:
        return finalize_system_prompt(SYSTEM_PROMPT_CONTROLLER)
    domain = str(task.get("domain") or "").lower()
    family = str(task.get("task_family") or "").lower()
    if domain == "pfdf":
        return finalize_system_prompt(SYSTEM_PROMPT_PFDF)
    if family in {"compose_hcg", "hcg_composition", "multi_step_compose"}:
        return finalize_system_prompt(SYSTEM_PROMPT_COMPOSE_HCG)
    return finalize_system_prompt(SYSTEM_PROMPT_WILDFIRE)


def build_instance_prompt(task: Mapping[str, Any], *, pack_root: str) -> str:
    """Build the user instance message from a solver_view task (no gold).

    P0-4 whitelist: opaque task_id, goal, scrubbed solver_visible, success_criteria only.
    Never include stratum / eligibility / anchor_id / _evaluator.
    """
    if "gold" in task:
        raise ValueError("solver task must not contain gold")
    meta = task.get("metadata") or {}
    domain = task.get("domain")
    clarify_mode = task.get("clarify_mode")
    instructions = (
        "Solve the user_facing_goal using tools. "
        "Finish with a typed terminal: submit_solution, submit_clarification, "
        "or submit_abstention (all three always available). "
        "Hints are non-binding. "
    )
    if str(clarify_mode) == "decision":
        instructions += (
            "Clarify-Decision mode: if information is missing, call "
            "submit_clarification with the single needed slot_id and question; "
            "you do not need a user reply to terminate. "
        )
    elif str(clarify_mode) == "resolution":
        instructions += (
            "Clarify-Resolution mode: call ask_user with the EXACT missing slot "
            "token from underspec_nl (e.g. event_time_window, not a shortened "
            "alias). After status=answered, call submit_clarification with that "
            "slot_id and question — do NOT keep re-asking. "
            "Wrong-slot replies return required_slots — copy them exactly. "
        )
    elif str(task.get("task_layer")) == "scenario_task":
        instructions += "Scenario layer: ask_user if critical slots are missing. "
    if str(domain or "").lower() == "pfdf":
        instructions += "For PFDF, sample_id ≡ record_id. "
    try:
        from hazardweaver.hwa.experiments.agent_strict_v2 import strict_instance_instructions

        strict_note = strict_instance_instructions(task)
        if strict_note:
            instructions += strict_note + " "
    except ImportError:
        pass
    if meta.get("hwb_headline_inventory"):
        cond = str(meta.get("same_llm_condition") or "")
        edges = (task.get("solver_visible") or {}).get("inputs", {}).get("allowed_edge_ids") or []
        non_schema = [e for e in edges if not str(e).startswith("schema_")]
        if cond == "full_hwa" or task.get("solver_visible", {}).get("controller_mode"):
            try:
                from hazardweaver.hwa.experiments.agent_strict_v2 import agent_strict_v2_enabled

                strict_v2 = agent_strict_v2_enabled()
            except ImportError:
                strict_v2 = False
            if strict_v2:
                instructions += (
                    "Headline Agent-Strict v2 (SCC): call controller_commit_route with route_id "
                    "AND explicit handles (record_id and/or scenario_id+split from user_facing_goal) "
                    "in one tool call — server runs admissibility check, commit, and execute. "
                    "Then submit_solution with route_id, execution_id, final_artifact_id from the result. "
                    "Skip separate enumerate/propose/run_capability when handles are known. "
                    "On bind failure, submit_abstention — do not use submit_answer or submit. "
                )
            else:
                instructions += (
                    "Headline controller mode: after controller_commit_route, copy execution_id and "
                    "final_artifact_id from result.execution; never call get_execution_result with "
                    "placeholder ids. On execution failure, submit_clarification (not NO_LEGAL_ROUTE abstain). "
                )
            if meta.get("headline_route_profile") == "g1_v1":
                cbr = str(meta.get("ds_cbr_capability") or "")
                if cbr:
                    rid = str(meta.get("ds_cbr_route_id") or f"route:cap:{cbr}")
                    instructions += (
                        "G1 CBR warm-start (validation retrieval prior, not the answer): "
                        f"capability {cbr} ({rid}) is the top DS-style route hint — "
                        "call controller_enumerate_routes, propose that route if admissible, "
                        "controller_commit_route, then submit_solution. "
                        "Do not submit_abstention while admissible routes exist. "
                    )
                else:
                    instructions += (
                        "G1 mode: enumerate allowed-edge routes before abstaining; "
                        "prefer the highest validation_utility route from enumerate output. "
                    )
        else:
            instructions += (
                "Headline ReAct mode: allowed_edge_ids lists authorized capabilities only. "
                "Use run_capability for applicable caps, then submit_solution. "
            )
            if len(non_schema) == 1:
                instructions += (
                    "Single non-schema capability listed — run it then submit_solution with tool outputs. "
                )

    payload = {
        "task_id": task.get("task_id"),
        "domain": domain,
        "user_facing_goal": task.get("user_facing_goal"),
        "solver_visible": task.get("solver_visible"),
        "success_criteria": task.get("success_criteria"),
        "pack_root": pack_root,
        "instructions": instructions.strip(),
    }
    if clarify_mode:
        payload["clarify_mode"] = clarify_mode
    label = (
        "PFDF water–fire agent bench task"
        if str(domain or "").lower() == "pfdf"
        else f"HWB headline task (track={task.get('headline_target') or meta.get('track') or 'multi'})"
        if meta.get("hwb_headline_inventory")
        else "Wildfire agent bench task"
    )
    return (
        f"{label} (solver view — gold stripped):\n"
        + json.dumps(payload, indent=2, ensure_ascii=False)
    )


def truncate_obs(obj: Any, max_chars: int) -> str:
    text = json.dumps(obj, ensure_ascii=False, default=str)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 20] + "...[truncated]"
