"""Controller tool surface policy (shared by tool_specs and ScientificController)."""

from __future__ import annotations

CONTROLLER_EXEC_TOOLS = frozenset(
    {
        "run_capability",
        "run_predictor",
        "tool_hcg_run_path",
    }
)

CONTROLLER_SEMANTIC_TOOLS = frozenset(
    {
        "controller_enumerate_routes",
        "controller_propose_route",
        "controller_commit_route",
        "controller_get_route_status",
    }
)

CONTROLLER_READONLY_TOOLS = frozenset(
    {
        "list_inventory",
        "read_card",
        "load_sample",
        "inspect_artifact",
        "get_execution_result",
        "tool_hcg_explain_edge",
        "tool_hcg_find_paths",
        "ask_user",
        "submit_answer",
        "submit",
        "submit_solution",
        "submit_clarification",
        "submit_abstention",
    }
)
