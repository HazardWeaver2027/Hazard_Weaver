"""OpenAI tool spec entries for controller semantic tools."""

from __future__ import annotations

from typing import Any, Dict, List

from hazardweaver.hwa.contracts.strict_tool_models_v1 import strict_tool_parameters

CONTROLLER_TOOL_SPECS: List[Dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "controller_enumerate_routes",
            "description": (
                "List candidate scientific routes with A_sci/A_cap admissibility annotations. "
                "Does not execute anything."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "sources": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Source artifact ids (optional; defaults to task sample refs)",
                    },
                    "target": {
                        "type": "string",
                        "description": "Target artifact id (optional; defaults to goal_artifacts[0])",
                    },
                    "max_paths": {"type": "integer", "default": 8},
                    "admissible_only": {
                        "type": "boolean",
                        "default": False,
                        "description": "If true, return only admissible routes",
                    },
                },
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "controller_propose_route",
            "description": (
                "Propose a route for scientific trade-off review. Does not execute. "
                "Route must be admissible (A_sci ∧ A_cap)."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "route_id": {"type": "string"},
                    "rationale": {"type": "string"},
                },
                "required": ["route_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "controller_commit_route",
            "description": (
                "Commit a previously proposed route for deterministic controller execution. "
                "Issues execution token; controller runs the route."
            ),
            "parameters": strict_tool_parameters("controller_commit_route"),
        },
    },
    {
        "type": "function",
        "function": {
            "name": "controller_get_route_status",
            "description": "Query active/pending route and session checkpoint state.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
]
