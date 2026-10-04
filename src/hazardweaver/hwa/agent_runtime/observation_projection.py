"""P0-3: project tool observations to planner/executor views (no test_* leakage).

Evaluator-only fields (test metrics, gold, oracle) must never enter solver observations.
Claim Guard: projection ≠ STRENGTH PASS.
"""

from __future__ import annotations

from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Set, Tuple

# Exact keys stripped from planner/executor observations
FORBIDDEN_EXACT_KEYS = frozenset(
    {
        "test_metric",
        "n_test",
        "test_used_for_selection",
        "reported_test_metric",
        "test_accuracy",
        "mae_test",
        "test_score",
        "test_split",
    }
)

FORBIDDEN_KEY_PREFIXES = ("test_", "reported_test_")


def is_forbidden_solver_key(key: str) -> bool:
    k = str(key or "")
    if k in FORBIDDEN_EXACT_KEYS:
        return True
    return any(k.startswith(p) for p in FORBIDDEN_KEY_PREFIXES)


def count_forbidden_keys(obj: Any, *, path: str = "") -> List[str]:
    """Return dotted paths of forbidden keys found in nested structures."""
    hits: List[str] = []
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            sk = str(k)
            p = f"{path}.{sk}" if path else sk
            if is_forbidden_solver_key(sk):
                hits.append(p)
            hits.extend(count_forbidden_keys(v, path=p))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            hits.extend(count_forbidden_keys(v, path=f"{path}[{i}]"))
    return hits


def project_planner_view(obj: Any) -> Any:
    """Deep-copy projection: drop forbidden keys; scrub split=='test' values."""
    if isinstance(obj, Mapping):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            sk = str(k)
            if is_forbidden_solver_key(sk):
                continue
            if sk == "split" and str(v).lower() == "test":
                out[sk] = "held_out_hidden"
                continue
            out[sk] = project_planner_view(v)
        return out
    if isinstance(obj, list):
        return [project_planner_view(v) for v in obj]
    return obj


def project_executor_view(obj: Any) -> Any:
    """Executor view currently shares the same anti-test projection as planner."""
    return project_planner_view(obj)


def assert_no_test_leak(obj: Any, *, context: str = "observation") -> None:
    hits = count_forbidden_keys(obj)
    if hits:
        raise AssertionError(f"{context}: solver-visible test_* leak at {hits[:12]}")
