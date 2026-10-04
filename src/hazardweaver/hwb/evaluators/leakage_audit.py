"""Three-view isolation and leakage audit for HWB TaskPacks."""

from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Set

from hazardweaver.hwb.agent_tasks import assert_no_gold_in_solver_view, solver_view

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_RULES_PATH = ROOT / "benchmark/schemas/anti_leak_rules.json"

HWB_FORBIDDEN_PLANNER_KEYS: Set[str] = {
    "reference_view",
    "accepted_witnesses",
    "reference_score",
    "selected_route_id",
    "expected_action",
    "checker",
    "witness",
    "evaluation_tier",
    "stratum",
    "gold",
    "hcg_oracle",
    "anchor_id",
    "eligibility",
    "paper_main_table_eligible",
    "strength_claim_eligible",
}

PLANNER_WHITELIST_TOP_KEYS: Set[str] = {
    "schema_version",
    "taskpack_id",
    "headline_target",
    "coverage_tier",
    "solver_view",
    "hib_is_grader",
    "metadata",
}


@dataclass
class LeakageFinding:
    severity: str  # critical | warning
    code: str
    detail: str


@dataclass
class LeakageReport:
    taskpack_id: str
    findings: List[LeakageFinding] = field(default_factory=list)

    @property
    def critical_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == "critical")

    @property
    def passed(self) -> bool:
        return self.critical_count == 0


def _load_rules(path: Optional[Path] = None) -> Dict[str, Any]:
    p = path or DEFAULT_RULES_PATH
    if not p.is_file():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def planner_view(taskpack: Mapping[str, Any]) -> Dict[str, Any]:
    """Planner-visible slice: solver_view + whitelisted metadata only."""
    view: Dict[str, Any] = {}
    for key in PLANNER_WHITELIST_TOP_KEYS:
        if key in taskpack:
            view[key] = deepcopy(taskpack[key])
    if "solver_view" not in view:
        view["solver_view"] = deepcopy(taskpack.get("solver_view") or {})
    return view


def executor_view(observation: Mapping[str, Any]) -> Dict[str, Any]:
    """Strip test/gold/reference fields from tool observations."""
    rules = _load_rules()
    forbidden_prefixes = tuple(rules.get("forbidden_key_prefixes") or ["gold_", "oracle_", "test_", "reported_test_"])
    forbidden_exact = set(rules.get("forbidden_exact_keys_extra") or [])
    forbidden_keys = set(rules.get("forbidden_keys") or []) | HWB_FORBIDDEN_PLANNER_KEYS

    def _scrub(obj: Any) -> Any:
        if isinstance(obj, Mapping):
            out: Dict[str, Any] = {}
            for k, v in obj.items():
                ks = str(k)
                if ks in forbidden_keys or ks in forbidden_exact:
                    continue
                if any(ks.startswith(p) for p in forbidden_prefixes):
                    continue
                out[ks] = _scrub(v)
            return out
        if isinstance(obj, list):
            return [_scrub(v) for v in obj]
        return obj

    return _scrub(observation)  # type: ignore[return-value]


def evaluator_view(taskpack: Mapping[str, Any]) -> Dict[str, Any]:
    """Evaluator-only slice."""
    return {
        "taskpack_id": taskpack.get("taskpack_id"),
        "reference_view": deepcopy(taskpack.get("reference_view") or {}),
    }


def _scan_forbidden_keys(obj: Any, forbidden: Set[str], *, path: str = "") -> List[str]:
    hits: List[str] = []
    if isinstance(obj, Mapping):
        for k, v in obj.items():
            p = f"{path}.{k}" if path else str(k)
            if str(k) in forbidden:
                hits.append(p)
            hits.extend(_scan_forbidden_keys(v, forbidden, path=p))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            hits.extend(_scan_forbidden_keys(v, forbidden, path=f"{path}[{i}]"))
    return hits


def audit_taskpack_leakage(
    taskpack: Mapping[str, Any],
    *,
    rules_path: Optional[Path] = None,
) -> LeakageReport:
    """Audit planner view for critical leakage (critical=0 required)."""
    taskpack_id = str(taskpack.get("taskpack_id") or "unknown")
    findings: List[LeakageFinding] = []
    rules = _load_rules(rules_path)
    forbidden = set(rules.get("forbidden_keys") or []) | HWB_FORBIDDEN_PLANNER_KEYS

    pv = planner_view(taskpack)
    leaked_keys = _scan_forbidden_keys(pv, forbidden)
    for key_path in leaked_keys:
        findings.append(
            LeakageFinding(
                severity="critical",
                code="forbidden_key_in_planner_view",
                detail=key_path,
            )
        )

    if "reference_view" in taskpack and "reference_view" in pv:
        findings.append(
            LeakageFinding(
                severity="critical",
                code="reference_view_in_planner",
                detail="reference_view must not appear in planner_view",
            )
        )

    # Reuse HWB gold leak assert when task has gold/hcg_oracle (agent_bench compat)
    compat = dict(taskpack)
    if "user_facing_goal" not in compat and (taskpack.get("solver_view") or {}).get("user_goal"):
        compat["user_facing_goal"] = taskpack["solver_view"]["user_goal"]
    if "solver_visible" not in compat:
        compat["solver_visible"] = {"inputs": {}, "allowed_inventory": {}}
    sv = solver_view(compat) if "gold" in compat or "hcg_oracle" in compat else pv
    try:
        if "gold" in compat or "hcg_oracle" in compat:
            assert_no_gold_in_solver_view(sv, full=compat)
    except AssertionError as exc:
        findings.append(
            LeakageFinding(severity="critical", code="gold_leak", detail=str(exc))
        )

    # NL substring scan on planner-facing text
    texts: List[str] = []
    solver = pv.get("solver_view") or {}
    if isinstance(solver.get("user_goal"), str):
        texts.append(solver["user_goal"])
    for pat in rules.get("forbidden_substrings_in_string_values") or []:
        for text in texts:
            if pat.lower() in text.lower():
                findings.append(
                    LeakageFinding(
                        severity="critical",
                        code="forbidden_substring",
                        detail=f"{pat} in planner text",
                    )
                )

    return LeakageReport(taskpack_id=taskpack_id, findings=findings)


def load_taskpack(path: Path | str) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
