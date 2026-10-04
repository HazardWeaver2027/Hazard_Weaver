"""Baseline fairness enforcer — reads HWB_BASELINE_FAIRNESS_PROTOCOL_v1.json."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PROTOCOL = (
    ROOT / "docs/engineering/benchmark/HWB_BASELINE_FAIRNESS_PROTOCOL_v1.json"
)


@dataclass
class FairnessCheckResult:
    passed: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)


def load_fairness_protocol(path: Optional[Path] = None) -> Dict[str, Any]:
    src = path or DEFAULT_PROTOCOL
    return json.loads(src.read_text(encoding="utf-8"))


def verify_baseline_run_manifest(
    manifest: Mapping[str, Any],
    *,
    protocol: Optional[Mapping[str, Any]] = None,
) -> FairnessCheckResult:
    """Validate a baseline run manifest against frozen fairness protocol fields."""
    pol = dict(protocol or load_fairness_protocol())
    factors = pol.get("control_factors") or {}
    errors: List[str] = []
    warnings: List[str] = []

    wall_limit = factors.get("wall_time_limit_seconds")
    reported_wall = manifest.get("wall_time_seconds")
    if wall_limit is not None and reported_wall is not None:
        if float(reported_wall) > float(wall_limit):
            errors.append("fairness:wall_time_exceeded")

    token_budget = factors.get("model_api_budget_tokens")
    reported_tokens = manifest.get("api_tokens_used") or manifest.get("token_budget_used")
    if token_budget is not None and reported_tokens is not None:
        if int(reported_tokens) > int(token_budget):
            errors.append("fairness:token_budget_exceeded")

    retry_max = factors.get("retry_policy_max")
    reported_retries = manifest.get("retry_count")
    if retry_max is not None and reported_retries is not None:
        if int(reported_retries) > int(retry_max):
            errors.append("fairness:retry_limit_exceeded")

    if manifest.get("evaluator_access") and manifest.get("evaluator_access") != factors.get(
        "evaluator_access"
    ):
        warnings.append("fairness:evaluator_access_mismatch")

    if pol.get("denominator_rules", {}).get("unsupported_counts_as_not_completed"):
        if manifest.get("unsupported") and manifest.get("counted_as_completed"):
            errors.append("fairness:unsupported_counted_as_completed")

    return FairnessCheckResult(passed=len(errors) == 0, errors=errors, warnings=warnings)
