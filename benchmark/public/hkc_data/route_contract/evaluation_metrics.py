"""HKC fused evaluation metrics (P1)."""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


def bootstrap_accuracy_ci(
    correct_flags: Sequence[bool],
    *,
    n_bootstrap: int = 1000,
    seed: int = 20260828,
    ci: float = 0.95,
) -> Dict[str, float]:
    """Bootstrap 95% CI on accuracy (Claude §4.1)."""
    n = len(correct_flags)
    if n == 0:
        return {"accuracy": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "n": 0}
    rng = random.Random(seed)
    accs: List[float] = []
    flags = list(correct_flags)
    for _ in range(n_bootstrap):
        sample = [flags[rng.randint(0, n - 1)] for _ in range(n)]
        accs.append(sum(sample) / n)
    accs.sort()
    lo_idx = int((1 - ci) / 2 * n_bootstrap)
    hi_idx = int((1 + ci) / 2 * n_bootstrap) - 1
    acc = sum(flags) / n
    return {
        "accuracy": acc,
        "ci_lower": accs[lo_idx],
        "ci_upper": accs[hi_idx],
        "n": n,
        "n_bootstrap": n_bootstrap,
    }


def false_admission_rate(
    rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Primary intrinsic metric (ChatGPT §22)."""
    gold_inapplicable = [r for r in rows if not r.get("expected_applicable")]
    if not gold_inapplicable:
        return {"far": None, "n": 0}
    false_admits = sum(1 for r in gold_inapplicable if r.get("predicted_adplicable"))
    return {
        "false_admission_rate": false_admits / len(gold_inapplicable),
        "n_gold_inapplicable": len(gold_inapplicable),
        "n_false_admit": false_admits,
    }


def risk_coverage_curve(
    scores: Sequence[float],
    labels: Sequence[bool],
    *,
    n_points: int = 20,
) -> List[Dict[str, float]]:
    """Risk-coverage points for admission layer (ChatGPT §23)."""
    if not scores:
        return []
    paired = sorted(zip(scores, labels), key=lambda x: -x[0])
    n = len(paired)
    curve: List[Dict[str, float]] = []
    for i in range(1, n_points + 1):
        k = max(1, int(n * i / n_points))
        subset = paired[:k]
        coverage = k / n
        prec = sum(1 for _, lab in subset if lab) / k if subset else 0.0
        curve.append({"coverage": coverage, "precision": prec, "threshold_rank": k})
    return curve


def load_per_arm_rows(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    arms = data.get("per_arm_rows") or data.get("arms") or {}
    if isinstance(arms, dict) and "no_hkc" in arms and "correct" in arms.get("no_hkc", {}):
        return {}
    return arms if isinstance(arms, dict) else {}


def augment_rq2_with_bootstrap(
    rq2_path: Path,
    out_path: Optional[Path] = None,
) -> Dict[str, Any]:
    data = json.loads(rq2_path.read_text(encoding="utf-8"))
    per_arm = data.get("per_arm_rows") or {}
    bootstrap: Dict[str, Any] = {}
    for arm, rows in per_arm.items():
        if not isinstance(rows, list):
            continue
        flags = [bool(r.get("predicted_applicable") == r.get("expected_applicable")) for r in rows]
        bootstrap[arm] = bootstrap_accuracy_ci(flags)
    data["bootstrap_ci"] = bootstrap
    out = out_path or rq2_path
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return data
