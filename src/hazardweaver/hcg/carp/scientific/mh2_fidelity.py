"""MH-2 scientific fidelity thresholds (DL-046) — no heuristic parity shortcuts."""

from __future__ import annotations

REPLAY_PARITY_MIN = 0.95
VBCI_REPLAY_PARITY_MIN = 0.95
PRIMARY_EVENT_ID = "ci38457511"
REFERENCE_MANIFEST = "reference_manifest.json"
FORBIDDEN_EVALUATOR_SUBSTRINGS = ("heuristic", "numpy_fusion", "not_installed", "dev_replay")


def evaluator_allowed(evaluator: str) -> bool:
    ev = str(evaluator or "").lower()
    return not any(tok in ev for tok in FORBIDDEN_EVALUATOR_SUBSTRINGS)


def parity_ok(parity: float | None, *, minimum: float = REPLAY_PARITY_MIN) -> bool:
    if parity is None:
        return False
    try:
        return float(parity) >= minimum
    except (TypeError, ValueError):
        return False


def metrics_fidelity_ok(metrics: dict) -> tuple[bool, str]:
    if metrics.get("synthetic_only") is not False:
        return False, "synthetic_only is not false"
    evaluator = str(metrics.get("evaluator", ""))
    if not evaluator_allowed(evaluator):
        return False, f"forbidden evaluator: {evaluator}"
    if not metrics.get("reference_present"):
        return False, "reference_present is not true"
    ref_kind = str(metrics.get("reference_kind", ""))
    if not ref_kind:
        return False, "missing reference_kind"
    parity = metrics.get("replay_parity")
    if parity is None:
        parity = metrics.get("metric_value")
    minimum = VBCI_REPLAY_PARITY_MIN if "vbci" in evaluator.lower() else REPLAY_PARITY_MIN
    if not parity_ok(parity, minimum=minimum):
        return False, f"replay_parity {parity} < {minimum}"
    return True, "ok"
