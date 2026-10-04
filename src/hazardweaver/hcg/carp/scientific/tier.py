"""Scientific validation_tier eligibility (DL-032)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from hazardweaver.hcg.carp.scientific.manifest_checks import (
    holdout_ok,
    load_manifest,
    official_test_ok,
)
from hazardweaver.hcg.carp.scientific.paths import (
    cap_scientific_dir,
    hwb_holdout_manifest,
    official_test_manifest,
    scientific_signoff_path,
)
from hazardweaver.hcg.carp.scientific.pi_scientific import load_scientific_signoff
from hazardweaver.hcg.carp.scientific.mh2_fidelity import metrics_fidelity_ok
from hazardweaver.hcg.carp.scientific.pin_resolver import pin_commit_ok, resolve_pin_path

DEV_REPLAY_SUFFIX = "_dev_replay_v1"


def _check_native_metrics(metrics: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if metrics is None:
        return {"ok": False, "reason": "missing native_metrics.json"}
    if metrics.get("blocked") is True:
        blocker = metrics.get("blocker")
        if not blocker or blocker == metrics.get("validation_tier"):
            blocker = metrics.get("reason") or metrics.get("note") or "runtime_blocked"
        return {"ok": False, "reason": f"runtime blocked: {blocker}"}
    if metrics.get("validation_tier") == "blocked":
        return {"ok": False, "reason": "validation_tier=blocked"}
    if metrics.get("ok") is False:
        return {"ok": False, "reason": metrics.get("note") or metrics.get("blocker") or "native_metrics ok=false"}
    if metrics.get("synthetic_only") is not False:
        return {"ok": False, "reason": "native_metrics synthetic_only is not false"}
    data_source = str(metrics.get("data_source") or "")
    if data_source.endswith(DEV_REPLAY_SUFFIX):
        return {
            "ok": False,
            "reason": f"data_source ends with {DEV_REPLAY_SUFFIX}",
            "data_source": data_source,
        }
    return {"ok": True, "data_source": data_source}


def _check_recipe_log(cap_dir: Path) -> Dict[str, Any]:
    log_path = cap_dir / "recipe_run.log"
    if not log_path.is_file():
        return {"ok": False, "reason": "missing recipe_run.log", "path": str(log_path)}
    data = json.loads(log_path.read_text(encoding="utf-8"))
    returncode = data.get("returncode")
    if returncode != 0:
        return {
            "ok": False,
            "reason": f"recipe_run.log returncode={returncode}",
            "path": str(log_path),
        }
    return {"ok": True, "path": str(log_path), "returncode": returncode}


def check_scientific_eligibility(
    taskpack: str,
    capability_id: str,
    *,
    batch2_root: Path,
    pin_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Return {eligible, checks, reasons} for validation_tier=scientific."""
    reasons: List[str] = []
    checks: Dict[str, Any] = {}

    official = load_manifest(official_test_manifest(taskpack))
    ok_official, msg_official = official_test_ok(official)
    checks["scientific_official_test"] = {
        "ok": ok_official,
        "reason": msg_official,
        "path": str(official_test_manifest(taskpack)),
    }
    if not ok_official:
        reasons.append(msg_official)

    signoff = load_scientific_signoff(taskpack)
    holdout = load_manifest(hwb_holdout_manifest(taskpack))
    ok_holdout, msg_holdout = holdout_ok(holdout, signoff)
    checks["scientific_holdout"] = {
        "ok": ok_holdout,
        "reason": msg_holdout,
        "path": str(hwb_holdout_manifest(taskpack)),
    }
    checks["scientific_signoff"] = {
        "ok": signoff is not None and signoff.get("approved") is True
        and signoff.get("approval_tier") == "scientific",
        "path": str(scientific_signoff_path(taskpack)),
        "approval_tier": (signoff or {}).get("approval_tier"),
    }
    if not ok_holdout:
        reasons.append(msg_holdout)

    metrics_path = cap_scientific_dir(taskpack, capability_id, runs_root=batch2_root) / "native_metrics.json"
    metrics = None
    if metrics_path.is_file():
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    checks["scientific_metrics"] = _check_native_metrics(metrics)
    if not checks["scientific_metrics"]["ok"]:
        reasons.append(checks["scientific_metrics"].get("reason", "native_metrics check failed"))

    if taskpack == "MH-2" and metrics:
        ok_fidelity, msg_fidelity = metrics_fidelity_ok(metrics)
        checks["scientific_fidelity"] = {"ok": ok_fidelity, "reason": msg_fidelity}
        if not ok_fidelity:
            reasons.append(msg_fidelity)

    cap_dir = cap_scientific_dir(taskpack, capability_id, runs_root=batch2_root)
    checks["scientific_recipe"] = _check_recipe_log(cap_dir)
    if not checks["scientific_recipe"]["ok"]:
        reasons.append(checks["scientific_recipe"].get("reason", "recipe check failed"))

    resolved_pin = pin_path or resolve_pin_path(taskpack, capability_id)
    ok_pin, msg_pin, pin_data = pin_commit_ok(resolved_pin)
    checks["scientific_pin"] = {
        "ok": ok_pin,
        "reason": msg_pin,
        "path": str(resolved_pin) if resolved_pin else None,
        "commit": (pin_data or {}).get("repo_commit") or (pin_data or {}).get("version_or_commit"),
    }
    if not ok_pin:
        reasons.append(msg_pin)

    eligible = all(v.get("ok") for v in checks.values())
    return {"eligible": eligible, "checks": checks, "reasons": reasons}
