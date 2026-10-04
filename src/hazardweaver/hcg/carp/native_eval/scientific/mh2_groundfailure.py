"""MH-2 Phase C scientific eval — official groundfailure + VBCI replay (DL-046)."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from hazardweaver.hcg.carp.batch2.replay_certificate import write_metrics, write_replay_manifest
from hazardweaver.hcg.carp.native_eval.blocked import eval_blocked_generic, write_blocked
from hazardweaver.hcg.carp.native_eval.scientific.mh2_gfail_worker import run_gfail_model
from hazardweaver.hcg.carp.native_eval.scientific.mh2_vbci_worker import run_vbci_ridgecrest
from hazardweaver.hcg.carp.scientific.mh2_fidelity import REPLAY_PARITY_MIN
from hazardweaver.hcg.carp.scientific.mh2_data import (
    TASKPACK,
    VBCI_EVENT_ID,
    event_dir,
    event_shakegrid_path,
    load_split_manifest,
)
from hazardweaver.hcg.carp.scientific.paths import SCIENTIFIC_RUNS_ROOT, cap_scientific_dir, taskpack_repo_pin

GF_CAPS = {
    "CAP-MH2-01": ("RF-EMPIRICAL-AREAL-COVERAGE-LOG", "nowicki_jessee_2018", "replay_parity"),
    "CAP-MH2-02": ("RF-EMPIRICAL-CELL-OCCURRENCE-LO", "nowicki_2014", "replay_parity"),
    "CAP-MH2-03": ("RF-EMPIRICAL-AREAL-COVERAGE-MOD", "godt_2008", "replay_parity"),
}
NEWMARK_CAP = "CAP-MH2-04"
BAYES_CAP = "CAP-MH2-05"
LIFELINE_CAP = "CAP-MH2-06"
DATA_SOURCE = "groundfailure_official_test_v1"
GF_REPO_COMMIT = "1.3.2"


def _official_events(split: str = "official_test") -> List[Dict[str, Any]]:
    manifest = load_split_manifest(split) or {}
    rows = []
    for eid in manifest.get("event_ids") or []:
        shake = event_shakegrid_path(split, str(eid))
        if not shake.is_file():
            continue
        rows.append(
            {
                "event_id": str(eid),
                "split": split,
                "shakefile": shake,
                "reference_dir": event_dir(split, str(eid)) / "reference",
            }
        )
    return rows


def _write_recipe_log(
    cap_dir: Path,
    *,
    capability_id: str,
    command: str,
    returncode: int = 0,
    stderr: str = "",
) -> Path:
    cap_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "taskpack_id": TASKPACK,
        "capability_id": capability_id,
        "official_command": command,
        "returncode": returncode,
        "status": "executed" if returncode == 0 else "failed",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "stderr_tail": stderr[-500:] if stderr else "",
        "note": "DL-046 faithful official replay",
    }
    path = cap_dir / "recipe_run.log"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _write_predictions(cap_dir: Path, *, split: str, event_id: str, prob: np.ndarray) -> None:
    pred_dir = cap_dir / "predictions" / split
    pred_dir.mkdir(parents=True, exist_ok=True)
    np.save(pred_dir / f"{event_id}_prob.npy", prob.astype(np.float32))
    meta = pred_dir / f"{event_id}_meta.json"
    meta.write_text(
        json.dumps({"event_id": event_id, "n_cells": int(prob.size), "split": split}, indent=2) + "\n",
        encoding="utf-8",
    )


def _vbci_runtime_ready() -> bool:
    try:
        from hazardweaver.hcg.carp.scientific.mh2_vbci_env import verify_vbci_runtime

        return not verify_vbci_runtime(require_runner=True)
    except Exception:
        return False


def _metrics_from_gfail_result(
    *,
    score: float,
    metric_name: str,
    evaluator: str,
    note: str,
    n_events: int,
    reference_present: bool,
    reference_kind: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    metrics = {
        "metric_name": metric_name,
        "metric_value": score,
        "synthetic_only": False,
        "data_source": DATA_SOURCE,
        "evaluator": evaluator,
        "replay_parity": score,
        "reference_present": reference_present,
        "reference_kind": reference_kind,
        "note": note,
        "n_events": n_events,
    }
    if extra:
        metrics.update(extra)
    return metrics


def _gfail_runtime_ready() -> bool:
    try:
        from hazardweaver.hcg.carp.scientific.mh2_gfail_env import verify_runtime

        return not verify_runtime()
    except Exception:
        return False


def _eval_gf_cap(capability_id: str, *, out_base: Optional[Path] = None) -> Dict[str, Any]:
    if capability_id not in GF_CAPS:
        return eval_blocked_generic(TASKPACK, capability_id, "not a GF replay cap", out_base=out_base)
    if not _gfail_runtime_ready():
        return write_blocked(
            TASKPACK,
            capability_id,
            "groundfailure runtime not ready; run install_groundfailure_vendor_deps.sh",
            out_base=out_base,
        )

    family_id, model_key, metric_name = GF_CAPS[capability_id]
    events = _official_events("official_test")
    if not events:
        return write_blocked(TASKPACK, capability_id, "no official_test events with grid.xml", out_base=out_base)

    out_root = out_base or SCIENTIFIC_RUNS_ROOT
    cap_dir = cap_scientific_dir(TASKPACK, capability_id, runs_root=out_root)
    commands: List[str] = []
    evaluator = "gfail@groundfailure"
    event_scores: List[float] = []
    n_events = 0

    for ev in events:
        result = run_gfail_model(
            shakefile=ev["shakefile"],
            model_key=model_key,
            out_dir=ev["shakefile"].parent.parent / "gfail" / model_key,
            reference_dir=ev["reference_dir"],
        )
        if not result.get("ok"):
            if ev["event_id"] == VBCI_EVENT_ID:
                return write_blocked(
                    TASKPACK,
                    capability_id,
                    result.get("error", "gfail replay failed"),
                    out_base=out_base,
                )
            continue
        prob = result["prob"]
        commands.append(str(result.get("official_command", "")))
        evaluator = str(result.get("evaluator", evaluator))
        _write_predictions(cap_dir, split="test", event_id=ev["event_id"], prob=prob)
        n_events += 1
        event_scores.append(float(result["replay_parity"]))

    if not event_scores:
        return write_blocked(
            TASKPACK,
            capability_id,
            f"no successful gfail replays for {model_key}",
            out_base=out_base,
        )

    score = float(sum(event_scores) / len(event_scores))
    primary_result = next((ev for ev in events if ev["event_id"] == VBCI_EVENT_ID), None)
    reference_present = False
    reference_kind = ""
    if primary_result:
        ref_result = run_gfail_model(
            shakefile=primary_result["shakefile"],
            model_key=model_key,
            out_dir=primary_result["shakefile"].parent.parent / "gfail" / model_key,
            reference_dir=primary_result["reference_dir"],
        )
        if ref_result.get("ok"):
            reference_present = bool(ref_result.get("reference_present"))
            reference_kind = str(ref_result.get("reference_kind", ""))
    if score < REPLAY_PARITY_MIN:
        return write_blocked(
            TASKPACK,
            capability_id,
            f"primary replay_parity {score} < {REPLAY_PARITY_MIN}",
            out_base=out_base,
        )

    cmd = "; ".join(commands)
    _write_recipe_log(cap_dir, capability_id=capability_id, command=cmd, returncode=0)
    metrics = _metrics_from_gfail_result(
        score=score,
        metric_name=metric_name,
        evaluator=evaluator,
        note=(
            f"official gfail replay ({model_key}); n_events={n_events}; "
            f"mean_replay_parity={score:.4f}; primary_event={VBCI_EVENT_ID}"
        ),
        n_events=n_events,
        reference_present=reference_present,
        reference_kind=reference_kind,
        extra={"primary_event": VBCI_EVENT_ID, "event_scores": event_scores},
    )
    write_metrics(TASKPACK, capability_id, metrics, base=out_root)
    write_replay_manifest(
        TASKPACK,
        capability_id,
        family_id=family_id,
        exec_ok=True,
        metric_name=metric_name,
        metric_value=score,
        notes=metrics["note"],
        base=out_root,
    )
    return {"ok": True, "metrics": metrics}


def _eval_newmark(capability_id: str, *, out_base: Optional[Path] = None) -> Dict[str, Any]:
    if not _gfail_runtime_ready():
        return write_blocked(
            TASKPACK,
            capability_id,
            "groundfailure runtime not ready; run install_groundfailure_vendor_deps.sh",
            out_base=out_base,
        )

    family_id = "RF-MECHANISTIC-NEWMARK-DISP"
    model_key = "newmark"
    metric_name = "replay_parity"
    events = _official_events("official_test")
    if not events:
        return write_blocked(TASKPACK, capability_id, "no official_test events with grid.xml", out_base=out_base)

    out_root = out_base or SCIENTIFIC_RUNS_ROOT
    cap_dir = cap_scientific_dir(TASKPACK, capability_id, runs_root=out_root)
    commands: List[str] = []
    evaluator = "gfail@groundfailure"
    event_scores: List[float] = []
    n_events = 0

    for ev in events:
        result = run_gfail_model(
            shakefile=ev["shakefile"],
            model_key=model_key,
            out_dir=ev["shakefile"].parent.parent / "gfail" / model_key,
            reference_dir=ev["reference_dir"],
        )
        if not result.get("ok"):
            if ev["event_id"] == VBCI_EVENT_ID:
                return write_blocked(
                    TASKPACK,
                    capability_id,
                    result.get("error", "Newmark NMdisp replay failed"),
                    out_base=out_base,
                )
            continue
        prob = result["prob"]
        commands.append(str(result.get("official_command", "")))
        evaluator = str(result.get("evaluator", evaluator))
        _write_predictions(cap_dir, split="test", event_id=ev["event_id"], prob=prob)
        n_events += 1
        event_scores.append(float(result["replay_parity"]))

    if not event_scores:
        return write_blocked(
            TASKPACK,
            capability_id,
            f"no successful Newmark replays; primary_event={VBCI_EVENT_ID}",
            out_base=out_base,
        )

    score = float(sum(event_scores) / len(event_scores))
    if score < REPLAY_PARITY_MIN:
        return write_blocked(
            TASKPACK,
            capability_id,
            f"primary replay_parity {score} < {REPLAY_PARITY_MIN}",
            out_base=out_base,
        )

    cmd = "; ".join(commands)
    _write_recipe_log(cap_dir, capability_id=capability_id, command=cmd, returncode=0)
    metrics = _metrics_from_gfail_result(
        score=score,
        metric_name=metric_name,
        evaluator=evaluator,
        note=(
            f"official godt.NMdisp replay; n_events={n_events}; "
            f"mean_replay_parity={score:.4f}; primary_event={VBCI_EVENT_ID}"
        ),
        n_events=n_events,
        reference_present=False,
        reference_kind="",
        extra={"primary_event": VBCI_EVENT_ID, "event_scores": event_scores},
    )
    write_metrics(TASKPACK, capability_id, metrics, base=out_root)
    write_replay_manifest(
        TASKPACK,
        capability_id,
        family_id=family_id,
        exec_ok=True,
        metric_name=metric_name,
        metric_value=score,
        notes=metrics["note"],
        base=out_root,
    )
    return {"ok": True, "metrics": metrics}


def _official_command_for_cap(capability_id: str) -> str:
    pin = taskpack_repo_pin(TASKPACK)
    commit = GF_REPO_COMMIT
    if pin.is_file():
        commit = json.loads(pin.read_text()).get("repo_commit", commit)
    if capability_id in GF_CAPS:
        _, model, _ = GF_CAPS[capability_id]
        return f"LogisticModelBase model={model} repo={commit}"
    if capability_id == NEWMARK_CAP:
        return f"LogisticModelBase model=newmark repo={commit}"
    if capability_id == BAYES_CAP:
        return f"VBCI updating.m event={VBCI_EVENT_ID}"
    if capability_id == LIFELINE_CAP:
        return "hwb_mh2_lifeline_chain --network frozen_v1"
    return "gfail faithful replay"


def _eval_bayesian(capability_id: str, *, out_base: Optional[Path] = None) -> Dict[str, Any]:
    if not _vbci_runtime_ready():
        return write_blocked(
            TASKPACK,
            capability_id,
            "VBCI runtime not ready; install vendor deps and load octave/matlab",
            out_base=out_base,
        )

    out_root = out_base or SCIENTIFIC_RUNS_ROOT
    cap_dir = cap_scientific_dir(TASKPACK, capability_id, runs_root=out_root)
    vbci_out = cap_dir / "vbci_replay"
    ref_dir = event_dir("official_test", VBCI_EVENT_ID) / "vbci" / "reference_posterior"

    result = run_vbci_ridgecrest(out_dir=vbci_out, reference_dir=ref_dir if ref_dir.is_dir() else None)
    if not result.get("ok"):
        detail = str(result.get("error", "VBCI replay failed"))
        stderr = str(result.get("stderr", ""))
        if stderr:
            detail = f"{detail} | stderr_tail={stderr[-800:]}"
        return write_blocked(TASKPACK, capability_id, detail, out_base=out_base)

    score = float(result["bayesian_replay_parity"])
    cmd = _official_command_for_cap(capability_id)
    _write_recipe_log(cap_dir, capability_id=capability_id, command=cmd, returncode=0)
    _write_predictions(cap_dir, split="test", event_id=VBCI_EVENT_ID, prob=result["posterior"])

    metrics = {
        "metric_name": "bayesian_replay_parity",
        "metric_value": score,
        "synthetic_only": False,
        "data_source": DATA_SOURCE,
        "evaluator": result.get("evaluator", "vbci"),
        "replay_parity": score,
        "reference_present": bool(result.get("reference_present")),
        "reference_kind": str(result.get("reference_kind", "")),
        "note": f"VBCI official Ridgecrest SVI replay; primary_event={VBCI_EVENT_ID}",
        "n_events": 1,
        "vbci_primary_event": VBCI_EVENT_ID,
    }
    write_metrics(TASKPACK, capability_id, metrics, base=out_root)
    write_replay_manifest(
        TASKPACK,
        capability_id,
        family_id="RF-BAYESIAN-CAUSAL-UPDATE",
        exec_ok=True,
        metric_name="bayesian_replay_parity",
        metric_value=score,
        notes=metrics["note"],
        base=out_root,
    )
    return {"ok": True, "metrics": metrics}


def _eval_lifeline_chain(capability_id: str, *, out_base: Optional[Path] = None) -> Dict[str, Any]:
    from hazardweaver.hcg.carp.native_eval.scientific.mh2_lifeline_chain import (
        eval_lifeline_chain_scientific,
        write_lifeline_blocked_certificate,
    )

    result = eval_lifeline_chain_scientific(capability_id, out_base=out_base)
    if not result.get("ok") and result.get("blocked"):
        write_lifeline_blocked_certificate(out_base=out_base)
    return result


def eval_mh2_scientific(capability_id: str, *, out_base: Optional[Path] = None) -> Dict[str, Any]:
    if capability_id in GF_CAPS:
        return _eval_gf_cap(capability_id, out_base=out_base)
    if capability_id == NEWMARK_CAP:
        return _eval_newmark(capability_id, out_base=out_base)
    if capability_id == BAYES_CAP:
        return _eval_bayesian(capability_id, out_base=out_base)
    if capability_id == LIFELINE_CAP:
        return _eval_lifeline_chain(capability_id, out_base=out_base)
    return eval_blocked_generic(TASKPACK, capability_id, "unknown MH-2 cap", out_base=out_base)


def run_all_scientific(*, out_base: Optional[Path] = None) -> Dict[str, Any]:
    caps = list(GF_CAPS) + [NEWMARK_CAP, BAYES_CAP, LIFELINE_CAP]
    results = {}
    for cap in caps:
        results[cap] = eval_mh2_scientific(cap, out_base=out_base)
    return results
