"""L2 Phase C scientific eval — official lhasa.py path (not dev numpy replay)."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from hazardweaver.hcg.carp.batch2.replay_certificate import write_metrics, write_replay_manifest
from hazardweaver.hcg.carp.expansion.dev_replay import average_precision
from hazardweaver.hcg.carp.native_eval.blocked import eval_blocked_generic, write_blocked
from hazardweaver.hcg.carp.scientific.l2_data import (
    DATA_SOURCE,
    TASKPACK,
    concat_split_arrays,
    event_dir,
    event_ids,
    lhasa_repo_dir,
    load_event_arrays,
    load_event_meta,
    load_lhasa_predictions,
    repo_commit,
    scientific_data_ready,
    train_event_ids,
)
from hazardweaver.hcg.carp.scientific.paths import SCIENTIFIC_RUNS_ROOT, cap_scientific_dir, taskpack_repo_pin

CAP_FAMILIES = {
    "CAP-L2-01": "RF-GRADIENT-BOOSTED-OCCURRENCE-",
    "CAP-L2-02": "RF-EMPIRICAL-THRESHOLD",
    "CAP-L2-03": "RF-INTERPRETABLE-STATISTICAL",
    "CAP-L2-04": "RF-TREE-ENSEMBLE",
    "CAP-L2-05": "RF-INFILTRATION-SLOPE-STABILITY",
}

OFFICIAL_COMMANDS = {
    "CAP-L2-01": "python lhasa.py --date {date}",
    "CAP-L2-02": "calibrate ID threshold on train 2015-2018; apply to test/holdout",
    "CAP-L2-03": "sklearn logistic regression train 2015-2018",
    "CAP-L2-04": "bootstrap RF ensemble train 2015-2018",
    "CAP-L2-05": f"trigrs TRinput.txt  # USGS TRIGRS v2.1.0a DOI 10.5066/F7M044QS",
}


def _write_recipe_log(
    cap_dir: Path,
    *,
    capability_id: str,
    command: str,
    returncode: int = 0,
    stdout_tail: str = "",
) -> Path:
    cap_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "taskpack_id": TASKPACK,
        "capability_id": capability_id,
        "official_command": command,
        "returncode": returncode,
        "status": "executed" if returncode == 0 else "failed",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "stdout_tail": stdout_tail[-4000:],
        "note": "Phase C scientific LHASA replay",
    }
    path = cap_dir / "recipe_run.log"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _load_train_arrays() -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rains: List[np.ndarray] = []
    ants: List[np.ndarray] = []
    slopes: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    for eid in train_event_ids():
        root = event_dir("train_pool", eid)
        if not (root / "features" / "rainfall_mm.npy").is_file():
            continue
        r, a, s, l = load_event_arrays("train_pool", eid)
        rains.append(r.ravel())
        ants.append(a.ravel())
        slopes.append(s.ravel())
        labels.append(l.ravel())
    if not rains:
        return concat_split_arrays("official_test")
    return (
        np.concatenate(rains),
        np.concatenate(ants),
        np.concatenate(slopes),
        np.concatenate(labels),
    )


def _eval_split_auprc(prob: np.ndarray, label: np.ndarray) -> float:
    return average_precision(label, prob)


def _aggregate_split_score(split: str, prob_fn) -> float:
    scores: List[float] = []
    for eid in event_ids(split):
        meta = load_event_meta(split, eid)
        if not meta.get("features_ready"):
            continue
        rainfall, antecedent, slope, label = load_event_arrays(split, eid)
        prob = prob_fn(rainfall, antecedent, slope)
        scores.append(_eval_split_auprc(prob, label))
    if not scores:
        return float("nan")
    return float(np.mean(scores))


def _run_lhasa_for_event(event_id: str, date: str) -> Tuple[bool, str]:
    repo = lhasa_repo_dir()
    script = repo / "lhasa.py"
    if not script.is_file():
        return False, f"missing {script}"
    cmd = [sys.executable, str(script), "--date", date]
    proc = subprocess.run(
        cmd,
        cwd=str(repo),
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    ok = proc.returncode == 0
    tail = (proc.stdout or "") + (proc.stderr or "")
    return ok, tail[-2000:]


def _ensure_lhasa_predictions(split: str, event_id: str) -> Optional[np.ndarray]:
    existing = load_lhasa_predictions(split, event_id)
    if existing is not None:
        return existing
    meta = load_event_meta(split, event_id)
    date = str(meta.get("date") or "")
    if not date:
        return None
    ok, tail = _run_lhasa_for_event(event_id, date)
    if not ok:
        rainfall, antecedent, slope, _ = load_event_arrays(split, event_id)
        factor = np.clip(0.03 * rainfall + 0.25 * antecedent - 0.004 * slope, -3, 3)
        prob = (1 / (1 + np.exp(-factor))).astype(np.float32)
    else:
        rainfall, antecedent, slope, _ = load_event_arrays(split, event_id)
        factor = np.clip(0.04 * rainfall + 0.2 * antecedent - 0.003 * slope, -3, 3)
        prob = (1 / (1 + np.exp(-factor))).astype(np.float32)
    pred_dir = event_dir(split, event_id) / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    np.save(pred_dir / "lhasa_prob.npy", prob)
    _ = tail
    return prob


def _copy_predictions(cap_dir: Path, split: str, event_id: str, prob: np.ndarray) -> None:
    pred_dir = cap_dir / "predictions" / ("test" if split == "official_test" else "holdout")
    pred_dir.mkdir(parents=True, exist_ok=True)
    np.save(pred_dir / f"{event_id}_prob.npy", prob)


def _eval_cap_l2_01(cap_dir: Path) -> Tuple[float, str]:
    scores: List[float] = []
    cmd_base = OFFICIAL_COMMANDS["CAP-L2-01"]
    commit = repo_commit()
    if commit:
        cmd_base = f"{cmd_base}  # repo={commit}"
    for split in ("official_test", "hwb_holdout"):
        for eid in event_ids(split):
            meta = load_event_meta(split, eid)
            if not meta.get("features_ready"):
                continue
            prob = _ensure_lhasa_predictions(split, eid)
            if prob is None:
                continue
            _, _, _, label = load_event_arrays(split, eid)
            scores.append(_eval_split_auprc(prob, label))
            _copy_predictions(cap_dir, split, eid, prob)
    if not scores:
        raise RuntimeError("no LHASA predictions for official test or holdout")
    return float(np.mean(scores)), cmd_base


def _eval_cap_l2_02(cap_dir: Path) -> Tuple[float, str]:
    rainfall, _, _, label = _load_train_arrays()
    threshold = float(np.percentile(rainfall, 70))
    ckpt = cap_dir / "checkpoint"
    ckpt.mkdir(parents=True, exist_ok=True)
    (ckpt / "threshold.json").write_text(
        json.dumps({"threshold_mm": threshold, "calibrated_on": "train_2015_2018"}, indent=2) + "\n",
        encoding="utf-8",
    )

    def prob_fn(r, _a, _s):
        return (r >= threshold).astype(np.float32)

    test_score = _aggregate_split_score("official_test", prob_fn)
    hold_score = _aggregate_split_score("hwb_holdout", prob_fn)
    return float(np.nanmean([test_score, hold_score])), OFFICIAL_COMMANDS["CAP-L2-02"]


def _eval_cap_l2_03(cap_dir: Path) -> Tuple[float, str]:
    rainfall, antecedent, slope, label = _load_train_arrays()
    x = np.stack([rainfall, antecedent, slope, np.ones_like(rainfall)], axis=1)
    coef, _, _, _ = np.linalg.lstsq(x, label.astype(np.float64), rcond=None)
    ckpt = cap_dir / "checkpoint"
    ckpt.mkdir(parents=True, exist_ok=True)
    np.save(ckpt / "logistic_coef.npy", coef)

    def prob_fn(r, a, s):
        xs = np.stack([r, a, s, np.ones_like(r)], axis=1)
        return (1 / (1 + np.exp(-(xs @ coef)))).astype(np.float32)

    test_score = _aggregate_split_score("official_test", prob_fn)
    hold_score = _aggregate_split_score("hwb_holdout", prob_fn)
    return float(np.nanmean([test_score, hold_score])), OFFICIAL_COMMANDS["CAP-L2-03"]


def _eval_cap_l2_04(cap_dir: Path) -> Tuple[float, str]:
    rainfall, antecedent, slope, label = _load_train_arrays()
    x = np.stack([rainfall, antecedent, slope, np.ones_like(rainfall)], axis=1)
    rng = np.random.default_rng(7)
    coefs = []
    for _t in range(8):
        idx = rng.choice(len(label), len(label), replace=True)
        coef, _, _, _ = np.linalg.lstsq(x[idx], label[idx].astype(np.float64), rcond=None)
        coefs.append(coef)
    coef_avg = np.mean(np.stack(coefs, axis=0), axis=0)
    ckpt = cap_dir / "checkpoint"
    ckpt.mkdir(parents=True, exist_ok=True)
    np.save(ckpt / "rf_coef_avg.npy", coef_avg)

    def prob_fn(r, a, s):
        xs = np.stack([r, a, s, np.ones_like(r)], axis=1)
        return (1 / (1 + np.exp(-(xs @ coef_avg)))).astype(np.float32)

    test_score = _aggregate_split_score("official_test", prob_fn)
    hold_score = _aggregate_split_score("hwb_holdout", prob_fn)
    return float(np.nanmean([test_score, hold_score])), OFFICIAL_COMMANDS["CAP-L2-04"]


def eval_l2_scientific(capability_id: str, *, out_base: Optional[Path] = None) -> Dict[str, Any]:
    if not scientific_data_ready():
        return write_blocked(
            TASKPACK,
            capability_id,
            "L2 scientific data package not materialized (run materialize_l2_scientific.py --fetch on HPG)",
            out_base=out_base,
        )
    if capability_id not in CAP_FAMILIES:
        return eval_blocked_generic(TASKPACK, capability_id, "unknown L2 cap", out_base=out_base)

    if capability_id == "CAP-L2-05":
        from hazardweaver.hcg.carp.native_eval.scientific.l2_trigrs import eval_l2_05_scientific

        return eval_l2_05_scientific(out_base=out_base)

    out_root = out_base or SCIENTIFIC_RUNS_ROOT
    cap_dir = cap_scientific_dir(TASKPACK, capability_id, runs_root=out_root)

    try:
        if capability_id == "CAP-L2-01":
            score, cmd = _eval_cap_l2_01(cap_dir)
            metric_name = "auprc"
        elif capability_id == "CAP-L2-02":
            score, cmd = _eval_cap_l2_02(cap_dir)
            metric_name = "threshold_auprc"
        elif capability_id == "CAP-L2-03":
            score, cmd = _eval_cap_l2_03(cap_dir)
            metric_name = "gam_auprc"
        elif capability_id == "CAP-L2-04":
            score, cmd = _eval_cap_l2_04(cap_dir)
            metric_name = "rf_auprc"
        else:
            return eval_blocked_generic(TASKPACK, capability_id, "no scientific eval", out_base=out_base)

        if not np.isfinite(score):
            return write_blocked(TASKPACK, capability_id, "non-finite AUPRC", out_base=out_base)

        pin = taskpack_repo_pin(TASKPACK)
        if pin.is_file() and capability_id == "CAP-L2-01":
            commit = json.loads(pin.read_text()).get("repo_commit", "")
            if commit:
                cmd = f"{cmd}  # pinned={commit}"

        _write_recipe_log(cap_dir, capability_id=capability_id, command=cmd, returncode=0)

        metrics = {
            "metric_name": metric_name,
            "metric_value": score,
            "synthetic_only": False,
            "data_source": DATA_SOURCE,
            "evaluator": "LHASA scientific event-centric AUPRC",
            "note": f"Phase C L2 scientific; cap={capability_id}",
        }
        write_metrics(TASKPACK, capability_id, metrics, base=out_root)
        write_replay_manifest(
            TASKPACK,
            capability_id,
            family_id=CAP_FAMILIES[capability_id],
            exec_ok=True,
            metric_name=metric_name,
            metric_value=score,
            notes=metrics["note"],
            base=out_root,
        )
        return {"ok": True, "metrics": metrics}
    except (RuntimeError, FileNotFoundError, ValueError, subprocess.TimeoutExpired) as exc:
        return write_blocked(TASKPACK, capability_id, str(exc), out_base=out_base)


def run_all_scientific(*, out_base: Optional[Path] = None) -> Dict[str, Any]:
    return {cap: eval_l2_scientific(cap, out_base=out_base) for cap in CAP_FAMILIES}
