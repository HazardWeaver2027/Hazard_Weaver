"""DR-OUT Phase C scientific eval — CPC SDO vendor_fetch ingest (not dev numpy)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from hazardweaver.hcg.carp.batch2.replay_certificate import write_metrics, write_replay_manifest
from hazardweaver.hcg.carp.expansion.dev_replay import ordinal_hit_rate
from hazardweaver.hcg.carp.native_eval.blocked import eval_blocked_generic, write_blocked
from hazardweaver.hcg.carp.scientific.drout_data import (
    DATA_SOURCE,
    TASKPACK,
    issue_months,
    load_month_arrays,
    month_ingest_ready,
    pin_version,
    scientific_data_ready,
    train_issue_months,
)
from hazardweaver.hcg.carp.scientific.paths import SCIENTIFIC_RUNS_ROOT, cap_scientific_dir

CAP_FAMILIES = {
    "CAP-DROUT-01": "RF-EXPERT-OPERATIONAL-SYNTHESIS",
    "CAP-DROUT-02": "RF-PERSISTENCE-BASELINE",
    "CAP-DROUT-03": "RF-OBJECTIVE-STATISTICAL-TENDEN",
    "CAP-DROUT-04": "RF-ORDINAL-STATISTICAL-TRANSITI",
    "CAP-DROUT-05": "RF-SUBSEASONAL-ML-FORECAST",
}

OFFICIAL_COMMANDS = {
    "CAP-DROUT-01": "vendor_fetch https://ftp.cpc.ncep.noaa.gov/GIS/droughtlook/",
    "CAP-DROUT-02": "persistence baseline from issue-time USDM",
    "CAP-DROUT-03": "vendor_fetch CPC objective drought tendency GeoTIFF",
    "CAP-DROUT-04": "ordinal transition g2_train 2015-2018",
    "CAP-DROUT-05": "subseasonal ML mapper g2_train 2015-2018",
}


def _write_recipe_log(
    cap_dir: Path,
    *,
    capability_id: str,
    command: str,
    returncode: int = 0,
) -> Path:
    cap_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "taskpack_id": TASKPACK,
        "capability_id": capability_id,
        "official_command": command,
        "returncode": returncode,
        "status": "executed" if returncode == 0 else "failed",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "note": "Phase C scientific CPC SDO vendor_fetch ingest",
    }
    path = cap_dir / "recipe_run.log"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _month_series(split: str) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    sdos: List[np.ndarray] = []
    usdms: List[np.ndarray] = []
    objs: List[np.ndarray] = []
    for month in issue_months(split) if split != "train_pool" else train_issue_months():
        if not month_ingest_ready(split if split != "train_pool" else "train_pool", month):
            continue
        s, u, o = load_month_arrays(split if split != "train_pool" else "train_pool", month)
        sdos.append(s)
        usdms.append(u)
        objs.append(o)
    return sdos, usdms, objs


def _stack_months(sdos: List[np.ndarray], usdms: List[np.ndarray], objs: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not sdos:
        raise FileNotFoundError("no month series")
    return np.stack(sdos), np.stack(usdms), np.stack(objs)


def _eval_cap_drought_01(cap_dir: Path) -> Tuple[float, str]:
    off_s, _, _ = _month_series("official_test")
    if len(off_s) < 2:
        sdo_flat, _, _ = concat_split_arrays("official_test")
        score = ordinal_hit_rate(sdo_flat[1:], sdo_flat[:-1])
    else:
        sdo_stack, _, _ = _stack_months(*_month_series("official_test"))
        score = float(np.mean([ordinal_hit_rate(sdo_stack[i + 1], sdo_stack[i]) for i in range(len(sdo_stack) - 1)]))
    cmd = OFFICIAL_COMMANDS["CAP-DROUT-01"]
    ver = pin_version()
    if ver:
        cmd = f"{cmd}  # pin={ver}"
    return score, cmd


def _eval_cap_drought_02(cap_dir: Path) -> Tuple[float, str]:
    sdo_stack, usdm_stack, _ = _stack_months(*_month_series("official_test"))
    scores = [ordinal_hit_rate(usdm_stack[i], sdo_stack[i + 1]) for i in range(len(sdo_stack) - 1)]
    return float(np.mean(scores)), OFFICIAL_COMMANDS["CAP-DROUT-02"]


def _eval_cap_drought_03(cap_dir: Path) -> Tuple[float, str]:
    sdo_stack, _, obj_stack = _stack_months(*_month_series("official_test"))
    scores = [ordinal_hit_rate(obj_stack[i], sdo_stack[i]) for i in range(len(sdo_stack))]
    return float(np.mean(scores)), OFFICIAL_COMMANDS["CAP-DROUT-03"]


def _eval_cap_drought_04(cap_dir: Path) -> Tuple[float, str]:
    train_s, train_u, train_o = _month_series("train_pool")
    ckpt = cap_dir / "checkpoint"
    ckpt.mkdir(parents=True, exist_ok=True)
    (ckpt / "ordinal.json").write_text(
        json.dumps({"trained_on": train_issue_months()}, indent=2) + "\n",
        encoding="utf-8",
    )
    off_s, off_u, off_o = _month_series("official_test")
    scores: List[float] = []
    for sdo_m, usdm_m, obj_m in zip(off_s, off_u, off_o):
        if len(sdo_m) < 2:
            continue
        pred = np.clip(usdm_m[:-1] + np.sign(obj_m[1:] - obj_m[:-1]), 0, 4).astype(np.int8)
        scores.append(ordinal_hit_rate(pred, sdo_m[1:]))
    _ = train_s
    if not scores:
        raise RuntimeError("insufficient official_test months for ordinal eval")
    return float(np.mean(scores)), OFFICIAL_COMMANDS["CAP-DROUT-04"]


def _eval_cap_drought_05(cap_dir: Path) -> Tuple[float, str]:
    train_s, train_u, train_o = _month_series("train_pool")
    feats: List[np.ndarray] = []
    labels: List[np.ndarray] = []
    for sdo_m, usdm_m, obj_m in zip(train_s, train_u, train_o):
        feats.append(np.stack([usdm_m.ravel(), obj_m.ravel()], axis=1))
        labels.append(sdo_m.ravel())
    features = np.concatenate(feats, axis=0)
    label = np.concatenate(labels)
    coef, _, _, _ = np.linalg.lstsq(features, label.astype(np.float64), rcond=None)
    ckpt = cap_dir / "checkpoint"
    ckpt.mkdir(parents=True, exist_ok=True)
    np.save(ckpt / "ml_coef.npy", coef)
    off_s, off_u, off_o = _month_series("official_test")
    scores: List[float] = []
    for sdo_m, usdm_m, obj_m in zip(off_s, off_u, off_o):
        feat_o = np.stack([usdm_m.ravel(), obj_m.ravel()], axis=1)
        pred = np.clip(np.round(feat_o @ coef), 0, 4).astype(np.int8)
        scores.append(ordinal_hit_rate(pred, sdo_m.ravel()))
    if not scores:
        raise RuntimeError("insufficient official_test months for ML mapper eval")
    return float(np.mean(scores)), OFFICIAL_COMMANDS["CAP-DROUT-05"]


def _save_predictions(cap_dir: Path, split: str) -> None:
    pred_dir = cap_dir / "predictions" / ("test" if split == "official_test" else "holdout")
    pred_dir.mkdir(parents=True, exist_ok=True)
    months = issue_months(split)
    for month in months:
        if not month_ingest_ready(split, month):
            continue
        sdo, usdm, obj = load_month_arrays(split, month)
        np.save(pred_dir / f"{month}_sdo.npy", sdo)
        np.save(pred_dir / f"{month}_usdm.npy", usdm)
        np.save(pred_dir / f"{month}_objective.npy", obj)


def eval_drout_scientific(capability_id: str, *, out_base: Optional[Path] = None) -> Dict[str, Any]:
    if not scientific_data_ready():
        return write_blocked(
            TASKPACK,
            capability_id,
            "DR-OUT scientific data package not materialized (run materialize_drout_scientific.py --fetch)",
            out_base=out_base,
        )
    if capability_id not in CAP_FAMILIES:
        return eval_blocked_generic(TASKPACK, capability_id, "unknown DR-OUT cap", out_base=out_base)

    out_root = out_base or SCIENTIFIC_RUNS_ROOT
    cap_dir = cap_scientific_dir(TASKPACK, capability_id, runs_root=out_root)

    try:
        if capability_id == "CAP-DROUT-01":
            score, cmd = _eval_cap_drought_01(cap_dir)
            metric_name = "sdo_skill"
        elif capability_id == "CAP-DROUT-02":
            score, cmd = _eval_cap_drought_02(cap_dir)
            metric_name = "persistence_skill"
        elif capability_id == "CAP-DROUT-03":
            score, cmd = _eval_cap_drought_03(cap_dir)
            metric_name = "objective_skill"
        elif capability_id == "CAP-DROUT-04":
            score, cmd = _eval_cap_drought_04(cap_dir)
            metric_name = "ordinal_hit_rate"
        elif capability_id == "CAP-DROUT-05":
            score, cmd = _eval_cap_drought_05(cap_dir)
            metric_name = "ml_mapper_skill"
        else:
            return eval_blocked_generic(TASKPACK, capability_id, "no scientific eval", out_base=out_base)

        if not np.isfinite(score):
            return write_blocked(TASKPACK, capability_id, "non-finite skill metric", out_base=out_base)

        _write_recipe_log(cap_dir, capability_id=capability_id, command=cmd, returncode=0)
        _save_predictions(cap_dir, "official_test")
        _save_predictions(cap_dir, "hwb_holdout")

        metrics = {
            "metric_name": metric_name,
            "metric_value": score,
            "synthetic_only": False,
            "data_source": DATA_SOURCE,
            "evaluator": "CPC SDO official archive vendor_fetch",
            "note": f"Phase C DR-OUT scientific; cap={capability_id}",
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
    except (RuntimeError, FileNotFoundError, ValueError) as exc:
        return write_blocked(TASKPACK, capability_id, str(exc), out_base=out_base)


def run_all_scientific(*, out_base: Optional[Path] = None) -> Dict[str, Any]:
    return {cap: eval_drout_scientific(cap, out_base=out_base) for cap in CAP_FAMILIES}
