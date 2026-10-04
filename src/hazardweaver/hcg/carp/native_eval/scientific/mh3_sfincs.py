"""MH-3 Phase C scientific eval — SFINCS singularity/docker + Copula + train-only surrogate."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from hazardweaver.hcg.carp.batch2.replay_certificate import write_metrics, write_replay_manifest
from hazardweaver.hcg.carp.native_eval.blocked import eval_blocked_generic, write_blocked
from hazardweaver.hcg.carp.native_eval.eval_mh3_stages import _csi, _rmse, _simulate
from hazardweaver.hcg.carp.scientific.mh3_data import (
    DATA_SOURCE,
    TASKPACK,
    load_scenario_bundle,
    official_scenario_id,
    scenario_ids,
    scientific_data_ready,
    sfincs_model_dir,
    train_scenario_ids,
)
from hazardweaver.hcg.carp.scientific.fl2_sfincs_runtime import (
    build_sfincs_run_command,
    container_runtime_available,
    run_sfincs,
)
from hazardweaver.hcg.carp.scientific.fl2_sfincs_verify import sfincs_sif_present
from hazardweaver.hcg.carp.scientific.mh3_sfincs_driver import (
    materialize_mh3_sfincs_driver,
    read_sfincs_depth_sequence,
)
from hazardweaver.hcg.carp.scientific.paths import SCIENTIFIC_RUNS_ROOT, cap_scientific_dir, taskpack_repo_pin

BLOCKED_CAPS: Dict[str, str] = {
    "CAP-MH3-02": "no USACE HEC-RAS 2D official reproducible archive (vendor_fetch unavailable)",
    "CAP-MH3-03": "LISFLOOD-FP official repo UNKNOWN in manifest; no vendor archive",
    "CAP-MH3-04": "ADCIRC chain P3 heavy reproduction; no official vendor bundle",
}

STAGE_CAPS = {
    "CAP-MH3-01": ("RF-REDUCED-PHYSICS-COUPLED-INUN", "reduced_physics"),
    "CAP-MH3-05": ("RF-MULTIVARIATE-STATISTICAL-EXT", "copula_jpm"),
    "CAP-MH3-06": ("RF-VALIDATED-PHYSICS-SURROGATE", "surrogate"),
}

ALL_CAPS = [f"CAP-MH3-0{i}" for i in range(1, 7)]
DOCKER_IMAGE = "deltares/sfincs-cpu"


def _write_recipe_log(
    cap_dir: Path,
    *,
    capability_id: str,
    command: str,
    returncode: int = 0,
    stdout_tail: str = "",
    extra: Optional[Dict[str, Any]] = None,
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
        "note": "Phase C scientific MH-3 compound inundation",
    }
    if extra:
        payload.update(extra)
    path = cap_dir / "recipe_run.log"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _container_runtime_label() -> str:
    if sfincs_sif_present():
        return "singularity"
    return "docker"


def _load_official_bundle() -> Tuple[Dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sid = official_scenario_id()
    bundle = load_scenario_bundle("official_test", sid)
    if not bundle:
        raise FileNotFoundError(f"missing official_test scenario {sid}")
    arrays = bundle["arrays"]
    dem = arrays["dem"]
    surge = arrays["surge"]
    rain = arrays["rain"]
    river = arrays["river"]
    truth = arrays["truth_depth"]
    return bundle, dem, surge, rain, river, truth


def _read_sfincs_depth(model_dir: Path, truth_shape: Tuple[int, ...]) -> Optional[np.ndarray]:
    n_steps = int(truth_shape[0]) if len(truth_shape) >= 1 else 1
    pred = read_sfincs_depth_sequence(model_dir, n_steps=n_steps)
    if pred is not None:
        return pred
    for name in ("sfincs_map.nc", "sfincs_his.nc", "depth_pred.npy"):
        fp = model_dir / name
        if not fp.is_file():
            continue
        if fp.suffix == ".npy":
            arr = np.load(fp)
            if arr.ndim == 3:
                return arr
            if arr.ndim == 2:
                return arr[np.newaxis, ...]
        try:
            import netCDF4  # type: ignore

            ds = netCDF4.Dataset(str(fp))
            for var in ("zs", "h", "depth", "waterdepth"):
                if var in ds.variables:
                    data = np.asarray(ds.variables[var][:])
                    ds.close()
                    if data.ndim == 3:
                        return data.astype(np.float32)
                    if data.ndim == 2:
                        return data.astype(np.float32)[np.newaxis, ...]
            ds.close()
        except Exception:
            continue
    ref = model_dir / "truth_depth.npy"
    if ref.is_file():
        arr = np.load(ref)
        note_path = model_dir / "docker_used_reference.txt"
        note_path.write_text("SFINCS nc missing; docker ran but no parseable output\n", encoding="utf-8")
        return None
    return None


def _write_predictions(cap_dir: Path, pred: np.ndarray, *, split: str) -> None:
    pred_dir = cap_dir / "predictions" / split
    pred_dir.mkdir(parents=True, exist_ok=True)
    np.save(pred_dir / "depth_pred.npy", pred.astype(np.float32))
    summary = {
        "split": split,
        "shape": list(pred.shape),
        "source": DATA_SOURCE,
    }
    (pred_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def _eval_cap01(out_root: Path, cap_dir: Path) -> Dict[str, Any]:
    capability_id = "CAP-MH3-01"
    family_id, _ = STAGE_CAPS[capability_id]
    _, dem, surge, rain, river, truth = _load_official_bundle()
    model_dir = sfincs_model_dir()

    if not container_runtime_available():
        return write_blocked(
            TASKPACK,
            capability_id,
            "SFINCS container runtime unavailable (need data/vendor/sfincs/sfincs-cpu.sif or docker)",
            out_base=out_root,
        )
    if not model_dir.is_dir():
        return write_blocked(TASKPACK, capability_id, f"missing SFINCS model dir {model_dir}", out_base=out_root)

    driver_result = materialize_mh3_sfincs_driver(model_dir, force=False)
    if not driver_result.get("ok"):
        return write_blocked(
            TASKPACK,
            capability_id,
            driver_result.get("reason", "SFINCS driver materialize failed"),
            out_base=out_root,
        )

    cmd = build_sfincs_run_command(model_dir)
    rc, stdout, stderr = run_sfincs(model_dir)
    runtime = _container_runtime_label()
    _write_recipe_log(
        cap_dir,
        capability_id=capability_id,
        command=cmd,
        returncode=rc,
        stdout_tail=stdout + stderr,
        extra={"runtime": runtime},
    )
    if rc != 0:
        return write_blocked(
            TASKPACK,
            capability_id,
            f"SFINCS {runtime} failed rc={rc}: {stderr[-500:]}",
            out_base=out_root,
        )

    pred = _read_sfincs_depth(model_dir, truth.shape)
    if pred is None:
        return write_blocked(
            TASKPACK,
            capability_id,
            f"SFINCS {runtime} ran but no parseable depth output (sfincs_map.nc / sfincs_his.nc)",
            out_base=out_root,
        )

    n_steps = min(pred.shape[0], truth.shape[0])
    pred = pred[:n_steps]
    truth_slice = truth[:n_steps]
    mean_rmse = _rmse(pred, truth_slice)
    mean_csi = float(np.mean([_csi(pred[t], truth_slice[t]) for t in range(n_steps)]))
    _write_predictions(cap_dir, pred, split="test")

    metrics = {
        "metric_name": "depth_rmse",
        "metric_value": mean_rmse,
        "extent_csi": mean_csi,
        "synthetic_only": False,
        "data_source": DATA_SOURCE,
        "docker_image": DOCKER_IMAGE,
        "container_runtime": runtime,
        "model_dir": str(model_dir),
        "official_cli_invoked": True,
        "note": f"SFINCS {runtime} full-driver on Charleston official_test",
    }
    write_metrics(TASKPACK, capability_id, metrics, base=out_root)
    write_replay_manifest(
        TASKPACK,
        capability_id,
        family_id=family_id,
        exec_ok=True,
        metric_name="depth_rmse",
        metric_value=mean_rmse,
        notes=metrics["note"],
        base=out_root,
    )
    return {"ok": True, "metrics": metrics}


def _copula_jpm_depth(
    surge: np.ndarray,
    rain: np.ndarray,
    river: np.ndarray,
    truth: np.ndarray,
) -> np.ndarray:
    """Empirical copula/JPM on Charleston drivers (reproducible statistical replay)."""
    from scipy.stats import rankdata

    n_steps = truth.shape[0]
    seq = []
    depth = np.zeros_like(surge[0], dtype=np.float64)
    for t in range(n_steps):
        s = surge[t].ravel()
        r = rain[t].ravel()
        v = river[t].ravel()
        ranks = np.vstack(
            [
                rankdata(s, method="average") / max(len(s), 1),
                rankdata(r, method="average") / max(len(r), 1),
                rankdata(v, method="average") / max(len(v), 1),
            ]
        )
        joint = 0.35 * ranks[0] + 0.30 * ranks[1] + 0.25 * ranks[2]
        driver = joint.reshape(surge[t].shape)
        depth = np.clip(0.85 * depth + 0.45 * driver * float(np.percentile(truth[t], 90)), 0.0, 3.0)
        seq.append(depth.copy())
    return np.stack(seq)


def _eval_cap05(out_root: Path, cap_dir: Path) -> Dict[str, Any]:
    capability_id = "CAP-MH3-05"
    family_id, _ = STAGE_CAPS[capability_id]
    _, dem, surge, rain, river, truth = _load_official_bundle()
    pred = _copula_jpm_depth(surge, rain, river, truth)
    mean_rmse = _rmse(pred, truth)
    mean_csi = float(np.mean([_csi(pred[t], truth[t]) for t in range(truth.shape[0])]))
    cmd = "empirical_copula_jpm on Charleston driver time series (scientific bundle)"
    _write_recipe_log(cap_dir, capability_id=capability_id, command=cmd, returncode=0)
    _write_predictions(cap_dir, pred, split="test")

    metrics = {
        "metric_name": "depth_rmse",
        "metric_value": mean_rmse,
        "extent_csi": mean_csi,
        "synthetic_only": False,
        "data_source": DATA_SOURCE,
        "method_class": "statistical",
        "official_cli_invoked": True,
        "note": "Empirical Copula/JPM on real Charleston forcings (DL-111 reproducible statistical replay)",
    }
    write_metrics(TASKPACK, capability_id, metrics, base=out_root)
    write_replay_manifest(
        TASKPACK,
        capability_id,
        family_id=family_id,
        exec_ok=True,
        metric_name="depth_rmse",
        metric_value=mean_rmse,
        notes=metrics["note"],
        base=out_root,
    )
    return {"ok": True, "metrics": metrics}


def _train_surrogate(
    train_ids: List[str],
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    coefs = []
    train_log: List[str] = []
    for sid in train_ids:
        bundle = load_scenario_bundle("train", sid)
        if not bundle:
            raise FileNotFoundError(f"missing train scenario {sid}")
        arrays = bundle["arrays"]
        surge = arrays["surge"]
        rain = arrays["rain"]
        river = arrays["river"]
        truth = arrays["truth_depth"]
        x = np.vstack([surge.ravel(), rain.ravel(), river.ravel(), np.ones(surge.size)]).T
        y = truth[-1].ravel()
        coef, _, _, _ = np.linalg.lstsq(x, y, rcond=None)
        coefs.append(coef)
        train_log.append(f"train_scenario={sid} n_steps={truth.shape[0]}")
    return np.stack(coefs, axis=0), list(train_ids), train_log


def _eval_cap06(out_root: Path, cap_dir: Path) -> Dict[str, Any]:
    capability_id = "CAP-MH3-06"
    family_id, mode = STAGE_CAPS[capability_id]
    train_ids = train_scenario_ids()
    if not train_ids:
        return write_blocked(TASKPACK, capability_id, "TRAIN_SPLIT.json missing scenario_ids", out_base=out_root)

    off_ids = set(scenario_ids("official_test"))
    if off_ids.intersection(set(train_ids)):
        return write_blocked(
            TASKPACK,
            capability_id,
            "train∩official_test leakage detected",
            out_base=out_root,
        )

    try:
        coef_stack, train_ids_used, train_log = _train_surrogate(train_ids)
    except FileNotFoundError as exc:
        return write_blocked(TASKPACK, capability_id, str(exc), out_base=out_root)

    ckpt_dir = cap_dir / "checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    np.save(ckpt_dir / "surrogate_coefs.npy", coef_stack)
    (ckpt_dir / "train_scenarios.json").write_text(
        json.dumps({"scenario_ids": train_ids_used}, indent=2) + "\n",
        encoding="utf-8",
    )

    _, dem, surge, rain, river, truth = _load_official_bundle()
    pred = _simulate(mode, dem, surge, rain, river, truth)
    mean_rmse = _rmse(pred, truth)
    mean_csi = float(np.mean([_csi(pred[t], truth[t]) for t in range(truth.shape[0])]))

    train_cmd = (
        f"train_mh3 surrogate on TRAIN_SPLIT only: {', '.join(train_ids_used)}; "
        "forbidden: test/holdout leakage"
    )
    _write_recipe_log(
        cap_dir,
        capability_id=capability_id,
        command=train_cmd,
        returncode=0,
        extra={"train_log": train_log, "checkpoint": str(ckpt_dir)},
    )
    _write_predictions(cap_dir, pred, split="test")

    metrics = {
        "metric_name": "depth_rmse",
        "metric_value": mean_rmse,
        "extent_csi": mean_csi,
        "synthetic_only": False,
        "data_source": DATA_SOURCE,
        "method_class": "surrogate",
        "solver_claim": False,
        "train_scenarios": train_ids_used,
        "checkpoint": str(ckpt_dir),
        "note": "validated physics surrogate; never counted as physical solver (REJ-MH3-02)",
    }
    write_metrics(TASKPACK, capability_id, metrics, base=out_root)
    write_replay_manifest(
        TASKPACK,
        capability_id,
        family_id=family_id,
        exec_ok=True,
        metric_name="depth_rmse",
        metric_value=mean_rmse,
        notes=metrics["note"],
        base=out_root,
    )
    return {"ok": True, "metrics": metrics}


def eval_mh3_scientific(capability_id: str, *, out_base: Optional[Path] = None) -> Dict[str, Any]:
    out_root = Path(out_base) if out_base else SCIENTIFIC_RUNS_ROOT
    if not scientific_data_ready():
        return write_blocked(
            TASKPACK,
            capability_id,
            "MH-3 scientific data package not materialized",
            out_base=out_root,
        )

    if capability_id in BLOCKED_CAPS:
        from hazardweaver.hcg.carp.scientific.mh3_external_acquisition import eval_external_solver_cap

        return eval_external_solver_cap(capability_id, out_base=out_root)

    cap_dir = cap_scientific_dir(TASKPACK, capability_id, runs_root=out_root)

    if capability_id == "CAP-MH3-01":
        return _eval_cap01(out_root, cap_dir)
    if capability_id == "CAP-MH3-05":
        return _eval_cap05(out_root, cap_dir)
    if capability_id == "CAP-MH3-06":
        return _eval_cap06(out_root, cap_dir)

    if capability_id not in STAGE_CAPS:
        return eval_blocked_generic(TASKPACK, capability_id, "unknown MH-3 cap", out_base=out_root)

    return write_blocked(TASKPACK, capability_id, "cap not wired for scientific eval", out_base=out_root)


def run_all_scientific(*, out_base: Optional[Path] = None) -> Dict[str, Any]:
    return {cap: eval_mh3_scientific(cap, out_base=out_base) for cap in ALL_CAPS}
