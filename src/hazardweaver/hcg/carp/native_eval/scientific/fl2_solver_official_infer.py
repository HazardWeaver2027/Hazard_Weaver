"""Official solver inference for CAP-FL2-04~06 (M3 HWB solver route seal).

Never uses sfincs_proxy / lisflood_proxy. When audit is BLOCKED, writes honest
recipe_run.log, per-scenario blocked metadata, and cap-level BLOCKED.json.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from hazardweaver.hcg.carp.native_eval.scientific.fl2_hand_faithful import compute_hand_depth_sequence
from hazardweaver.hcg.carp.native_eval.scientific.fl2_lisflood_driver import (
    materialize_lisflood_driver,
    read_lisflood_depth_sequence,
)
from hazardweaver.hcg.carp.native_eval.scientific.fl2_sfincs_driver import (
    materialize_sfincs_driver,
    read_sfincs_depth_sequence,
)
from hazardweaver.hcg.carp.scientific.fl2_lisflood_runtime import run_lisflood
from hazardweaver.hcg.carp.scientific.fl2_data import (
    PROJECT_ROOT,
    TASKPACK,
    container_runtime_available,
    load_scenario_arrays,
    lisflood_fp_vendor_ready,
    prediction_path,
    prediction_subdir,
    scenario_dir,
    scenario_ids,
    scenarios_ready,
    sfincs_vendor_ready,
    split_scenarios_ready,
)
from hazardweaver.hcg.carp.scientific.fl2_lisflood_verify import verify_lisflood_executable
from hazardweaver.hcg.carp.scientific.fl2_sfincs_verify import verify_sfincs_executable
from hazardweaver.hcg.carp.scientific.fl2_sfincs_runtime import run_sfincs
from hazardweaver.hcg.carp.scientific.fl2_official_commands import (
    build_hand_command,
    build_lisflood_command,
    build_sfincs_command,
)
from hazardweaver.hcg.carp.scientific.paths import cap_scientific_dir

SOLVER_CAPS = ["CAP-FL2-04", "CAP-FL2-05", "CAP-FL2-06"]
DEFAULT_SPLIT = "official_test"
AUDIT_DOC = (
    PROJECT_ROOT
    / "docs/engineering/hcg/external_expansion/FL-2/FL2_SOLVER_INFERENCE_AUDIT.md"
)

SPLIT_TO_PRED_SUBDIR = {
    "official_test": "test",
    "hwb_holdout": "holdout",
    "official_test_high": "test",
    "hwb_holdout_high": "holdout",
}


def solver_caps() -> List[str]:
    return list(SOLVER_CAPS)


def default_scenario_id(*, split: str = DEFAULT_SPLIT) -> str:
    ids = scenario_ids(split)
    if not ids:
        raise FileNotFoundError(f"no scenarios in split {split!r}")
    return ids[0]


def data_package_ready() -> bool:
    return scenarios_ready()


def predictions_subdir(split: str) -> str:
    return prediction_subdir(split)


def audit_solver_readiness(capability_id: str) -> Dict[str, Any]:
    if capability_id not in SOLVER_CAPS:
        return {
            "capability_id": capability_id,
            "conclusion": "BLOCKED",
            "blockers": ["UNKNOWN_CAPABILITY"],
            "audit_doc": str(AUDIT_DOC.relative_to(PROJECT_ROOT)),
        }

    blockers: List[str] = []
    base: Dict[str, Any] = {
        "capability_id": capability_id,
        "audit_doc": str(AUDIT_DOC.relative_to(PROJECT_ROOT)),
    }

    if capability_id == "CAP-FL2-06":
        if not scenarios_ready():
            blockers.append("SCENARIOS_NOT_READY")
        base["solver"] = "hand_faithful"
        base["substitutes_cap"] = None
        if blockers:
            return {**base, "conclusion": "BLOCKED", "blockers": blockers}
        return {**base, "conclusion": "OFFICIAL_READY", "blockers": []}

    if capability_id == "CAP-FL2-05":
        base["solver"] = "lisflood_fp_cli"
        if not lisflood_fp_vendor_ready():
            blockers.append("LISFLOOD_VENDOR_NOT_PASS")
        ok_exec, exec_report = verify_lisflood_executable()
        base["executable_verification"] = exec_report
        if not ok_exec:
            blockers.append("LISFLOOD_EXECUTABLE_NOT_VERIFIED")
        if blockers:
            return {**base, "conclusion": "BLOCKED", "blockers": blockers}
        return {**base, "conclusion": "OFFICIAL_READY", "blockers": []}

    # CAP-FL2-04 SFINCS
    base["solver"] = "sfincs_singularity"
    if not sfincs_vendor_ready():
        blockers.append("SFINCS_VENDOR_NOT_PASS")
    ok_exec, exec_report = verify_sfincs_executable()
    base["executable_verification"] = exec_report
    if not ok_exec:
        blockers.append("SFINCS_EXECUTABLE_NOT_VERIFIED")
    if not container_runtime_available():
        blockers.append("SFINCS_CONTAINER_RUNTIME_UNAVAILABLE")
    if blockers:
        return {**base, "conclusion": "BLOCKED", "blockers": blockers}
    return {**base, "conclusion": "OFFICIAL_READY", "blockers": []}


def build_solver_command(
    capability_id: str,
    *,
    scenario_id: str,
    split: str = DEFAULT_SPLIT,
    out_path: Optional[Path] = None,
) -> str:
    scen = scenario_dir(split, scenario_id)
    pred = out_path or prediction_path(cap_scientific_dir(TASKPACK, capability_id), split, scenario_id)
    if capability_id == "CAP-FL2-06":
        py = "python"
        worker = (
            Path(__file__).resolve().parent / "fl2_floodcast_worker.py"
        )
        return (
            f"{py} {worker} --capability-id {capability_id} "
            f"--mode hand_faithful --split {split} --scenario-id {scenario_id} "
            f"--output {pred}"
        )
    if capability_id == "CAP-FL2-05":
        driver = cap_scientific_dir(TASKPACK, capability_id) / "driver" / scenario_id
        return build_lisflood_command(str(driver))
    if capability_id == "CAP-FL2-04":
        driver = cap_scientific_dir(TASKPACK, capability_id) / "driver" / scenario_id
        return build_sfincs_command(str(driver))
    raise ValueError(f"unsupported solver capability {capability_id}")


def write_cap_blocked_json(capability_id: str, *, cap_dir: Optional[Path] = None) -> Path:
    audit = audit_solver_readiness(capability_id)
    cap_root = cap_dir or cap_scientific_dir(TASKPACK, capability_id)
    cap_root.mkdir(parents=True, exist_ok=True)
    template_cmd = build_solver_command(
        capability_id,
        scenario_id=default_scenario_id(split="official_test"),
        split="official_test",
    )
    payload = {
        "capability_id": capability_id,
        "conclusion": audit["conclusion"],
        "blockers": audit.get("blockers") or [],
        "audit_doc": audit.get("audit_doc"),
        "official_command_template": template_cmd,
        "n_scenarios_test": len(scenario_ids("official_test")),
        "n_scenarios_holdout": len(scenario_ids("hwb_holdout")),
        "solver": audit.get("solver"),
    }
    path = cap_root / "BLOCKED.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _write_recipe_log(
    cap_dir: Path,
    *,
    capability_id: str,
    command: str,
    returncode: int,
    inference_mode: str,
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    cap_dir.mkdir(parents=True, exist_ok=True)
    payload: Dict[str, Any] = {
        "taskpack_id": TASKPACK,
        "capability_id": capability_id,
        "official_command": command,
        "returncode": returncode,
        "inference_mode": inference_mode,
        "status": "executed" if returncode == 0 else "blocked",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "note": "M3 FL-2 solver infer (no proxy)",
    }
    if extra:
        payload.update(extra)
    path = cap_dir / "recipe_run.log"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _write_npz(
    path: Path,
    *,
    pred: np.ndarray,
    scenario_id: str,
    split: str,
    capability_id: str,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        pred_depth=pred.astype(np.float32),
        scenario_id=np.array(scenario_id),
        split=np.array(split),
        capability_id=np.array(capability_id),
    )


def _run_hand_infer(
    *,
    scenario_id: str,
    split: str,
    pred_path: Path,
) -> Dict[str, Any]:
    dem, initial, truth = load_scenario_arrays(split, scenario_id)
    pred = compute_hand_depth_sequence(dem, initial, truth.shape[0])
    _write_npz(
        pred_path,
        pred=pred,
        scenario_id=scenario_id,
        split=split,
        capability_id="CAP-FL2-06",
    )
    return {"ok": True, "shape": list(pred.shape)}


def _blocked_result(
    *,
    capability_id: str,
    scenario_id: str,
    split: str,
    cap_dir: Path,
    pred_dir: Path,
    command: str,
    audit: Dict[str, Any],
) -> Dict[str, Any]:
    write_cap_blocked_json(capability_id, cap_dir=cap_dir)
    recipe_log = _write_recipe_log(
        cap_dir,
        capability_id=capability_id,
        command=command,
        returncode=1,
        inference_mode="blocked",
        extra={
            "audit_conclusion": "BLOCKED",
            "blockers": audit["blockers"],
            "official_status": "BLOCKED_OFFICIAL",
            "scenario_id": scenario_id,
            "split": split,
        },
    )
    blocked_meta = pred_dir / f"{scenario_id}_blocked.json"
    blocked_meta.write_text(
        json.dumps(
            {
                "scenario_id": scenario_id,
                "split": split,
                "capability_id": capability_id,
                "blocked": True,
                "blockers": audit["blockers"],
                "official_command": command,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "ok": False,
        "blocked": True,
        "official_status": "BLOCKED_OFFICIAL",
        "blockers": audit["blockers"],
        "recipe_log": str(recipe_log),
        "official_command": command,
        "audit": audit,
        "blocked_meta": str(blocked_meta),
    }


def run_official_solver_infer(
    capability_id: str,
    *,
    scenario_id: str,
    split: str = DEFAULT_SPLIT,
    out_dir: Optional[Path] = None,
    write_npz: bool = True,
) -> Dict[str, Any]:
    if capability_id not in SOLVER_CAPS:
        return {"ok": False, "error": f"unsupported solver capability {capability_id}"}
    if not split_scenarios_ready(split):
        return {"ok": False, "error": f"FL-2 scenario data not ready for split {split!r}"}

    audit = audit_solver_readiness(capability_id)
    cap_dir = Path(out_dir) if out_dir else cap_scientific_dir(TASKPACK, capability_id)
    pred_path = prediction_path(cap_dir, split, scenario_id)
    pred_dir = pred_path.parent
    pred_dir.mkdir(parents=True, exist_ok=True)
    command = build_solver_command(
        capability_id,
        scenario_id=scenario_id,
        split=split,
        out_path=pred_path,
    )

    if audit["conclusion"] == "BLOCKED":
        return _blocked_result(
            capability_id=capability_id,
            scenario_id=scenario_id,
            split=split,
            cap_dir=cap_dir,
            pred_dir=pred_dir,
            command=command,
            audit=audit,
        )

    if pred_path.is_file() and not __import__("os").environ.get("FL2_FORCE_REINFER"):
        data = np.load(pred_path)
        pred_arr = data["pred_depth"] if "pred_depth" in data else None
        return {
            "ok": True,
            "official_status": "OFFICIAL_READY",
            "artifact_paths": [str(pred_path)],
            "pred_depth": str(pred_path),
            "recipe_log": str(cap_dir / "recipe_run.log"),
            "official_command": command,
            "shape": list(pred_arr.shape) if pred_arr is not None else None,
            "inference_mode": "reuse_batch_prediction",
        }

    if capability_id == "CAP-FL2-06":
        try:
            result = _run_hand_infer(scenario_id=scenario_id, split=split, pred_path=pred_path)
        except (FileNotFoundError, OSError) as exc:
            return {"ok": False, "error": str(exc), "blocked": True}
        recipe_log = _write_recipe_log(
            cap_dir,
            capability_id=capability_id,
            command=command,
            returncode=0,
            inference_mode="hand_faithful",
            extra={"scenario_id": scenario_id, "split": split},
        )
        return {
            "ok": True,
            "official_status": "OFFICIAL_READY",
            "artifact_paths": [str(pred_path)],
            "pred_depth": str(pred_path),
            "recipe_log": str(recipe_log),
            "official_command": command,
            "shape": result.get("shape"),
        }

    if capability_id == "CAP-FL2-05":
        scen = scenario_dir(split, scenario_id)
        dem, initial, truth = load_scenario_arrays(split, scenario_id)
        driver_root = cap_dir / "driver" / scenario_id
        par_path, meta = materialize_lisflood_driver(scen, out_dir=driver_root, max_dim=200)
        if not meta.get("ok"):
            return _blocked_result(
                capability_id=capability_id,
                scenario_id=scenario_id,
                split=split,
                cap_dir=cap_dir,
                pred_dir=pred_dir,
                command=command,
                audit={**audit, "conclusion": "BLOCKED", "blockers": ["LISFLOOD_DRIVER_MATERIALIZE_FAILED"]},
            )
        rc, stdout, stderr = run_lisflood(driver_root)
        pred = read_lisflood_depth_sequence(driver_root, n_steps=truth.shape[0])
        if pred is not None and write_npz:
            if pred.shape[1:] != truth.shape[1:]:
                from skimage.transform import resize

                out = np.zeros((pred.shape[0], *truth.shape[1:]), dtype=np.float32)
                for t in range(pred.shape[0]):
                    out[t] = resize(pred[t], truth.shape[1:], order=1, preserve_range=True, anti_aliasing=True)
                pred = out
            _write_npz(
                pred_path,
                pred=pred,
                scenario_id=scenario_id,
                split=split,
                capability_id=capability_id,
            )
        recipe_log = _write_recipe_log(
            cap_dir,
            capability_id=capability_id,
            command=command,
            returncode=rc,
            inference_mode="lisflood_cli",
            extra={
                "scenario_id": scenario_id,
                "split": split,
                "driver_meta": meta,
                "stdout_tail": stdout[-2000:],
                "stderr_tail": stderr[-2000:],
            },
        )
        if rc != 0 or pred is None or not pred_path.is_file():
            return _blocked_result(
                capability_id=capability_id,
                scenario_id=scenario_id,
                split=split,
                cap_dir=cap_dir,
                pred_dir=pred_dir,
                command=command,
                audit={**audit, "conclusion": "BLOCKED", "blockers": ["LISFLOOD_EXEC_FAILED"]},
            )
        return {
            "ok": True,
            "official_status": "OFFICIAL_READY",
            "artifact_paths": [str(pred_path)],
            "pred_depth": str(pred_path),
            "recipe_log": str(recipe_log),
            "official_command": command,
            "shape": list(pred.shape),
        }

    # CAP-FL2-04 SFINCS
    scen = scenario_dir(split, scenario_id)
    driver_root = cap_dir / "driver" / scenario_id
    dem, initial, truth = load_scenario_arrays(split, scenario_id)
    inp_path, meta = materialize_sfincs_driver(scen, out_dir=driver_root, max_dim=200)
    if not meta.get("ok"):
        return _blocked_result(
            capability_id=capability_id,
            scenario_id=scenario_id,
            split=split,
            cap_dir=cap_dir,
            pred_dir=pred_dir,
            command=command,
            audit={**audit, "conclusion": "BLOCKED", "blockers": ["SFINCS_DRIVER_MATERIALIZE_FAILED"]},
        )
    rc, stdout, stderr = run_sfincs(driver_root)
    pred = read_sfincs_depth_sequence(driver_root, n_steps=truth.shape[0])
    if pred is not None and write_npz:
        if pred.shape[1:] != truth.shape[1:]:
            from skimage.transform import resize

            out = np.zeros((pred.shape[0], *truth.shape[1:]), dtype=np.float32)
            for t in range(pred.shape[0]):
                out[t] = resize(pred[t], truth.shape[1:], order=1, preserve_range=True, anti_aliasing=True)
            pred = out
        _write_npz(
            pred_path,
            pred=pred,
            scenario_id=scenario_id,
            split=split,
            capability_id=capability_id,
        )
    recipe_log = _write_recipe_log(
        cap_dir,
        capability_id=capability_id,
        command=command,
        returncode=rc,
        inference_mode="sfincs_singularity",
        extra={
            "scenario_id": scenario_id,
            "split": split,
            "driver_meta": meta,
            "stdout_tail": stdout[-2000:],
            "stderr_tail": stderr[-2000:],
        },
    )
    if rc != 0 or pred is None or not pred_path.is_file():
        return _blocked_result(
            capability_id=capability_id,
            scenario_id=scenario_id,
            split=split,
            cap_dir=cap_dir,
            pred_dir=pred_dir,
            command=command,
            audit={**audit, "conclusion": "BLOCKED", "blockers": ["SFINCS_EXEC_FAILED"]},
        )
    return {
        "ok": True,
        "official_status": "OFFICIAL_READY",
        "artifact_paths": [str(pred_path)],
        "pred_depth": str(pred_path),
        "recipe_log": str(recipe_log),
        "official_command": command,
        "shape": list(pred.shape),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="CAP-FL2-04~06 official solver infer (M3)")
    ap.add_argument("--capability-id", required=True, choices=solver_caps())
    ap.add_argument("--split", default=DEFAULT_SPLIT)
    ap.add_argument("--scenario-id", default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    sid = args.scenario_id or default_scenario_id(split=args.split)
    out = run_official_solver_infer(
        args.capability_id,
        scenario_id=sid,
        split=args.split,
        out_dir=Path(args.out_dir) if args.out_dir else None,
    )
    print(json.dumps(out, indent=2))
    return 0 if out.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
