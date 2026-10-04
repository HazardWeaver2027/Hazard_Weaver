"""Official FloodCastBench G2 inference for CAP-FL2-01~03 (M2 neural route seal).

Never calls smoke/heuristic rollouts. When audit is BLOCKED, writes honest recipe_run.log,
per-scenario blocked metadata, and cap-level BLOCKED.json.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from hazardweaver.hcg.carp.scientific.fl2_data import (
    PROJECT_ROOT,
    TASKPACK,
    full_input_schema_ready,
    load_scenario_arrays,
    prediction_path,
    prediction_subdir,
    predictions_base_dir,
    scenario_dir,
    scenario_ids,
    scenarios_ready,
    split_scenarios_ready,
    taskpack_repo_pin,
)
from hazardweaver.hcg.carp.scientific.fl2_official_commands import CAP_MODEL_CONFIG, G2_CAPS
from hazardweaver.hcg.carp.scientific.paths import cap_scientific_dir

DEFAULT_SPLIT = "official_test"
AUDIT_DOC = (
    PROJECT_ROOT
    / "docs/engineering/hcg/external_expansion/FL-2/FL2_OFFICIAL_INFERENCE_AUDIT.md"
)
FLOODCAST_REPO = PROJECT_ROOT / "data/vendor/floodcastbench/floodcast_repo/repo"
BENCHMARK_MAIN = FLOODCAST_REPO / "FloodCast Benchmark" / "main"

MODEL_BY_CAP: Dict[str, str] = {
    "CAP-FL2-01": "unet",
    "CAP-FL2-02": "fno",
    "CAP-FL2-03": "fno_plus",
}

CKPT_NAME_BY_CAP: Dict[str, str] = {
    "CAP-FL2-01": "unet_best.pt",
    "CAP-FL2-02": "fno_best.pt",
    "CAP-FL2-03": "fno_plus_best.pt",
}

SPLIT_TO_PRED_SUBDIR = {
    "official_test": "test",
    "hwb_holdout": "holdout",
    "official_test_high": "test",
    "hwb_holdout_high": "holdout",
}


def g2_caps() -> List[str]:
    return list(G2_CAPS)


def default_scenario_id(*, split: str = DEFAULT_SPLIT) -> str:
    ids = scenario_ids(split)
    if not ids:
        raise FileNotFoundError(f"no scenarios in split {split!r}")
    return ids[0]


def data_package_ready() -> bool:
    """Scenario bundles + repo pin (does not require PI signoff)."""
    return scenarios_ready() and _read_training_code_status() != "BLOCKED"


def _read_training_code_status() -> str:
    pin = taskpack_repo_pin(TASKPACK)
    if not pin.is_file():
        return "BLOCKED"
    data = json.loads(pin.read_text(encoding="utf-8"))
    return str(data.get("training_code_status") or "BLOCKED")


def checkpoint_path(capability_id: str) -> Path:
    ckpt_name = CKPT_NAME_BY_CAP[capability_id]
    return cap_scientific_dir(TASKPACK, capability_id) / "checkpoint" / ckpt_name


def audit_official_readiness(capability_id: str) -> Dict[str, Any]:
    """Return per-cap audit conclusion matching FL2_OFFICIAL_INFERENCE_AUDIT.md."""
    if capability_id not in MODEL_BY_CAP:
        return {
            "capability_id": capability_id,
            "conclusion": "BLOCKED",
            "blockers": ["UNKNOWN_CAPABILITY"],
            "audit_doc": str(AUDIT_DOC.relative_to(PROJECT_ROOT)),
        }

    blockers: List[str] = []
    training_status = _read_training_code_status()
    ckpt = checkpoint_path(capability_id)

    infer_entry_ok = BENCHMARK_MAIN.is_file() and BENCHMARK_MAIN.stat().st_size > 10
    if not infer_entry_ok:
        blockers.append("NO_INFER_ENTRY")
    if training_status == "DATA_GEN_ONLY":
        blockers.append("TRAINING_CODE_DATA_GEN_ONLY")
    if not ckpt.is_file():
        blockers.append("NO_RELEASED_CKPT")
    schema_ok, schema_missing = full_input_schema_ready()
    if not schema_ok:
        blockers.append("INPUT_SCHEMA_MISMATCH")

    base = {
        "capability_id": capability_id,
        "model": MODEL_BY_CAP[capability_id],
        "training_code_status": training_status,
        "infer_entry": str(BENCHMARK_MAIN),
        "infer_entry_bytes": BENCHMARK_MAIN.stat().st_size if BENCHMARK_MAIN.is_file() else 0,
        "checkpoint_path": str(ckpt),
        "checkpoint_present": ckpt.is_file(),
        "audit_doc": str(AUDIT_DOC.relative_to(PROJECT_ROOT)),
    }
    if schema_missing:
        base["input_schema_missing_sample"] = schema_missing[:5]
        base["input_schema_missing_count"] = len(schema_missing)
    base["input_schema_ready"] = schema_ok
    if blockers:
        return {
            **base,
            "conclusion": "BLOCKED",
            "blockers": blockers,
            "cos_fl2_match": "mismatch",
        }
    return {
        **base,
        "conclusion": "OFFICIAL_READY",
        "blockers": [],
        "cos_fl2_match": "exact",
    }


def predictions_subdir(split: str) -> str:
    return prediction_subdir(split)


def build_official_command(
    capability_id: str,
    *,
    scenario_id: str,
    split: str = DEFAULT_SPLIT,
    out_path: Optional[Path] = None,
) -> str:
    if capability_id not in MODEL_BY_CAP:
        raise ValueError(f"unsupported capability {capability_id}")
    scen = scenario_dir(split, scenario_id)
    pred = out_path or prediction_path(cap_scientific_dir(TASKPACK, capability_id), split, scenario_id)
    model = MODEL_BY_CAP[capability_id]
    ckpt = checkpoint_path(capability_id)
    return (
        f"python {BENCHMARK_MAIN} "
        f"--model {model} "
        f"--checkpoint {ckpt} "
        f"--scenario-dir {scen} "
        f"--output {pred}"
    )


def write_cap_blocked_json(capability_id: str, *, cap_dir: Optional[Path] = None) -> Path:
    audit = audit_official_readiness(capability_id)
    cap_root = cap_dir or cap_scientific_dir(TASKPACK, capability_id)
    cap_root.mkdir(parents=True, exist_ok=True)
    template_cmd = build_official_command(
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
        "model": MODEL_BY_CAP.get(capability_id),
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
        "note": "M2 FL-2 official G2 infer (no smoke)",
    }
    if extra:
        payload.update(extra)
    path = cap_dir / "recipe_run.log"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _write_npz(path: Path, *, pred: np.ndarray, scenario_id: str, split: str, capability_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        pred_depth=pred.astype(np.float32),
        scenario_id=np.array(scenario_id),
        split=np.array(split),
        capability_id=np.array(capability_id),
    )


def run_official_g2_infer(
    capability_id: str,
    *,
    scenario_id: str,
    split: str = DEFAULT_SPLIT,
    out_dir: Optional[Path] = None,
    write_npz: bool = True,
) -> Dict[str, Any]:
    if capability_id not in MODEL_BY_CAP:
        return {"ok": False, "error": f"unsupported G2 capability {capability_id}"}
    if not split_scenarios_ready(split):
        return {"ok": False, "error": f"FL-2 scenario data not ready for split {split!r}"}

    audit = audit_official_readiness(capability_id)
    cap_dir = Path(out_dir) if out_dir else cap_scientific_dir(TASKPACK, capability_id)
    pred_path = prediction_path(cap_dir, split, scenario_id)
    pred_dir = pred_path.parent
    pred_dir.mkdir(parents=True, exist_ok=True)
    command = build_official_command(
        capability_id,
        scenario_id=scenario_id,
        split=split,
        out_path=pred_path,
    )

    if audit["conclusion"] == "BLOCKED":
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

    proc = subprocess.run(
        ["bash", "-lc", command],
        capture_output=True,
        text=True,
        timeout=3600,
        check=False,
    )
    recipe_log = _write_recipe_log(
        cap_dir,
        capability_id=capability_id,
        command=command,
        returncode=proc.returncode,
        inference_mode="official",
        extra={
            "stdout_tail": (proc.stdout or "")[-2000:],
            "stderr_tail": (proc.stderr or "")[-2000:],
            "scenario_id": scenario_id,
            "split": split,
        },
    )
    if proc.returncode != 0 or not pred_path.is_file():
        return {
            "ok": False,
            "blocked": True,
            "official_status": "BLOCKED_OFFICIAL",
            "error": f"official infer failed rc={proc.returncode}",
            "recipe_log": str(recipe_log),
            "official_command": command,
        }

    _, _, truth = load_scenario_arrays(split, scenario_id)
    if pred_path.suffix == ".npz":
        pred = np.load(pred_path)["pred_depth"]
    else:
        pred = np.load(pred_path)
    if pred.shape != truth.shape:
        return {
            "ok": False,
            "error": f"pred shape {pred.shape} != truth {truth.shape}",
            "recipe_log": str(recipe_log),
        }
    if write_npz and pred_path.suffix != ".npz":
        _write_npz(
            pred_path.with_suffix(".npz"),
            pred=pred,
            scenario_id=scenario_id,
            split=split,
            capability_id=capability_id,
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


# M1 compatibility aliases
def m1_default_scenario_id(*, split: str = DEFAULT_SPLIT) -> str:
    return default_scenario_id(split=split)


def audit_official_readiness_legacy() -> Dict[str, Any]:
    return audit_official_readiness("CAP-FL2-01")


def build_official_unet_command(
    *,
    scenario_id: str,
    split: str = DEFAULT_SPLIT,
    out_path: Optional[Path] = None,
) -> str:
    return build_official_command(
        "CAP-FL2-01",
        scenario_id=scenario_id,
        split=split,
        out_path=out_path,
    )


def run_official_unet_infer(
    *,
    capability_id: str = "CAP-FL2-01",
    scenario_id: str,
    split: str = DEFAULT_SPLIT,
    out_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    return run_official_g2_infer(
        capability_id,
        scenario_id=scenario_id,
        split=split,
        out_dir=out_dir,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="CAP-FL2-01~03 official G2 infer (M2)")
    ap.add_argument("--capability-id", default="CAP-FL2-01", choices=g2_caps())
    ap.add_argument("--split", default=DEFAULT_SPLIT)
    ap.add_argument("--scenario-id", default=None)
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    sid = args.scenario_id or default_scenario_id(split=args.split)
    out = run_official_g2_infer(
        args.capability_id,
        scenario_id=sid,
        split=args.split,
        out_dir=Path(args.out_dir) if args.out_dir else None,
    )
    print(json.dumps(out, indent=2))
    return 0 if out.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
