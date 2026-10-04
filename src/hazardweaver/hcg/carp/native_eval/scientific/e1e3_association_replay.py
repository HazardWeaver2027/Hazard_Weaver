"""Official PyOcto/GaMMA/REAL association replay for CAP-E1E3-05/06/07 (DL-111)."""

from __future__ import annotations

import json
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from hazardweaver.hcg.carp.batch2.replay_certificate import write_metrics, write_replay_manifest
from hazardweaver.hcg.carp.native_eval.blocked import write_blocked
from hazardweaver.hcg.carp.native_eval.scientific.e1e3_picker_eval import eval_picker_cap_both_splits
from hazardweaver.hcg.carp.scientific.e1e3_data import (
    DATA_SOURCE,
    TASKPACK,
    gamma_vendor_ready,
    official_test_trace_ids,
    pyocto_vendor_ready,
    real_vendor_ready,
    seisbench_python,
)
from hazardweaver.hcg.carp.scientific.e1e3_official_commands import CAP_MODEL_CONFIG, build_official_command, cap_family_id
from hazardweaver.hcg.carp.scientific.paths import SCIENTIFIC_RUNS_ROOT, cap_scientific_dir

WORKER = Path(__file__).resolve().parent / "e1e3_association_worker.py"
ASSOC_TIMEOUT_S = 3600


def _write_recipe_log(
    cap_dir: Path,
    *,
    capability_id: str,
    command: str,
    returncode: int = 0,
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
        "note": "DL-111 official association vendor replay",
    }
    if extra:
        payload.update(extra)
    path = cap_dir / "recipe_run.log"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _write_predictions(cap_dir: Path, *, predictions: List[Dict[str, Any]]) -> Path:
    out_dir = cap_dir / "predictions" / "test"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "picks.json"
    path.write_text(json.dumps({"split": "official_test", "predictions": predictions}, indent=2) + "\n", encoding="utf-8")
    return path


def _run_association_worker(
    capability_id: str,
    mode: str,
    picks_path: Path,
    work_dir: Path,
) -> Dict[str, Any]:
    py = seisbench_python()
    if not py.is_file():
        return {"ok": False, "reason": f"missing seisbench python {py}"}
    proc = subprocess.run(
        [
            str(py),
            str(WORKER),
            "--mode",
            mode,
            "--predictions",
            str(picks_path),
            "--work-dir",
            str(work_dir),
            "--split",
            "official_test",
        ],
        capture_output=True,
        text=True,
        timeout=ASSOC_TIMEOUT_S,
        check=False,
    )
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip().splitlines()
        return {"ok": False, "reason": err[-1] if err else "association worker failed"}
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except json.JSONDecodeError:
        return {"ok": False, "reason": f"bad worker output: {proc.stdout[:300]}"}


def eval_association_replay(capability_id: str, *, out_base: Optional[Path] = None) -> Dict[str, Any]:
    spec = CAP_MODEL_CONFIG[capability_id]
    mode = spec["mode"]
    if mode == "gamma_association" and not gamma_vendor_ready():
        return write_blocked(TASKPACK, capability_id, "GaMMA vendor missing", out_base=out_base)
    if mode == "real_association" and not real_vendor_ready():
        return write_blocked(TASKPACK, capability_id, "REAL vendor missing", out_base=out_base)
    if mode == "pyocto_association" and not pyocto_vendor_ready():
        return write_blocked(TASKPACK, capability_id, "PyOcto vendor missing", out_base=out_base)

    out_root = out_base or SCIENTIFIC_RUNS_ROOT
    cap_dir = cap_scientific_dir(TASKPACK, capability_id, runs_root=out_root)
    picks_path = cap_dir / "predictions" / "test" / "picks.json"
    if picks_path.is_file():
        predictions = json.loads(picks_path.read_text(encoding="utf-8")).get("predictions", [])
    else:
        picker = eval_picker_cap_both_splits("CAP-E1E3-02", out_base=out_base)
        if not picker.get("ok"):
            return write_blocked(TASKPACK, capability_id, "upstream picker failed", out_base=out_base)
        predictions = picker.get("predictions", [])
        _write_predictions(cap_dir, predictions=predictions)
    work_dir = cap_dir / "association_work"
    result = _run_association_worker(capability_id, mode, picks_path, work_dir)
    if not result.get("ok"):
        return write_blocked(TASKPACK, capability_id, result.get("reason", "association failed"), out_base=out_base)

    assoc_src = work_dir / "association_events.json"
    if assoc_src.is_file():
        (cap_dir / "association_events.json").write_text(assoc_src.read_text(encoding="utf-8"), encoding="utf-8")

    score = float(result["association_score"])
    cmd = result.get("official_command") or build_official_command(capability_id)
    _write_recipe_log(
        cap_dir,
        capability_id=capability_id,
        command=cmd,
        returncode=0,
        extra={"vendor": mode, "n_events_detected": result.get("n_events", 0)},
    )
    metrics = {
        "metric_name": spec["metric_name"],
        "metric_value": score,
        "stage": mode,
        "n_traces": len(official_test_trace_ids()),
        "n_events_detected": int(result.get("n_events", 0)),
        "synthetic_only": False,
        "data_source": DATA_SOURCE,
        "official_cli_invoked": True,
        "vendor_replay": mode,
        "note": f"{mode} official PyOcto vendor replay; association_score={score:.4f}",
    }
    write_metrics(TASKPACK, capability_id, metrics, base=out_root)
    write_replay_manifest(
        TASKPACK,
        capability_id,
        family_id=cap_family_id(capability_id),
        exec_ok=True,
        metric_name=spec["metric_name"],
        metric_value=score,
        notes=metrics["note"],
        base=out_root,
    )
    return {"ok": True, "metrics": metrics}
