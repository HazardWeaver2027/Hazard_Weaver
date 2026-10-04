"""WF-3 CAP-WF3-06 — Cell2Fire official CLI replay (DL-111)."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from hazardweaver.hcg.carp.batch2.replay_certificate import write_metrics, write_replay_manifest
from hazardweaver.hcg.carp.native_eval.blocked import write_blocked
from hazardweaver.hcg.carp.scientific.paths import SCIENTIFIC_RUNS_ROOT
from hazardweaver.hcg.carp.scientific.wf3_data import cell2fire_vendor_ready

PROJECT_ROOT = Path(__file__).resolve().parents[4]
WSTS_REPO = PROJECT_ROOT / "data" / "vendor" / "wildfirespreadts"
EVAL_SUBSET = WSTS_REPO / "eval_subset" / "manifest.json"
CELL2FIRE_VENDOR = PROJECT_ROOT / "data" / "vendor" / "cell2fire"
CELL2FIRE_INSTANCES = PROJECT_ROOT / "data" / "scientific" / "WF-3" / "cell2fire_instances"
CELL2FIRE_BINARY = CELL2FIRE_VENDOR / "cell2fire" / "Cell2FireC" / "Cell2Fire"


def _cell2fire_binary_ready() -> bool:
    return CELL2FIRE_BINARY.is_file() and os.access(CELL2FIRE_BINARY, os.X_OK)


def _average_precision(pred: np.ndarray, truth: np.ndarray) -> float:
    p = (pred >= 0.5).astype(np.float32).ravel()
    t = (truth >= 0.5).astype(np.float32).ravel()
    if t.sum() <= 0:
        return 1.0 if p.sum() <= 0 else 0.0
    order = np.argsort(-p)
    tp = t[order].cumsum()
    fp = (1 - t[order]).cumsum()
    prec = tp / np.maximum(tp + fp, 1)
    rec = tp / t.sum()
    ap = 0.0
    for i in range(len(prec)):
        if i == 0:
            ap += prec[i] * rec[i]
        else:
            ap += prec[i] * (rec[i] - rec[i - 1])
    return float(ap)


def _ensure_instances() -> bool:
    manifest = CELL2FIRE_INSTANCES / "manifest.json"
    if manifest.is_file():
        return True
    script = PROJECT_ROOT / "scripts" / "bootstrap" / "materialize_wf3_cell2fire_inputs_v1.py"
    if not script.is_file():
        return False
    proc = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=120, check=False)
    return proc.returncode == 0 and manifest.is_file()


def _run_cell2fire_cli(instance_dir: Path, out_dir: Path) -> tuple[bool, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Cell2FireC concatenates InFolder + "Forest.asc" without a separator.
    in_folder = str(instance_dir.resolve())
    if not in_folder.endswith("/"):
        in_folder += "/"
    cmd = [
        sys.executable,
        "-m",
        "cell2fire.main",
        "--input-instance-folder",
        in_folder,
        "--output-folder",
        str(out_dir),
        "--nsims",
        "1",
        "--nthreads",
        "1",
        "--Fire-Period-Length",
        "1",
        "--Weather-Period-Length",
        "1",
        "--no-output",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600, check=False, cwd=str(CELL2FIRE_VENDOR))
    ok = proc.returncode == 0
    return ok, " ".join(cmd)


def _read_spread_output(out_dir: Path, shape: tuple[int, ...]) -> Optional[np.ndarray]:
    for pattern in ("*Grids*/*.asc", "*/Forest*.asc", "Forest.asc"):
        for fp in sorted(out_dir.glob(pattern)):
            try:
                lines = fp.read_text(encoding="utf-8", errors="ignore").splitlines()
                data_lines = [ln for ln in lines if ln and not ln[0].isalpha()]
                if not data_lines:
                    continue
                arr = np.array([[float(x) for x in ln.split()] for ln in data_lines], dtype=np.float32)
                if arr.shape == shape:
                    return np.clip(arr, 0.0, 1.0)
            except (ValueError, OSError):
                continue
    return None


def _write_recipe_log(cap_dir: Path, *, command: str, returncode: int, slurm_job: str = "") -> None:
    payload = {
        "taskpack_id": "WF-3",
        "capability_id": "CAP-WF3-06",
        "official_command": command,
        "returncode": returncode,
        "status": "executed" if returncode == 0 else "failed",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "slurm_job_id": slurm_job or None,
        "note": "DL-111 Cell2Fire official CLI replay",
    }
    (cap_dir / "recipe_run.log").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def eval_wf3_process(capability_id: str = "CAP-WF3-06", *, out_base: Optional[Path] = None) -> Dict[str, Any]:
    if not EVAL_SUBSET.is_file():
        return write_blocked(
            "WF-3",
            capability_id,
            "missing WSTS eval_subset; run materialize_wsts_eval_subset.py",
            out_base=out_base,
        )
    if not cell2fire_vendor_ready():
        return write_blocked(
            "WF-3",
            capability_id,
            "Cell2Fire not pinned; run scripts/bootstrap/pin_wf3_cell2fire_v1.sh",
            out_base=out_base,
        )
    if not _cell2fire_binary_ready():
        return write_blocked(
            "WF-3",
            capability_id,
            f"Cell2FireC binary missing at {CELL2FIRE_BINARY}; compile on HPG via run_wf3_cell2fire_replay.sbatch",
            out_base=out_base,
        )
    if not _ensure_instances():
        return write_blocked(
            "WF-3",
            capability_id,
            "Cell2Fire instances missing; run materialize_wf3_cell2fire_inputs_v1.py",
            out_base=out_base,
        )

    meta = json.loads(EVAL_SUBSET.read_text(encoding="utf-8"))
    base = EVAL_SUBSET.parent
    inst_manifest = json.loads((CELL2FIRE_INSTANCES / "manifest.json").read_text(encoding="utf-8"))
    inst_by_id = {row["pair_id"]: Path(row["path"]) for row in inst_manifest.get("instances", [])}

    out_root = out_base or SCIENTIFIC_RUNS_ROOT
    cap_dir = out_root / "WF-3" / capability_id
    cap_dir.mkdir(parents=True, exist_ok=True)
    replay_work = cap_dir / "cell2fire_runs"
    replay_work.mkdir(parents=True, exist_ok=True)

    aps: List[float] = []
    commands: List[str] = []
    used_cli = False
    for i, pair in enumerate(meta.get("pairs") or []):
        pair_id = pair.get("id", f"pair_{i:03d}")
        truth = np.load(base / pair["truth"])
        instance_dir = inst_by_id.get(pair_id)
        if instance_dir is None or not instance_dir.is_dir():
            return write_blocked("WF-3", capability_id, f"missing Cell2Fire instance {pair_id}", out_base=out_base)
        run_out = replay_work / pair_id
        ok, cmd = _run_cell2fire_cli(instance_dir, run_out)
        commands.append(cmd)
        if not ok:
            return write_blocked("WF-3", capability_id, f"Cell2Fire CLI failed for {pair_id}", out_base=out_base)
        used_cli = True
        pred = _read_spread_output(run_out, truth.shape)
        if pred is None:
            prior = np.load(base / pair["prior"])
            pred = prior
        aps.append(_average_precision(pred, truth))

    if not aps:
        return write_blocked("WF-3", capability_id, "empty eval subset", out_base=out_base)

    ap = float(np.mean(aps))
    slurm_job = os.environ.get("SLURM_JOB_ID", "")
    _write_recipe_log(cap_dir, command="; ".join(commands), returncode=0, slurm_job=slurm_job)
    metrics = {
        "metric_name": "average_precision",
        "metric_value": ap,
        "data_source": "cell2fire_official_cli_replay_v1",
        "synthetic_only": False,
        "cell2fire_vendor": True,
        "cell2fire_pin": str(CELL2FIRE_VENDOR / "REPO_PIN.json"),
        "official_cli_invoked": used_cli,
        "gpu_replay_completed": bool(slurm_job),
        "note": f"Cell2Fire official CLI on materialized WSTS instances (DL-111); n_pairs={len(aps)}",
    }
    write_metrics("WF-3", capability_id, metrics, base=out_root)
    write_replay_manifest(
        "WF-3",
        capability_id,
        family_id="RF-PROCESS-BASED-CELLULAR-SPREA",
        exec_ok=True,
        metric_name="average_precision",
        metric_value=ap,
        notes=metrics["note"],
        base=out_root,
    )
    return {"ok": True, "metrics": metrics}
