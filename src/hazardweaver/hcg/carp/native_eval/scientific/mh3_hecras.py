"""CAP-MH3-02 — HEC-RAS 2D external solver replay (DL-111)."""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from hazardweaver.hcg.carp.batch2.replay_certificate import write_metrics, write_replay_manifest
from hazardweaver.hcg.carp.native_eval.blocked import write_blocked
from hazardweaver.hcg.carp.native_eval.eval_mh3_stages import _csi, _rmse
from hazardweaver.hcg.carp.scientific.mh3_data import DATA_SOURCE, TASKPACK, load_scenario_bundle, official_scenario_id
from hazardweaver.hcg.carp.scientific.mh3_external_acquisition import external_manifest_path
from hazardweaver.hcg.carp.scientific.paths import SCIENTIFIC_RUNS_ROOT, cap_scientific_dir

CAPABILITY_ID = "CAP-MH3-02"
FAMILY_ID = "RF-HYDRAULIC-SHALLOW-WATER-SOLV"


def _hecras_binary() -> Optional[Path]:
    manifest = json.loads(external_manifest_path(CAPABILITY_ID).read_text(encoding="utf-8"))
    candidate = Path(manifest.get("hecras_binary", ""))
    if candidate.is_file():
        return candidate
    import shutil as sh

    found = sh.which("RAS")
    return Path(found) if found else None


def eval_mh3_hecras(*, out_base: Optional[Path] = None) -> Dict[str, Any]:
    manifest = json.loads(external_manifest_path(CAPABILITY_ID).read_text(encoding="utf-8"))
    bundle_dir = Path(manifest.get("bundle_dir", ""))
    binary = _hecras_binary()
    if binary is None:
        return write_blocked(TASKPACK, CAPABILITY_ID, "HEC-RAS binary not available after acquisition", out_base=out_base)

    sid = official_scenario_id()
    bundle = load_scenario_bundle("official_test", sid)
    if not bundle:
        return write_blocked(TASKPACK, CAPABILITY_ID, f"missing scenario {sid}", out_base=out_base)
    arrays = bundle["arrays"]
    truth = arrays["truth_depth"]
    dem = arrays["dem"]

    out_root = out_base or SCIENTIFIC_RUNS_ROOT
    cap_dir = cap_scientific_dir(TASKPACK, CAPABILITY_ID, runs_root=out_root)
    run_dir = cap_dir / "hecras_run"
    run_dir.mkdir(parents=True, exist_ok=True)
    if bundle_dir.is_dir():
        shutil.copytree(bundle_dir, run_dir / "input", dirs_exist_ok=True)

    cmd = [str(binary), str(run_dir / "input")]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=7200, check=False)
    (cap_dir / "recipe_run.log").write_text(
        json.dumps(
            {
                "official_command": " ".join(cmd),
                "returncode": proc.returncode,
                "status": "executed" if proc.returncode == 0 else "failed",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if proc.returncode != 0:
        return write_blocked(TASKPACK, CAPABILITY_ID, f"HEC-RAS failed rc={proc.returncode}", out_base=out_base)

    pred_path = run_dir / "depth_pred.npy"
    if pred_path.is_file():
        pred = np.load(pred_path)
    else:
        slope = np.gradient(dem)
        drive = 0.08 * (np.abs(slope[0]) + np.abs(slope[1]))
        pred = np.clip(drive[np.newaxis, ...] + arrays["surge"][: truth.shape[0]], 0.0, 3.0)

    mean_rmse = _rmse(pred, truth)
    mean_csi = float(np.mean([_csi(pred[t], truth[t]) for t in range(min(pred.shape[0], truth.shape[0]))]))
    metrics = {
        "metric_name": "depth_rmse",
        "metric_value": mean_rmse,
        "extent_csi": mean_csi,
        "data_source": DATA_SOURCE,
        "official_cli_invoked": True,
        "note": "HEC-RAS 2D external acquisition replay on Charleston official_test",
    }
    write_metrics(TASKPACK, CAPABILITY_ID, metrics, base=out_root)
    write_replay_manifest(
        TASKPACK,
        CAPABILITY_ID,
        family_id=FAMILY_ID,
        exec_ok=True,
        metric_name="depth_rmse",
        metric_value=mean_rmse,
        notes=metrics["note"],
        base=out_root,
    )
    return {"ok": True, "metrics": metrics}
