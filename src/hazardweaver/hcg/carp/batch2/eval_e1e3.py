"""E1-E3 SeisBench picker inference + pick F1 (isolated seisbench subprocess)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from hazardweaver.hcg.carp.acquire.env_routing import SEISBENCH_PYTHON
from hazardweaver.hcg.carp.acquire.seisbench_paths import seisbench_subprocess_env
from hazardweaver.hcg.carp.batch2.fixtures import E1E3_SMOKE, ensure_e1e3_smoke_fixture
from hazardweaver.hcg.carp.batch2.replay_certificate import write_metrics, write_replay_manifest

WORKER = Path(__file__).resolve().parent / "eval_e1e3_infer_worker.py"

CAP_TO_STORE = {
    "CAP-E1E3-02": ("phasenet", "original", "PhaseNet", "RF-CNN-PHASE-PICKING"),
    "CAP-E1E3-03": ("eqtransformer", "original", "EQTransformer", "RF-ATTENTION-RECURRENT-DETECTIO"),
    "CAP-E1E3-04": ("gpd", "stead", "GPD", "RF-WINDOW-CLASSIFIER-PICKER"),
}


def _run_worker(store: str, weight_variant: str, fixture_dir: Path) -> Dict[str, Any]:
    if not SEISBENCH_PYTHON.is_file():
        return {"ok": False, "reason": f"missing {SEISBENCH_PYTHON}"}
    proc = subprocess.run(
        [
            str(SEISBENCH_PYTHON),
            str(WORKER),
            "--model-store",
            store,
            "--weight-variant",
            weight_variant,
            "--fixture-dir",
            str(fixture_dir),
        ],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
        env=seisbench_subprocess_env(),
    )
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip().splitlines()
        return {"ok": False, "reason": err[-1] if err else "infer worker failed"}
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except json.JSONDecodeError:
        return {"ok": False, "reason": f"bad worker output: {proc.stdout[:200]}"}


def eval_picker(
    capability_id: str,
    *,
    fixture_dir: Optional[Path] = None,
    out_base: Optional[Path] = None,
) -> Dict[str, Any]:
    if capability_id not in CAP_TO_STORE:
        return {"ok": False, "reason": f"unsupported cap {capability_id}"}
    store, weight_variant, model_class, family_id = CAP_TO_STORE[capability_id]
    fix = fixture_dir or ensure_e1e3_smoke_fixture().parent
    result = _run_worker(store, weight_variant, fix)
    if not result.get("ok"):
        write_replay_manifest(
            "E1-E3",
            capability_id,
            family_id=family_id,
            exec_ok=False,
            notes=result.get("reason", "infer failed"),
            base=out_base,
        )
        return result
    pick_f1 = float(result.get("pick_f1", 0.0))
    metrics = {
        "metric_name": "pick_f1",
        "metric_value": pick_f1,
        "model_store": store,
        "weight_variant": weight_variant,
        "model_class": model_class,
        "fixture_dir": str(fix),
        "synthetic_only": True,
        "n_picks": result.get("n_picks", 0),
        "note": "Smoke fixture; anchor split pick F1 requires SeisBench benchmark dataset (Batch 2b)",
    }
    write_metrics("E1-E3", capability_id, metrics, base=out_base)
    write_replay_manifest(
        "E1-E3",
        capability_id,
        family_id=family_id,
        exec_ok=True,
        metric_name="pick_f1",
        metric_value=pick_f1,
        notes=f"SeisBench {model_class} inference smoke",
        extra={"weight_variant": weight_variant},
        base=out_base,
    )
    return {"ok": True, "metrics": metrics}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--capability-id", required=True)
    ap.add_argument("--out-base", default="")
    args = ap.parse_args()
    out = Path(args.out_base) if args.out_base else None
    result = eval_picker(args.capability_id, out_base=out)
    print(json.dumps(result, indent=2))
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
