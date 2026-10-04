"""E1-E3 SeisBench anchor split pick F1 eval."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from hazardweaver.hcg.carp.acquire.env_routing import SEISBENCH_PYTHON
from hazardweaver.hcg.carp.acquire.seisbench_paths import seisbench_subprocess_env
from hazardweaver.hcg.carp.batch2.eval_e1e3 import CAP_TO_STORE, eval_picker
from hazardweaver.hcg.carp.native_eval.eval_e1e3_stages import eval_e1e3_stage
from hazardweaver.hcg.carp.batch2.replay_certificate import write_metrics, write_replay_manifest

PROJECT_ROOT = Path(__file__).resolve().parents[4]
ANCHOR_MANIFEST = PROJECT_ROOT / "data" / "processed" / "e1e3_seisbench_anchor_v1" / "manifest.json"
ANCHOR_FIXTURES = PROJECT_ROOT / "data" / "processed" / "e1e3_seisbench_anchor_v1" / "traces"
WORKER = Path(__file__).resolve().parent / "eval_e1e3_anchor_worker.py"


def _run_anchor_worker(store: str, weight_variant: str, trace_path: Path, meta: Dict[str, Any]) -> Dict[str, Any]:
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
            "--trace-path",
            str(trace_path),
            "--p-pick-s",
            str(meta.get("p_pick_s", 10.0)),
            "--s-pick-s",
            str(meta.get("s_pick_s", 20.0)),
            "--sample-rate",
            str(meta.get("sample_rate", 100.0)),
            "--tolerance-s",
            str(meta.get("tolerance_s", 0.5)),
        ],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
        env=seisbench_subprocess_env(),
    )
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip().splitlines()
        return {"ok": False, "reason": err[-1] if err else "anchor worker failed"}
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except json.JSONDecodeError:
        return {"ok": False, "reason": f"bad worker output: {proc.stdout[:200]}"}


def eval_e1e3_anchor(capability_id: str, *, out_base: Optional[Path] = None) -> Dict[str, Any]:
    if capability_id not in CAP_TO_STORE:
        return {"ok": False, "reason": f"unsupported cap {capability_id}"}
    if not ANCHOR_MANIFEST.is_file():
        return eval_picker(capability_id, out_base=out_base)
    store, weight_variant, model_class, family_id = CAP_TO_STORE[capability_id]
    manifest = json.loads(ANCHOR_MANIFEST.read_text(encoding="utf-8"))
    traces = manifest.get("traces") or []
    if not traces:
        return {"ok": False, "reason": "empty anchor manifest traces"}
    f1s = []
    for row in traces:
        trace_path = ANCHOR_FIXTURES / row["trace_file"]
        if not trace_path.is_file():
            continue
        result = _run_anchor_worker(store, weight_variant, trace_path, row)
        if result.get("ok"):
            f1s.append(float(result.get("pick_f1", 0.0)))
    if not f1s:
        return {"ok": False, "reason": "no anchor traces evaluated"}
    mean_f1 = float(sum(f1s) / len(f1s))
    metrics = {
        "metric_name": "pick_f1",
        "metric_value": mean_f1,
        "model_store": store,
        "weight_variant": weight_variant,
        "model_class": model_class,
        "manifest_path": str(ANCHOR_MANIFEST),
        "synthetic_only": False,
        "n_traces": len(f1s),
        "note": "SeisBench anchor dev subset",
    }
    write_metrics("E1-E3", capability_id, metrics, base=out_base)
    write_replay_manifest(
        "E1-E3",
        capability_id,
        family_id=family_id,
        exec_ok=True,
        metric_name="pick_f1",
        metric_value=mean_f1,
        notes=f"SeisBench {model_class} anchor pick F1",
        extra={"weight_variant": weight_variant},
        base=out_base,
    )
    return {"ok": True, "metrics": metrics}


def eval_e1e3_cap(capability_id: str, *, out_base: Optional[Path] = None) -> Dict[str, Any]:
    from hazardweaver.hcg.carp.scientific.e1e3_data import scientific_data_ready

    if scientific_data_ready():
        from hazardweaver.hcg.carp.native_eval.scientific.e1e3_seisbench import eval_e1e3_scientific
        from hazardweaver.hcg.carp.scientific.paths import SCIENTIFIC_RUNS_ROOT

        sci_base = out_base or SCIENTIFIC_RUNS_ROOT
        return eval_e1e3_scientific(capability_id, out_base=sci_base)

    if capability_id in CAP_TO_STORE:
        return eval_e1e3_anchor(capability_id, out_base=out_base)
    if capability_id in {
        "CAP-E1E3-01",
        "CAP-E1E3-05",
        "CAP-E1E3-06",
        "CAP-E1E3-07",
        "CAP-E1E3-08",
    }:
        return eval_e1e3_stage(capability_id, out_base=out_base)
    from hazardweaver.hcg.carp.native_eval.blocked import eval_blocked_generic

    if capability_id.startswith("CAP-E1E3-"):
        return eval_blocked_generic(
            "E1-E3",
            capability_id,
            "official_repo_replay chain not wired in Phase 3 slice",
            out_base=out_base,
        )
    return eval_blocked_generic("E1-E3", capability_id, "no native eval wired", out_base=out_base)
