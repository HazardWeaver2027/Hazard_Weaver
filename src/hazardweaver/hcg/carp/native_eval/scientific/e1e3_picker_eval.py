"""E1-E3 Phase C picker eval orchestrator (pyhazards → GPL worker)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

from hazardweaver.hcg.carp.acquire.env_routing import SEISBENCH_PYTHON
from hazardweaver.hcg.carp.acquire.seisbench_paths import seisbench_subprocess_env
from hazardweaver.hcg.carp.batch2.eval_e1e3 import CAP_TO_STORE
from hazardweaver.hcg.carp.native_eval.scientific.e1e3_picker_checkpoint import (
    aggregate_f1,
    checkpoint_dir_for,
    collect_predictions,
    progress_path_for,
)
from hazardweaver.hcg.carp.scientific.e1e3_data import (
    TASKPACK,
    load_split_traces,
    seisbench_python,
)
from hazardweaver.hcg.carp.scientific.paths import SCIENTIFIC_RUNS_ROOT, cap_scientific_dir

WORKER = Path(__file__).resolve().parent / "e1e3_picker_eval_worker.py"
DATA_ROOT = Path(__file__).resolve().parents[5] / "data" / "scientific" / TASKPACK
SPLIT_TIMEOUT_S = 6 * 3600


def _run_worker(args: List[str], *, use_seisbench: bool, timeout_s: int = 600) -> Dict[str, Any]:
    py = seisbench_python() if use_seisbench else SEISBENCH_PYTHON
    if not py.is_file():
        return {"ok": False, "reason": f"missing interpreter {py}"}
    env = seisbench_subprocess_env() if use_seisbench else None
    proc = subprocess.run(
        [str(py), str(WORKER), *args],
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
        env=env,
    )
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip().splitlines()
        return {"ok": False, "reason": err[-1] if err else "picker worker failed"}
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except json.JSONDecodeError:
        return {"ok": False, "reason": f"bad worker output: {proc.stdout[:200]}"}


def _run_split_batch(
    *,
    split: str,
    mode: str,
    model_store: str = "",
    weight_variant: str = "original",
    checkpoint_dir: Optional[Path] = None,
    progress_path: Optional[Path] = None,
) -> Dict[str, Any]:
    args = [
        "--mode",
        mode,
        "--split",
        split,
        "--data-root",
        str(DATA_ROOT),
    ]
    if mode == "seisbench":
        args.extend(["--model-store", model_store, "--weight-variant", weight_variant])
    if checkpoint_dir is not None:
        args.extend(["--checkpoint-dir", str(checkpoint_dir)])
    if progress_path is not None:
        args.extend(["--progress-path", str(progress_path)])
    return _run_worker(args, use_seisbench=True, timeout_s=SPLIT_TIMEOUT_S)


def eval_sta_lta_split(split: str, *, out_base: Optional[Path] = None) -> Dict[str, Any]:
    traces = load_split_traces(split)
    if not traces:
        return {"ok": False, "reason": f"no traces for split {split}"}
    cap_dir = cap_scientific_dir(TASKPACK, "CAP-E1E3-01", runs_root=out_base or SCIENTIFIC_RUNS_ROOT)
    ckpt_dir = checkpoint_dir_for(cap_dir, split)
    progress = progress_path_for(cap_dir, split)
    result = _run_split_batch(
        split=split,
        mode="sta_lta",
        checkpoint_dir=ckpt_dir,
        progress_path=progress,
    )
    if not result.get("ok"):
        return result
    predictions = collect_predictions(ckpt_dir)
    if len(predictions) < len(traces):
        return {
            "ok": False,
            "reason": f"STA/LTA incomplete: {len(predictions)}/{len(traces)} checkpoints",
        }
    return {
        "ok": True,
        "pick_f1": aggregate_f1(predictions),
        "n_traces": len(predictions),
        "split": split,
        "predictions": predictions,
    }


def eval_picker_cap(
    capability_id: str,
    *,
    split: str = "official_test",
    out_base: Optional[Path] = None,
) -> Dict[str, Any]:
    if capability_id not in CAP_TO_STORE:
        return {"ok": False, "reason": f"unsupported picker cap {capability_id}"}
    store, weight_variant, model_class, _family = CAP_TO_STORE[capability_id]
    traces = load_split_traces(split)
    if not traces:
        return {"ok": False, "reason": f"no traces for split {split}"}
    cap_dir = cap_scientific_dir(TASKPACK, capability_id, runs_root=out_base or SCIENTIFIC_RUNS_ROOT)
    ckpt_dir = checkpoint_dir_for(cap_dir, split)
    progress = progress_path_for(cap_dir, split)
    result = _run_split_batch(
        split=split,
        mode="seisbench",
        model_store=store,
        weight_variant=weight_variant,
        checkpoint_dir=ckpt_dir,
        progress_path=progress,
    )
    if not result.get("ok"):
        return result
    predictions = collect_predictions(ckpt_dir)
    if len(predictions) < len(traces):
        return {
            "ok": False,
            "reason": f"{capability_id} {split} incomplete: {len(predictions)}/{len(traces)} checkpoints",
        }
    pick_f1 = aggregate_f1(predictions)
    return {
        "ok": True,
        "pick_f1": pick_f1,
        "n_traces": len(predictions),
        "split": split,
        "model_store": store,
        "weight_variant": weight_variant,
        "model_class": model_class,
        "predictions": predictions,
    }


def eval_picker_cap_both_splits(
    capability_id: str,
    *,
    out_base: Optional[Path] = None,
) -> Dict[str, Any]:
    test = eval_picker_cap(capability_id, split="official_test", out_base=out_base)
    if not test.get("ok"):
        return test
    hold = eval_picker_cap(capability_id, split="hwb_holdout", out_base=out_base)
    hold_f1 = float(hold.get("pick_f1", 0.0)) if hold.get("ok") else 0.0
    return {
        **test,
        "holdout_pick_f1": hold_f1,
        "holdout_predictions": hold.get("predictions", []) if hold.get("ok") else [],
    }
