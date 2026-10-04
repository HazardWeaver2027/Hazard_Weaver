"""WF-3 WildfireSpreadTS official eval (minimal subset + REPLAY_PIN)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

from hazardweaver.hcg.carp.batch2.eval_wf3 import CAP_TO_FAMILY, _find_checkpoint_pin, eval_persistence
from hazardweaver.hcg.carp.batch2.replay_certificate import write_metrics, write_replay_manifest
from hazardweaver.hcg.carp.native_eval.blocked import write_blocked
from hazardweaver.hcg.carp.native_eval.eval_wf3_process import eval_wf3_process

PROJECT_ROOT = Path(__file__).resolve().parents[4]
WSTS_REPO = PROJECT_ROOT / "data" / "vendor" / "wildfirespreadts"
REPLAY_PIN = WSTS_REPO / "REPLAY_PIN.json"
EVAL_SUBSET = WSTS_REPO / "eval_subset" / "manifest.json"
WORKER = Path(__file__).resolve().parent / "eval_wf3_wsts_worker.py"


def _run_worker(mode: str, capability_id: str = "") -> Dict[str, Any]:
    py = PROJECT_ROOT / "envs" / "pyhazards" / "bin" / "python"
    if not py.is_file():
        return {"ok": False, "reason": f"missing {py}"}
    cmd = [str(py), str(WORKER), "--mode", mode]
    if capability_id:
        cmd.extend(["--capability-id", capability_id])
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600, check=False)
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "").strip().splitlines()
        return {"ok": False, "reason": err[-1] if err else "wsts worker failed"}
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except json.JSONDecodeError:
        return {"ok": False, "reason": f"bad worker output: {proc.stdout[:200]}"}


def eval_wf3_01_anchor(*, out_base: Optional[Path] = None) -> Dict[str, Any]:
    """CAP-WF3-01: prefer WSTS anchor AP; fallback NIFC persistence IoU."""
    capability_id = "CAP-WF3-01"
    if WSTS_REPO.is_dir() and REPLAY_PIN.is_file() and EVAL_SUBSET.is_file():
        result = _run_worker("persistence_ap", capability_id)
        if result.get("ok"):
            ap = float(result.get("average_precision", result.get("metric_value", 0.0)))
            metrics = {
                "metric_name": "average_precision",
                "metric_value": ap,
                "data_source": "wsts_eval_subset",
                "manifest_path": str(EVAL_SUBSET),
                "synthetic_only": False,
                "replay_pin": str(REPLAY_PIN),
                "note": "Official WSTS eval subset replay",
            }
            write_metrics("WF-3", capability_id, metrics, base=out_base)
            write_replay_manifest(
                "WF-3",
                capability_id,
                family_id="RF-PERSISTENCE",
                exec_ok=True,
                metric_name="average_precision",
                metric_value=ap,
                notes="WSTS anchor AP on eval subset",
                base=out_base,
            )
            return {"ok": True, "metrics": metrics}
    return eval_persistence(capability_id, out_base=out_base)


def eval_wf3_g2(capability_id: str, *, out_base: Optional[Path] = None) -> Dict[str, Any]:
    family_id = CAP_TO_FAMILY.get(capability_id, "unknown")
    if not WSTS_REPO.is_dir() or not REPLAY_PIN.is_file():
        return write_blocked(
            "WF-3",
            capability_id,
            "WildfireSpreadTS vendor missing; run scripts/bootstrap/pin_wildfirespreadts.sh",
            out_base=out_base,
        )
    ckpt_pin = _find_checkpoint_pin(capability_id)
    if ckpt_pin is None:
        return write_blocked(
            "WF-3",
            capability_id,
            "no CHECKPOINT_PIN.json; run run_hcg_batch2_wf3_train.sbatch",
            out_base=out_base,
        )
    pin = json.loads(ckpt_pin.read_text(encoding="utf-8"))
    if pin.get("status") != "checkpoint_ok":
        return write_blocked(
            "WF-3",
            capability_id,
            f"checkpoint pin status={pin.get('status')}",
            out_base=out_base,
        )
    if not EVAL_SUBSET.is_file():
        return write_blocked(
            "WF-3",
            capability_id,
            "missing eval_subset manifest; run materialize_wsts_eval_subset.py on HPG",
            out_base=out_base,
        )
    result = _run_worker("g2_ap", capability_id)
    if not result.get("ok"):
        return write_blocked("WF-3", capability_id, result.get("reason", "g2 eval failed"), out_base=out_base)
    ap = float(result.get("average_precision", result.get("metric_value", 0.0)))
    metrics = {
        "metric_name": "average_precision",
        "metric_value": ap,
        "capability_id": capability_id,
        "checkpoint_pin": str(ckpt_pin),
        "data_source": "wsts_eval_subset",
        "synthetic_only": False,
    }
    write_metrics("WF-3", capability_id, metrics, base=out_base)
    write_replay_manifest(
        "WF-3",
        capability_id,
        family_id=family_id,
        exec_ok=True,
        metric_name="average_precision",
        metric_value=ap,
        notes="WSTS g2 checkpoint official eval subset",
        extra={"checkpoint_pin": str(ckpt_pin)},
        base=out_base,
    )
    return {"ok": True, "metrics": metrics}


def _eval_wf3_from_cell_out_base(out_base: Path, capability_id: str) -> Optional[Dict[str, Any]]:
    """Read pinned native_metrics from a benchmark cell's inputs/hcg_scientific tree."""
    base = Path(out_base).resolve()
    metrics_path = base / "native_metrics.json"
    if not metrics_path.is_file():
        return None
    try:
        body = json.loads(metrics_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if body.get("ok") is False:
        return write_blocked(
            "WF-3",
            capability_id,
            str(body.get("note") or body.get("blocker") or "cell native_metrics not ok"),
            out_base=out_base,
        )
    metric_name = str(body.get("metric_name") or "average_precision")
    metric_value = body.get("metric_value")
    if metric_value is None:
        metric_value = body.get("average_precision") or body.get(metric_name)
    if metric_value is None:
        return None
    metrics = {
        "metric_name": metric_name,
        "metric_value": float(metric_value),
        "synthetic_only": bool(body.get("synthetic_only")),
        "data_source": str(body.get("data_source") or "cell_hcg_scientific_pinned"),
        "source_path": str(metrics_path),
        "capability_id": capability_id,
    }
    return {
        "ok": True,
        "metrics": metrics,
        "artifact_replay": True,
        "scientific_replay": True,
    }


def _wf3_agent_replay_mode(out_base: Optional[Path] = None) -> bool:
    import os

    if str(os.environ.get("HWA_WF3_AGENT_REPLAY", "")).strip().lower() in {"1", "true", "yes", "on"}:
        return True
    return False


def _wf3_pinned_replay_base(capability_id: str, out_base: Optional[Path]) -> Optional[Path]:
    if out_base is not None:
        base = Path(out_base)
        if (base / "native_metrics.json").is_file():
            return base
    pinned = Path("runs/carp/scientific/WF-3") / str(capability_id)
    if (pinned / "native_metrics.json").is_file():
        return pinned
    holdout = pinned / "predictions" / "holdout"
    if holdout.is_dir() and any(holdout.glob("*.npz")):
        return pinned
    return None


def eval_wf3_cap(capability_id: str, *, out_base: Optional[Path] = None) -> Dict[str, Any]:
    from hazardweaver.hcg.carp.scientific.wf3_data import scientific_data_ready

    replay_base = _wf3_pinned_replay_base(capability_id, out_base)
    if replay_base is not None:
        cell_eval = _eval_wf3_from_cell_out_base(replay_base, capability_id)
        if cell_eval is not None:
            return cell_eval

    if out_base is not None and not _wf3_agent_replay_mode(out_base):
        cell_eval = _eval_wf3_from_cell_out_base(Path(out_base), capability_id)
        if cell_eval is not None:
            return cell_eval

    if _wf3_agent_replay_mode(out_base):
        if capability_id == "CAP-WF3-01":
            return eval_wf3_01_anchor(out_base=replay_base or out_base)
        if capability_id in CAP_TO_FAMILY:
            return eval_wf3_g2(capability_id, out_base=replay_base or out_base)
        if capability_id == "CAP-WF3-06":
            return eval_wf3_process(capability_id, out_base=replay_base or out_base)
        return write_blocked(
            "WF-3",
            capability_id,
            "agent_replay_mode: no pinned native_metrics or WSTS replay",
            out_base=out_base,
        )

    if scientific_data_ready():
        from hazardweaver.hcg.carp.native_eval.scientific.wf3_wsts import eval_wf3_scientific
        from hazardweaver.hcg.carp.scientific.paths import SCIENTIFIC_RUNS_ROOT

        sci_base = out_base or SCIENTIFIC_RUNS_ROOT
        return eval_wf3_scientific(capability_id, out_base=sci_base)

    if capability_id == "CAP-WF3-01":
        return eval_wf3_01_anchor(out_base=out_base)
    if capability_id in CAP_TO_FAMILY:
        return eval_wf3_g2(capability_id, out_base=out_base)
    if capability_id == "CAP-WF3-06":
        return eval_wf3_process(capability_id, out_base=out_base)
    return write_blocked("WF-3", capability_id, "no native eval wired", out_base=out_base)
