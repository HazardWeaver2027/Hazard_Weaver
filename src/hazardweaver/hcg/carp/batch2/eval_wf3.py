"""WF-3 persistence baseline native eval (NIFC progression or smoke fixture)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from hazardweaver.hcg.carp.batch2.fixtures import ensure_wf3_smoke_fixture
from hazardweaver.hcg.carp.batch2.replay_certificate import write_metrics, write_replay_manifest
from hazardweaver.hwb.evaluators.wf3_spread_iou import score_spread_pair

PROJECT_ROOT = Path(__file__).resolve().parents[4]
NIFC_MANIFEST = PROJECT_ROOT / "data" / "processed" / "wf3_nifc_progression_v1" / "manifest.json"
WSTS_REPO = PROJECT_ROOT / "data" / "vendor" / "wildfirespreadts"


def _resolve_manifest() -> Tuple[Path, Dict[str, Any], str]:
    if NIFC_MANIFEST.is_file():
        data = json.loads(NIFC_MANIFEST.read_text(encoding="utf-8"))
        return NIFC_MANIFEST, data, "nifc_progression"
    smoke = ensure_wf3_smoke_fixture()
    data = json.loads(smoke.read_text(encoding="utf-8"))
    return smoke, data, "synthetic_smoke"


def _load_grid(path: Path, base: Path) -> np.ndarray:
    p = path if path.is_absolute() else base / path
    return np.load(p)


def _iter_fire_pairs(manifest: Dict[str, Any], base_dir: Path) -> List[tuple[np.ndarray, np.ndarray]]:
    pairs: List[tuple[np.ndarray, np.ndarray]] = []
    for fire in manifest.get("fires") or []:
        if fire.get("pairs"):
            for pair in fire["pairs"]:
                t0 = _load_grid(Path(pair["t0"]), base_dir)
                t1 = _load_grid(Path(pair["t1"]), base_dir)
                pairs.append((t0, t1))
            continue
        prior_rel = fire.get("prior_grid")
        prog_rel = fire.get("progression_grid")
        if prior_rel and prog_rel:
            prior = _load_grid(Path(prior_rel), base_dir)
            prog = _load_grid(Path(prog_rel), base_dir)
            t1 = np.clip(prior + prog, 0.0, 1.0)
            pairs.append((prior, t1))
    return pairs


def eval_persistence(
    capability_id: str = "CAP-WF3-01",
    *,
    out_base: Optional[Path] = None,
) -> Dict[str, Any]:
    manifest_path, manifest, source = _resolve_manifest()
    base_dir = manifest_path.parent
    ious: List[float] = []
    for t0, t1 in _iter_fire_pairs(manifest, base_dir):
        pred = t0  # persistence: next frame = previous mask
        scored = score_spread_pair(pred, t1)
        ious.append(float(scored["score"]))
    mean_iou = float(np.mean(ious)) if ious else 0.0
    exec_ok = len(ious) > 0
    metrics = {
        "metric_name": "spread_iou",
        "metric_value": mean_iou,
        "n_pairs": len(ious),
        "data_source": source,
        "manifest_path": str(manifest_path),
        "synthetic_only": bool(manifest.get("synthetic_only")),
        "wsts_repo_present": WSTS_REPO.is_dir(),
        "note": "Persistence baseline; full WSTS AP requires official repo evaluator (Batch 2b)",
    }
    write_metrics("WF-3", capability_id, metrics, base=out_base)
    write_replay_manifest(
        "WF-3",
        capability_id,
        family_id="RF-PERSISTENCE",
        exec_ok=exec_ok,
        metric_name="spread_iou",
        metric_value=mean_iou,
        notes=f"persistence eval source={source}",
        extra={"data_source": source},
        base=out_base,
    )
    return {"ok": exec_ok, "metrics": metrics}


CAP_TO_FAMILY = {
    "CAP-WF3-02": "RF-LINEAR-STATISTICAL",
    "CAP-WF3-03": "RF-SPATIAL-CONVOLUTIONAL-SEGMEN",
    "CAP-WF3-04": "RF-RECURRENT-TEMPORAL-SEGMENTAT",
    "CAP-WF3-05": "RF-TEMPORAL-ATTENTION-SEGMENTAT",
}


def _find_checkpoint_pin(capability_id: str) -> Optional[Path]:
    for root_name in ("batch1", "batch2"):
        root = PROJECT_ROOT / "runs" / "carp" / root_name
        if not root.is_dir():
            continue
        for job_dir in sorted(root.glob("_hpg_train_*"), reverse=True):
            pin = job_dir / "WF-3" / capability_id / "CHECKPOINT_PIN.json"
            if pin.is_file():
                return pin
    vendor_ckpt = PROJECT_ROOT / "data" / "vendor" / "wildfirespreadts" / "checkpoints" / capability_id
    pin = vendor_ckpt / "CHECKPOINT_PIN.json"
    return pin if pin.is_file() else None


def eval_g2_train_gate(
    capability_id: str,
    *,
    out_base: Optional[Path] = None,
) -> Dict[str, Any]:
    """Delegate to Phase 3 native WSTS g2 eval when pinned; else honest BLOCKED."""
    from hazardweaver.hcg.carp.native_eval.eval_wf3_wsts import eval_wf3_g2

    return eval_wf3_g2(capability_id, out_base=out_base)
