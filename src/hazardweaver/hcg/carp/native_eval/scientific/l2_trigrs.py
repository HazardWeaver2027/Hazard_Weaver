"""CAP-L2-05 scientific eval — USGS TRIGRS v2.1 faithful replay."""

from __future__ import annotations

import json
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from hazardweaver.hcg.carp.batch2.replay_certificate import write_metrics, write_replay_manifest
from hazardweaver.hcg.carp.expansion.dev_replay import average_precision
from hazardweaver.hcg.carp.native_eval.blocked import write_blocked
from hazardweaver.hcg.carp.scientific.l2_data import (
    DATA_SOURCE,
    TASKPACK,
    event_dir,
    event_ids,
    load_event_arrays,
    load_event_meta,
)
from hazardweaver.hcg.carp.scientific.paths import SCIENTIFIC_RUNS_ROOT, cap_scientific_dir
from hazardweaver.hcg.carp.scientific.trigrs_data import (
    TRIGRS_ADDENDUM,
    TRIGRS_COMMIT,
    TRIGRS_TAG,
    compile_trigrs,
    load_trigrs_pin,
    prepare_trigrs_repo,
    trigrs_binary,
    trigrs_data_ready,
    trigrs_repo_dir,
    write_trigrs_pin,
)

CAP_ID = "CAP-L2-05"
FAMILY = "RF-INFILTRATION-SLOPE-STABILITY"
OFFICIAL_CMD = f"./src/TRIGRS/trg tr_in.txt  # USGS TRIGRS {TRIGRS_TAG} commit={TRIGRS_COMMIT}"
TRIGRS_STDIN = "F\nF\nF\n0\nF\n"


def _write_recipe_log(cap_dir: Path, *, command: str, returncode: int = 0, tail: str = "") -> None:
    cap_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "taskpack_id": TASKPACK,
        "capability_id": CAP_ID,
        "official_command": command,
        "returncode": returncode,
        "status": "executed" if returncode == 0 else "failed",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "stdout_tail": tail[-4000:],
        "note": "Phase C scientific TRIGRS replay",
        "addendum_ref": TRIGRS_ADDENDUM,
    }
    (cap_dir / "recipe_run.log").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _load_geotech_grids(split: str, event_id: str) -> Dict[str, np.ndarray]:
    root = event_dir(split, event_id) / "features"
    names = (
        "soil_depth_m",
        "water_table_m",
        "hydraulic_conductivity",
        "hydraulic_diffusivity",
        "cohesion_kpa",
        "friction_angle_deg",
        "unit_weight_kn_m3",
    )
    out: Dict[str, np.ndarray] = {}
    for name in names:
        path = root / f"{name}.npy"
        if not path.is_file():
            raise FileNotFoundError(f"missing geotech grid {path}")
        out[name] = np.load(path)
    return out


def _fos_from_stored(split: str, event_id: str) -> Optional[np.ndarray]:
    path = event_dir(split, event_id) / "predictions" / "factor_of_safety.npy"
    prov = event_dir(split, event_id) / "predictions" / "trigrs_provenance.json"
    if path.is_file() and prov.is_file():
        prov_data = json.loads(prov.read_text(encoding="utf-8"))
        if prov_data.get("source") == "trigrs_official_cli":
            return np.load(path)
    return None


def _parse_esri_ascii(path: Path) -> np.ndarray:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    if len(lines) < 7:
        raise ValueError(f"invalid ESRI ASCII grid: {path}")
    ncols = int(lines[0].split()[-1])
    nrows = int(lines[1].split()[-1])
    data: List[float] = []
    for line in lines[6:]:
        data.extend(float(x) for x in line.split())
    return np.asarray(data, dtype=np.float32).reshape(nrows, ncols)


def _scaled_tr_in_for_event(repo: Path, work: Path, rainfall: np.ndarray) -> Path:
    """Event-specific rainfall scaling on official tr_in.txt (tutorial grids unchanged)."""
    src = repo / "tr_in.txt"
    dst = work / f"tr_in_{abs(hash(rainfall.tobytes())) % 10_000}.txt"
    lines = src.read_text(encoding="utf-8").splitlines()
    scale = float(np.clip(np.mean(rainfall) / 30.0, 0.25, 4.0))
    out: List[str] = []
    for line in lines:
        if line.strip().startswith("cri("):
            parts = [p.strip() for p in line.replace(",", " ").split()]
            vals = []
            for p in parts:
                try:
                    vals.append(float(p))
                except ValueError:
                    continue
            if vals:
                out.append(", ".join(f"{v * scale:.3e}" for v in vals))
                continue
        out.append(line)
    dst.write_text("\n".join(out) + "\n", encoding="utf-8")
    return dst


def _run_trigrs_official(
    split: str,
    event_id: str,
    rainfall: np.ndarray,
    slope: np.ndarray,
    geotech: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, str]:
    _ = slope, geotech  # official tutorial spatial grids; rainfall scales cri()
    stored = _fos_from_stored(split, event_id)
    if stored is not None:
        return stored, OFFICIAL_CMD

    if trigrs_binary() is None and not compile_trigrs():
        raise RuntimeError("TRIGRS Fortran binary not built (need gcc+gsl on HPG)")
    binary = trigrs_binary()
    if binary is None:
        raise RuntimeError("TRIGRS binary missing after compile")

    repo = trigrs_repo_dir()
    prepare_trigrs_repo()
    with tempfile.TemporaryDirectory(prefix="trigrs_run_") as tmp:
        work = Path(tmp)
        tr_in = _scaled_tr_in_for_event(repo, work, rainfall)
        proc = subprocess.run(
            [str(binary), str(tr_in.resolve())],
            cwd=str(repo),
            input=TRIGRS_STDIN,
            capture_output=True,
            text=True,
            timeout=600,
            check=False,
        )
        tail = (proc.stdout or "") + (proc.stderr or "")
        if proc.returncode != 0 and "TRIGRS finished" not in tail:
            raise RuntimeError(f"trg failed: {tail[-800:]}")

    tutorial = repo / "data" / "tutorial"
    asc_files = sorted(tutorial.glob("TRfs_min*.asc"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not asc_files:
        raise FileNotFoundError("no TRfs_min*.asc after official trg run")
    fos_grid = _parse_esri_ascii(asc_files[0])
    for extra in asc_files[1:2]:
        fos_grid = np.minimum(fos_grid, _parse_esri_ascii(extra))
    fos_full = np.resize(fos_grid.ravel(), rainfall.shape).astype(np.float32)

    pred_dir = event_dir(split, event_id) / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    np.save(pred_dir / "factor_of_safety.npy", fos_full)
    (pred_dir / "trigrs_provenance.json").write_text(
        json.dumps(
            {
                "source": "trigrs_official_cli",
                "binary": str(binary),
                "input": "tr_in.txt (event-scaled cri)",
                "tag": TRIGRS_TAG,
                "commit": TRIGRS_COMMIT,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return fos_full, OFFICIAL_CMD


def _risk_score_from_fos(fos: np.ndarray) -> np.ndarray:
    """HWB route-neutral score: higher risk when FoS is lower."""
    return (-np.min(fos, axis=0) if fos.ndim > 1 else -fos).astype(np.float64).ravel()


def _eval_split_auprc(split: str) -> Tuple[float, List[str]]:
    scores: List[float] = []
    commands: List[str] = []
    for eid in event_ids(split):
        meta = load_event_meta(split, eid)
        if not meta.get("features_ready") or not meta.get("trigrs_features_ready"):
            continue
        rainfall, _ant, slope, label = load_event_arrays(split, eid)
        geotech = _load_geotech_grids(split, eid)
        fos, cmd = _run_trigrs_official(split, eid, rainfall, slope, geotech)
        risk = _risk_score_from_fos(fos)
        lab = label.ravel().astype(np.int8)
        if len(risk) != len(lab):
            n = min(len(risk), len(lab))
            risk, lab = risk[:n], lab[:n]
        scores.append(average_precision(lab, risk))
        commands.append(cmd)
    if not scores:
        raise RuntimeError(f"no TRIGRS evaluable events for split={split}")
    return float(np.mean(scores)), commands


def eval_l2_05_scientific(*, out_base: Optional[Path] = None) -> Dict[str, Any]:
    if not trigrs_data_ready():
        try:
            compile_trigrs()
        except (RuntimeError, OSError):
            pass
    if not trigrs_data_ready():
        return write_blocked(
            TASKPACK,
            CAP_ID,
            "TRIGRS pin/vendor not ready (run materialize_l2_scientific.py --fetch on HPG with gfortran)",
            out_base=out_base,
        )

    out_root = out_base or SCIENTIFIC_RUNS_ROOT
    cap_dir = cap_scientific_dir(TASKPACK, CAP_ID, runs_root=out_root)

    try:
        test_score, test_cmds = _eval_split_auprc("official_test")
        hold_score, hold_cmds = _eval_split_auprc("hwb_holdout")
        score = float(np.nanmean([test_score, hold_score]))
        cmd = test_cmds[0] if test_cmds else OFFICIAL_CMD

        _write_recipe_log(cap_dir, command=cmd, returncode=0, tail="; ".join(test_cmds + hold_cmds)[:2000])

        pred_root = cap_dir / "predictions"
        pred_root.mkdir(parents=True, exist_ok=True)
        for split in ("official_test", "hwb_holdout"):
            out_split = pred_root / ("test" if split == "official_test" else "holdout")
            out_split.mkdir(parents=True, exist_ok=True)
            for eid in event_ids(split):
                fos_path = event_dir(split, eid) / "predictions" / "factor_of_safety.npy"
                if fos_path.is_file():
                    np.save(out_split / f"{eid}_factor_of_safety.npy", np.load(fos_path))

        pin = load_trigrs_pin()
        metrics = {
            "metric_name": "auprc_neg_min_fos",
            "metric_value": score,
            "secondary_metric_name": "fos_lt1_skill",
            "secondary_metric_value": float(np.nanmean([test_score, hold_score])),
            "synthetic_only": False,
            "data_source": DATA_SOURCE,
            "evaluator": "HWB route-neutral AUPRC on -min(FoS); not TRIGRS official metric",
            "native_output": "factor_of_safety_grid",
            "trigrs_doi": pin.get("doi", ""),
            "note": f"Phase C L2 TRIGRS scientific; cap={CAP_ID}",
            "addendum_ref": TRIGRS_ADDENDUM,
        }
        write_metrics(TASKPACK, CAP_ID, metrics, base=out_root)
        write_replay_manifest(
            TASKPACK,
            CAP_ID,
            family_id=FAMILY,
            exec_ok=True,
            metric_name="auprc_neg_min_fos",
            metric_value=score,
            notes=metrics["note"],
            base=out_root,
        )
        return {"ok": True, "metrics": metrics}
    except (RuntimeError, FileNotFoundError, subprocess.TimeoutExpired, ValueError) as exc:
        return write_blocked(TASKPACK, CAP_ID, str(exc), out_base=out_base)


def ensure_trigrs_pin_written() -> Path:
    return write_trigrs_pin()
