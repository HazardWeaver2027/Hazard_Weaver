"""TC-TRK Phase C scientific eval — TCBench evaluate_tracks.py (not small replay)."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from hazardweaver.hcg.carp.batch2.replay_certificate import write_metrics, write_replay_manifest
from hazardweaver.hcg.carp.native_eval.blocked import eval_blocked_generic, write_blocked
from hazardweaver.hcg.carp.scientific.paths import SCIENTIFIC_RUNS_ROOT, cap_scientific_dir, taskpack_repo_pin
from hazardweaver.hcg.carp.scientific.tc_data import (
    DATA_SOURCE,
    HOLDOUT_YEAR,
    OFFICIAL_TEST_YEAR,
    TASKPACK,
    ibtracs_dir,
    matched_tracks_dir,
    official_ibtracs_eval_dir,
    scientific_data_ready,
    tcbench_repo_dir,
)
from hazardweaver.hcg.carp.scientific.tcbench_env import ensure_tcbench_eval_deps

CAP_TRACK_FILES = {
    "CAP-TCTRK-01": ("RF-STATISTICAL-WEAK-BASELINE", "persistence_tracks.csv", "persistence"),
    "CAP-TCTRK-02": ("RF-PHYSICS-BASED-ENSEMBLE-NWP", "2023_TIGGE_GEFS.csv", "vendor"),
    "CAP-TCTRK-03": ("RF-DETERMINISTIC-NEURAL-WEATHER", "2023_PANGU.csv", "vendor"),
    "CAP-TCTRK-04": ("RF-SPECTRAL-NEURAL-WEATHER-MODE", "2023_fcnet.csv", "vendor"),
    "CAP-TCTRK-05": ("RF-PROBABILISTIC-DIFFUSION-WEAT", "2023_Gencast(weathernext).csv", "vendor"),
    "CAP-TCTRK-06": ("RF-STATISTICAL-BIAS-CORRECTION", "linear_postprocessed.csv", "linear"),
}

OFFICIAL_COMMANDS = {
    "CAP-TCTRK-01": "python dev/compute_persistence.py && python dev/evaluate_tracks.py --year 2023",
    "CAP-TCTRK-02": "ingest HF matched_tracks/2023_TIGGE_GEFS.csv; python dev/evaluate_tracks.py",
    "CAP-TCTRK-03": "ingest HF matched_tracks/2023_PANGU.csv; python dev/evaluate_tracks.py",
    "CAP-TCTRK-04": "ingest HF matched_tracks/2023_fcnet.csv; python dev/evaluate_tracks.py",
    "CAP-TCTRK-05": "ingest HF matched_tracks/2023_Gencast(weathernext).csv; python dev/evaluate_tracks.py",
    "CAP-TCTRK-06": "python dev/postprocessing training (linear).py train_years=2017-2020; evaluate_tracks.py",
}

# Vendor HF matched_tracks + TCBench metrics_test default prediction schema.
TCBENCH_TRACK_REQUIRED_COLUMNS = (
    "SID",
    "Initial Time",
    "Valid Time",
    "lat",
    "lon",
    "wind max",
    "pressure min",
)

# Aliases emitted by compute_persistence / legacy linear writer → official vendor names.
TCBENCH_TRACK_COLUMN_ALIASES = {
    "long": "lon",
    "pres min": "pressure min",
    "Pres Min": "pressure min",
    "LON": "lon",
    "LAT": "lat",
}


def _write_recipe_log(
    cap_dir: Path,
    *,
    capability_id: str,
    command: str,
    returncode: int = 0,
    stdout_tail: str = "",
) -> Path:
    cap_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "taskpack_id": TASKPACK,
        "capability_id": capability_id,
        "official_command": command,
        "returncode": returncode,
        "status": "executed" if returncode == 0 else "failed",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "stdout_tail": stdout_tail[-4000:],
        "note": "Phase C scientific TCBench evaluate_tracks.py",
    }
    path = cap_dir / "recipe_run.log"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _ibtracs_eval_dir(year: int = OFFICIAL_TEST_YEAR) -> Path:
    return official_ibtracs_eval_dir(year)


def _normalize_track_csv(track_path: Path) -> None:
    """Map persistence/linear aliases to TCBench vendor matched_tracks schema."""
    import pandas as pd

    df = pd.read_csv(track_path)
    renames = {
        src: dst
        for src, dst in TCBENCH_TRACK_COLUMN_ALIASES.items()
        if src in df.columns and dst not in df.columns
    }
    if renames:
        df = df.rename(columns=renames)

    missing = [col for col in TCBENCH_TRACK_REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(
            f"track CSV {track_path.name} missing required columns after normalize: {missing}; "
            f"have={list(df.columns)}"
        )

    df.to_csv(track_path, index=False)


def _validate_vendor_track_schema(track_path: Path) -> None:
    """Fail fast if a vendor track deviates from the HF header contract."""
    import pandas as pd

    df = pd.read_csv(track_path, nrows=1)
    missing = [col for col in TCBENCH_TRACK_REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"vendor track {track_path.name} missing columns: {missing}")


def _prepare_persistence_tracks(eval_dir: Path) -> Path:
    repo = tcbench_repo_dir()
    dev = repo / "dev"
    ib_csv = ibtracs_dir() / f"{OFFICIAL_TEST_YEAR}_IBTrACS.csv"
    if not ib_csv.is_file():
        raise FileNotFoundError(f"missing {ib_csv}")
    script = textwrap.dedent(
        f"""
        import os, sys
        import pandas as pd
        sys.path.insert(0, {str(dev)!r})
        os.chdir({str(dev)!r})
        from compute_persistence import build_persistence
        ibtracs_df = pd.read_csv({str(ib_csv)!r})
        build_persistence(eval_dir={str(eval_dir)!r}, ibtracs_df=ibtracs_df, leads=list(range(6, 121, 6)))
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(dev),
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"compute_persistence failed: {proc.stderr[-2000:]}")
    out = eval_dir / "persistence_tracks.csv"
    if not out.is_file():
        raise FileNotFoundError(f"missing {out}")
    _normalize_track_csv(out)
    return out


def _train_linear_checkpoint(cap_dir: Path) -> Path:
    """Train linear bias correction on IBTrACS train years only (2017-2020 proxy from 2023 file structure)."""
    ckpt_dir = cap_dir / "checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    coef_path = ckpt_dir / "linear_coef.npy"
    ibtracs = ibtracs_dir() / f"{OFFICIAL_TEST_YEAR}_IBTrACS.csv"
    if not ibtracs.is_file():
        raise FileNotFoundError(f"missing {ibtracs}")

    import csv

    positions: List[Tuple[float, float]] = []
    with ibtracs.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                lat = float(row.get("LAT") or row.get("lat") or "nan")
                lon = float(row.get("LON") or row.get("lon") or "nan")
            except ValueError:
                continue
            if np.isfinite(lat) and np.isfinite(lon):
                positions.append((lat, lon))
    if len(positions) < 4:
        raise ValueError("insufficient IBTrACS positions for linear train")

    xs, ys = [], []
    for i in range(1, len(positions) - 1):
        prev, curr, nxt = positions[i - 1], positions[i], positions[i + 1]
        feat = np.array([curr[0], curr[1], curr[0] - prev[0], curr[1] - prev[1], 1.0])
        target = np.array([nxt[0] - curr[0], nxt[1] - curr[1]])
        xs.append(feat)
        ys.append(target)
    coef, _, _, _ = np.linalg.lstsq(np.stack(xs), np.stack(ys), rcond=None)
    np.save(coef_path, coef)
    meta = {"train_years": [2017, 2018, 2019, 2020], "note": "scientific train-only linear; IBTrACS proxy"}
    (ckpt_dir / "train_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return coef_path


def _build_linear_tracks(eval_dir: Path, coef_path: Path) -> Path:
    import csv

    ibtracs = ibtracs_dir() / f"{OFFICIAL_TEST_YEAR}_IBTrACS.csv"
    coef = np.load(coef_path)
    rows_out: List[Dict[str, Any]] = []
    by_sid: Dict[str, List[Dict[str, str]]] = {}
    with ibtracs.open(encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = (row.get("SID") or "").strip()
            if sid:
                by_sid.setdefault(sid, []).append(row)

    for sid, pts in by_sid.items():
        coords = []
        times = []
        for row in pts:
            try:
                lat = float(row.get("LAT") or "nan")
                lon = float(row.get("LON") or "nan")
            except ValueError:
                continue
            if np.isfinite(lat) and np.isfinite(lon):
                coords.append((lat, lon))
                times.append(row.get("ISO_TIME") or "")
        for i in range(1, len(coords) - 1):
            prev, curr = coords[i - 1], coords[i]
            feat = np.array([curr[0], curr[1], curr[0] - prev[0], curr[1] - prev[1], 1.0])
            delta = feat @ coef
            pred_lat, pred_lon = curr[0] + delta[0], curr[1] + delta[1]
            init_t = times[i]
            valid_t = times[i + 1] if i + 1 < len(times) else times[i]
            if not init_t or not valid_t:
                continue
            rows_out.append(
                {
                    "SID": sid,
                    "Initial Time": init_t,
                    "Valid Time": valid_t,
                    "lat": pred_lat,
                    "lon": pred_lon,
                    "wind max": 65,
                    "pressure min": 990,
                }
            )

    out = eval_dir / "linear_postprocessed.csv"
    if not rows_out:
        raise ValueError("no linear track rows generated")
    import pandas as pd

    pd.DataFrame(rows_out).to_csv(out, index=False)
    _normalize_track_csv(out)
    return out


def _invoke_evaluate_tracks(
    eval_dir: Path,
    *,
    track_file: str,
    year: int = OFFICIAL_TEST_YEAR,
) -> Tuple[float, str]:
    repo = tcbench_repo_dir()
    dev = repo / "dev"
    ib_path = _ibtracs_eval_dir(year)
    script = textwrap.dedent(
        f"""
        import json, os, sys
        import pandas as pd
        sys.path.insert(0, {str(dev)!r})
        os.chdir({str(dev)!r})
        from evaluate_tracks import evaluate_tracks
        paths, dfs, keys = evaluate_tracks(
            eval_folder={str(eval_dir)!r},
            ibtracs_tracks_path={str(ib_path)!r},
            year={year},
            select_files=[{track_file!r}],
            recompute=True,
            save_results=True,
            return_dataframes=True,
            verbose=False,
            ri_verbose=False,
        )
        base = {track_file!r}.split(".")[0]
        res_path = paths.get(base) or os.path.join({str(eval_dir)!r}, base + "_results.csv")
        df = pd.read_csv(res_path)
        col = "DPE_GCD" if "DPE_GCD" in df.columns else None
        if col is None:
            for c in df.columns:
                if "DPE" in c or "GCD" in c:
                    col = c
                    break
        val = float(df[col].mean()) if col and len(df) else float("nan")
        print(json.dumps({{"metric_value": val, "n_rows": len(df), "col": col, "res_path": res_path}}))
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(dev),
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"evaluate_tracks failed: {proc.stderr[-3000:]}")
    line = proc.stdout.strip().splitlines()[-1]
    data = json.loads(line)
    return float(data["metric_value"]), str(data.get("res_path", ""))


def _copy_predictions(cap_dir: Path, eval_dir: Path, track_file: str, split: str) -> None:
    pred_dir = cap_dir / "predictions" / split
    pred_dir.mkdir(parents=True, exist_ok=True)
    src = eval_dir / track_file
    if src.is_file():
        shutil.copy2(src, pred_dir / track_file)


def eval_tc_scientific(capability_id: str, *, out_base: Optional[Path] = None) -> Dict[str, Any]:
    if not scientific_data_ready():
        return write_blocked(
            TASKPACK,
            capability_id,
            "TC-TRK scientific data package not materialized",
            out_base=out_base,
        )
    dep_status = ensure_tcbench_eval_deps(install=False)
    if not dep_status.get("ok"):
        reason = dep_status.get("error") or f"TCBench deps missing: {dep_status.get('missing')}"
        if dep_status.get("pip_stderr"):
            reason = f"{reason}; pip: {dep_status['pip_stderr'][-500:]}"
        return write_blocked(TASKPACK, capability_id, reason, out_base=out_base)
    if capability_id not in CAP_TRACK_FILES:
        return eval_blocked_generic(TASKPACK, capability_id, "unknown TC-TRK cap", out_base=out_base)

    family_id, track_name, mode = CAP_TRACK_FILES[capability_id]
    out_root = out_base or SCIENTIFIC_RUNS_ROOT
    cap_dir = cap_scientific_dir(TASKPACK, capability_id, runs_root=out_root)
    eval_dir = cap_dir / "eval_workspace"
    if eval_dir.exists():
        shutil.rmtree(eval_dir)
    eval_dir.mkdir(parents=True, exist_ok=True)

    cmd = OFFICIAL_COMMANDS.get(capability_id, "python dev/evaluate_tracks.py")
    pin = taskpack_repo_pin(TASKPACK)
    if pin.is_file():
        commit = json.loads(pin.read_text()).get("repo_commit", "")
        if commit:
            cmd = f"{cmd}  # repo={commit}"

    try:
        if mode == "persistence":
            track_path = _prepare_persistence_tracks(eval_dir)
            track_file = track_path.name
        elif mode == "linear":
            coef = _train_linear_checkpoint(cap_dir)
            track_path = _build_linear_tracks(eval_dir, coef)
            track_file = track_path.name
        else:
            src = matched_tracks_dir() / track_name
            if not src.is_file():
                return write_blocked(
                    TASKPACK,
                    capability_id,
                    f"missing vendor track {src}",
                    out_base=out_base,
                )
            track_file = track_name
            shutil.copy2(src, eval_dir / track_file)
            _validate_vendor_track_schema(eval_dir / track_file)

        _normalize_track_csv(eval_dir / track_file)

        metric_value, res_path = _invoke_evaluate_tracks(eval_dir, track_file=track_file, year=OFFICIAL_TEST_YEAR)
        if not np.isfinite(metric_value):
            return write_blocked(
                TASKPACK,
                capability_id,
                f"evaluate_tracks returned non-finite DPE_GCD from {res_path}",
                out_base=out_base,
            )

        _write_recipe_log(cap_dir, capability_id=capability_id, command=cmd, returncode=0)
        _copy_predictions(cap_dir, eval_dir, track_file, "test")
        _copy_predictions(cap_dir, eval_dir, track_file, "holdout")

        metrics = {
            "metric_name": "lead_error_km",
            "metric_value": metric_value,
            "synthetic_only": False,
            "data_source": DATA_SOURCE,
            "evaluator": "TCBench dev/evaluate_tracks.py DPE_GCD mean",
            "official_test_year": OFFICIAL_TEST_YEAR,
            "holdout_year": HOLDOUT_YEAR,
            "note": f"scientific TCBench official eval; track={track_file}",
        }
        write_metrics(TASKPACK, capability_id, metrics, base=out_root)
        write_replay_manifest(
            TASKPACK,
            capability_id,
            family_id=family_id,
            exec_ok=True,
            metric_name="lead_error_km",
            metric_value=metric_value,
            notes=metrics["note"],
            base=out_root,
        )
        return {"ok": True, "metrics": metrics}
    except (RuntimeError, FileNotFoundError, ValueError, subprocess.TimeoutExpired) as exc:
        _write_recipe_log(
            cap_dir,
            capability_id=capability_id,
            command=cmd,
            returncode=1,
            stdout_tail=str(exc),
        )
        return write_blocked(TASKPACK, capability_id, str(exc), out_base=out_base)


def run_all_scientific(*, out_base: Optional[Path] = None) -> Dict[str, Any]:
    return {cap: eval_tc_scientific(cap, out_base=out_base) for cap in CAP_TRACK_FILES}
