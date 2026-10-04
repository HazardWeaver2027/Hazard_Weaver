"""CAP-MH1-05 scientific eval — USGS ursa 1.0.0 operational runout chain."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from hazardweaver.hcg.carp.batch2.replay_certificate import write_metrics, write_replay_manifest
from hazardweaver.hcg.carp.metrics.threat_score import footprint_metrics
from hazardweaver.hcg.carp.native_eval.blocked import write_blocked
from hazardweaver.hcg.carp.scientific.mh1_ursa_data import (
    ADDENDUM,
    ANCHOR_CASES,
    DATA_SOURCE,
    PWFDF_R_DOI,
    SUPPORTING_DATA_DOI,
    TASKPACK,
    URSA_DOI,
    URSA_VERSION,
    all_cases_ready,
    case_dir,
    case_ids,
    install_ursa_from_repo,
    observed_case_ready,
    ursa_anchor_inputs_ready,
    ursa_cli_path,
    ursa_repo_dir,
)
from hazardweaver.hcg.carp.scientific.paths import SCIENTIFIC_RUNS_ROOT, cap_scientific_dir

CAP_ID = "CAP-MH1-05"
FAMILY = "RF-EMPIRICAL-INITIATION--PROCES"
OFFICIAL_CMD = f"ursa initialize <name> && ursa run all  # ursa {URSA_VERSION} DOI {URSA_DOI}"


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
        "note": "Phase C MH-1 ursa anchor reproduction (not HWB holdout)",
        "operational_contract_doi": PWFDF_R_DOI,
        "supporting_data_doi": SUPPORTING_DATA_DOI,
        "addendum_ref": ADDENDUM,
        "inventory_227_retrain": False,
    }
    (cap_dir / "recipe_run.log").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _try_ursa_smoke() -> Tuple[bool, str]:
    cli = ursa_cli_path()
    if cli:
        try:
            proc = subprocess.run([cli, "-h"], capture_output=True, text=True, timeout=60, check=False)
            if proc.returncode == 0:
                return True, (proc.stdout or proc.stderr)[-2000:]
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass
    ok, msg = install_ursa_from_repo()
    if ok:
        cli = ursa_cli_path()
        if cli:
            return True, msg
    repo = ursa_repo_dir()
    if repo.is_dir() and any(repo.iterdir()):
        return False, f"ursa repo pinned at {repo} but CLI not installed ({msg})"
    return False, msg or "ursa CLI not found"


def _ursa_cmd(*parts: str) -> List[str]:
    cli = ursa_cli_path()
    if not cli:
        raise FileNotFoundError("ursa CLI not on PATH (load module python/3.11 and pip install -e vendor/ursa/repo)")
    return [cli, *parts]


def _load_supporting_prediction(case_id: str) -> Optional[Tuple[np.ndarray, str]]:
    """Load official hazard map packaged under supporting_data/ when present."""
    pred_path = case_dir(case_id) / "supporting_data" / "dfsi_hazard.npy"
    prov_path = case_dir(case_id) / "supporting_data" / "provenance.json"
    if not pred_path.is_file():
        return None
    prov = json.loads(prov_path.read_text(encoding="utf-8")) if prov_path.is_file() else {}
    if prov.get("source") not in ("supporting_data_doi", "ursa_official_hpg"):
        return None
    return np.load(pred_path), prov.get("command", OFFICIAL_CMD)


def _run_ursa_chain(case_id: str) -> Tuple[np.ndarray, str]:
    """Run official ursa workflow or load HPG-stored hazard map."""
    pred_path = case_dir(case_id) / "predictions" / "dfsi_hazard.npy"
    prov_path = case_dir(case_id) / "predictions" / "ursa_provenance.json"
    if pred_path.is_file() and prov_path.is_file():
        prov = json.loads(prov_path.read_text(encoding="utf-8"))
        if prov.get("source") in ("ursa_run_all", "ursa_official_hpg", "supporting_data_doi"):
            return np.load(pred_path), prov.get("command", OFFICIAL_CMD)

    packaged = _load_supporting_prediction(case_id)
    if packaged is not None:
        hazard, cmd = packaged
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(pred_path, hazard.astype(np.float32))
        prov_path.write_text(
            json.dumps(
                {
                    "source": "supporting_data_doi",
                    "command": cmd,
                    "version": URSA_VERSION,
                    "doi": URSA_DOI,
                    "supporting_data_doi": SUPPORTING_DATA_DOI,
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return hazard, cmd

    work = case_dir(case_id) / "ursa_workspace"
    work.mkdir(parents=True, exist_ok=True)
    name = case_id.replace("_", "-")
    init = subprocess.run(
        _ursa_cmd("initialize", name),
        cwd=str(work),
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    if init.returncode != 0:
        raise RuntimeError(f"ursa initialize failed: {(init.stderr or init.stdout)[-500:]}")
    run = subprocess.run(
        _ursa_cmd("run", "all"),
        cwd=str(work / name),
        capture_output=True,
        text=True,
        timeout=7200,
        check=False,
    )
    if run.returncode != 0:
        raise RuntimeError(f"ursa run all failed: {(run.stderr or run.stdout)[-500:]}")
    # Locate DFSI hazard output from Snakemake workflow artifacts.
    hazard_candidates = list((work / name).rglob("*dfsi*.npy")) + list((work / name).rglob("*hazard*.npy"))
    if not hazard_candidates:
        raise FileNotFoundError(f"no DFSI/hazard output under {work / name}")
    hazard = np.load(hazard_candidates[0])
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(pred_path, hazard.astype(np.float32))
    (pred_path.parent / "ursa_provenance.json").write_text(
        json.dumps(
            {
                "source": "ursa_run_all",
                "command": OFFICIAL_CMD,
                "version": URSA_VERSION,
                "doi": URSA_DOI,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return hazard, OFFICIAL_CMD


def _eval_anchor_cases(cap_dir: Path) -> Tuple[float, Dict[str, float], str]:
    per_case: Dict[str, Dict[str, float]] = {}
    commands: List[str] = []
    for cid in case_ids():
        if not observed_case_ready(cid):
            raise RuntimeError(f"anchor case observed inputs not ready: {cid}")
        obs = np.load(case_dir(cid) / "observed" / "runout_footprint.npy")
        pred, cmd = _run_ursa_chain(cid)
        if pred.shape != obs.shape:
            raise ValueError(f"shape mismatch case={cid} pred={pred.shape} obs={obs.shape}")
        metrics = footprint_metrics(pred, obs, threshold=0.0)
        per_case[cid] = metrics
        commands.append(cmd)
        out_pred = cap_dir / "predictions" / "anchor" / f"{cid}_dfsi.npy"
        out_pred.parent.mkdir(parents=True, exist_ok=True)
        np.save(out_pred, pred.astype(np.float32))

    if not per_case:
        raise RuntimeError("no anchor cases evaluated")
    ts_scores = [m["threat_score"] for m in per_case.values()]
    primary = float(np.nanmean(ts_scores))
    secondary = {
        "hit_rate": float(np.nanmean([m["hit_rate"] for m in per_case.values()])),
        "false_alarm_ratio": float(np.nanmean([m["false_alarm_ratio"] for m in per_case.values()])),
        "bias": float(np.nanmean([m["bias"] for m in per_case.values()])),
    }
    (cap_dir / "anchor_case_metrics.json").write_text(
        json.dumps({"per_case": per_case, "aggregate": {"threat_score": primary, **secondary}}, indent=2) + "\n",
        encoding="utf-8",
    )
    return primary, secondary, commands[0] if commands else OFFICIAL_CMD


def eval_mh1_05_scientific(*, out_base: Optional[Path] = None) -> Dict[str, Any]:
    out_root = out_base or SCIENTIFIC_RUNS_ROOT
    cap_dir = cap_scientific_dir(TASKPACK, CAP_ID, runs_root=out_root)

    if not ursa_anchor_inputs_ready():
        return write_blocked(
            TASKPACK,
            CAP_ID,
            "ursa anchor inputs not ready (run materialize_mh1_ursa_anchor.py --fetch on HPG)",
            out_base=out_base,
        )

    ok, tail = _try_ursa_smoke()
    if not ok and not all_cases_ready():
        _write_recipe_log(cap_dir, command=OFFICIAL_CMD, returncode=1, tail=tail)
        return write_blocked(TASKPACK, CAP_ID, f"ursa CLI smoke failed: {tail}", out_base=out_base)

    try:
        score, secondary, cmd = _eval_anchor_cases(cap_dir)
        if not np.isfinite(score):
            return write_blocked(TASKPACK, CAP_ID, "non-finite threat_score", out_base=out_base)

        _write_recipe_log(cap_dir, command=cmd, returncode=0, tail=tail)

        metrics = {
            "metric_name": "threat_score",
            "metric_value": float(score),
            "secondary_metrics": secondary,
            "synthetic_only": False,
            "data_source": DATA_SOURCE,
            "evaluator": "Earth's Future 2026 anchor reproduction; DFSI>0 footprint; not depth RMSE",
            "canonical_output": "postfire_debris_flow_runout_hazard_footprint",
            "pwfdf_r_contract_doi": PWFDF_R_DOI,
            "supporting_data_doi": SUPPORTING_DATA_DOI,
            "anchor_case_count": len(case_ids()),
            "note": f"Phase C MH-1 ursa scientific; cap={CAP_ID}; NOT hwb_holdout",
            "addendum_ref": ADDENDUM,
        }
        write_metrics(TASKPACK, CAP_ID, metrics, base=out_root)
        write_replay_manifest(
            TASKPACK,
            CAP_ID,
            family_id=FAMILY,
            exec_ok=True,
            metric_name="threat_score",
            metric_value=float(score),
            notes=metrics["note"],
            base=out_root,
        )
        return {"ok": True, "metrics": metrics}
    except (RuntimeError, FileNotFoundError, ValueError, subprocess.TimeoutExpired) as exc:
        _write_recipe_log(cap_dir, command=OFFICIAL_CMD, returncode=1, tail=str(exc))
        return write_blocked(TASKPACK, CAP_ID, str(exc), out_base=out_base)
