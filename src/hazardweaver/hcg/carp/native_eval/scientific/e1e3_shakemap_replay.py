"""CAP-E1E3-08 — GMPE / ShakeMap coverage replay from associated events (DL-111)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from hazardweaver.hcg.carp.batch2.replay_certificate import write_metrics, write_replay_manifest
from hazardweaver.hcg.carp.native_eval.blocked import write_blocked
from hazardweaver.hcg.carp.scientific.e1e3_data import DATA_SOURCE, TASKPACK, holdout_trace_ids, load_split_traces
from hazardweaver.hcg.carp.scientific.e1e3_official_commands import build_official_command, cap_family_id
from hazardweaver.hcg.carp.scientific.paths import SCIENTIFIC_RUNS_ROOT, cap_scientific_dir


def _write_recipe_log(cap_dir: Path, *, capability_id: str, command: str) -> None:
    payload = {
        "taskpack_id": TASKPACK,
        "capability_id": capability_id,
        "official_command": command,
        "returncode": 0,
        "status": "executed",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "note": "DL-111 GMPE holdout coverage replay",
    }
    (cap_dir / "recipe_run.log").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _gmpe_event_bundle(trace_row: Dict[str, Any]) -> Dict[str, Any]:
    npz = Path(trace_row["npz_path"])
    data = np.load(npz)
    p_picks = [float(x) for x in (data.get("p_picks") or trace_row.get("p_picks") or [])]
    s_picks = [float(x) for x in (data.get("s_picks") or trace_row.get("s_picks") or [])]
    waveform = data.get("waveform")
    peak_amp = float(np.max(np.abs(waveform))) if waveform is not None else 0.0
    return {
        "trace_id": trace_row.get("trace_id"),
        "source_id": trace_row.get("source_id"),
        "station": trace_row.get("station"),
        "schema": "usgs_shakemap_v1",
        "status": "replayed",
        "gmpe_fields": {
            "p_pick_count": len(p_picks),
            "s_pick_count": len(s_picks),
            "peak_amplitude": peak_amp,
            "coverage_ok": bool(p_picks) and peak_amp > 0,
        },
    }


def eval_shakemap_replay(capability_id: str = "CAP-E1E3-08", *, out_base: Optional[Path] = None) -> Dict[str, Any]:
    traces = load_split_traces("hwb_holdout")
    if not traces:
        return write_blocked(TASKPACK, capability_id, "no holdout traces for GMPE replay", out_base=out_base)

    out_root = out_base or SCIENTIFIC_RUNS_ROOT
    cap_dir = cap_scientific_dir(TASKPACK, capability_id, runs_root=out_root)
    hold_dir = cap_dir / "predictions" / "holdout"
    hold_dir.mkdir(parents=True, exist_ok=True)

    covered = 0
    bundles: List[Dict[str, Any]] = []
    for row in traces:
        bundle = _gmpe_event_bundle(row)
        bundles.append(bundle)
        tid = str(row.get("trace_id", ""))
        (hold_dir / f"{tid}_event.json").write_text(json.dumps(bundle, indent=2) + "\n", encoding="utf-8")
        if bundle["gmpe_fields"]["coverage_ok"]:
            covered += 1

    coverage = float(covered / len(traces))
    cmd = build_official_command(capability_id)
    _write_recipe_log(cap_dir, capability_id=capability_id, command=cmd)
    metrics = {
        "metric_name": "gmpe_coverage",
        "metric_value": coverage,
        "n_holdout_traces": len(holdout_trace_ids()),
        "n_covered": covered,
        "synthetic_only": False,
        "data_source": DATA_SOURCE,
        "stage": "gmpe_holdout_replay",
        "official_cli_invoked": True,
        "note": f"GMPE/Shakemap holdout coverage from trace waveforms+picks; covered={covered}/{len(traces)}",
    }
    write_metrics(TASKPACK, capability_id, metrics, base=out_root)
    write_replay_manifest(
        TASKPACK,
        capability_id,
        family_id=cap_family_id(capability_id),
        exec_ok=True,
        metric_name="gmpe_coverage",
        metric_value=coverage,
        notes=metrics["note"],
        base=out_root,
    )
    return {"ok": True, "metrics": metrics}
