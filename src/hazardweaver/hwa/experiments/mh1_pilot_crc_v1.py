"""MH-1 pilot CRC trajectories — HWA #11 block (DL-081 handoff).

Runs PFDF agent on HWB StateVariant manifest (9 variants / 3 burn pairs).
Burn preconditioning: baseline vs scale_fraction_mod_high ±10%.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[3]
VARIANT_MANIFEST = PROJECT_ROOT / "hwb/manifests/MH1_STATE_VARIANTS_PILOT_v1.jsonl"
EXPERIMENT_ID = "mh1_pilot_crc_v1"

SCENARIO_PRIMARY_FIRE: Dict[str, str] = {
    "thomas_montecito_2017": "Thomas",
    "pipeline_copeland_2022": "Pipeline",
    "el_dorado_apple_birch_2020": "El Dorado",
}

PFDF_TOOLS: tuple[str, ...] = (
    "list_inventory",
    "load_sample",
    "run_burn_predictor",
    "run_volume_predictor",
    "compose_burn_to_volume",
    "submit_answer",
)

VALID_PFDF_ROUTE_FAMILIES: frozenset[str] = frozenset(
    {"pfdf_burn_volume_cascade", "pfdf_compose"}
)


def load_variants(path: Path | None = None) -> List[Dict[str, Any]]:
    p = path or VARIANT_MANIFEST
    return [json.loads(line) for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def preconditioning_factor(state_delta_kind: str) -> Optional[float]:
    kind = str(state_delta_kind or "")
    if kind == "baseline":
        return None
    if kind == "burn_severity_plus_10pct":
        return 1.10
    if kind == "burn_severity_minus_10pct":
        return 0.90
    return None


def resolve_record_id_for_scenario(scenario_id: str) -> str:
    """Map atlas scenario_id → USGS PFDF inventory record_id (primary fire)."""
    from hazardweaver.hwa.pfdf_agent.data_access import PfdfDataAccess

    fire = SCENARIO_PRIMARY_FIRE.get(scenario_id)
    if not fire:
        raise KeyError(f"unknown MH-1 scenario_id: {scenario_id}")
    data = PfdfDataAccess()
    for rid in data.list_record_ids(limit=8000):
        rec = data.get_record(rid)
        try:
            area = float(rec.get("Area_km2") or 0)
            rain = float(rec.get("i30RainfallAnomaly") or 0)
        except (TypeError, ValueError):
            continue
        if str(rec.get("FireName") or "") == fire and area > 0 and rain > 0:
            return str(rid)
    raise KeyError(f"no valid PFDF record for scenario={scenario_id} fire={fire}")


def build_solver_task(variant: Mapping[str, Any], *, record_id: str) -> Dict[str, Any]:
    vid = str(variant["variant_id"])
    scenario = str(variant["scenario_id"])
    kind = str(variant["state_delta_kind"])
    factor = preconditioning_factor(kind)
    pre: Dict[str, Any] = {}
    if factor is not None:
        pre = {
            "operator_id": "scale_fraction_mod_high",
            "factor": factor,
            "mh_coupling": "preconditioning",
        }
    return {
        "schema_version": "agent_bench_task/v1",
        "task_id": f"mh1_{vid}",
        "domain": "pfdf",
        "task_family": "mh1_preconditioning_crc",
        "track": "MH-1",
        "user_facing_goal": (
            f"Estimate postfire debris-flow log-volume for watershed {scenario} "
            f"under MH-1 burn preconditioning ({kind}). "
            "Use burn then volume tools; submit log_volume."
        ),
        "solver_visible": {
            "inputs": {
                "sample_refs": [
                    {
                        "dataset_id": "usgs_pfdf_inventory_v1",
                        "sample_id": record_id,
                        "notes": "MH-1 pilot; sample_id≡record_id",
                    }
                ],
                "scenario_id": scenario,
                "variant_id": vid,
                "pair_id": variant.get("pair_id"),
                "state_delta_kind": kind,
                "solver_visible_delta": dict(variant.get("solver_visible_delta") or {}),
                "mh1_burn_preconditioning": pre,
                "constraints": [],
                "hints": ["run_burn_predictor then run_volume_predictor"],
            },
            "allowed_inventory": {
                "dataset_ids": ["usgs_pfdf_inventory_v1"],
                "model_ids": ["burn_state_net_prithvi_v1", "pfdf_volume_gorr_v2"],
                "tool_ids": list(PFDF_TOOLS),
            },
            "legacy_mode": True,
            "pack_refs": [],
        },
        "metadata": {
            "mh1_pilot": True,
            "crc_eligible": bool(variant.get("crc_eligible", True)),
            "capability_snapshot_ref": variant.get("capability_snapshot_ref"),
        },
        "success_criteria": {
            "answer_type": "log_volume",
            "description": "submit log_volume",
            "metric": "mae",
        },
    }


def route_family_from_workdir(workdir: Path) -> str:
    """Derive CRC route_family from PFDF tool sequence."""
    path = Path(workdir) / "tool_calls.jsonl"
    if not path.is_file():
        return "unknown"
    tools: List[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if not row.get("ok"):
            continue
        name = str(row.get("name") or "")
        if name in {"run_burn_predictor", "run_volume_predictor", "compose_burn_to_volume"}:
            tools.append(name)
    if "compose_burn_to_volume" in tools:
        return "pfdf_compose"
    if "run_burn_predictor" in tools and "run_volume_predictor" in tools:
        return "pfdf_burn_volume_cascade"
    return "unknown"


def crc_eligible_submission(*, exit_reason: str, route_family: str) -> bool:
    """CRC counts only clean PFDF terminal solves with a complete route family."""
    return exit_reason == "submitted" and route_family in VALID_PFDF_ROUTE_FAMILIES


def trajectory_pair_from_workdir(
    variant: Mapping[str, Any],
    workdir: Path,
    *,
    submitted: bool,
) -> Dict[str, Any]:
    wd = Path(workdir)
    return {
        "variant_id": variant.get("variant_id"),
        "scenario_id": variant.get("scenario_id"),
        "pair_id": variant.get("pair_id"),
        "state_delta_kind": variant.get("state_delta_kind"),
        "route_family": route_family_from_workdir(wd),
        "route_id": route_family_from_workdir(wd),
        "submitted": bool(submitted),
        "workdir": str(wd),
    }


def build_trajectory_pairs(
    variants: Sequence[Mapping[str, Any]],
    rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    by_task = {str(r.get("task_id")): r for r in rows}
    pairs: List[Dict[str, Any]] = []
    for v in variants:
        tid = f"mh1_{v['variant_id']}"
        row = by_task.get(tid) or {}
        pairs.append(
            trajectory_pair_from_workdir(
                v,
                Path(row.get("workdir") or ""),
                submitted=bool(row.get("crc_eligible")),
            )
        )
    return pairs


def materialize_solver_views(
    out_dir: Path,
    variants: Sequence[Mapping[str, Any]] | None = None,
) -> List[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: List[Path] = []
    for v in variants or load_variants():
        rid = resolve_record_id_for_scenario(str(v["scenario_id"]))
        task = build_solver_task(v, record_id=rid)
        path = out_dir / f"{task['task_id']}.json"
        path.write_text(json.dumps(task, indent=2) + "\n", encoding="utf-8")
        paths.append(path)
    return paths


@dataclass(frozen=True)
class RunCell:
    variant: Dict[str, Any]
    task_path: Path
    workdir: Path


def iter_cells(
    *,
    solver_dir: Path,
    out_root: Path,
    variants: Sequence[Mapping[str, Any]] | None = None,
) -> Iterable[RunCell]:
    for v in variants or load_variants():
        tid = f"mh1_{v['variant_id']}"
        yield RunCell(
            variant=dict(v),
            task_path=solver_dir / f"{tid}.json",
            workdir=out_root / tid,
        )


def run_experiment(
    *,
    out_root: Path,
    max_steps: int = 50,
    max_wall_s: float = 1800.0,
    build_llm: Any = None,
    arm: str = "",
) -> Dict[str, Any]:
    from hazardweaver.hwa.agent_runtime.loop import AgentLimits, SWEAgentLoop

    out_root = Path(out_root)
    solver_dir = out_root / "solver_view"
    materialize_solver_views(solver_dir)
    variants = load_variants()

    if build_llm is None:
        from hazardweaver.hwa.agent_runtime.run import build_llm as _build_llm

        llm = _build_llm()
    else:
        llm = build_llm()

    loop = SWEAgentLoop(
        llm,
        limits=AgentLimits(max_steps=max_steps, max_wall_s=max_wall_s),
        pfdf_oracle_burn=True,
        controller_mode=False,
    )

    rows: List[Dict[str, Any]] = []
    for cell in iter_cells(solver_dir=solver_dir, out_root=out_root):
        task = json.loads(cell.task_path.read_text(encoding="utf-8"))
        try:
            result = loop.run(task, workdir=cell.workdir)
            route = route_family_from_workdir(result.workdir)
            row = {
                "task_id": result.task_id,
                "variant_id": cell.variant["variant_id"],
                "pair_id": cell.variant["pair_id"],
                "scenario_id": cell.variant["scenario_id"],
                "state_delta_kind": cell.variant["state_delta_kind"],
                "workdir": str(result.workdir),
                "loop_submitted": result.submitted,
                "submitted": result.submitted,
                "exit_reason": result.exit_reason,
                "n_steps": result.n_steps,
                "route_family": route,
            }
            row["crc_eligible"] = crc_eligible_submission(
                exit_reason=str(row["exit_reason"] or ""),
                route_family=route,
            )
        except Exception as exc:  # noqa: BLE001
            import traceback

            row = {
                "task_id": f"mh1_{cell.variant['variant_id']}",
                "variant_id": cell.variant["variant_id"],
                "workdir": str(cell.workdir),
                "loop_submitted": False,
                "submitted": False,
                "crc_eligible": False,
                "exit_reason": "exception",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc(limit=6),
            }
        rows.append(row)
        print(
            f"[{len(rows)}/9] {row.get('variant_id')} "
            f"crc_eligible={row.get('crc_eligible')} route={row.get('route_family')}",
            flush=True,
        )

    pairs = build_trajectory_pairs(variants, rows)
    crc_report: Dict[str, Any] = {"blocked": True}
    try:
        from hazardweaver.hwb.build.mh1_state_variant import crc_status

        crc_report = crc_status(trajectory_pairs=pairs)
    except Exception as exc:  # noqa: BLE001
        crc_report = {"blocked": True, "error": str(exc)}

    manifest = {
        "experiment_id": EXPERIMENT_ID,
        "arm": arm,
        "n_variants": len(variants),
        "n_loop_submitted": sum(1 for r in rows if r.get("loop_submitted")),
        "n_submitted": sum(1 for r in rows if r.get("crc_eligible")),
        "out_root": str(out_root),
        "solver_view_dir": str(solver_dir),
        "variant_manifest": str(VARIANT_MANIFEST),
        "trajectory_pairs": pairs,
        "crc_status": crc_report,
        "tasks": rows,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
    }
    man_path = out_root / "manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    pairs_path = out_root / "trajectory_pairs.json"
    pairs_path.write_text(json.dumps(pairs, indent=2) + "\n", encoding="utf-8")
    manifest["manifest_path"] = str(man_path)
    return manifest
