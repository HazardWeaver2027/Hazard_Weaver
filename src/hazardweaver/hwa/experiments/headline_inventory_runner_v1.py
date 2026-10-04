"""Run Full HWA vs ReAct on HWB ICLR headline inventory (218+ instances).

Bridges ``headline_inventory_v1.jsonl`` rows → HWA solver tasks (taskpack solver_view).
Controller routes use ``allowed_edge_ids`` only (no reference_view witness leakage).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from hazardweaver.hwb.registry.solver_allowed_edges_v1 import resolve_solver_allowed_edge_ids
from hazardweaver.hwb.run.eval_dca_submission_v1 import resolve_taskpack_for_inventory_row
from hazardweaver.hwa.experiments.headline_ds_cbr_v1 import attach_ds_cbr_metadata, should_attach_ds_cbr
from hazardweaver.hwa.experiments.headline_route_mode_v1 import (
    apply_graph_config_to_solver_visible,
    headline_route_class,
    hcg_readonly_tool_ids,
    infer_track_graph_config,
    resolve_headline_route_mode,
)

ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INVENTORY = ROOT / "runs/hwb/iclr_benchmark/headline_inventory_v1.jsonl"

VALID_CONDITIONS: Tuple[str, ...] = ("full_hwa", "react", "direct_answer")
DEFAULT_SEED = 11

_SHARED_READONLY_TOOLS: Tuple[str, ...] = (
    "inspect_artifact",
    "list_inventory",
    "get_execution_result",
    "submit",
    "submit_answer",
    "submit_solution",
    "submit_clarification",
    "submit_abstention",
)
_VCE_SOLVE_TOOLS: Tuple[str, ...] = tuple(
    t for t in _SHARED_READONLY_TOOLS if t != "submit_clarification"
)
_CONTROLLER_TOOLS: Tuple[str, ...] = (
    "controller_enumerate_routes",
    "controller_propose_route",
    "controller_commit_route",
    "controller_get_route_status",
)
_REACT_EXEC_TOOLS: Tuple[str, ...] = ("run_capability",)

TRACK_DOMAIN: Dict[str, str] = {
    "MH-1": "pfdf",
    "MH-2": "multi_hazard",
    "MH-3": "multi_hazard",
    "MH-4": "multi_hazard",
    "FL-2": "compound_flood",
    "WF-3": "wildfire",
    "L2": "landslide",
    "E1-E3": "earthquake",
    "TC-TRK": "cyclone",
    "DR-OUT": "drought",
    "HW-MED": "health",
}


def load_headline_inventory(path: Path | None = None) -> List[Dict[str, Any]]:
    src = Path(path or DEFAULT_INVENTORY)
    return [json.loads(line) for line in src.read_text(encoding="utf-8").splitlines() if line.strip()]


def opaque_id_for_instance(instance_id: str) -> str:
    digest = hashlib.sha256(instance_id.encode("utf-8")).hexdigest()
    num = int(digest[:8], 16) % 1_000_000
    return f"T_{num:06d}"


def inventory_workdir_key(inventory_row: Mapping[str, Any]) -> str:
    """Stable per-row workdir key; paired interventions must not collide on instance_id."""
    for key in (
        "rq4_cell_id",
        "hkc_pair_id",
        "hcg_pair_id",
        "src_dynamic_pair_key",
    ):
        val = str(inventory_row.get(key) or "").strip()
        if val:
            return val.replace(":", "__").replace("/", "_")
    return opaque_id_for_instance(str(inventory_row["instance_id"]))


def workdir_for_inventory_row(
    out_root: Path,
    *,
    seed: int,
    condition: str,
    inventory_row: Mapping[str, Any],
) -> Path:
    """One workdir per inventory row; paired interventions use explicit pair/cell ids."""
    base = Path(out_root) / f"seed_{seed}" / condition
    return base / inventory_workdir_key(inventory_row)


def controller_mode_for(condition: str) -> bool:
    if condition == "full_hwa":
        return True
    if condition in {"react", "direct_answer"}:
        return False
    raise ValueError(f"unknown condition: {condition!r}")


def infer_domain(*, track: str, taskpack: Mapping[str, Any]) -> str:
    tr = str(track or taskpack.get("headline_target") or "").upper()
    if tr in TRACK_DOMAIN:
        return TRACK_DOMAIN[tr]
    if tr.startswith("MH-"):
        return "multi_hazard"
    return "multi_hazard"


def prepare_headline_solver_task(
    inventory_row: Mapping[str, Any],
    taskpack: Mapping[str, Any],
    condition: str,
) -> Dict[str, Any]:
    if condition not in VALID_CONDITIONS:
        raise ValueError(f"unknown condition: {condition!r}")

    instance_id = str(inventory_row["instance_id"])
    opaque_id = opaque_id_for_instance(instance_id)
    sv = dict(taskpack.get("solver_view") or {})
    params = dict(sv.get("parameters") or {})
    allowed_edges = resolve_solver_allowed_edge_ids(taskpack, inventory_row=inventory_row)
    raw_goal = str(
        inventory_row.get("user_goal") or sv.get("user_goal") or taskpack.get("user_facing_goal") or ""
    )
    route_mode = resolve_headline_route_mode(inventory_row, taskpack, allowed_edges=allowed_edges)
    track = str(inventory_row.get("track") or taskpack.get("headline_target") or "")
    expected_action = str(
        (taskpack.get("reference_view") or {}).get("expected_action")
        or inventory_row.get("expected_action")
        or "solve"
    )

    if condition == "full_hwa":
        from hazardweaver.hwa.control.mode import is_vce_mode

        base_tools = (
            _VCE_SOLVE_TOOLS
            if is_vce_mode() and expected_action == "solve"
            else _SHARED_READONLY_TOOLS
        )
        tool_ids = list(base_tools) + list(_CONTROLLER_TOOLS)
        from hazardweaver.hwa.experiments.agent_strict_v2 import agent_strict_v2_enabled, strict_solver_tool_ids

        if agent_strict_v2_enabled():
            tool_ids = strict_solver_tool_ids(tool_ids)
        if route_mode == "hcg_multi_hop":
            tool_ids = tool_ids + hcg_readonly_tool_ids()
        controller_mode = True
    elif condition == "direct_answer":
        tool_ids = [
            t
            for t in _SHARED_READONLY_TOOLS
            if t in {"submit", "submit_answer", "submit_solution", "submit_abstention", "submit_clarification"}
        ]
        controller_mode = False
    else:
        tool_ids = list(_SHARED_READONLY_TOOLS) + list(_REACT_EXEC_TOOLS)
        # Fair ReAct baseline on multi-hop cells: same read-only HCG graph tools as full_hwa,
        # but no controller / SCC — execution stays direct run_capability on allowed edges.
        if route_mode == "hcg_multi_hop":
            tool_ids = tool_ids + hcg_readonly_tool_ids()
        controller_mode = False

    solver_visible: Dict[str, Any] = {
        "controller_mode": controller_mode,
        "inputs": {
            "allowed_edge_ids": allowed_edges,
            "parameters": params,
            "scenario_id": params.get("scenario_id") or inventory_row.get("scenario_id"),
            "split": params.get("split"),
        },
        "allowed_inventory": {"tool_ids": tool_ids},
        "constraints": {
            "no_gold_in_solver_view": True,
            "forbid_reference_score_fallback": True,
        },
    }
    if condition == "full_hwa" and route_mode == "hcg_multi_hop":
        graph_cfg = infer_track_graph_config(track, taskpack)
        if graph_cfg:
            solver_visible = apply_graph_config_to_solver_visible(solver_visible, graph_cfg)
            task_domain = str(graph_cfg.get("domain") or infer_domain(track=track, taskpack=taskpack))
        else:
            task_domain = infer_domain(track=track, taskpack=taskpack)
    elif condition == "react" and route_mode == "hcg_multi_hop":
        graph_cfg = infer_track_graph_config(track, taskpack)
        if graph_cfg:
            solver_visible = apply_graph_config_to_solver_visible(solver_visible, graph_cfg)
            task_domain = str(graph_cfg.get("domain") or infer_domain(track=track, taskpack=taskpack))
        else:
            task_domain = infer_domain(track=track, taskpack=taskpack)
    else:
        task_domain = infer_domain(track=track, taskpack=taskpack)

    meta_extra: Dict[str, Any] = {}
    if track == "MH-1" or any(
        str(e) in {"burn_state_net_prithvi_v1", "pfdf_volume_gorr_v2"} for e in allowed_edges
    ):
        task_domain = "pfdf"
        try:
            from hazardweaver.hwa.experiments.headline_pfdf_record_id_v1 import resolve_headline_pfdf_record_id

            meta_extra["pfdf_record_id"] = resolve_headline_pfdf_record_id(inventory_row)
        except KeyError:
            pass

    row_split = str(inventory_row.get("split") or params.get("split") or "").strip()

    task: Dict[str, Any] = {
        "task_id": opaque_id,
        "domain": task_domain,
        "user_facing_goal": raw_goal,
        "taskpack_id": str(taskpack.get("taskpack_id") or inventory_row.get("taskpack_id") or ""),
        "headline_target": track,
        "solver_visible": solver_visible,
        "metadata": {
            "hwb_headline_inventory": True,
            "same_llm_condition": condition,
            "instance_id": instance_id,
            "internal_task_id": instance_id,
            "taskpack_id": str(inventory_row.get("taskpack_id") or ""),
            "scenario_id": str(inventory_row.get("scenario_id") or params.get("scenario_id") or ""),
            "track": track,
            "difficulty_tier": str(inventory_row.get("difficulty_tier") or "L1"),
            "source": str(inventory_row.get("source") or ""),
            "routing_tier": str(inventory_row.get("routing_tier") or "headline"),
            "expected_action": expected_action,
            "headline_route_mode": route_mode if condition == "full_hwa" else "react_direct",
            "headline_route_class": headline_route_class(route_mode) if allowed_edges else "no_solver_edges",
            **meta_extra,
        },
    }
    if row_split:
        task["metadata"]["split"] = row_split
    if condition == "react" and route_mode == "hcg_multi_hop":
        task["metadata"]["headline_hcg_readonly_tools"] = True
    if condition == "react" and expected_action == "solve":
        sv = dict(task.get("solver_visible") or {})
        inv = dict((sv.get("allowed_inventory") or {}))
        inv["tool_ids"] = [
            t
            for t in list(inv.get("tool_ids") or [])
            if t not in {"submit_clarification"}
        ]
        sv["allowed_inventory"] = inv
        task["solver_visible"] = sv
        task["metadata"]["react_clarify_forbidden"] = True
    if expected_action == "clarify":
        task["clarify_mode"] = "decision"
    if condition == "react":
        task["metadata"]["headline_react_direct_execute"] = True
    if condition == "direct_answer":
        task["metadata"]["headline_direct_answer_only"] = True
        task["metadata"]["headline_route_mode"] = "direct_answer"
    from hazardweaver.hwa.experiments.headline_hwa_self_cbr_v1 import (
        attach_hwa_self_cbr_metadata,
        should_attach_hwa_self_cbr,
    )

    if should_attach_hwa_self_cbr(condition):
        attach_hwa_self_cbr_metadata(task, inventory_row)
    elif should_attach_ds_cbr(condition):
        attach_ds_cbr_metadata(task, inventory_row)
    from hazardweaver.hwa.experiments.agent_strict_v2 import (
        agent_strict_v2_enabled,
        apply_strict_solver_visible,
        apply_strict_vce_tools,
    )
    from hazardweaver.hwa.experiments.strict_solver_goal_v1 import (
        augment_user_facing_goal_for_strict,
        resolve_strict_solver_execution_hints,
    )

    if agent_strict_v2_enabled():
        from hazardweaver.hwa.experiments.agent_strict_v2 import strict_execution_hints_enabled

        hints = (
            resolve_strict_solver_execution_hints(
                inventory_row,
                taskpack_params=params,
            )
            if strict_execution_hints_enabled()
            else {}
        )
        task["user_facing_goal"] = (
            augment_user_facing_goal_for_strict(
                str(task.get("user_facing_goal") or raw_goal),
                hints,
            )
            if hints
            else str(task.get("user_facing_goal") or raw_goal)
        )
        if hints:
            task.setdefault("metadata", {})["strict_execution_hints"] = dict(hints)
        else:
            task.setdefault("metadata", {})["strict_execution_hints_disabled"] = True

    task = apply_strict_solver_visible(task)
    task = apply_strict_vce_tools(task)
    from hazardweaver.hwa.experiments.inventory_mechanism_overlay_v1 import apply_inventory_mechanism_overlays
    from hazardweaver.hwa.experiments.rq4_route_intervention_v1 import apply_rq4_intervention_from_row

    from hazardweaver.hwa.experiments.unified_benchmark_execution_v1 import (
        apply_unified_benchmark_overlays,
        seal_unified_runtime_allowed_edges,
    )

    task = apply_unified_benchmark_overlays(task, inventory_row)
    task = apply_inventory_mechanism_overlays(task, inventory_row)
    task = apply_rq4_intervention_from_row(task, inventory_row)
    if str(inventory_row.get("ablation_manual_pilot") or "") == "v1":
        from hazardweaver.hwa.experiments.ablation_manual_pilot_v1 import apply_ablation_manual_pilot_overlays

        task = apply_ablation_manual_pilot_overlays(task, inventory_row)
    from hazardweaver.hwa.scientific_controller.e1e3_hkc_bindings_v1 import apply_e1e3_hkc_task_overlays

    task = apply_e1e3_hkc_task_overlays(task, inventory_row)
    from hazardweaver.hwa.benchmark.unified_inventory_curator_v1 import apply_mtrack_unified_task_overlays

    task = apply_mtrack_unified_task_overlays(task, inventory_row)
    from hazardweaver.hwa.scientific_controller.mh_unified_hkc_bindings_v1 import apply_mh_unified_task_overlays

    task = apply_mh_unified_task_overlays(task, inventory_row)
    from hazardweaver.hwa.experiments.coupled_4m_mode_v1 import apply_coupled_row_metadata

    task = apply_coupled_row_metadata(task, inventory_row)
    task = seal_unified_runtime_allowed_edges(task, inventory_row)
    from hazardweaver.hwa.experiments.static_whitelist_s0_v1 import apply_static_whitelist_s0_freeze

    return apply_static_whitelist_s0_freeze(task)


def n_flat_cells(*, n_instances: int, conditions: Sequence[str] = VALID_CONDITIONS) -> int:
    return int(n_instances) * len(conditions)


def decode_flat_cell(
    flat_index: int,
    *,
    n_instances: int,
    conditions: Sequence[str] = VALID_CONDITIONS,
) -> Tuple[int, str]:
    if flat_index < 0 or flat_index >= n_flat_cells(n_instances=n_instances, conditions=conditions):
        raise IndexError(f"flat_index {flat_index} out of range for n_instances={n_instances}")
    instance_index = flat_index // len(conditions)
    condition = str(conditions[flat_index % len(conditions)])
    return instance_index, condition


@dataclass(frozen=True)
class HeadlineRunCell:
    flat_index: int
    instance_index: int
    condition: str
    seed: int
    inventory_row: Dict[str, Any]
    opaque_task_id: str
    workdir: Path


def iter_run_cells(
    inventory: Sequence[Mapping[str, Any]],
    *,
    conditions: Sequence[str] = VALID_CONDITIONS,
    seeds: Sequence[int] = (DEFAULT_SEED,),
    out_root: Path,
) -> Iterable[HeadlineRunCell]:
    out_root = Path(out_root)
    for seed in seeds:
        for flat_index in range(n_flat_cells(n_instances=len(inventory), conditions=conditions)):
            instance_index, condition = decode_flat_cell(
                flat_index, n_instances=len(inventory), conditions=conditions
            )
            row = dict(inventory[instance_index])
            iid = str(row["instance_id"])
            opaque = opaque_id_for_instance(iid)
            workdir = workdir_for_inventory_row(
                out_root, seed=int(seed), condition=condition, inventory_row=row
            )
            yield HeadlineRunCell(
                flat_index=flat_index,
                instance_index=instance_index,
                condition=condition,
                seed=int(seed),
                inventory_row=row,
                opaque_task_id=opaque,
                workdir=workdir,
            )


def cell_for_flat_index(
    inventory: Sequence[Mapping[str, Any]],
    flat_index: int,
    *,
    seed: int = DEFAULT_SEED,
    out_root: Path,
    conditions: Sequence[str] = VALID_CONDITIONS,
) -> HeadlineRunCell:
    instance_index, condition = decode_flat_cell(
        flat_index, n_instances=len(inventory), conditions=conditions
    )
    row = dict(inventory[instance_index])
    iid = str(row["instance_id"])
    opaque = opaque_id_for_instance(iid)
    return HeadlineRunCell(
        flat_index=flat_index,
        instance_index=instance_index,
        condition=condition,
        seed=int(seed),
        inventory_row=row,
        opaque_task_id=opaque,
        workdir=workdir_for_inventory_row(
            out_root, seed=int(seed), condition=condition, inventory_row=row
        ),
    )


def _patch_run_meta(workdir: Path, **fields: Any) -> None:
    meta_path = workdir / "run_meta.json"
    if not meta_path.is_file():
        return
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta.update(fields)
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")


def _inventory_row_for_sidecar(
    cell: HeadlineRunCell,
    task: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    row = dict(cell.inventory_row)
    meta = (task or {}).get("metadata") or {}
    for key in (
        "ablation_dca_tolerance",
        "ablation_gate_policy_override",
        "ablation_mechanism_primary",
    ):
        if meta.get(key) is not None:
            row[key] = meta[key]
    return row


def _write_sidecar_artifacts(
    workdir: Path,
    cell: HeadlineRunCell,
    taskpack: Mapping[str, Any],
    task: Optional[Mapping[str, Any]] = None,
) -> None:
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "inventory_row.json").write_text(
        json.dumps(_inventory_row_for_sidecar(cell, task), indent=2) + "\n", encoding="utf-8"
    )
    (workdir / "resolved_taskpack.json").write_text(
        json.dumps(taskpack, indent=2) + "\n", encoding="utf-8"
    )


def _write_dca_for_cell(
    workdir: Path,
    *,
    inventory_row: Mapping[str, Any],
    taskpack: Mapping[str, Any],
) -> Dict[str, Any]:
    """Emit dca_result.json when a submission exists ."""
    answer_path = workdir / "answer.json"
    if not answer_path.is_file():
        return {"skipped": True, "reason": "no_answer"}
    try:
        from hazardweaver.hwa.benchmark.dca_episode_eligibility_v1 import (
            incomplete_episode_dca_dict,
            is_dca_eligible,
        )
        from hazardweaver.hwa.benchmark.dca_mechanism_unify_v1 import apply_mechanism_veto_to_dca

        if not is_dca_eligible(workdir):
            meta = json.loads((workdir / "run_meta.json").read_text(encoding="utf-8"))
            dca_dict = incomplete_episode_dca_dict(
                taskpack_id=str(taskpack.get("taskpack_id") or ""),
                exit_reason=str(meta.get("exit_reason") or ""),
            )
            (workdir / "dca_result.json").write_text(
                json.dumps(dca_dict, indent=2) + "\n",
                encoding="utf-8",
            )
            return dca_dict

        from hazardweaver.hwa.runtime.unified_final_artifact_align_v1 import align_unified_final_artifact_if_needed
        from hazardweaver.hwb.bridge.hwa_workdir import load_hwa_submission
        from hazardweaver.hwb.evaluators.dca_scorer import evaluate_dca_submission

        align_unified_final_artifact_if_needed(workdir, inventory_row)
        submission = load_hwa_submission(workdir)
        tier = str(inventory_row.get("difficulty_tier") or "L1")
        dca = evaluate_dca_submission(
            taskpack,
            submission,
            agent_id="hazardweaver",
            difficulty_tier=tier,
            inventory_row=inventory_row,
        )
        dca_dict = apply_mechanism_veto_to_dca(dca.to_dict(), workdir)
        from hazardweaver.hwa.benchmark.unified_dca_route_gate_v1 import (
            apply_unified_route_gate_to_dca,
            infer_dca_route_condition,
        )

        dca_dict = apply_unified_route_gate_to_dca(
            dca_dict,
            workdir,
            inventory_row=inventory_row,
            condition=infer_dca_route_condition(workdir),
        )
        (workdir / "dca_result.json").write_text(
            json.dumps(dca_dict, indent=2) + "\n",
            encoding="utf-8",
        )
        return dca_dict
    except Exception as exc:  # noqa: BLE001
        err = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        (workdir / "dca_result.json").write_text(
            json.dumps(err, indent=2) + "\n",
            encoding="utf-8",
        )
        return err


def _agent_strict_cell_has_artifacts(workdir: Path) -> bool:
    """Run produced persisted meta + DCA artifacts (independent of success)."""
    if not workdir.is_dir():
        return False
    return (workdir / "run_meta.json").is_file() and (workdir / "dca_result.json").is_file()


def _agent_strict_react_cell_needs_repair(workdir: Path) -> bool:
    """ReAct solve ablation artifacts that are terminal but not comparable to full_hwa."""
    if not _agent_strict_cell_has_artifacts(workdir):
        return False
    try:
        meta = json.loads((workdir / "run_meta.json").read_text(encoding="utf-8"))
        ans = json.loads((workdir / "answer.json").read_text(encoding="utf-8")) if (workdir / "answer.json").is_file() else {}
    except (OSError, json.JSONDecodeError):
        return False
    if str(meta.get("same_llm_condition") or "") != "react":
        return False
    action = str((ans.get("answer") or {}).get("action") or "")
    emit = str(ans.get("emit_source") or "")
    exit_reason = str(meta.get("exit_reason") or "")
    if action == "clarify":
        return True
    if emit == "agent_loop_unterminated":
        return True
    if not meta.get("submitted") and exit_reason in {"limit_steps", "limit_wall", "parse_loop"}:
        return True
    return False


def _agent_strict_cell_terminal(workdir: Path) -> bool:
    """Completed run with a legal terminal exit (success or unsuccessful)."""
    if (workdir / "engineering_skip.json").is_file():
        return True
    if _agent_strict_react_cell_needs_repair(workdir):
        return False
    if not _agent_strict_cell_has_artifacts(workdir):
        return False
    try:
        meta = json.loads((workdir / "run_meta.json").read_text(encoding="utf-8"))
        dca = json.loads((workdir / "dca_result.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    exit_reason = str(meta.get("exit_reason") or "").strip()
    if not exit_reason or exit_reason == "unknown":
        return False
    # Pre-run harness error stub — not a finished agent episode.
    if dca.get("ok") is False and dca.get("error") and not dca.get("outcome"):
        return False
    return True


def _agent_strict_cell_success(workdir: Path) -> bool:
    """Strict solve success (task counted as solved under agent-strict protocol)."""
    if not _agent_strict_cell_terminal(workdir):
        return False
    try:
        dca = json.loads((workdir / "dca_result.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not dca.get("valid") or str(dca.get("outcome") or "") != "solve":
        return False
    ans_path = workdir / "answer.json"
    if ans_path.is_file():
        try:
            ans = json.loads(ans_path.read_text(encoding="utf-8"))
            action = str((ans.get("answer") or {}).get("action") or "")
            if action != "solve":
                return False
        except (OSError, json.JSONDecodeError):
            return False
    return True


def _agent_strict_cell_complete(workdir: Path) -> bool:
    """Backward-compatible alias: strict solve success (not terminal completion)."""
    return _agent_strict_cell_success(workdir)


def _engineering_blocked_row(row: Mapping[str, Any]) -> bool:
    from hazardweaver.hwb.registry.expand_parametric_headline_v1 import is_headline_engineering_blocked

    iid = str(row.get("instance_id") or "")
    sid = str(row.get("scenario_id") or "")
    cid = row.get("capability_id")
    return (
        is_headline_engineering_blocked(iid)
        or is_headline_engineering_blocked(sid, capability_id=cid)
    )


def run_headline_cell(
    cell: HeadlineRunCell,
    *,
    loop: Any,
    skip_if_submitted: bool = True,
) -> Dict[str, Any]:
    skip_marker = cell.workdir / "engineering_skip.json"
    if skip_marker.is_file():
        return {
            "flat_index": cell.flat_index,
            "instance_id": cell.inventory_row.get("instance_id"),
            "opaque_task_id": cell.opaque_task_id,
            "condition": cell.condition,
            "seed": cell.seed,
            "workdir": str(cell.workdir),
            "status": "engineering_blocked",
            "submitted": False,
            "exit_reason": "engineering_blocked",
        }

    from hazardweaver.hwa.experiments.ablation_manual_pilot_v1 import ablation_env_skip_reason

    env_skip = ablation_env_skip_reason(cell.inventory_row)
    if env_skip:
        cell.workdir.mkdir(parents=True, exist_ok=True)
        skip_marker.write_text(
            json.dumps(
                {
                    "status": "ENV_UNSUPPORTED",
                    "reason": env_skip,
                    "instance_id": cell.inventory_row.get("instance_id"),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return {
            "flat_index": cell.flat_index,
            "instance_id": cell.inventory_row.get("instance_id"),
            "opaque_task_id": cell.opaque_task_id,
            "condition": cell.condition,
            "seed": cell.seed,
            "workdir": str(cell.workdir),
            "status": "env_unsupported",
            "submitted": False,
            "exit_reason": "env_unsupported",
        }

    if _engineering_blocked_row(cell.inventory_row):
        from hazardweaver.hwb.registry.expand_parametric_headline_v1 import headline_block_reason

        cell.workdir.mkdir(parents=True, exist_ok=True)
        sid = str(cell.inventory_row.get("scenario_id") or "")
        cid = cell.inventory_row.get("capability_id")
        reason = headline_block_reason(sid, capability_id=cid) or "ENGINEERING_BLOCKED"
        skip_marker.write_text(
            json.dumps(
                {
                    "status": "ENGINEERING_BLOCKED",
                    "reason": reason,
                    "instance_id": cell.inventory_row.get("instance_id"),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        return {
            "flat_index": cell.flat_index,
            "instance_id": cell.inventory_row.get("instance_id"),
            "opaque_task_id": cell.opaque_task_id,
            "condition": cell.condition,
            "seed": cell.seed,
            "workdir": str(cell.workdir),
            "status": "engineering_blocked",
            "submitted": False,
            "exit_reason": "engineering_blocked",
        }

    answer_path = cell.workdir / "answer.json"
    from hazardweaver.hwa.experiments.agent_strict_v2 import agent_strict_v2_enabled

    if skip_if_submitted:
        if agent_strict_v2_enabled():
            if _agent_strict_cell_terminal(cell.workdir):
                meta_path = cell.workdir / "run_meta.json"
                meta = (
                    json.loads(meta_path.read_text(encoding="utf-8"))
                    if meta_path.is_file()
                    else {}
                )
                dca = json.loads((cell.workdir / "dca_result.json").read_text(encoding="utf-8"))
                return {
                    "flat_index": cell.flat_index,
                    "instance_id": cell.inventory_row.get("instance_id"),
                    "opaque_task_id": cell.opaque_task_id,
                    "condition": cell.condition,
                    "seed": cell.seed,
                    "workdir": str(cell.workdir),
                    "status": "skipped_existing",
                    "submitted": bool(meta.get("submitted")),
                    "exit_reason": meta.get("exit_reason"),
                    "dca_valid": bool(dca.get("valid")),
                }
        elif answer_path.is_file():
            meta_path = cell.workdir / "run_meta.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
            return {
                "flat_index": cell.flat_index,
                "instance_id": cell.inventory_row.get("instance_id"),
                "opaque_task_id": cell.opaque_task_id,
                "condition": cell.condition,
                "seed": cell.seed,
                "workdir": str(cell.workdir),
                "status": "skipped_existing",
                "submitted": bool(meta.get("submitted")),
                "exit_reason": meta.get("exit_reason"),
            }

    if not skip_if_submitted:
        from hazardweaver.hwa.experiments.headline_cell_rerun_hygiene_v1 import (
            prepare_workdir_for_force_rerun,
        )

        prepare_workdir_for_force_rerun(cell.workdir)

    tp = resolve_taskpack_for_inventory_row(cell.inventory_row)
    task = prepare_headline_solver_task(cell.inventory_row, tp, cell.condition)
    _write_sidecar_artifacts(cell.workdir, cell, tp, task)

    try:
        result = loop.run(task, workdir=cell.workdir)
        from hazardweaver.hwa.experiments.agent_strict_v2 import headline_control_mode_label
        from hazardweaver.hwa.experiments.agent_strict_ablation_protocol_v1 import run_meta_protocol_patch

        _patch_run_meta(
            result.workdir,
            experiment_id="headline_inventory_runner_v1",
            hwb_headline_inventory=True,
            same_llm_condition=cell.condition,
            experiment_seed=cell.seed,
            controller_mode=controller_mode_for(cell.condition),
            control_mode=headline_control_mode_label(),
            internal_task_id=str(cell.inventory_row.get("instance_id")),
            opaque_task_id=cell.opaque_task_id,
            taskpack_id=str(cell.inventory_row.get("taskpack_id") or ""),
            instance_id=str(cell.inventory_row.get("instance_id")),
            track=str(cell.inventory_row.get("track") or ""),
            scenario_id=str(cell.inventory_row.get("scenario_id") or ""),
            difficulty_tier=str(cell.inventory_row.get("difficulty_tier") or "L1"),
            **run_meta_protocol_patch(cell.condition),
        )
        dca_dict = _write_dca_for_cell(
            result.workdir,
            inventory_row=cell.inventory_row,
            taskpack=tp,
        )
        if dca_dict.get("valid") is not None:
            _patch_run_meta(
                result.workdir,
                dca_valid=bool(dca_dict.get("valid")),
                dca_outcome=str(dca_dict.get("outcome") or ""),
            )
        submission_integrity: Dict[str, Any] = {}
        from hazardweaver.hwa.experiments.agent_strict_v2 import (
            agent_strict_v2_enabled,
            validate_strict_submission_integrity,
        )

        if agent_strict_v2_enabled():
            from hazardweaver.hwa.runtime.trajectory_ledger import load_ledger_steps

            transcript: List[Dict[str, Any]] = []
            tpath = result.workdir / "transcript.jsonl"
            if tpath.is_file():
                transcript = [
                    json.loads(line)
                    for line in tpath.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ]
            submission_integrity = validate_strict_submission_integrity(
                transcript,
                ledger_steps=load_ledger_steps(result.workdir),
                submitted=bool(result.submitted),
            )
            _patch_run_meta(result.workdir, submission_integrity=submission_integrity)
        row = {
            "flat_index": cell.flat_index,
            "instance_id": cell.inventory_row.get("instance_id"),
            "opaque_task_id": cell.opaque_task_id,
            "condition": cell.condition,
            "seed": cell.seed,
            "workdir": str(result.workdir),
            "status": "ok",
            "exit_reason": result.exit_reason,
            "submitted": result.submitted,
            "n_steps": result.n_steps,
            "n_tool_calls": result.n_tool_calls,
            "model_id": result.run_meta.get("model_id"),
            "llm_provider": result.run_meta.get("llm_provider"),
        }
        if dca_dict.get("valid") is not None:
            row["dca_valid"] = bool(dca_dict.get("valid"))
            row["dca_outcome"] = dca_dict.get("outcome")
        if submission_integrity:
            row["submission_integrity"] = submission_integrity
        try:
            from hazardweaver.hwa.experiments.ablation_manual_pilot_integrity_v1 import finalize_ablation_pilot_run

            submitted, exit_reason, ablation_integrity = finalize_ablation_pilot_run(
                result.workdir,
                task,
                submitted=bool(result.submitted),
                exit_reason=str(result.exit_reason or row.get("exit_reason") or ""),
            )
            row["submitted"] = submitted
            row["exit_reason"] = exit_reason
            patch_meta: Dict[str, Any] = {
                "submitted": submitted,
                "exit_reason": exit_reason,
                "ablation_submission_integrity": ablation_integrity,
            }
            if ablation_integrity:
                row["ablation_submission_integrity"] = ablation_integrity
                if ablation_integrity.get("status") == "VALID_ABLATION_TERMINAL_EXECUTION":
                    submission_integrity = {
                        "valid": True,
                        "status": "VALID_ABLATION_TERMINAL_EXECUTION",
                        "errors": [],
                    }
                    row["submission_integrity"] = submission_integrity
                    patch_meta["submission_integrity"] = submission_integrity
            _patch_run_meta(result.workdir, **patch_meta)
        except ImportError:
            pass
        return row
    except Exception as exc:  # noqa: BLE001
        import traceback

        from hazardweaver.hwa.experiments.headline_cell_rerun_hygiene_v1 import seal_headline_cell_incomplete

        err = f"{type(exc).__name__}: {exc}"
        seal_headline_cell_incomplete(cell.workdir, exit_reason="exception", error=err)
        return {
            "flat_index": cell.flat_index,
            "instance_id": cell.inventory_row.get("instance_id"),
            "opaque_task_id": cell.opaque_task_id,
            "condition": cell.condition,
            "seed": cell.seed,
            "workdir": str(cell.workdir),
            "status": "error",
            "submitted": False,
            "exit_reason": "exception",
            "error": err,
            "traceback": traceback.format_exc(limit=8),
        }


def build_manifest_from_out_root(
    out_root: Path,
    *,
    inventory_path: Path,
    arm: str = "llama8b",
    seed: int = DEFAULT_SEED,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []
    root = Path(out_root)
    for seed_dir in sorted(root.glob("seed_*")):
        for cond_dir in seed_dir.iterdir():
            if not cond_dir.is_dir():
                continue
            condition = cond_dir.name
            for task_dir in sorted(cond_dir.iterdir()):
                if not task_dir.is_dir():
                    continue
                inv_path = task_dir / "inventory_row.json"
                if not inv_path.is_file():
                    continue
                inv_row = json.loads(inv_path.read_text(encoding="utf-8"))
                meta_path = task_dir / "run_meta.json"
                meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
                dca_path = task_dir / "dca_result.json"
                dca_valid = None
                if dca_path.is_file():
                    try:
                        dca_valid = bool(json.loads(dca_path.read_text(encoding="utf-8")).get("valid"))
                    except json.JSONDecodeError:
                        dca_valid = None
                rows.append(
                    {
                        "instance_id": inv_row.get("instance_id"),
                        "taskpack_id": inv_row.get("taskpack_id"),
                        "scenario_id": inv_row.get("scenario_id"),
                        "track": inv_row.get("track"),
                        "difficulty_tier": inv_row.get("difficulty_tier"),
                        "condition": condition,
                        "seed": int(seed_dir.name.replace("seed_", "") or seed),
                        "opaque_task_id": task_dir.name,
                        "workdir": str(task_dir),
                        "submitted": bool(meta.get("submitted")),
                        "exit_reason": meta.get("exit_reason"),
                        "model_id": meta.get("model_id"),
                        "dca_valid": dca_valid,
                    }
                )

    from collections import Counter

    dca_valid_n = sum(1 for r in rows if r.get("dca_valid") is True)
    by_condition: Counter[str] = Counter()
    by_track: Counter[str] = Counter()
    by_tier: Counter[str] = Counter()
    solve_by_cond: Counter[str] = Counter()
    for r in rows:
        cond = str(r.get("condition") or "?")
        if r.get("dca_valid") is True:
            by_condition[cond] += 1
            by_track[str(r.get("track") or "?")] += 1
            by_tier[str(r.get("difficulty_tier") or "?")] += 1
        ans = Path(str(r.get("workdir") or "")) / "answer.json"
        if ans.is_file():
            try:
                act = json.loads(ans.read_text(encoding="utf-8")).get("answer", {}).get("action")
                if act == "solve":
                    solve_by_cond[cond] += 1
            except (json.JSONDecodeError, AttributeError):
                pass

    manifest = {
        "schema_version": "HWB_HEADLINE_HWA_MANIFEST_v1",
        "experiment_id": "headline_inventory_runner_v1",
        "arm": arm,
        "inventory_path": str(inventory_path),
        "out_root": str(root),
        "n_tasks": len(rows),
        "n_submitted": sum(1 for r in rows if r.get("submitted")),
        "n_dca_valid": dca_valid_n,
        "dca_valid_rate": (dca_valid_n / len(rows)) if rows else 0.0,
        "by_condition_dca_valid": dict(by_condition),
        "by_track_dca_valid": dict(by_track),
        "by_tier_dca_valid": dict(by_tier),
        "by_condition_solve_action": dict(solve_by_cond),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "tasks": rows,
    }
    man_path = root / "manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    iclr_manifest = root / "hwa_manifest_v1.jsonl"
    with iclr_manifest.open("w", encoding="utf-8") as fh:
        for rec in rows:
            fh.write(
                json.dumps(
                    {
                        "instance_id": rec.get("instance_id"),
                        "workdir": rec.get("workdir"),
                        "condition": rec.get("condition"),
                        "seed": rec.get("seed"),
                        "taskpack_id": rec.get("taskpack_id"),
                        "scenario_id": rec.get("scenario_id"),
                        "track": rec.get("track"),
                        "difficulty_tier": rec.get("difficulty_tier"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    manifest["manifest_path"] = str(man_path)
    manifest["iclr_hwa_manifest_path"] = str(iclr_manifest)
    try:
        from scripts.benchmark.report_pred_usability_v1 import report_out_root

        pred = report_out_root(root)
        manifest["pred_usability_summary_path"] = str(root / "pred_usability_summary.json")
        manifest["pred_usability"] = pred["summary"]["overall"]
    except Exception as exc:  # noqa: BLE001
        manifest["pred_usability_error"] = f"{type(exc).__name__}: {exc}"
    return manifest
