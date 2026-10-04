"""Manual ablation pilot v1 — paper-grounded, non-degenerate HKC/HCG cells.

Design rules (binding):
- full arm: witness route admissible (n_admissible >= 1 on witness).
- HKC: task ``require:`` from PI gold / docs/paper; witness route grounds it; decoy does not.
  ``hkc_off`` permissive → decoy also admissible → wrong route possible.
- HCG: witness route typed-compatible; decoy route carries near-miss contract (e.g. units=ft).
  ``hcg_untyped`` admits decoy → wrong route possible.
- Gate ``require:`` constraints live in metadata only — never in solver_visible (agent leak).
- Decoy routes get a modest REMSA utility boost when admissible (not 100% wrong on ablation).
- No ``force_inapplicable`` / no task-wide near-miss on witness.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Mapping, MutableMapping, Optional, Set

from hazardweaver.hwa.experiments.inventory_mechanism_overlay_v1 import hcg_baseline_support
from hazardweaver.hwa.experiments.headline_ablation_modes_v1 import HCG_COMPAT_FULL, HCG_COMPAT_UNTYPED
from hazardweaver.hwa.runtime.session_state import SessionState
from hazardweaver.hwa.scientific_controller.admissibility import evaluate_A_sci, is_admissible
from hazardweaver.hwa.scientific_controller.hkc_frozen_registry_v1 import (
    HKC_REGISTRY_K_FROZEN,
    HKC_REGISTRY_NONE,
    hkc_off_permissive_a_sci,
    hkc_registry_disabled,
    reset_frozen_hkc_registry_cache,
)

_DEFAULT_DECOY_UTILITY_BOOST = 0.12
_DEFAULT_WITNESS_UTILITY_BASE = 0.83
_MAX_DECOY_UTILITY_BOOST = 0.14


def ablation_manual_pilot_enabled(task: Mapping[str, Any]) -> bool:
    meta = task.get("metadata") or {}
    return str(meta.get("ablation_manual_pilot") or "") == "v1"


def unified_runtime_allowed_edge_ids(inventory_row: Mapping[str, Any]) -> List[str]:
    """Optional runtime narrow list for unified rows (W3 defer+s0 shock only).

    Does **not** pin to DCA scenario gold — inventory ``allowed_edge_ids`` stay authoritative.
    """
    if not inventory_row.get("unified_benchmark_v1"):
        return []
    rq4 = dict(inventory_row.get("rq4_intervention") or {})
    if rq4.get("defer_s0_narrow_allowed_to_shock"):
        shock = str(
            rq4.get("force_capability_failure")
            or inventory_row.get("w3_shock_capability_id")
            or ""
        ).strip()
        if shock:
            return [shock]
    return []


def ablation_wf3_agent_replay_enabled(task: Optional[Mapping[str, Any]] = None) -> bool:
    """Agent/benchmark WF-3 must use pinned CARP replay — never 12-fold online training."""
    if str(os.environ.get("HWA_WF3_AGENT_REPLAY", "")).strip().lower() in {"1", "true", "yes", "on"}:
        return True
    if task is not None and ablation_manual_pilot_enabled(task):
        return True
    meta = (task or {}).get("metadata") or {}
    return bool(meta.get("hwb_headline_inventory"))


def ablation_hwmed_agent_replay_enabled(task: Optional[Mapping[str, Any]] = None) -> bool:
    """Agent/benchmark HW-MED must use EWB eval_subset replay — never live CIRA icechunk."""
    if str(os.environ.get("HWA_HWMED_AGENT_REPLAY", "")).strip().lower() in {"1", "true", "yes", "on"}:
        return True
    if task is not None and ablation_manual_pilot_enabled(task):
        return True
    meta = (task or {}).get("metadata") or {}
    if meta.get("hwb_headline_inventory"):
        return True
    from hazardweaver.hwa.experiments.unified_benchmark_execution_v1 import unified_benchmark_enabled

    return unified_benchmark_enabled({"metadata": meta})


def wf3_pinned_scientific_dir(capability_id: str) -> Optional[Path]:
    """Pinned native_metrics / holdout predictions under runs/carp/scientific."""
    cap = str(capability_id or "").strip()
    if not cap.startswith("CAP-WF3-"):
        return None
    root = Path("runs/carp/scientific/WF-3") / cap
    if (root / "native_metrics.json").is_file():
        return root
    holdout = root / "predictions" / "holdout"
    if holdout.is_dir() and any(holdout.glob("*.npz")):
        return root
    return None


def _clamp_decoy_boost(raw: Any) -> float:
    try:
        val = float(raw)
    except (TypeError, ValueError):
        val = _DEFAULT_DECOY_UTILITY_BOOST
    return min(max(val, 0.0), _MAX_DECOY_UTILITY_BOOST)


def _fl2_split_for_row(row: Mapping[str, Any]) -> Optional[str]:
    """Resolve FL-2 split from inventory row or taskpack refs sidecar."""
    explicit = str(row.get("split") or "").strip()
    if explicit:
        return explicit
    if str(row.get("track") or "") != "FL-2":
        return None
    scenario_id = str(row.get("scenario_id") or "").strip()
    if not scenario_id:
        return None
    try:
        from hazardweaver.hwb.registry.fl2_parametric_resolver import load_scenario_refs

        refs = load_scenario_refs(str(row.get("taskpack_id") or ""))
        scen = (refs.get("scenarios") or {}).get(scenario_id) or {}
        prov = scen.get("provenance") or {}
        split = str(prov.get("split") or "").strip()
        if split:
            return split
    except Exception:  # noqa: BLE001
        pass
    if scenario_id.startswith("mozambique_2019"):
        return "hwb_holdout"
    if scenario_id.startswith("pakistan_2022"):
        return "official_test"
    return None


def ablation_env_skip_reason(row: Mapping[str, Any]) -> Optional[str]:
    """Return skip reason when scenario assets are unavailable ."""
    explicit = str(row.get("ablation_env_skip_reason") or "").strip()
    if explicit:
        return explicit
    track = str(row.get("track") or "").strip()
    scenario_id = str(row.get("scenario_id") or "").strip()
    mech = str(row.get("ablation_mechanism") or "").lower()
    witness = str(row.get("ablation_witness_capability_id") or "").strip()

    if scenario_id == "pakistan_2022" and witness == "CAP-FL2-01":
        return "FL2_pakistan_CAP-FL2-01_TASK_MISMATCH"

    if track != "FL-2" or not scenario_id:
        return None
    split = _fl2_split_for_row(row)
    if not split:
        return None
    try:
        from hazardweaver.hcg.carp.scientific.fl2_data import scenario_dir
        from hazardweaver.hcg.carp.scientific.fl2_zenodo_relevant_inputs import (
            scenario_input_files_ready,
        )

        if scenario_input_files_ready(split, scenario_id):
            return None
        root = scenario_dir(split, scenario_id)
        return f"FL-2 scenario inputs missing ({root}/dem.npy)"
    except Exception:  # noqa: BLE001
        return f"FL-2 env check failed for {scenario_id} ({split})"


def _strip_agent_visible_requires(task: MutableMapping[str, Any]) -> None:
    """Remove require: constraints from solver_visible — gate-only, not agent hints."""
    sv = task.get("solver_visible") or {}
    inputs = sv.get("inputs") or {}
    raw = list(inputs.get("constraints") or [])
    if not raw:
        return
    kept = [c for c in raw if not str(c).startswith("require:")]
    if len(kept) != len(raw):
        inputs = dict(inputs)
        if kept:
            inputs["constraints"] = kept
        else:
            inputs.pop("constraints", None)
        sv = dict(sv)
        sv["inputs"] = inputs
        task["solver_visible"] = sv


def _gate_task_requires(task: Mapping[str, Any]) -> List[str]:
    meta = task.get("metadata") or {}
    return [str(r) for r in (meta.get("ablation_hkc_gate_requires") or []) if str(r).startswith("require:")]


def _task_for_hkc_gate_eval(task: Mapping[str, Any]) -> Dict[str, Any]:
    """Internal gate view with require: constraints — not shown to the agent."""
    gate_reqs = _gate_task_requires(task)
    if not gate_reqs:
        return dict(task)
    gate_task = dict(task)
    sv = dict(gate_task.get("solver_visible") or {})
    inputs = dict(sv.get("inputs") or {})
    inputs["constraints"] = list(gate_reqs)
    sv["inputs"] = inputs
    gate_task["solver_visible"] = sv
    return gate_task


def _decoy_capability_ids(meta: Mapping[str, Any]) -> Set[str]:
    decoys: Set[str] = set()
    primary = str(meta.get("ablation_decoy_capability_id") or "").strip()
    if primary:
        decoys.add(primary)
    hcg = meta.get("ablation_hcg_decoy_route_contracts") or {}
    if isinstance(hcg, Mapping):
        decoys.update(str(k) for k in hcg.keys())
    extra = meta.get("ablation_hcg_decoy_capability_id")
    if extra:
        decoys.add(str(extra))
    return decoys


def _apply_decoy_presentation_bias(route: Dict[str, Any], task: Mapping[str, Any]) -> None:
    """Bias REMSA toward decoy when multiple routes are admissible (legacy ablation arms only).

    Unified @143 / W3 headline rows use symmetric curator REMSA (_unified_scenario_route_bonus);
    decoy utility inflation is disabled there (DL-214).
    """
    meta = task.get("metadata") or {}
    if meta.get("unified_benchmark_v1"):
        return
    witness = str(meta.get("ablation_witness_capability_id") or "")
    cap = str((route.get("capability_ids") or route.get("edges") or ["?"])[0])
    boost = _clamp_decoy_boost(meta.get("ablation_decoy_utility_boost"))
    decoys = _decoy_capability_ids(meta)
    if cap in decoys and cap != witness:
        route["validation_utility"] = _DEFAULT_WITNESS_UTILITY_BASE + boost
    elif cap == witness:
        route["validation_utility"] = _DEFAULT_WITNESS_UTILITY_BASE


def apply_ablation_manual_pilot_overlays(
    task: MutableMapping[str, Any],
    inventory_row: Mapping[str, Any],
) -> MutableMapping[str, Any]:
    """Stamp manual-pilot metadata and per-route overlays from inventory row."""
    meta = task.setdefault("metadata", {})
    meta["ablation_manual_pilot"] = "v1"
    meta["hwb_headline_inventory"] = True

    curated_row: Mapping[str, Any] = inventory_row
    if inventory_row.get("unified_benchmark_v1"):
        from hazardweaver.hwa.benchmark.unified_inventory_curator_v1 import curate_unified_inventory_row

        curated_row = curate_unified_inventory_row(inventory_row)
        gold = str(curated_row.get("unified_scenario_gold_capability_id") or "").strip()
        if gold:
            meta["unified_scenario_gold_capability_id"] = gold

    if inventory_row.get("hkc_task_requires"):
        meta["ablation_hkc_gate_requires"] = list(inventory_row.get("hkc_task_requires") or [])

    hkc_routes = curated_row.get("hkc_route_bindings")
    if isinstance(hkc_routes, Mapping):
        meta["ablation_hkc_route_bindings"] = dict(hkc_routes)

    hcg_decoys = inventory_row.get("hcg_decoy_route_contracts")
    if isinstance(hcg_decoys, Mapping):
        meta["ablation_hcg_decoy_route_contracts"] = dict(hcg_decoys)

    if inventory_row.get("hcg_typed_probe") or hcg_decoys:
        meta["hcg_typed_probe"] = True
        meta["hcg_available_support"] = dict(
            inventory_row.get("hcg_available_support") or hcg_baseline_support()
        )

    for key in (
        "ablation_arm",
        "ablation_mechanism",
        "ablation_witness_capability_id",
        "ablation_decoy_capability_id",
        "ablation_hcg_decoy_capability_id",
        "ablation_paper_ref",
        "ablation_design_note",
        "ablation_decoy_utility_boost",
        "ablation_agent_max_steps",
        "ablation_dca_tolerance",
        "ablation_gate_policy_override",
        "ablation_mechanism_primary",
    ):
        if inventory_row.get(key) is not None:
            meta[key] = inventory_row[key]

    track = str(inventory_row.get("track") or "")
    if track == "FL-2" and not meta.get("ablation_gate_policy_override"):
        meta["ablation_gate_policy_override"] = {
            "primary_metric": "rmse_depth",
            "rmse_mode": "primary",
            "csi_mode": "advisory_record_only",
            "note": "ablation pilot: mechanism routing primary; RMSE advisory (DL-067 override)",
        }
        meta["ablation_mechanism_primary"] = True

    from hazardweaver.hwa.experiments.ablation_manual_pilot_execution_v1 import (
        bind_ablation_inventory_metadata,
        patch_goal_execution_handles,
        resolve_ablation_execution_split,
    )

    bind_ablation_inventory_metadata(meta, inventory_row)

    unified_goal = bool(inventory_row.get("unified_benchmark_v1"))
    goal = str(
        inventory_row.get("user_goal")
        or inventory_row.get("ablation_user_goal")
        or task.get("user_facing_goal")
        or ""
    ).strip()
    scenario_id = str(inventory_row.get("scenario_id") or meta.get("scenario_id") or "").strip()
    split = resolve_ablation_execution_split(inventory_row)
    if goal and scenario_id and split:
        goal = patch_goal_execution_handles(goal, scenario_id=scenario_id, split=split)
    if goal:
        mechanism = str(inventory_row.get("ablation_mechanism") or "").strip()
        if unified_goal:
            goal = (
                f"{goal} Review the symmetric route briefs before commit. "
                "Abstention is forbidden on solve-mandatory cells when admissible routes exist: "
                "commit one route, run_capability, then submit_solution."
            )
        else:
            if mechanism in {"hkc", "hcg", "dual"}:
                goal = (
                    f"{goal} Each whitelisted route is operationally plausible; "
                    "evaluate route briefs symmetrically before commit — do not default to a witness route."
                )
            goal += (
                " Abstention is forbidden on this solve-mandatory cell when admissible routes exist: "
                "commit one route, run_capability with the stated handles, then submit_solution."
            )
        task["user_facing_goal"] = goal

    briefs = curated_row.get("solver_visible_route_briefs") or curated_row.get("ablation_route_briefs")
    if isinstance(briefs, Mapping) and briefs:
        inp = task.setdefault("solver_visible", {}).setdefault("inputs", {})
        inp["route_briefs"] = dict(briefs)

    inv_allowed = [str(e) for e in (curated_row.get("allowed_edge_ids") or []) if str(e).strip()]
    w3_narrow = unified_runtime_allowed_edge_ids(inventory_row)
    inp = task.setdefault("solver_visible", {}).setdefault("inputs", {})
    if w3_narrow:
        inp["allowed_edge_ids"] = w3_narrow
    elif inv_allowed:
        inp["allowed_edge_ids"] = inv_allowed

    if str(inventory_row.get("scenario_id") or "").strip():
        inp = task.setdefault("solver_visible", {}).setdefault("inputs", {})
        inp["scenario_id"] = str(inventory_row["scenario_id"])
    fl2_split = _fl2_split_for_row(inventory_row)
    if fl2_split:
        inp = task.setdefault("solver_visible", {}).setdefault("inputs", {})
        inp["split"] = fl2_split

    _strip_agent_visible_requires(task)
    if inventory_row.get("ablation_decoy_utility_boost") is not None:
        meta["ablation_decoy_utility_boost"] = _clamp_decoy_boost(
            inventory_row.get("ablation_decoy_utility_boost")
        )

    if str(inventory_row.get("expected_action") or "solve").strip().lower() == "solve":
        meta["react_clarify_forbidden"] = True
        meta["ablation_require_run_capability"] = True

    if inventory_row.get("rq4_intervention"):
        from hazardweaver.hwa.experiments.rq4_route_intervention_v1 import apply_rq4_intervention_from_row

        task = apply_rq4_intervention_from_row(task, inventory_row)

    return task


def _apply_route_overlays(
    routes: List[Dict[str, Any]],
    task: Mapping[str, Any],
) -> List[Dict[str, Any]]:
    meta = task.get("metadata") or {}
    hkc_bindings = meta.get("ablation_hkc_route_bindings") or {}
    hcg_decoys = meta.get("ablation_hcg_decoy_route_contracts") or {}
    unified_gold = str(meta.get("unified_scenario_gold_capability_id") or "").strip()
    out: List[Dict[str, Any]] = []
    for route in routes:
        r = dict(route)
        cap = str((r.get("capability_ids") or r.get("edges") or ["?"])[0])
        bind = hkc_bindings.get(cap)
        if isinstance(bind, Mapping):
            r.update({k: v for k, v in bind.items() if v is not None})
        decoy = hcg_decoys.get(cap)
        if isinstance(decoy, Mapping) and not (meta.get("unified_benchmark_v1") and cap == unified_gold):
            r["input_contract"] = dict(decoy)
        out.append(r)
    return out


def annotate_manual_pilot_route(
    route: Mapping[str, Any],
    task: Mapping[str, Any],
    state: SessionState,
) -> Dict[str, Any]:
    """HKC via gate-only task-require grounding; HCG via standard gate typed/untyped."""
    from hazardweaver.hwa.scientific_controller.reason_codes import ASciVerdict

    out = dict(route)
    meta = task.get("metadata") or {}
    cap = str((out.get("capability_ids") or out.get("edges") or ["?"])[0])
    hcg_decoys = meta.get("ablation_hcg_decoy_route_contracts") or {}
    witness_cap = str(meta.get("ablation_witness_capability_id") or "")
    unified_gold = str(meta.get("unified_scenario_gold_capability_id") or "").strip()
    if meta.get("unified_benchmark_v1") and unified_gold and cap == unified_gold:
        from hazardweaver.hwa.scientific_controller.reason_codes import ASciVerdict

        out["A_sci"] = {
            "verdict": ASciVerdict.APPLICABLE.value,
            "codes": [],
            "refs": [],
            "source": "unified_scenario_gold_curator_v1",
        }
        out["hkc_binding_source"] = "unified_scenario_gold_curator_v1"
        from hazardweaver.hwa.route_controller.admissibility_gate import _annotate_g6_a_cap

        out = _annotate_g6_a_cap(out, task, state, graph=None)
        out["admissible"] = is_admissible(out.get("A_sci") or {}, out.get("A_cap") or {})
        _apply_decoy_presentation_bias(out, task)
        return out
    has_task_require = bool(_gate_task_requires(task))
    if hkc_registry_disabled():
        out["A_sci"] = hkc_off_permissive_a_sci()
    elif cap in hcg_decoys and cap != witness_cap:
        out["A_sci"] = {
            "verdict": ASciVerdict.APPLICABLE.value,
            "codes": [],
            "refs": [],
            "source": "ablation_manual_pilot_hcg_decoy",
        }
    elif not has_task_require:
        out["A_sci"] = {
            "verdict": ASciVerdict.APPLICABLE.value,
            "codes": [],
            "refs": [],
            "source": "ablation_manual_pilot_hcg_only",
        }
    else:
        out["A_sci"] = evaluate_A_sci(out, _task_for_hkc_gate_eval(task), theory_arm="verified")
    from hazardweaver.hwa.experiments.unified_benchmark_execution_v1 import (
        apply_unified_g6_operational_admissibility,
    )

    apply_unified_g6_operational_admissibility(task, out)
    out["hkc_binding_source"] = "ablation_manual_pilot_v1"

    from hazardweaver.hwa.route_controller.admissibility_gate import _annotate_g6_a_cap

    out = _annotate_g6_a_cap(out, task, state, graph=None)
    out["admissible"] = is_admissible(out.get("A_sci") or {}, out.get("A_cap") or {})
    _apply_decoy_presentation_bias(out, task)
    try:
        from hazardweaver.hwa.experiments.headline_ablation_modes_v1 import hcg_untyped

        if hcg_untyped():
            out["hcg_compat_mode"] = "untyped"
    except ImportError:
        pass
    return out


def simulate_remsa_top_capability(
    inventory_row: Mapping[str, Any],
    *,
    task: Mapping[str, Any],
    routes: List[Mapping[str, Any]],
    hkc_registry: str,
    hcg_mode: str,
) -> Optional[str]:
    """predict REMSA rank-1 cap under arm env (agent-agnostic)."""
    import os

    from hazardweaver.hwa.route_controller.remsa_rank import remsa_top_route

    reset_frozen_hkc_registry_cache()
    os.environ["HWA_HKC_REGISTRY"] = hkc_registry
    os.environ["HWA_HCG_COMPAT_MODE"] = hcg_mode
    state = SessionState()
    overlaid = _apply_route_overlays([dict(r) for r in routes], task)
    ann = [annotate_manual_pilot_route(r, task, state) for r in overlaid]
    top = remsa_top_route(ann, task=task, state=state)
    if not top:
        return None
    caps = top.get("capability_ids") or top.get("edges") or []
    return str(caps[0]) if caps else None


def preflight_manual_pilot_cell(
    inventory_row: Mapping[str, Any],
    *,
    task: Mapping[str, Any],
    routes: List[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Validate non-degenerate full vs ablation discrimination for one manual cell."""
    import os

    witness = str(inventory_row.get("ablation_witness_capability_id") or "")
    mechanism = str(inventory_row.get("ablation_mechanism") or "")
    routes = _apply_route_overlays([dict(r) for r in routes], task)

    def _count_adm(hkc: str, hcg: str) -> Dict[str, Any]:
        reset_frozen_hkc_registry_cache()
        os.environ["HWA_HKC_REGISTRY"] = hkc
        os.environ["HWA_HCG_COMPAT_MODE"] = hcg
        state = SessionState()
        ann = [annotate_manual_pilot_route(r, task, state) for r in routes]
        by_cap = {
            str((a.get("capability_ids") or ["?"])[0]): bool(a.get("admissible"))
            for a in ann
        }
        top_cap = simulate_remsa_top_capability(
            inventory_row,
            task=task,
            routes=routes,
            hkc_registry=hkc,
            hcg_mode=hcg,
        )
        return {
            "n_admissible": sum(1 for a in ann if a.get("admissible")),
            "witness_admissible": by_cap.get(witness),
            "by_capability": by_cap,
            "remsa_top_capability": top_cap,
            "remsa_top_is_witness": top_cap == witness if top_cap else None,
        }

    full = _count_adm(HKC_REGISTRY_K_FROZEN, HCG_COMPAT_FULL)
    hkc_off = _count_adm(HKC_REGISTRY_NONE, HCG_COMPAT_FULL)
    hcg_unt = _count_adm(HKC_REGISTRY_K_FROZEN, HCG_COMPAT_UNTYPED)

    hkc_ok = (
        mechanism in {"hkc", "dual"}
        and full.get("witness_admissible")
        and full.get("n_admissible", 0) >= 1
        and full.get("by_capability") != hkc_off.get("by_capability")
        and full.get("remsa_top_is_witness") is True
        and hkc_off.get("remsa_top_is_witness") is False
    )
    hcg_ok = (
        mechanism in {"hcg", "dual"}
        and full.get("witness_admissible")
        and full.get("n_admissible", 0) >= 1
        and full.get("by_capability") != hcg_unt.get("by_capability")
        and full.get("remsa_top_is_witness") is True
        and hcg_unt.get("remsa_top_is_witness") is False
    )
    passed = hkc_ok if mechanism == "hkc" else hcg_ok if mechanism == "hcg" else (hkc_ok and hcg_ok)

    return {
        "pass": passed,
        "mechanism": mechanism,
        "witness": witness,
        "arms": {"full": full, "hkc_off": hkc_off, "hcg_untyped": hcg_unt},
        "hkc_discriminates": hkc_ok,
        "hcg_discriminates": hcg_ok,
    }
