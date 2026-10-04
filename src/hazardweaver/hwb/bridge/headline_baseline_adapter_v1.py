"""Neutral headline task materialization for ICLR baselines (no HWA controller shortcuts)."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from hazardweaver.hwb.registry.hcg_solver_route_contracts_v1 import (
    build_allowed_capability_catalog,
    lookup_hcg_solver_route_contract,
    merge_solver_route_contracts,
)
from hazardweaver.hwb.registry.headline_route_contracts_v1 import lookup_route_contract
from hazardweaver.hwb.registry.solver_allowed_edges_v1 import resolve_solver_allowed_edge_ids
from hazardweaver.hwb.registry.track_parametric_resolver import load_scenario_refs

ROOT = Path(__file__).resolve().parents[3]

_SCORE_SUBMISSION_TEMPLATE = '''#!/usr/bin/env python3
"""Score baseline submission against headline DCA (thin CLI wrapper)."""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__PROJECT_ROOT__)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from hazardweaver.hwb.evaluators.dca_scorer import evaluate_dca_submission


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--trajectory", type=Path, default=Path("trajectory.json"))
    p.add_argument("--agent-id", default="baseline")
    p.add_argument("--output", type=Path, default=Path("dca_result.json"))
    args = p.parse_args()
    run_dir = Path(__file__).resolve().parent
    tp = json.loads((run_dir / "resolved_taskpack.json").read_text(encoding="utf-8"))
    inv = json.loads((run_dir / "inventory_row.json").read_text(encoding="utf-8"))
    traj = json.loads(args.trajectory.read_text(encoding="utf-8"))
    tier = str(inv.get("difficulty_tier") or "L1")
    result = evaluate_dca_submission(tp, traj, agent_id=args.agent_id, difficulty_tier=tier)
    payload = result.to_dict()
    args.output.write_text(json.dumps(payload, indent=2) + "\\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
'''


def _route_metadata(edge_id: str, taskpack: Mapping[str, Any]) -> Dict[str, Any]:
    solver = taskpack.get("solver_view") or {}
    contracts = solver.get("contracts") or {}
    meta = {"edge_id": edge_id, "kind": "capability" if edge_id.startswith("CAP-") else "schema"}
    contract = merge_solver_route_contracts(
        lookup_hcg_solver_route_contract(edge_id),
        lookup_route_contract(edge_id),
        contracts.get(edge_id) or {},
    )
    if contract:
        meta["contract"] = contract
        meta["summary"] = str(
            contract.get("summary")
            or contract.get("name")
            or contract.get("description")
            or edge_id
        )
    else:
        meta["summary"] = edge_id
    return meta


def _link_inputs(run_dir: Path, inventory_row: Mapping[str, Any], taskpack: Mapping[str, Any]) -> List[str]:
    inputs_dir = run_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    linked: List[str] = []
    taskpack_id = str(inventory_row.get("taskpack_id") or taskpack.get("taskpack_id") or "")
    scenario_id = str(
        inventory_row.get("scenario_id")
        or inventory_row.get("initial_state_id")
        or ""
    )
    base_sid = scenario_id.split("__", 1)[0] if "__" in scenario_id else scenario_id
    refs = load_scenario_refs(taskpack_id) if taskpack_id else {}
    sc = (refs.get("scenarios") or {}).get(base_sid) or {}
    hcg_dir = sc.get("hcg_scientific_dir") or (sc.get("provenance") or {}).get("hcg_scientific_dir")
    if hcg_dir:
        src = Path(str(hcg_dir))
        if src.is_dir():
            dest = inputs_dir / "hcg_scientific"
            if not dest.exists():
                dest.symlink_to(src.resolve(), target_is_directory=True)
            linked.append(str(dest))
    data_dir = (taskpack.get("solver_view") or {}).get("data_dir")
    if data_dir:
        src = Path(str(data_dir))
        if not src.is_absolute():
            src = ROOT / src
        if src.is_dir():
            dest = inputs_dir / "data"
            if not dest.exists():
                dest.symlink_to(src.resolve(), target_is_directory=True)
            linked.append(str(dest))
    return linked


def materialize_headline_baseline_task(
    inventory_row: Mapping[str, Any],
    taskpack: Mapping[str, Any],
    run_dir: Path,
) -> Dict[str, Any]:
    """Write neutral baseline-facing artifacts (task_desc, routes, inputs, scorer CLI)."""
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "inventory_row.json").write_text(
        json.dumps(dict(inventory_row), indent=2) + "\n",
        encoding="utf-8",
    )

    edges = resolve_solver_allowed_edge_ids(taskpack, inventory_row=inventory_row)
    routes = [_route_metadata(e, taskpack) for e in edges]
    contracts_by_edge = {
        str(r["edge_id"]): dict(r.get("contract") or {}) for r in routes if r.get("contract")
    }
    (run_dir / "routes.json").write_text(
        json.dumps(
            {
                "allowed_edge_ids": edges,
                "routes": routes,
                "catalog_schema": "hcg_solver_capability_catalog_v1",
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    catalog = build_allowed_capability_catalog(edges, contracts_by_edge=contracts_by_edge)
    catalog_path = run_dir / "capability_catalog.jsonl"
    with catalog_path.open("w", encoding="utf-8") as fh:
        for row in catalog:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    # V_q (check_v_cap) reads solver_view.allowed_edge_ids on the taskpack passed to
    # eval_baseline_run — must match routes.json or valid CAP picks fail invalid_capability.
    taskpack_out = dict(taskpack)
    solver_view = dict(taskpack_out.get("solver_view") or {})
    solver_view["allowed_edge_ids"] = list(edges)
    merged_contracts = dict(solver_view.get("contracts") or {})
    merged_contracts.update(contracts_by_edge)
    solver_view["contracts"] = merged_contracts
    taskpack_out["solver_view"] = solver_view
    (run_dir / "resolved_taskpack.json").write_text(
        json.dumps(taskpack_out, indent=2) + "\n",
        encoding="utf-8",
    )

    goal = str(
        inventory_row.get("user_goal")
        or taskpack.get("user_facing_goal")
        or taskpack.get("solver_view", {}).get("user_goal")
        or ""
    )
    tier = str(inventory_row.get("difficulty_tier") or "L1")
    track = str(inventory_row.get("track") or "")
    criteria = taskpack.get("success_criteria") or {}
    task_desc = f"""# HWB Headline Baseline Task (neutral adapter)

**Track:** {track} | **Tier:** {tier} | **Instance:** {inventory_row.get("instance_id")}

## Goal
{goal}

## Evaluation
Submit a trajectory scored by DCA (validity + outcome). Metric contracts are in routes.json metadata.
Do not reshuffle splits or access reference_view scores.

## Artifacts
- `routes.json` — authorized capability/schema edges with HCG solver-visible contracts (whitelist, not answer key)
- `capability_catalog.jsonl` — per-allowed-route capability cards (no benchmark scores)
- `inputs/` — read-only scientific inputs when available on disk
- `score_submission.py` — run after producing trajectory.json

## Success criteria (solver-visible)
{json.dumps(criteria, indent=2)}
"""
    (run_dir / "task_desc.md").write_text(task_desc, encoding="utf-8")
    score_path = run_dir / "score_submission.py"
    score_path.write_text(
        _SCORE_SUBMISSION_TEMPLATE.replace("__PROJECT_ROOT__", repr(str(ROOT.resolve()))),
        encoding="utf-8",
    )
    score_path.chmod(score_path.stat().st_mode | 0o111)

    linked = _link_inputs(run_dir, inventory_row, taskpack)
    return {
        "run_dir": str(run_dir),
        "n_routes": len(routes),
        "n_inputs_linked": len(linked),
        "task_desc": str(run_dir / "task_desc.md"),
        "routes_path": str(run_dir / "routes.json"),
        "capability_catalog_path": str(catalog_path),
        "n_catalog_rows": len(catalog),
    }


def select_baseline_l1l2_rows(
    inventory: List[Mapping[str, Any]],
    *,
    n_min: int = 30,
    n_max: int = 50,
) -> List[Dict[str, Any]]:
    """Pick L1/L2 solve rows for baseline matrix (reuse easy-50 scoring)."""
    from hazardweaver.hwb.run.select_headline_easy50_v1 import score_inventory_row, select_easy_rows

    l1l2 = [
        dict(r)
        for r in inventory
        if str(r.get("expected_action") or "solve") == "solve"
        and str(r.get("difficulty_tier") or "L1") in {"L1", "L2"}
    ]
    picked = select_easy_rows(l1l2, n=n_max, per_track_cap=10)
    if len(picked) < n_min:
        extra: List[Dict[str, Any]] = []
        seen = {str(r["instance_id"]) for r in picked}
        for row in l1l2:
            iid = str(row["instance_id"])
            if iid in seen:
                continue
            if score_inventory_row(row) is None:
                continue
            extra.append(dict(row))
            seen.add(iid)
            if len(picked) + len(extra) >= n_min:
                break
        picked = picked + extra
    return picked[:n_max]
