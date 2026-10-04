"""Recompute dca_result.json with unified eligibility + mechanism veto."""

from __future__ import annotations

import json
from pathlib import Path

from hazardweaver.hwa.benchmark.dca_episode_eligibility_v1 import incomplete_episode_dca_dict, is_dca_eligible
from hazardweaver.hwa.benchmark.dca_mechanism_unify_v1 import apply_mechanism_veto_to_dca
from hazardweaver.hwa.benchmark.unified_dca_route_gate_v1 import (
    apply_unified_route_gate_to_dca,
    infer_dca_route_condition,
)


def rescore_workdir_dca(
    workdir: Path,
    *,
    out_root_hint: str = "",
    inventory_row: dict | None = None,
) -> bool:
    """Return True if dca_result.json changed.

    Prefer the sealed submit inventory row when provided; workdir
    ``inventory_row.json`` can drift and yields inconsistent DCA vs replay.
    """
    dca_path = workdir / "dca_result.json"
    if not dca_path.is_file():
        return False
    old = json.loads(dca_path.read_text(encoding="utf-8"))
    inv_path = workdir / "inventory_row.json"
    inv_row = dict(inventory_row or {})
    if not inv_row and inv_path.is_file():
        inv_row = json.loads(inv_path.read_text(encoding="utf-8"))
    meta_path = workdir / "run_meta.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}

    if not is_dca_eligible(workdir):
        new = incomplete_episode_dca_dict(
            taskpack_id=str(inv_row.get("taskpack_id") or old.get("taskpack_id") or ""),
            exit_reason=str(meta.get("exit_reason") or ""),
        )
    else:
        try:
            from hazardweaver.hwb.bridge.hwa_workdir import load_hwa_submission
            from hazardweaver.hwb.evaluators.dca_scorer import evaluate_dca_submission
            from hazardweaver.hwb.run.eval_dca_submission_v1 import resolve_taskpack_for_inventory_row

            tp = resolve_taskpack_for_inventory_row(inv_row)
            sub = load_hwa_submission(workdir)
            tier = str(inv_row.get("difficulty_tier") or "L1")
            dca = evaluate_dca_submission(
                tp,
                sub,
                agent_id="hazardweaver",
                difficulty_tier=tier,
                inventory_row=inv_row,
                workdir=workdir,
            )
            new = apply_mechanism_veto_to_dca(dca.to_dict(), workdir)
        except Exception:
            new = apply_mechanism_veto_to_dca(old, workdir)

    route_cond = infer_dca_route_condition(workdir, out_root_hint=out_root_hint)
    new = apply_unified_route_gate_to_dca(
        new,
        workdir,
        inventory_row=inv_row,
        condition=route_cond,
    )
    if isinstance(new, dict):
        new["dca_tolerance_contract"] = "v2"
        new["dca_route_condition"] = route_cond

    if new == old:
        return False
    dca_path.write_text(json.dumps(new, indent=2) + "\n", encoding="utf-8")
    return True
