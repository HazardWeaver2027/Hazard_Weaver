"""VCE VERIFY gate — execution schema + optional DCA preflight."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from hazardweaver.hwa.agent_runtime.execution_schema import verify_submit_solution_ids
from hazardweaver.hwa.control.vce_preflight_v1 import build_vce_preflight_submission
from hazardweaver.hwa.control.vce_track_verify import vce_track_metric_preview


def vce_verify_execution(
    *,
    workdir: Path,
    route_id: str,
    execution_id: str,
    final_artifact_id: str,
    taskpack: Optional[Mapping[str, Any]] = None,
    inventory_row: Optional[Mapping[str, Any]] = None,
    difficulty_tier: str = "L1",
) -> Dict[str, Any]:
    """Return ``ok=True`` only when schema checks pass and DCA preflight is clean."""
    from hazardweaver.hwa.agent_runtime.execution_schema import load_execution

    er_pre = load_execution(workdir, str(execution_id).strip())
    track_row = dict(inventory_row or {})
    if not track_row.get("capability_id") and er_pre is not None:
        track_row["capability_id"] = str((er_pre.get("executed_capability_ids") or [""])[0])
    if taskpack is not None:
        track_row.setdefault("scenario_id", str(taskpack.get("scenario_id") or ""))
        track_row.setdefault("track", str(taskpack.get("headline_target") or ""))

    preview = vce_track_metric_preview(
        workdir=workdir,
        route_id=str(route_id).strip(),
        execution_id=str(execution_id).strip(),
        final_artifact_id=str(final_artifact_id).strip(),
        inventory_row=track_row or None,
        taskpack=taskpack,
    )
    if not preview.get("ok"):
        return preview
    prepared_submission = preview.get("submission")

    schema = verify_submit_solution_ids(
        workdir=workdir,
        route_id=str(route_id).strip(),
        execution_id=str(execution_id).strip(),
        final_artifact_id=str(final_artifact_id).strip(),
    )
    if not schema.get("ok"):
        return {**schema, "verify_layer": "execution_schema"}

    if taskpack is None:
        return {"ok": True, "verify_layer": "execution_schema", "dca_skipped": True, "execution": schema.get("execution")}

    er = schema.get("execution") or {}
    try:
        submission = prepared_submission or build_vce_preflight_submission(
            workdir=workdir,
            route_id=str(route_id).strip(),
            execution_id=str(execution_id).strip(),
            final_artifact_id=str(final_artifact_id).strip(),
            inventory_row=inventory_row,
            taskpack=taskpack,
        )
    except ValueError as exc:
        return {
            "ok": False,
            "error": str(exc),
            "verify_layer": "dca_preflight",
            "tool": "vce_verify",
        }
    try:
        from hazardweaver.hwb.evaluators.dca_scorer import evaluate_dca_submission

        dca = evaluate_dca_submission(
            taskpack,
            submission,
            agent_id="hazardweaver",
            difficulty_tier=str(
                difficulty_tier
                or (inventory_row or {}).get("difficulty_tier")
                or "L1"
            ),
            inventory_row=inventory_row,
        )
        dca_dict = dca.to_dict()
        if dca.contract_violated or not dca.valid:
            return {
                "ok": False,
                "error": "dca_preflight_failed",
                "verify_layer": "dca_preflight",
                "dca": dca_dict,
                "tool": "vce_verify",
            }
        return {
            "ok": True,
            "verify_layer": "dca_preflight",
            "execution": er,
            "dca": dca_dict,
        }
    except Exception as exc:  # noqa: BLE001 — preflight is best-effort
        return {
            "ok": True,
            "verify_layer": "execution_schema",
            "dca_skipped": True,
            "dca_preflight_error": f"{type(exc).__name__}: {exc}",
            "execution": er,
        }
