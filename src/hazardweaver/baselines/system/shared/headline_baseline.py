"""Headline neutral adapter helpers for system baselines (no reference_view leakage)."""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from shared.hwb_task_context import ensure_headline_task_desc, project_root

SCHEMA_VERSION = "HWA_TRAJECTORY_v1"


@dataclass(frozen=True)
class HeadlineEligibility:
    eligible: bool
    na_reason: str = ""
    taskpack_id: str = ""
    baseline_id: str = ""
    adapter_mode: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "eligible": self.eligible,
            "na_reason": self.na_reason,
            "taskpack_id": self.taskpack_id,
            "baseline_id": self.baseline_id,
            "adapter_mode": self.adapter_mode,
        }


def is_headline_run(run_dir: Path) -> bool:
    """True when neutral headline adapter materialized this run directory."""
    return ensure_headline_task_desc(Path(run_dir)) is not None


def headline_eligibility(
    taskpack: Mapping[str, Any],
    run_dir: Path,
    *,
    baseline_id: str,
) -> HeadlineEligibility:
    tid = str(taskpack.get("taskpack_id") or "")
    if is_headline_run(run_dir):
        return HeadlineEligibility(
            eligible=True,
            taskpack_id=tid,
            baseline_id=baseline_id,
            adapter_mode="headline_neutral_adapter",
        )
    return HeadlineEligibility(
        eligible=False,
        na_reason="N/A: run_dir missing headline adapter artifacts (routes.json + task_desc.md)",
        taskpack_id=tid,
        baseline_id=baseline_id,
    )


def build_headline_task_context(run_dir: Path) -> Dict[str, str]:
    run_dir = Path(run_dir)
    task_desc = run_dir / "task_desc.md"
    return {
        "data_dir": str(run_dir.resolve()),
        "goal": task_desc.read_text(encoding="utf-8"),
        "eval": "dca",
        "task_desc_path": str(task_desc.resolve()),
    }


def _metric_name_from_route(route_meta: Mapping[str, Any]) -> str:
    contract = route_meta.get("contract") or {}
    for key in ("metric", "metric_name", "primary_metric"):
        if contract.get(key):
            return str(contract[key])
    return "score"


def build_headline_smoke_trajectory(
    taskpack: Mapping[str, Any],
    run_dir: Path,
) -> Dict[str, Any]:
    """Minimal HWA_TRAJECTORY_v1 for pipeline smoke (no reference scores)."""
    run_dir = Path(run_dir)
    routes_payload = json.loads((run_dir / "routes.json").read_text(encoding="utf-8"))
    allowed = [str(e) for e in (routes_payload.get("allowed_edge_ids") or []) if e]
    if not allowed:
        raise ValueError(f"headline smoke: no allowed_edge_ids in {run_dir / 'routes.json'}")

    edge_id = allowed[0]
    route_meta: Mapping[str, Any] = next(
        (r for r in routes_payload.get("routes") or [] if str(r.get("edge_id")) == edge_id),
        {"edge_id": edge_id},
    )
    metric_name = _metric_name_from_route(route_meta)

    solver = taskpack.get("solver_view") or {}
    success = taskpack.get("success_criteria") or {}
    schema_id = str(
        success.get("artifact_schema_id")
        or (taskpack.get("reference_view") or {}).get("final_output_spec", {}).get("schema_id")
        or "hwa.final_artifact/v1"
    )

    step_kind = "capability" if str(edge_id).startswith("CAP-") else "tool"
    steps: list[Dict[str, Any]] = [
        {
            "step_index": 0,
            "kind": step_kind,
            "capability_id": edge_id,
            "tool": "headline_smoke",
            "ok": True,
            "inputs": {"mode": "headline_smoke", "edge_id": edge_id},
            "outputs": {"note": "pipeline smoke placeholder — not baseline output"},
        },
        {"step_index": 1, "kind": "submit", "ok": True},
    ]

    artifact_value = {
        "metric_name": metric_name,
        "score": 0.0,
        "selected_route": edge_id,
        "source": "headline_smoke",
    }

    return {
        "schema_version": SCHEMA_VERSION,
        "task_id": str(taskpack.get("taskpack_id") or ""),
        "steps": steps,
        "final_artifact": {
            "schema_id": schema_id,
            "value": artifact_value,
            "artifact_keys": list(artifact_value.keys()),
        },
        "route_summary": {
            "baseline_id": "headline_smoke",
            "selected_edge_id": edge_id,
            "adapter_mode": "headline_neutral_adapter",
        },
    }


def run_score_submission(
    run_dir: Path,
    *,
    agent_id: str = "baseline",
    trajectory_path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Invoke materialized score_submission.py and return parsed DCA payload."""
    run_dir = Path(run_dir).resolve()
    scorer = run_dir / "score_submission.py"
    if not scorer.is_file():
        raise FileNotFoundError(f"missing score_submission.py in {run_dir}")

    traj = (trajectory_path or (run_dir / "trajectory.json")).resolve()
    py = project_root() / "envs/pyhazards/bin/python"
    cmd = [
        str(py),
        str(scorer),
        "--trajectory",
        str(traj),
        "--agent-id",
        agent_id,
        "--output",
        str(run_dir / "dca_result.json"),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, cwd=str(run_dir), check=False)
    if proc.returncode != 0:
        raise RuntimeError(
            f"score_submission.py failed (exit {proc.returncode}): {(proc.stderr or proc.stdout)[-500:]}"
        )
    dca_path = run_dir / "dca_result.json"
    return json.loads(dca_path.read_text(encoding="utf-8"))


def write_headline_smoke_artifacts(
    *,
    baseline_id: str,
    taskpack: Mapping[str, Any],
    out_dir: Path,
) -> Dict[str, Any]:
    """Write trajectory.json, smoke_meta.json, and dca_result.json for headline smoke."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    trajectory = build_headline_smoke_trajectory(taskpack, out_dir)
    traj_path = out_dir / "trajectory.json"
    traj_path.write_text(json.dumps(trajectory, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    smoke_meta = {
        "mode": "headline_smoke",
        "baseline_id": baseline_id,
        "task_desc": str(out_dir / "task_desc.md"),
        "routes": str(out_dir / "routes.json"),
        "note": "Neutral headline adapter smoke — structural placeholder, not vendor output",
    }
    (out_dir / "smoke_meta.json").write_text(json.dumps(smoke_meta, indent=2) + "\n", encoding="utf-8")

    dca_payload = run_score_submission(out_dir, agent_id=baseline_id, trajectory_path=traj_path)
    return {"trajectory": trajectory, "trajectory_path": traj_path, "dca": dca_payload}
