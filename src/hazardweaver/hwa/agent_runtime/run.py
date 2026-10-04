"""CLI: run SWE-style HWA agent on wildfire or PFDF solver_view tasks."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from hazardweaver.hwa.agent_runtime.ablation_conditions import (
    VALID_CONDITIONS,
    apply_ablation_condition,
)
from hazardweaver.hwa.agent_runtime.env_factory import default_out_root
from hazardweaver.hwa.agent_runtime.loop import AgentLimits, SWEAgentLoop
from hazardweaver.hwa.llm.client import client_from_env


def load_solver_task(path: Path) -> Dict[str, Any]:
    task = json.loads(path.read_text(encoding="utf-8"))
    if "gold" in task:
        raise SystemExit(f"refusing task with gold (use solver_view): {path}")
    if not task.get("task_id"):
        raise SystemExit(f"task missing task_id: {path}")
    return task


def _patch_run_meta(workdir: Path, **fields: Any) -> None:
    meta_path = workdir / "run_meta.json"
    if not meta_path.is_file():
        return
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta.update(fields)
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")


def discover_tasks(tasks_dir: Path) -> List[Path]:
    paths = sorted(tasks_dir.glob("*.json"))
    if not paths:
        raise SystemExit(f"no *.json tasks under {tasks_dir}")
    return paths


def build_llm(*, mock: bool = False):
    if mock:
        raise SystemExit(
            "CLI --mock is not a ScriptedLLM. Use pytest ScriptedLLM for CI. "
            "For real solves, start vLLM and unset mock flags."
        )
    profile = os.environ.get("HW_LLM_PROFILE") or None
    return client_from_env(profile=profile)


def run_one(
    loop: SWEAgentLoop,
    task_path: Path,
    *,
    out_root: Path,
    condition: str = "full",
) -> Dict[str, Any]:
    raw = load_solver_task(task_path)
    task = apply_ablation_condition(raw, condition)
    workdir = out_root / str(task["task_id"])
    result = loop.run(task, workdir=workdir)
    _patch_run_meta(
        result.workdir,
        condition=condition,
        ablation_condition=condition,
        allowed_tool_ids=(
            (task.get("solver_visible") or {})
            .get("allowed_inventory", {})
            .get("tool_ids")
        ),
    )
    return {
        "task_id": result.task_id,
        "task_path": str(task_path),
        "domain": task.get("domain"),
        "condition": condition,
        "workdir": str(result.workdir),
        "exit_reason": result.exit_reason,
        "submitted": result.submitted,
        "n_steps": result.n_steps,
        "n_tool_calls": result.n_tool_calls,
        "answer_path": str(result.answer_path) if result.answer_path else None,
        "run_meta_path": str(result.workdir / "run_meta.json"),
        "llm_provider": result.run_meta.get("llm_provider"),
        "model_id": result.run_meta.get("model_id"),
        "pack_root": result.run_meta.get("pack_root"),
    }


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="HW SWE-style agent (wildfire W3 / PFDF MW4)"
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--task", type=Path, help="Single solver_view JSON")
    g.add_argument("--tasks-dir", type=Path, help="Directory of solver_view JSONs")
    p.add_argument(
        "--out-root",
        type=Path,
        default=None,
        help="Per-task workdirs under out-root/<task_id>/ (default runs/hw/agent)",
    )
    p.add_argument("--batch-id", type=str, default=None, help="Optional batch manifest id")
    p.add_argument(
        "--pack-root",
        type=Path,
        default=None,
        help="Override pack root (default: auto from task.domain)",
    )
    p.add_argument("--max-steps", type=int, default=30)
    p.add_argument("--max-wall-s", type=float, default=600.0)
    p.add_argument("--max-obs-chars", type=int, default=8000)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="LLM completion budget (default 512; lower reduces context blowups)",
    )
    p.add_argument(
        "--no-optional-tools",
        action="store_true",
        help="Hide suggest_plan/validate_plan when present in pack",
    )
    p.add_argument(
        "--live-burn",
        action="store_true",
        help="PFDF: use live Prithvi burn (default is oracle_burn for CI/report-safe)",
    )
    p.add_argument(
        "--pfdf-device",
        default=os.environ.get("HW_PFDF_DEVICE", "cpu"),
        help="Device for PFDF predictors when --live-burn",
    )
    p.add_argument(
        "--condition",
        choices=sorted(VALID_CONDITIONS),
        default="full",
        help=(
            "Ablation surface: full | no_hcg | no_compose_macro (Strength R3, "
            "default full) | S0 | S1 | S2 | S4 (Phase 3 W2 Hard g6_hard_v1 arms: "
            "S0=Theory ON alias-of-full, S1=Theory OFF, S2=adapter OFF, "
            "S4=select-one vs MultiDAG; see "
            "docs/engineering/hcg/ABLATION_HARD_G6_v1.yaml). "
            "AAAI/honest tasks ship without compose_* in allowed inventory "
            "(see docs/engineering/contracts/HONEST_TASK_SCHEMA.md); "
            "runtime force-strip of compose defaults is W-B1."
        ),
    )
    args = p.parse_args(argv)

    if args.task is None and args.tasks_dir is None:
        raise SystemExit("need --task or --tasks-dir")

    # Only explicit --pack-root freezes the pack. Default None → resolve per task.domain
    # (avoids locking a mixed batch to the first wildfire task's pack).
    pack_root_override = Path(args.pack_root) if args.pack_root is not None else None

    live = bool(args.live_burn)
    llm = build_llm()
    loop = SWEAgentLoop(
        llm,
        pack_root=pack_root_override,
        limits=AgentLimits(
            max_steps=args.max_steps,
            max_wall_s=args.max_wall_s,
            max_obs_chars=args.max_obs_chars,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        ),
        include_optional_tools=not args.no_optional_tools,
        pfdf_oracle_burn=not live,
        pfdf_device=args.pfdf_device,
        pfdf_require_checkpoint=live,
    )

    out_root = Path(args.out_root or default_out_root())
    out_root.mkdir(parents=True, exist_ok=True)

    condition = str(args.condition)

    if args.task:
        summary = run_one(
            loop, Path(args.task), out_root=out_root, condition=condition
        )
        print(json.dumps(summary, indent=2))
        return 0 if summary.get("submitted") else 2

    paths = discover_tasks(Path(args.tasks_dir))
    batch_id = args.batch_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    batch_dir = out_root / f"_batch_{batch_id}"
    batch_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []
    n_ok = 0
    for path in paths:
        try:
            row = run_one(loop, path, out_root=out_root, condition=condition)
        except Exception as exc:  # noqa: BLE001
            row = {
                "task_id": path.stem,
                "task_path": str(path),
                "condition": condition,
                "exit_reason": "exception",
                "submitted": False,
                "error": f"{type(exc).__name__}: {exc}",
            }
        rows.append(row)
        if row.get("submitted"):
            n_ok += 1
        print(
            f"[{len(rows)}/{len(paths)}] {row.get('task_id')} "
            f"cond={condition} exit={row.get('exit_reason')} "
            f"submitted={row.get('submitted')}",
            flush=True,
        )

    manifest = {
        "batch_id": batch_id,
        "tasks_dir": str(args.tasks_dir),
        "out_root": str(out_root),
        "condition": condition,
        "n_tasks": len(rows),
        "n_submitted": n_ok,
        "llm_provider": getattr(llm, "provider", None),
        "model_id": getattr(llm, "model_id", None),
        "pfdf_oracle_burn": not live,
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "tasks": rows,
    }
    man_path = batch_dir / "manifest.json"
    man_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {"batch_manifest": str(man_path), "n_submitted": n_ok, "n_tasks": len(rows)},
            indent=2,
        )
    )
    return 0 if n_ok > 0 else 2


if __name__ == "__main__":
    sys.exit(main())
