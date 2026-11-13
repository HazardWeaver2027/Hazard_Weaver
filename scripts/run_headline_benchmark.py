#!/usr/bin/env python3
"""Run HWB headline inventory cells (Full HWA vs baselines)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hazardweaver.paths import BENCHMARK_PUBLIC  # noqa: E402

DEFAULT_INVENTORY = BENCHMARK_PUBLIC / "manifest.jsonl"


def _configure_arm(arm: str, *, paper_run: bool) -> dict:
    from hazardweaver.hwa.llm.iclr_model_arms import apply_arm_env, list_arm_ids

    if arm not in list_arm_ids():
        raise SystemExit(f"unknown --model-arm {arm!r}; expected one of {list_arm_ids()}")
    meta = apply_arm_env(arm, paper_run=paper_run)
    if str(meta.get("backend") or "") == "vllm_local" and not os.environ.get("VLLM_BASE_URL"):
        raise SystemExit(f"model arm {arm!r} requires VLLM_BASE_URL")
    return meta


def main() -> int:
    parser = argparse.ArgumentParser(description="HWB headline inventory runner (141-cell headline config).")
    parser.add_argument("--config", type=Path, default=None, help="YAML stub under configs/ (optional).")
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--out-root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--cell-index", type=int, default=None, help="Flat index over inventory x conditions.")
    parser.add_argument("--flat-start", type=int, default=None)
    parser.add_argument("--flat-end", type=int, default=None)
    parser.add_argument("--conditions", nargs="+", default=None)
    parser.add_argument("--max-wall-s", type=float, default=1800.0)
    parser.add_argument("--model-arm", default="llama70b")
    parser.add_argument("--paper-run", action="store_true", help="Required for closed-API model arms.")
    parser.add_argument("--no-skip-existing", action="store_true")
    parser.add_argument("--build-manifest", action="store_true")
    args = parser.parse_args()

    if args.config is not None and args.config.is_file():
        try:
            import yaml

            cfg = yaml.safe_load(args.config.read_text(encoding="utf-8")) or {}
            args.seed = int(cfg.get("seed", args.seed))
            if cfg.get("inventory"):
                args.inventory = ROOT / str(cfg["inventory"])
            if cfg.get("conditions"):
                args.conditions = list(cfg["conditions"])
            if cfg.get("model_arm"):
                args.model_arm = str(cfg["model_arm"])
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(f"failed to load --config {args.config}: {exc}") from exc

    from hazardweaver.hwa.agent_runtime.loop import AgentLimits, SWEAgentLoop
    from hazardweaver.hwa.agent_runtime.run import build_llm
    from hazardweaver.hwa.experiments.headline_inventory_runner_v1 import (
        DEFAULT_SEED,
        VALID_CONDITIONS,
        build_manifest_from_out_root,
        cell_for_flat_index,
        controller_mode_for,
        load_headline_inventory,
        n_flat_cells,
        run_headline_cell,
    )
    from hazardweaver.hwa.llm.context_budget import agent_limits_for_headline

    conditions = args.conditions or list(VALID_CONDITIONS)
    inventory = load_headline_inventory(args.inventory)
    n_cells = n_flat_cells(n_instances=len(inventory), conditions=conditions)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.cell_index is not None:
        flat_indices = [int(args.cell_index)]
    elif args.flat_start is not None and args.flat_end is not None:
        flat_indices = list(range(int(args.flat_start), int(args.flat_end)))
    else:
        flat_indices = list(range(n_cells))

    skip_if_submitted = not args.no_skip_existing
    arm_meta = _configure_arm(args.model_arm, paper_run=bool(args.paper_run))
    llm = build_llm()
    base_limits = agent_limits_for_headline(arm_id=args.model_arm)
    limits = AgentLimits(
        max_steps=int(base_limits.max_steps),
        max_wall_s=args.max_wall_s,
        max_obs_chars=base_limits.max_obs_chars,
        temperature=base_limits.temperature,
        max_tokens=base_limits.max_tokens,
    )
    rows = []
    for flat_index in flat_indices:
        cell = cell_for_flat_index(
            inventory,
            flat_index,
            seed=int(args.seed or DEFAULT_SEED),
            out_root=out_root,
            conditions=conditions,
        )
        loop = SWEAgentLoop(
            llm,
            limits=limits,
            controller_mode=controller_mode_for(cell.condition),
        )
        row = run_headline_cell(cell, loop=loop, skip_if_submitted=skip_if_submitted)
        rows.append(row)
        print(
            json.dumps(
                {
                    "flat_index": flat_index,
                    "condition": cell.condition,
                    "instance_id": cell.inventory_row.get("instance_id"),
                    "submitted": row.get("submitted"),
                    "status": row.get("status"),
                    "exit_reason": row.get("exit_reason"),
                    "utc": datetime.now(timezone.utc).isoformat(),
                }
            ),
            flush=True,
        )

    summary_path = out_root / f"batch_summary_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    summary_path.write_text(
        json.dumps(
            {
                "arm": args.model_arm,
                "arm_meta": arm_meta,
                "inventory": str(args.inventory),
                "n_cells_attempted": len(rows),
                "n_submitted": sum(1 for r in rows if r.get("submitted")),
                "rows": rows,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    if args.build_manifest or args.cell_index is None:
        manifest = build_manifest_from_out_root(
            out_root,
            inventory_path=args.inventory,
            arm=args.model_arm,
            seed=args.seed,
        )
        print(json.dumps({"manifest_path": manifest["manifest_path"], "n_tasks": manifest["n_tasks"]}))

    return 0 if rows else 2


if __name__ == "__main__":
    raise SystemExit(main())
