#!/usr/bin/env python3
"""Verify released per-instance metrics; optionally replay DCA / E_q / V_q from workdirs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from hazardweaver.paths import BENCHMARK_PUBLIC, RELEASED_RESULTS, REPO_ROOT  # noqa: E402

PER_INST = RELEASED_RESULTS / "structured_trajectories" / "unified_141_per_instance_v1.jsonl"
OUT = REPO_ROOT / "outputs" / "evaluate_released"
EXPECTED = {"n": 141, "dca_valid": 126, "artifact_valid": 130, "trajectory_valid": 134}


def _load_rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _summarize(rows: list[dict]) -> dict[str, int]:
    return {
        "n": len(rows),
        "dca_valid": sum(1 for r in rows if r.get("dca_valid")),
        "artifact_valid": sum(1 for r in rows if r.get("artifact_valid")),
        "trajectory_valid": sum(1 for r in rows if r.get("trajectory_valid")),
    }


def _load_manifest() -> list[dict]:
    path = BENCHMARK_PUBLIC / "manifest.jsonl"
    if not path.is_file():
        raise FileNotFoundError(path)
    return _load_rows(path)


def _read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _scores_from_workdir(workdir: Path) -> tuple[bool, bool | None, bool | None]:
    """Prefer eval_report dual_gate sidecars; fall back to full replay when absent."""
    dca = _read_json(workdir / "dca_result.json")
    dca_valid = bool(dca.get("valid"))
    report = _read_json(workdir / "eval_report.json")
    dual = report.get("dual_gate") or {}
    eq_pass: bool | None
    vq_pass: bool | None
    if "E_q" in dual and "V_q" in dual:
        eq_pass = bool(dual.get("E_q"))
        vq_pass = bool(dual.get("V_q"))
        return dca_valid, eq_pass, vq_pass
    try:
        from hazardweaver.hwa.benchmark.hwa_workdir_dual_gate_v1 import (
            evaluate_hwa_workdir_dual_gate_layers,
        )

        dg = evaluate_hwa_workdir_dual_gate_layers(workdir)
        eq_pass = bool(dg.get("eq_pass")) if dg.get("eq_pass") is not None else None
        vq_pass = bool(dg.get("vq_pass")) if dg.get("vq_pass") is not None else None
    except Exception:
        eq_pass = None
        vq_pass = None
    return dca_valid, eq_pass, vq_pass


def _workdirs_by_instance(out_root: Path) -> dict[str, Path]:
    by_id: dict[str, Path] = {}
    for inv_path in Path(out_root).rglob("inventory_row.json"):
        inv = _read_json(inv_path)
        iid = str(inv.get("instance_id") or "")
        if iid:
            by_id[iid] = inv_path.parent
    return by_id


def _replay_from_workdirs(out_root: Path, *, seed: int, condition: str) -> dict[str, int]:
    inventory = _load_manifest()
    workdirs = _workdirs_by_instance(out_root)
    tallies = {k: 0 for k in EXPECTED}
    for row in inventory:
        workdir = workdirs.get(str(row.get("instance_id") or ""))
        if workdir is None or not (workdir / "answer.json").is_file():
            continue
        dca_valid, eq_pass, vq_pass = _scores_from_workdir(workdir)
        tallies["n"] += 1
        if dca_valid:
            tallies["dca_valid"] += 1
        if eq_pass:
            tallies["artifact_valid"] += 1
        if vq_pass:
            tallies["trajectory_valid"] += 1
    return tallies


def _matches_expected(summary: dict[str, int]) -> bool:
    return all(summary.get(k) == EXPECTED[k] for k in EXPECTED)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check or replay unified 141 headline metrics.")
    parser.add_argument(
        "--workdir-root",
        type=Path,
        default=None,
        help="Replay scoring from agent workdirs (seed_N/condition/T_*/answer.json).",
    )
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument("--condition", default="full_hwa")
    args = parser.parse_args()

    if not PER_INST.is_file():
        print(json.dumps({"status": "SKIPPED", "reason": "per_instance jsonl not present"}))
        return 0

    released = _summarize(_load_rows(PER_INST))
    summary: dict[str, object] = {
        "released": released,
        "expected": EXPECTED,
        "released_ok": _matches_expected(released),
    }

    exit_code = 0
    if not summary["released_ok"]:
        exit_code = 1
        for key in EXPECTED:
            if released.get(key) != EXPECTED[key]:
                print(
                    f"evaluate: released {key} {released.get(key)} != {EXPECTED[key]}",
                    file=sys.stderr,
                )

    if args.workdir_root is not None:
        if not args.workdir_root.is_dir():
            print(f"evaluate: missing workdir root {args.workdir_root}", file=sys.stderr)
            return 1
        replay = _replay_from_workdirs(args.workdir_root, seed=args.seed, condition=args.condition)
        summary["replay"] = replay
        summary["replay_ok"] = _matches_expected(replay)
        if not summary["replay_ok"]:
            exit_code = 1
            for key in EXPECTED:
                if replay.get(key) != EXPECTED[key]:
                    print(
                        f"evaluate: replay {key} {replay.get(key)} != {EXPECTED[key]}",
                        file=sys.stderr,
                    )

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    status = "OK" if exit_code == 0 else "MISMATCH"
    print(json.dumps({"status": status, "released": released, "expected": EXPECTED}))
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
