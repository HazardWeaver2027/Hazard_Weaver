"""Load ICLR model arm registry and apply runtime env for headline inventory runs."""

from __future__ import annotations

import json
import os
from functools import lru_cache
from glob import glob
from pathlib import Path
from typing import Any, Dict, Mapping

ROOT = Path(__file__).resolve().parents[3]
ARMS_PATH = ROOT / "hwb/registry/iclr_model_arms_v1.json"


@lru_cache(maxsize=1)
def load_model_arms_config() -> Dict[str, Any]:
    return json.loads(ARMS_PATH.read_text(encoding="utf-8"))


def list_arm_ids() -> list[str]:
    return sorted(load_model_arms_config().get("arms", {}).keys())


def get_arm(arm_id: str) -> Dict[str, Any]:
    arms = load_model_arms_config().get("arms") or {}
    if arm_id not in arms:
        raise KeyError(f"unknown model arm {arm_id!r}; expected one of {sorted(arms)}")
    cfg = dict(arms[arm_id])
    cfg["arm_id"] = arm_id
    shard_class = str(cfg.get("partition_class") or "rtx6000")
    defaults = dict((load_model_arms_config().get("shard_defaults") or {}).get(shard_class) or {})
    override = dict(cfg.get("shard_override") or {})
    cfg["shard"] = {**defaults, **override}
    return cfg


def _resolve_snapshot(pattern: str) -> str:
    rel = str(pattern or "")
    if not rel:
        return ""
    if "*" in rel:
        matches = sorted(glob(str(ROOT / rel)))
        for m in matches:
            if Path(m).is_dir():
                return m
        return ""
    p = ROOT / rel
    return str(p) if p.is_dir() else str(p)


def apply_arm_env(arm_id: str, *, paper_run: bool = True) -> Dict[str, Any]:
    """Configure process env for one model arm. Returns non-secret metadata."""
    cfg = get_arm(arm_id)
    backend = str(cfg.get("backend") or "vllm_local")

    if backend == "closed_api":
        if not paper_run:
            raise RuntimeError(f"arm {arm_id} requires --paper-run for closed API")
        provider = str(cfg.get("closed_provider") or "")
        model_id = str(cfg.get("model_id") or "")
        from hazardweaver.hwa.llm.closed_api import apply_closed_api_provider

        key_alias = str(cfg.get("closed_api_key_alias") or "").strip()
        existing_alias = os.environ.get("HW_CLOSED_API_KEY_ALIAS", "").strip()
        if existing_alias:
            key_alias = existing_alias
        elif key_alias:
            os.environ["HW_CLOSED_API_KEY_ALIAS"] = key_alias
        meta = apply_closed_api_provider(provider, model_id=model_id)
        meta["arm_id"] = arm_id
        meta["backend"] = backend
        if key_alias:
            meta["closed_api_key_alias"] = key_alias
        return meta

    profile = str(cfg.get("hw_llm_profile") or "l4_fast")
    model_id = str(cfg.get("model_id") or "")
    os.environ["HW_LLM_PROFILE"] = profile
    # Client must use HF model id; vLLM serve uses snapshot path + --served-model-name.
    os.environ["VLLM_MODEL_ID"] = model_id
    if not os.environ.get("VLLM_BASE_URL"):
        port = default_vllm_serve_port()
        os.environ["VLLM_BASE_URL"] = f"http://127.0.0.1:{port}/v1"
    from hazardweaver.hwa.llm.context_budget import apply_context_budget_env, context_budget_from_arm_cfg

    budget = context_budget_from_arm_cfg(cfg)
    apply_context_budget_env(budget)
    shard = cfg.get("shard") or {}
    if shard.get("parallel_jobs") is not None:
        os.environ.setdefault("PARALLEL_JOBS", str(shard.get("parallel_jobs")))
    return {
        "arm_id": arm_id,
        "backend": backend,
        "profile": profile,
        "model_id": model_id,
        "display": cfg.get("display"),
        "partition_class": cfg.get("partition_class"),
        "context_budget": budget.as_dict(),
        "parallel_jobs": shard.get("parallel_jobs"),
    }


def default_vllm_serve_port() -> int:
    """Per-job port to avoid 8000 collisions and stale listeners on shared nodes."""
    raw = os.environ.get("VLLM_SERVE_PORT")
    if raw:
        return int(raw)
    job = os.environ.get("SLURM_JOB_ID")
    if job and str(job).isdigit():
        return 18000 + (int(job) % 8000)
    return 8000


def vllm_base_url_for_port(port: int | None = None) -> str:
    p = port if port is not None else default_vllm_serve_port()
    return f"http://127.0.0.1:{p}/v1"


def vllm_serve_argv(arm_id: str) -> list[str]:
    """Build vllm serve CLI args for a local arm."""
    cfg = get_arm(arm_id)
    model_id = str(cfg.get("model_id") or "")
    if "olmo3" in arm_id or "Olmo-3" in model_id:
        from hazardweaver.hwa.llm.olmo3_vllm_patch_v1 import patch_olmo3_snapshot_config

        patch_olmo3_snapshot_config(str(cfg.get("snapshot_glob") or ""))
    snap = _resolve_snapshot(str(cfg.get("snapshot_glob") or ""))
    serve_path = snap if snap else model_id
    shard = cfg.get("shard") or {}
    gpu_util = str(
        shard.get("gpu_memory_utilization")
        or cfg.get("gpu_memory_utilization")
        or "0.90"
    )
    port = default_vllm_serve_port()
    parallel = int(shard.get("parallel_jobs") or os.environ.get("PARALLEL_JOBS") or 1)
    max_num_seqs = int(
        shard.get("vllm_max_num_seqs")
        or os.environ.get("VLLM_MAX_NUM_SEQS")
        or 0
    )
    if max_num_seqs <= 0:
        max_num_seqs = max(4, min(16, parallel * 4))
    args = [
        "serve",
        serve_path,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--served-model-name",
        model_id,
        "--max-model-len",
        str(cfg.get("max_model_len") or 8192),
        "--gpu-memory-utilization",
        gpu_util,
        "--max-num-seqs",
        str(max_num_seqs),
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        str(cfg.get("tool_call_parser") or "hermes"),
    ]
    quant = str(cfg.get("quantization") or "").strip()
    if quant:
        args.extend(["--quantization", quant])
    else:
        args.extend(["--dtype", "auto"])
    extra = cfg.get("vllm_extra_args") or []
    if extra:
        args.extend([str(x) for x in extra])
    # HPG Lustre: vLLM 0.25 auto-prefetches large checkpoints into page cache when
    # node RAM looks ample, but SLURM cgroup (--mem) is much smaller → oom_kill ~60s.
    if "--safetensors-load-strategy" not in args:
        args.extend(["--safetensors-load-strategy", "lazy"])
    return args
