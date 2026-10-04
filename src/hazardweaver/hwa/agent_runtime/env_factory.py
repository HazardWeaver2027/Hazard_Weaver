"""Resolve agent pack + tool env from task domain (wildfire vs pfdf)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Union

from hazardweaver.hwa.agent_runtime.env import GoldLeakageError, WildfireToolEnv
from hazardweaver.hwa.pfdf_agent.data_access import PACK_ROOT as PFDF_PACK_ROOT
from hazardweaver.hwa.wildfire.data_store import PACK_ROOT as WILDFIRE_PACK_ROOT
from hazardweaver.hwa.wildfire.data_store import PROJECT_ROOT

PackEnv = Union[WildfireToolEnv, Any]


def resolve_pack_root(
    task: Mapping[str, Any],
    *,
    pack_root: Path | None = None,
) -> Path:
    """Resolve pack for a task.

    ``domain=pfdf`` always uses the PFDF pack (ignores a mistaken wildfire
    override). Explicit ``--pack-root`` only applies when domain is not pfdf,
    or when the override path *is* the PFDF pack.
    """
    domain = str(task.get("domain") or "").lower()
    if domain == "pfdf":
        return Path(PFDF_PACK_ROOT)
    if pack_root is not None:
        return Path(pack_root)
    return Path(WILDFIRE_PACK_ROOT)


def make_tool_env(
    task: Mapping[str, Any],
    *,
    workdir: Path,
    pack_root: Path | None = None,
    max_obs_chars: int = 8000,
    pfdf_oracle_burn: bool = True,
    pfdf_device: str = "cpu",
    pfdf_require_checkpoint: bool = False,
) -> PackEnv:
    """Factory: WildfireToolEnv vs PfdfToolEnv."""
    from hazardweaver.hwa.agent_runtime.pfdf_env import PfdfToolEnv

    if "gold" in task:
        raise GoldLeakageError(
            f"refusing to start agent: task {task.get('task_id')!r} contains gold"
        )
    domain = str(task.get("domain") or "").lower()
    root = resolve_pack_root(task, pack_root=pack_root)
    from hazardweaver.hwa.experiments.agent_strict_v2 import agent_strict_v2_enabled

    oracle_burn = pfdf_oracle_burn
    if agent_strict_v2_enabled():
        oracle_burn = False
    if domain == "pfdf" or root.resolve() == Path(PFDF_PACK_ROOT).resolve():
        return PfdfToolEnv(
            task,
            workdir=workdir,
            pack_root=Path(PFDF_PACK_ROOT),
            max_obs_chars=max_obs_chars,
            default_oracle_burn=oracle_burn,
            device=pfdf_device,
            require_checkpoint=pfdf_require_checkpoint,
        )
    return WildfireToolEnv(
        task,
        workdir=workdir,
        pack_root=root,
        max_obs_chars=max_obs_chars,
    )


def default_out_root() -> Path:
    return PROJECT_ROOT / "runs" / "hw" / "agent"
