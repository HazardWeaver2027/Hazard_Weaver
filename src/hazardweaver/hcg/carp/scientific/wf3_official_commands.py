"""Official WildfireSpreadTS command templates per CAP-WF3-* (Phase C)."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from hazardweaver.hcg.carp.acquire.env_routing import python_for_taskpack
from hazardweaver.hcg.carp.scientific.paths import PROJECT_ROOT, SCIENTIFIC_RUNS_ROOT, cap_scientific_dir
from hazardweaver.hcg.carp.scientific.wf3_data import VENDOR_ROOT, hdf5_dir, repo_commit

TASKPACK = "WF-3"
WSTS_TRAIN_RUNNER = PROJECT_ROOT / "scripts" / "bootstrap" / "run_wsts_train.py"
PERSISTENCE_CONFIG = (
    PROJECT_ROOT
    / "experiments"
    / "hcg"
    / "carp"
    / "scientific"
    / "wsts_vendor_cfgs"
    / "persistence_monotemporal.yaml"
)
SCIENTIFIC_TRAINER_CONFIG = (
    PROJECT_ROOT
    / "experiments"
    / "hcg"
    / "carp"
    / "scientific"
    / "wsts_vendor_cfgs"
    / "trainer_single_gpu_scientific.yaml"
)

CAP_MODEL_CONFIG: Dict[str, Dict[str, Any]] = {
    "CAP-WF3-01": {
        "family_id": "RF-PERSISTENCE",
        # vendor train.py does not import PersistenceModel into LightningCLI; use direct eval.
        "mode": "persistence_eval",
        "config": str(PERSISTENCE_CONFIG),
        "data_overrides": ["--data.n_leading_observations=1"],
    },
    "CAP-WF3-02": {
        "family_id": "RF-LINEAR-STATISTICAL",
        "mode": "train_test",
        "config": "cfgs/LogisticRegression/full_run.yaml",
        "data_overrides": ["--data.n_leading_observations=1"],
    },
    "CAP-WF3-03": {
        "family_id": "RF-SPATIAL-CONVOLUTIONAL-SEGMEN",
        "mode": "train_test",
        "config": "cfgs/unet/res18_monotemporal.yaml",
        "data_overrides": ["--data.n_leading_observations=1"],
    },
    "CAP-WF3-04": {
        "family_id": "RF-RECURRENT-TEMPORAL-SEGMENTAT",
        "mode": "train_test",
        "config": "cfgs/convlstm/full_run.yaml",
        # Vendor wandb_table5: n_leading=5, batch_size=32 (not default 64).
        "data_overrides": [
            "--data.n_leading_observations=5",
            "--data.batch_size=32",
        ],
    },
    "CAP-WF3-05": {
        "family_id": "RF-TEMPORAL-ATTENTION-SEGMENTAT",
        "mode": "train_test",
        "config": "cfgs/UTAE/all_features.yaml",
        # Vendor wandb_table5: n_leading=5, batch_size=32, return_doy=True.
        "data_overrides": [
            "--data.n_leading_observations=5",
            "--data.batch_size=32",
            "--data.return_doy=true",
        ],
    },
    "CAP-WF3-06": {
        "family_id": "RF-PROCESS-BASED-CELLULAR-SPREA",
        "mode": "cell2fire_vendor",
        "config": None,
        "data_overrides": [],
    },
}

G2_TRAIN_CAPS = ["CAP-WF3-02", "CAP-WF3-03", "CAP-WF3-04", "CAP-WF3-05"]


def _wf3_python() -> str:
    return os.environ.get("PYTHON") or str(python_for_taskpack(TASKPACK))


def _resolve_config_path(config: str) -> str:
    path = Path(config)
    if path.is_file():
        return str(path.resolve())
    vendor = VENDOR_ROOT / config
    if vendor.is_file():
        return str(vendor.resolve())
    project = PROJECT_ROOT / config
    if project.is_file():
        return str(project.resolve())
    raise FileNotFoundError(f"WSTS config not found: {config}")


def vendor_train_py() -> Path:
    return VENDOR_ROOT / "src" / "train.py"


def fold_lightning_dir(
    capability_id: str,
    fold_id: int,
    *,
    runs_root: Optional[Path] = None,
) -> Path:
    """Per-fold Lightning root (checkpoints under .../checkpoints/)."""
    cap_dir = cap_scientific_dir(TASKPACK, capability_id, runs_root=runs_root or SCIENTIFIC_RUNS_ROOT)
    return cap_dir / "checkpoint" / f"fold_{fold_id:02d}" / "lightning_logs"


def find_resume_checkpoint(
    capability_id: str,
    fold_id: int,
    *,
    runs_root: Optional[Path] = None,
) -> Optional[Path]:
    """Return last/best checkpoint for in-fold resume after interrupt (same fold only)."""
    ckpt_dir = fold_lightning_dir(capability_id, fold_id, runs_root=runs_root) / "checkpoints"
    if not ckpt_dir.is_dir():
        return None
    last = ckpt_dir / "last.ckpt"
    if last.is_file():
        return last.resolve()
    candidates = sorted(ckpt_dir.glob("*.ckpt"), key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0].resolve() if candidates else None


def build_train_command(
    capability_id: str,
    *,
    fold_id: int,
    hdf5_path: Optional[Path] = None,
    do_train: bool = True,
    runs_root: Optional[Path] = None,
    resume: bool = True,
) -> str:
    spec = CAP_MODEL_CONFIG.get(capability_id)
    if spec is None:
        raise ValueError(f"unknown capability {capability_id}")
    if spec["mode"] == "persistence_eval":
        return build_persistence_command(fold_id=fold_id, hdf5_path=hdf5_path)
    if spec["mode"] == "cell2fire_vendor":
        commit = repo_commit() or "UNKNOWN"
        return (
            f"cell2fire --wsts-fold {fold_id} --data-dir {hdf5_path or hdf5_dir()} "
            f"--vendor-commit {commit}"
        )

    data_dir = hdf5_path or hdf5_dir()
    py = _wf3_python()
    num_workers = os.environ.get("WSTS_NUM_WORKERS", "0")
    config_path = _resolve_config_path(str(spec["config"]))
    trainer_path = _resolve_config_path(str(SCIENTIFIC_TRAINER_CONFIG))
    lightning_root = fold_lightning_dir(capability_id, fold_id, runs_root=runs_root)
    parts: List[str] = [
        f"{py} {WSTS_TRAIN_RUNNER}",
        f"--config={config_path}",
        f"--trainer={trainer_path}",
        "--data=cfgs/data_monotemporal_full_features.yaml",
        f"--data.data_fold_id={fold_id}",
        f"--data.data_dir={data_dir}",
        f"--data.num_workers={num_workers}",
        "--seed_everything=0",
        f"--do_train={str(do_train).lower()}",
        "--do_test=True",
        "--do_predict=False",
        "--trainer.enable_progress_bar=False",
        f"--trainer.default_root_dir={lightning_root}",
    ]
    if do_train and resume:
        ckpt = find_resume_checkpoint(capability_id, fold_id, runs_root=runs_root)
        if ckpt is not None:
            parts.append(f"--ckpt_path={ckpt}")
    parts.extend(spec.get("data_overrides") or [])
    return " ".join(parts)


def build_persistence_command(*, fold_id: int, hdf5_path: Optional[Path] = None) -> str:
    data_dir = hdf5_path or hdf5_dir()
    runner = (
        PROJECT_ROOT
        / "experiments"
        / "hcg"
        / "carp"
        / "native_eval"
        / "scientific"
        / "wf3_persistence_eval.py"
    )
    py = _wf3_python()
    return (
        f"{py} {runner} --fold-id {fold_id} --data-dir {data_dir} "
        f"--vendor-root {VENDOR_ROOT}"
    )


def build_fold_array_command(capability_id: str, fold_id: int) -> str:
    return build_train_command(capability_id, fold_id=fold_id, do_train=capability_id in G2_TRAIN_CAPS)


def all_fold_commands(capability_id: str) -> List[str]:
    return [build_fold_array_command(capability_id, fold_id) for fold_id in range(12)]


def cap_family_id(capability_id: str) -> str:
    spec = CAP_MODEL_CONFIG.get(capability_id) or {}
    return str(spec.get("family_id") or "unknown")
