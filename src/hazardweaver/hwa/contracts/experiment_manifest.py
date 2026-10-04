"""ExperimentManifest: reproducible experiment tracking."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class SplitSpec(BaseModel):
    """Grouped split definition."""

    strategy: str = "group_kfold"
    group_column: str = "burn_area_id"
    n_splits: int = 5
    outer_fold: Optional[int] = None
    train_groups: List[str] = Field(default_factory=list)
    val_groups: List[str] = Field(default_factory=list)
    test_groups: List[str] = Field(default_factory=list)
    split_hash: str = ""


class ConfidenceIntervalSpec(BaseModel):
    method: str = "bootstrap_fire_level"
    n_bootstrap: int = 1000
    confidence_level: float = 0.95


class ExperimentManifest(BaseModel):
    """All results must link back to this manifest."""

    experiment_id: str
    gate: str
    hypothesis: str = ""
    data_manifest_hash: str = ""
    split: SplitSpec = Field(default_factory=SplitSpec)
    models: List[str] = Field(default_factory=list)
    primary_metric: str = "mae_log_volume"
    parity_margin: Optional[float] = None
    ci: ConfidenceIntervalSpec = Field(default_factory=ConfidenceIntervalSpec)
    seed: int = 42
    git_commit: str = "UNVERIFIED"
    environment: Dict[str, str] = Field(default_factory=dict)
    hyperparameters: Dict[str, Any] = Field(default_factory=dict)
    oof_predictions_path: Optional[str] = None
    checkpoint_paths: Dict[str, str] = Field(default_factory=dict)
    locked_before_test: bool = False
    notes: str = ""
