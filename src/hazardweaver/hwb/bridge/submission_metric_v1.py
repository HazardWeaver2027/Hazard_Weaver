"""Single metric contract for HWB submission building (batch eval + VCE VERIFY)."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

# Same scalar, different capability vs taskpack field names.
METRIC_FAMILIES = (
    frozenset({"metric_value", "depth_rmse", "rmse_depth", "mae", "max_mae"}),
    frozenset({"log_volume_v1", "log_volume"}),
    frozenset({"rmse_depth", "depth_rmse"}),
    frozenset({"substrate_coverage", "sdo_skill", "log_mae"}),
    frozenset({"replay_parity", "gmpe_coverage"}),
    frozenset({"pick_f1"}),
    frozenset({"vendor_spatial_execution"}),
    frozenset({"burn_severity_summary_v1"}),
    frozenset({"auprc", "auprc_neg_min_fos", "trigrs_auprc", "rf_auprc", "gam_auprc", "threshold_auprc"}),
)


def taskpack_tolerance_metric(taskpack: Optional[Mapping[str, Any]]) -> str:
    if not taskpack:
        return ""
    tol = (taskpack.get("reference_view") or {}).get("tolerance") or {}
    return str(tol.get("metric") or "")


def metrics_compatible(source: str, target: str) -> bool:
    """True when capability metric and taskpack tolerance name the same score family."""
    src = str(source or "").strip()
    tgt = str(target or "").strip()
    if not src or not tgt:
        return True
    if src == tgt:
        return True
    for family in METRIC_FAMILIES:
        if src in family and tgt in family:
            return True
    return False


def value_has_score(value: Mapping[str, Any]) -> bool:
    for key in (
        "reference_score",
        "metric_value",
        "brier",
        "auprc",
        "log_volume_v1",
        "rmse_depth",
        "depth_rmse",
        "mae",
        "substrate_coverage",
        "log_mae",
        "replay_parity",
        "pick_f1",
        "vendor_spatial_execution",
        "score",
    ):
        val = value.get(key)
        if val is None:
            continue
        try:
            float(val)
            return True
        except (TypeError, ValueError):
            continue
    return False


def artifact_satisfies_taskpack(
    value: Mapping[str, Any],
    *,
    taskpack: Optional[Mapping[str, Any]],
) -> bool:
    """Reject cross-capability outputs (e.g. burn severity for a volume task)."""
    tol = taskpack_tolerance_metric(taskpack)
    if not value_has_score(value):
        return not tol
    if not tol:
        return True
    if tol == "log_volume_v1":
        return value.get("log_volume_v1") is not None or str(value.get("metric_name") or "") == "log_volume_v1"
    mn = str(value.get("metric_name") or "")
    if metrics_compatible(mn, tol):
        return True
    return tol in value


def normalize_submission_value(
    value: Dict[str, Any],
    *,
    taskpack: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Rename score keys to taskpack tolerance metric when same family."""
    out = dict(value)
    tol_metric = taskpack_tolerance_metric(taskpack)
    ref_out = ((taskpack or {}).get("reference_view") or {}).get("outputs") or {}
    if not tol_metric:
        for key in ("substrate_coverage", "rmse_depth", "metric_value", "reference_score", "log_volume_v1"):
            if key in ref_out:
                tol_metric = key
                break

    score_aliases = (
        "reference_score",
        "metric_value",
        tol_metric,
        "depth_rmse",
        "rmse_depth",
        "mae",
        "substrate_coverage",
        "log_volume_v1",
        "log_volume",
        "replay_parity",
        "pick_f1",
        "log_mae",
    )
    score = None
    for key in score_aliases:
        if key not in out or key == "metric_name":
            continue
        val = out.get(key)
        if val is None:
            continue
        try:
            score = float(val)
            break
        except (TypeError, ValueError):
            continue

    source_metric = str(out.get("metric_name") or "")
    if tol_metric and score is not None and metrics_compatible(source_metric, tol_metric):
        out["metric_name"] = tol_metric
        out["reference_score"] = score
        if tol_metric not in out:
            out[tol_metric] = score
    return out
