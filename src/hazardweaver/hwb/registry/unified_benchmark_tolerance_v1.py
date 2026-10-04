"""Unified benchmark v1 — authoritative DCA metric + tolerance binding (143 cells)."""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from hazardweaver.hwb.registry.table_a_v2_common import SUBSTRATE_ONLY_IDS

ROOT = Path(__file__).resolve().parents[3]

L2_AUPRC_ALIASES = frozenset(
    {
        "auprc",
        "rf_auprc",
        "auprc_neg_min_fos",
        "gam_auprc",
        "threshold_auprc",
        "trigrs_auprc",
    }
)

# Stratum-specific track headline metrics (unified v1 charter).
_STRATUM_TRACK_CANONICAL: Dict[Tuple[str, str], str] = {
    ("MH-1", "S"): "brier",
}
_TRACK_CANONICAL: Dict[str, str] = {
    "MH-4": "log_mae",
    "L2": "auprc",
    "HW-MED": "max_mae",
}

_MH1_BRIER_FALLBACK_CAP = "CAP-MH1-01"


def is_unified_benchmark_row(row: Optional[Mapping[str, Any]]) -> bool:
    return bool((row or {}).get("unified_benchmark_v1"))


def infer_unified_row_split(row: Mapping[str, Any]) -> str:
    """Resolve FloodCast / HWB split for DCA gold binding (DL-221).

    W3 FL-2 Pakistan ``*_test_*`` scenarios live under sealed ``test/``, not ``holdout/``.
    """
    split = str(row.get("split") or "").strip()
    if split:
        return split
    sid = str(row.get("scenario_id") or "").strip()
    if "_test_" in sid or sid.endswith("_test_000"):
        return "official_test"
    return "hwb_holdout"


def normalize_l2_metric_name(metric: str) -> str:
    name = str(metric or "").strip()
    if name in L2_AUPRC_ALIASES:
        return "auprc"
    return name


def canonical_metric_for_row(row: Mapping[str, Any]) -> Optional[str]:
    if not is_unified_benchmark_row(row):
        return None
    track = str(row.get("track") or "").upper()
    stratum = str(row.get("stratum") or "")
    if (track, stratum) in _STRATUM_TRACK_CANONICAL:
        return _STRATUM_TRACK_CANONICAL[(track, stratum)]
    if track in _TRACK_CANONICAL:
        canon = _TRACK_CANONICAL[track]
        if track == "MH-4":
            sid = str(row.get("scenario_id") or "")
            witness = str(row.get("ablation_witness_capability_id") or "")
            decoy = str(row.get("ablation_decoy_capability_id") or "")
            caps = {sid, witness, decoy} - {""}
            if caps & SUBSTRATE_ONLY_IDS:
                return "substrate_coverage"
        if track == "L2":
            return "auprc"
        return canon
    return None


def calibrate_tolerance(metric: str, reference_score: float) -> Dict[str, Any]:
    """Track/task-calibrated bands — DCA v2 contract (DL-193, tighter than v1)."""
    m = str(metric or "").strip()
    ref = float(reference_score)
    if m == "brier":
        floor = max(0.55, min(ref * 0.90, ref - 0.04)) if ref > 0.55 else 0.55
        return {"metric": "brier", "min_score": float(floor), "dca_tolerance_contract": "v2"}
    if m == "auprc":
        return {"metric": "auprc", "min_score": max(0.0, ref - 0.08), "dca_tolerance_contract": "v2"}
    if m in {"threshold_accuracy", "sdo_skill", "threat_score", "pick_f1", "iou"}:
        # E1-E3 / low-baseline pick_f1: HCG replay refs can be <0.05; do not apply a global floor
        if m == "pick_f1" and ref < 0.05:
            floor = max(0.0, ref - 0.08)
        else:
            floor = max(0.05, ref - 0.08)
        return {
            "metric": m,
            "min_score": float(floor),
            "dca_tolerance_contract": "v2",
        }
    if m in {"log_mae", "mae", "volume_mae"}:
        band = max(0.10, min(0.55, 0.25 * max(abs(ref), 0.05)))
        return {"metric": m, "max_abs_error": float(band), "dca_tolerance_contract": "v2"}
    if m == "log_volume_v1":
        return {"metric": "log_volume_v1", "max_abs_error": 0.4, "dca_tolerance_contract": "v2"}
    if m == "lead_error_km":
        return {
            "metric": "lead_error_km",
            "max_abs_error": max(10.0, 0.20 * max(ref, 1.0)),
            "dca_tolerance_contract": "v2",
        }
    if m == "rmse_depth":
        return {
            "metric": "rmse_depth",
            "max_abs_error": max(0.04, 0.15 * max(ref, 0.01)),
            "dca_tolerance_contract": "v2",
        }
    if m == "substrate_coverage":
        return {
            "metric": "substrate_coverage",
            "max_abs_error": max(0.25, 0.08 * max(ref, 1.0)),
            "dca_tolerance_contract": "v2",
        }
    if m in {"gmpe_coverage", "replay_parity"}:
        return {"metric": m, "min_score": max(0.0, ref - 0.08), "dca_tolerance_contract": "v2"}
    if m == "metric_value":
        return {
            "metric": "metric_value",
            "max_abs_error": max(0.08, 0.15 * max(abs(ref), 0.05)),
            "dca_tolerance_contract": "v2",
        }
    if m == "burn_severity_summary_v1":
        return {"metric": m, "max_abs_error": 0.12, "dca_tolerance_contract": "v2"}
    return {
        "metric": m or "metric_value",
        "max_abs_error": max(0.08, 0.15 * max(abs(ref), 0.05)),
        "dca_tolerance_contract": "v2",
    }


def _scenario_ref(taskpack_id: str, lookup: str) -> Dict[str, Any]:
    from hazardweaver.hwb.registry.track_parametric_resolver import load_scenario_refs

    refs = load_scenario_refs(taskpack_id)
    return dict((refs.get("scenarios") or {}).get(lookup) or {})


def _mh1_brier_ref() -> Dict[str, Any]:
    return _scenario_ref("hwb_mh1_parametric_v1", _MH1_BRIER_FALLBACK_CAP)


def _gold_lookup_key(row: Mapping[str, Any], executed_capability_id: Optional[str]) -> str:
    from hazardweaver.hwa.benchmark.unified_dca_route_gate_v1 import unified_scenario_gold_capability_id

    if executed_capability_id:
        cap = str(executed_capability_id).strip().split("__", 1)[0]
        if cap.startswith("CAP-"):
            return cap
    scenario_gold = unified_scenario_gold_capability_id(row)
    if scenario_gold:
        return scenario_gold
    for key in (
        "ablation_witness_capability_id",
        "scenario_id",
    ):
        val = str(row.get(key) or "").strip()
        if val:
            return val
    return str(row.get("scenario_id") or "").strip()


def _outputs_and_tolerance_from_ref(
    scenario_ref: Mapping[str, Any],
    *,
    canonical_metric: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    outputs = dict(scenario_ref.get("outputs") or {})
    native_metric = str(outputs.get("metric_name") or (scenario_ref.get("tolerance") or {}).get("metric") or "")
    metric = canonical_metric
    if metric == "auprc":
        native_metric = normalize_l2_metric_name(native_metric)
        for alias in L2_AUPRC_ALIASES:
            if alias in outputs:
                outputs["auprc"] = outputs[alias]
                outputs["metric_name"] = "auprc"
                outputs["reference_score"] = float(outputs[alias])
                break
    if native_metric and native_metric != metric and metric in outputs:
        pass
    elif native_metric and native_metric != metric:
        # Keep numeric gold under canonical key when same scalar family or explicit ref score.
        ref_val = outputs.get(native_metric)
        if ref_val is None:
            ref_val = outputs.get("reference_score")
        if ref_val is not None:
            outputs[metric] = float(ref_val)
            outputs["metric_name"] = metric
            outputs["reference_score"] = float(ref_val)
    ref_score = outputs.get(metric)
    if ref_score is None:
        ref_score = outputs.get("reference_score")
    if ref_score is None and native_metric:
        ref_score = outputs.get(native_metric)
    if ref_score is None:
        raise KeyError(f"missing gold for metric={metric} native={native_metric}")
    tol = calibrate_tolerance(metric, float(ref_score))
    outputs["metric_name"] = metric
    outputs["reference_score"] = float(ref_score)
    outputs[metric] = float(ref_score)
    return outputs, tol


def resolve_unified_gold_and_tolerance(
    row: Mapping[str, Any],
    taskpack: Mapping[str, Any],
    *,
    executed_capability_id: Optional[str] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Resolve canonical gold + tolerance for one unified inventory row."""
    canon = canonical_metric_for_row(row)
    taskpack_id = str(taskpack.get("taskpack_id") or row.get("taskpack_id") or "")
    lookup = _gold_lookup_key(row, executed_capability_id)

    track_u = str(row.get("track") or "").upper()
    if track_u in {"MH-1", "TC-TRK", "WF-3"}:
        from hazardweaver.hwb.registry.unified_agent_native_eval_reference_v1 import (
            TRACK_NATIVE_CAPS,
            native_eval_scenario_ref_for_capability,
            unified_agent_native_eval_authority,
        )

        if unified_agent_native_eval_authority(row):
            cap = ""
            if executed_capability_id:
                cap = str(executed_capability_id).strip().split("__", 1)[0]
            if not cap:
                cap = str(lookup or "").strip().split("__", 1)[0]
            if cap.startswith("CAP-") and cap in TRACK_NATIVE_CAPS.get(track_u, ()):
                scenario_ref = native_eval_scenario_ref_for_capability(
                    track_u,
                    cap,
                    inventory_row=row,
                )
                metric = str((scenario_ref.get("outputs") or {}).get("metric_name") or "metric_value")
                return _outputs_and_tolerance_from_ref(scenario_ref, canonical_metric=metric)

    if track_u == "FL-2" and taskpack_id == "hwb_fl2_parametric_v1":
        from hazardweaver.hwb.registry.fl2_parametric_agent_replay_reference_v1 import (
            fl2_parametric_agent_replay_eval_authority,
            fl2_parametric_replay_scenario_ref_for_capability,
        )

        if fl2_parametric_agent_replay_eval_authority(row):
            cap = str(executed_capability_id or lookup or "").strip().split("__", 1)[0]
            scenario_id = str(row.get("scenario_id") or "").strip()
            split = infer_unified_row_split(row)
            if cap.startswith("CAP-FL2-0") and scenario_id:
                scenario_ref = fl2_parametric_replay_scenario_ref_for_capability(
                    scenario_id,
                    cap,
                    split=split,
                    inventory_row=row,
                )
                return _outputs_and_tolerance_from_ref(
                    scenario_ref,
                    canonical_metric="rmse_depth",
                )

    if track_u == "FL-2" and taskpack_id == "hwb_fl2_solver_parametric_v1":
        from hazardweaver.hwb.registry.fl2_solver_agent_replay_reference_v1 import (
            fl2_solver_agent_replay_eval_authority,
            fl2_solver_replay_scenario_ref_for_capability,
        )

        if fl2_solver_agent_replay_eval_authority(row):
            cap = str(executed_capability_id or lookup or "").strip().split("__", 1)[0]
            scenario_id = str(row.get("scenario_id") or "").strip()
            split = infer_unified_row_split(row)
            if cap.startswith("CAP-FL2-") and scenario_id:
                scenario_ref = fl2_solver_replay_scenario_ref_for_capability(
                    scenario_id,
                    cap,
                    split=split,
                    inventory_row=row,
                )
                return _outputs_and_tolerance_from_ref(
                    scenario_ref,
                    canonical_metric="rmse_depth",
                )

    if not canon:
        lookup = _gold_lookup_key(row, executed_capability_id)
        for key in (lookup.split("__")[0], str(row.get("scenario_id") or "").strip()):
            if not key:
                continue
            scenario_ref = _scenario_ref(taskpack_id, key)
            outputs = dict((scenario_ref or {}).get("outputs") or {})
            if outputs:
                metric = str(outputs.get("metric_name") or "metric_value")
                return _outputs_and_tolerance_from_ref(scenario_ref, canonical_metric=metric)
        ref = dict((taskpack.get("reference_view") or {}))
        outputs = dict(ref.get("outputs") or {})
        tol = dict(ref.get("tolerance") or {})
        metric = str(tol.get("metric") or outputs.get("metric_name") or "metric_value")
        ref_score = outputs.get(metric) or outputs.get("reference_score")
        if ref_score is None:
            ref_score = 0.0
        return outputs, calibrate_tolerance(metric, float(ref_score))

    if taskpack_id == "hwb_mh1_atlas_state_variant_v1":
        sid = str(row.get("scenario_id") or "").strip()
        if sid:
            scenario_ref = _scenario_ref(taskpack_id, sid)
            if scenario_ref:
                return _outputs_and_tolerance_from_ref(
                    scenario_ref,
                    canonical_metric="log_volume_v1",
                )

    if str(row.get("track") or "").upper() == "HW-MED":
        from hazardweaver.hwb.registry.hwmed_agent_replay_reference_v1 import (
            hwmed_agent_replay_eval_authority,
            hwmed_replay_scenario_ref_for_capability,
        )

        if hwmed_agent_replay_eval_authority(row):
            cap = str(lookup or "").strip().split("__", 1)[0]
            if not cap and executed_capability_id:
                cap = str(executed_capability_id).strip().split("__", 1)[0]
            scenario_ref = hwmed_replay_scenario_ref_for_capability(cap)
            return _outputs_and_tolerance_from_ref(scenario_ref, canonical_metric="max_mae")

    if str(row.get("track") or "").upper() == "MH-1" and canon == "brier":
        scenario_ref = _scenario_ref(taskpack_id, lookup)
        if not scenario_ref or str((scenario_ref.get("tolerance") or {}).get("metric") or "") != "brier":
            scenario_ref = _mh1_brier_ref()
        return _outputs_and_tolerance_from_ref(scenario_ref, canonical_metric="brier")

    scenario_ref = _scenario_ref(taskpack_id, lookup.split("__")[0])
    if not scenario_ref and executed_capability_id:
        scenario_ref = _scenario_ref(taskpack_id, str(executed_capability_id))
    if not scenario_ref:
        ref = dict((taskpack.get("reference_view") or {}))
        scenario_ref = {
            "outputs": ref.get("outputs") or {},
            "tolerance": ref.get("tolerance") or {},
        }
    metric = canon
    if metric == "auprc":
        metric = normalize_l2_metric_name(
            str((scenario_ref.get("tolerance") or {}).get("metric") or (scenario_ref.get("outputs") or {}).get("metric_name") or "auprc")
        )
    return _outputs_and_tolerance_from_ref(scenario_ref, canonical_metric=metric)


def apply_unified_benchmark_eval_binding(
    taskpack: Mapping[str, Any],
    *,
    inventory_row: Mapping[str, Any],
    executed_capability_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Apply unified metric/tolerance binding after capability-aligned gold."""
    if not is_unified_benchmark_row(inventory_row):
        return dict(taskpack)
    out = deepcopy(dict(taskpack))
    outputs, tol = resolve_unified_gold_and_tolerance(
        inventory_row,
        out,
        executed_capability_id=executed_capability_id,
    )
    ref = dict(out.get("reference_view") or {})
    ref["outputs"] = outputs
    ref["tolerance"] = tol
    cap = str(
        executed_capability_id
        or outputs.get("scenario_id")
        or ref.get("capability_id")
        or ""
    ).strip().split("__", 1)[0]
    if cap:
        ref["capability_id"] = cap
    out["reference_view"] = ref
    return out


def annotate_inventory_row(row: Mapping[str, Any]) -> Dict[str, Any]:
    """Precompute unified_dca_tolerance on inventory rows (audit/submit)."""
    item = dict(row)
    if not is_unified_benchmark_row(item):
        return item
    from hazardweaver.hwb.run.eval_dca_submission_v1 import resolve_taskpack_for_inventory_row

    tp = resolve_taskpack_for_inventory_row(item)
    _, tol = resolve_unified_gold_and_tolerance(item, tp)
    item["unified_dca_tolerance"] = tol
    item["unified_canonical_metric"] = tol.get("metric")
    return item


def annotate_inventory_jsonl(path: Path, out_path: Optional[Path] = None) -> Dict[str, Any]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    annotated = [annotate_inventory_row(r) for r in rows]
    dest = out_path or path
    dest.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in annotated) + "\n", encoding="utf-8")
    from collections import Counter

    return {
        "path": str(dest),
        "n_rows": len(annotated),
        "by_metric": dict(Counter(str(r.get("unified_canonical_metric") or "?") for r in annotated)),
    }
