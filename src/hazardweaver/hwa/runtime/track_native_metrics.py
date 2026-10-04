"""Per-track native metrics for PFDF / DR-OUT / TC-TRK pilot slices."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

from hazardweaver.hwa.agent_runtime.execution_schema import artifacts_dir, load_execution, write_execution
from hazardweaver.hcg.runtime.portfolio_probe_resolver import PFDF_OFFICIAL_CAPS
from hazardweaver.hwa.route_controller.multi_track_slice_v1 import (
    DR_OUT_OFFICIAL_CAPS,
    PFDF_DEFAULT_RECORD_ID,
    TC_TRK_OFFICIAL_CAPS,
    track_from_capability,
)

_SEVEN_TRACK_NATIVE_EVAL = {
    "WF-3": ("experiments.hcg.carp.native_eval.eval_wf3_wsts", "eval_wf3_cap"),
    "E1-E3": ("experiments.hcg.carp.native_eval.eval_e1e3_anchor", "eval_e1e3_cap"),
    "MH-2": ("experiments.hcg.carp.native_eval.eval_mh2", "eval_mh2_cap"),
    "MH-4": ("experiments.hcg.carp.native_eval.eval_mh4", "eval_mh4_cap"),
}
_G6_TABULAR_EDGE_PREFIXES = (
    "ridge_",
    "rf_",
    "persist_",
    "heat_rule_",
    "spi_proxy_",
    "gorr_",
    "eval_wf_",
    "hw_wildfire_",
    "wf_firms_",
)

L2_CAPS = tuple(f"CAP-L2-{i:02d}" for i in range(1, 6))
FL2_CAPS = tuple(f"CAP-FL2-{i:02d}" for i in range(1, 7))
MH3_CAPS = tuple(f"CAP-MH3-{i:02d}" for i in range(1, 7))
MH1_CAPS = tuple(f"CAP-MH1-{i:02d}" for i in range(1, 6))
DR_OUT_PROXY_CAPS = frozenset({"CAP-DROUT-04", "CAP-DROUT-05"})


def _patch_workdir_metrics(
    workdir: Path,
    *,
    execution_id: str,
    final_artifact_id: str,
    capability_id: str,
    value: Dict[str, Any],
    provenance_extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if not final_artifact_id:
        raise RuntimeError("final_artifact_id_required")
    workdir = Path(workdir)
    art_dir = artifacts_dir(workdir)
    art_dir.mkdir(parents=True, exist_ok=True)
    art_path = art_dir / f"{final_artifact_id}.json"
    if art_path.is_file():
        body = json.loads(art_path.read_text(encoding="utf-8"))
    else:
        body = {"artifact_id": final_artifact_id, "schema_id": "hwa.final_artifact/v1"}
    body["value_or_uri"] = value
    body.pop("value", None)
    art_path.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")

    er = load_execution(workdir, execution_id)
    if er:
        fa = dict(er.get("final_artifact") or {})
        fa["artifact_id"] = final_artifact_id
        fa["schema_id"] = str(fa.get("schema_id") or "hwa.final_artifact/v1")
        fa["value_or_uri"] = value
        fa.setdefault("units_support", {"units": None, "support": {}})
        er["final_artifact"] = fa
        er["provenance"] = {
            **dict(er.get("provenance") or {}),
            "inference_mode": "hcg_native_eval",
            "capability_id": capability_id,
            **(provenance_extra or {}),
        }
        write_execution(workdir, er)
    return value


def score_dr_out_capability(*, capability_id: str) -> Dict[str, Any]:
    from hazardweaver.hcg.carp.native_eval.eval_dr_out import eval_dr_out_cap

    out = eval_dr_out_cap(capability_id)
    if not out.get("ok"):
        return {"status": "score_failed", "capability_id": capability_id, "detail": out}
    metrics = dict(out.get("metrics") or {})
    metric_name = str(metrics.get("metric_name") or "metric_value")
    metric_value = metrics.get("metric_value")
    if metric_value is None:
        return {"status": "score_failed", "capability_id": capability_id, "detail": out}
    return {
        "status": "ok",
        "capability_id": capability_id,
        "metric_name": metric_name,
        "metric_value": float(metric_value),
        "proxy_replay": capability_id in DR_OUT_PROXY_CAPS,
        "source": "eval_dr_out_cap",
    }


def score_tc_trk_capability(*, capability_id: str, scenario_id: str = "") -> Dict[str, Any]:
    from hazardweaver.hcg.carp.native_eval.eval_tc_tctrk import eval_tctrk_cap, eval_tctrk_persistence

    input_artifacts: Optional[Dict[str, Any]] = None
    sid = str(scenario_id or "").strip()
    if sid:
        input_artifacts = {"scenario_id": sid, "storm_sid": sid}
    out = eval_tctrk_cap(capability_id, input_artifacts=input_artifacts)
    if not out.get("ok") and capability_id == "CAP-TCTRK-01":
        out = eval_tctrk_persistence()
    if not out.get("ok"):
        return {"status": "score_failed", "capability_id": capability_id, "detail": out}
    metrics = dict(out.get("metrics") or {})
    metric_name = str(metrics.get("metric_name") or "track_error_km")
    metric_value = metrics.get("metric_value")
    if metric_value is None:
        for key in ("track_error_km", "mean_track_error_km", "reference_score"):
            if metrics.get(key) is not None:
                metric_value = metrics[key]
                metric_name = key
                break
    if metric_value is None:
        return {"status": "score_failed", "capability_id": capability_id, "detail": out}
    return {
        "status": "ok",
        "capability_id": capability_id,
        "metric_name": metric_name,
        "metric_value": float(metric_value),
        "artifact_replay": capability_id in {"CAP-TCTRK-03", "CAP-TCTRK-04", "CAP-TCTRK-05"},
        "source": "eval_tctrk_cap",
    }


def score_pfdf_capability(
    *,
    capability_id: str,
    record_id: str = "",
    produced_artifacts: Optional[Dict[str, Any]] = None,
    scientific_dir: Optional[str] = None,
) -> Dict[str, Any]:
    rid = record_id or PFDF_DEFAULT_RECORD_ID
    produced = dict(produced_artifacts or {})
    output = produced.get("output") if isinstance(produced.get("output"), dict) else produced

    if capability_id == "burn_state_net_prithvi_v1":
        burn_val = None
        if isinstance(output, dict):
            ws = output.get("watershed_summary")
            if isinstance(ws, dict):
                burn_val = ws.get("mean_dnbr", ws.get("MeandNBR"))
            if burn_val is None:
                burn_val = output.get("mean_dnbr", output.get("MeandNBR"))
        if burn_val is None:
            return {
                "status": "score_failed",
                "capability_id": capability_id,
                "record_id": rid,
                "error": "burn_severity_unavailable",
            }
        return {
            "status": "ok",
            "capability_id": capability_id,
            "record_id": rid,
            "metric_name": "burn_severity_summary_v1",
            "metric_value": float(burn_val),
            "source": "pfdf_prithvi_infer",
        }

    log_volume = None
    if isinstance(output, dict):
        log_volume = output.get("log_volume_v1")
        if log_volume is None:
            log_volume = output.get("log_volume")
        if log_volume is None and isinstance(output.get("burn_summary"), dict):
            log_volume = output["burn_summary"].get("log_volume_v1") or output["burn_summary"].get(
                "log_volume"
            )
    if log_volume is None and capability_id == "pfdf_volume_gorr_v2":
        return {
            "status": "score_failed",
            "capability_id": capability_id,
            "record_id": rid,
            "error": "log_volume_unavailable",
        }
    if log_volume is None:
        return {
            "status": "score_failed",
            "capability_id": capability_id,
            "record_id": rid,
            "error": "log_volume_unavailable",
        }
    return {
        "status": "ok",
        "capability_id": capability_id,
        "record_id": rid,
        "metric_name": "log_volume_v1",
        "log_volume_v1": float(log_volume),
        "source": "pfdf_portfolio_dispatch",
    }


def score_fl2_capability(
    *,
    capability_id: str,
    scientific_dir: Optional[str] = None,
) -> Dict[str, Any]:
    cap = str(capability_id or "").upper()
    if cap not in FL2_CAPS:
        return {"status": "score_failed", "capability_id": cap, "error": "not_fl2_cap"}
    sci_root = Path(scientific_dir) if scientific_dir else Path(f"runs/carp/scientific/FL-2/{cap}")
    from hazardweaver.hcg.carp.native_eval.eval_fl2 import eval_fl2_cap

    out = eval_fl2_cap(cap, out_base=sci_root if sci_root.is_dir() else None)
    if not out.get("ok"):
        return {"status": "score_failed", "capability_id": cap, "detail": out}
    metrics = dict(out.get("metrics") or {})
    metric_name = str(metrics.get("metric_name") or "rmse_depth")
    metric_value = metrics.get("metric_value")
    if metric_value is None:
        metric_value = metrics.get("rmse_depth") or metrics.get("reference_score")
    if metric_value is None:
        return {"status": "score_failed", "capability_id": cap, "detail": metrics}
    return {
        "status": "ok",
        "capability_id": cap,
        "metric_name": metric_name,
        "metric_value": float(metric_value),
        "source": "eval_fl2_cap",
    }


def score_mh3_capability(
    *,
    capability_id: str,
    scientific_dir: Optional[str] = None,
) -> Dict[str, Any]:
    cap = str(capability_id or "").upper()
    if cap not in MH3_CAPS:
        return {"status": "score_failed", "capability_id": cap, "error": "not_mh3_cap"}
    sci_root = Path(scientific_dir) if scientific_dir else Path(f"runs/carp/scientific/MH-3/{cap}")
    from hazardweaver.hcg.carp.native_eval.eval_mh3 import eval_mh3_cap

    out = eval_mh3_cap(cap, out_base=sci_root if sci_root.is_dir() else None)
    if not out.get("ok"):
        return {"status": "score_failed", "capability_id": cap, "detail": out}
    metrics = dict(out.get("metrics") or {})
    metric_name = str(metrics.get("metric_name") or "metric_value")
    metric_value = metrics.get("metric_value")
    if metric_value is None:
        metric_value = metrics.get("reference_score")
    if metric_value is None:
        return {"status": "score_failed", "capability_id": cap, "detail": metrics}
    return {
        "status": "ok",
        "capability_id": cap,
        "metric_name": metric_name,
        "metric_value": float(metric_value),
        "source": "eval_mh3_cap",
    }


def _is_g6_tabular_edge(capability_id: str) -> bool:
    cid = str(capability_id or "").strip().lower()
    return any(cid.startswith(p) for p in _G6_TABULAR_EDGE_PREFIXES)


def score_registry_track_capability(
    *,
    track: str,
    capability_id: str,
    scientific_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """Run HCG native eval for seven-track CAP-* capabilities."""
    mod_path, fn_name = _SEVEN_TRACK_NATIVE_EVAL.get(str(track or "").strip(), (None, None))
    cap = str(capability_id or "").strip()
    if not mod_path or not fn_name or not cap.upper().startswith("CAP-"):
        return {"status": "score_failed", "capability_id": cap, "error": "not_registry_track_cap"}
    import importlib

    mod = importlib.import_module(mod_path)
    eval_fn = getattr(mod, fn_name)
    sci_root = Path(scientific_dir) if scientific_dir else Path(f"runs/carp/scientific/{track}/{cap}")
    out = eval_fn(cap, out_base=sci_root if sci_root.is_dir() else None)
    if not out.get("ok"):
        return {"status": "score_failed", "capability_id": cap, "detail": out}
    metrics = dict(out.get("metrics") or {})
    metric_name = str(metrics.get("metric_name") or "metric_value")
    metric_value = metrics.get("metric_value")
    if metric_value is None:
        for key in ("reference_score", "rmse_depth", "pick_f1", "substrate_coverage"):
            if metrics.get(key) is not None:
                metric_value = metrics[key]
                metric_name = key
                break
    if metric_value is None:
        return {"status": "score_failed", "capability_id": cap, "detail": metrics}
    return {
        "status": "ok",
        "capability_id": cap,
        "metric_name": metric_name,
        "metric_value": float(metric_value),
        "source": mod_path,
    }


def score_l2_capability(
    *,
    capability_id: str,
    scientific_dir: Optional[str] = None,
) -> Dict[str, Any]:
    cap = str(capability_id or "").upper()
    if cap not in L2_CAPS:
        return {"status": "score_failed", "capability_id": cap, "error": "not_l2_cap"}
    sci_root = Path(scientific_dir) if scientific_dir else Path(f"runs/carp/scientific/L2/{cap}")
    from hazardweaver.hcg.carp.native_eval.eval_l2 import eval_l2_cap

    out = eval_l2_cap(cap, out_base=sci_root if sci_root.is_dir() else None)
    if not out.get("ok"):
        return {"status": "score_failed", "capability_id": cap, "detail": out}
    metrics = dict(out.get("metrics") or {})
    metric_name = str(metrics.get("metric_name") or "metric_value")
    metric_value = metrics.get("metric_value")
    if metric_value is None:
        return {"status": "score_failed", "capability_id": cap, "detail": metrics}
    return {
        "status": "ok",
        "capability_id": cap,
        "metric_name": metric_name,
        "metric_value": float(metric_value),
        "source": "eval_l2_cap",
    }


def patch_track_native_metrics(
    workdir: Path,
    *,
    execution_id: str,
    final_artifact_id: str,
    capability_id: str,
    scenario_id: str = "pilot_scenario_000",
    record_id: str = "",
    produced_artifacts: Optional[Dict[str, Any]] = None,
    scientific_dir: Optional[str] = None,
) -> Dict[str, Any]:
    track = track_from_capability(capability_id)
    pfdf_track = track in {"PFDF", "MH-1"} or capability_id in PFDF_OFFICIAL_CAPS
    if isinstance(produced_artifacts, Mapping):
        from hazardweaver.hwa.agent_runtime.execution_schema import _extract_submission_metrics

        metrics = _extract_submission_metrics({"produced_artifacts": produced_artifacts})
        if not metrics:
            metrics = _extract_submission_metrics(produced_artifacts)
        if metrics:
            metric_name = str(metrics.get("metric_name") or "metric_value")
            metric_value = metrics.get("reference_score")
            if metric_value is None:
                metric_value = metrics.get("metric_value")
            if metric_value is not None:
                value = {
                    "capability_id": capability_id,
                    "scenario_id": scenario_id,
                    "metric_name": metric_name,
                    "metric_value": float(metric_value),
                    "reference_score": float(metric_value),
                }
                if record_id:
                    value["record_id"] = record_id
                provenance_extra = {
                    "scenario_id": scenario_id,
                    "dispatch": "hcg_produced_artifacts",
                }
                return _patch_workdir_metrics(
                    workdir,
                    execution_id=execution_id,
                    final_artifact_id=final_artifact_id,
                    capability_id=capability_id,
                    value=value,
                    provenance_extra=provenance_extra,
                )
    if track == "DR-OUT" and capability_id in DR_OUT_OFFICIAL_CAPS:
        scored = score_dr_out_capability(capability_id=capability_id)
    elif track == "TC-TRK" and capability_id in TC_TRK_OFFICIAL_CAPS:
        scored = score_tc_trk_capability(capability_id=capability_id, scenario_id=scenario_id)
    elif track == "HW-MED":
        from hazardweaver.hcg.carp.native_eval.eval_hw_med import eval_hwmed_cap

        out = eval_hwmed_cap(capability_id)
        if not out.get("ok"):
            scored = {"status": "score_failed", "capability_id": capability_id, "detail": out}
        else:
            metrics = dict(out.get("metrics") or {})
            metric_value = metrics.get("metric_value")
            if metric_value is None:
                scored = {"status": "score_failed", "capability_id": capability_id, "detail": metrics}
            else:
                scored = {
                    "status": "ok",
                    "capability_id": capability_id,
                    "metric_name": str(metrics.get("metric_name") or "max_mae"),
                    "metric_value": float(metric_value),
                    "source": "eval_hwmed_cap",
                }
    elif track == "L2" and capability_id in L2_CAPS:
        scored = score_l2_capability(
            capability_id=capability_id,
            scientific_dir=scientific_dir,
        )
    elif track == "FL-2" and capability_id in FL2_CAPS:
        scored = score_fl2_capability(
            capability_id=capability_id,
            scientific_dir=scientific_dir,
        )
    elif track == "MH-3" and capability_id in MH3_CAPS:
        scored = score_mh3_capability(
            capability_id=capability_id,
            scientific_dir=scientific_dir,
        )
    elif track in _SEVEN_TRACK_NATIVE_EVAL and str(capability_id).upper().startswith("CAP-"):
        scored = score_registry_track_capability(
            track=str(track),
            capability_id=capability_id,
            scientific_dir=scientific_dir,
        )
    elif track == "MH-1" and capability_id in MH1_CAPS:
        from hazardweaver.hcg.carp.native_eval.eval_mh1 import eval_mh1_cap

        out = eval_mh1_cap(
            capability_id,
            out_base=Path(scientific_dir) if scientific_dir else None,
        )
        if not out.get("ok"):
            scored = {"status": "score_failed", "capability_id": capability_id, "detail": out}
        else:
            metrics = dict(out.get("metrics") or {})
            metric_value = metrics.get("metric_value")
            if metric_value is None:
                scored = {"status": "score_failed", "capability_id": capability_id, "detail": metrics}
            else:
                scored = {
                    "status": "ok",
                    "capability_id": capability_id,
                    "metric_name": str(metrics.get("metric_name") or "metric_value"),
                    "metric_value": float(metric_value),
                    "source": "eval_mh1_cap",
                }
    elif pfdf_track:
        scored = score_pfdf_capability(
            capability_id=capability_id,
            record_id=record_id,
            produced_artifacts=produced_artifacts,
            scientific_dir=scientific_dir,
        )
    else:
        from hazardweaver.hwa.agent_runtime.execution_schema import _extract_submission_metrics

        metrics = None
        if isinstance(produced_artifacts, Mapping):
            metrics = _extract_submission_metrics({"produced_artifacts": produced_artifacts})
            if not metrics:
                metrics = _extract_submission_metrics(produced_artifacts)
        if not metrics:
            raise RuntimeError(f"unsupported_track_capability:{capability_id}")
        metric_name = str(metrics.get("metric_name") or "metric_value")
        metric_value = metrics.get("reference_score")
        if metric_value is None:
            metric_value = metrics.get("metric_value")
        if metric_value is None:
            raise RuntimeError(f"track_native_metrics_failed:{metrics}")
        scored = {
            "status": "ok",
            "capability_id": capability_id,
            "metric_name": metric_name,
            "metric_value": float(metric_value),
            "artifact_replay": True,
            "source": "produced_artifacts",
        }

    if scored.get("status") != "ok":
        raise RuntimeError(f"track_native_metrics_failed:{scored}")

    if pfdf_track:
        rid_out = scored.get("record_id") or record_id or PFDF_DEFAULT_RECORD_ID
        if scored.get("metric_name") == "burn_severity_summary_v1":
            mv = float(scored["metric_value"])
            value = {
                "capability_id": capability_id,
                "record_id": rid_out,
                "metric_name": "burn_severity_summary_v1",
                "metric_value": mv,
                "reference_score": mv,
            }
        else:
            value = {
                "capability_id": capability_id,
                "record_id": rid_out,
                "metric_name": scored["metric_name"],
                "log_volume_v1": scored["log_volume_v1"],
                "reference_score": scored["log_volume_v1"],
            }
    else:
        value = {
            "capability_id": capability_id,
            "scenario_id": scenario_id,
            "metric_name": scored["metric_name"],
            "metric_value": scored["metric_value"],
            "reference_score": scored["metric_value"],
        }
        if scored.get("proxy_replay"):
            value["proxy_replay"] = True
        if scored.get("artifact_replay"):
            value["artifact_replay"] = True

    provenance_extra = {
        "scenario_id": scenario_id,
        "dispatch": f"{track.lower()}_native_eval",
    }
    if record_id:
        provenance_extra["record_id"] = record_id

    return _patch_workdir_metrics(
        workdir,
        execution_id=execution_id,
        final_artifact_id=final_artifact_id,
        capability_id=capability_id,
        value=value,
        provenance_extra=provenance_extra,
    )


def patch_headline_workdir_metrics_if_needed(
    workdir: Path,
    answer_body: Mapping[str, Any],
) -> Dict[str, Any]:
    """Backfill registry artifact scores from ledger/HCG output before trajectory emit."""
    from hazardweaver.hwa.agent_runtime.execution_schema import (
        _extract_submission_metrics,
        _value_or_uri_has_score,
    )
    from hazardweaver.hwa.runtime.trajectory_ledger import load_ledger_steps

    action = str(answer_body.get("action") or "")
    if action not in {"solve", "submit_solution"}:
        return {"patched": False, "reason": "not_solve"}
    execution_id = str(answer_body.get("execution_id") or "").strip()
    final_artifact_id = str(answer_body.get("final_artifact_id") or "").strip()
    if not execution_id or not final_artifact_id:
        return {"patched": False, "reason": "missing_ids"}

    er = load_execution(workdir, execution_id)
    if er is None:
        return {"patched": False, "reason": "unknown_execution"}
    fa = er.get("final_artifact") or {}
    capability_id = str(
        answer_body.get("capability_id")
        or (er.get("executed_capability_ids") or [""])[0]
        or ""
    ).strip()
    if _value_or_uri_has_score(fa.get("value_or_uri")):
        from hazardweaver.hwb.registry.unified_agent_native_eval_reference_v1 import (
            expected_metric_name_for_capability,
        )

        inv_row = {}
        inv_path = workdir / "inventory_row.json"
        if inv_path.is_file():
            try:
                inv_row = json.loads(inv_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                inv_row = {}
        track = track_from_capability(capability_id) if capability_id else ""
        expected = expected_metric_name_for_capability(
            str(track),
            capability_id,
            inventory_row=inv_row or None,
        )
        vou = fa.get("value_or_uri")
        submitted = str(vou.get("metric_name") or "") if isinstance(vou, Mapping) else ""
        if not expected or submitted == expected:
            return {"patched": False, "reason": "already_has_score"}
    produced_artifacts: Optional[Dict[str, Any]] = None
    for step in reversed(load_ledger_steps(workdir)):
        if str(step.get("execution_id") or "") != execution_id:
            continue
        ev = step.get("execution_event")
        if isinstance(ev, Mapping):
            nested = ev.get("produced_artifacts")
            if isinstance(nested, Mapping):
                produced_artifacts = dict(nested)
            metrics = _extract_submission_metrics(ev)
            if metrics and capability_id:
                value = {**metrics, "capability_id": capability_id}
                _patch_workdir_metrics(
                    workdir,
                    execution_id=execution_id,
                    final_artifact_id=final_artifact_id,
                    capability_id=capability_id,
                    value=value,
                    provenance_extra={"dispatch": "ledger_execution_event"},
                )
                return {"patched": True, "source": "ledger_execution_event"}
        break

    if capability_id:
        from hazardweaver.hwa.runtime.execution_score_backfill import backfill_final_artifact_from_execution

        out = backfill_final_artifact_from_execution(
            workdir,
            execution_id=execution_id,
            final_artifact_id=final_artifact_id,
            capability_id=capability_id,
        )
        if out.get("ok"):
            return {"patched": True, "source": "execution_produced_artifacts"}
    return {"patched": False, "reason": "no_metrics_source"}
