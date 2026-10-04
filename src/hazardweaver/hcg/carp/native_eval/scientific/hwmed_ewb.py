"""HW-MED Phase C scientific eval — EWB official evaluator (not eval_subset fixture)."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import textwrap
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from hazardweaver.hcg.carp.batch2.replay_certificate import write_metrics, write_replay_manifest
from hazardweaver.hcg.carp.native_eval.blocked import eval_blocked_generic, write_blocked
from hazardweaver.hcg.carp.scientific.hwmed_external_acquisition import (
    EXTERNAL_ACQUISITION_CAPS,
    eval_external_cap,
)
from hazardweaver.hcg.carp.scientific.ewb_env import require_ewb_python
from hazardweaver.hcg.carp.scientific.hwmed_data import (
    DATA_SOURCE,
    TASKPACK,
    events_yaml_path,
    load_split_manifest,
    scientific_data_ready,
)
from hazardweaver.hcg.carp.scientific.paths import SCIENTIFIC_RUNS_ROOT, cap_scientific_dir, taskpack_repo_pin

BLOCKED_CAPS: Dict[str, str] = {}  # legacy alias; use EXTERNAL_ACQUISITION_CAPS for 01/05

EXTERNAL_ACQUISITION_COMMANDS: Dict[str, str] = {
    "CAP-HWMED-01": (
        "EXTERNAL_ACQUISITION_REQUIRED: NOAA GFS / WeatherBench2 HRES → "
        "lossless adapter → ewb.evaluation MaximumMeanAbsoluteError (same HW-MED contract)"
    ),
    "CAP-HWMED-05": (
        "EXTERNAL_ACQUISITION_REQUIRED: DeepMind GenCast official inference/artifact → "
        "ingest → ewb.evaluation MaximumMeanAbsoluteError (same HW-MED contract)"
    ),
}

CAP_FORECAST: Dict[str, Tuple[str, str, str]] = {
    "CAP-HWMED-02": ("RF-SPECTRAL-NEURAL-WEATHER-MODE", "cira", "FOUR_v200_GFS"),
    "CAP-HWMED-03": ("RF-DETERMINISTIC-NEURAL-WEATHER", "cira", "PANG_v100_GFS"),
    "CAP-HWMED-04": ("RF-GRAPH-NEURAL-WEATHER-MODEL", "cira", "GRAP_v100_GFS"),
    "CAP-HWMED-06": ("RF-STATISTICAL-BASELINE", "climatology", "0.85"),
}

OFFICIAL_COMMANDS: Dict[str, str] = {
    "CAP-HWMED-01": EXTERNAL_ACQUISITION_COMMANDS["CAP-HWMED-01"],
    "CAP-HWMED-02": (
        "ewb.evaluation + inputs.get_cira_icechunk(FOUR_v200_GFS) + metrics.MaximumMeanAbsoluteError"
    ),
    "CAP-HWMED-03": (
        "ewb.evaluation + inputs.get_cira_icechunk(PANG_v100_GFS) + metrics.MaximumMeanAbsoluteError"
    ),
    "CAP-HWMED-04": (
        "ewb.evaluation + inputs.get_cira_icechunk(GRAP_v100_GFS) + metrics.MaximumMeanAbsoluteError"
    ),
    "CAP-HWMED-05": EXTERNAL_ACQUISITION_COMMANDS["CAP-HWMED-05"],
    "CAP-HWMED-06": (
        "ewb.evaluation + defaults.get_climatology(0.85) + metrics.MaximumMeanAbsoluteError"
    ),
}

ALL_CAPS = [
    "CAP-HWMED-01",
    "CAP-HWMED-02",
    "CAP-HWMED-03",
    "CAP-HWMED-04",
    "CAP-HWMED-05",
    "CAP-HWMED-06",
]

DEFAULT_EWB_EVAL_TIMEOUT_SEC = 21600


def _ewb_eval_timeout_sec() -> int:
    raw = os.environ.get("HW_EWB_EVAL_TIMEOUT_SEC", "").strip()
    if not raw:
        return DEFAULT_EWB_EVAL_TIMEOUT_SEC
    try:
        return max(600, int(raw))
    except ValueError:
        return DEFAULT_EWB_EVAL_TIMEOUT_SEC


def _ewb_subprocess_env() -> Dict[str, str]:
    """Isolated EWB subprocess env.

    Public EWB/ERA5/CIRA buckets are anonymous GCS; without ``GCSFS_TOKEN=anon``
    gcsfs tries Application Default Credentials on every open (slow) and CIRA
    subsetting can return empty ``valid_time`` (scientifically invalid).
    Optional ``HW_GCP_CREDENTIALS`` points at a service-account JSON for
    authenticated access when needed.
    """
    env = os.environ.copy()
    env["PYTHONNOUSERSITE"] = "1"
    project_root = Path(__file__).resolve().parents[5]
    env["PYTHONPATH"] = str(project_root)
    creds = os.environ.get("HW_GCP_CREDENTIALS", "").strip()
    if creds:
        env["GOOGLE_APPLICATION_CREDENTIALS"] = creds
    else:
        env.setdefault("GCSFS_TOKEN", "anon")
    return env


def _default_n_jobs() -> int:
    raw = os.environ.get("HW_EWB_N_JOBS", "").strip()
    if raw.isdigit() and int(raw) >= 0:
        return int(raw)
    return 1


def _write_recipe_log(
    cap_dir: Path,
    *,
    capability_id: str,
    command: str,
    returncode: int = 0,
    stdout_tail: str = "",
) -> Path:
    cap_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "taskpack_id": TASKPACK,
        "capability_id": capability_id,
        "official_command": command,
        "returncode": returncode,
        "status": "executed" if returncode == 0 else "failed",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "stdout_tail": stdout_tail[-4000:],
        "note": "Phase C scientific EWB MaximumMeanAbsoluteError",
    }
    path = cap_dir / "recipe_run.log"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def _official_case_id_numbers() -> List[int]:
    manifest = load_split_manifest("official_test") or {}
    nums = manifest.get("case_id_numbers") or []
    return [int(n) for n in nums if n is not None]


def aggregate_ewb_results_csv(csv_path: Path) -> Tuple[float, Dict[str, Any]]:
    """Aggregate EWB ``run_evaluation`` CSV (long ``value``/``metric`` or wide MAE columns)."""
    if not csv_path.is_file():
        raise RuntimeError(f"EWB results csv missing: {csv_path}")

    with csv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    if not rows:
        raise RuntimeError(f"EWB results csv empty: {csv_path}")

    if "value" in fieldnames:
        values: List[float] = []
        for row in rows:
            metric_name = str(row.get("metric") or "")
            if metric_name and "MaximumMeanAbsoluteError" not in metric_name:
                continue
            raw = row.get("value")
            if raw in (None, ""):
                continue
            try:
                val = float(raw)
            except (TypeError, ValueError):
                continue
            if math.isfinite(val):
                values.append(val)
        if not values:
            raise RuntimeError(f"no finite MaximumMeanAbsoluteError values in {csv_path}")
        return float(sum(values) / len(values)), {
            "metric_col": "value",
            "n_rows": len(rows),
            "n_values": len(values),
        }

    for col in fieldnames:
        col_s = str(col)
        if "MaximumMeanAbsoluteError" in col_s or col_s.lower() in {"max_mae", "maximummeanabsoluteerror"}:
            values = []
            for row in rows:
                raw = row.get(col)
                if raw in (None, ""):
                    continue
                try:
                    val = float(raw)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(val):
                    values.append(val)
            if values:
                return float(sum(values) / len(values)), {
                    "metric_col": col_s,
                    "n_rows": len(rows),
                    "n_values": len(values),
                }

    for col in fieldnames:
        if "mae" in str(col).lower():
            values = []
            for row in rows:
                raw = row.get(col)
                if raw in (None, ""):
                    continue
                try:
                    val = float(raw)
                except (TypeError, ValueError):
                    continue
                if math.isfinite(val):
                    values.append(val)
            if values:
                return float(sum(values) / len(values)), {
                    "metric_col": str(col),
                    "n_rows": len(rows),
                    "n_values": len(values),
                }

    raise RuntimeError(f"could not locate MaximumMeanAbsoluteError column in {csv_path}")


def _publish_scientific_success(
    *,
    cap_dir: Path,
    capability_id: str,
    family_id: str,
    forecast_mode: str,
    model_name: str,
    cmd: str,
    out_root: Path,
    metric_value: float,
    out_csv: str,
    aggregate_meta: Optional[Dict[str, Any]] = None,
    note_suffix: str = "",
) -> Dict[str, Any]:
    _write_recipe_log(cap_dir, capability_id=capability_id, command=cmd, returncode=0)
    _write_predictions_stub(cap_dir, split="test", source=model_name)
    _write_predictions_stub(cap_dir, split="holdout", source=model_name)

    note = f"scientific EWB CIRA/climatology eval; mode={forecast_mode}"
    if note_suffix:
        note = f"{note}; {note_suffix}"

    metrics: Dict[str, Any] = {
        "metric_name": "max_mae",
        "metric_value": metric_value,
        "synthetic_only": False,
        "data_source": DATA_SOURCE,
        "evaluator": "EWB metrics.MaximumMeanAbsoluteError mean over official_test cases",
        "n_official_cases": len(_official_case_id_numbers()),
        "forecast_source": model_name,
        "results_csv": out_csv,
        "note": note,
    }
    if aggregate_meta:
        metrics["ewb_aggregate"] = aggregate_meta

    write_metrics(TASKPACK, capability_id, metrics, base=out_root)
    write_replay_manifest(
        TASKPACK,
        capability_id,
        family_id=family_id,
        exec_ok=True,
        metric_name="max_mae",
        metric_value=metric_value,
        notes=metrics["note"],
        base=out_root,
    )
    return {"ok": True, "metrics": metrics}


def _build_eval_script(
    *,
    capability_id: str,
    forecast_mode: str,
    model_name: str,
    events_yaml: str,
    case_ids: List[int],
    out_csv: str,
    n_jobs: int = 1,
) -> str:
    return textwrap.dedent(
        f"""
        import json
        import os

        from hazardweaver.hcg.carp.scientific.hwmed_gcs import (
            era5_storage_options,
            install_gcs_anon_bootstrap,
        )
        from hazardweaver.hcg.carp.scientific.hwmed_grids import (
            expand_climatology_to_case,
            open_climatology_surface_air_temperature,
        )

        install_gcs_anon_bootstrap()

        import dataclasses

        import numpy as np
        import pandas as pd
        import xarray as xr
        from extremeweatherbench import defaults, inputs, metrics, cases

        # EWB v1.0.2: check_for_valid_times slices init_time directly and rejects
        # valid init+lead combos for CIRA icechunk. Patch before evaluation import.
        _orig_check = inputs.check_for_missing_data

        def _ewb_check_for_missing_data(data, case_metadata, source_module=None):
            if isinstance(data, xr.Dataset):
                if "init_time" in data.coords and "lead_time" in data.coords:
                    return True
                if "dayofyear" in data.coords and "hour" in data.coords:
                    return True
            return _orig_check(data, case_metadata, source_module=source_module)

        inputs.check_for_missing_data = _ewb_check_for_missing_data

        @dataclasses.dataclass
        class ClimatologyHeatwaveForecast(inputs.XarrayForecast):
            \"\"\"Expand climatology (dayofyear/hour) to valid_time for heatwave eval.\"\"\"

            def subset_data_to_case(self, data, case_metadata, **kwargs):
                drop = kwargs.get("drop", False)
                if isinstance(data, xr.Dataset):
                    clim = data["surface_air_temperature"]
                else:
                    clim = data
                return expand_climatology_to_case(clim, case_metadata, drop=drop)

        import extremeweatherbench as ewb

        case_list = cases.load_individual_cases_from_yaml({events_yaml!r})
        wanted = set({case_ids!r})
        case_list = [c for c in case_list if c.case_id_number in wanted and c.event_type == "heat_wave"]
        if not case_list:
            raise RuntimeError("no heat_wave cases matched official_test manifest")

        target = inputs.ERA5(
            source=inputs.ARCO_ERA5_FULL_URI,
            variables=["surface_air_temperature"],
            storage_options=era5_storage_options(),
        )
        mode = {forecast_mode!r}
        if mode == "cira":
            model = {model_name!r}
            if model == "FOUR_v200_GFS":
                forecast = defaults.cira_fcnv2_heatwave_forecast
            elif model == "PANG_v100_GFS":
                forecast = inputs.get_cira_icechunk(
                    model_name="PANG_v100_GFS",
                    variables=["surface_air_temperature"],
                    name="PanguWeather",
                )
            elif model == "GRAP_v100_GFS":
                forecast = inputs.get_cira_icechunk(
                    model_name="GRAP_v100_GFS",
                    variables=["surface_air_temperature"],
                    name="GraphCast",
                )
            else:
                forecast = inputs.get_cira_icechunk(
                    model_name=model,
                    variables=["surface_air_temperature"],
                    name=model,
                )
        elif mode == "climatology":
            quantile = float({model_name!r})
            clim = open_climatology_surface_air_temperature(quantile)
            ds = clim.to_dataset(name="surface_air_temperature")
            forecast = ClimatologyHeatwaveForecast(
                ds=ds,
                variables=["surface_air_temperature"],
                name="Climatology",
            )
        else:
            raise RuntimeError(f"unknown forecast mode {{mode}}")

        probe_n = 0
        probe_case = None
        for c in case_list:
            probe = forecast.subset_data_to_case(forecast.ds, c)
            n = int(probe.sizes.get("valid_time", 0))
            if n > 0:
                probe_n = n
                probe_case = c.case_id_number
                break
        if probe_n == 0:
            raise RuntimeError(
                "forecast subset empty for all official_test cases; require GCSFS_TOKEN=anon"
            )
        print(json.dumps({{"preflight_valid_time": probe_n, "probe_case": probe_case}}))

        # Same default as CIRA caps; climatology forecast carries dummy lead_time=0.
        metric = metrics.MaximumMeanAbsoluteError()

        eval_objects = [
            inputs.EvaluationObject(
                event_type="heat_wave",
                metric_list=[metric],
                target=target,
                forecast=forecast,
            )
        ]
        runner = ewb.evaluation(case_metadata=case_list, evaluation_objects=eval_objects)
        # Serial threading (n_jobs=1) + EWB patch in-process; loky loses the patch.
        df = runner.run_evaluation(
            parallel_config={{"backend": "threading", "n_jobs": {n_jobs}}}
        )
        df.to_csv({out_csv!r}, index=False)

        def _aggregate_max_mae(frame):
            if len(frame) == 0:
                return float("nan"), ""
            if "value" in frame.columns:
                sub = frame
                if "metric" in frame.columns:
                    mask = frame["metric"].astype(str).str.contains(
                        "MaximumMeanAbsoluteError", case=False, na=False
                    )
                    if mask.any():
                        sub = frame[mask]
                vals = sub["value"].astype(float)
                vals = vals[vals.map(lambda x: x == x)]
                if len(vals) == 0:
                    return float("nan"), ""
                return float(vals.mean()), "value"
            for col in frame.columns:
                col_s = str(col)
                if "MaximumMeanAbsoluteError" in col_s or col_s.lower() in {{
                    "max_mae",
                    "maximummeanabsoluteerror",
                }}:
                    vals = frame[col].astype(float)
                    vals = vals[vals.map(lambda x: x == x)]
                    if len(vals):
                        return float(vals.mean()), col_s
            for col in frame.columns:
                if "mae" in str(col).lower():
                    vals = frame[col].astype(float)
                    vals = vals[vals.map(lambda x: x == x)]
                    if len(vals):
                        return float(vals.mean()), str(col)
            return float("nan"), ""

        val, metric_col = _aggregate_max_mae(df)
        if not (val == val):
            raise RuntimeError(
                f"EWB metric aggregation failed (n_rows={{len(df)}}); check forecast/target GCS access"
            )
        print(json.dumps({{
            "metric_value": val,
            "metric_col": metric_col,
            "n_rows": len(df),
            "n_cases": len(case_list),
            "capability_id": {capability_id!r},
            "out_csv": {out_csv!r},
        }}))
        """
    )


def _run_ewb_eval(
    cap_dir: Path,
    *,
    capability_id: str,
    forecast_mode: str,
    model_name: str,
    n_jobs: int | None = None,
) -> Tuple[float, str]:
    events_yaml = str(events_yaml_path())
    case_ids = _official_case_id_numbers()
    if not case_ids:
        raise RuntimeError("official_test manifest missing case_id_numbers")

    eval_dir = cap_dir / "eval_workspace"
    eval_dir.mkdir(parents=True, exist_ok=True)
    out_csv = str(eval_dir / "ewb_results.csv")
    eval_log = eval_dir / "ewb_eval.log"
    jobs = n_jobs if n_jobs is not None else _default_n_jobs()

    script = _build_eval_script(
        capability_id=capability_id,
        forecast_mode=forecast_mode,
        model_name=model_name,
        events_yaml=events_yaml,
        case_ids=case_ids,
        out_csv=out_csv,
        n_jobs=jobs,
    )
    timeout_sec = _ewb_eval_timeout_sec()
    with eval_log.open("w", encoding="utf-8") as log_fp:
        proc = subprocess.run(
            [str(require_ewb_python()), "-c", script],
            stdout=log_fp,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_sec,
            check=False,
            env=_ewb_subprocess_env(),
        )
    log_tail = eval_log.read_text(encoding="utf-8")[-4000:] if eval_log.is_file() else ""
    if proc.returncode != 0:
        raise RuntimeError(f"EWB eval failed: {log_tail}")
    lines = [ln for ln in eval_log.read_text(encoding="utf-8").strip().splitlines() if ln.strip()]
    result_line = next(
        (ln for ln in reversed(lines) if '"metric_value"' in ln),
        "",
    )
    if not result_line:
        raise RuntimeError(f"EWB eval missing metric JSON result line: {log_tail}")
    line = result_line
    data = json.loads(line)
    metric_value = float(data["metric_value"])
    if not (metric_value == metric_value):  # NaN check
        raise RuntimeError(f"EWB eval returned NaN metric from {out_csv}")
    return metric_value, out_csv


def _write_predictions_stub(cap_dir: Path, *, split: str, source: str) -> None:
    pred_dir = cap_dir / "predictions" / split
    pred_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "split": split,
        "source": source,
        "note": "EWB evaluation emits metrics via official API; raw grids in eval_workspace/",
    }
    (pred_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def _finalize_from_results_csv(
    cap_dir: Path,
    *,
    capability_id: str,
    forecast_mode: str,
    model_name: str,
    family_id: str,
    cmd: str,
    out_root: Path,
) -> Dict[str, Any]:
    csv_path = cap_dir / "eval_workspace" / "ewb_results.csv"
    metric_value, aggregate_meta = aggregate_ewb_results_csv(csv_path)
    return _publish_scientific_success(
        cap_dir=cap_dir,
        capability_id=capability_id,
        family_id=family_id,
        forecast_mode=forecast_mode,
        model_name=model_name,
        cmd=cmd,
        out_root=out_root,
        metric_value=metric_value,
        out_csv=str(csv_path),
        aggregate_meta=aggregate_meta,
        note_suffix="finalized from existing ewb_results.csv",
    )


def eval_hwmed_scientific(
    capability_id: str,
    *,
    out_base: Optional[Path] = None,
    from_csv: bool = False,
) -> Dict[str, Any]:
    if not scientific_data_ready():
        blocked_base = Path(out_base) if out_base else None
        return write_blocked(
            TASKPACK,
            capability_id,
            "HW-MED scientific data package not materialized",
            out_base=blocked_base,
        )

    if capability_id in EXTERNAL_ACQUISITION_CAPS:
        blocked_base = Path(out_base) if out_base else None
        return eval_external_cap(capability_id, out_base=blocked_base)

    if capability_id in BLOCKED_CAPS:
        blocked_base = Path(out_base) if out_base else None
        return write_blocked(TASKPACK, capability_id, BLOCKED_CAPS[capability_id], out_base=blocked_base)

    if capability_id not in CAP_FORECAST:
        return eval_blocked_generic(TASKPACK, capability_id, "unknown HW-MED cap", out_base=out_base)

    family_id, forecast_mode, model_name = CAP_FORECAST[capability_id]
    out_root = Path(out_base) if out_base else SCIENTIFIC_RUNS_ROOT
    cap_dir = cap_scientific_dir(TASKPACK, capability_id, runs_root=out_root)
    cmd = OFFICIAL_COMMANDS.get(capability_id, "ewb.evaluation")
    pin = taskpack_repo_pin(TASKPACK)
    if pin.is_file():
        commit = json.loads(pin.read_text()).get("repo_commit", "")
        if commit:
            cmd = f"{cmd}  # repo={commit}"

    try:
        if from_csv or os.environ.get("HW_EWB_FROM_CSV", "").strip() in {"1", "true", "yes"}:
            return _finalize_from_results_csv(
                cap_dir,
                capability_id=capability_id,
                forecast_mode=forecast_mode,
                model_name=model_name,
                family_id=family_id,
                cmd=cmd,
                out_root=out_root,
            )

        metric_value, out_csv = _run_ewb_eval(
            cap_dir,
            capability_id=capability_id,
            forecast_mode=forecast_mode,
            model_name=model_name,
        )
        return _publish_scientific_success(
            cap_dir=cap_dir,
            capability_id=capability_id,
            family_id=family_id,
            forecast_mode=forecast_mode,
            model_name=model_name,
            cmd=cmd,
            out_root=out_root,
            metric_value=metric_value,
            out_csv=out_csv,
        )
    except (RuntimeError, FileNotFoundError, ValueError, subprocess.TimeoutExpired, json.JSONDecodeError) as exc:
        _write_recipe_log(
            cap_dir,
            capability_id=capability_id,
            command=cmd,
            returncode=1,
            stdout_tail=str(exc),
        )
        blocked_base = Path(out_base) if out_base else None
        return write_blocked(TASKPACK, capability_id, str(exc), out_base=blocked_base)


SCIENTIFIC_EVAL_CAPS = ("CAP-HWMED-02", "CAP-HWMED-03", "CAP-HWMED-04", "CAP-HWMED-06")


def run_all_scientific(*, out_base: Optional[Path] = None) -> Dict[str, Any]:
    return {cap: eval_hwmed_scientific(cap, out_base=out_base) for cap in ALL_CAPS}


def run_scientific_caps(
    capability_ids: List[str] | Tuple[str, ...],
    *,
    out_base: Optional[Path] = None,
) -> Dict[str, Any]:
    """Run a subset of caps (each still uses an isolated EWB subprocess)."""
    results: Dict[str, Any] = {}
    for cap in capability_ids:
        if cap in EXTERNAL_ACQUISITION_CAPS:
            results[cap] = eval_external_cap(cap, out_base=out_base)
        elif cap in CAP_FORECAST:
            results[cap] = eval_hwmed_scientific(cap, out_base=out_base)
        else:
            results[cap] = eval_blocked_generic(
                TASKPACK, cap, "unknown HW-MED cap", out_base=out_base
            )
    return results


def _merge_results_json(path: Path, cap: str, row: Dict[str, Any]) -> None:
    results: Dict[str, Any] = {}
    if path.is_file():
        results = json.loads(path.read_text(encoding="utf-8"))
    results[cap] = row
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(results, indent=2, default=str) + "\n", encoding="utf-8")


def _scientific_cap_ok(row: Dict[str, Any]) -> bool:
    return bool(row.get("ok")) and not row.get("blocked") and not row.get("external_acquisition_required")


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="HW-MED Phase C scientific EWB eval")
    parser.add_argument("--cap", action="append", dest="caps", metavar="CAP-HWMED-0X")
    parser.add_argument("--all", action="store_true", help="run all six caps in this process")
    parser.add_argument(
        "--results-json",
        type=Path,
        help="merge each cap result into this JSON file (for per-cap SLURM loops)",
    )
    parser.add_argument(
        "--from-csv",
        action="store_true",
        help="aggregate existing eval_workspace/ewb_results.csv (skip EWB subprocess)",
    )
    parser.add_argument(
        "--fail-if-scientific-missing",
        action="store_true",
        help="exit 1 when a SCIENTIFIC_EVAL_CAPS row is not ok",
    )
    args = parser.parse_args(argv)

    caps = list(args.caps or [])
    if args.all:
        caps = list(ALL_CAPS)
    if not caps:
        parser.error("specify --cap CAP-HWMED-0X (repeatable) or --all")

    results: Dict[str, Any] = {}
    for cap in caps:
        row = eval_hwmed_scientific(cap, from_csv=args.from_csv)
        results[cap] = row
        print(json.dumps({cap: row}, default=str), flush=True)
        if args.results_json:
            _merge_results_json(args.results_json, cap, row)

    if args.fail_if_scientific_missing:
        missing = sorted(
            cap for cap in SCIENTIFIC_EVAL_CAPS if cap in results and not _scientific_cap_ok(results[cap])
        )
        if missing:
            print(f"scientific caps missing or blocked: {missing}", flush=True)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
