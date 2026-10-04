"""Isolated subprocess worker for official USGS groundfailure model replay (MH-2)."""

from __future__ import annotations

import json
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from hazardweaver.hcg.carp.scientific.mh2_data import groundfailure_repo_dir
from hazardweaver.hcg.carp.scientific.mh2_fidelity import REPLAY_PARITY_MIN, parity_ok
from hazardweaver.hcg.carp.scientific.mh2_gfail_env import gfail_model_inputs_dir
from hazardweaver.hcg.carp.scientific.mh2_reference import (
    load_reference_prob,
    replay_parity,
    write_reference_manifest,
)

MODEL_SPECS: Dict[str, Dict[str, str]] = {
    "nowicki_jessee_2018": {
        "factory_key": "jessee_2018",
        "section": "jessee_2018",
        "config_relpath": "defaultconfigfiles/models/jessee_2018.ini",
        "runner": "logistic",
    },
    "nowicki_2014": {
        "factory_key": "nowicki_2014_global",
        "section": "nowicki_2014_global",
        "config_relpath": "defaultconfigfiles/models/nowicki_2014_global.ini",
        "runner": "logistic",
    },
    "godt_2008": {
        "factory_key": "godt2008",
        "section": "godt_2008",
        "config_relpath": "defaultconfigfiles/models/godt_2008.ini",
        "runner": "godt",
    },
    "newmark": {
        "factory_key": "godt2008",
        "section": "godt_2008",
        "config_relpath": "defaultconfigfiles/models/godt_2008.ini",
        "runner": "newmark",
    },
}


def _execute_gfail_subprocess(
    *,
    shakefile: Path,
    model_key: str,
    out_dir: Path,
    reference_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    repo = groundfailure_repo_dir()
    if not repo.is_dir():
        return {"ok": False, "error": f"missing groundfailure repo at {repo}"}

    spec = MODEL_SPECS.get(model_key)
    if spec is None:
        return {"ok": False, "error": f"unknown model_key {model_key}"}

    config_path = repo / spec["config_relpath"]
    if not config_path.is_file():
        return {"ok": False, "error": f"missing config {config_path}"}

    datapath = gfail_model_inputs_dir()
    if not (datapath / "global_grad.tif").is_file():
        return {
            "ok": False,
            "error": f"missing gfail model_inputs at {datapath}",
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    prob_out = out_dir / f"{model_key}_prob.npy"

    script = textwrap.dedent(
        f"""
        import json
        from pathlib import Path
        import numpy as np
        from configobj import ConfigObj
        from gfail.utilities import correct_config_filepaths

        shakefile = {str(shakefile)!r}
        config_path = {str(config_path)!r}
        datapath = {str(datapath)!r}
        prob_out = Path({str(prob_out)!r})
        model_key = {model_key!r}
        section_name = {spec["section"]!r}
        runner = {spec["runner"]!r}

        full = ConfigObj(config_path)
        correct_config_filepaths(datapath, full)
        config = full[section_name]
        if section_name == "nowicki_2014_global":
            config["layers"]["slope"]["file"] = str(Path(datapath) / "global_grad.tif")

        def _fixture_bounds():
            from mapio.shake import ShakeGrid
            from mapio.reader import get_file_geodict
            shake_gd = ShakeGrid.getFileGeoDict(shakefile)
            fix_gd = get_file_geodict(str(Path(datapath) / "global_grad.tif"))
            inter = shake_gd.getIntersection(fix_gd)
            if inter is None:
                return None
            return {{"xmin": inter.xmin, "xmax": inter.xmax, "ymin": inter.ymin, "ymax": inter.ymax}}

        def _godt_bounds():
            import os
            from mapio.shake import ShakeGrid
            from mapio.reader import get_file_geodict
            shake_gd = ShakeGrid.getFileGeoDict(shakefile)
            slope_path = full["godt_2008"]["layers"]["slope"]["filepath"]
            slpfile = os.path.join(slope_path, "slope_min.bil")
            base_gd = get_file_geodict(slpfile)
            inter = base_gd.getIntersection(shake_gd)
            if inter is None:
                return None
            return {{"xmin": inter.xmin, "xmax": inter.xmax, "ymin": inter.ymin, "ymax": inter.ymax}}

        bounds = _godt_bounds() if runner in ("godt", "newmark") else _fixture_bounds()
        if bounds is None:
            raise RuntimeError("ShakeMap has no overlap with staged gfail model_inputs")

        def _grid_data(layer):
            if hasattr(layer, "getData"):
                return np.array(layer.getData(), dtype=np.float32)
            if isinstance(layer, dict) and "grid" in layer:
                return _grid_data(layer["grid"])
            raise TypeError(f"unsupported layer type: {{type(layer)}}")

        if runner == "logistic":
            from gfail import Jessee2018Model, Nowicki2014Model
            cls = Jessee2018Model if section_name == "jessee_2018" else Nowicki2014Model
            model = cls(shakefile, config, bounds=bounds)
            layers = model.calculate()
            arr = _grid_data(layers["model"]).ravel()
            key = "model"
        elif runner == "godt":
            from gfail.models.godt import godt2008
            layers = godt2008(shakefile, full, bounds=bounds)
            arr = _grid_data(layers["model"]).ravel()
            key = "model"
        else:
            from gfail.models.godt import godt2008, NMdisp
            displmodel = config.get("displmodel", "J_PGA_M")
            layers = godt2008(shakefile, full, displmodel=displmodel, saveinputs=True, bounds=bounds)
            pga = _grid_data(layers["pga"])
            minfs = _grid_data(layers["minFS"])
            slope = _grid_data(layers["max slope"])
            ac = (minfs - 1.0) * np.sin(np.radians(slope))
            ac = np.maximum(ac, float(config.get("acthresh", 0.02)))
            pga_g = pga
            pgv = _grid_data(layers["pgv"]) if "pgv" in layers else None
            mag = float(config.get("magnitude", 7.0))
            try:
                from mapio.shake import ShakeGrid
                mag = float(ShakeGrid.load(shakefile).getEventDict().get("magnitude", mag))
            except Exception:
                pass
            dn, _, _ = NMdisp(ac, pga_g, model=displmodel, M=mag, PGV=pgv)
            arr = np.nanmax(dn, axis=2 if dn.ndim == 3 else 0).astype(np.float32).ravel()
            key = "newmark_displacement"

        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            raise RuntimeError("official gfail produced no finite cells")
        np.save(prob_out, arr)
        print(json.dumps({{"ok": True, "n_cells": int(arr.size), "layer": key}}))
        """
    )

    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=900,
        check=False,
    )
    if proc.returncode != 0:
        return {
            "ok": False,
            "error": "official gfail model subprocess failed",
            "stderr": (proc.stderr or "")[-2500:],
            "stdout": (proc.stdout or "")[-1000:],
        }

    if not prob_out.is_file():
        return {"ok": False, "error": f"missing output {prob_out}"}

    prob = np.load(prob_out).astype(np.float32)
    try:
        from hazardweaver.hcg.carp.scientific.mh2_gfail_env import gfail_version

        evaluator = f"gfail@{gfail_version() or '1.3.2'}"
    except Exception:
        evaluator = "gfail@1.3.2"

    return {
        "ok": True,
        "prob": prob,
        "prob_path": str(prob_out),
        "evaluator": evaluator,
        "model_key": model_key,
        "official_command": f"gfail MODEL_FACTORY[{spec['factory_key']}] shakefile={shakefile}",
    }


def run_gfail_model(
    *,
    shakefile: Path,
    model_key: str,
    out_dir: Path,
    reference_dir: Optional[Path] = None,
    pin_reference: bool = False,
    event_id: str = "",
) -> Dict[str, Any]:
    result = _execute_gfail_subprocess(
        shakefile=shakefile,
        model_key=model_key,
        out_dir=out_dir,
        reference_dir=reference_dir,
    )
    if not result.get("ok"):
        return result

    prob = result["prob"]
    ref_dir = reference_dir
    if pin_reference and ref_dir is not None:
        ref_dir.mkdir(parents=True, exist_ok=True)
        ref_path = ref_dir / f"{model_key}_prob.npy"
        np.save(ref_path, prob)
        write_reference_manifest(
            ref_dir,
            model_key=model_key,
            event_id=event_id or shakefile.parent.parent.name,
            reference_kind="pinned_official_gfail_replay_v1",
            prob_path=ref_path,
            extra={"shakefile": str(shakefile), "evaluator": result["evaluator"]},
        )
        result["reference_pinned"] = True
        result["reference_present"] = True
        result["reference_kind"] = "pinned_official_gfail_replay_v1"
        result["replay_parity"] = 1.0
        return result

    ref, ref_kind = load_reference_prob(ref_dir, model_key)
    parity, ref_present = replay_parity(prob, ref)
    result["reference_present"] = ref_present
    result["reference_kind"] = ref_kind
    result["replay_parity"] = parity

    if not ref_present:
        return {
            **result,
            "ok": False,
            "error": f"missing pinned reference for {model_key} under {ref_dir}",
        }
    if not parity_ok(parity):
        return {
            **result,
            "ok": False,
            "error": f"replay_parity {parity} < {REPLAY_PARITY_MIN}",
        }
    return result


def pin_gf_references_for_event(
    *,
    event_id: str,
    shakefile: Path,
    reference_dir: Path,
    model_keys: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Freeze official gfail outputs as replay references (materialize/bootstrap)."""
    keys = model_keys or list(MODEL_SPECS)
    rows = {}
    for model_key in keys:
        ref_path = reference_dir / f"{model_key}_prob.npy"
        if ref_path.is_file():
            rows[model_key] = {"ok": True, "skipped": True, "path": str(ref_path)}
            continue
        out_dir = reference_dir.parent / "gfail" / model_key
        row = run_gfail_model(
            shakefile=shakefile,
            model_key=model_key,
            out_dir=out_dir,
            reference_dir=reference_dir,
            pin_reference=True,
            event_id=event_id,
        )
        rows[model_key] = row
        if not row.get("ok"):
            return {"ok": False, "event_id": event_id, "models": rows}
    return {"ok": True, "event_id": event_id, "models": rows}


def run_gfail_models_for_events(
    events: List[Tuple[str, Path, Path]],
    model_key: str,
) -> List[Dict[str, Any]]:
    results = []
    for event_id, shakefile, ref_dir in events:
        out_dir = shakefile.parent.parent / "gfail" / model_key
        row = run_gfail_model(
            shakefile=shakefile,
            model_key=model_key,
            out_dir=out_dir,
            reference_dir=ref_dir,
        )
        row["event_id"] = event_id
        results.append(row)
    return results
