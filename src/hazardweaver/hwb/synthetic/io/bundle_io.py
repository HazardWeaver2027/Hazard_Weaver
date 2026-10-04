"""Scenario bundle read/write."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from hazardweaver.hwb.synthetic import __version__
from hazardweaver.hwb.synthetic.schemas import GenerationResult, MechanismCard, ScenarioBundleManifest, ScenarioSpec


def _hash_spec(spec: ScenarioSpec) -> str:
    payload = spec.model_dump_json()
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


def write_bundle(
    out_dir: Path,
    spec: ScenarioSpec,
    card: MechanismCard,
    result: GenerationResult,
) -> ScenarioBundleManifest:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "fields").mkdir(exist_ok=True)
    (out_dir / "hidden").mkdir(exist_ok=True)
    (out_dir / "counterfactuals").mkdir(exist_ok=True)
    (out_dir / "evaluation").mkdir(exist_ok=True)
    (out_dir / "provenance").mkdir(exist_ok=True)

    np.savez_compressed(out_dir / "fields" / "exposed_features.npz", **result.exposed)
    np.savez_compressed(out_dir / "fields" / "source_fields.npz", **{
        k: v for k, v in result.fields.items() if k in ("flood_depth", "fire_prob", "tc_track", "burn")
    } or result.fields)
    np.savez_compressed(out_dir / "hidden" / "labels.npz", labels=result.labels)

    hidden_extra = {k: v for k, v in result.hidden.items()}
    np.savez_compressed(out_dir / "hidden" / "generator_state.npz", **hidden_extra)

    with open(out_dir / "hidden" / "counterfactual_metadata.json", "w", encoding="utf-8") as f:
        json.dump(
            {name: {"dependency": result.dependency_scores.get(name)} for name in result.counterfactuals},
            f,
            indent=2,
        )

    for cf_name, cf_data in result.counterfactuals.items():
        cf_dir = out_dir / "counterfactuals" / cf_name
        cf_dir.mkdir(exist_ok=True)
        np.savez_compressed(cf_dir / "labels.npz", labels=cf_data["labels"])
        if "fields" in cf_data:
            np.savez_compressed(cf_dir / "fields.npz", **cf_data["fields"])

    public_spec = spec.model_dump()
    for key in ("hidden_fields",):
        public_spec.pop(key, None)

    with open(out_dir / "scenario_spec.json", "w", encoding="utf-8") as f:
        json.dump(public_spec, f, indent=2)

    with open(out_dir / "mechanism.json", "w", encoding="utf-8") as f:
        json.dump(card.model_dump(mode="json"), f, indent=2)

    mech_hash = _hash_spec(spec)
    manifest = ScenarioBundleManifest(
        scenario_id=spec.scenario_id,
        mechanism_id=spec.mechanism_id,
        seed=spec.seed,
        generator_version=__version__,
        mechanism_spec_hash=mech_hash,
        proxy_only=card.proxy_only,
        counterfactual_names=list(result.counterfactuals.keys()),
        exposed_field_names=list(result.exposed.keys()),
        label_type=spec.label_rule.type,
        n_samples=spec.n_samples,
    )

    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest.model_dump(), f, indent=2)

    grader_cfg = {
        "evaluation_metrics": spec.evaluation_metrics,
        "label_type": spec.label_rule.type,
        "dependency_scores": result.dependency_scores,
    }
    with open(out_dir / "evaluation" / "grader_config.json", "w", encoding="utf-8") as f:
        json.dump(grader_cfg, f, indent=2)

    prov = out_dir / "provenance"
    (prov / "generator_version.txt").write_text(__version__, encoding="utf-8")
    (prov / "mechanism_spec_hash.txt").write_text(mech_hash, encoding="utf-8")
    (prov / "random_seed.txt").write_text(str(spec.seed), encoding="utf-8")

    return manifest


def load_bundle(bundle_dir: Path) -> Dict[str, Any]:
    bundle_dir = Path(bundle_dir)
    with open(bundle_dir / "manifest.json", encoding="utf-8") as f:
        manifest = json.load(f)
    exposed = dict(np.load(bundle_dir / "fields" / "exposed_features.npz"))
    labels = np.load(bundle_dir / "hidden" / "labels.npz")["labels"]
    return {"manifest": manifest, "exposed": exposed, "labels": labels}


def load_counterfactual_labels(bundle_dir: Path, cf_name: str) -> np.ndarray:
    path = Path(bundle_dir) / "counterfactuals" / cf_name / "labels.npz"
    return np.load(path)["labels"]
