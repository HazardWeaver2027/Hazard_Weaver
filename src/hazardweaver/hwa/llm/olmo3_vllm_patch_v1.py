"""Prepare Olmo-3.1 HF snapshot for vLLM 0.25 + transformers Olmo3Config."""

from __future__ import annotations

import json
from glob import glob
from pathlib import Path
from typing import Any, Dict, Mapping

ROOT = Path(__file__).resolve().parents[3]
MARKER = ".vllm025_rope_parameters_patch_v4"
_LEGACY_MARKERS = (
    ".vllm025_rope_parameters_restore_v1",
    ".vllm025_rope_parameters_patch_v2",
    ".vllm025_rope_parameters_patch_v3",
)


def snapshot_config_path(snapshot_glob: str) -> Path:
    matches = sorted(glob(str(ROOT / snapshot_glob)))
    for snap in matches:
        cfg = Path(snap) / "config.json"
        if cfg.is_file():
            return cfg
    raise FileNotFoundError(f"no config.json for snapshot_glob={snapshot_glob!r}")


def _flat_rope_leaf(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    rope_theta = cfg.get("rope_theta")
    if rope_theta is None:
        raise ValueError("config missing rope_theta")
    scaling = dict(cfg.get("rope_scaling") or {})
    rope_type = str(scaling.pop("rope_type", None) or scaling.pop("type", None) or "default")
    out: Dict[str, Any] = {"rope_type": rope_type, "rope_theta": float(rope_theta)}
    for key, val in scaling.items():
        if key not in out and not isinstance(val, dict):
            out[key] = val
    return out


def rope_parameters_has_nested_dicts(rope_parameters: Any) -> bool:
    if not isinstance(rope_parameters, dict):
        return False
    return any(isinstance(val, dict) for val in rope_parameters.values())


def rope_parameters_vllm_safe(rope_parameters: Any) -> bool:
    if not isinstance(rope_parameters, dict):
        return False
    if rope_parameters.get("rope_theta") is None:
        return False
    return not rope_parameters_has_nested_dicts(rope_parameters)


def build_rope_parameters(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Nested Olmo3 rope_parameters for transformers-only callers/tests."""
    existing = cfg.get("rope_parameters")
    if isinstance(existing, dict) and isinstance(existing.get("full_attention"), dict):
        leaf = dict(existing["full_attention"])
    elif isinstance(existing, dict) and existing.get("rope_theta") is not None:
        leaf = {
            k: v
            for k, v in existing.items()
            if k not in {"full_attention", "sliding_attention"} and not isinstance(v, dict)
        }
    else:
        leaf = _flat_rope_leaf(cfg)
    return {
        "full_attention": dict(leaf),
        "sliding_attention": {
            "rope_type": "default",
            "rope_theta": float(leaf["rope_theta"]),
        },
    }


def build_vllm_compatible_rope_parameters(cfg: Mapping[str, Any]) -> Dict[str, Any]:
    """Nested Olmo3 rope_parameters plus top-level rope_theta for vLLM Olmo2Attention."""
    nested = build_rope_parameters(cfg)
    theta = nested["full_attention"].get("rope_theta")
    if theta is None:
        theta = nested["sliding_attention"].get("rope_theta")
    if theta is None:
        raise ValueError("config missing rope_theta for vLLM rope_parameters patch")
    out = dict(nested)
    out["rope_theta"] = float(theta)
    return out


def rope_parameters_vllm_compat(rope_parameters: Any) -> bool:
    if not isinstance(rope_parameters, dict):
        return False
    if rope_parameters.get("rope_theta") is None:
        return False
    full = rope_parameters.get("full_attention")
    sliding = rope_parameters.get("sliding_attention")
    return isinstance(full, dict) and isinstance(sliding, dict)


def ensure_olmo3_rope_parameters_compat(config: Any) -> None:
    """Inject top-level rope_theta when transformers built nested rope_parameters only."""
    rp = getattr(config, "rope_parameters", None)
    if not isinstance(rp, dict) or rp.get("rope_theta") is not None:
        return
    for key in ("sliding_attention", "full_attention"):
        leaf = rp.get(key)
        if isinstance(leaf, dict) and leaf.get("rope_theta") is not None:
            rp["rope_theta"] = float(leaf["rope_theta"])
            return
    top = getattr(config, "rope_theta", None)
    if top is not None:
        rp["rope_theta"] = float(top)


def patch_olmo3_snapshot_config(snapshot_glob: str, *, dry_run: bool = False) -> Dict[str, Any]:
    """Write vLLM-compatible nested rope_parameters (with top-level rope_theta) to HF snapshot."""
    path = snapshot_config_path(snapshot_glob)
    cfg = json.loads(path.read_text(encoding="utf-8"))
    marker = path.parent / MARKER
    legacy_markers = [path.parent / name for name in _LEGACY_MARKERS]
    had_legacy = any(m.is_file() for m in legacy_markers)
    desired = build_vllm_compatible_rope_parameters(cfg)
    current = cfg.get("rope_parameters")
    already_ok = marker.is_file() and rope_parameters_vllm_compat(current) and not had_legacy
    if already_ok:
        return {"path": str(path), "status": "already_patched"}

    cfg["rope_parameters"] = desired
    report: Dict[str, Any] = {
        "path": str(path.relative_to(ROOT)),
        "status": "patched",
        "rope_theta": desired["rope_theta"],
        "runtime_patch": "hwa.llm.vllm_cli_entry_v1",
    }
    if not dry_run:
        path.write_text(json.dumps(cfg, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        marker.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        for legacy in legacy_markers:
            if legacy.is_file():
                legacy.unlink()
    return report


def restore_olmo3_snapshot_config(snapshot_glob: str, *, dry_run: bool = False) -> Dict[str, Any]:
    """Backward-compatible alias for ensure_olmo3_vllm_config_v1."""
    return patch_olmo3_snapshot_config(snapshot_glob, dry_run=dry_run)
