"""Utility module."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from hazardweaver.hcg.carp.batch2 import FIXTURES_ROOT

WF3_SMOKE = FIXTURES_ROOT / "wf3_persistence_smoke"
E1E3_SMOKE = FIXTURES_ROOT / "e1e3_pick_smoke"


def ensure_wf3_smoke_fixture() -> Path:
    WF3_SMOKE.mkdir(parents=True, exist_ok=True)
    grids = WF3_SMOKE / "grids"
    grids.mkdir(exist_ok=True)
    t0 = np.array([[1.0, 1.0, 0.0, 0.0], [1.0, 0.5, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    t1 = np.array([[1.0, 1.0, 0.5, 0.0], [1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
    np.save(grids / "fire0_t0.npy", t0)
    np.save(grids / "fire0_t1.npy", t1)
    manifest = {
        "source": "synthetic_smoke",
        "synthetic_only": True,
        "fires": [
            {
                "fire_id": "fire0",
                "pairs": [{"t0": "grids/fire0_t0.npy", "t1": "grids/fire0_t1.npy"}],
            }
        ],
    }
    (WF3_SMOKE / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return WF3_SMOKE / "manifest.json"


def ensure_e1e3_smoke_fixture() -> Path:
    E1E3_SMOKE.mkdir(parents=True, exist_ok=True)
    # 30 s trace @ 100 Hz, simple pulse + noise for PhaseNet smoke
    sr = 100.0
    n = 3000
    t = np.arange(n) / sr
    wave = 0.3 * np.sin(2 * np.pi * 4 * t) + 0.05 * np.random.default_rng(42).standard_normal(n)
    wave = wave.astype(np.float32)
    np.save(E1E3_SMOKE / "trace.npy", wave)
    meta = {
        "source": "synthetic_smoke",
        "synthetic_only": True,
        "sample_rate": sr,
        "p_pick_s": 10.0,
        "s_pick_s": 20.0,
        "tolerance_s": 0.5,
        "models": {
            "CAP-E1E3-02": {"store": "phasenet", "weight_variant": "original", "class": "PhaseNet"},
            "CAP-E1E3-03": {"store": "eqtransformer", "weight_variant": "original", "class": "EQTransformer"},
            "CAP-E1E3-04": {"store": "gpd", "weight_variant": "stead", "class": "GPD"},
        },
    }
    (E1E3_SMOKE / "fixture.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    return E1E3_SMOKE / "fixture.json"
