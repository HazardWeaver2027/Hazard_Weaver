"""Paths and constants for expansion A2 tracks."""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]

EXPANSION_TRACKS = ("L2", "DR-OUT", "MH-1", "MH-2")

DEV_ROOTS = {
    "L2": PROJECT_ROOT / "data" / "processed" / "l2_hwb_dev_v1",
    "DR-OUT": PROJECT_ROOT / "data" / "processed" / "dr_out_hwb_dev_v1",
    "MH-1": PROJECT_ROOT / "data" / "processed" / "mh1_hwb_dev_v1",
    "MH-2": PROJECT_ROOT / "data" / "processed" / "mh2_hwb_dev_v1",
}

PIN_ROOTS = {
    "L2": PROJECT_ROOT / "data" / "vendor" / "lhasa",
    "DR-OUT": PROJECT_ROOT / "data" / "vendor" / "cpc_sdo",
    "MH-1": PROJECT_ROOT / "data" / "vendor" / "ocelote_pfdf",
    "MH-2": PROJECT_ROOT / "data" / "vendor" / "groundfailure",
}

G2_TRAIN_CAPS = frozenset(
    {
        "CAP-L2-03",
        "CAP-L2-04",
        "CAP-DROUT-04",
        "CAP-DROUT-05",
        "CAP-MH1-04",
        "CAP-MH1-05",
        "CAP-MH2-04",
        "CAP-MH2-05",
        "CAP-MH2-06",
    }
)


def expansion_dev_root(taskpack_id: str) -> Path:
    return DEV_ROOTS[taskpack_id]


def pi_signoff_path(taskpack_id: str) -> Path:
    return expansion_dev_root(taskpack_id) / "PI_HOLDOUT_SIGNOFF.json"


def pin_manifest_path(taskpack_id: str) -> Path:
    root = PIN_ROOTS[taskpack_id]
    for name in ("CHECKPOINT_PIN.json", "FETCH_PIN.json", "REPLAY_PIN.json"):
        p = root / name
        if p.is_file():
            return p
    return root / "CHECKPOINT_PIN.json"
