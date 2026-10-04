"""Canonical HKC pilot slice paths for HWA/HCG downstream (DL-063, DL-107)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, FrozenSet, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[3]

PILOT_BULK_FAMILIES = frozenset(
    {
        "fl2_v1",
        "pfdf_v1",
        "wf3_process_v1",
        "tc_tctrk_v1",
        "dr_out_v1",
    }
)

SEVEN_TRACK_FAMILIES = frozenset(
    {
        "wf3_process_v1",
        "l2_v1",
        "eq_e1e3_v1",
        "hw_med_v1",
        "mh2_v1",
        "mh3_v1",
        "mh4_v1",
    }
)

PILOT_FAMILIES = frozenset(PILOT_BULK_FAMILIES | SEVEN_TRACK_FAMILIES)

# Taskpack label → HKC family_id (seven-track binding)
SEVEN_TRACK_TASKPACK_TO_FAMILY: Dict[str, str] = {
    "WF-3": "wf3_process_v1",
    "L2": "l2_v1",
    "E1-E3": "eq_e1e3_v1",
    "HW-MED": "hw_med_v1",
    "MH-2": "mh2_v1",
    "MH-3": "mh3_v1",
    "MH-4": "mh4_v1",
}

FAMILY_TO_SLICE_REL: Dict[str, Path] = {
    "wf3_process_v1": Path("runs/eskc_compiler/route_cards/wf3_process_v1_slice_v1.jsonl"),
    "l2_v1": Path("runs/eskc_compiler/route_cards/l2_v1_slice_v1.jsonl"),
    "eq_e1e3_v1": Path("runs/eskc_compiler/route_cards/eq_e1e3_v1_slice_v1.jsonl"),
    "hw_med_v1": Path("runs/eskc_compiler/route_cards/hw_med_v1_slice_v1.jsonl"),
    "mh2_v1": Path("runs/eskc_compiler/route_cards/mh2_v1_slice_v1.jsonl"),
    "mh3_v1": Path("runs/eskc_compiler/route_cards/mh3_v1_slice_v1.jsonl"),
    "mh4_v1": Path("runs/eskc_compiler/route_cards/mh4_v1_slice_v1.jsonl"),
}

FAMILY_TO_CONTRACTS_REL: Dict[str, Path] = {
    fam: Path(f"runs/eskc_compiler/route_contracts/{fam}_slice_contracts_v1.jsonl")
    for fam in SEVEN_TRACK_FAMILIES
}

PILOT_ROUTE_CARDS_REL = Path("runs/eskc_compiler/route_cards/pilot_slice_v1.jsonl")
PILOT_CONTRACTS_REL = Path("runs/eskc_compiler/route_contracts/pilot_slice_contracts_v1.jsonl")
PILOT_EVIDENCE_REL = Path("runs/eskc_compiler/route_cards/pilot_slice_evidence_v1.jsonl")
PILOT_MANIFEST_REL = Path("runs/eskc_compiler/route_cards/pilot_slice_manifest_v1.json")

LEGACY_ANNOTATION_ROUTE_CARDS = Path(
    "benchmark/public/hkc_data/08_ANNOTATION_PACKAGES/route_cards_compiled_pilot_v1.jsonl"
)


@dataclass(frozen=True)
class HKCPilotPaths:
    route_cards: Path
    contracts: Path
    evidence: Path
    manifest: Path

    def route_cards_exists(self) -> bool:
        return self.route_cards.is_file()

    def contracts_exists(self) -> bool:
        return self.contracts.is_file()

    def configured(self) -> bool:
        return self.route_cards_exists() and self.contracts_exists()


def resolve_hkc_pilot_paths(root: Path | None = None) -> HKCPilotPaths:
    base = root or PROJECT_ROOT
    return HKCPilotPaths(
        route_cards=base / PILOT_ROUTE_CARDS_REL,
        contracts=base / PILOT_CONTRACTS_REL,
        evidence=base / PILOT_EVIDENCE_REL,
        manifest=base / PILOT_MANIFEST_REL,
    )


def resolve_family_slice_paths(
    family_id: str,
    root: Path | None = None,
) -> Tuple[Path, Path]:
    base = root or PROJECT_ROOT
    fid = str(family_id or "").strip()
    cards = base / FAMILY_TO_SLICE_REL.get(fid, Path(f"runs/eskc_compiler/route_cards/{fid}_slice_v1.jsonl"))
    contracts = base / FAMILY_TO_CONTRACTS_REL.get(
        fid, Path(f"runs/eskc_compiler/route_contracts/{fid}_slice_contracts_v1.jsonl")
    )
    return cards, contracts


def is_pilot_family(family_id: str) -> bool:
    return str(family_id or "").strip() in PILOT_FAMILIES


def is_seven_track_family(family_id: str) -> bool:
    return str(family_id or "").strip() in SEVEN_TRACK_FAMILIES


def pilot_assets_configured(root: Path | None = None) -> bool:
    return resolve_hkc_pilot_paths(root).configured()
