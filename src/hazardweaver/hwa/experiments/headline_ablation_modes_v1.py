"""Env flags for RQ2 component ablations (E4–E6) on headline Agent-Strict."""

from __future__ import annotations

import os

HCG_COMPAT_FULL = "full"
HCG_COMPAT_INTERFACE_ONLY = "interface_only"
HCG_COMPAT_UNTYPED = "untyped"

SRC_GATE_ON = "on"
SRC_GATE_OFF = "off"

HKC_SCI_FULL = "full"
HKC_SCI_FORCE_APPLICABLE = "force_applicable"


def hcg_compat_mode() -> str:
    raw = (
        os.environ.get("HWA_HCG_COMPAT_MODE")
        or os.environ.get("ICLR_HWA_HCG_COMPAT_MODE")
        or HCG_COMPAT_FULL
    ).strip().lower()
    if raw in {HCG_COMPAT_INTERFACE_ONLY, "interface", "shallow"}:
        return HCG_COMPAT_INTERFACE_ONLY
    if raw in {HCG_COMPAT_UNTYPED, "no_semantics", "shallow_typed"}:
        return HCG_COMPAT_UNTYPED
    return HCG_COMPAT_FULL


def hcg_interface_only() -> bool:
    return hcg_compat_mode() == HCG_COMPAT_INTERFACE_ONLY


def hcg_untyped() -> bool:
    """Round2 HCG ablation: skip 8-field semantic compatibility; keep executability."""
    return hcg_compat_mode() == HCG_COMPAT_UNTYPED


def src_eligibility_gate_mode() -> str:
    raw = (
        os.environ.get("HWA_SRC_ELIGIBILITY_GATE")
        or os.environ.get("ICLR_HWA_SRC_ELIGIBILITY_GATE")
        or SRC_GATE_ON
    ).strip().lower()
    if raw in {SRC_GATE_OFF, "0", "false", "no"}:
        return SRC_GATE_OFF
    return SRC_GATE_ON


def src_eligibility_gate_off() -> bool:
    return src_eligibility_gate_mode() == SRC_GATE_OFF


def hkc_sci_applicability_mode() -> str:
    raw = (
        os.environ.get("HWA_HKC_SCI_ABLATION")
        or os.environ.get("ICLR_HWA_HKC_SCI_ABLATION")
        or HKC_SCI_FULL
    ).strip().lower()
    if raw in {HKC_SCI_FORCE_APPLICABLE, "off", "all_applicable", "applicable"}:
        return HKC_SCI_FORCE_APPLICABLE
    return HKC_SCI_FULL


def hkc_sci_force_applicable() -> bool:
    """Legacy E4 only — Round2 uses HWA_HKC_REGISTRY=none instead."""
    if hkc_registry_disabled():
        return False
    return hkc_sci_applicability_mode() == HKC_SCI_FORCE_APPLICABLE


def hkc_registry_disabled() -> bool:
    from hazardweaver.hwa.scientific_controller.hkc_frozen_registry_v1 import hkc_registry_disabled as _off

    return _off()


def hkc_registry_enabled() -> bool:
    from hazardweaver.hwa.scientific_controller.hkc_frozen_registry_v1 import hkc_registry_enabled as _on

    return _on()


ROUTE_ELIGIBILITY_DYNAMIC = "dynamic"
ROUTE_ELIGIBILITY_STATIC_S0 = "static_s0"
ROUTE_ELIGIBILITY_STATIC_WHITELIST_S0 = "static_whitelist_s0"


def route_eligibility_mode() -> str:
    raw = (
        os.environ.get("HWA_ROUTE_ELIGIBILITY_MODE")
        or os.environ.get("ICLR_ROUTE_ELIGIBILITY_MODE")
        or ROUTE_ELIGIBILITY_DYNAMIC
    ).strip().lower()
    if raw in {
        ROUTE_ELIGIBILITY_STATIC_WHITELIST_S0,
        "static_whitelist",
        "whitelist_s0",
        "static_wl_s0",
    }:
        return ROUTE_ELIGIBILITY_STATIC_WHITELIST_S0
    if raw in {ROUTE_ELIGIBILITY_STATIC_S0, "static", "frozen_s0", "s0_only"}:
        return ROUTE_ELIGIBILITY_STATIC_S0
    return ROUTE_ELIGIBILITY_DYNAMIC


def route_eligibility_static_s0() -> bool:
    return route_eligibility_mode() == ROUTE_ELIGIBILITY_STATIC_S0


def route_eligibility_static_whitelist_s0() -> bool:
    return route_eligibility_mode() == ROUTE_ELIGIBILITY_STATIC_WHITELIST_S0


def route_eligibility_static_frozen() -> bool:
    """Any static eligibility-freeze ablation arm (legacy replay or whitelist-s0)."""
    return route_eligibility_static_s0() or route_eligibility_static_whitelist_s0()
