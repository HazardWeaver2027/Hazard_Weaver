"""Π_adm gate — HKC A_sci + HCG A_cap (tri-state preserved)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from hazardweaver.hcg.api import evaluate_reachability
from hazardweaver.hwa.runtime.hcg_certificates import attach_reachability_certificate
from hazardweaver.hwa.runtime.session_state import SessionState
from hazardweaver.hwa.scientific_controller.admissibility import evaluate_A_cap, evaluate_A_sci, is_admissible
from hazardweaver.hwa.route_controller.drout_pilot_slice import is_drout_pilot_task, resolve_hkc_family_id as resolve_drout_family_id
from hazardweaver.hwa.route_controller.fl2_pilot_slice import is_fl2_pilot_task, resolve_hkc_family_id as resolve_fl2_family_id
from hazardweaver.hwa.route_controller.pfdf_pilot_slice import is_pfdf_pilot_task, resolve_hkc_family_id as resolve_pfdf_family_id
from hazardweaver.hwa.route_controller.seven_track_pilot_slice import is_seven_track_pilot_task, resolve_seven_track_family_id
from hazardweaver.hwa.route_controller.tctrk_pilot_slice import is_tctrk_pilot_task, resolve_hkc_family_id as resolve_tctrk_family_id
from hazardweaver.hwa.scientific_controller.hkc_pilot_assets import is_pilot_family, pilot_assets_configured
from hazardweaver.hwa.scientific_controller.hkc_route_card import (
    RouteCardIndex,
    evaluate_A_sci_with_route_card,
    load_route_card_index,
)
from hazardweaver.hwa.scientific_controller.hkc_route_contract import (
    RouteContractIndex,
    evaluate_A_sci_with_contract,
    load_route_contract_index,
)
from hazardweaver.hwa.scientific_controller.reason_codes import ASciVerdict
from hazardweaver.hwa.experiments.headline_hkc_mode_v1 import headline_hkc_off
from hazardweaver.hwa.route_controller.route_metric_contract_v1 import apply_route_metric_contract


def resolve_hkc_family_id(
    task: Mapping[str, Any],
    route: Optional[Mapping[str, Any]] = None,
) -> str:
    """Unified HKC family resolution for pilot-track tasks."""
    if is_fl2_pilot_task(task):
        return resolve_fl2_family_id(task, route)
    if is_pfdf_pilot_task(task):
        return resolve_pfdf_family_id(task, route)
    if is_drout_pilot_task(task):
        return resolve_drout_family_id(task, route)
    if is_tctrk_pilot_task(task):
        return resolve_tctrk_family_id(task, route)
    if is_seven_track_pilot_task(task):
        return resolve_seven_track_family_id(task, route)
    if route is not None:
        for resolver in (
            resolve_fl2_family_id,
            resolve_pfdf_family_id,
            resolve_drout_family_id,
            resolve_tctrk_family_id,
            resolve_seven_track_family_id,
        ):
            fam = resolver(task, route)
            if fam:
                return fam
    return resolve_seven_track_family_id(task, route) or resolve_fl2_family_id(task, route)


def _family_id(route: Mapping[str, Any], task: Mapping[str, Any]) -> str:
    resolved = resolve_hkc_family_id(task, route)
    if resolved:
        return resolved
    return str(
        route.get("route_family_id")
        or route.get("hkc_family_id")
        or route.get("family_id")
        or (task.get("solver_visible") or {}).get("route_family_id")
        or task.get("task_family")
        or ""
    )


def _paper_id(route: Mapping[str, Any], task: Mapping[str, Any]) -> str:
    return str(
        route.get("paper_id")
        or route.get("hkc_paper_id")
        or (task.get("solver_visible") or {}).get("paper_id")
        or task.get("paper_id")
        or ""
    )


def _benchmark_curated_a_sci() -> Dict[str, Any]:
    return {
        "verdict": ASciVerdict.APPLICABLE.value,
        "codes": [],
        "refs": [],
        "source": "benchmark_curated",
    }


def _missing_binding_a_sci(code: str) -> Dict[str, Any]:
    return {
        "verdict": ASciVerdict.UNKNOWN_PENDING_THEORY.value,
        "codes": [code],
        "refs": [],
    }


def _annotate_g6_a_cap(
    out: Dict[str, Any],
    task: Mapping[str, Any],
    state: SessionState,
    *,
    graph: Any = None,
) -> Dict[str, Any]:
    from hazardweaver.hwa.experiments.g6_coreexec_controller_bridge import g6_requires_typed_acap

    from hazardweaver.hwa.experiments.unified_benchmark_execution_v1 import unified_e12_shock_s0_operational_a_cap

    shock_cap = unified_e12_shock_s0_operational_a_cap(task, out)
    if shock_cap is not None:
        out["A_cap"] = shock_cap
        return out
    if not out.get("g6_coreexec") or not g6_requires_typed_acap(task, out):
        preset = dict(out.get("A_cap") or {})
        if not preset:
            from hazardweaver.hwa.scientific_controller.reason_codes import ACapVerdict

            preset = {
                "verdict": ACapVerdict.REACHABLE.value,
                "codes": [],
                "missing_artifacts": [],
                "source": "allowed_edge_ids",
            }
        out["A_cap"] = preset
        return out
    out["A_cap"] = evaluate_A_cap(out, state, graph=graph)
    return out


class AdmissibilityGate:
    """Single source for A_sci / A_cap annotation on route candidates."""

    def __init__(
        self,
        task: Mapping[str, Any],
        *,
        route_card_index: Optional[RouteCardIndex] = None,
        route_card_path: Optional[Path] = None,
        contract_index: Optional[RouteContractIndex] = None,
        contract_path: Optional[Path] = None,
    ):
        self.task = dict(task)
        self.route_card_index = route_card_index
        if route_card_index is None and route_card_path and route_card_path.is_file():
            self.route_card_index = load_route_card_index(route_card_path)
        self.contract_index = contract_index
        if contract_index is None and contract_path and contract_path.is_file():
            self.contract_index = load_route_contract_index(contract_path)

    def _resolve_contract_row(self, route: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        if self.contract_index is None:
            return None
        fid = _family_id(route, self.task)
        item_id = str(route.get("item_id") or route.get("route_card_item_id") or "")
        if fid and item_id:
            row = self.contract_index.get(fid, item_id)
            if row is not None:
                return row
        paper_id = _paper_id(route, self.task)
        if fid and paper_id:
            return self.contract_index.get_by_paper(fid, paper_id)
        return None

    def _route_card_for(self, route: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
        if self.route_card_index is None:
            return None
        fid = _family_id(route, self.task)
        item_id = str(route.get("item_id") or route.get("route_card_item_id") or "")
        if fid and item_id:
            card = self.route_card_index.get(fid, item_id)
            if card is not None:
                return card
        paper_id = _paper_id(route, self.task)
        if fid and paper_id:
            cards = self.route_card_index.for_paper(fid, paper_id)
            if cards:
                return cards[0]
        return None

    def annotate_route(
        self,
        route: Mapping[str, Any],
        state: SessionState,
        *,
        graph: Any = None,
    ) -> Dict[str, Any]:
        try:
            from hazardweaver.hwa.experiments.ablation_manual_pilot_v1 import (
                _apply_route_overlays,
                ablation_manual_pilot_enabled,
                annotate_manual_pilot_route,
            )

            if ablation_manual_pilot_enabled(self.task):
                patched = _apply_route_overlays([dict(route)], self.task)[0]
                return annotate_manual_pilot_route(patched, self.task, state)
        except ImportError:
            pass
        out = dict(route)
        theory_arm = getattr(state, "theory_arm", "verified")
        fid = _family_id(out, self.task)
        contract_row = self._resolve_contract_row(out)
        card = None if contract_row is not None else self._route_card_for(out)

        binding_source = "dev_stub"
        try:
            from hazardweaver.hwa.scientific_controller.hkc_frozen_registry_v1 import (
                evaluate_a_sci_round2,
                get_frozen_hkc_registry,
                hkc_registry_disabled,
                hkc_registry_enabled,
            )

            if hkc_registry_enabled() or hkc_registry_disabled():
                if hkc_registry_enabled():
                    frozen = get_frozen_hkc_registry()
                    self.contract_index = frozen.contract_index
                    contract_row = self._resolve_contract_row(out)
                else:
                    contract_row = None
                if hkc_registry_enabled() and self.contract_index is not None:
                    from hazardweaver.hwa.scientific_controller.hkc_frozen_registry_v1 import resolve_frozen_contract_row

                    contract_row = resolve_frozen_contract_row(
                        out,
                        self.task,
                        self.contract_index,
                        explicit_row=contract_row,
                    )
                a_sci = evaluate_a_sci_round2(
                    out,
                    self.task,
                    contract_row=contract_row,
                    contract_index=self.contract_index if hkc_registry_enabled() else None,
                    theory_arm=theory_arm,
                )
                binding_source = str(a_sci.get("source") or "frozen_k_hkc_v1")
                out["A_sci"] = a_sci
                from hazardweaver.hwa.experiments.unified_benchmark_execution_v1 import (
                    apply_unified_g6_operational_admissibility,
                )

                apply_unified_g6_operational_admissibility(self.task, out)
                if str((out.get("A_sci") or {}).get("source") or "").startswith("unified_"):
                    binding_source = str(out["A_sci"].get("source") or "unified_g6_operational")
                if contract_row is not None:
                    out["scientific_route_contract_id"] = contract_row.get("contract_id")
                if out.get("g6_coreexec"):
                    out = _annotate_g6_a_cap(out, self.task, state, graph=graph)
                else:
                    cert = evaluate_reachability(out, state, graph=graph)
                    out = attach_reachability_certificate(out, cert)
                out["admissible"] = is_admissible(out["A_sci"], out["A_cap"])
                try:
                    from hazardweaver.hwa.experiments.headline_ablation_modes_v1 import hcg_interface_only, hcg_untyped

                    if hcg_interface_only():
                        a_cap = out.get("A_cap") or {}
                        verdict = str(a_cap.get("verdict") or "").upper()
                        out["admissible"] = verdict in {"REACHABLE", "OK", "PASS"} or bool(out.get("g6_coreexec"))
                        out["hcg_compat_mode"] = "interface_only"
                    elif hcg_untyped():
                        out["hcg_compat_mode"] = "untyped"
                except ImportError:
                    pass
                out = apply_route_metric_contract(out, self.task)
                out["hkc_binding_source"] = binding_source
                if not out.get("route_id"):
                    edges = out.get("edges") or out.get("capability_ids") or []
                    out["route_id"] = "route:" + ("+".join(edges[:3]) if edges else "unknown")
                return out
        except (ImportError, FileNotFoundError):
            pass
        try:
            from hazardweaver.hwa.experiments.headline_ablation_modes_v1 import hkc_sci_force_applicable

            if hkc_sci_force_applicable():
                a_sci = _benchmark_curated_a_sci()
                binding_source = "hkc_sci_ablation_force_applicable"
                out["A_sci"] = a_sci
                if out.get("g6_coreexec"):
                    out = _annotate_g6_a_cap(out, self.task, state, graph=graph)
                else:
                    cert = evaluate_reachability(out, state, graph=graph)
                    out = attach_reachability_certificate(out, cert)
                out["admissible"] = is_admissible(out["A_sci"], out["A_cap"])
                try:
                    from hazardweaver.hwa.experiments.headline_ablation_modes_v1 import hcg_interface_only

                    if hcg_interface_only():
                        a_cap = out.get("A_cap") or {}
                        verdict = str(a_cap.get("verdict") or "").upper()
                        out["admissible"] = verdict in {"REACHABLE", "OK", "PASS"} or bool(out.get("g6_coreexec"))
                        out["hcg_compat_mode"] = "interface_only"
                except ImportError:
                    pass
                out = apply_route_metric_contract(out, self.task)
                out["hkc_binding_source"] = binding_source
                if not out.get("route_id"):
                    edges = out.get("edges") or out.get("capability_ids") or []
                    out["route_id"] = "route:" + ("+".join(edges[:3]) if edges else "unknown")
                return out
        except ImportError:
            pass
        if headline_hkc_off(self.task):
            if contract_row is not None:
                a_sci = evaluate_A_sci_with_contract(out, self.task, contract_row, theory_arm=theory_arm)
                binding_source = "scientific_route_contract"
                out["scientific_route_contract_id"] = contract_row.get("contract_id")
            elif card is not None:
                a_sci = evaluate_A_sci_with_route_card(out, self.task, card, theory_arm=theory_arm)
                binding_source = "route_card"
            else:
                a_sci = _benchmark_curated_a_sci()
                binding_source = "benchmark_curated"
        elif contract_row is not None:
            a_sci = evaluate_A_sci_with_contract(out, self.task, contract_row, theory_arm=theory_arm)
            binding_source = "scientific_route_contract"
            out["scientific_route_contract_id"] = contract_row.get("contract_id")
        elif card is not None:
            a_sci = evaluate_A_sci_with_route_card(out, self.task, card, theory_arm=theory_arm)
            binding_source = "route_card"
        elif is_pilot_family(fid) or is_fl2_pilot_task(self.task) or is_pfdf_pilot_task(self.task) or is_drout_pilot_task(self.task) or is_tctrk_pilot_task(self.task) or is_seven_track_pilot_task(self.task):
            if pilot_assets_configured():
                a_sci = _missing_binding_a_sci("missing_route_binding")
                binding_source = "missing_binding"
            else:
                a_sci = _missing_binding_a_sci("missing_route_contract")
                binding_source = "missing_binding"
        else:
            a_sci = evaluate_A_sci(out, self.task, theory_arm=theory_arm)
            binding_source = "dev_stub"

        out["A_sci"] = a_sci
        if out.get("g6_coreexec"):
            out = _annotate_g6_a_cap(out, self.task, state, graph=graph)
        else:
            cert = evaluate_reachability(out, state, graph=graph)
            out = attach_reachability_certificate(out, cert)
        out["admissible"] = is_admissible(out["A_sci"], out["A_cap"])
        try:
            from hazardweaver.hwa.experiments.headline_ablation_modes_v1 import hcg_interface_only, hcg_untyped

            if hcg_interface_only():
                a_cap = out.get("A_cap") or {}
                verdict = str(a_cap.get("verdict") or "").upper()
                out["admissible"] = verdict in {"REACHABLE", "OK", "PASS"} or bool(out.get("g6_coreexec"))
                out["hcg_compat_mode"] = "interface_only"
            elif hcg_untyped():
                out["hcg_compat_mode"] = "untyped"
        except ImportError:
            pass
        out = apply_route_metric_contract(out, self.task)
        out["hkc_binding_source"] = binding_source
        if card is not None:
            out["route_card_id"] = card.get("route_card_id")
        if not out.get("route_id"):
            edges = out.get("edges") or out.get("capability_ids") or []
            out["route_id"] = "route:" + ("+".join(edges[:3]) if edges else "unknown")
        return out

    def admissible_routes(
        self,
        routes: Sequence[Mapping[str, Any]],
        state: SessionState,
        *,
        graph: Any = None,
        admissible_only: bool = False,
    ) -> List[Dict[str, Any]]:
        annotated = [self.annotate_route(r, state, graph=graph) for r in routes]
        if admissible_only:
            return [r for r in annotated if r.get("admissible")]
        return annotated

    def pi_adm_nonempty(self, routes: Sequence[Mapping[str, Any]]) -> bool:
        return any(r.get("admissible") for r in routes)
