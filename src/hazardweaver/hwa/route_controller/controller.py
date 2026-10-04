"""ScientificRouteController — default HWA route state machine (Fusion Task 1)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from hazardweaver.hwa.route_controller.minimal_repair import dispatch_minimal_repair
from hazardweaver.hwa.route_controller.remsa_rank import remsa_soft_rank
from hazardweaver.hwa.route_controller.admissibility_gate import AdmissibilityGate
from hazardweaver.hwa.route_controller.care_stop import care_should_stop
from hazardweaver.hwa.route_controller.pareto_prune import pareto_prune
from hazardweaver.hwa.route_controller.route_decision import build_route_decision_packet
from hazardweaver.hwa.route_controller.task_context import TaskContext
from hazardweaver.hwa.route_controller.tie_break import needs_clarification, tie_break_routes
from hazardweaver.hwa.runtime.lease_manager import require_lease
from hazardweaver.hwa.runtime.hcg_execute_bridge import run_headline_capability, event_to_run_capability_result
from hazardweaver.hwa.scientific_controller.controller import ScientificController
from hazardweaver.hwa.scientific_controller.hkc_pilot_assets import resolve_hkc_pilot_paths
from hazardweaver.hwa.scientific_controller.reason_codes import ControllerAction
from hazardweaver.hwa.scientific_controller.reachability import find_reachable_paths


class ScientificRouteController(ScientificController):
    """Certificate-grounded controller: HKC A_sci + HCG A_cap + lease execution."""

    def __init__(
        self,
        task: Mapping[str, Any],
        workdir: Path,
        *,
        pack_root: Optional[Path] = None,
        store: Any = None,
        env: Any = None,
        route_card_path: Optional[Path] = None,
        contract_path: Optional[Path] = None,
    ):
        super().__init__(task, workdir, pack_root=pack_root, store=store, env=env)
        self.task_ctx = TaskContext.from_task(task)
        from hazardweaver.hwa.experiments.headline_hkc_mode_v1 import headline_hkc_mode, headline_hkc_off
        from hazardweaver.hwa.scientific_controller.hkc_frozen_registry_v1 import (
            hkc_registry_enabled,
            get_frozen_hkc_registry,
        )

        self.hkc_mode = headline_hkc_mode()
        self.route_card_loaded = False
        pilot_paths = resolve_hkc_pilot_paths()
        rc_path = route_card_path
        contract = contract_path
        if hkc_registry_enabled():
            frozen = get_frozen_hkc_registry()
            contract = frozen.contracts_path
            rc_path = None
        elif not headline_hkc_off(task):
            if rc_path is None and pilot_paths.route_cards_exists():
                rc_path = pilot_paths.route_cards
            if rc_path is None:
                legacy = Path("benchmark/public/hkc_data/08_ANNOTATION_PACKAGES/route_cards_compiled_pilot_v1.jsonl")
                if legacy.is_file():
                    rc_path = legacy
                else:
                    candidate = Path("benchmark/public/hkc_data/08_ANNOTATION_PACKAGES/route_cards_compiled.jsonl")
                    if candidate.is_file():
                        rc_path = candidate
            if contract is None and pilot_paths.contracts_exists():
                contract = pilot_paths.contracts
        else:
            rc_path = None
            contract = None
        self.gate = AdmissibilityGate(
            task,
            route_card_path=rc_path,
            contract_path=contract,
        )
        self.route_card_loaded = bool(
            self.gate.route_card_index is not None or self.gate.contract_index is not None
        )
        self._last_decision_packet: Optional[Dict[str, Any]] = None
        self._s0_frozen_routes: Optional[List[Dict[str, Any]]] = None
        from hazardweaver.hwa.runtime.runtime_closure import init_headline_graph_state, init_pfdf_controller_state

        init_headline_graph_state(self)
        init_pfdf_controller_state(self)

    def refresh_admissible_surface(self) -> Dict[str, Any]:
        """Per-step Π_adm refresh (SayCan tool surface; invalidation-aware)."""
        try:
            from hazardweaver.hwa.experiments.headline_ablation_modes_v1 import route_eligibility_static_frozen

            if route_eligibility_static_frozen() and self._s0_frozen_routes is not None:
                return self._return_frozen_s0_routes(admissible_only=False)
        except ImportError:
            pass
        return self.enumerate_routes(admissible_only=False)

    def _route_key(self, route: Mapping[str, Any]) -> str:
        rid = str(route.get("route_id") or "").strip()
        if rid:
            return rid
        edges = route.get("edges") or route.get("capability_ids") or []
        return "edges:" + "+".join(str(e) for e in edges)

    def _freeze_s0_routes_if_needed(self, ranked: Sequence[Mapping[str, Any]]) -> None:
        try:
            from hazardweaver.hwa.experiments.headline_ablation_modes_v1 import route_eligibility_static_frozen
        except ImportError:
            return
        if not route_eligibility_static_frozen() or self._s0_frozen_routes is not None:
            return
        self._s0_frozen_routes = [dict(r) for r in ranked if r.get("admissible")]

    def _return_frozen_s0_routes(self, *, admissible_only: bool) -> Dict[str, Any]:
        from hazardweaver.hwa.experiments.static_whitelist_s0_v1 import filter_routes_by_invalidated

        frozen = list(self._s0_frozen_routes or [])
        invalidated = set(getattr(self.state, "invalidated_capabilities", None) or ())
        if invalidated:
            frozen = filter_routes_by_invalidated(frozen, invalidated)
        ranked = [dict(r) for r in frozen if self._route_key(r) not in self._rejected_route_ids]
        self._cache_routes(ranked)
        self._freeze_s0_routes_if_needed(ranked)
        self._last_decision_packet = build_route_decision_packet(
            ranked,
            task_id=self.task_ctx.task_id,
            checkpoint=self.state.checkpoint,
        )
        if admissible_only:
            ranked = [r for r in ranked if r.get("admissible")]
        return self._finalize_enumerate(
            {
                "ok": True,
                "sources": list(self.state.sources),
                "target": str(self.state.target or ""),
                "n_routes": len(ranked),
                "routes": ranked,
                "decision_packet": self._last_decision_packet,
                "checkpoint": self.state.checkpoint,
                "static_s0_frozen": True,
            }
        )

    def handle_execution_failure(
        self,
        route: Mapping[str, Any],
        *,
        failed_capability_id: str = "",
        reason: str = "",
    ) -> Dict[str, Any]:
        from hazardweaver.hwa.experiments.g6_coreexec_controller_bridge import uses_allowed_edge_route_synthesis
        from hazardweaver.hwa.scientific_controller.reason_codes import ControllerAction
        from hazardweaver.hwa.scientific_controller.reinstantiate import invalidate_capability

        if not uses_allowed_edge_route_synthesis(self.task):
            return super().handle_execution_failure(
                route,
                failed_capability_id=failed_capability_id,
                reason=reason,
            )

        from hazardweaver.hwa.experiments.rq4_route_intervention_v1 import rq4_skip_reinstantiate

        if rq4_skip_reinstantiate(self.task):
            cid = str(failed_capability_id or "").strip()
            from hazardweaver.hwa.experiments.headline_ablation_modes_v1 import route_eligibility_static_whitelist_s0

            if route_eligibility_static_whitelist_s0():
                from hazardweaver.hwa.experiments.static_whitelist_s0_v1 import apply_static_whitelist_s0_post_shock_failure

                edges = list(route.get("edges") or route.get("capability_ids") or [])
                invalidate_capability(
                    self.state,
                    cid or (edges[0] if edges else "unknown"),
                    reason=reason,
                    also_invalidate_edges=edges,
                )
                _post_allowed, frozen = apply_static_whitelist_s0_post_shock_failure(
                    self.task,
                    self.state,
                    frozen_routes=self._s0_frozen_routes,
                )
                self._cache_routes(frozen)
                self._purge_stale_route_cache()
                self.enumerate_routes(admissible_only=False)
                self._log(
                    ControllerAction.INVALIDATE.value,
                    route=route,
                    notes=reason or "static_whitelist_s0_post_shock",
                )
                return {
                    "ok": False,
                    "invalidated": cid,
                    "reason": reason,
                    "n_reinstantiated_routes": 0,
                    "routes": frozen,
                    "rq4_static_control": True,
                    "static_whitelist_s0_post_shock": True,
                    "n_post_shock_frozen_routes": len(frozen),
                }
            return {
                "ok": False,
                "invalidated": cid,
                "reason": reason,
                "n_reinstantiated_routes": 0,
                "routes": [],
                "rq4_static_control": True,
            }

        cid = str(failed_capability_id or "").strip()
        edges = list(route.get("edges") or route.get("capability_ids") or [])
        invalidate_capability(
            self.state,
            cid or (edges[0] if edges else "unknown"),
            reason=reason,
            also_invalidate_edges=edges,
        )
        self._expand_rq4_allowed_edges_after_shock_failure()
        self._log(
            ControllerAction.INVALIDATE.value,
            route=route,
            notes=reason or "execution_failure",
        )
        enum = self.enumerate_routes(admissible_only=False)
        new_routes = list(enum.get("routes") or [])
        invalidated = set(self.state.invalidated_capabilities or ())
        if invalidated:
            new_routes = [
                r
                for r in new_routes
                if not any(str(c) in invalidated for c in (r.get("capability_ids") or r.get("edges") or []))
            ]
        self._cache_routes(new_routes)
        self._purge_stale_route_cache()
        self._log(
            ControllerAction.REINSTANTIATE.value,
            notes=f"g6_coreexec n_routes={len(new_routes)}",
            extra={"invalidated": cid},
        )
        return {
            "ok": False,
            "invalidated": cid,
            "reason": reason,
            "n_reinstantiated_routes": len(new_routes),
            "routes": new_routes,
            "g6_coreexec_reinstantiate": True,
        }

    def propose_route(
        self,
        route_id: str,
        *,
        rationale: Optional[str] = None,
        route_intent: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        """P0-6: LLM route choice must lie on last Pareto frontier."""
        from hazardweaver.hwa.route_controller.route_decision import parse_route_intent
        from hazardweaver.hwa.runtime.runtime_closure import counterfactual_hint_for_route

        rid = str(route_id or "").strip()
        if route_intent:
            parsed = parse_route_intent(route_intent)
            if parsed:
                rid = parsed
        if rid:
            from hazardweaver.hwa.experiments.unified_e12_agent_nudge_v1 import unified_e12_s0_shock_only_violation

            cap = rid.split("route:cap:", 1)[-1] if rid.startswith("route:cap:") else ""
            s0_probe = {"route_id": rid, "capability_ids": [cap] if cap else []}
            s0_err = unified_e12_s0_shock_only_violation(self.task, s0_probe)
            if s0_err:
                self._log(ControllerAction.REJECT.value, notes=s0_err, extra={"route_id": rid})
                return {
                    "ok": False,
                    "error": s0_err,
                    "route_id": rid,
                    "hint": "E12 s0: commit the forced shock route only, then recover post-failure.",
                }
        packet = self._last_decision_packet or {}
        frontier = {str(x) for x in (packet.get("pareto_frontier_ids") or []) if x}
        skip_pareto = False
        try:
            from hazardweaver.hwa.experiments.headline_ablation_modes_v1 import route_eligibility_static_whitelist_s0
            from hazardweaver.hwa.experiments.unified_e12_agent_nudge_v1 import unified_e12_post_shock_context

            if route_eligibility_static_whitelist_s0() and unified_e12_post_shock_context(
                self.task, self.state
            ):
                skip_pareto = True
        except ImportError:
            pass
        if frontier and rid and rid not in frontier and not skip_pareto:
            self._log(
                ControllerAction.REJECT.value,
                notes=f"off_pareto_frontier:{rid}",
            )
            return {
                "ok": False,
                "error": "route_off_pareto_frontier",
                "route_id": rid,
                "pareto_frontier_ids": sorted(frontier),
                "hint": "call controller_enumerate_routes and choose a frontier route_id",
            }
        result = super().propose_route(rid, rationale=rationale)
        if not result.get("ok"):
            route = self._route_cache.get(rid)
            cf = counterfactual_hint_for_route(route, controller=self)
            if cf:
                result["counterfactual"] = cf
                slots = cf.get("user_recoverable_slots") or cf.get("clarify_slots") or []
                if slots:
                    result["hint"] = (
                        "A_sci UNKNOWN — call ask_user with user_recoverable_slots "
                        f"from counterfactual: {slots[:5]}"
                    )
        return result

    def enumerate_routes(
        self,
        sources: Optional[Sequence[str]] = None,
        target: Optional[str] = None,
        *,
        max_paths: int = 8,
        admissible_only: bool = False,
    ) -> Dict[str, Any]:
        try:
            from hazardweaver.hwa.experiments.headline_ablation_modes_v1 import route_eligibility_static_frozen

            if route_eligibility_static_frozen() and self._s0_frozen_routes is not None:
                return self._return_frozen_s0_routes(admissible_only=admissible_only)
        except ImportError:
            pass
        from hazardweaver.hwa.experiments.g6_coreexec_controller_bridge import (
            headline_hcg_hybrid_g6_fallback_eligible,
            uses_allowed_edge_route_synthesis,
        )
        try:
            from hazardweaver.hwa.experiments.headline_ablation_modes_v1 import src_eligibility_gate_off

            if src_eligibility_gate_off():
                admissible_only = False
        except ImportError:
            pass

        if uses_allowed_edge_route_synthesis(self.task):
            return self._enumerate_g6_coreexec(
                sources=sources,
                target=target,
                admissible_only=admissible_only,
                notes="g6_coreexec",
            )

        src = list(sources or self.state.sources)
        tgt = str(target or self.state.target or "")
        if not tgt:
            goals = self.task_ctx.goal_artifacts
            if goals:
                tgt = goals[0]
                self.state.target = tgt

        routes = find_reachable_paths(
            src,
            tgt,
            self.state,
            store=self.store,
            max_paths=max_paths,
            packs=self.state.packs,
        )
        annotated = self.gate.admissible_routes(
            routes,
            self.state,
            graph=self.graph,
            admissible_only=admissible_only,
        )
        pruned = pareto_prune(annotated)
        ranked = remsa_soft_rank(
            pruned,
            task=self.task,
            state=self.state,
            committed_route=self._route_cache.get(self.state.active_route_id or ""),
        )
        self._cache_routes(ranked)
        self._freeze_s0_routes_if_needed(ranked)
        stop = care_should_stop(ranked, n_steps=len(ranked), max_steps=max_paths)
        self._last_decision_packet = build_route_decision_packet(
            ranked,
            task_id=self.task_ctx.task_id,
            checkpoint=self.state.checkpoint,
        )
        self._log(
            ControllerAction.ENUMERATE.value,
            notes=f"n={len(ranked)} admissible={sum(1 for r in ranked if r.get('admissible'))}",
            extra={
                "care_stop": stop,
                "tie_clarify": needs_clarification(ranked),
                "remsa_top": (ranked[0].get("route_id") if ranked else None),
            },
        )
        n_admissible = sum(1 for r in ranked if r.get("admissible"))
        if admissible_only:
            ranked = [r for r in ranked if r.get("admissible")]
        if n_admissible == 0:
            from hazardweaver.hwa.experiments.coupled_4m_mode_v1 import coupled_typed_allowed_edge_eligible
            from hazardweaver.hwa.experiments.g6_coreexec_controller_bridge import routes_from_allowed_edges

            if coupled_typed_allowed_edge_eligible(self.task) and routes_from_allowed_edges(self.task):
                return self._enumerate_g6_coreexec(
                    sources=src,
                    target=tgt,
                    admissible_only=admissible_only,
                    notes="coupled_typed_allowed_edge",
                    hcg_hybrid_fallback=False,
                )
        if n_admissible == 0 and headline_hcg_hybrid_g6_fallback_eligible(self.task):
            return self._enumerate_g6_coreexec(
                sources=src,
                target=tgt,
                admissible_only=admissible_only,
                notes="hcg_hybrid_g6_fallback",
                hcg_hybrid_fallback=True,
            )
        return self._finalize_enumerate(
            {
                "ok": True,
                "sources": src,
                "target": tgt,
                "n_routes": len(ranked),
                "routes": ranked,
                "decision_packet": self._last_decision_packet,
                "checkpoint": self.state.checkpoint,
                "remsa_ranked": True,
            }
        )

    def _enumerate_g6_coreexec(
        self,
        *,
        sources: Optional[Sequence[str]],
        target: Optional[str],
        admissible_only: bool,
        notes: str,
        hcg_hybrid_fallback: bool = False,
    ) -> Dict[str, Any]:
        from hazardweaver.hwa.experiments.g6_coreexec_controller_bridge import routes_from_allowed_edges

        g6_routes = routes_from_allowed_edges(self.task)
        g6_routes = self.gate.admissible_routes(g6_routes, self.state)
        ranked = remsa_soft_rank(
            g6_routes,
            task=self.task,
            state=self.state,
            committed_route=self._route_cache.get(self.state.active_route_id or ""),
        )
        self._cache_routes(ranked)
        self._freeze_s0_routes_if_needed(ranked)
        self._last_decision_packet = build_route_decision_packet(
            ranked,
            task_id=self.task_ctx.task_id,
            checkpoint=self.state.checkpoint,
        )
        self._log(
            ControllerAction.ENUMERATE.value,
            notes=f"{notes} n={len(ranked)}",
            extra={
                "remsa_top": (ranked[0].get("route_id") if ranked else None),
                "hcg_hybrid_fallback": hcg_hybrid_fallback,
            },
        )
        if admissible_only:
            ranked = [r for r in ranked if r.get("admissible")]
        out: Dict[str, Any] = {
            "ok": True,
            "sources": list(sources or []),
            "target": str(target or ""),
            "n_routes": len(ranked),
            "routes": ranked,
            "decision_packet": self._last_decision_packet,
            "checkpoint": self.state.checkpoint,
            "remsa_ranked": True,
            "g6_coreexec": True,
        }
        if hcg_hybrid_fallback:
            out["hcg_hybrid_fallback"] = True
        meta = self.task.get("metadata") or {}
        cbr_rid = meta.get("ds_cbr_route_id")
        if cbr_rid and str(meta.get("headline_route_profile") or "") == "g1_v1":
            out["g1_cbr_recommended_route_id"] = cbr_rid
        return self._apply_post_enumerate_autocommit(out)

    def _finalize_enumerate(self, result: Dict[str, Any]) -> Dict[str, Any]:
        return self._apply_post_enumerate_autocommit(result)

    def _apply_post_enumerate_autocommit(self, result: Dict[str, Any]) -> Dict[str, Any]:
        from hazardweaver.hwa.control.mode import is_vce_mode
        from hazardweaver.hwa.experiments.headline_g1_autoroute_v1 import apply_g1_autocommit_if_eligible
        from hazardweaver.hwa.experiments.headline_hcg_bind_autocommit_v1 import apply_hcg_bind_autocommit_if_eligible
        from hazardweaver.hwa.experiments.unified_e12_agent_nudge_v1 import apply_unified_e12_post_shock_enumerate_overlay

        out = apply_unified_e12_post_shock_enumerate_overlay(self, result)
        if is_vce_mode():
            return out
        out = apply_g1_autocommit_if_eligible(self, out)
        return apply_hcg_bind_autocommit_if_eligible(self, out)

    def get_decision_packet(self) -> Optional[Dict[str, Any]]:
        return self._last_decision_packet

    def execute_with_lease(
        self,
        capability_id: str,
        handles: Mapping[str, Any],
        *,
        smoke_mode: bool = False,
    ) -> Dict[str, Any]:
        """PFDF / bench shared execution path with lease gate."""
        from hazardweaver.hwa.experiments.agent_strict_v2 import agent_strict_v2_enabled

        lease = require_lease(self.env or self)
        if lease is None:
            return {
                "ok": False,
                "error": "scoped_execution_lease_required",
                "capability_id": capability_id,
            }
        if agent_strict_v2_enabled():
            from hazardweaver.hcg.runtime.execute_capability import ExecuteCapabilityBundle
            from hazardweaver.hwa.runtime.hwa_headline_infer_v2 import (
                hwa_strict_result_to_run_capability,
                run_hwa_strict_headline_infer,
            )

            try:
                bundle = run_hwa_strict_headline_infer(
                    capability_id,
                    handles,
                    lease,
                    host=self.env or self,
                    smoke_mode=smoke_mode,
                )
            except ValueError as exc:
                return {
                    "ok": False,
                    "error": "agent_strict_v2_validation_failed",
                    "capability_id": capability_id,
                    "message": str(exc),
                }
            return hwa_strict_result_to_run_capability(bundle, capability_id=capability_id)
        event = run_headline_capability(
            capability_id,
            handles,
            lease,
            smoke_mode=smoke_mode,
            full_certificates=True,
        )
        if hasattr(event, "event"):
            return event_to_run_capability_result(event.event, capability_id=capability_id, bundle=event)
        return event_to_run_capability_result(event, capability_id=capability_id)

    def handle_tool_failure(
        self,
        capability_id: str,
        error: str,
    ) -> Dict[str, Any]:
        rid = str(self.state.active_route_id or "")
        route = self._route_cache.get(rid) or self._proposed or {}
        return dispatch_minimal_repair(
            self,
            route,
            failed_capability_id=capability_id,
            reason=error,
        )

    def deterministic_tie_break(self) -> Optional[Dict[str, Any]]:
        if self._last_decision_packet is None:
            self.enumerate_routes()
        routes = (self._last_decision_packet or {}).get("candidates") or []
        committed = self._route_cache.get(self.state.active_route_id or "")
        return tie_break_routes(
            routes,
            task=self.task,
            state=self.state,
            committed_route=committed,
        )

    def bump_segment(self) -> str:
        """Late-binding: increment segment_id on state for next lease."""
        seg = int(self.state.segment_id or "0")
        self.state.segment_id = str(seg + 1)
        return self.state.segment_id
