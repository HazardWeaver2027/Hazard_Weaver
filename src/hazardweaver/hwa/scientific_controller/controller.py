"""Deterministic Scientific Controller — sole holder of execution authority."""

from __future__ import annotations

import hashlib
import json
import secrets
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from hazardweaver.hwa.scientific_controller.tool_policy import (
    CONTROLLER_EXEC_TOOLS,
    CONTROLLER_SEMANTIC_TOOLS,
)
from hazardweaver.hwa.scientific_controller.admissibility import (
    admissible_routes,
    annotate_route,
    is_admissible,
)
from hazardweaver.hwa.scientific_controller.logging import append_decision, load_decisions
from hazardweaver.hwa.scientific_controller.reason_codes import ControllerAction
from hazardweaver.hwa.scientific_controller.reachability import (
    find_reachable_paths,
    init_state_from_task,
    make_store_for_task,
)
from hazardweaver.hwa.scientific_controller.reinstantiate import (
    invalidate_capability,
    reinstantiate_routes,
)


def _coerce_bool(val: Any, *, default: bool = True) -> bool:
    if val is None:
        return default
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    s = str(val).strip().lower()
    if s in {"true", "1", "yes", "on"}:
        return True
    if s in {"false", "0", "no", "off"}:
        return False
    return default
from hazardweaver.hwa.scientific_controller.state import SessionState


class ScientificController:
    """Hard-logic controller: reachability, compatibility, execution, invalidation."""

    def __init__(
        self,
        task: Mapping[str, Any],
        workdir: Path,
        *,
        pack_root: Optional[Path] = None,
        store: Any = None,
        env: Any = None,
    ):
        self.task = dict(task)
        self.workdir = Path(workdir)
        self.pack_root = pack_root
        self.env = env
        self.state = init_state_from_task(task)
        from hazardweaver.hwa.runtime.runtime_closure import hydrate_session_graph_from_task

        hydrate_session_graph_from_task(task, self.state)
        self.store = store or make_store_for_task(task, state=self.state)
        self._route_cache: Dict[str, Dict[str, Any]] = {}
        self._execution_tokens: Dict[str, Dict[str, Any]] = {}
        self._proposed: Optional[Dict[str, Any]] = None
        self._rejected_route_ids: set[str] = set()
        self._last_enumerate_checkpoint: Optional[int] = None

    @property
    def graph(self) -> Any:
        return self.store.graph if self.store is not None else None

    def n_decisions(self) -> int:
        return len(load_decisions(self.workdir))

    def _cache_routes(self, routes: Sequence[Mapping[str, Any]]) -> None:
        for r in routes:
            rid = str(r.get("route_id") or "")
            if rid:
                self._route_cache[rid] = dict(r)

    def _route_capability_ids(self, route: Mapping[str, Any]) -> List[str]:
        return [
            str(c).strip()
            for c in (route.get("capability_ids") or route.get("edges") or [])
            if str(c).strip() and not str(c).startswith("schema")
        ]

    def _current_allowed_capability_ids(self) -> set[str]:
        sv = (self.task.get("solver_visible") or {}).get("inputs") or {}
        return {str(c).strip() for c in (sv.get("allowed_edge_ids") or []) if str(c).strip()}

    def _route_surface_violation(self, route: Mapping[str, Any]) -> Optional[str]:
        from hazardweaver.hwa.experiments.unified_e12_agent_nudge_v1 import unified_e12_s0_shock_only_violation

        s0_err = unified_e12_s0_shock_only_violation(self.task, route)
        if s0_err:
            return s0_err
        invalidated = set(self.state.invalidated_capabilities or ())
        allowed = self._current_allowed_capability_ids()
        caps = self._route_capability_ids(route)
        if not caps:
            return "route_has_no_capabilities"
        for cap in caps:
            if cap in invalidated:
                return "route_uses_invalidated_capability"
        if allowed and any(cap not in allowed for cap in caps):
            return "route_not_in_allowed_surface"
        return None

    def _purge_stale_route_cache(self) -> None:
        for rid in list(self._route_cache.keys()):
            route = self._route_cache.get(rid) or {}
            if not self._route_surface_violation(route):
                continue
            self._route_cache.pop(rid, None)
            if str(self.state.pending_route_id or "") == rid:
                self.state.pending_route_id = None
            if self._proposed and str(self._proposed.get("route_id") or "") == rid:
                self._proposed = None

    def _mark_route_rejected(self, route_id: str, *, reason: str) -> None:
        rid = str(route_id or "").strip()
        if rid:
            self._rejected_route_ids.add(rid)

    def remaining_admissible_route_ids(self) -> List[str]:
        out: List[str] = []
        for rid, route in self._route_cache.items():
            if rid in self._rejected_route_ids:
                continue
            if is_admissible(route.get("A_sci") or {}, route.get("A_cap") or {}):
                out.append(rid)
        return out

    def all_candidates_exhausted(self) -> bool:
        if not self._route_cache:
            return False
        return len(self.remaining_admissible_route_ids()) == 0

    def candidates_exhausted_recovery(self) -> Dict[str, Any]:
        """Terminal/recovery surface when every cached route was rejected."""
        self._log(
            ControllerAction.REJECT.value,
            notes="all_candidates_exhausted",
        )
        return {
            "ok": False,
            "error": "all_candidates_exhausted",
            "rejected_route_ids": sorted(self._rejected_route_ids),
            "next_step": (
                "No admissible routes remain at this checkpoint. "
                "Call controller_reinstantiate_routes, submit_clarification, "
                "or submit_abstention with an evidence-backed reason."
            ),
        }

    def _log(
        self,
        action: str,
        *,
        route: Optional[Mapping[str, Any]] = None,
        notes: Optional[str] = None,
        execution_id: Optional[str] = None,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        cp = self.state.bump_checkpoint()
        merged_extra = dict(extra or {})
        if route and route.get("reachability_certificate"):
            merged_extra["reachability_certificate"] = route.get("reachability_certificate")
        return append_decision(
            self.workdir,
            checkpoint=cp,
            action=action,
            route_id=str(route.get("route_id")) if route else None,
            capability_ids=list(route.get("capability_ids") or route.get("edges") or []) if route else None,
            a_sci=route.get("A_sci") if route else None,
            a_cap=route.get("A_cap") if route else None,
            theory_arm=self.state.theory_arm,
            execution_id=execution_id,
            notes=notes,
            extra=merged_extra,
        )

    def enumerate_routes(
        self,
        sources: Optional[Sequence[str]] = None,
        target: Optional[str] = None,
        *,
        max_paths: int = 8,
        admissible_only: bool = False,
    ) -> Dict[str, Any]:
        from hazardweaver.hwa.agent_runtime.hcg_tools import coerce_artifact_id_sequence, normalize_artifact_id

        raw_src = sources if sources is not None else self.state.sources
        src = [normalize_artifact_id(v) for v in coerce_artifact_id_sequence(raw_src)]
        tgt = normalize_artifact_id(target or self.state.target or "")
        if not tgt:
            inputs = (self.task.get("solver_visible") or {}).get("inputs") or {}
            goals = inputs.get("goal_artifacts") or []
            if goals:
                tgt = str(goals[0])
                self.state.target = tgt

        routes = find_reachable_paths(
            src,
            tgt,
            self.state,
            store=self.store,
            max_paths=max_paths,
            packs=self.state.packs,
        )
        annotated = admissible_routes(
            routes,
            self.task,
            self.state,
            graph=self.graph,
            admissible_only=admissible_only,
        )
        visible: List[Dict[str, Any]] = []
        for route in annotated:
            rid = str(route.get("route_id") or "")
            if rid in self._rejected_route_ids:
                continue
            visible.append(dict(route))
        self._cache_routes(annotated)
        self._last_enumerate_checkpoint = int(self.state.checkpoint)
        self._log(ControllerAction.ENUMERATE.value, notes=f"n={len(visible)}")
        payload: Dict[str, Any] = {
            "ok": True,
            "sources": src,
            "target": tgt,
            "n_routes": len(visible),
            "routes": visible,
            "checkpoint": self.state.checkpoint,
            "rejected_route_ids": sorted(self._rejected_route_ids),
        }
        if self.all_candidates_exhausted():
            payload["all_candidates_exhausted"] = True
            payload["recovery"] = self.candidates_exhausted_recovery()
        return payload

    def _canonical_route_id(self, route_id: Optional[str]) -> str:
        from hazardweaver.hwa.contracts.route_execution import normalize_controller_route_id

        raw = str(route_id or "").strip()
        rid = normalize_controller_route_id(raw) or raw
        if rid in self._route_cache:
            return rid
        if raw.startswith("CAP-"):
            from hazardweaver.hwa.contracts.route_execution import make_route_id_for_capability

            alt = str(make_route_id_for_capability(raw))
            if alt in self._route_cache:
                return alt
        return rid

    def propose_route(
        self,
        route_id: str,
        *,
        rationale: Optional[str] = None,
        route_intent: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        rid = self._canonical_route_id(route_id)
        route = self._route_cache.get(rid)
        if route is None:
            self.enumerate_routes()
            route = self._route_cache.get(rid)
        if route is None:
            self._log(
                ControllerAction.REJECT.value,
                notes=f"unknown_route_id:{rid}",
            )
            return {
                "ok": False,
                "error": "unknown_route_id",
                "route_id": rid,
                "hint": "call controller_enumerate_routes first",
            }
        if rid in self._rejected_route_ids:
            recovery = (
                self.candidates_exhausted_recovery()
                if self.all_candidates_exhausted()
                else {
                    "ok": False,
                    "error": "route_already_rejected",
                    "route_id": rid,
                    "hint": "choose a different route_id from controller_enumerate_routes",
                }
            )
            return recovery

        from hazardweaver.hwa.experiments.g6_coreexec_controller_bridge import prepare_route_for_controller

        route = prepare_route_for_controller(route, self.task, self.state, graph=self.graph)
        self._route_cache[rid] = route
        surface_err = self._route_surface_violation(route)
        if surface_err:
            self._mark_route_rejected(rid, reason=surface_err)
            self._log(ControllerAction.REJECT.value, route=route, notes=surface_err)
            return {
                "ok": False,
                "error": surface_err,
                "route_id": rid,
                "hint": "call controller_enumerate_routes after shock failure reinstantiate",
            }
        if not is_admissible(route.get("A_sci") or {}, route.get("A_cap") or {}):
            from hazardweaver.hwa.scientific_controller.admissibility import admissibility_reason_code

            reason_code = admissibility_reason_code(route.get("A_sci") or {}, route.get("A_cap") or {})
            self._mark_route_rejected(rid, reason=reason_code or "not_admissible")
            self._log(ControllerAction.REJECT.value, route=route, notes=rationale or "not_admissible")
            result = {
                "ok": False,
                "error": "route_not_admissible",
                "reason_code": reason_code,
                "route_id": rid,
                "A_sci": route.get("A_sci"),
                "A_cap": route.get("A_cap"),
                "rationale": rationale,
            }
            if self.all_candidates_exhausted():
                result["all_candidates_exhausted"] = True
                result["recovery"] = self.candidates_exhausted_recovery()
            return result

        self._proposed = route
        self.state.pending_route_id = rid
        self._log(ControllerAction.PROPOSE.value, route=route, notes=rationale)
        return {
            "ok": True,
            "route_id": rid,
            "status": "proposed",
            "A_sci": route.get("A_sci"),
            "A_cap": route.get("A_cap"),
            "rationale": rationale,
            "next_step": "controller_commit_route",
        }

    def mint_execution_token(self, route_id: str) -> str:
        token = secrets.token_hex(16)
        self._execution_tokens[token] = {
            "route_id": route_id,
            "created_at": time.time(),
            "used": False,
        }
        return token

    def execution_token_rejection_reason(
        self, token: Optional[str], route_id: str
    ) -> Optional[str]:
        """Why a token would be rejected (None = would validate). Does not consume."""
        if not token:
            return "controller_token_missing"
        rec = self._execution_tokens.get(str(token))
        if rec is None:
            return "controller_token_unknown"
        if rec.get("used"):
            return "controller_token_already_consumed"
        if rec.get("route_id") != route_id:
            return "controller_token_route_mismatch"
        if time.time() - float(rec.get("created_at", 0)) > 600:
            return "controller_token_expired"
        return None

    def validate_execution_token(self, token: Optional[str], route_id: str) -> bool:
        if self.execution_token_rejection_reason(token, route_id) is not None:
            return False
        rec = self._execution_tokens.get(str(token))
        if rec is None:
            return False
        rec["used"] = True
        return True

    def pending_unused_token(self, route_id: str) -> Optional[str]:
        """Return an unused execution token for ``route_id`` (strict null-literal bridge)."""
        rid = str(route_id or "").strip()
        if not rid:
            return None
        for token, rec in self._execution_tokens.items():
            if rec.get("used"):
                continue
            if str(rec.get("route_id") or "") == rid:
                return str(token)
        return None

    def commit_route(self, route_id: Optional[str] = None) -> Dict[str, Any]:
        rid = self._canonical_route_id(route_id or self.state.pending_route_id)
        if not rid and self._proposed:
            rid = str(self._proposed.get("route_id") or "")
        route = self._route_cache.get(rid) or self._proposed
        if route is None and rid:
            prop = self.propose_route(rid)
            if not prop.get("ok"):
                return prop
            route = self._route_cache.get(rid) or self._proposed
        if route is None:
            return {"ok": False, "error": "no_proposed_route", "route_id": rid}

        from hazardweaver.hwa.experiments.g6_coreexec_controller_bridge import prepare_route_for_controller

        route = prepare_route_for_controller(route, self.task, self.state, graph=self.graph)
        surface_err = self._route_surface_violation(route)
        if surface_err:
            self._mark_route_rejected(rid, reason=surface_err)
            self._log(ControllerAction.REJECT.value, route=route, notes=surface_err)
            return {
                "ok": False,
                "error": surface_err,
                "route_id": rid,
                "hint": "choose a route on the post-shock admissible surface",
            }
        if not is_admissible(route.get("A_sci") or {}, route.get("A_cap") or {}):
            self._mark_route_rejected(rid, reason="commit_not_admissible")
            self._log(ControllerAction.REJECT.value, route=route, notes="commit_not_admissible")
            result = {
                "ok": False,
                "error": "route_not_admissible_at_commit",
                "route_id": rid,
                "A_sci": route.get("A_sci"),
                "A_cap": route.get("A_cap"),
            }
            if self.all_candidates_exhausted():
                result["all_candidates_exhausted"] = True
                result["recovery"] = self.candidates_exhausted_recovery()
            return result

        token = self.mint_execution_token(rid)
        self.state.active_route_id = rid
        from hazardweaver.hwa.experiments.unified_e12_agent_nudge_v1 import reset_unified_e12_post_shock_enum_streak

        reset_unified_e12_post_shock_enum_streak(self, route)
        caps = list(route.get("capability_ids") or route.get("edges") or [])
        try:
            from hazardweaver.hwa.runtime.lease_manager import bind_controller_commit

            lease = bind_controller_commit(self, rid, caps)
            lease_id = lease.lease_id
        except Exception:  # noqa: BLE001
            lease_id = None
        self._log(ControllerAction.COMMIT.value, route=route, extra={"execution_token_issued": True, "lease_id": lease_id})
        return {
            "ok": True,
            "route_id": rid,
            "status": "committed",
            "execution_token": token,
            "lease_id": lease_id,
            "allowed_capability_ids": caps,
            "message": "Controller will execute on next internal dispatch; token required for run_capability.",
        }

    def execute_route(
        self,
        route_id: Optional[str] = None,
        *,
        execution_token: Optional[str] = None,
        host: Any = None,
        out_dir: Any = None,
        fixture: str = "wf_hard",
        handles: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Execute a committed route via HCG path runner."""
        from hazardweaver.hwa.agent_runtime.hcg_tools import tool_hcg_run_path
        from hazardweaver.hwa.experiments.g6_coreexec_controller_bridge import execute_g6_coreexec_route

        rid = str(route_id or self.state.active_route_id or "").strip()
        route = self._route_cache.get(rid) or self._proposed
        if route is None:
            return {"ok": False, "error": "no_active_route", "route_id": rid}

        if route.get("g6_coreexec"):
            return execute_g6_coreexec_route(
                self,
                route,
                execution_token=execution_token,
                host=host,
                out_dir=out_dir or self.workdir,
                handles=handles,
            )

        if execution_token and not self.validate_execution_token(execution_token, rid):
            self._log(ControllerAction.REJECT.value, route=route, notes="invalid_execution_token")
            return {"ok": False, "error": "invalid_execution_token", "route_id": rid}

        edges = list(route.get("edges") or route.get("capability_ids") or [])
        src = list(self.state.sources)
        tgt = str(self.state.target or "")
        result = tool_hcg_run_path(
            src,
            tgt,
            store=self.store,
            packs=self.state.packs or ["graph_eval_v0"],
            fixture=fixture,
            handles=handles,
            host=host or self.env,
            out_dir=out_dir or self.workdir,
            edge_ids=edges if edges else None,
            route_id=rid,
            a_sci=route.get("A_sci"),
            a_cap=route.get("A_cap"),
        )

        if result.get("ok"):
            self.state.active_route_id = rid
            if tgt and result.get("answer"):
                self.state.produced_handles[tgt] = result.get("answer")
            self._log(
                ControllerAction.EXECUTE.value,
                route=route,
                execution_id=str(result.get("execution_id") or ""),
                notes="execution_ok",
            )
        else:
            self.handle_execution_failure(
                route,
                failed_capability_id=edges[0] if edges else "",
                reason=str(result.get("abstain_reason") or result.get("error") or "execution_failed"),
            )
            result["reinstantiated"] = self.enumerate_routes(admissible_only=False)
        return result

    def _expand_rq4_allowed_edges_after_shock_failure(self) -> None:
        """After E12 shock failure, Full arm narrows to scenario gold; static keeps s0 freeze."""
        from hazardweaver.hwa.experiments.unified_benchmark_execution_v1 import unified_e12_post_shock_allowed_edge_ids

        meta = dict(self.task.get("metadata") or {})
        full_allowed = list(meta.get("rq4_full_allowed_edge_ids") or [])
        if not full_allowed:
            return
        invalidated = set(self.state.invalidated_capabilities or ())
        post_shock = unified_e12_post_shock_allowed_edge_ids(
            self.task,
            full_allowed=full_allowed,
            invalidated=invalidated,
        )
        inputs = self.task.setdefault("solver_visible", {}).setdefault("inputs", {})
        inputs["allowed_edge_ids"] = list(post_shock)
        rq4 = dict(meta.get("rq4_intervention") or {})
        rq4["defer_s0_narrow_active"] = False
        meta["rq4_intervention"] = rq4
        meta["rq4_defer_s0_narrow_active"] = False
        self.task["metadata"] = meta
        constraints = self.task.setdefault("solver_visible", {}).setdefault("constraints", {})
        crq4 = constraints.setdefault("rq4_intervention", {})
        crq4["defer_s0_narrow_active"] = False
        self._purge_stale_route_cache()

    def handle_execution_failure(
        self,
        route: Mapping[str, Any],
        *,
        failed_capability_id: str = "",
        reason: str = "",
    ) -> Dict[str, Any]:
        from hazardweaver.hwa.experiments.rq4_route_intervention_v1 import rq4_skip_reinstantiate
        from hazardweaver.hwa.scientific_controller.reinstantiate import invalidate_capability

        if rq4_skip_reinstantiate(self.task):
            cid = str(failed_capability_id or "").strip()
            edges = list(route.get("edges") or route.get("capability_ids") or [])
            from hazardweaver.hwa.experiments.headline_ablation_modes_v1 import route_eligibility_static_whitelist_s0
            from hazardweaver.hwa.experiments.static_whitelist_s0_v1 import static_whitelist_s0_frozen_edge_ids

            if route_eligibility_static_whitelist_s0():
                from hazardweaver.hwa.experiments.static_whitelist_s0_v1 import apply_static_whitelist_s0_post_shock_failure

                invalidate_capability(
                    self.state,
                    cid or (edges[0] if edges else "unknown"),
                    reason=reason,
                    also_invalidate_edges=edges,
                )
                _post_allowed, frozen = apply_static_whitelist_s0_post_shock_failure(
                    self.task,
                    self.state,
                    frozen_routes=getattr(self, "_s0_frozen_routes", None),
                )
                self._cache_routes(frozen)
                self._purge_stale_route_cache()
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
        edges = list(route.get("edges") or [])
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
        new_routes = reinstantiate_routes(self.task, self.state, store=self.store)
        self._cache_routes(new_routes)
        self._purge_stale_route_cache()
        self._log(
            ControllerAction.REINSTANTIATE.value,
            notes=f"n_routes={len(new_routes)}",
            extra={"invalidated": cid},
        )
        return {
            "ok": False,
            "invalidated": cid,
            "reason": reason,
            "n_reinstantiated_routes": len(new_routes),
            "routes": new_routes,
        }

    def get_route_status(self) -> Dict[str, Any]:
        self._log(ControllerAction.STATUS.value)
        return {
            "ok": True,
            "checkpoint": self.state.checkpoint,
            "active_route_id": self.state.active_route_id,
            "pending_route_id": self.state.pending_route_id,
            "invalidated_edges": sorted(self.state.invalidated_edges),
            "invalidated_capabilities": sorted(self.state.invalidated_capabilities),
            "state": self.state.to_dict(),
        }

    def reject_direct_execution(self, tool_name: str, action: Mapping[str, Any]) -> Dict[str, Any]:
        self._log(
            ControllerAction.REJECT.value,
            notes=f"direct_execution_forbidden:{tool_name}",
            extra={"arguments_keys": sorted((action.get("arguments") or {}).keys())},
        )
        return {
            "ok": False,
            "error": "direct_execution_forbidden",
            "tool": tool_name,
            "message": (
                "LLM cannot execute capabilities directly. Use controller_enumerate_routes → "
                "controller_propose_route → controller_commit_route."
            ),
            "allowed_semantic_tools": sorted(CONTROLLER_SEMANTIC_TOOLS),
        }

    def handle_semantic_tool(
        self,
        name: str,
        args: Mapping[str, Any],
        *,
        action_id: Any = None,
    ) -> Dict[str, Any]:
        if name == "controller_enumerate_routes":
            from hazardweaver.hwa.experiments.unified_e12_agent_nudge_v1 import block_unified_e12_post_shock_enumerate_spin

            spin_block = block_unified_e12_post_shock_enumerate_spin(self)
            if spin_block is not None:
                result = spin_block
            else:
                result = self.enumerate_routes(
                    sources=args.get("sources"),
                    target=args.get("target"),
                    max_paths=int(args.get("max_paths") or 8),
                    admissible_only=bool(args.get("admissible_only", False)),
                )
        elif name == "controller_propose_route":
            result = self.propose_route(
                str(args.get("route_id") or ""),
                rationale=args.get("rationale"),
                route_intent=args.get("route_intent"),
            )
        elif name == "controller_commit_route":
            result = self.commit_route(route_id=args.get("route_id"))
            exec_result: Dict[str, Any] = {}
            from hazardweaver.hwa.experiments.g6_coreexec_controller_bridge import uses_allowed_edge_route_synthesis

            execute = _coerce_bool(args.get("execute"), default=True)
            chain_handles: Optional[Dict[str, Any]] = None
            from hazardweaver.hwa.experiments.agent_strict_v2 import (
                agent_strict_v2_chained_commit_enabled,
                agent_strict_v2_enabled,
                extract_chained_commit_handles,
            )

            if agent_strict_v2_enabled():
                chain_handles = (
                    extract_chained_commit_handles(args)
                    if agent_strict_v2_chained_commit_enabled()
                    else None
                )
                if chain_handles:
                    from hazardweaver.hwa.experiments.ablation_manual_pilot_execution_v1 import (
                        ablation_authoritative_execution_handles,
                    )
                    from hazardweaver.hwa.experiments.ablation_manual_pilot_v1 import ablation_manual_pilot_enabled

                    if ablation_manual_pilot_enabled(self.task):
                        meta = self.task.get("metadata") or {}
                        inv = meta.get("grader_hidden") or meta
                        chain_handles = ablation_authoritative_execution_handles(
                            chain_handles,
                            inventory_row=inv if isinstance(inv, Mapping) else meta,
                            task_metadata=meta if isinstance(meta, Mapping) else None,
                        )
                if not chain_handles and uses_allowed_edge_route_synthesis(self.task):
                    from hazardweaver.hwa.experiments.unified_benchmark_execution_v1 import (
                        solve_mandatory_episode_policy_enabled,
                    )

                    if solve_mandatory_episode_policy_enabled(self.task):
                        from hazardweaver.hwa.experiments.ablation_manual_pilot_execution_v1 import (
                            ablation_authoritative_execution_handles,
                        )

                        meta = self.task.get("metadata") or {}
                        inv = meta.get("grader_hidden") or meta
                        chain_handles = ablation_authoritative_execution_handles(
                            {},
                            inventory_row=inv if isinstance(inv, Mapping) else meta,
                            task_metadata=meta if isinstance(meta, Mapping) else None,
                        )
                execute = bool(chain_handles)
            elif uses_allowed_edge_route_synthesis(self.task):
                execute = True
            if result.get("ok") and execute:
                exec_result = self.execute_route(
                    route_id=result.get("route_id"),
                    execution_token=result.get("execution_token"),
                    host=self.env,
                    out_dir=self.workdir,
                    fixture=str(args.get("fixture") or "wf_hard"),
                    handles=chain_handles,
                )
                result = {**result, "execution": exec_result}
                if exec_result.get("ok"):
                    result["execution_id"] = exec_result.get("execution_id")
                    result["final_artifact_id"] = exec_result.get("final_artifact_id")
                    result["next_step"] = "submit_solution"
                    result["message"] = (
                        "Execution succeeded. Call submit_solution with route_id, "
                        "execution_id, and final_artifact_id from this response."
                    )
                else:
                    result["ok"] = False
                    result["error"] = exec_result.get("error") or "execution_failed"
                    result["execution_error"] = exec_result.get("error") or "execution_failed"
                    result["failure_class"] = exec_result.get("failure_class", "unknown")
                    result["next_step"] = "run_capability"
                    from hazardweaver.hwa.experiments.unified_benchmark_execution_v1 import (
                        solve_mandatory_episode_policy_enabled,
                    )

                    skip_reenumerate = (
                        uses_allowed_edge_route_synthesis(self.task)
                        and solve_mandatory_episode_policy_enabled(self.task)
                    )
                    if skip_reenumerate:
                        result["message"] = (
                            "Execution failed after commit. Retry run_capability on the "
                            "committed route with execution handles from the goal; do not "
                            "re-enumerate routes."
                        )
                    else:
                        result["next_step"] = "submit_clarification"
                        try:
                            from hazardweaver.hwa.runtime.abstention_gate import pi_adm_empty

                            enum = self.enumerate_routes(admissible_only=False)
                            routes = list(enum.get("routes") or [])
                            if not pi_adm_empty(routes):
                                result["abstention_blocked"] = True
                                result["message"] = (
                                    "Execution failed but admissible routes remain. "
                                    "Retry execution or submit_clarification; do not use "
                                    "submit_abstention(NO_LEGAL_ROUTE)."
                                )
                            else:
                                result["message"] = (
                                    "Execution failed. Use submit_clarification or a "
                                    "failure-specific abstain reason; do not use "
                                    "NO_LEGAL_ROUTE after commit."
                                )
                        except Exception:  # noqa: BLE001
                            result["message"] = (
                                "Execution failed. Read execution.failure_class and "
                                "execution.error; use submit_clarification or appropriate abstain."
                            )
            elif result.get("ok"):
                result["execution"] = {"ok": False, "skipped": True, "reason": "execute=false"}
        elif name == "controller_get_route_status":
            result = self.get_route_status()
        else:
            result = {"ok": False, "error": "unknown_controller_tool", "tool": name}

        try:
            from hazardweaver.hwa.runtime.trajectory_ledger import record_controller_decision

            record_controller_decision(
                self.workdir,
                {
                    "action": name.replace("controller_", ""),
                    "route_id": result.get("route_id"),
                    "checkpoint": self.state.checkpoint,
                    "A_sci": result.get("A_sci"),
                    "A_cap": result.get("A_cap"),
                    "execution_id": result.get("execution_id"),
                    "lease_id": result.get("lease_id"),
                    "extra": {
                        "tool": name,
                        "ok": bool(result.get("ok", True)),
                        "error": result.get("error"),
                        "execution": result.get("execution"),
                        "failure_class": result.get("failure_class"),
                        "next_step": result.get("next_step"),
                    },
                },
            )
        except Exception:  # noqa: BLE001
            pass

        inner_ok = bool(result.get("ok", True))
        content = json.dumps({"ok": inner_ok, "name": name, "result": result}, default=str)
        return {
            "tool_call_id": action_id,
            "name": name,
            "ok": inner_ok,
            "content": content,
        }

    def is_exec_tool(self, name: str) -> bool:
        return name in CONTROLLER_EXEC_TOOLS

    def is_semantic_tool(self, name: str) -> bool:
        return name in CONTROLLER_SEMANTIC_TOOLS

    def update_state_from_observation(self, tool_name: str, result: Mapping[str, Any]) -> None:
        if not isinstance(result, Mapping):
            return
        if tool_name in {"load_sample", "inspect_artifact"} and result.get("artifact_id"):
            self.state.available_artifacts.add(str(result["artifact_id"]))
        inner = result.get("result") if isinstance(result.get("result"), Mapping) else result
        if isinstance(inner, Mapping) and inner.get("execution_id"):
            self.state.produced_handles[str(inner.get("final_artifact_id") or "")] = inner


def controller_mode_from_task(task: Mapping[str, Any]) -> bool:
    solver = task.get("solver_visible") or {}
    if solver.get("controller_mode") is True:
        return True
    sv = str(task.get("schema_version") or "")
    if "controller_v1" in sv:
        return True
    return False
