"""HW bridge: thin tools wrapping the independent ``hcg`` package."""

from __future__ import annotations

import ast
import json
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from hazardweaver.hcg import (
    GraphStore,
    PathExecutor,
    emit_infer_job,
    emit_train_job,
    find_paths,
    path_edge_ids,
    search_composition_paths,
)
from hazardweaver.hcg.models import SupportSpec
from hazardweaver.hcg.route.instantiate import instantiate_route, list_admissible_routes, on_state_delta
from hazardweaver.hcg.route.state import ExecutionState, StateDelta, from_fixtures
from hazardweaver.hcg.runtime import fixtures as fx
from hazardweaver.hcg.runtime import validate_path
from hazardweaver.hcg.search import CompositionPath, PathStep


def normalize_artifact_id(value: Any) -> str:
    """Strip whitespace and wrapping quotes LLMs often put around artifact ids."""
    s = str(value).strip()
    if len(s) >= 2 and s[0] == s[-1] and s[0] in {"\"", "'"}:
        s = s[1:-1].strip()
    return s


def coerce_artifact_id_sequence(values: Any) -> List[Any]:
    """Coerce LLM tool args into a flat artifact-id sequence.

    Small local models (e.g. Llama-8B) often emit ``sources`` as a Python-list
    *string* such as ``"['registry_stub_input_v1']"`` instead of a JSON array.
    Passing that string through ``list(str)`` yields per-character tokens and
    triggers ``unknown artifact in find_paths: [`` KeyErrors.
    """
    if values is None:
        return []
    if isinstance(values, (list, tuple)):
        return list(values)
    if isinstance(values, str):
        s = values.strip()
        if not s:
            return []
        if s[0] in "[({":
            try:
                parsed = ast.literal_eval(s)
            except (ValueError, SyntaxError):
                parsed = None
            if isinstance(parsed, (list, tuple)):
                return list(parsed)
            if parsed is not None:
                return [parsed]
            try:
                parsed_json = json.loads(s)
            except json.JSONDecodeError:
                parsed_json = None
            if isinstance(parsed_json, list):
                return parsed_json
            if parsed_json is not None:
                return [parsed_json]
        return [s]
    return [values]


def normalize_artifact_ids(values: Sequence[Any]) -> List[str]:
    flat = coerce_artifact_id_sequence(values)
    out = [normalize_artifact_id(v) for v in flat]
    if any(not x for x in out):
        raise ValueError("empty artifact id after normalize")
    return out


def make_default_store(
    *,
    include_holdouts: bool = False,
    packs: Optional[Sequence[str]] = None,
) -> GraphStore:
    store = GraphStore.load_v0(include_holdouts=include_holdouts)
    for name in packs or ():
        store.merge_pack(name)
    return store


def tool_hcg_list_admissible_routes(
    task_id: str,
    s_k: Optional[ExecutionState] = None,
    *,
    store: Optional[GraphStore] = None,
) -> List[str]:
    state = s_k or from_fixtures(task_id)
    return list_admissible_routes(task_id, state, store=store)


def tool_hcg_instantiate_route(
    family_id: str,
    s_k: Optional[ExecutionState] = None,
    *,
    task_id: str = "WF-3",
    store: Optional[GraphStore] = None,
) -> Dict[str, Any]:
    state = s_k or from_fixtures(task_id)
    result = instantiate_route(family_id, state, store=store)
    out: Dict[str, Any] = {
        "ok": result.ok,
        "abstain": result.abstain,
        "clarify": result.clarify,
        "reason": result.reason,
    }
    if result.route:
        out.update(
            {
                "route_id": result.route.route_id,
                "family_id": result.route.family_id,
                "edges": result.route.edge_ids(),
                "target": result.route.target,
                "log": result.route.log.to_dict(),
            }
        )
    return out


def tool_hcg_on_state_delta(
    s_k: ExecutionState,
    delta: StateDelta,
    *,
    family_id: str,
    prior_route_id: str,
    store: Optional[GraphStore] = None,
) -> Dict[str, Any]:
    result = on_state_delta(
        s_k,
        delta,
        family_id=family_id,
        prior_route_id=prior_route_id,
        store=store,
    )
    out: Dict[str, Any] = {
        "action": result.action,
        "reason": result.reason,
        "prior_route_id": result.prior_route_id,
    }
    if result.route:
        out.update(
            {
                "route_id": result.route.route_id,
                "edges": result.route.edge_ids(),
                "log": result.route.log.to_dict(),
            }
        )
    return out


def _path_from_edge_ids(graph: Any, edge_ids: Sequence[str]) -> Optional[CompositionPath]:
    """Reconstruct CompositionPath from edge id list using graph metadata."""
    steps: List[PathStep] = []
    available: set[str] = set()
    for eid in edge_ids:
        eid = str(eid)
        if eid in graph.capabilities:
            cap = graph.capabilities[eid]
            steps.append(
                PathStep(
                    kind="capability",
                    edge_id=eid,
                    consumes=tuple(cap.consumes),
                    produces=tuple(cap.produces),
                )
            )
            available.update(cap.produces)
        elif eid in graph.adapters:
            ad = graph.adapters[eid]
            steps.append(
                PathStep(
                    kind="adapter",
                    edge_id=eid,
                    consumes=tuple(ad.consumes),
                    produces=tuple(ad.produces),
                )
            )
            available.update(ad.produces)
        else:
            return None
    if not steps:
        return None
    return CompositionPath(steps=steps, available=available)


def tool_hcg_find_paths(
    sources: Sequence[str],
    target: str,
    *,
    store: Optional[GraphStore] = None,
    max_paths: int = 5,
    packs: Optional[Sequence[str]] = None,
    executable_only: bool = False,
    require_all_sources: bool = False,
) -> List[Dict[str, Any]]:
    sources_n = normalize_artifact_ids(sources)
    target_n = normalize_artifact_id(target)
    if not target_n:
        raise ValueError("empty target artifact id after normalize")
    store = store or make_default_store(packs=packs)
    try:
        paths = search_composition_paths(
            store.graph,
            sources_n,
            target_n,
            max_paths=max_paths,
            executable_only=executable_only,
            require_all_sources=require_all_sources,
        )
    except KeyError:
        # LLM may pass capability_id as target; treat as no paths (do not kill VCE loop).
        return []
    return [
        {
            "edges": path_edge_ids(p),
            "available": sorted(p.available),
            "valid": validate_path(store.graph, p).ok,
        }
        for p in paths
    ]


def tool_hcg_run_path(
    sources: Sequence[str],
    target: str,
    *,
    store: Optional[GraphStore] = None,
    packs: Optional[Sequence[str]] = None,
    require_all_sources: bool = True,
    fixture: str = "wf_hard",
    handles: Optional[Dict[str, Any]] = None,
    host: Any = None,
    out_dir: Any = None,
    edge_ids: Optional[Sequence[str]] = None,
    route_id: Optional[str] = None,
    a_sci: Optional[Mapping[str, Any]] = None,
    a_cap: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Execute composition path (Hard default: AND sources).

    P0-6: successful runs mint ExecutionResult into the task workdir registry.
    When ``edge_ids`` is provided, execute that exact path (Scientific Controller).
    """
    from hazardweaver.hwa.agent_runtime.execution_schema import (
        build_route_candidate,
        mint_and_register_execution,
    )

    sources_n = normalize_artifact_ids(sources)
    target_n = normalize_artifact_id(target)
    if not target_n:
        return {"ok": False, "abstain_reason": "EMPTY_TARGET_ID", "edges": []}
    store = store or make_default_store(packs=packs or ["graph_eval_v0"])

    path: Optional[CompositionPath] = None
    if edge_ids:
        path = _path_from_edge_ids(store.graph, edge_ids)
        if path is None:
            return {"ok": False, "abstain_reason": "INVALID_EDGE_IDS", "edges": list(edge_ids)}
    else:
        paths = search_composition_paths(
            store.graph,
            sources_n,
            target_n,
            max_paths=8,
            max_depth=16,
            executable_only=True,
            require_all_sources=require_all_sources,
            explore_or_edges=True,
        )
        preferred = None
        for p in paths:
            if "hw_wildfire_spread_aspp_bridge_v0" in path_edge_ids(p):
                preferred = p
                break
        path = preferred or (paths[0] if paths else None)

    if path is None:
        return {"ok": False, "abstain_reason": "NO_EXECUTABLE_AND_PATH", "edges": []}

    if handles is None:
        if fixture == "wf_hard":
            handles = fx.wf_hard_initial_handles()
        elif fixture == "eq_chain":
            handles = fx.eq_chain_initial_handles()
        else:
            handles = {s: 0.0 for s in sources_n}
    result = PathExecutor(store.graph).run(path, handles)
    answer: Dict[str, Any] = {}
    if result.ok and target_n in result.handles:
        answer[target_n] = float(np.asarray(result.handles[target_n]).reshape(-1)[0])
    edges = path_edge_ids(path)
    rid = route_id or ("hcg:" + ("+".join(edges[:3]) if edges else target_n))
    out: Dict[str, Any] = {
        "ok": result.ok,
        "edges": edges,
        "abstain_reason": result.abstain_reason,
        "answer": answer,
        "route_id": rid,
    }
    if result.ok:
        route_candidate = build_route_candidate(
            route_id=rid,
            capability_ids=[e for e in edges if e in store.graph.capabilities],
            adapter_ids=[e for e in edges if e in store.graph.adapters],
            theory_alignment_refs=list((a_sci or {}).get("refs") or []),
        )
        minted = mint_and_register_execution(
            host=host,
            out_dir=out_dir,
            raw_result={"answer": answer, "predicted_value": next(iter(answer.values()), None)},
            capability_ids=list(edges),
            route_id=rid,
            ok=True,
            provenance={
                "tool": "tool_hcg_run_path",
                "target": target_n,
                "A_sci": dict(a_sci) if a_sci else None,
                "A_cap": dict(a_cap) if a_cap else None,
            },
            route_candidate=route_candidate,
        )
        out.update(minted)
    return out


def tool_hcg_reinstantiate(
    sources: Sequence[str],
    target: str,
    *,
    state: Any,
    task: Mapping[str, Any],
    store: Optional[GraphStore] = None,
    max_paths: int = 8,
) -> Dict[str, Any]:
    """Controller-internal reinstantiate (not exposed to LLM tool surface)."""
    from hazardweaver.hwa.scientific_controller.reinstantiate import tool_hcg_reinstantiate as _reinst

    return _reinst(
        sources,
        target,
        state=state,
        task=task,
        store=store or make_default_store(),
        max_paths=max_paths,
    )


def tool_hcg_explain_edge(edge_id: str, *, store: Optional[GraphStore] = None) -> Dict[str, Any]:
    store = store or make_default_store()
    eid = normalize_artifact_id(edge_id)
    if eid in store.graph.capabilities:
        return {"kind": "capability", **store.get_capability(eid).model_dump(mode="json")}
    if eid in store.graph.adapters:
        return {"kind": "adapter", **store.get_adapter(eid).model_dump(mode="json")}
    raise KeyError(f"unknown edge: {eid}")


def tool_hcg_emit_train_job(
    capability_id: str,
    dataset_id: str,
    *,
    store: Optional[GraphStore] = None,
    overrides: Optional[Dict[str, Any]] = None,
    target_support: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    store = store or make_default_store()
    ts = SupportSpec.model_validate(target_support) if target_support else None
    job = emit_train_job(
        store.graph,
        capability_id,
        dataset_id,
        overrides=overrides,
        target_support=ts,
    )
    return job.to_dict()


def tool_hcg_emit_infer_job(
    sources: Sequence[str],
    target: str,
    sample_refs: Dict[str, str],
    *,
    store: Optional[GraphStore] = None,
) -> Dict[str, Any]:
    store = store or make_default_store()
    sources_n = normalize_artifact_ids(sources)
    target_n = normalize_artifact_id(target)
    paths = find_paths(store.graph, sources_n, target_n, max_paths=1)
    if not paths:
        raise ValueError(f"no path from {sources_n} to {target_n}")
    return emit_infer_job(
        paths[0],
        sample_refs=sample_refs,
        source_artifacts=sources_n,
        target_artifact=target_n,
    ).to_dict()
