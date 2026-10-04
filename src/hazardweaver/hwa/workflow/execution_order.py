"""Topological execution order for compiled workflow DAGs."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Dict, List

from hazardweaver.hwa.contracts import WorkflowSpec


def topological_node_order(workflow: WorkflowSpec) -> List[str]:
    """Return node ids in dependency order (predecessors before successors)."""

    node_ids = {node.id for node in workflow.nodes}
    indegree: Dict[str, int] = {nid: 0 for nid in node_ids}
    outgoing: Dict[str, List[str]] = defaultdict(list)

    for edge in workflow.edges:
        if edge.from_node not in node_ids or edge.to_node not in node_ids:
            continue
        outgoing[edge.from_node].append(edge.to_node)
        indegree[edge.to_node] += 1

    queue = deque(sorted(nid for nid, deg in indegree.items() if deg == 0))
    order: List[str] = []
    while queue:
        current = queue.popleft()
        order.append(current)
        for nxt in sorted(outgoing[current]):
            indegree[nxt] -= 1
            if indegree[nxt] == 0:
                queue.append(nxt)

    if len(order) != len(node_ids):
        raise ValueError("workflow_contains_cycle_or_disconnected_nodes")
    return order


def node_by_id(workflow: WorkflowSpec) -> Dict[str, object]:
    return {node.id: node for node in workflow.nodes}
