"""Cluster bootstrap CI — resample unit = scenario_id / event_id."""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import numpy as np


def cluster_bootstrap_ci(
    records: Sequence[Dict[str, object]],
    *,
    cluster_key: str = "scenario_id",
    value_key: str = "valid",
    n_bootstrap: int = 500,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Bootstrap CI over cluster units (not individual variants).

    Returns (point_estimate, ci_low, ci_high).
    """
    if not records:
        return float("nan"), float("nan"), float("nan")

    clusters: Dict[str, List[float]] = {}
    for rec in records:
        cid = str(rec.get(cluster_key) or "unknown")
        val = float(rec.get(value_key) or 0.0)
        clusters.setdefault(cid, []).append(val)

    cluster_ids = sorted(clusters.keys())
    cluster_means = [float(np.mean(clusters[c])) for c in cluster_ids]
    point = float(np.mean(cluster_means))

    rng = np.random.default_rng(seed)
    boot: List[float] = []
    for _ in range(n_bootstrap):
        sampled = rng.choice(cluster_ids, size=len(cluster_ids), replace=True)
        vals = [float(np.mean(clusters[c])) for c in sampled]
        boot.append(float(np.mean(vals)))
    if not boot:
        return point, float("nan"), float("nan")
    return point, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))
