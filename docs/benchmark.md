# Benchmark and evaluation

## HWB inventory (sealed)

| | Count |
|---|------:|
| Instances | 141 |
| Single-hazard | 90 |
| Multi-hazard | 51 |
| One route at s₀ | 45 |
| Multi-route at s₀ | 96 |

**Tracks (11):** DR-OUT, E1-E3, FL-2, HW-MED, L2, MH-1, MH-2, MH-3, MH-4, TC-TRK, WF-3.

Files: `benchmark/public/manifest.jsonl`, taskpacks under `benchmark/public/taskpacks/`.

## Solver vs evaluator

- **Public** material is everything an agent may see before submitting an answer (task text, route briefs, tool schema).
- **Sealed** material under `benchmark/sealed_eval/` holds gold routes, tolerances, and reference bindings. Agent code must not read it; scoring runs after a terminal submission.

See `benchmark/README.md` and `tests/leakage/`.

## DCA and aggregates

**DCA (v2)** marks an instance correct when the terminal artifact passes the task-native checker and, when a gold route exists, the committed capability matches that route.

Reported aggregates follow the paper: Wilson intervals on instance rates, macro DCA as the unweighted mean over the 11 tracks (bootstrap over tracks, seed 42), and paired McNemar tests on the same 141 instances.

## Baselines

Table 1 lists adapted planning agents (Plan-and-Execute, Self-Consistency, DisasterBench-ToT, MLE-STAR, AutoML-Agent, Reflexion). Table 2 lists shared-tool controls (AIDE-style, DS-Agent-style, R&D-Agent-style, ReAct with the same tool surface as HWA). All use the same Llama-3.3-70B policy and HCG replay path as in the paper; implementations are under `src/hazardweaver/baselines/`.

## Data sources

Instances are built from public hazard datasets (WildfireSpreadTS, FloodCastBench, LHASA, SeisBench, TCBench, NOAA CPC, ExtremeWeatherBench, USGS post-fire and ground-failure products, SFINCS, NOAA Storm Events, etc.). Full rasters and checkpoints are obtained separately; smoke tests use small fixtures (`hwb_*_fixture_v1` taskpacks).

## Limitations

The 141-instance set is frozen for the paper; it does not cover every region or sensor configuration. Validity is contract-based on sealed evaluator definitions, not fresh expert re-labeling of every case. Operational use requires updated data and local review.
