# mhbench Implementation Report

**Date:** 2026-06-15  
**Version:** 0.1.0  
**Evidence tags:** confirmed_by_execution where noted

---

## What was implemented

| Module | Status |
|--------|--------|
| `mhbench/schemas/` | Pydantic v2: MechanismCard, ScenarioSpec, ScenarioBundleManifest, ParameterBounds, DependencyTestSpec |
| `mhbench/registry/` | 3 mechanism YAML cards + `synthetic_geography.json` + loader |
| `mhbench/compiler/operators.py` | Full temporal/spatial/coupling/counterfactual operators |
| `mhbench/compiler/field_synthesizer.py` | Independent RNG base fields (graph/raster/track) |
| `mhbench/compiler/validators.py` | Blocking gates: bounds, leakage, dependency ordering, negative control |
| `mhbench/compiler/scenario_compiler.py` | Orchestrates plugins → validate → bundle |
| `mhbench/generators/` | 3 plugins with **all** declared counterfactual families |
| `mhbench/io/bundle_io.py` | Standard ScenarioBundle layout |
| `mhbench/evaluators/` | Frozen grader (no LLM) |
| `mhbench/baselines/` | 5 reproducible mechanism-aware baselines |
| `mhbench/scripts/` | generate_scenarios, validate_scenarios, run_baselines |
| `mhbench/llm_harness/` | C2 offline draft ingest (not on hot path) |
| `mhbench/examples/` | 3 scenario YAML specs |
| `MCCS/*.md` | Upgraded to v1.1 |

---

## What was verified by execution

- **confirmed_by_execution:** `pytest mhbench/tests/` — 20/20 passed
- **confirmed_by_execution:** Generated bundles:
  - `runs/mhbench/compound_seed0`
  - `runs/mhbench/tc_flood_seed42`
  - `runs/mhbench/wf_proxy_seed7`
- **confirmed_by_execution:** All bundles pass strict validation (dependency ordering, counterfactual non-triviality)
- **confirmed_by_execution:** `test_no_pyhazards_import` — generation works with `pyhazards` blocked
- **confirmed_by_execution:** Determinism — same seed → identical labels

---

## What remains design-only

- RQ-B0 C2 LLM-draft path (harness stub only; no LLM wired)
- Historical event archetype scenarios (T3 validation tier)
- E5 held-out mechanism transfer experiments
- Solver-facing multi-agent evaluation (E6)
- PyHazards optional shape integration test extra `mhbench[pyhazards]`

---

## Where PyHazards was used

- **Not used at runtime** in generator path.
- Shapes vendored from `asserts/PYHAZARDS_DATA_ONTOLOGY.md` into `schemas/modality_contracts.py`.

---

## Where PyHazards was deliberately not used

- No `load_dataset()`, `build_model()`, or `BenchmarkRunner()` in generator.
- No PyHazards tensor values as seed.
- No agent-call-model loop in generator.

---

## How circularity was avoided

1. Generator uses independent field synthesizer + frozen label rules.
2. Solver path (future) may use PyHazards models; generator does not.
3. `mechanism_oracle_baseline` uses hidden state for **sanity upper bound only**, not solver evaluation.
4. LLM limited to offline spec drafts (C2 harness), never numeric labels.

---

## Which assumptions are weak

1. **Synthetic realism:** Fields are statistically plausible, not physically simulated.
2. **WF proxy:** Post-fire susceptibility is a raster proxy, not debris-flow physics.
3. **TC–Flood alignment:** `synthetic_geography.json` is a schematic basin–grid map.
4. **Dependency metrics:** Correlation-based proxies, not causal identification.
5. **Literature citations:** Placeholder strings pending T2 author verification.

---

## Validation tiers (implemented)

| Tier | Implementation |
|------|----------------|
| T1 | `validators.py` + `validate_scenarios.py` + pytest |
| T2 | Citations in `mechanism_registry.yaml` — author review |
| T3 | Deferred (historical archetype) |
| T4 | Deferred (domain expert) |

---

## What Jojo / ChatGPT need to decide next

1. T2 literature citation completion for mechanism cards
2. Whether to add historical event archetype specs (after advisor input)
3. RQ-B0: wire LLM harness to actual LLM API for C2 ablation
4. Solver system design against frozen bundles
5. Paper figure selection from `runs/mhbench/*` bundles

---

## Quick commands

```bash
cd 
conda activate ./envs/pyhazards
PYTHONPATH=. python -m pytest mhbench/tests/ -q
PYTHONPATH=. python -m mhbench.scripts.generate_scenarios \
  --spec mhbench/examples/flood_wildfire_compound_example.yaml \
  --out runs/mhbench/compound_seed0
PYTHONPATH=. python -m mhbench.scripts.validate_scenarios \
  --bundle runs/mhbench/compound_seed0 \
  --spec mhbench/examples/flood_wildfire_compound_example.yaml
PYTHONPATH=. python -m mhbench.scripts.run_baselines \
  --bundle runs/mhbench/compound_seed0
```
