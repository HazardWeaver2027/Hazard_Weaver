# mhbench Engineering Design Memo

**Version:** 0.1.0  
**Date:** 2026-06-15  
**Status:** design decision + implementation baseline

---

## 0.1 System boundary

| Component | Role |
|-----------|------|
| **Generator** | Mechanism simulator + scenario compiler; deterministic label/counterfactual production |
| **Solver** (future) | Multi-agent / PyHazards model orchestration |
| **LLM** | Offline only: mechanism card + DSL drafts → validator → frozen spec |

**Hard rules:**

- Generator **never** calls `build_model()`, `BenchmarkRunner`, or agent-call-model loops.
- Generator **never** calls `load_dataset()` for tensor values.
- Generator uses **independent field synthesizer** (matching PyHazards ontology shapes, own RNG).
- Solver may use PyHazards; solver cannot read `hidden/`, label rules, or counterfactual metadata during prediction.

Evidence tags used in this repo: `confirmed_by_code`, `confirmed_by_execution`, `inferred_from_audit`, `design_decision`, `unsupported_future_work`.

---

## 0.2 MVP mechanism templates

| ID | Modality | Label | Claim boundary |
|----|----------|-------|----------------|
| `flood_wildfire_spatial_compound` | raster | compound risk mask | spatial overlap + interaction |
| `tc_to_flood` | graph + track | flood risk regression/binary | TC forcing → basin risk |
| `wildfire_to_postfire_flood_proxy` | raster | susceptibility proxy | **proxy only**, not debris-flow |

**Not in scope:** landslide, tsunami, arbitrary global city generator, historical event reproduction.

**Implementation order:** compound (flagship) → tc_to_flood → wildfire proxy.

---

## 0.3 PyHazards dependency boundary

**Allowed:** vendored shapes in `schemas/modality_contracts.py` (from `asserts/PYHAZARDS_DATA_ONTOLOGY.md`), metric vocabulary, modality naming.

**Forbidden:** `load_dataset` tensor values, `build_model`, `BenchmarkRunner`, generator agent loop.

**Default:** `mhbench` runs with **zero runtime PyHazards import**. Optional `mhbench[pyhazards]` for shape integration tests only.

---

## 0.4 Validation tiers

| Tier | Gate | Blocking? |
|------|------|-----------|
| T1 | dependency ordering, negative control, leakage, determinism | **Yes** (CI) |
| T2 | literature citation check on mechanism cards | Soft (author) |
| T3 | historical event archetype | No (future) |
| T4 | domain expert field review | No (camera-ready optional) |

---

## 0.5 No-dummy principle

Even at MVP, every deliverable must be production-quality:

- All declared counterfactual families implemented (not a subset).
- Real label rules with configurable parameters recorded in provenance.
- Baselines are reproducible mechanism-aware models, not constant stubs.
- Validators are blocking gates; failed bundles are not written.
- Grader is frozen and LLM-free.

---

## 0.6 Key risks and mitigations

| Risk | Mitigation |
|------|------------|
| Label leakage | `exposed_field_manifest` + correlation threshold in validators |
| Weak counterfactuals | Full counterfactual families + dependency ordering CI |
| Arbitrary parameters | `parameter_bounds` in registry + compiler reject |
| TC–Flood heterogeneity | `registry/synthetic_geography.json` |
| WF proxy overclaim | manifest `proxy_only: true` + mechanism card notes |
| Circularity with solver | Separate generator/solver paths; no shared model loop |
| Synthetic realism gap | Counterfactual self-validation, not expert field review |

---

## 0.7 Package layout

```text
mhbench/
  schemas/          # Pydantic v2 models + modality contracts
  registry/         # mechanism YAML + synthetic_geography.json
  compiler/         # operators, field_synthesizer, validators, scenario_compiler
  generators/       # 3 mechanism plugins
  io/               # ScenarioBundle read/write
  evaluators/       # frozen grader
  baselines/        # sanity-check baselines
  scripts/          # CLI entry points
  examples/         # scenario YAML specs
  llm_harness/      # C2 offline ingest (no hot path)
  tests/
```

---

## 0.8 Acceptance criteria (per template)

1. Same seed → identical manifest hash  
2. All registry counterfactuals present with non-trivial label change  
3. Dependency ordering passes (original > remove_source; template-specific)  
4. Negative control below margin  
5. Leakage correlation below threshold  
6. No PyHazards tensor in generation path  
7. Baseline layering observable at high coupling  
8. `mhbench_implementation_report.md` documents execution evidence  
