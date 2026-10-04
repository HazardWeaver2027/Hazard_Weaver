# Benchmark boundary

## Public (`benchmark/public/`)

Solver-visible material only: task specifications, initial state, route briefs, capability interfaces, and public metric hints.

## Sealed evaluation (`benchmark/sealed_eval/`)

Evaluator-only contracts, reference bindings, and scoring manifest. **Solvers must not read this directory.**

Evaluation runs only after a terminal submission is committed.

## Counts (sealed @141)

- 141 instances, 11 tracks
- 90 single-hazard / 51 multi-hazard
- 45 single-route / 96 multi-route at initial state
