# Curie baseline — INFRA_BLOCKED (Docker)

**Status:** `INFRA_BLOCKED` on HPG login nodes and typical compute partitions without Docker.

## Why blocked

1. **Infrastructure:** Curie requires Docker (`curie/ExpDockerfile_default`) to execute generated experiment code. HPG login nodes have no Docker (`docker ps` → `NO_DOCKER`). Feasibility evidence: `runs/benchmark/system_baselines/feasibility_v1/curie_llama8b_trial.json`.
2. **Charter (superseded):** `docs/system_baseline.md` §4 originally deferred Curie until venue confirmation. User directive **2026-09-02** overrides this for feasibility work only; full HWB headline matrix remains **blocked** until Docker path PASS on compute.

## What is delivered instead

- Vendor pin + `health_check.py` (vendor import OK)
- Headline **smoke-only** adapter skeleton (`adapter.py`, `run_fixture.py`)
- LLM-only probe script in `scripts/benchmark/run_feasibility_trials_v1.py` (not a Curie PASS)

## Unblock checklist

1. Run on a node with Docker: `docker ps` succeeds
2. Build image: `docker build -t exp-agent-image -f curie/ExpDockerfile_default .` under `vendor/baselines/curie`
3. Point `OPENAI_BASE_URL` at vLLM Llama-8B (`scripts/hpg/serve_vllm_llama8b_rtx6000.sbatch`)
4. Re-run `scripts/hpg/run_curie_llama8b_feasibility_v1.sbatch` → update `curie_llama8b_trial.json` to PASS
5. Replace this file with full `adapter.py` live path

## References

- Paper: arXiv:2502.16069
- Repo: https://github.com/Just-Curieous/Curie
- Vendor pin: `vendor_pin.json`
