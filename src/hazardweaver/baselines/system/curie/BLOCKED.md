# Curie baseline — INFRA_BLOCKED (Docker)

**Status:** `INFRA_BLOCKED` on environments without Docker.

## Why blocked

Curie requires Docker (`curie/ExpDockerfile_default`) to execute generated experiment code. Nodes without a working Docker daemon cannot run the vendor workflow end-to-end.

## What is delivered instead

- Vendor pin + `health_check.py` (vendor import OK)
- Headline **smoke-only** adapter skeleton (`adapter.py`, `run_fixture.py`)
- Feasibility probe hooks documented for maintainers (not a Curie PASS)

## Unblock checklist

1. Run on a host where `docker ps` succeeds
2. Build the vendor image under `vendor/baselines/curie`
3. Point `OPENAI_BASE_URL` at a vLLM OpenAI-compatible endpoint
4. Re-run feasibility and record PASS in your private run logs

## References

- Paper: arXiv:2502.16069
- Repo: https://github.com/Just-Curieous/Curie
- Vendor pin: `vendor_pin.json`
