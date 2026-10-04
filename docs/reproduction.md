# Reproduction

## Tables and figures without models

1. Install the package (`pip install -e ".[dev]"`).
2. `make verify` — manifest size, sealed contracts, released stats checksums, path hygiene.
3. `make reproduce-paper` — compares `released_results/` to `paper_artifacts/paper_manifest.yaml`.

Shipped SVG/TeX under `paper_artifacts/` matches the submission; the manifest documents the headline counts (e.g. Llama HWA **126/141** DCA).

## Re-scoring released trajectories

```bash
make evaluate-released
```

Reads `released_results/structured_trajectories/unified_141_per_instance_v1.jsonl` and writes `outputs/evaluate_released/summary.json`. It does not modify the released bundle.

## Re-running agents (optional)

Requires GPU or API access, model weights, and downloaded scientific data:

```bash
pip install -e ".[inference]"
python scripts/run_experiment.py --instance-id <id> --model-config configs/models/llama70b.yaml
```

Environment variables for vLLM/API endpoints are described in model config stubs under `configs/` (never commit secrets). A full 141×condition sweep is compute-heavy and not required to audit the paper tables.

## External data

`scripts/download_external_data.py` prints pointers to dataset acquisition steps; it does not download bulk files on the login node.
