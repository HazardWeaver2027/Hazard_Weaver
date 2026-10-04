# HazardWeaver

**State-dependent scientific route selection for hazard-analysis agents.**

HazardWeaver is a framework for selecting and revising model-backed scientific workflows as an analysis evolves. Given a single- or multi-hazard task and the current analysis state, it grounds route applicability in scientific evidence, composes heterogeneous models and tools through typed capability relations, and selects, executes, and revises eligible routes as evidence and execution conditions change. This repository provides the HazardWeaver implementation, the sealed 141-instance Hazard Weaver Benchmark (HWB), the unified Decision-Constrained Accuracy (DCA) evaluator, and scripts for reproducing the paper’s results.


![HazardWeaver system overview](paper_artifacts/figures/HW_framework.png)

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Set `PYTHON` if needed, e.g. `export PYTHON=.venv/bin/python`.

## Verify and reproduce paper numbers

```bash
make verify PYTHON=$PYTHON
make reproduce-paper PYTHON=$PYTHON
```

Optional checks:

```bash
make smoke PYTHON=$PYTHON
make evaluate-released PYTHON=$PYTHON
python scripts/check_public_paths.py
pytest tests -q
```

- **`make smoke`** — loads a small public taskpack, runs the leakage audit (solver view must not see sealed fields), and checks core imports. No GPU.
- **`make evaluate-released`** — reads the released per-instance JSONL and checks that headline Llama DCA still matches `released_results/` (expected **k = 126**).

## Layout

| Directory | Contents |
|-----------|----------|
| `src/hazardweaver/` | HKC, HCG, HWA, HWB, baselines |
| `benchmark/public/` | Manifest, solver taskpacks (sealed inventory) |
| `benchmark/sealed_eval/` | Evaluator contracts (post-submission scoring only) |
| `released_results/` | JSON/JSONL used in the paper |
| `paper_artifacts/` | Figures, LaTeX/table sources, manifest |

### Release overview

**HWB** is a decision-layer benchmark: each instance is a hazard decision task with a public solver view (instructions, route briefs, tool schema) and post-submission scoring against evaluator contracts. The inventory spans **11 tracks** (wildfire, flood, landslide, drought, hurricane, heat, earthquake–landslide chains, multi-hazard coupling MH-1–4, etc.) across **90 single-hazard** and **51 multi-hazard** cases, with **45** instances having a single admissible route at \(s_0\) and **96** multi-route at \(s_0\).

**HazardWeaver (HWA)** is the headline agent: one LLM policy selects among HCG-backed capabilities under dual scientific/capability gates, with the same tool surface as the paper baselines. This repo ships the evaluation stack (HKC + HCG runtime hooks + HWB graders), **adapted planning baselines** and **shared-tool controls** from Tables 1–2, and **released aggregates** aligned with the paper tables.

**Data and models.** Tasks are grounded in public hazard products—WildfireSpreadTS, FloodCastBench, LHASA, SeisBench, TCBench, NOAA CPC, ExtremeWeatherBench, USGS post-fire and ground-failure layers, SFINCS, NOAA Storm Events, and related catalogs. Full rasters, HCG scientific run trees, and **Llama / vLLM weights** are downloaded or mounted separately; see [docs/benchmark.md](docs/benchmark.md) and `scripts/download_external_data.py`. CPU **fixture taskpacks** support smoke tests without multi-terabyte assets.

**Paper artifacts.** Table and figure sources, headline JSON, and per-instance trajectory summaries live under `paper_artifacts/` and `released_results/`. Re-scoring and optional agent re-runs: [docs/reproduction.md](docs/reproduction.md).

*Figure 1 — Global coverage of HWB.*

![Global coverage of HWB](paper_artifacts/figures/global_coverage_map.png)

## License

MIT ([LICENSE](LICENSE)). For anonymous review, `configs/release.yaml` uses a placeholder copyright; use `configs/release_lab.yaml` when publishing under your lab name.
