# Unified @141 — 图与表数据（DL-677 后）

`built_utc`: 2026-09-23T20:20:27Z · N=141

## 图文件

| 图 | 文件 | 数据节 |
|----|------|--------|
| RQ2 Fig 4a | `rq2_fig4a_scientific_constraints.svg` | RQ2 @96 |
| RQ2 Fig 4b | `rq2_fig4b_route_updating_table.tex` | RQ2 W3-40 |
| RQ3 Fig 5a | `rq3_fig5a_hazard_composition_grouped.svg` | RQ3 边际 Single/Multi |
| RQ3 Fig 5b | `rq3_fig5b_route_multiplicity_grouped.svg` | RQ3 边际 One/Multi-route |
| RQ3 Fig 6 | `rq3_fig6_track_dca_radar.svg` | RQ3 Track-level |
| RQ3 Fig 6 heatmap | `rq3_fig6_track_dca_heatmap.svg` | Track DCA % matrix |
| Appendix inventory | `appendix_hwb_inventory_bar.svg` | Sealed 141 track N |
| Appendix joint strata | `appendix_joint_strata_heatmap.svg` | 2×2 hazard×route counts |
| Appendix joint DCA | `appendix_joint_stratum_dca_bars.svg` | HWA / Plan / AIDE × 4 strata |
| Appendix McNemar | `appendix_mcnemar_dumbbell.svg` | Discordant paired counts |
| Appendix info sep. | `appendix_information_separation.svg` | Agent vs evaluator matrix |
| Appendix RQ2 | `rq2_fig4a_scientific_constraints.svg` | replaces appendix RQ2 table |
| Appendix | `rq3_decision_steps.svg` | 决策步数 |

## Table 1 — Adapted system baselines

| Method | Single-hazard | Multi-hazard | One route | Multiple routes | Overall DCA |
|--------|---------------|--------------|-----------|-----------------|-------------|
| Plan-and-Execute faithful | 37/90 (41.1% [31.5, 51.4]) | 21/51 (41.2% [28.8, 54.8]) | 26/45 (57.8% [43.3, 71.0]) | 32/96 (33.3% [24.7, 43.2]) | 41.1% [33.4, 49.4] |
| Self-Consistency faithful | 30/90 (33.3% [24.4, 43.6]) | 20/51 (39.2% [27.0, 52.9]) | 19/45 (42.2% [29.0, 56.7]) | 31/96 (32.3% [23.8, 42.2]) | 35.5% [28.1, 43.6] |
| DisasterBench-ToT faithful | 32/90 (35.6% [26.4, 45.9]) | 18/51 (35.3% [23.6, 49.0]) | 19/45 (42.2% [29.0, 56.7]) | 31/96 (32.3% [23.8, 42.2]) | 35.5% [28.1, 43.6] |
| MLE-STAR faithful | 32/90 (35.6% [26.4, 45.9]) | 15/51 (29.4% [18.7, 43.0]) | 16/45 (35.6% [23.2, 50.2]) | 31/96 (32.3% [23.8, 42.2]) | 33.3% [26.1, 41.5] |
| AutoML-Agent faithful | 24/90 (26.7% [18.6, 36.6]) | 14/51 (27.4% [17.1, 41.0]) | 11/45 (24.4% [14.2, 38.7]) | 27/96 (28.1% [20.1, 37.8]) | 26.9% [20.3, 34.8] |
| Reflexion faithful | 27/90 (30.0% [21.5, 40.1]) | 18/51 (35.3% [23.6, 49.0]) | 23/45 (51.1% [37.0, 65.0]) | 22/96 (22.9% [15.7, 32.3]) | 31.9% [24.8, 40.0] |
| HazardWeaver | 82/90 (91.1% [83.4, 95.4]) | 44/51 (86.3% [74.3, 93.2]) | 36/45 (80.0% [66.2, 89.1]) | 90/96 (93.8% [87.0, 97.1]) | 89.4% [83.2, 93.5] |

## Table 2 — Shared-tool control baselines

| Method | Single-hazard | Multi-hazard | One route | Multiple routes | Overall DCA |
|--------|---------------|--------------|-----------|-----------------|-------------|
| AIDE faithful | 20/90 (22.2% [14.9, 31.9]) | 12/51 (23.5% [14.0, 36.8]) | 13/45 (28.9% [17.7, 43.4]) | 19/96 (19.8% [13.1, 28.9]) | 22.7% [16.6, 30.3] |
| DS-Agent faithful | 30/90 (33.3% [24.4, 43.6]) | 17/51 (33.3% [22.0, 47.0]) | 18/45 (40.0% [27.0, 54.5]) | 29/96 (30.2% [21.9, 40.0]) | 33.3% [26.1, 41.5] |
| R&D-Agent faithful | 30/90 (33.3% [24.4, 43.6]) | 18/51 (35.3% [23.6, 49.0]) | 19/45 (42.2% [29.0, 56.7]) | 29/96 (30.2% [21.9, 40.0]) | 34.0% [26.7, 42.2] |
| ReAct same-tool | 26/90 (28.9% [20.5, 39.0]) | 16/51 (31.4% [20.3, 45.0]) | 14/45 (31.1% [19.5, 45.7]) | 28/96 (29.2% [21.0, 38.9]) | 29.8% [22.9, 37.8] |
| HazardWeaver | 82/90 (91.1% [83.4, 95.4]) | 44/51 (86.3% [74.3, 93.2]) | 36/45 (80.0% [66.2, 89.1]) | 90/96 (93.8% [87.0, 97.1]) | 89.4% [83.2, 93.5] |

## Paired McNemar（HWA vs baseline @141）

HWA pass: 126/141

| Baseline | HW-only | BL-only | Discordant n | p-value |
|----------|--------:|--------:|-------------:|--------:|
| Plan-and-Execute faithful | 75 | 7 | 82 | 1.37e-13 |
| Self-Consistency faithful | 80 | 4 | 84 | 2.76e-16 |
| DisasterBench-ToT faithful | 80 | 4 | 84 | 2.76e-16 |
| MLE-STAR faithful | 80 | 1 | 81 | 4.45e-18 |
| AutoML-Agent faithful | 93 | 5 | 98 | 1.52e-18 |
| Reflexion faithful | 89 | 8 | 97 | 4.56e-16 |
| AIDE faithful | 94 | 0 | 94 | 8.62e-22 |
| DS-Agent faithful | 82 | 3 | 85 | 2.67e-17 |
| R&D-Agent faithful | 82 | 4 | 86 | 1.01e-16 |
| ReAct same-tool | 84 | 0 | 84 | 1.35e-19 |

## 四格联合分解（DCA k/n）

分母：single×one 29 · single×multi 61 · multi×one 16 · multi×multi 35

| 条件 | HWA | Plan-Execute | AIDE |
|------|----:|-------------:|-----:|
| single × one route | 24/29 | 13/29 | 7/29 |
| single × multi-route | 58/61 | 24/61 | 13/61 |
| multi × one route | 12/16 | 13/16 | 6/16 |
| multi × multi-route | 32/35 | 8/35 | 6/35 |

## HWA per-track（Llama @141）

| Track | k/N |
|-------|----:|
| WF-3 | 12/13 |
| FL-2 | 6/12 |
| L2 | 14/14 |
| E1-E3 | 13/14 |
| TC-TRK | 6/6 |
| DR-OUT | 14/14 |
| HW-MED | 17/17 |
| MH-1 | 12/16 |
| MH-2 | 10/12 |
| MH-3 | 10/11 |
| MH-4 | 12/12 |

## RQ2 — Stratum M @96（gold-match composite，= @141 M 层）

| Backbone | HKC off k/96 (%) | HCG untyped k/96 (%) | Full HWA k/96 (%) |
|----------|-----------------:|---------------------:|------------------:|
| Llama-3.3-70B | 44 (45.8%) | 46 (47.9%) | 90 (93.8%) |
| Gemma-3-31B | 50 (52.1%) | 55 (57.3%) | 76 (79.2%) |
| Mixtral-8x22 | 34 (35.4%) | 32 (33.3%) | 65 (67.7%) |
| OLMo-3-32B | 38 (39.6%) | 39 (40.6%) | 49 (51.0%) |
| DeepSeek-V4.1 | 68 (70.8%) | 71 (74.0%) | 86 (89.6%) |

## RQ2 — W3-40 route updating（Fig 4b 表）

分母：n=40 gated cells

| Backbone | Fixed initial eligibility | Full HWA |
|----------|--------------------------|----------|
| Mixtral-8x22B | 2/40 (5.0% [1.4, 16.5]) | 6/40 (15.0% [7.1, 29.1]) |
| OLMo-3.1-32B | 7/40 (17.5% [8.8, 31.9]) | 10/40 (25.0% [14.2, 40.2]) |
| DeepSeek-V4.1-Flash | 9/40 (22.5% [12.3, 37.5]) | 14/40 (35.0% [22.1, 50.5]) |
| Gemma-4-31B-it | 8/40 (20.0% [10.5, 34.8]) | 17/40 (42.5% [28.5, 57.8]) |
| Llama-3.3-70B | 6/40 (15.0% [7.1, 29.1]) | 36/40 (90.0% [77.0, 96.0]) |

## RQ3 — Main table（五骨干 Full HWA @141）

| Backbone | DCA ↑ [95% CI] | Macro DCA ↑ [95% CI] | E_q ↑ [95% CI] | V_q ↑ [95% CI] |
|----------|----------------|----------------------|----------------|----------------|
| OLMo | 60.3% [52.0, 68.0] | 61.9% [54.4, 71.2] | 78.0% [70.5, 84.1] | 91.5% [85.7, 95.1] |
| Mixtral | 70.2% [62.2, 77.1] | 70.8% [57.6, 83.9] | 79.4% [72.0, 85.3] | 92.2% [86.6, 95.6] |
| Gemma | 79.4% [72.0, 85.3] | 78.6% [65.2, 91.5] | 86.5% [79.9, 91.2] | 92.9% [87.4, 96.1] |
| DeepSeek | 86.5% [79.9, 91.2] | 86.6% [77.2, 94.3] | 90.8% [84.9, 94.5] | 92.2% [86.6, 95.6] |
| Llama | 89.4% [83.2, 93.5] | 89.5% [79.7, 97.0] | 92.2% [86.6, 95.6] | 95.0% [90.1, 97.6] |

## RQ3 — E_q / V_q / dual_gate / DCA（k/141）

| Backbone | E_q | V_q | dual_gate | DCA |
|----------|----:|----:|----------:|----:|
| OLMo | 110 | 129 | 110 | 85 |
| Mixtral | 112 | 130 | 112 | 99 |
| Gemma | 122 | 131 | 122 | 112 |
| DeepSeek | 128 | 130 | 127 | 122 |
| Llama | 130 | 134 | 130 | 126 |

## RQ3 — Hazard / Route 边际（DCA %）

| Backbone | Single | Multi | One route | Multi-route |
|----------|-------:|------:|----------:|------------:|
| OLMo | 63.3 | 54.9 | 80.0 | 51.0 |
| Mixtral | 82.2 | 49.0 | 75.6 | 67.7 |
| Gemma | 91.1 | 58.8 | 80.0 | 79.2 |
| DeepSeek | 90.0 | 80.4 | 80.0 | 89.6 |
| Llama | 91.1 | 86.3 | 80.0 | 93.8 |

## RQ3 — 四格联合（Llama k）

| 格 | k |
|----|--:|
| single × one route | 24 |
| single × multi-route | 58 |
| multi × one route | 12 |
| multi × multi-route | 32 |

## RQ3 — Track-level DCA（k/N）

| Track | OLMo | Mixtral | Gemma | DeepSeek | Llama |
|-------|-----:|--------:|------:|---------:|------:|
| WF-3 | 7/13 | 8/13 | 12/13 | 12/13 | 12/13 |
| FL-2 | 6/12 | 6/12 | 6/12 | 6/12 | 6/12 |
| L2 | 10/14 | 12/14 | 14/14 | 14/14 | 14/14 |
| E1-E3 | 8/14 | 12/14 | 13/14 | 12/14 | 13/14 |
| TC-TRK | 6/6 | 6/6 | 6/6 | 6/6 | 6/6 |
| DR-OUT | 9/14 | 14/14 | 14/14 | 14/14 | 14/14 |
| HW-MED | 11/17 | 16/17 | 17/17 | 17/17 | 17/17 |
| MH-1 | 9/16 | 6/16 | 12/16 | 12/16 | 12/16 |
| MH-2 | 5/12 | 6/12 | 6/12 | 10/12 | 10/12 |
| MH-3 | 6/11 | 8/11 | 6/11 | 10/11 | 10/11 |
| MH-4 | 8/12 | 5/12 | 6/12 | 9/12 | 12/12 |

## 决策步数（median steps）

| Backbone | median valid | median invalid | timeout_rate @141 |
|----------|-------------:|---------------:|------------------:|
| OLMo | 2 | 2 | 0.0% |
| Mixtral | 5 | 7 | 0.0% |
| Gemma | 2 | 2 | 0.0% |
| DeepSeek | 2 | 5 | 0.0% |
| Llama | 3 | 3 | 0.0% |

## Failure-stage 计数（HWA 五骨干 @141）

| 阶段 | Llama | Gemma | Mixtral | OLMo | DeepSeek |
|------|------:|------:|--------:|-----:|---------:|
| success | 126 | 112 | 100 | 85 | 122 |
| route_selection | 0 | 0 | 14 | 0 | 0 |
| scientific_execution | 1 | 2 | 4 | 1 | 3 |
| recovery | 12 | 26 | 21 | 43 | 15 |
| abstention | 1 | 1 | 0 | 1 | 0 |
| budget_timeout | 1 | 0 | 2 | 11 | 0 |

## Baseline failure bucket @141

| Method | PASS | ROUTE_WRONG | EXEC_FAIL |
|--------|-----:|------------:|----------:|
| Plan-and-Execute faithful | 58 | 83 | 0 |
| Self-Consistency faithful | 50 | 91 | 0 |
| DisasterBench-ToT faithful | 50 | 91 | 0 |
| MLE-STAR faithful | 47 | 94 | 0 |
| AutoML-Agent faithful | 38 | 103 | 0 |
| Reflexion faithful | 45 | 96 | 0 |
| AIDE faithful | 32 | 103 | 6 |
| DS-Agent faithful | 47 | 94 | 0 |
| R&D-Agent faithful | 48 | 93 | 0 |
| ReAct same-tool | 42 | 99 | 0 |
