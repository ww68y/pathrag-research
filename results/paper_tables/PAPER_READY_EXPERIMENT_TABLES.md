# Paper-Ready Experiment Tables

> Generated: 2026-09-19 16:52:07
> Scope: offline consolidation from completed result artifacts; no API calls.

## Recommended Claim Framing

- Headline candidate: `SAGE v33-preservation`, because it is reproducible, strong on Mix130, and safer than v56 despite v56's exploratory aggregate gains.
- Final holdouts: locked unseen75 is the primary post-freeze zero-overlap holdout; Repeat1-8 add strict zero-overlap robustness/stress evidence with complete judge and pairwise coverage across 326 rows.
- VSR contribution: use `full_v2_rescue` as the verification/repair/rescue mechanism layer, not as a blanket F1-superiority method against strong retrieval baselines.
- Metric framing: report lexical F1/ROUGE-L together with faithfulness, abstention, semantic judge, and cost/latency; do not mix old LLM-judge baselines with current lexical scores as equivalent metrics.

## Table 1. Strict VSR Ablation on UltraDomain Mix130

| Config | Rows | Status | F1 | ROUGE-L | Faithfulness | Repair% | Abstain% | Runtime s/q |
|---|---|---|---:|---:|---:|---:|---:|---:|
| w/o_all | 130/130 | complete | 0.2276 | 0.1658 | 0.5000 | 0.0% | 21.5% | 9.0 |
| w/o_evidence_extraction | 130/130 | complete | 0.2285 | 0.1612 | 0.5353 | 66.2% | 40.0% | 37.2 |
| w/o_verification | 130/130 | complete | 0.2138 | 0.1552 | 0.5000 | 0.0% | 18.5% | 17.6 |
| w/o_repair | 130/130 | complete | 0.2206 | 0.1591 | 0.8516 | 0.0% | 23.1% | 36.2 |
| full | 130/130 | complete | 0.2437 | 0.1749 | 0.8516 | 34.6% | 27.7% | 44.2 |
| full_v2 | 130/130 | complete | 0.2824 | 0.2107 | 0.9728 | 40.0% | 8.5% | 41.0 |
| full_v2_rescue | 130/130 | complete | 0.2878 | 0.2194 | 0.9756 | 27.7% | 0.8% | 41.7 |

## Table 2. Full vs Ablations Paired Tests

| Comparison | N | F1 Delta | F1 p | F1 d | ROUGE-L Delta | Faithfulness Delta | Faithfulness p |
|---|---:|---:|---:|---:|---:|---:|---:|
| full - w/o_all | 130 | +0.0161 | 0.0277 | +0.19 | +0.0091 | +0.3516 | <1e-15 |
| full - w/o_evidence_extraction | 130 | +0.0152 | 0.0641 | +0.16 | +0.0138 | +0.3163 | 1.78e-15 |
| full - w/o_verification | 130 | +0.0299 | 5.51e-07 | +0.44 | +0.0198 | +0.3516 | <1e-15 |
| full - w/o_repair | 130 | +0.0231 | 5.18e-05 | +0.35 | +0.0158 | +0.0000 | 1.0000 |
| full - full_v2 | 130 | -0.0386 | 2.51e-05 | -0.37 | -0.0357 | -0.1211 | 2.03e-07 |
| full - full_v2_rescue | 130 | -0.0441 | 3.46e-05 | -0.36 | -0.0444 | -0.1239 | 4.53e-07 |

## Table 3. VSR Rescue Cross-Dataset Internal Validation

| Dataset | N | w/o Repair F1 | Full v2 F1 | Full v2 Rescue F1 | Rescue Delta | p | d |
|---|---:|---:|---:|---:|---:|---:|---:|
| UltraDomain mix | 130 | 0.2206 | 0.2824 | 0.2878 | +0.0671 | 1.12e-11 | +0.60 |
| Biology full context | 50 | 0.1557 | 0.2121 | 0.2287 | +0.0729 | 3.85e-09 | +0.83 |
| Legal full context | 20 | 0.1791 | 0.2407 | 0.3496 | +0.1705 | 3.00e-04 | +0.81 |
| Domain expansion windowed | 50 | 0.1629 | 0.1986 | 0.2337 | +0.0708 | 1.95e-05 | +0.60 |

## Table 4. External Retrieval Baselines and SAGE v33

| Dataset | N | Naive F1 | HyDE F1 | VSR Rescue F1 | SAGE v33 F1 | Best Retrieval | SAGE-Best | p vs Best | VSR-Best |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| Legal full context | 20 | 0.3632 | 0.3760 | 0.3496 | 0.3925 | hyde | +0.0164 | 0.3035 | -0.0264 |
| Biology full context | 50 | 0.2365 | 0.2385 | 0.2287 | 0.2385 | hyde | +0.0000 | 1.0000 | -0.0098 |
| Domain expansion windowed | 50 | 0.2711 | 0.2549 | 0.2337 | 0.2881 | naive | +0.0170 | 0.2051 | -0.0374 |

## Table 5. GraphRAG-Lite / Legacy Overlap Supplement

| Dataset | N | GraphRAG-lite F1 | ROUGE-L | Judge Overall | PathRAG Pairwise Win | Runtime s/q | Best Method | Best F1 | VSR Rescue F1 | Legacy PathRAG F1 |
|---|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| UltraDomain mix | 130 | 0.3334 | 0.2525 | 2.646 | 89.38% | 10.3 | legacy_direct_context | 0.4905 | 0.2878 | 0.1815 |
| Legal full context | 20 | 0.3238 | 0.2745 | 2.600 | 94.50% | 6.0 | sage_v18_cost_aware_prerouter | 0.4496 | 0.3496 | 0.1308 |
| Biology full context | 50 | 0.2794 | 0.2286 | 2.720 | 97.40% | 8.7 | legacy_hyde | 0.3268 | 0.2287 | 0.1069 |
| Domain expansion windowed | 50 | 0.2928 | 0.2296 | 2.560 | 95.40% | 6.7 | legacy_naive | 0.2948 | 0.2337 | 0.1263 |

## Table 6. LightRAG Current Heavy Baseline Supplement

| Method | N | F1 | ROUGE-L | Judge Overall | PathRAG Pairwise Win | Avg Chars | Note |
|---|---:|---:|---:|---:|---:|---:|---|
| LightRAG current-130 | 130 | 0.2217 | 0.1691 | 4.277 | 30.69% | 1909.1 | Representative heavy graph-RAG baseline; evaluated current subset. |
| SAGE v33-preservation | 130 | 0.4646 | 0.3824 | N/A | N/A | 529.1 | Same evaluated 130-line LightRAG subset; current headline candidate. |
| GraphRAG-lite | 130 | 0.3334 | 0.2525 | N/A | N/A | 310.1 | Same evaluated 130-line LightRAG subset; lightweight community baseline. |
| PathRAG current | 130 | 0.1815 | 0.1254 | N/A | N/A | 2017.0 | Same evaluated 130-line LightRAG subset; pairwise reference side. |

## Table 7. SAGE Candidate Position

| Item | N | F1 | ROUGE-L | Runtime s/q | Delta / Note |
|---|---:|---:|---:|---:|---|
| SAGE v33-preservation on Mix130 | 130 | 0.4646 | 0.3824 | 13.8 | vs VSR rescue +0.1768; wins 113, losses 17 |
| Exploratory v56 weighted external replay | 120 | 0.3019 | 0.2402 | N/A | vs v33 F1 +0.0171; judge overall +0.0875; keep exploratory |

## Table 8. Locked Unseen75 Final Holdout

| Method | N | F1 | ROUGE-L | Judge Correct | Judge Coverage | Judge Overall | Excellent | Poor |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| naive | 75 | 0.2903 | 0.2223 | 3.253 | 3.027 | 3.133 | 26 | 28 |
| hyde | 75 | 0.2963 | 0.2257 | 3.427 | 3.173 | 3.280 | 25 | 21 |
| sage_v33_preservation_delta | 75 | 0.3096 | 0.2428 | 4.107 | 3.760 | 3.880 | 37 | 8 |

## Table 9. Locked Unseen75 Paired Gains

| Comparison | N | F1 Delta | ROUGE-L Delta | Judge Overall Delta | Overall p | Overall d | Pairwise Win Rate | Wins | Losses | Ties |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| SAGE v33 - HyDE | 75 | +0.0133 | +0.0171 | +0.600 | 1.27e-04 | +0.44 | 58.40% | 27 | 7 | 41 |
| SAGE v33 - Naive | 75 | +0.0193 | +0.0206 | +0.747 | 1.87e-06 | +0.55 | 63.47% | 33 | 8 | 34 |

## Table 10. Locked Unseen25 Repeat1 Supplement

| Method | N | F1 | ROUGE-L | Judge Overall | Runtime s/q | Avg Chars |
|---|---:|---:|---:|---:|---:|---:|
| naive | 25 | 0.2590 | 0.1938 | 3.040 | 12.4 | 868.4 |
| hyde | 25 | 0.2543 | 0.1938 | 3.040 | 23.5 | 907.4 |
| sage_v33_preservation_delta | 25 | 0.2797 | 0.2224 | 3.920 | 23.5 | 959.8 |

### Repeat1 Paired/Pairwise Gains

| Comparison | N | F1 Delta | F1 p | Pairwise Win Rate | Wins/Losses/Ties |
|---|---:|---:|---:|---:|---|
| SAGE v33 - HyDE | 25 | +0.0254 | 0.0966 | 62.80% | 6 / 1 / 18 |
| SAGE v33 - Naive | 25 | +0.0208 | 0.3329 | 69.60% | 14 / 11 / 0 |

## Table 11. Locked Unseen40 Repeat2 Supplement

| Method | N | F1 | ROUGE-L | Judge Overall | Runtime s/q | Avg Chars |
|---|---:|---:|---:|---:|---:|---:|
| naive | 40 | 0.2712 | 0.2019 | 3.050 | 8.3 | 710.3 |
| hyde | 40 | 0.2685 | 0.1908 | 3.025 | 19.2 | 716.9 |
| sage_v33_preservation_delta | 40 | 0.3059 | 0.2244 | 3.800 | 19.2 | 833.7 |

### Repeat2 Paired/Pairwise Gains

| Comparison | N | F1 Delta | F1 p | Pairwise Win Rate | Wins/Losses/Ties |
|---|---:|---:|---:|---:|---|
| SAGE v33 - HyDE | 40 | +0.0374 | 0.0065 | 62.75% | 11 / 2 / 27 |
| SAGE v33 - Naive | 40 | +0.0346 | 0.0475 | 66.50% | 23 / 17 / 0 |

## Table 12. Locked Unseen24 Repeat3 Supplement

| Method | N | F1 | ROUGE-L | Judge Overall | Runtime s/q | Avg Chars |
|---|---:|---:|---:|---:|---:|---:|
| naive | 24 | 0.2978 | 0.2338 | 3.125 | 14.9 | 595.1 |
| hyde | 24 | 0.3042 | 0.2467 | 2.917 | 32.6 | 623.2 |
| sage_v33_preservation_delta | 24 | 0.3541 | 0.3016 | 3.750 | 32.6 | 712.8 |

### Repeat3 Paired/Pairwise Gains

| Comparison | N | F1 Delta | F1 p | Pairwise Win Rate | Wins/Losses/Ties |
|---|---:|---:|---:|---:|---|
| SAGE v33 - HyDE | 24 | +0.0499 | 0.0952 | 64.58% | 5 / 2 / 17 |
| SAGE v33 - Naive | 24 | +0.0563 | 0.0626 | 56.25% | 16 / 8 / 0 |

## Table 13. Locked Unseen24 Repeat4 Supplement

| Method | N | F1 | ROUGE-L | Judge Overall | Runtime s/q | Avg Chars |
|---|---:|---:|---:|---:|---:|---:|
| naive | 24 | 0.2852 | 0.2290 | 3.167 | 27.1 | 579.0 |
| hyde | 24 | 0.2725 | 0.2215 | 3.167 | 48.5 | 629.2 |
| sage_v33_preservation_delta | 24 | 0.2725 | 0.2215 | 3.125 | 48.5 | 629.2 |

### Repeat4 Paired/Pairwise Gains

| Comparison | N | F1 Delta | F1 p | Pairwise Win Rate | Wins/Losses/Ties |
|---|---:|---:|---:|---:|---|
| SAGE v33 - HyDE | 24 | +0.0000 | 1.0000 | 50.00% | 0 / 0 / 24 |
| SAGE v33 - Naive | 24 | -0.0127 | 0.4251 | 49.58% | 9 / 14 / 1 |

## Table 14. Locked Legal40 Repeat5 Stress Holdout

| Method | N | F1 | ROUGE-L | Judge Overall | Runtime s/q | Avg Chars |
|---|---:|---:|---:|---:|---:|---:|
| naive | 40 | 0.3454 | 0.2772 | 2.975 | 11.6 | 603.5 |
| hyde | 40 | 0.3403 | 0.2741 | 2.975 | 19.9 | 659.2 |
| sage_v33_preservation_delta | 40 | 0.3659 | 0.3037 | 3.800 | 19.9 | 756.7 |

### Repeat5 Legal40 Paired/Pairwise Gains

| Comparison | N | F1 Delta | F1 p | Pairwise Win Rate | Wins/Losses/Ties |
|---|---:|---:|---:|---:|---|
| SAGE v33 - HyDE | 40 | +0.0256 | 0.0514 | 63.75% | 9 / 2 / 29 |
| SAGE v33 - Naive | 40 | +0.0205 | 0.2566 | 71.25% | 19 / 20 / 1 |

## Table 15. Locked History4 Repeat6 Tail Holdout

| Method | N | F1 | ROUGE-L | Judge Overall | Runtime s/q | Avg Chars |
|---|---:|---:|---:|---:|---:|---:|
| naive | 4 | 0.2183 | 0.1677 | 2.500 | 13.0 | 1020.5 |
| hyde | 4 | 0.2378 | 0.1956 | 2.250 | 36.8 | 755.2 |
| sage_v33_preservation_delta | 4 | 0.2298 | 0.1951 | 3.000 | 36.8 | 980.2 |

### Repeat6 History4 Paired/Pairwise Gains

| Comparison | N | F1 Delta | F1 p | Pairwise Win Rate | Wins/Losses/Ties |
|---|---:|---:|---:|---:|---|
| SAGE v33 - HyDE | 4 | -0.0080 | 0.3173 | 62.50% | 0 / 1 / 3 |
| SAGE v33 - Naive | 4 | +0.0115 | 0.5683 | 62.50% | 2 / 2 / 0 |

## Table 16. Locked Legal40 Repeat7 Stress Holdout

| Method | N | F1 | ROUGE-L | Judge Overall | Runtime s/q | Avg Chars |
|---|---:|---:|---:|---:|---:|---:|
| naive | 40 | 0.3283 | 0.2773 | 3.000 | 13.1 | 670.4 |
| hyde | 40 | 0.3430 | 0.2849 | 2.875 | 23.6 | 683.0 |
| sage_v33_preservation_delta | 40 | 0.3661 | 0.3144 | 3.975 | 23.6 | 817.8 |

### Repeat7 Legal40 Paired/Pairwise Gains

| Comparison | N | F1 Delta | F1 p | Pairwise Win Rate | Wins/Losses/Ties |
|---|---:|---:|---:|---:|---|
| SAGE v33 - HyDE | 40 | +0.0231 | 0.1272 | 68.50% | 12 / 4 / 24 |
| SAGE v33 - Naive | 40 | +0.0378 | 0.0502 | 67.50% | 26 / 14 / 0 |

## Table 17. Locked Legal54 Repeat8 Exhaustive Stress Holdout

| Method | N | F1 | ROUGE-L | Judge Overall | Runtime s/q | Avg Chars |
|---|---:|---:|---:|---:|---:|---:|
| naive | 54 | 0.2958 | 0.2344 | 2.444 | 13.4 | 621.7 |
| hyde | 54 | 0.2862 | 0.2174 | 2.352 | 29.4 | 621.0 |
| sage_v33_preservation_delta | 54 | 0.2862 | 0.2174 | 2.407 | 29.4 | 621.0 |

### Repeat8 Legal54 Paired/Pairwise Gains

| Comparison | N | F1 Delta | F1 p | Pairwise Win Rate | Wins/Losses/Ties |
|---|---:|---:|---:|---:|---|
| SAGE v33 - HyDE | 54 | +0.0000 | 1.0000 | 50.00% | 0 / 0 / 54 |
| SAGE v33 - Naive | 54 | -0.0097 | 0.4986 | 47.22% | 28 / 26 / 0 |

## Table 18. Locked Holdout Meta Summary (Unseen75 + Repeats 1-8)

| Method | N | F1 | ROUGE-L | Runtime s/q | Avg Chars |
|---|---:|---:|---:|---:|---:|
| naive | 326 | 0.2972 | 0.2338 | 14.2 | 665.9 |
| hyde | 326 | 0.2972 | 0.2317 | 25.1 | 687.6 |
| sage_v33_preservation_delta | 326 | 0.3164 | 0.2532 | 25.1 | 764.3 |

### Locked Meta Coverage

| Holdout | N | Methods | Judge | Pairwise | SAGE-HyDE F1 | SAGE-Naive F1 |
|---|---:|---|---|---|---:|---:|
| Locked Unseen75 | 75 | complete | complete | complete | +0.0133 | +0.0193 |
| Repeat1 N25 | 25 | complete | complete | complete | +0.0254 | +0.0208 |
| Repeat2 N40 | 40 | complete | complete | complete | +0.0374 | +0.0346 |
| Repeat3 N24 | 24 | complete | complete | complete | +0.0499 | +0.0563 |
| Repeat4 N24 | 24 | complete | complete | complete | +0.0000 | -0.0127 |
| Repeat5 Legal40 | 40 | complete | complete | complete | +0.0256 | +0.0205 |
| Repeat6 History4 | 4 | complete | complete | complete | -0.0080 | +0.0115 |
| Repeat7 Legal40 | 40 | complete | complete | complete | +0.0231 | +0.0378 |
| Repeat8 Legal54 | 54 | complete | complete | complete | +0.0000 | -0.0097 |

### Complete-Judge Weighted View

Judge-complete rows: `326/326`; incomplete judge values are not imputed.

| Method | N | Correct | Coverage | Overall |
|---|---:|---:|---:|---:|
| naive | 326 | 3.092 | 2.856 | 2.960 |
| hyde | 326 | 3.110 | 2.856 | 2.942 |
| sage_v33_preservation_delta | 326 | 3.816 | 3.469 | 3.555 |

### Complete-Pairwise Weighted View

| Comparison | N | Avg Target Win Rate |
|---|---:|---:|
| sage_v33_vs_hyde | 326 | 59.66% |
| sage_v33_vs_naive | 326 | 61.50% |

## Table 19. Cost and Latency Evidence

| Dataset | N | Naive s/q | HyDE s/q | VSR Rescue s/q | VSR/Naive | VSR/HyDE | VSR-Best F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Legal full context | 20 | 6.0 | 14.4 | 59.6 | 9.89 | 4.14 | -0.0264 |
| Biology full context | 50 | 6.8 | 20.7 | 46.9 | 6.88 | 2.26 | -0.0098 |
| Domain expansion windowed | 50 | 5.6 | 14.7 | 43.3 | 7.66 | 2.94 | -0.0374 |

### Token/Call Sample

| Sample | Method | N | LLM Calls/q | Tokens/q | Runtime s/q | Repair% | Rescue% | Faithfulness |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Biology full-context cost sample | w/o_repair | 5 | 3.0 | 6693.2 | 37.0 | 0.0% | N/A | 0.622 |
| Biology full-context cost sample | full_v2 | 5 | 5.0 | 9085.8 | 49.8 | 100.0% | N/A | 0.734 |
| Biology full-context cost sample | full_v2_rescue | 5 | 5.0 | 9085.8 | 49.8 | 100.0% | 0.0% | 0.734 |
| Domain expansion windowed cost sample | w/o_repair | 5 | 3.0 | 6950.6 | 41.5 | 0.0% | N/A | 0.877 |
| Domain expansion windowed cost sample | full_v2 | 5 | 3.8 | 7348.4 | 43.4 | 40.0% | N/A | 1.000 |
| Domain expansion windowed cost sample | full_v2_rescue | 5 | 3.8 | 7348.4 | 43.4 | 40.0% | 0.0% | 1.000 |
| Legal full-context cost sample | w/o_repair | 10 | 3.0 | 5825.3 | 40.6 | 0.0% | N/A | 0.959 |
| Legal full-context cost sample | full_v2 | 10 | 3.6 | 6387.2 | 42.8 | 30.0% | N/A | 1.000 |
| Legal full-context cost sample | full_v2_rescue | 10 | 6.5 | 16898.2 | 61.1 | 40.0% | 70.0% | 0.867 |

## Remaining Work and API Intensity

| Priority | Item | Status | API Intensity | Next Action |
|---|---|---|---|---|
| P0 | Freeze final method definition | Complete for paper planning: freeze SAGE v33-preservation; keep v56 exploratory. | Offline / done unless promoting a new guard | Use the v33 freeze/holdout audit as the method boundary and reserve v56 for discussion/future work. |
| P0 | Same-protocol baselines | Naive/HyDE complete on external slices; GraphRAG-lite covers Mix130, Legal, Biology, and Domain; LightRAG current-130 is complete as an evaluated representative heavy baseline; 130 valid predictions have been generated in total, with all 130 judge-evaluated and all 130 pairwise-evaluated; no judge, pairwise, or retry rows remain. Full official GraphRAG/LightRAG comparability remains approximate. | Low for reporting completed GraphRAG-lite/LightRAG subset evidence; high only for official GraphRAG or full LightRAG reruns. | Use GraphRAG-lite as the lightweight community baseline and LightRAG current-130 as the evaluated high-cost representative comparator unless budget supports full official reruns. |
| P0 | Strict ablation table | Complete: all seven UltraDomain mix130 aligned configs have 130/130 valid rows and paired tests. | Done / offline only | Use this as the main internal mechanism ablation table. |
| P0 | Multi-dataset validation | Complete: prior external slices plus locked unseen75 and Repeat1-8 strict holdouts now include lexical, same-rubric judge, and pairwise A/B evaluation. | Done / offline unless adding a different frozen method or more judge replications. | Use locked unseen75 as the primary final post-freeze holdout and Repeat1-8 as supplementary robustness/stress evidence; do not tune on any locked holdout. |
| P0 | Final paper tables | Consolidated in this output and exported as include-ready LaTeX fragments in `paper/generated_experiment_tables.tex`. | Offline | Input or copy the generated fragments into the paper draft after final claim wording is frozen. |
| P1 | Cost and latency | Runtime complete; exact token accounting now sampled across Legal N=10, Biology N=5, and Domain-windowed N=5 with per-stage call/token breakdown. | Offline for current summary; medium only if expanding any slice beyond the current sample. | Use cross-domain N=20 accounting now; optionally expand Domain/Biology only if reviewers require tighter cost intervals. |
| P1 | Error analysis | 15-row LLM audit, longer-context diagnostic, cost-quality case audit, and exploratory i18n abstention/language guard audit are available. | Offline unless adding human/LLM judge cases. | Use the i18n guard only as future-validation framing; select representative cases for the paper discussion section. |
| P1 | Reproducibility manifest | Complete: manifest refreshed after Repeat8 judge and pairwise artifacts were added. | Offline | Refresh again only after future API reruns, table regeneration, or artifact changes. |

## Source Files

- `strict_ablation`: `results/enhanced_ablation_aligned/aligned_ablation_study_n130.json`
- `strict_ablation_stats`: `results/enhanced_ablation_aligned/aligned_ablation_stat_tests_n130.json`
- `cross_dataset_vsr`: `results/enhanced_ablation_aligned/cross_dataset_aligned_progress_summary.json`
- `retrieval_baselines`: `results/retrieval_baselines/retrieval_baseline_aligned_summary.json`
- `sage_final_candidate`: `results/sage_final_candidate/sage_final_candidate_summary.json`
- `cost_latency`: `results/cost_latency/cost_latency_summary.json`
- `cost_quality_case_audit`: `results/cost_latency/cost_quality_case_audit.json`
- `i18n_abstention_language_guard_audit`: `results/cost_latency/i18n_abstention_language_guard_audit.json`
- `legacy_overlap`: `results/legacy_baseline_overlap/legacy_baseline_overlap_summary.json`
- `lightrag_current_subset`: `results/historical_baselines/lightrag_current_extension_report.md`
- `locked_unseen75_lexical`: `results/sage_locked_unseen75/sage_locked_unseen75_baseline_summary.json`
- `locked_unseen75_judge`: `results/sage_locked_unseen75/sage_locked_unseen75_judge_summary.json`
- `locked_unseen75_pairwise`: `results/sage_locked_unseen75/sage_locked_unseen75_pairwise_summary.json`
- `locked_unseen25_repeat1`: `results/sage_locked_unseen25_repeat1/sage_locked_unseen25_repeat1_summary.json`
- `locked_unseen40_repeat2`: `results/sage_locked_unseen40_repeat2/sage_locked_unseen40_repeat2_summary.json`
- `locked_unseen24_repeat3`: `results/sage_locked_unseen24_repeat3/sage_locked_unseen24_repeat3_summary.json`
- `locked_unseen24_repeat4`: `results/sage_locked_unseen24_repeat4/sage_locked_unseen24_repeat4_summary.json`
- `locked_legal40_repeat5`: `results/sage_locked_legal40_repeat5/sage_locked_legal40_repeat5_summary.json`
- `locked_history4_repeat6`: `results/sage_locked_history4_repeat6/sage_locked_history4_repeat6_summary.json`
- `locked_legal40_repeat7`: `results/sage_locked_legal40_repeat7/sage_locked_legal40_repeat7_summary.json`
- `locked_legal54_repeat8`: `results/sage_locked_legal54_repeat8/sage_locked_legal54_repeat8_summary.json`
- `locked_unseen_meta`: `results/sage_locked_unseen_meta/sage_locked_unseen_meta_summary.json`
