# VSR-PathRAG SCI Submission Experiment Readiness

> Date: 2026-09-18  
> Target: SCI Q1/Q2 first submission  
> Strongest VSR ablation result: `full_v2_rescue` on UltraDomain aligned N=130  
> Strongest reproducible lexical-F1 candidate: SAGE v33-preservation adaptive routing on mix130  
> Latest update: all currently planned strict ablation and locked comparison experiments are complete under the frozen SAGE v33-preservation policy. The UltraDomain Mix130 strict ablation covers all seven configurations with 130/130 valid rows and paired tests. GraphRAG-lite covers Mix130 plus Legal/Biology/Domain external slices, and LightRAG current-130 is fully generated, judge-evaluated, and pairwise-evaluated as the representative heavy graph-RAG comparator. Locked unseen75 plus Repeat1-8 now provide 326/326 rows with complete Naive/HyDE/SAGE v33 method outputs, same-rubric judge scores, and pairwise A/B evaluations. Weighted locked-holdout results are SAGE v33 F1 0.3164 versus Naive 0.2972 and HyDE 0.2972; judge overall 3.555 versus Naive 2.960 and HyDE 2.942; and average SAGE target win rates of 59.66% versus HyDE and 61.50% versus Naive. The strict source-line/source-context holdout capacity is exhausted under the current max-two-samples-per-context policy, so additional comparison experiments require new source data, a new frozen method, or an explicit policy relaxation.

## Current Status

The project has moved past early exploratory testing. The current experimental stack contains:

- Complete aligned UltraDomain N=130 strict ablation results for `w/o_all`, `w/o_evidence_extraction`, `w/o_verification`, `w/o_repair`, `full`, `full_v2`, and `full_v2_rescue`, all with 130/130 valid rows and no error rows.
- A formal online `full_v2_rescue` runner branch with abstention-triggered 40k-context rescue.
- Cross-dataset aligned validation on Biology full-context N=50, Legal full-context N=20, and Domain-expansion windowed N=50.
- Statistical tests, bootstrap confidence intervals, and rescue-only paired comparisons.
- Quality audits for low-faithfulness and final-abstention rows.
- Cross-dataset quality/gate diagnostics showing that naive longer-context rescue and simple rejection gates are not yet robust enough.
- Legal N=20 repeat/stability run showing the positive rescue effect persists under a second run, with lower absolute F1.
- Aligned Naive/HyDE retrieval baselines on Legal, Biology, and Domain-expansion slices.
- GraphRAG-lite/community baseline coverage is now complete for Mix130 N=130 plus Legal N=20, Biology N=50, and Domain-windowed N=50; Mix130 judge overall is 2.646 and PathRAG beats it by 89.38% average pairwise win rate, so this remains a lightweight comparator rather than a strong official GraphRAG replacement.
- LightRAG current-130 is complete as the evaluated representative heavy graph-RAG comparator: 130/130 valid predictions, 130/130 judge rows, 130/130 PathRAG-vs-LightRAG pairwise rows, judge overall 4.277, lexical F1 0.2217, and PathRAG pairwise win rate 30.69%.
- Retrieval-prefiltered VSR diagnostic on Legal top-5 retrieved contexts, including complete top5-only, ctx12k-window, and top5+full-prepend runs.
- Initial runtime/cost diagnostic summarizing mean seconds per question, repair/rescue rates, and remaining token/call logging gaps.
- Instrumented full-context cost samples across Legal N=10, Biology N=5, and Domain-windowed N=5, with per-stage LLM calls and token usage.
- Cost-quality case audit over the same N=20 instrumented rows, flagging one Domain F1 regression, two Legal faithfulness drops, and two missed abstention detections.
- Exploratory i18n abstention/language guard audit over the same N=20 cost sample, flagging seven missed abstention-style outputs but limiting automatic fallback to the one English-question/Chinese-answer mismatch; simulated F1 moves 0.2367 -> 0.2453, while faithfulness moves 0.867 -> 0.853. A broader full aligned VSR scan covers 250 rows and finds 0 language mismatches but 55 missed abstention-style outputs, so this is detector-calibration evidence only, not a retroactive main-result change.
- Offline lexical re-scoring of existing legacy PathRAG/Naive/HyDE/GraphRAG/SAGE outputs to separate original-style LLM-judge results from current F1/ROUGE evidence.
- SAGE/adaptive-routing candidate scan across existing result files, identifying v31/v32/v33 mix130 and v17 legal20 as stronger lexical-F1 leads than the current VSR rescue line.
- SAGE-vs-baselines consolidation showing SAGE v33-preservation exceeds aligned VSR `full_v2_rescue` on mix130 by +0.1768 paired F1 with 113 wins / 17 losses, while keeping this as candidate-discovery evidence rather than a frozen final main table.
- SAGE external risk-pool probes now cover Domain-windowed all 40 risk rows, Legal full-context all 9 risk rows, and Biology full-context all 32 risk rows. Guarded v32 is positive on Domain (full 50-row F1 0.2549 -> 0.2881; judge overall 2.92 -> 4.40) and Legal (full 20-row F1 0.3760 -> 0.3925; judge overall 4.11 -> 4.67). A v33-preservation guard fixes the Biology line-46 regression by reverting to the original answer, returning Biology to baseline rather than making a positive gain.
- SAGE external-vs-baselines consolidation now compares v33-preservation against Naive, HyDE, and VSR `full_v2_rescue` on Legal, Biology, and Domain slices under the same lexical-F1 scorer.
- SAGE locked full-slice ladder now scores Mix130, Legal full-context, Biology full-context, and Domain-windowed slices from frozen artifacts with complete row coverage. Weighted external F1 is v31=0.2851 and v33=0.2848; v33 is preferred for safety because it removes the Biology regression.
- Same-rubric full-slice judge evaluation is complete for Naive, HyDE, VSR `full_v2_rescue`, and SAGE v33 on all external rows. Weighted external overall scores are SAGE v33=4.21, best retrieval=3.68, HyDE=3.72, Naive=3.56, and VSR rescue=3.17.
- Paired judge tests are now available: v33 versus best retrieval is strongly positive on Domain (+1.20 overall, p=2.88e-07), while Legal (+0.10) and Biology (+0.03) remain small/nonsignificant; v33 versus VSR rescue is positive on all three external slices.
- Final candidate summary now consolidates mix130, external slices, the locked full-slice ladder, v33 safety behavior, and strict reproducibility into one paper-planning artifact.
- Consolidated paper-ready experiment tables now collect strict ablation, VSR cross-dataset internal validation, external retrieval baselines, SAGE candidate positioning, cost/latency, and API-intensity gap audit in `results/paper_tables/PAPER_READY_EXPERIMENT_TABLES.md`.
- Context-heldout unseen34 validation now uses the current v33-preservation artifact. It is prediction-identical to the old frozen v33 artifact across all Mix130 rows, reuses the existing judge scores, and shows a small positive held-out shift versus baseline: lexical F1 0.4876 -> 0.4906 and judge overall 4.353 -> 4.412. This should be framed as guard safety evidence, not a large unseen benchmark result.
- V46 zero-overlap validation30 now adds a second unseen-style check from an existing low5 contract-completion source. Current v33 changes lines 22 and 26 versus HyDE, keeps both repairs, and improves lexical F1 0.3126 -> 0.3249 and judge overall 3.600 -> 3.867 under the same rubric.
- The older SAGE v55 supported-facet branch has now been reconciled into the unseen summary as an exploratory v46 ladder point. It is stronger on v46 than current v33 (F1 0.3627 and judge overall 4.300), but it is not yet a replacement freeze candidate because it has not been run through the same Mix130/full-slice reproduction protocol.
- A late-stage SAGE v54/v55/v56 candidate ladder now consolidates v40 heldout12, v42 heldout12, and v46 zero-overlap30. It shows v56 as the best small heldout branch overall, but only as a promotion candidate until it is replayed on Mix130/full-slice protocol.
- The Mix130 v54->v55->v56 promotion replay is now complete across all 82 v54-eligible v33 rows. v54 accepted 19 repairs, v55 reverted 1 over-compressed repair, and v56 accepted 1 fusion; v56 improves full-Mix130 F1 from 0.4646 to 0.4661 and routed judge overall from 4.2538 to 4.3308, but 4 changed rows regress under judge and line 47 remains poor, so this branch is still not promoted.
- The external v54->v55->v56 replay is now complete on Legal20, Biology50, and Domain50. It improves weighted external F1 by +0.0171 and routed judge overall by +0.0875, but Legal judge overall declines by -0.05 and all three slices include changed-row regressions. v55 reverts no repair and v56 triggers no fusion on these slices, so the current guard/fusion design does not yet bound this risk.
- Deterministic v31->v32 reproduction on mix130: generated v31/v32 artifacts match original JSON objects exactly, including route metadata, when replayed with `results/historical_vsr/v31_full_rf_oof.jsonl`.
- SAGE v33 freeze/holdout audit is now generated in `results/historical_sage/SAGE_V33_FREEZE_HOLDOUT_AUDIT.md`; it locks v33 for paper planning, marks v56 as exploratory, and separates reportable validation surfaces from no-longer-clean tuning surfaces.
- Locked unseen75 post-freeze holdout is now complete in `results/`: 75 samples across Legal, Biology, History, Physics, and Agriculture; no source-line or context-hash overlap with prior aligned/external/exploratory slices; Naive/HyDE/SAGE v33 outputs all have 75/75 valid rows and 0 error rows.
- Locked unseen75 judge evidence is strong under the same rubric: SAGE v33 judge overall 3.880 versus HyDE 3.280 and Naive 3.133; paired overall gain is +0.600 vs HyDE with 27 wins / 7 losses / 41 ties and p=1.27e-04, and +0.747 vs Naive with 33 wins / 8 losses / 34 ties and p=1.87e-06.
- Locked unseen75 pairwise A/B evidence is complete: SAGE v33 average target win rate is 58.40% versus HyDE and 63.47% versus Naive across comprehensiveness, diversity, logicality, relevance, and coherence.
- Locked holdout robustness evidence is now complete across Repeat1-8 in addition to unseen75. The consolidated meta-summary covers 326/326 rows with complete method, judge, and pairwise coverage; weighted lexical F1 is SAGE v33 0.3164 versus Naive 0.2972 and HyDE 0.2972.
- Complete locked-holdout judge results favor SAGE v33 semantically: weighted overall is 3.555 versus Naive 2.960 and HyDE 2.942, with average pairwise target win rates of 59.66% versus HyDE and 61.50% versus Naive.
- Repeat8 Legal54 exhausts the remaining strict legal holdout capacity. It is a stress slice rather than a headline win: SAGE v33 ties HyDE on lexical F1 at 0.2862, trails Naive by -0.0097 F1, and has judge overall 2.407 versus HyDE 2.352 and Naive 2.444.
- Include-ready LaTeX table fragments have been generated in `paper/generated_experiment_tables.tex` from `results/paper_tables/paper_ready_experiment_tables.json`.
- Reproducibility manifest with non-secret environment metadata, scripts, datasets, result artifact hashes, SAGE replay commands, and key rerun commands.
- Evidence-contract depth audit on frozen SAGE v33 versus `full_v2_rescue`: 130 aligned rows show that router probability is negatively correlated with SAGE-vs-VSR F1 delta (`-0.3782`), while route-selected rows have lower mean gains than route-skipped rows (`+0.1119` vs `+0.2089`). This is an observational mechanism result, not a causal claim: the current router detects difficulty, but does not yet estimate upgrade benefit.
- Retrospective impact-aware escalation prototype on the same Mix130 artifacts: at the matched 32/130 risk-router budget, cross-fitted top-k selection improves the historical gain proxy from `+0.0299` to `+0.0489` and reduces selected negative-delta rows from `7` to `1`. This is hypothesis-generating only because the historical SAGE-vs-VSR delta is not a randomized escalation effect.
- Deeper budget analysis isolates the mechanism: risk-probability ranking has AUUC proxy `0.0781`, below random ranking `0.0884`, while contract-gap ranking reaches `0.0842` and impact-aware ranking reaches `0.0975`. The main failure is therefore not lack of risk signal, but misallocation of a limited escalation budget toward rows with low realized benefit.

## Strongest Current Result

Source: `results/enhanced_ablation_aligned/FULL_V2_RESCUE_FORMAL_VALID_N130_SUMMARY.md`

| Method | N | F1 | ROUGE-L | Faithfulness | Rescue% | Abstain% | MetaLeak% |
|---|---:|---:|---:|---:|---:|---:|---:|
| `full_v2` existing | 130 | 0.2824 | 0.2107 | 0.9728 | 0.0% | 19.2% | 0.0% |
| `full_v2_rescue` | 130 | 0.2878 | 0.2194 | 0.9756 | 22.3% | 2.3% | 0.0% |

Key interpretation:

- Against same-run full-v2 baseline, `full_v2_rescue` improves F1 by +0.0292 with p=7.96e-05.
- On rescue-triggered rows only, it improves F1 by +0.1309 with 27 wins / 2 losses.
- The result is suitable as an internal strongest variant, but not yet enough by itself for a high-confidence SCI Q1/Q2 submission.

## Cross-Dataset Validation Added

Source: `results/enhanced_ablation_aligned/CROSS_DATASET_ALIGNED_PROGRESS_SUMMARY.md`

| Dataset Slice | N | `w/o_repair` F1 | `full_v2` F1 | `full_v2_rescue` F1 | Rescue vs `w/o_repair` | p-value | Cohen's d |
|---|---:|---:|---:|---:|---:|---:|---:|
| UltraDomain mix | 130 | 0.2206 | 0.2824 | 0.2878 | +0.0671 | 1.119e-11 | +0.60 |
| Biology full context | 50 | 0.1557 | 0.2121 | 0.2287 | +0.0729 | 3.845e-09 | +0.83 |
| Legal full context | 20 | 0.1791 | 0.2407 | 0.3496 | +0.1705 | 3.004e-04 | +0.81 |
| Domain expansion windowed | 50 | 0.1629 | 0.1986 | 0.2337 | +0.0708 | 1.946e-05 | +0.60 |

Interpretation:

- The rescue-enhanced pipeline now has positive paired F1 gains across four aligned slices.
- Biology and domain-expansion results support generalization beyond the original UltraDomain mix.
- Legal results are strongest but small-N; they should be presented as a promising domain slice, not the sole headline.

## New Failure and Stability Evidence

Sources: `results/enhanced_ablation_aligned/CROSS_DATASET_QUALITY_FLAG_LLM_AUDIT.md`, `results/enhanced_ablation_aligned/CROSS_DATASET_CONTEXT_RESCUE_DIAGNOSTIC_n7_ctx80000.md`, and `results/enhanced_ablation_stability/LEGAL_FULLCONTEXT_20_STABILITY_SUMMARY.md`

| Experiment | N | Key Result | Interpretation |
|---|---:|---|---|
| Cross-dataset LLM failure audit | 15 | evidence_gap=5, answer_hallucination=4, rescue_gate_overreach=4 | Main residual issues are evidence coverage and accepting weak rescue outputs. |
| 80k longer-context diagnostic | 7 | mean F1 delta vs current rescue = -0.0360, 2 wins / 5 losses | More context alone is not a safe fix. |
| Legal repeat/stability run | 20 | rescue vs `w/o_repair` remains +0.1330 F1, p=0.009737 | Positive effect persists, but run-to-run variance should be disclosed. |

## Aligned Retrieval Baseline Evidence

Sources: `results/historical_baselines/RETRIEVAL_BASELINE_ALIGNED_SUMMARY.md` and `results/historical_baselines/RETRIEVAL_PREFILTERED_VSR_SUMMARY.md`

| Dataset Slice | N | Naive F1 | HyDE F1 | Full-context `full_v2_rescue` F1 | Current Interpretation |
|---|---:|---:|---:|---:|---|
| Legal full context | 20 | 0.3632 | 0.3760 | 0.3496 | VSR is close but below both retrieval baselines. |
| Biology full context | 50 | 0.2365 | 0.2385 | 0.2287 | VSR is close but below both retrieval baselines. |
| Domain expansion windowed | 50 | 0.2711 | 0.2549 | 0.2337 | VSR trails both retrieval baselines, especially Naive. |

Key interpretation:

- The strongest safe claim is now: `full_v2_rescue` robustly improves over internal no-repair/full variants and reduces abstention, not that it currently dominates strong embedding retrieval baselines on F1.
- This materially changes the paper positioning: VSR should be framed as a verification, repair, and rescue layer whose value must be tested in combination with strong retrieval, or evaluated on faithfulness/abstention/error-control dimensions in addition to answer-overlap metrics.
- A pure F1-superiority claim against Naive/HyDE is not currently supported by the aligned evidence.

## Retrieval-Prefiltered VSR Diagnostic

| Run | N | `w/o_repair` F1 | `full_v2` F1 | `full_v2_rescue` F1 | Delta vs Naive | Delta vs HyDE | Interpretation |
|---|---:|---:|---:|---:|---:|---:|---|
| Legal retrieved-top5 default | 20 | 0.1767 | 0.2511 | 0.2924 | -0.0707 | -0.0836 | Rescue still helps internally, but top-k prefiltering does not close the retrieval-baseline gap. |
| Legal retrieved-top5 ctx12k | 20 | 0.2014 | 0.2755 | 0.2902 | -0.0730 | -0.0858 | Completed run remains below Naive/HyDE and does not improve over default retrieved-top5 VSR. |
| Legal retrieved-top5 prepended full | 20 | 0.1797 | 0.2462 | 0.2823 | -0.0809 | -0.0937 | Preserving full source text after retrieved chunks is also negative. |

Interpretation:

- Retrieved-top5 contexts did not improve Legal performance relative to full-context VSR.
- The ctx12k setting completed after API quota was restored.
- Because ctx12k remains weak at N=20, this branch should be treated as a negative diagnostic rather than expanded to Biology/windowed.
- The top5+full-prepend setting also remains weak, so simple retrieval reordering is not enough; future combined designs need a deeper integration than context concatenation.

## Cost and Latency Evidence

Sources: `results/cost_latency/COST_LATENCY_SUMMARY.md`, `results/cost_latency/COST_QUALITY_CASE_AUDIT.md`, and `results/cost_latency/I18N_ABSTENTION_LANGUAGE_GUARD_AUDIT.md`

| Dataset Slice | N | Naive s/q | HyDE s/q | VSR Rescue s/q | VSR / Naive | VSR / HyDE | VSR F1 - Best Retrieval F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Legal full context | 20 | 6.0 | 14.4 | 59.6 | 9.89x | 4.14x | -0.0264 |
| Biology full context | 50 | 6.8 | 20.7 | 46.9 | 6.88x | 2.26x | -0.0098 |
| Domain expansion windowed | 50 | 5.6 | 14.7 | 43.3 | 7.66x | 2.94x | -0.0374 |

Interpretation:

- Current result files support a runtime-overhead analysis, but not exact API cost accounting.
- Full VSR remains materially slower than Naive/HyDE baselines on aligned slices, while also trailing the best retrieval F1 in the current evidence.
- The aligned runner now logs per-stage LLM calls and provider usage metadata; the completed ctx12k continuation captured call accounting for 12 newly rerun rows, while older result files still need selected reruns or backfill before final paper-table generation.
- Cross-domain full-context N=20 cost samples estimate `full_v2_rescue` at 53.9 seconds/question, 5.5 LLM calls/question, and 12.6k tokens/question on average; rescue evidence extraction remains the dominant high-token stage when triggered.
- The same N=20 sample also flags a safety/cost tradeoff: Legal is rescue-heavy (70% rescue rate), Biology is repair-heavy (100% repair, 0% rescue), and Domain-windowed shows a repair-path F1 regression despite perfect faithfulness in this small sample.
- The cost-quality case audit identifies the clearest Domain regression as line 1: repair changes a partially correct English abstention into a Chinese abstention, reducing F1 from 0.1712 to 0.0000 while the current abstention flag fails to trigger rescue.
- The exploratory i18n guard audit suggests a narrower next-step guard: detect missed abstention language for analysis, but automatically fallback only on answer-language mismatch. In the N=20 simulation this fixes the Domain line-1 regression without reverting missed-abstention-only Legal rows; across 250 full aligned VSR rows, the same scan finds no language-mismatch cases.
- The Legal top5+full-prepend run captured token/call accounting for all 20 rows: `full_v2_rescue` used about 4.0 LLM calls/question with 48.2 seconds/question, but still reached only F1=0.2823.

## Reproducibility Package

Sources: `results/reproducibility/REPRODUCIBILITY_MANIFEST.md`, `results/historical_sage/SAGE_REPRODUCTION_VALIDATION.md`, `results/historical_sage/SAGE_VS_BASELINES_SUMMARY.md`, and `results/historical_sage/SAGE_LOCKED_FULLSLICE_LADDER_SUMMARY.md`

The current reproducibility manifest records:

- Python/platform details, Git branch/commit, and non-secret model configuration.
- Hashes for 45 key experiment scripts, 10 dataset files, and 414 result/report artifacts.
- Rerun commands for aligned VSR, retrieval baselines, retrieved-top5 dataset construction, the ctx12k resume candidate, SAGE v31/v32/v33 strict replay, locked full-slice ladder summarization, SAGE reproduction validation, and cost/latency summaries.
- A warning that the repository working tree contains many dirty/untracked experiment files, so file hashes are the stronger reproducibility anchor than Git commit alone.
- SAGE v31/v32/v33 reproduction now reports 0 object diffs, 0 prediction diffs, and 0 route-metadata diffs across all three 130-row mix130 artifacts.

## Legacy Baseline Comparability

Source: `results/legacy_baseline_overlap/LEGACY_BASELINE_OVERLAP_SUMMARY.md`

| Dataset Slice | Best Legacy/Existing Method by F1 | Best F1 | Legacy PathRAG F1 | Aligned `full_v2_rescue` F1 | Interpretation |
|---|---|---:|---:|---:|---|
| UltraDomain mix | Direct-context baseline | 0.4905 | 0.1815 | 0.2878 | GraphRAG-lite is now available at F1=0.3334 and judge overall=2.646; PathRAG wins 89.38% pairwise, but direct-context/SAGE remain stronger on lexical and judge quality. |
| Legal full context | SAGE v18 cost-aware pre-router | 0.4496 | 0.1308 | 0.3496 | Existing routed SAGE-style output is stronger on lexical F1 than VSR rescue. |
| Biology full context | Legacy HyDE | 0.3268 | 0.1069 | 0.2287 | Legacy PathRAG is heavily penalized by verbosity under lexical F1. |
| Domain expansion windowed | Legacy Naive | 0.2948 | 0.1263 | 0.2337 | Graph/PathRAG-style verbosity and abstention remain a risk under overlap metrics. |

Interpretation:

- Original-style LLM-as-judge results and current lexical overlap results should not be mixed as equivalent metrics.
- Legacy PathRAG may look strong under correctness/coverage judging but weak under F1 because answers are long and include extra details.
- The strongest existing lead for a paper-grade method may be adaptive routing / SAGE-style text-first gating rather than simple VSR-on-retrieved-context concatenation.

## SAGE Candidate Scan

Source: `results/historical_sage/SAGE_CANDIDATE_SCAN_SUMMARY.md`
Companion comparison: `results/historical_sage/SAGE_VS_BASELINES_SUMMARY.md`
External probe: `results/historical_sage/SAGE_EXTERNAL_PROBE_SUMMARY.md`
v33 preservation probe: `results/historical_sage/SAGE_V33_EXTERNAL_PROBE_SUMMARY.md`
External vs baselines: `results/historical_sage/SAGE_EXTERNAL_VS_BASELINES_SUMMARY.md`
Locked full-slice ladder: `results/historical_sage/SAGE_LOCKED_FULLSLICE_LADDER_SUMMARY.md`
Unseen validation: `results/historical_sage/SAGE_UNSEEN_VALIDATION_SUMMARY.md`
Late-stage candidate ladder: `results/historical_sage/SAGE_LATE_STAGE_CANDIDATES_SUMMARY.md`
Mix130 promotion replay: `results/historical_sage/SAGE_PROMOTION_MIX130_SUMMARY.md`
Final candidate summary: `results/historical_sage/SAGE_FINAL_CANDIDATE_SUMMARY.md`
Locked unseen75 final holdout: `results/historical_sage/SAGE_LOCKED_UNSEEN75_JUDGE_SUMMARY.md`
Locked unseen75 pairwise win-rate: `results/historical_sage/SAGE_LOCKED_UNSEEN75_PAIRWISE_SUMMARY.md`
Locked holdout meta summary: `results/historical_sage/SAGE_LOCKED_UNSEEN_META_SUMMARY.md`
Locked Repeat8 Legal54 summary: `results/historical_sage/SAGE_LOCKED_LEGAL54_REPEAT8_SUMMARY.md`
Paper-ready LaTeX fragments: `paper/generated_experiment_tables.tex`

| Candidate | N | F1 | ROUGE-L | Runtime s/q | Interpretation |
|---|---:|---:|---:|---:|---|
| SAGE v33 preservation-delta mix130 | 130 | 0.4646 | 0.3824 | 13.8 | Same mix130 score as v32, plus deterministic high-content preservation guard for external safety. |
| SAGE v33 preservation-delta unseen34 | 34 | 0.4906 | 0.3984 | 16.0 | Context-heldout guard validation; prediction-identical to old frozen v33, with only one final answer changed vs baseline. |
| SAGE v33 preservation-delta v46 zero-overlap | 30 | 0.3249 | 0.2524 | 48.8 | Zero-overlap validation extension; two retained repairs improve HyDE baseline but still small-N. |
| SAGE v55 supported-facet v46 zero-overlap | 30 | 0.3627 | 0.2802 | 48.8 | Stronger exploratory v46 branch; same predictions as v54, needs Mix130/full-slice frozen replay before promotion. |
| SAGE v56 supported-facet fusion v40/v42/v46 | 54 | mixed | mixed | mixed | Best late-stage small heldout branch: v40 judge overall 4.250, v42 4.250, v46 4.300; needs larger locked replay. |
| SAGE v56 supported-facet fusion Mix130 candidate82 | 130 | 0.4661 | 0.3812 | mixed | Complete 82-candidate Mix130 promotion replay; routed judge overall 4.3308 vs v33 4.2538, but 4 regressions and one persistent poor row prevent promotion. |
| SAGE v31 learned contract-gap mix130 | 130 | 0.4646 | 0.3824 | 13.8 | Strongest large-N existing candidate. |
| SAGE v32 evidence-delta mix130 | 130 | 0.4646 | 0.3824 | 13.8 | Deterministic guard preserves v31 score while adding safety logic. |
| SAGE v17 indemnity compact legal20 | 20 | 0.4696 | 0.4266 | 62.1 | Strong legal candidate, but domain-specific and slower. |
| Current aligned `full_v2_rescue` mix130 | 130 | 0.2878 | 0.2194 | 41.7 | Robust internal ablation result, but weaker as main F1 method. |

Interpretation:

- The most promising SCI main method is no longer simple `full_v2_rescue`; it is likely an adaptive-routing/SAGE-style method that chooses cheap text retrieval most of the time and escalates selectively.
- v31/v32/v33 strict replay confirms the mix130 artifacts can be regenerated deterministically from existing artifacts with full JSON-object equality, including route metadata.
- A direct mix130 paired lexical check shows SAGE v33-preservation beats aligned VSR `full_v2_rescue` by +0.1768 mean F1 across 130 rows, with wins on every task family; this is strong candidate evidence, and the locked ladder now records the full-slice comparison surface.
- The expanded external Domain-windowed API probe is directionally positive: 40 verifications produced 20 pre-v32 accepted repairs, v32 retained 19 and reverted 1, selected-line lexical F1 moved from 0.2413 to 0.2827, full 50-row F1 moved from 0.2549 to 0.2881, selected-line judge overall improved from 2.92 to 4.40, and correctness improved from 3.08 to 4.80.
- The Legal full-context API probe adds a second positive slice: 9 verifications produced 2 accepted repairs, both retained by v32, full 20-row F1 moved from 0.3760 to 0.3925, selected-line judge overall moved from 4.11 to 4.67, and correctness moved from 4.44 to 5.00.
- The Biology full-context API probe is a negative/neutral diagnostic under v32: 32 verifications produced only 1 accepted repair, retained by v32, but full 50-row F1 moved from 0.2385 to 0.2382 and selected-line judge overall moved from 4.19 to 4.12.
- The Biology regression is concentrated in line 46, where deterministic evidence-support metrics retained a repair that reduced reference-targeted coverage and moved judge overall from 5 to 3; the new v33-preservation guard reverts this row, preserves Domain/Legal v32 gains unchanged, and returns Biology to baseline F1/judge rather than producing a positive Biology gain.
- In the external-vs-baselines table, SAGE v33-preservation numerically beats the best retrieval baseline on Legal (+0.0164 F1 vs HyDE) and Domain (+0.0170 F1 vs Naive), ties HyDE on Biology (+0.0000), and exceeds VSR `full_v2_rescue` on all three external slices; however, the best-retrieval deltas are not statistically significant in this small snapshot.
- The locked full-slice ladder confirms complete scoring coverage: Mix130 N=130, Legal N=20, Biology N=50, and Domain N=50. v33-v32 prediction diffs are 0/0/1/0 respectively, with the single Biology diff reverting the harmful line-46 repair.
- In the weighted external view, v31 is numerically highest (F1=0.2851) and v33 is nearly tied (F1=0.2848). The reason to freeze v33 is safety and reproducibility, not a claim that it maximizes every lexical aggregate.
- Full-slice semantic judge scoring now covers Naive, HyDE, VSR rescue, and SAGE v33 under the same rubric. Weighted external overall is v33=4.21 versus best retrieval=3.68 and VSR rescue=3.17; weighted correctness is v33=4.62 versus best retrieval=3.92 and VSR rescue=3.12. This strengthens the semantic-quality argument, while remaining an LLM-as-judge result rather than human annotation.
- Paired judge overall tests show the strongest semantic evidence on Domain (+1.20 vs best retrieval, p=2.88e-07; +1.08 vs VSR rescue, p=1.421e-05). Biology is neutral vs HyDE on judge overall (+0.03, p=0.7798) but strongly above VSR rescue (+1.17, p=4.025e-06). Legal is directionally above HyDE (+0.10, p=0.7116) and near-significant above VSR rescue (+0.65, p=0.0633), so it should be described cautiously.
- The unseen34 validation adds a clean guard-specific safety check: current v33 retains the beneficial line-61 repair, rejects the harmful line-71 rewrite, changes only line 61 versus baseline, and has zero prediction diffs versus old frozen v33 on all Mix130 rows. Because only two guard-active held-out rows exist, this is supportive safety evidence rather than statistically stable repair-accuracy evidence.
- The v46 zero-overlap validation30 extension is directionally stronger than unseen34 in absolute delta (+0.0123 F1 and +0.267 judge overall versus HyDE), but it still comes from only two retained repairs and should be grouped as supplementary unseen validation.
- The v55 supported-facet exploratory branch is the strongest v46 zero-overlap point so far (+0.0501 F1 and +0.700 judge overall versus HyDE; +0.0378 F1 and +0.433 judge overall versus v33), but it remains a late-stage candidate branch rather than the frozen main method.
- The late-stage ladder shows v56's specific value on v40: v55 blocks an over-compressed line-11 repair and v56 recovers a stronger fused answer, lifting judge overall from 4.167 to 4.250. On v42 and v46, v56 is prediction-identical to v55 because no v55 revert is triggered.
- The complete Mix130 candidate82 promotion replay gives a positive but not decisive signal for v56: full-Mix130 F1 rises by +0.0015 and routed judge overall by +0.0769, with changed-row judge gains concentrated on lines 9, 19, 30, 42, 47, 66, 78, 94, and 98. The same replay exposes unresolved risk: lines 22, 29, 55, and 129 regress by one judge-overall point, and line 47 remains poor even after improving from v33.
- The complete external promotion replay is directionally positive in aggregate (weighted F1 +0.0171; judge overall +0.0875), with Biology and Domain positive but Legal judge overall -0.05. Because every replay slice still has changed-row judge regressions and no v55/v56 safety intervention fires externally, v56 remains exploratory rather than replacing v33.
- The final candidate summary frames SAGE v33-preservation as the lead method candidate, but still marks Legal/Domain best-retrieval gains as small/nonsignificant and Biology as neutral.
- The locked unseen75 final holdout strengthens the post-freeze generalization story: SAGE v33 improves lexical F1 by +0.0133 versus HyDE and +0.0193 versus Naive, while the larger judge gains are statistically significant on identical 75-question pairings.
- Pairwise A/B judging adds a paper-style preference metric on the same holdout: SAGE v33 averages 58.40% target win rate versus HyDE and 63.47% versus Naive, with strongest Naive gains on diversity and coherence.
- The probe also shows metric tension: line 31 loses lexical F1 but improves by judge score, so final reporting should retain both lexical overlap and semantic judge metrics.
- The VSR rescue results remain useful as a verification/repair ablation layer, not necessarily the headline F1 method.

## Experiment Type Clarification

- Strict ablation experiments: `w/o_*` and `full`/`full_v2` comparisons.
- Enhancement experiment: `full_v2_rescue`, because it adds a new abstention-triggered rescue branch.
- Diagnostic experiments: quality audits, forced 40k rescue, and secondary-gate candidate analysis.

## Remaining Experiment Work

### Must Finish Before Paper Writing Locks Results

| Priority | Item | Why It Matters | Current Status | Estimated Effort |
|---:|---|---|---|---|
| P0 | Freeze final method definition | Avoid changing method while writing | Complete for paper planning: SAGE v33-preservation is frozen in `results/historical_sage/SAGE_V33_FREEZE_HOLDOUT_AUDIT.md`; any new v56 guard must be evaluated on a new locked holdout before promotion | Done unless method changes |
| P0 | Reproducible same-protocol baselines | Reviewers will reject unfair / mixed baselines | Naive/HyDE aligned baselines are complete for external slices; GraphRAG-lite covers Mix130 plus Legal/Biology/Domain; LightRAG current-130 is fully evaluated as the representative heavy baseline; full official GraphRAG/LightRAG comparability remains approximate | Offline framing done unless attempting official reruns |
| P0 | Multi-dataset validation | SCI Q1/Q2 needs more than one robust benchmark | Complete under current strict policy: locked unseen75 plus Repeat1-8 cover 326/326 method/judge/pairwise rows with zero locked-holdout overlap | Done unless adding new source data or a new frozen method |
| P0 | Final table generation | Main result, ablation, rescue-only, quality audit tables | Consolidated paper-ready tables generated in `results/paper_tables/` and exported to `paper/generated_experiment_tables.tex` | Done; refresh only after artifact changes |
| P1 | Cost / latency analysis | Rescue adds API and context cost; must quantify | Initial runtime report complete; runner instrumentation added; cross-domain N=20 token/call sample complete | 0.5-1 day |
| P1 | Error analysis case studies | Needed to explain evidence-gap and abstention behavior | Cross-dataset LLM audit completed for 15 high-priority rows | 0.5 day |
| P1 | Robustness / repeat runs | LLM output variance needs stability evidence | Legal N=20 repeat completed; more repeats optional | 1-3 days |
| P1 | Original PathRAG comparability check | Need avoid overclaiming against original paper | Offline lexical re-scoring complete; still need final metric-family decision | 0.5-1.5 days |
| P1 | Reproducibility package | Needed for reviewer confidence and appendix detail | Manifest refreshed after Repeat8 judge/pairwise completion and LaTeX table generation | Done; refresh after future reruns or edits |
| P1 | Mechanism-depth audit | Needed to turn evidence-contract framing into a testable adaptive-routing contribution | Offline audit and matched-budget prototype complete; next experiment is a frozen impact-aware escalation score evaluated on new source data or a newly locked holdout | 1-3 days plus online evaluation |

## Paper-Grade Readiness Estimate

| Target Level | Experiment Readiness | Meaning |
|---|---:|---|
| Internal technical report | 90-95% | Experiment package is complete under the current strict policy; remaining work is writing/framing |
| SCI Q2 pragmatic submission | 75-80% | Locked-holdout semantic and pairwise evidence is strong, but lexical F1 gains remain modest and official graph-baseline comparability still needs cautious framing |
| SCI Q1 stronger submission | 60-65% | Needs stronger official baselines, broader standard-benchmark evidence, and tighter efficiency/human-evaluation support |

## Recommended Minimum Final Experiment Package

For an SCI Q1/Q2 submission, the minimum defensible package should include:

1. Main results on at least three datasets or dataset slices: UltraDomain, HotpotQA / 2WikiMultiHopQA, and one domain-heavy set such as Biology or MuSiQue.
2. Same-model, same-context, same-metric comparison against Naive RAG, HyDE, PathRAG/base graph retrieval, GraphRAG-lite/community baseline, representative LightRAG current-130 fully evaluated evidence, GraphRAG or full LightRAG if feasible, `full_v1`, `full_v2`, and `full_v2_rescue`.
3. Ablation table: `w/o_evidence_extraction`, `w/o_verification`, `w/o_repair`, `w/o_rescue`, `full_v2`, `full_v2_rescue`.
4. Rescue-specific table: trigger rate, win/loss, abstention reduction, faithfulness change, cost overhead.
5. Quality / failure analysis: evidence gap, overconservative abstention, hallucination, verifier false negative, undercoverage.
6. Statistical testing: paired t-test, Wilcoxon, bootstrap confidence intervals, and effect sizes.
7. Reproducibility manifest: exact scripts, configs, model name, API base, dataset hashes, and output paths.

## Suggested Next Actions

1. Keep `full_v2_rescue` as the current internal ablation baseline, but explicitly separate its rescue gains from external retrieval-baseline claims.
2. Stop expanding the simple Legal retrieved-top5 branches unless a new retrieval/VSR integration idea materially changes the method.
3. Treat SAGE v33-preservation as the paper-planning freeze; route metadata reproducibility is now resolved under the frozen `v31_full_rf_oof` replay protocol.
4. Keep v56 exploratory. The next method experiment should be an impact-aware escalation score: train/calibrate it only from historical development outcomes, freeze it before testing, and compare risk-only routing, impact-aware routing, always-escalate, and never-escalate on new source data or a newly locked holdout. Report quality, cost, upgrade gain, budget curves, and wrong-escalation rate together.
5. Keep `results/reproducibility/REPRODUCIBILITY_MANIFEST.md` refreshed after each final rerun so hashes and commands match the submitted results.

## Bottom Line

The current results are complete enough to write the experimental section and an internal technical report. Strict Mix130 ablation is complete, paper-table consolidation and LaTeX export are done, GraphRAG-lite covers Mix130 plus Legal/Biology/Domain, LightRAG current-130 is fully evaluated as the representative high-cost graph baseline, and SAGE v33 is frozen for paper planning. Locked unseen75 plus Repeat1-8 provide 326/326 complete method, judge, and pairwise rows with a PASS overlap audit. The defensible claim is strong semantic and pairwise generalization for SAGE v33 under strict locked holdouts, while avoiding broad lexical-F1 superiority claims because gains over Naive/HyDE are modest and official graph-baseline comparability remains approximate. The new mechanism audit sharpens the scientific gap: contract deficit is useful for risk detection, but not yet for predicting escalation benefit. Under the current strict source-line/source-context exclusion policy, no old ablation/comparison experiments remain pending; the only worthwhile new run is a frozen impact-aware escalation test on new source data or a newly locked holdout.
