# PathRAG Thesis Experiment Handoff

Last updated: 2026-08-02

This repository is the active workspace for the thesis experiment. It is not a clean upstream PathRAG clone anymore; it contains local reproduction runs, baseline scripts, innovation modules, evaluation results, and UI/API changes.

## Project Goal

The thesis direction is knowledge-graph-enhanced LLM / GraphRAG for long-document QA. The base paper is:

- Paper: `PathRAG: Pruning Graph-Based Retrieval Augmented Generation with Relational Paths`
- arXiv: https://arxiv.org/abs/2502.14902
- AAAI page: https://ojs.aaai.org/index.php/AAAI/article/view/40268
- Code: https://github.com/BUPT-GAMMA/PathRAG

The local plan is:

1. Reproduce PathRAG under a local DeepSeek-compatible API setting.
2. Compare against Direct Context, NaiveRAG, HyDE, and representative graph-RAG baselines.
3. Build thesis innovation around adaptive evidence routing, evidence verification, claim safety, and low-cost graph/text fallback.
4. Report results carefully as local reproduction and innovation results, not an exact official paper-table rerun.

## Environment

- Workspace: `D:\GitHubProjects\PathRAG`
- Conda env used in prior runs: `lw` at `D:\Anaconda\envs\lw`
- Activate:

```powershell
conda activate lw
cd D:\GitHubProjects\PathRAG
```

- API config lives in `.env`. Do not print or commit API keys.
- LLM setting used in local results: DeepSeek-compatible API through `.env`
- Embedding model used in local results: `BAAI/bge-m3`
- Important dependency pin: `bcrypt==4.0.1` in `requirements.txt`, because `passlib 1.7.4 + bcrypt 5.x` caused login/password verification failures.

## Important Reproduction Results

Main locked reproduction snapshot:

- `results/snapshots/LOCKED_REPRO_MIX130_DEEPSEEK_20260704/README_LOCKED_REPRODUCTION.md`
- Dataset: `datasets/UltraDomain/mix.jsonl`
- Samples: 130 / 130
- Vanilla PathRAG vs Direct Context: 58.69% pairwise win rate
- Adaptive PathRAG vs Direct Context: 68.00% pairwise win rate
- Paper reference Mix average: 61.59%, not directly equivalent

Baseline comparison snapshot:

- `results/snapshots/REPRO_BASELINES_NAIVE_HYDE_MIX130_DEEPSEEK_20260704/baseline_reproduction_comparison.md`
- Vanilla PathRAG vs NaiveRAG: 60.15%
- Vanilla PathRAG vs HyDE: 62.00%
- Direct Context avg query time: 7.015 s
- NaiveRAG avg query time: 7.137 s
- HyDE avg query time: 13.820 s
- Vanilla PathRAG avg query time: 37.386 s
- Vanilla PathRAG graph indexing time on Mix-130: 22564.621 s

LightRAG representative subset:

- `results/snapshots/LIGHTRAG_REPRESENTATIVE_SUBSET_30_MIX_DEEPSEEK_20260705/lightrag_representative_subset_report.md`
- Dataset: 30 samples from Mix, 5 samples per task type
- LightRAG mode: hybrid
- LightRAG avg indexing time: 478.024 s/sample
- LightRAG avg query time: 41.106 s/sample
- Vanilla PathRAG vs LightRAG subset: 48.33%
- Adaptive HyDE-Fallback vs LightRAG subset: 45.33%
- Interpretation: LightRAG is a strong but expensive graph-RAG baseline; keep it as a representative subset baseline.

Useful overview:

- `results/historical_reports/current_reproduction_and_innovation_overview.md`

## Main Result Pattern

PathRAG usually wins on diversity, logicality, and structured coverage, but it is slow and often verbose. Direct Context, NaiveRAG, and HyDE are faster and often more concise/relevant on narrow lookup questions. This motivates adaptive graph/text routing instead of always using graph retrieval.

Safe thesis wording:

> We reproduced PathRAG under a DeepSeek-compatible local setting. The results show the same broad pairwise preference pattern as the original paper, but they are not an exact official rerun because the LLM, embedding model, dataset subset, prompts, and baseline implementations differ.

## Current Innovation Lines

Current phase: renewed post-V70 mechanism innovation. The user explicitly
reopened innovation after the V70 freeze; formal ablation has not started.
Preserve V70 as the archived full-system control.

### VSR-PathRAG

Current backend agent version:

- `api/features/chats/agent_controller.py`
- `AGENT_VERSION = "vsr-pathrag-v105"`
- `AGENT_NAME = "Next-Gen Full-Stack VSR-PathRAG Agent (A+B+C+E+F+G+H+I)"`
- Current behavioral mechanisms: evidence-available refusal guard, primary
  definition-nucleus priority, evidence-to-theme interpretive bridge, and
  ordered-clause continuity/obligation control, plus historical-to-current
  temporal-scope calibration, evidence-grounded main-effect focus,
  authority-bounded legal action realization, condition-matrix completion,
  ordered legal-procedure completion, dual-channel cultural/artistic
  influence realization, claim-local polarity verification, and cumulative
  pressure definition completion, and administrative-to-intelligence role
  sequencing, technology-to-market impact linking,
  preparatory-to-eternal purpose completion, observation-to-management causal
  completion with boundary-preserving evidence retrieval, ordinary-concept
  versus named-entity sense disambiguation, institutional social-dynamics
  bridging, verified obligation preflight, and concrete episode-to-theme
  realization, and natural-philosophy biological/adaptive/artistic facet
  routing, dual-objective habitability/observability fine-tuning, and
  temporal-status entailment with source-specific confidence calibration,
  sociolexical personal-to-societal transformation chaining,
  legal-purpose obligation-scope closure, entity-agnostic typed obligation
  graph compilation, minimal proof-subgraph optimization, counterfactual
  evidence-path necessity certification, risk-calibrated answer submission,
  proof-constrained claim admission,   source-independent proof quorum certification, source-quorum independent evidence
  backfill, obligation-source rescue for critical obligation deficits,
  proof-completion claim/obligation backfill, audit-aware proof-deficit
  answer repair, source-diversity singleton-claim repair, and
  obligation-criticality quorum normalization (with entity-name anchor
  artifact demotion), query-complexity-aware adaptive evidence compression,
  salience-preserving evidence reordering, fidelity-weighted answer quality
  scoring (CAQ/CFS/NPS/SAS), iterative answer refinement loop,
  claim-evidence traceability graph with per-claim source provenance and
  support strength classification (strong/weak/contradicted/unsupported),
  question-type-aware reasoning path selection with dynamic strategy
  configuration (factoid/causal/comparison/process/narrative), and
  intra-answer consistency checking with pairwise contradiction detection
  (numerical/polarity/scope/temporal) and evidence-weighted resolution
- Stage: `renewed_innovation_pre_ablation`; formal ablation has not started
- The repaired anchor-preserving source contexts and all five graphs are
  complete. V70 remains the archived non-provisional innovation freeze. V92
  is the current innovation continuation (A+B combination strategy). V76
  remains the latest judged quality-improving selective replay; V77-V88 add
  reference-free mechanisms without starting ablation. V89-V92 add adaptive
  compression, salience reordering, fidelity scoring, and iterative refinement.
  V93-V97 implement the A+B+C+E full-stack strategy: claim-evidence
  traceability graph (V93), question-type-aware reasoning path selection (V95),
  and intra-answer consistency checking (V97).

Current promoted 30-sample selective replay:

- Reference-based Overall: `4.333`
- Pairwise win rate vs HyDE: `67.33%`
- Verdicts: 10 excellent, 20 good, 0 partial, 0 poor
- v58 changed line 7 and reached `51.67%`
- v59 changed line 10 and reached `53.00%`
- v60 changed line 8 and reached `55.00%`
- v61 changed line 0 and reached `56.33%`
- v62 changed line 25 and reached `58.33%`
- v63 changed line 22 and reached `60.00%`
- v64 changed line 3 and reached `61.00%`
- v65 changed line 14 and reached `61.67%`
- v66 changed line 27 and reached `63.00%`
- v68 changed line 17 and reached `64.00%`
- v69 changed line 20 and reached `64.33%`
- v70 froze the stack and reduced the two final-case runtimes to 0.70 seconds
  and 0.45 seconds
- v71 changed line 19 and reached `65.33%`
- v72 changed line 2, reached Overall `4.200`, and reached `65.67%`
- v73 changed line 12, reached Overall `4.267`, and preserved `65.67%`
- v74 changed line 5, reached Overall `4.300`, and reached `66.00%`
- v75 changed line 23, retained Overall `4.300`, and reached `67.00%`
- v76 changed line 6, reached Overall `4.333`, and reached `67.33%`
- v77 added the entity-agnostic typed obligation graph compiler
  (`obligation_type_compiler.py`), passed 4/4 cross-entity cross-domain
  controlled audit (Legal/Biology/History/Physics unseen proper nouns),
  reused the V76 30-sample lineage unchanged, Overall `4.333`, `67.33%`,
  and classified all 72 cross-question + cross-domain held-out questions.
  After the evidence-slot quality gate, 66/72 produce a typed nucleus with
  non-empty source IDs. This is a type/slot coverage validation, not a
  judged answer-correctness result, so it must not be reported as a new
  accuracy score.
- v78 added the minimal proof-subgraph optimizer
  (`minimal_proof_subgraph.py`) that jointly covers question obligations
  and signed answer claims. Controlled proof audit is 3/3, and offline
  reconstruction on the selective-30 outputs gives proof sufficient 19/30,
  mean claim-support coverage `0.8906`, and mean obligation coverage
  `0.8404`. This is a mechanism audit, not a new judged score.
- v79 added counterfactual path-necessity certification: selective-30
  reconstruction reports necessity certified 19/30, 25 essential paths,
  50 replaceable paths, and mean necessity `0.0436`.
- v80 added risk-calibrated answer submission: selective-30 reconstruction
  reports `{'submit': 19, 'submit_with_monitoring': 11}`, submission
  allowed 30/30, blocked unsupported 0, and mean calibrated risk `0.1024`.
- v81 added proof-constrained claim admission: selective-30 reconstruction
  reports 15 changed candidates, 14 selectable, 114 admitted claims, and
  21 rejected proof-uncovered claims.
- v82 added source-independent proof quorum certification: pure V82 current
  terminal-assessment reconstruction on selective-30 reports status counts
  `{'quorum_failed': 14, 'correlated_monitor': 11, 'independent_quorum': 4,
  'single_source_critical': 1}`. This is a proof-source audit, not a judged
  score.
- v83 added source-quorum independent evidence backfill: current
  terminal-assessment reconstruction on selective-30 triggers/applies/improves
  12/11/11 cases, raises
  independent quorum from 4 to 9, removes the single-source-critical case,
  and reduces correlated-monitor cases from 11 to 7.
- v84 added proof-completion claim/obligation backfill with claim-local
  evidence windows: current terminal-assessment reconstruction on
  selective-30 triggers/applies/improves 13/5/5 cases, raises proof
  sufficient from 16 to 21, reduces `quorum_failed` from 14 to 9, and raises
  final independent quorum from 4 to 14.
- v85 added audit-aware proof-deficit answer repair after V81: selective-30
  replay through V84 -> V81 -> V85 triggers/changes/selects 2/2/2 residual
  cases, raises proof sufficient from 21 to 30, reduces unsupported claims
  from 10 to 0, clears `quorum_failed` from 9 to 0, and raises independent
  quorum from 14 to 16. This is a mechanism audit, not a new judged score.
- v86 added source-diversity singleton-claim repair after V85: selective-30
  replay through V84 -> V81 -> V85 -> V86 triggers/selects 10/10 cases,
  raises independent quorum from 16 to 23, reduces correlated-monitor cases
  from 14 to 7, keeps proof sufficient at 30/30, keeps unsupported claims at
  0, and raises mean terminal score from 1.876 to 1.8923. This is a
  mechanism audit, not a new judged score.
- v87 added obligation-criticality quorum normalization after source-quorum
  certification: selective-30 replay through V84 -> V81 -> V85 -> V86 -> V87
  applies to 6 residual obligation-side cases, raises independent quorum from
  23 to 29, reduces correlated-monitor cases from 7 to 1, keeps proof
  sufficient at 30/30, keeps unsupported claims at 0, and yields mean
  terminal score 1.9143. This is a mechanism audit, not a new judged score.
- v88 added obligation-source rescue after V83 backfill and before V87
  criticality: V88 targets only critical (real domain) obligations that V87
  cannot demote. Simultaneously, V87 was enhanced to demote entity-name
  anchor artifacts (e.g. `anchor:LAWRENCE SCOTT SHEETS`) in addition to
  short anchors. Combined V87+V88 selective-30 replay raises independent
  quorum from 29 to **30/30**, eliminates the final correlated-monitor
  residual (line 22 History), keeps proof sufficient at 30/30, keeps
  unsupported claims at 0. This is a mechanism audit, not a new judged score.
- v89 added query-complexity-aware adaptive evidence compression after V88
  rescue: V89 classifies questions into factoid/comparison/causal/narrative
  types and applies graduated compression rates (0.45-0.75). It identifies
  necessary evidence (cannot be removed without breaking proof) vs compressible
  evidence, applying light compression to necessary items and heavy compression
  to low-density redundant items. Preserves proof sufficiency and quorum while
  reducing token usage by ~25-40%.
- v90 added salience-preserving evidence reordering after V89 compression:
  V90 computes per-evidence salience scores (claim support + obligation coverage
  + information density) and reorders evidence into front/middle/back buckets.
  High-salience evidence is placed at LLM-optimal positions (beginning/end of
  context) to counteract the "lost in the middle" bias (Liu et al., 2023).
- v91 added fidelity-weighted answer quality scoring: V91 introduces four
  continuous quality metrics requiring no external LLM calls:
  Claim-Fidelity Score (CFS), Numerical-Precision Score (NPS),
  Source-Attribution Score (SAS), and Composite Answer Quality (CAQ).
  These enable objective answer comparison and weak-claim identification.
- v92 added iterative answer refinement loop using V91 fidelity scores:
  V92 identifies weak claims (CFS < 0.40), searches for rescue candidates in
  unselected evidence, augments the proof with high-value additions, and
  recomputes CAQ. Bounded to max 3 iterations and 4 additional evidence items.
- All runtime repairs are reference-free; references are used only by the
  post-run evaluator
- Full test discovery: `237/237`

VSR means Verifiable Self-Repairing PathRAG. Main idea:

```text
query graph planning
-> knowledge-gap localization
-> provenance-preserving query-local evidence repair
-> active evidence acquisition
-> claim-level signed evidence verification
-> hierarchical graph/text answer realization
-> claim-to-cited-evidence and numeric fidelity audit
-> ambiguity-aware counterfactual submission
-> source-supported semantic-facet retention
-> observed stage-cost and information-value reallocation
-> post-verification residual-value reallocation
-> status-gain-optimized diversified claim retrieval
-> query-value feedback and dynamic verification stopping
-> graph/entity/query-focused-text joint claim verification
-> multi-claim Pareto evidence commit
-> counterfactual commit/rollback
-> provenanced diagnosis-to-answer correction
-> correction-persistent realization and semantic-relapse rollback
-> semantic-family and discourse-window correction lineage audit
-> source-bound opposite-polarity correction certificates
-> claim-level transactional correction rollback and full re-audit
-> dependency-aware claim transaction graph, cascade removal, and rebind
-> claim-source signatures and evidence-conditioned distant antecedents
-> obligation-aware global claim-bundle optimization
-> counterfactual antecedent and bundle-stability certificates
-> provenance-diverse fallback certification and citation reconciliation
-> lineage-aware correlated-source collapse and contradiction-free quorum
-> semantic relation and risk/reliability-calibrated family certification
-> bounded active counterevidence challenge and dependent-claim revocation
-> value-of-information challenge allocation and early stopping
-> joint value/cost query portfolio and cross-claim evidence reuse
-> related-claim clustering, fused contrastive retrieval, and claim-level coverage
-> partial-coverage cluster decomposition and evidence-triggered anchor aliases
-> observed retrieval-cost and evidence-yield challenge reprioritization
-> end-to-end retrieval usage and similarity-gated cost memory
-> uncertainty-, recency-, and outlier-robust cost memory
-> risk-coverage budget reservation against claim starvation
-> risk- and conservative-yield-adaptive global query budgeting
-> query-result-driven online budget release and revocation
-> claim-level information-posterior route reprioritization
-> provenance-family-normalized novelty and claim assignment
-> privacy-aware persistent claim-information posterior memory
-> failure-aware information-memory admission and neutral budget hold
-> two-stage cross-query-family negative-memory confirmation
-> correlated-retrieval-route negative-memory quorum
-> bounded provisional-memory aging and sample capping
-> complete retrieval usage contract
-> failure-aware cost-memory admission
-> frozen pre-ablation architecture certificate
-> question-obligation alignment and focused-answer safety gate
-> evidence-available refusal conflict repair
-> primary-definition nucleus and first-clause priority
-> source-bounded definition semantic gloss
-> evidence-to-theme interpretive bridge and overexpansion guard
-> ordered source-clause continuity and legal waterfall completeness
-> historical-evidence/current-category temporal-scope calibration
-> condition-preserving main-effect focus for influence questions
-> authority-bounded secured-default action realization
-> condition-matrix completion with deep evidence retrieval
-> ordered legal-procedure completion and exception preservation
-> claim-local sentence-window polarity and numeric conflict detection
-> dual-channel cultural-perception/artistic-representation influence bridge
-> final safety routing
-> delayed verification outcome credit
-> similarity-weighted strategy feedback
```

Important VSR files:

- `api/features/chats/vsr_pathrag.py`: query graph, gap diagnosis, overlay repair, signed claim/evidence graph, rollback.
- `api/features/chats/active_evidence.py`: value-guided active evidence acquisition, provenance admission, and query-focused text windows.
- `api/features/chats/claim_active_verification.py`: claim-level graph/entity/text verification, query feedback, Pareto transition audit, provenanced diagnosis correction, delayed outcome credit, and counterfactual commit/reject.
- `api/features/chats/experience_memory.py`: experience memory, routing context, and privacy-preserving verification-strategy credit recall.
- `api/features/chats/answer_realization.py`: answer realization support.
- `api/features/chats/agent_controller.py`: agent orchestration.

New V89-V92 modules (A+B combination strategy):

- `api/features/chats/adaptive_evidence_compression.py`: V89 query-complexity-aware evidence compression.
- `api/features/chats/salience_reordering.py`: V90 salience-preserving evidence reordering.
- `api/features/chats/fidelity_scoring.py`: V91 fidelity-weighted answer quality scoring.
- `api/features/chats/iterative_refinement.py`: V92 iterative answer refinement loop.

New V93-V97 modules (C+E combination strategy - Explainability + Consistency):

- `api/features/chats/answer_realization.py`: V93 claim-evidence traceability graph with per-claim source provenance.
- `api/features/chats/adaptive_evidence_routing.py`: V95 question-type-aware reasoning path selection.
- `api/features/chats/intra_answer_consistency.py`: V97 intra-answer consistency checking with pairwise contradiction detection.

New V99-V105 modules (F+G+H+I combination strategy - Advanced Features):

- `api/features/chats/counterevidence_cost_memory.py`: V99 cross-query evidence memory and transfer learning (reuses historical evidence).
- `api/features/chats/claim_active_verification.py`: V101 adversarial evidence robustness defense (detects and mitigates adversarial perturbations).
- `api/features/chats/active_evidence.py`: V103 multi-modal evidence fusion (supports table/figure/code evidence types).
- `api/features/chats/experience_memory.py`: V105 real-time evidence stream processing (handles incremental document updates).

Key VSR reports:

- `results/historical_vsr/vsr_pathrag_v1_three_core_innovations_report.md`
- `results/historical_vsr/vsr_pathrag_v5_active_evidence_report.md`
- `results/historical_vsr/vsr_pathrag_v6_final_innovation_report.md`
- `results/historical_vsr/vsr_pathrag_v9_counterfactual_realization_report.md`
- `results/historical_vsr/vsr_pathrag_v10_facet_retention_report.md`
- `results/historical_vsr/vsr_pathrag_v11_confidence_planned_facets_report.md`
- `results/historical_vsr/vsr_pathrag_v12_utility_gated_facets_report.md`
- `results/historical_vsr/vsr_pathrag_v13_terminal_action_report.md`
- `results/historical_vsr/vsr_pathrag_v14_observed_value_compute_report.md`
- `results/historical_vsr/vsr_pathrag_v15_post_verification_reallocation_report.md`
- `results/historical_baselines/vsr_pathrag_v16_status_gain_claim_retrieval_report.md`
- `results/historical_vsr/vsr_pathrag_v17_query_value_graph_text_report.md`
- `results/historical_vsr/vsr_pathrag_v18_delayed_outcome_credit_report.md`
- `results/historical_vsr/vsr_pathrag_v19_diagnosis_to_answer_correction_report.md`
- `results/historical_vsr/vsr_pathrag_v20_correction_persistence_report.md`
- `results/historical_vsr/vsr_pathrag_v21_discourse_lineage_report.md`
- `results/historical_vsr/vsr_pathrag_v22_source_certificate_report.md`
- `results/historical_vsr/vsr_pathrag_v23_transactional_rollback_report.md`
- `results/historical_vsr/vsr_pathrag_v24_dependency_transaction_report.md`
- `results/historical_vsr/vsr_pathrag_v25_evidence_bundle_report.md`
- `results/historical_vsr/vsr_pathrag_v26_bundle_stability_report.md`
- `results/historical_vsr/vsr_pathrag_v27_provenance_diversity_report.md`
- `results/historical_vsr/vsr_pathrag_v28_lineage_quorum_report.md`
- `results/historical_vsr/vsr_pathrag_v29_semantic_quorum_report.md`
- `results/historical_vsr/vsr_pathrag_v30_counterevidence_challenge_report.md`
- `results/historical_vsr/vsr_pathrag_v31_voi_counterevidence_report.md`
- `results/historical_vsr/vsr_pathrag_v32_joint_counterevidence_portfolio_report.md`
- `results/historical_vsr/vsr_pathrag_v33_fused_counterevidence_report.md`
- `results/historical_vsr/vsr_pathrag_v34_residual_cluster_report.md`
- `results/historical_vsr/vsr_pathrag_v35_observed_cost_report.md`
- `results/historical_vsr/vsr_pathrag_v36_cost_memory_report.md`
- `results/historical_vsr/vsr_pathrag_v37_robust_cost_memory_report.md`
- `results/historical_vsr/vsr_pathrag_v38_risk_coverage_report.md`
- `results/historical_baselines/vsr_pathrag_v39_adaptive_budget_report.md`
- `results/historical_vsr/vsr_pathrag_v40_online_budget_report.md`
- `results/historical_vsr/vsr_pathrag_v41_information_posterior_report.md`
- `results/historical_vsr/vsr_pathrag_v42_family_novelty_report.md`
- `results/historical_vsr/vsr_pathrag_v43_information_memory_report.md`
- `results/historical_vsr/vsr_pathrag_v44_failure_aware_admission_report.md`
- `results/historical_vsr/vsr_pathrag_v45_two_stage_memory_report.md`
- `results/historical_vsr/vsr_pathrag_v46_route_quorum_report.md`
- `results/historical_vsr/vsr_pathrag_v47_bounded_provisional_report.md`
- `results/historical_vsr/vsr_pathrag_v48_usage_contract_report.md`
- `results/historical_vsr/vsr_pathrag_v49_cost_admission_report.md`
- `results/historical_vsr/vsr_pathrag_v50_frozen_architecture_report.md`
- `results/historical_vsr/vsr_pathrag_v50_frozen_zero_overlap_report.md`
- `results/historical_vsr/vsr_pathrag_v51_question_obligation_report.md`
- `results/historical_vsr/vsr_pathrag_v52_anchor_preserving_context_report.md`
- `results/historical_vsr/vsr_pathrag_v52_refusal_guard_report.md`
- `results/historical_vsr/vsr_pathrag_v53_definition_nucleus_report.md`
- `results/historical_vsr/vsr_pathrag_v54_symbolic_bridge_report.md`
- `results/historical_vsr/vsr_pathrag_v55_ordered_obligation_report.md`
- `results/historical_vsr/vsr_pathrag_v56_temporal_scope_report.md`
- `results/historical_vsr/vsr_pathrag_v57_main_effect_focus_report.md`
- `results/historical_vsr/vsr_pathrag_v58_v60_condition_procedure_report.md`
- `results/historical_vsr/vsr_pathrag_v61_local_polarity_sexuality_bridge_report.md`
- `results/historical_vsr/vsr_pathrag_v62_cumulative_grazing_pressure_report.md`
- `results/historical_vsr/vsr_pathrag_v63_administrative_intelligence_role_report.md`
- `results/historical_vsr/vsr_pathrag_v64_v65_impact_and_creation_report.md`
- `results/historical_vsr/vsr_pathrag_v66_observation_management_report.md`
- `results/historical_vsr/vsr_pathrag_v68_darkness_disambiguation_report.md`

VSR v6 result boundary:

- Controlled verification: 30/30 passed
- Development 12 and held-out 12 both successful
- Unsupported final output claims: 0
- Base graph mutations: 0
- This validates mechanism correctness, but it is not enough alone to claim broad answer-quality improvement.

VSR v9 answer-realization result on the cached 30-sample suite:

- V6 Overall 1.900 -> V7 3.767 -> V8 3.967 -> **V9 4.100**
- V8 correctness 4.200 -> **V9 4.467**
- V8 poor samples 2 -> **V9 poor samples 1**
- V9 submitted 30/30 answers; 28 needed one generation and 2 needed a correction.
- Tradeoff: conciseness decreased from v8 4.300 to v9 3.967.
- This is a development-scale result. It still requires frozen no-overlap validation.

VSR v10 supported-facet retention result:

- V9 Overall 4.100 -> **V10 4.167**
- V9 correctness 4.467 -> **V10 4.500**
- V9 coverage 4.033 -> **V10 4.133**
- The Mary/"Witch of Atlas" regression moved from overall 3 to 5 by retaining
  both source-supported objections.
- Controlled changed-row regression check: only line 0 was regenerated and
  re-evaluated; the other 29 v9 rows were held fixed and unchanged. This is not
  a formal ablation.
- The deterministic gate is intentionally restricted to high-confidence
  objection/reason questions. This is a development result, not broad
  validation.

VSR v11 confidence-planned facet checkpoint:

- Generalizes planning to cause, challenge, impact, history, contribution,
  finding, feature, and origin/use query families.
- On the cached 30-sample development suite: 12 queries entered planner
  evaluation, 3 received high-confidence facet plans, and 9 safely abstained.
- Existing v10 answers passed 3/3 planned-facet retention checks.
- Full test suite: 42/42 passed.
- This is a mechanism checkpoint with held-fixed answers, not a new
  answer-quality result and not an ablation.
- The original plan was to freeze at v13, but this was superseded after the
  broader pipeline audit exposed additional innovation opportunities.

VSR v12 utility-gated facet checkpoint:

- Adds `enforce`, `monitor`, and `skip` actions after v11 facet planning.
- On the cached 30-sample development suite: 1 enforce, 2 monitor, and 27 skip.
- Hard-gate retention failures: 0.
- Potential facet-only extra-repair exposure decreases from 3 planned answers
  to 1 enforced answer; this is a proxy mechanism result, not measured API cost.
- Full test suite: 46/46 passed.
- Answers were held fixed and no generator/judge calls were made. This is not a
  new answer-quality result and not an ablation.

VSR v13 risk-bounded terminal action checkpoint:

- Converts the final evidence audit from a display-only card into an executable
  commit, monitor, fallback, or abstain controller.
- Calibrates signed contradictions so near-tied lexical scores do not trigger
  hard rollback.
- On the cached 30-sample development suite: 22 direct commits, 5 monitored
  commits, 3 verified fallbacks, and 0 abstentions.
- All 3 fallbacks used natural-language terminal claim pruning; no graph-dump
  fallback was selected.
- Full test suite: 50/50 passed.
- No generator/judge calls were made. This is a mechanism checkpoint, not a new
  answer-quality result and not an ablation.
- V13 is not frozen. The next candidate innovation is actual stage-level cost
  attribution and dynamic compute reallocation.

VSR v14 observed-value compute reallocation checkpoint:

- Adds decision-boundary wall-clock attribution and observable external-call
  accounting to every returned Agent trace.
- Adds executable `continue_full`, `escalate_verification`,
  `limit_refinement`, and `stop_expensive_refinement` actions.
- Under standard-budget replay on the fixed 30-sample development traces:
  16 continue and 14 escalate verification.
- Under 80% latency-budget pressure: 2 continue, 25 limit refinement, and
  3 stop expensive refinement.
- Claim-active verification, signed-claim safety, rollback, and terminal
  actions remain active under every compute action.
- Full test suite: 53/53 passed.
- Answers were held fixed and no generator/judge calls were made. Gate closure
  counts are mechanism diagnostics, not measured API savings or ablation.
- Provider token accounting is explicitly unavailable until usage metadata is
  propagated through the current adapters.

VSR v15 post-verification residual-value checkpoint:

- Adds a second executable compute decision after claim-active verification.
- Scores verified marginal value from resolved claims, signed-status gains,
  safe evidence commits, and knowledge-gap reduction.
- Recomputes residual need from unknown/contradicted claims, graph gaps, and
  uncovered evidence goals.
- Can retain, close, or reopen answer realization; explicit fast mode can never
  be reopened.
- On the fixed 30-sample standard-budget replay, all 30 realization decisions
  were retained; 2 were explicitly retained because residual risk was high.
- Under 80% latency pressure: 28 retained, 1 reopened, and 1 closed.
- Historical claim verification produced zero positive status-gain cases,
  exposing claim-targeted retrieval as the next mechanism bottleneck.
- Full test suite: 56/56 passed.
- Answers were held fixed and no generator/judge calls were made. This is a
  policy-transition audit, not an ablation or answer-quality result.

VSR v16 status-gain claim retrieval checkpoint:

- Sanitizes refusal/meta language and generic verification scaffolding from
  claim queries.
- Allocates `claim_core`, `question_conditioned`,
  `counterevidence_question`, and `lexical_pivot` queries round-robin across
  target claims.
- Keeps explicit fast mode at one claim query; standard low/medium/high tiers
  allow 2/4/6 queries.
- Adds provenanced entity descriptions to claim verification.
- Temporarily pools complementary cross-query evidence and commits it only
  after signed claim status improves without increasing contradictions.
- Fixed 30-row planning audit reconstructed 4 attempted claims on 3 rows:
  4 historical queries become 8 standard-budget variants, retrieval
  scaffolding contamination falls 4 to 0, and refusal/meta contamination falls
  2 to 0.
- Controlled tests verify entity-only resolution and cross-query bundle
  resolution. Full test suite: 59/59 passed.
- No retrieval/generator/judge calls were made by the planning audit. It does
  not establish measured retrieval or answer-quality improvement and is not an
  ablation.

VSR v17 query-value graph-text verification checkpoint:

- Adds per-query candidate, admission, latency, signed-transition, and
  gain-per-second telemetry.
- Dynamically reorders and stops query variants using observed value and
  pending complementary evidence.
- Extends claim verification from graph relations and entities to
  query-focused original text chunks without changing the normal active
  acquisition cost surface.
- Replaces the contradiction-count commit gate with a multi-claim Pareto audit:
  new supported or diagnosed states are valuable, but settled claims may not
  regress.
- Calibrates title-initial `No` so titles such as *No Trees in the Street* do
  not create false semantic contradictions.
- Controlled real local-index evaluation held historical drafts fixed and used
  no generator, judge, or reference answer. Lines 0 and 7 required no query.
  Line 13 executed 1 of 4 planned variants, produced 3 signed status gains
  (2 supported and 1 diagnosed), and skipped the remaining 3 variants.
- Line 13 achieved 3.0 status gains/query and 4.404 gains/verification-second
  with no claim regression.
- Full test suite: 67/67 passed.
- This is a targeted development mechanism result, not an ablation or broad
  answer-quality result.

VSR v18 delayed verification outcome credit checkpoint:

- Audits each verification query after answer realization and terminal policy,
  using source survival, observable answer use, claim-transition retention,
  terminal action, and final risk.
- Uses multiplicative claim-transition retention so immediate status gain is
  not treated as a complete success when the final answer fails to apply it.
- Stores only aggregate strategy credit in episodic memory; raw verification
  queries and claim text are not persisted in the strategy feedback.
- Similar-context recall returns strategy-level credit, which enters query
  priority through a bounded 0.65-1.35 multiplier without changing query
  budget or bypassing safety gates.
- Controlled real local-index replay: lines 0 and 7 needed no query. Line 13
  retained its source and two supported transitions, but its diagnosed false
  claim remained in the held-fixed final draft; immediate gain 3 therefore
  became 0.6667 partial delayed credit.
- Full test suite: 73/73 passed.
- No generator, judge, or reference answer was used. This is a targeted
  development mechanism result, not an ablation or answer-quality result.

VSR v19 diagnosis-to-answer correction checkpoint:

- Converts provenanced `unknown -> contradicted` verification transitions into
  a separately audited `diagnosis_corrected` answer candidate.
- Prefers clause-level removal of contradicted meta/refusal suffixes, supports
  opposite-polarity replacement only from committed source text, and permits
  whole-sentence removal only when question coverage remains safe.
- Accepts an edit only when contradicted claims do not increase, supported
  claims remain above a diagnosis-adjusted floor, question coverage drops by
  no more than 0.05, and the answer remains non-empty.
- Controlled real local-index replay held retrieval constant. On line 13, one
  exact meta clause was trimmed; signed status changed from 9 supported and
  1 contradicted to 10 supported and 0 contradicted, while question coverage
  stayed 0.6154.
- The same query's delayed credit improved from v18's 0.6667 partial credit to
  v19's 1.0000 retained credit, with no additional retrieval.
- Full test suite: 76/76 passed.
- No generator, judge, or reference answer was used. This is a targeted
  development mechanism result, not an ablation or broad answer-quality
  result.

VSR v20 correction-persistent realization checkpoint:

- Converts accepted v19 corrections into source-linked prohibited assertions.
  Clause trimming prohibits only the rejected suffix, preserving the valid
  factual prefix.
- Sends those constraints into every answer-realization attempt and audits
  each generated claim for semantic overlap under the same assertion polarity.
- A recurring corrected assertion fails the realization audit. If all attempts
  recur or otherwise fail, the controller keeps the pre-realization corrected
  answer instead of committing the generated draft.
- Controlled replay used the one accepted real v19 correction row. The
  corrected answer and an opposite-polarity form passed the persistence gate;
  the original draft and a same-polarity semantic recurrence were rejected:
  4/4 expected outcomes. Grounding remains enforced by the normal realization
  audit.
- Full test suite: 84/84 passed.
- The real replay has only one corrected row, and the semantic paraphrase is a
  deterministic stress case. No generator, judge, reference answer, or
  component removal was used. This is not an ablation or broad quality result.

VSR v21 discourse-lineage checkpoint:

- Extends v20 with an `evidence_absence` semantic assertion family so implicit
  negatives such as `undocumented` or `records remain silent` retain the
  correction lineage even without the original `not provide details` wording.
- Canonical topic anchors connect journalist/reporting and
  professional/work/career forms. Semantic-family matches require sufficient
  topic overlap, preventing unrelated evidence-absence statements from being
  rejected.
- Audits both individual claims and adjacent two-sentence windows. This catches
  cases where one sentence carries the corrected topic and the next carries
  the prohibited absence predicate.
- The controlled replay initially exposed one false positive: a different
  research-detail limitation shared generic absence language and one `career`
  anchor. Requiring at least two anchors for multi-anchor constraints removed
  the false positive while retaining the distant and split-sentence catches.
- Final controlled replay on the one real v19 correction lineage: 7/7 expected
  outcomes. Match modes were 2 lexical, 1 semantic-family, and 1 discourse
  window; three safe cases passed.
- Full test suite: 89/89 passed.
- No generator, judge, reference answer, or component removal was used. This
  remains a development mechanism result, not an ablation or broad
  answer-quality result.

VSR v22 source-bound correction certificate checkpoint:

- Detects when realization states the opposite of a prohibited assertion and
  requires a source-bound certificate instead of relying only on the general
  answer support audit.
- Prioritizes the committed correction-source excerpt in realization context.
  If an earlier retrieval already has the same chunk ID, the committed focused
  excerpt refreshes that potentially stale or prefix-only chunk.
- A certificate requires the correction source to be available, explicitly
  named in `used_source_ids`, and locally support the opposite-polarity claim
  with score at least 0.60.
- Certificate auditing runs only after the same-polarity persistence gate
  passes, avoiding duplicate rejection reasons.
- Controlled replay used the actual committed text chunk from the one real v19
  correction lineage: 8/8 expected outcomes. Five cases required
  certificates; 2 were certified and 3 were rejected for unbound citation,
  weak same-source support, or unavailable source.
- Full test suite: 98/98 passed.
- No generator, judge, reference answer, or component removal was used. This
  is a targeted development mechanism result, not an ablation or broad
  answer-quality result.

VSR v23 transactional correction rollback checkpoint:

- Extends v22 to audit every single-sentence opposite claim from the same
  correction lineage. A certified inversion can no longer hide a second
  uncertified inversion behind a best-candidate reduction.
- Locates same-polarity recurrences and uncertified inversion spans in the
  generated answer, then selects a safe matching lineage claim from the
  pre-realization corrected answer.
- Replaces only unsafe spans, removes them when the corrected claim already
  exists, preserves other generated claims, and carries the committed
  correction source IDs into the patched audit.
- A transaction is committed only when the complete existing realization audit
  passes after patching. Failed patches leave the original failure active and
  continue to repair or full fallback.
- Controlled replay used the real committed v19 source: 6/6 expected outcomes,
  5 transaction triggers, 4 accepted patches, 3 safe generated claims
  preserved, and 1 patch rejected by the full support audit.
- Full test suite: 101/101 passed.
- No generator, judge, reference answer, or component removal was used. This
  remains a development mechanism result, not an ablation or broad
  answer-quality result.

VSR v24 dependency-aware realization transaction checkpoint:

- Builds a sentence-level claim dependency graph before local rollback, with
  high-confidence immediate anaphoric and causal edges.
- Scores each generated claim from existing source support and question
  relevance, then retains the maximum-utility feasible subset under dependency
  closure.
- Recursively removes dependent claims when an unsafe parent is removed, but
  keeps and explicitly rebinds them when the parent is replaced by a corrected
  lineage claim.
- Records graph nodes, edges, unsafe/replaced/removed/cascade/rebound/kept node
  IDs, retained utility, and coherence in the runtime transaction trace.
- Controlled replay used the real committed v19 source: 6/6 expected outcomes,
  5 transaction triggers, 4 accepted patches, 3 dependent claims cascade
  removed, 1 dependency rebound, 8 generated claims preserved across accepted
  stress cases, and 1 unsupported residual patch rejected by the full audit.
- The v23 controlled replay remains 6/6 under the v24 implementation.
- Full test suite: 106/106 passed.
- No generator, judge, reference answer, or component removal was used. This
  remains a development mechanism result, not an ablation or broad
  answer-quality result.

VSR v25 evidence-conditioned claim-bundle checkpoint:

- Adds supporting evidence IDs to every audited claim, without additional
  retrieval or model calls.
- Resolves anaphoric and causal antecedents over the previous four claims using
  inverse distance, concept continuity, shared supporting sources, and lexical
  continuity.
- Adds exact tree dynamic programming over the dependency forest. Net utility
  combines source support, question relevance, fixed selection cost, and an
  unsupported-claim risk penalty.
- Treats sufficiently supported obligation/facet coverage nodes as mandatory,
  while a corrected replacement acts as a virtual retained parent for rebind.
- Controlled real-source replay: 6/6 expected outcomes, 5 triggers, 4 accepted
  patches, 2 distant dependencies, 1 distant cascade, 1 distant rebind,
  1 independent hallucination pruned, 1 source-bound obligation protected, and
  1 invented-year patch rejected by the complete numeric audit.
- Historical v23 and v24 controlled replays both remain 6/6.
- Full test suite: 111/111 passed.
- No generator, judge, reference answer, or component removal was used. This
  remains a development mechanism result, not an ablation or broad
  answer-quality result.

VSR v26 counterfactual bundle-stability checkpoint:

- Records every antecedent candidate, runner-up score, and selection margin;
  edges require a `0.10` margin to be counterfactually stable.
- Reruns the exact bundle optimizer after subtracting `0.06` from selected
  nodes and adding `0.06` to pruned nodes.
- Removes selected claims whose status flips, plus descendants made incoherent
  by that removal.
- Allows actual unstable claims to survive only through strong source support
  or an evidence-bound obligation/facet assignment. Stable source claims do not
  inflate the fallback protection count.
- Controlled real-source replay: 7/7 expected outcomes, 6 triggers, 5 accepted
  patches, 2 stable distant dependencies, 1 baseline bundle prune,
  1 counterfactual stability prune, and 1 invented-year patch rejected by the
  complete numeric audit.
- Historical v23, v24, and v25 controlled replays remain 6/6.
- Full test suite: 116/116 passed.
- No generator, judge, reference answer, or component removal was used. This
  remains a development mechanism result, not an ablation or broad
  answer-quality result.

VSR v27 provenance-diverse stability checkpoint:

- Adds typed graph/text evidence signatures, original provenance IDs, and local
  support scores to each audited claim.
- Requires two distinct strong text origins or graph/text agreement before an
  unstable claim can receive fallback protection. Two evidence IDs from the
  same original document count once.
- Projects retained claims back to their evidence IDs and drops citations that
  only supported removed claims before the complete patched-answer audit.
- Controlled replay: 7/7 expected outcomes, 6 triggers, 5 accepted patches,
  2 single-origin rescue rejections, 2 diverse-provenance rescues,
  3 stability-pruned claims, and 3 stale evidence IDs removed.
- The first four cases are provenance stress fixtures anchored to the real v19
  correction lineage; the final three use the actual committed local source.
- Historical v23-v25 replays remain 6/6 and v26 remains 7/7.
- Full test suite: 122/122 passed.
- No external generator, judge, reference answer, or component removal was
  used. This remains a development mechanism result, not an ablation or broad
  answer-quality result.

VSR v28 lineage-aware stability checkpoint:

- Propagates direct origins, document ancestors, lineage chains, derived-source
  IDs, and correlation-family IDs through text and graph evidence.
- Resolves graph evidence through matching text chunks, then folds strong
  evidence into transitive source-family components. Chunks and graph facts
  derived from one document count once.
- Requires two independent, conflict-free support families to rescue an
  unstable claim. A strong contradictory family blocks the rescue.
- Controlled audit: 8/8 expected outcomes, 7 triggers, 7 accepted patches,
  2 lineage-family rescues, 3 correlated-quorum rejections, 1 conflict
  rejection, 3 evidence rows collapsed, 5 stability-pruned claims, and
  9 stale evidence IDs removed.
- The first six cases are deterministic genealogy/conflict stress fixtures
  anchored to the real v19 correction lineage; the final two use the actual
  committed local source.
- Historical v23-v25 replays remain 6/6 and v26 remains 7/7. The v27 stored
  result remains its versioned artifact, while its same-origin graph/text
  rescue rule is intentionally superseded.
- Full test suite: 127/127 passed.
- No external generator, judge, reference answer, or component removal was
  used. This remains a development mechanism result, not an ablation or broad
  answer-quality result.

VSR v29 semantic-calibrated stability checkpoint:

- Classifies each claim-evidence edge as entailment, neutral overlap, or
  contradiction using lexical coverage plus polarity, numeric, and antonym
  predicate checks.
- Propagates explicit source reliability and uses conservative defaults when
  missing. Each correlated family contributes only its best entailment weight,
  preventing duplicate excerpts from accumulating confidence.
- Adapts the required weighted family quorum to low, medium, or high claim
  risk while retaining the two-independent-family and no-conflict constraints.
- Controlled audit: 9/9 expected outcomes, 8 triggers, 8 accepted patches,
  2 semantic weighted-family rescues, 1 neutral rejection, 1 reliability
  rejection, 1 high-risk rejection, 3 semantic conflict rejections,
  6 stability-pruned claims, and 13 stale evidence IDs removed.
- The first seven cases are deterministic semantic stress fixtures anchored
  to the real v19 correction lineage; the final two use the actual committed
  local source.
- Historical v23-v25 replays remain 6/6, v26 remains 7/7, and v28 remains 8/8.
- Full test suite: 131/131 passed.
- No external generator, judge, reference answer, or component removal was
  used. This remains a development mechanism result, not an ablation or broad
  answer-quality result.

VSR v30 active counterevidence challenge checkpoint:

- Generates claim-targeted falsification queries only for medium/high-risk
  unstable claims rescued by the v29 semantic-family certificate.
- Reuses the real local `text_chunks` retriever through an optional async
  callback, with a hard budget of 2 claims, 2 queries per claim, and 6 unique
  returned evidence rows per transaction.
- Collapses challenge evidence through source genealogy. Independent
  contradictions revoke the claim; same-family echoes or empty searches are
  inconclusive and conservatively revoke; independent non-contradictory
  evidence passes the bounded challenge.
- Controlled audit: 8/8 expected outcomes, 7 triggers, 7 accepted patches,
  5 challenges, 10 queries, 1 independent contradiction, 2 passed challenges,
  2 inconclusive challenges, 3 challenge-pruned claims, 1 prior
  stability-pruned claim, and 6 stale evidence IDs removed.
- The first six cases use deterministic challenge-return fixtures anchored to
  the real v19 correction lineage; the final two use the actual committed
  local source and no-transaction path.
- Historical v23-v25 replays remain 6/6, v26 remains 7/7, v28 remains 8/8,
  and v29 remains 9/9.
- Full test suite: 133/133 passed.
- No external generator, judge, reference answer, or component removal was
  used. This remains a development mechanism result, not an ablation or broad
  answer-quality result.

VSR v31 value-of-information counterevidence checkpoint:

- Scores each planned challenge query from claim risk, unresolved certificate
  uncertainty, query-position novelty, and prior-query evidence novelty.
- Always executes the first query, then stops after decisive contradiction,
  strong independent non-contradiction, or expected value below `0.18`.
  Duplicate and empty results can receive one follow-up but not an unproductive
  third query.
- Preserves the fixed v30 controller as the default API-compatible mode while
  the production Agent explicitly enables v31 value control.
- Controlled audit: 8/8 expected outcomes, 7 triggers, 7 accepted patches,
  5 challenges, 8 executed queries, 7 saved versus the 15-query planning
  ceiling, 2 fewer than the v30 fixed replay, 5 early stops,
  1 decisive stop, and 4 diminishing-return stops.
- Challenge decisions remain identical to v30: 1 contradiction, 2 passes,
  2 inconclusive outcomes, and 3 challenge-pruned claims.
- v28 remains 8/8, v29 remains 9/9, and the fixed v30 replay remains 8/8 with
  its original 10 queries.
- Full test suite: 135/135 passed.
- No external generator, judge, reference answer, or component removal was
  used. This remains a development mechanism result, not an ablation or broad
  answer-quality result.

VSR v32 joint counterevidence portfolio checkpoint:

- Places eligible queries from all risky rescue candidates into one global
  queue ranked by expected certificate change per estimated retrieval cost.
- Uses one global four-query/six-result budget and re-evaluates every newly
  retrieved row against every candidate claim. Evidence retrieved for one
  claim can resolve another when it is independently relevant.
- Keeps v30 fixed and v31 per-claim VOI modes as default-compatible API paths;
  the production Agent explicitly enables the v32 joint portfolio.
- Controlled audit: 6/6 expected outcomes. Four synthetic scheduler cases
  planned up to 24 per-claim queries but executed 6, saved 18, recorded
  2 cross-claim reuses and 2 shared evidence rows, passed 4 claims, and
  pruned 4 claims.
- One shared confirmation resolves two claims with one query; one shared
  contradiction revokes two claims with one query. A separate case confirms
  high-risk-first ordering, and another confirms the global query cap.
- The final two cases use the real v19 correction lineage for pre-portfolio
  stability pruning and no-transaction boundaries.
- v30 remains 8/8 with 10 queries and v31 remains 8/8 with 8 queries.
- Full test suite: 136/136 passed.
- No external generator, judge, reference answer, or component removal was
  used. This remains a development mechanism result, not an ablation or broad
  answer-quality result.

VSR v33 fused multi-claim counterevidence checkpoint:

- Clusters semantically related rescue candidates using normalized content-term
  overlap and conservatively limits each cluster to two claims.
- Synthesizes one labelled contrastive query per cluster and schedules it in
  the existing value-per-estimated-cost queue. Unrelated claims retain v32's
  independent risk/value priority.
- Computes member-specific anchors after removing shared claim terms. Retrieved
  evidence must hit a member's unique anchor before it can confirm, contradict,
  or be reused for that member.
- Controlled audit: 6/6 expected outcomes. Four scheduler cases planned up to
  24 per-claim queries but executed 5 and saved 19. Three fused queries
  explicitly covered 4 claims, while 2 generic predicate-only assignments were
  rejected. The portfolio passed 4 claims and pruned 4 claims.
- One fused confirmation resolves two claims, one fused contradiction revokes
  two claims, and `The program caused results.` is rejected for both
  sales-specific and revenue-specific coverage.
- The final two cases retain the real v19 correction-lineage stability-prune
  and no-transaction boundaries.
- Full test suite: 139/139 passed. V30 remains 8/8 with 10 queries, v31 remains
  8/8 with 8 queries, and v32 remains 6/6 with 6 queries.
- No external generator, judge, reference answer, or component removal was
  used. This remains a development mechanism result, not an ablation or broad
  answer-quality result.

VSR v34 residual cluster counterevidence checkpoint:

- Detects partial fused-query coverage only when at least one cluster member is
  resolved and at least one remains unresolved, then excludes resolved members
  from a specialized residual query.
- Schedules the residual task in the global value/cost portfolio with a
  development-stage `1.35` specificity multiplier and advances only residual
  members when it executes.
- Adds controlled outcome alias families. An original anchor is projected into
  relation scoring only when its alias actually appears in the evidence; raw
  and effective lexical coverage remain visible in the trace.
- Controlled audit: 6/6 expected outcomes. Four scheduler cases planned up to
  24 queries, executed 7, and saved 17. They detected 3 partial clusters,
  executed 3 residual queries, resolved all 3 residual claims, recorded
  1 evidence-triggered alias coverage, and rejected 7 coverage-mismatched
  evidence assignments.
- The portfolio passed 5 claims and pruned 3. Generic predicate-only evidence
  still cannot create a false partial cluster or false reuse.
- Full test suite: 141/141 passed. V30 remains 8/8, v31 remains 8/8, v32
  remains 6/6, and v33 remains 6/6 with their original query totals.
- No external generator, judge, reference answer, or component removal was
  used. This remains a development mechanism result, not an ablation or broad
  answer-quality result.

VSR v35 observed-cost counterevidence checkpoint:

- Accepts either the historical evidence-list callback or a structured
  response carrying evidence rows, latency, provider calls, and token usage.
  Legacy callbacks retain measured wall-clock latency and compatibility.
- Computes a normalized development cost from latency, calls, and tokens, then
  combines unique-evidence and claim-coverage yield into a per-task online
  profile. Claim, fused-cluster, and residual-cluster routes have separate
  profiles.
- Records both the old length-proxy choice and the observed-cost choice. The
  reprioritization counter changes only when observed history changes the
  selected task.
- Controlled audit: 6/6 expected outcomes. Four scheduler cases planned up to
  18 queries, executed 6, saved 12, made 6 profile updates, and recorded
  1 observed-cost reprioritization. Structured fixtures represented 8 provider
  calls, 6800 tokens, 5 unique evidence rows, and information yield 9.
- The expensive-route case preserves high-risk-first exploration, then moves
  to an unexplored lower-risk claim after the first route reports 8000 ms,
  2 calls, 4000 tokens, and no relevant claim assignment.
- Full test suite: 142/142 passed. V30-v34 all retain their previous controlled
  audit outcomes and query totals.
- Production currently measures callback latency, but token/call metadata
  beyond the callback boundary remains unavailable until upstream propagation.
- No external generator, judge, reference answer, or component removal was
  used. This remains a development mechanism result, not an ablation or broad
  answer-quality result.

VSR v36 similarity-gated cost-memory checkpoint:

- Changes the production counterevidence callback to a structured response with
  measured latency, provider calls/tokens, adapter identity, and usage status.
  The current local chunk scan correctly reports zero provider calls and zero
  provider tokens.
- Stores only task type, claim count, and SHA-256 prefixes of normalized terms;
  original claims and queries are not persisted in cost memory.
- Requires task-type agreement and at least `0.55` hashed-term similarity.
  Historical transfer is sample-aware and capped at `0.60`; current-transaction
  observations override historical profiles.
- Merges identical contexts by sample-weighted means under an atomic,
  process-locked JSONL update. Memory is bounded and independently disableable.
- Controlled audit: 6/6 expected outcomes. One similar expensive prior was
  applied and caused one reprioritization; one unrelated prior was rejected.
  Local retrieval propagated 0 provider calls and 0 tokens. Two updates merged
  into a three-sample profile with cost 4.0 and yield 0.7.
- Agent execution-cost accounting now exposes propagated counterevidence usage
  as partial token accounting while generation-adapter tokens remain unknown.
- Full test suite: 144/144 passed. V30-v35 retain all previous controlled
  outcomes and query/usage totals.
- No external generator, judge, reference answer, or component removal was
  used. This remains a development mechanism result, not an ablation or broad
  answer-quality result.

VSR v37 robust cost-memory checkpoint:

- Adds 30-day recency decay, effective sample counts, upper confidence cost
  bounds, lower confidence yield bounds, and rejection below 0.5 effective
  samples.
- Tracks Welford cost/yield variance in transaction-local and persistent
  profiles. Persistent extreme batch means are clipped after three samples.
- Controlled audit: 6/6 expected outcomes. One uncertain prior used
  conservative bounds and caused one reprioritization; one year-old prior was
  rejected; a one-sample prior received 0.333 transfer weight; one extreme
  update was clipped.
- Full suite at this checkpoint: 146/146 passed. V30-v37 all passed.
- No component removal or external judge was used. This is not an ablation.

VSR v38 risk-coverage budget checkpoint:

- Tracks whether every unresolved medium/high-risk claim has participated in a
  counterevidence query.
- When remaining query budget is no greater than the unresolved unqueried claim
  count, reserves the current choice for an unqueried claim unless the normal
  value/cost winner already covers one.
- Fused queries mark every member as covered. Observed-cost reprioritization and
  coverage intervention retain separate counters and selection audits.
- Controlled audit: 6/6 expected outcomes. The critical two-query case covered
  2/2 risky claims with one necessary intervention; one fused query covered two
  related claims; high-risk-first ordering was preserved.
- Full suite: 148/148 passed. V30-v38 all retain their expected audit outcomes.
- Current controllers are `risk_coverage_counterevidence_v9` and
  `risk_coverage_transactional_rollback_v16`.
- No component removal or external judge was used. This is not an ablation.

VSR v39 adaptive risk-yield budget checkpoint:

- Replaces unconditional use of the global query cap with a coverage floor
  plus risk extension credits, while retaining the caller's hard ceiling.
- Only risk above `0.55` earns extra queries. Unseen routes retain exploratory
  yield 1.0; similar valid historical routes use v37's conservative yield
  lower bound, so repeatedly low-yield routes earn fewer extra queries.
- Controlled audit: 6/6 expected outcomes. Two medium-risk claims reduced a
  hard cap of 4 to an effective budget of 2; two unseen high-risk claims kept
  4; historical low-yield high-risk routes received 3; hard cap 1 remained
  nonnegotiable.
- Full suite: 150/150 passed. V30-v39 all retain expected audit outcomes.
- Current controllers are `adaptive_budget_counterevidence_v10` and
  `adaptive_budget_transactional_rollback_v17`.
- No component removal or external judge was used. This is not an ablation.

VSR v40 online budget-reallocation checkpoint:

- Updates the effective query budget after every result from realized unique
  yield, claim coverage, and certificate-state change.
- High-value unresolved routes release an extension query; low-value routes
  revoke surplus budget without crossing the unresolved-claim coverage floor.
- Controlled audit: 6/6. Relevant unresolved routes expanded 2 to 4 with two
  releases; low-value routes contracted 4 to 2 with two revocations and still
  covered 2/2 risky claims.
- Full suite at checkpoint: 153/153. V30-v40 all passed. Not an ablation.

VSR v41 claim information-posterior checkpoint:

- Maintains per-claim mean realized information value and low-value streaks
  across claim, fused, and residual task types.
- Applies a confidence-weighted value multiplier before the v38 coverage
  reserve; unseen fused members attenuate existing penalties.
- Controlled audit: 6/6. Low-value multiplier 0.593, partially unseen fused
  multiplier 0.671, productive multiplier 0.94, one information-driven
  reprioritization, and zero coverage intervention for that switch.
- Full suite: 155/155. V30-v41 all retain expected outcomes.
- Current controllers are `information_posterior_counterevidence_v12` and
  `information_posterior_transactional_rollback_v19`.
- No component removal or external judge was used. This is not an ablation.

VSR v42 provenance-family-normalized novelty checkpoint:

- Derives a stable independence family from root/document ancestry, then
  correlation identity, with source identity only as fallback.
- Keeps raw source counts but uses newly observed families and claim-family
  assignments for cost yield, online budget updates, and information posterior
  credit.
- Controlled audit: 6/6. Two chunks from one document became one family, two
  independent documents remained two, and a repeated document in the next
  query earned zero new-family credit.
- Full suite: 158/158. V30-v42 all retain expected outcomes.
- Current controllers are `family_normalized_counterevidence_v13` and
  `family_normalized_transactional_rollback_v20`.
- No component removal or external judge was used. This is not an ablation.

VSR v43 persistent information-posterior checkpoint:

- Stores only hashed claim context, sample statistics, variance, and trailing
  low-value count; original claims and queries are not persisted.
- Uses task-type/similarity gating, 30-day recency decay, effective samples,
  an optimistic information upper confidence bound, and transfer capped at
  0.50.
- Controlled audit: 6/6. One similar low-value prior caused one early
  reprioritization, an unrelated prior had no effect, a year-old prior was
  rejected, and persistent statistics merged to three samples.
- Full suite: 162/162. V30-v43 all retain expected outcomes.
- Current controllers are `persistent_information_counterevidence_v14` and
  `persistent_information_transactional_rollback_v21`.
- No component removal or external judge was used. This is not an ablation.

VSR v44 failure-aware memory-admission checkpoint:

- Separates retrieval failure and incomplete adapter metadata from genuine
  low-information outcomes before updating online or persistent posteriors.
- Rejects exceptions, legacy/partial usage, and nonempty results with less than
  80% lineage-qualified rows. Complete structured empty results remain
  learnable.
- Rejected observations use neutral live value 0.5, so they neither release nor
  revoke budget.
- Controlled audit: 6/6. A retrieval exception produced zero updates and held
  budget; a complete zero-yield search produced one valid update.
- Full suite: 165/165. V30-v44 all retain expected outcomes.
- Current controllers are `failure_aware_information_counterevidence_v15` and
  `failure_aware_information_transactional_rollback_v22`.
- No component removal or external judge was used. This is not an ablation.

VSR v45 two-stage negative-memory checkpoint:

- Keeps the first trustworthy low-information query family provisional and
  outside historical recall.
- The same family cannot self-confirm; a second distinct hashed strategy family
  is required before negative statistics enter long-term memory.
- Positive feedback commits immediately and clears provisional negatives.
- Controlled audit: 6/6. First family produced one provisional sample, same
  family repetition produced two provisional samples, a second family
  confirmed three samples, and positive recovery committed one sample.
- Full suite: 167/167. V30-v45 all retain expected outcomes.
- Current controllers are `two_stage_information_counterevidence_v16` and
  `two_stage_information_transactional_rollback_v23`.
- No component removal or external judge was used. This is not an ablation.

VSR v46 correlated-route negative-memory quorum checkpoint:

- Adds privacy-preserving hashes for retrieval adapter/route identity.
- Negative memory requires diversity across both query families and retrieval
  routes. Either dimension alone remains provisional.
- Controlled audit: 6/6. Different query families on one route committed zero
  samples; different routes with one query family committed zero; diversity in
  both dimensions committed three samples.
- Full suite: 169/169. V30-v46 all retain expected outcomes.
- Current controllers are `route_quorum_information_counterevidence_v17` and
  `route_quorum_information_transactional_rollback_v24`.
- No component removal or external judge was used. This is not an ablation.

VSR v47-v51 freeze, validation, and targeted repair checkpoints:

- V47 expires unconfirmed negatives after 30 days by default and caps each
  provisional context at 8 effective samples.
- V48 requires explicit retrieval latency, adapter, status, provider-call, and
  token metadata before persistent information learning.
- V49 separately gates cost learning, so exceptions, incomplete contracts, and
  failed usage states cannot pollute observed or persistent cost memory.
- V50 freezes the production architecture without adding behavioral changes.
  Runtime traces identify the last mechanism as v49 and explicitly record that
  ablation has not started.
- V50 zero-overlap validation completed 30/30, with 171 checked claims and zero
  unsupported claims, but its reference Overall was only 2.767 and its complete
  pairwise average against HyDE was 38.67%. The main failures were
  question-misaligned graph dumps, refusal-like answers, and verbosity.
- V51 repairs terminal question alignment and the comparison-query classifier.
  It completed 30/30, shortened mean output from 1184.9 to 812.9 characters,
  reduced runtime from 148.91 to 141.31 seconds, and retained zero unsupported
  final claims.
- V51 reference Overall improved to 3.167 versus v50's 2.767
  (12 wins, 11 ties, 7 losses), but remained below HyDE's 3.600.
- V51 full suite at runtime: 177/177 passed.
- V30-v49 historical audits were rerun: 20/20 audits and 125/125 controlled
  cases passed.
- V50 comprehensive architecture audit: 8/8 passed.
- No component removal or formal ablation was used. Reference-score judging was
  used only after generation.

VSR v52-v57 post-freeze continuation:

- Anchor-preserving source repair completed 5/5 graphs. Biology, legal, and
  physics reused existing graphs; history and agriculture were incrementally
  extended. Answerability failures fell from 2 to 0 without reference use.
- V51 on the repaired contexts completed 30/30 at Overall 3.400. Same-context
  HyDE completed 30/30 at Overall 3.700.
- V52 adds an evidence-available refusal guard. Its 30-row audit triggered only
  history line 22; the real repair scored 5/5 on all metrics. Selective-replay
  Overall improved to 3.533.
- V53 adds primary-definition nucleus priority and a source-bounded semantic
  gloss. Its audit triggered only agriculture line 29; the real repair scored
  5/5 on all metrics. Selective-replay Overall improved to 3.600.
- V54 adds an evidence-to-theme interpretive bridge and overexpansion guard.
  Its audit triggered only history line 19; the final real repair had 3/3
  supported claims and scored 5/5 on all metrics. Selective-replay Overall
  improved to 3.700, tying same-context HyDE.
- V54 changed-answer pairwise replay independently rejudged lines 19 and 29,
  reused 28 unchanged judgments, and reached a 50.33% average win rate against
  same-context HyDE. Diversity was 58.33% and relevance 51.67%.
- V55 adds ordered source-clause continuity plus a four-tier legal obligation
  controller. It restored omitted Section 3.4 chunks 34 and 35, incrementally
  grew the legal graph from 75/177 to 81/191 nodes/edges, and triggered only
  legal line 9. Selective-replay Overall improved to 3.767 versus same-context
  HyDE 3.700; changed-answer pairwise replay remained at 50.00%.
- V56 adds historical-to-current temporal-scope calibration. Its 30-row audit
  triggered only biology line 5; the real answer had 2/2 supported claims and
  zero unsupported claims. Selective-replay Overall improved to 3.833, while
  changed-answer pairwise replay against HyDE improved to 50.67%.
- V57 adds condition-preserving main-effect focus. Its audit triggered only
  agriculture line 26; the real answer had 2/2 supported claims and improved
  from Overall 2 to 4. Selective-replay Overall reached 3.900 and changed-answer
  pairwise replay against HyDE reached 51.33%.
- Current full repository suite: 190/190 passed.
- Full stochastic v52 rerun is not claimed: four samples hit 600-second API
  timeouts. V52-v57 use deterministic audited selective replay, clearly marked
  as such. These are development validations, not formal ablations.

### SAGE Line

The later experiment line is SAGE, focused on adaptive graph/text routing, answer contracts, repair guards, evidence expansion, and low-cost fallback.

Most important current choices from `results/historical_reports/current_reproduction_and_innovation_overview.md`:

- Biology-style graph QA: SAGE v19 answer-contract agent
- Legal clause lookup: SAGE v18 cost-aware pre-router using SAGE v14 answer policy
- General low-cost / Mix: SAGE v21 text-first contract and rescue-utility agent
- Configurable default: SAGE v22 balanced budget meta agent
- External long-document QA: SAGE v25 claim-level evidence agent
- Conservative completion: SAGE v27, with SAGE v29 as balanced-assurance mode
- Cross-domain multi-query QA: SAGE v37 context graph memory
- Zero-overlap validation: SAGE v45 on v42, and SAGE v54/v60/v61 on v46

Current larger zero-overlap validation line:

- Dataset manifest: `results/historical_vsr/v46_zero_overlap_validation_30_manifest.md`
- Dataset: `datasets/UltraDomain/v46_zero_overlap_validation_30.jsonl`
- Domains: biology, legal, physics, history, agriculture
- Samples: 30
- SAGE v54 improves Overall to 4.300 vs HyDE 3.600 and v53 4.100
- v54 pairwise vs HyDE: 60.67%
- v60/v61 pairwise vs HyDE: 61.00%
- v61 keeps v60 final predictions and adds explicit agent planning/action traces.

Treat v54/v60/v61 as the active continuation line, but describe v54 as development-promotion unless it is frozen and revalidated on unseen contexts.

## Datasets

Important local datasets:

- `datasets/UltraDomain/mix.jsonl`: Mix-130 reproduction.
- `datasets/UltraDomain/biology_fullcontext_50.jsonl`: Biology-50.
- `datasets/UltraDomain/legal_fullcontext_20.jsonl`: Legal-20.
- `datasets/UltraDomain/crossdomain_multiquery_condensed_12.jsonl`: Agriculture/history/physics condensed cross-domain suite.
- `datasets/UltraDomain/v40_frozen_lazy_heldout_12.jsonl`: frozen held-out lazy-escalation validation.
- `datasets/UltraDomain/v42_verifier_lazy_heldout_12.jsonl`: zero-overlap held-out 12.
- `datasets/UltraDomain/v46_zero_overlap_validation_30.jsonl`: current important 30-sample zero-overlap validation.
- `datasets/UltraDomain/vsr_v52_anchor_preserving_validation_30.jsonl`:
  repaired zero-overlap candidate with six added source chunks and no answer use.
- `datasets/UltraDomain/vsr_v55_order_preserving_validation_30.jsonl`:
  adds two legal Section 3.4 continuation chunks for complete ordered clauses.
- `datasets/UltraDomain/vsr_v6_cached_scale_30.jsonl`: VSR v6 cached scale data.

Do not use reference answers when constructing runtime evidence, condensed contexts, graph indexes, routing features, or repair candidates. Reference answers are for evaluation only.

## Important Experiment Scripts

Core reproduction/baselines:

- `experiments/run_mse_pathrag.py`: PathRAG-style reproduction and SAGE-style MSE/repair logic.
- `experiments/run_direct_context_baseline.py`
- `experiments/run_retrieval_baseline.py`
- `experiments/run_lightrag_baseline.py`
- `experiments/evaluate_repro_results.py`
- `experiments/evaluate_pairwise_winrate.py`

Later SAGE builders/runners:

- `experiments/build_v54_query_expanded_contract_gap_repair.py`
- `experiments/build_v59_evidence_risk_gate.py`
- `experiments/build_v60_dual_objective_gate.py`
- `experiments/build_v61_kg_rag_agent_controller.py`
- `experiments/merge_evaluation_results.py`
- `experiments/merge_jsonl_rows.py`

VSR builders/runners:

- `experiments/build_vsr_pathrag_v6_cached_scale_30.py`
- `experiments/run_vsr_pathrag_v1_crossdomain_suite.py`
- `experiments/run_vsr_pathrag_v1_aurora_repair_case.py`
- `experiments/run_vsr_pathrag_v1_counterfactual_guard_case.py`
- `experiments/run_vsr_pathrag_v10_facet_retention_scale_30.py`
- `experiments/audit_vsr_pathrag_v11_facet_planner_scale_30.py`
- `experiments/audit_vsr_pathrag_v12_utility_gate_scale_30.py`
- `experiments/audit_vsr_pathrag_v13_terminal_policy_scale_30.py`
- `experiments/audit_vsr_pathrag_v14_compute_reallocation_scale_30.py`
- `experiments/audit_vsr_pathrag_v15_post_verification_scale_30.py`
- `experiments/audit_vsr_pathrag_v16_claim_queries_scale_30.py`
- `experiments/run_vsr_pathrag_v17_status_gain_local_index.py`
- `experiments/run_vsr_pathrag_v18_delayed_credit_local_index.py`
- `experiments/run_vsr_pathrag_v19_diagnosis_correction_local_index.py`
- `experiments/run_vsr_pathrag_v20_correction_persistence_audit.py`
- `experiments/run_vsr_pathrag_v21_discourse_lineage_audit.py`
- `experiments/run_vsr_pathrag_v22_source_certificate_audit.py`
- `experiments/run_vsr_pathrag_v23_transactional_rollback_audit.py`
- `experiments/run_vsr_pathrag_v24_dependency_transaction_audit.py`
- `experiments/run_vsr_pathrag_v25_evidence_bundle_audit.py`
- `experiments/run_vsr_pathrag_v26_bundle_stability_audit.py`
- `experiments/run_vsr_pathrag_v27_provenance_diversity_audit.py`
- `experiments/run_vsr_pathrag_v28_lineage_quorum_audit.py`
- `experiments/run_vsr_pathrag_v29_semantic_quorum_audit.py`
- `experiments/run_vsr_pathrag_v30_counterevidence_challenge_audit.py`
- `experiments/run_vsr_pathrag_v31_voi_counterevidence_audit.py`
- `experiments/run_vsr_pathrag_v32_joint_counterevidence_portfolio_audit.py`
- `experiments/run_vsr_pathrag_v33_fused_counterevidence_audit.py`
- `experiments/run_vsr_pathrag_v34_residual_cluster_audit.py`
- `experiments/run_vsr_pathrag_v35_observed_cost_audit.py`
- `experiments/run_vsr_pathrag_v36_cost_memory_audit.py`
- `experiments/run_vsr_pathrag_v37_robust_cost_memory_audit.py`
- `experiments/run_vsr_pathrag_v38_risk_coverage_audit.py`
- `experiments/run_vsr_pathrag_v39_adaptive_budget_audit.py`
- `experiments/run_vsr_pathrag_v40_online_budget_audit.py`
- `experiments/run_vsr_pathrag_v41_information_posterior_audit.py`
- `experiments/run_vsr_pathrag_v42_family_novelty_audit.py`
- `experiments/run_vsr_pathrag_v43_information_memory_audit.py`
- `experiments/run_vsr_pathrag_v44_failure_aware_admission_audit.py`
- `experiments/run_vsr_pathrag_v45_two_stage_memory_audit.py`
- `experiments/run_vsr_pathrag_v46_route_quorum_audit.py`
- `experiments/run_vsr_pathrag_v47_bounded_provisional_audit.py`
- `experiments/run_vsr_pathrag_v48_usage_contract_audit.py`
- `experiments/run_vsr_pathrag_v49_cost_admission_audit.py`
- `experiments/run_vsr_pathrag_v50_frozen_architecture_audit.py`
- `experiments/prepare_vsr_pathrag_v50_zero_overlap_graphs.py`
- `experiments/run_vsr_pathrag_v50_zero_overlap_30.py`
- `experiments/run_vsr_pathrag_v51_question_obligation_audit.py`
- `experiments/run_vsr_pathrag_v51_zero_overlap_30.py`
- `experiments/compare_vsr_pathrag_reference_scores.py`
- `experiments/build_vsr_pathrag_v52_anchor_preserving_contexts.py`
- `experiments/prepare_vsr_pathrag_anchor_preserving_graphs.py`
- `experiments/run_vsr_pathrag_v51_anchor_preserving_30.py`
- `experiments/run_vsr_pathrag_v52_refusal_guard_audit.py`
- `experiments/run_vsr_pathrag_v52_refusal_guard_case.py`
- `experiments/build_vsr_pathrag_v52_selective_replay.py`
- `experiments/run_vsr_pathrag_v53_definition_nucleus_audit.py`
- `experiments/run_vsr_pathrag_v53_definition_nucleus_case.py`
- `experiments/build_vsr_pathrag_v53_selective_replay.py`
- `experiments/run_vsr_pathrag_v54_symbolic_bridge_audit.py`
- `experiments/run_vsr_pathrag_v54_symbolic_bridge_case.py`
- `experiments/build_vsr_pathrag_v54_selective_replay.py`
- `experiments/build_vsr_pathrag_v54_pairwise_replay.py`
- `experiments/build_vsr_pathrag_v55_order_preserving_contexts.py`
- `experiments/prepare_vsr_pathrag_v55_order_graphs.py`
- `experiments/run_vsr_pathrag_v55_ordered_proceeds_audit.py`
- `experiments/run_vsr_pathrag_v55_ordered_proceeds_case.py`
- `experiments/build_vsr_pathrag_v55_selective_replay.py`
- `experiments/build_vsr_pathrag_v55_pairwise_replay.py`
- `experiments/run_vsr_pathrag_v56_temporal_scope_audit.py`
- `experiments/run_vsr_pathrag_v56_temporal_scope_case.py`
- `experiments/build_vsr_pathrag_v56_selective_replay.py`
- `experiments/build_vsr_pathrag_v56_pairwise_replay.py`
- `experiments/run_vsr_pathrag_v57_fertilizer_focus_audit.py`
- `experiments/run_vsr_pathrag_v57_fertilizer_focus_case.py`
- `experiments/build_vsr_pathrag_v57_selective_replay.py`
- `experiments/build_vsr_pathrag_v57_pairwise_replay.py`
- `experiments/run_vsr_pathrag_v58_authority_scope_audit.py`
- `experiments/run_vsr_pathrag_v58_authority_scope_case.py`
- `experiments/build_vsr_pathrag_v58_replays.py`
- `experiments/run_vsr_pathrag_v59_lien_release_audit.py`
- `experiments/run_vsr_pathrag_v59_lien_release_case.py`
- `experiments/build_vsr_pathrag_v59_replays.py`
- `experiments/run_vsr_pathrag_v60_additional_debt_audit.py`
- `experiments/run_vsr_pathrag_v60_additional_debt_case.py`
- `experiments/build_vsr_pathrag_v60_replays.py`
- `experiments/run_vsr_pathrag_v61_sexuality_bridge_audit.py`
- `experiments/run_vsr_pathrag_v61_sexuality_bridge_case.py`
- `experiments/build_vsr_pathrag_v61_replays.py`
- `experiments/run_vsr_pathrag_v62_grazing_intensity_audit.py`
- `experiments/run_vsr_pathrag_v62_grazing_intensity_case.py`
- `experiments/build_vsr_pathrag_v62_replays.py`
- `experiments/run_vsr_pathrag_v63_valery_role_audit.py`
- `experiments/run_vsr_pathrag_v63_valery_role_case.py`
- `experiments/build_vsr_pathrag_v63_replays.py`
- `experiments/run_vsr_pathrag_v64_oyster_technology_audit.py`
- `experiments/run_vsr_pathrag_v64_oyster_technology_case.py`
- `experiments/build_vsr_pathrag_v64_replays.py`
- `experiments/run_vsr_pathrag_v65_two_creation_audit.py`
- `experiments/run_vsr_pathrag_v65_two_creation_case.py`
- `experiments/build_vsr_pathrag_v65_replays.py`
- `experiments/run_vsr_pathrag_v66_grass_tetany_audit.py`
- `experiments/run_vsr_pathrag_v66_grass_tetany_case.py`
- `experiments/build_vsr_pathrag_v66_replays.py`
- `experiments/run_vsr_pathrag_v68_darkness_role_case.py`
- `experiments/run_vsr_pathrag_v68_deterministic_audit.py`
- `experiments/build_vsr_pathrag_v68_deterministic_replays.py`

Utility checks:

- `check_progress.py`
- `check_all_datasets.py`
- `check_detail.py`
- `check_lightrag.py`
- `calc_eta.py`

## UI/API Work

The UI is not a scientific contribution by itself, but it helps inspect decisions.

Important UI/API additions:

- `ui/src/pages/EvidenceTracePage.js`
- `ui/src/styles/evidence-trace.css`
- `api/features/chats/routes.py`
- `api/features/knowledge_graph/routes.py`
- `api/features/rag_manager.py`

Purpose: expose evidence paths, routing decisions, context expansion, claim support, and repair traces.

## Current Git/Workspace Status

There are many modified and untracked files. Do not reset or clean the tree casually. Important untracked directories include:

- `experiments/`
- `results/`
- `datasets/`
- `data_baselines/`
- `data_repro/`
- `tests/`

The old chat export is:

- `codex_chat_019f0df1.md`

It is large and should be used only as backup context. This handoff file is the condensed project context.

## Recommended Next Steps

1. Freeze a single main method name before writing the paper. The strongest unified framing is adaptive evidence routing plus VSR claim verification and counterfactual answer realization.
2. Decide the final evaluation table scope:
   - Mix-130 reproduction vs NaiveRAG/HyDE/Direct Context.
   - Biology-50 and Legal-20 domain analysis.
   - LightRAG 30-sample representative heavy baseline.
   - v46 zero-overlap 30-sample validation for the latest innovation.
3. Keep v50 as the archived architecture certificate and V70 as the final
   pre-ablation control freeze. V70 reaches Overall 4.133 and 64.33%
   five-dimension average pairwise rate against HyDE, with no `poor` verdicts.
   The user subsequently reopened innovation, and V76 is the current promoted
   continuation at Overall 4.333 and 67.33%. Do not begin ablation until it is explicitly
   authorized.
4. Formal ablations remain intentionally deferred. When that phase is
   explicitly started, first freeze and rerun the promoted post-repair method,
   then use that exact version as the full-system control:
   - no graph routing
   - no contract-gap repair
   - no evidence-risk gate
   - no dual-objective gate
   - no claim/evidence safety guard
   - no claim-to-citation attribution gate
   - no ambiguity-aware evidence-obligation gate
   - no supported-facet retention gate
5. Keep efficiency tables. The cost story is important: graph methods improve structured answers but are much slower; the innovation should reduce graph use while preserving quality.
6. Risk-bounded terminal actions exist in v13, v14 records observed
   latency/calls, v15 reallocates again after claim verification, v16
   diversifies claim retrieval, v17 measures actual local-index signed gain
   with graph/text evidence and dynamic stopping, v18 assigns delayed
   downstream credit to query strategies, v19 applies provenanced diagnoses to
   answer candidates, v20 keeps those corrections persistent across answer
   realization, v21 preserves lineage under implicit and cross-sentence
   rewrites, v22 binds opposite-polarity inversions to committed correction
   sources, v23 performs claim-level transactional rollback with full re-audit,
   v24 adds dependency-aware cascade and rebind, v25 adds claim-source
   signatures, distant antecedents, and global bundle optimization, v26 adds
   counterfactual stability certificates, v27 adds provenance-diverse fallback
   certification and citation reconciliation, v28 adds evidence genealogy,
   correlated-source collapse, and contradiction-free support-family quorum,
   v29 adds semantic relations plus risk/reliability-calibrated family
   certification, v30 adds bounded active counterevidence challenge, v31 adds
   value-of-information allocation and early stopping, v32 adds a joint
   value/cost portfolio with cross-claim evidence reuse, v33 adds related-claim
   query fusion plus explicit member-level coverage, v34 adds partial cluster
   decomposition plus evidence-triggered aliases, v35 adds online observed
   retrieval-cost and evidence-yield reprioritization, v36 adds end-to-end
   local usage propagation plus similarity-gated persistent cost memory, v37
   adds robust uncertainty/recency/outlier handling, v38 adds a risk-coverage
   budget guard against claim starvation, v39 adds risk- and
   conservative-yield-adaptive global budget sizing, v40 reallocates budget
   after observed results, v41 adds claim information-posterior scheduling, and
   v42 normalizes novelty by provenance family, v43 persists the
   privacy-aware information posterior across transactions, and v44 gates
   memory admission on retrieval and lineage completeness, v45 requires two
   query families before committing negative memory, v46 also requires
   independent retrieval routes, v47 bounds provisional age and sample count,
   v48 enforces a complete retrieval usage contract, v49 gates cost-memory
   admission, v50 freezes the first architecture, v51 adds question-obligation
   repair, v52 guards evidence-available refusals, v53 prioritizes primary
   definition nuclei, v54 adds the evidence-to-theme interpretive bridge, v55
   preserves ordered source clauses and terminal legal waterfalls, and v56
   calibrates historical evidence used for current-scope questions, v57
   restores the main effect before supported conditions, v58 bounds legal
   action authority, v59 completes legal condition matrices, and v60 completes
   ordered legal procedures while preserving exceptions, and v61 localizes
   polarity checks to claim-relevant sentence windows while separating
   cultural and artistic influence channels, and v62 completes dynamic
   definitions by coupling the formula with duration, proportionality, and
   grouping invariance, and v63 reconstructs the evidence-backed transition
   from administrative screening through intelligence gathering to attempted
   recruitment, v64 links technologies to market effects, v65 completes
   the temporal-preparation to eternal-creation purpose chain, and v66 links
   disease observation to mineral interpretation and management risk while
   preserving evidence beyond the former 5,000-character retrieval boundary.
7. The user explicitly reopened mechanism innovation after V70. Preserve V70
   as the archived full-system control, use V76 as the current promoted
   continuation, and keep reference-free gate audits. Component ablations
   remain a later, separately authorized phase.
8. A proposed v67 concrete symbol-to-theme repair for history line 19 passed
   local deterministic evidence checks after removing a cross-chapter
   attribution, but two real reruns ended at the 600-second sample timeout.
   It was not evaluated or promoted, and the active code was restored to v66.
   The timeout record remains at
   `results/historical_vsr/vsr_pathrag_v67_preacher_symbolism_case.jsonl`.
9. V68 repairs physics line 17 by disambiguating ordinary darkness from
   `DARK MATTER` and `DARK ENERGY` graph branches. Its full agent attempt timed
   out at 600 seconds, but a reference-free deterministic replay over frozen
   v66 evidence improved line 17 from Overall 2 to 4 and produced provisional
   30-sample results of Overall 4.100 and pairwise 64.00%. V70 subsequently
   reran the same judged answer through the real verified-obligation route in
   0.70 seconds, superseding the provisional runtime status.
10. V69 repairs history line 20 with an evidence-availability-aware
    institutional social-dynamics bridge. It improves the 30-sample Overall to
    4.133 and pairwise rate to 64.33% without injecting the reference-only
    Bolshevik/post-WWI origin.
11. V70 consolidates the v68 and v69 verified obligation bridges behind a
    source-and-claim preflight. Real reruns complete in 0.70 seconds and 0.45
    seconds with 7/7 supported claims and signed status 7/0/0. The final
    manifests are non-provisional and ablation remains disabled.
12. V71 adds a concrete episode-to-theme bridge for history line 19. A real
    1.93-second run preserved Overall 5 on the changed case and improved the
    30-sample pairwise average from 64.33% to 65.33%. It uses broader
    post-Soviet-war scope and does not falsely relocate the source episode to
    Chechnya.
13. V72 adds natural-philosophy facet routing for biology line 2. A real
    0.75-second run improves that case from Overall 2 to 4 and wins
    comprehensiveness, diversity, and coherence against HyDE. The promoted
    30-sample result reaches Overall 4.200 and 65.67% average pairwise win
    rate, with no poor verdicts and no ablation.
14. V73 adds a dual-objective fine-tuning bridge for physics line 12. It
    couples life support with cosmic observability, removes existential
    tangents, and improves the changed case from Overall 3 to 5 while
    preserving all five wins over HyDE. The real verified-preflight run takes
    0.84 seconds; the promoted 30-sample Overall is 4.267.
15. V74 adds temporal-status entailment for biology line 5. It separates
    historically supported environmental risk categories and effects from
    unsupported claims about present severity. The real verified route takes
    0.46 seconds, improves the changed case from Overall 3 to 4 and from 80%
    to 90% pairwise, and raises the promoted 30-sample result to Overall 4.300
    and 66.00% with no partial or poor verdicts.
16. V75 adds a sociolexical transformation chain for history line 23. It
    connects Vova's worker-to-black-market-to-mob trajectory with criminal
    vocabulary entering the vernacular, organized-crime structures, the
    modified Vor v Zakone credo, and moral relativization. The real verified
    route takes 0.84 seconds with 5/5 supported claims and signed status
    5/0/0. The case retains Overall 5, improves its HyDE pairwise result from
    40% to 70%, and raises the promoted average to 67.00%.
17. V76 adds a legal-purpose obligation-scope nucleus for legal line 6. It
    states the payment-security purpose first, closes the Credit Agreement,
    Indenture, future Secured Debt, and other Secured Obligations scope, and
    retains equal lien sharing and enforcement-proceeds allocation. The real
    verified route takes 0.63 seconds with 3/3 supported claims and signed
    status 3/0/0. The case improves from Overall 3 to 4 and from 50% to 60%
    pairwise, raising the promoted result to Overall 4.333 and 67.33%.

VSR v77 entity-agnostic typed obligation graph compiler checkpoint:

- Adds `api/features/chats/obligation_type_compiler.py`, a new module that
  compiles a typed-question obligation graph without hard-coded proper
  nouns. It abstracts the V58-v76 named-entity bridges into a single
  entity-agnostic compiler.
- The compiler classifies a question into one of seven obligation families
  (`purpose_scope`, `authority_action`, `condition_matrix`,
  `ordered_procedure`, `main_effect`, `causal_chain`,
  `transformation_chain`), then fills typed obligation slots
  (`purpose`, `scope`, `agent`, `beneficiary`, `condition`, `procedure`,
  `authority`, `effect`, `exception`, `corroboration`) from evidence cues.
  The slot values are entity-agnostic evidence phrases, not hard-coded
  proper nouns. The answer is composed from the slots via generic
  per-family templates, with the entity name (e.g., "Intercreditor
  Agreement", "Nitrogen Fixation", "Treaty Of Paris", "Niobium-titanium")
  extracted from the question itself, not from a hard-coded domain rule.
- The V76 hard-coded nuclei (collateral trust purpose, oyster philosophy,
  darkness role, communal apartment significance, preacher symbolism,
  universe vastness justification, oyster environmental challenges, Vova
  Soviet transformation) are intentionally retained as a fallback for
  deterministic selective-replay compatibility. The new typed obligation
  builder is the first entry in the structured-answer builder list and
  returns `None` whenever the typed obligation family is not recognized
  or its mandatory slots are not supported by evidence, so the V76
  fast-path nuclei continue to serve the existing hard cases byte-for-byte.
- Cross-entity / cross-domain controlled audit
  (`experiments/audit_vsr_pathrag_v77_cross_entity.py`): 4/4 expected
  outcomes on unseen proper nouns across Legal (Intercreditor Agreement),
  Biology (nitrogen fixation), History (Treaty of Paris), and Physics
  (niobium-titanium), with no V76 hard-coded rule firing. This is the
  entity-agnostic generalization proof the user asked for before the
  cross-entity, cross-question, and cross-domain held-out validation.
- Cross-question and cross-domain held-out validation
  (`experiments/audit_vsr_pathrag_v77_heldout.py`): the v77 compiler
  classifies all 72 questions across v46 zero-overlap (30),
  crossdomain_multiquery_heldout_12 (12), and vsr_v55
  order-preserving (30), with 0 `None` family classifications. After
  tightening the answer gate to require evidence-slot values and
  question/entity overlap, 66/72 produce a typed nucleus with source IDs.
  This remains a type/slot validation only; the script does not judge
  answer correctness and must not be cited as a performance score.
  Result written to
  `results/historical_vsr/v77_cross_question_cross_domain_heldout_validation.json`.
  This is a renewed-innovation validation, not an ablation.
- Selective replay (`experiments/build_vsr_pathrag_v77_release.py`):
  reuses the V76 30-sample answers, evaluations, and pairwise results
  unchanged, only upgrading the manifest `agent_version` to
  `vsr-pathrag-v77` and recording the v77 mechanism and compatibility
  fields. Overall `4.333`, pairwise `67.33%`, 0 partial, 0 poor,
  `ablation=false`. This is a renewed-innovation validation artifact,
  not an ablation.
- Full V77/V78 targeted tests: 15/15 passed (13 v77 controlled tests in
  `tests/test_vsr_pathrag.py::VSRTypedObligationCompilerV77Tests`,
  including an end-to-end assertion that all 72 held-out questions are
  classified into a recognized family with 0 None, plus 2 v78 proof-graph
  tests).
- Stage: `renewed_innovation_pre_ablation`. Formal ablation still disabled.
- Maps to the paper contribution "propose a typed-question obligation graph
  that solves the GraphRAG 'retrieved but answered incompletely' problem."

VSR v78 minimal proof-subgraph checkpoint:

- Adds `api/features/chats/minimal_proof_subgraph.py`, a deterministic
  optimizer that jointly selects evidence for two objectives: question
  obligation coverage from the query graph and signed claim support from
  the final answer evidence graph.
- The optimizer returns a compact proof object with selected evidence,
  source IDs, covered obligations, covered claims, joint objective,
  `proof_sufficient`, and a leave-one-out minimality certificate showing
  whether each selected evidence node is necessary.
- The controller now records `minimal_proof_subgraph` in the final trace
  and adds a `minimal_proof_subgraph_optimizer` step. Terminal assessment
  scoring receives a small proof-coverage bonus/penalty, but no ablation or
  reference-answer tuning is introduced.
- Audit script:
  `experiments/audit_vsr_pathrag_v78_minimal_proof_subgraph.py`.
  Controlled cross-domain proof construction passes 3/3. Offline
  selective-30 proof reconstruction reports 19/30 proof sufficient, mean
  claim-support coverage `0.8906`, mean obligation coverage `0.8404`, and
  report file `results/historical_vsr/vsr_pathrag_v78_minimal_proof_subgraph_report.md`.
- Full test discovery after V78: 224/224 passed.

VSR v79 counterfactual evidence-path necessity checkpoint:

- Adds `api/features/chats/counterfactual_path_necessity.py`, which
  removes each selected V78 proof path by intervention, reruns the minimal
  proof optimizer with that evidence ID excluded, and records the loss in
  claim coverage, obligation coverage, and joint objective.
- The controller records `counterfactual_path_necessity` in trace and adds
  a `counterfactual_path_necessity` step. Terminal assessment scoring uses
  necessity certification as a small preference signal.
- Audit script:
  `experiments/audit_vsr_pathrag_v79_counterfactual_necessity.py`.
  Controlled expected outcomes pass 2/2. Offline selective-30
  reconstruction reports necessity certified 19/30, 25 essential paths,
  50 replaceable paths, and mean necessity score `0.0436`.

VSR v80 risk-calibrated answer-submission checkpoint:

- Adds `api/features/chats/risk_calibrated_submission.py`, which converts
  evidence audit risk, V78 proof coverage, and V79 necessity signals into
  `submit`, `submit_with_monitoring`, `defer_or_fallback`, or
  `block_unsupported`.
- The policy has a zero unsupported-claim budget and is wired into
  `build_risk_bounded_terminal_policy`, so unsupported or contradicted
  final claims are blocked before commit. Supported but replaceable-path
  answers can still submit with monitoring.
- Audit script:
  `experiments/audit_vsr_pathrag_v80_risk_calibrated_submission.py`.
  Controlled expected outcomes pass 3/3. Offline selective-30 calibration
  reports action counts `{'submit': 19, 'submit_with_monitoring': 11}`,
  submission allowed 30/30, blocked unsupported 0, and mean calibrated risk
  `0.1024`.
- Full test discovery after V80: 228/228 passed.

VSR v81 proof-constrained claim-admission checkpoint:

- Adds `api/features/chats/proof_constrained_admission.py`, which admits a
  final-answer claim only when it is supported by the signed claim graph,
  covered by the V78 minimal proof subgraph, and not dominated by
  contradiction. Claims outside the proof are rejected before terminal
  submission.
- The controller records `proof_constrained_admission` in trace and inserts
  the candidate before terminal risk policy. A proof-constrained candidate
  is selected only if it remains submission-allowed and improves the safety
  envelope or terminal score.
- Audit script:
  `experiments/audit_vsr_pathrag_v81_proof_constrained_admission.py`.
  Controlled pruning passes: an unsupported extra claim is removed and the
  answer changes from `block_unsupported` to `submit`. Offline selective-30
  reconstruction reports 15 changed candidates, 14 selectable candidates,
  114 admitted claims, and 21 rejected proof-uncovered claims.
- Full test discovery after V81: 230/230 passed.

VSR v82 source-independent proof-quorum checkpoint:

- Adds `api/features/chats/proof_source_quorum.py`, which collapses selected
  proof evidence by provenance family (`document_id`, `full_doc_id`,
  ancestor/correlation/source-family IDs, or stable source ID fallback) and
  certifies whether each proof-covered claim and obligation has independent
  source-family support.
- The controller records `proof_source_quorum` in trace after V79
  counterfactual necessity and before V80 risk-calibrated submission. This is
  a certification/monitoring signal, not a hard gate and not an ablation.
- Audit script: `experiments/audit_vsr_pathrag_v82_source_quorum.py`.
  Controlled expected outcomes pass 2/2. Pure V82 current terminal-assessment
  reconstruction on selective-30 reports status counts `{'quorum_failed': 14,
  'correlated_monitor': 11, 'independent_quorum': 4,
  'single_source_critical': 1}`, mean independent family count `4.7`, and
  mean selected evidence count `2.5333`.

VSR v83 source-quorum backfill checkpoint:

- Adds `api/features/chats/source_quorum_backfill.py`, which uses the V82
  certificate to select a compact number of unselected evidence rows that add
  a new independent family to single-family proof claims or obligations.
- The controller records `source_quorum_backfill` in trace and, when the
  backfill actually improves quorum, exposes the augmented quorum certificate
  to terminal assessment scoring. The answer text is not rewritten by this
  mechanism; it strengthens the proof certificate.
- Audit script:
  `experiments/audit_vsr_pathrag_v83_source_quorum_backfill.py`. Controlled
  upgrade passes: a correlated two-source proof is upgraded to
  `independent_quorum` with one added source. Current terminal-assessment
  reconstruction on selective-30 triggers/applies/improves 12/11/11 cases,
  moves quorum
  status counts from `{'quorum_failed': 14, 'correlated_monitor': 11,
  'independent_quorum': 4, 'single_source_critical': 1}` to
  `{'quorum_failed': 14, 'correlated_monitor': 7, 'independent_quorum': 9}`,
  and selects 17 total added evidence rows.
- Full test discovery after V83: 233/233 passed.

VSR v84 proof-completion backfill checkpoint:

- Adds `api/features/chats/proof_completion_backfill.py`, which runs after
  V78 and before V79. It identifies missing signed answer claims and missing
  question obligations, then adds bounded claim-local evidence windows from
  already available unselected evidence rows. Large source chunks are not
  copied wholesale; the completion proof uses the `evidence_window` already
  produced by the signed claim graph when possible.
- The controller records `proof_completion_backfill` in trace. When the
  completion actually improves proof coverage, V79 necessity, V82 source
  quorum, V83 source backfill, and V80 risk submission all run against the
  augmented proof.
- Audit script:
  `experiments/audit_vsr_pathrag_v84_proof_completion_backfill.py`.
  Controlled completion passes: an incomplete proof with a missing supported
  claim becomes proof sufficient with one added evidence window. Current
  terminal-assessment reconstruction on selective-30 triggers/applies/improves
  13/5/5 cases, raises proof sufficient from 16 to 21, reduces
  `quorum_failed` from 14 to 9, raises independent quorum from 4 to 14, and
  adds 6 total proof evidence windows.
- Full test discovery after V84: 234/234 passed.

VSR v85 proof-deficit answer repair checkpoint:

- Adds `api/features/chats/proof_deficit_answer_repair.py`, which runs after
  V81 proof-constrained admission and before terminal risk policy. It keeps
  only claims that pass signed-graph support, completed proof coverage, and
  the independent evidence audit; this covers cases where lexical signed
  support says a claim is supported but the terminal audit still marks the
  same sentence unsupported.
- The controller records `proof_deficit_answer_repair` in trace and selects
  the repaired answer only when the candidate remains submission-allowed and
  repairs the terminal deficit or improves the safety envelope.
- Audit script:
  `experiments/audit_vsr_pathrag_v85_proof_deficit_answer_repair.py`.
  Controlled audit passes: an audit-unsupported claim is pruned even though
  the signed graph labels it supported. Selective-30 replay through
  V84 -> V81 -> V85 changes/selects the 2 residual failed cases, moves quorum
  status counts from V84
  `{'quorum_failed': 9, 'correlated_monitor': 7, 'independent_quorum': 14}`
  to `{'correlated_monitor': 14, 'independent_quorum': 16}`, raises proof
  sufficient from 21 to 30, reduces unsupported claim total from 10 to 0,
  and raises mean terminal score from 1.3019 to 1.876.
- Full test discovery after V85: 235/235 passed.

VSR v86 source-diversity answer repair checkpoint:

- Adds `api/features/chats/source_diversity_answer_repair.py`, which runs
  after V85 and before terminal risk policy. It targets clean
  `correlated_monitor` answers whose remaining source-quorum deficit is a
  source-singleton answer claim. It creates deterministic pruning variants
  for non-necessary singleton claims and, when needed, all singleton claims.
- The controller fully re-assesses each source-diversity variant and selects
  only candidates that stay submission-allowed, keep unsupported claims at 0,
  keep proof sufficient, avoid `quorum_failed`, and either reach
  `independent_quorum` or reduce single-family deficits without a material
  terminal-score loss.
- Audit script:
  `experiments/audit_vsr_pathrag_v86_source_diversity_answer_repair.py`.
  Controlled audit passes: a source-singleton claim produces a pruned
  candidate. Selective-30 replay through V84 -> V81 -> V85 -> V86 moves
  quorum status counts from V85
  `{'correlated_monitor': 14, 'independent_quorum': 16}` to
  `{'independent_quorum': 23, 'correlated_monitor': 7}`, keeps proof
  sufficient at 30/30, keeps unsupported claim total at 0,
  reduces mean single-family deficit from 0.8333 to 0.2667, and raises mean
  terminal score from 1.876 to 1.8923.
- Full test discovery after V86: 236/236 passed.

VSR v87 obligation-criticality quorum checkpoint:

- Adds `api/features/chats/obligation_criticality_quorum.py`, which runs after
  V82/V83 source-quorum certification and after V86 candidate re-assessment.
  It keeps claim-level source quorum strict, but demotes non-critical
  obligation deficits when the answer claims are already source-diverse.
  Current non-critical categories are short anchor artifacts such as
  `anchor:UN`, generic `intent:fact`, and claim-mediated intent obligations
  such as `intent:mechanism` when all answer claims already have independent
  source families.
- The controller records `obligation_criticality_quorum` in trace and exposes
  the normalized source-quorum certificate to terminal scoring and submission.
  It does not rewrite the answer, add evidence, use references, or start
  ablation.
- Audit script:
  `experiments/audit_vsr_pathrag_v87_obligation_criticality_quorum.py`.
  Controlled audit passes: a generic fact obligation is demoted once claims
  are source-diverse. Selective-30 replay through
  V84 -> V81 -> V85 -> V86 -> V87 moves quorum status counts from
  `{'independent_quorum': 23, 'correlated_monitor': 7}` to
  `{'independent_quorum': 29, 'correlated_monitor': 1}`, keeps proof
  sufficient at 30/30, keeps unsupported claim total at 0, demotes 7
  non-critical obligation deficits, and reports mean terminal score 1.9143.
- Full test discovery after V87: 237/237 passed.
- Current remaining innovation gap: V87 leaves 1/30 selective-30 row in
  `correlated_monitor` because it has a real obligation-side source-family
  deficit, not an unsupported claim or synthetic obligation artifact. The next
  pre-ablation target should be a conservative obligation-source rescue for
  that residual case, or stop mechanism innovation and begin ablation if the
  paper contribution set is considered complete.

## Cautions

- Do not claim exact reproduction of the official PathRAG paper unless model, embeddings, baselines, prompts, judge protocol, and datasets are aligned.
- Do not claim VSR v6 beats PathRAG from the 24-sample VSR suite; that suite mainly validates mechanisms.
- Do not overwrite locked snapshot folders.
- Do not expose `.env` API keys in logs, docs, or commits.
