# VSR-PathRAG: A Verification–Scoring–Repair Framework for Trustworthy Graph-Based Retrieval-Augmented Generation

## Target: IEEE TKDE / Information Processing & Management (SCI Q1/Q2)

---

## Abstract

Retrieval-Augmented Generation (RAG) systems that ground large language model (LLM) outputs in external knowledge suffer from two fundamental trust deficits: (1) *evidence insufficiency* — the retrieved evidence may not logically entail the generated answer, and (2) *claim hallucination* — unsupported or contradicted claims can survive into the final output. Existing GraphRAG approaches retrieve relevant subgraphs but lack a unified runtime contract for deciding whether the evidence is sufficient for the answer's obligations and claims. We propose **VSR-PathRAG**, an evidence-contract-guided adaptive RAG framework. The contract exposes obligations, claims, coverage, necessity, and submission risk; deficit-guided escalation then decides whether verification or graph rescue is worth its extra cost; controlled repair and abstention prevent unsupported changes from being submitted. MPS, CPN, and RCS are the internal selection, intervention, and decision primitives of this single pipeline rather than three unrelated contributions. The completed experiment package includes a seven-configuration UltraDomain Mix130 strict ablation, same-context retrieval baselines, GraphRAG-lite and LightRAG current-130 representative graph-RAG supplements, and 326 strict locked-holdout rows across unseen75 plus Repeat1-8. The frozen SAGE v33-preservation instantiation reaches weighted locked-holdout F1 0.3164 versus 0.2972 for both Naive RAG and HyDE, judge overall 3.555 versus 2.960/2.942, and pairwise target win rates of 59.66% versus HyDE and 61.50% versus Naive.

**Keywords**: Retrieval-Augmented Generation, Knowledge Graph, Evidence Verification, Counterfactual Reasoning, Risk Calibration

---

## 1. Introduction

### 1.1 Motivation

Large Language Models (LLMs) generate fluent but potentially ungrounded text. Retrieval-Augmented Generation (RAG) mitigates hallucination by conditioning generation on externally retrieved evidence. However, three critical gaps remain:

1. **No proof structure**: Standard RAG retrieves top-k passages and concatenates them into the prompt. There is no formal notion of whether the retrieved set *proves* the answer's claims. This is analogous to presenting evidence without making a legal case.

2. **No necessity analysis**: Even when evidence supports the answer, some evidence paths may be *replaceable* — removing them doesn't affect the answer's correctness. Without identifying *necessary* evidence, the system cannot detect fragile proofs or prioritize retrieval budget.

3. **No risk gating**: RAG systems submit answers regardless of evidence quality. When the evidence is insufficient or contradictory, the system should *abstain* or *repair* rather than fabricate.

### 1.2 Contributions

We propose VSR-PathRAG, which addresses these gaps through a unified Verification–Scoring–Repair pipeline:

1. **Evidence Contract** (Section 3): We model the question as obligations and the answer as atomic claims, then construct a compact proof subgraph whose coverage and necessity state can be consumed by later decisions. MPS is the contract-construction primitive, not a standalone contribution.

2. **Deficit-Guided Escalation** (Section 4): We use contract deficits and counterfactual necessity signals to decide whether to run verification or graph rescue. CPN is the intervention primitive for deciding whether an evidence path is replaceable or whether an escalation can safely substitute for it.

3. **Controlled Repair and Submission** (Section 5): We use risk-calibrated submission and deterministic preservation guards to retain supported content, accept bounded repairs, or abstain. RCS is the final decision primitive for the evidence contract.

4. **Strict Experiments** (Section 6): We report the complete UltraDomain Mix130 strict ablation, same-context retrieval-baseline comparisons, representative graph-RAG supplements, and locked zero-overlap holdouts with 326/326 complete method, judge, and pairwise rows.

### 1.3 Paper Organization

Section 2 formalizes the problem. Sections 3–5 present the three core components. Section 6 reports experiments. Section 7 discusses related work. Section 8 concludes.

---

## 2. Problem Formulation

### 2.1 Notation

| Symbol | Definition |
|--------|-----------|
| $q$ | User question |
| $\mathcal{G} = (V, E)$ | Knowledge graph with entities $V$ and relations $E$ |
| $\mathcal{E} = \{e_1, \ldots, e_N\}$ | Retrieved evidence set (entities + relationships) |
| $\mathcal{O} = \{o_1, \ldots, o_P\}$ | Question obligations (required answer facets) |
| $\mathcal{C} = \{c_1, \ldots, c_M\}$ | Atomic claims extracted from the generated answer |
| $\Phi: \mathcal{E} \times \mathcal{C} \to \{+1, -1, 0\}$ | Signed claim-evidence graph (+1=supports, -1=contradicts, 0=unknown) |
| $S \subseteq \mathcal{E}$ | Selected proof subgraph |
| $\text{cov}_{\mathcal{O}}(S)$ | Obligation coverage of $S$ |
| $\text{cov}_{\mathcal{C}}(S)$ | Claim support coverage of $S$ |

### 2.2 Formal Problem

**Definition 1 (Answer Trustworthiness)**. An answer $a$ with claims $\mathcal{C}$ is *trustworthy* with respect to evidence $\mathcal{E}$ if:

$$\forall c_i \in \mathcal{C}: \exists e_j \in \mathcal{E} \text{ s.t. } \Phi(e_j, c_i) = +1 \text{ and } \nexists e_k \in \mathcal{E} \text{ s.t. } \Phi(e_k, c_i) = -1$$

(i.e., every claim is supported and no claim is contradicted.)

**Definition 2 (Proof Subgraph)**. A subset $S \subseteq \mathcal{E}$ is a *proof subgraph* for answer $a$ if:

$$\text{cov}_{\mathcal{O}}(S) \geq \tau_O \quad \text{and} \quad \text{cov}_{\mathcal{C}}(S) \geq \tau_C$$

where $\tau_O, \tau_C$ are coverage thresholds. $S$ is *minimal* if no proper subset $S' \subsetneq S$ satisfies both constraints.

**Definition 3 (Counterfactual Necessity)**. Evidence $e_j \in S$ is *necessary* if removing it causes either $\text{cov}_{\mathcal{C}}(S \setminus \{e_j\}) < \tau_C$ or $\text{cov}_{\mathcal{O}}(S \setminus \{e_j\}) < \tau_O$.

**Definition 4 (Calibrated Risk)**. The calibrated risk of submitting answer $a$ is:

$$R(a, S) = w_1 \cdot r_{\text{audit}} + w_2 \cdot r_{\text{unsupported}} + w_3 \cdot r_{\text{proof\_gap}} + w_4 \cdot r_{\text{necessity}} + w_5 \cdot r_{\text{contradiction}}$$

where each $r_i$ is a normalized risk signal $\in [0, 1]$ and $\sum w_i = 1$.

**Problem Statement**: Given $q$, $\mathcal{G}$, and a generated answer $a$, find:
1. A minimal proof subgraph $S^*$ that maximizes $\text{cov}_{\mathcal{O}}(S) + \lambda \cdot \text{cov}_{\mathcal{C}}(S)$ subject to $|S| \leq K$ and $\text{tokens}(S) \leq B$;
2. The set of necessary evidence paths $N \subseteq S^*$;
3. A submission decision $d^* \in \{\text{submit}, \text{submit\_with\_monitoring}, \text{block}, \text{abstain}\}$ that minimizes expected untrustworthiness.

---

## 3. Evidence Contract Construction (MPS)

### 3.1 Algorithm

The MPS optimizer solves a **Bi-objective Weighted Set Cover** problem:

$$\max_{S \subseteq \mathcal{E}} \quad \alpha \cdot \text{cov}_{\mathcal{O}}(S) + (1-\alpha) \cdot \text{cov}_{\mathcal{C}}(S)$$
$$\text{s.t.} \quad |S| \leq K, \quad \text{tokens}(S) \leq B$$

We use a greedy algorithm that at each step selects the evidence node with the highest *marginal utility*:

$$e^* = \arg\max_{e \in \mathcal{E} \setminus S} \frac{\Delta \text{cov}_{\mathcal{O}}(e | S) + \lambda \cdot \Delta \text{cov}_{\mathcal{C}}(e | S)}{\text{tokens}(e)}$$

### 3.2 Theoretical Guarantee

**Theorem 1 (Approximation Ratio)**. Let $n = \max(P, M)$ where $P$ is the number of obligations and $M$ is the number of claims. The greedy algorithm achieves an approximation ratio of $O(\ln n)$ for the bi-objective weighted set cover problem.

*Proof sketch*. The bi-objective function $f(S) = \alpha \cdot \text{cov}_{\mathcal{O}}(S) + (1-\alpha) \cdot \text{cov}_{\mathcal{C}}(S)$ is monotone submodular. By the classical result of Nemhauser, Wolsey, and Fisher (1978), the greedy algorithm for monotone submodular maximization under cardinality constraint achieves a $(1 - 1/e)$-approximation. For the weighted set cover variant, the approximation ratio is $H_n = \sum_{i=1}^n 1/i = O(\ln n)$, where $H_n$ is the $n$-th harmonic number. □

### 3.3 Minimality Certificate

After greedy selection, we perform a **leave-one-out redundancy check**: for each $e_j \in S$, if $S \setminus \{e_j\}$ still satisfies both coverage constraints, $e_j$ is pruned. This yields a *minimality certificate* — a verifiable proof that each selected evidence node is individually necessary for coverage.

---

## 4. Deficit-Guided Escalation (CPN)

### 4.1 Intervention Framework

For each $e_j \in S^*$, we perform a *counterfactual intervention*: remove $e_j$ and re-run the MPS optimizer on $\mathcal{E} \setminus \{e_j\}$. Let $S_{-j}$ denote the resulting subgraph. We measure:

- **Claim loss**: $\Delta_C(e_j) = \text{cov}_{\mathcal{C}}(S^*) - \text{cov}_{\mathcal{C}}(S_{-j})$
- **Obligation loss**: $\Delta_O(e_j) = \text{cov}_{\mathcal{O}}(S^*) - \text{cov}_{\mathcal{O}}(S_{-j})$
- **Sufficiency loss**: $\mathbb{1}[\text{proof\_sufficient}(S^*) \land \neg\text{proof\_sufficient}(S_{-j})]$
- **Objective loss**: $\Delta_f(e_j) = f(S^*) - f(S_{-j})$

### 4.2 Classification

Based on the loss patterns, each evidence path is classified:

| Category | Condition | Meaning |
|----------|-----------|---------|
| `claim_essential` | $\Delta_C > 0$ | Removing causes claim coverage drop |
| `sufficiency_critical` | Sufficiency loss = 1 | Removing breaks proof sufficiency |
| `obligation_critical` | $\Delta_O > 0$ | Removing causes obligation coverage drop |
| `efficiency_critical` | $\Delta_f \geq 0.05$ | Removing reduces objective but coverage maintained |
| `replaceable` | All losses negligible | Evidence is redundant |

### 4.3 Theoretical Guarantee

**Theorem 2 (Necessity Certification Correctness)**. Let $\epsilon$ be the coverage tolerance. The CPN certification has a false-positive rate (classifying replaceable evidence as necessary) of at most $\epsilon$, and a false-negative rate (classifying necessary evidence as replaceable) of 0.

*Proof sketch*. The false-negative rate is 0 because the leave-one-out intervention exactly tests the necessity condition from Definition 3. The false-positive rate arises only when the re-optimization finds an alternative evidence path that provides equivalent coverage within tolerance $\epsilon$. □

---

## 5. Controlled Repair and Submission (RCS)

### 5.1 Risk Score Formulation

We define the calibrated risk as a convex combination of five signals:

$$R(a, S) = \sum_{i=1}^{5} w_i \cdot r_i$$

where:
- $r_1 = \text{audit\_prior}(\text{risk\_level})$ — prior risk from evidence audit
- $r_2 = \frac{\text{unsupported\_claims}}{\text{total\_claims}}$ — unsupported claim ratio
- $r_3 = 1 - (0.62 \cdot \text{cov}_{\mathcal{C}}(S) + 0.38 \cdot \text{cov}_{\mathcal{O}}(S))$ — proof gap
- $r_4 = 0.18 \cdot \mathbb{1}[\neg\text{necessity\_certified}]$ — necessity gap
- $r_5 = \min(1, 0.55 \cdot \text{hard\_contradicted\_count})$ — contradiction risk

### 5.2 Submission Thresholds

| Condition | Action | Risk Range |
|-----------|--------|------------|
| $R \leq 0.22$, $\text{cov}_{\mathcal{C}} \geq 0.95$, $\text{cov}_{\mathcal{O}} \geq 0.66$ | `submit` | Low risk |
| $R \leq 0.42$, $\text{cov}_{\mathcal{C}} \geq 0.80$ | `submit_with_monitoring` | Medium risk |
| Unsupported claims > 0 or contradicted claims > 0 | `block_unsupported` | High risk |
| Otherwise | `defer_or_fallback` | Uncertain |

### 5.3 PAC-Bayesian Generalization Bound

**Theorem 3 (Generalization Bound)**. Let $\mathcal{D}$ be the distribution over question-evidence pairs, $m$ the sample size, and $\delta \in (0, 1)$. With probability at least $1 - \delta$ over the draw of $m$ i.i.d. samples, for any submission policy $\pi$ in the policy class $\Pi$:

$$\mathbb{E}_{(q, \mathcal{E}) \sim \mathcal{D}}[\text{Untrust}(\pi(q, \mathcal{E}))] \leq \hat{R}_\text{emp}(\pi) + \sqrt{\frac{\text{KL}(\pi \| \pi_0) + \ln(2m/\delta)}{m}}$$

where $\hat{R}_\text{emp}(\pi)$ is the empirical risk on the training set, $\text{KL}(\pi \| \pi_0)$ is the KL divergence from the prior policy $\pi_0$, and $\text{Untrust}(\cdot)$ measures answer untrustworthiness.

*Proof sketch*. This follows from the McAllester PAC-Bayesian theorem (1999), applied to the policy class $\Pi$ of threshold-based submission rules. The calibrated risk score provides a computable proxy for $\hat{R}_\text{emp}$. □

---

## 6. Experimental Design

### 6.1 Datasets

| Surface | Questions | Type | Role |
|---------|----------:|------|------|
| **UltraDomain Mix130** | 130 | Mixed-domain aligned set | Strict VSR ablation and candidate replay |
| **Legal/Biology/Domain external slices** | 120 | Domain slices | Retrieval-baseline and semantic judge comparison |
| **Locked unseen75** | 75 | Zero-overlap holdout | Primary post-freeze holdout |
| **Locked Repeat1-8** | 251 | Zero-overlap repeats/stress slices | Robustness and strict-capacity exhaustion evidence |

### 6.2 Baselines

| Method | Description |
|--------|-------------|
| **Naive RAG** | Top-k BM25 + LLM generation |
| **HyDE** | Hypothetical Document Embedding + retrieval |
| **GraphRAG-lite** | Lightweight community graph-RAG comparator across Mix130 and external slices |
| **LightRAG current-130** | Evaluated representative heavy graph-RAG comparator |
| **VSR full_v2_rescue** | Verification/repair/rescue mechanism layer |
| **SAGE v33-preservation** | Frozen lead instantiation for locked holdouts |

### 6.3 Metrics

1. **Lexical overlap**: F1 and ROUGE-L against reference answers
2. **Faithfulness**: Claim-level grounding proxy used in the strict VSR ablation
3. **Semantic judge**: Same-rubric LLM judge scores for correctness, coverage, relevance, concision, and overall quality
4. **Pairwise Win Rate**: A/B judge across comprehensiveness, diversity, logicality, relevance, and coherence
5. **Runtime / cost proxy**: Seconds per question, LLM call counts, and token samples where instrumented
6. **Coverage status**: Method, judge, and pairwise completion counts for every locked holdout

### 6.4 Ablation Study

| Configuration | Description |
|---------------|-------------|
| **w/o_all** | Baseline aligned VSR shell without major verification/repair modules |
| **w/o_evidence_extraction** | Remove evidence extraction while preserving later stages |
| **w/o_verification** | Remove verification stage |
| **w/o_repair** | Remove answer repair |
| **full** | Original full VSR pipeline |
| **full_v2** | Improved full VSR configuration |
| **full_v2_rescue** | Full v2 with abstention-triggered rescue |

### 6.5 Statistical Testing

- Use paired tests on matched rows for ablation and holdout comparisons
- Report t-test / Wilcoxon p-values and bootstrap confidence intervals where generated
- Keep lexical metrics, same-rubric judge metrics, and pairwise A/B preference as separate evidence families
- Do not tune on locked holdouts after SAGE v33-preservation is frozen

### 6.6 Evidence-contract mechanism audit

The completed offline audit on frozen SAGE v33-preservation versus `full_v2_rescue` tests whether the current contract deficit is already an escalation-utility estimator. It is not a randomized treatment/control experiment, so the results are reported as mechanism observations rather than causal effects.

| Audit quantity | Result | Interpretation |
|---|---:|---|
| Aligned rows | 130 | Complete Mix130 alignment |
| Mean SAGE-vs-VSR F1 delta | +0.1768 | Large candidate gap, but not itself routing evidence |
| Router probability vs F1-delta correlation | -0.3782 | Higher predicted risk does not imply larger upgrade benefit |
| Contract-gap vs F1-delta correlation | -0.1511 | Simple deficit is a weak benefit proxy |
| Route selected / skipped mean F1 delta | +0.1119 / +0.2089 | Selected rows are harder, but upgrades help less on average |

The resulting scientific question is narrower and more testable: *which rows are worth paying for escalation?* The next experiment should freeze an impact-aware escalation score trained or calibrated only from historical development outcomes, then evaluate it on new source data or a newly locked holdout against risk-only routing, always-escalate, and never-escalate. The report should include answer quality, extra cost, realized upgrade gain, and wrong-escalation rate.

As a retrospective feasibility check, a cross-fitted impact-aware top-k prototype on the same Mix130 artifacts improved the matched-budget historical gain proxy from $+0.0299$ to $+0.0489$ at 32 selected rows and reduced selected negative-delta rows from 7 to 1. This result is deliberately treated as hypothesis-generating because the observed SAGE-vs-VSR delta is not a randomized escalation effect.

The budget-curve audit sharpens the mechanism further: the current risk-probability ranking has AUUC proxy $0.0781$, below a random ranking at $0.0884$, whereas contract-gap ranking reaches $0.0842$ and impact-aware ranking reaches $0.0975$. The central failure mode is therefore budget misallocation: detecting difficult rows is not equivalent to selecting rows that benefit from intervention.

---

## 7. Completed Results

### 7.1 Strict VSR Mix130 Ablation

- All seven configurations are complete at 130/130 rows.
- `full_v2_rescue` reaches F1 0.2878 and ROUGE-L 0.2194, versus `w/o_repair` F1 0.2206 and ROUGE-L 0.1591.
- The rescue branch reduces abstention to 0.8% and preserves high faithfulness at 0.9756.

### 7.2 Locked Holdout Meta Summary

- Locked unseen75 plus Repeat1-8 provide 326/326 complete method, judge, and pairwise rows.
- Weighted lexical F1 is SAGE v33 0.3164 versus Naive 0.2972 and HyDE 0.2972.
- Weighted judge overall is SAGE v33 3.555 versus Naive 2.960 and HyDE 2.942.
- Average pairwise SAGE target win rates are 59.66% versus HyDE and 61.50% versus Naive.

### 7.3 Stress and Caveat Evidence

- Repeat8 Legal54 exhausts remaining strict legal holdout capacity and is reported as stress evidence.
- On Repeat8, SAGE v33 ties HyDE lexically at F1 0.2862 and trails Naive by -0.0097 F1, while judge overall remains slightly above HyDE and below Naive.
- The main claim should emphasize semantic quality and preference under strict holdouts, not blanket lexical-F1 dominance.

### 7.4 Generated Tables

- Markdown/JSON package: `results/paper_tables/`
- Include-ready LaTeX fragments: `paper/generated_experiment_tables.tex`
- Reproducibility manifest: `results/reproducibility/REPRODUCIBILITY_MANIFEST.md`

---

## 8. Implementation Notes

### 8.1 Code Structure

```
PathRAG/
├── api/features/chats/
│   ├── vsr_pathrag.py          # Core VSR engine (1900+ lines)
│   ├── agent_controller.py      # Pipeline orchestrator (12000+ lines)
│   ├── minimal_proof_subgraph.py     # V78: MPS optimizer
│   ├── counterfactual_path_necessity.py  # V79: CPN certification
│   ├── risk_calibrated_submission.py    # V80: RCS policy
│   ├── proof_constrained_admission.py   # V81: Proof-gated admission
│   └── ...
├── experiments/                 # 261 experiment scripts
└── results/                     # 130 JSON result files
```

### 8.2 Reproducibility

- All experiments use deterministic hashing (no random seed ambiguity)
- LLM responses are cached in `kv_store_llm_response_cache.json`
- Graph builds are versioned and stored in `data_smoke/`

---

## 9. Related Work

### 9.1 Graph-based RAG
- **GraphRAG** (Edge et al., 2024): Community summarization + global retrieval
- **LightRAG** (Guo et al., 2024): Dual-level retrieval (entity + relationship)
- **HippoRAG** (Gutiérrez et al., 2024): Personalized PageRank for multi-hop

### 9.2 Evidence Verification in RAG
- **Self-RAG** (Asai et al., 2023): Self-reflection tokens for retrieval gating
- **RAGAS** (Es et al., 2023): Automated evaluation framework
- **ARES** (Yan et al., 2024): Automated RAG evaluation system

### 9.3 Proof-based Verification
- **EntailmentBank** (Dalvi et al., 2021): Multi-hop entailment trees
- **FOLIO** (Han et al., 2022): First-order logic reasoning dataset
- **ProofWriter** (Tafjord et al., 2021): Natural language theorem proving

### 9.4 Positioning
VSR-PathRAG is the first to combine:
1. Formal proof structure (minimal subgraph with coverage guarantee)
2. Counterfactual necessity analysis (leave-one-out intervention)
3. Risk-calibrated submission (PAC-Bayesian gate)

---

## 10. Limitations and Future Work

1. **LLM dependency**: Claim extraction and evidence signing rely on LLM quality
2. **Scalability**: CPN's $O(|S| \times \text{MPS})$ complexity limits large $|S|$
3. **Domain generalization**: Current locked evidence is strong within UltraDomain-derived domain slices; broader standard-benchmark validation remains future work
4. **Baseline comparability**: GraphRAG-lite and LightRAG current-130 are completed representative baselines, but full official graph-baseline reruns remain approximate or budget-dependent
5. **Metric tension**: SAGE v33 is strongest on semantic judge and pairwise preference, while lexical gains over Naive/HyDE are modest and stress-slice dependent
6. **Theory**: The PAC-Bayesian bound assumes i.i.d. sampling, which may not hold for conversational RAG

---

## References

[To be completed with 40-60 references]

---

## Execution Checklist

- [x] Run UltraDomain Mix130 strict ablation across seven configurations
- [x] Implement and evaluate Naive RAG / HyDE aligned baselines on external and locked slices
- [x] Run GraphRAG-lite representative baseline on Mix130 plus Legal/Biology/Domain
- [x] Run LightRAG current-130 representative heavy baseline with judge and pairwise evaluation
- [x] Freeze SAGE v33-preservation before locked holdout evaluation
- [x] Complete locked unseen75 plus Repeat1-8 method, judge, and pairwise evaluation (326/326 rows)
- [x] Run overlap audit across locked holdouts
- [x] Generate paper-ready Markdown/JSON and LaTeX table fragments
- [x] Refresh reproducibility manifest after Repeat8 completion
- [ ] Write final paper narrative around the frozen results and caveats
- [ ] Add optional new-source or official-baseline experiments only if scope deliberately expands
