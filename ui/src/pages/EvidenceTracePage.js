import React, { useEffect, useMemo, useState } from 'react';
import { Button, ButtonGroup, Loader, Message, SelectPicker, Tag } from 'rsuite';
import Layout from '../components/Layout';
import { knowledgeGraphAPI } from '../services/api';
import '../styles/evidence-trace.css';

const DEFAULT_TRACE_FILE = 'mse_pathrag_sage_v8_uncertainty_hyde_biology_20.jsonl';
const ROUTER_TRACE_FILES = {
  v22: {
    economy: 'mse_pathrag_sage_v22_budget_meta_economy_biology_20.jsonl',
    balanced: 'mse_pathrag_sage_v22_budget_meta_balanced_biology_20.jsonl',
    quality: 'mse_pathrag_sage_v22_budget_meta_quality_biology_20.jsonl',
  },
  v23: {
    economy: 'mse_pathrag_sage_v23_hybrid_dual_gate_economy_domain50.jsonl',
    balanced: 'mse_pathrag_sage_v23_hybrid_dual_gate_balanced_domain50.jsonl',
    quality: 'mse_pathrag_sage_v23_hybrid_dual_gate_quality_domain50.jsonl',
  },
  v24: {
    economy: 'mse_pathrag_sage_v24_context_budget_economy_domain50.jsonl',
    balanced: 'mse_pathrag_sage_v24_context_budget_balanced_domain50.jsonl',
    quality: 'mse_pathrag_sage_v24_context_budget_quality_domain50.jsonl',
  },
  v25: {
    balanced: 'mse_pathrag_sage_v25_claim_evidence_agent_domain50.jsonl',
  },
  v27: {
    balanced: 'mse_pathrag_sage_v27_conservative_contract_completion_domain50.jsonl',
  },
  v28: {
    balanced: 'mse_pathrag_sage_v28_postrepair_consensus_v2_domain50.jsonl',
  },
  v29: {
    balanced: 'mse_pathrag_sage_v29_selective_consensus_domain50.jsonl',
  },
  v30: {
    domain: 'mse_pathrag_sage_v30_learned_contract_gap_domain50.jsonl',
    biology: 'mse_pathrag_sage_v30_learned_contract_gap_biology_validation30.jsonl',
  },
  v31: {
    domain: 'mse_pathrag_sage_v31_learned_contract_gap_domain50.jsonl',
    'bio-val': 'mse_pathrag_sage_v31_learned_contract_gap_biology_validation30.jsonl',
    'bio-dev': 'mse_pathrag_sage_v31_learned_contract_gap_biology_dev20.jsonl',
    mix: 'mse_pathrag_sage_v31_learned_contract_gap_mix130.jsonl',
  },
  v32: {
    domain: 'mse_pathrag_sage_v32_evidence_delta_domain50.jsonl',
    'bio-val': 'mse_pathrag_sage_v32_evidence_delta_biology_validation30.jsonl',
    'bio-dev': 'mse_pathrag_sage_v32_evidence_delta_biology_dev20.jsonl',
    mix: 'mse_pathrag_sage_v32_evidence_delta_mix130.jsonl',
  },
  v33: {
    heldout: 'mse_pathrag_sage_v33_frozen_evidence_delta_mix_unseen34_trace.jsonl',
  },
  v35: {
    arbiter: 'mse_pathrag_sage_v35_candidate_conditioned_arbiter_crossdomain12.jsonl',
  },
  v37: {
    memory: 'mse_pathrag_sage_v37_context_graph_memory_crossdomain12.jsonl',
  },
  v38: {
    heldout: 'mse_pathrag_sage_v38_frozen_context_graph_memory_heldout12.jsonl',
  },
  v39: {
    dev: 'mse_pathrag_sage_v39_evidence_calibrated_lazy_dev12.jsonl',
    heldout: 'mse_pathrag_sage_v39_frozen_lazy_v40_heldout12.jsonl',
  },
  v41: {
    dev: 'mse_pathrag_sage_v41_verifier_lazy_dev12.jsonl',
    heldout: 'mse_pathrag_sage_v41_verifier_lazy_v40_heldout12.jsonl',
  },
  v42: {
    pregraph: 'mse_pathrag_sage_v41_verifier_lazy_v42_pregraph_only12.jsonl',
  },
  v43: {
    expanded: 'mse_pathrag_sage_v43_text_expansion_v42_12.jsonl',
  },
  v44: {
    mismatch: 'mse_pathrag_sage_v44_evidence_mismatch_v42_12.jsonl',
  },
  v45: {
    compact: 'mse_pathrag_sage_v45_conservative_compression_v42_12.jsonl',
  },
  v47: {
    heldout30: 'mse_pathrag_sage_v47_question_anchored_v46_30.jsonl',
  },
  v48: {
    guarded: 'mse_pathrag_sage_v48_question_chunk_completion_guarded_v46_30.jsonl',
  },
  v49: {
    compression: 'mse_pathrag_sage_v49_fact_preserving_completion_compression_v46_30.jsonl',
  },
  v53: {
    refusal: 'mse_pathrag_sage_v53_substance_refusal_repair_v46_30.jsonl',
  },
  v54: {
    contract: 'mse_pathrag_sage_v54_contract_gap_repair_v46_30.jsonl',
  },
  v55: {
    heldout30: 'mse_pathrag_sage_v55_supported_facet_guard_v46_30.jsonl',
    'v40-frozen12': 'mse_pathrag_sage_v55_supported_facet_guard_v40_12.jsonl',
    'v42-frozen12': 'mse_pathrag_sage_v55_supported_facet_guard_v42_12.jsonl',
  },
  v56: {
    heldout30: 'mse_pathrag_sage_v56_supported_facet_fusion_v46_30.jsonl',
    'v40-frozen12': 'mse_pathrag_sage_v56_supported_facet_fusion_v40_12.jsonl',
    'v42-frozen12': 'mse_pathrag_sage_v56_supported_facet_fusion_v42_12.jsonl',
  },
  v57: {
    heldout30: 'mse_pathrag_sage_v57_facet_balanced_compaction_v46_30.jsonl',
    'v40-frozen12': 'mse_pathrag_sage_v57_facet_balanced_compaction_v40_12.jsonl',
    'v42-frozen12': 'mse_pathrag_sage_v57_facet_balanced_compaction_v42_12.jsonl',
  },
  v58: {
    heldout30: 'mse_pathrag_sage_v58_residual_hardcase_editor_v46_30.jsonl',
  },
  v59: {
    heldout30: 'mse_pathrag_sage_v59_evidence_risk_gate_v46_30.jsonl',
  },
  v60: {
    heldout30: 'mse_pathrag_sage_v60_dual_objective_gate_v46_30.jsonl',
  },
  v61: {
    heldout30: 'mse_pathrag_sage_v61_kg_rag_agent_controller_v46_30.jsonl',
  },
};

const formatNumber = (value, digits = 3) => {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return '0';
  }
  return Number(value).toFixed(digits);
};

const scorePercent = (edge) => {
  const score = edge?.score_breakdown?.final_score || 0;
  return `${Math.max(4, Math.min(100, score * 100))}%`;
};

const average = (items, selector) => {
  if (!items.length) return 0;
  return items.reduce((sum, item) => sum + Number(selector(item) || 0), 0) / items.length;
};

const EvidenceEdge = ({ edge, muted = false }) => (
  <div className={`trace-edge ${muted ? 'trace-edge-muted' : ''}`}>
    <div className="trace-edge-head">
      <span className="trace-edge-path">{edge.path}</span>
      <span className={`trace-role ${edge.evidence_role === 'repaired' ? 'repaired' : 'primary'}`}>
        {edge.evidence_role || 'primary'}
      </span>
      <span className={`trace-prompt-flag ${edge.used_in_prompt === false ? 'muted' : ''}`}>
        {edge.used_in_prompt === false ? 'trace only' : 'in prompt'}
      </span>
      <span className="trace-rank">#{edge.rank}</span>
    </div>
    <div className="trace-score-row">
      <div className="trace-score-bar">
        <div className="trace-score-fill" style={{ width: scorePercent(edge) }} />
      </div>
      <span>{formatNumber(edge?.score_breakdown?.final_score, 4)}</span>
    </div>
    <div className="trace-score-grid">
      <span>lex {formatNumber(edge?.score_breakdown?.lexical_score, 3)}</span>
      <span>rel {formatNumber(edge?.score_breakdown?.relation_support, 3)}</span>
      <span>pen {formatNumber(edge?.score_breakdown?.counterfactual_penalty, 3)}</span>
    </div>
    {edge.keywords && <div className="trace-keywords">{edge.keywords}</div>}
    {edge.description && <p className="trace-description">{edge.description}</p>}
    {edge.matched_query_terms?.length > 0 && (
      <div className="trace-term-row">
        {edge.matched_query_terms.slice(0, 10).map((term) => (
          <Tag key={term} size="sm">{term}</Tag>
        ))}
      </div>
    )}
    {edge.repair_new_terms?.length > 0 && (
      <div className="trace-term-row">
        {edge.repair_new_terms.map((term) => (
          <Tag key={term} size="sm" color="green">{term}</Tag>
        ))}
      </div>
    )}
  </div>
);

const ClaimEvidence = ({ agent }) => {
  const claims = agent?.expanded_claim_traces || agent?.claim_traces || [];
  if (!agent?.enabled) return null;
  return (
    <section className="trace-panel">
      <div className="trace-section-title">Claim Evidence Audit</div>
      <div className="claim-audit-summary">
        <div><span>Mean support</span><strong>{formatNumber(agent.mean_support, 3)}</strong></div>
        <div><span>Supported</span><strong>{agent.supported_claims || 0}</strong></div>
        <div><span>Partial</span><strong>{agent.partial_claims || 0}</strong></div>
        <div><span>Unsupported</span><strong>{agent.unsupported_claims || 0}</strong></div>
      </div>
      {agent.context_expansion?.enabled && (
        <div className="context-expansion-banner">
          <strong>Full-context expansion</strong>
          <span>
            {agent.context_expansion.full_context_chars} chars / {agent.context_expansion.full_text_windows} windows
          </span>
          <span>
            question score {formatNumber(agent.context_expansion.window_question_score, 3)}
            {' -> '}
            {formatNumber(agent.context_expansion.full_question_score, 3)}
          </span>
        </div>
      )}
      <div className="claim-audit-list">
        {claims.map((claim, index) => {
          const topText = claim.text_evidence?.[0];
          const topGraph = claim.graph_evidence?.[0];
          return (
            <article className={`claim-audit-item ${claim.status}`} key={`${index}-${claim.claim}`}>
              <div className="claim-audit-head">
                <Tag color={claim.status === 'supported' ? 'green' : claim.status === 'partial' ? 'orange' : 'red'}>
                  {claim.status}
                </Tag>
                <strong>{formatNumber(claim.support_score, 3)}</strong>
              </div>
              <p>{claim.claim}</p>
              {topText && (
                <div className="claim-evidence-snippet">
                  <span>Text {formatNumber(topText.score, 3)}</span>
                  <p>{topText.text}</p>
                </div>
              )}
              {topGraph && (
                <div className="claim-evidence-snippet graph">
                  <span>Graph {formatNumber(topGraph.score, 3)}</span>
                  <p>{topGraph.text}</p>
                </div>
              )}
            </article>
          );
        })}
      </div>
    </section>
  );
};

const ContractCompletion = ({ agent }) => {
  if (!agent?.enabled) return null;
  const verification = agent.verification || {};
  const policy = agent.conservative_policy || {};
  const guard = agent.repair_guard || {};
  return (
    <section className="trace-panel">
      <div className="trace-section-title">Contract Completion</div>
      <div className="contract-score-grid">
        <div><span>Contract</span><strong>{agent.contract?.type || 'general'}</strong></div>
        <div><span>Coverage</span><strong>{formatNumber(verification.contract_coverage, 3)}</strong></div>
        <div><span>Grounding</span><strong>{formatNumber(verification.evidence_grounding, 3)}</strong></div>
        <div><span>Decision</span><strong>{policy.action || 'not routed'}</strong></div>
      </div>
      {policy.severe_gap && (
        <div className="contract-gap-banner">
          <strong>Severe contract gap</strong>
          <span>
            coverage &le; {formatNumber(policy.max_contract_coverage, 2)} and grounding &le;{' '}
            {formatNumber(policy.max_evidence_grounding, 2)}
          </span>
        </div>
      )}
      {(verification.missing_supported_facts || []).length > 0 && (
        <div className="contract-fact-list">
          <span>Missing supported facts</span>
          {verification.missing_supported_facts.map((fact, index) => (
            <p key={`${index}-${fact}`}>{fact}</p>
          ))}
        </div>
      )}
      {(verification.unsupported_or_off_topic_claims || []).length > 0 && (
        <div className="contract-fact-list warning">
          <span>Unsupported or off-topic claims</span>
          {verification.unsupported_or_off_topic_claims.map((claim, index) => (
            <p key={`${index}-${claim}`}>{claim}</p>
          ))}
        </div>
      )}
      {guard.checks && (
        <div className="contract-check-list">
          {Object.entries(guard.checks).map(([name, passed]) => (
            <Tag key={name} color={passed ? 'green' : 'red'}>{name.replaceAll('_', ' ')}</Tag>
          ))}
        </div>
      )}
    </section>
  );
};

const PostRepairConsensus = ({ agent }) => {
  if (!agent?.enabled) return null;
  return (
    <section className="trace-panel">
      <div className="trace-section-title">Post-Repair Consensus</div>
      <div className="consensus-vote-grid">
        <div><span>Repair votes</span><strong>{agent.repair_votes || 0}</strong></div>
        <div><span>Original votes</span><strong>{agent.original_votes || 0}</strong></div>
        <div><span>Ties</span><strong>{agent.tie_votes || 0}</strong></div>
        <div><span>Utility gain</span><strong>{formatNumber(agent.utility_gain, 3)}</strong></div>
      </div>
      <div className={`consensus-decision ${agent.final_action === 'use_repair' ? 'accepted' : 'reverted'}`}>
        <strong>{agent.final_action}</strong>
        <span>generator/verifier evidence parity with reversed A/B order</span>
      </div>
      {(agent.views || []).map((view, index) => (
        <div className="consensus-view" key={`${index}-${view.repair_is_a}`}>
          <div>
            <strong>View {index + 1}</strong>
            <Tag color={view.winner === 'repair' ? 'green' : view.winner === 'original' ? 'red' : 'orange'}>
              {view.winner}
            </Tag>
          </div>
          <p>{view.reason}</p>
        </div>
      ))}
    </section>
  );
};

const SelectiveConsensus = ({ router }) => {
  if (!router?.enabled) return null;
  return (
    <section className="trace-panel">
      <div className="trace-section-title">Selective Consensus Router</div>
      <div className="selective-router-grid">
        <div><span>First winner</span><strong>{router.primary_winner || 'unknown'}</strong></div>
        <div><span>Utility gain</span><strong>{formatNumber(router.primary_utility_gain, 3)}</strong></div>
        <div><span>Second view</span><strong>{router.second_view_used ? 'used' : 'skipped'}</strong></div>
        <div><span>Final action</span><strong>{router.final_action}</strong></div>
      </div>
      <div className={`selective-router-decision ${router.second_view_used ? 'escalated' : 'direct'}`}>
        <strong>{router.second_view_used ? 'Escalated to reversed view' : 'Accepted after primary view'}</strong>
        <span>repair shown as Answer {router.primary_repair_is_a ? 'A' : 'B'}</span>
      </div>
      {(router.risk_reasons || []).length > 0 && (
        <div className="contract-check-list">
          {router.risk_reasons.map((reason) => (
            <Tag key={reason} color="orange">{reason.replaceAll('_', ' ')}</Tag>
          ))}
        </div>
      )}
    </section>
  );
};

const LearnedContractGate = ({ router }) => {
  if (!router?.enabled) return null;
  return (
    <section className="trace-panel">
      <div className="trace-section-title">Learned Contract-Gap Gate</div>
      <div className="learned-gap-grid">
        <div><span>OOF probability</span><strong>{formatNumber(router.oof_probability, 3)}</strong></div>
        <div><span>Threshold</span><strong>{formatNumber(router.threshold, 3)}</strong></div>
        <div><span>Fold</span><strong>{router.fold || '-'}</strong></div>
        <div><span>Action</span><strong>{router.action}</strong></div>
      </div>
      <div className={`learned-gap-decision ${router.verification_selected ? 'selected' : 'skipped'}`}>
        <strong>{router.verification_selected ? 'Run contract verifier' : 'Skip contract verifier'}</strong>
        <span>{router.repair_used ? 'repair accepted' : 'original retained'}</span>
      </div>
    </section>
  );
};

const EvidenceDeltaGuard = ({ guard }) => {
  if (!guard?.enabled) return null;
  const metrics = guard.metrics || {};
  const retained = guard.action === 'retain_repair';
  return (
    <section className="trace-panel">
      <div className="trace-section-title">Independent Evidence Delta Guard</div>
      <div className="evidence-delta-grid">
        <div>
          <span>Mean support</span>
          <strong>{formatNumber(metrics.pre_mean_support, 3)} -&gt; {formatNumber(metrics.post_mean_support, 3)}</strong>
        </div>
        <div>
          <span>Supported claims</span>
          <strong>{metrics.pre_supported_claims || 0} -&gt; {metrics.post_supported_claims || 0}</strong>
        </div>
        <div>
          <span>Content retained</span>
          <strong>{formatNumber(metrics.supported_capacity_retention, 3)}</strong>
        </div>
        <div>
          <span>Length ratio</span>
          <strong>{formatNumber(metrics.answer_length_ratio, 3)}</strong>
        </div>
      </div>
      <div className={`evidence-delta-decision ${retained ? 'accepted' : 'reverted'}`}>
        <strong>{retained ? 'Repair retained' : 'Original restored'}</strong>
        <span>deterministic evidence statistics, no extra LLM call</span>
      </div>
      <div className="contract-check-list">
        {Object.entries(guard.checks || {}).map(([name, passed]) => (
          <Tag key={name} color={passed ? 'green' : 'red'}>{name.replaceAll('_', ' ')}</Tag>
        ))}
        {(guard.rejection_reasons || []).map((reason) => (
          <Tag key={reason} color="red">{reason.replaceAll('_', ' ')}</Tag>
        ))}
      </div>
    </section>
  );
};

const CandidateArbiter = ({ agent }) => {
  if (!agent?.enabled) return null;
  const decision = agent.decision || {};
  const result = agent.result || {};
  return (
    <section className="trace-panel">
      <div className="trace-section-title">Evidence-Calibrated Candidate Arbitration</div>
      <div className="candidate-arbiter-grid">
        <div><span>Raw winner</span><strong>{decision.raw_winner || 'unknown'}</strong></div>
        <div><span>Selected</span><strong>{decision.selected_candidate || 'text'}</strong></div>
        <div><span>Confidence</span><strong>{formatNumber(result.confidence, 3)}</strong></div>
        <div><span>Utility gain</span><strong>{formatNumber(decision.utility_gain, 3)}</strong></div>
      </div>
      <div className={`candidate-arbiter-decision ${decision.selected_candidate || 'text'}`}>
        <strong>{decision.winner_accepted ? 'Candidate accepted' : 'Low-cost text retained'}</strong>
        <span>anonymous order, shared evidence, no reference answer</span>
      </div>
      <div className="contract-check-list">
        {Object.entries(agent.candidate_order || {}).map(([label, name]) => (
          <Tag key={label} color={name === decision.selected_candidate ? 'green' : 'blue'}>
            {label}: {name}
          </Tag>
        ))}
      </div>
      {result.reason && <p className="candidate-arbiter-reason">{result.reason}</p>}
    </section>
  );
};

const ContextGraphMemory = ({ agent }) => {
  if (!agent?.enabled) return null;
  return (
    <section className="trace-panel">
      <div className="trace-section-title">Context-Level Graph Utility Memory</div>
      <div className="context-memory-grid">
        <div><span>Memory status</span><strong>{agent.graph_memory_active ? 'active' : 'inactive'}</strong></div>
        <div><span>Seeds</span><strong>{agent.seed_count || 0}/{agent.minimum_seeds || 0}</strong></div>
        <div><span>Arbiter choice</span><strong>{agent.arbiter_candidate || 'text'}</strong></div>
        <div><span>Final choice</span><strong>{agent.selected_candidate || 'text'}</strong></div>
      </div>
      <div className={`context-memory-decision ${agent.graph_memory_active ? 'active' : 'inactive'}`}>
        <strong>{agent.action}</strong>
        <span>context {agent.context_hash}</span>
      </div>
      {(agent.seed_lines || []).length > 0 && (
        <div className="contract-check-list">
          {agent.seed_lines.map((line) => <Tag key={line} color="violet">seed line {line}</Tag>)}
        </div>
      )}
    </section>
  );
};

const EvidenceCalibratedLazyAgent = ({ agent, plan }) => {
  if (!agent?.enabled && !plan?.enabled) return null;
  const active = agent?.enabled ? agent : plan;
  const veto = plan?.verification_veto || active?.pregraph_plan?.verification_veto;
  return (
    <section className="trace-panel">
      <div className="trace-section-title">Evidence-Calibrated Lazy Escalation</div>
      <div className="candidate-arbiter-grid">
        <div><span>Pre-graph action</span><strong>{plan?.action || active?.pregraph_plan?.action || 'n/a'}</strong></div>
        <div><span>Final action</span><strong>{agent?.action || active?.action || 'n/a'}</strong></div>
        <div><span>Selected</span><strong>{agent?.selected_candidate || (plan?.repair_dominance ? 'repair' : 'text')}</strong></div>
        <div><span>Graph attempt</span><strong>{agent?.graph_attempted || plan?.requires_graph ? 'yes' : 'no'}</strong></div>
      </div>
      <div className={`candidate-arbiter-decision ${agent?.selected_candidate || 'text'}`}>
        <strong>{plan?.repair_dominance ? 'Supported text repair dominates' : 'Lazy evidence route'}</strong>
        <span>text-only verification before optional graph construction</span>
      </div>
      <div className="contract-check-list">
        {Object.entries(plan?.repair_dominance_checks || {}).map(([name, passed]) => (
          <Tag key={name} color={passed ? 'green' : 'red'}>{name.replaceAll('_', ' ')}</Tag>
        ))}
      </div>
      {veto?.enabled && (
        <div className="contract-check-list">
          {Object.entries(veto.checks || {}).map(([name, passed]) => (
            <Tag key={name} color={passed ? 'green' : 'red'}>{name.replaceAll('_', ' ')}</Tag>
          ))}
        </div>
      )}
    </section>
  );
};

const TextEvidenceExpansionFallback = ({ agent }) => {
  if (!agent?.enabled) return null;
  const guard = agent.guard || {};
  const metrics = guard.post_claim_metrics || {};
  const selectedEvidence = agent.selected_evidence || [];
  return (
    <section className="trace-panel">
      <div className="trace-section-title">Text-Evidence Expansion Fallback</div>
      <div className="candidate-arbiter-grid">
        <div><span>Action</span><strong>{agent.action || 'n/a'}</strong></div>
        <div><span>Accepted</span><strong>{agent.accepted ? 'yes' : 'no'}</strong></div>
        <div><span>Full-text windows</span><strong>{agent.evidence_windows || 0}</strong></div>
        <div><span>Selected evidence</span><strong>{selectedEvidence.length}</strong></div>
      </div>
      <div className={`candidate-arbiter-decision ${agent.accepted ? 'text' : 'reverted'}`}>
        <strong>{agent.accepted ? 'Expanded text answer accepted' : 'Original answer retained'}</strong>
        <span>triggered only after a graph-pending sample could not be completed cheaply</span>
      </div>
      <div className="claim-audit-summary text-expansion-guard">
        <div><span>Claims</span><strong>{metrics.claims || 0}</strong></div>
        <div><span>Supported</span><strong>{metrics.supported_claims || 0}</strong></div>
        <div><span>Mean support</span><strong>{formatNumber(metrics.mean_support, 3)}</strong></div>
        <div><span>Unsupported ratio</span><strong>{formatNumber(metrics.unsupported_ratio, 3)}</strong></div>
      </div>
      <div className="contract-check-list">
        {Object.entries(guard.checks || {}).map(([name, passed]) => (
          <Tag key={name} color={passed ? 'green' : 'red'}>{name.replaceAll('_', ' ')}</Tag>
        ))}
        {guard.support_pruning_applied && <Tag color="violet">support pruning</Tag>}
      </div>
      {selectedEvidence.length > 0 && (
        <div className="text-expansion-evidence-list">
          {selectedEvidence.slice(0, 5).map((item, index) => (
            <article key={`${item.source_id || index}-${index}`} className="text-expansion-evidence">
              <div>
                <strong>{item.source_id || `evidence-${index + 1}`}</strong>
                <span>{formatNumber(item.score, 4)}</span>
              </div>
              <p>{item.text}</p>
            </article>
          ))}
        </div>
      )}
    </section>
  );
};

const EvidenceMismatchRepair = ({ agent }) => {
  if (!agent?.enabled) return null;
  const guard = agent.guard || {};
  const arbiter = guard.arbiter || {};
  const currentMetrics = guard.current_claim_metrics || {};
  const candidateMetrics = guard.candidate_claim_metrics || {};
  const delta = guard.metric_delta || {};
  const selectedEvidence = agent.selected_evidence || [];
  return (
    <section className="trace-panel">
      <div className="trace-section-title">Evidence-Mismatch Repair</div>
      <div className="candidate-arbiter-grid">
        <div><span>Action</span><strong>{agent.action || 'n/a'}</strong></div>
        <div><span>Accepted</span><strong>{agent.accepted ? 'yes' : 'no'}</strong></div>
        <div><span>Arbiter</span><strong>{arbiter.winner || 'tie'}</strong></div>
        <div><span>Confidence</span><strong>{formatNumber(arbiter.confidence, 2)}</strong></div>
      </div>
      <div className={`candidate-arbiter-decision ${agent.accepted ? 'text' : 'reverted'}`}>
        <strong>{agent.accepted ? 'Evidence repair accepted' : 'Current answer retained'}</strong>
        <span>{arbiter.reason || 'evidence-only arbitration without reference answers'}</span>
      </div>
      <div className="claim-audit-summary text-expansion-guard">
        <div>
          <span>Mean support</span>
          <strong>
            {formatNumber(currentMetrics.mean_support, 3)}
            {' -> '}
            {formatNumber(candidateMetrics.mean_support, 3)}
          </strong>
        </div>
        <div>
          <span>Unsupported</span>
          <strong>
            {formatNumber(currentMetrics.unsupported_ratio, 3)}
            {' -> '}
            {formatNumber(candidateMetrics.unsupported_ratio, 3)}
          </strong>
        </div>
        <div><span>Support delta</span><strong>{formatNumber(delta.mean_support_delta, 3)}</strong></div>
        <div><span>Evidence windows</span><strong>{agent.evidence_windows || 0}</strong></div>
      </div>
      <div className="contract-check-list">
        {Object.entries(guard.checks || {}).map(([name, passed]) => (
          <Tag key={name} color={passed ? 'green' : 'red'}>{name.replaceAll('_', ' ')}</Tag>
        ))}
      </div>
      {selectedEvidence.length > 0 && (
        <div className="text-expansion-evidence-list">
          {selectedEvidence.slice(0, 5).map((item, index) => (
            <article key={`${item.source_id || index}-${index}`} className="text-expansion-evidence">
              <div>
                <strong>{item.source_id || `evidence-${index + 1}`}</strong>
                <span>{formatNumber(item.score, 4)}</span>
              </div>
              <p>{item.text}</p>
            </article>
          ))}
        </div>
      )}
    </section>
  );
};

const ConservativeRepairCompression = ({ agent }) => {
  if (!agent?.enabled) return null;
  const guard = agent.guard || {};
  const verifier = guard.semantic_fidelity_verifier || {};
  const selectedEvidence = agent.selected_evidence || [];
  return (
    <section className="trace-panel">
      <div className="trace-section-title">Conservative Repair Compression</div>
      <div className="candidate-arbiter-grid">
        <div><span>Action</span><strong>{agent.action || 'n/a'}</strong></div>
        <div><span>Accepted</span><strong>{agent.accepted ? 'yes' : 'no'}</strong></div>
        <div>
          <span>Chars</span>
          <strong>{guard.original_chars || 0} -&gt; {guard.compressed_chars || 0}</strong>
        </div>
        <div><span>Reduction</span><strong>{formatNumber((guard.length_reduction || 0) * 100, 2)}%</strong></div>
      </div>
      <div className={`candidate-arbiter-decision ${agent.accepted ? 'text' : 'reverted'}`}>
        <strong>{agent.accepted ? 'Compressed repair accepted' : 'Repair retained'}</strong>
        <span>{verifier.reason || 'semantic fidelity checked before replacing the answer'}</span>
      </div>
      <div className="claim-audit-summary text-expansion-guard">
        <div><span>Fact coverage</span><strong>{formatNumber(verifier.essential_fact_coverage, 3)}</strong></div>
        <div><span>Lost facts</span><strong>{(verifier.lost_essential_facts || []).length}</strong></div>
        <div><span>New claims</span><strong>{(verifier.unsupported_new_claims || []).length}</strong></div>
        <div><span>Contradictions</span><strong>{(verifier.contradictions || []).length}</strong></div>
      </div>
      <div className="contract-check-list">
        {Object.entries(guard.checks || {}).map(([name, passed]) => (
          <Tag key={name} color={passed ? 'green' : 'red'}>{name.replaceAll('_', ' ')}</Tag>
        ))}
      </div>
      {selectedEvidence.length > 0 && (
        <div className="text-expansion-evidence-list">
          {selectedEvidence.slice(0, 4).map((item, index) => (
            <article key={`${item.source_id || index}-${index}`} className="text-expansion-evidence">
              <div>
                <strong>{item.source_id || `evidence-${index + 1}`}</strong>
                <span>{formatNumber(item.score, 4)}</span>
              </div>
              <p>{item.text}</p>
            </article>
          ))}
        </div>
      )}
    </section>
  );
};

const QuestionAnchoredRepair = ({ agent }) => {
  if (!agent?.enabled) return null;
  const guard = agent.guard || {};
  const metrics = guard.post_claim_metrics || {};
  const selectedEvidence = agent.selected_evidence || [];
  return (
    <section className="trace-panel">
      <div className="trace-section-title">Question-Anchored Repair</div>
      <div className="candidate-arbiter-grid">
        <div><span>Action</span><strong>{agent.action || 'n/a'}</strong></div>
        <div><span>Accepted</span><strong>{agent.accepted ? 'yes' : 'no'}</strong></div>
        <div><span>Term coverage</span><strong>{formatNumber(guard.question_term_coverage, 3)}</strong></div>
        <div><span>Evidence windows</span><strong>{agent.evidence_windows || 0}</strong></div>
      </div>
      <div className={`candidate-arbiter-decision ${agent.accepted ? 'text' : 'reverted'}`}>
        <strong>{agent.accepted ? 'Question-anchored answer accepted' : 'Failed expansion reverted'}</strong>
        <span>question-only evidence retrieval with setting-consistency checks</span>
      </div>
      <div className="claim-audit-summary text-expansion-guard">
        <div><span>Claims</span><strong>{metrics.claims || 0}</strong></div>
        <div><span>Supported</span><strong>{metrics.supported_claims || 0}</strong></div>
        <div><span>Mean support</span><strong>{formatNumber(metrics.mean_support, 3)}</strong></div>
        <div><span>Unsupported ratio</span><strong>{formatNumber(metrics.unsupported_ratio, 3)}</strong></div>
      </div>
      <div className="contract-check-list">
        {Object.entries(guard.checks || {}).map(([name, passed]) => (
          <Tag key={name} color={passed ? 'green' : 'red'}>{name.replaceAll('_', ' ')}</Tag>
        ))}
        {(guard.question_terms || []).map((term) => (
          <Tag key={term} color="blue">{term}</Tag>
        ))}
      </div>
      {selectedEvidence.length > 0 && (
        <div className="text-expansion-evidence-list">
          {selectedEvidence.slice(0, 5).map((item, index) => (
            <article key={`${item.source_id || index}-${index}`} className="text-expansion-evidence">
              <div>
                <strong>{item.source_id || `evidence-${index + 1}`}</strong>
                <span>{formatNumber(item.score, 4)}</span>
              </div>
              <p>{item.text}</p>
            </article>
          ))}
        </div>
      )}
    </section>
  );
};

const QuestionChunkCompletion = ({ agent }) => {
  if (!agent?.enabled) return null;
  const guard = agent.guard || {};
  const currentMetrics = guard.current_claim_metrics || {};
  const candidateMetrics = guard.candidate_claim_metrics || {};
  const selectedEvidence = agent.selected_evidence || [];
  return (
    <section className="trace-panel">
      <div className="trace-section-title">Question-Chunk Completion</div>
      <div className="candidate-arbiter-grid">
        <div><span>Action</span><strong>{agent.action || 'n/a'}</strong></div>
        <div><span>Accepted</span><strong>{agent.accepted ? 'yes' : 'no'}</strong></div>
        <div><span>Term coverage</span><strong>{formatNumber(guard.question_term_coverage, 3)}</strong></div>
        <div><span>Evidence</span><strong>{selectedEvidence.length}</strong></div>
      </div>
      <div className={`candidate-arbiter-decision ${agent.accepted ? 'text' : 'reverted'}`}>
        <strong>{agent.accepted ? 'Completion accepted' : 'Current answer retained'}</strong>
        <span>question-constrained chunks with positive-utility guard</span>
      </div>
      <div className="claim-audit-summary text-expansion-guard">
        <div>
          <span>Claims</span>
          <strong>{currentMetrics.claims || 0} -&gt; {candidateMetrics.claims || 0}</strong>
        </div>
        <div>
          <span>Mean support</span>
          <strong>{formatNumber(currentMetrics.mean_support, 3)} -&gt; {formatNumber(candidateMetrics.mean_support, 3)}</strong>
        </div>
        <div>
          <span>Unsupported</span>
          <strong>
            {formatNumber(currentMetrics.unsupported_ratio, 3)}
            {' -> '}
            {formatNumber(candidateMetrics.unsupported_ratio, 3)}
          </strong>
        </div>
        <div><span>Current coverage</span><strong>{formatNumber(guard.current_question_term_coverage, 3)}</strong></div>
      </div>
      <div className="contract-check-list">
        {Object.entries(guard.checks || {}).map(([name, passed]) => (
          <Tag key={name} color={passed ? 'green' : 'red'}>{name.replaceAll('_', ' ')}</Tag>
        ))}
      </div>
      {selectedEvidence.length > 0 && (
        <div className="text-expansion-evidence-list">
          {selectedEvidence.slice(0, 5).map((item, index) => (
            <article key={`${item.source_id || index}-${index}`} className="text-expansion-evidence">
              <div>
                <strong>{item.source_id || `evidence-${index + 1}`}</strong>
                <span>{formatNumber(item.score, 4)}</span>
              </div>
              <p>{item.text}</p>
            </article>
          ))}
        </div>
      )}
    </section>
  );
};

const CompressionGuardPanel = ({ agent, title }) => {
  if (!agent?.enabled) return null;
  const guard = agent.guard || {};
  const verifier = guard.semantic_fidelity_verifier || {};
  const selectedEvidence = agent.selected_evidence || [];
  const generation = agent.generation || {};
  const afterChars = guard.compressed_chars ?? guard.revised_chars ?? guard.compact_chars ?? 0;
  const words = guard.compressed_words ?? guard.revised_words ?? guard.compact_words;
  const preservedFacts = (
    generation.must_preserve
    || generation.essential_facts_kept
    || generation.restored_facts
    || generation.kept_facets
    || []
  );
  return (
    <section className="trace-panel">
      <div className="trace-section-title">{title}</div>
      <div className="candidate-arbiter-grid">
        <div><span>Action</span><strong>{agent.action || 'n/a'}</strong></div>
        <div><span>Accepted</span><strong>{agent.accepted ? 'yes' : 'no'}</strong></div>
        <div><span>Chars</span><strong>{guard.original_chars || 0} -&gt; {afterChars}</strong></div>
        <div><span>Reduction</span><strong>{formatNumber((guard.length_reduction || 0) * 100, 2)}%</strong></div>
        {words !== undefined && <div><span>Words</span><strong>{words}</strong></div>}
      </div>
      <div className={`candidate-arbiter-decision ${agent.accepted ? 'text' : 'reverted'}`}>
        <strong>{agent.accepted ? 'Compression accepted' : 'Compression rejected'}</strong>
        <span>{verifier.reason || 'semantic fidelity and evidence support checked before replacement'}</span>
      </div>
      <div className="claim-audit-summary text-expansion-guard">
        <div><span>Fact coverage</span><strong>{formatNumber(verifier.essential_fact_coverage, 3)}</strong></div>
        <div><span>Lost facts</span><strong>{(verifier.lost_essential_facts || []).length}</strong></div>
        <div><span>New claims</span><strong>{(verifier.unsupported_new_claims || []).length}</strong></div>
        <div><span>Contradictions</span><strong>{(verifier.contradictions || []).length}</strong></div>
      </div>
      <div className="contract-check-list">
        {Object.entries(guard.checks || {}).map(([name, passed]) => (
          <Tag key={name} color={passed ? 'green' : 'red'}>{name.replaceAll('_', ' ')}</Tag>
        ))}
      </div>
      {preservedFacts.length > 0 && (
        <div className="contract-fact-list">
          <span>Preserved facts</span>
          {preservedFacts.slice(0, 8).map((fact, index) => (
            <p key={`${index}-${fact}`}>{fact}</p>
          ))}
        </div>
      )}
      {selectedEvidence.length > 0 && (
        <div className="text-expansion-evidence-list">
          {selectedEvidence.slice(0, 4).map((item, index) => (
            <article key={`${item.source_id || index}-${index}`} className="text-expansion-evidence">
              <div>
                <strong>{item.source_id || `evidence-${index + 1}`}</strong>
                <span>{formatNumber(item.score, 4)}</span>
              </div>
              <p>{item.text}</p>
            </article>
          ))}
        </div>
      )}
    </section>
  );
};

const RefusalRepairPanel = ({ agent, title, subtitle }) => {
  if (!agent?.enabled) return null;
  const guard = agent.guard || {};
  const metrics = guard.post_claim_metrics || {};
  const selectedEvidence = agent.selected_evidence || [];
  const expansion = agent.query_expansion || {};
  return (
    <section className="trace-panel">
      <div className="trace-section-title">{title}</div>
      <div className="candidate-arbiter-grid">
        <div><span>Action</span><strong>{agent.action || 'n/a'}</strong></div>
        <div><span>Accepted</span><strong>{agent.accepted ? 'yes' : 'no'}</strong></div>
        <div><span>Term coverage</span><strong>{formatNumber(guard.question_term_coverage, 3)}</strong></div>
        <div><span>Evidence</span><strong>{selectedEvidence.length}</strong></div>
      </div>
      <div className={`candidate-arbiter-decision ${agent.accepted ? 'text' : 'reverted'}`}>
        <strong>{agent.accepted ? 'Refusal repaired' : 'Current answer retained'}</strong>
        <span>{subtitle}</span>
      </div>
      <div className="claim-audit-summary text-expansion-guard">
        <div><span>Claims</span><strong>{metrics.claims || 0}</strong></div>
        <div><span>Supported</span><strong>{metrics.supported_claims || 0}</strong></div>
        <div><span>Mean support</span><strong>{formatNumber(metrics.mean_support, 3)}</strong></div>
        <div><span>Unsupported ratio</span><strong>{formatNumber(metrics.unsupported_ratio, 3)}</strong></div>
      </div>
      <div className="contract-check-list">
        {Object.entries(guard.checks || {}).map(([name, passed]) => (
          <Tag key={name} color={passed ? 'green' : 'red'}>{name.replaceAll('_', ' ')}</Tag>
        ))}
        {(guard.question_terms || []).map((term) => (
          <Tag key={term} color="blue">{term}</Tag>
        ))}
      </div>
      {(expansion.search_phrases || expansion.must_cover_terms) && (
        <div className="contract-fact-list">
          <span>Query expansion</span>
          {(expansion.search_phrases || []).slice(0, 5).map((phrase, index) => (
            <p key={`phrase-${index}-${phrase}`}>{phrase}</p>
          ))}
          {(expansion.must_cover_terms || []).slice(0, 10).map((term) => (
            <Tag key={term} size="sm" color="cyan">{term}</Tag>
          ))}
        </div>
      )}
      {selectedEvidence.length > 0 && (
        <div className="text-expansion-evidence-list">
          {selectedEvidence.slice(0, 5).map((item, index) => (
            <article key={`${item.source_id || index}-${index}`} className="text-expansion-evidence">
              <div>
                <strong>{item.source_id || `evidence-${index + 1}`}</strong>
                <span>{formatNumber(item.score, 4)}</span>
              </div>
              <p>{item.text}</p>
            </article>
          ))}
        </div>
      )}
    </section>
  );
};

const ContractGapRepairPanel = ({ agent }) => {
  if (!agent?.enabled) return null;
  const guard = agent.guard || {};
  const metrics = guard.candidate_claim_metrics || {};
  const verification = agent.verification || {};
  const selectedEvidence = agent.selected_evidence || [];
  const expansion = agent.query_expansion || {};
  return (
    <section className="trace-panel">
      <div className="trace-section-title">Query-Expanded Contract-Gap Repair</div>
      <div className="candidate-arbiter-grid">
        <div><span>Action</span><strong>{agent.action || 'n/a'}</strong></div>
        <div><span>Accepted</span><strong>{agent.accepted ? 'yes' : 'no'}</strong></div>
        <div><span>Coverage</span><strong>{formatNumber(verification.contract_coverage, 3)}</strong></div>
        <div><span>Grounding</span><strong>{formatNumber(verification.evidence_grounding, 3)}</strong></div>
      </div>
      <div className={`candidate-arbiter-decision ${agent.accepted ? 'text' : 'reverted'}`}>
        <strong>{agent.accepted ? 'Contract-gap repair accepted' : 'Current answer retained'}</strong>
        <span>{verification.reason || 'question-expanded evidence and contract verifier'}</span>
      </div>
      <div className="claim-audit-summary text-expansion-guard">
        <div><span>Missing facts</span><strong>{(verification.missing_supported_facts || []).length}</strong></div>
        <div><span>Unsupported claims</span><strong>{(verification.unsupported_or_off_topic_claims || []).length}</strong></div>
        <div><span>Mean support</span><strong>{formatNumber(metrics.mean_support, 3)}</strong></div>
        <div><span>Unsupported ratio</span><strong>{formatNumber(metrics.unsupported_ratio, 3)}</strong></div>
      </div>
      <div className="contract-check-list">
        {Object.entries(guard.checks || {}).map(([name, passed]) => (
          <Tag key={name} color={passed ? 'green' : 'red'}>{name.replaceAll('_', ' ')}</Tag>
        ))}
        {(guard.question_terms || []).map((term) => (
          <Tag key={term} color="blue">{term}</Tag>
        ))}
      </div>
      {(verification.missing_supported_facts || []).length > 0 && (
        <div className="contract-fact-list">
          <span>Missing supported facts</span>
          {verification.missing_supported_facts.slice(0, 8).map((fact, index) => (
            <p key={`${index}-${fact}`}>{fact}</p>
          ))}
        </div>
      )}
      {(verification.unsupported_or_off_topic_claims || []).length > 0 && (
        <div className="contract-fact-list warning">
          <span>Unsupported or off-topic claims</span>
          {verification.unsupported_or_off_topic_claims.slice(0, 8).map((claim, index) => (
            <p key={`${index}-${claim}`}>{claim}</p>
          ))}
        </div>
      )}
      {(expansion.search_phrases || expansion.must_cover_terms) && (
        <div className="contract-fact-list">
          <span>Query expansion</span>
          {(expansion.search_phrases || []).slice(0, 5).map((phrase, index) => (
            <p key={`contract-phrase-${index}-${phrase}`}>{phrase}</p>
          ))}
          {(expansion.must_cover_terms || []).slice(0, 10).map((term) => (
            <Tag key={term} size="sm" color="cyan">{term}</Tag>
          ))}
        </div>
      )}
      {selectedEvidence.length > 0 && (
        <div className="text-expansion-evidence-list">
          {selectedEvidence.slice(0, 5).map((item, index) => (
            <article key={`${item.source_id || index}-${index}`} className="text-expansion-evidence">
              <div>
                <strong>{item.source_id || `evidence-${index + 1}`}</strong>
                <span>{formatNumber(item.score, 4)}</span>
              </div>
              <p>{item.text}</p>
            </article>
          ))}
        </div>
      )}
    </section>
  );
};

const SupportedFacetGuardPanel = ({ guard }) => {
  if (!guard?.enabled) return null;
  const stats = guard.stats || {};
  const reverted = guard.action === 'revert_to_pre_repair_answer';
  return (
    <section className="trace-panel">
      <div className="trace-section-title">Supported-Facet Retention Guard</div>
      <div className="candidate-arbiter-grid">
        <div><span>Action</span><strong>{guard.action || 'n/a'}</strong></div>
        <div><span>Retention</span><strong>{formatNumber(stats.supported_facet_retention, 3)}</strong></div>
        <div><span>Compression</span><strong>{formatNumber(stats.compression_ratio, 3)}</strong></div>
        <div><span>Supported terms</span><strong>{stats.supported_before_terms || 0}</strong></div>
      </div>
      <div className={`candidate-arbiter-decision ${reverted ? 'reverted' : 'text'}`}>
        <strong>{reverted ? 'Repair reverted' : 'Repair kept'}</strong>
        <span>{guard.reason || 'checks evidence-supported facets before accepting repair'}</span>
      </div>
      <div className="claim-audit-summary text-expansion-guard">
        <div><span>Before chars</span><strong>{stats.before_chars || 0}</strong></div>
        <div><span>After chars</span><strong>{stats.after_chars || 0}</strong></div>
        <div><span>Retained terms</span><strong>{stats.retained_supported_terms || 0}</strong></div>
        <div><span>Dropped terms</span><strong>{stats.dropped_supported_terms || 0}</strong></div>
      </div>
      {(stats.dropped_supported_term_examples || []).length > 0 && (
        <div className="contract-check-list">
          {stats.dropped_supported_term_examples.slice(0, 20).map((term) => (
            <Tag key={term} color={reverted ? 'orange' : 'blue'}>{term}</Tag>
          ))}
        </div>
      )}
    </section>
  );
};

const SupportedFacetFusionPanel = ({ agent }) => {
  if (!agent?.enabled) return null;
  const guard = agent.guard || {};
  const stats = guard.supported_facet_stats || {};
  const contractChecks = guard.contract_guard?.checks || {};
  return (
    <section className="trace-panel">
      <div className="trace-section-title">Supported-Facet Fusion</div>
      <div className="candidate-arbiter-grid">
        <div><span>Action</span><strong>{agent.action || 'n/a'}</strong></div>
        <div><span>Accepted</span><strong>{agent.accepted ? 'yes' : 'no'}</strong></div>
        <div><span>Retention</span><strong>{formatNumber(stats.supported_facet_retention, 3)}</strong></div>
        <div><span>Required</span><strong>{formatNumber(guard.required_retention_floor, 3)}</strong></div>
      </div>
      <div className={`candidate-arbiter-decision ${agent.accepted ? 'text' : 'reverted'}`}>
        <strong>{agent.accepted ? 'Fusion accepted' : 'Fusion rejected'}</strong>
        <span>
          original facets + verified missing facts; reference answer not used
        </span>
      </div>
      <div className="claim-audit-summary text-expansion-guard">
        <div><span>Before chars</span><strong>{stats.before_chars || 0}</strong></div>
        <div><span>After chars</span><strong>{stats.after_chars || 0}</strong></div>
        <div><span>Compression</span><strong>{formatNumber(stats.compression_ratio, 3)}</strong></div>
        <div><span>Reverted retention</span><strong>{formatNumber(guard.reverted_repair_retention, 3)}</strong></div>
      </div>
      <div className="contract-check-list">
        {Object.entries(guard.checks || {}).map(([name, passed]) => (
          <Tag key={name} color={passed ? 'green' : 'red'}>{name.replaceAll('_', ' ')}</Tag>
        ))}
        {Object.entries(contractChecks).slice(0, 10).map(([name, passed]) => (
          <Tag key={`contract-${name}`} color={passed ? 'cyan' : 'orange'}>
            {name.replaceAll('_', ' ')}
          </Tag>
        ))}
      </div>
      {(stats.dropped_supported_term_examples || []).length > 0 && (
        <div className="contract-check-list">
          {stats.dropped_supported_term_examples.slice(0, 20).map((term) => (
            <Tag key={term} color="blue">{term}</Tag>
          ))}
        </div>
      )}
    </section>
  );
};

const ResidualHardcaseEditorPanel = ({ agent }) => {
  if (!agent?.enabled) return null;
  const guard = agent.guard || {};
  const decision = agent.decision || {};
  const stats = guard.supported_facet_stats || {};
  const selectedEvidence = agent.selected_evidence || [];
  return (
    <section className="trace-panel">
      <div className="trace-section-title">Residual Hard-Case Editor</div>
      <div className="candidate-arbiter-grid">
        <div><span>Action</span><strong>{agent.action || 'n/a'}</strong></div>
        <div><span>Accepted</span><strong>{agent.accepted ? 'yes' : 'no'}</strong></div>
        <div><span>Edit type</span><strong>{decision.edit_type || 'n/a'}</strong></div>
        <div><span>Missing coverage</span><strong>{formatNumber(guard.missing_fact_coverage, 3)}</strong></div>
      </div>
      <div className={`candidate-arbiter-decision ${agent.accepted ? 'text' : 'reverted'}`}>
        <strong>{agent.accepted ? 'Edit accepted' : 'Edit rejected'}</strong>
        <span>{decision.reason || 'local evidence retrieval + conservative answer edit'}</span>
      </div>
      <div className="claim-audit-summary text-expansion-guard">
        <div><span>Chars</span><strong>{guard.current_chars || 0} -> {guard.candidate_chars || 0}</strong></div>
        <div><span>Q terms</span><strong>{formatNumber(guard.current_question_term_coverage, 3)} -> {formatNumber(guard.candidate_question_term_coverage, 3)}</strong></div>
        <div><span>Retention</span><strong>{formatNumber(stats.supported_facet_retention, 3)}</strong></div>
        <div><span>Support</span><strong>{formatNumber(guard.candidate_claim_metrics?.mean_support, 3)}</strong></div>
      </div>
      <div className="contract-check-list">
        {Object.entries(guard.checks || {}).map(([name, passed]) => (
          <Tag key={name} color={passed ? 'green' : 'red'}>{name.replaceAll('_', ' ')}</Tag>
        ))}
      </div>
      {(guard.rejection_reasons || []).length > 0 && (
        <div className="contract-check-list">
          {guard.rejection_reasons.map((reason) => (
            <Tag key={reason} color="orange">{reason.replaceAll('_', ' ')}</Tag>
          ))}
        </div>
      )}
      {selectedEvidence.length > 0 && (
        <div className="text-expansion-evidence-list">
          {selectedEvidence.slice(0, 4).map((item, index) => (
            <article key={`${item.source_id || index}-${index}`} className="text-expansion-evidence">
              <div>
                <strong>{item.source_id || `evidence-${index + 1}`}</strong>
                <span>{formatNumber(item.score, 4)}</span>
              </div>
              <p>{item.text}</p>
            </article>
          ))}
        </div>
      )}
    </section>
  );
};

const EvidenceGatePanel = ({ gate, title }) => {
  if (!gate?.enabled) return null;
  const accepted = Boolean(gate.accepted);
  return (
    <section className="trace-panel">
      <div className="trace-section-title">{title}</div>
      <div className="candidate-arbiter-grid">
        <div><span>Action</span><strong>{gate.action || 'n/a'}</strong></div>
        <div><span>Accepted</span><strong>{accepted ? 'yes' : 'no'}</strong></div>
        <div><span>Compression</span><strong>{formatNumber(gate.compression_ratio, 3)}</strong></div>
        <div><span>Retention</span><strong>{formatNumber(gate.supported_facet_retention, 3)}</strong></div>
      </div>
      <div className={`candidate-arbiter-decision ${accepted ? 'text' : 'reverted'}`}>
        <strong>{accepted ? 'Candidate kept' : 'Candidate reverted'}</strong>
        <span>runtime-only evidence checks; reference answer and judge score not used</span>
      </div>
      <div className="claim-audit-summary text-expansion-guard">
        <div><span>Q terms</span><strong>{formatNumber(gate.current_question_term_coverage, 3)} -> {formatNumber(gate.candidate_question_term_coverage, 3)}</strong></div>
        <div><span>Support</span><strong>{formatNumber(gate.current_mean_support, 3)} -> {formatNumber(gate.candidate_mean_support, 3)}</strong></div>
        <div><span>Unsupported</span><strong>{formatNumber(gate.current_unsupported_ratio, 3)} -> {formatNumber(gate.candidate_unsupported_ratio, 3)}</strong></div>
        <div><span>Supported terms</span><strong>{gate.supported_before_terms || 0}</strong></div>
      </div>
      <div className="contract-check-list">
        {Object.entries(gate.checks || {}).map(([name, passed]) => (
          <Tag key={name} color={passed ? 'green' : 'red'}>{name.replaceAll('_', ' ')}</Tag>
        ))}
      </div>
      {(gate.rejection_reasons || []).length > 0 && (
        <div className="contract-check-list">
          {gate.rejection_reasons.map((reason) => (
            <Tag key={reason} color="orange">{reason.replaceAll('_', ' ')}</Tag>
          ))}
        </div>
      )}
    </section>
  );
};

const AgentControllerPanel = ({ agent }) => {
  if (!agent?.enabled) return null;
  const planner = agent.planner || {};
  const observation = agent.initial_observation || {};
  const finalDecision = agent.final_decision || {};
  const riskFlags = observation.risk_flags || [];
  const actions = agent.actions || [];
  return (
    <section className="trace-panel">
      <div className="trace-section-title">KG-RAG Agent Controller</div>
      <div className="candidate-arbiter-grid">
        <div><span>Agent</span><strong>{agent.agent_name || 'KG-RAG Agent'}</strong></div>
        <div><span>Question type</span><strong>{planner.question_type || 'unknown'}</strong></div>
        <div><span>Final policy</span><strong>{finalDecision.policy || 'n/a'}</strong></div>
        <div><span>Actions</span><strong>{actions.length}</strong></div>
      </div>
      <div className={`candidate-arbiter-decision ${finalDecision.rolled_back ? 'reverted' : 'text'}`}>
        <strong>{agent.agent_loop || 'observe -> plan -> act -> verify -> answer'}</strong>
        <span>{planner.goal || 'adaptive evidence-aware answer control'}</span>
      </div>
      <div className="contract-check-list">
        {(planner.selected_tools || []).map((tool) => (
          <Tag key={tool} color="blue">{tool.replaceAll('_', ' ')}</Tag>
        ))}
      </div>
      {riskFlags.length > 0 && (
        <div className="contract-check-list">
          {riskFlags.map((flag) => (
            <Tag key={flag} color="orange">{flag.replaceAll('_', ' ')}</Tag>
          ))}
        </div>
      )}
      <div className="claim-audit-summary text-expansion-guard">
        <div><span>Graph nodes</span><strong>{observation.graph_nodes || 0}</strong></div>
        <div><span>Graph edges</span><strong>{observation.graph_edges || 0}</strong></div>
        <div><span>Answer chars</span><strong>{finalDecision.answer_chars || 0}</strong></div>
        <div><span>Rollback</span><strong>{finalDecision.rolled_back ? 'yes' : 'no'}</strong></div>
      </div>
      <div className="claim-audit-list">
        {actions.map((step, index) => (
          <article className="claim-audit-item supported" key={`${index}-${step.tool}-${step.decision}`}>
            <div className="claim-audit-head">
              <Tag color={step.decision?.includes('revert') || step.decision?.includes('rollback') ? 'orange' : 'green'}>
                {step.tool?.replaceAll('_', ' ')}
              </Tag>
              <strong>{step.decision}</strong>
            </div>
            <p>{step.observation}</p>
            <div className="contract-check-list">
              {step.accepted !== undefined && (
                <Tag color={step.accepted ? 'green' : 'red'}>
                  {step.accepted ? 'accepted' : 'rejected'}
                </Tag>
              )}
              {(step.rejection_reasons || []).map((reason) => (
                <Tag key={reason} color="orange">{reason.replaceAll('_', ' ')}</Tag>
              ))}
              {step.compression_ratio !== undefined && (
                <Tag color="cyan">compression {formatNumber(step.compression_ratio, 3)}</Tag>
              )}
              {step.supported_facet_retention !== undefined && (
                <Tag color="cyan">retention {formatNumber(step.supported_facet_retention, 3)}</Tag>
              )}
            </div>
          </article>
        ))}
      </div>
    </section>
  );
};

const EvidenceTracePage = () => {
  const [fileName, setFileName] = useState(DEFAULT_TRACE_FILE);
  const [availableFiles, setAvailableFiles] = useState([]);
  const [traces, setTraces] = useState([]);
  const [selectedLineNo, setSelectedLineNo] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    const fetchTraces = async () => {
      setLoading(true);
      setError(null);
      try {
        const response = await knowledgeGraphAPI.getEvidenceTraces(fileName, 80);
        const data = response.data || {};
        setAvailableFiles(data.available_files || []);
        setTraces(data.traces || []);
        setSelectedLineNo((data.traces || [])[0]?.line_no ?? null);
      } catch (err) {
        console.error('Error fetching evidence traces:', err);
        setError('Failed to load evidence traces');
      } finally {
        setLoading(false);
      }
    };

    fetchTraces();
  }, [fileName]);

  const selectedTrace = useMemo(
    () => traces.find((trace) => trace.line_no === selectedLineNo) || traces[0],
    [traces, selectedLineNo]
  );

  const activeRouterFamily = Object.entries(ROUTER_TRACE_FILES)
    .find(([, files]) => Object.values(files).includes(fileName))?.[0] || 'v22';
  const activeBudget = Object.entries(ROUTER_TRACE_FILES[activeRouterFamily])
    .find(([, value]) => value === fileName)?.[0] || 'balanced';

  const summary = useMemo(() => {
    const counterfactualCount = traces.reduce(
      (sum, trace) => sum + (trace.filtered_counterfactual_edges?.length || 0),
      0
    );
    const repairCount = traces.reduce(
      (sum, trace) => sum + (trace.sufficiency?.triggered ? 1 : 0),
      0
    );
    const fallbackCount = traces.reduce(
      (sum, trace) => sum + (trace.uncertainty_fallback?.action?.startsWith('use_') ? 1 : 0),
      0
    );
    const routerCount = traces.reduce(
      (sum, trace) => sum + (trace.query_router?.action?.startsWith('use_') ? 1 : 0),
      0
    );
    const compressionCount = traces.reduce(
      (sum, trace) => sum + (trace.rescue_compression?.action === 'use_compressed' ? 1 : 0),
      0
    );
    const legalRouterCount = traces.reduce(
      (sum, trace) => sum + (trace.legal_error_router?.action?.startsWith('use_') ? 1 : 0),
      0
    );
    const contractRescueCount = traces.reduce(
      (sum, trace) => sum + (trace.answer_contract_agent?.action?.endsWith('_rescue') ? 1 : 0),
      0
    );
    const textFirstGraphAttempts = traces.reduce(
      (sum, trace) => sum + (trace.text_first_contract_agent?.graph_rescue_attempted ? 1 : 0),
      0
    );
    const textFirstGraphAccepts = traces.reduce(
      (sum, trace) => sum + (trace.text_first_contract_agent?.graph_rescue_accepted ? 1 : 0),
      0
    );
    const metaGraphRoutes = traces.reduce(
      (sum, trace) => sum + (trace.budget_meta_agent?.chosen_policy === 'graph_first' ? 1 : 0),
      0
    );
    const learnedGraphAttempts = traces.reduce(
      (sum, trace) => sum + (trace.learned_dual_gate?.graph_attempted ? 1 : 0),
      0
    );
    const learnedGraphAccepts = traces.reduce(
      (sum, trace) => sum + (trace.learned_dual_gate?.graph_accepted ? 1 : 0),
      0
    );
    const allContextHashes = new Set(
      traces.map((trace) => trace.context_budget_agent?.context_hash).filter(Boolean)
    );
    const selectedContextHashes = new Set(
      traces
        .filter((trace) => trace.context_budget_agent?.context_selected)
        .map((trace) => trace.context_budget_agent?.context_hash)
        .filter(Boolean)
    );
    const claimRepairs = traces.reduce(
      (sum, trace) => sum + (trace.claim_evidence_agent?.repair_applied ? 1 : 0),
      0
    );
    const contextExpansions = traces.reduce(
      (sum, trace) => sum + (trace.claim_evidence_agent?.context_expansion?.enabled ? 1 : 0),
      0
    );
    const unsupportedClaims = traces.reduce(
      (sum, trace) => sum + Number(trace.claim_evidence_agent?.unsupported_claims || 0),
      0
    );
    const contractVerifications = traces.reduce(
      (sum, trace) => sum + (trace.contract_completion_agent?.verification_attempted ? 1 : 0),
      0
    );
    const contractRepairs = traces.reduce(
      (sum, trace) => sum + (trace.contract_completion_agent?.repair_accepted ? 1 : 0),
      0
    );
    const consensusChecks = traces.reduce(
      (sum, trace) => sum + (trace.post_repair_consensus_agent?.enabled ? 1 : 0),
      0
    );
    const consensusReverts = traces.reduce(
      (sum, trace) => sum + (trace.post_repair_consensus_agent?.final_action === 'retain_original' ? 1 : 0),
      0
    );
    const selectiveChecks = traces.reduce(
      (sum, trace) => sum + (trace.selective_consensus_router?.enabled ? 1 : 0),
      0
    );
    const selectiveEscalations = traces.reduce(
      (sum, trace) => sum + (trace.selective_consensus_router?.second_view_used ? 1 : 0),
      0
    );
    const learnedGapChecks = traces.reduce(
      (sum, trace) => sum + (trace.learned_contract_gap_router?.enabled ? 1 : 0),
      0
    );
    const learnedGapSelected = traces.reduce(
      (sum, trace) => sum + (trace.learned_contract_gap_router?.verification_selected ? 1 : 0),
      0
    );
    const evidenceDeltaChecks = traces.reduce(
      (sum, trace) => sum + (trace.evidence_delta_guard?.enabled ? 1 : 0),
      0
    );
    const evidenceDeltaReverts = traces.reduce(
      (sum, trace) => sum + (trace.evidence_delta_guard?.action === 'retain_original' ? 1 : 0),
      0
    );
    const candidateArbitrations = traces.reduce(
      (sum, trace) => sum + (trace.evidence_candidate_arbiter?.enabled ? 1 : 0),
      0
    );
    const arbiterGraphSelections = traces.reduce(
      (sum, trace) => sum + (trace.evidence_candidate_arbiter?.decision?.selected_candidate === 'graph' ? 1 : 0),
      0
    );
    const memoryGraphRoutes = traces.reduce(
      (sum, trace) => sum + (trace.context_graph_memory_agent?.selected_candidate === 'graph' ? 1 : 0),
      0
    );
    const activeMemoryContexts = new Set(
      traces
        .filter((trace) => trace.context_graph_memory_agent?.graph_memory_active)
        .map((trace) => trace.context_graph_memory_agent?.context_hash)
        .filter(Boolean)
    );
    const lazyGraphAttempts = traces.reduce(
      (sum, trace) => sum + (trace.evidence_calibrated_lazy_agent?.graph_attempted ? 1 : 0),
      0
    );
    const lazyTextRepairs = traces.reduce(
      (sum, trace) => sum + (trace.evidence_calibrated_lazy_agent?.selected_candidate === 'repair' ? 1 : 0),
      0
    );
    const lazyVerifierVetoes = traces.reduce(
      (sum, trace) => sum + (
        trace.evidence_calibrated_lazy_agent?.pregraph_plan?.action === 'keep_text_verifier_veto'
        || trace.v41_pregraph_plan?.action === 'keep_text_verifier_veto'
          ? 1
          : 0
      ),
      0
    );
    const textEvidenceExpansions = traces.reduce(
      (sum, trace) => sum + (trace.text_evidence_expansion_fallback?.attempted ? 1 : 0),
      0
    );
    const textEvidenceAccepted = traces.reduce(
      (sum, trace) => sum + (trace.text_evidence_expansion_fallback?.accepted ? 1 : 0),
      0
    );
    const evidenceMismatchAttempts = traces.reduce(
      (sum, trace) => sum + (trace.evidence_mismatch_repair?.attempted ? 1 : 0),
      0
    );
    const evidenceMismatchAccepted = traces.reduce(
      (sum, trace) => sum + (trace.evidence_mismatch_repair?.accepted ? 1 : 0),
      0
    );
    const conservativeCompressionAttempts = traces.reduce(
      (sum, trace) => sum + (trace.conservative_repair_compression?.attempted ? 1 : 0),
      0
    );
    const conservativeCompressionAccepted = traces.reduce(
      (sum, trace) => sum + (trace.conservative_repair_compression?.accepted ? 1 : 0),
      0
    );
    const questionAnchoredAttempts = traces.reduce(
      (sum, trace) => sum + (trace.question_anchored_repair?.attempted ? 1 : 0),
      0
    );
    const questionAnchoredAccepted = traces.reduce(
      (sum, trace) => sum + (trace.question_anchored_repair?.accepted ? 1 : 0),
      0
    );
    const questionAnchoredReverts = traces.reduce(
      (sum, trace) => sum + (trace.question_anchored_repair?.action === 'revert_failed_expansion' ? 1 : 0),
      0
    );
    const questionChunkAttempts = traces.reduce(
      (sum, trace) => sum + (trace.question_chunk_completion?.attempted ? 1 : 0),
      0
    );
    const questionChunkAccepted = traces.reduce(
      (sum, trace) => sum + (trace.question_chunk_completion?.accepted ? 1 : 0),
      0
    );
    return {
      samples: traces.length,
      avgCoverage: average(traces, (trace) => trace.selection_summary?.term_coverage),
      avgSelectedEdges: average(traces, (trace) => trace.selection_summary?.selected_edges),
      counterfactualCount,
      repairCount,
      fallbackCount,
      routerCount,
      compressionCount,
      legalRouterCount,
      contractRescueCount,
      textFirstGraphAttempts,
      textFirstGraphAccepts,
      metaGraphRoutes,
      learnedGraphAttempts,
      learnedGraphAccepts,
      contextCount: allContextHashes.size,
      selectedContextCount: selectedContextHashes.size,
      claimRepairs,
      contextExpansions,
      unsupportedClaims,
      avgClaimSupport: average(traces, (trace) => trace.claim_evidence_agent?.mean_support),
      contractVerifications,
      contractRepairs,
      consensusChecks,
      consensusReverts,
      selectiveChecks,
      selectiveEscalations,
      learnedGapChecks,
      learnedGapSelected,
      evidenceDeltaChecks,
      evidenceDeltaReverts,
      candidateArbitrations,
      arbiterGraphSelections,
      memoryGraphRoutes,
      activeMemoryContexts: activeMemoryContexts.size,
      lazyGraphAttempts,
      lazyTextRepairs,
      lazyVerifierVetoes,
      textEvidenceExpansions,
      textEvidenceAccepted,
      evidenceMismatchAttempts,
      evidenceMismatchAccepted,
      conservativeCompressionAttempts,
      conservativeCompressionAccepted,
      questionAnchoredAttempts,
      questionAnchoredAccepted,
      questionAnchoredReverts,
      questionChunkAttempts,
      questionChunkAccepted,
    };
  }, [traces]);

  const fileOptions = availableFiles
    .filter((name) => name.includes('trace') || name.includes('mse_pathrag'))
    .map((name) => ({ label: name, value: name }));

  return (
    <Layout>
      <div className="evidence-trace-page">
        <div className="trace-page-header">
          <div>
            <h2>Evidence Trace</h2>
            <p>MSE-PathRAG evidence selection and counterfactual path filtering</p>
          </div>
          <div className="trace-header-controls">
            <ButtonGroup className="trace-budget-control">
              {Object.keys(ROUTER_TRACE_FILES).map((family) => (
                <Button
                  key={family}
                  appearance={activeRouterFamily === family ? 'primary' : 'default'}
                  onClick={() => setFileName(
                    ROUTER_TRACE_FILES[family][activeBudget]
                    || ROUTER_TRACE_FILES[family].balanced
                    || Object.values(ROUTER_TRACE_FILES[family])[0]
                  )}
                >
                  {family.toUpperCase()}
                </Button>
              ))}
            </ButtonGroup>
            <ButtonGroup className="trace-budget-control">
              {Object.entries(ROUTER_TRACE_FILES[activeRouterFamily]).map(([mode, traceFile]) => (
                <Button
                  key={mode}
                  appearance={activeBudget === mode ? 'primary' : 'default'}
                  onClick={() => setFileName(traceFile)}
                >
                  {mode.charAt(0).toUpperCase() + mode.slice(1)}
                </Button>
              ))}
            </ButtonGroup>
            <SelectPicker
              data={fileOptions}
              value={fileName}
              onChange={(value) => value && setFileName(value)}
              cleanable={false}
              searchable
              className="trace-file-picker"
              placeholder="Trace file"
            />
          </div>
        </div>

        {error && <Message type="error">{error}</Message>}

        {loading ? (
          <div className="trace-loading">
            <Loader content="Loading evidence traces..." />
          </div>
        ) : !selectedTrace ? (
          <div className="trace-empty">No evidence trace data available.</div>
        ) : (
          <>
            <div className="trace-metrics">
              <div>
                <span>Samples</span>
                <strong>{summary.samples}</strong>
              </div>
              <div>
                <span>Avg Coverage</span>
                <strong>{formatNumber(summary.avgCoverage, 3)}</strong>
              </div>
              <div>
                <span>Avg Selected Edges</span>
                <strong>{formatNumber(summary.avgSelectedEdges, 2)}</strong>
              </div>
              <div>
                <span>Counterfactual Paths</span>
                <strong>{summary.counterfactualCount}</strong>
              </div>
              <div>
                <span>Repair Triggered</span>
                <strong>{summary.repairCount}</strong>
              </div>
              <div>
                <span>Fallback Used</span>
                <strong>{summary.fallbackCount}</strong>
              </div>
              <div>
                <span>Router Used</span>
                <strong>{summary.routerCount}</strong>
              </div>
              <div>
                <span>Compression Used</span>
                <strong>{summary.compressionCount}</strong>
              </div>
              <div>
                <span>Error Router Used</span>
                <strong>{summary.legalRouterCount}</strong>
              </div>
              <div>
                <span>Contract Rescue</span>
                <strong>{summary.contractRescueCount}</strong>
              </div>
              <div>
                <span>Graph Rescue</span>
                <strong>{summary.textFirstGraphAccepts}/{summary.textFirstGraphAttempts}</strong>
              </div>
              <div>
                <span>Meta Graph Routes</span>
                <strong>{summary.metaGraphRoutes}</strong>
              </div>
              <div>
                <span>Learned Graph</span>
                <strong>{summary.learnedGraphAccepts}/{summary.learnedGraphAttempts}</strong>
              </div>
              <div>
                <span>Context Budget</span>
                <strong>{summary.selectedContextCount}/{summary.contextCount}</strong>
              </div>
              <div>
                <span>Claim Support</span>
                <strong>{formatNumber(summary.avgClaimSupport, 3)}</strong>
              </div>
              <div>
                <span>Unsupported Claims</span>
                <strong>{summary.unsupportedClaims}</strong>
              </div>
              <div>
                <span>Context Expansion</span>
                <strong>{summary.contextExpansions}</strong>
              </div>
              <div>
                <span>Claim Repairs</span>
                <strong>{summary.claimRepairs}</strong>
              </div>
              <div>
                <span>Contract Checks</span>
                <strong>{summary.contractVerifications}</strong>
              </div>
              <div>
                <span>Contract Repairs</span>
                <strong>{summary.contractRepairs}</strong>
              </div>
              <div>
                <span>Consensus Checks</span>
                <strong>{summary.consensusChecks}</strong>
              </div>
              <div>
                <span>Repair Reverts</span>
                <strong>{summary.consensusReverts}</strong>
              </div>
              <div>
                <span>Selective Checks</span>
                <strong>{summary.selectiveChecks}</strong>
              </div>
              <div>
                <span>Second Views</span>
                <strong>{summary.selectiveEscalations}</strong>
              </div>
              <div>
                <span>Learned Gap Checks</span>
                <strong>{summary.learnedGapChecks}</strong>
              </div>
              <div>
                <span>Verifier Selected</span>
                <strong>{summary.learnedGapSelected}</strong>
              </div>
              <div>
                <span>Delta Guard Checks</span>
                <strong>{summary.evidenceDeltaChecks}</strong>
              </div>
              <div>
                <span>Delta Guard Reverts</span>
                <strong>{summary.evidenceDeltaReverts}</strong>
              </div>
              <div>
                <span>Candidate Arbitration</span>
                <strong>{summary.candidateArbitrations}</strong>
              </div>
              <div>
                <span>Arbiter Graph Picks</span>
                <strong>{summary.arbiterGraphSelections}</strong>
              </div>
              <div>
                <span>Memory Graph Routes</span>
                <strong>{summary.memoryGraphRoutes}</strong>
              </div>
              <div>
                <span>Active Memory Contexts</span>
                <strong>{summary.activeMemoryContexts}</strong>
              </div>
              <div>
                <span>Lazy Graph Attempts</span>
                <strong>{summary.lazyGraphAttempts}</strong>
              </div>
              <div>
                <span>Pre-Graph Repairs</span>
                <strong>{summary.lazyTextRepairs}</strong>
              </div>
              <div>
                <span>Verifier Vetoes</span>
                <strong>{summary.lazyVerifierVetoes}</strong>
              </div>
              <div>
                <span>Text Expansion</span>
                <strong>{summary.textEvidenceAccepted}/{summary.textEvidenceExpansions}</strong>
              </div>
              <div>
                <span>Mismatch Repair</span>
                <strong>{summary.evidenceMismatchAccepted}/{summary.evidenceMismatchAttempts}</strong>
              </div>
              <div>
                <span>Repair Compression</span>
                <strong>{summary.conservativeCompressionAccepted}/{summary.conservativeCompressionAttempts}</strong>
              </div>
              <div>
                <span>Question Anchor</span>
                <strong>{summary.questionAnchoredAccepted}/{summary.questionAnchoredAttempts}</strong>
              </div>
              <div>
                <span>Anchor Reverts</span>
                <strong>{summary.questionAnchoredReverts}</strong>
              </div>
              <div>
                <span>Chunk Completion</span>
                <strong>{summary.questionChunkAccepted}/{summary.questionChunkAttempts}</strong>
              </div>
            </div>

            <div className="trace-layout">
              <aside className="trace-sample-list">
                <div className="trace-section-title">Samples</div>
                {traces.map((trace) => (
                  <button
                    key={trace.line_no}
                    className={`trace-sample-button ${trace.line_no === selectedTrace.line_no ? 'active' : ''}`}
                    onClick={() => setSelectedLineNo(trace.line_no)}
                  >
                    <span>Line {trace.line_no}</span>
                    <small>{trace.question}</small>
                  </button>
                ))}
              </aside>

              <main className="trace-main">
                <section className="trace-panel trace-query-panel">
                  <div className="trace-section-title">Query Analysis</div>
                  <h3>{selectedTrace.question}</h3>
                  <div className="trace-chip-row">
                    <Tag color="blue">{selectedTrace.policy?.mode || 'unknown'}</Tag>
                    <Tag color="green">
                      coverage {formatNumber(selectedTrace.selection_summary?.term_coverage, 3)}
                    </Tag>
                    <Tag color="violet">
                      selected {selectedTrace.selection_summary?.selected_edges || 0}
                    </Tag>
                    <Tag color="orange">
                      max {selectedTrace.policy?.max_edges || 0}
                    </Tag>
                    <Tag color={selectedTrace.sufficiency?.triggered ? 'red' : 'cyan'}>
                      repair {selectedTrace.sufficiency?.triggered ? 'on' : 'off'}
                    </Tag>
                    <Tag color={selectedTrace.uncertainty_fallback?.action?.startsWith('use_') ? 'red' : 'cyan'}>
                      fallback {selectedTrace.uncertainty_fallback?.action?.startsWith('use_') ? 'on' : 'off'}
                    </Tag>
                    <Tag color={selectedTrace.query_router?.action?.startsWith('use_') ? 'red' : 'cyan'}>
                      router {selectedTrace.query_router?.action?.startsWith('use_') ? 'on' : 'off'}
                    </Tag>
                    {selectedTrace.query_router?.question_type && (
                      <Tag color="yellow">{selectedTrace.query_router.question_type}</Tag>
                    )}
                    <Tag color={selectedTrace.rescue_compression?.action === 'use_compressed' ? 'green' : 'cyan'}>
                      compression {selectedTrace.rescue_compression?.action === 'use_compressed' ? 'on' : 'off'}
                    </Tag>
                    <Tag color={selectedTrace.legal_error_router?.action?.startsWith('use_') ? 'red' : 'cyan'}>
                      error-router {selectedTrace.legal_error_router?.action?.startsWith('use_') ? 'on' : 'off'}
                    </Tag>
                    {selectedTrace.cost_aware_prerouter?.route && (
                      <Tag color={selectedTrace.cost_aware_prerouter.requires_graph ? 'orange' : 'green'}>
                        route {selectedTrace.cost_aware_prerouter.route}
                      </Tag>
                    )}
                    {selectedTrace.answer_contract_agent?.contract?.type && (
                      <Tag color={selectedTrace.answer_contract_agent.action?.endsWith('_rescue') ? 'red' : 'green'}>
                        contract {selectedTrace.answer_contract_agent.contract.type}
                      </Tag>
                    )}
                    {selectedTrace.text_first_contract_agent?.contract?.type && (
                      <Tag color={selectedTrace.text_first_contract_agent.graph_rescue_accepted ? 'red' : 'green'}>
                        text-first {selectedTrace.text_first_contract_agent.contract.type}
                      </Tag>
                    )}
                    {selectedTrace.budget_meta_agent?.budget && (
                      <Tag color={selectedTrace.budget_meta_agent.chosen_policy === 'graph_first' ? 'orange' : 'green'}>
                        {selectedTrace.budget_meta_agent.budget}: {selectedTrace.budget_meta_agent.chosen_policy}
                      </Tag>
                    )}
                    {selectedTrace.learned_dual_gate?.enabled && (
                      <Tag color={selectedTrace.learned_dual_gate.graph_accepted ? 'orange' : 'green'}>
                        learned {selectedTrace.learned_dual_gate.chosen_policy}
                      </Tag>
                    )}
                    {selectedTrace.context_budget_agent?.enabled && (
                      <Tag color={selectedTrace.context_budget_agent.context_selected ? 'orange' : 'green'}>
                        context {selectedTrace.context_budget_agent.context_selected ? 'selected' : 'skipped'}
                      </Tag>
                    )}
                    {selectedTrace.claim_evidence_agent?.enabled && (
                      <Tag color={selectedTrace.claim_evidence_agent.repair_applied ? 'red' : 'green'}>
                        claim guard {selectedTrace.claim_evidence_agent.repair_applied ? 'repaired' : 'verified'}
                      </Tag>
                    )}
                    {selectedTrace.contract_completion_agent?.enabled && (
                      <Tag color={selectedTrace.contract_completion_agent.repair_accepted ? 'red' : 'green'}>
                        contract {selectedTrace.contract_completion_agent.repair_accepted ? 'completed' : 'retained'}
                      </Tag>
                    )}
                    {selectedTrace.post_repair_consensus_agent?.enabled && (
                      <Tag color={selectedTrace.post_repair_consensus_agent.final_action === 'use_repair' ? 'green' : 'red'}>
                        consensus {selectedTrace.post_repair_consensus_agent.final_action}
                      </Tag>
                    )}
                    {selectedTrace.selective_consensus_router?.enabled && (
                      <Tag color={selectedTrace.selective_consensus_router.second_view_used ? 'orange' : 'green'}>
                        selective {selectedTrace.selective_consensus_router.second_view_used ? 'escalated' : 'primary'}
                      </Tag>
                    )}
                    {selectedTrace.learned_contract_gap_router?.enabled && (
                      <Tag color={selectedTrace.learned_contract_gap_router.verification_selected ? 'orange' : 'green'}>
                        learned-gap {selectedTrace.learned_contract_gap_router.verification_selected ? 'verify' : 'skip'}
                      </Tag>
                    )}
                    {selectedTrace.text_evidence_expansion_fallback?.enabled && (
                      <Tag color={selectedTrace.text_evidence_expansion_fallback.accepted ? 'green' : 'orange'}>
                        text-expansion {selectedTrace.text_evidence_expansion_fallback.accepted ? 'accepted' : 'checked'}
                      </Tag>
                    )}
                    {selectedTrace.evidence_mismatch_repair?.enabled && (
                      <Tag color={selectedTrace.evidence_mismatch_repair.accepted ? 'green' : 'orange'}>
                        mismatch {selectedTrace.evidence_mismatch_repair.accepted ? 'repaired' : 'checked'}
                      </Tag>
                    )}
                    {selectedTrace.conservative_repair_compression?.enabled && (
                      <Tag color={selectedTrace.conservative_repair_compression.accepted ? 'green' : 'orange'}>
                        compact {selectedTrace.conservative_repair_compression.accepted ? 'accepted' : 'checked'}
                      </Tag>
                    )}
                    {selectedTrace.question_anchored_repair?.enabled && (
                      <Tag color={selectedTrace.question_anchored_repair.accepted ? 'green' : 'red'}>
                        anchored {selectedTrace.question_anchored_repair.accepted ? 'accepted' : 'reverted'}
                      </Tag>
                    )}
                    {selectedTrace.question_chunk_completion?.enabled && (
                      <Tag color={selectedTrace.question_chunk_completion.accepted ? 'green' : 'orange'}>
                        chunk-complete {selectedTrace.question_chunk_completion.accepted ? 'accepted' : 'checked'}
                      </Tag>
                    )}
                  </div>
                  {selectedTrace.uncertainty_fallback?.action?.startsWith('use_') && (
                    <div className="trace-fallback-banner">
                      <strong>{selectedTrace.uncertainty_fallback.action}</strong>
                      <span>
                        {(selectedTrace.uncertainty_fallback.reasons || []).join('; ')}
                      </span>
                    </div>
                  )}
                  {selectedTrace.query_router?.action?.startsWith('use_') && (
                    <div className="trace-router-banner">
                      <strong>{selectedTrace.query_router.action}</strong>
                      <span>
                        {(selectedTrace.query_router.reasons || []).join('; ')}
                      </span>
                    </div>
                  )}
                  {selectedTrace.rescue_compression?.action === 'use_compressed' && (
                    <div className="trace-compression-banner">
                      <strong>{selectedTrace.rescue_compression.action}</strong>
                      <span>
                        {selectedTrace.rescue_compression.original_chars}
                        {' -> '}
                        {selectedTrace.rescue_compression.final_chars}
                        {' chars'}
                      </span>
                    </div>
                  )}
                  {selectedTrace.legal_error_router?.action?.startsWith('use_') && (
                    <div className="trace-error-router-banner">
                      <strong>{selectedTrace.legal_error_router.action}</strong>
                      <span>
                        {selectedTrace.legal_error_router.pattern}
                        {'; '}
                        {(selectedTrace.legal_error_router.reasons || []).join('; ')}
                      </span>
                    </div>
                  )}
                  {selectedTrace.cost_aware_prerouter?.route && (
                    <div className="trace-cost-router-banner">
                      <strong>{selectedTrace.cost_aware_prerouter.route}</strong>
                      <span>
                        estimated {selectedTrace.cost_aware_prerouter.estimated_query_seconds || 0}s
                        {'; graph '}
                        {selectedTrace.cost_aware_prerouter.requires_graph ? 'required' : 'skipped'}
                      </span>
                      <span>
                        {(selectedTrace.cost_aware_prerouter.reasons || []).join('; ')}
                      </span>
                    </div>
                  )}
                  {selectedTrace.answer_contract_agent?.contract?.type && (
                    <div className="trace-router-banner">
                      <strong>{selectedTrace.answer_contract_agent.action}</strong>
                      <span>
                        {selectedTrace.answer_contract_agent.contract.type}
                        {'; coverage '}
                        {formatNumber(selectedTrace.answer_contract_agent.verification?.contract_coverage, 3)}
                        {'; grounding '}
                        {formatNumber(selectedTrace.answer_contract_agent.verification?.grounding_score, 3)}
                      </span>
                      {(selectedTrace.answer_contract_agent.verification?.missing_facets || []).length > 0 && (
                        <span>
                          missing: {selectedTrace.answer_contract_agent.verification.missing_facets.join(', ')}
                        </span>
                      )}
                    </div>
                  )}
                  {selectedTrace.text_first_contract_agent?.contract?.type && (
                    <div className="trace-router-banner">
                      <strong>{selectedTrace.text_first_contract_agent.action}</strong>
                      <span>
                        {selectedTrace.text_first_contract_agent.contract.type}
                        {'; graph attempted '}
                        {selectedTrace.text_first_contract_agent.graph_rescue_attempted ? 'yes' : 'no'}
                        {'; accepted '}
                        {selectedTrace.text_first_contract_agent.graph_rescue_accepted ? 'yes' : 'no'}
                      </span>
                      {selectedTrace.text_first_contract_agent.rescue_utility_guard?.enabled && (
                        <span>
                          utility gain {formatNumber(
                            selectedTrace.text_first_contract_agent.rescue_utility_guard.utility_gain,
                            3
                          )}
                          {'; length ratio '}
                          {formatNumber(
                            selectedTrace.text_first_contract_agent.rescue_utility_guard.length_ratio,
                            2
                          )}
                        </span>
                      )}
                    </div>
                  )}
                  {selectedTrace.budget_meta_agent?.budget && (
                    <div className="trace-cost-router-banner">
                      <strong>{selectedTrace.budget_meta_agent.chosen_policy}</strong>
                      <span>
                        budget {selectedTrace.budget_meta_agent.budget}
                        {'; estimated '}
                        {formatNumber(selectedTrace.budget_meta_agent.estimated_query_seconds, 3)}s
                        {'; graph '}
                        {selectedTrace.budget_meta_agent.graph_required ? 'required' : 'skipped'}
                      </span>
                      <span>{selectedTrace.budget_meta_agent.reason}</span>
                    </div>
                  )}
                  {selectedTrace.learned_dual_gate?.enabled && (
                    <div className="trace-cost-router-banner">
                      <strong>{selectedTrace.learned_dual_gate.chosen_policy}</strong>
                      <span>
                        pre {formatNumber(selectedTrace.learned_dual_gate.pre_graph_probability, 3)}
                        {' / '}
                        {formatNumber(selectedTrace.learned_dual_gate.pre_threshold, 3)}
                        {'; post '}
                        {formatNumber(selectedTrace.learned_dual_gate.post_accept_probability, 3)}
                        {' / '}
                        {formatNumber(selectedTrace.learned_dual_gate.post_threshold, 3)}
                      </span>
                      <span>
                        attempted {selectedTrace.learned_dual_gate.graph_attempted ? 'yes' : 'no'}
                        {'; accepted '}
                        {selectedTrace.learned_dual_gate.graph_accepted ? 'yes' : 'no'}
                        {'; safety '}
                        {selectedTrace.learned_dual_gate.learned_safety_passed ? 'passed' : 'not used'}
                      </span>
                    </div>
                  )}
                  {selectedTrace.context_budget_agent?.enabled && (
                    <div className="trace-cost-router-banner">
                      <strong>{selectedTrace.context_budget_agent.final_policy}</strong>
                      <span>
                        budget {selectedTrace.context_budget_agent.budget}
                        {'; context '}
                        {selectedTrace.context_budget_agent.context_hash}
                        {'; selected '}
                        {selectedTrace.context_budget_agent.context_selected ? 'yes' : 'no'}
                      </span>
                      <span>
                        value {formatNumber(selectedTrace.context_budget_agent.context_value, 3)}
                        {'; cost '}
                        {formatNumber(selectedTrace.context_budget_agent.context_cost, 3)}s
                        {'; density '}
                        {formatNumber(selectedTrace.context_budget_agent.context_density, 6)}
                      </span>
                    </div>
                  )}
                  {selectedTrace.text_evidence_expansion_fallback?.enabled && (
                    <div className="trace-cost-router-banner">
                      <strong>{selectedTrace.text_evidence_expansion_fallback.action}</strong>
                      <span>
                        full-text windows {selectedTrace.text_evidence_expansion_fallback.evidence_windows || 0}
                        {'; selected '}
                        {(selectedTrace.text_evidence_expansion_fallback.selected_evidence || []).length}
                        {'; accepted '}
                        {selectedTrace.text_evidence_expansion_fallback.accepted ? 'yes' : 'no'}
                      </span>
                    </div>
                  )}
                  {selectedTrace.evidence_mismatch_repair?.enabled && (
                    <div className="trace-cost-router-banner">
                      <strong>{selectedTrace.evidence_mismatch_repair.action}</strong>
                      <span>
                        arbiter {selectedTrace.evidence_mismatch_repair.guard?.arbiter?.winner || 'tie'}
                        {'; confidence '}
                        {formatNumber(selectedTrace.evidence_mismatch_repair.guard?.arbiter?.confidence, 2)}
                        {'; accepted '}
                        {selectedTrace.evidence_mismatch_repair.accepted ? 'yes' : 'no'}
                      </span>
                    </div>
                  )}
                  {selectedTrace.conservative_repair_compression?.enabled && (
                    <div className="trace-cost-router-banner">
                      <strong>{selectedTrace.conservative_repair_compression.action}</strong>
                      <span>
                        chars {selectedTrace.conservative_repair_compression.guard?.original_chars || 0}
                        {' -> '}
                        {selectedTrace.conservative_repair_compression.guard?.compressed_chars || 0}
                        {'; accepted '}
                        {selectedTrace.conservative_repair_compression.accepted ? 'yes' : 'no'}
                      </span>
                    </div>
                  )}
                  {selectedTrace.question_anchored_repair?.enabled && (
                    <div className="trace-cost-router-banner">
                      <strong>{selectedTrace.question_anchored_repair.action}</strong>
                      <span>
                        term coverage {formatNumber(
                          selectedTrace.question_anchored_repair.guard?.question_term_coverage,
                          2
                        )}
                        {'; windows '}
                        {selectedTrace.question_anchored_repair.evidence_windows || 0}
                        {'; accepted '}
                        {selectedTrace.question_anchored_repair.accepted ? 'yes' : 'no'}
                      </span>
                    </div>
                  )}
                  {selectedTrace.question_chunk_completion?.enabled && (
                    <div className="trace-cost-router-banner">
                      <strong>{selectedTrace.question_chunk_completion.action}</strong>
                      <span>
                        evidence {(selectedTrace.question_chunk_completion.selected_evidence || []).length}
                        {'; accepted '}
                        {selectedTrace.question_chunk_completion.accepted ? 'yes' : 'no'}
                        {'; utility '}
                        {selectedTrace.question_chunk_completion.guard?.checks?.positive_utility ? 'positive' : 'blocked'}
                      </span>
                    </div>
                  )}
                  <div className="trace-term-block">
                    <span>Covered terms</span>
                    <div>
                      {(selectedTrace.selection_summary?.covered_terms || []).slice(0, 18).map((term) => (
                        <Tag key={term} size="sm">{term}</Tag>
                      ))}
                    </div>
                  </div>
                  <div className="trace-term-block">
                    <span>Missing terms</span>
                    <div>
                      {(selectedTrace.selection_summary?.missing_terms || []).slice(0, 18).map((term) => (
                        <Tag key={term} size="sm" color="red">{term}</Tag>
                      ))}
                    </div>
                  </div>
                  {(selectedTrace.answer_aspects || []).length > 0 && (
                    <div className="trace-term-block">
                      <span>Answer aspects</span>
                      <div>
                        {(selectedTrace.answer_aspects || []).map((aspect) => (
                          <Tag key={aspect.label} size="sm" color="orange">
                            {aspect.label}
                          </Tag>
                        ))}
                      </div>
                    </div>
                  )}
                  {selectedTrace.aspect_preservation?.added_edges > 0 && (
                    <div className="trace-term-block">
                      <span>Aspect preserved</span>
                      <div>
                        {(selectedTrace.aspect_preservation?.added || []).map((item) => (
                          <Tag key={`${item.aspect}-${item.source}-${item.target}`} size="sm" color="green">
                            {item.aspect}: {item.source} -> {item.target}
                          </Tag>
                        ))}
                      </div>
                    </div>
                  )}
                </section>

                <section className="trace-panel">
                  <div className="trace-section-title">Answer</div>
                  <p className="trace-answer-text">{selectedTrace.prediction}</p>
                  {selectedTrace.original_prediction && (
                    <div className="trace-term-block">
                      <span>Original graph answer before fallback</span>
                      <p className="trace-original-answer">{selectedTrace.original_prediction}</p>
                    </div>
                  )}
                  {selectedTrace.graph_prediction_before_router && (
                    <div className="trace-term-block">
                      <span>Graph answer before query router</span>
                      <p className="trace-original-answer">{selectedTrace.graph_prediction_before_router}</p>
                    </div>
                  )}
                  {selectedTrace.prediction_before_candidate_router && (
                    <div className="trace-term-block">
                      <span>Answer before legal error router</span>
                      <p className="trace-original-answer">{selectedTrace.prediction_before_candidate_router}</p>
                    </div>
                  )}
                  {selectedTrace.prediction_before_rescue && (
                    <div className="trace-term-block">
                      <span>Answer before graph rescue</span>
                      <p className="trace-original-answer">{selectedTrace.prediction_before_rescue}</p>
                    </div>
                  )}
                  {selectedTrace.prediction_before_compression && (
                    <div className="trace-term-block">
                      <span>Answer before rescue compression</span>
                      <p className="trace-original-answer">{selectedTrace.prediction_before_compression}</p>
                    </div>
                  )}
                  {selectedTrace.prediction_before_answer_contract_rescue && (
                    <div className="trace-term-block">
                      <span>Answer before contract rescue</span>
                      <p className="trace-original-answer">
                        {selectedTrace.prediction_before_answer_contract_rescue}
                      </p>
                    </div>
                  )}
                  {selectedTrace.prediction_before_text_first_graph_rescue && (
                    <div className="trace-term-block">
                      <span>Text answer before accepted graph rescue</span>
                      <p className="trace-original-answer">
                        {selectedTrace.prediction_before_text_first_graph_rescue}
                      </p>
                    </div>
                  )}
                  {selectedTrace.rejected_graph_rescue_prediction && (
                    <div className="trace-term-block">
                      <span>Rejected graph rescue answer</span>
                      <p className="trace-original-answer">
                        {selectedTrace.rejected_graph_rescue_prediction}
                      </p>
                    </div>
                  )}
                  {selectedTrace.prediction_before_budget_meta_route && (
                    <div className="trace-term-block">
                      <span>Text-policy answer before budget meta route</span>
                      <p className="trace-original-answer">
                        {selectedTrace.prediction_before_budget_meta_route}
                      </p>
                    </div>
                  )}
                  {selectedTrace.prediction_before_learned_graph_accept && (
                    <div className="trace-term-block">
                      <span>Text answer before learned graph acceptance</span>
                      <p className="trace-original-answer">
                        {selectedTrace.prediction_before_learned_graph_accept}
                      </p>
                    </div>
                  )}
                  {selectedTrace.rejected_learned_graph_prediction && (
                    <div className="trace-term-block">
                      <span>Graph answer rejected by learned safety gate</span>
                      <p className="trace-original-answer">
                        {selectedTrace.rejected_learned_graph_prediction}
                      </p>
                    </div>
                  )}
                  {selectedTrace.suppressed_graph_prediction_by_context_budget && (
                    <div className="trace-term-block">
                      <span>Graph answer suppressed by context budget</span>
                      <p className="trace-original-answer">
                        {selectedTrace.suppressed_graph_prediction_by_context_budget}
                      </p>
                    </div>
                  )}
                  {selectedTrace.prediction_before_claim_guard && (
                    <div className="trace-term-block">
                      <span>Answer before claim-evidence repair</span>
                      <p className="trace-original-answer">{selectedTrace.prediction_before_claim_guard}</p>
                    </div>
                  )}
                  {selectedTrace.prediction_before_contract_completion && (
                    <div className="trace-term-block">
                      <span>Answer before contract completion</span>
                      <p className="trace-original-answer">{selectedTrace.prediction_before_contract_completion}</p>
                    </div>
                  )}
                  {selectedTrace.rejected_contract_completion_prediction && (
                    <div className="trace-term-block">
                      <span>Contract completion rejected by conservative policy</span>
                      <p className="trace-original-answer">{selectedTrace.rejected_contract_completion_prediction}</p>
                    </div>
                  )}
                  {selectedTrace.prediction_before_postrepair_consensus && (
                    <div className="trace-term-block">
                      <span>Repair rejected by post-repair consensus</span>
                      <p className="trace-original-answer">{selectedTrace.prediction_before_postrepair_consensus}</p>
                    </div>
                  )}
                  {selectedTrace.prediction_before_selective_consensus && (
                    <div className="trace-term-block">
                      <span>Repair rejected by selective consensus</span>
                      <p className="trace-original-answer">{selectedTrace.prediction_before_selective_consensus}</p>
                    </div>
                  )}
                  {selectedTrace.suppressed_contract_completion_prediction && (
                    <div className="trace-term-block">
                      <span>Contract completion suppressed by learned gate</span>
                      <p className="trace-original-answer">{selectedTrace.suppressed_contract_completion_prediction}</p>
                    </div>
                  )}
                  {selectedTrace.rejected_evidence_delta_prediction && (
                    <div className="trace-term-block">
                      <span>Repair rejected by independent evidence-delta guard</span>
                      <p className="trace-original-answer">{selectedTrace.rejected_evidence_delta_prediction}</p>
                    </div>
                  )}
                  {selectedTrace.prediction_before_candidate_arbitration && (
                    <div className="trace-term-block">
                      <span>Text answer before candidate arbitration</span>
                      <p className="trace-original-answer">{selectedTrace.prediction_before_candidate_arbitration}</p>
                    </div>
                  )}
                  {selectedTrace.prediction_before_context_graph_memory && (
                    <div className="trace-term-block">
                      <span>Answer before context graph-memory route</span>
                      <p className="trace-original-answer">{selectedTrace.prediction_before_context_graph_memory}</p>
                    </div>
                  )}
                  {selectedTrace.prediction_before_text_evidence_expansion_fallback && (
                    <div className="trace-term-block">
                      <span>Answer before text-evidence expansion fallback</span>
                      <p className="trace-original-answer">
                        {selectedTrace.prediction_before_text_evidence_expansion_fallback}
                      </p>
                    </div>
                  )}
                  {selectedTrace.prediction_before_evidence_mismatch_repair && (
                    <div className="trace-term-block">
                      <span>Answer before evidence-mismatch repair</span>
                      <p className="trace-original-answer">
                        {selectedTrace.prediction_before_evidence_mismatch_repair}
                      </p>
                    </div>
                  )}
                  {selectedTrace.rejected_evidence_mismatch_repair_prediction && (
                    <div className="trace-term-block">
                      <span>Evidence-mismatch repair rejected</span>
                      <p className="trace-original-answer">
                        {selectedTrace.rejected_evidence_mismatch_repair_prediction}
                      </p>
                    </div>
                  )}
                  {selectedTrace.prediction_before_conservative_repair_compression && (
                    <div className="trace-term-block">
                      <span>Answer before conservative repair compression</span>
                      <p className="trace-original-answer">
                        {selectedTrace.prediction_before_conservative_repair_compression}
                      </p>
                    </div>
                  )}
                  {selectedTrace.rejected_conservative_repair_compression && (
                    <div className="trace-term-block">
                      <span>Conservative compression rejected</span>
                      <p className="trace-original-answer">
                        {selectedTrace.rejected_conservative_repair_compression}
                      </p>
                    </div>
                  )}
                  {selectedTrace.prediction_before_question_anchored_repair && (
                    <div className="trace-term-block">
                      <span>Answer before question-anchored repair</span>
                      <p className="trace-original-answer">
                        {selectedTrace.prediction_before_question_anchored_repair}
                      </p>
                    </div>
                  )}
                  {selectedTrace.prediction_before_question_chunk_completion && (
                    <div className="trace-term-block">
                      <span>Answer before question-chunk completion</span>
                      <p className="trace-original-answer">
                        {selectedTrace.prediction_before_question_chunk_completion}
                      </p>
                    </div>
                  )}
                  {selectedTrace.rejected_question_chunk_completion_prediction && (
                    <div className="trace-term-block">
                      <span>Question-chunk completion rejected</span>
                      <p className="trace-original-answer">
                        {selectedTrace.rejected_question_chunk_completion_prediction}
                      </p>
                    </div>
                  )}
                  {selectedTrace.prediction_before_question_completion_compression && (
                    <div className="trace-term-block">
                      <span>Answer before fact-preserving compression</span>
                      <p className="trace-original-answer">
                        {selectedTrace.prediction_before_question_completion_compression}
                      </p>
                    </div>
                  )}
                  {selectedTrace.rejected_question_completion_compression && (
                    <div className="trace-term-block">
                      <span>Fact-preserving compression rejected</span>
                      <p className="trace-original-answer">
                        {selectedTrace.rejected_question_completion_compression}
                      </p>
                    </div>
                  )}
                  {selectedTrace.prediction_before_direct_fact_compression && (
                    <div className="trace-term-block">
                      <span>Answer before direct fact compression</span>
                      <p className="trace-original-answer">
                        {selectedTrace.prediction_before_direct_fact_compression}
                      </p>
                    </div>
                  )}
                  {selectedTrace.rejected_direct_fact_compression && (
                    <div className="trace-term-block">
                      <span>Direct fact compression rejected</span>
                      <p className="trace-original-answer">
                        {selectedTrace.rejected_direct_fact_compression}
                      </p>
                    </div>
                  )}
                  {selectedTrace.prediction_before_self_corrected_fact_compression && (
                    <div className="trace-term-block">
                      <span>Answer before self-corrected compression</span>
                      <p className="trace-original-answer">
                        {selectedTrace.prediction_before_self_corrected_fact_compression}
                      </p>
                    </div>
                  )}
                  {selectedTrace.rejected_self_corrected_fact_compression && (
                    <div className="trace-term-block">
                      <span>Self-corrected compression rejected</span>
                      <p className="trace-original-answer">
                        {selectedTrace.rejected_self_corrected_fact_compression}
                      </p>
                    </div>
                  )}
                  {selectedTrace.prediction_before_query_expanded_refusal_repair && (
                    <div className="trace-term-block">
                      <span>Answer before query-expanded refusal repair</span>
                      <p className="trace-original-answer">
                        {selectedTrace.prediction_before_query_expanded_refusal_repair}
                      </p>
                    </div>
                  )}
                  {selectedTrace.rejected_query_expanded_refusal_repair && (
                    <div className="trace-term-block">
                      <span>Query-expanded refusal repair rejected</span>
                      <p className="trace-original-answer">
                        {selectedTrace.rejected_query_expanded_refusal_repair}
                      </p>
                    </div>
                  )}
                  {selectedTrace.prediction_before_substance_refusal_repair && (
                    <div className="trace-term-block">
                      <span>Answer before substance refusal repair</span>
                      <p className="trace-original-answer">
                        {selectedTrace.prediction_before_substance_refusal_repair}
                      </p>
                    </div>
                  )}
                  {selectedTrace.rejected_substance_refusal_repair && (
                    <div className="trace-term-block">
                      <span>Substance refusal repair rejected</span>
                      <p className="trace-original-answer">
                        {selectedTrace.rejected_substance_refusal_repair}
                      </p>
                    </div>
                  )}
                  {selectedTrace.prediction_before_query_expanded_contract_gap_repair && (
                    <div className="trace-term-block">
                      <span>Answer before query-expanded contract-gap repair</span>
                      <p className="trace-original-answer">
                        {selectedTrace.prediction_before_query_expanded_contract_gap_repair}
                      </p>
                    </div>
                  )}
                  {selectedTrace.rejected_query_expanded_contract_gap_repair && (
                    <div className="trace-term-block">
                      <span>Query-expanded contract-gap repair rejected</span>
                      <p className="trace-original-answer">
                        {selectedTrace.rejected_query_expanded_contract_gap_repair}
                      </p>
                    </div>
                  )}
                  {selectedTrace.prediction_before_supported_facet_guard && (
                    <div className="trace-term-block">
                      <span>Repair candidate reverted by supported-facet guard</span>
                      <p className="trace-original-answer">
                        {selectedTrace.prediction_before_supported_facet_guard}
                      </p>
                    </div>
                  )}
                  {selectedTrace.prediction_before_supported_facet_fusion && (
                    <div className="trace-term-block">
                      <span>Answer before supported-facet fusion</span>
                      <p className="trace-original-answer">
                        {selectedTrace.prediction_before_supported_facet_fusion}
                      </p>
                    </div>
                  )}
                  {selectedTrace.rejected_supported_facet_fusion && (
                    <div className="trace-term-block">
                      <span>Supported-facet fusion rejected</span>
                      <p className="trace-original-answer">
                        {selectedTrace.rejected_supported_facet_fusion}
                      </p>
                    </div>
                  )}
                  {selectedTrace.prediction_before_facet_balanced_compaction && (
                    <div className="trace-term-block">
                      <span>Answer before facet-balanced compaction</span>
                      <p className="trace-original-answer">
                        {selectedTrace.prediction_before_facet_balanced_compaction}
                      </p>
                    </div>
                  )}
                  {selectedTrace.rejected_facet_balanced_compaction && (
                    <div className="trace-term-block">
                      <span>Facet-balanced compaction rejected</span>
                      <p className="trace-original-answer">
                        {selectedTrace.rejected_facet_balanced_compaction}
                      </p>
                    </div>
                  )}
                  {selectedTrace.prediction_before_residual_hardcase_editor && (
                    <div className="trace-term-block">
                      <span>Answer before residual hard-case editor</span>
                      <p className="trace-original-answer">
                        {selectedTrace.prediction_before_residual_hardcase_editor}
                      </p>
                    </div>
                  )}
                  {selectedTrace.prediction_before_evidence_risk_gate && (
                    <div className="trace-term-block">
                      <span>Candidate before evidence risk gate</span>
                      <p className="trace-original-answer">
                        {selectedTrace.prediction_before_evidence_risk_gate}
                      </p>
                    </div>
                  )}
                  {selectedTrace.rejected_evidence_risk_gate_prediction && (
                    <div className="trace-term-block">
                      <span>Evidence risk gate reverted candidate</span>
                      <p className="trace-original-answer">
                        {selectedTrace.rejected_evidence_risk_gate_prediction}
                      </p>
                    </div>
                  )}
                  {selectedTrace.prediction_before_dual_objective_gate && (
                    <div className="trace-term-block">
                      <span>Candidate before dual-objective gate</span>
                      <p className="trace-original-answer">
                        {selectedTrace.prediction_before_dual_objective_gate}
                      </p>
                    </div>
                  )}
                  {selectedTrace.rejected_dual_objective_gate_prediction && (
                    <div className="trace-term-block">
                      <span>Dual-objective gate reverted candidate</span>
                      <p className="trace-original-answer">
                        {selectedTrace.rejected_dual_objective_gate_prediction}
                      </p>
                    </div>
                  )}
                  {selectedTrace.prediction_before_kg_rag_agent && (
                    <div className="trace-term-block">
                      <span>Answer entering KG-RAG agent controller</span>
                      <p className="trace-original-answer">
                        {selectedTrace.prediction_before_kg_rag_agent}
                      </p>
                    </div>
                  )}
                </section>

                <ClaimEvidence agent={selectedTrace.claim_evidence_agent} />
                <ContractCompletion agent={selectedTrace.contract_completion_agent} />
                <PostRepairConsensus agent={selectedTrace.post_repair_consensus_agent} />
                <SelectiveConsensus router={selectedTrace.selective_consensus_router} />
                <LearnedContractGate router={selectedTrace.learned_contract_gap_router} />
                <EvidenceDeltaGuard guard={selectedTrace.evidence_delta_guard} />
                <CandidateArbiter agent={selectedTrace.evidence_candidate_arbiter} />
                <ContextGraphMemory agent={selectedTrace.context_graph_memory_agent} />
                <EvidenceCalibratedLazyAgent
                  agent={selectedTrace.evidence_calibrated_lazy_agent}
                  plan={selectedTrace.v41_pregraph_plan || selectedTrace.v39_pregraph_plan}
                />
                <TextEvidenceExpansionFallback agent={selectedTrace.text_evidence_expansion_fallback} />
                <EvidenceMismatchRepair agent={selectedTrace.evidence_mismatch_repair} />
                <ConservativeRepairCompression agent={selectedTrace.conservative_repair_compression} />
                <QuestionAnchoredRepair agent={selectedTrace.question_anchored_repair} />
                <QuestionChunkCompletion agent={selectedTrace.question_chunk_completion} />
                <CompressionGuardPanel
                  agent={selectedTrace.question_completion_compression}
                  title="Fact-Preserving Completion Compression"
                />
                <CompressionGuardPanel
                  agent={selectedTrace.direct_fact_compression}
                  title="Direct Fact Compression"
                />
                <CompressionGuardPanel
                  agent={selectedTrace.self_corrected_fact_compression}
                  title="Self-Corrected Fact Compression"
                />
                <RefusalRepairPanel
                  agent={selectedTrace.query_expanded_refusal_repair}
                  title="Query-Expanded Refusal Repair"
                  subtitle="question-only expansion, full-source evidence search, no reference answer"
                />
                <RefusalRepairPanel
                  agent={selectedTrace.substance_refusal_repair}
                  title="Substance-Oriented Refusal Repair"
                  subtitle="answers the substantive question from relevant evidence instead of repeating abstention"
                />
                <ContractGapRepairPanel agent={selectedTrace.query_expanded_contract_gap_repair} />
                <SupportedFacetGuardPanel guard={selectedTrace.supported_facet_guard} />
                <SupportedFacetFusionPanel agent={selectedTrace.supported_facet_fusion} />
                <CompressionGuardPanel
                  agent={selectedTrace.facet_balanced_compaction}
                  title="Facet-Balanced Compaction"
                />
                <ResidualHardcaseEditorPanel agent={selectedTrace.residual_hardcase_editor} />
                <EvidenceGatePanel
                  gate={selectedTrace.evidence_risk_gate}
                  title="Evidence Risk Gate"
                />
                <EvidenceGatePanel
                  gate={selectedTrace.dual_objective_gate}
                  title="Dual-Objective Evidence Gate"
                />
                <AgentControllerPanel agent={selectedTrace.kg_rag_agent} />

                <section className="trace-panel">
                  <div className="trace-section-title">Selected Evidence Paths</div>
                  {(selectedTrace.selected_edges || []).map((edge) => (
                    <EvidenceEdge key={`${edge.rank}-${edge.path}`} edge={edge} />
                  ))}
                </section>
              </main>

              <aside className="trace-side">
                <section className="trace-panel">
                  <div className="trace-section-title">Sufficiency Repair</div>
                  <div className="trace-repair-card">
                    <div>
                      <span>Status</span>
                      <strong>{selectedTrace.sufficiency?.after?.status || 'not checked'}</strong>
                    </div>
                    <div>
                      <span>Score</span>
                      <strong>{formatNumber(selectedTrace.sufficiency?.after?.score, 3)}</strong>
                    </div>
                    <div>
                      <span>Coverage</span>
                      <strong>
                        {formatNumber(selectedTrace.sufficiency?.coverage_before, 3)}
                        {' -> '}
                        {formatNumber(selectedTrace.sufficiency?.coverage_after, 3)}
                      </strong>
                    </div>
                    <div>
                      <span>Added Edges</span>
                      <strong>{selectedTrace.sufficiency?.added_edges || 0}</strong>
                    </div>
                  </div>
                  {(selectedTrace.sufficiency?.before?.reasons || []).length > 0 && (
                    <div className="trace-term-block">
                      <span>Trigger reasons</span>
                      <div>
                        {selectedTrace.sufficiency.before.reasons.map((reason) => (
                          <Tag key={reason} size="sm" color="orange">{reason}</Tag>
                        ))}
                      </div>
                    </div>
                  )}
                  {(selectedTrace.sufficiency?.actions || []).length > 0 && (
                    <div className="trace-repair-actions">
                      {selectedTrace.sufficiency.actions.map((action, index) => (
                        <div key={`${action.path}-${index}`}>
                          <strong>{action.path}</strong>
                          <div>
                            {(action.new_terms || []).map((term) => (
                              <Tag key={term} size="sm" color="green">{term}</Tag>
                            ))}
                          </div>
                        </div>
                      ))}
                    </div>
                  )}
                </section>

                <section className="trace-panel">
                  <div className="trace-section-title">Top Candidates</div>
                  {(selectedTrace.top_candidates || []).slice(0, 5).map((edge) => (
                    <EvidenceEdge key={`${edge.rank}-${edge.path}`} edge={edge} muted={!edge.selected} />
                  ))}
                </section>

                <section className="trace-panel">
                  <div className="trace-section-title">Filtered Counterfactual Paths</div>
                  {(selectedTrace.filtered_counterfactual_edges || []).length === 0 ? (
                    <p className="trace-muted-text">No penalized distracting path for this sample.</p>
                  ) : (
                    selectedTrace.filtered_counterfactual_edges.slice(0, 5).map((edge) => (
                      <EvidenceEdge key={`${edge.rank}-${edge.path}`} edge={edge} muted />
                    ))
                  )}
                </section>

                <section className="trace-panel">
                  <div className="trace-section-title">Supporting Chunks</div>
                  {(selectedTrace.supporting_chunks || []).slice(0, 4).map((chunk) => (
                    <div className="trace-chunk" key={chunk.chunk_id}>
                      <div>
                        <strong>Chunk {chunk.rank}</strong>
                        <span>{chunk.term_overlap} terms</span>
                      </div>
                      <p>{chunk.snippet}</p>
                    </div>
                  ))}
                </section>
              </aside>
            </div>
          </>
        )}
      </div>
    </Layout>
  );
};

export default EvidenceTracePage;
