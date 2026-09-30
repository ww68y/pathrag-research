import React, { useState, useEffect, useRef, useCallback } from 'react';
import { Loader, Panel, Grid, Row, Col } from 'rsuite';
import { useParams, useNavigate } from 'react-router-dom';
import Layout from '../components/Layout';
import ChatMessage from '../components/chat/ChatMessage';
import ChatInput from '../components/chat/ChatInput';
import AgentStatusPanel from '../components/agent/AgentStatusPanel';
import RealTimeDetector from '../components/agent/RealTimeDetector';
import { chatAPI } from '../services/api';
import { saveThreadUUID, getThreadUUID } from '../utils/threadStorage';
import '../styles/agent-chat.css';

// ─── 真实处理步骤标签映射 ───
// 后端 agent_controller.py 中 add_step 的 tool 名称 → 前端展示名称
const STEP_LABEL_MAP = {
  'question_analyzer': { name: 'Query Analysis', description: 'Classifying question type and extracting routing hints' },
  'initial_profile_selector': { name: 'Profile Selection', description: 'Choosing initial retrieval profile' },
  'graph_evidence_probe': { name: 'Evidence Retrieval', description: 'Collecting runtime evidence from knowledge graph' },
  'query_graph_planner': { name: 'Query Planning', description: 'Building multi-hop query plan' },
  'knowledge_gap_locator': { name: 'Gap Detection', description: 'Identifying knowledge gaps' },
  'source_conflict_graph_builder': { name: 'Conflict Detection', description: 'Detecting source conflicts' },
  'compute_budget_planner': { name: 'Budget Planning', description: 'Allocating compute budget' },
  'active_evidence_acquisition': { name: 'Active Evidence', description: 'Acquiring additional evidence' },
  'hierarchical_answer_realization': { name: 'Answer Draft', description: 'Generating hierarchical answer' },
  'claim_active_verification': { name: 'Claim Verification', description: 'Verifying claims against evidence' },
  'diagnosis_answer_correction': { name: 'Diagnosis', description: 'Diagnosing and correcting answer issues' },
  'evidence_repair_overlay': { name: 'Evidence Repair', description: 'Repairing evidence gaps' },
  'evidence_portfolio_optimizer': { name: 'Portfolio Optimization', description: 'Optimizing evidence portfolio' },
  'cross_query_evidence_memory': { name: 'Cross-Query Memory', description: 'Checking historical evidence cache' },
  'adversarial_robustness_defense': { name: 'Adversarial Defense', description: 'Validating evidence authenticity' },
  'multimodal_evidence_fusion': { name: 'Multi-Modal Fusion', description: 'Integrating table/figure/code evidence' },
  'realtime_evidence_stream': { name: 'Stream Processing', description: 'Checking evidence freshness' },
  'collaborative_reasoning': { name: 'Collaborative Reasoning', description: 'Multi-agent collaborative analysis' },
  'uncertainty_quantification': { name: 'Uncertainty', description: 'Quantifying answer uncertainty' },
  'personalized_adaptation': { name: 'Personalization', description: 'Adapting response to user profile' },
  'signed_claim_evidence_graph': { name: 'Claim Safety', description: 'Building signed claim-evidence graph' },
  'minimal_proof_subgraph_optimizer': { name: 'Proof Subgraph', description: 'Extracting minimal proof subgraph' },
  'proof_completion_backfill': { name: 'Proof Backfill', description: 'Completing proof gaps' },
  'counterfactual_path_necessity': { name: 'Path Necessity', description: 'Certifying counterfactual path necessity' },
  'proof_source_quorum': { name: 'Source Quorum', description: 'Certifying proof source quorum' },
  'obligation_source_rescue': { name: 'Source Rescue', description: 'Rescuing obligation sources' },
  'obligation_criticality_quorum': { name: 'Obligation Quorum', description: 'Normalizing obligation criticality' },
  'risk_bounded_terminal_controller': { name: 'Risk Assessment', description: 'Applying terminal risk policy' },
  'evidence_audit_card': { name: 'Evidence Audit', description: 'Auditing evidence quality' },
  'final_answer_controller': { name: 'Answer Generation', description: 'Committing final answer' },
  // V136-V143 创新模块
  'v136_multimodal_fusion': { name: 'V136 Multimodal Fusion', description: 'Extracting tables, formulas, and code from evidence' },
  'v137_temporal_reasoning': { name: 'V137 Temporal Reasoning', description: 'Building timeline and causal chains' },
  'v138_cross_lingual_alignment': { name: 'V138 Cross-Lingual', description: 'Aligning bilingual entities' },
  'v139_privacy_preserving': { name: 'V139 Privacy Protection', description: 'Detecting PII and anonymizing' },
  'v140_explainability_dashboard': { name: 'V140 Explainability', description: 'Building evidence tree and reasoning path' },
  'v141_rl_optimizer_v2': { name: 'V141 RL Optimizer', description: 'Selecting optimal action via PPO policy' },
  'v143_edge_deployment': { name: 'V143 Edge Deployment', description: 'Detecting system resources and optimizing config' },
};

// ─── 从后端 trace.steps 构建前端展示步骤 ───
function buildProcessingStepsFromTrace(traceSteps) {
  if (!traceSteps || !Array.isArray(traceSteps)) return [];

  return traceSteps.map((step, index) => {
    const tool = step.tool || '';
    const labelInfo = STEP_LABEL_MAP[tool] || {
      name: tool.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase()),
      description: step.observation || '',
    };

    return {
      name: labelInfo.name,
      description: labelInfo.description,
      tool,
      decision: step.decision || '',
      observation: step.observation || '',
      progress: 100,
      index,
    };
  });
}

// ─── 从后端 trace 提取真实指标 ───
function extractMetricsFromTrace(trace) {
  if (!trace) return null;

  const runtimeSeconds = trace.runtime_seconds || 0;
  const candidateCount = trace.candidate_count || 0;
  const evidenceSummary = trace.evidence_summary || {};
  const entityCount = (evidenceSummary.entities || []).length;
  const relationshipCount = (evidenceSummary.relationships || []).length;
  const graphNodes = evidenceSummary.graph_nodes || 0;
  const graphEdges = evidenceSummary.graph_edges || 0;
  const claimVerification = trace.claim_verification || {};
  const facetCoverage = trace.facet_coverage || {};
  const structuredConfidence = trace.structured_confidence || {};
  const steps = trace.steps || [];

  // 统计 V136-V143 模块执行情况
  const v136Executed = steps.some(s => s.tool === 'v136_multimodal_fusion' && s.decision === 'executed');
  const v137Executed = steps.some(s => s.tool === 'v137_temporal_reasoning' && s.decision === 'executed');
  const v138Executed = steps.some(s => s.tool === 'v138_cross_lingual_alignment' && s.decision === 'executed');
  const v139Executed = steps.some(s => s.tool === 'v139_privacy_preserving' && s.decision === 'executed');
  const v140Executed = steps.some(s => s.tool === 'v140_explainability_dashboard' && s.decision === 'executed');
  const v141Executed = steps.some(s => s.tool === 'v141_rl_optimizer_v2' && s.decision === 'executed');
  const v143Executed = steps.some(s => s.tool === 'v143_edge_deployment' && s.decision === 'executed');

  const vModulesTotal = 7;
  const vModulesExecuted = [v136Executed, v137Executed, v138Executed, v139Executed, v140Executed, v141Executed, v143Executed].filter(Boolean).length;

  return {
    runtimeSeconds: Math.round(runtimeSeconds * 10) / 10,
    candidateCount,
    entityCount,
    relationshipCount,
    graphNodes,
    graphEdges,
    claimSupport: `${claimVerification.supported_count || 0}/${claimVerification.claim_count || 0}`,
    facetCoverage: `${facetCoverage.covered_count || 0}/${facetCoverage.facet_count || 0}`,
    confidence: structuredConfidence.score || 'N/A',
    confidenceThreshold: structuredConfidence.threshold || 'N/A',
    totalSteps: steps.length,
    vModulesExecuted,
    vModulesTotal,
    repairTriggered: trace.repair_triggered || false,
    earlyExit: trace.early_exit_used || false,
    budgetPolicy: trace.budget_policy || 'refinement',
    selectedMode: trace.selected_mode || 'N/A',
  };
}

const AgentChatPage = () => {
  const { threadId } = useParams();
  const navigate = useNavigate();
  const [currentThreadId, setCurrentThreadId] = useState(threadId);
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(true);
  const [sending, setSending] = useState(false);
  const [searchContext, setSearchContext] = useState('agent');
  const [currentInput, setCurrentInput] = useState('');
  const [detectedIssues, setDetectedIssues] = useState([]);
  const [processingStep, setProcessingStep] = useState(-1);
  const [realSteps, setRealSteps] = useState([]);
  const [lastTraceMetrics, setLastTraceMetrics] = useState(null);

  // 累计指标
  const [agentMetrics, setAgentMetrics] = useState({
    totalQueries: 0,
    avgResponseTime: 0,
    totalSteps: 0,
    evidenceEntities: 0,
    evidenceRelationships: 0,
    vModulesExecuted: 0,
    vModulesTotal: 7,
  });

  const messagesEndRef = useRef(null);
  const messagesContainerRef = useRef(null);
  const inputRef = useRef(null);

  // Initialize thread
  useEffect(() => {
    const initializeThread = async () => {
      if (threadId) {
        setCurrentThreadId(threadId);
        saveThreadUUID(threadId);
        return;
      }

      const storedThreadId = getThreadUUID();
      if (storedThreadId) {
        navigate(`/agent-chat/${storedThreadId}`, { replace: true });
        return;
      }

      try {
        const response = await chatAPI.createThread("Agent Chat");
        if (response?.data?.uuid) {
          const newThreadId = response.data.uuid;
          saveThreadUUID(newThreadId);
          navigate(`/agent-chat/${newThreadId}`, { replace: true });
        }
      } catch (error) {
        console.error("Failed to create new chat thread", error);
      }
    };

    initializeThread();
  }, [threadId, navigate]);

  // Fetch chat history
  useEffect(() => {
    const fetchThreadChats = async () => {
      if (!currentThreadId) return;

      setLoading(true);
      try {
        const response = await chatAPI.getThread(currentThreadId);
        if (response?.data?.chats) {
          const formattedMessages = response.data.chats.map((chat) => ({
            id: `${chat.id}`,
            content: chat.message,
            isUser: chat.role === 'user',
            metadata: chat.metadata || {}
          }));
          setMessages(formattedMessages);
        }
      } catch (error) {
        console.error('Error fetching thread chats:', error);
      } finally {
        setLoading(false);
      }
    };

    if (currentThreadId) {
      fetchThreadChats();
    }
  }, [currentThreadId]);

  // Scroll to bottom
  const scrollToBottom = useCallback(() => {
    if (messagesEndRef.current) {
      messagesEndRef.current.scrollIntoView({
        behavior: 'smooth',
        block: 'end'
      });
    }
    if (messagesContainerRef.current) {
      messagesContainerRef.current.scrollTop = messagesContainerRef.current.scrollHeight;
    }
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, sending, scrollToBottom]);

  // ─── 渐进式展示处理步骤（从后端真实 trace 驱动） ───
  const animateRealSteps = useCallback(async (stepsData) => {
    if (!stepsData || stepsData.length === 0) return;

    const totalSteps = stepsData.length;
    // 根据步骤总数动态调整每步展示时间（总时长约 3-5 秒）
    const perStep = Math.max(100, Math.min(300, 3000 / totalSteps));

    setProcessingStep(0);
    for (let i = 0; i < totalSteps; i++) {
      await new Promise(resolve => setTimeout(resolve, perStep));
      setProcessingStep(i + 1);
    }
    // 保持最后一步展示一会儿
    await new Promise(resolve => setTimeout(resolve, 400));
    setProcessingStep(-1);
  }, []);

  // Handle issues detected
  const handleIssuesDetected = useCallback((issues) => {
    setDetectedIssues(prev => {
      const newIssues = issues.filter(issue =>
        !prev.some(p => p.id === issue.id)
      );
      return [...prev, ...newIssues];
    });
  }, []);

  // ─── 发送消息（从后端获取真实 trace） ───
  const handleSendMessage = async (message) => {
    if (!message.trim() || sending || !currentThreadId) return;

    const userMessage = {
      id: `temp-${Date.now()}-user`,
      content: message,
      isUser: true,
    };

    setMessages(prev => [...prev, userMessage]);
    setSending(true);
    setCurrentInput('');
    setDetectedIssues([]);
    setRealSteps([]);
    setProcessingStep(0);

    try {
      // 发送到 agent API
      const response = await chatAPI.sendAgentChatToThread(
        currentThreadId,
        searchContext,
        message
      );

      if (response?.data) {
        // 获取最新 trace（从 agent-traces API）
        let traceData = null;
        try {
          const tracesResponse = await chatAPI.getRecentAgentTraces(1);
          if (tracesResponse?.data?.traces && tracesResponse.data.traces.length > 0) {
            traceData = tracesResponse.data.traces[0].trace;
          }
        } catch (traceErr) {
          console.warn('Could not fetch agent trace:', traceErr);
        }

        // 从 trace 构建真实处理步骤
        let stepsData = [];
        let metrics = null;
        if (traceData) {
          stepsData = buildProcessingStepsFromTrace(traceData.steps);
          metrics = extractMetricsFromTrace(traceData);
          setRealSteps(stepsData);
          setLastTraceMetrics(metrics);
        }

        // 渐进式展示处理步骤
        if (stepsData.length > 0) {
          await animateRealSteps(stepsData);
        } else {
          // 如果没有 trace 数据，使用简短等待
          setProcessingStep(-1);
        }

        // 刷新聊天历史
        const threadResponse = await chatAPI.getThread(currentThreadId);
        if (threadResponse?.data?.chats) {
          const formattedMessages = threadResponse.data.chats.map((chat) => ({
            id: `${chat.id}`,
            content: chat.message,
            isUser: chat.role === 'user',
            metadata: chat.metadata || {}
          }));
          setMessages(formattedMessages);
        }

        // ─── 从真实 trace 更新累计指标 ───
        if (metrics) {
          setAgentMetrics(prev => {
            const newTotal = prev.totalQueries + 1;
            const newAvgTime = (prev.avgResponseTime * prev.totalQueries + metrics.runtimeSeconds) / newTotal;
            return {
              totalQueries: newTotal,
              avgResponseTime: Math.round(newAvgTime * 10) / 10,
              totalSteps: prev.totalSteps + metrics.totalSteps,
              evidenceEntities: prev.evidenceEntities + metrics.entityCount,
              evidenceRelationships: prev.evidenceRelationships + metrics.relationshipCount,
              vModulesExecuted: prev.vModulesExecuted + metrics.vModulesExecuted,
              vModulesTotal: metrics.vModulesTotal,
            };
          });
        } else {
          // 没有 trace 时至少增加查询计数
          setAgentMetrics(prev => ({
            ...prev,
            totalQueries: prev.totalQueries + 1,
          }));
        }
      }
    } catch (error) {
      console.error('Error sending message:', error);

      const errorMessage = {
        id: `error-${Date.now()}`,
        content: 'I encountered an error processing your request. Please try again or rephrase your question.\n\nError: ' + (error.response?.data?.detail || error.message || 'Unknown error'),
        isUser: false,
        metadata: { error: true }
      };

      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setSending(false);
      setProcessingStep(-1);
    }
  };

  // Handle input change for real-time detection
  const handleInputChange = (value) => {
    setCurrentInput(value);
  };

  // ─── 快捷提问按钮 ───
  const handleQuickQuestion = (question) => {
    setCurrentInput(question);
    // 直接发送
    handleSendMessage(question);
  };

  // 供 AgentStatusPanel 使用的处理步骤
  const displaySteps = realSteps.length > 0 ? realSteps : [];

  return (
    <Layout>
      <div className="agent-chat-page">
        <Grid fluid className="agent-chat-grid">
          <Row className="agent-chat-row">
            {/* Left Panel - Agent Status */}
            <Col xs={24} md={8} className="agent-left-panel">
              <AgentStatusPanel
                isProcessing={sending}
                currentStep={processingStep}
                processingSteps={displaySteps}
                detectedIssues={detectedIssues}
                onIssueClick={(issue) => {
                  console.log('Issue clicked:', issue);
                }}
                onQuickQuestion={handleQuickQuestion}
              />

              {/* Agent Metrics — 从真实 trace 提取 */}
              <Panel className="agent-metrics-panel" header="Agent Metrics">
                <div className="metrics-grid">
                  <div className="metric-item">
                    <span className="metric-value">{agentMetrics.totalQueries}</span>
                    <span className="metric-label">Queries</span>
                  </div>
                  <div className="metric-item">
                    <span className="metric-value">{agentMetrics.avgResponseTime}s</span>
                    <span className="metric-label">Avg Response</span>
                  </div>
                  <div className="metric-item">
                    <span className="metric-value">{agentMetrics.totalSteps}</span>
                    <span className="metric-label">Total Steps</span>
                  </div>
                  <div className="metric-item">
                    <span className="metric-value">{agentMetrics.evidenceEntities}</span>
                    <span className="metric-label">Entities Found</span>
                  </div>
                  <div className="metric-item">
                    <span className="metric-value">{agentMetrics.evidenceRelationships}</span>
                    <span className="metric-label">Relations Found</span>
                  </div>
                  <div className="metric-item">
                    <span className="metric-value">{agentMetrics.vModulesExecuted}/{agentMetrics.vModulesTotal}</span>
                    <span className="metric-label">V136-V143 Runs</span>
                  </div>
                </div>
              </Panel>

              {/* Last Trace Summary — 显示最近一次查询的详细指标 */}
              {lastTraceMetrics && (
                <Panel className="quick-tips-panel" header="📋 Last Query Details">
                  <div className="trace-details">
                    <div className="trace-detail-row">
                      <span className="trace-label">Runtime:</span>
                      <span className="trace-value">{lastTraceMetrics.runtimeSeconds}s</span>
                    </div>
                    <div className="trace-detail-row">
                      <span className="trace-label">Candidates:</span>
                      <span className="trace-value">{lastTraceMetrics.candidateCount}</span>
                    </div>
                    <div className="trace-detail-row">
                      <span className="trace-label">Graph Nodes:</span>
                      <span className="trace-value">{lastTraceMetrics.graphNodes}</span>
                    </div>
                    <div className="trace-detail-row">
                      <span className="trace-label">Graph Edges:</span>
                      <span className="trace-value">{lastTraceMetrics.graphEdges}</span>
                    </div>
                    <div className="trace-detail-row">
                      <span className="trace-label">Claim Support:</span>
                      <span className="trace-value">{lastTraceMetrics.claimSupport}</span>
                    </div>
                    <div className="trace-detail-row">
                      <span className="trace-label">Facet Coverage:</span>
                      <span className="trace-value">{lastTraceMetrics.facetCoverage}</span>
                    </div>
                    <div className="trace-detail-row">
                      <span className="trace-label">Confidence:</span>
                      <span className="trace-value">{lastTraceMetrics.confidence}/{lastTraceMetrics.confidenceThreshold}</span>
                    </div>
                    <div className="trace-detail-row">
                      <span className="trace-label">Processing Steps:</span>
                      <span className="trace-value">{lastTraceMetrics.totalSteps}</span>
                    </div>
                    <div className="trace-detail-row">
                      <span className="trace-label">V136-V143 Executed:</span>
                      <span className="trace-value">{lastTraceMetrics.vModulesExecuted}/{lastTraceMetrics.vModulesTotal}</span>
                    </div>
                    <div className="trace-detail-row">
                      <span className="trace-label">Selected Mode:</span>
                      <span className="trace-value">{lastTraceMetrics.selectedMode}</span>
                    </div>
                    <div className="trace-detail-row">
                      <span className="trace-label">Budget Policy:</span>
                      <span className="trace-value">{lastTraceMetrics.budgetPolicy}</span>
                    </div>
                    {lastTraceMetrics.repairTriggered && (
                      <div className="trace-detail-row warning">
                        <span className="trace-label">⚠ Repair Triggered</span>
                      </div>
                    )}
                    {lastTraceMetrics.earlyExit && (
                      <div className="trace-detail-row info">
                        <span className="trace-label">⚡ Early Exit Used</span>
                      </div>
                    )}
                  </div>
                </Panel>
              )}

              {/* Quick Tips */}
              <Panel className="quick-tips-panel" header="💡 Evidence-Contract Tips">
                <ul className="tips-list">
                  <li>Ask a focused question with clear answer requirements</li>
                  <li>Use Evidence Trace to inspect coverage and source support</li>
                  <li>Check which items were selected under the verification budget</li>
                  <li>Repair is accepted only when evidence support improves</li>
                  <li>Rollback keeps the original answer when the guard fails</li>
                </ul>
              </Panel>
            </Col>

            {/* Right Panel - Chat */}
            <Col xs={24} md={16} className="agent-right-panel">
              <div className="chat-container">
                {/* Real-time Detector */}
                {currentInput.trim().length > 3 && (
                  <RealTimeDetector
                    query={currentInput}
                    isProcessing={sending}
                    onIssueDetected={handleIssuesDetected}
                    showAnalysis={true}
                  />
                )}

                {/* Messages */}
                <div className="messages-container" ref={messagesContainerRef}>
                  {loading ? (
                    <div className="loading-message">
                      <Loader content="Loading chat history..." vertical />
                    </div>
                  ) : messages.length === 0 ? (
                    <div className="empty-chat">
                      <div className="welcome-section">
                        <h2>Evidence-Contract Agent</h2>
                        <p>
                          PathRAG first identifies the information required by the question, then verifies only the
                          highest-risk evidence gaps before deciding whether to repair, retain, or roll back the answer.
                        </p>
                        <div className="capability-cards">
                          <div className="capability-card-item">
                            <span className="capability-icon">🔍</span>
                            <span>Evidence Contract</span>
                          </div>
                          <div className="capability-card-item">
                            <span className="capability-icon">🛡️</span>
                            <span>Risk-Based Verification</span>
                          </div>
                          <div className="capability-card-item">
                            <span className="capability-icon">🧠</span>
                            <span>Guarded Repair</span>
                          </div>
                          <div className="capability-card-item">
                            <span className="capability-icon">📊</span>
                            <span>Evidence Trace</span>
                          </div>
                        </div>
                        <div className="quick-questions-section">
                          <p className="quick-questions-title">Try a traceable question:</p>
                          <div className="quick-questions-buttons">
                            <button
                              className="quick-question-btn"
                              onClick={() => handleQuickQuestion('Which claims in the answer are not fully supported by the available evidence?')}
                            >
                              Find evidence gaps
                            </button>
                            <button
                              className="quick-question-btn"
                              onClick={() => handleQuickQuestion('Should the current answer be repaired under a limited verification budget? Explain why.')}
                            >
                              Explain a repair decision
                            </button>
                            <button
                              className="quick-question-btn"
                              onClick={() => handleQuickQuestion('Show the evidence path and final accept-or-rollback decision for this answer.')}
                            >
                              Trace the final decision
                            </button>
                          </div>
                        </div>
                        <p className="start-hint">Ask a question below to inspect the full decision trace.</p>
                      </div>
                    </div>
                  ) : (
                    <div className="conversation-thread">
                      {messages.map((message) => (
                        <div
                          key={message.id}
                          className={`message-wrapper ${message.isUser ? 'user-message-wrapper' : 'ai-message-wrapper'}`}
                        >
                          <ChatMessage
                            message={message.content}
                            isUser={message.isUser}
                            metadata={message.metadata}
                          />
                        </div>
                      ))}

                      {sending && (
                        <div className="message-wrapper ai-message-wrapper">
                          <div className="loading-message">
                            <Loader size="md" content="Checking evidence gaps and repair guards..." vertical />
                          </div>
                        </div>
                      )}

                      <div ref={messagesEndRef} style={{ height: 1, clear: 'both' }} />
                    </div>
                  )}
                </div>

                {/* Chat Input */}
                <div className="chat-input-container">
                  <ChatInput
                    onSendMessage={handleSendMessage}
                    onChange={handleInputChange}
                    isLoading={sending}
                    searchContext={searchContext}
                    setSearchContext={setSearchContext}
                    placeholder="Ask me anything about your documents..."
                  />
                </div>
              </div>
            </Col>
          </Row>
        </Grid>
      </div>
    </Layout>
  );
};

export default AgentChatPage;
