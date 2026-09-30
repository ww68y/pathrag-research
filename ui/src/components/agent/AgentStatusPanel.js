import React, { useState, useEffect } from 'react';
import { Panel, Badge, Timeline, Progress, IconButton, Tooltip, Whisper } from 'rsuite';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import {
  faRobot,
  faBrain,
  faShieldAlt,
  faMemory,
  faStream,
  faTable,
  faCode,
  faImage,
  faCheckCircle,
  faExclamationTriangle,
  faSpinner,
  faClock,
  faGlobe,
  faLock,
  faChartLine,
  faBullseye,
  faMobileAlt,
  faNetworkWired,
  faFlask,
  faAtom,
  faDna,
  faInfinity
} from '@fortawesome/free-solid-svg-icons';
import '../../styles/agent.css';

// Agent capability status component
const CapabilityCard = ({ icon, title, status, description, active }) => (
  <div className={`capability-card ${active ? 'active' : 'inactive'}`}>
    <div className="capability-icon">
      <FontAwesomeIcon icon={icon} />
    </div>
    <div className="capability-info">
      <div className="capability-title">
        {title}
        {active && <Badge className="status-badge" content="ON" style={{ backgroundColor: '#4CAF50' }} />}
      </div>
      <div className="capability-description">{description}</div>
    </div>
    <div className="capability-status">
      {active ? (
        <FontAwesomeIcon icon={faCheckCircle} style={{ color: '#4CAF50' }} />
      ) : (
        <FontAwesomeIcon icon={faExclamationTriangle} style={{ color: '#FF9800' }} />
      )}
    </div>
  </div>
);

// Real-time processing step component — 从后端 trace 真实数据驱动
const ProcessingStep = ({ step, index, currentStep }) => {
  const isActive = index === currentStep;
  const isCompleted = index < currentStep;

  return (
    <Timeline.Item
      className={`processing-step ${isActive ? 'active' : ''} ${isCompleted ? 'completed' : ''}`}
    >
      <div className="step-content">
        <div className="step-header">
          <span className="step-name">{step.name}</span>
          {isActive && <FontAwesomeIcon icon={faSpinner} spin className="step-spinner" />}
          {isCompleted && <FontAwesomeIcon icon={faCheckCircle} style={{ color: '#4CAF50', fontSize: '0.85rem' }} />}
        </div>
        <div className="step-details">
          {step.description}
          {step.decision && step.decision !== 'executed' && step.decision !== 'skipped' && (
            <span className="step-decision"> → {step.decision}</span>
          )}
          {step.observation && (
            <div className="step-observation">{step.observation}</div>
          )}
          {step.progress > 0 && (
            <Progress.Line
              percent={step.progress}
              strokeColor={isActive ? '#2196F3' : '#4CAF50'}
              showInfo={false}
              style={{ marginTop: 8 }}
            />
          )}
        </div>
      </div>
    </Timeline.Item>
  );
};

const AgentStatusPanel = ({
  isProcessing,
  currentStep,
  processingSteps,
  agentCapabilities,
  detectedIssues,
  onIssueClick,
  onQuickQuestion
}) => {
  const [expanded, setExpanded] = useState(true);
  const [activeTab, setActiveTab] = useState('status'); // 'status', 'issues', 'capabilities'

  // Use real steps from backend trace if available, otherwise empty
  const steps = processingSteps || [];

  // Updated capabilities — V136-V143 plus core modules
  const defaultCapabilities = [
    // Core VSR modules
    { icon: faBrain, title: 'Adaptive Compression', description: 'Smart evidence compression (V89)', active: true },
    { icon: faShieldAlt, title: 'Adversarial Defense', description: 'Evidence authenticity check (V101)', active: true },
    { icon: faMemory, title: 'Cross-Query Memory', description: 'Historical evidence reuse (V99)', active: true },
    { icon: faStream, title: 'Real-Time Stream', description: 'Incremental updates (V105)', active: true },
    // V128-V135 (conceptual)
    { icon: faAtom, title: 'Quantum Retrieval', description: 'Quantum-inspired path exploration (V128)', active: false },
    { icon: faDna, title: 'Bio Evolution', description: 'Genetic algorithm optimization (V130)', active: false },
    // V136-V143 (newly integrated)
    { icon: faTable, title: 'V136 Multimodal Fusion', description: 'Table/formula/code extraction & fusion', active: true },
    { icon: faClock, title: 'V137 Temporal Reasoning', description: 'Timeline & causal chain inference', active: true },
    { icon: faGlobe, title: 'V138 Cross-Lingual', description: 'Bilingual entity alignment', active: true },
    { icon: faLock, title: 'V139 Privacy Protection', description: 'PII detection & anonymization', active: true },
    { icon: faChartLine, title: 'V140 Explainability', description: 'Evidence tree & reasoning path', active: true },
    { icon: faBullseye, title: 'V141 RL Optimizer', description: 'PPO-based action selection', active: true },
    { icon: faNetworkWired, title: 'V142 Incremental Update', description: 'Graph version management', active: false },
    { icon: faMobileAlt, title: 'V143 Edge Deployment', description: 'Resource-adaptive deployment', active: true },
  ];

  const capabilities = agentCapabilities || defaultCapabilities;
  const activeCount = capabilities.filter(c => c.active).length;

  // Quick questions for idle state
  const quickQuestions = [
    'What is PathRAG?',
    'Analyze this document',
    'Compare two concepts',
    'Explain the VSR mechanism',
    'What innovation modules are available?'
  ];

  return (
    <Panel
      className="agent-status-panel"
      header={
        <div className="panel-header">
          <div className="panel-title">
            <FontAwesomeIcon icon={faRobot} className="agent-icon" />
            <span>PathRAG Agent</span>
            {isProcessing && (
              <Badge
                content="Processing"
                style={{ backgroundColor: '#2196F3', marginLeft: 8 }}
              />
            )}
          </div>
          <div className="panel-tabs">
            <button
              className={`tab-btn ${activeTab === 'status' ? 'active' : ''}`}
              onClick={() => setActiveTab('status')}
            >
              Status
            </button>
            <button
              className={`tab-btn ${activeTab === 'issues' ? 'active' : ''}`}
              onClick={() => setActiveTab('issues')}
            >
              Issues
              {detectedIssues?.length > 0 && (
                <Badge content={detectedIssues.length} style={{ marginLeft: 4 }} />
              )}
            </button>
            <button
              className={`tab-btn ${activeTab === 'capabilities' ? 'active' : ''}`}
              onClick={() => setActiveTab('capabilities')}
            >
              Capabilities
            </button>
          </div>
        </div>
      }
      collapsible
      expanded={expanded}
      onSelect={() => setExpanded(!expanded)}
    >
      {activeTab === 'status' && (
        <div className="status-content">
          {isProcessing ? (
            <>
              <div className="processing-header">
                <FontAwesomeIcon icon={faSpinner} spin className="processing-icon" />
                <span>
                  {steps.length > 0
                    ? `Processing through ${steps.length} steps...`
                    : 'Agent is processing your query...'}
                </span>
              </div>
              {steps.length > 0 ? (
                <Timeline className="processing-timeline">
                  {steps.map((step, index) => (
                    <ProcessingStep
                      key={index}
                      step={step}
                      index={index}
                      currentStep={currentStep}
                    />
                  ))}
                </Timeline>
              ) : (
                <div className="processing-placeholder">
                  <FontAwesomeIcon icon={faFlask} spin style={{ fontSize: '2rem', color: '#667eea' }} />
                  <p>Running 60+ innovation modules...</p>
                </div>
              )}
            </>
          ) : (
            <div className="idle-status">
              <FontAwesomeIcon icon={faRobot} className="idle-icon" />
              <p>Agent is ready. Type a message to start.</p>
              <div className="quick-actions">
                <span className="quick-action-hint">Try asking:</span>
                {quickQuestions.map((q, idx) => (
                  <button
                    key={idx}
                    className="quick-action-btn"
                    onClick={() => onQuickQuestion && onQuickQuestion(q)}
                  >
                    {q}
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
      )}

      {activeTab === 'issues' && (
        <div className="issues-content">
          {detectedIssues && detectedIssues.length > 0 ? (
            <div className="issues-list">
              {detectedIssues.map((issue, index) => (
                <div
                  key={index}
                  className={`issue-item ${issue.severity}`}
                  onClick={() => onIssueClick && onIssueClick(issue)}
                >
                  <div className="issue-header">
                    <FontAwesomeIcon
                      icon={issue.severity === 'high' ? faExclamationTriangle : faCheckCircle}
                      className={`issue-icon ${issue.severity}`}
                    />
                    <span className="issue-title">{issue.title}</span>
                    <span className={`issue-severity ${issue.severity}`}>
                      {issue.severity.toUpperCase()}
                    </span>
                  </div>
                  <div className="issue-description">{issue.message}</div>
                  {issue.suggestion && (
                    <div className="issue-suggestion">
                      <strong>Suggestion:</strong> {issue.suggestion}
                    </div>
                  )}
                </div>
              ))}
            </div>
          ) : (
            <div className="no-issues">
              <FontAwesomeIcon icon={faCheckCircle} className="no-issues-icon" />
              <p>No issues detected</p>
              <span className="no-issues-hint">
                The agent will automatically detect and report issues during processing
              </span>
            </div>
          )}
        </div>
      )}

      {activeTab === 'capabilities' && (
        <div className="capabilities-content">
          <div className="capabilities-grid">
            {capabilities.map((cap, index) => (
              <CapabilityCard key={index} {...cap} />
            ))}
          </div>
          <div className="capabilities-footer">
            <span className="version-info">
              Agent Version: <strong>vsr-pathrag-v143</strong>
            </span>
            <span className="innovation-count">
              {activeCount}/{capabilities.length} Modules Active · 60+ Total Innovations
            </span>
          </div>
        </div>
      )}
    </Panel>
  );
};

export default AgentStatusPanel;
