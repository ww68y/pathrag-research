import React, { useState, useEffect, useCallback } from 'react';
import { Panel, Alert, Message, IconButton, Tooltip, Whisper, Badge } from 'rsuite';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import {
  faExclamationTriangle,
  faCheckCircle,
  faInfoCircle,
  faBug,
  faShieldAlt,
  faBolt,
  faEye,
  faTimes,
  faBell
} from '@fortawesome/free-solid-svg-icons';
import '../../styles/agent.css';

// Issue types and their configurations
const ISSUE_TYPES = {
  AMBIGUITY: {
    icon: faInfoCircle,
    color: '#2196F3',
    title: 'Query Ambiguity Detected'
  },
  MISSING_CONTEXT: {
    icon: faExclamationTriangle,
    color: '#FF9800',
    title: 'Missing Context'
  },
  ADVERSARIAL: {
    icon: faShieldAlt,
    color: '#F44336',
    title: 'Potential Adversarial Input'
  },
  COMPLEXITY: {
    icon: faBolt,
    color: '#9C27B0',
    title: 'High Complexity Query'
  },
  MULTIMODAL: {
    icon: faEye,
    color: '#4CAF50',
    title: 'Multi-Modal Evidence Available'
  },
  ERROR: {
    icon: faBug,
    color: '#F44336',
    title: 'Processing Error'
  }
};

// Real-time issue detection component
const IssueAlert = ({ issue, onDismiss, onAction }) => {
  const config = ISSUE_TYPES[issue.type] || ISSUE_TYPES.ERROR;
  
  return (
    <Message
      showIcon
      type={issue.severity === 'error' ? 'error' : issue.severity === 'warning' ? 'warning' : 'info'}
      className="issue-alert"
      style={{ borderLeftColor: config.color }}
    >
      <div className="issue-alert-content">
        <div className="issue-alert-header">
          <FontAwesomeIcon icon={config.icon} style={{ color: config.color }} />
          <span className="issue-title">{issue.title || config.title}</span>
          <Whisper
            placement="top"
            trigger="hover"
            speaker={<Tooltip>Dismiss</Tooltip>}
          >
            <IconButton
              icon={<FontAwesomeIcon icon={faTimes} />}
              size="xs"
              appearance="subtle"
              onClick={() => onDismiss(issue.id)}
              className="dismiss-btn"
            />
          </Whisper>
        </div>
        <div className="issue-alert-body">
          <p>{issue.message}</p>
          {issue.suggestion && (
            <div className="issue-suggestion">
              <strong>💡 Suggestion:</strong> {issue.suggestion}
            </div>
          )}
        </div>
        {issue.action && (
          <div className="issue-alert-actions">
            <button 
              className="issue-action-btn"
              onClick={() => onAction(issue)}
            >
              {issue.action.label}
            </button>
          </div>
        )}
      </div>
    </Message>
  );
};

// Query analysis result component
const QueryAnalysisResult = ({ analysis }) => {
  if (!analysis) return null;
  
  return (
    <div className="query-analysis-result">
      <div className="analysis-header">
        <FontAwesomeIcon icon={faEye} />
        <span>Query Analysis</span>
      </div>
      <div className="analysis-content">
        <div className="analysis-item">
          <span className="analysis-label">Type:</span>
          <span className="analysis-value">{analysis.queryType}</span>
        </div>
        <div className="analysis-item">
          <span className="analysis-label">Complexity:</span>
          <span className={`analysis-value complexity-${analysis.complexity}`}>
            {analysis.complexity}
          </span>
        </div>
        <div className="analysis-item">
          <span className="analysis-label">Intent:</span>
          <span className="analysis-value">{analysis.intent}</span>
        </div>
        {analysis.entities && analysis.entities.length > 0 && (
          <div className="analysis-item">
            <span className="analysis-label">Entities:</span>
            <div className="entity-tags">
              {analysis.entities.map((entity, idx) => (
                <span key={idx} className="entity-tag">{entity}</span>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

const RealTimeDetector = ({ 
  query, 
  isProcessing, 
  onIssueDetected, 
  onAnalysisComplete,
  showAnalysis = true 
}) => {
  const [issues, setIssues] = useState([]);
  const [analysis, setAnalysis] = useState(null);
  const [dismissedIssues, setDismissedIssues] = useState(new Set());

  // Real-time query analysis
  const analyzeQuery = useCallback((queryText) => {
    if (!queryText || queryText.trim().length < 3) {
      setAnalysis(null);
      return;
    }

    const lowerQuery = queryText.toLowerCase();
    const newAnalysis = {
      queryType: 'general',
      complexity: 'low',
      intent: 'information seeking',
      entities: []
    };

    // Detect query type
    if (lowerQuery.match(/\b(what|who|when|where|which)\b/)) {
      newAnalysis.queryType = 'factoid';
    } else if (lowerQuery.match(/\b(why|how|cause|reason)\b/)) {
      newAnalysis.queryType = 'causal';
      newAnalysis.intent = 'explanation seeking';
    } else if (lowerQuery.match(/\b(compare|difference|versus|vs)\b/)) {
      newAnalysis.queryType = 'comparison';
      newAnalysis.intent = 'comparative analysis';
    } else if (lowerQuery.match(/\b(process|step|method|how to)\b/)) {
      newAnalysis.queryType = 'procedural';
      newAnalysis.intent = 'procedure seeking';
    }

    // Detect complexity
    const wordCount = queryText.split(/\s+/).length;
    if (wordCount > 20) {
      newAnalysis.complexity = 'high';
    } else if (wordCount > 10) {
      newAnalysis.complexity = 'medium';
    }

    // Extract simple entities (capitalized words)
    const entityMatches = queryText.match(/\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b/g);
    if (entityMatches) {
      newAnalysis.entities = [...new Set(entityMatches)].slice(0, 5);
    }

    setAnalysis(newAnalysis);
    if (onAnalysisComplete) {
      onAnalysisComplete(newAnalysis);
    }
  }, [onAnalysisComplete]);

  // Real-time issue detection
  const detectIssues = useCallback((queryText) => {
    if (!queryText || queryText.trim().length < 5) {
      setIssues([]);
      return;
    }

    const detectedIssues = [];
    const lowerQuery = queryText.toLowerCase();

    // Check for ambiguity
    if (lowerQuery.match(/\b(it|this|that|they|them)\b/) && !lowerQuery.match(/\b(it is|this is)\b/)) {
      detectedIssues.push({
        id: `ambiguity-${Date.now()}`,
        type: 'AMBIGUITY',
        severity: 'info',
        message: 'Your query contains ambiguous pronouns. Consider being more specific.',
        suggestion: 'Replace "it/they/this" with the actual subject you\'re referring to.',
        action: {
          label: 'Show Examples',
          handler: () => {}
        }
      });
    }

    // Check for complexity
    const wordCount = queryText.split(/\s+/).length;
    if (wordCount > 25) {
      detectedIssues.push({
        id: `complexity-${Date.now()}`,
        type: 'COMPLEXITY',
        severity: 'warning',
        message: 'This is a complex query with multiple parts.',
        suggestion: 'Consider breaking it down into simpler questions for better results.',
        action: {
          label: 'Split Query',
          handler: () => {}
        }
      });
    }

    // Check for potential adversarial patterns
    if (lowerQuery.match(/\b(ignore|forget|disregard)\b.*\b(instruction|rule|guideline)\b/)) {
      detectedIssues.push({
        id: `adversarial-${Date.now()}`,
        type: 'ADVERSARIAL',
        severity: 'error',
        message: 'Potential adversarial pattern detected.',
        suggestion: 'Please ensure your query is genuine and not attempting to manipulate the system.',
        action: null
      });
    }

    // Check for missing context indicators
    if (lowerQuery.match(/\b(the|this|that)\s+\w+\b/) && !lowerQuery.match(/\b(document|file|paper|article)\b/)) {
      const hasContext = detectedIssues.length === 0;
      if (!hasContext) {
        detectedIssues.push({
          id: `context-${Date.now()}`,
          type: 'MISSING_CONTEXT',
          severity: 'info',
          message: 'Your query might benefit from more context.',
          suggestion: 'Consider specifying which document or topic you\'re referring to.',
          action: null
        });
      }
    }

    // Filter out dismissed issues
    const filteredIssues = detectedIssues.filter(issue => !dismissedIssues.has(issue.id));
    setIssues(filteredIssues);

    if (onIssueDetected && filteredIssues.length > 0) {
      onIssueDetected(filteredIssues);
    }
  }, [dismissedIssues, onIssueDetected]);

  // Effect: Analyze query when it changes
  useEffect(() => {
    const timeoutId = setTimeout(() => {
      analyzeQuery(query);
      detectIssues(query);
    }, 300); // Debounce for 300ms

    return () => clearTimeout(timeoutId);
  }, [query, analyzeQuery, detectIssues]);

  // Dismiss an issue
  const handleDismiss = (issueId) => {
    setDismissedIssues(prev => new Set([...prev, issueId]));
    setIssues(prev => prev.filter(issue => issue.id !== issueId));
  };

  // Handle issue action
  const handleAction = (issue) => {
    if (issue.action && issue.action.handler) {
      issue.action.handler();
    }
  };

  // Clear all issues
  const clearAllIssues = () => {
    issues.forEach(issue => setDismissedIssues(prev => new Set([...prev, issue.id])));
    setIssues([]);
  };

  if (!query || query.trim().length < 3) {
    return null;
  }

  return (
    <div className="real-time-detector">
      {/* Query Analysis */}
      {showAnalysis && analysis && (
        <QueryAnalysisResult analysis={analysis} />
      )}

      {/* Issues Panel */}
      {issues.length > 0 && (
        <Panel 
          className="issues-panel"
          header={
            <div className="issues-panel-header">
              <div className="issues-title">
                <FontAwesomeIcon icon={faBell} />
                <span>Detected Issues</span>
                <Badge content={issues.length} />
              </div>
              <Whisper
                placement="top"
                trigger="hover"
                speaker={<Tooltip>Clear all</Tooltip>}
              >
                <IconButton
                  icon={<FontAwesomeIcon icon={faTimes} />}
                  size="xs"
                  appearance="subtle"
                  onClick={clearAllIssues}
                />
              </Whisper>
            </div>
          }
        >
          <div className="issues-list">
            {issues.map(issue => (
              <IssueAlert
                key={issue.id}
                issue={issue}
                onDismiss={handleDismiss}
                onAction={handleAction}
              />
            ))}
          </div>
        </Panel>
      )}

      {/* Processing Indicator */}
      {isProcessing && (
        <div className="processing-indicator">
          <FontAwesomeIcon icon={faBolt} spin />
          <span>Agent is analyzing your query...</span>
        </div>
      )}
    </div>
  );
};

export default RealTimeDetector;
