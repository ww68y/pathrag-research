import React, { useEffect } from 'react';
import { Avatar, IconButton } from 'rsuite';
import MoreIcon from '@rsuite/icons/More';
import { useAuth } from '../../context/AuthContext';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { vscDarkPlus } from 'react-syntax-highlighter/dist/esm/styles/prism';
import remarkGfm from 'remark-gfm';
import mermaid from 'mermaid';

// Initialize mermaid
mermaid.initialize({
  startOnLoad: true,
  theme: 'dark',
  securityLevel: 'loose',
  fontFamily: 'monospace'
});

const parseAgentTrace = (content) => {
  if (typeof content !== 'string') {
    return { answer: '', trace: null };
  }
  const marker = '\n---\n### KG-RAG Agent Trace';
  const markerIndex = content.indexOf(marker);
  if (markerIndex === -1) {
    return { answer: content, trace: null };
  }

  const answer = content.slice(0, markerIndex).trim();
  const traceText = content.slice(markerIndex + '\n---\n'.length).trim();
  const meta = {};
  traceText.split('\n').forEach((line) => {
    const match = line.match(/^- \*\*(.+?):\*\* `?(.+?)`?$/);
    if (match) {
      meta[match[1].toLowerCase()] = match[2];
    }
  });

  const rows = traceText
    .split('\n')
    .filter((line) => line.startsWith('|') && !line.includes('---'))
    .slice(1)
    .map((line) => line.split('|').slice(1, -1).map((cell) => cell.trim()))
    .filter((cells) => cells.length >= 4)
    .map(([index, tool, decision, observation]) => ({
      index,
      tool,
      decision,
      observation,
    }));

  return {
    answer,
    trace: {
      questionType: meta['question type'] || 'unknown',
      answerContract: meta['answer contract'] || 'balanced',
      budgetPolicy: meta['budget policy'] || 'refinement',
      finalPolicy: meta['final policy'] || 'unknown',
      selectedMode: meta['selected mode'] || 'unknown',
      selectedProfile: meta['selected profile'] || 'unknown',
      candidateCount: meta['candidate count'] || '1',
      evidenceEdges: meta['evidence edges'] || '0',
      queryPlan: meta['query plan'] || 'n/a',
      knowledgeGap: meta['knowledge gap'] || 'n/a',
      localGraphRepair: meta['local graph repair'] || '0 accepted/0 rejected',
      counterfactualRollback: meta['counterfactual rollback'] || 'no',
      evidencePortfolio: meta['evidence portfolio'] || 'n/a',
      sourceConflicts: meta['source conflicts'] || 'n/a',
      computeBudget: meta['compute budget'] || 'n/a',
      activeAcquisition: meta['active acquisition'] || 'n/a',
      claimVerification: meta['claim verification'] || 'n/a',
      claimSafety: meta['claim safety'] || 'n/a',
      experienceMemory: meta['experience memory'] || 'n/a',
      experienceRoute: meta['experience route'] || 'n/a',
      evidenceRepair: meta['evidence repair'] || 'unknown',
      claimPrune: meta['claim prune'] || 'unknown',
      facetCompletion: meta['facet completion'] || 'unknown',
      claimSupport: meta['claim support'] || '0/0',
      unsupportedClaims: meta['unsupported claims'] || '0',
      facetCoverage: meta['facet coverage'] || '0/0',
      structuredConfidence: meta['structured confidence'] || 'n/a',
      evidenceAudit: meta['evidence audit'] || 'n/a',
      riskFlags: meta['risk flags'] || 'none',
      runtime: meta.runtime || '0s',
      rows,
    },
  };
};

const AgentTracePanel = ({ trace }) => {
  if (!trace) return null;
  return (
    <details className="agent-trace-panel">
      <summary>
        <span>KG-RAG Agent Trace</span>
        <strong>{trace.finalPolicy}</strong>
      </summary>
      <div className="agent-trace-meta">
        <span>Type: {trace.questionType}</span>
        <span>Contract: {trace.answerContract}</span>
        <span>Budget: {trace.budgetPolicy}</span>
        <span>Mode: {trace.selectedMode}</span>
        <span>Profile: {trace.selectedProfile}</span>
        <span>Candidates: {trace.candidateCount}</span>
        <span>Evidence: {trace.evidenceEdges}</span>
        <span>Query plan: {trace.queryPlan}</span>
        <span>Graph gap: {trace.knowledgeGap}</span>
        <span>Local repair: {trace.localGraphRepair}</span>
        <span>Rollback: {trace.counterfactualRollback}</span>
        <span>Portfolio: {trace.evidencePortfolio}</span>
        <span>Conflicts: {trace.sourceConflicts}</span>
        <span>Compute: {trace.computeBudget}</span>
        <span>Active evidence: {trace.activeAcquisition}</span>
        <span>Claim verification: {trace.claimVerification}</span>
        <span>Claim safety: {trace.claimSafety}</span>
        <span>Memory: {trace.experienceMemory}</span>
        <span>Memory route: {trace.experienceRoute}</span>
        <span>Repair: {trace.evidenceRepair}</span>
        <span>Prune: {trace.claimPrune}</span>
        <span>Facet: {trace.facetCompletion}</span>
        <span>Claims: {trace.claimSupport}</span>
        <span>Unsupported: {trace.unsupportedClaims}</span>
        <span>Coverage: {trace.facetCoverage}</span>
        <span>Confidence: {trace.structuredConfidence}</span>
        <span>Audit: {trace.evidenceAudit}</span>
        <span>Risk: {trace.riskFlags}</span>
        <span>Runtime: {trace.runtime}</span>
      </div>
      <div className="agent-trace-steps">
        {trace.rows.map((row) => (
          <article className="agent-trace-step" key={`${row.index}-${row.tool}-${row.decision}`}>
            <div>
              <span>#{row.index}</span>
              <strong>{row.tool}</strong>
              <em>{row.decision}</em>
            </div>
            <p>{row.observation}</p>
          </article>
        ))}
      </div>
    </details>
  );
};

const ChatMessage = ({ message, isUser }) => {
  const { currentUser } = useAuth();
  const { answer, trace } = !isUser ? parseAgentTrace(message) : { answer: message, trace: null };

  useEffect(() => {
    // Process any mermaid diagrams after component mounts
    setTimeout(() => {
      try {
        mermaid.contentLoaded();
      } catch (error) {
        console.error('Mermaid processing error:', error);
      }
    }, 0);
  }, [message]);

  // Function to process mermaid code blocks
  const processMermaidBlocks = (content) => {
    if (typeof content !== 'string') return '';

    const mermaidRegex = /```mermaid\n([\s\S]*?)```/g;
    let lastIndex = 0;
    const parts = [];
    let match;

    while ((match = mermaidRegex.exec(content)) !== null) {
      // Add text before the mermaid block
      if (match.index > lastIndex) {
        parts.push(content.substring(lastIndex, match.index));
      }

      // Add mermaid diagram with unique ID
      const diagramId = `mermaid-diagram-${Date.now()}-${Math.floor(Math.random() * 1000)}`;
      parts.push(
        `<div class="mermaid-diagram">
          <div class="mermaid" id="${diagramId}">
            ${match[1]}
          </div>
        </div>`
      );

      lastIndex = match.index + match[0].length;
    }

    // Add any remaining text
    if (lastIndex < content.length) {
      parts.push(content.substring(lastIndex));
    }

    return parts.join('');
  };

  // Process content for mermaid diagrams if it's a bot message
  const processedContent = !isUser ? processMermaidBlocks(answer) : answer;

  return (
    <div className={`chat-message ${isUser ? 'user-message' : 'ai-message'}`}>
      {!isUser && (
        <div className="message-avatar">
          <Avatar circle src="/logo192.png">
            AI
          </Avatar>
        </div>
      )}

      <div className="message-content">
        <div className="message-header">
          <span className="message-sender">{isUser ? currentUser?.username : 'Enterprise KG AI'}</span>
          <span className="message-time">{new Date().toLocaleTimeString()}</span>
        </div>

        <div className="message-bubble">
          {!isUser ? (
            <div className="markdown-content">
              <ReactMarkdown
                remarkPlugins={[remarkGfm]}
                components={{
                  code({ node, inline, className, children, ...props }) {
                    const match = /language-(\w+)/.exec(className || '');
                    return !inline && match ? (
                      <SyntaxHighlighter
                        style={vscDarkPlus}
                        language={match[1]}
                        PreTag="div"
                        {...props}
                      >
                        {String(children).replace(/\n$/, '')}
                      </SyntaxHighlighter>
                    ) : (
                      <code className={className} {...props}>
                        {children}
                      </code>
                    );
                  }
                }}
              >
                {processedContent}
              </ReactMarkdown>
              <AgentTracePanel trace={trace} />
            </div>
          ) : (
            <div className="text-content">{answer}</div>
          )}
        </div>

        <div className="message-actions">
          <IconButton icon={<MoreIcon />} circle size="xs" />
        </div>
      </div>

      {isUser && (
        <div className="message-avatar">
          <Avatar circle>
            {currentUser?.username?.charAt(0).toUpperCase()}
          </Avatar>
        </div>
      )}
    </div>
  );
};

export default ChatMessage;
