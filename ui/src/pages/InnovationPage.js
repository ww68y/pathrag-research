import React, { useState } from 'react';
import { Panel, Modal, Grid, Row, Col, Tag, Progress, Timeline, Badge } from 'rsuite';
import Layout from '../components/Layout';
import InnovationShowcase from '../components/agent/InnovationShowcase';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import '../styles/innovation-page.css';

const InnovationPage = () => {
  const [selectedModule, setSelectedModule] = useState(null);
  const [showDetailModal, setShowDetailModal] = useState(false);

  const handleModuleSelect = (module) => {
    setSelectedModule(module);
    setShowDetailModal(true);
  };

  const closeModal = () => {
    setShowDetailModal(false);
    setSelectedModule(null);
  };

  // Detailed module information
  const moduleDetails = {
    'V89': {
      fullDescription: 'Adaptive Evidence Compression dynamically adjusts the compression ratio based on query complexity and evidence importance. It uses a multi-factor scoring system to determine optimal compression levels.',
      technicalDetails: [
        'Query complexity analysis using entropy metrics',
        'Evidence importance scoring based on relevance and novelty',
        'Dynamic compression ratio selection (0.3x - 0.9x)',
        'Preservation of critical information markers'
      ],
      benefits: [
        '30-50% reduction in token usage',
        'Maintained answer quality',
        'Faster processing times',
        'Reduced API costs'
      ],
      useCases: [
        'Long document summarization',
        'Multi-document synthesis',
        'Real-time query processing'
      ]
    },
    'V90': {
      fullDescription: 'Salience Reordering uses attention mechanisms to dynamically reorder evidence based on query relevance, ensuring the most important information is processed first.',
      technicalDetails: [
        'Query-evidence attention scoring',
        'Dynamic priority queue management',
        'Context-aware relevance calculation',
        'Real-time reordering during processing'
      ],
      benefits: [
        'Improved answer relevance',
        'Better early termination opportunities',
        'Enhanced user experience',
        'Optimized resource utilization'
      ],
      useCases: [
        'Priority evidence retrieval',
        'Interactive query refinement',
        'Progressive answer generation'
      ]
    },
    'V91': {
      fullDescription: 'Fidelity Scoring System provides multi-dimensional quality assessment using four key metrics: Claim Accuracy Quotient (CAQ), Contextual Fidelity Score (CFS), Novelty Preservation Score (NPS), and Semantic Alignment Score (SAS).',
      technicalDetails: [
        'CAQ: Fact verification against sources',
        'CFS: Context preservation measurement',
        'NPS: Novel information detection',
        'SAS: Semantic similarity scoring'
      ],
      benefits: [
        'Comprehensive quality metrics',
        'Identifiable improvement areas',
        'Benchmarking capabilities',
        'Quality assurance automation'
      ],
      useCases: [
        'Answer quality evaluation',
        'System performance monitoring',
        'A/B testing framework'
      ]
    },
    'V107': {
      fullDescription: 'Collaborative Multi-Agent Reasoning (CMAR) simulates multiple expert agents with different specializations debating and collaborating to reach consensus on complex queries.',
      technicalDetails: [
        '5 Expert Agent Types: Factual, Logical, Creative, Critical, Synthesis',
        'Multi-round debate mechanism',
        'Consensus building algorithms',
        'Confidence-weighted voting'
      ],
      benefits: [
        'Reduced bias through diversity',
        'Higher quality conclusions',
        'Identified edge cases',
        'Comprehensive analysis'
      ],
      useCases: [
        'Complex reasoning tasks',
        'Controversial topic analysis',
        'Multi-perspective synthesis'
      ]
    },
    'V108': {
      fullDescription: 'Uncertainty Quantification & Confidence Calibration provides statistical confidence intervals for answers, distinguishing between epistemic (knowledge) and aleatoric (inherent) uncertainty.',
      technicalDetails: [
        'Bayesian confidence estimation',
        'Temperature scaling calibration',
        'Confidence interval generation',
        'Uncertainty decomposition'
      ],
      benefits: [
        'Honest uncertainty representation',
        'Better decision support',
        'Risk-aware responses',
        'Improved user trust'
      ],
      useCases: [
        'High-stakes decision support',
        'Scientific query answering',
        'Medical information retrieval'
      ]
    },
    'V109': {
      fullDescription: 'Personalized Adaptation & User Modeling builds dynamic user profiles to adapt responses based on expertise level, preferred domains, and interaction patterns.',
      technicalDetails: [
        'Dynamic user profiling',
        'Expertise level detection',
        'Domain preference learning',
        'Adaptive response generation'
      ],
      benefits: [
        'Personalized user experience',
        'Appropriate explanation depth',
        'Relevant recommendations',
        'Improved learning outcomes'
      ],
      useCases: [
        'Educational applications',
        'Professional research tools',
        'Multi-audience platforms'
      ]
    }
  };

  return (
    <Layout>
      <div className="innovation-page">
        <div className="page-header">
          <h1>🚀 Innovation Showcase</h1>
          <p>Explore the 14 cutting-edge innovation modules powering PathRAG Agent</p>
        </div>

        <InnovationShowcase onModuleSelect={handleModuleSelect} />

        {/* Detail Modal */}
        <Modal
          open={showDetailModal}
          onClose={closeModal}
          size="lg"
          className="module-detail-modal"
        >
          {selectedModule && (
            <>
              <Modal.Header>
                <Modal.Title>
                  <div className="modal-title-content">
                    <div 
                      className="modal-icon"
                      style={{ backgroundColor: `${selectedModule.color}20`, color: selectedModule.color }}
                    >
                      <FontAwesomeIcon icon={selectedModule.icon} />
                    </div>
                    <div className="modal-title-text">
                      <span className="module-id">{selectedModule.id}</span>
                      <h3>{selectedModule.name}</h3>
                    </div>
                  </div>
                </Modal.Title>
              </Modal.Header>
              <Modal.Body>
                <div className="module-detail-content">
                  <Tag 
                    size="lg"
                    style={{ 
                      backgroundColor: selectedModule.color,
                      color: 'white',
                      marginBottom: 16
                    }}
                  >
                    {selectedModule.category}
                  </Tag>

                  <p className="module-description">
                    {moduleDetails[selectedModule.id]?.fullDescription || selectedModule.description}
                  </p>

                  {moduleDetails[selectedModule.id] && (
                    <>
                      <div className="detail-section">
                        <h4>🔧 Technical Details</h4>
                        <ul>
                          {moduleDetails[selectedModule.id].technicalDetails.map((detail, idx) => (
                            <li key={idx}>{detail}</li>
                          ))}
                        </ul>
                      </div>

                      <div className="detail-section">
                        <h4>✨ Benefits</h4>
                        <ul className="benefits-list">
                          {moduleDetails[selectedModule.id].benefits.map((benefit, idx) => (
                            <li key={idx}>{benefit}</li>
                          ))}
                        </ul>
                      </div>

                      <div className="detail-section">
                        <h4>🎯 Use Cases</h4>
                        <div className="use-cases">
                          {moduleDetails[selectedModule.id].useCases.map((useCase, idx) => (
                            <span key={idx} className="use-case-tag">{useCase}</span>
                          ))}
                        </div>
                      </div>
                    </>
                  )}

                  <div className="module-metrics">
                    <div className="metric">
                      <span className="metric-label">Status</span>
                      <Badge 
                        content={selectedModule.status.toUpperCase()}
                        style={{ 
                          backgroundColor: selectedModule.status === 'active' ? '#27AE60' : '#F39C12'
                        }}
                      />
                    </div>
                    <div className="metric">
                      <span className="metric-label">Impact</span>
                      <Badge 
                        content={selectedModule.impact.toUpperCase()}
                        style={{ 
                          backgroundColor: 
                            selectedModule.impact === 'very-high' ? '#E74C3C' :
                            selectedModule.impact === 'high' ? '#27AE60' : '#F39C12'
                        }}
                      />
                    </div>
                  </div>
                </div>
              </Modal.Body>
            </>
          )}
        </Modal>
      </div>
    </Layout>
  );
};

export default InnovationPage;
