import React, { useState } from 'react';
import { Panel, Grid, Row, Col, Tag, Progress, Timeline, Badge } from 'rsuite';
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome';
import {
  faRocket,
  faBrain,
  faShieldAlt,
  faMemory,
  faStream,
  faTable,
  faCode,
  faUsers,
  faChartLine,
  faUserCog,
  faCheckCircle,
  faFlask,
  faProjectDiagram,
  faNetworkWired,
  faEye,
  faUserSecret,
  faAtom,
  faCube,
  faDna,
  faExchangeAlt,
  faInfinity,
  faShieldVirus,
  faProjectDiagram as faGraph,
  faServer
} from '@fortawesome/free-solid-svg-icons';
import '../../styles/innovation-showcase.css';

// Innovation module data
const innovationModules = [
  {
    id: 'V89',
    name: 'Adaptive Evidence Compression',
    icon: faRocket,
    color: '#FF6B6B',
    category: 'Efficiency',
    description: 'Smart compression of evidence based on query complexity',
    status: 'active',
    impact: 'high'
  },
  {
    id: 'V90',
    name: 'Salience Reordering',
    icon: faBrain,
    color: '#4ECDC4',
    category: 'Efficiency',
    description: 'Dynamic reordering of evidence by importance',
    status: 'active',
    impact: 'high'
  },
  {
    id: 'V91',
    name: 'Fidelity Scoring System',
    icon: faCheckCircle,
    color: '#45B7D1',
    category: 'Quality',
    description: 'Multi-dimensional quality assessment (CAQ/CFS/NPS/SAS)',
    status: 'active',
    impact: 'high'
  },
  {
    id: 'V92',
    name: 'Iterative Refinement',
    icon: faFlask,
    color: '#96CEB4',
    category: 'Quality',
    description: 'Self-improving answer generation through iteration',
    status: 'active',
    impact: 'high'
  },
  {
    id: 'V93',
    name: 'Claim Evidence Traceability',
    icon: faStream,
    color: '#FFEAA7',
    category: 'Explainability',
    description: 'Fine-grained claim-to-evidence mapping',
    status: 'active',
    impact: 'high'
  },
  {
    id: 'V95',
    name: 'Adaptive Reasoning Path',
    icon: faBrain,
    color: '#DDA0DD',
    category: 'Adaptivity',
    description: 'Question-type aware reasoning strategies',
    status: 'active',
    impact: 'medium'
  },
  {
    id: 'V97',
    name: 'Intra-Answer Consistency',
    icon: faCheckCircle,
    color: '#98D8C8',
    category: 'Consistency',
    description: 'Global consistency checking within answers',
    status: 'active',
    impact: 'high'
  },
  {
    id: 'V99',
    name: 'Cross-Query Memory',
    icon: faMemory,
    color: '#F7DC6F',
    category: 'Memory',
    description: 'Evidence reuse across conversations',
    status: 'active',
    impact: 'high'
  },
  {
    id: 'V101',
    name: 'Adversarial Robustness',
    icon: faShieldAlt,
    color: '#BB8FCE',
    category: 'Security',
    description: 'Defense against adversarial attacks',
    status: 'active',
    impact: 'high'
  },
  {
    id: 'V103',
    name: 'Multi-Modal Fusion',
    icon: faTable,
    color: '#85C1E9',
    category: 'Multi-Modal',
    description: 'Integration of tables, figures, and code',
    status: 'active',
    impact: 'high'
  },
  {
    id: 'V105',
    name: 'Real-Time Stream Processing',
    icon: faStream,
    color: '#F8B739',
    category: 'Real-Time',
    description: 'Incremental evidence updates',
    status: 'active',
    impact: 'medium'
  },
  {
    id: 'V107',
    name: 'Collaborative Multi-Agent Reasoning',
    icon: faUsers,
    color: '#6C5CE7',
    category: 'Advanced',
    description: 'Multi-expert debate and consensus building',
    status: 'active',
    impact: 'very-high'
  },
  {
    id: 'V108',
    name: 'Uncertainty Quantification',
    icon: faChartLine,
    color: '#A29BFE',
    category: 'Advanced',
    description: 'Statistical confidence intervals and calibration',
    status: 'active',
    impact: 'very-high'
  },
  {
    id: 'V109',
    name: 'Personalized Adaptation',
    icon: faUserCog,
    color: '#74B9FF',
    category: 'Advanced',
    description: 'User modeling and adaptive responses',
    status: 'active',
    impact: 'high'
  },
  {
    id: 'V110',
    name: 'Causal Reasoning Enhancement',
    icon: faProjectDiagram,
    color: '#00B894',
    category: 'Advanced',
    description: 'Causal discovery and counterfactual reasoning',
    status: 'active',
    impact: 'very-high'
  },
  {
    id: 'V111',
    name: 'Knowledge Graph Expansion',
    icon: faNetworkWired,
    color: '#E17055',
    category: 'Advanced',
    description: 'Dynamic entity and relation extraction',
    status: 'active',
    impact: 'high'
  },
  {
    id: 'V112',
    name: 'Meta-Learning Optimization',
    icon: faBrain,
    color: '#6C5CE7',
    category: 'Advanced',
    description: 'Few-shot learning and adaptive optimization',
    status: 'active',
    impact: 'very-high'
  },
  {
    id: 'V113',
    name: 'Explainability Visualization',
    icon: faEye,
    color: '#FD79A8',
    category: 'Advanced',
    description: 'Interactive reasoning visualization engine',
    status: 'active',
    impact: 'high'
  },
  {
    id: 'V114',
    name: 'Federated Learning & Privacy',
    icon: faUserSecret,
    color: '#00CEC9',
    category: 'Advanced',
    description: 'Distributed learning with privacy preservation',
    status: 'active',
    impact: 'very-high'
  },
  // V128-V135: 方案三 - 终极体验创新
  {
    id: 'V128',
    name: 'Quantum-Inspired Retrieval',
    icon: faAtom,
    color: '#9B59B6',
    category: 'Ultimate',
    description: 'Quantum superposition, entanglement, and tunneling for retrieval',
    status: 'active',
    impact: 'breakthrough'
  },
  {
    id: 'V129',
    name: 'NeRF Knowledge Visualization',
    icon: faCube,
    color: '#E74C3C',
    category: 'Ultimate',
    description: '3D neural radiance fields for immersive knowledge exploration',
    status: 'active',
    impact: 'breakthrough'
  },
  {
    id: 'V130',
    name: 'Bio-Inspired Evolution',
    icon: faDna,
    color: '#27AE60',
    category: 'Ultimate',
    description: 'Genetic algorithms, immune systems, and ant colony optimization',
    status: 'active',
    impact: 'breakthrough'
  },
  {
    id: 'V131',
    name: 'Zero-Shot Cross-Modal Transfer',
    icon: faExchangeAlt,
    color: '#F39C12',
    category: 'Ultimate',
    description: 'Unified representation space for text, image, audio, and video',
    status: 'active',
    impact: 'breakthrough'
  },
  {
    id: 'V132',
    name: 'Continual Lifelong Learning',
    icon: faInfinity,
    color: '#3498DB',
    category: 'Ultimate',
    description: 'EWC, progressive networks, and memory replay for continuous learning',
    status: 'active',
    impact: 'breakthrough'
  },
  {
    id: 'V133',
    name: 'Adversarial Robustness Training',
    icon: faShieldVirus,
    color: '#E67E22',
    category: 'Ultimate',
    description: 'FGSM, PGD attacks and certified defenses for robust retrieval',
    status: 'active',
    impact: 'breakthrough'
  },
  {
    id: 'V134',
    name: 'GNN Reasoning Enhancement',
    icon: faGraph,
    color: '#1ABC9C',
    category: 'Ultimate',
    description: 'GCN, GAT, GIN, and heterogeneous GNN for knowledge graph reasoning',
    status: 'active',
    impact: 'breakthrough'
  },
  {
    id: 'V135',
    name: 'Edge Computing Deployment',
    icon: faServer,
    color: '#34495E',
    category: 'Ultimate',
    description: 'Model quantization, partitioning, and edge-cloud orchestration',
    status: 'active',
    impact: 'breakthrough'
  }
];

// Ultimate innovation statistics
const ultimateStats = {
  totalUltimateModules: 8,
  breakthroughTechnologies: 8,
  newDimensions: 8,
  version: 'V135'
};

// Category colors
const categoryColors = {
  'Efficiency': '#FF6B6B',
  'Quality': '#4ECDC4',
  'Explainability': '#FFEAA7',
  'Adaptivity': '#DDA0DD',
  'Consistency': '#98D8C8',
  'Memory': '#F7DC6F',
  'Security': '#BB8FCE',
  'Multi-Modal': '#85C1E9',
  'Real-Time': '#F8B739',
  'Advanced': '#6C5CE7',
  'Ultimate': '#E74C3C'
};

const InnovationCard = ({ module, onClick }) => (
  <div 
    className={`innovation-card ${module.status}`}
    onClick={() => onClick && onClick(module)}
    style={{ borderLeftColor: module.color }}
  >
    <div className="innovation-header">
      <div 
        className="innovation-icon"
        style={{ backgroundColor: `${module.color}20`, color: module.color }}
      >
        <FontAwesomeIcon icon={module.icon} />
      </div>
      <div className="innovation-meta">
        <span className="innovation-id">{module.id}</span>
        <Tag 
          size="sm"
          style={{ 
            backgroundColor: categoryColors[module.category],
            color: module.category === 'Efficiency' ? 'white' : '#333'
          }}
        >
          {module.category}
        </Tag>
      </div>
    </div>
    <h4 className="innovation-name">{module.name}</h4>
    <p className="innovation-description">{module.description}</p>
    <div className="innovation-footer">
      <Badge 
        content={module.impact.toUpperCase()}
        style={{ 
          backgroundColor: 
            module.impact === 'breakthrough' ? '#9B59B6' :
            module.impact === 'very-high' ? '#E74C3C' :
            module.impact === 'high' ? '#27AE60' : '#F39C12'
        }}
      />
      <span className={`status-indicator ${module.status}`}>
        <span className="status-dot"></span>
        {module.status}
      </span>
    </div>
  </div>
);

const InnovationShowcase = ({ onModuleSelect }) => {
  const [selectedCategory, setSelectedCategory] = useState('all');
  const [viewMode, setViewMode] = useState('grid'); // 'grid' or 'timeline'

  // Filter modules by category
  const filteredModules = selectedCategory === 'all' 
    ? innovationModules 
    : innovationModules.filter(m => m.category === selectedCategory);

  // Group modules by category
  const modulesByCategory = innovationModules.reduce((acc, module) => {
    if (!acc[module.category]) {
      acc[module.category] = [];
    }
    acc[module.category].push(module);
    return acc;
  }, {});

  // Calculate statistics
  const stats = {
    total: innovationModules.length,
    active: innovationModules.filter(m => m.status === 'active').length,
    highImpact: innovationModules.filter(m => m.impact === 'high' || m.impact === 'very-high' || m.impact === 'breakthrough').length,
    breakthrough: innovationModules.filter(m => m.impact === 'breakthrough').length,
    categories: Object.keys(modulesByCategory).length,
    ultimateModules: innovationModules.filter(m => m.category === 'Ultimate').length
  };

  return (
    <div className="innovation-showcase">
      <Panel className="showcase-header" header="🚀 PathRAG Innovation Showcase - Ultimate Experience V128-V135">
        <div className="showcase-stats">
          <div className="stat-item">
            <span className="stat-value">{stats.total}</span>
            <span className="stat-label">Total Modules</span>
          </div>
          <div className="stat-item">
            <span className="stat-value" style={{ color: '#27AE60' }}>{stats.active}</span>
            <span className="stat-label">Active</span>
          </div>
          <div className="stat-item">
            <span className="stat-value" style={{ color: '#E74C3C' }}>{stats.highImpact}</span>
            <span className="stat-label">High Impact</span>
          </div>
          <div className="stat-item">
            <span className="stat-value" style={{ color: '#9B59B6' }}>{stats.breakthrough}</span>
            <span className="stat-label">Breakthrough</span>
          </div>
          <div className="stat-item">
            <span className="stat-value">{stats.categories}</span>
            <span className="stat-label">Categories</span>
          </div>
          <div className="stat-item ultimate-badge">
            <span className="stat-value" style={{ color: '#E74C3C' }}>{stats.ultimateModules}</span>
            <span className="stat-label">Ultimate V128-V135</span>
          </div>
        </div>

        <div className="category-filters">
          <button 
            className={`filter-btn ${selectedCategory === 'all' ? 'active' : ''}`}
            onClick={() => setSelectedCategory('all')}
          >
            All
          </button>
          {Object.keys(modulesByCategory).map(category => (
            <button
              key={category}
              className={`filter-btn ${selectedCategory === category ? 'active' : ''}`}
              onClick={() => setSelectedCategory(category)}
              style={{ 
                '--category-color': categoryColors[category]
              }}
            >
              {category}
            </button>
          ))}
        </div>

        <div className="view-toggle">
          <button 
            className={`view-btn ${viewMode === 'grid' ? 'active' : ''}`}
            onClick={() => setViewMode('grid')}
          >
            Grid
          </button>
          <button 
            className={`view-btn ${viewMode === 'timeline' ? 'active' : ''}`}
            onClick={() => setViewMode('timeline')}
          >
            Timeline
          </button>
        </div>
      </Panel>

      {viewMode === 'grid' ? (
        <Grid fluid className="innovations-grid">
          <Row>
            {filteredModules.map(module => (
              <Col xs={24} sm={12} md={8} lg={6} key={module.id}>
                <InnovationCard 
                  module={module} 
                  onClick={onModuleSelect}
                />
              </Col>
            ))}
          </Row>
        </Grid>
      ) : (
        <Panel className="timeline-panel">
          <Timeline className="innovation-timeline">
            {filteredModules.map((module, index) => (
              <Timeline.Item
                key={module.id}
                dot={
                  <div 
                    className="timeline-dot"
                    style={{ backgroundColor: module.color }}
                  >
                    <FontAwesomeIcon icon={module.icon} />
                  </div>
                }
              >
                <div className="timeline-content">
                  <div className="timeline-header">
                    <span className="timeline-id">{module.id}</span>
                    <Tag 
                      size="sm"
                      style={{ 
                        backgroundColor: categoryColors[module.category],
                        color: 'white'
                      }}
                    >
                      {module.category}
                    </Tag>
                  </div>
                  <h4>{module.name}</h4>
                  <p>{module.description}</p>
                  <div className="timeline-meta">
                    <Badge 
                      content={module.impact.toUpperCase()}
                      style={{ 
                        backgroundColor: 
                          module.impact === 'very-high' ? '#E74C3C' :
                          module.impact === 'high' ? '#27AE60' : '#F39C12'
                      }}
                    />
                  </div>
                </div>
              </Timeline.Item>
            ))}
          </Timeline>
        </Panel>
      )}

      {/* Innovation Categories Overview */}
      <Panel className="categories-overview" header="📊 Category Overview">
        <Grid fluid>
          <Row>
            {Object.entries(modulesByCategory).map(([category, modules]) => (
              <Col xs={24} sm={12} md={8} key={category}>
                <div 
                  className="category-card"
                  style={{ borderTopColor: categoryColors[category] }}
                >
                  <div className="category-header">
                    <h4 style={{ color: categoryColors[category] }}>{category}</h4>
                    <span className="module-count">{modules.length} modules</span>
                  </div>
                  <div className="category-modules">
                    {modules.slice(0, 3).map(m => (
                      <div key={m.id} className="mini-module">
                        <FontAwesomeIcon icon={m.icon} style={{ color: m.color }} />
                        <span>{m.name}</span>
                      </div>
                    ))}
                    {modules.length > 3 && (
                      <div className="more-modules">
                        +{modules.length - 3} more
                      </div>
                    )}
                  </div>
                  <Progress.Line
                    percent={(modules.filter(m => m.status === 'active').length / modules.length) * 100}
                    strokeColor={categoryColors[category]}
                    showInfo={false}
                  />
                </div>
              </Col>
            ))}
          </Row>
        </Grid>
      </Panel>
    </div>
  );
};

export default InnovationShowcase;
