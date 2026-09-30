/**
 * V136-V143 创新模块展示中心
 * 
 * 展示 8 大创新领域的功能演示和交互界面
 */

import React, { useState } from 'react';
import { PanelGroup, Panel } from 'rsuite';
import 'rsuite/dist/rsuite.min.css';
import '../styles/innovation-hub.css';

// 模块配置数据
const INNOVATION_MODULES = [
  {
    id: 'V136',
    name: '多模态知识融合引擎',
    nameEn: 'Multimodal Knowledge Fusion',
    icon: '📊',
    color: '#4CAF50',
    description: '将图像、表格、公式、代码作为一等公民融入知识图谱推理过程',
    capabilities: ['表格提取', '公式解析', '代码分析', '跨模态融合'],
    endpoint: '/api/innovation/v136/multimodal-fusion',
    demoData: {
      question: 'PathRAG的性能如何？',
      evidence_chunks: [
        {
          text: `## 实验结果对比表\n\n| 模型 | 准确率 | 召回率 | F1分数 |\n|------|--------|--------|--------|\n| PathRAG v1 | 0.85 | 0.78 | 0.81 |\n| PathRAG v2 | 0.92 | 0.89 | 0.90 |\n\n性能提升公式：$F_1 = \\frac{2 \\cdot P \\cdot R}{P + R}$`,
          source_id: 'doc_001'
        },
        {
          text: `\`\`\`python\ndef retrieve_evidence(query, graph):\n    nodes = graph.search(query)\n    paths = find_relevant_paths(nodes)\n    return rank_by_relevance(paths)\n\`\`\`\n\n该算法的时间复杂度为 O(n log n)`,
          source_id: 'doc_002'
        }
      ]
    }
  },
  {
    id: 'V137',
    name: '时序推理增强',
    nameEn: 'Temporal Reasoning Enhancement',
    icon: '⏱️',
    color: '#2196F3',
    description: '支持时间维度的事件因果链推理，包括时间线构建与趋势预测',
    capabilities: ['时间表达式提取', '事件排序', '因果推断', '趋势分析'],
    endpoint: '/api/innovation/v137/temporal-reasoning',
    demoData: {
      question: 'PathRAG项目的发展历程是怎样的？',
      evidence_chunks: [
        { text: 'PathRAG 项目于 2023 年 3 月启动，最初版本在 6 月发布。', source_id: 't1' },
        { text: '2024 年初，团队引入了 VSR 机制，显著提升了答案准确性。', source_id: 't2' },
        { text: '到 2024 年底，系统已处理超过 100 万次查询请求。', source_id: 't3' }
      ]
    }
  },
  {
    id: 'V138',
    name: '多语言对齐层',
    nameEn: 'Cross-Lingual Alignment Layer',
    icon: '🌐',
    color: '#FF9800',
    description: '支持中英双语实体的深度语义对齐，包括翻译验证与文化感知',
    capabilities: ['双语实体识别', '术语翻译', '语义对齐', '文化感知'],
    endpoint: '/api/innovation/v138/cross-lingual',
    demoData: {
      question: '什么是PathRAG的核心技术？',
      evidence_chunks: [
        { text: 'PathRAG uses a novel retrieval-augmented generation approach with knowledge graphs.', source_id: 'en1' },
        { text: '该系统采用创新的检索增强生成方法，结合了知识图谱技术。', source_id: 'zh1' },
        { text: 'The neural network architecture includes transformer layers and attention mechanisms.', source_id: 'en2' }
      ]
    }
  },
  {
    id: 'V139',
    name: '隐私保护检索',
    nameEn: 'Privacy-Preserving Retrieval',
    icon: '🔒',
    color: '#9C27B0',
    description: '实现联邦学习+差分隐私的知识图谱查询，包括PII检测与脱敏',
    capabilities: ['PII检测', '数据脱敏', '访问控制', '审计日志'],
    endpoint: '/api/innovation/v139/privacy-retrieval',
    demoData: {
      question: '查询用户张三的信息',
      evidence_chunks: [
        { text: '用户张三（邮箱：zhangsan@example.com，电话：13812345678）于2024年1月15日提交了申请。', source_id: 'priv_1' },
        { text: '系统管理员李四的API密钥为 sk-abc123xyz456，请妥善保管。密码为 password@123', source_id: 'priv_2' },
        { text: '北京市海淀区中关村大街1号，公司账户余额为 ¥1,234,567.89', source_id: 'priv_3' }
      ],
      user_info: { user_id: 'admin_001', roles: ['admin'] }
    }
  },
  {
    id: 'V140',
    name: '可解释性仪表盘',
    nameEn: 'Explainability Dashboard',
    icon: '📈',
    color: '#00BCD4',
    description: '为前端提供完整的推理过程可视化数据，包括证据溯源与置信度分解',
    capabilities: ['证据树', '推理路径', '置信度分解', '模块贡献'],
    endpoint: '/api/innovation/v140/explainability-dashboard',
    demoData: {
      query: 'PathRAG的工作原理是什么？',
      answer: 'PathRAG 是一种基于知识图谱路径的检索增强生成系统，通过 VSR 机制确保答案质量。',
      final_confidence: 0.89,
      evidence_chunks: [
        { text: 'PathRAG 通过知识图谱路径检索来增强 RAG 系统的推理能力。', score: 0.92, source_id: 'ev_1' },
        { text: 'VSR (Verification, Scoring, Repair) 机制确保答案的准确性和完整性。', score: 0.88, source_id: 'ev_2' },
        { text: '证据融合模块整合来自多个来源的信息，提高答案的可靠性。', score: 0.75, source_id: 'ev_3' }
      ]
    }
  },
  {
    id: 'V141',
    name: '强化学习优化器 v2',
    nameEn: 'RL Optimizer v2',
    icon: '🎯',
    color: '#F44336',
    description: '基于PPO的端到端优化框架，动态调整检索策略与答案生成参数',
    capabilities: ['策略网络', '多目标奖励', '在线学习', '动作选择'],
    endpoint: '/api/innovation/v141/rl-optimizer',
    demoData: {
      question: '如何优化PathRAG的性能？',
      evidence_chunks: [
        { text: '证据1：关于PathRAG的基本介绍', score: 0.7 },
        { text: '证据2：详细的技术实现方案', score: 0.85 },
        { text: '证据3：实验结果对比分析', score: 0.65 }
      ],
      history_performance: [0.78, 0.82, 0.80, 0.85, 0.88]
    }
  },
  {
    id: 'V142',
    name: '增量图更新机制',
    nameEn: 'Incremental Graph Update',
    icon: '🔄',
    color: '#795548',
    description: '支持知识图谱的增量式更新，无需全量重建，包括版本管理与回滚',
    capabilities: ['变更检测', '影响评估', '版本管理', '一致性维护'],
    endpoint: '/api/innovation/v142/incremental-update',
    demoData: {
      current_entities: [
        { entity_id: 'ent_1', name: 'PathRAG', entity_type: 'system', properties: { version: 'v1.0' }, source_id: 'doc_old' },
        { entity_id: 'ent_2', name: 'Knowledge Graph', entity_type: 'component', properties: {}, source_id: 'doc_old' }
      ],
      current_relations: [
        { relation_id: 'rel_1', source_entity_id: 'ent_1', target_entity_id: 'ent_2', relation_type: 'uses', weight: 0.9 }
      ],
      new_documents: [
        { doc_id: 'doc_new_1', text: 'PathRAG v2.0 引入了新的 VSR 机制，大幅提升了系统的准确性和可解释性。新版本还支持多模态输入，包括图像、表格和代码。' }
      ]
    }
  },
  {
    id: 'V143',
    name: '边缘部署支持',
    nameEn: 'Edge Deployment Support',
    icon: '📱',
    color: '#607D8B',
    description: '支持轻量级模型部署到边缘设备，包括模型量化压缩与离线模式',
    capabilities: ['环境检测', '模型压缩', '缓存优化', '离线模式'],
    endpoint: '/api/innovation/v143/edge-deployment',
    demoData: {}
  }
];

function InnovationHubPage() {
  const [selectedModule, setSelectedModule] = useState(INNOVATION_MODULES[0]);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('demo');

  // 调用 API
  const callAPI = async (module) => {
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      let response;
      const baseUrl = process.env.REACT_APP_API_URL || 'http://localhost:8000';

      switch (module.id) {
        case 'V136':
          response = await fetch(`${baseUrl}${module.endpoint}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(module.demoData)
          });
          break;
        
        case 'V137':
          response = await fetch(`${baseUrl}${module.endpoint}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(module.demoData)
          });
          break;
        
        case 'V138':
          response = await fetch(`${baseUrl}${module.endpoint}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(module.demoData)
          });
          break;
        
        case 'V139':
          response = await fetch(`${baseUrl}${module.endpoint}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(module.demoData)
          });
          break;
        
        case 'V140':
          response = await fetch(`${baseUrl}${module.endpoint}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(module.demoData)
          });
          break;
        
        case 'V141':
          response = await fetch(`${baseUrl}${module.endpoint}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(module.demoData)
          });
          break;
        
        case 'V142':
          response = await fetch(`${baseUrl}${module.endpoint}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(module.demoData)
          });
          break;
        
        case 'V143':
          response = await fetch(`${baseUrl}/api/innovation/v143/system-resources`);
          break;
        
        default:
          throw new Error('Unknown module');
      }

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setResult(data);

    } catch (err) {
      console.error('API call error:', err);
      setError(err.message || 'Failed to call API');
    } finally {
      setLoading(false);
    }
  };

  // 渲染结果
  const renderResult = () => {
    if (loading) {
      return (
        <div className="loading-container">
          <div className="spinner"></div>
          <p>正在调用 {selectedModule.name} ...</p>
        </div>
      );
    }

    if (error) {
      return (
        <div className="error-container">
          <span className="error-icon">❌</span>
          <p>错误: {error}</p>
        </div>
      );
    }

    if (!result) {
      return (
        <div className="placeholder-container">
          <span className="placeholder-icon">🚀</span>
          <p>点击"运行演示"按钮查看 {selectedModule.name} 的效果</p>
        </div>
      );
    }

    // 根据不同模块渲染不同的结果
    return renderModuleResult(result.data);
  };

  const renderModuleResult = (data) => {
    if (!data) return null;

    switch (selectedModule.id) {
      case 'V136':
        return (
          <div className="result-content">
            <h4>📊 多模态检测结果</h4>
            <div className="stats-grid">
              {Object.entries(data.modality_statistics || {}).map(([key, value]) => (
                value > 0 && (
                  <div key={key} className="stat-card">
                    <span className="stat-value">{value}</span>
                    <span className="stat-label">{key}</span>
                  </div>
                )
              ))}
            </div>
            <h5>跨模态关系: {data.cross_modal_relations?.length || 0} 个</h5>
            <h5>融合证据: {data.fused_evidence?.length || 0} 个</h5>
            <pre className="result-json">{JSON.stringify(data, null, 2).substring(0, 2000)}</pre>
          </div>
        );

      case 'V137':
        return (
          <div className="result-content">
            <h4>⏱️ 时序分析结果</h4>
            {(data.timeline?.events || []).slice(0, 5).map((event, idx) => (
              <div key={idx} className="timeline-event">
                <span className="event-time">{event.inferred_time?.substring(0, 10) || '未知'}</span>
                <span className="event-text">{event.text?.substring(0, 80)}...</span>
              </div>
            ))}
            <h5>因果链: {data.causal_chains?.length || 0} 个</h5>
            <h5>趋势: {(data.trends || []).length} 个</h5>
          </div>
        );

      case 'V138':
        return (
          <div className="result-content">
            <h4>🌐 多语言对齐结果</h4>
            <div className="stats-grid">
              <div className="stat-card">
                <span className="stat-value">{data.statistics?.total_entities || 0}</span>
                <span className="stat-label">总实体数</span>
              </div>
              <div className="stat-card">
                <span className="stat-value">{data.statistics?.chinese_entities || 0}</span>
                <span className="stat-label">中文实体</span>
              </div>
              <div className="stat-card">
                <span className="stat-value">{data.statistics?.english_entities || 0}</span>
                <span className="stat-label">英文实体</span>
              </div>
              <div className="stat-card">
                <span className="stat-value">{data.statistics?.aligned_pairs || 0}</span>
                <span className="stat-label">对齐对数</span>
              </div>
            </div>
          </div>
        );

      case 'V139':
        return (
          <div className="result-content">
            <h4>🔒 隐私保护结果</h4>
            <div className="stats-grid">
              <div className="stat-card">
                <span className="stat-value">{data.statistics?.total_sensitive_entities || 0}</span>
                <span className="stat-label">敏感实体</span>
              </div>
              <div className="stat-card">
                <span className="stat-value">{data.anonymized_chunks?.length || 0}</span>
                <span className="stat-label">脱敏块数</span>
              </div>
            </div>
            {(data.anonymized_chunks || []).slice(0, 2).map((chunk, idx) => (
              <div key={idx} className="anonymized-example">
                <p><strong>原始:</strong> {chunk.original_text?.substring(0, 60)}...</p>
                <p><strong>脱敏:</strong> {chunk.anonymized_text?.substring(0, 60)}...</p>
              </div>
            ))}
          </div>
        );

      case 'V140':
        return (
          <div className="result-content">
            <h4>📈 可解释性分析</h4>
            <div className="stats-grid">
              <div className="stat-card">
                <span className="stat-value">{data.visualizations?.evidence_tree?.tree_info?.total_nodes || 0}</span>
                <span className="stat-label">证据节点</span>
              </div>
              <div className="stat-card">
                <span className="stat-value">{data.visualizations?.reasoning_path?.total_steps || 0}</span>
                <span className="stat-label">推理步骤</span>
              </div>
              <div className="stat-card">
                <span className="stat-value">{(data.visualizations?.confidence_breakdown?.total_confidence * 100)?.toFixed(0) || 0}%</span>
                <span className="stat-label">置信度</span>
              </div>
            </div>
            {(data.visualizations?.reasoning_path?.steps || []).map((step, idx) => (
              <div key={idx} className="reasoning-step">
                <span>{idx + 1}. {step.module_name}</span>
                <span className="confidence-change">({step.confidence_change > 0 ? '+' : ''}{(step.confidence_change * 100).toFixed(1)}%)</span>
              </div>
            ))}
          </div>
        );

      case 'V141':
        return (
          <div className="result-content">
            <h4>🎯 RL 优化建议</h4>
            <div className="stats-grid">
              <div className="stat-card">
                <span className="stat-value">{data.selected_action?.action_type || 'N/A'}</span>
                <span className="stat-label">推荐动作</span>
              </div>
              <div className="stat-card">
                <span className="stat-value">{(data.selected_action?.confidence * 100)?.toFixed(0) || 0}%</span>
                <span className="stat-label">置信度</span>
              </div>
            </div>
            {(data.optimization_suggestions || []).slice(0, 3).map((s, idx) => (
              <div key={idx} className={`suggestion ${s.priority}`}>
                [{s.priority}] {s.message}
              </div>
            ))}
          </div>
        );

      case 'V142':
        return (
          <div className="result-content">
            <h4>🔄 增量更新分析</h4>
            <div className="stats-grid">
              <div className="stat-card">
                <span className="stat-value">{data.detected_changes?.length || 0}</span>
                <span className="stat-label">检测到的变更</span>
              </div>
              <div className="stat-card">
                <span className="stat-value">{data.snapshot_info?.entity_count_after || 0}</span>
                <span className="stat-label">更新后实体数</span>
              </div>
              <div className="stat-card">
                <span className="stat-value">{data.snapshot_info?.new_version || 'N/A'}</span>
                <span className="stat-label">新版本号</span>
              </div>
            </div>
            {(data.recommendations || []).slice(0, 2).map((r, idx) => (
              <div key={idx} className={`recommendation ${r.type}`}>
                [{r.type}] {r.message}
              </div>
            ))}
          </div>
        );

      case 'V143':
        return (
          <div className="result-content">
            <h4>📱 边缘部署检测</h4>
            <div className="stats-grid">
              <div className="stat-card">
                <span className="stat-value">{data.system_resources?.cpu_count || 'N/A'}</span>
                <span className="stat-label">CPU 核心数</span>
              </div>
              <div className="stat-card">
                <span className="stat-value">{data.system_resources?.total_memory_mb || 0}</span>
                <span className="stat-label">总内存 (MB)</span>
              </div>
              <div className="stat-card">
                <span className="stat-value">{data.deployment_config?.device_tier || 'N/A'}</span>
                <span className="stat-label">设备层级</span>
              </div>
              <div className="stat-card">
                <span className="stat-value">{data.readiness_score || 0}</span>
                <span className="stat-label">就绪分数</span>
              </div>
            </div>
            <h5>部署模式: {data.deployment_config?.mode}</h5>
            <h5>启用模块 ({(data.deployment_config?.enabled_modules || []).length}个)</h5>
          </div>
        );

      default:
        return <pre className="result-json">{JSON.stringify(data, null, 2).substring(0, 3000)}</pre>;
    }
  };

  return (
    <div className="innovation-hub">
      {/* 头部 */}
      <header className="hub-header">
        <h1>🚀 PathRAG 创新中心</h1>
        <p>V136-V143 八大创新领域 · 功能演示与交互体验</p>
      </header>

      <PanelGroup direction="horizontal" style={{ height: 'calc(100vh - 120px)' }}>
        {/* 左侧：模块选择 */}
        <Panel defaultSize={25} minSize={200}>
          <div className="module-selector">
            <h3>📦 创新模块</h3>
            <div className="module-list">
              {INNOVATION_MODULES.map((module) => (
                <div
                  key={module.id}
                  className={`module-card ${selectedModule.id === module.id ? 'active' : ''}`}
                  onClick={() => {
                    setSelectedModule(module);
                    setResult(null);
                    setError(null);
                  }}
                  style={{ '--module-color': module.color }}
                >
                  <span className="module-icon">{module.icon}</span>
                  <div className="module-info">
                    <span className="module-id">{module.id}</span>
                    <span className="module-name">{module.name}</span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        </Panel>

        <div className="innovation-panel-divider" aria-hidden="true" />

        {/* 右侧：详情面板 */}
        <Panel defaultSize={75}>
          <div className="detail-panel">
            {/* 模块信息头部 */}
            <div className="module-header" style={{ background: `linear-gradient(135deg, ${selectedModule.color}22, ${selectedModule.color}44)` }}>
              <div className="header-left">
                <span className="large-icon">{selectedModule.icon}</span>
                <div>
                  <h2>{selectedModule.name}</h2>
                  <p className="name-en">{selectedModule.nameEn}</p>
                </div>
              </div>
              <button
                className="run-button"
                onClick={() => callAPI(selectedModule)}
                disabled={loading}
                style={{ backgroundColor: selectedModule.color }}
              >
                {loading ? '⏳ 运行中...' : '▶️ 运行演示'}
              </button>
            </div>

            {/* 标签页 */}
            <div className="tabs">
              <button
                className={`tab ${activeTab === 'demo' ? 'active' : ''}`}
                onClick={() => setActiveTab('demo')}
              >
                🧪 功能演示
              </button>
              <button
                className={`tab ${activeTab === 'info' ? 'active' : ''}`}
                onClick={() => setActiveTab('info')}
              >
                📖 模块介绍
              </button>
              <button
                className={`tab ${activeTab === 'api' ? 'active' : ''}`}
                onClick={() => setActiveTab('api')}
              >
                🔗 API 文档
              </button>
            </div>

            {/* 内容区域 */}
            <div className="content-area">
              {activeTab === 'demo' && (
                <div className="tab-content">
                  {renderResult()}
                </div>
              )}

              {activeTab === 'info' && (
                <div className="tab-content info-content">
                  <section>
                    <h4>📝 模块描述</h4>
                    <p>{selectedModule.description}</p>
                  </section>

                  <section>
                    <h4>✨ 核心能力</h4>
                    <ul className="capability-list">
                      {selectedModule.capabilities.map((cap, idx) => (
                        <li key={idx}>{cap}</li>
                      ))}
                    </ul>
                  </section>

                  <section>
                    <h4>🔧 技术细节</h4>
                    <table className="tech-table">
                      <tbody>
                        <tr>
                          <td>模块编号</td>
                          <td><code>{selectedModule.id}</code></td>
                        </tr>
                        <tr>
                          <td>API 端点</td>
                          <td><code>{selectedModule.endpoint}</code></td>
                        </tr>
                        <tr>
                          <td>状态</td>
                          <td><span className="status-badge active">Active</span></td>
                        </tr>
                      </tbody>
                    </table>
                  </section>
                </div>
              )}

              {activeTab === 'api' && (
                <div className="tab-content api-content">
                  <h4>API 端点</h4>
                  <code className="endpoint-code">POST {selectedModule.endpoint}</code>

                  <h4>请求示例</h4>
                  <pre className="code-block">
{JSON.stringify(selectedModule.demoData, null, 2).substring(0, 1500)}
                  </pre>

                  <h4>响应格式</h4>
                  <pre className="code-block">
{`{
  "success": true,
  "module": "${selectedModule.id}_...",
  "timestamp": "2024-01-01T00:00:00",
  "data": { ... }
}`}
                  </pre>
                </div>
              )}
            </div>
          </div>
        </Panel>
      </PanelGroup>
    </div>
  );
}

export default InnovationHubPage;
