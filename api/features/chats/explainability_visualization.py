"""
V113: Explainability Visualization Engine (EVE)
可解释性可视化引擎 - 将推理过程可视化

创新点：
1. 推理路径可视化：图形化展示推理步骤
2. 注意力热力图：展示证据重要性分布
3. 证据溯源图：交互式证据链展示
4. 置信度仪表盘：实时置信度监控
"""

import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import math


class VisualizationType(Enum):
    """可视化类型"""
    REASONING_PATH = "reasoning_path"
    ATTENTION_HEATMAP = "attention_heatmap"
    EVIDENCE_GRAPH = "evidence_graph"
    CONFIDENCE_GAUGE = "confidence_gauge"
    COMPARISON_CHART = "comparison_chart"


@dataclass
class Node:
    """图节点"""
    id: str
    label: str
    node_type: str
    x: float = 0
    y: float = 0
    properties: Dict[str, Any] = field(default_factory=dict)
    color: str = "#4A90E2"
    size: float = 20


@dataclass
class Edge:
    """图边"""
    source: str
    target: str
    label: str = ""
    weight: float = 1.0
    color: str = "#999"
    style: str = "solid"


@dataclass
class ReasoningStep:
    """推理步骤"""
    step_number: int
    title: str
    description: str
    input_data: Dict[str, Any]
    output_data: Dict[str, Any]
    confidence: float
    duration_ms: int
    sub_steps: List['ReasoningStep'] = field(default_factory=list)


class ReasoningPathVisualizer:
    """推理路径可视化器"""
    
    def __init__(self):
        self.steps: List[ReasoningStep] = []
    
    def add_step(
        self,
        title: str,
        description: str,
        input_data: Dict,
        output_data: Dict,
        confidence: float,
        duration_ms: int
    ) -> ReasoningStep:
        """添加推理步骤"""
        step = ReasoningStep(
            step_number=len(self.steps) + 1,
            title=title,
            description=description,
            input_data=input_data,
            output_data=output_data,
            confidence=confidence,
            duration_ms=duration_ms
        )
        self.steps.append(step)
        return step
    
    def visualize(self) -> Dict[str, Any]:
        """生成推理路径可视化数据"""
        # 创建节点和边
        nodes = []
        edges = []
        
        for i, step in enumerate(self.steps):
            # 步骤节点
            node = Node(
                id=f"step_{step.step_number}",
                label=step.title,
                node_type="reasoning_step",
                x=i * 150,
                y=100,
                properties={
                    'description': step.description,
                    'confidence': step.confidence,
                    'duration': step.duration_ms
                },
                color=self._get_confidence_color(step.confidence),
                size=30 + step.confidence * 20
            )
            nodes.append(node)
            
            # 连接边
            if i > 0:
                edge = Edge(
                    source=f"step_{self.steps[i-1].step_number}",
                    target=f"step_{step.step_number}",
                    label="leads to",
                    weight=step.confidence,
                    color=self._get_confidence_color(step.confidence)
                )
                edges.append(edge)
        
        return {
            'type': 'reasoning_path',
            'nodes': [
                {
                    'id': n.id,
                    'label': n.label,
                    'type': n.node_type,
                    'x': n.x,
                    'y': n.y,
                    'color': n.color,
                    'size': n.size,
                    'properties': n.properties
                }
                for n in nodes
            ],
            'edges': [
                {
                    'source': e.source,
                    'target': e.target,
                    'label': e.label,
                    'weight': e.weight,
                    'color': e.color
                }
                for e in edges
            ],
            'steps_summary': [
                {
                    'number': s.step_number,
                    'title': s.title,
                    'confidence': s.confidence,
                    'duration_ms': s.duration_ms
                }
                for s in self.steps
            ],
            'overall_confidence': sum(s.confidence for s in self.steps) / len(self.steps) if self.steps else 0
        }
    
    def _get_confidence_color(self, confidence: float) -> str:
        """根据置信度获取颜色"""
        if confidence >= 0.8:
            return "#27AE60"  # 绿色
        elif confidence >= 0.6:
            return "#F39C12"  # 橙色
        else:
            return "#E74C3C"  # 红色


class AttentionHeatmapGenerator:
    """注意力热力图生成器"""
    
    def __init__(self):
        self.attention_weights: Dict[str, List[float]] = {}
    
    def add_attention_layer(
        self,
        layer_name: str,
        tokens: List[str],
        weights: List[float]
    ):
        """添加注意力层数据"""
        self.attention_weights[layer_name] = {
            'tokens': tokens,
            'weights': weights
        }
    
    def generate_heatmap(
        self,
        layer_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """生成热力图数据"""
        if layer_name and layer_name in self.attention_weights:
            data = self.attention_weights[layer_name]
            return self._format_heatmap_data(layer_name, data)
        
        # 生成所有层的热力图
        heatmaps = []
        for name, data in self.attention_weights.items():
            heatmaps.append(self._format_heatmap_data(name, data))
        
        return {
            'type': 'attention_heatmap',
            'layers': heatmaps,
            'summary': {
                'total_layers': len(heatmaps),
                'average_attention_sparsity': self._calculate_sparsity()
            }
        }
    
    def _format_heatmap_data(
        self,
        layer_name: str,
        data: Dict
    ) -> Dict[str, Any]:
        """格式化热力图数据"""
        tokens = data['tokens']
        weights = data['weights']
        
        # 归一化权重到 0-1
        max_weight = max(weights) if weights else 1
        normalized_weights = [w / max_weight for w in weights]
        
        # 生成颜色
        colors = [self._weight_to_color(w) for w in normalized_weights]
        
        return {
            'layer_name': layer_name,
            'tokens': [
                {
                    'text': token,
                    'weight': weight,
                    'normalized_weight': norm_weight,
                    'color': color,
                    'intensity': int(norm_weight * 255)
                }
                for token, weight, norm_weight, color in zip(
                    tokens, weights, normalized_weights, colors
                )
            ],
            'statistics': {
                'max_attention': max(weights) if weights else 0,
                'min_attention': min(weights) if weights else 0,
                'mean_attention': sum(weights) / len(weights) if weights else 0,
                'focused_tokens': sum(1 for w in normalized_weights if w > 0.7)
            }
        }
    
    def _weight_to_color(self, weight: float) -> str:
        """将权重转换为颜色"""
        # 使用蓝-黄-红渐变
        if weight < 0.5:
            # 蓝色到黄色
            r = int(255 * weight * 2)
            g = int(255 * weight * 2)
            b = int(255 * (1 - weight * 2))
        else:
            # 黄色到红色
            r = 255
            g = int(255 * (1 - (weight - 0.5) * 2))
            b = 0
        
        return f"#{r:02x}{g:02x}{b:02x}"
    
    def _calculate_sparsity(self) -> float:
        """计算注意力稀疏度"""
        all_weights = []
        for data in self.attention_weights.values():
            all_weights.extend(data['weights'])
        
        if not all_weights:
            return 0.0
        
        # 计算有多少权重集中在少数token上
        sorted_weights = sorted(all_weights, reverse=True)
        top_20_percent = int(len(sorted_weights) * 0.2)
        top_sum = sum(sorted_weights[:top_20_percent])
        total_sum = sum(sorted_weights)
        
        return top_sum / total_sum if total_sum > 0 else 0.0


class EvidenceGraphBuilder:
    """证据图谱构建器"""
    
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []
    
    def add_evidence_node(
        self,
        evidence_id: str,
        content: str,
        source: str,
        confidence: float,
        evidence_type: str = "text"
    ):
        """添加证据节点"""
        node = Node(
            id=evidence_id,
            label=content[:50] + "..." if len(content) > 50 else content,
            node_type="evidence",
            properties={
                'full_content': content,
                'source': source,
                'confidence': confidence,
                'evidence_type': evidence_type
            },
            color=self._get_evidence_color(confidence, evidence_type),
            size=20 + confidence * 15
        )
        self.nodes[evidence_id] = node
    
    def add_claim_node(
        self,
        claim_id: str,
        claim_text: str,
        confidence: float
    ):
        """添加声明节点"""
        node = Node(
            id=claim_id,
            label=claim_text[:50] + "..." if len(claim_text) > 50 else claim_text,
            node_type="claim",
            properties={
                'full_text': claim_text,
                'confidence': confidence
            },
            color="#9B59B6",
            size=25
        )
        self.nodes[claim_id] = node
    
    def add_support_edge(
        self,
        evidence_id: str,
        claim_id: str,
        support_strength: float
    ):
        """添加支持边"""
        edge = Edge(
            source=evidence_id,
            target=claim_id,
            label="supports",
            weight=support_strength,
            color=self._get_support_color(support_strength),
            style="solid" if support_strength > 0.7 else "dashed"
        )
        self.edges.append(edge)
    
    def build_graph(self) -> Dict[str, Any]:
        """构建证据图谱"""
        # 计算布局（简化的力导向布局）
        self._calculate_layout()
        
        return {
            'type': 'evidence_graph',
            'nodes': [
                {
                    'id': n.id,
                    'label': n.label,
                    'type': n.node_type,
                    'x': n.x,
                    'y': n.y,
                    'color': n.color,
                    'size': n.size,
                    'properties': n.properties
                }
                for n in self.nodes.values()
            ],
            'edges': [
                {
                    'source': e.source,
                    'target': e.target,
                    'label': e.label,
                    'weight': e.weight,
                    'color': e.color,
                    'style': e.style
                }
                for e in self.edges
            ],
            'statistics': {
                'total_evidence': sum(1 for n in self.nodes.values() if n.node_type == 'evidence'),
                'total_claims': sum(1 for n in self.nodes.values() if n.node_type == 'claim'),
                'total_relations': len(self.edges),
                'average_confidence': sum(n.properties.get('confidence', 0) for n in self.nodes.values()) / len(self.nodes) if self.nodes else 0
            }
        }
    
    def _calculate_layout(self):
        """计算节点布局"""
        # 简化的布局：证据在左，声明在右
        evidence_nodes = [n for n in self.nodes.values() if n.node_type == 'evidence']
        claim_nodes = [n for n in self.nodes.values() if n.node_type == 'claim']
        
        # 证据节点垂直排列
        for i, node in enumerate(evidence_nodes):
            node.x = 100
            node.y = 100 + i * 80
        
        # 声明节点垂直排列
        for i, node in enumerate(claim_nodes):
            node.x = 400
            node.y = 100 + i * 100
    
    def _get_evidence_color(self, confidence: float, evidence_type: str) -> str:
        """获取证据颜色"""
        base_colors = {
            'text': '#3498DB',
            'table': '#E67E22',
            'figure': '#27AE60',
            'code': '#9B59B6'
        }
        
        # 根据置信度调整透明度
        base_color = base_colors.get(evidence_type, '#3498DB')
        return base_color
    
    def _get_support_color(self, strength: float) -> str:
        """获取支持强度颜色"""
        if strength >= 0.8:
            return "#27AE60"
        elif strength >= 0.5:
            return "#F39C12"
        else:
            return "#E74C3C"


class ConfidenceGaugeBuilder:
    """置信度仪表盘构建器"""
    
    def __init__(self):
        self.metrics: Dict[str, float] = {}
    
    def add_metric(self, name: str, value: float, threshold_good: float = 0.8):
        """添加指标"""
        self.metrics[name] = {
            'value': value,
            'threshold': threshold_good,
            'status': 'good' if value >= threshold_good else 'warning' if value >= 0.6 else 'poor'
        }
    
    def build_gauge(self) -> Dict[str, Any]:
        """构建仪表盘"""
        gauges = []
        
        for name, data in self.metrics.items():
            gauge = {
                'metric_name': name,
                'value': data['value'],
                'percentage': data['value'] * 100,
                'threshold': data['threshold'],
                'status': data['status'],
                'color': self._get_status_color(data['status']),
                'angle': data['value'] * 180  # 半圆仪表盘
            }
            gauges.append(gauge)
        
        # 计算总体评分
        overall_score = sum(d['value'] for d in self.metrics.values()) / len(self.metrics) if self.metrics else 0
        
        return {
            'type': 'confidence_gauge',
            'gauges': gauges,
            'overall_score': overall_score,
            'overall_status': 'good' if overall_score >= 0.8 else 'warning' if overall_score >= 0.6 else 'poor',
            'summary': {
                'metrics_count': len(gauges),
                'good_metrics': sum(1 for g in gauges if g['status'] == 'good'),
                'warning_metrics': sum(1 for g in gauges if g['status'] == 'warning'),
                'poor_metrics': sum(1 for g in gauges if g['status'] == 'poor')
            }
        }
    
    def _get_status_color(self, status: str) -> str:
        """获取状态颜色"""
        colors = {
            'good': '#27AE60',
            'warning': '#F39C12',
            'poor': '#E74C3C'
        }
        return colors.get(status, '#95A5A6')


# 主引擎类
class ExplainabilityVisualizationEngine:
    """可解释性可视化引擎"""
    
    def __init__(self):
        self.reasoning_visualizer = ReasoningPathVisualizer()
        self.attention_generator = AttentionHeatmapGenerator()
        self.evidence_builder = EvidenceGraphBuilder()
        self.gauge_builder = ConfidenceGaugeBuilder()
    
    def create_full_visualization(
        self,
        reasoning_steps: List[Dict],
        attention_data: Dict[str, Any],
        evidence_data: List[Dict],
        confidence_metrics: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        创建完整的可视化
        
        这是V113创新的主要入口函数
        """
        # 1. 构建推理路径可视化
        for step in reasoning_steps:
            self.reasoning_visualizer.add_step(
                title=step.get('title', f'Step {step.get("number", 0)}'),
                description=step.get('description', ''),
                input_data=step.get('input', {}),
                output_data=step.get('output', {}),
                confidence=step.get('confidence', 0.5),
                duration_ms=step.get('duration_ms', 0)
            )
        
        reasoning_viz = self.reasoning_visualizer.visualize()
        
        # 2. 生成注意力热力图
        for layer_name, data in attention_data.items():
            self.attention_generator.add_attention_layer(
                layer_name,
                data.get('tokens', []),
                data.get('weights', [])
            )
        
        attention_viz = self.attention_generator.generate_heatmap()
        
        # 3. 构建证据图谱
        for evidence in evidence_data:
            self.evidence_builder.add_evidence_node(
                evidence_id=evidence.get('id', ''),
                content=evidence.get('content', ''),
                source=evidence.get('source', ''),
                confidence=evidence.get('confidence', 0),
                evidence_type=evidence.get('type', 'text')
            )
            
            # 添加声明节点和支持边
            for claim in evidence.get('supports', []):
                claim_id = f"claim_{claim.get('id', '')}"
                self.evidence_builder.add_claim_node(
                    claim_id=claim_id,
                    claim_text=claim.get('text', ''),
                    confidence=claim.get('confidence', 0)
                )
                self.evidence_builder.add_support_edge(
                    evidence_id=evidence.get('id', ''),
                    claim_id=claim_id,
                    support_strength=claim.get('support_strength', 0.5)
                )
        
        evidence_viz = self.evidence_builder.build_graph()
        
        # 4. 构建置信度仪表盘
        for metric_name, value in confidence_metrics.items():
            self.gauge_builder.add_metric(metric_name, value)
        
        gauge_viz = self.gauge_builder.build_gauge()
        
        return {
            'version': 'V113',
            'name': 'Explainability Visualization Engine (EVE)',
            'visualizations': {
                'reasoning_path': reasoning_viz,
                'attention_heatmap': attention_viz,
                'evidence_graph': evidence_viz,
                'confidence_gauge': gauge_viz
            },
            'interactive_elements': [
                {
                    'type': 'node_click',
                    'description': 'Click on nodes to see detailed information'
                },
                {
                    'type': 'edge_hover',
                    'description': 'Hover over edges to see relationship strength'
                },
                {
                    'type': 'zoom_pan',
                    'description': 'Zoom and pan to explore the graph'
                }
            ],
            'export_formats': ['json', 'svg', 'png', 'interactive_html']
        }
    
    def generate_comparison_visualization(
        self,
        baseline_results: Dict[str, Any],
        improved_results: Dict[str, Any],
        metrics: List[str]
    ) -> Dict[str, Any]:
        """生成对比可视化"""
        comparison_data = []
        
        for metric in metrics:
            baseline_val = baseline_results.get(metric, 0)
            improved_val = improved_results.get(metric, 0)
            improvement = ((improved_val - baseline_val) / baseline_val * 100) if baseline_val > 0 else 0
            
            comparison_data.append({
                'metric': metric,
                'baseline': baseline_val,
                'improved': improved_val,
                'improvement': improvement,
                'improvement_type': 'positive' if improvement > 0 else 'negative' if improvement < 0 else 'neutral'
            })
        
        return {
            'type': 'comparison_chart',
            'data': comparison_data,
            'chart_type': 'bar',
            'summary': {
                'metrics_improved': sum(1 for d in comparison_data if d['improvement'] > 0),
                'metrics_degraded': sum(1 for d in comparison_data if d['improvement'] < 0),
                'average_improvement': sum(d['improvement'] for d in comparison_data) / len(comparison_data) if comparison_data else 0
            }
        }


# 便捷函数
def build_explainability_visualization(
    reasoning_steps: List[Dict],
    attention_data: Dict[str, Any],
    evidence_data: List[Dict],
    confidence_metrics: Dict[str, float]
) -> Dict[str, Any]:
    """
    构建可解释性可视化
    
    这是V113创新的便捷入口函数
    """
    engine = ExplainabilityVisualizationEngine()
    return engine.create_full_visualization(
        reasoning_steps,
        attention_data,
        evidence_data,
        confidence_metrics
    )


# 示例用法
if __name__ == "__main__":
    # 示例数据
    reasoning_steps = [
        {
            'title': 'Query Analysis',
            'description': 'Analyzing query intent and type',
            'input': {'query': 'What is PathRAG?'},
            'output': {'query_type': 'factual', 'complexity': 'low'},
            'confidence': 0.9,
            'duration_ms': 150
        },
        {
            'title': 'Evidence Retrieval',
            'description': 'Retrieving relevant evidence from knowledge graph',
            'input': {'query_type': 'factual'},
            'output': {'evidence_count': 5, 'sources': ['paper', 'documentation']},
            'confidence': 0.85,
            'duration_ms': 300
        },
        {
            'title': 'Answer Generation',
            'description': 'Synthesizing final answer',
            'input': {'evidence': 5},
            'output': {'answer': 'PathRAG is...', 'confidence': 0.88},
            'confidence': 0.88,
            'duration_ms': 200
        }
    ]
    
    attention_data = {
        'layer_1': {
            'tokens': ['What', 'is', 'PathRAG', '?'],
            'weights': [0.2, 0.3, 0.8, 0.1]
        }
    }
    
    evidence_data = [
        {
            'id': 'ev1',
            'content': 'PathRAG is a novel retrieval-augmented generation system',
            'source': 'research_paper',
            'confidence': 0.95,
            'type': 'text',
            'supports': [
                {'id': 'c1', 'text': 'PathRAG is a RAG system', 'confidence': 0.9, 'support_strength': 0.95}
            ]
        }
    ]
    
    confidence_metrics = {
        'evidence_quality': 0.88,
        'reasoning_consistency': 0.92,
        'answer_completeness': 0.85,
        'source_reliability': 0.90
    }
    
    result = build_explainability_visualization(
        reasoning_steps,
        attention_data,
        evidence_data,
        confidence_metrics
    )
    
    print(json.dumps(result, indent=2))
