"""
V140 - Explainability Dashboard API (可解释性仪表盘)
====================================================

为前端提供完整的推理过程可视化数据，包括：
- 证据溯源追踪
- 推理路径可视化
- 置信度分解
- 模块贡献分析
- 对比实验支持

核心能力：
- 生成结构化的推理过程数据
- 计算各模块的贡献分数
- 提供证据链的可视化格式
- 支持导出详细报告
"""

from __future__ import annotations

import json
import time
import hashlib
from typing import Any, Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from collections import defaultdict


class VisualizationType(Enum):
    """可视化类型"""
    EVIDENCE_TREE = "evidence_tree"           # 证据树
    REASONING_PATH = "reasoning_path"         # 推理路径图
    CONFIDENCE_BREAKDOWN = "confidence_breakdown"  # 置信度分解
    MODULE_CONTRIBUTION = "module_contribution"    # 模块贡献热力图
    TIMELINE_VIEW = "timeline_view"           # 时间线视图
    COMPARISON_TABLE = "comparison_table"     # 对比表格
    KNOWLEDGE_GRAPH_SNAPSHOT = "kg_snapshot"  # 知识图谱快照


@dataclass
class EvidenceNode:
    """证据节点（用于证据树）"""
    node_id: str
    content: str
    source_id: str
    relevance_score: float
    confidence: float = 0.0
    node_type: str = "evidence"  # evidence, intermediate, final
    children: List[str] = field(default_factory=list)  # 子节点ID列表
    parent_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'node_id': self.node_id,
            'content': self.content[:200],
            'source_id': self.source_id,
            'relevance_score': round(self.relevance_score, 4),
            'confidence': round(self.confidence, 4),
            'node_type': self.node_type,
            'children': self.children,
            'parent_id': self.parent_id,
            'metadata': self.metadata
        }


@dataclass
class ReasoningStep:
    """推理步骤"""
    step_id: int
    module_name: str
    description: str
    input_summary: str
    output_summary: str
    duration_ms: float
    confidence_change: float  # 该步骤对置信度的改变
    key_decisions: List[Dict[str, Any]] = field(default_factory=list)
    artifacts: Dict[str, Any] = field(default_factory=dict)  # 中间产物
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'step_id': self.step_id,
            'module_name': self.module_name,
            'description': self.description,
            'input_summary': self.input_summary[:100],
            'output_summary': self.output_summary[:100],
            'duration_ms': round(self.duration_ms, 2),
            'confidence_change': round(self.confidence_change, 4),
            'key_decisions': self.key_decisions,
            'artifacts': self.artifacts
        }


@dataclass
class ModuleContribution:
    """模块贡献记录"""
    module_name: str
    version: str
    execution_time_ms: float
    input_size: int  # 输入数据量
    output_size: int  # 输出数据量
    confidence_contribution: float  # 对最终置信度的贡献
    quality_score: float  # 输出质量评分
    enabled: bool = True
    errors: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'module_name': self.module_name,
            'version': self.version,
            'execution_time_ms': round(self.execution_time_ms, 2),
            'input_size': self.input_size,
            'output_size': self.output_size,
            'confidence_contribution': round(self.confidence_contribution, 4),
            'quality_score': round(self.quality_score, 4),
            'enabled': self.enabled,
            'errors': self.errors
        }


@dataclass
class ExplanationReport:
    """解释报告"""
    report_id: str
    query: str
    answer: str
    final_confidence: float
    timestamp: datetime
    total_processing_time_ms: float
    evidence_nodes: List[EvidenceNode]
    reasoning_steps: List[ReasoningStep]
    module_contributions: List[ModuleContribution]
    summary: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


class EvidenceTreeBuilder:
    """证据树构建器"""
    
    @classmethod
    def build_tree(
        cls,
        question: str,
        evidence_chunks: List[Dict[str, Any]],
        relevance_scores: Optional[List[float]] = None,
        max_depth: int = 3
    ) -> Tuple[List[EvidenceNode], Dict[str, Any]]:
        """
        构建证据树
        
        Returns:
            (节点列表, 树结构信息)
        """
        nodes = []
        
        if relevance_scores is None:
            relevance_scores = [0.5] * len(evidence_chunks)
        
        # 创建根节点（问题）
        root_node = EvidenceNode(
            node_id="root",
            content=question,
            source_id="query",
            relevance_score=1.0,
            confidence=1.0,
            node_type="query_root"
        )
        nodes.append(root_node)
        
        # 创建证据节点
        for idx, chunk in enumerate(evidence_chunks):
            chunk_text = chunk.get('text', '')
            chunk_id = chunk.get('source_id', f'chunk_{idx}')
            
            # 截断长文本
            display_text = chunk_text[:300] + "..." if len(chunk_text) > 300 else chunk_text
            
            node = EvidenceNode(
                node_id=f"ev_{idx}",
                content=display_text,
                source_id=chunk_id,
                relevance_score=relevance_scores[idx] if idx < len(relevance_scores) else 0.5,
                confidence=chunk.get('confidence', 0.7),
                node_type="evidence",
                parent_id="root",
                metadata={
                    'original_length': len(chunk_text),
                    'has_code': any(kw in chunk_text for kw in ['def ', 'function', 'class ', '{', '}']),
                    'has_table': '|' in chunk_text and '--' in chunk_text,
                    'has_formula': any(c in chunk_text for c in ['∑', '∫', '√', '∂'])
                }
            )
            nodes.append(node)
            root_node.children.append(node.node_id)
        
        # 构建子关系（基于内容相似性）
        cls._build_child_relations(nodes, max_depth)
        
        tree_info = {
            'total_nodes': len(nodes),
            'max_depth': cls._calculate_max_depth(nodes),
            'branching_factor': sum(len(n.children) for n in nodes if n.node_type == "root") / max(1, len([n for n in nodes if n.node_type == "root"])),
            'leaf_count': sum(1 for n in nodes if not n.children),
            'avg_relevance': sum(n.relevance_score for n in nodes if n.node_type == "evidence") / max(1, sum(1 for n in nodes if n.node_type == "evidence"))
        }
        
        return nodes, tree_info
    
    @classmethod
    def _build_child_relations(cls, nodes: List[EvidenceNode], max_depth: int):
        """构建子节点关系"""
        evidence_nodes = [n for n in nodes if n.node_type == "evidence"]
        
        for i, node_i in enumerate(evidence_nodes):
            for j, node_j in enumerate(evidence_nodes[i+1:], i+1):
                # 基于简单的内容重叠判断是否应该建立父子关系
                overlap = cls._content_overlap(node_i.content, node_j.content)
                
                if overlap > 0.3 and len(node_i.children) < 2:  # 限制分支数
                    node_i.children.append(node_j.node_id)
                    node_j.parent_id = node_i.node_id
                    node_j.node_type = "supporting_evidence"
    
    @staticmethod
    def _content_overlap(text1: str, text2: str) -> float:
        """计算文本内容重叠度"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2:
            return 0.0
        intersection = words1 & words2
        union = words1 | words2
        return len(intersection) / len(union)
    
    @staticmethod
    def _calculate_max_depth(nodes: List[EvidenceNode]) -> int:
        """计算最大深度"""
        depths = {}
        
        def get_depth(node_id: str) -> int:
            if node_id in depths:
                return depths[node_id]
            node = next((n for n in nodes if n.node_id == node_id), None)
            if node is None or node.parent_id is None:
                return 0
            depth = get_depth(node.parent_id) + 1
            depths[node_id] = depth
            return depth
        
        return max((get_depth(n.node_id) for n in nodes), default=0)


class ReasoningPathBuilder:
    """推理路径构建器"""
    
    @classmethod
    def build_path(
        cls,
        module_executions: List[Dict[str, Any]],
        initial_confidence: float = 0.5,
        final_confidence: float = 0.85
    ) -> List[ReasoningStep]:
        """
        构建推理路径
        
        Args:
            module_executions: 各模块执行记录 [{module_name, input, output, duration, ...}]
            initial_confidence: 初始置信度
            final_confidence: 最终置信度
        
        Returns:
            推理步骤列表
        """
        steps = []
        current_confidence = initial_confidence
        
        if not module_executions:
            # 默认路径
            steps.append(ReasoningStep(
                step_id=0,
                module_name="default_pipeline",
                description="Default reasoning pipeline",
                input_summary="Question and evidence",
                output_summary="Generated answer",
                duration_ms=100.0,
                confidence_change=final_confidence - initial_confidence
            ))
            return steps
        
        # 计算每个步骤的置信度变化
        total_steps = len(module_executions)
        remaining_delta = final_confidence - initial_confidence
        
        for idx, exec_record in enumerate(module_executions):
            is_last = (idx == total_steps - 1)
            
            if is_last:
                conf_change = remaining_delta
            else:
                # 分配部分置信度增量
                weight = exec_record.get('quality_score', 0.7)
                conf_change = (remaining_delta * weight) / total_steps
                remaining_delta -= conf_change
            
            step = ReasoningStep(
                step_id=idx,
                module_name=exec_record.get('module_name', f'module_{idx}'),
                description=exec_record.get('description', f'Step {idx}'),
                input_summary=exec_record.get('input_summary', ''),
                output_summary=exec_record.get('output_summary', ''),
                duration_ms=exec_record.get('duration_ms', 0),
                confidence_change=conf_change,
                key_decisions=exec_record.get('key_decisions', []),
                artifacts=exec_record.get('artifacts', {})
            )
            steps.append(step)
            current_confidence += conf_change
        
        return steps


class ConfidenceDecomposer:
    """置信度分解器"""
    
    @classmethod
    def decompose(
        cls,
        final_confidence: float,
        module_contributions: List[ModuleContribution],
        evidence_quality_scores: List[float]
    ) -> Dict[str, Any]:
        """
        分解最终置信度到各个来源
        
        Returns:
            {
                'total_confidence': float,
                'components': {...},
                'dominant_factor': str,
                'weaknesses': [...]
            }
        """
        components = {}
        
        # 1. 模块贡献组件
        module_total = sum(mc.confidence_contribution for mc in module_contributions)
        if module_total > 0:
            for mc in module_contributions:
                normalized = mc.confidence_contribution / module_total
                components[f"module_{mc.module_name}"] = {
                    'raw_contribution': mc.confidence_contribution,
                    'normalized': round(normalized, 4),
                    'weight_in_final': round(normalized * final_confidence, 4)
                }
        
        # 2. 证据质量组件
        if evidence_quality_scores:
            avg_evidence_quality = sum(evidence_quality_scores) / len(evidence_quality_scores)
            components['evidence_quality'] = {
                'average_score': round(avg_evidence_quality, 4),
                'count': len(evidence_quality_scores),
                'min': round(min(evidence_quality_scores), 4),
                'max': round(max(evidence_quality_scores), 4)
            }
        
        # 3. 综合分析
        dominant_factor = cls._find_dominant_factor(components)
        weaknesses = cls._identify_weaknesses(components)
        
        return {
            'total_confidence': round(final_confidence, 4),
            'components': components,
            'dominant_factor': dominant_factor,
            'weaknesses': weaknesses,
            'recommendation': cls._generate_recommendation(dominant_factor, weaknesses)
        }
    
    @staticmethod
    def _find_dominant_factor(components: Dict) -> str:
        """找出主导因素"""
        max_component = ""
        max_value = 0
        
        for key, value in components.items():
            if isinstance(value, dict):
                if 'weight_in_final' in value and value['weight_in_final'] > max_value:
                    max_value = value['weight_in_final']
                    max_component = key
                elif 'average_score' in value and value['average_score'] > max_value:
                    max_value = value['average_score']
                    max_component = key
        
        return max_component or "unknown"
    
    @staticmethod
    def _identify_weaknesses(components: Dict) -> List[str]:
        """识别弱点"""
        weaknesses = []
        
        for key, value in components.items():
            if isinstance(value, dict):
                if 'normalized' in value and value['normalized'] < 0.05:
                    weaknesses.append(f"{key}: Low contribution ({value['normalized']:.2%})")
                elif 'min' in value and value['min'] < 0.3:
                    weaknesses.append(f"{key}: Has low-quality items (min={value['min']:.2f})")
        
        return weaknesses
    
    @staticmethod
    def _generate_recommendation(dominant_factor: str, weaknesses: List[str]) -> str:
        """生成建议"""
        if not weaknesses:
            return "System performance is well-balanced."
        
        if dominant_factor.startswith("module_"):
            return f"Dominated by {dominant_factor}. Consider balancing other modules."
        
        return f"Weaknesses found: {'; '.join(weaknesses[:3])}"


def build_explainability_dashboard_data(
    query: str,
    answer: str,
    final_confidence: float,
    evidence_chunks: List[Dict[str, Any]],
    module_executions: Optional[List[Dict[str, Any]]] = None,
    processing_metadata: Optional[Dict[str, Any]] = None,
    include_visualizations: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    构建可解释性仪表盘数据
    
    Args:
        query: 用户查询
        answer: 生成的答案
        final_confidence: 最终置信度
        evidence_chunks: 使用的证据块
        module_executions: 各模块执行记录
        processing_metadata: 处理元数据
        include_visualizations: 要包含的可视化类型
    
    Returns:
        完整的仪表盘数据结构
    """
    start_time = time.time()
    
    if include_visualizations is None:
        include_visualizations = [
            VisualizationType.EVIDENCE_TREE.value,
            VisualizationType.REASONING_PATH.value,
            VisualizationType.CONFIDENCE_BREAKDOWN.value,
            VisualizationType.MODULE_CONTRIBUTION.value
        ]
    
    result = {
        'dashboard_version': 'V140_Explainability_Dashboard_v1',
        'report_id': hashlib.md5(f"{query}_{time.time()}".encode()).hexdigest()[:12],
        'timestamp': datetime.now().isoformat(),
        'query': query,
        'answer': answer,
        'final_confidence': final_confidence,
        'visualizations': {},
        'summary': {},
        'export_options': ['json', 'detailed_json']
    }
    
    # 1. 证据树
    if VisualizationType.EVIDENCE_TREE.value in include_visualizations:
        relevance_scores = [c.get('relevance_score', c.get('score', 0.5)) 
                          for c in evidence_chunks]
        evidence_nodes, tree_info = EvidenceTreeBuilder.build_tree(
            query, evidence_chunks, relevance_scores
        )
        result['visualizations']['evidence_tree'] = {
            'nodes': [n.to_dict() for n in evidence_nodes],
            'tree_info': tree_info
        }
    
    # 2. 推理路径
    if VisualizationType.REASONING_PATH.value in include_visualizations:
        if module_executions:
            reasoning_steps = ReasoningPathBuilder.build_path(
                module_executions,
                initial_confidence=processing_metadata.get('initial_confidence', 0.3) if processing_metadata else 0.3,
                final_confidence=final_confidence
            )
        else:
            # 从元数据推断模块执行
            inferred_modules = []
            if processing_metadata:
                modules_used = processing_metadata.get('modules_used', [])
                for mod in modules_used:
                    inferred_modules.append({
                        'module_name': mod.get('name', mod) if isinstance(mod, dict) else mod,
                        'description': mod.get('description', '') if isinstance(mod, dict) else '',
                        'duration_ms': mod.get('duration_ms', 50) if isinstance(mod, dict) else 50,
                        'quality_score': mod.get('quality_score', 0.8) if isinstance(mod, dict) else 0.8
                    })
            reasoning_steps = ReasoningPathBuilder.build_path(
                inferred_modules or [{'module_name': 'pathrag_core'}],
                initial_confidence=0.3,
                final_confidence=final_confidence
            )
        
        result['visualizations']['reasoning_path'] = {
            'steps': [s.to_dict() for s in reasoning_steps],
            'total_steps': len(reasoning_steps),
            'total_duration_ms': sum(s.duration_ms for s in reasoning_steps)
        }
    
    # 3. 置信度分解
    if VisualizationType.CONFIDENCE_BREAKDOWN.value in include_visualizations:
        module_contribs = []
        if module_executions:
            for me in module_executions:
                module_contribs.append(ModuleContribution(
                    module_name=me.get('module_name', 'unknown'),
                    version=me.get('version', 'v1'),
                    execution_time_ms=me.get('duration_ms', 0),
                    input_size=me.get('input_size', 0),
                    output_size=me.get('output_size', 0),
                    confidence_contribution=me.get('conf_delta', 0.1),
                    quality_score=me.get('quality_score', 0.8)
                ))
        
        evidence_scores = [c.get('score', c.get('relevance_score', 0.7)) 
                         for c in evidence_chunks]
        
        decomposition = ConfidenceDecomposer.decompose(
            final_confidence, module_contribs, evidence_scores
        )
        result['visualizations']['confidence_breakdown'] = decomposition
    
    # 4. 模块贡献
    if VisualizationType.MODULE_CONTRIBUTION.value in include_visualizations:
        contributions = []
        if module_executions:
            for me in module_executions:
                contributions.append(ModuleContribution(
                    module_name=me.get('module_name', 'unknown'),
                    version=me.get('version', 'v1'),
                    execution_time_ms=me.get('duration_ms', 0),
                    input_size=me.get('input_size', 0),
                    output_size=me.get('output_size', 0),
                    confidence_contribution=me.get('conf_delta', 0),
                    quality_score=me.get('quality_score', 0.8),
                    errors=me.get('errors', [])
                ).to_dict())
        else:
            # 默认显示 PathRAG 核心模块
            default_modules = [
                ('VSR_PathRAG', 'v2.0', 120, 10, 5, 0.35, 0.9),
                ('Evidence_Retrieval', 'v1.5', 80, 20, 15, 0.25, 0.85),
                ('Answer_Generation', 'v1.3', 60, 5, 1, 0.25, 0.88),
            ]
            for name, ver, dur, inp, outp, conf, qual in default_modules:
                contributions.append(ModuleContribution(
                    module_name=name, version=ver, execution_time_ms=dur,
                    input_size=inp, output_size=outp,
                    confidence_contribution=conf, quality_score=qual
                ).to_dict())
        
        result['visualizations']['module_contribution'] = {
            'contributions': contributions,
            'total_modules': len(contributions),
            'total_execution_time_ms': sum(c['execution_time_ms'] for c in contributions),
            'avg_quality': round(sum(c['quality_score'] for c in contributions) / max(len(contributions), 1), 3)
        }
    
    # 5. 生成摘要
    result['summary'] = _generate_dashboard_summary(result)
    
    # 总处理时间
    result['dashboard_generation_time_ms'] = (time.time() - start_time) * 1000
    
    return result


def _generate_dashboard_summary(dashboard_data: Dict[str, Any]) -> Dict[str, Any]:
    """生成仪表盘摘要"""
    viz = dashboard_data.get('visualizations', {})
    
    summary = {
        'evidence_count': 0,
        'reasoning_steps': 0,
        'modules_involved': 0,
        'overall_health': 'good',
        'key_insights': [],
        'actionable_items': []
    }
    
    # 证据统计
    if 'evidence_tree' in viz:
        et = viz['evidence_tree']
        summary['evidence_count'] = et.get('tree_info', {}).get('total_nodes', 0) - 1  # 减去根节点
        avg_rel = et.get('tree_info', {}).get('avg_relevance', 0)
        if avg_rel < 0.5:
            summary['key_insights'].append("Evidence relevance is below average")
            summary['actionable_items'].append("Consider improving retrieval queries")
    
    # 推理统计
    if 'reasoning_path' in viz:
        rp = viz['reasoning_path']
        summary['reasoning_steps'] = rp.get('total_steps', 0)
        total_dur = rp.get('total_duration_ms', 0)
        if total_dur > 5000:
            summary['key_insights'].append(f"Processing took {total_dur/1000:.1f}s - consider optimization")
    
    # 模块统计
    if 'module_contribution' in viz:
        mc = viz['module_contribution']
        summary['modules_involved'] = mc.get('total_modules', 0)
        avg_qual = mc.get('avg_quality', 0)
        if avg_qual < 0.7:
            summary['overall_health'] = 'warning'
            summary['actionable_items'].append("Some modules have low quality scores")
    
    # 置信度分析
    if 'confidence_breakdown' in viz:
        cb = viz['confidence_breakdown']
        if cb.get('weaknesses'):
            summary['overall_health'] = 'warning'
            summary['key_insights'].extend(cb['weaknesses'][:2])
    
    return summary


# 导出别名
run_explainability_dashboard = build_explainability_dashboard_data

__all__ = [
    'VisualizationType',
    'EvidenceNode',
    'ReasoningStep',
    'ModuleContribution',
    'ExplanationReport',
    'EvidenceTreeBuilder',
    'ReasoningPathBuilder',
    'ConfidenceDecomposer',
    'build_explainability_dashboard_data',
    'run_explainability_dashboard',
]
