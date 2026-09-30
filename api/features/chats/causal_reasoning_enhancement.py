"""
V110: Causal Reasoning Enhancement (CaRE)
因果推理增强 - 识别因果关系，进行反事实推理

创新点：
1. 因果发现：从文本中自动识别因果关系
2. 因果图构建：构建因果图模型
3. 反事实推理：回答"如果...会怎样"的问题
4. 干预分析：评估干预措施的效果
"""

import re
import json
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict
import networkx as nx
from enum import Enum


class CausalRelationType(Enum):
    """因果关系类型"""
    CAUSES = "causes"                    # A 导致 B
    PREVENTS = "prevents"                # A 阻止 B
    ENABLES = "enables"                  # A 使能 B
    INHIBITS = "inhibits"                # A 抑制 B
    CORRELATES = "correlates_with"       # A 与 B 相关（非因果）


@dataclass
class CausalRelation:
    """因果关系数据结构"""
    cause: str
    effect: str
    relation_type: CausalRelationType
    confidence: float
    evidence: List[str] = field(default_factory=list)
    context: str = ""
    temporal_order: Optional[str] = None  # 时间顺序信息
    strength: str = "moderate"            # strong, moderate, weak


@dataclass
class CausalGraph:
    """因果图数据结构"""
    nodes: Set[str] = field(default_factory=set)
    edges: List[CausalRelation] = field(default_factory=list)
    graph: nx.DiGraph = field(default_factory=nx.DiGraph)
    
    def add_relation(self, relation: CausalRelation):
        """添加因果关系到图中"""
        self.nodes.add(relation.cause)
        self.nodes.add(relation.effect)
        self.edges.append(relation)
        
        # 添加到 networkx 图
        self.graph.add_edge(
            relation.cause, 
            relation.effect,
            relation_type=relation.relation_type.value,
            confidence=relation.confidence,
            strength=relation.strength
        )
    
    def find_causal_paths(
        self, 
        source: str, 
        target: str,
        max_length: int = 5
    ) -> List[List[str]]:
        """找到从 source 到 target 的所有因果路径"""
        try:
            paths = list(nx.all_simple_paths(
                self.graph, 
                source, 
                target, 
                cutoff=max_length
            ))
            return paths
        except nx.NetworkXNoPath:
            return []
    
    def find_common_causes(self, effects: List[str]) -> List[str]:
        """找到多个效果的共同原因"""
        if len(effects) < 2:
            return []
        
        # 找到所有可能的原因
        common_causes = set(self.graph.predecessors(effects[0]))
        for effect in effects[1:]:
            common_causes &= set(self.graph.predecessors(effect))
        
        return list(common_causes)
    
    def calculate_causal_effect(
        self, 
        intervention: str, 
        outcome: str
    ) -> Dict[str, Any]:
        """计算干预效果"""
        # 简化的因果效应计算
        try:
            # 检查是否有直接路径
            if nx.has_path(self.graph, intervention, outcome):
                paths = self.find_causal_paths(intervention, outcome)
                
                # 计算路径强度
                total_effect = 0
                for path in paths:
                    path_strength = 1.0
                    for i in range(len(path) - 1):
                        edge_data = self.graph.get_edge_data(path[i], path[i+1])
                        if edge_data:
                            confidence = edge_data.get('confidence', 0.5)
                            path_strength *= confidence
                    total_effect += path_strength
                
                return {
                    'has_effect': True,
                    'effect_size': min(total_effect, 1.0),
                    'path_count': len(paths),
                    'paths': paths,
                    'confidence': self._calculate_path_confidence(paths)
                }
            else:
                return {
                    'has_effect': False,
                    'effect_size': 0,
                    'path_count': 0,
                    'paths': [],
                    'confidence': 0
                }
        except Exception as e:
            return {
                'has_effect': False,
                'error': str(e),
                'effect_size': 0,
                'confidence': 0
            }
    
    def _calculate_path_confidence(self, paths: List[List[str]]) -> float:
        """计算路径的整体置信度"""
        if not paths:
            return 0
        
        confidences = []
        for path in paths:
            path_confidence = 1.0
            for i in range(len(path) - 1):
                edge_data = self.graph.get_edge_data(path[i], path[i+1])
                if edge_data:
                    path_confidence *= edge_data.get('confidence', 0.5)
            confidences.append(path_confidence)
        
        return max(confidences) if confidences else 0


class CausalDiscoveryEngine:
    """因果发现引擎"""
    
    # 因果指示词
    CAUSAL_INDICATORS = {
        CausalRelationType.CAUSES: [
            'causes', 'leads to', 'results in', 'produces', 'generates',
            'induces', 'triggers', 'brings about', 'gives rise to',
            '导致', '引起', '产生', '造成', '引发'
        ],
        CausalRelationType.PREVENTS: [
            'prevents', 'stops', 'blocks', 'inhibits', 'hinders',
            'precludes', 'avoids', 'prevents', '阻碍', '阻止', '防止'
        ],
        CausalRelationType.ENABLES: [
            'enables', 'allows', 'permits', 'facilitates', 'helps',
            'supports', 'promotes', '使能', '促进', '有助于'
        ],
        CausalRelationType.INHIBITS: [
            'inhibits', 'reduces', 'decreases', 'suppresses', 'dampens',
            'limits', 'restricts', '抑制', '减少', '降低'
        ]
    }
    
    def __init__(self):
        self.causal_graph = CausalGraph()
    
    def discover_causal_relations(
        self, 
        text: str,
        min_confidence: float = 0.5
    ) -> List[CausalRelation]:
        """
        从文本中发现因果关系
        
        Args:
            text: 输入文本
            min_confidence: 最小置信度阈值
        
        Returns:
            发现的因果关系列表
        """
        relations = []
        sentences = self._split_sentences(text)
        
        for sentence in sentences:
            sentence_relations = self._extract_relations_from_sentence(sentence)
            relations.extend(sentence_relations)
        
        # 过滤低置信度的关系
        relations = [r for r in relations if r.confidence >= min_confidence]
        
        # 添加到因果图
        for relation in relations:
            self.causal_graph.add_relation(relation)
        
        return relations
    
    def _split_sentences(self, text: str) -> List[str]:
        """将文本分割成句子"""
        # 简单的句子分割
        sentences = re.split(r'[.!?。！？]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def _extract_relations_from_sentence(
        self, 
        sentence: str
    ) -> List[CausalRelation]:
        """从单个句子中提取因果关系"""
        relations = []
        sentence_lower = sentence.lower()
        
        for relation_type, indicators in self.CAUSAL_INDICATORS.items():
            for indicator in indicators:
                if indicator.lower() in sentence_lower:
                    # 尝试提取 cause 和 effect
                    cause, effect = self._extract_cause_effect(
                        sentence, 
                        indicator
                    )
                    
                    if cause and effect:
                        confidence = self._calculate_relation_confidence(
                            sentence, 
                            indicator, 
                            cause, 
                            effect
                        )
                        
                        relation = CausalRelation(
                            cause=cause,
                            effect=effect,
                            relation_type=relation_type,
                            confidence=confidence,
                            evidence=[sentence],
                            context=sentence,
                            strength=self._determine_strength(confidence)
                        )
                        relations.append(relation)
        
        return relations
    
    def _extract_cause_effect(
        self, 
        sentence: str, 
        indicator: str
    ) -> Tuple[Optional[str], Optional[str]]:
        """提取原因和结果"""
        # 简化的提取逻辑
        parts = sentence.lower().split(indicator.lower())
        
        if len(parts) == 2:
            # 假设原因在指示词前，结果在指示词后
            cause = parts[0].strip()
            effect = parts[1].strip()
            
            # 清理提取的文本
            cause = self._clean_entity(cause)
            effect = self._clean_entity(effect)
            
            return cause, effect
        
        return None, None
    
    def _clean_entity(self, text: str) -> str:
        """清理实体文本"""
        # 移除常见的停用词和标点
        stop_words = {'the', 'a', 'an', 'this', 'that', 'these', 'those'}
        words = text.split()
        words = [w for w in words if w.lower() not in stop_words]
        return ' '.join(words[:5])  # 限制长度
    
    def _calculate_relation_confidence(
        self, 
        sentence: str, 
        indicator: str,
        cause: str, 
        effect: str
    ) -> float:
        """计算因果关系的置信度"""
        confidence = 0.5  # 基础置信度
        
        # 指示词的明确性
        if indicator in ['causes', 'leads to', 'results in']:
            confidence += 0.2
        
        # 实体长度（太短可能不准确）
        if len(cause) > 3 and len(effect) > 3:
            confidence += 0.1
        
        # 句子长度（适中长度通常更可靠）
        words = len(sentence.split())
        if 5 <= words <= 20:
            confidence += 0.1
        
        # 是否有时间顺序词
        temporal_words = ['before', 'after', 'then', 'subsequently', '首先', '然后', '之后']
        if any(word in sentence.lower() for word in temporal_words):
            confidence += 0.1
        
        return min(confidence, 1.0)
    
    def _determine_strength(self, confidence: float) -> str:
        """确定因果关系的强度"""
        if confidence >= 0.8:
            return "strong"
        elif confidence >= 0.6:
            return "moderate"
        else:
            return "weak"


class CounterfactualReasoner:
    """反事实推理器"""
    
    def __init__(self, causal_graph: CausalGraph):
        self.causal_graph = causal_graph
    
    def reason_counterfactual(
        self,
        premise: str,
        hypothetical: str
    ) -> Dict[str, Any]:
        """
        进行反事实推理
        
        Args:
            premise: 事实前提
            hypothetical: 假设条件（通常以"如果"开头）
        
        Returns:
            反事实推理结果
        """
        # 解析假设
        intervention = self._parse_intervention(hypothetical)
        
        if not intervention:
            return {
                'valid': False,
                'reason': 'Could not parse intervention from hypothetical',
                'conclusion': None
            }
        
        # 找到可能的影响
        affected_nodes = self._find_affected_nodes(intervention)
        
        # 生成反事实结论
        conclusions = []
        for node in affected_nodes:
            effect = self.causal_graph.calculate_causal_effect(intervention, node)
            if effect['has_effect']:
                conclusions.append({
                    'outcome': node,
                    'likelihood': effect['effect_size'],
                    'confidence': effect['confidence'],
                    'paths': effect['paths']
                })
        
        # 排序结论
        conclusions.sort(key=lambda x: x['likelihood'], reverse=True)
        
        return {
            'valid': True,
            'intervention': intervention,
            'affected_outcomes': len(conclusions),
            'conclusions': conclusions[:5],  # 返回前5个最可能的结果
            'reasoning_chain': self._build_reasoning_chain(intervention, conclusions)
        }
    
    def _parse_intervention(self, hypothetical: str) -> Optional[str]:
        """从假设条件中解析干预"""
        # 移除常见的假设词
        prefixes = ['if', '假如', '如果', '假设', 'suppose', 'assuming']
        
        text = hypothetical.lower()
        for prefix in prefixes:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
        
        # 移除标点
        text = text.strip('.,!?。！？')
        
        return text if text else None
    
    def _find_affected_nodes(self, intervention: str) -> List[str]:
        """找到可能受干预影响的节点"""
        affected = []
        
        # 直接匹配
        for node in self.causal_graph.nodes:
            if intervention.lower() in node.lower() or node.lower() in intervention.lower():
                # 找到从该节点可达的所有节点
                descendants = nx.descendants(self.causal_graph.graph, node)
                affected.extend(descendants)
        
        return list(set(affected))
    
    def _build_reasoning_chain(
        self, 
        intervention: str, 
        conclusions: List[Dict]
    ) -> str:
        """构建推理链说明"""
        if not conclusions:
            return f"No clear causal path found from '{intervention}' to any outcomes."
        
        chains = []
        for i, conclusion in enumerate(conclusions[:3], 1):
            chain = f"{i}. If '{intervention}' occurs, then '{conclusion['outcome']}' "
            chain += f"would likely occur (likelihood: {conclusion['likelihood']:.2f})"
            chains.append(chain)
        
        return "\n".join(chains)


class InterventionAnalyzer:
    """干预分析器"""
    
    def __init__(self, causal_graph: CausalGraph):
        self.causal_graph = causal_graph
    
    def analyze_intervention(
        self,
        intervention: str,
        target_outcome: str,
        alternative_interventions: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        分析干预措施的效果
        
        Args:
            intervention: 拟议的干预
            target_outcome: 目标结果
            alternative_interventions: 替代干预方案
        
        Returns:
            干预分析结果
        """
        # 计算主要干预的效果
        main_effect = self.causal_graph.calculate_causal_effect(
            intervention, 
            target_outcome
        )
        
        # 分析替代方案
        alternatives_analysis = []
        if alternative_interventions:
            for alt in alternative_interventions:
                alt_effect = self.causal_graph.calculate_causal_effect(
                    alt, 
                    target_outcome
                )
                alternatives_analysis.append({
                    'intervention': alt,
                    'effect': alt_effect,
                    'comparison': self._compare_interventions(
                        main_effect, 
                        alt_effect
                    )
                })
        
        # 排序替代方案
        alternatives_analysis.sort(
            key=lambda x: x['effect'].get('effect_size', 0), 
            reverse=True
        )
        
        return {
            'target_outcome': target_outcome,
            'proposed_intervention': {
                'name': intervention,
                'effect': main_effect
            },
            'alternatives': alternatives_analysis,
            'recommendation': self._generate_recommendation(
                intervention,
                main_effect,
                alternatives_analysis
            ),
            'side_effects': self._identify_side_effects(intervention)
        }
    
    def _compare_interventions(
        self, 
        main: Dict, 
        alternative: Dict
    ) -> str:
        """比较两个干预方案"""
        main_size = main.get('effect_size', 0)
        alt_size = alternative.get('effect_size', 0)
        
        if alt_size > main_size * 1.2:
            return "significantly_better"
        elif alt_size > main_size:
            return "better"
        elif alt_size < main_size * 0.8:
            return "significantly_worse"
        elif alt_size < main_size:
            return "worse"
        else:
            return "comparable"
    
    def _generate_recommendation(
        self,
        proposed: str,
        proposed_effect: Dict,
        alternatives: List[Dict]
    ) -> str:
        """生成干预建议"""
        proposed_size = proposed_effect.get('effect_size', 0)
        
        if proposed_size > 0.7:
            return f"'{proposed}' is highly effective and recommended."
        elif proposed_size > 0.4:
            if alternatives and alternatives[0]['comparison'] in ['better', 'significantly_better']:
                better_alt = alternatives[0]['intervention']
                return f"'{proposed}' is moderately effective, but consider '{better_alt}' as a potentially better alternative."
            return f"'{proposed}' is moderately effective and worth considering."
        else:
            if alternatives:
                better_alt = alternatives[0]['intervention']
                return f"'{proposed}' has limited effectiveness. Consider '{better_alt}' instead."
            return f"'{proposed}' has limited effectiveness. Consider alternative approaches."
    
    def _identify_side_effects(self, intervention: str) -> List[Dict]:
        """识别干预的副作用"""
        side_effects = []
        
        # 找到干预可能影响的所有结果
        for node in self.causal_graph.nodes:
            if node != intervention:
                effect = self.causal_graph.calculate_causal_effect(
                    intervention, 
                    node
                )
                if effect['has_effect'] and effect['effect_size'] > 0.3:
                    side_effects.append({
                        'outcome': node,
                        'likelihood': effect['effect_size'],
                        'type': 'intended' if effect['effect_size'] > 0.7 else 'unintended'
                    })
        
        # 按可能性排序
        side_effects.sort(key=lambda x: x['likelihood'], reverse=True)
        return side_effects[:5]  # 返回前5个


# 主引擎类
class CausalReasoningEngine:
    """因果推理引擎"""
    
    def __init__(self):
        self.discovery_engine = CausalDiscoveryEngine()
        self.counterfactual_reasoner = None
        self.intervention_analyzer = None
    
    def build_causal_model(
        self, 
        evidence: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        从证据构建因果模型
        
        Args:
            evidence: 证据列表
        
        Returns:
            因果模型构建结果
        """
        all_relations = []
        
        for ev in evidence:
            content = ev.get('content', '')
            relations = self.discovery_engine.discover_causal_relations(content)
            all_relations.extend(relations)
        
        # 初始化推理器
        causal_graph = self.discovery_engine.causal_graph
        self.counterfactual_reasoner = CounterfactualReasoner(causal_graph)
        self.intervention_analyzer = InterventionAnalyzer(causal_graph)
        
        return {
            'version': 'V110',
            'name': 'Causal Reasoning Enhancement (CaRE)',
            'relations_discovered': len(all_relations),
            'nodes': len(causal_graph.nodes),
            'edges': len(causal_graph.edges),
            'causal_graph': {
                'nodes': list(causal_graph.nodes),
                'edges': [
                    {
                        'cause': r.cause,
                        'effect': r.effect,
                        'type': r.relation_type.value,
                        'confidence': r.confidence,
                        'strength': r.strength
                    }
                    for r in all_relations
                ]
            }
        }
    
    def answer_causal_question(
        self,
        question: str,
        question_type: str = 'why'
    ) -> Dict[str, Any]:
        """
        回答因果问题
        
        Args:
            question: 问题
            question_type: 问题类型 (why, what_if, how_to)
        
        Returns:
            因果推理结果
        """
        if question_type == 'why':
            return self._answer_why_question(question)
        elif question_type == 'what_if':
            return self._answer_what_if_question(question)
        elif question_type == 'how_to':
            return self._answer_how_to_question(question)
        else:
            return {
                'error': f'Unsupported question type: {question_type}'
            }
    
    def _answer_why_question(self, question: str) -> Dict[str, Any]:
        """回答"为什么"问题"""
        # 提取结果
        effect = self._extract_effect_from_question(question)
        
        if not effect:
            return {'error': 'Could not identify effect in question'}
        
        # 找到可能的原因
        causes = self.discovery_engine.causal_graph.find_common_causes([effect])
        
        # 构建因果链
        causal_chains = []
        for cause in causes:
            paths = self.discovery_engine.causal_graph.find_causal_paths(
                cause, 
                effect
            )
            for path in paths:
                causal_chains.append({
                    'cause': cause,
                    'path': path,
                    'explanation': self._build_explanation(path)
                })
        
        return {
            'question_type': 'why',
            'effect': effect,
            'possible_causes': causes,
            'causal_chains': causal_chains,
            'answer': self._synthesize_why_answer(effect, causal_chains)
        }
    
    def _answer_what_if_question(self, question: str) -> Dict[str, Any]:
        """回答"如果...会怎样"问题"""
        if not self.counterfactual_reasoner:
            return {'error': 'Causal model not initialized'}
        
        # 解析假设
        result = self.counterfactual_reasoner.reason_counterfactual(
            "current situation",
            question
        )
        
        return {
            'question_type': 'what_if',
            'result': result,
            'answer': result.get('reasoning_chain', 'No clear causal path found.')
        }
    
    def _answer_how_to_question(self, question: str) -> Dict[str, Any]:
        """回答"如何实现"问题"""
        # 提取目标
        target = self._extract_target_from_question(question)
        
        if not target:
            return {'error': 'Could not identify target in question'}
        
        if not self.intervention_analyzer:
            return {'error': 'Causal model not initialized'}
        
        # 找到可能的干预措施
        # 简化的实现：找到所有可能导致目标的节点
        possible_interventions = []
        for node in self.discovery_engine.causal_graph.nodes:
            if node != target:
                effect = self.discovery_engine.causal_graph.calculate_causal_effect(
                    node, 
                    target
                )
                if effect['has_effect'] and effect['effect_size'] > 0.3:
                    possible_interventions.append({
                        'intervention': node,
                        'effectiveness': effect['effect_size'],
                        'confidence': effect['confidence'],
                        'paths': effect['paths']
                    })
        
        # 排序
        possible_interventions.sort(
            key=lambda x: x['effectiveness'], 
            reverse=True
        )
        
        return {
            'question_type': 'how_to',
            'target': target,
            'recommended_interventions': possible_interventions[:5],
            'answer': self._synthesize_how_to_answer(target, possible_interventions[:3])
        }
    
    def _extract_effect_from_question(self, question: str) -> Optional[str]:
        """从问题中提取效果"""
        # 简化的提取
        patterns = [
            r'why (?:does|did|is|was) (.+?)\?',
            r'why (.+?) happen',
            r'什么原因导致(.+?)',
            r'为什么(.+?)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, question, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _extract_target_from_question(self, question: str) -> Optional[str]:
        """从问题中提取目标"""
        patterns = [
            r'how (?:can|to) (?:we|I|you) (.+?)\?',
            r'how (?:do|does) (?:one|we) (.+?)\?',
            r'如何实现(.+?)',
            r'怎样才能(.+?)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, question, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        return None
    
    def _build_explanation(self, path: List[str]) -> str:
        """构建因果解释"""
        if len(path) < 2:
            return ""
        
        explanations = []
        for i in range(len(path) - 1):
            explanations.append(f"{path[i]} leads to {path[i+1]}")
        
        return " → ".join(explanations)
    
    def _synthesize_why_answer(
        self, 
        effect: str, 
        chains: List[Dict]
    ) -> str:
        """综合"为什么"答案"""
        if not chains:
            return f"The causes of '{effect}' are not clearly established in the available evidence."
        
        answer = f"'{effect}' can be explained by the following causal mechanisms:\n\n"
        
        for i, chain in enumerate(chains[:3], 1):
            answer += f"{i}. {chain['explanation']}\n"
        
        return answer
    
    def _synthesize_how_to_answer(
        self, 
        target: str, 
        interventions: List[Dict]
    ) -> str:
        """综合"如何实现"答案"""
        if not interventions:
            return f"No clear intervention paths found to achieve '{target}'."
        
        answer = f"To achieve '{target}', consider the following approaches:\n\n"
        
        for i, intervention in enumerate(interventions, 1):
            effectiveness = intervention['effectiveness']
            confidence = intervention['confidence']
            answer += f"{i}. {intervention['intervention']} "
            answer += f"(effectiveness: {effectiveness:.2f}, confidence: {confidence:.2f})\n"
        
        return answer


# 便捷函数
def build_causal_reasoning_enhancement(
    evidence: List[Dict[str, Any]],
    question: Optional[str] = None,
    question_type: str = 'why'
) -> Dict[str, Any]:
    """
    构建因果推理增强
    
    这是V110创新的主要入口函数
    """
    engine = CausalReasoningEngine()
    
    # 构建因果模型
    model_result = engine.build_causal_model(evidence)
    
    result = {
        'version': 'V110',
        'name': 'Causal Reasoning Enhancement (CaRE)',
        'causal_model': model_result
    }
    
    # 如果提供了问题，进行推理
    if question:
        reasoning_result = engine.answer_causal_question(question, question_type)
        result['reasoning'] = reasoning_result
    
    return result


# 示例用法
if __name__ == "__main__":
    evidence = [
        {
            'content': 'Smoking causes lung cancer. Heavy smoking increases the risk significantly.',
            'source': 'medical_study'
        },
        {
            'content': 'Lung cancer leads to respiratory failure. Respiratory failure can cause death.',
            'source': 'medical_textbook'
        },
        {
            'content': 'Quitting smoking reduces the risk of lung cancer over time.',
            'source': 'health_report'
        }
    ]
    
    # 构建因果模型
    result = build_causal_reasoning_enhancement(evidence)
    print("Causal Model:")
    print(json.dumps(result['causal_model'], indent=2))
    
    # 回答因果问题
    engine = CausalReasoningEngine()
    engine.build_causal_model(evidence)
    
    why_result = engine.answer_causal_question(
        "Why does smoking cause health problems?",
        "why"
    )
    print("\nWhy Question:")
    print(json.dumps(why_result, indent=2))
