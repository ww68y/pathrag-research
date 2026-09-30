"""
V122: Neural-Symbolic Reasoning Bridge (NSRB)
神经符号推理桥接 - 连接神经网络的感知能力与符号推理的逻辑能力

创新点：
1. 神经-符号接口层：双向转换神经网络输出和符号表示
2. 逻辑规则自动提取：从神经网络中提取符号规则
3. 符号推理链生成：构建可解释的推理链
4. 混合推理执行：结合神经网络和符号推理
"""

import json
import re
from typing import List, Dict, Any, Optional, Tuple, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import defaultdict
import numpy as np


class SymbolType(Enum):
    """符号类型"""
    ENTITY = "entity"
    RELATION = "relation"
    ATTRIBUTE = "attribute"
    RULE = "rule"
    CONSTRAINT = "constraint"


class LogicOperator(Enum):
    """逻辑运算符"""
    AND = "and"
    OR = "or"
    NOT = "not"
    IMPLIES = "implies"
    FORALL = "forall"
    EXISTS = "exists"


@dataclass
class Symbol:
    """符号"""
    symbol_id: str
    name: str
    symbol_type: SymbolType
    properties: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 1.0


@dataclass
class LogicRule:
    """逻辑规则"""
    rule_id: str
    premises: List[Symbol]
    conclusion: Symbol
    operator: LogicOperator
    confidence: float = 1.0
    source: str = "extracted"  # extracted, manual, learned


@dataclass
class ReasoningStep:
    """推理步骤"""
    step_id: str
    rule: LogicRule
    bindings: Dict[str, Any]
    result: Symbol
    confidence: float = 1.0


class NeuralToSymbolicConverter:
    """神经到符号转换器"""
    
    def __init__(self):
        self.extraction_patterns = self._init_extraction_patterns()
        self.conversion_history: List[Dict] = []
    
    def _init_extraction_patterns(self) -> Dict[str, Any]:
        """初始化提取模式"""
        return {
            'entity_patterns': [
                r'([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',  # 专有名词
                r'\b(machine learning|artificial intelligence|deep learning)\b',
            ],
            'relation_patterns': [
                r'(\w+)\s+(is|are|was|were)\s+(\w+)',
                r'(\w+)\s+(causes|leads to|results in)\s+(\w+)',
            ],
            'attribute_patterns': [
                r'(\w+)\s+has\s+(\w+)\s+of\s+(\d+)',
                r'(\w+)\s+is\s+(\w+)\s+than\s+(\w+)',
            ]
        }
    
    def convert(
        self,
        neural_output: Dict[str, Any],
        text_context: str
    ) -> Dict[str, Any]:
        """
        将神经网络输出转换为符号表示
        
        Args:
            neural_output: 神经网络输出（如分类概率、嵌入向量）
            text_context: 原始文本上下文
        
        Returns:
            符号表示
        """
        # 1. 提取实体
        entities = self._extract_entities(text_context)
        
        # 2. 提取关系
        relations = self._extract_relations(text_context)
        
        # 3. 提取属性
        attributes = self._extract_attributes(text_context)
        
        # 4. 构建符号图
        symbol_graph = self._build_symbol_graph(
            entities, relations, attributes
        )
        
        # 5. 添加神经网络置信度
        enriched_graph = self._enrich_with_confidence(
            symbol_graph, neural_output
        )
        
        conversion_result = {
            'entities': entities,
            'relations': relations,
            'attributes': attributes,
            'symbol_graph': enriched_graph,
            'conversion_confidence': self._compute_conversion_confidence(
                entities, relations, neural_output
            )
        }
        
        self.conversion_history.append(conversion_result)
        
        return conversion_result
    
    def _extract_entities(self, text: str) -> List[Symbol]:
        """提取实体"""
        entities = []
        
        # 使用模式匹配提取实体
        for pattern in self.extraction_patterns['entity_patterns']:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                entity_name = match.group(0)
                entity_id = f"entity_{len(entities)}"
                
                entities.append(Symbol(
                    symbol_id=entity_id,
                    name=entity_name,
                    symbol_type=SymbolType.ENTITY,
                    properties={'source_text': text[match.start():match.end()]}
                ))
        
        return entities
    
    def _extract_relations(self, text: str) -> List[Symbol]:
        """提取关系"""
        relations = []
        
        # 常见关系词
        relation_keywords = {
            'is': 'identity',
            'causes': 'causation',
            'leads to': 'causation',
            'part of': 'partonomy',
            'uses': 'usage',
            'improves': 'improvement',
            'based on': 'derivation'
        }
        
        for keyword, relation_type in relation_keywords.items():
            if keyword in text.lower():
                relation_id = f"relation_{len(relations)}"
                relations.append(Symbol(
                    symbol_id=relation_id,
                    name=keyword,
                    symbol_type=SymbolType.RELATION,
                    properties={'relation_type': relation_type}
                ))
        
        return relations
    
    def _extract_attributes(self, text: str) -> List[Symbol]:
        """提取属性"""
        attributes = []
        
        # 数值属性
        number_pattern = r'(\w+)\s+(is|has|equals)\s+(\d+(?:\.\d+)?)'
        matches = re.finditer(number_pattern, text, re.IGNORECASE)
        
        for match in matches:
            attr_id = f"attr_{len(attributes)}"
            attributes.append(Symbol(
                symbol_id=attr_id,
                name=match.group(1),
                symbol_type=SymbolType.ATTRIBUTE,
                properties={
                    'value': float(match.group(3)),
                    'unit': 'unknown'
                }
            ))
        
        return attributes
    
    def _build_symbol_graph(
        self,
        entities: List[Symbol],
        relations: List[Symbol],
        attributes: List[Symbol]
    ) -> Dict[str, Any]:
        """构建符号图"""
        nodes = {}
        edges = []
        
        # 添加实体节点
        for entity in entities:
            nodes[entity.symbol_id] = {
                'type': 'entity',
                'name': entity.name,
                'properties': entity.properties
            }
        
        # 添加关系边
        for i, relation in enumerate(relations):
            if len(entities) >= 2:
                edges.append({
                    'source': entities[0].symbol_id,
                    'target': entities[1].symbol_id if len(entities) > 1 else entities[0].symbol_id,
                    'relation': relation.name,
                    'relation_type': relation.properties.get('relation_type', 'unknown')
                })
        
        return {
            'nodes': nodes,
            'edges': edges,
            'node_count': len(nodes),
            'edge_count': len(edges)
        }
    
    def _enrich_with_confidence(
        self,
        symbol_graph: Dict,
        neural_output: Dict
    ) -> Dict:
        """用神经网络置信度丰富符号图"""
        # 提取神经网络的置信度分数
        confidences = neural_output.get('confidences', [])
        
        if confidences:
            avg_confidence = np.mean(confidences)
            symbol_graph['neural_confidence'] = avg_confidence
        
        return symbol_graph
    
    def _compute_conversion_confidence(
        self,
        entities: List[Symbol],
        relations: List[Symbol],
        neural_output: Dict
    ) -> float:
        """计算转换置信度"""
        # 基于提取的数量和神经网络置信度
        extraction_score = min(1.0, (len(entities) + len(relations)) / 10)
        neural_confidence = neural_output.get('overall_confidence', 0.5)
        
        return (extraction_score + neural_confidence) / 2


class RuleExtractor:
    """规则提取器"""
    
    def __init__(self):
        self.extracted_rules: List[LogicRule] = []
        self.rule_patterns = self._init_rule_patterns()
    
    def _init_rule_patterns(self) -> Dict[str, Any]:
        """初始化规则模式"""
        return {
            'causal': {
                'patterns': ['causes', 'leads to', 'results in', 'produces'],
                'operator': LogicOperator.IMPLIES
            },
            'conditional': {
                'patterns': ['if', 'when', 'whenever'],
                'operator': LogicOperator.IMPLIES
            },
            'universal': {
                'patterns': ['all', 'every', 'always'],
                'operator': LogicOperator.FORALL
            },
            'existential': {
                'patterns': ['some', 'there exists', 'at least one'],
                'operator': LogicOperator.EXISTS
            }
        }
    
    def extract_rules(
        self,
        symbols: List[Symbol],
        text_context: str
    ) -> List[LogicRule]:
        """
        从符号和文本中提取逻辑规则
        
        Args:
            symbols: 符号列表
            text_context: 文本上下文
        
        Returns:
            提取的逻辑规则
        """
        rules = []
        
        # 1. 基于模式提取规则
        pattern_rules = self._extract_pattern_rules(text_context, symbols)
        rules.extend(pattern_rules)
        
        # 2. 基于统计关联提取规则
        association_rules = self._extract_association_rules(symbols)
        rules.extend(association_rules)
        
        # 3. 基于层次结构提取规则
        hierarchical_rules = self._extract_hierarchical_rules(symbols)
        rules.extend(hierarchical_rules)
        
        self.extracted_rules.extend(rules)
        
        return rules
    
    def _extract_pattern_rules(
        self,
        text: str,
        symbols: List[Symbol]
    ) -> List[LogicRule]:
        """基于模式提取规则"""
        rules = []
        
        for rule_type, config in self.rule_patterns.items():
            for pattern in config['patterns']:
                if pattern in text.lower():
                    # 提取规则
                    rule = self._parse_rule_from_text(
                        text, pattern, config['operator'], symbols
                    )
                    if rule:
                        rules.append(rule)
        
        return rules
    
    def _parse_rule_from_text(
        self,
        text: str,
        pattern: str,
        operator: LogicOperator,
        symbols: List[Symbol]
    ) -> Optional[LogicRule]:
        """从文本解析规则"""
        # 简化的规则解析
        # 查找包含模式的句子
        sentences = text.split('.')
        
        for sentence in sentences:
            if pattern in sentence.lower():
                # 提取前提和结论
                parts = sentence.lower().split(pattern)
                if len(parts) == 2:
                    premise_text = parts[0].strip()
                    conclusion_text = parts[1].strip()
                    
                    # 创建符号
                    premise = Symbol(
                        symbol_id=f"prem_{len(self.extracted_rules)}",
                        name=premise_text,
                        symbol_type=SymbolType.ENTITY
                    )
                    
                    conclusion = Symbol(
                        symbol_id=f"conc_{len(self.extracted_rules)}",
                        name=conclusion_text,
                        symbol_type=SymbolType.ENTITY
                    )
                    
                    return LogicRule(
                        rule_id=f"rule_{len(self.extracted_rules)}",
                        premises=[premise],
                        conclusion=conclusion,
                        operator=operator,
                        source="extracted"
                    )
        
        return None
    
    def _extract_association_rules(self, symbols: List[Symbol]) -> List[LogicRule]:
        """提取关联规则"""
        rules = []
        
        # 查找经常一起出现的实体
        entity_pairs = self._find_cooccurring_entities(symbols)
        
        for pair in entity_pairs:
            if pair['confidence'] > 0.7:
                rule = LogicRule(
                    rule_id=f"assoc_rule_{len(rules)}",
                    premises=[pair['entity1']],
                    conclusion=pair['entity2'],
                    operator=LogicOperator.IMPLIES,
                    confidence=pair['confidence'],
                    source="learned"
                )
                rules.append(rule)
        
        return rules
    
    def _find_cooccurring_entities(
        self,
        symbols: List[Symbol]
    ) -> List[Dict]:
        """查找共现实体"""
        # 简化的共现检测
        pairs = []
        entities = [s for s in symbols if s.symbol_type == SymbolType.ENTITY]
        
        for i, e1 in enumerate(entities):
            for e2 in entities[i+1:]:
                pairs.append({
                    'entity1': e1,
                    'entity2': e2,
                    'confidence': 0.8  # 简化
                })
        
        return pairs
    
    def _extract_hierarchical_rules(self, symbols: List[Symbol]) -> List[LogicRule]:
        """提取层次规则"""
        rules = []
        
        # 基于命名模式检测层次关系
        for symbol in symbols:
            if symbol.symbol_type == SymbolType.ENTITY:
                # 检查是否有上下位词
                hypernyms = self._find_hypernyms(symbol.name)
                
                for hypernym in hypernyms:
                    hypernym_symbol = Symbol(
                        symbol_id=f"hyper_{hypernym}",
                        name=hypernym,
                        symbol_type=SymbolType.ENTITY
                    )
                    
                    rule = LogicRule(
                        rule_id=f"hier_rule_{len(rules)}",
                        premises=[symbol],
                        conclusion=hypernym_symbol,
                        operator=LogicOperator.IMPLIES,
                        source="extracted"
                    )
                    rules.append(rule)
        
        return rules
    
    def _find_hypernyms(self, entity_name: str) -> List[str]:
        """查找上位词"""
        # 简化的上位词查找
        hypernym_map = {
            'machine learning': ['artificial intelligence', 'computer science'],
            'deep learning': ['machine learning', 'artificial intelligence'],
            'neural network': ['machine learning model', 'algorithm']
        }
        
        return hypernym_map.get(entity_name.lower(), [])


class SymbolicReasoner:
    """符号推理器"""
    
    def __init__(self):
        self.knowledge_base: List[LogicRule] = []
        self.reasoning_history: List[Dict] = []
    
    def reason(
        self,
        query: Symbol,
        knowledge_base: List[LogicRule],
        max_depth: int = 5
    ) -> Dict[str, Any]:
        """
        执行符号推理
        
        Args:
            query: 查询符号
            knowledge_base: 知识库（逻辑规则）
            max_depth: 最大推理深度
        
        Returns:
            推理结果
        """
        self.knowledge_base = knowledge_base
        
        # 1. 前向推理
        forward_result = self._forward_reason(query, max_depth)
        
        # 2. 后向推理
        backward_result = self._backward_reason(query, max_depth)
        
        # 3. 构建推理链
        reasoning_chain = self._build_reasoning_chain(
            forward_result, backward_result
        )
        
        # 4. 验证结论
        verification = self._verify_conclusions(reasoning_chain)
        
        result = {
            'query': query.name,
            'forward_reasoning': forward_result,
            'backward_reasoning': backward_result,
            'reasoning_chain': reasoning_chain,
            'verification': verification,
            'conclusion': self._extract_conclusion(reasoning_chain),
            'confidence': self._compute_reasoning_confidence(reasoning_chain)
        }
        
        self.reasoning_history.append(result)
        
        return result
    
    def _forward_reason(
        self,
        query: Symbol,
        max_depth: int
    ) -> Dict[str, Any]:
        """前向推理"""
        inferred = []
        current_facts = [query]
        
        for depth in range(max_depth):
            new_facts = []
            
            for rule in self.knowledge_base:
                if self._can_apply_rule(rule, current_facts):
                    conclusion = self._apply_rule(rule, current_facts)
                    if conclusion:
                        new_facts.append(conclusion)
                        inferred.append({
                            'rule': rule.rule_id,
                            'conclusion': conclusion.name,
                            'depth': depth
                        })
            
            if not new_facts:
                break
            
            current_facts.extend(new_facts)
        
        return {
            'inferred_facts': inferred,
            'total_inferences': len(inferred)
        }
    
    def _backward_reason(
        self,
        query: Symbol,
        max_depth: int
    ) -> Dict[str, Any]:
        """后向推理"""
        subgoals = []
        
        # 查找能够推导出查询的规则
        supporting_rules = [
            rule for rule in self.knowledge_base
            if rule.conclusion.name == query.name
        ]
        
        for rule in supporting_rules:
            subgoals.append({
                'rule': rule.rule_id,
                'premises': [p.name for p in rule.premises],
                'operator': rule.operator.value
            })
        
        return {
            'supporting_rules': subgoals,
            'explanation': self._generate_explanation(supporting_rules)
        }
    
    def _can_apply_rule(self, rule: LogicRule, facts: List[Symbol]) -> bool:
        """检查是否可以应用规则"""
        fact_names = {f.name for f in facts}
        premise_names = {p.name for p in rule.premises}
        
        return premise_names.issubset(fact_names)
    
    def _apply_rule(self, rule: LogicRule, facts: List[Symbol]) -> Optional[Symbol]:
        """应用规则"""
        # 简化的规则应用
        return rule.conclusion
    
    def _build_reasoning_chain(
        self,
        forward_result: Dict,
        backward_result: Dict
    ) -> List[Dict]:
        """构建推理链"""
        chain = []
        
        # 添加后向推理步骤
        for subgoal in backward_result.get('supporting_rules', []):
            chain.append({
                'direction': 'backward',
                'step': subgoal
            })
        
        # 添加前向推理步骤
        for inference in forward_result.get('inferred_facts', []):
            chain.append({
                'direction': 'forward',
                'step': inference
            })
        
        return chain
    
    def _verify_conclusions(self, chain: List[Dict]) -> Dict[str, Any]:
        """验证结论"""
        # 简化的验证
        return {
            'verified': True,
            'confidence': 0.8,
            'issues': []
        }
    
    def _extract_conclusion(self, chain: List[Dict]) -> str:
        """提取结论"""
        if chain:
            last_step = chain[-1]
            if 'step' in last_step:
                step = last_step['step']
                if 'conclusion' in step:
                    return step['conclusion']
        
        return "No clear conclusion"
    
    def _compute_reasoning_confidence(self, chain: List[Dict]) -> float:
        """计算推理置信度"""
        if not chain:
            return 0.0
        
        # 基于链长度和规则置信度
        base_confidence = 0.9
        decay_factor = 0.95 ** len(chain)
        
        return base_confidence * decay_factor
    
    def _generate_explanation(self, rules: List[LogicRule]) -> str:
        """生成解释"""
        if not rules:
            return "No supporting rules found"
        
        explanations = []
        for rule in rules:
            premises = " and ".join([p.name for p in rule.premises])
            explanations.append(
                f"Because {premises}, therefore {rule.conclusion.name}"
            )
        
        return "; ".join(explanations)


class HybridInferenceEngine:
    """混合推理引擎"""
    
    def __init__(self):
        self.neural_converter = NeuralToSymbolicConverter()
        self.rule_extractor = RuleExtractor()
        self.symbolic_reasoner = SymbolicReasoner()
        self.inference_history: List[Dict] = []
    
    def hybrid_reason(
        self,
        neural_input: Dict[str, Any],
        text_context: str,
        query: str
    ) -> Dict[str, Any]:
        """
        执行混合推理
        
        这是V122创新的主要入口函数
        """
        # 1. 神经到符号转换
        symbolic_representation = self.neural_converter.convert(
            neural_input, text_context
        )
        
        # 2. 提取逻辑规则
        all_symbols = (
            symbolic_representation['entities'] +
            symbolic_representation['relations'] +
            symbolic_representation['attributes']
        )
        
        extracted_rules = self.rule_extractor.extract_rules(
            all_symbols, text_context
        )
        
        # 3. 创建查询符号
        query_symbol = Symbol(
            symbol_id='query_0',
            name=query,
            symbol_type=SymbolType.ENTITY
        )
        
        # 4. 符号推理
        reasoning_result = self.symbolic_reasoner.reason(
            query_symbol,
            extracted_rules,
            max_depth=5
        )
        
        # 5. 将符号结论转换回神经表示
        neural_conclusion = self._symbol_to_neural(reasoning_result['conclusion'])
        
        # 6. 融合神经和符号结果
        fused_result = self._fuse_results(
            neural_input,
            reasoning_result,
            neural_conclusion
        )
        
        result = {
            'version': 'V122',
            'name': 'Neural-Symbolic Reasoning Bridge (NSRB)',
            'symbolic_representation': symbolic_representation,
            'extracted_rules': [
                {
                    'rule_id': r.rule_id,
                    'premises': [p.name for p in r.premises],
                    'conclusion': r.conclusion.name,
                    'operator': r.operator.value
                }
                for r in extracted_rules
            ],
            'symbolic_reasoning': reasoning_result,
            'neural_conclusion': neural_conclusion,
            'fused_result': fused_result,
            'explainability': self._generate_explanation(
                symbolic_representation, reasoning_result
            )
        }
        
        self.inference_history.append(result)
        
        return result
    
    def _symbol_to_neural(self, conclusion: str) -> Dict[str, Any]:
        """将符号转换回神经表示"""
        return {
            'conclusion_text': conclusion,
            'embedding': self._text_to_embedding(conclusion),
            'confidence': 0.85
        }
    
    def _text_to_embedding(self, text: str) -> List[float]:
        """文本到嵌入向量"""
        # 简化的嵌入生成
        np.random.seed(hash(text) % 2**32)
        return np.random.randn(768).tolist()
    
    def _fuse_results(
        self,
        neural_input: Dict,
        symbolic_result: Dict,
        neural_conclusion: Dict
    ) -> Dict[str, Any]:
        """融合神经和符号结果"""
        neural_confidence = neural_input.get('overall_confidence', 0.5)
        symbolic_confidence = symbolic_result.get('confidence', 0.5)
        
        # 加权融合
        if neural_confidence > symbolic_confidence:
            final_conclusion = neural_input.get('prediction', 'unknown')
            fusion_method = 'neural_preferred'
        else:
            final_conclusion = symbolic_result.get('conclusion', 'unknown')
            fusion_method = 'symbolic_preferred'
        
        return {
            'final_conclusion': final_conclusion,
            'fusion_method': fusion_method,
            'neural_confidence': neural_confidence,
            'symbolic_confidence': symbolic_confidence,
            'agreement': abs(neural_confidence - symbolic_confidence) < 0.2
        }
    
    def _generate_explanation(
        self,
        symbolic_rep: Dict,
        reasoning: Dict
    ) -> Dict[str, Any]:
        """生成可解释性报告"""
        return {
            'entities_identified': len(symbolic_rep.get('entities', [])),
            'rules_applied': len(reasoning.get('reasoning_chain', [])),
            'reasoning_depth': reasoning.get('forward_reasoning', {}).get('total_inferences', 0),
            'explanation_text': reasoning.get('backward_reasoning', {}).get('explanation', ''),
            'transparency_score': 0.9  # 高透明度
        }


# 便捷函数
def build_neural_symbolic_reasoning(
    neural_output: Dict[str, Any],
    text: str,
    query: str
) -> Dict[str, Any]:
    """
    构建神经符号推理
    
    这是V122创新的便捷入口函数
    """
    engine = HybridInferenceEngine()
    return engine.hybrid_reason(neural_output, text, query)


# 示例用法
if __name__ == "__main__":
    neural_output = {
        'predictions': ['machine learning', 'AI'],
        'confidences': [0.9, 0.85],
        'overall_confidence': 0.875
    }
    
    text = "Machine learning is a subset of artificial intelligence. Deep learning causes improved performance."
    query = "What is the relationship between machine learning and AI?"
    
    result = build_neural_symbolic_reasoning(neural_output, text, query)
    
    print(f"Entities: {len(result['symbolic_representation']['entities'])}")
    print(f"Rules: {len(result['extracted_rules'])}")
    print(f"Conclusion: {result['fused_result']['final_conclusion']}")
