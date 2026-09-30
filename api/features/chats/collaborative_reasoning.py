"""
V107: Collaborative Multi-Agent Reasoning (CMAR)
协同多智能体推理 - 模拟多个专家智能体协同解决问题

创新点：
1. 多视角推理：不同"专家"从不同角度分析问题
2. 辩论机制：智能体之间进行逻辑辩论，找出最佳答案
3. 共识构建：通过投票和证据权重达成共识
4. 认知多样性：模拟不同领域的专家思维方式
"""

import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from enum import Enum
import json


class ExpertType(Enum):
    """专家类型枚举"""
    FACTUAL = "factual"           # 事实核查专家
    LOGICAL = "logical"           # 逻辑推理专家
    CREATIVE = "creative"         # 创造性思维专家
    CRITICAL = "critical"         # 批判性思维专家
    SYNTHESIS = "synthesis"       # 综合分析专家


@dataclass
class ExpertOpinion:
    """专家意见数据结构"""
    expert_type: ExpertType
    confidence: float
    reasoning: str
    evidence_support: List[str]
    counter_arguments: List[str] = field(default_factory=list)
    agreement_with_others: Dict[str, float] = field(default_factory=dict)


@dataclass
class DebateRound:
    """辩论轮次数据结构"""
    round_number: int
    opinions: List[ExpertOpinion]
    consensus_score: float
    disagreements: List[str]
    refinements: List[str]


class ExpertAgent:
    """专家智能体基类"""
    
    def __init__(self, expert_type: ExpertType, system_prompt: str):
        self.expert_type = expert_type
        self.system_prompt = system_prompt
        self.memory: List[Dict] = []
    
    async def analyze(
        self, 
        query: str, 
        evidence: List[Dict],
        other_opinions: Optional[List[ExpertOpinion]] = None
    ) -> ExpertOpinion:
        """
        分析问题并给出专家意见
        
        Args:
            query: 用户查询
            evidence: 相关证据列表
            other_opinions: 其他专家的意见（用于辩论）
        
        Returns:
            ExpertOpinion: 专家意见
        """
        # 根据专家类型调整分析策略
        if self.expert_type == ExpertType.FACTUAL:
            return await self._factual_analysis(query, evidence, other_opinions)
        elif self.expert_type == ExpertType.LOGICAL:
            return await self._logical_analysis(query, evidence, other_opinions)
        elif self.expert_type == ExpertType.CREATIVE:
            return await self._creative_analysis(query, evidence, other_opinions)
        elif self.expert_type == ExpertType.CRITICAL:
            return await self._critical_analysis(query, evidence, other_opinions)
        elif self.expert_type == ExpertType.SYNTHESIS:
            return await self._synthesis_analysis(query, evidence, other_opinions)
        else:
            raise ValueError(f"Unknown expert type: {self.expert_type}")
    
    async def _factual_analysis(
        self, 
        query: str, 
        evidence: List[Dict],
        other_opinions: Optional[List[ExpertOpinion]]
    ) -> ExpertOpinion:
        """事实核查专家分析"""
        # 检查证据的事实准确性
        verified_facts = []
        uncertain_claims = []
        
        for ev in evidence:
            # 模拟事实验证
            confidence = self._calculate_factual_confidence(ev)
            if confidence > 0.8:
                verified_facts.append(ev.get('content', ''))
            else:
                uncertain_claims.append(ev.get('content', ''))
        
        # 检查与其他专家意见的一致性
        counter_args = []
        if other_opinions:
            for opinion in other_opinions:
                if opinion.confidence < 0.6:
                    counter_args.append(
                        f"{opinion.expert_type.value} expert's low confidence ({opinion.confidence:.2f}) "
                        f"suggests potential factual gaps"
                    )
        
        confidence = len(verified_facts) / max(len(evidence), 1) if evidence else 0.5
        
        return ExpertOpinion(
            expert_type=self.expert_type,
            confidence=min(confidence + 0.2, 1.0),
            reasoning=f"Verified {len(verified_facts)} facts, found {len(uncertain_claims)} uncertain claims",
            evidence_support=verified_facts,
            counter_arguments=counter_args
        )
    
    async def _logical_analysis(
        self, 
        query: str, 
        evidence: List[Dict],
        other_opinions: Optional[List[ExpertOpinion]]
    ) -> ExpertOpinion:
        """逻辑推理专家分析"""
        # 构建逻辑推理链
        reasoning_chain = []
        logical_gaps = []
        
        # 检查证据之间的逻辑关系
        for i in range(len(evidence) - 1):
            current = evidence[i]
            next_ev = evidence[i + 1]
            
            # 检查逻辑连贯性
            if self._check_logical_connection(current, next_ev):
                reasoning_chain.append(f"{current.get('id')} → {next_ev.get('id')}")
            else:
                logical_gaps.append(f"Gap between {current.get('id')} and {next_ev.get('id')}")
        
        # 检查循环论证
        circular_reasoning = self._detect_circular_reasoning(evidence)
        
        confidence = 1.0 - (len(logical_gaps) * 0.1) - (len(circular_reasoning) * 0.2)
        confidence = max(confidence, 0.3)
        
        return ExpertOpinion(
            expert_type=self.expert_type,
            confidence=confidence,
            reasoning=f"Logical chain: {' → '.join(reasoning_chain)}. Gaps: {len(logical_gaps)}",
            evidence_support=reasoning_chain,
            counter_arguments=circular_reasoning + logical_gaps
        )
    
    async def _creative_analysis(
        self, 
        query: str, 
        evidence: List[Dict],
        other_opinions: Optional[List[ExpertOpinion]]
    ) -> ExpertOpinion:
        """创造性思维专家分析"""
        # 寻找证据中的创新关联
        novel_connections = []
        alternative_interpretations = []
        
        # 跨领域联想
        for i, ev1 in enumerate(evidence):
            for ev2 in evidence[i+1:]:
                connection = self._find_novel_connection(ev1, ev2)
                if connection:
                    novel_connections.append(connection)
        
        # 提出替代解释
        for ev in evidence:
            alt = self._generate_alternative_interpretation(ev)
            if alt:
                alternative_interpretations.append(alt)
        
        confidence = min(0.5 + len(novel_connections) * 0.1, 0.9)
        
        return ExpertOpinion(
            expert_type=self.expert_type,
            confidence=confidence,
            reasoning=f"Found {len(novel_connections)} novel connections, proposed {len(alternative_interpretations)} alternatives",
            evidence_support=novel_connections,
            counter_arguments=[]
        )
    
    async def _critical_analysis(
        self, 
        query: str, 
        evidence: List[Dict],
        other_opinions: Optional[List[ExpertOpinion]]
    ) -> ExpertOpinion:
        """批判性思维专家分析"""
        # 识别偏见和假设
        biases = []
        assumptions = []
        weaknesses = []
        
        for ev in evidence:
            # 检查来源偏见
            source_bias = self._detect_source_bias(ev)
            if source_bias:
                biases.append(source_bias)
            
            # 识别隐含假设
            assumption = self._identify_assumptions(ev)
            if assumption:
                assumptions.append(assumption)
            
            # 找出证据弱点
            weakness = self._identify_weaknesses(ev)
            if weakness:
                weaknesses.append(weakness)
        
        # 批判其他专家的意见
        if other_opinions:
            for opinion in other_opinions:
                if opinion.confidence > 0.9:
                    weaknesses.append(
                        f"Overconfidence in {opinion.expert_type.value} analysis may indicate bias"
                    )
        
        confidence = max(0.4, 1.0 - len(biases) * 0.15)
        
        return ExpertOpinion(
            expert_type=self.expert_type,
            confidence=confidence,
            reasoning=f"Identified {len(biases)} biases, {len(assumptions)} assumptions, {len(weaknesses)} weaknesses",
            evidence_support=[],
            counter_arguments=biases + assumptions + weaknesses
        )
    
    async def _synthesis_analysis(
        self, 
        query: str, 
        evidence: List[Dict],
        other_opinions: Optional[List[ExpertOpinion]]
    ) -> ExpertOpinion:
        """综合分析专家分析"""
        if not other_opinions:
            return ExpertOpinion(
                expert_type=self.expert_type,
                confidence=0.5,
                reasoning="Waiting for other experts' opinions to synthesize",
                evidence_support=[],
                counter_arguments=[]
            )
        
        # 综合所有专家意见
        synthesis_points = []
        conflicts = []
        
        # 找出共识
        high_confidence_areas = [
            op for op in other_opinions if op.confidence > 0.7
        ]
        
        # 识别冲突
        for i, op1 in enumerate(other_opinions):
            for op2 in other_opinions[i+1:]:
                if abs(op1.confidence - op2.confidence) > 0.3:
                    conflicts.append(
                        f"Disagreement between {op1.expert_type.value} and {op2.expert_type.value}"
                    )
        
        # 构建综合结论
        synthesis_points = [
            f"Consensus on {len(high_confidence_areas)} key points",
            f"Average confidence: {sum(op.confidence for op in other_opinions) / len(other_opinions):.2f}"
        ]
        
        confidence = len(high_confidence_areas) / len(other_opinions) if other_opinions else 0.5
        
        return ExpertOpinion(
            expert_type=self.expert_type,
            confidence=confidence,
            reasoning="; ".join(synthesis_points),
            evidence_support=synthesis_points,
            counter_arguments=conflicts
        )
    
    # Helper methods
    def _calculate_factual_confidence(self, evidence: Dict) -> float:
        """计算证据的事实置信度"""
        # 基于来源可靠性、时间戳、交叉验证等
        base_confidence = 0.7
        
        # 如果有多个来源支持，提高置信度
        if evidence.get('source_count', 1) > 1:
            base_confidence += 0.1
        
        # 如果证据较新，提高置信度
        if evidence.get('is_recent', False):
            base_confidence += 0.1
        
        return min(base_confidence, 1.0)
    
    def _check_logical_connection(self, ev1: Dict, ev2: Dict) -> bool:
        """检查两个证据之间的逻辑连接"""
        # 检查关键词重叠、因果关系等
        content1 = ev1.get('content', '').lower()
        content2 = ev2.get('content', '').lower()
        
        # 简单的关键词匹配
        words1 = set(content1.split())
        words2 = set(content2.split())
        overlap = len(words1 & words2)
        
        return overlap >= 2  # 至少有2个共同词
    
    def _detect_circular_reasoning(self, evidence: List[Dict]) -> List[str]:
        """检测循环论证"""
        circular = []
        for i, ev1 in enumerate(evidence):
            for j, ev2 in enumerate(evidence[i+1:], i+1):
                if ev1.get('id') == ev2.get('references', [None])[0]:
                    circular.append(f"Potential circular reference: {ev1.get('id')} ↔ {ev2.get('id')}")
        return circular
    
    def _find_novel_connection(self, ev1: Dict, ev2: Dict) -> Optional[str]:
        """寻找证据之间的新颖连接"""
        # 模拟跨领域联想
        content1 = ev1.get('content', '')
        content2 = ev2.get('content', '')
        
        # 如果来自不同领域但有相似概念
        if ev1.get('domain') != ev2.get('domain'):
            return f"Cross-domain link: {ev1.get('domain')} × {ev2.get('domain')}"
        return None
    
    def _generate_alternative_interpretation(self, evidence: Dict) -> Optional[str]:
        """生成替代解释"""
        # 模拟创造性思维
        if evidence.get('ambiguity_score', 0) > 0.5:
            return f"Alternative interpretation of {evidence.get('id')}: [context-dependent meaning]"
        return None
    
    def _detect_source_bias(self, evidence: Dict) -> Optional[str]:
        """检测来源偏见"""
        source = evidence.get('source', '')
        if 'blog' in source.lower() or 'opinion' in source.lower():
            return f"Potential bias in {source}: non-peer-reviewed source"
        return None
    
    def _identify_assumptions(self, evidence: Dict) -> Optional[str]:
        """识别隐含假设"""
        content = evidence.get('content', '')
        # 检测假设性语言
        if 'obviously' in content.lower() or 'clearly' in content.lower():
            return f"Assumption in {evidence.get('id')}: unstated premise"
        return None
    
    def _identify_weaknesses(self, evidence: Dict) -> Optional[str]:
        """识别证据弱点"""
        weaknesses = []
        
        if not evidence.get('citations'):
            weaknesses.append("no citations")
        
        if evidence.get('age_days', 0) > 365:
            weaknesses.append("potentially outdated")
        
        if weaknesses:
            return f"Weakness in {evidence.get('id')}: {', '.join(weaknesses)}"
        return None


class CollaborativeReasoningEngine:
    """
    协同推理引擎
    协调多个专家智能体进行协作推理
    """
    
    def __init__(self, max_rounds: int = 3):
        self.max_rounds = max_rounds
        self.experts: List[ExpertAgent] = []
        self.debate_history: List[DebateRound] = []
        self._initialize_experts()
    
    def _initialize_experts(self):
        """初始化专家智能体"""
        expert_configs = [
            (ExpertType.FACTUAL, "You are a fact-checking expert focused on accuracy and verification."),
            (ExpertType.LOGICAL, "You are a logical reasoning expert focused on consistency and validity."),
            (ExpertType.CREATIVE, "You are a creative thinking expert focused on novel connections and insights."),
            (ExpertType.CRITICAL, "You are a critical thinking expert focused on identifying biases and weaknesses."),
            (ExpertType.SYNTHESIS, "You are a synthesis expert focused on integrating diverse perspectives.")
        ]
        
        for expert_type, prompt in expert_configs:
            self.experts.append(ExpertAgent(expert_type, prompt))
    
    async def reason(
        self, 
        query: str, 
        evidence: List[Dict],
        consensus_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """
        执行协同推理
        
        Args:
            query: 用户查询
            evidence: 证据列表
            consensus_threshold: 共识阈值
        
        Returns:
            推理结果，包含最终结论、置信度、专家意见等
        """
        all_opinions: List[ExpertOpinion] = []
        
        # 多轮辩论
        for round_num in range(self.max_rounds):
            round_opinions: List[ExpertOpinion] = []
            
            # 每个专家发表意见
            for expert in self.experts:
                opinion = await expert.analyze(query, evidence, all_opinions)
                round_opinions.append(opinion)
            
            # 计算共识度
            consensus_score = self._calculate_consensus(round_opinions)
            
            # 识别分歧
            disagreements = self._identify_disagreements(round_opinions)
            
            # 记录改进建议
            refinements = self._generate_refinements(round_opinions, disagreements)
            
            # 保存辩论轮次
            debate_round = DebateRound(
                round_number=round_num + 1,
                opinions=round_opinions,
                consensus_score=consensus_score,
                disagreements=disagreements,
                refinements=refinements
            )
            self.debate_history.append(debate_round)
            
            all_opinions = round_opinions
            
            # 如果达到共识阈值，提前结束
            if consensus_score >= consensus_threshold:
                break
        
        # 生成最终结论
        final_conclusion = self._generate_final_conclusion(all_opinions)
        
        return {
            'final_conclusion': final_conclusion,
            'confidence': self._calculate_final_confidence(all_opinions),
            'expert_opinions': [
                {
                    'type': op.expert_type.value,
                    'confidence': op.confidence,
                    'reasoning': op.reasoning,
                    'evidence_support': op.evidence_support,
                    'counter_arguments': op.counter_arguments
                }
                for op in all_opinions
            ],
            'debate_rounds': [
                {
                    'round': r.round_number,
                    'consensus_score': r.consensus_score,
                    'disagreements': r.disagreements,
                    'refinements': r.refinements
                }
                for r in self.debate_history
            ],
            'reasoning_depth': len(self.debate_history),
            'collaboration_score': self._calculate_collaboration_score(all_opinions)
        }
    
    def _calculate_consensus(self, opinions: List[ExpertOpinion]) -> float:
        """计算专家之间的共识度"""
        if len(opinions) < 2:
            return 1.0
        
        confidences = [op.confidence for op in opinions]
        avg_confidence = sum(confidences) / len(confidences)
        variance = sum((c - avg_confidence) ** 2 for c in confidences) / len(confidences)
        
        # 共识度 = 平均置信度 - 方差惩罚
        consensus = avg_confidence - (variance * 0.5)
        return max(consensus, 0.0)
    
    def _identify_disagreements(self, opinions: List[ExpertOpinion]) -> List[str]:
        """识别专家之间的分歧"""
        disagreements = []
        
        for i, op1 in enumerate(opinions):
            for op2 in opinions[i+1:]:
                if abs(op1.confidence - op2.confidence) > 0.3:
                    disagreements.append(
                        f"{op1.expert_type.value} vs {op2.expert_type.value}: "
                        f"confidence gap {abs(op1.confidence - op2.confidence):.2f}"
                    )
        
        return disagreements
    
    def _generate_refinements(
        self, 
        opinions: List[ExpertOpinion], 
        disagreements: List[str]
    ) -> List[str]:
        """生成改进建议"""
        refinements = []
        
        # 基于分歧生成改进建议
        for disagreement in disagreements:
            refinements.append(f"Address: {disagreement}")
        
        # 基于低置信度专家的意见
        low_confidence_ops = [op for op in opinions if op.confidence < 0.5]
        for op in low_confidence_ops:
            refinements.append(
                f"Investigate {op.expert_type.value} concerns: {op.reasoning[:50]}..."
            )
        
        return refinements
    
    def _generate_final_conclusion(self, opinions: List[ExpertOpinion]) -> str:
        """生成最终结论"""
        # 综合所有专家意见
        synthesis_opinion = next(
            (op for op in opinions if op.expert_type == ExpertType.SYNTHESIS),
            None
        )
        
        if synthesis_opinion:
            return synthesis_opinion.reasoning
        
        # 如果没有综合专家，使用加权平均
        weighted_confidences = []
        for op in opinions:
            weight = 1.0
            if op.expert_type == ExpertType.FACTUAL:
                weight = 1.2  # 事实专家权重更高
            elif op.expert_type == ExpertType.CRITICAL:
                weight = 1.1  # 批判专家权重较高
            weighted_confidences.append(op.confidence * weight)
        
        avg_confidence = sum(weighted_confidences) / sum(
            1.2 if op.expert_type == ExpertType.FACTUAL else 
            1.1 if op.expert_type == ExpertType.CRITICAL else 1.0
            for op in opinions
        )
        
        if avg_confidence > 0.8:
            return "High confidence conclusion based on strong expert consensus"
        elif avg_confidence > 0.6:
            return "Moderate confidence conclusion with some expert reservations"
        else:
            return "Low confidence conclusion due to significant expert disagreement"
    
    def _calculate_final_confidence(self, opinions: List[ExpertOpinion]) -> float:
        """计算最终置信度"""
        confidences = [op.confidence for op in opinions]
        
        # 使用加权平均
        weights = {
            ExpertType.FACTUAL: 1.3,
            ExpertType.LOGICAL: 1.2,
            ExpertType.SYNTHESIS: 1.2,
            ExpertType.CRITICAL: 1.1,
            ExpertType.CREATIVE: 0.8
        }
        
        weighted_sum = sum(
            op.confidence * weights.get(op.expert_type, 1.0)
            for op in opinions
        )
        total_weight = sum(weights.get(op.expert_type, 1.0) for op in opinions)
        
        return min(weighted_sum / total_weight, 1.0)
    
    def _calculate_collaboration_score(self, opinions: List[ExpertOpinion]) -> float:
        """计算协作质量分数"""
        # 基于专家多样性、共识度、推理深度
        diversity = len(set(op.expert_type for op in opinions)) / len(ExpertType)
        consensus = self._calculate_consensus(opinions)
        depth = len(self.debate_history) / self.max_rounds
        
        return (diversity * 0.3 + consensus * 0.5 + depth * 0.2)


# 便捷函数
async def build_collaborative_reasoning(
    query: str,
    evidence: List[Dict],
    max_rounds: int = 3
) -> Dict[str, Any]:
    """
    构建协同推理
    
    这是V107创新的主要入口函数
    """
    engine = CollaborativeReasoningEngine(max_rounds=max_rounds)
    result = await engine.reason(query, evidence)
    
    return {
        'version': 'V107',
        'name': 'Collaborative Multi-Agent Reasoning (CMAR)',
        'description': 'Multi-expert collaborative reasoning with debate and consensus',
        'result': result
    }


# 示例用法
if __name__ == "__main__":
    async def test():
        query = "What are the implications of transformer architectures in NLP?"
        evidence = [
            {'id': 'ev1', 'content': 'Transformers use self-attention mechanisms', 'source': 'paper', 'domain': 'ml'},
            {'id': 'ev2', 'content': 'Self-attention allows parallel processing', 'source': 'blog', 'domain': 'cs'},
            {'id': 'ev3', 'content': 'BERT and GPT are based on transformers', 'source': 'paper', 'domain': 'nlp'},
        ]
        
        result = await build_collaborative_reasoning(query, evidence)
        print(json.dumps(result, indent=2))
    
    asyncio.run(test())
