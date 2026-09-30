"""
V127: Hierarchical Multi-Agent Debate (HMAD)
层次化多智能体辩论 - 层次化多智能体辩论系统

创新点：
1. 层次化辩论结构：多层次辩论架构
2. 角色专业化分工：不同层级的专业智能体
3. 共识达成机制：通过辩论达成共识
4. 质量评估与选择：评估和选择最佳答案
"""

import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import defaultdict
import numpy as np


class AgentRole(Enum):
    """智能体角色"""
    STRATEGIC = "strategic"  # 战略层
    TACTICAL = "tactical"    # 战术层
    OPERATIONAL = "operational"  # 执行层


class DebateStance(Enum):
    """辩论立场"""
    PRO = "pro"
    CON = "con"
    NEUTRAL = "neutral"
    SYNTHESIS = "synthesis"


class DebatePhase(Enum):
    """辩论阶段"""
    OPENING = "opening"
    ARGUMENTATION = "argumentation"
    REBUTTAL = "rebuttal"
    CROSS_EXAMINATION = "cross_examination"
    CLOSING = "closing"
    DELIBERATION = "deliberation"


@dataclass
class Argument:
    """论点"""
    argument_id: str
    content: str
    stance: DebateStance
    supporting_evidence: List[str]
    confidence: float
    agent_id: str
    timestamp: datetime


@dataclass
class DebateAgent:
    """辩论智能体"""
    agent_id: str
    name: str
    role: AgentRole
    expertise: List[str]
    stance: DebateStance
    arguments: List[Argument] = field(default_factory=list)
    credibility_score: float = 1.0


@dataclass
class DebateRound:
    """辩论轮次"""
    round_id: str
    phase: DebatePhase
    arguments: List[Argument]
    rebuttals: List[Dict[str, Any]]
    timestamp: datetime


class HierarchicalDebateStructure:
    """层次化辩论结构"""
    
    def __init__(self):
        self.levels: Dict[AgentRole, List[DebateAgent]] = {
            AgentRole.STRATEGIC: [],
            AgentRole.TACTICAL: [],
            AgentRole.OPERATIONAL: []
        }
        self.debate_history: List[DebateRound] = []
    
    def create_hierarchy(
        self,
        topic: str,
        num_strategic: int = 2,
        num_tactical: int = 4,
        num_operational: int = 6
    ) -> Dict[str, Any]:
        """
        创建层次化辩论结构
        
        Args:
            topic: 辩论主题
            num_strategic: 战略层智能体数量
            num_tactical: 战术层智能体数量
            num_operational: 执行层智能体数量
        
        Returns:
            层次结构配置
        """
        # 1. 创建战略层智能体
        strategic_agents = self._create_strategic_agents(topic, num_strategic)
        self.levels[AgentRole.STRATEGIC] = strategic_agents
        
        # 2. 创建战术层智能体
        tactical_agents = self._create_tactical_agents(topic, num_tactical)
        self.levels[AgentRole.TACTICAL] = tactical_agents
        
        # 3. 创建执行层智能体
        operational_agents = self._create_operational_agents(topic, num_operational)
        self.levels[AgentRole.OPERATIONAL] = operational_agents
        
        # 4. 建立层级关系
        hierarchy = self._establish_hierarchy()
        
        return {
            'topic': topic,
            'levels': {
                'strategic': [
                    {'agent_id': a.agent_id, 'name': a.name, 'expertise': a.expertise}
                    for a in strategic_agents
                ],
                'tactical': [
                    {'agent_id': a.agent_id, 'name': a.name, 'expertise': a.expertise}
                    for a in tactical_agents
                ],
                'operational': [
                    {'agent_id': a.agent_id, 'name': a.name, 'expertise': a.expertise}
                    for a in operational_agents
                ]
            },
            'hierarchy_map': hierarchy,
            'total_agents': len(strategic_agents) + len(tactical_agents) + len(operational_agents)
        }
    
    def _create_strategic_agents(self, topic: str, count: int) -> List[DebateAgent]:
        """创建战略层智能体"""
        agents = []
        
        strategic_roles = [
            ('Strategic Visionary', ['long_term_planning', 'goal_setting', 'risk_assessment']),
            ('Policy Architect', ['policy_design', 'system_thinking', 'constraint_analysis'])
        ]
        
        for i in range(min(count, len(strategic_roles))):
            name, expertise = strategic_roles[i]
            agents.append(DebateAgent(
                agent_id=f"strategic_{i}",
                name=name,
                role=AgentRole.STRATEGIC,
                expertise=expertise,
                stance=DebateStance.NEUTRAL
            ))
        
        return agents
    
    def _create_tactical_agents(self, topic: str, count: int) -> List[DebateAgent]:
        """创建战术层智能体"""
        agents = []
        
        tactical_roles = [
            ('Evidence Analyst', ['evidence_evaluation', 'source_verification', 'fact_checking']),
            ('Logic Specialist', ['logical_reasoning', 'fallacy_detection', 'argument_structure']),
            ('Context Expert', ['domain_knowledge', 'historical_context', 'cultural_awareness']),
            ('Perspective Balancer', ['viewpoint_analysis', 'bias_detection', 'fairness_evaluation'])
        ]
        
        for i in range(min(count, len(tactical_roles))):
            name, expertise = tactical_roles[i]
            agents.append(DebateAgent(
                agent_id=f"tactical_{i}",
                name=name,
                role=AgentRole.TACTICAL,
                expertise=expertise,
                stance=DebateStance.PRO if i % 2 == 0 else DebateStance.CON
            ))
        
        return agents
    
    def _create_operational_agents(self, topic: str, count: int) -> List[DebateAgent]:
        """创建执行层智能体"""
        agents = []
        
        operational_roles = [
            'Data Collector', 'Fact Checker', 'Example Provider',
            'Counter-example Finder', 'Implementation Planner', 'Feasibility Assessor'
        ]
        
        for i in range(min(count, len(operational_roles))):
            agents.append(DebateAgent(
                agent_id=f"operational_{i}",
                name=operational_roles[i],
                role=AgentRole.OPERATIONAL,
                expertise=['data_collection', 'verification', 'implementation'],
                stance=DebateStance.NEUTRAL
            ))
        
        return agents
    
    def _establish_hierarchy(self) -> Dict[str, List[str]]:
        """建立层级关系"""
        hierarchy = {}
        
        # 战略层指导战术层
        for strategic in self.levels[AgentRole.STRATEGIC]:
            hierarchy[strategic.agent_id] = [
                a.agent_id for a in self.levels[AgentRole.TACTICAL]
            ]
        
        # 战术层指导执行层
        for tactical in self.levels[AgentRole.TACTICAL]:
            hierarchy[tactical.agent_id] = [
                a.agent_id for a in self.levels[AgentRole.OPERATIONAL]
            ]
        
        return hierarchy
    
    def get_agents_by_level(self, role: AgentRole) -> List[DebateAgent]:
        """获取特定层级的智能体"""
        return self.levels.get(role, [])


class SpecializedRoleManager:
    """专业化角色管理器"""
    
    def __init__(self):
        self.role_definitions: Dict[str, Dict] = {}
        self.specialization_matrix: Dict[str, List[str]] = defaultdict(list)
    
    def define_role(
        self,
        role_name: str,
        responsibilities: List[str],
        required_expertise: List[str],
        debate_style: str
    ) -> Dict[str, Any]:
        """定义角色"""
        role_def = {
            'role_name': role_name,
            'responsibilities': responsibilities,
            'required_expertise': required_expertise,
            'debate_style': debate_style,
            'created_at': datetime.now()
        }
        
        self.role_definitions[role_name] = role_def
        
        # 更新专业化矩阵
        for expertise in required_expertise:
            self.specialization_matrix[expertise].append(role_name)
        
        return role_def
    
    def assign_specialized_roles(
        self,
        agents: List[DebateAgent],
        topic: str
    ) -> List[DebateAgent]:
        """分配专业化角色"""
        # 基于主题分析分配角色
        topic_keywords = self._extract_topic_keywords(topic)
        
        for agent in agents:
            # 匹配最适合的专业领域
            best_match = self._find_best_expertise_match(
                agent.expertise,
                topic_keywords
            )
            
            # 更新智能体的专业化信息
            agent.expertise.append(best_match)
        
        return agents
    
    def _extract_topic_keywords(self, topic: str) -> List[str]:
        """提取主题关键词"""
        # 简化的关键词提取
        words = topic.lower().split()
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been'}
        return [w for w in words if w not in stopwords and len(w) > 3]
    
    def _find_best_expertise_match(
        self,
        agent_expertise: List[str],
        topic_keywords: List[str]
    ) -> str:
        """找到最佳专业匹配"""
        best_match = 'general'
        max_overlap = 0
        
        for expertise in agent_expertise:
            overlap = sum(1 for keyword in topic_keywords if keyword in expertise.lower())
            if overlap > max_overlap:
                max_overlap = overlap
                best_match = expertise
        
        return best_match


class ConsensusMechanism:
    """共识机制"""
    
    def __init__(self):
        self.consensus_threshold: float = 0.7
        self.deliberation_history: List[Dict] = []
    
    def reach_consensus(
        self,
        arguments: List[Argument],
        agents: List[DebateAgent],
        max_rounds: int = 5
    ) -> Dict[str, Any]:
        """
        达成共识
        
        Args:
            arguments: 论点列表
            agents: 智能体列表
            max_rounds: 最大协商轮数
        
        Returns:
            共识结果
        """
        round_num = 0
        current_positions = self._initialize_positions(arguments, agents)
        
        while round_num < max_rounds:
            round_num += 1
            
            # 1. 立场表达
            positions = self._express_positions(current_positions, agents)
            
            # 2. 协商讨论
            discussion = self._facilitate_discussion(positions, arguments)
            
            # 3. 立场调整
            adjustments = self._adjust_positions(discussion, agents)
            
            # 4. 检查是否达成共识
            consensus_check = self._check_consensus(adjustments)
            
            if consensus_check['consensus_reached']:
                return {
                    'consensus_reached': True,
                    'rounds': round_num,
                    'final_position': consensus_check['consensus_position'],
                    'agreement_rate': consensus_check['agreement_rate'],
                    'dissenting_views': consensus_check.get('dissent', []),
                    'deliberation_process': self.deliberation_history
                }
            
            current_positions = adjustments
            
            # 记录本轮
            self.deliberation_history.append({
                'round': round_num,
                'positions': positions,
                'discussion': discussion,
                'adjustments': adjustments
            })
        
        # 未达成共识，返回最佳妥协方案
        return self._generate_compromise(current_positions, arguments)
    
    def _initialize_positions(
        self,
        arguments: List[Argument],
        agents: List[DebateAgent]
    ) -> Dict[str, Any]:
        """初始化立场"""
        positions = {}
        
        for agent in agents:
            agent_arguments = [a for a in arguments if a.agent_id == agent.agent_id]
            positions[agent.agent_id] = {
                'stance': agent.stance.value,
                'arguments': [a.content for a in agent_arguments],
                'confidence': np.mean([a.confidence for a in agent_arguments]) if agent_arguments else 0.5
            }
        
        return positions
    
    def _express_positions(
        self,
        positions: Dict,
        agents: List[DebateAgent]
    ) -> Dict[str, Any]:
        """表达立场"""
        expressed = {}
        
        for agent_id, position in positions.items():
            agent = next((a for a in agents if a.agent_id == agent_id), None)
            if agent:
                expressed[agent_id] = {
                    'agent_name': agent.name,
                    'role': agent.role.value,
                    'position': position,
                    'credibility': agent.credibility_score
                }
        
        return expressed
    
    def _facilitate_discussion(
        self,
        positions: Dict,
        arguments: List[Argument]
    ) -> Dict[str, Any]:
        """促进讨论"""
        # 识别分歧点
        disagreements = self._identify_disagreements(positions)
        
        # 生成讨论点
        discussion_points = []
        for disagreement in disagreements:
            discussion_points.append({
                'topic': disagreement['topic'],
                'conflicting_positions': disagreement['positions'],
                'suggested_resolution': self._suggest_resolution(disagreement)
            })
        
        return {
            'disagreements': disagreements,
            'discussion_points': discussion_points,
            'common_ground': self._find_common_ground(positions, arguments)
        }
    
    def _identify_disagreements(self, positions: Dict) -> List[Dict]:
        """识别分歧"""
        disagreements = []
        
        # 基于立场差异识别分歧
        stances = defaultdict(list)
        for agent_id, pos in positions.items():
            stances[pos['position']['stance']].append(agent_id)
        
        if len(stances) > 1:
            disagreements.append({
                'topic': 'overall_stance',
                'positions': dict(stances),
                'severity': 'high' if len(stances) > 2 else 'medium'
            })
        
        return disagreements
    
    def _suggest_resolution(self, disagreement: Dict) -> str:
        """建议解决方案"""
        if disagreement['severity'] == 'high':
            return "需要深入讨论和妥协"
        else:
            return "可以通过澄清和微调达成一致"
    
    def _find_common_ground(
        self,
        positions: Dict,
        arguments: List[Argument]
    ) -> List[str]:
        """寻找共同点"""
        common = []
        
        # 提取所有论点中的共同证据
        all_evidence = []
        for arg in arguments:
            all_evidence.extend(arg.supporting_evidence)
        
        # 找出被多方引用的证据
        evidence_count = defaultdict(int)
        for evidence in all_evidence:
            evidence_count[evidence] += 1
        
        common = [e for e, count in evidence_count.items() if count > 1]
        
        return common
    
    def _adjust_positions(
        self,
        discussion: Dict,
        agents: List[DebateAgent]
    ) -> Dict[str, Any]:
        """调整立场"""
        adjusted = {}
        
        # 基于讨论结果调整立场
        for agent in agents:
            # 简化的调整逻辑
            adjusted[agent.agent_id] = {
                'stance': agent.stance.value,
                'adjustment': 'minor_shift',
                'confidence_change': 0.05
            }
        
        return adjusted
    
    def _check_consensus(self, positions: Dict) -> Dict[str, Any]:
        """检查是否达成共识"""
        # 统计立场分布
        stance_counts = defaultdict(int)
        for agent_id, pos in positions.items():
            stance = pos.get('stance', 'neutral')
            stance_counts[stance] += 1
        
        total = len(positions)
        max_stance = max(stance_counts.items(), key=lambda x: x[1])
        agreement_rate = max_stance[1] / total
        
        consensus_reached = agreement_rate >= self.consensus_threshold
        
        return {
            'consensus_reached': consensus_reached,
            'agreement_rate': agreement_rate,
            'consensus_position': max_stance[0] if consensus_reached else None,
            'dissent': [
                stance for stance, count in stance_counts.items()
                if stance != max_stance[0]
            ] if consensus_reached else []
        }
    
    def _generate_compromise(
        self,
        positions: Dict,
        arguments: List[Argument]
    ) -> Dict[str, Any]:
        """生成妥协方案"""
        # 找到中间立场
        all_stances = [pos.get('stance', 'neutral') for pos in positions.values()]
        
        # 简化的妥协逻辑
        compromise = {
            'consensus_reached': False,
            'compromise_position': 'balanced_view',
            'supporting_arguments': [a.content for a in arguments[:3]],
            'agreement_rate': 0.5,
            'dissenting_views': list(set(all_stances)),
            'note': 'Partial consensus reached through compromise'
        }
        
        return compromise


class QualityAssessor:
    """质量评估器"""
    
    def __init__(self):
        self.quality_criteria = {
            'logical_soundness': 0.25,
            'evidence_strength': 0.25,
            'relevance': 0.2,
            'clarity': 0.15,
            'originality': 0.15
        }
        self.assessment_history: List[Dict] = []
    
    def assess_argument_quality(self, argument: Argument) -> Dict[str, Any]:
        """评估论点质量"""
        scores = {}
        
        # 逻辑合理性
        scores['logical_soundness'] = self._assess_logical_soundness(argument)
        
        # 证据强度
        scores['evidence_strength'] = self._assess_evidence_strength(argument)
        
        # 相关性
        scores['relevance'] = self._assess_relevance(argument)
        
        # 清晰度
        scores['clarity'] = self._assess_clarity(argument)
        
        # 原创性
        scores['originality'] = self._assess_originality(argument)
        
        # 计算加权总分
        total_score = sum(
            scores[criterion] * weight
            for criterion, weight in self.quality_criteria.items()
        )
        
        assessment = {
            'argument_id': argument.argument_id,
            'scores': scores,
            'total_score': total_score,
            'quality_grade': self._grade_quality(total_score),
            'strengths': self._identify_strengths(scores),
            'weaknesses': self._identify_weaknesses(scores)
        }
        
        self.assessment_history.append(assessment)
        
        return assessment
    
    def _assess_logical_soundness(self, argument: Argument) -> float:
        """评估逻辑合理性"""
        # 简化的逻辑评估
        content = argument.content.lower()
        
        # 检查逻辑连接词
        logical_markers = ['because', 'therefore', 'thus', 'hence', 'since']
        has_logical_structure = any(marker in content for marker in logical_markers)
        
        base_score = 0.7
        if has_logical_structure:
            base_score += 0.2
        
        return min(1.0, base_score)
    
    def _assess_evidence_strength(self, argument: Argument) -> float:
        """评估证据强度"""
        evidence_count = len(argument.supporting_evidence)
        
        if evidence_count >= 3:
            return 0.9
        elif evidence_count >= 2:
            return 0.7
        elif evidence_count >= 1:
            return 0.5
        else:
            return 0.3
    
    def _assess_relevance(self, argument: Argument) -> float:
        """评估相关性"""
        # 基于置信度推断相关性
        return argument.confidence
    
    def _assess_clarity(self, argument: Argument) -> float:
        """评估清晰度"""
        content = argument.content
        
        # 基于句子长度和复杂度评估
        sentences = content.split('.')
        avg_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
        
        if avg_length < 10:
            return 0.9
        elif avg_length < 20:
            return 0.7
        else:
            return 0.5
    
    def _assess_originality(self, argument: Argument) -> float:
        """评估原创性"""
        # 简化的原创性评估
        return 0.7
    
    def _grade_quality(self, score: float) -> str:
        """质量分级"""
        if score >= 0.9:
            return 'A+ (Excellent)'
        elif score >= 0.8:
            return 'A (Very Good)'
        elif score >= 0.7:
            return 'B (Good)'
        elif score >= 0.6:
            return 'C (Fair)'
        else:
            return 'D (Needs Improvement)'
    
    def _identify_strengths(self, scores: Dict) -> List[str]:
        """识别优势"""
        strengths = []
        
        for criterion, score in scores.items():
            if score >= 0.8:
                strengths.append(criterion)
        
        return strengths
    
    def _identify_weaknesses(self, scores: Dict) -> List[str]:
        """识别弱点"""
        weaknesses = []
        
        for criterion, score in scores.items():
            if score < 0.6:
                weaknesses.append(criterion)
        
        return weaknesses
    
    def select_best_answer(
        self,
        arguments: List[Argument],
        consensus_result: Dict
    ) -> Dict[str, Any]:
        """选择最佳答案"""
        # 评估所有论点
        assessed_arguments = [
            (arg, self.assess_argument_quality(arg))
            for arg in arguments
        ]
        
        # 排序
        assessed_arguments.sort(
            key=lambda x: x[1]['total_score'],
            reverse=True
        )
        
        # 选择最佳
        best_argument, best_assessment = assessed_arguments[0]
        
        return {
            'best_argument': {
                'content': best_argument.content,
                'agent': best_argument.agent_id,
                'stance': best_argument.stance.value
            },
            'quality_assessment': best_assessment,
            'consensus_info': consensus_result,
            'alternative_arguments': [
                {
                    'content': arg.content,
                    'score': assessment['total_score']
                }
                for arg, assessment in assessed_arguments[1:4]
            ]
        }


# 主引擎类
class HierarchicalMultiAgentDebateEngine:
    """层次化多智能体辩论引擎"""
    
    def __init__(self):
        self.hierarchy = HierarchicalDebateStructure()
        self.role_manager = SpecializedRoleManager()
        self.consensus_mechanism = ConsensusMechanism()
        self.quality_assessor = QualityAssessor()
        self.debate_history: List[Dict] = []
    
    def conduct_debate(
        self,
        topic: str,
        initial_arguments: List[str],
        max_rounds: int = 5
    ) -> Dict[str, Any]:
        """
        进行层次化辩论
        
        这是V127创新的主要入口函数
        """
        # 1. 创建层次化结构
        hierarchy_config = self.hierarchy.create_hierarchy(topic)
        
        # 2. 分配专业化角色
        all_agents = (
            self.hierarchy.get_agents_by_level(AgentRole.STRATEGIC) +
            self.hierarchy.get_agents_by_level(AgentRole.TACTICAL) +
            self.hierarchy.get_agents_by_level(AgentRole.OPERATIONAL)
        )
        
        specialized_agents = self.role_manager.assign_specialized_roles(
            all_agents, topic
        )
        
        # 3. 生成初始论点
        arguments = self._generate_arguments(
            initial_arguments,
            specialized_agents
        )
        
        # 4. 执行多轮辩论
        debate_rounds = self._execute_debate_rounds(
            arguments,
            specialized_agents,
            max_rounds
        )
        
        # 5. 达成共识
        consensus_result = self.consensus_mechanism.reach_consensus(
            arguments,
            specialized_agents,
            max_rounds
        )
        
        # 6. 质量评估和选择
        final_selection = self.quality_assessor.select_best_answer(
            arguments,
            consensus_result
        )
        
        result = {
            'version': 'V127',
            'name': 'Hierarchical Multi-Agent Debate (HMAD)',
            'topic': topic,
            'hierarchy': hierarchy_config,
            'debate_rounds': debate_rounds,
            'consensus': consensus_result,
            'final_selection': final_selection,
            'debate_statistics': self._generate_statistics(arguments, debate_rounds),
            'key_insights': self._extract_insights(arguments, consensus_result)
        }
        
        self.debate_history.append(result)
        
        return result
    
    def _generate_arguments(
        self,
        initial_arguments: List[str],
        agents: List[DebateAgent]
    ) -> List[Argument]:
        """生成论点"""
        arguments = []
        
        for i, content in enumerate(initial_arguments):
            agent = agents[i % len(agents)]
            
            argument = Argument(
                argument_id=f"arg_{i}",
                content=content,
                stance=agent.stance,
                supporting_evidence=[f"evidence_{i}_{j}" for j in range(3)],
                confidence=0.7 + np.random.random() * 0.2,
                agent_id=agent.agent_id,
                timestamp=datetime.now()
            )
            
            arguments.append(argument)
            agent.arguments.append(argument)
        
        return arguments
    
    def _execute_debate_rounds(
        self,
        arguments: List[Argument],
        agents: List[DebateAgent],
        max_rounds: int
    ) -> List[Dict]:
        """执行辩论轮次"""
        rounds = []
        
        phases = [
            DebatePhase.OPENING,
            DebatePhase.ARGUMENTATION,
            DebatePhase.REBUTTAL,
            DebatePhase.CROSS_EXAMINATION,
            DebatePhase.CLOSING
        ]
        
        for round_num in range(min(max_rounds, len(phases))):
            phase = phases[round_num]
            
            # 模拟辩论轮次
            round_result = {
                'round': round_num + 1,
                'phase': phase.value,
                'arguments_presented': len(arguments),
                'agent_participation': len(agents),
                'key_developments': self._simulate_debate_developments(phase)
            }
            
            rounds.append(round_result)
        
        return rounds
    
    def _simulate_debate_developments(self, phase: DebatePhase) -> List[str]:
        """模拟辩论发展"""
        developments = {
            DebatePhase.OPENING: ["各方陈述初始立场", "明确辩论焦点"],
            DebatePhase.ARGUMENTATION: ["提出核心论点", "展示证据支持"],
            DebatePhase.REBUTTAL: ["反驳对方观点", "指出逻辑漏洞"],
            DebatePhase.CROSS_EXAMINATION: ["深入质疑", "澄清误解"],
            DebatePhase.CLOSING: ["总结主要观点", "强调关键证据"]
        }
        
        return developments.get(phase, ["继续辩论"])
    
    def _generate_statistics(
        self,
        arguments: List[Argument],
        rounds: List[Dict]
    ) -> Dict[str, Any]:
        """生成统计信息"""
        return {
            'total_arguments': len(arguments),
            'total_rounds': len(rounds),
            'avg_confidence': np.mean([a.confidence for a in arguments]),
            'stance_distribution': self._compute_stance_distribution(arguments),
            'agent_participation': len(set(a.agent_id for a in arguments))
        }
    
    def _compute_stance_distribution(
        self,
        arguments: List[Argument]
    ) -> Dict[str, int]:
        """计算立场分布"""
        distribution = defaultdict(int)
        
        for arg in arguments:
            distribution[arg.stance.value] += 1
        
        return dict(distribution)
    
    def _extract_insights(
        self,
        arguments: List[Argument],
        consensus: Dict
    ) -> List[str]:
        """提取洞察"""
        insights = []
        
        if consensus.get('consensus_reached'):
            insights.append("通过层次化辩论成功达成共识")
        else:
            insights.append("存在显著分歧，但通过妥协达成部分共识")
        
        # 基于论点质量添加洞察
        avg_confidence = np.mean([a.confidence for a in arguments])
        if avg_confidence > 0.8:
            insights.append("整体论点质量较高，证据充分")
        
        return insights


# 便捷函数
def build_hierarchical_multi_agent_debate(
    topic: str,
    initial_arguments: List[str]
) -> Dict[str, Any]:
    """
    构建层次化多智能体辩论
    
    这是V127创新的便捷入口函数
    """
    engine = HierarchicalMultiAgentDebateEngine()
    return engine.conduct_debate(topic, initial_arguments)


# 示例用法
if __name__ == "__main__":
    result = build_hierarchical_multi_agent_debate(
        "Should artificial intelligence be regulated?",
        [
            "AI should be regulated to ensure safety and ethical use",
            "Over-regulation may stifle innovation and progress",
            "A balanced approach with industry standards is needed"
        ]
    )
    
    print(f"Topic: {result['topic']}")
    print(f"Total Agents: {result['hierarchy']['total_agents']}")
    print(f"Consensus Reached: {result['consensus'].get('consensus_reached', False)}")
    print(f"Best Argument: {result['final_selection']['best_argument']['content'][:100]}...")
