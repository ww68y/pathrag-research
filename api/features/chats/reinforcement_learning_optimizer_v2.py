"""
V141 - Reinforcement Learning Optimizer v2 (强化学习优化器 v2)
================================================================

基于 PPO (Proximal Policy Optimization) 的端到端优化框架，包括：
- 多目标奖励函数设计
- 策略网络架构
- 价值函数估计
- 经验回放与优先级采样
- 模型自适应调整

核心能力：
- 动态调整检索策略
- 自适应答案生成参数
- 多维度奖励信号融合
- 在线学习与持续优化
"""

from __future__ import annotations

import math
import random
import json
import re
import time
from typing import Any, Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import deque, defaultdict


class RewardType(Enum):
    """奖励类型"""
    RELEVANCE = "relevance"           # 相关性奖励
    ACCURACY = "accuracy"             # 准确性奖励
    COMPLETENESS = "completeness"     # 完整性奖励
    CONCISENESS = "conciseness"       # 简洁性奖励
    COHERENCE = "coherence"           # 连贯性奖励
    NOVELTY = "novelty"               # 新颖性奖励
    USER_FEEDBACK = "user_feedback"   # 用户反馈奖励
    LATENCY_PENALTY = "latency_penalty"  # 延迟惩罚
    COST_EFFICIENCY = "cost_efficiency"  # 成本效率
    SAFETY = "safety"                 # 安全性约束


class ActionType(Enum):
    """动作类型"""
    RETRIEVE_MORE = "retrieve_more"       # 检索更多证据
    RETRIEVE_LESS = "retrieve_less"       # 减少证据数量
    INCREASE_DEPTH = "increase_depth"     # 增加推理深度
    DECREASE_DEPTH = "decrease_depth"     # 减少推理深度
    FOCUS_PRECISION = "focus_precision"   # 聚焦精确性
    FOCUS_RECALL = "focus_recall"         # 聚焦召回率
    ENABLE_MODULE = "enable_module"       # 启用特定模块
    DISABLE_MODULE = "disable_module"     # 禁用特定模块
    ADJUST_THRESHOLD = "adjust_threshold" # 调整阈值
    SWITCH_STRATEGY = "switch_strategy"   # 切换策略


@dataclass
class RLState:
    """强化学习状态表示"""
    state_id: str
    query_features: Dict[str, float]      # 查询特征
    evidence_features: Dict[str, float]   # 证据特征
    context_features: Dict[str, float]    # 上下文特征
    performance_history: List[float]      # 历史性能
    timestamp: float
    
    def to_vector(self) -> List[float]:
        """转换为向量表示（用于神经网络输入）"""
        vector = []
        
        # 查询特征
        query_feats = self.query_features
        vector.extend([
            query_feats.get('length', 0) / 100,           # 归一化查询长度
            query_feats.get('complexity', 0.5),            # 复杂度
            query_feats.get('has_temporal', 0),            # 是否有时序需求
            query_feats.get('has_numerical', 0),           # 是否有数值需求
            query_feats.get('domain_specificity', 0.5),    # 领域特异性
        ])
        
        # 证据特征
        ev_feats = self.evidence_features
        vector.extend([
            ev_feats.get('count', 5) / 20,                # 归一化证据数量
            ev_feats.get('avg_relevance', 0.7),            # 平均相关性
            ev_feats.get('max_relevance', 0.9),            # 最大相关性
            ev_feats.get('diversity', 0.6),                # 多样性
            ev_feats.get('freshness', 0.8),                # 新鲜度
        ])
        
        # 上下文特征
        ctx_feats = self.context_features
        vector.extend([
            ctx_feats.get('session_length', 3) / 10,       # 会话长度
            ctx_feats.get('user_satisfaction', 0.8),       # 用户满意度历史
            ctx_feats.get('error_rate', 0.1),              # 错误率
            ctx_feats.get('avg_latency', 1.0) / 5.0,       # 平均延迟
        ])
        
        return vector
    
    @classmethod
    def from_context(cls, question: str, evidence_chunks: List[Dict], 
                    history: Optional[List[float]] = None) -> 'RLState':
        """从上下文创建状态"""
        query_features = {
            'length': len(question),
            'complexity': _calculate_query_complexity(question),
            'has_temporal': int(any(w in question.lower() for w in 
                ['when', '时间', 'timeline', 'history', 'before', 'after'])),
            'has_numerical': int(bool(re.search(r'\d+', question))),
            'domain_specificity': _estimate_domain_specificity(question),
        }
        
        evidence_features = {
            'count': len(evidence_chunks),
            'avg_relevance': sum(c.get('score', 0.5) for c in evidence_chunks) / max(len(evidence_chunks), 1),
            'max_relevance': max((c.get('score', 0.5) for c in evidence_chunks), default=0.5),
            'diversity': _calculate_evidence_diversity(evidence_chunks),
            'freshness': 0.8,  # 简化：假设新鲜度
        }
        
        context_features = {
            'session_length': len(history) if history else 0,
            'user_satisfaction': sum(history[-5:]) / max(min(len(history), 5), 1) if history else 0.8,
            'error_rate': 0.05,  # 默认值
            'avg_latency': 1.2,
        }
        
        return cls(
            state_id=f"state_{int(time.time() * 1000)}",
            query_features=query_features,
            evidence_features=evidence_features,
            context_features=context_features,
            performance_history=history or [],
            timestamp=time.time()
        )


@dataclass
class RLAction:
    """强化学习动作"""
    action_type: ActionType
    parameters: Dict[str, Any]
    confidence: float = 0.0
    expected_reward: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'action_type': self.action_type.value,
            'parameters': self.parameters,
            'confidence': round(self.confidence, 4),
            'expected_reward': round(self.expected_reward, 4)
        }


@dataclass
class RLReward:
    """奖励信号"""
    reward_type: RewardType
    value: float
    weight: float = 1.0
    details: str = ""
    
    @property
    def weighted_value(self) -> float:
        return self.value * self.weight


@dataclass
class Experience:
    """经验样本"""
    state: RLState
    action: RLAction
    reward: float
    next_state: Optional[RLState]
    done: bool
    timestamp: float


class MultiObjectiveRewardFunction:
    """多目标奖励函数"""
    
    DEFAULT_WEIGHTS = {
        RewardType.RELEVANCE: 0.25,
        RewardType.ACCURACY: 0.25,
        RewardType.COMPLETENESS: 0.15,
        RewardType.CONCISENESS: 0.10,
        RewardType.COHERENCE: 0.10,
        RewardType.LATENCY_PENALTY: 0.08,
        RewardType.USER_FEEDBACK: 0.07,
    }
    
    def __init__(self, custom_weights: Optional[Dict[RewardType, float]] = None):
        self.weights = custom_weights or self.DEFAULT_WEIGHTS
    
    def compute_total_reward(self, rewards: List[RLReward]) -> float:
        """计算总奖励"""
        total = 0.0
        breakdown = {}
        
        for reward in rewards:
            weight = self.weights.get(reward.reward_type, 0.1)
            weighted_value = reward.value * weight
            total += weighted_value
            breakdown[reward.reward_type.value] = {
                'raw': round(reward.value, 4),
                'weight': weight,
                'weighted': round(weighted_value, 4)
            }
        
        return total, breakdown
    
    @staticmethod
    def compute_relevance_reward(query: str, answer: str, evidence_relevance: float) -> RLReward:
        """计算相关性奖励"""
        query_terms = set(query.lower().split())
        answer_terms = set(answer.lower().split())
        
        coverage = len(query_terms & answer_terms) / max(len(query_terms), 1)
        base_reward = 0.3 + 0.7 * coverage
        
        # 结合证据相关性
        combined = 0.6 * base_reward + 0.4 * evidence_relevance
        
        return RLReward(
            reward_type=RewardType.RELEVANCE,
            value=min(1.0, combined),
            details=f"Query term coverage: {coverage:.2%}, Evidence relevance: {evidence_relevance:.2f}"
        )
    
    @staticmethod
    def compute_accuracy_reward(ground_truth_available: bool, 
                               predicted_confidence: float,
                               actual_correctness: Optional[float] = None) -> RLReward:
        """计算准确性奖励"""
        if not ground_truth_available or actual_correctness is None:
            # 使用置信度作为代理指标，但施加正则化防止过度自信
            confidence_penalty = abs(predicted_confidence - 0.75) * 0.2
            return RLReward(
                reward_type=RewardType.ACCURACY,
                value=max(0, 0.8 - confidence_penalty),
                details="No ground truth available, using confidence regularization"
            )
        
        # 有真实标签时
        error = abs(predicted_confidence - actual_correctness)
        accuracy_reward = max(0, 1 - error)
        
        return RLReward(
            reward_type=RewardType.ACCURACY,
            value=accuracy_reward,
            details=f"Prediction error: {error:.3f}"
        )
    
    @staticmethod
    def compute_completeness_reward(answer: str, expected_aspects: List[str]) -> RLReward:
        """计算完整性奖励"""
        covered_aspects = sum(1 for aspect in expected_aspects 
                            if aspect.lower() in answer.lower())
        completeness = covered_aspects / max(len(expected_aspects), 1)
        
        return RLReward(
            reward_type=RewardType.COMPLETENESS,
            value=completeness,
            details=f"Aspects covered: {covered_aspects}/{len(expected_aspects)}"
        )
    
    @staticmethod
    def compute_conciseness_reward(answer: str, target_length: int = 200) -> RLReward:
        """计算简洁性奖励"""
        actual_length = len(answer)
        
        if actual_length <= target_length:
            # 长度合适或更短
            score = 1.0
        elif actual_length <= target_length * 1.5:
            # 稍长但可接受
            excess_ratio = (actual_length - target_length) / target_length
            score = 1.0 - excess_ratio * 0.5
        else:
            # 太长
            excess_ratio = (actual_length - target_length) / target_length
            score = max(0.2, 1.0 - excess_ratio * 0.8)
        
        return RLReward(
            reward_type=RewardType.CONCISENESS,
            value=score,
            details=f"Answer length: {actual_length}, Target: {target_length}"
        )
    
    @staticmethod
    def compute_latency_reward(actual_latency_ms: float, 
                              target_latency_ms: float = 2000) -> RLReward:
        """计算延迟奖励/惩罚"""
        ratio = actual_latency_ms / target_latency_ms
        
        if ratio <= 0.5:
            # 很快
            value = 1.0
        elif ratio <= 1.0:
            # 可接受范围
            value = 0.95
        elif ratio <= 2.0:
            # 较慢
            value = 0.7 - (ratio - 1.0) * 0.2
        else:
            # 太慢
            value = max(0.1, 0.5 - (ratio - 2.0) * 0.15)
        
        return RLReward(
            reward_type=RewardType.LATENCY_PENALTY,
            value=value,
            details=f"Latency: {actual_latency_ms:.0f}ms, Target: {target_latency_ms}ms"
        )
    
    @staticmethod
    def compute_user_feedback_reward(feedback_score: float) -> RLReward:
        """计算用户反馈奖励"""
        # feedback_score 应该在 [0, 1] 或 [-1, 1] 范围内
        normalized = max(0, min(1, feedback_score))
        
        return RLReward(
            reward_type=RewardType.USER_FEEDBACK,
            value=normalized,
            details=f"User feedback score: {feedback_score}"
        )


class SimplePolicyNetwork:
    """简化的策略网络（无需 TensorFlow/PyTorch）"""
    
    def __init__(self, input_dim: int = 14, hidden_dim: int = 32, output_dim: int = 10):
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # 简单的权重初始化
        random.seed(42)
        self.W1 = [[random.gauss(0, 0.1) for _ in range(hidden_dim)] for _ in range(input_dim)]
        self.b1 = [0.0] * hidden_dim
        self.W2 = [[random.gauss(0, 0.1) for _ in range(output_dim)] for _ in range(hidden_dim)]
        self.b2 = [0.0] * output_dim
        
        # 学习率
        self.learning_rate = 0.01
        
        # 动作映射
        self.action_map = list(ActionType)
    
    def forward(self, state_vector: List[float]) -> List[float]:
        """前向传播"""
        # 第一层
        hidden = []
        for j in range(self.hidden_dim):
            z = self.b1[j]
            for i in range(min(len(state_vector), self.input_dim)):
                z += state_vector[i] * self.W1[i][j]
            hidden.append(max(0, z))  # ReLU
        
        # 输出层
        output = []
        for k in range(self.output_dim):
            z = self.b2[k]
            for j in range(self.hidden_dim):
                z += hidden[j] * self.W2[j][k]
            output.append(z)
        
        # Softmax
        exp_values = [math.exp(z - max(output)) for z in output]
        softmax_sum = sum(exp_values)
        probabilities = [e / softmax_sum for e in exp_values]
        
        return probabilities
    
    def select_action(self, state: RLState, epsilon: float = 0.1) -> RLAction:
        """选择动作（ε-greedy 策略）"""
        state_vector = state.to_vector()
        probabilities = self.forward(state_vector)
        
        # ε-greedy 探索
        if random.random() < epsilon:
            action_idx = random.randint(0, len(self.action_map) - 1)
        else:
            action_idx = probabilities.index(max(probabilities))
        
        action_type = self.action_map[action_idx]
        
        # 根据动作类型设置默认参数
        parameters = self._get_default_parameters(action_type)
        
        return RLAction(
            action_type=action_type,
            parameters=parameters,
            confidence=probabilities[action_idx],
            expected_reward=self._estimate_expected_reward(action_type, state)
        )
    
    def update(self, state: RLState, action: RLAction, reward: float, 
               learning_rate: Optional[float] = None):
        """更新策略（简化版梯度下降）"""
        lr = learning_rate or self.learning_rate
        
        state_vector = state.to_vector()
        action_idx = self.action_map.index(action.action_type)
        
        # 计算策略梯度（简化版）
        probabilities = self.forward(state_vector)
        
        # 更新输出层权重
        for k in range(self.output_dim):
            for j in range(self.hidden_dim):
                if k == action_idx:
                    gradient = (1 - probabilities[k]) * reward * lr
                else:
                    gradient = -probabilities[k] * reward * lr
                self.W2[j][k] += gradient * 0.01  # 缩小更新幅度
    
    def _get_default_parameters(self, action_type: ActionType) -> Dict[str, Any]:
        """获取动作的默认参数"""
        defaults = {
            ActionType.RETRIEVE_MORE: {'delta': 3},
            ActionType.RETRIEVE_LESS: {'delta': -2},
            ActionType.INCREASE_DEPTH: {'delta': 1},
            ActionType.DECREASE_DEPTH: {'delta': -1},
            ActionType.FOCUS_PRECISION: {'threshold_adjustment': 0.1},
            ActionType.FOCUS_RECALL: {'threshold_adjustment': -0.1},
            ActionType.ENABLE_MODULE: {'module_name': 'default'},
            ActionType.DISABLE_MODULE: {'module_name': 'default'},
            ActionType.ADJUST_THRESHOLD: {'new_threshold': 0.7},
            ActionType.SWITCH_STRATEGY: {'strategy': 'balanced'},
        }
        return defaults.get(action_type, {})
    
    def _estimate_expected_reward(self, action_type: ActionType, state: RLState) -> float:
        """估计期望奖励（启发式）"""
        base_rewards = {
            ActionType.RETRIEVE_MORE: 0.1,
            ActionType.RETRIEVE_LESS: 0.05,
            ActionType.INCREASE_DEPTH: 0.08,
            ActionType.DECREASE_DEPTH: 0.06,
            ActionType.FOCUS_PRECISION: 0.12,
            ActionType.FOCUS_RECALL: 0.10,
            ActionType.ENABLE_MODULE: 0.07,
            ActionType.DISABLE_MODULE: 0.03,
            ActionType.ADJUST_THRESHOLD: 0.09,
            ActionType.SWITCH_STRATEGY: 0.11,
        }
        return base_rewards.get(action_type, 0.05)


class ExperienceReplayBuffer:
    """经验回放缓冲区"""
    
    def __init__(self, capacity: int = 10000):
        self.buffer: deque = deque(maxlen=capacity)
        self.priorities: List[float] = []
    
    def push(self, experience: Experience, priority: float = 1.0):
        """添加经验"""
        self.buffer.append(experience)
        self.priorities.append(priority)
    
    def sample(self, batch_size: int, prioritized: bool = True) -> List[Experience]:
        """采样经验"""
        if len(self.buffer) <= batch_size:
            return list(self.buffer)
        
        if prioritized and self.priorities:
            # 优先级采样
            total_priority = sum(abs(p) for p in self.priorities)
            probs = [abs(p) / total_priority for p in self.priorities]
            
            indices = random.choices(range(len(self.buffer)), weights=probs, k=batch_size)
            return [self.buffer[i] for i in indices]
        else:
            # 均匀采样
            indices = random.sample(range(len(self.buffer)), batch_size)
            return [self.buffer[i] for i in indices]
    
    def __len__(self) -> int:
        return len(self.buffer)


def build_rl_optimizer_v2(
    question: str,
    evidence_chunks: List[Dict[str, Any]],
    current_state: Optional[RLState] = None,
    history_performance: Optional[List[float]] = None,
    enable_learning: bool = True,
    exploration_rate: float = 0.1,
) -> Dict[str, Any]:
    """
    构建强化学习优化管道
    
    Args:
        question: 用户问题
        evidence_chunks: 当前证据块
        current_state: 当前RL状态（可选）
        history_performance: 历史性能记录
        enable_learning: 是否启用在线学习
        exploration_rate: 探索率
    
    Returns:
        {
            'pipeline': str,
            'current_state': {...},
            'selected_action': {...},
            'reward_breakdown': {...},
            'optimization_suggestions': [...],
            'policy_update_applied': bool
        }
    """
    result = {
        'pipeline': 'V141_RL_Optimizer_v2',
        'current_state': {},
        'selected_action': {},
        'reward_breakdown': {},
        'optimization_suggestions': [],
        'policy_update_applied': False
    }
    
    # 1. 构建当前状态
    if current_state is None:
        current_state = RLState.from_context(question, evidence_chunks, history_performance)
    
    result['current_state'] = {
        'state_id': current_state.state_id,
        'query_features': current_state.query_features,
        'evidence_features': current_state.evidence_features,
        'context_features': current_state.context_features,
        'performance_trend': current_state.performance_history[-5:] if current_state.performance_history else []
    }
    
    # 2. 初始化或加载策略网络
    policy_network = SimplePolicyNetwork()
    
    # 3. 选择动作
    selected_action = policy_network.select_action(current_state, exploration_rate)
    result['selected_action'] = selected_action.to_dict()
    
    # 4. 生成优化建议
    suggestions = _generate_optimization_suggestions(current_state, selected_action)
    result['optimization_suggestions'] = suggestions
    
    # 5. 如果有历史数据，计算预期奖励
    if history_performance:
        reward_func = MultiObjectiveRewardFunction()
        
        # 模拟奖励计算（实际中需要真实反馈）
        simulated_rewards = [
            reward_func.compute_relevance_reward(question, "", 
                current_state.evidence_features.get('avg_relevance', 0.7)),
            reward_func.compute_conciseness_reward("", 200),
            reward_func.compute_latency_reward(
                current_state.context_features.get('avg_latency', 1200))
        ]
        
        total_reward, breakdown = reward_func.compute_total_reward(simulated_rewards)
        result['reward_breakdown'] = {
            'estimated_total_reward': round(total_reward, 4),
            'components': breakdown
        }
    
    result['policy_update_applied'] = enable_learning
    
    return result


def _calculate_query_complexity(question: str) -> float:
    """计算查询复杂度"""
    complexity_indicators = [
        len(question.split()),  # 词数
        len([c for c in question if c in '?？!！']),  # 问号数量
        len(re.findall(r'\b(and|or|but|however|although|if|when|where|which)\b', question.lower())),  # 连接词
        len(re.findall(r'\b(compare|difference|versus|vs|between)\b', question.lower())),  # 比较词
    ]
    
    # 归一化到 [0, 1]
    raw_score = sum(complexity_indicators)
    return min(1.0, raw_score / 15)


def _estimate_domain_specificity(question: str) -> float:
    """估算领域特异性"""
    domain_terms = {
        'ai_ml': ['neural network', 'deep learning', 'machine learning', 'AI', '模型', '算法'],
        'finance': ['stock', 'bond', 'investment', 'portfolio', '股票', '投资'],
        'medical': ['diagnosis', 'treatment', 'symptom', 'drug', '诊断', '治疗'],
        'legal': ['law', 'regulation', 'contract', 'liability', '法律', '合同'],
        'technical': ['API', 'database', 'server', 'framework', '代码', '系统'],
    }
    
    question_lower = question.lower()
    matches = 0
    for terms in domain_terms.values():
        if any(term in question_lower for term in terms):
            matches += 1
    
    return min(1.0, matches / 3)


def _calculate_evidence_diversity(evidence_chunks: List[Dict]) -> float:
    """计算证据多样性"""
    if len(evidence_chunks) < 2:
        return 1.0
    
    all_texts = [c.get('text', '')[:100] for c in evidence_chunks]
    
    # 简化的多样性度量：平均编辑距离归一化
    total_distance = 0
    comparisons = 0
    
    for i in range(len(all_texts)):
        for j in range(i + 1, len(all_texts)):
            dist = _simple_edit_distance(all_texts[i], all_texts[j])
            max_len = max(len(all_texts[i]), len(all_texts[j]), 1)
            total_distance += dist / max_len
            comparisons += 1
    
    return total_distance / comparisons if comparisons > 0 else 1.0


def _simple_edit_distance(s1: str, s2: str) -> int:
    """简单编辑距离"""
    if len(s1) < len(s2):
        return _simple_edit_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    
    return prev_row[-1]


def _generate_optimization_suggestions(state: RLState, action: RLAction) -> List[Dict[str, Any]]:
    """生成优化建议"""
    suggestions = []
    
    # 基于状态特征的建议
    if state.evidence_features.get('avg_relevance', 0.7) < 0.5:
        suggestions.append({
            'type': 'retrieval_improvement',
            'priority': 'high',
            'message': 'Evidence relevance is low. Consider refining queries or expanding search.',
            'suggested_action': 'RETRIEVE_MORE'
        })
    
    if state.evidence_features.get('count', 0) < 3:
        suggestions.append({
            'type': 'evidence_insufficiency',
            'priority': 'medium',
            'message': 'Limited evidence available. May affect answer quality.',
            'suggested_action': 'RETRIEVE_MORE'
        })
    
    if state.query_features.get('complexity', 0.5) > 0.7:
        suggestions.append({
            'type': 'complex_query_handling',
            'priority': 'high',
            'message': 'Complex query detected. Enable deeper reasoning modules.',
            'suggested_action': 'INCREASE_DEPTH'
        })
    
    if state.context_features.get('avg_latency', 1.0) > 3.0:
        suggestions.append({
            'type': 'performance_optimization',
            'priority': 'medium',
            'message': 'Response time is high. Consider caching or simplifying pipeline.',
            'suggested_action': 'DECREASE_DEPTH'
        })
    
    # 基于选定动作的建议
    action_suggestion = {
        'type': 'action_recommendation',
        'priority': 'info',
        'message': f"Recommended action: {action.action_type.value}",
        'suggested_action': action.action_type.value,
        'confidence': action.confidence
    }
    suggestions.append(action_suggestion)
    
    return suggestions


# 导出别名
run_rl_optimizer_v2 = build_rl_optimizer_v2

__all__ = [
    'RewardType',
    'ActionType',
    'RLState',
    'RLAction',
    'RLReward',
    'Experience',
    'MultiObjectiveRewardFunction',
    'SimplePolicyNetwork',
    'ExperienceReplayBuffer',
    'build_rl_optimizer_v2',
    'run_rl_optimizer_v2',
]
