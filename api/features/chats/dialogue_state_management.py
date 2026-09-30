"""
V117: Dialogue State Management & Multi-turn Reasoning (DSM)
对话状态管理与多轮推理 - 维护对话上下文，支持复杂多轮交互

创新点：
1. 对话状态追踪：维护对话的完整状态
2. 意图演化检测：追踪用户意图的变化
3. 多轮推理链：支持跨多轮的复杂推理
4. 对话恢复：从错误中恢复对话
"""

import json
import hashlib
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from collections import defaultdict


class DialogueState(Enum):
    """对话状态"""
    GREETING = "greeting"
    INFORMATION_GATHERING = "information_gathering"
    REASONING = "reasoning"
    CLARIFICATION = "clarification"
    CONFIRMATION = "confirmation"
    CONCLUSION = "conclusion"
    ERROR_RECOVERY = "error_recovery"
    IDLE = "idle"


class IntentType(Enum):
    """意图类型"""
    QUERY = "query"
    CLARIFY = "clarify"
    CONFIRM = "confirm"
    REJECT = "reject"
    EXPAND = "expand"
    SHIFT = "shift"
    REPEAT = "repeat"
    GREET = "greet"
    FAREWELL = "farewell"


@dataclass
class Turn:
    """对话轮次"""
    turn_id: str
    user_utterance: str
    system_response: str
    timestamp: datetime
    state: DialogueState
    intent: IntentType
    entities: Dict[str, Any] = field(default_factory=dict)
    context_updates: Dict[str, Any] = field(default_factory=dict)
    reasoning_chain: List[str] = field(default_factory=list)


@dataclass
class DialogueContext:
    """对话上下文"""
    session_id: str
    current_state: DialogueState
    state_history: List[DialogueState] = field(default_factory=list)
    turns: List[Turn] = field(default_factory=list)
    entity_memory: Dict[str, Any] = field(default_factory=dict)
    intent_history: List[IntentType] = field(default_factory=list)
    pending_clarifications: List[str] = field(default_factory=list)
    completed_goals: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_updated: datetime = field(default_factory=datetime.now)


@dataclass
class ReasoningNode:
    """推理节点"""
    node_id: str
    premise: str
    conclusion: str
    evidence: List[str]
    confidence: float
    dependencies: List[str] = field(default_factory=list)
    sub_nodes: List['ReasoningNode'] = field(default_factory=list)


class StateTracker:
    """状态追踪器"""
    
    def __init__(self):
        self.active_sessions: Dict[str, DialogueContext] = {}
        self.state_transitions: Dict[Tuple[DialogueState, IntentType], DialogueState] = self._init_transitions()
    
    def _init_transitions(self) -> Dict[Tuple[DialogueState, IntentType], DialogueState]:
        """初始化状态转移规则"""
        return {
            # 从问候状态
            (DialogueState.GREETING, IntentType.QUERY): DialogueState.INFORMATION_GATHERING,
            (DialogueState.GREETING, IntentType.GREET): DialogueState.GREETING,
            
            # 从信息收集状态
            (DialogueState.INFORMATION_GATHERING, IntentType.QUERY): DialogueState.REASONING,
            (DialogueState.INFORMATION_GATHERING, IntentType.CLARIFY): DialogueState.CLARIFICATION,
            
            # 从推理状态
            (DialogueState.REASONING, IntentType.CONFIRM): DialogueState.CONCLUSION,
            (DialogueState.REASONING, IntentType.CLARIFY): DialogueState.CLARIFICATION,
            (DialogueState.REASONING, IntentType.EXPAND): DialogueState.REASONING,
            (DialogueState.REASONING, IntentType.REJECT): DialogueState.ERROR_RECOVERY,
            
            # 从澄清状态
            (DialogueState.CLARIFICATION, IntentType.CONFIRM): DialogueState.REASONING,
            (DialogueState.CLARIFICATION, IntentType.QUERY): DialogueState.REASONING,
            
            # 从结论状态
            (DialogueState.CONCLUSION, IntentType.QUERY): DialogueState.INFORMATION_GATHERING,
            (DialogueState.CONCLUSION, IntentType.EXPAND): DialogueState.REASONING,
            (DialogueState.CONCLUSION, IntentType.FAREWELL): DialogueState.IDLE,
            
            # 从错误恢复状态
            (DialogueState.ERROR_RECOVERY, IntentType.QUERY): DialogueState.INFORMATION_GATHERING,
            (DialogueState.ERROR_RECOVERY, IntentType.REPEAT): DialogueState.REASONING,
        }
    
    def create_session(self, session_id: Optional[str] = None) -> DialogueContext:
        """创建新对话会话"""
        if not session_id:
            session_id = hashlib.md5(
                str(datetime.now().timestamp()).encode()
            ).hexdigest()[:12]
        
        context = DialogueContext(
            session_id=session_id,
            current_state=DialogueState.GREETING
        )
        
        self.active_sessions[session_id] = context
        return context
    
    def get_session(self, session_id: str) -> Optional[DialogueContext]:
        """获取对话会话"""
        return self.active_sessions.get(session_id)
    
    def update_state(
        self,
        session_id: str,
        user_intent: IntentType,
        utterance: str,
        system_response: str,
        entities: Optional[Dict] = None
    ) -> Tuple[DialogueState, DialogueContext]:
        """
        更新对话状态
        
        Args:
            session_id: 会话ID
            user_intent: 用户意图
            utterance: 用户话语
            system_response: 系统响应
            entities: 提取的实体
        
        Returns:
            (新状态, 更新后的上下文)
        """
        context = self.get_session(session_id)
        if not context:
            context = self.create_session(session_id)
        
        current_state = context.current_state
        
        # 确定新状态
        transition_key = (current_state, user_intent)
        new_state = self.state_transitions.get(
            transition_key,
            current_state  # 默认保持当前状态
        )
        
        # 创建新的对话轮次
        turn = Turn(
            turn_id=f"{session_id}_{len(context.turns)}",
            user_utterance=utterance,
            system_response=system_response,
            timestamp=datetime.now(),
            state=new_state,
            intent=user_intent,
            entities=entities or {}
        )
        
        # 更新上下文
        context.turns.append(turn)
        context.state_history.append(current_state)
        context.current_state = new_state
        context.intent_history.append(user_intent)
        context.last_updated = datetime.now()
        
        # 更新实体记忆
        if entities:
            context.entity_memory.update(entities)
        
        return new_state, context
    
    def get_state_summary(self, session_id: str) -> Dict[str, Any]:
        """获取对话状态摘要"""
        context = self.get_session(session_id)
        if not context:
            return {'error': 'Session not found'}
        
        return {
            'session_id': session_id,
            'current_state': context.current_state.value,
            'turn_count': len(context.turns),
            'duration_seconds': (
                datetime.now() - context.created_at
            ).total_seconds(),
            'state_history': [s.value for s in context.state_history],
            'intent_history': [i.value for i in context.intent_history],
            'entity_memory': context.entity_memory,
            'completed_goals': context.completed_goals,
            'pending_clarifications': context.pending_clarifications
        }


class IntentEvolutionDetector:
    """意图演化检测器"""
    
    def __init__(self):
        self.intent_patterns: Dict[str, List[IntentType]] = {
            'deepening': [IntentType.QUERY, IntentType.EXPAND, IntentType.EXPAND],
            'clarifying': [IntentType.QUERY, IntentType.CLARIFY, IntentType.CONFIRM],
            'exploring': [IntentType.QUERY, IntentType.SHIFT, IntentType.QUERY],
            'confirming': [IntentType.QUERY, IntentType.CONFIRM, IntentType.CONFIRM],
            'rejecting': [IntentType.QUERY, IntentType.REJECT, IntentType.QUERY]
        }
    
    def detect_intent_evolution(
        self,
        intent_history: List[IntentType]
    ) -> Dict[str, Any]:
        """
        检测意图演化模式
        
        Args:
            intent_history: 意图历史
        
        Returns:
            演化分析结果
        """
        if len(intent_history) < 2:
            return {
                'evolution_detected': False,
                'pattern': 'insufficient_data'
            }
        
        # 检查匹配的模式
        matched_patterns = []
        for pattern_name, pattern_sequence in self.intent_patterns.items():
            if self._sequence_matches(intent_history, pattern_sequence):
                matched_patterns.append(pattern_name)
        
        # 分析意图变化
        recent_intents = intent_history[-3:]
        intent_shifts = sum(
            1 for i in range(len(recent_intents) - 1)
            if recent_intents[i] != recent_intents[i + 1]
        )
        
        # 检测意图漂移
        intent_drift = self._detect_intent_drift(intent_history)
        
        return {
            'evolution_detected': len(matched_patterns) > 0 or intent_shifts > 0,
            'matched_patterns': matched_patterns,
            'recent_intent_shifts': intent_shifts,
            'intent_stability': 1 - (intent_shifts / max(len(recent_intents) - 1, 1)),
            'intent_drift': intent_drift,
            'evolution_summary': self._summarize_evolution(
                intent_history, matched_patterns
            )
        }
    
    def _sequence_matches(
        self,
        history: List[IntentType],
        pattern: List[IntentType]
    ) -> bool:
        """检查序列是否匹配模式"""
        if len(history) < len(pattern):
            return False
        
        # 检查最近的意图是否匹配模式
        recent = history[-len(pattern):]
        return all(r == p for r, p in zip(recent, pattern))
    
    def _detect_intent_drift(
        self,
        intent_history: List[IntentType]
    ) -> Dict[str, Any]:
        """检测意图漂移"""
        if len(intent_history) < 5:
            return {'drift_detected': False}
        
        # 比较前半段和后半段的主要意图
        mid = len(intent_history) // 2
        first_half = intent_history[:mid]
        second_half = intent_history[mid:]
        
        first_main = max(set(first_half), key=first_half.count)
        second_main = max(set(second_half), key=second_half.count)
        
        drift_detected = first_main != second_main
        
        return {
            'drift_detected': drift_detected,
            'from_intent': first_main.value if drift_detected else None,
            'to_intent': second_main.value if drift_detected else None,
            'drift_severity': 'high' if drift_detected else 'none'
        }
    
    def _summarize_evolution(
        self,
        intent_history: List[IntentType],
        patterns: List[str]
    ) -> str:
        """总结意图演化"""
        if not patterns:
            if len(set(intent_history)) == 1:
                return f"User consistently shows {intent_history[0].value} intent"
            else:
                return "User intent shows mixed patterns"
        
        if 'deepening' in patterns:
            return "User is progressively deepening their inquiry"
        elif 'clarifying' in patterns:
            return "User is seeking clarification and confirmation"
        elif 'exploring' in patterns:
            return "User is exploring different aspects of the topic"
        else:
            return f"Detected patterns: {', '.join(patterns)}"


class MultiTurnReasoningEngine:
    """多轮推理引擎"""
    
    def __init__(self):
        self.reasoning_chains: Dict[str, List[ReasoningNode]] = {}
        self.incomplete_reasonings: Dict[str, ReasoningNode] = {}
    
    def build_reasoning_chain(
        self,
        session_id: str,
        turns: List[Turn]
    ) -> List[ReasoningNode]:
        """
        构建跨多轮的推理链
        
        Args:
            session_id: 会话ID
            turns: 对话轮次列表
        
        Returns:
            推理节点链
        """
        chain = []
        
        for i, turn in enumerate(turns):
            # 提取推理步骤
            if turn.reasoning_chain:
                for step in turn.reasoning_chain:
                    node = ReasoningNode(
                        node_id=f"{session_id}_node_{i}_{len(chain)}",
                        premise=turn.user_utterance,
                        conclusion=step,
                        evidence=[turn.system_response],
                        confidence=0.8  # 默认置信度
                    )
                    
                    # 连接依赖关系
                    if chain:
                        node.dependencies.append(chain[-1].node_id)
                    
                    chain.append(node)
        
        self.reasoning_chains[session_id] = chain
        return chain
    
    def continue_reasoning(
        self,
        session_id: str,
        current_node_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        继续未完成的推理
        
        Args:
            session_id: 会话ID
            current_node_id: 当前节点ID（可选）
        
        Returns:
            继续推理的建议
        """
        chain = self.reasoning_chains.get(session_id, [])
        
        if not chain:
            return {
                'can_continue': False,
                'reason': 'No existing reasoning chain'
            }
        
        # 找到当前节点
        if current_node_id:
            current_idx = next(
                (i for i, node in enumerate(chain) if node.node_id == current_node_id),
                len(chain) - 1
            )
        else:
            current_idx = len(chain) - 1
        
        current_node = chain[current_idx]
        
        # 检查是否有子节点
        if current_node.sub_nodes:
            return {
                'can_continue': True,
                'next_steps': [
                    {'node_id': sub.node_id, 'conclusion': sub.conclusion}
                    for sub in current_node.sub_nodes
                ],
                'suggestion': 'Explore sub-reasoning paths'
            }
        
        # 检查是否有后续节点
        if current_idx < len(chain) - 1:
            next_node = chain[current_idx + 1]
            return {
                'can_continue': True,
                'next_step': {
                    'node_id': next_node.node_id,
                    'conclusion': next_node.conclusion
                },
                'suggestion': 'Continue to next reasoning step'
            }
        
        # 推理链完成
        return {
            'can_continue': False,
            'reason': 'Reasoning chain is complete',
            'summary': self._summarize_reasoning_chain(chain)
        }
    
    def _summarize_reasoning_chain(self, chain: List[ReasoningNode]) -> Dict[str, Any]:
        """总结推理链"""
        return {
            'total_steps': len(chain),
            'average_confidence': sum(n.confidence for n in chain) / len(chain) if chain else 0,
            'key_conclusions': [n.conclusion for n in chain[-3:]],  # 最近3个结论
            'reasoning_depth': max(
                len(node.dependencies) for node in chain
            ) if chain else 0
        }
    
    def merge_parallel_reasonings(
        self,
        session_id: str,
        reasoning_branches: List[List[ReasoningNode]]
    ) -> List[ReasoningNode]:
        """合并并行推理分支"""
        # 简化的合并：取所有分支的并集
        merged = []
        seen_conclusions = set()
        
        for branch in reasoning_branches:
            for node in branch:
                if node.conclusion not in seen_conclusions:
                    merged.append(node)
                    seen_conclusions.add(node.conclusion)
        
        # 按置信度排序
        merged.sort(key=lambda n: n.confidence, reverse=True)
        
        self.reasoning_chains[session_id] = merged
        return merged


class DialogueRecoveryEngine:
    """对话恢复引擎"""
    
    def __init__(self):
        self.error_patterns: Dict[str, str] = {
            'misunderstanding': 'clarify_intent',
            'irrelevant_response': 'redirect',
            'technical_error': 'apologize_and_retry',
            'user_frustration': 'empathize_and_escalate'
        }
    
    def detect_error(
        self,
        context: DialogueContext,
        last_turn: Turn
    ) -> Dict[str, Any]:
        """
        检测对话错误
        
        Args:
            context: 对话上下文
            last_turn: 最后一轮对话
        
        Returns:
            错误检测结果
        """
        errors = []
        
        # 检查是否重复相同响应
        if len(context.turns) >= 2:
            prev_turn = context.turns[-2]
            if prev_turn.system_response == last_turn.system_response:
                errors.append({
                    'type': 'repetition',
                    'severity': 'medium',
                    'description': 'System is repeating the same response'
                })
        
        # 检查用户是否表达了困惑
        confusion_indicators = ['confused', 'don\'t understand', 'unclear', '不明白', '困惑']
        if any(indicator in last_turn.user_utterance.lower() for indicator in confusion_indicators):
            errors.append({
                'type': 'misunderstanding',
                'severity': 'high',
                'description': 'User expressed confusion'
            })
        
        # 检查状态循环
        if len(context.state_history) >= 3:
            recent_states = context.state_history[-3:]
            if len(set(recent_states)) == 1:
                errors.append({
                    'type': 'state_loop',
                    'severity': 'medium',
                    'description': 'Dialogue is stuck in the same state'
                })
        
        # 检查意图快速变化
        if len(context.intent_history) >= 3:
            recent_intents = context.intent_history[-3:]
            if len(set(recent_intents)) == 3:  # 三个不同的意图
                errors.append({
                    'type': 'intent_instability',
                    'severity': 'low',
                    'description': 'User intent is rapidly changing'
                })
        
        return {
            'errors_detected': len(errors) > 0,
            'error_count': len(errors),
            'errors': errors,
            'recovery_recommended': any(e['severity'] in ['high', 'medium'] for e in errors)
        }
    
    def generate_recovery_strategy(
        self,
        error_type: str,
        context: DialogueContext
    ) -> Dict[str, Any]:
        """生成恢复策略"""
        strategies = {
            'misunderstanding': {
                'action': 'clarify_intent',
                'response_template': "I apologize for the confusion. Let me rephrase. Are you asking about {topic}?",
                'next_state': DialogueState.CLARIFICATION,
                'steps': [
                    'Acknowledge confusion',
                    'Ask clarifying question',
                    'Provide alternative interpretations'
                ]
            },
            'repetition': {
                'action': 'change_approach',
                'response_template': "Let me try a different approach to help you with {topic}.",
                'next_state': DialogueState.ERROR_RECOVERY,
                'steps': [
                    'Acknowledge repetition',
                    'Offer alternative solution',
                    'Escalate if needed'
                ]
            },
            'state_loop': {
                'action': 'force_transition',
                'response_template': "I notice we've been going in circles. Let me summarize what we've discussed so far.",
                'next_state': DialogueState.CONCLUSION,
                'steps': [
                    'Summarize progress',
                    'Identify stuck point',
                    'Propose way forward'
                ]
            },
            'intent_instability': {
                'action': 'refocus',
                'response_template': "I want to make sure I help you effectively. What would you like to focus on first?",
                'next_state': DialogueState.INFORMATION_GATHERING,
                'steps': [
                    'Acknowledge multiple interests',
                    'Help prioritize',
                    'Focus on one topic at a time'
                ]
            }
        }
        
        strategy = strategies.get(error_type, strategies['misunderstanding'])
        
        # 填充模板
        topic = context.entity_memory.get('main_topic', 'your question')
        strategy['response_template'] = strategy['response_template'].format(topic=topic)
        
        return strategy
    
    def recover_dialogue(
        self,
        session_id: str,
        error_analysis: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        执行对话恢复
        
        Args:
            session_id: 会话ID
            error_analysis: 错误分析结果
        
        Returns:
            恢复结果
        """
        if not error_analysis['recovery_recommended']:
            return {
                'recovery_needed': False,
                'reason': 'No significant errors detected'
            }
        
        # 获取最严重的错误
        serious_errors = [
            e for e in error_analysis['errors']
            if e['severity'] in ['high', 'medium']
        ]
        
        if not serious_errors:
            return {
                'recovery_needed': False,
                'reason': 'No serious errors'
            }
        
        primary_error = serious_errors[0]
        
        # 生成恢复策略
        context = DialogueContext(session_id=session_id)  # 简化
        recovery_strategy = self.generate_recovery_strategy(
            primary_error['type'],
            context
        )
        
        return {
            'recovery_needed': True,
            'error_type': primary_error['type'],
            'severity': primary_error['severity'],
            'recovery_strategy': recovery_strategy,
            'suggested_response': recovery_strategy['response_template'],
            'next_state': recovery_strategy['next_state'].value
        }


# 主引擎类
class DialogueStateManagementEngine:
    """对话状态管理引擎"""
    
    def __init__(self):
        self.state_tracker = StateTracker()
        self.intent_detector = IntentEvolutionDetector()
        self.reasoning_engine = MultiTurnReasoningEngine()
        self.recovery_engine = DialogueRecoveryEngine()
    
    def process_turn(
        self,
        session_id: str,
        user_utterance: str,
        detected_intent: IntentType,
        entities: Optional[Dict] = None,
        system_response: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        处理对话轮次
        
        这是V117创新的主要入口函数
        """
        # 1. 获取或创建会话
        context = self.state_tracker.get_session(session_id)
        if not context:
            context = self.state_tracker.create_session(session_id)
        
        # 2. 更新状态
        new_state, updated_context = self.state_tracker.update_state(
            session_id,
            detected_intent,
            user_utterance,
            system_response or "[System response placeholder]",
            entities
        )
        
        # 3. 检测意图演化
        intent_evolution = self.intent_detector.detect_intent_evolution(
            updated_context.intent_history
        )
        
        # 4. 检查是否需要恢复
        if updated_context.turns:
            error_analysis = self.recovery_engine.detect_error(
                updated_context,
                updated_context.turns[-1]
            )
            
            if error_analysis['recovery_recommended']:
                recovery_plan = self.recovery_engine.recover_dialogue(
                    session_id,
                    error_analysis
                )
            else:
                recovery_plan = None
        else:
            error_analysis = None
            recovery_plan = None
        
        # 5. 构建或继续推理链
        if len(updated_context.turns) > 1:
            reasoning_chain = self.reasoning_engine.build_reasoning_chain(
                session_id,
                updated_context.turns
            )
        else:
            reasoning_chain = []
        
        # 6. 生成状态摘要
        state_summary = self.state_tracker.get_state_summary(session_id)
        
        return {
            'version': 'V117',
            'name': 'Dialogue State Management & Multi-turn Reasoning (DSM)',
            'session_id': session_id,
            'turn_info': {
                'turn_number': len(updated_context.turns),
                'current_state': new_state.value,
                'detected_intent': detected_intent.value
            },
            'intent_evolution': intent_evolution,
            'error_analysis': error_analysis,
            'recovery_plan': recovery_plan,
            'reasoning_chain': {
                'node_count': len(reasoning_chain),
                'nodes': [
                    {
                        'id': node.node_id,
                        'conclusion': node.conclusion,
                        'confidence': node.confidence
                    }
                    for node in reasoning_chain[-3:]  # 最近3个节点
                ]
            },
            'state_summary': state_summary,
            'recommendations': self._generate_recommendations(
                updated_context, intent_evolution, error_analysis
            )
        }
    
    def _generate_recommendations(
        self,
        context: DialogueContext,
        intent_evolution: Dict,
        error_analysis: Optional[Dict]
    ) -> List[str]:
        """生成建议"""
        recommendations = []
        
        # 基于意图演化
        if intent_evolution.get('intent_drift', {}).get('drift_detected'):
            recommendations.append("User intent has shifted significantly. Confirm understanding.")
        
        if intent_evolution.get('recent_intent_shifts', 0) > 2:
            recommendations.append("High intent instability. Consider focusing the conversation.")
        
        # 基于错误分析
        if error_analysis and error_analysis['errors_detected']:
            for error in error_analysis['errors']:
                if error['severity'] == 'high':
                    recommendations.append(f"Address {error['type']}: {error['description']}")
        
        # 基于对话长度
        if len(context.turns) > 10:
            recommendations.append("Long conversation. Consider summarizing progress.")
        
        return recommendations


# 便捷函数
def build_dialogue_state_management(
    session_id: str,
    user_utterance: str,
    detected_intent: str = "query",
    entities: Optional[Dict] = None
) -> Dict[str, Any]:
    """
    构建对话状态管理
    
    这是V117创新的便捷入口函数
    """
    engine = DialogueStateManagementEngine()
    
    intent_map = {
        'query': IntentType.QUERY,
        'clarify': IntentType.CLARIFY,
        'confirm': IntentType.CONFIRM,
        'reject': IntentType.REJECT,
        'expand': IntentType.EXPAND,
        'shift': IntentType.SHIFT
    }
    
    intent = intent_map.get(detected_intent.lower(), IntentType.QUERY)
    
    return engine.process_turn(session_id, user_utterance, intent, entities)


# 示例用法
if __name__ == "__main__":
    engine = DialogueStateManagementEngine()
    session_id = "test_session_001"
    
    # 模拟多轮对话
    turns = [
        ("Hello!", IntentType.GREET),
        ("I want to know about machine learning", IntentType.QUERY),
        ("What exactly is deep learning?", IntentType.EXPAND),
        ("I'm confused about neural networks", IntentType.CLARIFY),
        ("Can you explain it simpler?", IntentType.CLARIFY)
    ]
    
    for utterance, intent in turns:
        result = engine.process_turn(session_id, utterance, intent)
        print(f"\nUser: {utterance}")
        print(f"State: {result['turn_info']['current_state']}")
        print(f"Intent: {result['turn_info']['detected_intent']}")
        if result['intent_evolution']['evolution_detected']:
            print(f"Evolution: {result['intent_evolution']['evolution_summary']}")
