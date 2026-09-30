"""
V116: Emotion-Aware Reasoning (EAR)
情感感知推理 - 识别用户情感并调整推理策略

创新点：
1. 情感识别：检测用户查询中的情感状态
2. 情感推理：基于情感的上下文推理
3. 情感响应：生成情感适配的答案
4. 共情计算：计算和提升系统共情能力
"""

import re
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import numpy as np


class EmotionType(Enum):
    """情感类型"""
    NEUTRAL = "neutral"
    HAPPY = "happy"
    SAD = "sad"
    ANGRY = "angry"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    CONFUSED = "confused"
    FRUSTRATED = "frustrated"
    CURIOUS = "curious"
    URGENT = "urgent"


class EmotionIntensity(Enum):
    """情感强度"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    EXTREME = 4


@dataclass
class EmotionState:
    """情感状态"""
    primary_emotion: EmotionType
    secondary_emotions: List[Tuple[EmotionType, float]] = field(default_factory=list)
    intensity: EmotionIntensity = EmotionIntensity.MEDIUM
    confidence: float = 0.0
    valence: float = 0.0  # -1 (negative) to 1 (positive)
    arousal: float = 0.0  # 0 (calm) to 1 (excited)
    triggers: List[str] = field(default_factory=list)


@dataclass
class EmpathyScore:
    """共情分数"""
    cognitive_empathy: float  # 理解他人情感的能力
    emotional_empathy: float  # 感受他人情感的能力
    compassionate_empathy: float  # 帮助他人解决问题的意愿
    overall_score: float
    suggestions: List[str] = field(default_factory=list)


class EmotionRecognizer:
    """情感识别器"""
    
    # 情感词典
    EMOTION_LEXICON = {
        EmotionType.HAPPY: {
            'words': ['happy', 'joy', 'excited', 'glad', 'delighted', 'pleased', 
                     'wonderful', 'great', 'awesome', 'fantastic', 'love', '喜欢', '开心', '高兴'],
            'intensity_boosters': ['very', 'extremely', 'so', 'really', '太', '非常'],
            'intensity_dampeners': ['somewhat', 'a bit', '有点', '稍微']
        },
        EmotionType.SAD: {
            'words': ['sad', 'unhappy', 'depressed', 'disappointed', 'upset', 'miserable',
                     'heartbroken', 'gloomy', 'sorry', '难过', '伤心', '失望', '沮丧'],
            'intensity_boosters': ['very', 'deeply', 'extremely', '太', '非常'],
            'intensity_dampeners': ['a little', 'somewhat', '有点']
        },
        EmotionType.ANGRY: {
            'words': ['angry', 'mad', 'furious', 'annoyed', 'irritated', 'frustrated',
                     'outraged', 'livid', '生气', '愤怒', '恼火', '烦躁'],
            'intensity_boosters': ['very', 'extremely', 'really', '太', '非常'],
            'intensity_dampeners': ['a bit', 'somewhat', '有点']
        },
        EmotionType.FEAR: {
            'words': ['afraid', 'scared', 'terrified', 'anxious', 'worried', 'nervous',
                     'panic', 'fearful', '害怕', '担心', '焦虑', '恐惧'],
            'intensity_boosters': ['very', 'extremely', '太', '非常'],
            'intensity_dampeners': ['a little', 'somewhat', '有点']
        },
        EmotionType.SURPRISE: {
            'words': ['surprised', 'amazed', 'astonished', 'shocked', 'stunned',
                     'wow', 'unexpected', '惊讶', '震惊', '意外'],
            'intensity_boosters': ['very', 'completely', 'totally', '太', '非常'],
            'intensity_dampeners': ['somewhat', 'a bit', '有点']
        },
        EmotionType.CONFUSED: {
            'words': ['confused', 'puzzled', 'perplexed', 'lost', 'unclear',
                     'don\'t understand', '不明白', '困惑', '迷茫', '不清楚'],
            'intensity_boosters': ['very', 'completely', 'totally', '太', '完全'],
            'intensity_dampeners': ['a bit', 'somewhat', '有点']
        },
        EmotionType.FRUSTRATED: {
            'words': ['frustrated', 'stuck', 'blocked', 'helpless', 'hopeless',
                     'can\'t figure out', '解决不了', '卡住', '无助'],
            'intensity_boosters': ['very', 'extremely', '太', '非常'],
            'intensity_dampeners': ['a bit', 'somewhat', '有点']
        },
        EmotionType.CURIOUS: {
            'words': ['curious', 'interested', 'wondering', 'want to know',
                     'how does', 'why is', '想知道', '好奇', '感兴趣'],
            'intensity_boosters': ['very', 'really', '太', '非常'],
            'intensity_dampeners': ['somewhat', 'a bit', '有点']
        },
        EmotionType.URGENT: {
            'words': ['urgent', 'asap', 'immediately', 'quickly', 'hurry',
                     'deadline', 'emergency', '紧急', '马上', '立即', '尽快'],
            'intensity_boosters': ['very', 'extremely', '太', '非常'],
            'intensity_dampeners': []
        }
    }
    
    # 标点符号情感权重
    PUNCTUATION_PATTERNS = {
        '!!!': (EmotionType.ANGRY, 0.3),
        '???': (EmotionType.CONFUSED, 0.2),
        '!?': (EmotionType.SURPRISE, 0.2),
        '...': (EmotionType.SAD, 0.1),
        '!!': (EmotionType.HAPPY, 0.2),
        '??': (EmotionType.CONFUSED, 0.15)
    }
    
    def __init__(self):
        self.emotion_history: List[EmotionState] = []
    
    def recognize_emotion(self, text: str, context: Optional[List[str]] = None) -> EmotionState:
        """
        识别文本中的情感
        
        Args:
            text: 输入文本
            context: 上下文对话历史
        
        Returns:
            情感状态
        """
        text_lower = text.lower()
        emotion_scores = defaultdict(float)
        triggers = []
        
        # 1. 基于词典的情感识别
        for emotion, lexicon in self.EMOTION_LEXICON.items():
            score = 0.0
            
            # 检查情感词
            for word in lexicon['words']:
                if word in text_lower:
                    count = text_lower.count(word)
                    score += count * 1.0
                    triggers.append(word)
            
            # 检查强度修饰词
            for booster in lexicon['intensity_boosters']:
                if booster in text_lower:
                    score *= 1.5
            
            for dampener in lexicon['intensity_dampeners']:
                if dampener in text_lower:
                    score *= 0.7
            
            emotion_scores[emotion] = score
        
        # 2. 基于标点符号的情感识别
        for pattern, (emotion, weight) in self.PUNCTUATION_PATTERNS.items():
            if pattern in text:
                emotion_scores[emotion] += weight
                triggers.append(pattern)
        
        # 3. 基于文本特征的识别
        # 全大写通常表示强烈情感
        if text.isupper() and len(text) > 3:
            emotion_scores[EmotionType.ANGRY] += 0.5
            emotion_scores[EmotionType.URGENT] += 0.3
        
        # 4. 确定主要情感
        if emotion_scores:
            primary_emotion = max(emotion_scores.items(), key=lambda x: x[1])
            
            # 计算置信度
            total_score = sum(emotion_scores.values())
            confidence = primary_emotion[1] / total_score if total_score > 0 else 0
            
            # 确定强度
            intensity = self._calculate_intensity(primary_emotion[1], len(text))
            
            # 计算次要情感
            secondary = sorted(
                [(e, s) for e, s in emotion_scores.items() if e != primary_emotion[0]],
                key=lambda x: x[1],
                reverse=True
            )[:2]
            
            # 计算 valence 和 arousal
            valence, arousal = self._calculate_valence_arousal(primary_emotion[0], intensity)
            
            emotion_state = EmotionState(
                primary_emotion=primary_emotion[0],
                secondary_emotions=secondary,
                intensity=intensity,
                confidence=min(confidence, 1.0),
                valence=valence,
                arousal=arousal,
                triggers=list(set(triggers))
            )
        else:
            emotion_state = EmotionState(
                primary_emotion=EmotionType.NEUTRAL,
                intensity=EmotionIntensity.LOW,
                confidence=0.5
            )
        
        # 保存历史
        self.emotion_history.append(emotion_state)
        
        return emotion_state
    
    def _calculate_intensity(self, score: float, text_length: int) -> EmotionIntensity:
        """计算情感强度"""
        normalized_score = score / (text_length / 10 + 1)
        
        if normalized_score > 2.0:
            return EmotionIntensity.EXTREME
        elif normalized_score > 1.0:
            return EmotionIntensity.HIGH
        elif normalized_score > 0.5:
            return EmotionIntensity.MEDIUM
        else:
            return EmotionIntensity.LOW
    
    def _calculate_valence_arousal(
        self, 
        emotion: EmotionType, 
        intensity: EmotionIntensity
    ) -> Tuple[float, float]:
        """计算 valence 和 arousal"""
        # Valence: 情感的愉悦程度 (-1 to 1)
        valence_map = {
            EmotionType.HAPPY: 0.8,
            EmotionType.CURIOUS: 0.6,
            EmotionType.SURPRISE: 0.3,
            EmotionType.NEUTRAL: 0.0,
            EmotionType.CONFUSED: -0.2,
            EmotionType.FRUSTRATED: -0.4,
            EmotionType.SAD: -0.6,
            EmotionType.ANGRY: -0.7,
            EmotionType.FEAR: -0.8,
            EmotionType.DISGUST: -0.9
        }
        
        # Arousal: 情感的激活程度 (0 to 1)
        arousal_map = {
            EmotionType.NEUTRAL: 0.1,
            EmotionType.SAD: 0.2,
            EmotionType.CONFUSED: 0.4,
            EmotionType.FEAR: 0.8,
            EmotionType.ANGRY: 0.9,
            EmotionType.SURPRISE: 0.8,
            EmotionType.HAPPY: 0.7,
            EmotionType.CURIOUS: 0.6,
            EmotionType.URGENT: 0.9
        }
        
        base_valence = valence_map.get(emotion, 0.0)
        base_arousal = arousal_map.get(emotion, 0.5)
        
        # 根据强度调整
        intensity_factor = intensity.value / 2
        
        return base_valence, min(base_arousal * intensity_factor, 1.0)
    
    def detect_emotion_shift(self, window_size: int = 3) -> Optional[Dict[str, Any]]:
        """检测情感变化"""
        if len(self.emotion_history) < window_size:
            return None
        
        recent = self.emotion_history[-window_size:]
        emotions = [e.primary_emotion for e in recent]
        
        # 检查是否有显著变化
        if len(set(emotions)) > 1:
            return {
                'shift_detected': True,
                'from': emotions[0].value,
                'to': emotions[-1].value,
                'shift_pattern': [e.value for e in emotions],
                'possible_cause': 'context_change' if emotions[-1] != emotions[0] else 'fluctuation'
            }
        
        return None


class EmotionReasoningEngine:
    """情感推理引擎"""
    
    def __init__(self):
        self.emotion_strategies = self._init_emotion_strategies()
    
    def _init_emotion_strategies(self) -> Dict[EmotionType, Dict]:
        """初始化情感策略"""
        return {
            EmotionType.HAPPY: {
                'reasoning_depth': 'standard',
                'explanation_style': 'enthusiastic',
                'include_examples': True,
                'tone': 'positive'
            },
            EmotionType.SAD: {
                'reasoning_depth': 'detailed',
                'explanation_style': 'supportive',
                'include_examples': True,
                'tone': 'empathetic',
                'offer_help': True
            },
            EmotionType.ANGRY: {
                'reasoning_depth': 'concise',
                'explanation_style': 'direct',
                'include_examples': False,
                'tone': 'calm',
                'acknowledge_frustration': True
            },
            EmotionType.CONFUSED: {
                'reasoning_depth': 'very_detailed',
                'explanation_style': 'step_by_step',
                'include_examples': True,
                'tone': 'patient',
                'check_understanding': True
            },
            EmotionType.FRUSTRATED: {
                'reasoning_depth': 'detailed',
                'explanation_style': 'solution_focused',
                'include_examples': True,
                'tone': 'encouraging',
                'offer_alternatives': True
            },
            EmotionType.CURIOUS: {
                'reasoning_depth': 'comprehensive',
                'explanation_style': 'exploratory',
                'include_examples': True,
                'tone': 'engaging',
                'suggest_related': True
            },
            EmotionType.URGENT: {
                'reasoning_depth': 'essential_only',
                'explanation_style': 'direct',
                'include_examples': False,
                'tone': 'efficient',
                'prioritize_speed': True
            },
            EmotionType.NEUTRAL: {
                'reasoning_depth': 'standard',
                'explanation_style': 'balanced',
                'include_examples': True,
                'tone': 'professional'
            }
        }
    
    def adapt_reasoning_strategy(
        self,
        emotion_state: EmotionState,
        query_complexity: str = 'medium'
    ) -> Dict[str, Any]:
        """
        根据情感状态调整推理策略
        
        Args:
            emotion_state: 情感状态
            query_complexity: 查询复杂度
        
        Returns:
            调整后的推理策略
        """
        base_strategy = self.emotion_strategies.get(
            emotion_state.primary_emotion,
            self.emotion_strategies[EmotionType.NEUTRAL]
        )
        
        # 根据情感强度调整
        if emotion_state.intensity == EmotionIntensity.EXTREME:
            if emotion_state.primary_emotion in [EmotionType.ANGRY, EmotionType.FEAR]:
                # 极端负面情绪需要特别处理
                base_strategy['reasoning_depth'] = 'minimal'
                base_strategy['tone'] = 'very_calm'
                base_strategy['acknowledge_emotion'] = True
        
        # 根据复杂度调整
        complexity_adjustments = {
            'low': {'reasoning_depth': 'concise'},
            'medium': {'reasoning_depth': 'standard'},
            'high': {'reasoning_depth': 'detailed'}
        }
        
        if query_complexity in complexity_adjustments:
            base_strategy.update(complexity_adjustments[query_complexity])
        
        return {
            'strategy': base_strategy,
            'emotion_adaptations': self._generate_adaptations(emotion_state),
            'priority_adjustments': self._adjust_priorities(emotion_state)
        }
    
    def _generate_adaptations(self, emotion_state: EmotionState) -> List[str]:
        """生成情感适配建议"""
        adaptations = []
        
        if emotion_state.primary_emotion == EmotionType.CONFUSED:
            adaptations.extend([
                "Break down complex concepts into simpler parts",
                "Use analogies and examples",
                "Check for understanding at each step"
            ])
        elif emotion_state.primary_emotion == EmotionType.FRUSTRATED:
            adaptations.extend([
                "Acknowledge the difficulty",
                "Provide alternative approaches",
                "Offer step-by-step guidance"
            ])
        elif emotion_state.primary_emotion == EmotionType.ANGRY:
            adaptations.extend([
                "Stay calm and professional",
                "Focus on solutions",
                "Avoid defensive language"
            ])
        
        return adaptations
    
    def _adjust_priorities(self, emotion_state: EmotionState) -> Dict[str, float]:
        """根据情感调整优先级"""
        base_priorities = {
            'accuracy': 0.4,
            'speed': 0.3,
            'empathy': 0.2,
            'detail': 0.1
        }
        
        if emotion_state.primary_emotion == EmotionType.URGENT:
            base_priorities['speed'] = 0.6
            base_priorities['accuracy'] = 0.3
        elif emotion_state.primary_emotion == EmotionType.CONFUSED:
            base_priorities['detail'] = 0.4
            base_priorities['empathy'] = 0.3
        elif emotion_state.primary_emotion == EmotionType.SAD:
            base_priorities['empathy'] = 0.5
            base_priorities['accuracy'] = 0.3
        
        return base_priorities


class EmpathyCalculator:
    """共情计算器"""
    
    def __init__(self):
        self.empathy_history: List[EmpathyScore] = []
    
    def calculate_empathy(
        self,
        user_emotion: EmotionState,
        system_response: str,
        user_feedback: Optional[Dict] = None
    ) -> EmpathyScore:
        """
        计算系统的共情能力
        
        Args:
            user_emotion: 用户情感状态
            system_response: 系统响应
            user_feedback: 用户反馈（可选）
        
        Returns:
            共情分数
        """
        # 1. 认知共情：系统是否正确理解了用户情感
        cognitive = self._assess_cognitive_empathy(user_emotion, system_response)
        
        # 2. 情感共情：系统是否表达了适当的情感回应
        emotional = self._assess_emotional_empathy(user_emotion, system_response)
        
        # 3. 同情共情：系统是否提供了有帮助的回应
        compassionate = self._assess_compassionate_empathy(
            user_emotion, system_response, user_feedback
        )
        
        # 计算总体分数
        overall = (cognitive * 0.4 + emotional * 0.3 + compassionate * 0.3)
        
        # 生成改进建议
        suggestions = self._generate_empathy_suggestions(
            cognitive, emotional, compassionate, user_emotion
        )
        
        empathy_score = EmpathyScore(
            cognitive_empathy=cognitive,
            emotional_empathy=emotional,
            compassionate_empathy=compassionate,
            overall_score=overall,
            suggestions=suggestions
        )
        
        self.empathy_history.append(empathy_score)
        
        return empathy_score
    
    def _assess_cognitive_empathy(
        self,
        user_emotion: EmotionState,
        response: str
    ) -> float:
        """评估认知共情"""
        score = 0.5  # 基础分
        
        # 检查是否承认了用户的情感
        emotion_words = self._get_emotion_synonyms(user_emotion.primary_emotion)
        if any(word in response.lower() for word in emotion_words):
            score += 0.3
        
        # 检查是否理解了情感原因
        if user_emotion.triggers:
            trigger_words = [t.lower() for t in user_emotion.triggers]
            if any(trigger in response.lower() for trigger in trigger_words):
                score += 0.2
        
        return min(score, 1.0)
    
    def _assess_emotional_empathy(
        self,
        user_emotion: EmotionState,
        response: str
    ) -> float:
        """评估情感共情"""
        score = 0.5
        
        # 检查语调是否匹配
        if user_emotion.primary_emotion in [EmotionType.SAD, EmotionType.FRUSTRATED]:
            if any(word in response.lower() for word in ['understand', 'sorry', 'help']):
                score += 0.3
        
        if user_emotion.primary_emotion == EmotionType.HAPPY:
            if any(word in response.lower() for word in ['great', 'wonderful', 'glad']):
                score += 0.3
        
        # 检查是否表达了适当的情感支持
        if user_emotion.valence < 0:  # 负面情绪
            if 'support' in response.lower() or 'here to help' in response.lower():
                score += 0.2
        
        return min(score, 1.0)
    
    def _assess_compassionate_empathy(
        self,
        user_emotion: EmotionState,
        response: str,
        user_feedback: Optional[Dict]
    ) -> float:
        """评估同情共情"""
        score = 0.5
        
        # 检查是否提供了解决方案
        solution_indicators = ['solution', 'approach', 'try', 'can help', 'recommend']
        if any(indicator in response.lower() for indicator in solution_indicators):
            score += 0.3
        
        # 如果有用户反馈，使用反馈评估
        if user_feedback:
            helpfulness = user_feedback.get('helpfulness', 0.5)
            score = score * 0.5 + helpfulness * 0.5
        
        return min(score, 1.0)
    
    def _get_emotion_synonyms(self, emotion: EmotionType) -> List[str]:
        """获取情感同义词"""
        synonyms = {
            EmotionType.HAPPY: ['happy', 'glad', 'pleased', 'delighted', 'joyful'],
            EmotionType.SAD: ['sad', 'unhappy', 'sorry', 'upset', 'disappointed'],
            EmotionType.ANGRY: ['angry', 'frustrated', 'annoyed', 'upset', 'mad'],
            EmotionType.CONFUSED: ['confused', 'unclear', 'puzzled', 'lost'],
            EmotionType.FEAR: ['afraid', 'worried', 'anxious', 'concerned']
        }
        return synonyms.get(emotion, [emotion.value])
    
    def _generate_empathy_suggestions(
        self,
        cognitive: float,
        emotional: float,
        compassionate: float,
        user_emotion: EmotionState
    ) -> List[str]:
        """生成共情改进建议"""
        suggestions = []
        
        if cognitive < 0.7:
            suggestions.append("Explicitly acknowledge the user's emotional state")
        
        if emotional < 0.7:
            if user_emotion.valence < 0:
                suggestions.append("Use more supportive and understanding language")
            else:
                suggestions.append("Match the user's positive energy")
        
        if compassionate < 0.7:
            suggestions.append("Offer more concrete solutions or next steps")
        
        if not suggestions:
            suggestions.append("Maintain current empathetic approach")
        
        return suggestions
    
    def get_empathy_trend(self) -> Dict[str, Any]:
        """获取共情能力趋势"""
        if len(self.empathy_history) < 2:
            return {'trend': 'insufficient_data'}
        
        recent_scores = [e.overall_score for e in self.empathy_history[-10:]]
        
        if len(recent_scores) >= 3:
            # 简单线性趋势
            x = np.arange(len(recent_scores))
            slope = np.polyfit(x, recent_scores, 1)[0]
            
            if slope > 0.05:
                trend = 'improving'
            elif slope < -0.05:
                trend = 'declining'
            else:
                trend = 'stable'
        else:
            trend = 'stable'
        
        return {
            'trend': trend,
            'average_score': np.mean(recent_scores),
            'current_score': recent_scores[-1],
            'best_score': max(recent_scores),
            'samples': len(recent_scores)
        }


# 主引擎类
class EmotionAwareReasoningEngine:
    """情感感知推理引擎"""
    
    def __init__(self):
        self.emotion_recognizer = EmotionRecognizer()
        self.reasoning_engine = EmotionReasoningEngine()
        self.empathy_calculator = EmpathyCalculator()
    
    def process_with_emotion_awareness(
        self,
        query: str,
        context: Optional[List[str]] = None,
        generate_response: bool = True
    ) -> Dict[str, Any]:
        """
        情感感知处理
        
        这是V116创新的主要入口函数
        """
        # 1. 识别情感
        emotion_state = self.emotion_recognizer.recognize_emotion(query, context)
        
        # 2. 检测情感变化
        emotion_shift = self.emotion_recognizer.detect_emotion_shift()
        
        # 3. 调整推理策略
        reasoning_strategy = self.reasoning_engine.adapt_reasoning_strategy(
            emotion_state
        )
        
        # 4. 生成情感适配的响应（如果需要）
        response = None
        if generate_response:
            response = self._generate_emotion_aware_response(
                emotion_state,
                reasoning_strategy
            )
        
        # 5. 计算共情分数
        if response:
            empathy_score = self.empathy_calculator.calculate_empathy(
                emotion_state,
                response
            )
        else:
            empathy_score = None
        
        return {
            'version': 'V116',
            'name': 'Emotion-Aware Reasoning (EAR)',
            'emotion_analysis': {
                'primary_emotion': emotion_state.primary_emotion.value,
                'intensity': emotion_state.intensity.name,
                'confidence': emotion_state.confidence,
                'valence': emotion_state.valence,
                'arousal': emotion_state.arousal,
                'triggers': emotion_state.triggers
            },
            'emotion_shift': emotion_shift,
            'reasoning_strategy': reasoning_strategy,
            'generated_response': response,
            'empathy_assessment': {
                'cognitive': empathy_score.cognitive_empathy if empathy_score else None,
                'emotional': empathy_score.emotional_empathy if empathy_score else None,
                'compassionate': empathy_score.compassionate_empathy if empathy_score else None,
                'overall': empathy_score.overall_score if empathy_score else None,
                'suggestions': empathy_score.suggestions if empathy_score else []
            },
            'recommendations': self._generate_recommendations(emotion_state)
        }
    
    def _generate_emotion_aware_response(
        self,
        emotion_state: EmotionState,
        strategy: Dict[str, Any]
    ) -> str:
        """生成情感感知响应"""
        strategy_config = strategy.get('strategy', {})
        tone = strategy_config.get('tone', 'professional')
        
        # 根据情感生成开场白
        opening = self._generate_opening(emotion_state, tone)
        
        # 生成主体内容（简化）
        body = "[Answer content based on reasoning strategy]"
        
        # 生成结尾
        closing = self._generate_closing(emotion_state, strategy)
        
        return f"{opening} {body} {closing}"
    
    def _generate_opening(
        self, 
        emotion_state: EmotionState, 
        tone: str
    ) -> str:
        """生成开场白"""
        openings = {
            EmotionType.SAD: [
                "I understand this might be difficult for you.",
                "I'm sorry to hear you're going through this."
            ],
            EmotionType.ANGRY: [
                "I understand your frustration.",
                "I can see why you're upset."
            ],
            EmotionType.CONFUSED: [
                "I understand this can be confusing.",
                "Let me help clarify this for you."
            ],
            EmotionType.FRUSTRATED: [
                "I know this is frustrating.",
                "I understand you're having difficulty."
            ],
            EmotionType.HAPPY: [
                "I'm glad to help!",
                "Great question!"
            ],
            EmotionType.NEUTRAL: [
                "I'd be happy to help.",
                "Let me assist you with that."
            ]
        }
        
        import random
        options = openings.get(emotion_state.primary_emotion, openings[EmotionType.NEUTRAL])
        return random.choice(options)
    
    def _generate_closing(
        self, 
        emotion_state: EmotionState, 
        strategy: Dict[str, Any]
    ) -> str:
        """生成结尾"""
        strategy_config = strategy.get('strategy', {})
        
        if strategy_config.get('check_understanding'):
            return "Does this help clarify things? Let me know if you need more explanation."
        elif strategy_config.get('offer_help'):
            return "Is there anything else I can help you with?"
        elif strategy_config.get('suggest_related'):
            return "Would you like to explore related topics?"
        else:
            return "I hope this helps!"
    
    def _generate_recommendations(self, emotion_state: EmotionState) -> List[str]:
        """生成建议"""
        recommendations = []
        
        if emotion_state.intensity == EmotionIntensity.HIGH:
            recommendations.append("Consider offering additional support resources")
        
        if emotion_state.primary_emotion == EmotionType.CONFUSED:
            recommendations.append("Provide visual aids or step-by-step breakdowns")
        
        if emotion_state.primary_emotion == EmotionType.ANGRY:
            recommendations.append("Escalate to human support if issue persists")
        
        return recommendations


# 便捷函数
def build_emotion_aware_reasoning(
    query: str,
    context: Optional[List[str]] = None
) -> Dict[str, Any]:
    """
    构建情感感知推理
    
    这是V116创新的便捷入口函数
    """
    engine = EmotionAwareReasoningEngine()
    return engine.process_with_emotion_awareness(query, context)


# 示例用法
if __name__ == "__main__":
    # 测试情感识别
    test_queries = [
        "I'm so happy with the results!",
        "I'm confused about how this works...",
        "This is really frustrating!",
        "I'm worried about the deadline.",
        "What is machine learning?"  # 中性
    ]
    
    engine = EmotionAwareReasoningEngine()
    
    for query in test_queries:
        result = engine.process_with_emotion_awareness(query)
        print(f"\nQuery: {query}")
        print(f"Emotion: {result['emotion_analysis']['primary_emotion']}")
        print(f"Intensity: {result['emotion_analysis']['intensity']}")
        print(f"Confidence: {result['emotion_analysis']['confidence']:.2f}")
