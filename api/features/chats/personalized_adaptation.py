"""
V109: Personalized Adaptation & User Modeling (PAUM)
个性化适应与用户建模 - 根据用户偏好和历史行为自适应调整

创新点：
1. 用户画像构建：基于交互历史构建用户知识背景和偏好模型
2. 自适应解释：根据用户专业水平调整解释深度
3. 个性化推荐：推荐用户可能感兴趣的相关问题
4. 学习路径优化：为用户定制知识学习路径
"""

import json
import hashlib
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime, timedelta
import numpy as np


@dataclass
class UserProfile:
    """用户画像数据结构"""
    user_id: str
    expertise_level: str = 'intermediate'  # novice, intermediate, expert
    preferred_domains: List[str] = field(default_factory=list)
    knowledge_gaps: List[str] = field(default_factory=list)
    interaction_style: str = 'balanced'  # concise, detailed, exploratory
    learning_goals: List[str] = field(default_factory=list)
    query_history_summary: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)


@dataclass
class InteractionPattern:
    """交互模式数据结构"""
    query_types: Dict[str, int] = field(default_factory=lambda: defaultdict(int))
    topic_distribution: Dict[str, float] = field(default_factory=dict)
    avg_query_length: float = 0.0
    follow_up_rate: float = 0.0
    satisfaction_indicators: Dict[str, float] = field(default_factory=dict)
    common_follow_up_patterns: List[str] = field(default_factory=list)


class UserModelingEngine:
    """用户建模引擎"""
    
    def __init__(self):
        self.user_profiles: Dict[str, UserProfile] = {}
        self.interaction_history: Dict[str, List[Dict]] = defaultdict(list)
        self.pattern_cache: Dict[str, InteractionPattern] = {}
    
    def create_or_update_profile(
        self, 
        user_id: str, 
        interaction: Dict[str, Any]
    ) -> UserProfile:
        """
        创建或更新用户画像
        
        Args:
            user_id: 用户ID
            interaction: 当前交互数据
        
        Returns:
            UserProfile: 更新后的用户画像
        """
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserProfile(user_id=user_id)
        
        profile = self.user_profiles[user_id]
        
        # 更新交互历史
        self.interaction_history[user_id].append({
            'timestamp': datetime.now(),
            'query': interaction.get('query', ''),
            'response_type': interaction.get('response_type', ''),
            'feedback': interaction.get('feedback', {}),
            'duration': interaction.get('duration', 0)
        })
        
        # 限制历史长度
        if len(self.interaction_history[user_id]) > 100:
            self.interaction_history[user_id] = self.interaction_history[user_id][-100:]
        
        # 分析并更新用户画像
        self._analyze_expertise_level(profile, user_id)
        self._analyze_preferred_domains(profile, user_id)
        self._analyze_knowledge_gaps(profile, user_id)
        self._analyze_interaction_style(profile, user_id)
        
        profile.updated_at = datetime.now()
        
        return profile
    
    def _analyze_expertise_level(self, profile: UserProfile, user_id: str):
        """分析用户专业水平"""
        history = self.interaction_history[user_id]
        if len(history) < 3:
            return
        
        # 基于查询复杂度判断
        complex_terms_count = 0
        total_queries = len(history)
        
        expert_indicators = [
            'algorithm', 'architecture', 'implementation', 'optimization',
            'hyperparameter', 'gradient', 'backpropagation', 'attention mechanism',
            'transformer', 'bert', 'gpt', 'fine-tuning', 'pre-training'
        ]
        
        for interaction in history:
            query = interaction.get('query', '').lower()
            complex_terms_count += sum(1 for term in expert_indicators if term in query)
        
        avg_complexity = complex_terms_count / total_queries if total_queries > 0 else 0
        
        # 更新专业水平
        if avg_complexity > 2:
            profile.expertise_level = 'expert'
        elif avg_complexity > 0.5:
            profile.expertise_level = 'intermediate'
        else:
            profile.expertise_level = 'novice'
    
    def _analyze_preferred_domains(self, profile: UserProfile, user_id: str):
        """分析用户偏好领域"""
        history = self.interaction_history[user_id]
        
        domain_keywords = {
            'machine_learning': ['ml', 'machine learning', 'model', 'training', 'neural'],
            'nlp': ['nlp', 'natural language', 'text', 'tokenization', 'embedding'],
            'computer_vision': ['cv', 'image', 'vision', 'cnn', 'resnet'],
            'data_science': ['data', 'analysis', 'statistics', 'visualization'],
            'software_engineering': ['code', 'programming', 'software', 'api', 'system']
        }
        
        domain_counts = defaultdict(int)
        
        for interaction in history:
            query = interaction.get('query', '').lower()
            for domain, keywords in domain_keywords.items():
                if any(kw in query for kw in keywords):
                    domain_counts[domain] += 1
        
        # 更新偏好领域（取前3）
        sorted_domains = sorted(domain_counts.items(), key=lambda x: x[1], reverse=True)
        profile.preferred_domains = [d[0] for d in sorted_domains[:3]]
    
    def _analyze_knowledge_gaps(self, profile: UserProfile, user_id: str):
        """分析用户知识缺口"""
        history = self.interaction_history[user_id]
        
        # 识别重复询问的基础概念
        basic_concepts = defaultdict(int)
        
        for interaction in history:
            query = interaction.get('query', '').lower()
            
            # 检测基础问题模式
            if any(pattern in query for pattern in ['what is', 'how to', 'explain', 'difference between']):
                # 提取主题
                words = query.split()
                for i, word in enumerate(words):
                    if word in ['what', 'how', 'explain'] and i + 2 < len(words):
                        concept = ' '.join(words[i+1:i+3])
                        basic_concepts[concept] += 1
        
        # 频繁询问的基础概念可能是知识缺口
        profile.knowledge_gaps = [
            concept for concept, count in basic_concepts.items()
            if count >= 2
        ][:5]  # 最多5个
    
    def _analyze_interaction_style(self, profile: UserProfile, user_id: str):
        """分析用户交互风格"""
        history = self.interaction_history[user_id]
        if len(history) < 5:
            return
        
        # 分析查询长度
        query_lengths = [len(i.get('query', '')) for i in history]
        avg_length = np.mean(query_lengths)
        
        # 分析追问率
        follow_ups = sum(1 for i in history if i.get('is_follow_up', False))
        follow_up_rate = follow_ups / len(history)
        
        # 判断交互风格
        if avg_length < 30 and follow_up_rate < 0.3:
            profile.interaction_style = 'concise'
        elif avg_length > 80 or follow_up_rate > 0.5:
            profile.interaction_style = 'exploratory'
        else:
            profile.interaction_style = 'detailed'
    
    def get_interaction_pattern(self, user_id: str) -> InteractionPattern:
        """获取用户交互模式"""
        if user_id in self.pattern_cache:
            return self.pattern_cache[user_id]
        
        history = self.interaction_history[user_id]
        pattern = InteractionPattern()
        
        if not history:
            return pattern
        
        # 查询类型分布
        for interaction in history:
            query = interaction.get('query', '').lower()
            if 'what' in query:
                pattern.query_types['factual'] += 1
            elif 'how' in query:
                pattern.query_types['procedural'] += 1
            elif 'why' in query:
                pattern.query_types['causal'] += 1
            elif 'compare' in query or 'difference' in query:
                pattern.query_types['comparative'] += 1
            else:
                pattern.query_types['general'] += 1
        
        # 平均查询长度
        pattern.avg_query_length = np.mean([
            len(i.get('query', '')) for i in history
        ])
        
        # 追问率
        follow_ups = sum(1 for i in history if i.get('is_follow_up', False))
        pattern.follow_up_rate = follow_ups / len(history)
        
        self.pattern_cache[user_id] = pattern
        return pattern


class PersonalizedResponseGenerator:
    """个性化响应生成器"""
    
    def __init__(self, user_modeling: UserModelingEngine):
        self.user_modeling = user_modeling
    
    def adapt_response(
        self,
        user_id: str,
        base_response: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        根据用户画像调整响应
        
        Args:
            user_id: 用户ID
            base_response: 基础响应
            context: 上下文信息
        
        Returns:
            个性化调整后的响应
        """
        profile = self.user_modeling.user_profiles.get(user_id)
        if not profile:
            return {
                'response': base_response,
                'adaptations': [],
                'personalization_score': 0.0
            }
        
        adaptations = []
        adapted_response = base_response
        
        # 根据专业水平调整
        if profile.expertise_level == 'novice':
            adapted_response = self._simplify_for_novice(adapted_response)
            adaptations.append('simplified_language')
        elif profile.expertise_level == 'expert':
            adapted_response = self._add_technical_depth(adapted_response, context)
            adaptations.append('technical_depth')
        
        # 根据交互风格调整
        if profile.interaction_style == 'concise':
            adapted_response = self._make_concise(adapted_response)
            adaptations.append('concise_format')
        elif profile.interaction_style == 'exploratory':
            adapted_response = self._add_exploratory_elements(adapted_response, context)
            adaptations.append('exploratory_elements')
        
        # 根据偏好领域添加相关内容
        if profile.preferred_domains:
            related_content = self._add_domain_specific_content(
                adapted_response, 
                profile.preferred_domains,
                context
            )
            if related_content:
                adapted_response += f"\n\n{related_content}"
                adaptations.append('domain_specific_content')
        
        # 针对知识缺口提供补充
        if profile.knowledge_gaps:
            supplement = self._provide_knowledge_supplement(
                adapted_response,
                profile.knowledge_gaps,
                context
            )
            if supplement:
                adapted_response += f"\n\n💡 **Helpful Context:** {supplement}"
                adaptations.append('knowledge_gap_supplement')
        
        personalization_score = len(adaptations) / 5.0  # 最多5个适应维度
        
        return {
            'response': adapted_response,
            'adaptations': adaptations,
            'personalization_score': round(personalization_score, 2),
            'user_profile_summary': {
                'expertise_level': profile.expertise_level,
                'interaction_style': profile.interaction_style,
                'preferred_domains': profile.preferred_domains
            }
        }
    
    def _simplify_for_novice(self, response: str) -> str:
        """为新手简化语言"""
        # 添加解释性注释
        technical_terms = {
            'transformer': ' (a type of neural network architecture)',
            'attention mechanism': ' (a technique that helps models focus on relevant parts of input)',
            'embedding': ' (a numerical representation of text)',
            'fine-tuning': ' (adjusting a pre-trained model for a specific task)',
            'gradient descent': ' (an optimization algorithm)'
        }
        
        simplified = response
        for term, explanation in technical_terms.items():
            if term in simplified.lower():
                # 只在第一次出现时添加解释
                simplified = simplified.replace(term, f"{term}{explanation}", 1)
        
        return simplified
    
    def _add_technical_depth(self, response: str, context: Dict) -> str:
        """为专家添加技术深度"""
        # 添加技术细节和引用
        technical_addendum = "\n\n**Technical Details:**\n"
        
        if 'evidence' in context:
            evidence = context['evidence']
            if evidence:
                technical_addendum += f"- Evidence sources: {len(evidence)} items\n"
                technical_addendum += f"- Confidence score: {context.get('confidence', 'N/A')}\n"
        
        if 'methodology' in context:
            technical_addendum += f"- Method: {context['methodology']}\n"
        
        return response + technical_addendum
    
    def _make_concise(self, response: str) -> str:
        """使响应更简洁"""
        # 提取关键句子
        sentences = response.split('. ')
        
        # 保留最重要的句子（简化处理：保留前2-3句）
        if len(sentences) > 3:
            concise = '. '.join(sentences[:3]) + '.'
            return concise
        
        return response
    
    def _add_exploratory_elements(self, response: str, context: Dict) -> str:
        """添加探索性元素"""
        exploratory_section = "\n\n**Related Questions to Explore:**\n"
        
        # 基于上下文生成相关问题
        related_questions = [
            "How does this compare to alternative approaches?",
            "What are the limitations of this method?",
            "Can you provide a concrete example?",
            "What recent developments have there been in this area?"
        ]
        
        for i, question in enumerate(related_questions[:3], 1):
            exploratory_section += f"{i}. {question}\n"
        
        return response + exploratory_section
    
    def _add_domain_specific_content(
        self, 
        response: str, 
        domains: List[str],
        context: Dict
    ) -> Optional[str]:
        """添加领域特定内容"""
        domain_content = {
            'machine_learning': "From an ML perspective, this relates to model generalization and feature learning.",
            'nlp': "In NLP terms, this involves semantic understanding and language modeling.",
            'computer_vision': "In computer vision, similar principles apply to visual feature extraction.",
            'data_science': "From a data science viewpoint, this involves statistical inference and pattern recognition.",
            'software_engineering': "From a software engineering perspective, this relates to system design and architecture."
        }
        
        relevant_contents = []
        for domain in domains:
            if domain in domain_content:
                relevant_contents.append(domain_content[domain])
        
        if relevant_contents:
            return "**Domain Connection:** " + " ".join(relevant_contents)
        
        return None
    
    def _provide_knowledge_supplement(
        self,
        response: str,
        knowledge_gaps: List[str],
        context: Dict
    ) -> Optional[str]:
        """提供知识补充"""
        # 检查响应中是否涉及用户的知识缺口
        for gap in knowledge_gaps:
            if gap.lower() in response.lower():
                return f"Since you've asked about '{gap}' before, here's a quick refresher: [Key concept explained simply]"
        
        return None


class PersonalizedRecommendationEngine:
    """个性化推荐引擎"""
    
    def __init__(self, user_modeling: UserModelingEngine):
        self.user_modeling = user_modeling
    
    def recommend_next_steps(
        self,
        user_id: str,
        current_query: str,
        current_response: str
    ) -> Dict[str, Any]:
        """
        推荐下一步操作
        
        Args:
            user_id: 用户ID
            current_query: 当前查询
            current_response: 当前响应
        
        Returns:
            推荐结果
        """
        profile = self.user_modeling.user_profiles.get(user_id)
        pattern = self.user_modeling.get_interaction_pattern(user_id)
        
        recommendations = {
            'follow_up_questions': [],
            'related_topics': [],
            'learning_resources': [],
            'suggested_actions': []
        }
        
        # 基于查询类型推荐
        query_lower = current_query.lower()
        
        if 'what is' in query_lower:
            recommendations['follow_up_questions'] = [
                "How does this work in practice?",
                "What are the main components?",
                "Can you give me an example?"
            ]
        elif 'how to' in query_lower:
            recommendations['follow_up_questions'] = [
                "What are common pitfalls to avoid?",
                "Are there alternative methods?",
                "How do I evaluate the results?"
            ]
        elif 'why' in query_lower:
            recommendations['follow_up_questions'] = [
                "What evidence supports this?",
                "Are there different interpretations?",
                "How does this compare to other theories?"
            ]
        
        # 基于用户画像推荐
        if profile:
            # 根据专业水平调整推荐
            if profile.expertise_level == 'novice':
                recommendations['learning_resources'] = [
                    "📚 Beginner's Guide to Key Concepts",
                    "🎥 Video Tutorial: Fundamentals",
                    "📖 Glossary of Terms"
                ]
            elif profile.expertise_level == 'expert':
                recommendations['learning_resources'] = [
                    "📄 Recent Research Papers",
                    "🔬 Advanced Implementation Details",
                    "📊 Benchmark Datasets and Results"
                ]
            
            # 根据偏好领域推荐相关主题
            for domain in profile.preferred_domains[:2]:
                recommendations['related_topics'].append(
                    f"Explore {domain.replace('_', ' ').title()}"
                )
        
        # 基于交互模式推荐
        if pattern.follow_up_rate > 0.5:
            recommendations['suggested_actions'].append(
                "Continue this exploration thread"
            )
        
        if pattern.avg_query_length > 60:
            recommendations['suggested_actions'].append(
                "Try breaking down complex queries into simpler ones"
            )
        
        return recommendations
    
    def generate_learning_path(
        self,
        user_id: str,
        topic: str
    ) -> Dict[str, Any]:
        """
        生成个性化学习路径
        
        Args:
            user_id: 用户ID
            topic: 学习主题
        
        Returns:
            学习路径
        """
        profile = self.user_modeling.user_profiles.get(user_id)
        
        # 根据专业水平生成不同难度的路径
        if not profile or profile.expertise_level == 'novice':
            path = {
                'difficulty': 'beginner',
                'estimated_time': '2-3 hours',
                'steps': [
                    {'order': 1, 'title': f'Introduction to {topic}', 'type': 'concept'},
                    {'order': 2, 'title': 'Basic Terminology', 'type': 'vocabulary'},
                    {'order': 3, 'title': 'Simple Examples', 'type': 'example'},
                    {'order': 4, 'title': 'Guided Practice', 'type': 'exercise'},
                    {'order': 5, 'title': 'Review and Quiz', 'type': 'assessment'}
                ]
            }
        elif profile.expertise_level == 'intermediate':
            path = {
                'difficulty': 'intermediate',
                'estimated_time': '4-5 hours',
                'steps': [
                    {'order': 1, 'title': f'Core Concepts of {topic}', 'type': 'concept'},
                    {'order': 2, 'title': 'Technical Deep Dive', 'type': 'technical'},
                    {'order': 3, 'title': 'Case Studies', 'type': 'case_study'},
                    {'order': 4, 'title': 'Implementation Project', 'type': 'project'},
                    {'order': 5, 'title': 'Best Practices', 'type': 'guideline'}
                ]
            }
        else:  # expert
            path = {
                'difficulty': 'advanced',
                'estimated_time': '6-8 hours',
                'steps': [
                    {'order': 1, 'title': f'Advanced {topic} Theory', 'type': 'theory'},
                    {'order': 2, 'title': 'Research Frontiers', 'type': 'research'},
                    {'order': 3, 'title': 'Complex Problem Solving', 'type': 'problem'},
                    {'order': 4, 'title': 'Optimization Techniques', 'type': 'optimization'},
                    {'order': 5, 'title': 'Innovation Project', 'type': 'innovation'}
                ]
            }
        
        return path


# 主引擎类
class PersonalizedAdaptationEngine:
    """个性化适应引擎"""
    
    def __init__(self):
        self.user_modeling = UserModelingEngine()
        self.response_generator = PersonalizedResponseGenerator(self.user_modeling)
        self.recommendation_engine = PersonalizedRecommendationEngine(self.user_modeling)
    
    def process_interaction(
        self,
        user_id: str,
        query: str,
        base_response: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        处理交互并提供个性化适应
        
        这是V109创新的主要入口函数
        """
        # 1. 更新用户画像
        interaction_data = {
            'query': query,
            'response_type': context.get('response_type', 'standard'),
            'feedback': context.get('feedback', {}),
            'duration': context.get('duration', 0),
            'is_follow_up': context.get('is_follow_up', False)
        }
        
        profile = self.user_modeling.create_or_update_profile(user_id, interaction_data)
        
        # 2. 生成个性化响应
        personalized = self.response_generator.adapt_response(
            user_id, 
            base_response, 
            context
        )
        
        # 3. 生成推荐
        recommendations = self.recommendation_engine.recommend_next_steps(
            user_id,
            query,
            base_response
        )
        
        # 4. 生成学习路径（如果适用）
        learning_path = None
        if context.get('suggest_learning_path', False):
            # 从查询中提取主题
            topic = query.replace('what is', '').replace('how to', '').strip()
            learning_path = self.recommendation_engine.generate_learning_path(
                user_id, 
                topic
            )
        
        return {
            'version': 'V109',
            'name': 'Personalized Adaptation & User Modeling',
            'user_profile': {
                'expertise_level': profile.expertise_level,
                'preferred_domains': profile.preferred_domains,
                'knowledge_gaps': profile.knowledge_gaps,
                'interaction_style': profile.interaction_style
            },
            'personalized_response': personalized,
            'recommendations': recommendations,
            'learning_path': learning_path,
            'adaptation_summary': {
                'total_interactions': len(self.user_modeling.interaction_history[user_id]),
                'profile_confidence': self._calculate_profile_confidence(profile),
                'personalization_applied': len(personalized.get('adaptations', [])) > 0
            }
        }
    
    def _calculate_profile_confidence(self, profile: UserProfile) -> float:
        """计算用户画像的置信度"""
        # 基于数据新鲜度和完整度
        age_hours = (datetime.now() - profile.updated_at).total_seconds() / 3600
        freshness = max(0, 1 - age_hours / 168)  # 一周内新鲜度线性下降
        
        completeness = 0
        if profile.expertise_level != 'intermediate':
            completeness += 0.25
        if profile.preferred_domains:
            completeness += 0.25
        if profile.knowledge_gaps:
            completeness += 0.25
        if profile.interaction_style != 'balanced':
            completeness += 0.25
        
        return round(freshness * completeness, 2)


# 便捷函数
def personalize_response(
    user_id: str,
    query: str,
    base_response: str,
    context: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    个性化响应
    
    这是V109创新的便捷入口函数
    """
    engine = PersonalizedAdaptationEngine()
    return engine.process_interaction(
        user_id, 
        query, 
        base_response, 
        context or {}
    )


# 示例用法
if __name__ == "__main__":
    engine = PersonalizedAdaptationEngine()
    
    # 模拟多次交互
    user_id = "user_123"
    
    interactions = [
        {
            'query': 'What is machine learning?',
            'response': 'Machine learning is a subset of AI that enables systems to learn from data.',
            'context': {'response_type': 'definition'}
        },
        {
            'query': 'How does gradient descent work?',
            'response': 'Gradient descent is an optimization algorithm that minimizes loss functions.',
            'context': {'response_type': 'explanation', 'is_follow_up': True}
        },
        {
            'query': 'Explain transformer architecture and attention mechanisms',
            'response': 'Transformers use self-attention to process sequences in parallel.',
            'context': {'response_type': 'technical', 'is_follow_up': True}
        }
    ]
    
    for interaction in interactions:
        result = engine.process_interaction(
            user_id,
            interaction['query'],
            interaction['response'],
            interaction['context']
        )
        print(f"\nQuery: {interaction['query']}")
        print(f"Profile: {result['user_profile']}")
        print(f"Adaptations: {result['personalized_response']['adaptations']}")
