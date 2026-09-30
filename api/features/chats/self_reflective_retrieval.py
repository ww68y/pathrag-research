"""
V120: Self-Reflective Retrieval Optimization (SRRO)
自反思检索优化 - 系统能够自我反思并优化检索策略

创新点：
1. 检索结果自我评估：评估检索质量
2. 失败案例分析：分析检索失败原因
3. 策略自动调整：根据反馈优化检索策略
4. 查询重写优化：改进查询表达
"""

import json
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from collections import defaultdict
import numpy as np


class RetrievalQuality(Enum):
    """检索质量等级"""
    EXCELLENT = "excellent"
    GOOD = "good"
    FAIR = "fair"
    POOR = "poor"
    FAILED = "failed"


class FailureType(Enum):
    """失败类型"""
    IRRELEVANT = "irrelevant_results"
    INCOMPLETE = "incomplete_coverage"
    CONFLICTING = "conflicting_information"
    OUTDATED = "outdated_information"
    TOO_BROAD = "query_too_broad"
    TOO_SPECIFIC = "query_too_specific"
    AMBIGUOUS = "ambiguous_query"


@dataclass
class RetrievalAttempt:
    """检索尝试记录"""
    attempt_id: str
    query: str
    retrieved_documents: List[Dict]
    quality_score: float
    quality_label: RetrievalQuality
    failure_types: List[FailureType]
    timestamp: datetime
    reflection_notes: List[str] = field(default_factory=list)


@dataclass
class ReflectionInsight:
    """反思洞察"""
    insight_type: str
    description: str
    confidence: float
    recommended_action: str
    expected_improvement: float


class RetrievalQualityEvaluator:
    """检索质量评估器"""
    
    def __init__(self):
        self.evaluation_history: List[Dict] = []
        self.quality_thresholds = {
            RetrievalQuality.EXCELLENT: 0.9,
            RetrievalQuality.GOOD: 0.7,
            RetrievalQuality.FAIR: 0.5,
            RetrievalQuality.POOR: 0.3
        }
    
    def evaluate_retrieval(
        self,
        query: str,
        retrieved_docs: List[Dict],
        answer: Optional[str] = None,
        user_feedback: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        评估检索质量
        
        Args:
            query: 原始查询
            retrieved_docs: 检索到的文档
            answer: 生成的答案（可选）
            user_feedback: 用户反馈分数（可选）
        
        Returns:
            质量评估结果
        """
        # 1. 相关性评分
        relevance_score = self._compute_relevance_score(query, retrieved_docs)
        
        # 2. 多样性评分
        diversity_score = self._compute_diversity_score(retrieved_docs)
        
        # 3. 覆盖度评分
        coverage_score = self._compute_coverage_score(query, retrieved_docs)
        
        # 4. 时效性评分
        freshness_score = self._compute_freshness_score(retrieved_docs)
        
        # 5. 综合质量分数
        overall_score = (
            relevance_score * 0.4 +
            diversity_score * 0.2 +
            coverage_score * 0.25 +
            freshness_score * 0.15
        )
        
        # 6. 确定质量等级
        quality_label = self._determine_quality_label(overall_score)
        
        # 7. 识别失败类型
        failure_types = self._identify_failure_types(
            query, retrieved_docs, relevance_score, coverage_score
        )
        
        # 8. 生成评估细节
        evaluation_details = {
            'relevance': {
                'score': relevance_score,
                'details': self._get_relevance_details(query, retrieved_docs)
            },
            'diversity': {
                'score': diversity_score,
                'top_sources': self._get_source_distribution(retrieved_docs)
            },
            'coverage': {
                'score': coverage_score,
                'missing_aspects': self._identify_missing_aspects(query, retrieved_docs)
            },
            'freshness': {
                'score': freshness_score,
                'avg_doc_age_days': self._compute_avg_doc_age(retrieved_docs)
            }
        }
        
        result = {
            'overall_score': overall_score,
            'quality_label': quality_label.value,
            'failure_types': [ft.value for ft in failure_types],
            'evaluation_details': evaluation_details,
            'improvement_suggestions': self._generate_suggestions(
                failure_types, evaluation_details
            )
        }
        
        self.evaluation_history.append(result)
        return result
    
    def _compute_relevance_score(
        self,
        query: str,
        docs: List[Dict]
    ) -> float:
        """计算相关性分数"""
        if not docs:
            return 0.0
        
        query_terms = set(query.lower().split())
        scores = []
        
        for doc in docs:
            content = doc.get('content', '').lower()
            doc_terms = set(content.split())
            
            # Jaccard相似度
            intersection = len(query_terms & doc_terms)
            union = len(query_terms | doc_terms)
            
            if union > 0:
                score = intersection / union
                scores.append(score)
        
        return np.mean(scores) if scores else 0.0
    
    def _compute_diversity_score(self, docs: List[Dict]) -> float:
        """计算多样性分数"""
        if len(docs) <= 1:
            return 1.0
        
        # 基于来源的多样性
        sources = [doc.get('source', 'unknown') for doc in docs]
        unique_sources = len(set(sources))
        
        return unique_sources / len(docs)
    
    def _compute_coverage_score(
        self,
        query: str,
        docs: List[Dict]
    ) -> float:
        """计算覆盖度分数"""
        # 提取查询的关键方面
        query_aspects = self._extract_query_aspects(query)
        
        if not query_aspects:
            return 0.5
        
        covered_aspects = set()
        
        for doc in docs:
            content = doc.get('content', '').lower()
            for aspect in query_aspects:
                if aspect.lower() in content:
                    covered_aspects.add(aspect)
        
        return len(covered_aspects) / len(query_aspects)
    
    def _extract_query_aspects(self, query: str) -> List[str]:
        """提取查询的关键方面"""
        # 简化的方面提取
        # 移除停用词
        stopwords = {'what', 'is', 'the', 'how', 'to', 'a', 'an', 'in', 'on', 'of'}
        words = [w for w in query.lower().split() if w not in stopwords]
        
        # 返回名词性短语（简化处理）
        return words[:5]  # 取前5个关键词
    
    def _compute_freshness_score(self, docs: List[Dict]) -> float:
        """计算时效性分数"""
        if not docs:
            return 0.0
        
        scores = []
        current_time = datetime.now()
        
        for doc in docs:
            doc_time = doc.get('timestamp')
            if doc_time:
                if isinstance(doc_time, str):
                    try:
                        doc_time = datetime.fromisoformat(doc_time)
                    except:
                        scores.append(0.5)
                        continue
                
                age_days = (current_time - doc_time).days
                # 30天内为最新，超过1年为过时
                if age_days <= 30:
                    scores.append(1.0)
                elif age_days >= 365:
                    scores.append(0.0)
                else:
                    scores.append(1.0 - (age_days / 365))
            else:
                scores.append(0.5)  # 未知时间戳
        
        return np.mean(scores) if scores else 0.5
    
    def _determine_quality_label(self, score: float) -> RetrievalQuality:
        """确定质量标签"""
        if score >= self.quality_thresholds[RetrievalQuality.EXCELLENT]:
            return RetrievalQuality.EXCELLENT
        elif score >= self.quality_thresholds[RetrievalQuality.GOOD]:
            return RetrievalQuality.GOOD
        elif score >= self.quality_thresholds[RetrievalQuality.FAIR]:
            return RetrievalQuality.FAIR
        elif score >= self.quality_thresholds[RetrievalQuality.POOR]:
            return RetrievalQuality.POOR
        else:
            return RetrievalQuality.FAILED
    
    def _identify_failure_types(
        self,
        query: str,
        docs: List[Dict],
        relevance_score: float,
        coverage_score: float
    ) -> List[FailureType]:
        """识别失败类型"""
        failures = []
        
        if relevance_score < 0.3:
            failures.append(FailureType.IRRELEVANT)
        
        if coverage_score < 0.4:
            failures.append(FailureType.INCOMPLETE)
        
        # 检查冲突
        if self._has_conflicting_info(docs):
            failures.append(FailureType.CONFLICTING)
        
        # 检查查询特征
        query_words = len(query.split())
        if query_words < 3:
            failures.append(FailureType.TOO_BROAD)
        elif query_words > 15:
            failures.append(FailureType.TOO_SPECIFIC)
        
        # 检查歧义
        if self._is_ambiguous(query):
            failures.append(FailureType.AMBIGUOUS)
        
        return failures
    
    def _has_conflicting_info(self, docs: List[Dict]) -> bool:
        """检查是否有冲突信息"""
        # 简化的冲突检测
        # 检查是否有矛盾的数值或事实
        return False  # 简化实现
    
    def _is_ambiguous(self, query: str) -> bool:
        """检查查询是否歧义"""
        ambiguous_indicators = ['it', 'they', 'them', 'that', 'this']
        return any(indicator in query.lower() for indicator in ambiguous_indicators)
    
    def _get_relevance_details(
        self,
        query: str,
        docs: List[Dict]
    ) -> List[Dict]:
        """获取相关性详情"""
        details = []
        
        for i, doc in enumerate(docs[:5]):  # 前5个文档
            content = doc.get('content', '')[:200]  # 前200字符
            details.append({
                'doc_id': i,
                'preview': content,
                'source': doc.get('source', 'unknown')
            })
        
        return details
    
    def _get_source_distribution(self, docs: List[Dict]) -> Dict[str, int]:
        """获取来源分布"""
        sources = defaultdict(int)
        
        for doc in docs:
            source = doc.get('source', 'unknown')
            sources[source] += 1
        
        return dict(sources)
    
    def _identify_missing_aspects(
        self,
        query: str,
        docs: List[Dict]
    ) -> List[str]:
        """识别缺失的方面"""
        query_aspects = self._extract_query_aspects(query)
        covered_aspects = set()
        
        for doc in docs:
            content = doc.get('content', '').lower()
            for aspect in query_aspects:
                if aspect.lower() in content:
                    covered_aspects.add(aspect)
        
        missing = [aspect for aspect in query_aspects if aspect not in covered_aspects]
        return missing
    
    def _compute_avg_doc_age(self, docs: List[Dict]) -> float:
        """计算平均文档年龄"""
        ages = []
        current_time = datetime.now()
        
        for doc in docs:
            doc_time = doc.get('timestamp')
            if doc_time and isinstance(doc_time, datetime):
                age = (current_time - doc_time).days
                ages.append(age)
        
        return np.mean(ages) if ages else -1
    
    def _generate_suggestions(
        self,
        failure_types: List[FailureType],
        details: Dict
    ) -> List[str]:
        """生成改进建议"""
        suggestions = []
        
        for failure in failure_types:
            if failure == FailureType.IRRELEVANT:
                suggestions.append("考虑使用更具体的查询词或添加同义词")
            elif failure == FailureType.INCOMPLETE:
                suggestions.append("扩展查询以覆盖更多相关方面")
            elif failure == FailureType.CONFLICTING:
                suggestions.append("添加冲突解决机制或来源可信度评估")
            elif failure == FailureType.OUTDATED:
                suggestions.append("优先检索最新文档或添加时间过滤器")
            elif failure == FailureType.TOO_BROAD:
                suggestions.append("缩小查询范围，添加更多限定条件")
            elif failure == FailureType.TOO_SPECIFIC:
                suggestions.append("简化查询，移除不必要的细节")
            elif failure == FailureType.AMBIGUOUS:
                suggestions.append("澄清查询中的代词或模糊引用")
        
        return suggestions


class FailureCaseAnalyzer:
    """失败案例分析器"""
    
    def __init__(self):
        self.failure_cases: List[RetrievalAttempt] = []
        self.pattern_memory: Dict[str, List[Dict]] = defaultdict(list)
    
    def analyze_failure(
        self,
        attempt: RetrievalAttempt
    ) -> Dict[str, Any]:
        """
        分析失败案例
        
        Args:
            attempt: 检索尝试记录
        
        Returns:
            失败分析结果
        """
        analysis = {
            'attempt_id': attempt.attempt_id,
            'failure_types': [ft.value for ft in attempt.failure_types],
            'root_causes': [],
            'patterns': [],
            'lessons_learned': []
        }
        
        # 1. 根因分析
        for failure_type in attempt.failure_types:
            root_cause = self._identify_root_cause(
                attempt.query,
                attempt.retrieved_documents,
                failure_type
            )
            analysis['root_causes'].append({
                'failure_type': failure_type.value,
                'root_cause': root_cause
            })
        
        # 2. 模式识别
        patterns = self._identify_patterns(attempt)
        analysis['patterns'] = patterns
        
        # 3. 生成经验教训
        lessons = self._generate_lessons(attempt, patterns)
        analysis['lessons_learned'] = lessons
        
        # 4. 存储案例
        self.failure_cases.append(attempt)
        
        # 5. 更新模式记忆
        for pattern in patterns:
            self.pattern_memory[pattern['type']].append({
                'query': attempt.query,
                'failure': attempt.failure_types[0].value if attempt.failure_types else 'unknown',
                'timestamp': attempt.timestamp
            })
        
        return analysis
    
    def _identify_root_cause(
        self,
        query: str,
        docs: List[Dict],
        failure_type: FailureType
    ) -> str:
        """识别根因"""
        causes = {
            FailureType.IRRELEVANT: "查询与文档库语义空间不匹配",
            FailureType.INCOMPLETE: "知识库覆盖度不足或查询过于复杂",
            FailureType.CONFLICTING: "知识库中存在矛盾信息",
            FailureType.OUTDATED: "文档更新不及时",
            FailureType.TOO_BROAD: "查询缺乏具体性",
            FailureType.TOO_SPECIFIC: "查询包含过多限制条件",
            FailureType.AMBIGUOUS: "查询表述不清晰"
        }
        
        return causes.get(failure_type, "未知原因")
    
    def _identify_patterns(self, attempt: RetrievalAttempt) -> List[Dict]:
        """识别模式"""
        patterns = []
        
        # 查询长度模式
        query_len = len(attempt.query.split())
        if query_len < 4:
            patterns.append({
                'type': 'short_query',
                'description': '短查询模式',
                'confidence': 0.8
            })
        elif query_len > 12:
            patterns.append({
                'type': 'long_query',
                'description': '长查询模式',
                'confidence': 0.8
            })
        
        # 失败类型模式
        if attempt.failure_types:
            patterns.append({
                'type': f'failure_{attempt.failure_types[0].value}',
                'description': f'{attempt.failure_types[0].value}失败模式',
                'confidence': 0.9
            })
        
        return patterns
    
    def _generate_lessons(
        self,
        attempt: RetrievalAttempt,
        patterns: List[Dict]
    ) -> List[str]:
        """生成经验教训"""
        lessons = []
        
        for pattern in patterns:
            if pattern['type'] == 'short_query':
                lessons.append("短查询需要扩展以获取更多上下文")
            elif pattern['type'] == 'long_query':
                lessons.append("长查询可能需要分解为子查询")
        
        # 基于失败类型的教训
        for failure in attempt.failure_types:
            if failure == FailureType.IRRELEVANT:
                lessons.append("考虑使用查询扩展或语义重写")
            elif failure == FailureType.INCOMPLETE:
                lessons.append("需要多轮检索或主动证据获取")
        
        return lessons
    
    def get_common_failure_patterns(self, n: int = 5) -> List[Dict]:
        """获取常见失败模式"""
        pattern_counts = defaultdict(int)
        
        for case in self.failure_cases:
            for failure in case.failure_types:
                pattern_counts[failure.value] += 1
        
        sorted_patterns = sorted(
            pattern_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        return [
            {'pattern': p[0], 'count': p[1]}
            for p in sorted_patterns[:n]
        ]


class StrategyOptimizer:
    """策略优化器"""
    
    def __init__(self):
        self.strategies: Dict[str, Dict] = {}
        self.optimization_history: List[Dict] = []
        self.current_strategy = self._init_default_strategy()
    
    def _init_default_strategy(self) -> Dict:
        """初始化默认策略"""
        return {
            'retrieval_depth': 'standard',
            'similarity_threshold': 0.7,
            'max_documents': 10,
            'diversity_weight': 0.3,
            'recency_weight': 0.2,
            'query_expansion': False,
            'reranking': True
        }
    
    def optimize_strategy(
        self,
        evaluation_results: List[Dict],
        failure_analyses: List[Dict]
    ) -> Dict[str, Any]:
        """
        优化检索策略
        
        Args:
            evaluation_results: 评估结果列表
            failure_analyses: 失败分析列表
        
        Returns:
            优化后的策略
        """
        # 1. 分析趋势
        trend_analysis = self._analyze_trends(evaluation_results)
        
        # 2. 识别需要调整的参数
        adjustments = self._identify_adjustments(
            trend_analysis,
            failure_analyses
        )
        
        # 3. 应用调整
        new_strategy = self._apply_adjustments(
            self.current_strategy,
            adjustments
        )
        
        # 4. 验证策略有效性
        validation = self._validate_strategy(new_strategy)
        
        # 5. 更新当前策略
        self.current_strategy = new_strategy
        
        optimization_record = {
            'timestamp': datetime.now(),
            'previous_strategy': self.current_strategy,
            'new_strategy': new_strategy,
            'adjustments': adjustments,
            'validation': validation,
            'expected_improvement': trend_analysis.get('predicted_improvement', 0)
        }
        
        self.optimization_history.append(optimization_record)
        
        return {
            'optimized_strategy': new_strategy,
            'adjustments_made': adjustments,
            'validation_result': validation,
            'optimization_record': optimization_record
        }
    
    def _analyze_trends(self, evaluations: List[Dict]) -> Dict:
        """分析趋势"""
        if not evaluations:
            return {'trend': 'stable'}
        
        scores = [e.get('overall_score', 0) for e in evaluations]
        
        # 计算趋势
        if len(scores) >= 3:
            recent_avg = np.mean(scores[-3:])
            older_avg = np.mean(scores[:-3]) if len(scores) > 3 else scores[0]
            
            if recent_avg > older_avg + 0.05:
                trend = 'improving'
            elif recent_avg < older_avg - 0.05:
                trend = 'declining'
            else:
                trend = 'stable'
        else:
            trend = 'insufficient_data'
        
        return {
            'trend': trend,
            'average_score': np.mean(scores),
            'score_std': np.std(scores),
            'predicted_improvement': 0.1 if trend == 'improving' else 0.05
        }
    
    def _identify_adjustments(
        self,
        trend_analysis: Dict,
        failure_analyses: List[Dict]
    ) -> List[Dict]:
        """识别需要调整的参数"""
        adjustments = []
        
        # 基于趋势的调整
        if trend_analysis.get('trend') == 'declining':
            adjustments.append({
                'parameter': 'retrieval_depth',
                'current_value': self.current_strategy['retrieval_depth'],
                'new_value': 'deep',
                'reason': '质量下降，需要更深入检索'
            })
        
        # 基于失败分析的调整
        common_failures = defaultdict(int)
        for analysis in failure_analyses:
            for failure in analysis.get('failure_types', []):
                common_failures[failure] += 1
        
        if common_failures.get('irrelevant_results', 0) > 2:
            adjustments.append({
                'parameter': 'similarity_threshold',
                'current_value': self.current_strategy['similarity_threshold'],
                'new_value': min(0.9, self.current_strategy['similarity_threshold'] + 0.1),
                'reason': '不相关结果过多，提高相似度阈值'
            })
        
        if common_failures.get('incomplete_coverage', 0) > 2:
            adjustments.append({
                'parameter': 'max_documents',
                'current_value': self.current_strategy['max_documents'],
                'new_value': self.current_strategy['max_documents'] + 5,
                'reason': '覆盖度不足，增加检索文档数量'
            })
        
        return adjustments
    
    def _apply_adjustments(
        self,
        strategy: Dict,
        adjustments: List[Dict]
    ) -> Dict:
        """应用调整"""
        new_strategy = strategy.copy()
        
        for adjustment in adjustments:
            param = adjustment['parameter']
            new_value = adjustment['new_value']
            new_strategy[param] = new_value
        
        return new_strategy
    
    def _validate_strategy(self, strategy: Dict) -> Dict:
        """验证策略"""
        # 简化的验证
        validation_issues = []
        
        if strategy['similarity_threshold'] > 0.95:
            validation_issues.append("相似度阈值过高，可能导致召回率下降")
        
        if strategy['max_documents'] > 20:
            validation_issues.append("文档数量过多，可能影响处理速度")
        
        return {
            'valid': len(validation_issues) == 0,
            'issues': validation_issues
        }


class QueryRewriter:
    """查询重写器"""
    
    def __init__(self):
        self.rewrite_history: List[Dict] = []
        self.rewrite_templates = self._init_templates()
    
    def _init_templates(self) -> Dict[str, List[str]]:
        """初始化重写模板"""
        return {
            'expand': [
                "{query} 的定义和原理",
                "{query} 的应用和案例",
                "{query} 的优缺点"
            ],
            'clarify': [
                "请解释 {query}",
                "什么是 {query}",
                "详细说明 {query}"
            ],
            'decompose': [
                "{query} 的第一部分",
                "{query} 的第二部分"
            ]
        }
    
    def rewrite_query(
        self,
        query: str,
        failure_type: FailureType,
        context: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        重写查询
        
        Args:
            query: 原始查询
            failure_type: 失败类型
            context: 上下文信息
        
        Returns:
            重写结果
        """
        rewrite_strategy = self._select_rewrite_strategy(failure_type)
        
        rewritten_queries = []
        
        if rewrite_strategy == 'expand':
            rewritten_queries = self._expand_query(query)
        elif rewrite_strategy == 'clarify':
            rewritten_queries = self._clarify_query(query)
        elif rewrite_strategy == 'decompose':
            rewritten_queries = self._decompose_query(query)
        elif rewrite_strategy == 'synonym':
            rewritten_queries = self._add_synonyms(query)
        
        result = {
            'original_query': query,
            'failure_type': failure_type.value,
            'rewrite_strategy': rewrite_strategy,
            'rewritten_queries': rewritten_queries,
            'expected_improvement': 0.15
        }
        
        self.rewrite_history.append(result)
        
        return result
    
    def _select_rewrite_strategy(self, failure_type: FailureType) -> str:
        """选择重写策略"""
        strategy_map = {
            FailureType.TOO_BROAD: 'clarify',
            FailureType.TOO_SPECIFIC: 'decompose',
            FailureType.IRRELEVANT: 'synonym',
            FailureType.INCOMPLETE: 'expand',
            FailureType.AMBIGUOUS: 'clarify'
        }
        
        return strategy_map.get(failure_type, 'expand')
    
    def _expand_query(self, query: str) -> List[str]:
        """扩展查询"""
        templates = self.rewrite_templates['expand']
        return [template.format(query=query) for template in templates]
    
    def _clarify_query(self, query: str) -> List[str]:
        """澄清查询"""
        templates = self.rewrite_templates['clarify']
        return [template.format(query=query) for template in templates]
    
    def _decompose_query(self, query: str) -> List[str]:
        """分解查询"""
        # 简化的查询分解
        parts = query.split('，')
        if len(parts) > 1:
            return parts
        
        templates = self.rewrite_templates['decompose']
        return [template.format(query=query) for template in templates[:2]]
    
    def _add_synonyms(self, query: str) -> List[str]:
        """添加同义词"""
        # 简化的同义词扩展
        synonyms_map = {
            'machine learning': ['ML', 'machine intelligence'],
            'artificial intelligence': ['AI', 'cognitive computing'],
            'deep learning': ['deep neural networks', 'DL']
        }
        
        rewritten = [query]
        
        for term, synonyms in synonyms_map.items():
            if term in query.lower():
                for synonym in synonyms:
                    rewritten.append(query.lower().replace(term, synonym))
        
        return rewritten[:3]  # 最多返回3个


# 主引擎类
class SelfReflectiveRetrievalEngine:
    """自反思检索引擎"""
    
    def __init__(self):
        self.quality_evaluator = RetrievalQualityEvaluator()
        self.failure_analyzer = FailureCaseAnalyzer()
        self.strategy_optimizer = StrategyOptimizer()
        self.query_rewriter = QueryRewriter()
        self.attempt_history: List[RetrievalAttempt] = []
    
    def process_with_reflection(
        self,
        query: str,
        retrieved_docs: List[Dict],
        answer: Optional[str] = None,
        user_feedback: Optional[float] = None
    ) -> Dict[str, Any]:
        """
        带反思的检索处理
        
        这是V120创新的主要入口函数
        """
        # 1. 评估检索质量
        quality_evaluation = self.quality_evaluator.evaluate_retrieval(
            query, retrieved_docs, answer, user_feedback
        )
        
        # 2. 创建检索尝试记录
        attempt = RetrievalAttempt(
            attempt_id=f"attempt_{len(self.attempt_history)}",
            query=query,
            retrieved_documents=retrieved_docs,
            quality_score=quality_evaluation['overall_score'],
            quality_label=RetrievalQuality(quality_evaluation['quality_label']),
            failure_types=[FailureType(ft) for ft in quality_evaluation['failure_types']],
            timestamp=datetime.now()
        )
        
        # 3. 如果质量不好，进行失败分析
        failure_analysis = None
        query_rewrite = None
        
        if attempt.quality_label in [RetrievalQuality.POOR, RetrievalQuality.FAILED]:
            failure_analysis = self.failure_analyzer.analyze_failure(attempt)
            
            # 4. 重写查询
            if attempt.failure_types:
                query_rewrite = self.query_rewriter.rewrite_query(
                    query,
                    attempt.failure_types[0]
                )
        
        # 5. 优化策略
        strategy_optimization = self.strategy_optimizer.optimize_strategy(
            [quality_evaluation],
            [failure_analysis] if failure_analysis else []
        )
        
        # 6. 生成反思洞察
        insights = self._generate_insights(
            attempt,
            quality_evaluation,
            failure_analysis
        )
        
        # 7. 保存尝试记录
        self.attempt_history.append(attempt)
        
        return {
            'version': 'V120',
            'name': 'Self-Reflective Retrieval Optimization (SRRO)',
            'quality_evaluation': quality_evaluation,
            'failure_analysis': failure_analysis,
            'query_rewrite': query_rewrite,
            'strategy_optimization': strategy_optimization,
            'reflection_insights': insights,
            'recommended_next_action': self._recommend_next_action(attempt),
            'learning_progress': self._get_learning_progress()
        }
    
    def _generate_insights(
        self,
        attempt: RetrievalAttempt,
        evaluation: Dict,
        failure_analysis: Optional[Dict]
    ) -> List[ReflectionInsight]:
        """生成反思洞察"""
        insights = []
        
        # 基于质量的洞察
        if attempt.quality_score < 0.5:
            insights.append(ReflectionInsight(
                insight_type='quality_improvement',
                description='检索质量较低，需要改进策略',
                confidence=0.9,
                recommended_action='应用查询重写和策略优化',
                expected_improvement=0.2
            ))
        
        # 基于失败类型的洞察
        if attempt.failure_types:
            for failure in attempt.failure_types:
                if failure == FailureType.IRRELEVANT:
                    insights.append(ReflectionInsight(
                        insight_type='relevance_issue',
                        description='检索结果与查询不相关',
                        confidence=0.85,
                        recommended_action='提高相似度阈值，使用语义匹配',
                        expected_improvement=0.25
                    ))
        
        # 基于历史的洞察
        if len(self.attempt_history) > 5:
            recent_scores = [a.quality_score for a in self.attempt_history[-5:]]
            avg_score = sum(recent_scores) / len(recent_scores)
            
            if avg_score < 0.6:
                insights.append(ReflectionInsight(
                    insight_type='systemic_issue',
                    description='近期检索质量持续偏低',
                    confidence=0.8,
                    recommended_action='进行全面的策略审查和调整',
                    expected_improvement=0.3
                ))
        
        return insights
    
    def _recommend_next_action(self, attempt: RetrievalAttempt) -> str:
        """推荐下一步行动"""
        if attempt.quality_label == RetrievalQuality.EXCELLENT:
            return "保持当前策略，继续监控"
        elif attempt.quality_label in [RetrievalQuality.GOOD, RetrievalQuality.FAIR]:
            return "进行微调优化"
        else:
            return "使用重写的查询重新检索，并应用优化后的策略"
    
    def _get_learning_progress(self) -> Dict:
        """获取学习进度"""
        if not self.attempt_history:
            return {'status': 'no_data'}
        
        recent_scores = [a.quality_score for a in self.attempt_history[-10:]]
        
        return {
            'total_attempts': len(self.attempt_history),
            'average_quality': sum(recent_scores) / len(recent_scores),
            'improvement_trend': 'positive' if len(recent_scores) > 1 and recent_scores[-1] > recent_scores[0] else 'stable',
            'common_patterns': self.failure_analyzer.get_common_failure_patterns(3)
        }


# 便捷函数
def build_self_reflective_retrieval(
    query: str,
    retrieved_docs: List[Dict],
    answer: Optional[str] = None
) -> Dict[str, Any]:
    """
    构建自反思检索优化
    
    这是V120创新的便捷入口函数
    """
    engine = SelfReflectiveRetrievalEngine()
    return engine.process_with_reflection(query, retrieved_docs, answer)


# 示例用法
if __name__ == "__main__":
    # 测试数据
    test_docs = [
        {'content': 'Machine learning is a subset of AI', 'source': 'wiki'},
        {'content': 'Deep learning uses neural networks', 'source': 'paper'}
    ]
    
    result = build_self_reflective_retrieval(
        "What is machine learning?",
        test_docs
    )
    
    print(f"Quality Score: {result['quality_evaluation']['overall_score']:.2f}")
    print(f"Quality Label: {result['quality_evaluation']['quality_label']}")
    print(f"Insights: {len(result['reflection_insights'])}")
