"""
V112: Meta-Learning & Adaptive Optimization (MeLAO)
元学习与自适应优化 - 让系统学会如何学习

创新点：
1. 少样本学习：快速适应新领域
2. 超参数优化：自动调整系统参数
3. 策略学习：学习最优检索和生成策略
4. 持续学习：不断从交互中改进
"""

import json
import numpy as np
from typing import List, Dict, Any, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime, timedelta
import random


@dataclass
class TaskRepresentation:
    """任务表示"""
    task_id: str
    domain: str
    query_patterns: List[str]
    performance_history: List[float] = field(default_factory=list)
    optimal_config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MetaKnowledge:
    """元知识"""
    task_similarities: Dict[str, float] = field(default_factory=dict)
    transfer_rules: List[Dict] = field(default_factory=list)
    learned_strategies: Dict[str, Any] = field(default_factory=dict)


class FewShotLearner:
    """少样本学习器"""
    
    def __init__(self):
        self.task_prototypes: Dict[str, np.ndarray] = {}
        self.support_sets: Dict[str, List[Dict]] = defaultdict(list)
    
    def learn_from_few_examples(
        self, 
        task_name: str,
        examples: List[Dict],
        embedding_fn: Callable[[str], np.ndarray]
    ) -> Dict[str, Any]:
        """
        从少量示例学习
        
        Args:
            task_name: 任务名称
            examples: 示例列表
            embedding_fn: 嵌入函数
        
        Returns:
            学习结果
        """
        if len(examples) < 2:
            return {
                'success': False,
                'reason': 'Insufficient examples (minimum 2 required)'
            }
        
        # 计算任务原型（示例的平均嵌入）
        embeddings = []
        for ex in examples:
            text = ex.get('query', '') + ' ' + ex.get('context', '')
            emb = embedding_fn(text)
            embeddings.append(emb)
        
        prototype = np.mean(embeddings, axis=0)
        self.task_prototypes[task_name] = prototype
        self.support_sets[task_name] = examples
        
        # 计算类内方差（评估学习质量）
        if len(embeddings) > 1:
            variance = np.var(embeddings, axis=0).mean()
            compactness = 1 / (1 + variance)  # 方差越小，紧凑度越高
        else:
            compactness = 1.0
        
        return {
            'success': True,
            'task_name': task_name,
            'prototype_shape': prototype.shape,
            'examples_used': len(examples),
            'compactness': round(compactness, 4),
            'learned_patterns': self._extract_patterns(examples)
        }
    
    def adapt_to_new_task(
        self,
        new_task_query: str,
        embedding_fn: Callable[[str], np.ndarray],
        k: int = 3
    ) -> Dict[str, Any]:
        """
        适应新任务
        
        Args:
            new_task_query: 新任务查询
            embedding_fn: 嵌入函数
            k: 最近邻数量
        
        Returns:
            适应结果
        """
        query_emb = embedding_fn(new_task_query)
        
        # 找到最相似的任务
        similarities = {}
        for task_name, prototype in self.task_prototypes.items():
            similarity = self._cosine_similarity(query_emb, prototype)
            similarities[task_name] = similarity
        
        # 排序并选择 top-k
        sorted_tasks = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        top_k_tasks = sorted_tasks[:k]
        
        # 构建适应策略
        adaptation_strategy = {
            'most_similar_task': top_k_tasks[0][0] if top_k_tasks else None,
            'similarity_score': top_k_tasks[0][1] if top_k_tasks else 0,
            'suggested_config': self._suggest_config(top_k_tasks),
            'transfer_knowledge': self._gather_transfer_knowledge(top_k_tasks)
        }
        
        return adaptation_strategy
    
    def _cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """计算余弦相似度"""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return np.dot(a, b) / (norm_a * norm_b)
    
    def _extract_patterns(self, examples: List[Dict]) -> List[str]:
        """从示例中提取模式"""
        patterns = []
        
        # 简单的模式提取：共同关键词
        all_words = []
        for ex in examples:
            query = ex.get('query', '').lower()
            all_words.extend(query.split())
        
        # 统计词频
        word_freq = defaultdict(int)
        for word in all_words:
            if len(word) > 3:  # 忽略短词
                word_freq[word] += 1
        
        # 返回高频词作为模式
        common_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        patterns = [word for word, freq in common_words[:5] if freq > 1]
        
        return patterns
    
    def _suggest_config(
        self, 
        similar_tasks: List[tuple]
    ) -> Dict[str, Any]:
        """基于相似任务建议配置"""
        if not similar_tasks:
            return {}
        
        # 获取最相似任务的配置
        most_similar = similar_tasks[0][0]
        support_examples = self.support_sets.get(most_similar, [])
        
        if not support_examples:
            return {}
        
        # 聚合配置
        configs = [ex.get('optimal_config', {}) for ex in support_examples]
        
        # 简单的多数投票
        suggested = {}
        for key in ['retrieval_depth', 'compression_ratio', 'reasoning_strategy']:
            values = [c.get(key) for c in configs if c.get(key) is not None]
            if values:
                # 对于数值取平均，对于类别取众数
                if isinstance(values[0], (int, float)):
                    suggested[key] = np.mean(values)
                else:
                    suggested[key] = max(set(values), key=values.count)
        
        return suggested
    
    def _gather_transfer_knowledge(
        self, 
        similar_tasks: List[tuple]
    ) -> List[Dict]:
        """收集可迁移的知识"""
        knowledge = []
        
        for task_name, similarity in similar_tasks:
            examples = self.support_sets.get(task_name, [])
            for ex in examples:
                if ex.get('successful', False):
                    knowledge.append({
                        'from_task': task_name,
                        'similarity': similarity,
                        'technique': ex.get('technique_used'),
                        'effectiveness': ex.get('performance', 0)
                    })
        
        # 按相似度和效果排序
        knowledge.sort(key=lambda x: x['similarity'] * x['effectiveness'], reverse=True)
        return knowledge[:5]


class HyperparameterOptimizer:
    """超参数优化器"""
    
    def __init__(self):
        self.param_space = {
            'retrieval_depth': {'type': 'int', 'min': 3, 'max': 20, 'default': 10},
            'compression_ratio': {'type': 'float', 'min': 0.3, 'max': 0.9, 'default': 0.7},
            'evidence_threshold': {'type': 'float', 'min': 0.5, 'max': 0.95, 'default': 0.7},
            'max_reasoning_steps': {'type': 'int', 'min': 3, 'max': 15, 'default': 7},
            'consistency_weight': {'type': 'float', 'min': 0.1, 'max': 1.0, 'default': 0.5}
        }
        self.performance_history: List[Dict] = []
        self.best_config: Dict[str, Any] = {}
    
    def optimize(
        self,
        evaluation_fn: Callable[[Dict], float],
        n_trials: int = 20,
        strategy: str = 'bayesian'
    ) -> Dict[str, Any]:
        """
        优化超参数
        
        Args:
            evaluation_fn: 评估函数，接收配置返回性能分数
            n_trials: 试验次数
            strategy: 优化策略
        
        Returns:
            优化结果
        """
        if strategy == 'random':
            return self._random_search(evaluation_fn, n_trials)
        elif strategy == 'grid':
            return self._grid_search(evaluation_fn)
        else:  # bayesian
            return self._bayesian_optimization(evaluation_fn, n_trials)
    
    def _random_search(
        self, 
        evaluation_fn: Callable[[Dict], float],
        n_trials: int
    ) -> Dict[str, Any]:
        """随机搜索"""
        best_score = -float('inf')
        best_config = {}
        all_results = []
        
        for trial in range(n_trials):
            config = self._sample_random_config()
            score = evaluation_fn(config)
            
            all_results.append({
                'trial': trial,
                'config': config,
                'score': score
            })
            
            if score > best_score:
                best_score = score
                best_config = config.copy()
        
        self.best_config = best_config
        
        return {
            'strategy': 'random_search',
            'trials': n_trials,
            'best_score': best_score,
            'best_config': best_config,
            'all_results': all_results,
            'improvement_over_default': self._calculate_improvement(best_score)
        }
    
    def _bayesian_optimization(
        self, 
        evaluation_fn: Callable[[Dict], float],
        n_trials: int
    ) -> Dict[str, Any]:
        """简化的贝叶斯优化"""
        # 简化的实现：使用历史信息指导搜索
        best_score = -float('inf')
        best_config = {}
        all_results = []
        
        # 前30%随机探索
        n_random = n_trials // 3
        for trial in range(n_random):
            config = self._sample_random_config()
            score = evaluation_fn(config)
            all_results.append({'trial': trial, 'config': config, 'score': score})
            
            if score > best_score:
                best_score = score
                best_config = config.copy()
        
        # 后70%基于历史最优进行局部搜索
        for trial in range(n_random, n_trials):
            if best_config:
                config = self._local_search_around(best_config)
            else:
                config = self._sample_random_config()
            
            score = evaluation_fn(config)
            all_results.append({'trial': trial, 'config': config, 'score': score})
            
            if score > best_score:
                best_score = score
                best_config = config.copy()
        
        self.best_config = best_config
        
        return {
            'strategy': 'bayesian_optimization',
            'trials': n_trials,
            'best_score': best_score,
            'best_config': best_config,
            'all_results': all_results,
            'improvement_over_default': self._calculate_improvement(best_score)
        }
    
    def _sample_random_config(self) -> Dict[str, Any]:
        """随机采样配置"""
        config = {}
        for param, spec in self.param_space.items():
            if spec['type'] == 'int':
                config[param] = random.randint(spec['min'], spec['max'])
            else:
                config[param] = random.uniform(spec['min'], spec['max'])
        return config
    
    def _local_search_around(
        self, 
        center_config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """在中心配置周围进行局部搜索"""
        config = {}
        for param, spec in self.param_space.items():
            center_val = center_config.get(param, spec['default'])
            
            if spec['type'] == 'int':
                # 在中心值附近随机扰动
                perturbation = random.randint(-2, 2)
                config[param] = max(spec['min'], min(spec['max'], center_val + perturbation))
            else:
                # 连续值的扰动
                range_size = spec['max'] - spec['min']
                perturbation = random.uniform(-0.1, 0.1) * range_size
                config[param] = max(spec['min'], min(spec['max'], center_val + perturbation))
        
        return config
    
    def _calculate_improvement(self, best_score: float) -> float:
        """计算相对于默认配置的改进"""
        default_config = {k: v['default'] for k, v in self.param_space.items()}
        # 这里简化处理，实际应该重新评估默认配置
        return round(best_score * 0.1, 4)  # 假设10%的改进


class StrategyLearner:
    """策略学习器"""
    
    def __init__(self):
        self.strategies: Dict[str, Dict] = {}
        self.strategy_performance: Dict[str, List[float]] = defaultdict(list)
        self.context_strategy_map: Dict[str, str] = {}
    
    def learn_strategy(
        self,
        strategy_name: str,
        context_features: Dict[str, Any],
        actions: List[str],
        outcome: float
    ) -> Dict[str, Any]:
        """
        学习策略
        
        Args:
            strategy_name: 策略名称
            context_features: 上下文特征
            actions: 执行的动作序列
            outcome: 结果（性能分数）
        
        Returns:
            学习结果
        """
        # 存储策略
        self.strategies[strategy_name] = {
            'context_features': context_features,
            'actions': actions,
            'success_rate': outcome,
            'usage_count': 1,
            'last_used': datetime.now()
        }
        
        self.strategy_performance[strategy_name].append(outcome)
        
        # 更新上下文-策略映射
        context_key = self._encode_context(context_features)
        if context_key not in self.context_strategy_map:
            self.context_strategy_map[context_key] = strategy_name
        else:
            # 比较性能，保留更好的策略
            current_best = self.context_strategy_map[context_key]
            current_avg = np.mean(self.strategy_performance[current_best])
            new_avg = np.mean(self.strategy_performance[strategy_name])
            
            if new_avg > current_avg:
                self.context_strategy_map[context_key] = strategy_name
        
        return {
            'strategy_learned': strategy_name,
            'context_key': context_key,
            'performance': outcome,
            'is_best_for_context': self.context_strategy_map[context_key] == strategy_name
        }
    
    def select_strategy(
        self,
        current_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        为当前上下文选择最佳策略
        
        Args:
            current_context: 当前上下文
        
        Returns:
            选择的策略
        """
        context_key = self._encode_context(current_context)
        
        # 检查是否有精确匹配
        if context_key in self.context_strategy_map:
            strategy_name = self.context_strategy_map[context_key]
            return {
                'strategy_name': strategy_name,
                'confidence': 0.9,
                'actions': self.strategies[strategy_name]['actions'],
                'expected_performance': np.mean(self.strategy_performance[strategy_name]),
                'match_type': 'exact'
            }
        
        # 寻找最相似的上下文
        best_match = None
        best_similarity = 0
        
        for known_context, strategy_name in self.context_strategy_map.items():
            similarity = self._context_similarity(context_key, known_context)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = strategy_name
        
        if best_match and best_similarity > 0.7:
            return {
                'strategy_name': best_match,
                'confidence': best_similarity,
                'actions': self.strategies[best_match]['actions'],
                'expected_performance': np.mean(self.strategy_performance[best_match]),
                'match_type': 'similar'
            }
        
        # 返回默认策略
        return {
            'strategy_name': 'default',
            'confidence': 0.5,
            'actions': ['retrieve', 'reason', 'generate'],
            'expected_performance': 0.7,
            'match_type': 'default'
        }
    
    def _encode_context(self, context: Dict[str, Any]) -> str:
        """编码上下文为字符串键"""
        # 简化的编码：连接关键特征
        key_parts = []
        
        if 'query_type' in context:
            key_parts.append(f"type:{context['query_type']}")
        if 'domain' in context:
            key_parts.append(f"domain:{context['domain']}")
        if 'complexity' in context:
            key_parts.append(f"complexity:{context['complexity']}")
        
        return '|'.join(key_parts) if key_parts else 'default'
    
    def _context_similarity(self, context1: str, context2: str) -> float:
        """计算上下文相似度"""
        parts1 = set(context1.split('|'))
        parts2 = set(context2.split('|'))
        
        if not parts1 or not parts2:
            return 0.0
        
        intersection = len(parts1 & parts2)
        union = len(parts1 | parts2)
        
        return intersection / union if union > 0 else 0.0


class ContinualLearner:
    """持续学习器"""
    
    def __init__(self):
        self.experience_buffer: List[Dict] = []
        self.learned_patterns: Dict[str, Any] = {}
        self.performance_trends: Dict[str, List[tuple]] = defaultdict(list)
        self.update_schedule = {
            'last_update': datetime.now(),
            'update_interval': timedelta(hours=1),
            'min_samples': 10
        }
    
    def accumulate_experience(self, experience: Dict):
        """积累交互经验"""
        experience['timestamp'] = datetime.now()
        self.experience_buffer.append(experience)
        
        # 限制缓冲区大小
        if len(self.experience_buffer) > 1000:
            self.experience_buffer = self.experience_buffer[-1000:]
    
    def should_update(self) -> bool:
        """检查是否应该更新模型"""
        now = datetime.now()
        time_since_update = now - self.update_schedule['last_update']
        
        return (
            time_since_update >= self.update_schedule['update_interval'] and
            len(self.experience_buffer) >= self.update_schedule['min_samples']
        )
    
    def continual_update(self) -> Dict[str, Any]:
        """执行持续更新"""
        if not self.should_update():
            return {
                'updated': False,
                'reason': 'Update conditions not met',
                'buffer_size': len(self.experience_buffer),
                'time_since_update': (datetime.now() - self.update_schedule['last_update']).total_seconds()
            }
        
        # 分析最近的经验
        recent_experiences = self.experience_buffer[-100:]
        
        # 提取新的模式
        new_patterns = self._extract_new_patterns(recent_experiences)
        
        # 更新性能趋势
        self._update_performance_trends(recent_experiences)
        
        # 检测性能退化
        degradation = self._detect_performance_degradation()
        
        # 更新学习到的模式
        self.learned_patterns.update(new_patterns)
        
        self.update_schedule['last_update'] = datetime.now()
        
        return {
            'updated': True,
            'patterns_learned': len(new_patterns),
            'new_patterns': new_patterns,
            'performance_degradation': degradation,
            'buffer_size': len(self.experience_buffer)
        }
    
    def _extract_new_patterns(self, experiences: List[Dict]) -> Dict[str, Any]:
        """从经验中提取新模式"""
        patterns = {}
        
        # 分析成功和失败的案例
        successful = [e for e in experiences if e.get('success', False)]
        failed = [e for e in experiences if not e.get('success', False)]
        
        # 提取成功模式
        if successful:
            common_features = self._find_common_features(successful)
            patterns['success_indicators'] = common_features
        
        # 提取失败模式
        if failed:
            common_features = self._find_common_features(failed)
            patterns['failure_indicators'] = common_features
        
        return patterns
    
    def _find_common_features(self, experiences: List[Dict]) -> List[str]:
        """找到共同特征"""
        feature_counts = defaultdict(int)
        
        for exp in experiences:
            features = exp.get('features', {})
            for key, value in features.items():
                feature_str = f"{key}:{value}"
                feature_counts[feature_str] += 1
        
        # 返回出现频率高的特征
        threshold = len(experiences) * 0.6
        common = [f for f, count in feature_counts.items() if count >= threshold]
        
        return common[:10]
    
    def _update_performance_trends(self, experiences: List[Dict]):
        """更新性能趋势"""
        for exp in experiences:
            domain = exp.get('domain', 'general')
            performance = exp.get('performance', 0)
            timestamp = exp.get('timestamp', datetime.now())
            
            self.performance_trends[domain].append((timestamp, performance))
    
    def _detect_performance_degradation(self) -> Dict[str, Any]:
        """检测性能退化"""
        degradation_alerts = []
        
        for domain, trends in self.performance_trends.items():
            if len(trends) < 10:
                continue
            
            # 比较最近和早期的性能
            recent = [p for t, p in trends[-10:]]
            earlier = [p for t, p in trends[-20:-10]] if len(trends) >= 20 else [p for t, p in trends[:10]]
            
            recent_avg = np.mean(recent)
            earlier_avg = np.mean(earlier)
            
            if recent_avg < earlier_avg * 0.9:  # 10%退化
                degradation_alerts.append({
                    'domain': domain,
                    'recent_avg': recent_avg,
                    'earlier_avg': earlier_avg,
                    'degradation': (earlier_avg - recent_avg) / earlier_avg
                })
        
        return {
            'has_degradation': len(degradation_alerts) > 0,
            'alerts': degradation_alerts
        }


# 主引擎类
class MetaLearningEngine:
    """元学习引擎"""
    
    def __init__(self):
        self.few_shot_learner = FewShotLearner()
        self.hyperparameter_optimizer = HyperparameterOptimizer()
        self.strategy_learner = StrategyLearner()
        self.continual_learner = ContinualLearner()
    
    def meta_learn(
        self,
        task_description: Dict[str, Any],
        examples: List[Dict],
        embedding_fn: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """
        执行元学习
        
        这是V112创新的主要入口函数
        """
        results = {
            'version': 'V112',
            'name': 'Meta-Learning & Adaptive Optimization (MeLAO)',
            'components': {}
        }
        
        # 1. 少样本学习
        if examples and embedding_fn:
            few_shot_result = self.few_shot_learner.learn_from_few_examples(
                task_description.get('name', 'unknown_task'),
                examples,
                embedding_fn
            )
            results['components']['few_shot_learning'] = few_shot_result
        
        # 2. 策略学习
        if 'strategy' in task_description:
            strategy_result = self.strategy_learner.learn_strategy(
                task_description['strategy']['name'],
                task_description.get('context', {}),
                task_description['strategy'].get('actions', []),
                task_description['strategy'].get('outcome', 0)
            )
            results['components']['strategy_learning'] = strategy_result
        
        # 3. 持续学习更新
        continual_result = self.continual_learner.continual_update()
        results['components']['continual_learning'] = continual_result
        
        # 4. 生成元知识总结
        results['meta_knowledge'] = self._summarize_meta_knowledge()
        
        return results
    
    def adapt_to_task(
        self,
        query: str,
        context: Dict[str, Any],
        embedding_fn: Optional[Callable] = None
    ) -> Dict[str, Any]:
        """适应新任务"""
        adaptation = {}
        
        # 少样本适应
        if embedding_fn:
            few_shot_adaptation = self.few_shot_learner.adapt_to_new_task(
                query, 
                embedding_fn
            )
            adaptation['few_shot_adaptation'] = few_shot_adaptation
        
        # 策略选择
        strategy_selection = self.strategy_learner.select_strategy(context)
        adaptation['strategy_selection'] = strategy_selection
        
        return adaptation
    
    def _summarize_meta_knowledge(self) -> Dict[str, Any]:
        """总结元知识"""
        return {
            'learned_tasks': len(self.few_shot_learner.task_prototypes),
            'learned_strategies': len(self.strategy_learner.strategies),
            'experience_accumulated': len(self.continual_learner.experience_buffer),
            'performance_domains': list(self.continual_learner.performance_trends.keys())
        }


# 便捷函数
def build_meta_learning_optimization(
    task_description: Dict[str, Any],
    examples: Optional[List[Dict]] = None,
    embedding_fn: Optional[Callable] = None
) -> Dict[str, Any]:
    """
    构建元学习优化
    
    这是V112创新的便捷入口函数
    """
    engine = MetaLearningEngine()
    return engine.meta_learn(task_description, examples or [], embedding_fn)


# 示例用法
if __name__ == "__main__":
    # 示例任务描述
    task = {
        'name': 'medical_qa',
        'domain': 'healthcare',
        'context': {
            'query_type': 'factual',
            'domain': 'medical',
            'complexity': 'high'
        },
        'strategy': {
            'name': 'evidence_based_reasoning',
            'actions': ['retrieve_medical_literature', 'verify_facts', 'generate_answer'],
            'outcome': 0.85
        }
    }
    
    # 示例数据
    examples = [
        {
            'query': 'What are the symptoms of diabetes?',
            'context': 'medical',
            'optimal_config': {'retrieval_depth': 15, 'consistency_weight': 0.8},
            'successful': True
        },
        {
            'query': 'How does insulin work?',
            'context': 'medical',
            'optimal_config': {'retrieval_depth': 12, 'consistency_weight': 0.7},
            'successful': True
        }
    ]
    
    # 简化的嵌入函数
    def mock_embedding(text: str) -> np.ndarray:
        return np.random.randn(128)
    
    result = build_meta_learning_optimization(task, examples, mock_embedding)
    print(json.dumps(result, indent=2, default=str))
