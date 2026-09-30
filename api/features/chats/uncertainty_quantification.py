"""
V108: Uncertainty Quantification & Confidence Calibration (UQCC)
不确定性量化与置信度校准 - 为每个答案提供统计置信区间

创新点：
1. 贝叶斯置信度估计：基于证据质量和数量的统计置信度
2. 置信区间生成：为数值答案提供置信区间
3. 认知不确定性建模：区分偶然不确定性和认知不确定性
4. 动态置信度更新：随着新证据到来更新置信度
"""

import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from scipy import stats
import json


@dataclass
class UncertaintyEstimate:
    """不确定性估计数据结构"""
    mean: float
    std: float
    confidence_interval: Tuple[float, float]
    confidence_level: float
    epistemic_uncertainty: float  # 认知不确定性（模型知识不足）
    aleatoric_uncertainty: float  # 偶然不确定性（数据固有噪声）
    total_uncertainty: float
    sample_size: int
    reliability_score: float


@dataclass
class CalibratedConfidence:
    """校准后的置信度数据结构"""
    raw_confidence: float
    calibrated_confidence: float
    calibration_method: str
    expected_accuracy: float
    is_well_calibrated: bool
    calibration_error: float


class BayesianConfidenceEstimator:
    """贝叶斯置信度估计器"""
    
    def __init__(self, prior_strength: float = 1.0):
        self.prior_strength = prior_strength
        self.posterior_history: List[Dict] = []
    
    def estimate_confidence(
        self,
        evidence: List[Dict],
        query_type: str = 'factual'
    ) -> UncertaintyEstimate:
        """
        基于贝叶斯方法估计置信度
        
        Args:
            evidence: 证据列表
            query_type: 查询类型
        
        Returns:
            UncertaintyEstimate: 不确定性估计
        """
        if not evidence:
            return self._default_uncertainty()
        
        # 提取证据质量分数
        quality_scores = [self._extract_quality_score(ev) for ev in evidence]
        
        # 计算先验分布参数
        prior_mean = 0.5
        prior_var = 0.25
        
        # 计算似然
        likelihood_mean = np.mean(quality_scores)
        likelihood_var = np.var(quality_scores) if len(quality_scores) > 1 else 0.1
        
        # 贝叶斯更新
        n = len(quality_scores)
        posterior_var = 1 / (1/prior_var + n/likelihood_var) if likelihood_var > 0 else prior_var
        posterior_mean = posterior_var * (prior_mean/prior_var + n*likelihood_mean/likelihood_var) if likelihood_var > 0 else prior_mean
        
        # 计算置信区间
        confidence_level = 0.95
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        margin_error = z_score * np.sqrt(posterior_var)
        ci_lower = max(0, posterior_mean - margin_error)
        ci_upper = min(1, posterior_mean + margin_error)
        
        # 分解不确定性
        epistemic = self._calculate_epistemic_uncertainty(evidence, posterior_var)
        aleatoric = self._calculate_aleatoric_uncertainty(quality_scores)
        total_unc = np.sqrt(epistemic**2 + aleatoric**2)
        
        # 计算可靠性分数
        reliability = self._calculate_reliability(evidence, posterior_mean, posterior_var)
        
        estimate = UncertaintyEstimate(
            mean=posterior_mean,
            std=np.sqrt(posterior_var),
            confidence_interval=(ci_lower, ci_upper),
            confidence_level=confidence_level,
            epistemic_uncertainty=epistemic,
            aleatoric_uncertainty=aleatoric,
            total_uncertainty=total_unc,
            sample_size=n,
            reliability_score=reliability
        )
        
        self.posterior_history.append({
            'mean': posterior_mean,
            'var': posterior_var,
            'evidence_count': n
        })
        
        return estimate
    
    def _extract_quality_score(self, evidence: Dict) -> float:
        """从证据中提取质量分数"""
        # 综合多个因素
        scores = []
        
        # 来源可靠性
        source_reliability = evidence.get('source_reliability', 0.7)
        scores.append(source_reliability)
        
        # 相关性分数
        relevance = evidence.get('relevance_score', 0.5)
        scores.append(relevance)
        
        # 时效性
        recency = evidence.get('recency_score', 0.8)
        scores.append(recency)
        
        # 一致性
        consistency = evidence.get('consistency_score', 0.9)
        scores.append(consistency)
        
        return np.mean(scores)
    
    def _calculate_epistemic_uncertainty(
        self, 
        evidence: List[Dict], 
        posterior_var: float
    ) -> float:
        """计算认知不确定性（可以通过更多数据减少）"""
        # 基于证据数量和多样性
        n = len(evidence)
        
        # 证据多样性
        sources = set(ev.get('source', 'unknown') for ev in evidence)
        diversity = len(sources) / max(n, 1)
        
        # 认知不确定性随证据增加而减少
        epistemic = posterior_var * (1 + 1/max(n, 1)) * (2 - diversity)
        
        return min(epistemic, 1.0)
    
    def _calculate_aleatoric_uncertainty(self, quality_scores: List[float]) -> float:
        """计算偶然不确定性（数据固有噪声）"""
        if len(quality_scores) < 2:
            return 0.3  # 默认噪声水平
        
        # 使用分数的标准差作为偶然不确定性
        aleatoric = np.std(quality_scores)
        
        # 考虑分数分布的偏度
        if len(quality_scores) > 3:
            skewness = stats.skew(quality_scores)
            aleatoric *= (1 + abs(skewness) * 0.1)
        
        return min(aleatoric, 1.0)
    
    def _calculate_reliability(
        self, 
        evidence: List[Dict], 
        posterior_mean: float, 
        posterior_var: float
    ) -> float:
        """计算可靠性分数"""
        # 基于证据数量、质量和置信度
        n = len(evidence)
        
        # 证据数量因子
        count_factor = min(n / 10, 1.0)  # 最多10个证据达到满分
        
        # 置信度因子（中等置信度最好）
        confidence_factor = 1 - abs(posterior_mean - 0.7) / 0.7
        
        # 不确定性因子（低不确定性更好）
        uncertainty_factor = 1 - np.sqrt(posterior_var)
        
        reliability = (count_factor * 0.3 + 
                      confidence_factor * 0.4 + 
                      uncertainty_factor * 0.3)
        
        return min(reliability, 1.0)
    
    def _default_uncertainty(self) -> UncertaintyEstimate:
        """默认不确定性（无证据时）"""
        return UncertaintyEstimate(
            mean=0.5,
            std=0.5,
            confidence_interval=(0.0, 1.0),
            confidence_level=0.95,
            epistemic_uncertainty=0.5,
            aleatoric_uncertainty=0.3,
            total_uncertainty=0.58,
            sample_size=0,
            reliability_score=0.3
        )


class ConfidenceCalibrator:
    """置信度校准器"""
    
    def __init__(self):
        self.calibration_history: List[Dict] = []
        self.temperature: float = 1.0  # 温度参数用于温度缩放
    
    def calibrate(
        self, 
        raw_confidence: float, 
        evidence: List[Dict],
        method: str = 'temperature_scaling'
    ) -> CalibratedConfidence:
        """
        校准置信度
        
        Args:
            raw_confidence: 原始置信度
            evidence: 证据列表
            method: 校准方法
        
        Returns:
            CalibratedConfidence: 校准后的置信度
        """
        if method == 'temperature_scaling':
            calibrated = self._temperature_scaling(raw_confidence)
        elif method == 'platt_scaling':
            calibrated = self._platt_scaling(raw_confidence, evidence)
        elif method == 'isotonic':
            calibrated = self._isotonic_regression(raw_confidence, evidence)
        else:
            calibrated = raw_confidence
        
        # 计算期望准确率
        expected_accuracy = self._estimate_expected_accuracy(calibrated, evidence)
        
        # 计算校准误差
        calibration_error = abs(calibrated - expected_accuracy)
        
        # 判断是否良好校准
        is_well_calibrated = calibration_error < 0.1
        
        result = CalibratedConfidence(
            raw_confidence=raw_confidence,
            calibrated_confidence=calibrated,
            calibration_method=method,
            expected_accuracy=expected_accuracy,
            is_well_calibrated=is_well_calibrated,
            calibration_error=calibration_error
        )
        
        self.calibration_history.append({
            'raw': raw_confidence,
            'calibrated': calibrated,
            'method': method,
            'error': calibration_error
        })
        
        return result
    
    def _temperature_scaling(self, confidence: float) -> float:
        """温度缩放校准"""
        # 使用温度参数软化置信度
        # 高温度 = 更保守的置信度
        logits = np.log(confidence + 1e-10) - np.log(1 - confidence + 1e-10)
        scaled_logits = logits / self.temperature
        calibrated = 1 / (1 + np.exp(-scaled_logits))
        return float(calibrated)
    
    def _platt_scaling(
        self, 
        confidence: float, 
        evidence: List[Dict]
    ) -> float:
        """Platt缩放（逻辑回归校准）"""
        # 简化的Platt缩放
        # 在实际应用中，这需要训练数据来学习参数
        a = 1.0  # 斜率参数
        b = 0.0  # 偏移参数
        
        # 根据证据质量调整参数
        if evidence:
            avg_quality = np.mean([ev.get('quality', 0.7) for ev in evidence])
            a = 0.8 + avg_quality * 0.4
            b = (0.5 - avg_quality) * 0.2
        
        logits = np.log(confidence + 1e-10) - np.log(1 - confidence + 1e-10)
        scaled_logits = a * logits + b
        calibrated = 1 / (1 + np.exp(-scaled_logits))
        return float(calibrated)
    
    def _isotonic_regression(
        self, 
        confidence: float, 
        evidence: List[Dict]
    ) -> float:
        """保序回归校准"""
        # 简化的保序回归
        # 使用单调变换
        if confidence < 0.5:
            # 对低置信度进行提升
            calibrated = confidence * 1.2
        elif confidence > 0.9:
            # 对过高置信度进行抑制
            calibrated = 0.9 + (confidence - 0.9) * 0.5
        else:
            calibrated = confidence
        
        return min(calibrated, 1.0)
    
    def _estimate_expected_accuracy(
        self, 
        confidence: float, 
        evidence: List[Dict]
    ) -> float:
        """估计期望准确率"""
        # 基于置信度和证据质量估计期望准确率
        base_accuracy = confidence
        
        # 证据质量调整
        if evidence:
            quality_boost = np.mean([ev.get('quality', 0.7) for ev in evidence]) * 0.1
            base_accuracy += quality_boost
        
        # 历史校准调整
        if self.calibration_history:
            recent_errors = [h['error'] for h in self.calibration_history[-10:]]
            avg_error = np.mean(recent_errors)
            base_accuracy -= avg_error * 0.5
        
        return min(max(base_accuracy, 0), 1)


class UncertaintyQuantificationEngine:
    """不确定性量化引擎"""
    
    def __init__(self):
        self.bayesian_estimator = BayesianConfidenceEstimator()
        self.confidence_calibrator = ConfidenceCalibrator()
    
    def quantify_uncertainty(
        self,
        answer: str,
        evidence: List[Dict],
        query_type: str = 'factual'
    ) -> Dict[str, Any]:
        """
        量化答案的不确定性
        
        Args:
            answer: 生成的答案
            evidence: 支持证据
            query_type: 查询类型
        
        Returns:
            包含不确定性量化结果的字典
        """
        # 1. 贝叶斯置信度估计
        uncertainty = self.bayesian_estimator.estimate_confidence(evidence, query_type)
        
        # 2. 置信度校准
        raw_confidence = uncertainty.mean
        calibrated = self.confidence_calibrator.calibrate(
            raw_confidence, 
            evidence, 
            method='temperature_scaling'
        )
        
        # 3. 为数值答案生成置信区间
        numerical_ci = None
        if self._is_numerical_answer(answer):
            numerical_ci = self._generate_numerical_ci(answer, uncertainty)
        
        # 4. 生成不确定性报告
        report = self._generate_uncertainty_report(
            uncertainty, 
            calibrated, 
            numerical_ci
        )
        
        return {
            'version': 'V108',
            'name': 'Uncertainty Quantification & Confidence Calibration',
            'uncertainty_estimate': {
                'mean_confidence': round(uncertainty.mean, 4),
                'standard_deviation': round(uncertainty.std, 4),
                'confidence_interval': {
                    'lower': round(uncertainty.confidence_interval[0], 4),
                    'upper': round(uncertainty.confidence_interval[1], 4),
                    'level': uncertainty.confidence_level
                },
                'epistemic_uncertainty': round(uncertainty.epistemic_uncertainty, 4),
                'aleatoric_uncertainty': round(uncertainty.aleatoric_uncertainty, 4),
                'total_uncertainty': round(uncertainty.total_uncertainty, 4),
                'sample_size': uncertainty.sample_size,
                'reliability_score': round(uncertainty.reliability_score, 4)
            },
            'calibrated_confidence': {
                'raw': round(calibrated.raw_confidence, 4),
                'calibrated': round(calibrated.calibrated_confidence, 4),
                'method': calibrated.calibration_method,
                'expected_accuracy': round(calibrated.expected_accuracy, 4),
                'is_well_calibrated': calibrated.is_well_calibrated,
                'calibration_error': round(calibrated.calibration_error, 4)
            },
            'numerical_confidence_interval': numerical_ci,
            'uncertainty_report': report,
            'recommendations': self._generate_recommendations(uncertainty, calibrated)
        }
    
    def _is_numerical_answer(self, answer: str) -> bool:
        """检查答案是否包含数值"""
        import re
        # 查找数字模式
        patterns = [
            r'\d+\.?\d*\s*(?:%|percent)',
            r'\d+\.?\d*\s*(?:million|billion|thousand)',
            r'\$\d+\.?\d*',
            r'\d+\.?\d*\s*(?:years?|months?|days?)',
        ]
        return any(re.search(pattern, answer, re.IGNORECASE) for pattern in patterns)
    
    def _generate_numerical_ci(
        self, 
        answer: str, 
        uncertainty: UncertaintyEstimate
    ) -> Dict[str, Any]:
        """为数值答案生成置信区间"""
        import re
        
        # 提取数值
        numbers = re.findall(r'\d+\.?\d*', answer)
        if not numbers:
            return None
        
        main_number = float(numbers[0])
        
        # 基于不确定性计算置信区间
        margin = main_number * uncertainty.total_uncertainty
        
        return {
            'original_value': main_number,
            'unit': self._extract_unit(answer),
            'confidence_interval': {
                'lower': round(main_number - margin, 2),
                'upper': round(main_number + margin, 2)
            },
            'confidence_level': uncertainty.confidence_level,
            'interpretation': f"We are {uncertainty.confidence_level*100:.0f}% confident the true value lies between {main_number - margin:.2f} and {main_number + margin:.2f}"
        }
    
    def _extract_unit(self, answer: str) -> str:
        """提取单位"""
        import re
        unit_patterns = [
            r'(?:\d+\.?\d*)\s*(%|percent)',
            r'(?:\d+\.?\d*)\s*(million|billion|thousand)',
            r'\$\s*(?:\d+\.?\d*)',
            r'(?:\d+\.?\d*)\s*(years?|months?|days?|hours?|minutes?)',
        ]
        for pattern in unit_patterns:
            match = re.search(pattern, answer, re.IGNORECASE)
            if match:
                return match.group(1)
        return 'unknown'
    
    def _generate_uncertainty_report(
        self,
        uncertainty: UncertaintyEstimate,
        calibrated: CalibratedConfidence,
        numerical_ci: Optional[Dict]
    ) -> str:
        """生成不确定性报告"""
        report_parts = []
        
        # 总体评估
        if calibrated.is_well_calibrated:
            report_parts.append("✓ The confidence estimate is well-calibrated.")
        else:
            report_parts.append("⚠ The confidence estimate may be over- or under-confident.")
        
        # 不确定性分解
        if uncertainty.epistemic_uncertainty > uncertainty.aleatoric_uncertainty:
            report_parts.append(
                "The primary source of uncertainty is epistemic (lack of knowledge), "
                "which can be reduced by gathering more evidence."
            )
        else:
            report_parts.append(
                "The primary source of uncertainty is aleatoric (inherent randomness), "
                "which represents fundamental unpredictability in the data."
            )
        
        # 可靠性评估
        if uncertainty.reliability_score > 0.8:
            report_parts.append("The evidence base is highly reliable.")
        elif uncertainty.reliability_score > 0.5:
            report_parts.append("The evidence base has moderate reliability.")
        else:
            report_parts.append("The evidence base has low reliability - exercise caution.")
        
        return " ".join(report_parts)
    
    def _generate_recommendations(
        self,
        uncertainty: UncertaintyEstimate,
        calibrated: CalibratedConfidence
    ) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        if uncertainty.epistemic_uncertainty > 0.3:
            recommendations.append(
                "Gather additional evidence from diverse sources to reduce epistemic uncertainty"
            )
        
        if not calibrated.is_well_calibrated:
            recommendations.append(
                "Review confidence calibration - consider adjusting confidence estimates"
            )
        
        if uncertainty.sample_size < 5:
            recommendations.append(
                f"Current sample size ({uncertainty.sample_size}) is small - seek more supporting evidence"
            )
        
        if uncertainty.reliability_score < 0.5:
            recommendations.append(
                "Evidence quality is low - verify sources and check for potential biases"
            )
        
        if not recommendations:
            recommendations.append("Current uncertainty level is acceptable for most applications")
        
        return recommendations


# 便捷函数
def quantify_answer_uncertainty(
    answer: str,
    evidence: List[Dict],
    query_type: str = 'factual'
) -> Dict[str, Any]:
    """
    量化答案的不确定性
    
    这是V108创新的主要入口函数
    """
    engine = UncertaintyQuantificationEngine()
    return engine.quantify_uncertainty(answer, evidence, query_type)


# 示例用法
if __name__ == "__main__":
    answer = "The transformer architecture has achieved 95% accuracy on the benchmark."
    evidence = [
        {
            'content': 'BERT achieved 94.9% on SQuAD',
            'source_reliability': 0.95,
            'relevance_score': 0.9,
            'recency_score': 0.8,
            'consistency_score': 0.95,
            'quality': 0.9
        },
        {
            'content': 'GPT-3 achieved 95.2% on similar tasks',
            'source_reliability': 0.9,
            'relevance_score': 0.85,
            'recency_score': 0.9,
            'consistency_score': 0.9,
            'quality': 0.85
        }
    ]
    
    result = quantify_answer_uncertainty(answer, evidence)
    print(json.dumps(result, indent=2))
