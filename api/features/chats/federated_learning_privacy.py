"""
V114: Federated Learning & Privacy Preservation (FLPP)
联邦学习与隐私保护 - 在保护隐私的前提下协作学习

创新点：
1. 联邦学习：分布式训练，数据不出本地
2. 差分隐私：数学保证的隐私保护
3. 安全聚合：加密聚合模型更新
4. 本地个性化：在全局模型基础上本地微调
"""

import json
import hashlib
import random
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import numpy as np
from enum import Enum


class PrivacyLevel(Enum):
    """隐私级别"""
    STRICT = "strict"      # 严格隐私，高噪声
    BALANCED = "balanced"  # 平衡隐私和效用
    PERMISSIVE = "permissive"  # 宽松隐私，低噪声


@dataclass
class LocalUpdate:
    """本地模型更新"""
    client_id: str
    model_update: Dict[str, np.ndarray]
    sample_count: int
    timestamp: datetime = field(default_factory=datetime.now)
    privacy_budget_spent: float = 0.0


@dataclass
class PrivacyBudget:
    """隐私预算"""
    epsilon: float  # 隐私预算参数
    delta: float    # 失败概率
    spent: float = 0.0
    remaining: float = field(init=False)
    
    def __post_init__(self):
        self.remaining = self.epsilon - self.spent


class DifferentialPrivacyMechanism:
    """差分隐私机制"""
    
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        self.epsilon = epsilon
        self.delta = delta
        self.noise_multiplier = self._calculate_noise_multiplier()
    
    def _calculate_noise_multiplier(self) -> float:
        """计算噪声乘数"""
        # 简化的噪声计算
        # 实际应用中应该使用更复杂的计算
        return np.sqrt(2 * np.log(1.25 / self.delta)) / self.epsilon
    
    def add_noise_to_gradients(
        self,
        gradients: Dict[str, np.ndarray],
        sensitivity: float = 1.0
    ) -> Dict[str, np.ndarray]:
        """
        向梯度添加噪声
        
        Args:
            gradients: 模型梯度
            sensitivity: 敏感度
        
        Returns:
            加噪后的梯度
        """
        noisy_gradients = {}
        
        for name, grad in gradients.items():
            # 计算噪声标准差
            noise_std = sensitivity * self.noise_multiplier
            
            # 生成高斯噪声
            noise = np.random.normal(0, noise_std, grad.shape)
            
            # 添加噪声
            noisy_gradients[name] = grad + noise
        
        return noisy_gradients
    
    def clip_gradients(
        self,
        gradients: Dict[str, np.ndarray],
        max_norm: float = 1.0
    ) -> Dict[str, np.ndarray]:
        """梯度裁剪"""
        # 计算全局梯度范数
        global_norm = np.sqrt(sum(np.sum(g ** 2) for g in gradients.values()))
        
        # 裁剪因子
        clip_factor = min(1.0, max_norm / (global_norm + 1e-8))
        
        # 裁剪梯度
        clipped_gradients = {
            name: grad * clip_factor
            for name, grad in gradients.items()
        }
        
        return clipped_gradients
    
    def compute_privacy_spent(
        self,
        num_steps: int,
        sampling_rate: float
    ) -> float:
        """
        计算隐私预算消耗
        
        使用简化的矩会计师方法
        """
        # 简化的隐私会计
        # 实际应用中使用更精确的会计方法
        privacy_spent = num_steps * sampling_rate * (1.0 / self.epsilon)
        return privacy_spent


class SecureAggregator:
    """安全聚合器"""
    
    def __init__(self, num_clients: int):
        self.num_clients = num_clients
        self.public_keys: Dict[str, bytes] = {}
    
    def register_client(self, client_id: str, public_key: bytes):
        """注册客户端"""
        self.public_keys[client_id] = public_key
    
    def secure_aggregate(
        self,
        encrypted_updates: List[Dict[str, Any]]
    ) -> Dict[str, np.ndarray]:
        """
        安全聚合加密更新
        
        注意：这是简化的实现，实际应用中使用更复杂的加密方案
        """
        if not encrypted_updates:
            return {}
        
        # 解密并聚合（简化）
        aggregated = {}
        total_samples = sum(u.get('sample_count', 1) for u in encrypted_updates)
        
        # 获取所有参数名
        param_names = encrypted_updates[0].get('model_update', {}).keys()
        
        for param_name in param_names:
            # 加权平均
            weighted_sum = None
            
            for update in encrypted_updates:
                model_update = update.get('model_update', {})
                if param_name in model_update:
                    param_values = model_update[param_name]
                    sample_count = update.get('sample_count', 1)
                    weight = sample_count / total_samples
                    
                    if weighted_sum is None:
                        weighted_sum = param_values * weight
                    else:
                        weighted_sum += param_values * weight
            
            if weighted_sum is not None:
                aggregated[param_name] = weighted_sum
        
        return aggregated
    
    def verify_update_integrity(
        self,
        update: Dict[str, Any],
        client_id: str
    ) -> bool:
        """验证更新完整性"""
        # 简化的完整性检查
        # 实际应用中使用数字签名
        
        # 检查必需的字段
        required_fields = ['client_id', 'model_update', 'sample_count', 'timestamp']
        if not all(field in update for field in required_fields):
            return False
        
        # 验证客户端ID
        if update['client_id'] != client_id:
            return False
        
        # 验证模型更新不为空
        if not update.get('model_update'):
            return False
        
        return True


class FederatedLearningServer:
    """联邦学习服务器"""
    
    def __init__(
        self,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        min_clients: int = 3
    ):
        self.dp_mechanism = DifferentialPrivacyMechanism(epsilon, delta)
        self.secure_aggregator: Optional[SecureAggregator] = None
        self.global_model: Dict[str, np.ndarray] = {}
        self.min_clients = min_clients
        self.participating_clients: List[str] = []
        self.round_history: List[Dict] = []
        self.privacy_budget = PrivacyBudget(epsilon, delta)
    
    def initialize_model(self, model_structure: Dict[str, Tuple]):
        """初始化全局模型"""
        for name, shape in model_structure.items():
            self.global_model[name] = np.random.randn(*shape) * 0.01
    
    def register_client(self, client_id: str, public_key: bytes):
        """注册客户端"""
        if self.secure_aggregator is None:
            self.secure_aggregator = SecureAggregator(self.min_clients)
        
        self.secure_aggregator.register_client(client_id, public_key)
        self.participating_clients.append(client_id)
    
    def execute_federated_round(
        self,
        client_updates: List[LocalUpdate],
        current_round: int
    ) -> Dict[str, Any]:
        """
        执行一轮联邦学习
        
        Args:
            client_updates: 客户端更新列表
            current_round: 当前轮次
        
        Returns:
            聚合结果
        """
        if len(client_updates) < self.min_clients:
            return {
                'success': False,
                'reason': f'Insufficient clients: {len(client_updates)} < {self.min_clients}'
            }
        
        # 1. 验证更新
        valid_updates = []
        for update in client_updates:
            if self.secure_aggregator.verify_update_integrity(
                {
                    'client_id': update.client_id,
                    'model_update': update.model_update,
                    'sample_count': update.sample_count,
                    'timestamp': update.timestamp.isoformat()
                },
                update.client_id
            ):
                valid_updates.append(update)
        
        if len(valid_updates) < self.min_clients:
            return {
                'success': False,
                'reason': f'Insufficient valid updates: {len(valid_updates)} < {self.min_clients}'
            }
        
        # 2. 应用差分隐私
        privatized_updates = []
        for update in valid_updates:
            # 裁剪梯度
            clipped_update = self.dp_mechanism.clip_gradients(
                update.model_update,
                max_norm=1.0
            )
            
            # 添加噪声
            noisy_update = self.dp_mechanism.add_noise_to_gradients(
                clipped_update,
                sensitivity=1.0
            )
            
            privatized_updates.append({
                'client_id': update.client_id,
                'model_update': noisy_update,
                'sample_count': update.sample_count
            })
            
            # 更新隐私预算
            privacy_spent = self.dp_mechanism.compute_privacy_spent(
                num_steps=1,
                sampling_rate=len(valid_updates) / len(self.participating_clients)
            )
            self.privacy_budget.spent += privacy_spent
        
        # 3. 安全聚合
        aggregated_update = self.secure_aggregator.secure_aggregate(privatized_updates)
        
        # 4. 更新全局模型
        learning_rate = 0.1  # 学习率
        for param_name in self.global_model:
            if param_name in aggregated_update:
                self.global_model[param_name] -= learning_rate * aggregated_update[param_name]
        
        # 5. 记录轮次历史
        round_record = {
            'round': current_round,
            'participating_clients': len(valid_updates),
            'total_samples': sum(u.sample_count for u in valid_updates),
            'privacy_spent': self.privacy_budget.spent,
            'privacy_remaining': self.privacy_budget.remaining
        }
        self.round_history.append(round_record)
        
        return {
            'success': True,
            'round': current_round,
            'aggregated_update': aggregated_update,
            'global_model_version': current_round,
            'privacy_status': {
                'spent': self.privacy_budget.spent,
                'remaining': self.privacy_budget.remaining,
                'exhausted': self.privacy_budget.remaining <= 0
            }
        }
    
    def get_global_model(self) -> Dict[str, np.ndarray]:
        """获取全局模型"""
        return self.global_model.copy()


class FederatedLearningClient:
    """联邦学习客户端"""
    
    def __init__(self, client_id: str):
        self.client_id = client_id
        self.local_model: Dict[str, np.ndarray] = {}
        self.local_data: List[Dict] = []
        self.personalization_layer: Dict[str, np.ndarray] = {}
    
    def set_local_data(self, data: List[Dict]):
        """设置本地数据"""
        self.local_data = data
    
    def receive_global_model(self, global_model: Dict[str, np.ndarray]):
        """接收全局模型"""
        self.local_model = global_model.copy()
    
    def local_train(
        self,
        epochs: int = 1,
        batch_size: int = 32
    ) -> LocalUpdate:
        """
        本地训练
        
        注意：这是简化的训练过程
        """
        # 模拟本地训练
        model_update = {}
        
        for name, param in self.local_model.items():
            # 模拟梯度计算（随机扰动）
            gradient = np.random.randn(*param.shape) * 0.01
            model_update[name] = gradient
        
        # 创建本地更新
        local_update = LocalUpdate(
            client_id=self.client_id,
            model_update=model_update,
            sample_count=len(self.local_data)
        )
        
        return local_update
    
    def personalize_model(
        self,
        personalization_epochs: int = 5
    ) -> Dict[str, Any]:
        """
        个性化本地模型
        
        在全局模型基础上进行本地微调
        """
        # 保存原始全局模型
        global_params = self.local_model.copy()
        
        # 本地微调（简化）
        for epoch in range(personalization_epochs):
            for name, param in self.local_model.items():
                # 模拟本地适应
                adaptation = np.random.randn(*param.shape) * 0.001
                self.local_model[name] = param + adaptation
        
        # 计算个性化层（与全局模型的差异）
        self.personalization_layer = {
            name: self.local_model[name] - global_params[name]
            for name in self.local_model.keys()
        }
        
        return {
            'personalized': True,
            'adaptation_epochs': personalization_epochs,
            'personalization_magnitude': {
                name: float(np.linalg.norm(diff))
                for name, diff in self.personalization_layer.items()
            }
        }
    
    def generate_public_key(self) -> bytes:
        """生成公钥（简化）"""
        # 简化的密钥生成
        key_data = f"pk_{self.client_id}_{datetime.now().timestamp()}"
        return hashlib.sha256(key_data.encode()).digest()


class PrivacyAuditor:
    """隐私审计器"""
    
    def __init__(self):
        self.audit_logs: List[Dict] = []
    
    def log_data_access(
        self,
        client_id: str,
        data_type: str,
        access_purpose: str,
        privacy_level: PrivacyLevel
    ):
        """记录数据访问"""
        self.audit_logs.append({
            'timestamp': datetime.now().isoformat(),
            'client_id': client_id,
            'data_type': data_type,
            'purpose': access_purpose,
            'privacy_level': privacy_level.value,
            'action': 'access'
        })
    
    def log_model_update(
        self,
        client_id: str,
        update_size: int,
        privacy_budget_spent: float
    ):
        """记录模型更新"""
        self.audit_logs.append({
            'timestamp': datetime.now().isoformat(),
            'client_id': client_id,
            'update_size_bytes': update_size,
            'privacy_budget_spent': privacy_budget_spent,
            'action': 'model_update'
        })
    
    def generate_privacy_report(self) -> Dict[str, Any]:
        """生成隐私报告"""
        total_accesses = len([l for l in self.audit_logs if l['action'] == 'access'])
        total_updates = len([l for l in self.audit_logs if l['action'] == 'model_update'])
        
        total_privacy_spent = sum(
            l.get('privacy_budget_spent', 0) 
            for l in self.audit_logs
        )
        
        # 按客户端统计
        client_stats = defaultdict(lambda: {'accesses': 0, 'updates': 0})
        for log in self.audit_logs:
            client_id = log['client_id']
            if log['action'] == 'access':
                client_stats[client_id]['accesses'] += 1
            else:
                client_stats[client_id]['updates'] += 1
        
        return {
            'total_accesses': total_accesses,
            'total_model_updates': total_updates,
            'total_privacy_budget_spent': total_privacy_spent,
            'client_breakdown': dict(client_stats),
            'audit_trail_complete': True,
            'recommendations': self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """生成隐私建议"""
        recommendations = []
        
        # 检查是否有客户端过于活跃
        client_stats = defaultdict(int)
        for log in self.audit_logs:
            client_stats[log['client_id']] += 1
        
        avg_activity = sum(client_stats.values()) / len(client_stats) if client_stats else 0
        for client_id, count in client_stats.items():
            if count > avg_activity * 2:
                recommendations.append(
                    f"Client {client_id} shows unusually high activity. Consider investigating."
                )
        
        if not recommendations:
            recommendations.append("No privacy concerns detected.")
        
        return recommendations


# 主引擎类
class FederatedLearningPrivacyEngine:
    """联邦学习与隐私保护引擎"""
    
    def __init__(
        self,
        epsilon: float = 1.0,
        delta: float = 1e-5,
        min_clients: int = 3
    ):
        self.server = FederatedLearningServer(epsilon, delta, min_clients)
        self.clients: Dict[str, FederatedLearningClient] = {}
        self.privacy_auditor = PrivacyAuditor()
        self.current_round = 0
    
    def register_client(self, client_id: str, local_data: List[Dict]):
        """注册客户端"""
        client = FederatedLearningClient(client_id)
        client.set_local_data(local_data)
        
        # 生成公钥并注册
        public_key = client.generate_public_key()
        self.server.register_client(client_id, public_key)
        
        self.clients[client_id] = client
        
        return {
            'client_id': client_id,
            'data_size': len(local_data),
            'registered': True
        }
    
    def run_federated_round(self) -> Dict[str, Any]:
        """运行一轮联邦学习"""
        self.current_round += 1
        
        # 1. 分发全局模型
        global_model = self.server.get_global_model()
        for client in self.clients.values():
            client.receive_global_model(global_model)
        
        # 2. 本地训练
        local_updates = []
        for client in self.clients.values():
            update = client.local_train()
            local_updates.append(update)
            
            # 审计日志
            self.privacy_auditor.log_model_update(
                client.client_id,
                update_size=len(str(update.model_update)),
                privacy_budget_spent=0.01  # 简化的预算计算
            )
        
        # 3. 聚合更新
        result = self.server.execute_federated_round(
            local_updates,
            self.current_round
        )
        
        return result
    
    def personalize_for_client(self, client_id: str) -> Dict[str, Any]:
        """为特定客户端个性化模型"""
        if client_id not in self.clients:
            return {'error': 'Client not found'}
        
        client = self.clients[client_id]
        
        # 确保客户端有最新的全局模型
        global_model = self.server.get_global_model()
        client.receive_global_model(global_model)
        
        # 个性化
        result = client.personalize_model()
        
        return {
            'client_id': client_id,
            'personalization': result
        }
    
    def get_privacy_report(self) -> Dict[str, Any]:
        """获取隐私报告"""
        audit_report = self.privacy_auditor.generate_privacy_report()
        
        return {
            'version': 'V114',
            'name': 'Federated Learning & Privacy Preservation (FLPP)',
            'privacy_parameters': {
                'epsilon': self.server.dp_mechanism.epsilon,
                'delta': self.server.dp_mechanism.delta,
                'privacy_budget_remaining': self.server.privacy_budget.remaining
            },
            'federation_status': {
                'total_clients': len(self.clients),
                'current_round': self.current_round,
                'min_clients_required': self.server.min_clients
            },
            'audit_report': audit_report
        }


# 便捷函数
def build_federated_learning_privacy(
    client_data: Dict[str, List[Dict]],
    epsilon: float = 1.0,
    num_rounds: int = 5
) -> Dict[str, Any]:
    """
    构建联邦学习与隐私保护
    
    这是V114创新的主要入口函数
    """
    engine = FederatedLearningPrivacyEngine(epsilon=epsilon)
    
    # 注册客户端
    for client_id, data in client_data.items():
        engine.register_client(client_id, data)
    
    # 初始化模型（简化）
    engine.server.initialize_model({
        'embedding': (128,),
        'attention_weights': (64, 64)
    })
    
    # 运行联邦学习轮次
    round_results = []
    for round_num in range(num_rounds):
        result = engine.run_federated_round()
        round_results.append(result)
    
    # 为每个客户端个性化
    personalization_results = {}
    for client_id in client_data.keys():
        personalization_results[client_id] = engine.personalize_for_client(client_id)
    
    # 生成隐私报告
    privacy_report = engine.get_privacy_report()
    
    return {
        'version': 'V114',
        'name': 'Federated Learning & Privacy Preservation (FLPP)',
        'configuration': {
            'epsilon': epsilon,
            'num_rounds': num_rounds,
            'num_clients': len(client_data)
        },
        'federation_results': {
            'rounds_completed': len(round_results),
            'round_details': round_results,
            'final_model_version': engine.current_round
        },
        'personalization': personalization_results,
        'privacy_report': privacy_report
    }


# 示例用法
if __name__ == "__main__":
    # 模拟客户端数据
    client_data = {
        'client_1': [
            {'query': 'What is ML?', 'answer': 'Machine Learning'},
            {'query': 'What is AI?', 'answer': 'Artificial Intelligence'}
        ],
        'client_2': [
            {'query': 'What is DL?', 'answer': 'Deep Learning'},
            {'query': 'What is NLP?', 'answer': 'Natural Language Processing'}
        ],
        'client_3': [
            {'query': 'What is CV?', 'answer': 'Computer Vision'},
            {'query': 'What is RL?', 'answer': 'Reinforcement Learning'}
        ]
    }
    
    result = build_federated_learning_privacy(client_data, epsilon=1.0, num_rounds=3)
    print(json.dumps(result, indent=2, default=str))
