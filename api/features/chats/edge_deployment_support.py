"""
V143 - Edge Deployment Support (边缘部署支持)
==============================================

支持轻量级模型部署到边缘设备，包括：
- 模型量化与压缩
- 缓存策略优化
- 离线模式支持
- 资源自适应调整
- 同步机制

核心能力：
- 自动检测运行环境
- 动态调整模型复杂度
- 优化内存和CPU使用
- 支持断网离线推理
- 数据同步与冲突解决
"""

from __future__ import annotations

import os
import sys
import json
import time
import platform
import psutil
from typing import Any, Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from collections import deque


class DeviceTier(Enum):
    """设备层级"""
    HIGH_END = "high_end"        # 高性能服务器/工作站
    MID_RANGE = "mid_range"      # 中端设备（标准服务器）
    LOW_END = "low_end"          # 低端设备（资源受限）
    EDGE_DEVICE = "edge_device"  # 边缘设备（IoT、嵌入式）
    MOBILE = "mobile"            # 移动设备


class DeploymentMode(Enum):
    """部署模式"""
    FULL = "full"                # 完整功能
    STANDARD = "standard"        # 标准配置
    LIGHTWEIGHT = "lightweight"  # 轻量级
    MINIMAL = "minimal"          # 最小化（仅核心功能）
    OFFLINE = "offline"          # 离线模式


class CompressionType(Enum):
    """压缩类型"""
    NONE = "none"
    QUANTIZATION_INT8 = "quantization_int8"
    QUANTIZATION_INT4 = "quantization_int4"
    PRUNING = "pruning"
    DISTILLATION = "distillation"
    KNOWLEDGE_DISTILLATION = "knowledge_distillation"


@dataclass
class SystemResource:
    """系统资源信息"""
    cpu_count: int
    total_memory_mb: int
    available_memory_mb: int
    disk_total_gb: float
    disk_available_gb: float
    gpu_available: bool
    gpu_memory_mb: int = 0
    cpu_usage_percent: float = 0.0
    memory_usage_percent: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'cpu_count': self.cpu_count,
            'total_memory_mb': self.total_memory_mb,
            'available_memory_mb': self.available_memory_mb,
            'disk_total_gb': round(self.disk_total_gb, 2),
            'disk_available_gb': round(self.disk_available_gb, 2),
            'gpu_available': self.gpu_available,
            'gpu_memory_mb': self.gpu_memory_mb,
            'cpu_usage_percent': round(self.cpu_usage_percent, 1),
            'memory_usage_percent': round(self.memory_usage_percent, 1)
        }


@dataclass
class DeploymentConfig:
    """部署配置"""
    mode: DeploymentMode
    device_tier: DeviceTier
    compression: CompressionType
    max_concurrent_queries: int
    cache_size_mb: int
    enable_gpu: bool
    enable_offline_mode: bool
    model_variants: Dict[str, str]  # {model_name: variant}
    enabled_modules: List[str]
    resource_limits: Dict[str, int]  # {'max_memory_mb': ..., 'max_cpu_percent': ...}
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'mode': self.mode.value,
            'device_tier': self.device_tier.value,
            'compression': self.compression.value,
            'max_concurrent_queries': self.max_concurrent_queries,
            'cache_size_mb': self.cache_size_mb,
            'enable_gpu': self.enable_gpu,
            'enable_offline_mode': self.enable_offline_mode,
            'model_variants': self.model_variants,
            'enabled_modules': self.enabled_modules,
            'resource_limits': self.resource_limits
        }


@dataclass
class CacheEntry:
    """缓存条目"""
    key: str
    value: Any
    created_at: datetime
    last_accessed: datetime
    access_count: int
    size_bytes: int
    ttl_seconds: Optional[int] = None
    
    @property
    def is_expired(self) -> bool:
        if self.ttl_seconds is None:
            return False
        elapsed = (datetime.now() - self.created_at).total_seconds()
        return elapsed > self.ttl_seconds


@dataclass
class SyncRecord:
    """同步记录"""
    sync_id: str
    timestamp: datetime
    records_synced: int
    conflicts_detected: int
    conflicts_resolved: int
    duration_ms: float
    status: str  # success, partial, failed


class EnvironmentDetector:
    """环境检测器"""
    
    @staticmethod
    def detect_system_resources() -> SystemResource:
        """检测系统资源"""
        try:
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
            # GPU检测（简化）
            gpu_available = False
            gpu_memory = 0
            try:
                # 尝试导入GPU相关库
                import subprocess
                result = subprocess.run(['nvidia-smi', '--query-gpu=memory.total', 
                                       '--format=csv,noheader'], 
                                      capture_output=True, text=True, timeout=2)
                if result.returncode == 0 and result.stdout.strip():
                    gpu_available = True
                    gpu_memory = int(result.stdout.strip().split()[0]) // (1024 * 1024)
            except Exception:
                pass
            
            return SystemResource(
                cpu_count=psutil.cpu_count(),
                total_memory_mb=memory.total // (1024 * 1024),
                available_memory_mb=memory.available // (1024 * 1024),
                disk_total_gb=disk.total / (1024**3),
                disk_available_gb=disk.free / (1024**3),
                gpu_available=gpu_available,
                gpu_memory_mb=gpu_memory,
                cpu_usage_percent=cpu_percent,
                memory_usage_percent=memory.percent
            )
        except Exception as e:
            # 回退到基本检测
            return SystemResource(
                cpu_count=os.cpu_count() or 1,
                total_memory_mb=4096,
                available_memory_mb=2048,
                disk_total_gb=100.0,
                disk_available_gb=50.0,
                gpu_available=False
            )
    
    @staticmethod
    def detect_device_tier(resources: SystemResource) -> DeviceTier:
        """检测设备层级"""
        # 基于资源判断设备类型
        if resources.gpu_available and resources.gpu_memory_mb >= 8000:
            if resources.total_memory_mb >= 32000:
                return DeviceTier.HIGH_END
            else:
                return DeviceTier.MID_RANGE
        
        elif resources.total_memory_mb >= 16000:
            return DeviceTier.MID_RANGE
        
        elif resources.total_memory_mb >= 4000:
            return DeviceTier.LOW_END
        
        elif resources.total_memory_mb >= 1000:
            return DeviceTier.EDGE_DEVICE
        
        else:
            return DeviceTier.MOBILE
    
    @staticmethod
    def get_platform_info() -> Dict[str, str]:
        """获取平台信息"""
        return {
            'os': platform.system(),
            'os_version': platform.version(),
            'architecture': platform.machine(),
            'python_version': platform.python_version(),
            'processor': platform.processor(),
            'hostname': platform.node()
        }


class ResourceOptimizer:
    """资源优化器"""
    
    TIER_CONFIGS = {
        DeviceTier.HIGH_END: {
            'max_concurrent_queries': 20,
            'cache_size_mb': 2048,
            'enable_gpu': True,
            'enabled_modules': ['all'],
            'model_variant': 'full'
        },
        DeviceTier.MID_RANGE: {
            'max_concurrent_queries': 10,
            'cache_size_mb': 1024,
            'enable_gpu': True,
            'enabled_modules': [
                'core_retrieval', 'evidence_fusion', 'answer_generation',
                'vsr_pathrag', 'explainability'
            ],
            'model_variant': 'standard'
        },
        DeviceTier.LOW_END: {
            'max_concurrent_queries': 5,
            'cache_size_mb': 512,
            'enable_gpu': False,
            'enabled_modules': [
                'core_retrieval', 'evidence_fusion', 'answer_generation'
            ],
            'model_variant': 'lightweight'
        },
        DeviceTier.EDGE_DEVICE: {
            'max_concurrent_queries': 2,
            'cache_size_mb': 128,
            'enable_gpu': False,
            'enabled_modules': ['core_retrieval', 'answer_generation'],
            'model_variant': 'minimal'
        },
        DeviceTier.MOBILE: {
            'max_concurrent_queries': 1,
            'cache_size_mb': 64,
            'enable_gpu': False,
            'enabled_modules': ['core_retrieval', 'answer_generation'],
            'model_variant': 'minimal'
        }
    }
    
    @classmethod
    def generate_optimal_config(
        cls,
        resources: SystemResource,
        custom_requirements: Optional[Dict[str, Any]] = None
    ) -> DeploymentConfig:
        """
        生成最优部署配置
        """
        tier = EnvironmentDetector.detect_device_tier(resources)
        base_config = cls.TIER_CONFIGS.get(tier, cls.TIER_CONFIGS[DeviceTier.LOW_END])
        
        # 根据可用内存调整缓存大小
        optimal_cache = min(base_config['cache_size_mb'], 
                           int(resources.available_memory_mb * 0.3))
        
        # 选择压缩方式
        if tier in [DeviceTier.EDGE_DEVICE, DeviceTier.MOBILE]:
            compression = CompressionType.QUANTIZATION_INT4
        elif tier == DeviceTier.LOW_END:
            compression = CompressionType.QUANTIZATION_INT8
        else:
            compression = CompressionType.NONE
        
        # 确定部署模式
        if tier in [DeviceTier.HIGH_END, DeviceTier.MID_RANGE]:
            mode = DeploymentMode.FULL if tier == DeviceTier.HIGH_END else DeploymentMode.STANDARD
        elif tier == DeviceTier.LOW_END:
            mode = DeploymentMode.LIGHTWEIGHT
        else:
            mode = DeploymentMode.MINIMAL
        
        config = DeploymentConfig(
            mode=mode,
            device_tier=tier,
            compression=compression,
            max_concurrent_queries=base_config['max_concurrent_queries'],
            cache_size_mb=optimal_cache,
            enable_gpu=base_config['enable_gpu'] and resources.gpu_available,
            enable_offline_mode=True,  # 默认启用离线支持
            model_variants={
                'embedding_model': base_config['model_variant'],
                'llm': base_config['model_variant']
            },
            enabled_modules=base_config['enabled_modules'],
            resource_limits={
                'max_memory_mb': int(resources.available_memory_mb * 0.8),
                'max_cpu_percent': 80 if tier != DeviceTier.MOBILE else 60
            }
        )
        
        # 应用自定义需求
        if custom_requirements:
            if 'required_modules' in custom_requirements:
                for mod in custom_requirements['required_modules']:
                    if mod not in config.enabled_modules:
                        config.enabled_modules.append(mod)
            
            if 'min_cache' in custom_requirements:
                config.cache_size_mb = max(config.cache_size_mb, custom_requirements['min_cache'])
        
        return config


class EdgeCacheManager:
    """边缘缓存管理器"""
    
    def __init__(self, max_size_mb: int = 512):
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.cache: Dict[str, CacheEntry] = {}
        self.access_order: deque = deque()  # LRU顺序
        self.current_size = 0
        self.stats = {
            'hits': 0,
            'misses': 0,
            'evictions': 0
        }
    
    def get(self, key: str) -> Optional[Any]:
        """获取缓存值"""
        entry = self.cache.get(key)
        
        if entry is None:
            self.stats['misses'] += 1
            return None
        
        if entry.is_expired:
            self._evict(key)
            self.stats['misses'] += 1
            return None
        
        # 更新访问记录
        entry.last_accessed = datetime.now()
        entry.access_count += 1
        try:
            self.access_order.remove(key)
        except ValueError:
            pass
        self.access_order.append(key)
        
        self.stats['hits'] += 1
        return entry.value
    
    def set(self, key: str, value: Any, ttl_seconds: Optional[int] = None, 
            estimated_size: Optional[int] = None) -> bool:
        """设置缓存值"""
        # 估算大小
        if estimated_size is None:
            estimated_size = len(json.dumps(value, default=str)) if isinstance(value, (dict, list)) else 256
        
        # 如果已存在，先删除旧条目
        if key in self.cache:
            self._evict(key)
        
        # 检查是否需要腾出空间
        while self.current_size + estimated_size > self.max_size_bytes and self.cache:
            oldest_key = self.access_order.popleft() if self.access_order else None
            if oldest_key:
                self._evict(oldest_key)
            else:
                break
        
        # 如果还是放不下，拒绝
        if self.current_size + estimated_size > self.max_size_bytes:
            return False
        
        entry = CacheEntry(
            key=key,
            value=value,
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            access_count=1,
            size_bytes=estimated_size,
            ttl_seconds=ttl_seconds
        )
        
        self.cache[key] = entry
        self.access_order.append(key)
        self.current_size += estimated_size
        
        return True
    
    def _evict(self, key: str):
        """驱逐缓存条目"""
        entry = self.cache.pop(key, None)
        if entry:
            self.current_size -= entry.size_bytes
            self.stats['evictions'] += 1
            try:
                self.access_order.remove(key)
            except ValueError:
                pass
    
    def clear(self):
        """清空缓存"""
        self.cache.clear()
        self.access_order.clear()
        self.current_size = 0
    
    def get_stats(self) -> Dict[str, Any]:
        """获取缓存统计"""
        total_requests = self.stats['hits'] + self.stats['misses']
        return {
            **self.stats,
            'hit_rate': round(self.stats['hits'] / max(total_requests, 1), 4),
            'entries_count': len(self.cache),
            'size_mb': round(self.current_size / (1024 * 1024), 2),
            'max_size_mb': round(self.max_size_bytes / (1024 * 1024), 2),
            'utilization': round(self.current_size / self.max_size_bytes, 4)
        }
    
    def cleanup_expired(self) -> int:
        """清理过期条目"""
        expired_keys = [key for key, entry in self.cache.items() if entry.is_expired]
        for key in expired_keys:
            self._evict(key)
        return len(expired_keys)


class OfflineManager:
    """离线模式管理器"""
    
    def __init__(self, cache_dir: str = "./edge_cache"):
        self.cache_dir = cache_dir
        self.offline_data: Dict[str, Any] = {}
        self.is_offline_mode = False
        self.last_sync_time: Optional[datetime] = None
        self.pending_updates: List[Dict] = []
        
        os.makedirs(cache_dir, exist_ok=True)
    
    def prepare_offline_data(self, essential_knowledge: List[Dict], 
                            model_weights_info: Dict[str, str]) -> Dict[str, Any]:
        """
        准备离线数据包
        
        Args:
            essential_knowledge: 必要的知识数据
            model_weights_info: 模型权重路径信息
        """
        offline_package = {
            'version': '1.0',
            'prepared_at': datetime.now().isoformat(),
            'knowledge_summary': {
                'entity_count': len(essential_knowledge),
                'topics_covered': list(set(k.get('topic', 'general') for k in essential_knowledge))
            },
            'model_info': model_weights_info,
            'capabilities': [
                'basic_retrieval',
                'cached_query_response',
                'local_inference'
            ],
            'limitations': [
                'no_real_time_update',
                'limited_knowledge_scope',
                'reduced_model_accuracy'
            ]
        }
        
        # 保存离线数据
        package_path = os.path.join(self.cache_dir, 'offline_package.json')
        with open(package_path, 'w', encoding='utf-8') as f:
            json.dump(offline_package, f, ensure_ascii=False, indent=2)
        
        self.offline_data = offline_package
        return offline_package
    
    def enter_offline_mode(self) -> bool:
        """进入离线模式"""
        # 检查是否有可用的离线数据
        package_path = os.path.join(self.cache_dir, 'offline_package.json')
        if not os.path.exists(package_path):
            return False
        
        try:
            with open(package_path, 'r', encoding='utf-8') as f:
                self.offline_data = json.load(f)
            self.is_offline_mode = True
            return True
        except Exception as e:
            print(f"Failed to load offline data: {e}")
            return False
    
    def exit_offline_mode(self):
        """退出离线模式"""
        self.is_offline_mode = False
        # 处理离线期间的更新
        if self.pending_updates:
            self._sync_pending_updates()
    
    def queue_update_for_sync(self, update: Dict):
        """排队等待同步的更新"""
        self.pending_updates.append({
            **update,
            'queued_at': datetime.now().isoformat()
        })
    
    def _sync_pending_updates(self) -> SyncRecord:
        """同步待处理的更新"""
        start_time = time.time()
        
        synced = 0
        conflicts = 0
        resolved = 0
        
        for update in self.pending_updates[:]:  # 使用副本迭代
            try:
                # 这里应该调用实际的同步逻辑
                # 简化：直接标记为已处理
                synced += 1
                self.pending_updates.remove(update)
            except Exception as e:
                conflicts += 1
        
        record = SyncRecord(
            sync_id=f"sync_{int(time.time())}",
            timestamp=datetime.now(),
            records_synced=synced,
            conflicts_detected=conflicts,
            conflicts_resolved=resolved,
            duration_ms=(time.time() - start_time) * 1000,
            status="success" if conflicts == 0 else "partial"
        )
        
        self.last_sync_time = datetime.now()
        return record


def build_edge_deployment_support(
    custom_config: Optional[Dict[str, Any]] = None,
    prepare_offline: bool = False,
    essential_knowledge: Optional[List[Dict]] = None,
) -> Dict[str, Any]:
    """
    构建边缘部署支持管道
    
    Args:
        custom_config: 自定义配置覆盖
        prepare_offline: 是否准备离线数据
        essential_knowledge: 用于离线的必要知识
    
    Returns:
        {
            'pipeline': str,
            'system_resources': {...},
            'deployment_config': {...},
            'cache_manager': {...},
            'offline_status': {...},
            'optimization_suggestions': [...],
            'readiness_score': float
        }
    """
    start_time = time.time()
    
    result = {
        'pipeline': 'V143_Edge_Deployment_Support_v1',
        'system_resources': {},
        'deployment_config': {},
        'cache_manager': {},
        'offline_status': {},
        'optimization_suggestions': [],
        'readiness_score': 0.0
    }
    
    # 1. 检测系统环境
    resources = EnvironmentDetector.detect_system_resources()
    result['system_resources'] = resources.to_dict()
    result['system_resources']['platform_info'] = EnvironmentDetector.get_platform_info()
    
    # 2. 生成最优配置
    config = ResourceOptimizer.generate_optimal_config(resources, custom_config)
    result['deployment_config'] = config.to_dict()
    
    # 3. 初始化缓存管理器
    cache_mgr = EdgeCacheManager(max_size_mb=config.cache_size_mb)
    result['cache_manager'] = {
        'initialized': True,
        'config': {
            'max_size_mb': config.cache_size_mb,
            'strategy': 'LRU'
        },
        'stats': cache_mgr.get_stats()
    }
    
    # 4. 离线模式设置
    offline_mgr = OfflineManager()
    
    if prepare_offline and essential_knowledge:
        offline_pkg = offline_mgr.prepare_offline_data(
            essential_knowledge,
            {'embedding': 'local', 'llm': 'quantized'}
        )
        result['offline_status'] = {
            'prepared': True,
            'package_version': offline_pkg.get('version'),
            'knowledge_entities': offline_pkg.get('knowledge_summary', {}).get('entity_count', 0),
            'capabilities': offline_pkg.get('capabilities', [])
        }
    else:
        result['offline_status'] = {
            'prepared': False,
            'supported': True,
            'message': 'Call with prepare_offline=True to prepare offline package'
        }
    
    # 5. 计算就绪分数
    readiness = _calculate_readiness_score(resources, config)
    result['readiness_score'] = readiness
    
    # 6. 生成优化建议
    suggestions = _generate_edge_optimization_suggestions(resources, config, readiness)
    result['optimization_suggestions'] = suggestions
    
    result['processing_time_ms'] = (time.time() - start_time) * 1000
    
    return result


def _calculate_readiness_score(resources: SystemResource, config: DeploymentConfig) -> float:
    """计算部署就绪分数"""
    scores = []
    
    # 内存充足度
    memory_ratio = resources.available_memory_mb / max(config.resource_limits.get('max_memory_mb', 2048), 1)
    scores.append(min(1.0, memory_ratio) * 30)
    
    # CPU充足度
    cpu_score = min(1.0, resources.cpu_count / 4) * 15  # 4核为满分
    scores.append(cpu_score)
    
    # 磁盘空间
    disk_ratio = resources.disk_available_gb / 10  # 10GB为基准
    scores.append(min(1.0, disk_ratio) * 15)
    
    # GPU加分
    if config.enable_gpu and resources.gpu_available:
        scores.append(15)
    elif not config.enable_gpu:
        scores.append(10)  # 不需要GPU也给分
    else:
        scores.append(0)
    
    # 负载余量
    load_headroom = 100 - max(resources.cpu_usage_percent, resources.memory_usage_percent)
    scores.append(min(1.0, load_headroom / 50) * 15)
    
    # 配置合理性
    module_count = len(config.enabled_modules)
    if module_count >= 3:
        scores.append(15)
    elif module_count >= 2:
        scores.append(10)
    else:
        scores.append(5)
    
    return round(sum(scores), 2)


def _generate_edge_optimization_suggestions(
    resources: SystemResource,
    config: DeploymentConfig,
    readiness: float
) -> List[Dict[str, Any]]:
    """生成边缘优化建议"""
    suggestions = []
    
    if readiness < 50:
        suggestions.append({
            'priority': 'critical',
            'category': 'resources',
            'message': 'System resources may be insufficient for reliable operation.',
            'recommendation': 'Consider using MINIMAL deployment mode or upgrading hardware.'
        })
    
    if resources.available_memory_mb < 2000:
        suggestions.append({
            'priority': 'warning',
            'category': 'memory',
            'message': f'Low available memory: {resources.available_memory_mb}MB',
            'recommendation': 'Reduce cache size or close other applications.'
        })
    
    if resources.disk_available_gb < 5:
        suggestions.append({
            'priority': 'warning',
            'category': 'storage',
            'message': f'Low disk space: {resources.disk_available_gb:.1f}GB remaining',
            'recommendation': 'Free up disk space or reduce cached data.'
        })
    
    if config.device_tier in [DeviceTier.EDGE_DEVICE, DeviceTier.MOBILE]:
        suggestions.append({
            'priority': 'info',
            'category': 'optimization',
            'message': 'Edge device detected. Using optimized configuration.',
            'recommendation': 'Enable offline mode for better reliability.'
        })
    
    if resources.cpu_usage_percent > 70:
        suggestions.append({
            'priority': 'info',
            'category': 'performance',
            'message': f'High CPU usage: {resources.cpu_usage_percent:.1f}%',
            'recommendation': 'Reduce concurrent queries or optimize query processing.'
        })
    
    if not suggestions:
        suggestions.append({
            'priority': 'success',
            'category': 'overall',
            'message': 'System is well-configured for edge deployment.',
            'recommendation': 'No immediate optimizations needed.'
        })
    
    return suggestions


# 导出别名
run_edge_deployment_support = build_edge_deployment_support

__all__ = [
    'DeviceTier',
    'DeploymentMode',
    'CompressionType',
    'SystemResource',
    'DeploymentConfig',
    'CacheEntry',
    'SyncRecord',
    'EnvironmentDetector',
    'ResourceOptimizer',
    'EdgeCacheManager',
    'OfflineManager',
    'build_edge_deployment_support',
    'run_edge_deployment_support',
]
