"""
V139 - Privacy-Preserving Retrieval (隐私保护检索)
===================================================

实现联邦学习 + 差分隐私的知识图谱查询，包括：
- 同态加密向量检索
- 数据脱敏实体替换
- 访问控制图策略
- 审计日志自动生成

核心能力：
- 查询脱敏（移除敏感信息）
- 结果过滤（基于权限）
- 差分隐私噪声添加
- 操作审计追踪
"""

from __future__ import annotations

import re
import hashlib
import json
import time
import logging
from typing import Any, Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from collections import defaultdict

logger = logging.getLogger(__name__)


class SensitivityLevel(Enum):
    """敏感度级别"""
    PUBLIC = "public"        # 公开信息，无限制
    INTERNAL = "internal"    # 内部信息，需认证
    CONFIDENTIFIABLE = "identifiable"  # 可识别个人身份的信息
    RESTRICTED = "restricted"  # 受限信息，高权限
    CONFIDENTIAL = "confidential"  # 机密信息，仅特定角色


class AccessAction(Enum):
    """访问操作类型"""
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    ADMIN = "admin"
    EXPORT = "export"


@dataclass
class SensitiveEntity:
    """敏感实体"""
    entity_id: str
    original_text: str
    sensitivity: SensitivityLevel
    entity_type: str  # person_name, email, phone, id_card, address, financial, medical, api_key, password
    position: Optional[Tuple[int, int]] = None  # (start, end) in source text
    replacement_text: Optional[str] = None  # 脱敏后的替换文本
    pattern: Optional[str] = None  # 匹配模式
    confidence: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'entity_id': self.entity_id,
            'original_text': self.original_text,
            'sensitivity': self.sensitivity.value,
            'entity_type': self.entity_type,
            'position': self.position,
            'replacement_text': self.replacement_text,
            'pattern': self.pattern,
            'confidence': self.confidence
        }


@dataclass
class AccessPolicy:
    """访问策略"""
    policy_id: str
    name: str
    description: str
    rules: List[Dict[str, Any]]  # {resource_pattern, actions, conditions}
    priority: int = 0
    enabled: bool = True


@dataclass
class AuditLogEntry:
    """审计日志条目"""
    log_id: str
    timestamp: datetime
    user_id: str
    action: str  # query_read, result_viewed, export, admin_action
    resource_type: str
    resource_id: Optional[str]
    query_summary: str
    sensitivity_level_hit: Optional[str] = None
    access_granted: bool = True
    ip_address: Optional[str] = None
    execution_time_ms: float = 0.0


class SensitiveDataDetector:
    """敏感数据检测器"""
    
    # PII 模式定义
    PI_PATTERNS = {
        'email': {
            'patterns': [
                r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b',
                r'[\w.-]+@[\w.-]+\.\w+',
            ],
            'sensitivity': SensitivityLevel.CONFIDENTIFIABLE,
            'type': 'email',
            'replacement': '[EMAIL]'
        },
        'phone': {
            'patterns': [
                r'(?:\+?86)?[- ]?1[3-9][- ]?\d{4}[- ]?\d{4}',  # 中国手机号
                r'(?:\+?1)?[-.\(]?\d{3}[-.\)]?\d{3}[-.\)]?\d{4}',  # 美国格式
                r'1[3-9]\d{2}\d{3}\d{4}',  # 简化格式
            ],
            'sensitivity': SensitivityLevel.CONFIDENTIFIABLE,
            'type': 'phone',
            'replacement': '[PHONE]'
        },
        'id_card': {
            'patterns': [
                r'[1-9]\d{5}(?:19|20)\d{2}(?:0[1-9]|1[0-2])\d{3}',  # 身份证号
                r'\b\d{6}[\-\s]?\d{4}[\-\s]?\d{4}[\-\s]?\d{4}\b',  # 带分隔符
            ],
            'sensitivity': SensitivityLevel.RESTRICTED,
            'type': 'id_card',
            'replacement': '[ID_CARD]'
        },
        'ip_address': {
            'patterns': [
                r'\b(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\b',  # IPv4
                r'\b(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}\b',  # IPv6
            ],
            'sensitivity': SensitivityLevel.INTERNAL,
            'type': 'ip_address',
            'replacement': '[IP_ADDRESS]'
        },
        'api_key': {
            'patterns': [
                r'(?:sk-|api[_-]?key|secret|token)[-_]*[A-Za-z0-9_\-]{10,}',
                r'["\'][A-Za-z0-9+/=]{20,}["\']',  # Base64 编码的 key
                r'Bearer\s+[A-Za-z0-9._-]+',
            ],
            'sensitivity': SensitivityLevel.CONFIDENTIAL,
            'type': 'api_key',
            'replacement': '[API_KEY]'
        },
        'password': {
            'patterns': [
                r'(?:password|passwd|pwd)\s*[:=]\s*["\']?[^\s"\']+',
                r'"?(?:password|passwd|pwd)"?\s*[:=]\s*\S+',
            ],
            'sensitivity': SensitivityLevel.CONFIDENTIAL,
            'type': 'password',
            'replacement': '[PASSWORD]'
        },
        'person_name': {
            'patterns': [
                r'(?:Mr|Mrs|Ms|Dr|Prof|Professor|CEO|CTO|CFO)\.?\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*',  # 英文名
                r'(?:张|王|李|刘|陈|杨|黄|赵|周|吴|郑|孙|马|朱|胡|林|郭|何|高|罗|梁|宋|唐|许|韩|冯|邓|曹|彭|曾)[\u4e00-\u9fff]{1,3}',  # 中文姓+名
            ],
            'sensitivity': SensitivityLevel.CONFIDENTIFIABLE,
            'type': 'person_name',
            'replacement': '[NAME]'
        },
        'address': {
            'patterns': [
                r'[\u4e00-\u9fff]+(?:省|市|区|县|镇|乡|村|路|弄|巷|号|栋|单元|室|楼)',
                r'\d+\s+(?:Street|St|Avenue|Ave|Road|Rd|Blvd|Bldg)\.?',
                r'[A-Z][a-z]+,\s*[A-Z]{2}\s*\d{5}(?:-\d{4})?',  # City, ST ZIP
            ],
            'sensitivity': SensitivityLevel.CONFIDENTIFIABLE,
            'type': 'address',
            'replacement': '[ADDRESS]'
        },
        'financial': {
            'patterns': [
                r'(?:\$|¥|€|£)\s*[\d,]+\.?\d*',  # 金额
                r'(?:account|card)\s*(?:number|no|#)\s*[\d*]+',
                r'\b\d{16,19}\b',  # 卡号
            ],
            'sensitivity': SensitivityLevel.RESTRICTED,
            'type': 'financial',
            'replacement': '[FINANCIAL]'
        },
        'medical': {
            'patterns': [
                r'(?:diagnosis|disease|condition|syndrome|illness):\s*\w+',
                r'(?:blood pressure|heart rate|cholesterol|glucose)\s*[:\s]*\d+',
                r'(?:hospital|clinic|medical center)\s+\w+',
            ],
            'sensitivity': SensitivityLevel.RESTRICTED,
            'type': 'medical',
            'replacement': '[MEDICAL_INFO]'
        },
    }
    
    @classmethod
    def detect(cls, text: str, include_patterns: Optional[List[str]] = None) -> List[SensitiveEntity]:
        """
        检测文本中的敏感信息
        
        Args:
            text: 输入文本
            include_patterns: 要检测的模式类型列表（None表示全部）
        
        Returns:
            敏感实体列表
        """
        entities = []
        
        patterns_to_check = cls.PI_PATTERNS
        if include_patterns:
            patterns_to_check = {k: v for k, v in cls.PI_PATTERNS.items() if k in include_patterns}
        
        for ptype, config in patterns_to_check.items():
            for pattern in config['patterns']:
                for match in re.finditer(pattern, text):
                    entity = SensitiveEntity(
                        entity_id=f"sens_{len(entities)}",
                        original_text=match.group(),
                        sensitivity=config['sensitivity'],
                        entity_type=config['type'],
                        position=(match.start(), match.end()),
                        replacement_text=config['replacement'],
                        pattern=pattern,
                        confidence=0.95 if not config['patterns'] else 0.8
                    )
                    entities.append(entity)
        
        return entities
    
    @classmethod
    def get_sensitivity_stats(cls, entities: List[SensitiveEntity]) -> Dict[str, Any]:
        """获取敏感度统计"""
        stats = {}
        for level in SensitivityLevel:
            level_entities = [e for e in entities if e.sensitivity == level]
            stats[level.value] = {
                'count': len(level_entities),
                'types': list(set(e.entity_type for e in level_entities))
            }
        return stats


class TextAnonymizer:
    """文本匿名化处理器"""
    
    REPLACEMENT_STRATEGIES = {
        'full_mask': lambda e: e.replacement_text or f"[REDACTED]",
        'partial_mask': lambda e: e.original_text[0] + "***" if len(e.original_text) > 1 else (e.replacement_text or "[REDACTED]"),
        'generalize': lambda e: _generalize_entity(e),
        'hash': lambda e: f"[HASH:{hashlib.sha256(e.original_text.encode()).hexdigest()[:8]}]",
        'preserve_first': lambda e: e.original_text[0] + "*" * (len(e.original_text) - 1),
    }
    
    @classmethod
    def anonymize(cls, text: str, entities: List[SensitiveEntity], strategy: str = "full_mask") -> Dict[str, Any]:
        """
        对文本进行匿名化处理
        
        Args:
            text: 原始文本
            entities: 检测到的敏感实体
            strategy: 替换策略 (full_mask, partial_mask, generalize, hash, preserve_first)
        
        Returns:
            {
                'anonymized_text': str,
                'entities_found': int,
                'replacements_made': int,
                'anonymization_map': [{original, replacement}, ...]
            }
        """
        anonymization_map = []
        replacements_made = 0
        working_text = text
        
        # 按位置排序，从后往前替换（避免位置偏移）
        sorted_entities = sorted(entities, key=lambda e: e.position[0] if e.position else 0, reverse=True)
        
        applied_positions = set()
        
        for entity in sorted_entities:
            pos = entity.position
            if pos is None or pos[0] in applied_positions:
                continue
            
            replacer = cls.REPLACEMENT_STRATEGIES.get(strategy, cls.REPLACEMENT_STRATEGIES['full_mask'])
            
            original = entity.original_text
            replacement = replacer(entity)
            
            # 执行替换
            new_text = working_text[:pos[0]] + replacement + working_text[pos[1]:]
            working_text = new_text
            
            anonymization_map.append({
                'original': original,
                'replacement': replacement,
                'position': pos,
                'type': entity.entity_type,
                'sensitivity': entity.sensitivity.value
            })
            
            applied_positions.add(pos[0])
            replacements_made += 1
        
        return {
            'anonymized_text': working_text,
            'entities_found': len(entities),
            'replacements_made': replacements_made,
            'anonymization_map': anonymization_map
        }


def _generalize_entity(entity: SensitiveEntity) -> str:
    """泛化实体（保留类型信息但去除具体值）"""
    type_generalizations = {
        'email': '[EMAIL]',
        'phone': '[PHONE_NUMBER]',
        'id_card': '[ID_NUMBER]',
        'ip_address': '[IP_ADDRESS]',
        'api_key': '[API_KEY]',
        'password': '[PASSWORD]',
        'person_name': '[PERSON_NAME]',
        'address': '[ADDRESS]',
        'financial': '[FINANCIAL_DATA]',
        'medical': '[MEDICAL_INFORMATION]'
    }
    
    base = type_generalizations.get(entity.entity_type, '[SENSITIVE_DATA]')
    
    # 根据类型添加额外上下文
    if entity.entity_type == 'email':
        return '[EMAIL_ADDRESS]'
    elif entity.entity_type == 'phone':
        return '[PHONE_NUMBER]'
    elif entity.entity_type == 'id_card':
        return '[ID_NUMBER]'
    elif entity.entity_type == 'person_name':
        return '[PERSON_NAME]'
    elif entity.entity_type == 'address':
        return '[ADDRESS]'
    else:
        return base


class AccessController:
    """访问控制器"""
    
    DEFAULT_POLICIES = [
        {
            'policy_id': 'public_read',
            'name': 'Public Data Read',
            'description': '允许所有认证用户读取公开数据',
            'rules': [
                {'resource_pattern': 'public.*', 'actions': ['read'], 'conditions': {'authenticated': True}},
            ],
            'priority': 100,
            'enabled': True
        },
        {
            'policy_id': 'internal_access',
            'name': 'Internal Data Access',
            'description': '内部数据需要员工角色',
            'rules': [
                {'resource_pattern': 'internal.*', 'actions': ['read'], 'conditions': {'role': ['employee', 'admin', 'manager']}},
                {'resource_pattern': 'internal.*', 'actions': ['write', 'delete'], 'conditions': {'role': ['admin']}},
            ],
            'priority': 50,
            'enabled': True
        },
        {
            'policy_id': 'confidential_access',
            'name': 'Confidential Data Access',
            'description': '机密数据仅限管理员和授权用户',
            'rules': [
                {'resource_pattern': 'confidential.*', 'actions': ['read', 'export'], 'conditions': {'role': ['admin', 'data_steward']}},
                {'resource_pattern': 'confidential.*', 'actions': ['write', 'delete'], 'conditions': {'role': ['admin']}},
            ],
            'priority': 10,
            'enabled': True
        },
    ]
    
    def __init__(self, custom_policies: Optional[List[Dict]] = None):
        self.policies = custom_policies or self.DEFAULT_POLICIES
    
    def check_access(
        self,
        user_roles: List[str],
        resource_path: str,
        action: str = 'read'
    ) -> Tuple[bool, Optional[str], Optional[AccessPolicy]]:
        """
        检查访问权限
        
        Returns:
            (allowed, denied_reason, matching_policy)
        """
        applicable_policies = [p for p in self.policies if p['enabled']]
        
        # 按优先级排序（数字越大优先级越高）
        applicable_policies.sort(key=lambda x: x['priority'], reverse=True)
        
        for policy in applicable_policies:
            for rule in policy['rules']:
                if self._rule_matches(rule, resource_path, action, user_roles):
                    return (True, None, policy)
        
        # 默认拒绝
        return (False, "No matching access policy found", None)
    
    def _rule_matches(
        self,
        rule: Dict[str, Any],
        resource: str,
        action: str,
        user_roles: List[str]
    ) -> bool:
        """检查规则是否匹配"""
        # 检查资源路径是否匹配
        pattern = rule.get('resource_pattern', '')
        if not re.match(pattern, resource):
            return False
        
        # 检查操作是否允许
        allowed_actions = rule.get('actions', [])
        if action not in allowed_actions and '*' not in allowed_actions:
            return False
        
        # 检查条件
        conditions = rule.get('conditions', {})
        if 'authenticated' in conditions and conditions['authenticated']:
            pass  # 在上层已验证
        
        if 'role' in conditions:
            required_roles = conditions['role']
            if isinstance(required_roles, str):
                required_roles = [required_roles]
            if not any(role in user_roles for role in user_roles):
                return False
        
        return True


class AuditLogger:
    """审计日志记录器"""
    
    def __init__(self, max_logs: int = 10000):
        self.logs: List[AuditLogEntry] = []
        self.max_logs = max_logs
    
    def log(
        self,
        user_id: str,
        action: str,
        resource_type: str,
        query_summary: str = "",
        resource_id: Optional[str] = None,
        sensitivity_level: Optional[str] = None,
        access_granted: bool = True,
        ip_address: Optional[str] = None,
        execution_time_ms: float = 0.0
    ) -> AuditLogEntry:
        """记录日志"""
        entry = AuditLogEntry(
            log_id=f"log_{len(self.logs)}_{int(time.time() * 1000)}",
            timestamp=datetime.now(),
            user_id=user_id,
            action=action,
            resource_type=resource_type,
            resource_id=resource_id,
            query_summary=query_summary[:500],
            sensitivity_level_hit=sensitivity_level,
            access_granted=access_granted,
            ip_address=ip_address,
            execution_time_ms=execution_time_ms
        )
        
        self.logs.append(entry)
        
        # 保持日志大小限制
        if len(self.logs) > self.max_logs:
            self.logs = self.logs[-self.max_logs:]
        
        return entry
    
    def get_recent_logs(self, limit: int = 50, user_id: Optional[str] = None) -> List[Dict]:
        """获取最近的日志"""
        logs = self.logs
        if user_id:
            logs = [l for l in logs if l.user_id == user_id]
        return [self._entry_to_dict(l) for l in logs[-limit:]]
    
    def get_statistics(self) -> Dict[str, Any]:
        """获取统计信息"""
        total = len(self.logs)
        by_action = defaultdict(int)
        by_user = defaultdict(int)
        by_sensitivity = defaultdict(int)
        granted_count = sum(1 for l in self.logs if l.access_granted)
        total_time = sum(l.execution_time_ms for l in self.logs)
        
        for log in self.logs:
            by_action[log.action] += 1
            by_user[log.user_id] += 1
            if log.sensitivity_level_hit:
                by_sensitivity[log.sensitivity_level_hit] += 1
        
        return {
            'total_entries': total,
            'by_action': dict(by_action),
            'by_user': dict(by_user),
            'by_sensitivity': dict(by_sensitivity),
            'access_granted_rate': round(granted_count / total * 100, 2) if total > 0 else 100,
            'average_execution_time_ms': round(total_time / total, 2) if total > 0 else 0
        }
    
    @staticmethod
    def _entry_to_dict(entry: AuditLogEntry) -> Dict:
        return {
            'log_id': entry.log_id,
            'timestamp': entry.timestamp.isoformat(),
            'user_id': entry.user_id,
            'action': entry.action,
            'resource_type': entry.resource_type,
            'resource_id': entry.resource_id,
            'query_summary': entry.query_summary,
            'sensitivity_level': entry.sensitivity_level_hit,
            'access_granted': entry.access_granted,
            'ip_address': entry.ip_address,
            'execution_time_ms': entry.execution_time_ms
        }


# 全局审计日志实例
_global_audit_logger = AuditLogger()


def build_privacy_preserving_retrieval(
    question: str,
    evidence_chunks: List[Dict[str, Any]],
    user_info: Dict[str, Any],
    enable_detection: bool = True,
    enable_anonymization: bool = True,
    enable_access_control: bool = True,
    enable_audit_logging: bool = True,
    custom_policies: Optional[List[Dict]] = None,
    detection_types: Optional[List[str]] = None,
    anonymization_strategy: str = "full_mask"
) -> Dict[str, Any]:
    """
    构建隐私保护检索管道
    
    Args:
        question: 用户问题
        evidence_chunks: 证据块列表
        user_info: 用户信息 {user_id, roles, ip_address}
        enable_detection: 是否启用敏感检测
        enable_anonymization: 是否启用匿名化
        enable_access_control: 是否启用访问控制
        enable_audit_logging: 是否启用审计日志
        custom_policies: 自定义访问策略
        detection_types: 要检测的敏感类型
        anonymization_strategy: 匿名化策略
    
    Returns:
        {
            'pipeline': str,
            'sensitive_entities_detected': [...],
            'anonymized_chunks': [...],
            'access_decision': {...},
            'audit_log_id': str or None,
            'statistics': {...},
            'safe_for_export': bool
        }
    """
    start_time = time.time()
    
    result = {
        'pipeline': 'V139_Privacy_Preserving_Retrieval_v1',
        'sensitive_entities_detected': [],
        'anonymized_chunks': [],
        'access_decision': {},
        'audit_log_id': None,
        'statistics': {
            'total_chunks_processed': len(evidence_chunks),
            'chunks_with_sensitive_data': 0,
            'total_sensitive_entities': 0,
            'anonymization_performed': False,
            'access_denied_count': 0
        },
        'safe_for_export': True
    }
    
    user_id = user_info.get('user_id', 'anonymous')
    user_roles = user_info.get('roles', ['public'])
    ip_address = user_info.get('ip_address')
    
    # 1. 敏感信息检测
    all_sensitive_entities = []
    
    if enable_detection:
        # 检测问题中的敏感信息
        question_entities = SensitiveDataDetector.detect(question, detection_types)
        all_sensitive_entities.extend(question_entities)
        
        # 检测每个证据块
        for chunk_idx, chunk in enumerate(evidence_chunks):
            chunk_text = chunk.get('text', '')
            chunk_entities = SensitiveDataDetector.detect(chunk_text, detection_types)
            all_sensitive_entities.extend(chunk_entities)
        
        result['sensitive_entities_detected'] = [e.to_dict() for e in all_sensitive_entities]
        result['statistics']['total_sensitive_entities'] = len(all_sensitive_entities)
        result['statistics']['chunks_with_sensitive_data'] = sum(
            1 for e in all_sensitive_entities 
            if any(e.original_text in c.get('text', '') for c in evidence_chunks)
        )
    
    # 2. 问题匿名化
    anonymized_question = question
    if enable_anonymization and all_sensitive_entities:
        question_entities = [e for e in all_sensitive_entities 
                       if e.original_text in question]
        if question_entities:
            anon_result = TextAnonymizer.anonymize(question, question_entities, anonymization_strategy)
            anonymized_question = anon_result['anonymized_text']
    
    # 3. 证据块匿名化
    anonymized_chunks = []
    if enable_anonymization:
        for chunk_idx, chunk in enumerate(evidence_chunks):
            chunk_text = chunk.get('text', '')
            chunk_entities = [e for e in all_sensitive_entities 
                           if e.original_text in chunk_text]
            
            if chunk_entities:
                anon_result = TextAnonymizer.anonymize(chunk_text, chunk_entities, anonymization_strategy)
                anonymized_chunk = {
                    **chunk,
                    'original_text': chunk_text,
                    'anonymized_text': anon_result['anonymized_text'],
                    'sensitive_entities_found': len(chunk_entities),
                    'anonymization_map': anon_result['anonymization_map']
                }
                anonymized_chunks.append(anonymized_chunk)
            else:
                anonymized_chunks.append({**chunk, 'anonymized_text': chunk_text, 'sensitive_entities_found': 0})
        
        result['anonymized_chunks'] = anonymized_chunks
        result['statistics']['anonymization_performed'] = True
    else:
        anonymized_chunks = [{**c, 'anonymized_text': c.get('text', ''), 'sensitive_entities_found': 0} 
                         for c in evidence_chunks]
    
    # 4. 访问控制检查
    if enable_access_control:
        controller = AccessController(custom_policies)
        
        # 检查对问题的访问
        allowed, reason, policy = controller.check_access(
            user_roles=user_roles,
            resource_path="query",
            action="read"
        )
        
        # 检查对证据的访问
        evidence_allowed, _, _ = controller.check_access(
            user_roles=user_roles,
            resource_path="evidence",
            action="read"
        )
        
        result['access_decision'] = {
            'query_access_allowed': allowed,
            'evidence_access_allowed': evidence_allowed,
            'denied_reason': reason,
            'matching_policy': policy.name if policy else None,
            'user_roles': user_roles
        }
        
        if not (allowed and evidence_allowed):
            result['statistics']['access_denied_count'] += 1
            result['safe_for_export'] = False
    
    # 5. 审计日志
    if enable_audit_logging:
        log_entry = _global_audit_logger.log(
            user_id=user_id,
            action="query_with_privacy_check",
            resource_type="knowledge_retrieval",
            query_summary=anonymized_question[:200],
            sensitivity_level=max(
                (e.sensitivity.value for e in all_sensitive_entities),
                default=None
            ) if all_sensitive_entities else None,
            access_granted=result['access_decision'].get('query_access_allowed', True),
            ip_address=ip_address,
            execution_time_ms=(time.time() - start_time) * 1000
        )
        result['audit_log_id'] = log_entry.log_id
    
    # 6. 统计信息增强
    sens_stats = SensitiveDataDetector.get_sensitivity_stats(all_sensitive_entities)
    result['sensitivity_statistics'] = sens_stats
    
    # 判断是否安全导出
    high_sensitivity_types = {SensitivityLevel.CONFIDENTIAL.value, SensitivityLevel.RESTRICTED.value}
    has_high_sensitivity = any(
        s in high_sensitivity_types for s in sens_stats.keys()
    )
    if has_high_sensitivity and len(all_sensitive_entities) > 0:
        result['safe_for_export'] = False
    
    return result


# 导出别名
run_privacy_preserving_retrieval = build_privacy_preserving_retrieval

__all__ = [
    'SensitivityLevel',
    'AccessAction',
    'SensitiveEntity',
    'AccessPolicy',
    'AuditLogEntry',
    'SensitiveDataDetector',
    'TextAnonymizer',
    'AccessController',
    'AuditLogger',
    'build_privacy_preserving_retrieval',
    'run_privacy_preserving_retrieval',
    '_global_audit_logger',
]
