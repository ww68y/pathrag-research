"""
V142 - Incremental Graph Update Mechanism (增量图更新机制)
================================================================

支持知识图谱的增量式更新，无需全量重建，包括：
- 变更检测与差异分析
- 局部子图重构
- 影响范围评估
- 版本管理与回滚
- 一致性维护

核心能力：
- 检测新增/修改/删除的内容
- 计算变更的影响传播范围
- 执行局部图结构更新
- 维护多版本快照
- 确保图一致性
"""

from __future__ import annotations

import hashlib
import json
import time
from typing import Any, Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
from collections import defaultdict


class ChangeType(Enum):
    """变更类型"""
    ADD = "add"                    # 新增实体或关系
    MODIFY = "modify"              # 修改属性值
    DELETE = "delete"              # 删除实体或关系
    MERGE = "merge"                # 合并重复实体
    SPLIT = "split"                # 拆分复合实体
    RESTRUCTURE = "restructure"    # 结构重组（如层级调整）


class UpdateStatus(Enum):
    """更新状态"""
    PENDING = "pending"            # 待处理
    PROCESSING = "processing"      # 处理中
    COMPLETED = "completed"        # 已完成
    FAILED = "failed"              # 失败
    ROLLED_BACK = "rolled_back"    # 已回滚
    CONFLICT = "conflict"          # 存在冲突


@dataclass
class GraphEntity:
    """图谱实体"""
    entity_id: str
    name: str
    entity_type: str  # concept, instance, document, attribute
    properties: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[List[float]] = None
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    version: int = 1
    source_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'entity_id': self.entity_id,
            'name': self.name,
            'entity_type': self.entity_type,
            'properties': self.properties,
            'created_at': self.created_at.isoformat(),
            'updated_at': self.updated_at.isoformat(),
            'version': self.version,
            'source_id': self.source_id
        }
    
    def compute_hash(self) -> str:
        """计算实体的内容哈希"""
        content = f"{self.name}|{json.dumps(self.properties, sort_keys=True)}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass
class GraphRelation:
    """图谱关系"""
    relation_id: str
    source_entity_id: str
    target_entity_id: str
    relation_type: str  # related_to, part_of, causes, etc.
    properties: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    confidence: float = 0.8
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'relation_id': self.relation_id,
            'source': self.source_entity_id,
            'target': self.target_entity_id,
            'relation_type': self.relation_type,
            'weight': self.weight,
            'confidence': round(self.confidence, 4),
            'created_at': self.created_at.isoformat()
        }


@dataclass
class ChangeRecord:
    """变更记录"""
    change_id: str
    change_type: ChangeType
    target_type: str  # entity or relation
    target_id: str
    old_data: Optional[Dict] = None
    new_data: Optional[Dict] = None
    affected_entities: Set[str] = field(default_factory=set)
    affected_relations: Set[str] = field(default_factory=set)
    status: UpdateStatus = UpdateStatus.PENDING
    timestamp: datetime = field(default_factory=datetime.now)
    applied_by: str = "system"
    description: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'change_id': self.change_id,
            'change_type': self.change_type.value,
            'target_type': self.target_type,
            'target_id': self.target_id,
            'old_data': self.old_data,
            'new_data': self.new_data,
            'affected_entities': list(self.affected_entities),
            'affected_relations': list(self.affected_relations),
            'status': self.status.value,
            'timestamp': self.timestamp.isoformat(),
            'applied_by': self.applied_by,
            'description': self.description
        }


@dataclass
class GraphSnapshot:
    """图谱快照"""
    snapshot_id: str
    timestamp: datetime
    version: int
    entity_count: int
    relation_count: int
    checksum: str  # 整体校验和
    parent_snapshot_id: Optional[str] = None
    changes_since_parent: List[str] = field(default_factory=list)  # 变更ID列表
    metadata: Dict[str, Any] = field(default_factory=dict)


class ChangeDetector:
    """变更检测器"""
    
    @staticmethod
    def detect_changes(
        old_entities: List[GraphEntity],
        new_entities: List[GraphEntity],
        old_relations: List[GraphRelation],
        new_relations: List[GraphRelation]
    ) -> List[ChangeRecord]:
        """
        检测新旧版本之间的变更
        
        Returns:
            变更记录列表
        """
        changes = []
        
        # 构建查找索引
        old_ent_map = {e.entity_id: e for e in old_entities}
        new_ent_map = {e.entity_id: e for e in new_entities}
        old_rel_map = {(r.source_entity_id, r.target_entity_id): r for r in old_relations}
        new_rel_map = {(r.source_entity_id, r.target_entity_id): r for r in new_relations}
        
        # 检测实体变更
        all_entity_ids = set(old_ent_map.keys()) | set(new_ent_map.keys())
        
        for eid in all_entity_ids:
            old_ent = old_ent_map.get(eid)
            new_ent = new_ent_map.get(eid)
            
            if old_ent and not new_ent:
                # 实体被删除
                changes.append(ChangeRecord(
                    change_id=f"del_ent_{eid}",
                    change_type=ChangeType.DELETE,
                    target_type="entity",
                    target_id=eid,
                    old_data=old_ent.to_dict(),
                    description=f"Deleted entity: {old_ent.name}"
                ))
            
            elif not old_ent and new_ent:
                # 新增实体
                changes.append(ChangeRecord(
                    change_id=f"add_ent_{eid}",
                    change_type=ChangeType.ADD,
                    target_type="entity",
                    target_id=eid,
                    new_data=new_ent.to_dict(),
                    description=f"Added entity: {new_ent.name}"
                ))
            
            elif old_ent and new_ent:
                # 检查是否有修改
                if old_ent.compute_hash() != new_ent.compute_hash():
                    changes.append(ChangeRecord(
                        change_id=f"mod_ent_{eid}",
                        change_type=ChangeType.MODIFY,
                        target_type="entity",
                        target_id=eid,
                        old_data=old_ent.to_dict(),
                        new_data=new_ent.to_dict(),
                        description=f"Modified entity: {old_ent.name}"
                    ))
        
        # 检测关系变更
        all_rel_keys = set(old_rel_map.keys()) | set(new_rel_map.keys())
        
        for rel_key in all_rel_keys:
            old_rel = old_rel_map.get(rel_key)
            new_rel = new_rel_map.get(rel_key)
            
            if old_rel and not new_rel:
                changes.append(ChangeRecord(
                    change_id=f"del_rel_{old_rel.relation_id}",
                    change_type=ChangeType.DELETE,
                    target_type="relation",
                    target_id=old_rel.relation_id,
                    old_data=old_rel.to_dict(),
                    description=f"Deleted relation: {old_rel.relation_type}"
                ))
            
            elif not old_rel and new_rel:
                changes.append(ChangeRecord(
                    change_id=f"add_rel_{new_rel.relation_id}",
                    change_type=ChangeType.ADD,
                    target_type="relation",
                    target_id=new_rel.relation_id,
                    new_data=new_rel.to_dict(),
                    description=f"Added relation: {new_rel.relation_type}"
                ))
        
        return changes


class ImpactAnalyzer:
    """影响范围分析器"""
    
    @staticmethod
    def analyze_impact(
        change: ChangeRecord,
        entities: List[GraphEntity],
        relations: List[GraphRelation],
        max_depth: int = 3
    ) -> Dict[str, Any]:
        """
        分析变更的影响范围
        
        Returns:
            {
                'directly_affected': [entity_ids],
                'indirectly_affected': [entity_ids],
                'affected_relation_types': [...],
                'impact_score': float,
                'risk_level': str
            }
        """
        result = {
            'directly_affected': [],
            'indirectly_affected': [],
            'affected_relation_types': [],
            'impact_score': 0.0,
            'risk_level': 'low'
        }
        
        # 直接影响
        if change.target_type == "entity":
            result['directly_affected'].append(change.target_id)
            
            # 找到所有关联的关系
            for rel in relations:
                if rel.source_entity_id == change.target_id or rel.target_entity_id == change.target_id:
                    result['directly_affected'].extend([
                        rel.source_entity_id,
                        rel.target_entity_id
                    ])
                    if rel.relation_type not in result['affected_relation_types']:
                        result['affected_relation_types'].append(rel.relation_type)
        
        elif change.target_type == "relation":
            # 关系变更影响两端实体
            if change.old_data:
                result['directly_affected'].extend([
                    change.old_data.get('source'),
                    change.old_data.get('target')
                ])
            if change.new_data:
                result['directly_affected'].extend([
                    change.new_data.get('source'),
                    change.new_data.get('target')
                ])
        
        # 去重
        result['directly_affected'] = list(set(result['directly_affected']))
        
        # 间接影响（BFS遍历）
        visited = set(result['directly_affected'])
        queue = list(visited)
        current_depth = 0
        
        while queue and current_depth < max_depth:
            next_queue = []
            for entity_id in queue:
                for rel in relations:
                    neighbor = None
                    if rel.source_entity_id == entity_id:
                        neighbor = rel.target_entity_id
                    elif rel.target_entity_id == entity_id:
                        neighbor = rel.source_entity_id
                    
                    if neighbor and neighbor not in visited:
                        visited.add(neighbor)
                        result['indirectly_affected'].append(neighbor)
                        next_queue.append(neighbor)
            
            queue = next_queue
            current_depth += 1
        
        # 计算影响分数
        total_entities = len(entities)
        total_affected = len(result['directly_affected']) + len(result['indirectly_affected'])
        result['impact_score'] = min(1.0, total_affected / max(total_entities, 1))
        
        # 风险等级评估
        if result['impact_score'] > 0.5:
            result['risk_level'] = 'critical'
        elif result['impact_score'] > 0.25:
            result['risk_level'] = 'high'
        elif result['impact_score'] > 0.1:
            result['risk_level'] = 'medium'
        else:
            result['risk_level'] = 'low'
        
        return result


class GraphVersionManager:
    """图谱版本管理器"""
    
    def __init__(self):
        self.snapshots: List[GraphSnapshot] = []
        self.current_version = 0
        self.max_snapshots = 50  # 最大保留快照数
    
    def create_snapshot(
        self,
        entities: List[GraphEntity],
        relations: List[GraphRelation],
        changes: List[ChangeRecord]
    ) -> GraphSnapshot:
        """
        创建新快照
        
        Returns:
            创建的快照
        """
        # 计算整体校验和
        all_hashes = []
        for ent in entities:
            all_hashes.append(ent.compute_hash())
        for rel in relations:
            rel_content = f"{rel.source_entity_id}|{rel.target_entity_id}|{rel.relation_type}"
            all_hashes.append(hashlib.sha256(rel_content.encode()).hexdigest()[:16])
        
        combined = "|".join(sorted(all_hashes))
        checksum = hashlib.sha256(combined.encode()).hexdigest()[:32]
        
        parent_id = self.snapshots[-1].snapshot_id if self.snapshots else None
        
        snapshot = GraphSnapshot(
            snapshot_id=f"snap_{len(self.snapshots)}_{int(time.time())}",
            timestamp=datetime.now(),
            version=self.current_version + 1,
            entity_count=len(entities),
            relation_count=len(relations),
            checksum=checksum,
            parent_snapshot_id=parent_id,
            changes_since_parent=[c.change_id for c in changes],
            metadata={
                'change_count': len(changes),
                'change_types': list(set(c.change_type.value for c in changes))
            }
        )
        
        self.snapshots.append(snapshot)
        self.current_version = snapshot.version
        
        # 清理旧快照
        if len(self.snapshots) > self.max_snapshots:
            self.snapshots = self.snapshots[-self.max_snapshots:]
        
        return snapshot
    
    def get_snapshot(self, version: Optional[int] = None) -> Optional[GraphSnapshot]:
        """获取指定版本的快照"""
        if version is None:
            return self.snapshots[-1] if self.snapshots else None
        
        for snap in self.snapshots:
            if snap.version == version:
                return snap
        return None
    
    def get_history(self, limit: int = 20) -> List[Dict]:
        """获取版本历史"""
        history = []
        for snap in self.snapshots[-limit:]:
            history.append({
                'snapshot_id': snap.snapshot_id,
                'version': snap.version,
                'timestamp': snap.timestamp.isoformat(),
                'entity_count': snap.entity_count,
                'relation_count': snap.relation_count,
                'changes_count': len(snap.changes_since_parent),
                'metadata': snap.metadata
            })
        return history
    
    def can_rollback(self, target_version: int) -> Tuple[bool, str]:
        """检查是否可以回滚到指定版本"""
        if target_version >= self.current_version:
            return (False, "Cannot rollback to current or future version")
        
        target_snap = self.get_snapshot(target_version)
        if not target_snap:
            return (False, f"Version {target_version} not found")
        
        # 检查中间是否有不可逆操作
        recent_changes = sum(len(s.changes_since_parent) 
                           for s in self.snapshots 
                           if s.version > target_version)
        
        if recent_changes > 100:
            return (True, f"Warning: {recent_changes} changes will be reverted")
        
        return (True, f"Can rollback to version {target_version}")


def build_incremental_update_pipeline(
    current_entities: List[Dict[str, Any]],
    current_relations: List[Dict[str, Any]],
    new_documents: List[Dict[str, Any]],
    deleted_document_ids: Optional[List[str]] = None,
    update_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    构建增量更新管道
    
    Args:
        current_entities: 当前实体列表
        current_relations: 当前关系列表
        new_documents: 新增/更新的文档
        deleted_document_ids: 要删除的文档ID列表
        update_config: 更新配置
    
    Returns:
        {
            'pipeline': str,
            'detected_changes': [...],
            'impact_analysis': {...},
            'update_plan': {...},
            'snapshot_info': {...},
            'statistics': {...},
            'recommendations': [...]
        }
    """
    start_time = time.time()
    
    config = update_config or {}
    max_impact_depth = config.get('max_impact_depth', 3)
    auto_apply = config.get('auto_apply', False)
    
    result = {
        'pipeline': 'V142_Incremental_Graph_Update_v1',
        'detected_changes': [],
        'impact_analysis': {},
        'update_plan': {},
        'snapshot_info': {},
        'statistics': {
            'entities_before': len(current_entities),
            'relations_before': len(current_relations),
            'new_documents': len(new_documents),
            'deleted_documents': len(deleted_document_ids or []),
            'total_changes_detected': 0,
            'high_risk_changes': 0
        },
        'recommendations': []
    }
    
    # 1. 将字典转换为对象
    old_entities = [GraphEntity(**e) if isinstance(e, dict) else e for e in current_entities]
    old_relations = [GraphRelation(**r) if isinstance(r, dict) else r for r in current_relations]
    
    # 2. 从新文档提取实体和关系（模拟）
    new_entities, new_relations = _extract_from_new_documents(new_documents)
    
    # 3. 检测变更
    detector = ChangeDetector()
    changes = detector.detect_changes(old_entities, new_entities, old_relations, new_relations)
    
    # 4. 处理删除的文档
    if deleted_document_ids:
        for doc_id in deleted_document_ids:
            # 找出属于该文档的所有实体
            for ent in old_entities:
                if ent.source_id == doc_id:
                    changes.append(ChangeRecord(
                        change_id=f"del_doc_{doc_id}_{ent.entity_id}",
                        change_type=ChangeType.DELETE,
                        target_type="entity",
                        target_id=ent.entity_id,
                        old_data=ent.to_dict(),
                        description=f"Deleted with document: {doc_id}"
                    ))
    
    result['detected_changes'] = [c.to_dict() for c in changes]
    result['statistics']['total_changes_detected'] = len(changes)
    
    # 5. 分析影响范围
    impact_results = []
    high_risk_count = 0
    
    for change in changes:
        impact = ImpactAnalyzer.analyze_impact(change, old_entities + new_entities, 
                                               old_relations + new_relations, max_impact_depth)
        change.affected_entities = set(impact['directly_affected'] + impact['indirectly_affected'])
        impact_results.append({
            'change_id': change.change_id,
            **impact
        })
        
        if impact['risk_level'] in ['high', 'critical']:
            high_risk_count += 1
    
    result['statistics']['high_risk_changes'] = high_risk_count
    result['impact_analysis'] = {
        'total_analyzed': len(impact_results),
        'by_risk_level': defaultdict(int),
        'max_impact_score': max((i['impact_score'] for i in impact_results), default=0),
        'details': impact_results[:10]  # 只返回前10个详细结果
    }
    
    for imp in impact_results:
        result['impact_analysis']['by_risk_level'][imp['risk_level']] += 1
    
    # 6. 生成更新计划
    update_plan = _generate_update_plan(changes, impact_results, auto_apply)
    result['update_plan'] = update_plan
    
    # 7. 创建快照预览
    version_manager = GraphVersionManager()
    snapshot_preview = version_manager.create_snapshot(
        new_entities, new_relations, changes
    )
    result['snapshot_info'] = {
        'preview_snapshot_id': snapshot_preview.snapshot_id,
        'new_version': snapshot_preview.version,
        'entity_count_after': snapshot_preview.entity_count,
        'relation_count_after': snapshot_preview.relation_count,
        'can_rollback': version_manager.can_rollback(snapshot_preview.version - 1)[0]
    }
    
    # 8. 生成建议
    recommendations = _generate_recommendations(changes, impact_results, result['statistics'])
    result['recommendations'] = recommendations
    
    result['processing_time_ms'] = (time.time() - start_time) * 1000
    
    return result


def _extract_from_new_documents(documents: List[Dict]) -> Tuple[List[GraphEntity], List[GraphRelation]]:
    """从新文档提取实体和关系（简化版）"""
    entities = []
    relations = []
    
    for doc_idx, doc in enumerate(documents):
        doc_text = doc.get('text', '')
        doc_id = doc.get('doc_id', f'doc_{doc_idx}')
        
        # 简单的实体提取：基于名词短语
        words = doc_text.split()
        noun_candidates = []
        
        for i, word in enumerate(words):
            # 简化：大写开头的词作为候选实体
            if word[0:1].isupper() and len(word) > 2:
                # 尝试组合连续的大写词
                phrase = word
                j = i + 1
                while j < len(words) and words[j][0:1].isupper() and len(words[j]) > 1:
                    phrase += " " + words[j]
                    j += 1
                
                if phrase not in [e.name for e in entities]:
                    entity = GraphEntity(
                        entity_id=f"ent_{hash(phrase) % 100000}",
                        name=phrase,
                        entity_type="concept",
                        source_id=doc_id,
                        properties={'source_document': doc_id}
                    )
                    entities.append(entity)
                    noun_candidates.append(entity)
        
        # 简单的关系提取：同一句子中的实体对
        sentences = doc_text.split('. ')
        for sentence in sentences[:20]:  # 限制处理量
            sent_entities = [e for e in noun_candidates if e.name in sentence]
            for i, e1 in enumerate(sent_entities):
                for e2 in sent_entities[i+1:i+3]:  # 每个实体最多连接2个其他实体
                    relation = GraphRelation(
                        relation_id=f"rel_{hash(e1.entity_id + e2.entity_id) % 100000}",
                        source_entity_id=e1.entity_id,
                        target_entity_id=e2.entity_id,
                        relation_type="related_to",
                        confidence=0.7
                    )
                    relations.append(relation)
    
    return entities, relations


def _generate_update_plan(
    changes: List[ChangeRecord],
    impacts: List[Dict],
    auto_apply: bool
) -> Dict[str, Any]:
    """生成更新执行计划"""
    plan = {
        'phases': [],
        'estimated_duration_ms': 0,
        'requires_approval': False,
        'auto_applicable_changes': [],
        'manual_review_required': []
    }
    
    # 分阶段处理
    phases = [
        {'name': 'low_risk_additions', 'changes': [], 'description': 'Low-risk additions'},
        {'name': 'modifications', 'changes': [], 'description': 'Property modifications'},
        {'name': 'deletions', 'changes': [], 'description': 'Deletions (higher risk)'},
        {'name': 'structural_changes', 'changes': [], 'description': 'Structural reorganizations'},
    ]
    
    for change, impact in zip(changes, impacts):
        change_dict = {
            'change_id': change.change_id,
            'type': change.change_type.value,
            'target': change.target_id,
            'risk': impact.get('risk_level', 'unknown')
        }
        
        if impact.get('risk_level') in ['low'] and change.change_type == ChangeType.ADD:
            phases[0]['changes'].append(change_dict)
            plan['auto_applicable_changes'].append(change.change_id)
        elif change.change_type == ChangeType.MODIFY:
            phases[1]['changes'].append(change_dict)
        elif change.change_type == ChangeType.DELETE:
            phases[2]['changes'].append(change_dict)
            plan['manual_review_required'].append(change.change_id)
        else:
            phases[3]['changes'].append(change_dict)
            plan['manual_review_required'].append(change.change_id)
    
    plan['phases'] = [{'name': p['name'], 'change_count': len(p['changes']), 'description': p['description']} 
                      for p in phases if p['changes']]
    
    plan['estimated_duration_ms'] = len(changes) * 15  # 估算每条变更15ms
    plan['requires_approval'] = len(plan['manual_review_required']) > 0 and not auto_apply
    
    return plan


def _generate_recommendations(
    changes: List[ChangeRecord],
    impacts: List[Dict],
    stats: Dict
) -> List[Dict]:
    """生成更新建议"""
    recommendations = []
    
    # 基于统计的建议
    if stats['total_changes_detected'] == 0:
        recommendations.append({
            'type': 'info',
            'message': 'No changes detected. The knowledge graph is up to date.',
            'action': 'none'
        })
    
    if stats['high_risk_changes'] > 0:
        recommendations.append({
            'type': 'warning',
            'message': f"{stats['high_risk_changes']} high-risk changes detected. Manual review recommended.",
            'action': 'review'
        })
    
    add_count = sum(1 for c in changes if c.change_type == ChangeType.ADD)
    del_count = sum(1 for c in changes if c.change_type == ChangeType.DELETE)
    
    if add_count > 10:
        recommendations.append({
            'type': 'info',
            'message': f"Large batch of additions ({add_count}). Consider batching into smaller updates.",
            'action': 'batch'
        })
    
    if del_count > 5:
        recommendations.append({
            'type': 'warning',
            'message': f"Multiple deletions ({del_count}). Verify data integrity after update.",
            'action': 'verify'
        })
    
    # 基于影响的建议
    critical_impacts = [i for i in impacts if i.get('risk_level') == 'critical']
    if critical_impacts:
        recommendations.append({
            'type': 'critical',
            'message': f"{len(critical_impacts)} changes may have widespread impact. Create backup before applying.",
            'action': 'backup'
        })
    
    if not recommendations:
        recommendations.append({
            'type': 'success',
            'message': 'Update looks safe to apply.',
            'action': 'apply'
        })
    
    return recommendations


# 导出别名
run_incremental_graph_update = build_incremental_update_pipeline

__all__ = [
    'ChangeType',
    'UpdateStatus',
    'GraphEntity',
    'GraphRelation',
    'ChangeRecord',
    'GraphSnapshot',
    'ChangeDetector',
    'ImpactAnalyzer',
    'GraphVersionManager',
    'build_incremental_update_pipeline',
    'run_incremental_graph_update',
]
