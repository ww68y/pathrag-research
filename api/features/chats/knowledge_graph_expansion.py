"""
V111: Dynamic Knowledge Graph Expansion (DKGE)
知识图谱动态扩展 - 自动发现和添加新知识到图谱

创新点：
1. 实体链接：将文本中的实体链接到知识图谱
2. 关系抽取：自动发现实体间的新关系
3. 知识融合：整合外部知识源
4. 冲突检测：识别并解决知识冲突
"""

import json
import hashlib
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict
from datetime import datetime
import re


@dataclass
class Entity:
    """实体数据结构"""
    id: str
    name: str
    entity_type: str
    aliases: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)
    source: str = ""
    confidence: float = 1.0
    created_at: datetime = field(default_factory=datetime.now)
    
    def generate_id(self) -> str:
        """生成实体ID"""
        content = f"{self.name}:{self.entity_type}"
        return hashlib.md5(content.encode()).hexdigest()[:12]


@dataclass
class Relation:
    """关系数据结构"""
    source_id: str
    target_id: str
    relation_type: str
    confidence: float
    evidence: List[str] = field(default_factory=list)
    properties: Dict[str, Any] = field(default_factory=dict)
    source: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    
    def generate_id(self) -> str:
        """生成关系ID"""
        content = f"{self.source_id}:{self.relation_type}:{self.target_id}"
        return hashlib.md5(content.encode()).hexdigest()[:12]


@dataclass
class KnowledgeConflict:
    """知识冲突数据结构"""
    conflict_type: str  # 'contradiction', 'duplicate', 'inconsistency'
    entities_involved: List[str]
    conflicting_statements: List[Dict]
    severity: str  # 'high', 'medium', 'low'
    suggested_resolution: str = ""


class EntityExtractor:
    """实体提取器"""
    
    # 实体类型模式
    ENTITY_PATTERNS = {
        'PERSON': [
            r'\b[A-Z][a-z]+ [A-Z][a-z]+\b',  # 人名
            r'\b(?:Mr\.?|Mrs\.?|Ms\.?|Dr\.?|Prof\.?) [A-Z][a-z]+\b'
        ],
        'ORGANIZATION': [
            r'\b[A-Z][a-z]* (?:University|College|Institute|Company|Corp|Inc|Ltd)\b',
            r'\b(?:Google|Microsoft|Apple|Amazon|Facebook|OpenAI|DeepMind)\b'
        ],
        'TECHNOLOGY': [
            r'\b(?:Transformer|BERT|GPT|LLM|RAG|GraphRAG|PathRAG|CNN|RNN|LSTM)\b',
            r'\b(?:Machine Learning|Deep Learning|Neural Network|AI|NLP)\b'
        ],
        'CONCEPT': [
            r'\b(?:Attention Mechanism|Self-Attention|Embedding|Tokenization)\b',
            r'\b(?:Knowledge Graph|Vector Database|Retrieval|Generation)\b'
        ],
        'DATASET': [
            r'\b(?:Wikipedia|Common Crawl|BookCorpus|C4|RedPajama)\b',
            r'\b(?:SQuAD|GLUE|SuperGLUE|MSMARCO|Natural Questions)\b'
        ]
    }
    
    def __init__(self):
        self.extracted_entities: Dict[str, Entity] = {}
    
    def extract_entities(self, text: str) -> List[Entity]:
        """从文本中提取实体"""
        entities = []
        
        for entity_type, patterns in self.ENTITY_PATTERNS.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    name = match.group()
                    
                    # 检查是否已存在
                    entity_key = f"{name.lower()}:{entity_type}"
                    if entity_key in self.extracted_entities:
                        continue
                    
                    entity = Entity(
                        id="",  # 稍后生成
                        name=name,
                        entity_type=entity_type,
                        aliases=[name.lower()],
                        source="text_extraction"
                    )
                    entity.id = entity.generate_id()
                    
                    self.extracted_entities[entity_key] = entity
                    entities.append(entity)
        
        return entities
    
    def link_to_existing(
        self, 
        new_entity: Entity, 
        existing_entities: List[Entity]
    ) -> Optional[Entity]:
        """将新实体链接到现有实体"""
        for existing in existing_entities:
            # 检查名称相似度
            if self._name_similarity(new_entity.name, existing.name) > 0.8:
                return existing
            
            # 检查别名
            if new_entity.name.lower() in [a.lower() for a in existing.aliases]:
                return existing
        
        return None
    
    def _name_similarity(self, name1: str, name2: str) -> float:
        """计算名称相似度"""
        # 简化的相似度计算
        set1 = set(name1.lower().split())
        set2 = set(name2.lower().split())
        
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0


class RelationExtractor:
    """关系提取器"""
    
    # 关系类型模式
    RELATION_PATTERNS = {
        'developed_by': [
            r'(\w+) (?:was |is )?developed by (\w+)',
            r'(\w+) created by (\w+)',
            r'(\w+) 由 (\w+) 开发'
        ],
        'uses': [
            r'(\w+) uses (\w+)',
            r'(\w+) employs (\w+)',
            r'(\w+) utilizes (\w+)',
            r'(\w+) 使用 (\w+)'
        ],
        'improves': [
            r'(\w+) improves (?:upon )?(\w+)',
            r'(\w+) outperforms (\w+)',
            r'(\w+) better than (\w+)',
            r'(\w+) 优于 (\w+)'
        ],
        'based_on': [
            r'(\w+) (?:is )?based on (\w+)',
            r'(\w+) built on (\w+)',
            r'(\w+) derived from (\w+)',
            r'(\w+) 基于 (\w+)'
        ],
        'part_of': [
            r'(\w+) is part of (\w+)',
            r'(\w+) component of (\w+)',
            r'(\w+) belongs to (\w+)'
        ],
        'related_to': [
            r'(\w+) related to (\w+)',
            r'(\w+) associated with (\w+)',
            r'(\w+) connected to (\w+)'
        ]
    }
    
    def __init__(self, entity_extractor: EntityExtractor):
        self.entity_extractor = entity_extractor
    
    def extract_relations(
        self, 
        text: str, 
        entities: List[Entity]
    ) -> List[Relation]:
        """从文本中提取关系"""
        relations = []
        entity_map = {e.name.lower(): e for e in entities}
        
        for relation_type, patterns in self.RELATION_PATTERNS.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    if len(match.groups()) >= 2:
                        source_name = match.group(1)
                        target_name = match.group(2)
                        
                        # 查找对应的实体
                        source_entity = entity_map.get(source_name.lower())
                        target_entity = entity_map.get(target_name.lower())
                        
                        if source_entity and target_entity:
                            relation = Relation(
                                source_id=source_entity.id,
                                target_id=target_entity.id,
                                relation_type=relation_type,
                                confidence=0.7,  # 基础置信度
                                evidence=[match.group()],
                                source="pattern_extraction"
                            )
                            relations.append(relation)
        
        return relations
    
    def infer_relations(
        self, 
        entities: List[Entity],
        existing_relations: List[Relation]
    ) -> List[Relation]:
        """基于已有关系推断新关系"""
        inferred = []
        
        # 传递性推断
        relation_graph = defaultdict(list)
        for rel in existing_relations:
            relation_graph[rel.source_id].append((rel.relation_type, rel.target_id))
        
        # 如果 A -> B 且 B -> C，推断 A -> C
        for source_id, connections in relation_graph.items():
            for rel_type, intermediate_id in connections:
                if intermediate_id in relation_graph:
                    for next_rel_type, target_id in relation_graph[intermediate_id]:
                        # 检查是否已存在直接关系
                        exists = any(
                            r.source_id == source_id and r.target_id == target_id
                            for r in existing_relations
                        )
                        if not exists:
                            inferred_relation = Relation(
                                source_id=source_id,
                                target_id=target_id,
                                relation_type=f"inferred_{rel_type}",
                                confidence=0.5,  # 推断的置信度较低
                                evidence=[f"Inferred from {source_id} -> {intermediate_id} -> {target_id}"],
                                source="transitive_inference"
                            )
                            inferred.append(inferred_relation)
        
        return inferred


class KnowledgeFusionEngine:
    """知识融合引擎"""
    
    def __init__(self):
        self.knowledge_sources: Dict[str, Dict] = {}
    
    def add_knowledge_source(
        self, 
        source_name: str, 
        entities: List[Entity],
        relations: List[Relation]
    ):
        """添加知识源"""
        self.knowledge_sources[source_name] = {
            'entities': {e.id: e for e in entities},
            'relations': relations,
            'added_at': datetime.now()
        }
    
    def fuse_knowledge(
        self, 
        target_entities: List[Entity],
        target_relations: List[Relation]
    ) -> Dict[str, Any]:
        """融合来自多个源的知识"""
        fused_entities = {e.id: e for e in target_entities}
        fused_relations = list(target_relations)
        conflicts = []
        
        for source_name, source_data in self.knowledge_sources.items():
            # 融合实体
            for entity_id, entity in source_data['entities'].items():
                if entity_id in fused_entities:
                    # 合并属性
                    fused_entities[entity_id].properties.update(entity.properties)
                    # 合并别名
                    fused_entities[entity_id].aliases = list(set(
                        fused_entities[entity_id].aliases + entity.aliases
                    ))
                else:
                    fused_entities[entity_id] = entity
            
            # 融合关系
            for relation in source_data['relations']:
                # 检查是否已存在
                exists = any(
                    r.source_id == relation.source_id and 
                    r.target_id == relation.target_id and
                    r.relation_type == relation.relation_type
                    for r in fused_relations
                )
                if not exists:
                    fused_relations.append(relation)
        
        return {
            'entities': list(fused_entities.values()),
            'relations': fused_relations,
            'conflicts': conflicts,
            'source_count': len(self.knowledge_sources)
        }


class ConflictDetector:
    """冲突检测器"""
    
    def detect_conflicts(
        self, 
        entities: List[Entity],
        relations: List[Relation]
    ) -> List[KnowledgeConflict]:
        """检测知识冲突"""
        conflicts = []
        
        # 检测实体冲突（重复或矛盾定义）
        entity_conflicts = self._detect_entity_conflicts(entities)
        conflicts.extend(entity_conflicts)
        
        # 检测关系冲突
        relation_conflicts = self._detect_relation_conflicts(relations)
        conflicts.extend(relation_conflicts)
        
        # 检测逻辑不一致
        logical_conflicts = self._detect_logical_inconsistencies(entities, relations)
        conflicts.extend(logical_conflicts)
        
        return conflicts
    
    def _detect_entity_conflicts(self, entities: List[Entity]) -> List[KnowledgeConflict]:
        """检测实体冲突"""
        conflicts = []
        entity_groups = defaultdict(list)
        
        # 按名称分组
        for entity in entities:
            entity_groups[entity.name.lower()].append(entity)
        
        # 检查同名但不同类型
        for name, group in entity_groups.items():
            if len(group) > 1:
                types = set(e.entity_type for e in group)
                if len(types) > 1:
                    conflict = KnowledgeConflict(
                        conflict_type='duplicate',
                        entities_involved=[e.id for e in group],
                        conflicting_statements=[
                            {'entity': e.name, 'type': e.entity_type}
                            for e in group
                        ],
                        severity='medium',
                        suggested_resolution=f"Merge entities or disambiguate '{name}'"
                    )
                    conflicts.append(conflict)
        
        return conflicts
    
    def _detect_relation_conflicts(self, relations: List[Relation]) -> List[KnowledgeConflict]:
        """检测关系冲突"""
        conflicts = []
        
        # 检查矛盾关系
        relation_pairs = defaultdict(list)
        for rel in relations:
            pair_key = tuple(sorted([rel.source_id, rel.target_id]))
            relation_pairs[pair_key].append(rel)
        
        for pair, rels in relation_pairs.items():
            if len(rels) > 1:
                # 检查是否有矛盾的关系类型
                relation_types = set(r.relation_type for r in rels)
                contradictory_pairs = [
                    ('improves', 'worsens'),
                    ('causes', 'prevents'),
                    ('enables', 'inhibits')
                ]
                
                for type1, type2 in contradictory_pairs:
                    if type1 in relation_types and type2 in relation_types:
                        conflict = KnowledgeConflict(
                            conflict_type='contradiction',
                            entities_involved=list(pair),
                            conflicting_statements=[
                                {'relation': r.relation_type, 'confidence': r.confidence}
                                for r in rels
                            ],
                            severity='high',
                            suggested_resolution="Review evidence and determine correct relation"
                        )
                        conflicts.append(conflict)
        
        return conflicts
    
    def _detect_logical_inconsistencies(
        self, 
        entities: List[Entity], 
        relations: List[Relation]
    ) -> List[KnowledgeConflict]:
        """检测逻辑不一致"""
        conflicts = []
        
        # 简化的逻辑检查：检查循环关系
        # 实际实现中可以使用图算法检测更复杂的模式
        
        return conflicts


# 主引擎类
class KnowledgeGraphExpansionEngine:
    """知识图谱扩展引擎"""
    
    def __init__(self):
        self.entity_extractor = EntityExtractor()
        self.relation_extractor = RelationExtractor(self.entity_extractor)
        self.fusion_engine = KnowledgeFusionEngine()
        self.conflict_detector = ConflictDetector()
        
        self.knowledge_graph = {
            'entities': {},
            'relations': []
        }
    
    def expand_from_text(
        self, 
        text: str,
        source: str = "unknown"
    ) -> Dict[str, Any]:
        """
        从文本扩展知识图谱
        
        Args:
            text: 输入文本
            source: 知识来源
        
        Returns:
            扩展结果
        """
        # 1. 提取实体
        entities = self.entity_extractor.extract_entities(text)
        
        # 2. 链接到现有实体
        linked_entities = []
        new_entities = []
        
        for entity in entities:
            existing = self.entity_extractor.link_to_existing(
                entity, 
                list(self.knowledge_graph['entities'].values())
            )
            if existing:
                linked_entities.append(existing)
            else:
                new_entities.append(entity)
                self.knowledge_graph['entities'][entity.id] = entity
        
        # 3. 提取关系
        all_entities = linked_entities + new_entities
        relations = self.relation_extractor.extract_relations(text, all_entities)
        
        # 4. 推断新关系
        inferred_relations = self.relation_extractor.infer_relations(
            all_entities,
            self.knowledge_graph['relations'] + relations
        )
        
        all_relations = relations + inferred_relations
        self.knowledge_graph['relations'].extend(all_relations)
        
        # 5. 检测冲突
        conflicts = self.conflict_detector.detect_conflicts(
            list(self.knowledge_graph['entities'].values()),
            self.knowledge_graph['relations']
        )
        
        return {
            'version': 'V111',
            'name': 'Dynamic Knowledge Graph Expansion (DKGE)',
            'source': source,
            'extraction_results': {
                'entities_extracted': len(entities),
                'new_entities': len(new_entities),
                'linked_entities': len(linked_entities),
                'relations_extracted': len(relations),
                'inferred_relations': len(inferred_relations)
            },
            'new_entities': [
                {
                    'id': e.id,
                    'name': e.name,
                    'type': e.entity_type,
                    'confidence': e.confidence
                }
                for e in new_entities
            ],
            'new_relations': [
                {
                    'source': r.source_id,
                    'target': r.target_id,
                    'type': r.relation_type,
                    'confidence': r.confidence,
                    'inferred': r.source == 'transitive_inference'
                }
                for r in all_relations
            ],
            'conflicts_detected': len(conflicts),
            'conflicts': [
                {
                    'type': c.conflict_type,
                    'severity': c.severity,
                    'suggestion': c.suggested_resolution
                }
                for c in conflicts
            ],
            'graph_stats': {
                'total_entities': len(self.knowledge_graph['entities']),
                'total_relations': len(self.knowledge_graph['relations'])
            }
        }
    
    def integrate_external_knowledge(
        self, 
        external_knowledge: Dict[str, Any],
        source_name: str
    ) -> Dict[str, Any]:
        """整合外部知识"""
        # 转换外部知识
        entities = [
            Entity(
                id=e.get('id', ''),
                name=e.get('name', ''),
                entity_type=e.get('type', 'unknown'),
                properties=e.get('properties', {}),
                source=source_name
            )
            for e in external_knowledge.get('entities', [])
        ]
        
        relations = [
            Relation(
                source_id=r.get('source', ''),
                target_id=r.get('target', ''),
                relation_type=r.get('type', 'related_to'),
                confidence=r.get('confidence', 0.5),
                source=source_name
            )
            for r in external_knowledge.get('relations', [])
        ]
        
        # 添加到融合引擎
        self.fusion_engine.add_knowledge_source(source_name, entities, relations)
        
        # 融合知识
        current_entities = list(self.knowledge_graph['entities'].values())
        current_relations = self.knowledge_graph['relations']
        
        fusion_result = self.fusion_engine.fuse_knowledge(
            current_entities,
            current_relations
        )
        
        # 更新知识图谱
        self.knowledge_graph['entities'] = {e.id: e for e in fusion_result['entities']}
        self.knowledge_graph['relations'] = fusion_result['relations']
        
        return {
            'source': source_name,
            'entities_added': len(entities),
            'relations_added': len(relations),
            'total_sources': fusion_result['source_count'],
            'conflicts': fusion_result['conflicts']
        }


# 便捷函数
def build_knowledge_graph_expansion(
    text: str,
    existing_graph: Optional[Dict] = None,
    source: str = "unknown"
) -> Dict[str, Any]:
    """
    构建知识图谱扩展
    
    这是V111创新的主要入口函数
    """
    engine = KnowledgeGraphExpansionEngine()
    
    # 如果提供了现有图谱，加载它
    if existing_graph:
        for entity_data in existing_graph.get('entities', []):
            entity = Entity(
                id=entity_data.get('id', ''),
                name=entity_data.get('name', ''),
                entity_type=entity_data.get('type', 'unknown')
            )
            engine.knowledge_graph['entities'][entity.id] = entity
    
    result = engine.expand_from_text(text, source)
    return result


# 示例用法
if __name__ == "__main__":
    text = """
    PathRAG is a novel retrieval-augmented generation system developed by researchers.
    It improves upon traditional RAG by using path-based retrieval.
    PathRAG uses knowledge graphs for better context understanding.
    BERT is a language model developed by Google.
    Transformer architecture is the foundation of BERT and GPT.
    """
    
    result = build_knowledge_graph_expansion(text, source="research_paper")
    print(json.dumps(result, indent=2, default=str))
