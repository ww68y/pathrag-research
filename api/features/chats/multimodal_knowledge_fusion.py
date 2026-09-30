"""
V136 - Multimodal Knowledge Fusion Engine (多模态知识融合引擎)
==========================================================

将图像、表格、公式、代码作为一等公民融入知识图谱推理过程。
支持图表理解、数学公式语义解析、代码关系建模等。

核心能力：
- 图表实体抽取与语义理解
- 表格结构化知识提取
- 数学公式符号解析
- 代码依赖关系建模
- 跨模态证据对齐与融合
"""

from __future__ import annotations

import re
import base64
import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum


class ModalityType(Enum):
    """模态类型枚举"""
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    FORMULA = "formula"
    CODE = "code"
    CHART = "chart"
    DIAGRAM = "diagram"


@dataclass
class MultimodalEntity:
    """多模态实体表示"""
    entity_id: str
    name: str
    modality: ModalityType
    content: str  # 原始内容或描述
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_region: Optional[Dict[str, int]] = None  # {x, y, width, height} for images
    confidence: float = 0.0
    extracted_by: str = "base"  # base, vision_llm, parser


@dataclass
class CrossModalRelation:
    """跨模态关系"""
    relation_id: str
    source_entity_id: str
    target_entity_id: str
    source_modality: ModalityType
    target_modality: ModalityType
    relation_type: str  # refers_to, describes, explains, implements, visualizes
    description: str
    confidence: float = 0.0
    alignment_score: float = 0.0  # 跨模态对齐分数


@dataclass
class FusionEvidence:
    """融合后的证据"""
    evidence_id: str
    primary_modality: ModalityType
    content_summary: str
    supporting_entities: List[str]
    cross_modal_links: List[str]
    fused_embedding: Optional[List[float]] = None
    relevance_score: float = 0.0
    completeness_score: float = 0.0  # 多模态信息完整度


class TableStructureExtractor:
    """表格结构化提取器"""
    
    # 表格模式正则
    TABLE_PATTERNS = {
        'markdown_table': r'\|(.+)\|\n\|[\s\-:]+\|',
        'csv_like': r'^(.+?,)+.+$',
        'html_table': r'<table.*?>.*?</table>',
        'aligned_table': r'^[\s\|]*:?-+:?[\s\|]*$',
    }
    
    # 数值模式
    NUMERIC_PATTERN = r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?%?'
    
    @classmethod
    def extract(cls, table_text: str) -> Dict[str, Any]:
        """
        从表格文本中提取结构化知识
        
        Returns:
            {
                'headers': [...],
                'rows': [[...], ...],
                'entities': [{'name', 'type', 'value', ...}],
                'relations': [{'header', 'row_key', 'value'}],
                'statistics': {'row_count', 'col_count', 'numeric_cols'}
            }
        """
        result = {
            'headers': [],
            'rows': [],
            'entities': [],
            'relations': [],
            'statistics': {}
        }
        
        # 尝试解析 Markdown 表格
        parsed = cls._parse_markdown_table(table_text)
        if parsed:
            result.update(parsed)
        
        # 提取数值实体和关系
        entities, relations = cls._extract_table_knowledge(
            result.get('headers', []),
            result.get('rows', [])
        )
        result['entities'] = entities
        result['relations'] = relations
        
        # 统计信息
        result['statistics'] = {
            'row_count': len(result['rows']),
            'col_count': len(result['headers']),
            'numeric_entities': sum(1 for e in entities if e.get('type') == 'numeric')
        }
        
        return result
    
    @classmethod
    def _parse_markdown_table(cls, text: str) -> Optional[Dict[str, Any]]:
        """解析 Markdown 格式表格"""
        lines = text.strip().split('\n')
        if len(lines) < 2:
            return None
        
        # 检查是否是表格格式
        if not all('|' in line for line in lines[:min(3, len(lines))]):
            return None
        
        # 解析表头
        headers = [h.strip() for h in lines[0].split('|')]
        headers = [h for h in headers if h]  # 移除空字符串
        
        # 跳过分隔行
        data_lines = [line for line in lines[1:] if not set(line.strip()) <= {'|', '-', ' ', ':'}]
        
        rows = []
        for line in data_lines:
            cells = [c.strip() for c in line.split('|')]
            cells = [c for c in cells if c or True]  # 保留空单元格位置
            if len(cells) >= len(headers):
                rows.append(cells[:len(headers)])
        
        return {'headers': headers, 'rows': rows} if rows else None
    
    @classmethod
    def _extract_table_knowledge(cls, headers: List[str], rows: List[List[str]]) -> Tuple[List[Dict], List[Dict]]:
        """从表格中提取知识和关系"""
        entities = []
        relations = []
        
        # 为每个表头创建属性实体
        for header in headers:
            entities.append({
                'name': header,
                'type': 'table_attribute',
                'category': cls._infer_header_type(header),
                'source': 'table'
            })
        
        # 提取每行的数据作为实体和关系
        for row_idx, row in enumerate(rows):
            row_key = row[0] if row else f"row_{row_idx}"
            
            # 行键作为实体
            entities.append({
                'name': row_key,
                'type': 'table_row_entity',
                'source': 'table'
            })
            
            # 每个单元格值创建关系
            for col_idx, cell in enumerate(row):
                if col_idx >= len(headers):
                    continue
                header = headers[col_idx]
                
                # 提取数值
                numeric_match = re.search(cls.NUMERIC_PATTERN, cell)
                
                relations.append({
                    'subject': row_key,
                    'predicate': header,
                    'object': cell,
                    'object_type': 'numeric' if numeric_match else 'text',
                    'numeric_value': float(numeric_match.group()) if numeric_match else None,
                    'row_index': row_idx,
                    'col_index': col_idx
                })
        
        return entities, relations
    
    @classmethod
    def _infer_header_type(cls, header: str) -> str:
        """推断表头类型"""
        header_lower = header.lower()
        if any(w in header_lower for w in ['amount', 'price', 'cost', 'revenue', '$']):
            return 'financial'
        elif any(w in header_lower for w in ['date', 'time', 'year', 'month']):
            return 'temporal'
        elif any(w in header_lower for w in ['count', 'number', 'quantity', '%', 'rate']):
            return 'metric'
        elif any(w in header_lower for w in ['name', 'title', 'id', 'description']):
            return 'identifier'
        else:
            return 'general'


class FormulaSemanticParser:
    """数学公式语义解析器"""
    
    # 常见数学符号映射
    SYMBOL_MAP = {
        '+': 'addition',
        '-': 'subtraction',
        '*': 'multiplication',
        '/': 'division',
        '=': 'equals',
        '≈': 'approximately_equals',
        '>': 'greater_than',
        '<': 'less_than',
        '≥': 'greater_equal',
        '≤': 'less_equal',
        '∑': 'summation',
        '∏': 'product',
        '∫': 'integral',
        '∂': 'partial_derivative',
        '∇': 'gradient',
        '√': 'square_root',
        '∞': 'infinity',
        'π': 'pi',
        '∈': 'element_of',
        '⊂': 'subset_of',
        '∪': 'union',
        '∩': 'intersection',
        '→': 'implies',
        '⇔': 'iff',
        '∀': 'for_all',
        '∃': 'exists',
        'lim': 'limit',
        'sin': 'sine',
        'cos': 'cosine',
        'tan': 'tangent',
        'log': 'logarithm',
        'ln': 'natural_log',
        'exp': 'exponential',
    }
    
    # LaTeX 模式
    LATEX_PATTERNS = [
        r'\\frac\{([^}]+)\}\{([^}]+)\}',  # 分数
        r'\\sqrt\{([^}]+)\}',  # 平方根
        r'\^(\{[^}]+\}|\w)',  # 上标
        r'_(\{[^}]+\}|\w)',  # 下标
        r'\\int_[^{}]*(?:\^{[^}]*)?\{([^}]*)\}',  # 积分
        r'\\sum_(?:[^=]*)=(?:[^{}]*)\^(?:\{([^}]*)\}|(\w+))',  # 求和
    ]
    
    @classmethod
    def parse(cls, formula_str: str) -> Dict[str, Any]:
        """
        解析数学公式，提取语义结构
        
        Returns:
            {
                'original': str,
                'symbols': [{symbol, meaning, position}],
                'structure': 'equation | inequality | expression | identity',
                'variables': [str],
                'constants': [str],
                'operations': [{op, operands, result_type}],
                'domain': 'algebraic | calculus | statistical | geometric',
                'natural_description': str
            }
        """
        result = {
            'original': formula_str,
            'symbols': [],
            'structure': cls._infer_structure(formula_str),
            'variables': [],
            'constants': [],
            'operations': [],
            'domain': cls._infer_domain(formula_str),
            'natural_description': ''
        }
        
        # 提取符号
        for symbol, meaning in cls.SYMBOL_MAP.items():
            positions = [m.start() for m in re.finditer(re.escape(symbol), formula_str)]
            for pos in positions:
                result['symbols'].append({
                    'symbol': symbol,
                    'meaning': meaning,
                    'position': pos
                })
        
        # 提取变量（单个小写/大写字母，排除已知常量）
        known_constants = {'π', 'e', 'i', '∞'}
        variable_pattern = r'\b([a-hj-zA-Z])\b(?![{(])'
        variables = set(re.findall(variable_pattern, formula_str))
        result['variables'] = [v for v in variables if v not in known_constants]
        
        # 提取常量
        result['constants'] = [v for v in variables if v in known_constants]
        
        # 提取操作
        result['operations'] = cls._extract_operations(formula_str)
        
        # 生成自然语言描述
        result['natural_description'] = cls._generate_description(
            formula_str, 
            result['structure'],
            result['operations']
        )
        
        return result
    
    @classmethod
    def _infer_structure(cls, formula: str) -> str:
        """推断公式类型"""
        if '=' in formula and '→' not in formula:
            if '<' in formula or '>' in formula:
                return 'inequality'
            return 'equation'
        elif '→' in formula or '⇔' in formula:
            return 'implication'
        elif re.search(r'[+\-*/]', formula):
            return 'expression'
        else:
            return 'identity'
    
    @classmethod
    def _infer_domain(cls, formula: str) -> str:
        """推断数学领域"""
        if any(op in formula for op in ['∫', '∂', 'd/', 'lim']):
            return 'calculus'
        elif any(op in formula for op in ['∑', '∏', 'E[', 'Var[', 'Pr(']):
            return 'statistical'
        elif any(op in formula for op in ['∇', '⋅', '×', '‖']):
            return 'geometric'
        elif any(op in formula for op in ['∧', '∨', '¬', '⊢']):
            return 'logical'
        else:
            return 'algebraic'
    
    @classmethod
    def _extract_operations(cls, formula: str) -> List[Dict]:
        """提取数学操作"""
        operations = []
        
        # 简单算术操作
        arithmetic_ops = [(r'([a-zA-Z]\w*)\s*\+\s*([a-zA-Z\d]+)', 'addition'),
                         (r'([a-zA-Z]\w*)\s*-\s*([a-zA-Z\d]+)', 'subtraction'),
                         (r'([a-zA-Z]\w*)\s*\*\s*([a-zA-Z\d]+)', 'multiplication'),
                         (r'([a-zA-Z]\w*)\s*/\s*([a-zA-Z\d]+)', 'division')]
        
        for pattern, op_name in arithmetic_ops:
            matches = re.findall(pattern, formula)
            for match in matches:
                operations.append({
                    'operation': op_name,
                    'operands': list(match),
                    'result_type': 'scalar'
                })
        
        return operations
    
    @classmethod
    def _generate_description(cls, formula: str, structure: str, operations: List[Dict]) -> str:
        """生成公式的自然语言描述"""
        desc_parts = []
        
        if structure == 'equation':
            parts = formula.split('=')
            if len(parts) == 2:
                desc_parts.append(f"defines the equality between {parts[0].strip()} and {parts[1].strip()}")
        elif structure == 'inequality':
            desc_parts.append("specifies a constraint or bound")
        
        if operations:
            op_names = [op['operation'] for op in operations[:3]]
            desc_parts.append(f"involving {', '.join(op_names)}")
        
        return "This formula ".join(desc_parts) if desc_parts else f"Mathematical expression: {formula}"


class CodeDependencyAnalyzer:
    """代码依赖关系分析器"""
    
    # 代码模式
    CODE_PATTERNS = {
        'function_def': r'(?:def|function|func)\s+(\w+)\s*\(',
        'class_def': r'class\s+(\w+)(?:\s*[\(:]\s*(\w+)?)?',
        'variable_assign': r'(\w+)\s*=.*',
        'import_statement': r'(?:import|from)\s+([\w.]+)',
        'function_call': r'(\w+)\s*\(',
        'comment': r'#.*$|//.*$|/\*.*\*/',
    }
    
    @classmethod
    def analyze(cls, code_str: str, language: str = "python") -> Dict[str, Any]:
        """
        分析代码，提取结构和依赖关系
        
        Returns:
            {
                'language': str,
                'functions': [{name, params, calls, complexity_estimate}],
                'classes': [{name, inherits, methods}],
                'imports': [str],
                'variables': [{name, type_hint}],
                'dependencies': [{from_func, to_func, type: calls|overrides|uses}],
                'call_graph': {node: [targets]},
                'entities_for_kg': [...],
                'relations_for_kg': [...]
            }
        """
        result = {
            'language': language,
            'functions': [],
            'classes': [],
            'imports': [],
            'variables': [],
            'dependencies': [],
            'call_graph': {},
            'entities_for_kg': [],
            'relations_for_kg': []
        }
        
        # 提取导入
        result['imports'] = re.findall(cls.CODE_PATTERNS['import_statement'], code_str, re.MULTILINE)
        
        # 提取函数定义
        func_matches = re.finditer(cls.CODE_PATTERNS['function_def'], code_str)
        for match in func_matches:
            func_name = match.group(1)
            # 找到函数体（简化处理）
            func_info = {
                'name': func_name,
                'params': cls._extract_params(code_str, match.end()),
                'calls': cls._find_function_calls_in_scope(code_str, match.end()),
                'complexity_estimate': cls._estimate_complexity(code_str, match.end())
            }
            result['functions'].append(func_info)
            
            # 创建 KG 实体
            result['entities_for_kg'].append({
                'name': func_name,
                'type': 'function',
                'modality': 'code',
                'metadata': {'language': language, 'params': func_info['params']}
            })
        
        # 提取类定义
        class_matches = re.finditer(cls.CODE_PATTERNS['class_def'], code_str)
        for match in class_matches:
            class_name = match.group(1)
            parent_class = match.group(2) if match.lastindex >= 2 else None
            
            class_info = {
                'name': class_name,
                'inherits': parent_class,
                'methods': []
            }
            result['classes'].append(class_info)
            
            # 创建 KG 实体
            result['entities_for_kg'].append({
                'name': class_name,
                'type': 'class',
                'modality': 'code',
                'metadata': {'language': language, 'parent': parent_class}
            })
            
            if parent_class:
                result['relations_for_kg'].append({
                    'subject': class_name,
                    'predicate': 'inherits_from',
                    'object': parent_class,
                    'modality': 'code'
                })
        
        # 构建调用图
        for func in result['functions']:
            callers = [f['name'] for f in result['functions'] if func['name'] in f['calls']]
            result['call_graph'][func['name']] = callers
            
            for called_func in func['calls']:
                result['dependencies'].append({
                    'from_func': func['name'],
                    'to_func': called_func,
                    'type': 'calls'
                })
                
                result['relations_for_kg'].append({
                    'subject': func['name'],
                    'predicate': 'calls',
                    'object': called_func,
                    'modality': 'code'
                })
        
        return result
    
    @classmethod
    def _extract_params(cls, code: str, start_pos: int) -> List[str]:
        """提取函数参数"""
        # 找到匹配的括号
        depth = 0
        param_start = start_pos
        for i in range(start_pos, min(start_pos + 500, len(code))):
            if code[i] == '(':
                depth += 1
            elif code[i] == ')':
                depth -= 1
                if depth == 0:
                    param_str = code[param_start:i]
                    # 简单分割参数
                    params = [p.split(':')[0].strip().replace('*', '').replace('**', '') 
                             for p in param_str.split(',') if p.strip()]
                    return params
        return []
    
    @classmethod
    def _find_function_calls_in_scope(cls, code: str, start_pos: int) -> List[str]:
        """查找函数体内的函数调用"""
        # 简化：找到下一个顶层 def/class 或文件结尾
        scope_end = len(code)
        depth = 0
        in_string = False
        string_char = None
        
        for i in range(start_pos, min(start_pos + 2000, len(code))):
            char = code[i]
            
            if char in '"\'':
                if not in_string:
                    in_string = True
                    string_char = char
                elif char == string_char:
                    in_string = False
                continue
            
            if in_string:
                continue
                
            if char in '([{':
                depth += 1
            elif char in ')]}':
                depth -= 1
                if depth < 0:
                    scope_end = i
                    break
            elif depth == 0 and (code[i:i+4] == 'def ' or code[i:i+6] == 'class '):
                scope_end = i
                break
        
        scope_code = code[start_pos:scope_end]
        calls = re.findall(r'(\w+)\s*\(', scope_code)
        
        # 过滤掉 Python 关键字和非函数调用
        keywords = {'if', 'else', 'elif', 'for', 'while', 'with', 'return', 'print', 'assert'}
        return [c for c in calls if c not in keywords and not c[0].isupper()]
    
    @classmethod
    def _estimate_complexity(cls, code: str, start_pos: int) -> str:
        """估算函数复杂度"""
        # 简化的圈复杂度估算
        scope_end = min(start_pos + 2000, len(code))
        scope_code = code[start_pos:scope_end]
        
        complexity = 1  # 基础复杂度
        complexity += len(re.findall(r'\bif\b', scope_code))
        complexity += len(re.findall(r'\bfor\b', scope_code))
        complexity += len(re.findall(r'\bwhile\b', scope_code))
        complexity += len(re.findall(r'\belif\b', scope_code))
        complexity += len(re.findall(r'\bexcept\b', scope_code))
        
        if complexity <= 3:
            return "low"
        elif complexity <= 8:
            return "medium"
        else:
            return "high"


class ImageRegionAnalyzer:
    """图像区域分析器（简化版，实际使用时需接入视觉模型）"""
    
    # 图像中的常见元素模式
    VISUAL_ELEMENT_PATTERNS = {
        'figure_caption': r'(?:Figure|Fig\.|图)\s*(\d+)[.:]\s*(.+)',
        'table_reference': r'(?:Table|表)\s*(\d+)[.:]\s*(.+)',
        'equation_reference': r'(?:Equation|Eq\.|公式|式)\s*(\d+)',
        'region_description': r'(?:left|right|top|bottom|above|below|center)\s+(?:panel|part|section)\s*(\w)',
    }
    
    @classmethod
    def analyze_image_reference(cls, context_text: str, image_data: Optional[bytes] = None) -> Dict[str, Any]:
        """
        分析图像引用及其上下文
        
        Returns:
            {
                'figure_number': Optional[int],
                'caption': str,
                'referenced_elements': [str],
                'surrounding_entities': [str],
                'visual_description_hints': [str],
                'likely_content_type': str
            }
        """
        result = {
            'figure_number': None,
            'caption': '',
            'referenced_elements': [],
            'surrounding_entities': [],
            'visual_description_hints': [],
            'likely_content_type': 'unknown'
        }
        
        # 提取图片标题
        caption_match = re.search(cls.VISUAL_ELEMENT_PATTERNS['figure_caption'], context_text, re.IGNORECASE)
        if caption_match:
            result['figure_number'] = int(caption_match.group(1))
            result['caption'] = caption_match.group(2).strip()
        
        # 提取表格引用
        table_refs = re.findall(cls.VISUAL_ELEMENT_PATTERNS['table_reference'], context_text, re.IGNORECASE)
        result['referenced_elements'].extend([f"Table:{ref[0]}" for ref in table_refs])
        
        # 提取公式引用
        eq_refs = re.findall(cls.VISUAL_ELEMENT_PATTERNS['equation_reference'], context_text, re.IGNORECASE)
        result['referenced_elements'].extend([f"Eq:{ref}" for ref in eq_refs])
        
        # 推断图像内容类型
        caption_lower = result['caption'].lower()
        if any(w in caption_lower for w in ['plot', 'graph', 'chart', 'curve', 'trend']):
            result['likely_content_type'] = 'visualization'
        elif any(w in caption_lower for w in ['architecture', 'system', 'network', 'pipeline']):
            result['likely_content_type'] = 'diagram'
        elif any(w in caption_lower for w in ['result', 'output', 'example', 'sample']):
            result['likely_content_type'] = 'example_output'
        elif any(w in caption_lower for w in ['comparison', 'vs', 'versus']):
            result['likely_content_type'] = 'comparison'
        
        # 提取周围的实体名称（大写开头的词）
        surrounding_entities = re.findall(r'\b([A-Z][a-z]+(?:[A-Z][a-z]+)*)(?:\s+(?:system|model|method|algorithm|framework))?', context_text)
        result['surrounding_entities'] = list(set(surrounding_entities))[:10]
        
        return result


def build_multimodal_evidence_fusion(
    question: str,
    evidence_chunks: List[Dict[str, Any]],
    query_graph: Optional[Dict[str, Any]] = None,
    enable_vision: bool = False,
    enable_code_analysis: bool = True,
    enable_table_extraction: bool = True,
    enable_formula_parsing: bool = True,
) -> Dict[str, Any]:
    """
    构建多模态证据融合管道
    
    Args:
        question: 用户问题
        evidence_chunks: 原始证据块列表
        query_graph: 查询图（可选）
        enable_vision: 是否启用图像分析
        enable_code_analysis: 是否启用代码分析
        enable_table_extraction: 是否启用表格提取
        enable_formula_parsing: 是否启用公式解析
    
    Returns:
        {
            'fusion_pipeline': str,
            'multimodal_entities': [MultimodalEntity],
            'cross_modal_relations': [CrossModalRelation],
            'fused_evidence': [FusionEvidence],
            'modality_statistics': {...},
            'enhanced_context': str
        }
    """
    pipeline_result = {
        'fusion_pipeline': 'V136_Multimodal_Knowledge_Fusion_v1',
        'multimodal_entities': [],
        'cross_modal_relations': [],
        'fused_evidence': [],
        'modality_statistics': {
            ModalityType.TEXT.value: 0,
            ModalityType.IMAGE.value: 0,
            ModalityType.TABLE.value: 0,
            ModalityType.FORMULA.value: 0,
            ModalityType.CODE.value: 0,
            ModalityType.CHART.value: 0,
        },
        'enhanced_context': ''
    }
    
    entity_counter = 0
    all_entities: Dict[str, MultimodalEntity] = {}
    
    # 处理每个证据块
    for chunk_idx, chunk in enumerate(evidence_chunks):
        chunk_text = chunk.get('text', '')
        chunk_source = chunk.get('source_id', f'chunk_{chunk_idx}')
        
        # 1. 文本模态（基础）
        text_entity = MultimodalEntity(
            entity_id=f"mm_ent_{entity_counter}",
            name=f"text_chunk_{chunk_idx}",
            modality=ModalityType.TEXT,
            content=chunk_text[:500],
            confidence=1.0,
            extracted_by="base"
        )
        all_entities[text_entity.entity_id] = text_entity
        pipeline_result['modality_statistics'][ModalityType.TEXT.value] += 1
        entity_counter += 1
        
        # 2. 表格检测与提取
        if enable_table_extraction:
            table_sections = _detect_table_sections(chunk_text)
            for table_text in table_sections:
                table_data = TableStructureExtractor.extract(table_text)
                
                table_entity = MultimodalEntity(
                    entity_id=f"mm_ent_{entity_counter}",
                    name=f"table_{chunk_idx}_{len(pipeline_result['multimodal_entities'])}",
                    modality=ModalityType.TABLE,
                    content=json.dumps(table_data, ensure_ascii=False)[:1000],
                    metadata={
                        'row_count': table_data['statistics'].get('row_count', 0),
                        'col_count': table_data['statistics'].get('col_count', 0),
                        'entity_count': len(table_data.get('entities', [])),
                        'relation_count': len(table_data.get('relations', [])),
                        'parsed_data': table_data
                    },
                    confidence=0.85,
                    extracted_by="table_extractor"
                )
                all_entities[table_entity.entity_id] = table_entity
                pipeline_result['modality_statistics'][ModalityType.TABLE.value] += 1
                entity_counter += 1
                
                # 创建文本-表格跨模态关系
                relation = CrossModalRelation(
                    relation_id=f"cm_rel_{len(pipeline_result['cross_modal_relations'])}",
                    source_entity_id=text_entity.entity_id,
                    target_entity_id=table_entity.entity_id,
                    source_modality=ModalityType.TEXT,
                    target_modality=ModalityType.TABLE,
                    relation_type="contains",
                    description=f"Text chunk contains structured table data",
                    confidence=0.9,
                    alignment_score=0.85
                )
                pipeline_result['cross_modal_relations'].append(relation)
        
        # 3. 公式检测与解析
        if enable_formula_parsing:
            formula_sections = _detect_formula_sections(chunk_text)
            for formula_text in formula_sections:
                formula_data = FormulaSemanticParser.parse(formula_text)
                
                formula_entity = MultimodalEntity(
                    entity_id=f"mm_ent_{entity_counter}",
                    name=formula_data.get('variables', ['formula'])[0] if formula_data.get('variables') else f"formula_{entity_counter}",
                    modality=ModalityType.FORMULA,
                    content=formula_text,
                    metadata={
                        'structure': formula_data.get('structure'),
                        'domain': formula_data.get('domain'),
                        'variables': formula_data.get('variables'),
                        'operations_count': len(formula_data.get('operations', [])),
                        'parsed_data': formula_data
                    },
                    confidence=0.8,
                    extracted_by="formula_parser"
                )
                all_entities[formula_entity.entity_id] = formula_entity
                pipeline_result['modality_statistics'][ModalityType.FORMULA.value] += 1
                entity_counter += 1
        
        # 4. 代码检测与分析
        if enable_code_analysis:
            code_sections = _detect_code_sections(chunk_text)
            for code_lang, code_text in code_sections:
                code_data = CodeDependencyAnalyzer.analyze(code_text, language=code_lang)
                
                code_entity = MultimodalEntity(
                    entity_id=f"mm_ent_{entity_counter}",
                    name=code_data.get('functions', [{}])[0].get('name', f"code_{entity_counter}") if code_data.get('functions') else f"code_block_{entity_counter}",
                    modality=ModalityType.CODE,
                    content=code_text[:800],
                    metadata={
                        'language': code_lang,
                        'function_count': len(code_data.get('functions', [])),
                        'class_count': len(code_data.get('classes', [])),
                        'complexity': code_data.get('functions', [{}])[0].get('complexity_estimate', 'unknown') if code_data.get('functions') else 'unknown',
                        'parsed_data': code_data
                    },
                    confidence=0.82,
                    extracted_by="code_analyzer"
                )
                all_entities[code_entity.entity_id] = code_entity
                pipeline_result['modality_statistics'][ModalityType.CODE.value] += 1
                entity_counter += 1
        
        # 5. 图像引用检测
        if enable_vision:
            image_refs = ImageRegionAnalyzer.analyze_image_reference(chunk_text)
            if image_refs.get('figure_number') is not None:
                image_entity = MultimodalEntity(
                    entity_id=f"mm_ent_{entity_counter}",
                    name=f"figure_{image_refs['figure_number']}",
                    modality=ModalityType.IMAGE,
                    content=image_refs.get('caption', ''),
                    metadata=image_refs,
                    confidence=0.7,
                    extracted_by="image_ref_analyzer"
                )
                all_entities[image_entity.entity_id] = image_entity
                pipeline_result['modality_statistics'][ModalityType.IMAGE.value] += 1
                entity_counter += 1
    
    # 收集所有实体
    pipeline_result['multimodal_entities'] = list(all_entities.values())
    
    # 构建融合证据
    fusion_evidences = _build_fusion_evidences(question, all_entities, pipeline_result['cross_modal_relations'])
    pipeline_result['fused_evidence'] = fusion_evidences
    
    # 生成增强上下文
    enhanced_context = _generate_enhanced_context(
        question, 
        pipeline_result['multimodal_entities'],
        pipeline_result['cross_modal_relations']
    )
    pipeline_result['enhanced_context'] = enhanced_context
    
    return pipeline_result


def _detect_table_sections(text: str) -> List[str]:
    """检测文本中的表格区域"""
    tables = []
    lines = text.split('\n')
    current_table = []
    in_table = False
    
    for line in lines:
        if '|' in line and '--' in line:
            in_table = True
            current_table.append(line)
        elif in_table and '|' in line:
            current_table.append(line)
        elif in_table:
            if len(current_table) >= 3:  # 至少有表头、分隔符、一行数据
                tables.append('\n'.join(current_table))
            current_table = []
            in_table = False
    
    # 处理最后一个表格
    if in_table and len(current_table) >= 3:
        tables.append('\n'.join(current_table))
    
    return tables


def _detect_formula_sections(text: str) -> List[str]:
    """检测文本中的数学公式"""
    formulas = []
    
    # LaTeX 公式 $...$ 或 \(...\) 或 \[...\]
    latex_patterns = [
        r'\$(?:\$|[^\$])+?\$',  # $...$ 或 $$...$$
        r'\\\((.+?)\\\)',  # \(...\)
        r'\\\[(.+?)\\\]',  # \[...\]
    ]
    
    for pattern in latex_patterns:
        formulas.extend(re.findall(pattern, text))
    
    # 行内独立公式（包含多个数学符号）
    independent_formula_pattern = r'^[^a-zA-Z\n]*[$]?[a-zA-Z]+\s*[=+\-*/≤≥<>∫∑∏√∇∂][^$\n]{5,100}[$]?$'
    for line in text.split('\n'):
        if re.match(independent_formula_pattern, line.strip()) and line.strip() not in formulas:
            formulas.append(line.strip())
    
    return list(set(formulas))


def _detect_code_sections(text: str) -> List[Tuple[str, str]]:
    """检测文本中的代码块"""
    code_blocks = []
    
    # Markdown 代码块 ```language ... ```
    md_code_pattern = r'```(\w*)\s*\n(.*?)```'
    matches = re.findall(md_code_pattern, text, re.DOTALL)
    code_blocks.extend([(lang or 'python', code) for lang, code in matches])
    
    # 缩进代码块（4个空格或tab）
    indented_lines = []
    current_code = []
    in_indented = False
    
    for line in text.split('\n'):
        if line.startswith('    ') or line.startswith('\t'):
            in_indented = True
            current_code.append(line.lstrip())
        elif in_indented:
            if current_code:
                code_blocks.append(('python', '\n'.join(current_code)))
            current_code = []
            in_indented = False
    
    if in_indented and current_code:
        code_blocks.append(('python', '\n'.join(current_code)))
    
    return code_blocks


def _build_fusion_evidences(
    question: str,
    entities: Dict[str, MultimodalEntity],
    relations: List[CrossModalRelation],
) -> List[FusionEvidence]:
    """构建融合证据"""
    fusion_evidences = []
    question_terms = set(re.findall(r'\w+', question.lower()))
    
    # 按模态分组
    entities_by_modality: Dict[ModalityType, List[MultimodalEntity]] = {}
    for entity in entities.values():
        if entity.modality not in entities_by_modality:
            entities_by_modality[entity.modality] = []
        entities_by_modality[entity.modality].append(entity)
    
    # 为每种主要模态创建融合证据
    primary_modalities = [ModalityType.TEXT, ModalityType.TABLE, ModalityType.CODE, ModalityType.FORMULA]
    
    for modality in primary_modalities:
        modality_entities = entities_by_modality.get(modality, [])
        if not modality_entities:
            continue
        
        # 计算相关性
        relevant_entities = []
        for entity in modality_entities:
            entity_terms = set(re.findall(r'\w+', entity.content.lower()))
            overlap = len(question_terms & entity_terms) / max(len(question_terms), 1)
            if overlap > 0.1 or modality == ModalityType.TEXT:
                relevant_entities.append((entity, overlap))
        
        # 排序并选择 top-k
        relevant_entities.sort(key=lambda x: x[1], reverse=True)
        top_entities = relevant_entities[:3]
        
        if top_entities:
            # 查找跨模态链接
            cross_modal_links = []
            for entity, _ in top_entities:
                for rel in relations:
                    if rel.source_entity_id == entity.entity_id or rel.target_entity_id == entity.entity_id:
                        cross_modal_links.append(rel.relation_id)
            
            # 计算完整度分数
            linked_modalities = set()
            for rel in relations:
                for entity, _ in top_entities:
                    if rel.source_entity_id == entity.entity_id:
                        linked_modalities.add(rel.target_modality)
                    elif rel.target_entity_id == entity.entity_id:
                        linked_modalities.add(rel.source_modality)
            
            completeness = min(1.0, len(linked_modalities) / 3)
            
            evidence = FusionEvidence(
                evidence_id=f"fuse_{modality.value}_{len(fusion_evidences)}",
                primary_modality=modality,
                content_summary=f"Fused evidence from {len(top_entities)} {modality.value} sources",
                supporting_entities=[e.entity_id for e, _ in top_entities],
                cross_modal_links=cross_modal_links,
                relevance_score=max(s for _, s in top_entities) if top_entities else 0,
                completeness_score=completeness
            )
            fusion_evidences.append(evidence)
    
    return fusion_evidences


def _generate_enhanced_context(
    question: str,
    entities: List[MultimodalEntity],
    relations: List[CrossModalRelation],
) -> str:
    """生成增强的多模态上下文"""
    sections = []
    
    # 问题分析
    sections.append(f"# Query: {question}")
    sections.append("")
    
    # 多模态证据摘要
    modality_counts = {}
    for entity in entities:
        modality = entity.modality.value
        modality_counts[modality] = modality_counts.get(modality, 0) + 1
    
    if modality_counts:
        sections.append("## Multimodal Evidence Summary:")
        for modality, count in sorted(modality_counts.items(), key=lambda x: -x[1]):
            sections.append(f"- **{modality.capitalize()}**: {count} items")
        sections.append("")
    
    # 关键表格数据
    tables = [e for e in entities if e.modality == ModalityType.TABLE][:3]
    if tables:
        sections.append("## Structured Data from Tables:")
        for table in tables:
            parsed = table.metadata.get('parsed_data', {})
            if parsed.get('relations'):
                key_relations = parsed['relations'][:5]
                for rel in key_relations:
                    sections.append(f"- {rel.get('subject')} → {rel.get('predicate')} → {rel.get('object')}")
        sections.append("")
    
    # 关键公式
    formulas = [e for e in entities if e.modality == ModalityType.FORMULA][:3]
    if formulas:
        sections.append("## Mathematical Formulas:")
        for formula in formulas:
            parsed = formula.metadata.get('parsed_data', {})
            desc = parsed.get('natural_description', formula.content)
            sections.append(f"- `{formula.content[:80]}...`: {desc}")
        sections.append("")
    
    # 代码结构
    codes = [e for e in entities if e.modality == ModalityType.CODE][:2]
    if codes:
        sections.append("## Code Structures:")
        for code in codes:
            parsed = code.metadata.get('parsed_data', {})
            funcs = [f['name'] for f in parsed.get('functions', [])]
            classes = [c['name'] for c in parsed.get('classes', [])]
            if funcs or classes:
                sections.append(f"- Functions: {', '.join(funcs)}; Classes: {', '.join(classes)}")
        sections.append("")
    
    # 跨模态关系
    if relations:
        sections.append("## Cross-Modal Relationships:")
        for rel in relations[:5]:
            sections.append(f"- [{rel.source_modality.value}] → [{rel.relation_type}] → [{rel.target_modality.value}]")
        sections.append("")
    
    return '\n'.join(sections)


# 导出别名以兼容 agent_controller
run_multimodal_knowledge_fusion = build_multimodal_evidence_fusion

__all__ = [
    'MultimodalEntity',
    'CrossModalRelation',
    'FusionEvidence',
    'ModalityType',
    'TableStructureExtractor',
    'FormulaSemanticParser',
    'CodeDependencyAnalyzer',
    'ImageRegionAnalyzer',
    'build_multimodal_evidence_fusion',
    'run_multimodal_knowledge_fusion',
]
