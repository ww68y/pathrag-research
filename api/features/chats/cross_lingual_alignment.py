"""
V138 - Cross-Lingual Alignment Layer (多语言对齐层)
==================================================

支持中英双语实体的深度语义对齐，包括：
- 跨语言实体消歧
- 翻译等价性验证
- 文化背景感知
- 低资源语言适配

核心能力：
- 中英文实体识别与标准化
- 翻译候选生成与排序
- 跨语言语义相似度计算
- 代码/术语混合文本处理
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict


class Language(Enum):
    """支持的语言"""
    CHINESE = "zh"
    ENGLISH = "en"
    JAPANESE = "ja"
    KOREAN = "ko"
    FRENCH = "fr"
    GERMAN = "de"
    SPANISH = "es"
    UNKNOWN = "unknown"


@dataclass
class CrossLingualEntity:
    """跨语言实体"""
    entity_id: str
    original_text: str
    language: Language
    normalized_text: str  # 标准化后的文本（去除变体等）
    translations: Dict[Language, List[str]] = field(default_factory=dict)  # 各语言的翻译候选
    canonical_form: Optional[str] = None  # 规范形式（英文或主要形式）
    entity_type: str = "general"  # general, person, organization, location, technical, acronym
    confidence: float = 0.0
    source_context: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'entity_id': self.entity_id,
            'original_text': self.original_text,
            'language': self.language.value,
            'normalized_text': self.normalized_text,
            'translations': {lang.value: trans for lang, trans in self.translations.items()},
            'canonical_form': self.canonical_form,
            'entity_type': self.entity_type,
            'confidence': self.confidence
        }


@dataclass
class AlignmentScore:
    """对齐分数"""
    source_entity_id: str
    target_entity_id: str
    score: float  # 0-1
    alignment_type: str  # exact_translation, semantic_equivalent, partial_match, context_inferred
    evidence: str = ""  # 对齐证据


@dataclass
class TranslationCandidate:
    """翻译候选"""
    text: str
    language: Language
    source_entity_id: str
    confidence: float
    source: str = "dictionary"  # dictionary, embedding, rule_based, context
    is_verified: bool = False


class ChineseTextProcessor:
    """中文文本处理器"""
    
    # 常见中文变体映射
    VARIANT_MAP = {
        # 简体-繁体常见映射
        '计算机': ['電腦', '計算機'],
        '人工智能': ['人工智慧'],
        '机器学习': ['機器學習'],
        '深度学习': ['深度學習'],
        '神经网络': ['神經網絡'],
        '数据库': ['數據庫', '資料庫'],
        '算法': ['演算法'],
        '软件': ['軟體', '軟件'],
        '硬件': ['硬體'],
        '网络': ['網路', '網絡'],
        '数据': ['數據', '資料'],
        '信息': ['資訊', '信息'],
        '程序': ['程式', '程序'],
        '系统': ['系統', '系統'],
        '文件': ['檔案', '文件'],
        '用户': ['使用者', '用戶'],
        '服务器': ['伺服器'],
        '客户端': ['客戶端'],
        '应用': ['應用', '應用程式'],
        '接口': ['介面', '接口'],
        '框架': ['框架'],
        '库': ['庫', '函式庫'],
        '函数': ['函數', '函式'],
        '变量': ['變數', '變量'],
        '对象': ['物件', '對象'],
        '类別': ['類別', '類'],
        '实例': ['實例', '實例'],
        '异常': ['異常', '例外'],
        '缓存': ['快取', '緩存'],
        '线程': ['線程'],
        '进程': ['進程', '行程'],
        '队列': ['佇列'],
        '堆栈': ['堆疊'],
        '指针': ['指標', '指針'],
        '数组': ['陣列', '陣列'],
        '字符串': ['字串', '字符串'],
        '布尔': ['布林', '布爾'],
        '字符': ['字元', '字符'],
        '浮点': ['浮點數'],
        '整数': ['整數', '整數'],
        '空值': ['空值', 'Null'],
    }
    
    # 中文停用词（用于实体识别时过滤）
    STOP_WORDS = {
        '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都',
        '一', '一个', '上', '也', '很', '到', '说', '要', '去', '你', '会',
        '着', '没有', '看', '好', '自己', '这', '他', '她', '它', '们',
        '那', '什么', '被', '从', '把', '与', '而', '但', '以', '及',
        '可以', '对', '能', '为', '于', '中', '如', '则', '或', '等'
    }
    
    @classmethod
    def detect_language(cls, text: str) -> Tuple[Language, float]:
        """
        检测文本是否为中文
        
        Returns:
            (语言, 中文比例)
        """
        if not text:
            return (Language.UNKNOWN, 0.0)
        
        total_chars = len(text)
        chinese_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
        
        ratio = chinese_chars / total_chars if total_chars > 0 else 0
        
        if ratio > 0.3:
            return (Language.CHINESE, round(ratio, 3))
        else:
            return (Language.UNKNOWN, round(ratio, 3))
    
    @classmethod
    def normalize(cls, text: str) -> str:
        """
        标准化中文文本（繁转简、全角转半角）
        """
        # 全角转半角
        text = cls._fullwidth_to_halfwidth(text)
        
        # 繁体转简体（简化版，实际应使用 opencc 或类似库）
        text = cls._traditional_to_simplified(text)
        
        # 统一空白
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    @classmethod
    def _fullwidth_to_halfwidth(cls, text: str) -> str:
        """全角字符转半角"""
        result = []
        for char in text:
            code = ord(char)
            if 0xFF01 <= code <= 0xFF5E:  # 全角字母/数字/符号
                result.append(chr(code - 0xFEE0))
            elif code == 0x3000:  # 全角空格
                result.append(' ')
            elif 0xFF10 <= code <= 0xFF19:  # 全角数字
                result.append(chr(code - 0xFEE0))
            elif 0xFF21 <= code <= 0xFF3A:  # 全角大写字母
                result.append(chr(code - 0xFEE0))
            elif 0xFF41 <= code <= 0xFF5A:  # 全角小写字母
                result.append(chr(code - 0xFEE0))
            else:
                result.append(char)
        return ''.join(result)
    
    @classmethod
    def _traditional_to_simplified(cls, text: str) -> str:
        """繁体转简体（简化版）"""
        # 常见繁简映射
        trad_simp_map = {
            '電腦': '电脑', '資訊': '信息', '資料': '数据', '網路': '网络',
            '軟體': '软件', '硬體': '硬件', '程式': '程序', '檔案': '文件',
            '佇列': '队列', '堆疊': '堆栈', '指標': '指针', '陣列': '数组',
            '字串': '字符串', '布林': '布尔', '異常': '异常', '物件': '对象',
            '類別': '类别', '實例': '实例', '變數': '变量', '對象': '对象',
            '函數': '函数', '函式': '函数', '介面': '界面', '使用者': '用户',
            '用戶': '用户', '伺服器': '服务器', '客戶端': '客户端',
            '學習': '学习', '機器': '机器', '智慧': '智能', '網絡': '网络',
            '識別': '识别', '標準': '标准', '編碼': '编码', '解碼': '解码',
            '連結': '连接', '通訊': '通讯', '傳輸': '传输', '處理': '处理',
            '運算': '运算', '優化': '优化', '配置': '配置', '設定': '设置',
            '選項': '选项', '參數': '参数', '變更': '变更', '查詢': '查询',
            '顯示': '显示', '隱藏': '隐藏', '啟動': '启动', '關閉': '关闭',
            '開啟': '开启', '點擊': '点击', '雙擊': '双击', '拖曳': '拖拽',
            '載入': '加载', '儲存': '存储', '備份': '备份', '還原': '还原',
            '匯入': '导入', '匯出': '導出', '列印': '打印', '掃描': '扫描',
            '圖片': '图片', '影像': '图像', '視頻': '视频', '音頻': '音频',
            '網頁': '网页', '網站': '网站', '瀏覽器': '浏览器', '搜尋': '搜索',
        }
        
        for trad, simp in trad_simp_map.items():
            text = text.replace(trad, simp)
        
        return text
    
    @classmethod
    def extract_entities(cls, text: str) -> List[Tuple[str, int, int]]:
        """
        从中文文本中提取实体候选
        
        Returns:
            [(实体文本, 开始位置, 结束位置), ...]
        """
        entities = []
        
        # 模式1：中文名词短语（2-6个连续汉字）
        cn_noun_pattern = r'[\u4e00-\u9fff]{2,6}'
        for match in re.finditer(cn_noun_pattern, text):
            word = match.group()
            if word not in cls.STOP_WORDS and not cls._is_common_word(word):
                entities.append((word, match.start(), match.end()))
        
        # 模式2：中英混合技术术语
        mixed_pattern = r'[\u4e00-\u9fff]+[A-Za-z0-9_.]+|[A-Za-z][A-Za-z0-9_]*[\u4e00-\u9fff]+'
        for match in re.finditer(mixed_pattern, text):
            entities.append((match.group(), match.start(), match.end()))
        
        # 模式3：带括号的英文缩写
        acronym_pattern = r'[\u4e00-\u9fff]+\(([A-Za-z]+)\)|([A-Za-z]+)[\u4e00-\u9fff]*'
        for match in re.finditer(acronym_pattern, text):
            entities.append((match.group().strip('()'), match.start(), match.end()))
        
        # 去重并按位置排序
        seen = set()
        unique_entities = []
        for entity in entities:
            if entity[0] not in seen:
                seen.add(entity[0])
                unique_entities.append(entity)
        
        return unique_entities
    
    @classmethod
    def _is_common_word(cls, word: str) -> bool:
        """判断是否为常见词"""
        common_words = {
            '这个', '那个', '什么', '怎么', '如何', '为什么', '哪里', '哪个',
            '可以', '可能', '应该', '需要', '必须', '能够', '已经', '正在',
            '进行', '使用', '通过', '根据', '关于', '对于', '由于', '因此',
            '但是', '然而', '而且', '或者', '如果', '虽然', '即使', '除非',
            '一个', '一些', '每个', '所有', '任何', '其他', '某些',
            '第一', '第二', '第三', '以上', '以下', '之中', '之间',
        }
        return word in common_words or len(word) < 2


class EnglishTextProcessor:
    """英文文本处理器"""
    
    STOP_WORDS = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
        'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her',
        'us', 'them', 'my', 'your', 'his', 'its', 'our', 'their',
        'this', 'that', 'these', 'those', 'what', 'which', 'who', 'whom',
        'and', 'or', 'but', 'if', 'then', 'else', 'when', 'where', 'why',
        'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most',
        'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
        'same', 'so', 'than', 'too', 'very', 'just', 'also', 'now',
        'here', 'there', 'to', 'of', 'in', 'for', 'on', 'with', 'at',
        'by', 'from', 'up', 'about', 'into', 'over', 'after', 'before',
        'between', 'through', 'during', 'without', 'within', 'along',
        'across', 'behind', 'below', 'beneath', 'under', 'above'
    }
    
    @classmethod
    def detect_language(cls, text: str) -> Tuple[Language, float]:
        """检测是否为英文"""
        if not text:
            return (Language.UNKNOWN, 0.0)
        
        alpha_chars = sum(1 for c in text if c.isalpha())
        english_ratio = alpha_chars / len(text) if len(text) > 0 else 0
        
        if english_ratio > 0.5:
            return (Language.ENGLISH, round(english_ratio, 3))
        else:
            return (Language.UNKNOWN, round(english_ratio, 3))
    
    @classmethod
    def normalize(cls, text: str) -> str:
        """标准化英文文本"""
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        return text
    
    @classmethod
    def extract_entities(cls, text: str) -> List[Tuple[str, int, int]]:
        """从英文文本中提取实体候选"""
        entities = []
        
        # 模式1：首字母大写的名词短语
        proper_noun_pattern = r'\b[A-Z][a-z]+(?:\s+[A-Z]?[a-z]+)*\b'
        for match in re.finditer(proper_noun_pattern, text):
            word = match.group()
            if word not in cls.STOP_WORDS and len(word) > 1:
                entities.append((word, match.start(), match.end()))
        
        # 模式2：技术术语（通常包含连字符或驼峰命名）
        tech_pattern = r'\b[a-z]+(?:-[a-z]+)+\b|\b[a-z]+(?:[A-Z][a-z]+)+\b'
        for match in re.finditer(tech_pattern, text):
            entities.append((match.group(), match.start(), match.end()))
        
        # 模式3：缩写词
        acronym_pattern = r'\b[A-Z]{2,}\b'
        for match in re.finditer(acronym_pattern, text):
            entities.append((match.group(), match.start(), match.end()))
        
        # 去重
        seen = set()
        unique_entities = []
        for entity in entities:
            if entity[0].lower() not in seen:
                seen.add(entity[0].lower())
                unique_entities.append(entity)
        
        return unique_entities


class BilingualDictionary:
    """双语词典（简化版）"""
    
    # 技术术语词典
    TECH_DICTIONARY: Dict[str, List[str]] = {
        # AI/ML
        'artificial intelligence': ['人工智能', 'AI'],
        'machine learning': ['机器学习', 'ML'],
        'deep learning': ['深度学习', 'DL'],
        'neural network': ['神经网络', 'NN'],
        'convolutional neural network': ['卷积神经网络', 'CNN'],
        'recurrent neural network': ['循环神经网络', 'RNN'],
        'transformer': ['Transformer', '变换器'],
        'attention mechanism': ['注意力机制'],
        'large language model': ['大语言模型', 'LLM'],
        'natural language processing': ['自然语言处理', 'NLP'],
        'computer vision': ['计算机视觉', 'CV'],
        'reinforcement learning': ['强化学习', 'RL'],
        'supervised learning': ['监督学习'],
        'unsupervised learning': ['无监督学习'],
        'transfer learning': ['迁移学习'],
        'federated learning': ['联邦学习'],
        'generative model': ['生成模型'],
        'diffusion model': ['扩散模型'],
        'embedding': ['嵌入', '向量表示'],
        'vector database': ['向量数据库'],
        'knowledge graph': ['知识图谱'],
        'retrieval augmented generation': ['检索增强生成', 'RAG'],
        
        # 编程
        'algorithm': ['算法'],
        'data structure': ['数据结构'],
        'function': ['函数'],
        'variable': ['变量'],
        'object-oriented': ['面向对象'],
        'class': ['类'],
        'instance': ['实例'],
        'inheritance': ['继承'],
        'polymorphism': ['多态'],
        'encapsulation': ['封装'],
        'abstraction': ['抽象'],
        'interface': ['接口'],
        'framework': ['框架'],
        'library': ['库'],
        'api': ['应用程序接口', 'API'],
        'database': ['数据库'],
        'cache': ['缓存'],
        'thread': ['线程'],
        'process': ['进程'],
        'concurrency': ['并发'],
        'parallelism': ['并行'],
        'asynchronous': ['异步'],
        'synchronous': ['同步'],
        'callback': ['回调'],
        'promise': ['Promise'],
        'exception': ['异常'],
        'error handling': ['错误处理'],
        'debugging': ['调试'],
        'testing': ['测试'],
        'deployment': ['部署'],
        
        # 系统
        'server': ['服务器'],
        'client': ['客户端'],
        'frontend': ['前端'],
        'backend': ['后端'],
        'microservice': ['微服务'],
        'container': ['容器'],
        'virtual machine': ['虚拟机'],
        'cloud computing': ['云计算'],
        'distributed system': ['分布式系统'],
        'load balancing': ['负载均衡'],
        'scalability': ['可扩展性'],
        'availability': ['可用性'],
        'reliability': ['可靠性'],
        'security': ['安全性'],
        'authentication': ['认证'],
        'authorization': ['授权'],
        'encryption': ['加密'],
        'firewall': ['防火墙'],
    }
    
    @classmethod
    def translate(cls, term: str, source_lang: Language, target_lang: Language) -> List[TranslationCandidate]:
        """
        查找翻译候选
        
        Returns:
            翻译候选列表
        """
        candidates = []
        normalized_term = term.lower().strip()
        
        if source_lang == Language.ENGLISH and target_lang == Language.CHINESE:
            # 英译中
            if normalized_term in cls.TECH_DICTIONARY:
                translations = cls.TECH_DICTIONARY[normalized_term]
                for trans in translations:
                    candidates.append(TranslationCandidate(
                        text=trans,
                        language=target_lang,
                        source_entity_id="",
                        confidence=0.95,
                        source="dictionary",
                        is_verified=True
                    ))
            
            # 尝试反向查找（中文->英文词典的逆映射）
            for en_term, zh_terms in cls.TECH_DICTIONARY.items():
                if normalized_term in [t.lower() for t in zh_terms]:
                    candidates.append(TranslationCandidate(
                        text=en_term,
                        language=target_lang,
                        source_entity_id="",
                        confidence=0.85,
                        source="reverse_dictionary"
                    ))
                    
        elif source_lang == Language.CHINESE and target_lang == Language.ENGLISH:
            # 中译英
            simplified = ChineseTextProcessor.normalize(term)
            
            for en_term, zh_terms in cls.TECH_DICTIONARY.items():
                if simplified in zh_terms or term in zh_terms:
                    candidates.append(TranslationCandidate(
                        text=en_term,
                        language=target_lang,
                        source_entity_id="",
                        confidence=0.95,
                        source="dictionary",
                        is_verified=True
                    ))
        
        return candidates


class CrossLingualAligner:
    """跨语言对齐器"""
    
    @classmethod
    def align_entities(
        cls,
        source_entities: List[CrossLingualEntity],
        target_entities: List[CrossLingualEntity],
    ) -> List[AlignmentScore]:
        """
        对齐两个语言列表中的实体
        
        Returns:
            对齐分数列表
        """
        alignments = []
        
        for src_ent in source_entities:
            best_match = None
            best_score = 0.0
            
            for tgt_ent in target_entities:
                score = cls._compute_alignment_score(src_ent, tgt_ent)
                
                if score > best_score and score > 0.3:  # 阈值
                    best_score = score
                    best_match = tgt_ent
            
            if best_match:
                align_type = cls._determine_alignment_type(best_score)
                alignments.append(AlignmentScore(
                    source_entity_id=src_ent.entity_id,
                    target_entity_id=best_match.entity_id,
                    score=round(best_score, 3),
                    alignment_type=align_type,
                    evidence=f"{src_ent.original_text} <-> {best_match.original_text}"
                ))
        
        return alignments
    
    @classmethod
    def _compute_alignment_score(cls, ent1: CrossLingualEntity, ent2: CrossLingualEntity) -> float:
        """计算两个实体的对齐分数"""
        scores = []
        
        # 1. 字符串相似度（如果一个是另一个的翻译）
        norm1 = ent1.normalized_text.lower()
        norm2 = ent2.normalized_text.lower()
        
        # 完全匹配
        if norm1 == norm2:
            scores.append(1.0)
        # 包含关系
        elif norm1 in norm2 or norm2 in norm1:
            scores.append(0.8)
        # 编辑距离相似度
        else:
            edit_sim = 1 - cls._edit_distance(norm1, norm2) / max(len(norm1), len(norm2), 1)
            scores.append(edit_sim * 0.5)
        
        # 2. 翻译匹配
        translation_match = 0.0
        for lang, translations in ent1.translations.items():
            for trans in translations:
                if trans.lower() in norm2 or norm2 in trans.lower():
                    translation_match = max(translation_match, 0.9)
        for lang, translations in ent2.translations.items():
            for trans in translations:
                if trans.lower() in norm1 or norm1 in trans.lower():
                    translation_match = max(translation_match, 0.9)
        scores.append(translation_match)
        
        # 3. 类型一致性
        if ent1.entity_type == ent2.entity_type:
            scores.append(0.1)
        
        return sum(scores) / len(scores) if scores else 0.0
    
    @staticmethod
    def _edit_distance(s1: str, s2: str) -> int:
        """计算编辑距离"""
        if len(s1) < len(s2):
            return CrossLingualAligner._edit_distance(s2, s1)
        if len(s2) == 0:
            return len(s1)
        
        previous_row = range(len(s2) + 1)
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    @classmethod
    def _determine_alignment_type(cls, score: float) -> str:
        """确定对齐类型"""
        if score >= 0.9:
            return "exact_translation"
        elif score >= 0.7:
            return "semantic_equivalent"
        elif score >= 0.5:
            return "partial_match"
        else:
            return "context_inferred"


def build_cross_lingual_alignment(
    question: str,
    evidence_chunks: List[Dict[str, Any]],
    primary_language: Language = Language.UNKNOWN,
    enable_entity_extraction: bool = True,
    enable_translation: bool = True,
    enable_alignment: bool = True,
) -> Dict[str, Any]:
    """
    构建跨语言对齐管道
    
    Args:
        question: 用户问题
        evidence_chunks: 证据块列表
        primary_language: 主要语言（自动检测）
        enable_entity_extraction: 是否启用实体提取
        enable_translation: 是否启用翻译
        enable_alignment: 是否启用对齐
    
    Returns:
        {
            'pipeline': str,
            'detected_languages': {...},
            'entities_by_language': {...},
            'alignments': [...],
            'enhanced_query': {...},
            'statistics': {...}
        }
    """
    result = {
        'pipeline': 'V138_Cross_Lingual_Alignment_v1',
        'detected_languages': {
            'question': {'language': Language.UNKNOWN.value, 'confidence': 0.0},
            'evidence': {}
        },
        'entities_by_language': {
            Language.CHINESE.value: [],
            Language.ENGLISH.value: [],
            Language.UNKNOWN.value: []
        },
        'alignments': [],
        'enhanced_query': {},
        'statistics': {
            'total_entities': 0,
            'chinese_entities': 0,
            'english_entities': 0,
            'aligned_pairs': 0,
            'translation_candidates': 0
        }
    }
    
    # 1. 检测问题语言
    q_lang, q_conf = ChineseTextProcessor.detect_language(question)
    if q_lang == Language.UNKNOWN:
        q_lang, q_conf = EnglishTextProcessor.detect_language(question)
    result['detected_languages']['question'] = {'language': q_lang.value, 'confidence': q_conf}
    
    if primary_language == Language.UNKNOWN:
        primary_language = q_lang if q_conf > 0.3 else Language.ENGLISH
    
    # 2. 处理证据块
    all_text = question + "\n" + "\n".join(c.get('text', '') for c in evidence_chunks)
    
    chinese_entities = []
    english_entities = []
    
    for chunk_idx, chunk in enumerate(evidence_chunks):
        chunk_text = chunk.get('text', '')
        chunk_id = chunk.get('source_id', f'chunk_{chunk_idx}')
        
        # 检测语言
        cn_lang, cn_ratio = ChineseTextProcessor.detect_language(chunk_text)
        en_lang, en_ratio = EnglishTextProcessor.detect_language(chunk_text)
        
        dominant_lang = cn_lang if cn_ratio > en_ratio else en_lang
        result['detected_languages']['evidence'][chunk_id] = {
            'dominant': dominant_lang.value,
            'chinese_ratio': cn_ratio,
            'english_ratio': en_ratio
        }
        
        # 提取实体
        if enable_entity_extraction:
            # 中文实体
            cn_entities = ChineseTextProcessor.extract_entities(chunk_text)
            for ent_text, start, end in cn_entities:
                entity = CrossLingualEntity(
                    entity_id=f"cn_ent_{len(chinese_entities)}",
                    original_text=ent_text,
                    language=Language.CHINESE,
                    normalized_text=ChineseTextProcessor.normalize(ent_text),
                    entity_type=_infer_entity_type(ent_text, Language.CHINESE),
                    source_context=chunk_text[max(0, start-20):end+20],
                    confidence=min(0.9, 0.6 + len(ent_text) * 0.05)
                )
                
                # 查找翻译
                if enable_translation:
                    translations = BilingualDictionary.translate(
                        ent_text, Language.CHINESE, Language.ENGLISH
                    )
                    entity.translations[Language.ENGLISH] = [t.text for t in translations]
                    result['statistics']['translation_candidates'] += len(translations)
                
                chinese_entities.append(entity)
            
            # 英文实体
            en_entities = EnglishTextProcessor.extract_entities(chunk_text)
            for ent_text, start, end in en_entities:
                entity = CrossLingualEntity(
                    entity_id=f"en_ent_{len(english_entities)}",
                    original_text=ent_text,
                    language=Language.ENGLISH,
                    normalized_text=EnglishTextProcessor.normalize(ent_text),
                    entity_type=_infer_entity_type(ent_text, Language.ENGLISH),
                    source_context=chunk_text[max(0, start-20):end+20],
                    confidence=min(0.9, 0.6 + len(ent_text) * 0.05)
                )
                
                # 查找翻译
                if enable_translation:
                    translations = BilingualDictionary.translate(
                        ent_text, Language.ENGLISH, Language.CHINESE
                    )
                    entity.translations[Language.CHINESE] = [t.text for t in translations]
                    result['statistics']['translation_candidates'] += len(translations)
                
                english_entities.append(entity)
    
    result['entities_by_language'][Language.CHINESE.value] = [e.to_dict() for e in chinese_entities]
    result['entities_by_language'][Language.ENGLISH.value] = [e.to_dict() for e in english_entities]
    
    result['statistics']['total_entities'] = len(chinese_entities) + len(english_entities)
    result['statistics']['chinese_entities'] = len(chinese_entities)
    result['statistics']['english_entities'] = len(english_entities)
    
    # 3. 执行对齐
    if enable_alignment and chinese_entities and english_entities:
        aligner = CrossLingualAligner()
        alignments = aligner.align_entities(chinese_entities, english_entities)
        result['alignments'] = [{
            'source': a.source_entity_id,
            'target': a.target_entity_id,
            'score': a.score,
            'type': a.alignment_type,
            'evidence': a.evidence
        } for a in alignments]
        result['statistics']['aligned_pairs'] = len(alignments)
    
    # 4. 生成增强查询
    enhanced_queries = _generate_enhanced_queries(
        question, primary_language, chinese_entities, english_entities, result['alignments']
    )
    result['enhanced_query'] = enhanced_queries
    
    return result


def _infer_entity_type(text: str, language: Language) -> str:
    """推断实体类型"""
    text_lower = text.lower()
    
    # 组织机构模式
    org_patterns = [
        r'(?:company|inc|corp|ltd|group|organization|university|institute|lab|laboratory)',
        r'(?:公司|集团|组织|大学|研究院|实验室|研究所)',
        r'(?:Inc\.?|Corp\.?|Ltd\.?|Co\.?)$'
    ]
    for pattern in org_patterns:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return 'organization'
    
    # 人物模式
    person_patterns = [
        r'(?:mr|mrs|ms|dr|professor|ceo|cto|cfo)',
        r'(?:先生|女士|博士|教授|经理|总裁|CEO|CTO)'
    ]
    for pattern in person_patterns:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return 'person'
    
    # 地点模式
    location_patterns = [
        r'(?:city|country|state|province|region|river|mountain|ocean)',
        r'(?:市|省|州|国|地区|河|山|洋|湖)'
    ]
    for pattern in location_patterns:
        if re.search(pattern, text_lower, re.IGNORECASE):
            return 'location'
    
    # 技术术语模式
    tech_indicators = [
        'algorithm', 'model', 'framework', 'library', 'api', 'function',
        'class', 'method', 'system', 'network', 'protocol', 'format',
        '算法', '模型', '框架', '库', '函数', '类', '方法', '系统', '网络', '协议'
    ]
    if any(ind in text_lower for ind in tech_indicators):
        return 'technical'
    
    # 缩写检测
    if re.match(r'^[A-Z]{2,}$', text) or re.match(r'^[A-Za-z]+[0-9]+$', text):
        return 'acronym'
    
    return 'general'


def _generate_enhanced_queries(
    question: str,
    primary_language: Language,
    chinese_entities: List[CrossLingualEntity],
    english_entities: List[CrossLingualEntity],
    alignments: List[Dict]
) -> Dict[str, Any]:
    """生成增强查询"""
    enhanced = {
        'original': question,
        'translated_versions': [],
        'expanded_with_entities': [],
        'key_terms': {
            'chinese': [],
            'english': []
        }
    }
    
    # 收集关键术语
    enhanced['key_terms']['chinese'] = list(set(e.original_text for e in chinese_entities[:15]))
    enhanced['key_terms']['english'] = list(set(e.original_text for e in english_entities[:15]))
    
    # 如果问题主要是中文，添加英文关键词扩展
    if primary_language == Language.CHINESE:
        english_expansions = []
        for ce in chinese_entities[:10]:
            for trans in ce.translations.get(Language.ENGLISH, []):
                if trans not in question:
                    english_expansions.append(trans)
        if english_expansions:
            enhanced['expanded_with_entities'] = english_expansions[:8]
    
    elif primary_language == Language.ENGLISH:
        chinese_expansions = []
        for ee in english_entities[:10]:
            for trans in ee.translations.get(Language.CHINESE, []):
                if trans not in question:
                    chinese_expansions.append(trans)
        if chinese_expansions:
            enhanced['expanded_with_entities'] = chinese_expansions[:8]
    
    return enhanced


# 导出别名
run_cross_lingual_alignment = build_cross_lingual_alignment

__all__ = [
    'CrossLingualEntity',
    'AlignmentScore',
    'TranslationCandidate',
    'Language',
    'ChineseTextProcessor',
    'EnglishTextProcessor',
    'BilingualDictionary',
    'CrossLingualAligner',
    'build_cross_lingual_alignment',
    'run_cross_lingual_alignment',
]
