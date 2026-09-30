"""
V115: Multilingual & Cross-lingual Retrieval (MCR)
多语言与跨语言检索 - 支持多语言查询和跨语言知识迁移

创新点：
1. 语言检测：自动检测查询语言
2. 跨语言嵌入：统一的多语言语义空间
3. 机器翻译集成：查询和文档的自动翻译
4. 文化适配：考虑文化差异的答案生成
"""

import re
import json
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import numpy as np


class LanguageCode(Enum):
    """语言代码"""
    EN = "en"
    ZH = "zh"
    JA = "ja"
    KO = "ko"
    FR = "fr"
    DE = "de"
    ES = "es"
    RU = "ru"
    AR = "ar"
    PT = "pt"
    IT = "it"
    NL = "nl"
    UNKNOWN = "unknown"


@dataclass
class LanguageProfile:
    """语言配置文件"""
    code: LanguageCode
    name: str
    name_native: str
    script: str
    direction: str  # 'ltr' or 'rtl'
    common_encodings: List[str] = field(default_factory=list)


@dataclass
class TranslatedContent:
    """翻译内容"""
    original_text: str
    original_language: LanguageCode
    translated_text: str
    target_language: LanguageCode
    translation_confidence: float
    back_translation: Optional[str] = None
    semantic_preservation_score: float = 0.0


class LanguageDetector:
    """语言检测器"""
    
    # 语言特征模式
    LANGUAGE_PATTERNS = {
        LanguageCode.ZH: {
            'script': r'[\u4e00-\u9fff]+',
            'common_words': ['的', '是', '在', '和', '了', '有', '我', '不', '人', '这']
        },
        LanguageCode.JA: {
            'script': r'[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9faf]+',
            'common_words': ['の', 'は', 'を', 'が', 'に', 'で', 'と', 'た', 'て', 'です']
        },
        LanguageCode.KO: {
            'script': r'[\uac00-\ud7af]+',
            'common_words': ['의', '이', '가', '은', '는', '을', '를', '에', '로', '과']
        },
        LanguageCode.AR: {
            'script': r'[\u0600-\u06ff\u0750-\u077f]+',
            'direction': 'rtl'
        },
        LanguageCode.RU: {
            'script': r'[\u0400-\u04ff]+',
            'common_words': ['в', 'и', 'не', 'на', 'я', 'быть', 'он', 'с', 'что', 'а']
        },
        LanguageCode.EN: {
            'script': r'[a-zA-Z]+',
            'common_words': ['the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i']
        }
    }
    
    def __init__(self):
        self.language_profiles = self._init_language_profiles()
    
    def _init_language_profiles(self) -> Dict[LanguageCode, LanguageProfile]:
        """初始化语言配置文件"""
        profiles = {
            LanguageCode.EN: LanguageProfile(
                code=LanguageCode.EN,
                name="English",
                name_native="English",
                script="Latin",
                direction="ltr"
            ),
            LanguageCode.ZH: LanguageProfile(
                code=LanguageCode.ZH,
                name="Chinese",
                name_native="中文",
                script="Han",
                direction="ltr"
            ),
            LanguageCode.JA: LanguageProfile(
                code=LanguageCode.JA,
                name="Japanese",
                name_native="日本語",
                script="Mixed",
                direction="ltr"
            ),
            LanguageCode.KO: LanguageProfile(
                code=LanguageCode.KO,
                name="Korean",
                name_native="한국어",
                script="Hangul",
                direction="ltr"
            ),
            LanguageCode.FR: LanguageProfile(
                code=LanguageCode.FR,
                name="French",
                name_native="Français",
                script="Latin",
                direction="ltr"
            ),
            LanguageCode.DE: LanguageProfile(
                code=LanguageCode.DE,
                name="German",
                name_native="Deutsch",
                script="Latin",
                direction="ltr"
            ),
            LanguageCode.ES: LanguageProfile(
                code=LanguageCode.ES,
                name="Spanish",
                name_native="Español",
                script="Latin",
                direction="ltr"
            ),
            LanguageCode.RU: LanguageProfile(
                code=LanguageCode.RU,
                name="Russian",
                name_native="Русский",
                script="Cyrillic",
                direction="ltr"
            ),
            LanguageCode.AR: LanguageProfile(
                code=LanguageCode.AR,
                name="Arabic",
                name_native="العربية",
                script="Arabic",
                direction="rtl"
            )
        }
        return profiles
    
    def detect_language(self, text: str) -> Tuple[LanguageCode, float]:
        """
        检测文本语言
        
        Args:
            text: 输入文本
        
        Returns:
            (语言代码, 置信度)
        """
        if not text or not text.strip():
            return LanguageCode.UNKNOWN, 0.0
        
        scores = defaultdict(float)
        
        for lang_code, patterns in self.LANGUAGE_PATTERNS.items():
            score = 0.0
            
            # 脚本匹配
            if 'script' in patterns:
                script_matches = len(re.findall(patterns['script'], text))
                score += script_matches * 2.0
            
            # 常用词匹配
            if 'common_words' in patterns:
                text_lower = text.lower()
                word_matches = sum(1 for word in patterns['common_words'] if word in text_lower)
                score += word_matches * 1.5
            
            scores[lang_code] = score
        
        # 选择最高分
        if scores:
            best_lang = max(scores.items(), key=lambda x: x[1])
            total_score = sum(scores.values())
            confidence = best_lang[1] / total_score if total_score > 0 else 0
            
            # 如果置信度太低，返回未知
            if confidence < 0.3:
                return LanguageCode.UNKNOWN, confidence
            
            return best_lang[0], min(confidence, 1.0)
        
        return LanguageCode.UNKNOWN, 0.0
    
    def detect_mixed_language(self, text: str) -> List[Tuple[str, LanguageCode]]:
        """检测混合语言文本"""
        segments = []
        
        # 简单的句子分割
        sentences = re.split(r'[.!?。！？]+', text)
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                lang, _ = self.detect_language(sentence)
                segments.append((sentence, lang))
        
        return segments


class CrossLingualEmbedder:
    """跨语言嵌入器"""
    
    def __init__(self):
        self.language_mappings: Dict[Tuple[LanguageCode, LanguageCode], Any] = {}
        self.semantic_spaces: Dict[LanguageCode, Any] = {}
    
    def embed_multilingual(
        self,
        texts: List[str],
        languages: List[LanguageCode]
    ) -> List[np.ndarray]:
        """
        生成多语言嵌入
        
        注意：这是简化的实现，实际应用中使用多语言模型如 mBERT、XLM-R
        """
        embeddings = []
        
        for text, lang in zip(texts, languages):
            # 简化的嵌入生成（实际应用中使用预训练模型）
            # 这里使用文本特征的哈希作为模拟
            embedding = self._generate_mock_embedding(text, lang)
            embeddings.append(embedding)
        
        return embeddings
    
    def _generate_mock_embedding(
        self, 
        text: str, 
        lang: LanguageCode
    ) -> np.ndarray:
        """生成模拟嵌入"""
        # 使用文本的哈希值生成确定性嵌入
        hash_val = hash(text + lang.value) % 10000
        np.random.seed(hash_val)
        
        # 生成768维嵌入（标准BERT维度）
        embedding = np.random.randn(768)
        
        # 归一化
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding
    
    def compute_cross_lingual_similarity(
        self,
        text1: str,
        lang1: LanguageCode,
        text2: str,
        lang2: LanguageCode
    ) -> float:
        """计算跨语言相似度"""
        emb1 = self._generate_mock_embedding(text1, lang1)
        emb2 = self._generate_mock_embedding(text2, lang2)
        
        # 余弦相似度
        similarity = np.dot(emb1, emb2)
        
        return float(similarity)
    
    def align_semantic_space(
        self,
        parallel_texts: List[Tuple[str, str]],
        src_lang: LanguageCode,
        tgt_lang: LanguageCode
    ) -> Dict[str, Any]:
        """
        对齐语义空间
        
        使用平行语料对齐两种语言的语义空间
        """
        # 简化的对齐过程
        alignment_quality = len(parallel_texts) / 1000  # 假设需要1000对平行文本
        
        return {
            'source_language': src_lang.value,
            'target_language': tgt_lang.value,
            'parallel_samples': len(parallel_texts),
            'alignment_quality': min(alignment_quality, 1.0),
            'aligned': alignment_quality > 0.5
        }


class NeuralMachineTranslator:
    """神经机器翻译器"""
    
    def __init__(self):
        self.translation_cache: Dict[str, str] = {}
        self.supported_pairs: set = self._init_supported_pairs()
    
    def _init_supported_pairs(self) -> set:
        """初始化支持的翻译语言对"""
        # 简化的支持列表
        pairs = set()
        languages = [LanguageCode.EN, LanguageCode.ZH, LanguageCode.JA, LanguageCode.KO, 
                     LanguageCode.FR, LanguageCode.DE, LanguageCode.ES]
        
        for src in languages:
            for tgt in languages:
                if src != tgt:
                    pairs.add((src, tgt))
        
        return pairs
    
    def translate(
        self,
        text: str,
        source_lang: LanguageCode,
        target_lang: LanguageCode
    ) -> TranslatedContent:
        """
        翻译文本
        
        注意：这是简化的实现，实际应用中使用NMT模型
        """
        # 检查缓存
        cache_key = f"{text}:{source_lang.value}:{target_lang.value}"
        if cache_key in self.translation_cache:
            translated = self.translation_cache[cache_key]
        else:
            # 模拟翻译
            translated = self._mock_translate(text, source_lang, target_lang)
            self.translation_cache[cache_key] = translated
        
        # 计算翻译质量
        confidence = self._estimate_translation_quality(
            text, translated, source_lang, target_lang
        )
        
        # 回译检查
        back_translated = self._mock_translate(
            translated, target_lang, source_lang
        )
        
        # 计算语义保持分数
        semantic_score = self._compute_semantic_preservation(
            text, back_translated
        )
        
        return TranslatedContent(
            original_text=text,
            original_language=source_lang,
            translated_text=translated,
            target_language=target_lang,
            translation_confidence=confidence,
            back_translation=back_translated,
            semantic_preservation_score=semantic_score
        )
    
    def _mock_translate(
        self,
        text: str,
        source_lang: LanguageCode,
        target_lang: LanguageCode
    ) -> str:
        """模拟翻译"""
        # 简化的模拟翻译
        translations = {
            (LanguageCode.EN, LanguageCode.ZH): {
                'hello': '你好',
                'world': '世界',
                'machine learning': '机器学习',
                'artificial intelligence': '人工智能'
            },
            (LanguageCode.ZH, LanguageCode.EN): {
                '你好': 'hello',
                '世界': 'world',
                '机器学习': 'machine learning',
                '人工智能': 'artificial intelligence'
            }
        }
        
        pair = (source_lang, target_lang)
        if pair in translations:
            for src, tgt in translations[pair].items():
                if src.lower() in text.lower():
                    text = text.replace(src, tgt)
        
        # 如果没有找到特定翻译，添加标记
        if text and not text.startswith('['):
            text = f"[{target_lang.value.upper()}] {text}"
        
        return text
    
    def _estimate_translation_quality(
        self,
        source: str,
        target: str,
        src_lang: LanguageCode,
        tgt_lang: LanguageCode
    ) -> float:
        """估计翻译质量"""
        # 基于长度比和语言对支持度
        length_ratio = len(target) / (len(source) + 1)
        
        # 理想的长度比接近1
        length_score = 1 - abs(1 - length_ratio)
        
        # 语言对支持度
        pair_supported = (src_lang, tgt_lang) in self.supported_pairs
        support_score = 1.0 if pair_supported else 0.5
        
        return (length_score * 0.5 + support_score * 0.5)
    
    def _compute_semantic_preservation(
        self,
        original: str,
        back_translated: str
    ) -> float:
        """计算语义保持度"""
        # 简化的语义保持度计算
        # 实际应用中使用语义相似度模型
        
        orig_words = set(original.lower().split())
        back_words = set(back_translated.lower().split())
        
        if not orig_words:
            return 0.0
        
        overlap = len(orig_words & back_words)
        return overlap / len(orig_words)
    
    def batch_translate(
        self,
        texts: List[str],
        source_lang: LanguageCode,
        target_lang: LanguageCode
    ) -> List[TranslatedContent]:
        """批量翻译"""
        return [
            self.translate(text, source_lang, target_lang)
            for text in texts
        ]


class CulturalAdapter:
    """文化适配器"""
    
    def __init__(self):
        self.cultural_profiles: Dict[LanguageCode, Dict] = self._init_cultural_profiles()
    
    def _init_cultural_profiles(self) -> Dict[LanguageCode, Dict]:
        """初始化文化配置文件"""
        return {
            LanguageCode.EN: {
                'formality': 'medium',
                'directness': 'high',
                'context_dependence': 'low',
                'time_orientation': 'monochronic',
                'communication_style': 'explicit'
            },
            LanguageCode.ZH: {
                'formality': 'high',
                'directness': 'low',
                'context_dependence': 'high',
                'time_orientation': 'polychronic',
                'communication_style': 'implicit'
            },
            LanguageCode.JA: {
                'formality': 'very_high',
                'directness': 'very_low',
                'context_dependence': 'very_high',
                'time_orientation': 'polychronic',
                'communication_style': 'highly_implicit'
            },
            LanguageCode.AR: {
                'formality': 'high',
                'directness': 'medium',
                'context_dependence': 'high',
                'time_orientation': 'polychronic',
                'communication_style': 'contextual'
            }
        }
    
    def adapt_answer_style(
        self,
        answer: str,
        target_language: LanguageCode,
        target_culture: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        适配答案风格到目标文化
        
        Args:
            answer: 原始答案
            target_language: 目标语言
            target_culture: 目标文化（可选）
        
        Returns:
            适配后的答案和元数据
        """
        profile = self.cultural_profiles.get(target_language, {})
        
        adapted_answer = answer
        adaptations = []
        
        # 根据正式程度调整
        if profile.get('formality') == 'very_high':
            adapted_answer = self._make_formal(adapted_answer)
            adaptations.append('increased_formality')
        elif profile.get('formality') == 'low':
            adapted_answer = self._make_casual(adapted_answer)
            adaptations.append('decreased_formality')
        
        # 根据直接程度调整
        if profile.get('directness') == 'low':
            adapted_answer = self._make_indirect(adapted_answer)
            adaptations.append('made_indirect')
        
        # 添加上下文信息
        if profile.get('context_dependence') == 'high':
            adapted_answer = self._add_context(adapted_answer)
            adaptations.append('added_context')
        
        return {
            'original_answer': answer,
            'adapted_answer': adapted_answer,
            'target_language': target_language.value,
            'cultural_profile': profile,
            'adaptations_applied': adaptations,
            'cultural_distance': self._calculate_cultural_distance(
                LanguageCode.EN, target_language
            )
        }
    
    def _make_formal(self, text: str) -> str:
        """使文本更正式"""
        # 简化的正式化
        formal_replacements = {
            "don't": "do not",
            "can't": "cannot",
            "won't": "will not",
            "it's": "it is"
        }
        
        for informal, formal in formal_replacements.items():
            text = text.replace(informal, formal)
            text = text.replace(informal.capitalize(), formal.capitalize())
        
        return text
    
    def _make_casual(self, text: str) -> str:
        """使文本更随意"""
        # 简化的随意化
        return text.replace("Therefore", "So").replace("However", "But")
    
    def _make_indirect(self, text: str) -> str:
        """使文本更间接"""
        # 添加缓和语
        indirect_prefixes = [
            "It seems that ",
            "One might consider that ",
            "It appears that "
        ]
        
        if not text.startswith(tuple(indirect_prefixes)):
            text = "It seems that " + text[0].lower() + text[1:]
        
        return text
    
    def _add_context(self, text: str) -> str:
        """添加上下文信息"""
        # 简化的上下文添加
        return "Considering the broader context, " + text
    
    def _calculate_cultural_distance(
        self,
        source: LanguageCode,
        target: LanguageCode
    ) -> float:
        """计算文化距离"""
        if source == target:
            return 0.0
        
        src_profile = self.cultural_profiles.get(source, {})
        tgt_profile = self.cultural_profiles.get(target, {})
        
        if not src_profile or not tgt_profile:
            return 0.5  # 默认中等距离
        
        # 计算属性差异
        differences = 0
        total_attrs = 0
        
        for key in src_profile:
            if key in tgt_profile:
                total_attrs += 1
                if src_profile[key] != tgt_profile[key]:
                    differences += 1
        
        return differences / total_attrs if total_attrs > 0 else 0.5


# 主引擎类
class MultilingualCrosslingualEngine:
    """多语言跨语言引擎"""
    
    def __init__(self):
        self.language_detector = LanguageDetector()
        self.cross_lingual_embedder = CrossLingualEmbedder()
        self.translator = NeuralMachineTranslator()
        self.cultural_adapter = CulturalAdapter()
    
    def process_multilingual_query(
        self,
        query: str,
        target_answer_language: Optional[LanguageCode] = None
    ) -> Dict[str, Any]:
        """
        处理多语言查询
        
        这是V115创新的主要入口函数
        """
        # 1. 检测查询语言
        detected_lang, confidence = self.language_detector.detect_language(query)
        
        # 2. 如果是混合语言，分割处理
        if detected_lang == LanguageCode.UNKNOWN:
            segments = self.language_detector.detect_mixed_language(query)
        else:
            segments = [(query, detected_lang)]
        
        # 3. 翻译到英语（用于检索）
        english_queries = []
        for segment, lang in segments:
            if lang != LanguageCode.EN:
                translation = self.translator.translate(
                    segment, lang, LanguageCode.EN
                )
                english_queries.append(translation.translated_text)
            else:
                english_queries.append(segment)
        
        # 4. 生成跨语言嵌入
        embeddings = self.cross_lingual_embedder.embed_multilingual(
            [query],
            [detected_lang]
        )
        
        # 5. 确定答案语言
        answer_language = target_answer_language or detected_lang
        
        # 6. 准备检索用的查询
        retrieval_query = " ".join(english_queries)
        
        return {
            'version': 'V115',
            'name': 'Multilingual & Cross-lingual Retrieval (MCR)',
            'query_analysis': {
                'original_query': query,
                'detected_language': detected_lang.value,
                'detection_confidence': confidence,
                'is_mixed_language': len(segments) > 1,
                'segments': [
                    {'text': seg, 'language': lang.value}
                    for seg, lang in segments
                ]
            },
            'translation': {
                'retrieval_query': retrieval_query,
                'translations': [
                    {
                        'original': seg,
                        'translated': trans,
                        'language': lang.value
                    }
                    for (seg, lang), trans in zip(segments, english_queries)
                ]
            },
            'embedding': {
                'dimension': len(embeddings[0]) if embeddings else 0,
                'cross_lingual_capable': True
            },
            'answer_configuration': {
                'target_language': answer_language.value,
                'requires_translation': answer_language != LanguageCode.EN
            }
        }
    
    def generate_multilingual_answer(
        self,
        english_answer: str,
        target_language: LanguageCode,
        adapt_culturally: bool = True
    ) -> Dict[str, Any]:
        """生成多语言答案"""
        # 1. 翻译答案
        if target_language != LanguageCode.EN:
            translation = self.translator.translate(
                english_answer,
                LanguageCode.EN,
                target_language
            )
            translated_answer = translation.translated_text
            translation_quality = translation.translation_confidence
        else:
            translated_answer = english_answer
            translation_quality = 1.0
        
        # 2. 文化适配
        if adapt_culturally:
            cultural_adaptation = self.cultural_adapter.adapt_answer_style(
                translated_answer,
                target_language
            )
            final_answer = cultural_adaptation['adapted_answer']
        else:
            cultural_adaptation = None
            final_answer = translated_answer
        
        return {
            'english_answer': english_answer,
            'translated_answer': translated_answer,
            'final_answer': final_answer,
            'target_language': target_language.value,
            'translation_quality': translation_quality,
            'cultural_adaptation': cultural_adaptation,
            'multilingual_support': True
        }
    
    def cross_lingual_retrieval(
        self,
        query: str,
        documents: List[Dict],
        top_k: int = 5
    ) -> Dict[str, Any]:
        """
        跨语言检索
        
        检索与查询语义相关的文档，即使它们使用不同语言
        """
        # 检测查询语言
        query_lang, _ = self.language_detector.detect_language(query)
        
        # 生成查询嵌入
        query_emb = self.cross_lingual_embedder._generate_mock_embedding(
            query, query_lang
        )
        
        # 计算与所有文档的相似度
        scored_documents = []
        for doc in documents:
            doc_text = doc.get('content', '')
            doc_lang = LanguageCode(doc.get('language', 'en'))
            
            # 生成文档嵌入
            doc_emb = self.cross_lingual_embedder._generate_mock_embedding(
                doc_text, doc_lang
            )
            
            # 计算跨语言相似度
            similarity = np.dot(query_emb, doc_emb)
            
            scored_documents.append({
                'document': doc,
                'similarity': similarity,
                'language': doc_lang.value
            })
        
        # 排序并返回top-k
        scored_documents.sort(key=lambda x: x['similarity'], reverse=True)
        
        return {
            'query_language': query_lang.value,
            'total_documents': len(documents),
            'retrieved_documents': scored_documents[:top_k],
            'cross_lingual_matches': sum(
                1 for d in scored_documents[:top_k]
                if d['language'] != query_lang.value
            )
        }


# 便捷函数
def build_multilingual_crosslingual_retrieval(
    query: str,
    documents: Optional[List[Dict]] = None,
    target_language: Optional[str] = None
) -> Dict[str, Any]:
    """
    构建多语言跨语言检索
    
    这是V115创新的便捷入口函数
    """
    engine = MultilingualCrosslingualEngine()
    
    # 处理查询
    query_result = engine.process_multilingual_query(
        query,
        LanguageCode(target_language) if target_language else None
    )
    
    result = {
        'version': 'V115',
        'name': 'Multilingual & Cross-lingual Retrieval (MCR)',
        'query_processing': query_result
    }
    
    # 如果提供了文档，执行跨语言检索
    if documents:
        retrieval_result = engine.cross_lingual_retrieval(query, documents)
        result['retrieval'] = retrieval_result
    
    return result


# 示例用法
if __name__ == "__main__":
    # 测试多语言处理
    queries = [
        "What is machine learning?",
        "什么是机器学习？",
        "機械学習とは何ですか？",
        "Qu'est-ce que l'apprentissage automatique?"
    ]
    
    engine = MultilingualCrosslingualEngine()
    
    for query in queries:
        result = engine.process_multilingual_query(query)
        print(f"\nQuery: {query}")
        print(f"Detected: {result['query_analysis']['detected_language']}")
        print(f"Confidence: {result['query_analysis']['detection_confidence']:.2f}")
