"""
V137 - Temporal Reasoning Enhancement (时序推理增强)
====================================================

支持时间维度的事件因果链推理，包括：
- 时间线构建与事件排序
- 因果时序依赖检测
- 趋势预测与异常检测
- 版本化知识追踪
- 时间锚定查询理解

核心能力：
- 从文本中提取时间表达式和事件
- 构建时间线并检测时间关系（before, after, during, overlaps）
- 基于时间顺序的因果推理
- 趋势分析和预测
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict


class TemporalRelation(Enum):
    """时间关系类型"""
    BEFORE = "before"
    AFTER = "after"
    DURING = "during"
    OVERLAPS = "overlaps"
    CONTAINS = "contains"
    STARTS = "starts"
    ENDS = "ends"
    EQUALS = "equals"
    SIMULTANEOUS = "simultaneous"
    VAGUELY_BEFORE = "vaguely_before"  # 大约在...之前
    VAGUELY_AFTER = "vaguely_after"


class TimeUnit(Enum):
    """时间粒度"""
    YEAR = "year"
    QUARTER = "quarter"
    MONTH = "month"
    WEEK = "week"
    DAY = "day"
    HOUR = "hour"
    MINUTE = "minute"
    SECOND = "second"
    EPOCH = "epoch"  # 无具体时间，仅相对顺序
    UNKNOWN = "unknown"


@dataclass
class TemporalExpression:
    """时间表达式"""
    text: str
    normalized_value: Optional[datetime] = None
    time_unit: TimeUnit = TimeUnit.UNKNOWN
    is_duration: bool = False  # 是否是时间段而非时间点
    duration_days: Optional[int] = None
    confidence: float = 0.0
    context: str = ""  # 周围上下文
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'text': self.text,
            'normalized_value': self.normalized_value.isoformat() if self.normalized_value else None,
            'time_unit': self.time_unit.value,
            'is_duration': self.is_duration,
            'duration_days': self.duration_days,
            'confidence': self.confidence
        }


@dataclass
class TemporalEvent:
    """时间事件"""
    event_id: str
    text: str
    event_type: str  # state_change, process_start, process_end, observation, milestone
    temporal_expressions: List[TemporalExpression] = field(default_factory=list)
    inferred_time: Optional[datetime] = None
    time_anchor: Optional[str] = None  # 锚定到的时间点或事件
    participants: List[str] = field(default_factory=list)
    location: Optional[str] = None
    causality_links: List[str] = field(default_factory=list)  # 可能的因果关联事件ID
    confidence: float = 0.0
    source_chunk_id: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'event_id': self.event_id,
            'text': self.text,
            'event_type': self.event_type,
            'temporal_expressions': [te.to_dict() for te in self.temporal_expressions],
            'inferred_time': self.inferred_time.isoformat() if self.inferred_time else None,
            'time_anchor': self.time_anchor,
            'participants': self.participants,
            'location': self.location,
            'causality_links': self.causality_links,
            'confidence': self.confidence
        }


@dataclass
class TemporalEdge:
    """时间图边（事件间的关系）"""
    edge_id: str
    source_event_id: str
    target_event_id: str
    relation: TemporalRelation
    confidence: float = 0.0
    evidence_text: str = ""
    inferred: bool = False  # 是否是推断出的关系
    gap_duration: Optional[int] = None  # 时间间隔（天）


@dataclass
class Timeline:
    """时间线"""
    timeline_id: str
    events: List[TemporalEvent] = field(default_factory=list)
    edges: List[TemporalEdge] = field(default_factory=list)
    title: str = ""
    time_span: Tuple[Optional[datetime], Optional[datetime]] = (None, None)
    coverage_score: float = 0.0  # 时间覆盖完整度
    
    def get_sorted_events(self) -> List[TemporalEvent]:
        """按时间排序的事件列表"""
        valid_events = [e for e in self.events if e.inferred_time]
        return sorted(valid_events, key=lambda e: e.inferred_time or datetime.min)


class TemporalExpressionExtractor:
    """时间表达式提取器"""
    
    # 绝对时间模式
    ABSOLUTE_PATTERNS = [
        # ISO 格式
        (r'\b(\d{4})-(\d{1,2})-(\d{1,2})\b', '%Y-%m-%d'),
        (r'\b(\d{1,2})/(\d{1,2})/(\d{4})\b', '%m/%d/%Y'),
        (r'\b(\d{1,2})\.(\d{1,2})\.(\d{4})\b', '%d.%m.%Y'),
        # 年份
        (r'\b(19|20)\d{2}\b', '%Y'),
        # 季度
        (r'\bQ([1-4])[\s\-](19|20)\d{2}\b', 'Q%Y'),
        # 月份
        (r'\b(January|February|March|April|May|June|July|August|September|October|November|December)\s+((?:19|20)\d{2})?\b', '%B %Y'),
        (r'\b(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[.\s]+((?:19|20)\d{2})?\b', '%b %Y'),
    ]
    
    # 相对时间模式
    RELATIVE_PATTERNS = [
        (r'\b(\d+)\s*(years?|months?|weeks?|days?|hours?|minutes?)\s+ago\b', -1),
        (r'\bin\s+the\s+(last|past)\s+(\d+)\s*(years?|months?|weeks?|days?)\b', -1),
        (r'\b(\d+)\s*(years?|months?|weeks?|days?|hours?|minutes?)\s+(later|after|from now)\b', 1),
        (r'\bin\s+the\s+next\s+(\d+)\s*(years?|months?|weeks?|days?)\b', 1),
    ]
    
    # 模糊时间模式
    VAGUE_PATTERNS = [
        (r'\b(recently|lately|currently|now|today|presently)\b', 'present'),
        (r'\b(in the past|previously|formerly|historically)\b', 'past'),
        (r'\b(in the future|soon|upcoming|eventually)\b', 'future'),
        (r'\b(early|mid|late)\s+(19|20)\d{2}s?\b', 'decade_period'),
        (r'\b(the\s+)?(19|20)\d{2}s\b', 'decade'),
    ]
    
    # 时间段/持续时间
    DURATION_PATTERNS = [
        (r'\bfor\s+(\d+)\s*(years?|months?|weeks?|days?|hours?|minutes?|seconds?)\b'),
        (r'\bover\s+(the\s+)?(?:course\s+of\s+)?(\d+)\s*(years?|months?|weeks?|days?)\b'),
        (r'\bduring\s+(a\s+)?period\s+of\s+(\d+)\s*(years?|months?|weeks?|days?)\b'),
        (r'\bspanning\s+(\d+)\s*(years?|months?|weeks?|days?)\b'),
    ]
    
    # 时间顺序词
    SEQUENCE_INDICATORS = [
        (r'\b(first|initially|originally|to begin with|at first)\b', 1),
        (r'\b(next|then|subsequently|afterwards|following this)\b', 2),
        (r'\b(finally|eventually|in the end|ultimately|lastly)\b', 3),
        (r'\b(before(prior|hand)|earlier|preceding)\b', -1),
        (r'\b(after(ward)?|later|subsequent)\b', 1),
        (r'\b(simultaneously|at the same time|concurrently|meanwhile)\b', 0),
    ]
    
    @classmethod
    def extract(cls, text: str, reference_time: Optional[datetime] = None) -> List[TemporalExpression]:
        """
        从文本中提取所有时间表达式
        
        Args:
            text: 输入文本
            reference_time: 参考时间（默认为当前时间）
        
        Returns:
            时间表达式列表
        """
        if reference_time is None:
            reference_time = datetime.now()
        
        expressions = []
        seen_positions = set()
        
        # 提取绝对时间
        for pattern, fmt in cls.ABSOLUTE_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                pos = match.start()
                if pos in seen_positions:
                    continue
                seen_positions.add(pos)
                
                try:
                    if fmt == '%Y':
                        value = datetime(int(match.group()), 1, 1)
                        unit = TimeUnit.YEAR
                    elif fmt == 'Q%Y':
                        quarter = int(match.group(1))
                        year = int(match.group(2))
                        month = quarter * 3 - 2
                        value = datetime(year, month, 1)
                        unit = TimeUnit.QUARTER
                    elif fmt.startswith('%B') or fmt.startswith('%b'):
                        value = datetime.strptime(match.group().strip(), fmt)
                        unit = TimeUnit.MONTH if '%B' in fmt else TimeUnit.MONTH
                    else:
                        value = datetime.strptime(match.group(), fmt)
                        unit = cls._infer_time_unit(fmt)
                    
                    expressions.append(TemporalExpression(
                        text=match.group(),
                        normalized_value=value,
                        time_unit=unit,
                        confidence=0.95,
                        context=text[max(0, pos-30):pos+len(match.group())+30]
                    ))
                except (ValueError, IndexError):
                    continue
        
        # 提取相对时间
        for pattern, direction in cls.RELATIVE_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                pos = match.start()
                if pos in seen_positions:
                    continue
                seen_positions.add(pos)
                
                try:
                    amount = int(match.group(1))
                    unit_str = match.group(2).lower()
                    
                    # 计算偏移
                    delta = cls._parse_timedelta(amount, unit_str)
                    if direction < 0:
                        value = reference_time - delta
                    else:
                        value = reference_time + delta
                    
                    expressions.append(TemporalExpression(
                        text=match.group(),
                        normalized_value=value,
                        time_unit=cls._infer_unit_from_string(unit_str),
                        confidence=0.8,
                        context=text[max(0, pos-30):pos+len(match.group())+30]
                    ))
                except (ValueError, AttributeError):
                    continue
        
        # 提取模糊时间
        for pattern, category in cls.VAGUE_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                pos = match.start()
                if pos in seen_positions:
                    continue
                seen_positions.add(pos)
                
                expressions.append(TemporalExpression(
                    text=match.group(),
                    normalized_value=None,
                    time_unit=TimeUnit.EPOCH,
                    confidence=0.6,
                    context=text[max(0, pos-30):pos+len(match.group())+30]
                ))
        
        # 提取持续时间
        for pattern in cls.DURATION_PATTERNS:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                pos = match.start()
                if pos in seen_positions:
                    continue
                seen_positions.add(pos)
                
                try:
                    amount = int(match.group(1))
                    unit_str = match.group(2).lower() if match.lastindex >= 2 else "days"
                    delta = cls._parse_timedelta(amount, unit_str)
                    
                    expressions.append(TemporalExpression(
                        text=match.group(),
                        normalized_value=None,
                        time_unit=TimeUnit.DAY,
                        is_duration=True,
                        duration_days=delta.days,
                        confidence=0.75,
                        context=text[max(0, pos-30):pos+len(match.group())+30]
                    ))
                except (ValueError, AttributeError):
                    continue
        
        return expressions
    
    @classmethod
    def _infer_time_unit(cls, fmt: str) -> TimeUnit:
        """从格式字符串推断时间单位"""
        if '%Y' in fmt and ('%m' not in fmt and '%d' not in fmt):
            return TimeUnit.YEAR
        elif '%m' in fmt and '%d' not in fmt:
            return TimeUnit.MONTH
        elif '%H' in fmt or '%M' in fmt or '%S' in fmt:
            return TimeUnit.HOUR
        else:
            return TimeUnit.DAY
    
    @classmethod
    def _infer_unit_from_string(cls, unit_str: str) -> TimeUnit:
        """从字符串推断时间单位"""
        if unit_str.startswith('year'):
            return TimeUnit.YEAR
        elif unit_str.startswith('month'):
            return TimeUnit.MONTH
        elif unit_str.startswith('week'):
            return TimeUnit.WEEK
        elif unit_str.startswith('day'):
            return TimeUnit.DAY
        elif unit_str.startswith('hour'):
            return TimeUnit.HOUR
        elif unit_str.startswith('minute'):
            return TimeUnit.MINUTE
        else:
            return TimeUnit.DAY
    
    @classmethod
    def _parse_timedelta(cls, amount: int, unit_str: str) -> timedelta:
        """解析时间增量"""
        unit_str = unit_str.rstrip('s')  # 移除复数
        
        if unit_str.startswith('year'):
            return timedelta(days=amount * 365)
        elif unit_str.startswith('month'):
            return timedelta(days=amount * 30)
        elif unit_str.startswith('week'):
            return timedelta(weeks=amount)
        elif unit_str.startswith('day'):
            return timedelta(days=amount)
        elif unit_str.startswith('hour'):
            return timedelta(hours=amount)
        elif unit_str.startswith('minute'):
            return timedelta(minutes=amount)
        elif unit_str.startswith('second'):
            return timedelta(seconds=amount)
        else:
            return timedelta(days=amount)


class EventExtractor:
    """事件提取器"""
    
    # 事件触发词
    EVENT_TRIGGERS = {
        'state_change': [
            'became', 'changed', 'transformed', 'evolved', 'transitioned',
            'converted', 'shifted', 'turned', 'grew', 'declined'
        ],
        'process_start': [
            'started', 'began', 'launched', 'initiated', 'commenced',
            'introduced', 'adopted', 'implemented', 'established', 'founded'
        ],
        'process_end': [
            'ended', 'finished', 'completed', 'closed', 'terminated',
            'discontinued', 'shut down', 'ceased', 'stopped', 'concluded'
        ],
        'milestone': [
            'reached', 'achieved', 'attained', 'hit', 'surpassed',
            'announced', 'released', 'published', 'launched', 'unveiled'
        ],
        'observation': [
            'observed', 'noticed', 'detected', 'found', 'discovered',
            'reported', 'recorded', 'identified', 'recognized', 'showed'
        ]
    }
    
    # 句子分割模式
    SENTENCE_PATTERN = r'(?<=[.!?])\s+|(?<=[.!?])\n|\n\n'
    
    @classmethod
    def extract_events(cls, text: str, temporal_expressions: List[TemporalExpression]) -> List[TemporalEvent]:
        """
        从文本中提取事件
        
        Args:
            text: 输入文本
            temporal_expressions: 已提取的时间表达式
        
        Returns:
            事件列表
        """
        events = []
        sentences = re.split(cls.SENTENCE_PATTERN, text)
        
        event_counter = 0
        for sent_idx, sentence in enumerate(sentences):
            sentence = sentence.strip()
            if len(sentence) < 10:  # 太短的句子跳过
                continue
            
            # 检测事件类型
            event_type = cls._detect_event_type(sentence)
            if event_type is None:
                continue
            
            # 找到相关的时间表达式
            related_temps = []
            for te in temporal_expressions:
                if te.context and (sentence in te.context or te.text in sentence):
                    related_temps.append(te)
            
            # 推断时间
            inferred_time = None
            time_anchor = None
            if related_temps:
                # 使用最具体的时间
                concrete_temps = [t for t in related_temps if t.normalized_value is not None]
                if concrete_temps:
                    # 选择置信度最高的
                    best_temp = max(concrete_temps, key=lambda t: t.confidence)
                    inferred_time = best_temp.normalized_value
                    time_anchor = best_temp.text
            
            # 提取参与者（大写开头的名词短语）
            participants = cls._extract_participants(sentence)
            
            event = TemporalEvent(
                event_id=f"evt_{event_counter}",
                text=sentence[:200],  # 截断长句子
                event_type=event_type,
                temporal_expressions=related_temps,
                inferred_time=inferred_time,
                time_anchor=time_anchor,
                participants=participants,
                confidence=min(0.9, 0.5 + len(related_temps) * 0.15)
            )
            events.append(event)
            event_counter += 1
        
        return events
    
    @classmethod
    def _detect_event_type(cls, sentence: str) -> Optional[str]:
        """检测句子的事件类型"""
        sentence_lower = sentence.lower()
        
        scores = {}
        for etype, triggers in cls.EVENT_TRIGGERS.items():
            score = sum(1 for trigger in triggers if trigger in sentence_lower)
            if score > 0:
                scores[etype] = score
        
        if not scores:
            return None
        
        return max(scores, key=scores.get)
    
    @classmethod
    def _extract_participants(cls, sentence: str) -> List[str]:
        """提取参与者（简化版）"""
        # 匹配大写字母开头的名词短语
        patterns = [
            r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)(?:\s+(?:system|model|algorithm|company|organization|team))?',
            r'\b([A-Z]{2,})\b',  # 缩写
        ]
        
        participants = set()
        for pattern in patterns:
            matches = re.findall(pattern, sentence)
            for m in matches:
                if isinstance(m, tuple):
                    m = m[0]
                # 过滤掉常见非参与者词汇
                skip_words = {'The', 'This', 'That', 'It', 'There', 'Here', 'These', 'Those'}
                if m not in skip_words and len(m) > 1:
                    participants.add(m)
        
        return list(participants)[:5]


class CausalTemporalReasoner:
    """因果时序推理器"""
    
    CAUSAL_INDICATORS = [
        'because of', 'due to', 'as a result of', 'resulting from',
        'caused by', 'led to', 'triggered by', 'resulted in',
        'therefore', 'thus', 'hence', 'consequently', 'accordingly',
        'enabled', 'allowed', 'facilitated', 'prompted'
    ]
    
    @classmethod
    def infer_causal_chains(cls, timeline: Timeline) -> List[List[str]]:
        """
        推断因果链
        
        Args:
            timeline: 时间线
        
        Returns:
            因果链列表，每条链是一个事件ID序列
        """
        chains = []
        events_by_id = {e.event_id: e for e in timeline.events}
        
        # 基于时间顺序的潜在因果
        sorted_events = timeline.get_sorted_events()
        
        for i, event in enumerate(sorted_events):
            potential_causes = []
            
            # 在之前发生的事件可能是原因
            for j in range(max(0, i-5), i):  # 只看前5个事件
                prev_event = sorted_events[j]
                
                # 检查是否有因果指示词
                combined_text = f"{prev_event.text} {event.text}"
                has_causal_indicator = any(indicator in combined_text.lower() 
                                           for indicator in cls.CAUSAL_INDICATORS)
                
                # 检查实体重叠
                entity_overlap = set(prev_event.participants) & set(event.participants)
                
                # 检查时间合理性（不超过太长时间）
                time_gap = None
                if prev_event.inferred_time and event.inferred_time:
                    time_gap = (event.inferred_time - prev_event.inferred_time).days
                
                # 判断是否可能存在因果关系
                if has_causal_indicator or (entity_overlap and (time_gap is None or 0 <= time_gap <= 365)):
                    potential_causes.append(prev_event.event_id)
            
            if potential_causes:
                chains.append([potential_causes[-1], event.event_id])
        
        return chains
    
    @classmethod
    def detect_trends(cls, timeline: Timeline, metric_keywords: List[str] = None) -> List[Dict[str, Any]]:
        """
        检测趋势
        
        Args:
            timeline: 时间线
            metric_keywords: 要追踪的关键词（如增长率、收入等）
        
        Returns:
            趋势分析结果
        """
        if metric_keywords is None:
            metric_keywords = ['growth', 'increase', 'decrease', 'rise', 'fall', 
                            'revenue', 'profit', 'users', 'market share']
        
        trends = []
        sorted_events = timeline.get_sorted_events()
        
        for keyword in metric_keywords:
            relevant_events = []
            for event in sorted_events:
                if keyword in event.text.lower():
                    relevant_events.append(event)
            
            if len(relevant_events) >= 2:
                # 分析趋势方向
                trend_direction = cls._analyze_trend_direction(keyword, relevant_events)
                
                trends.append({
                    'metric': keyword,
                    'direction': trend_direction,
                    'data_points': len(relevant_events),
                    'start_time': relevant_events[0].inferred_time.isoformat() if relevant_events[0].inferred_time else None,
                    'end_time': relevant_events[-1].inferred_time.isoformat() if relevant_events[-1].inferred_time else None,
                    'events': [e.event_id for e in relevant_events]
                })
        
        return trends
    
    @classmethod
    def _analyze_trend_direction(cls, keyword: str, events: List[TemporalEvent]) -> str:
        """分析趋势方向"""
        increase_indicators = ['increase', 'rise', 'grow', 'gain', 'improve', 'up']
        decrease_indicators = ['decrease', 'fall', 'drop', 'decline', 'lose', 'down']
        
        increase_count = sum(1 for e in events 
                            if any(ind in e.text.lower() for ind in increase_indicators))
        decrease_count = sum(1 for e in events 
                            if any(ind in e.text.lower() for ind in decrease_indicators))
        
        if increase_count > decrease_count * 1.5:
            return "increasing"
        elif decrease_count > increase_count * 1.5:
            return "decreasing"
        else:
            return "stable/mixed"


def build_temporal_reasoning_enhancement(
    question: str,
    evidence_chunks: List[Dict[str, Any]],
    query_graph: Optional[Dict[str, Any]] = None,
    enable_causal_inference: bool = True,
    enable_trend_analysis: bool = True,
    max_timeline_events: int = 50,
) -> Dict[str, Any]:
    """
    构建时序推理增强管道
    
    Args:
        question: 用户问题
        evidence_chunks: 证据块列表
        query_graph: 查询图（可选）
        enable_causal_inference: 是否启用因果推断
        enable_trend_analysis: 是否启用趋势分析
        max_timeline_events: 最大事件数量
    
    Returns:
        {
            'pipeline': str,
            'timeline': Timeline.to_dict(),
            'temporal_statistics': {...},
            'question_temporal_analysis': {...},
            'causal_chains': [...],
            'trends': [...],
            'enhanced_context': str,
            'temporal_answer_constraints': [...]
        }
    """
    result = {
        'pipeline': 'V137_Temporal_Reasoning_Enhancement_v1',
        'timeline': None,
        'temporal_statistics': {
            'total_expressions': 0,
            'absolute_times': 0,
            'relative_times': 0,
            'vague_times': 0,
            'durations': 0,
            'total_events': 0,
            'events_with_time': 0,
            'time_coverage_ratio': 0.0
        },
        'question_temporal_analysis': {},
        'causal_chains': [],
        'trends': [],
        'enhanced_context': '',
        'temporal_answer_constraints': []
    }
    
    # 1. 分析问题中的时间需求
    question_temporal = _analyze_question_temporality(question)
    result['question_temporal_analysis'] = question_temporal
    
    # 2. 从证据中提取时间表达式和事件
    all_expressions = []
    all_events = []
    
    for chunk_idx, chunk in enumerate(evidence_chunks):
        chunk_text = chunk.get('text', '')
        chunk_id = chunk.get('source_id', f'chunk_{chunk_idx}')
        
        # 提取时间表达式
        expressions = TemporalExpressionExtractor.extract(chunk_text)
        for expr in expressions:
            expr.source_chunk = chunk_id
        all_expressions.extend(expressions)
        
        # 提取事件
        events = EventExtractor.extract_events(chunk_text, expressions)
        for event in events:
            event.source_chunk_id = chunk_id
        all_events.extend(events)
    
    # 更新统计
    result['temporal_statistics']['total_expressions'] = len(all_expressions)
    result['temporal_statistics']['absolute_times'] = sum(1 for e in all_expressions if e.normalized_value is not None and not e.is_duration)
    result['temporal_statistics']['relative_times'] = sum(1 for e in all_expressions if e.normalized_value is not None and any(ind in e.text.lower() for ind in ['ago', 'later', 'from now']))
    result['temporal_statistics']['vague_times'] = sum(1 for e in all_expressions if e.time_unit == TimeUnit.EPOCH)
    result['temporal_statistics']['durations'] = sum(1 for e in all_expressions if e.is_duration)
    result['temporal_statistics']['total_events'] = len(all_events)
    result['temporal_statistics']['events_with_time'] = sum(1 for e in all_events if e.inferred_time is not None)
    
    # 3. 构建时间线
    timeline = Timeline(
        timeline_id="main_timeline",
        events=all_events[:max_timeline_events],
        title=f"Timeline for: {question[:50]}"
    )
    
    # 计算时间跨度
    times = [e.inferred_time for e in timeline.events if e.inferred_time]
    if times:
        timeline.time_span = (min(times), max(times))
        result['temporal_statistics']['time_coverage_ratio'] = _calculate_time_coverage(timeline)
    
    # 构建时间边（事件间关系）
    timeline.edges = _build_temporal_edges(timeline.events)
    
    result['timeline'] = _serialize_timeline(timeline)
    
    # 4. 因果推断
    if enable_causal_inference:
        causal_reasoner = CausalTemporalReasoner()
        result['causal_chains'] = causal_reasoner.infer_causal_chains(timeline)
    
    # 5. 趋势分析
    if enable_trend_analysis:
        causal_reasoner = CausalTemporalReasoner()
        # 从问题中提取可能的指标关键词
        metric_keywords = [w for w in question.split() if len(w) > 3]
        result['trends'] = causal_reasoner.detect_trends(timeline, metric_keywords)
    
    # 6. 生成时间约束答案
    result['temporal_answer_constraints'] = _generate_temporal_constraints(
        question, timeline, question_temporal
    )
    
    # 7. 生成增强上下文
    result['enhanced_context'] = _generate_temporal_enhanced_context(
        question, timeline, result['causal_chains'], result['trends']
    )
    
    return result


def _analyze_question_temporality(question: str) -> Dict[str, Any]:
    """分析问题的时间特性"""
    analysis = {
        'has_temporal_intent': False,
        'temporal_question_type': None,
        'time_expressions_in_question': [],
        'requested_time_frame': None,
        'sequence_required': False,
        'comparison_required': False
    }
    
    question_lower = question.lower()
    
    # 时间问题类型
    temporal_patterns = [
        (r'\bwhen\b.*\b(happen|occur|start|end|begin|finish|release|introduce|launch)\b', 'when_did'),
        (r'\bhow long\b', 'duration'),
        (r'\b(since|from|starting)\b.*\b(to|until|till)\b', 'period'),
        (r'\b(before|after|during|while)\b', 'temporal_relation'),
        (r'\b(timeline|chronology|history|evolution|progression)\b', 'sequence'),
        (r'\b(trend|change|increase|decrease|grow|decline)\b', 'trend'),
        (r'\b(compare|versus|vs\.)\b.*\b(before|after|over time)\b', 'temporal_comparison'),
        (r'\b(first|initial|original|latest|recent|current|newest|oldest)\b', 'extreme_time'),
    ]
    
    for pattern, qtype in temporal_patterns:
        if re.search(pattern, question_lower):
            analysis['has_temporal_intent'] = True
            analysis['temporal_question_type'] = qtype
            break
    
    # 提取问题中的时间表达式
    analysis['time_expressions_in_question'] = [
        te.to_dict() for te in TemporalExpressionExtractor.extract(question)
    ]
    
    # 判断是否需要序列
    sequence_words = ['sequence', 'order', 'step', 'phase', 'stage', 'first', 'then', 'finally']
    analysis['sequence_required'] = any(w in question_lower for w in sequence_words)
    
    # 判断是否需要比较
    comparison_words = ['compare', 'difference', 'change', 'versus', 'vs.', 'before vs after']
    analysis['comparison_required'] = any(w in question_lower for w in comparison_words)
    
    return analysis


def _build_temporal_edges(events: List[TemporalEvent]) -> List[TemporalEdge]:
    """构建事件间的时间关系边"""
    edges = []
    sorted_events = sorted(
        [(i, e) for i, e in enumerate(events) if e.inferred_time],
        key=lambda x: x[1].inferred_time or datetime.min
    )
    
    for i, (idx_i, event_i) in enumerate(sorted_events):
        for idx_j, event_j in sorted_events[i+1:i+6]:  # 只连接相邻的几个事件
            if not event_i.inferred_time or not event_j.inferred_time:
                continue
            
            time_diff = (event_j.inferred_time - event_i.inferred_time).days
            
            if time_diff < 0:
                relation = TemporalRelation.BEFORE
            elif time_diff == 0:
                relation = TemporalRelation.SIMULTANEOUS
            elif time_diff <= 7:
                relation = TemporalRelation.OVERLAPS
            elif time_diff <= 30:
                relation = TemporalRelation.VAGUELY_AFTER
            else:
                relation = TemporalRelation.AFTER
            
            edge = TemporalEdge(
                edge_id=f"tedge_{len(edges)}",
                source_event_id=event_i.event_id,
                target_event_id=event_j.event_id,
                relation=relation,
                confidence=0.8 if time_diff != 0 else 0.9,
                gap_duration=time_diff,
                inferred=True
            )
            edges.append(edge)
    
    return edges


def _calculate_time_coverage(timeline: Timeline) -> float:
    """计算时间覆盖度"""
    times = [e.inferred_time for e in timeline.events if e.inferred_time]
    if len(times) < 2:
        return 1.0 if len(times) == 1 else 0.0
    
    time_span = (max(times) - min(times)).days
    if time_span == 0:
        return 1.0
    
    # 简化：检查事件分布的均匀性
    expected_gap = time_span / (len(times) - 1)
    actual_gaps = [(times[i+1] - times[i]).days for i in range(len(times)-1)]
    
    variance = sum((g - expected_gap) ** 2 for g in actual_gaps) / len(actual_gaps)
    std_dev = variance ** 0.5
    
    # 覆盖度基于标准差与期望间隔的比例
    coverage = max(0, 1 - (std_dev / expected_gap) * 0.5) if expected_gap > 0 else 1.0
    return round(min(1.0, coverage), 3)


def _generate_temporal_constraints(
    question: str,
    timeline: Timeline,
    question_analysis: Dict[str, Any]
) -> List[Dict[str, Any]]:
    """生成时间约束条件"""
    constraints = []
    
    if not question_analysis.get('has_temporal_intent'):
        return constraints
    
    sorted_events = timeline.get_sorted_events()
    if not sorted_events:
        return constraints
    
    qtype = question_analysis.get('temporal_question_type')
    
    if qtype == 'when_did':
        # 需要找到特定事件的最早/最晚时间
        for event in sorted_events[:3]:  # 返回最早的相关事件
            if event.inferred_time:
                constraints.append({
                    'type': 'time_point',
                    'event_id': event.event_id,
                    'time': event.inferred_time.isoformat(),
                    'confidence': event.confidence
                })
    
    elif qtype == 'duration':
        # 需要计算持续时间
        if len(sorted_events) >= 2:
            start_event = sorted_events[0]
            end_event = sorted_events[-1]
            if start_event.inferred_time and end_event.inferred_time:
                duration = (end_event.inferred_time - start_event.inferred_time).days
                constraints.append({
                    'type': 'duration_range',
                    'start_event': start_event.event_id,
                    'end_event': end_event.event_id,
                    'min_duration_days': duration,
                    'confidence': min(start_event.confidence, end_event.confidence)
                })
    
    elif qtype in ['sequence', 'timeline']:
        # 需要按时间顺序排列事件
        event_sequence = []
        for event in sorted_events[:10]:
            event_sequence.append({
                'event_id': event.event_id,
                'order': len(event_sequence),
                'time': event.inferred_time.isoformat() if event.inferred_time else None,
                'summary': event.text[:100]
            })
        constraints.append({
            'type': 'event_sequence',
            'sequence': event_sequence
        })
    
    elif qtype == 'trend':
        # 需要趋势信息
        constraints.append({
            'type': 'trend_constraint',
            'event_count': len(sorted_events),
            'time_span_days': (sorted_events[-1].inferred_time - sorted_events[0].inferred_time).days 
                              if len(sorted_events) >= 2 and sorted_events[0].inferred_time and sorted_events[-1].inferred_time else None
        })
    
    return constraints


def _generate_temporal_enhanced_context(
    question: str,
    timeline: Timeline,
    causal_chains: List[List[str]],
    trends: List[Dict[str, Any]]
) -> str:
    """生成时序增强的上下文"""
    sections = []
    
    sections.append(f"# Temporal Analysis for Query: {question}")
    sections.append("")
    
    # 时间线摘要
    sorted_events = timeline.get_sorted_events()
    if sorted_events:
        sections.append("## Timeline Summary:")
        sections.append(f"- **Total Events**: {len(timeline.events)}")
        sections.append(f"- **Events with Time Anchors**: {sum(1 for e in timeline.events if e.inferred_time)}")
        
        if timeline.time_span[0] and timeline.time_span[1]:
            span = timeline.time_span[1] - timeline.time_span[0]
            sections.append(f"- **Time Span**: {timeline.time_span[0].strftime('%Y-%m-%d')} to {timeline.time_span[1].strftime('%Y-%m-%d')} ({span.days} days)")
        sections.append("")
        
        # 关键时间节点
        sections.append("### Key Temporal Nodes:")
        for event in sorted_events[:8]:
            time_str = event.inferred_time.strftime('%Y-%m-%d') if event.inferred_time else '[unknown]'
            sections.append(f"- [{time_str}] {event.text[:80]}...")
        sections.append("")
    
    # 因果链
    if causal_chains:
        sections.append("## Inferred Causal Chains:")
        events_map = {e.event_id: e for e in timeline.events}
        for chain in causal_chains[:5]:
            chain_desc = " → ".join([
                events_map[eid].text[:40] + "..." if eid in events_map else eid
                for eid in chain
            ])
            sections.append(f"- {chain_desc}")
        sections.append("")
    
    # 趋势
    if trends:
        sections.append("## Detected Trends:")
        for trend in trends[:5]:
            emoji = "📈" if trend['direction'] == 'increasing' else "📉" if trend['direction'] == 'decreasing' else "➡️"
            sections.append(f"- {emoji} **{trend['metric']}**: {trend['direction']} ({trend['data_points']} data points)")
        sections.append("")
    
    return '\n'.join(sections)


def _serialize_timeline(timeline: Timeline) -> Dict[str, Any]:
    """序列化时间线"""
    return {
        'timeline_id': timeline.timeline_id,
        'title': timeline.title,
        'events': [e.to_dict() for e in timeline.events],
        'edges': [{
            'edge_id': e.edge_id,
            'source': e.source_event_id,
            'target': e.target_event_id,
            'relation': e.relation.value,
            'confidence': e.confidence,
            'gap_duration': e.gap_duration
        } for e in timeline.edges],
        'time_span': (
            timeline.time_span[0].isoformat() if timeline.time_span[0] else None,
            timeline.time_span[1].isoformat() if timeline.time_span[1] else None
        ),
        'coverage_score': timeline.coverage_score
    }


# 导出别名
run_temporal_reasoning = build_temporal_reasoning_enhancement

__all__ = [
    'TemporalExpression',
    'TemporalEvent',
    'TemporalEdge',
    'Timeline',
    'TemporalRelation',
    'TimeUnit',
    'TemporalExpressionExtractor',
    'EventExtractor',
    'CausalTemporalReasoner',
    'build_temporal_reasoning_enhancement',
    'run_temporal_reasoning',
]
