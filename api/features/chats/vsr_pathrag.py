from __future__ import annotations

import hashlib
import json
import re
from copy import deepcopy
from typing import Any

from PathRAG.prompt import GRAPH_FIELD_SEP


STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "between",
    "by",
    "did",
    "do",
    "does",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}

INTENT_KEYWORDS = {
    "cause_effect": {
        "affect",
        "because",
        "cause",
        "consequence",
        "effect",
        "impact",
        "lead",
        "reason",
        "result",
    },
    "mechanism": {
        "add",
        "adding",
        "approach",
        "how",
        "improve",
        "improved",
        "mechanism",
        "method",
        "monitor",
        "process",
        "through",
        "using",
        "via",
    },
    "comparison": {
        "compare",
        "contrast",
        "differ",
        "difference",
        "similar",
        "versus",
    },
    "role": {
        "direct",
        "manage",
        "operate",
        "provide",
        "responsible",
        "role",
        "supply",
    },
    "time": {
        "after",
        "before",
        "date",
        "during",
        "month",
        "time",
        "timeline",
        "when",
        "year",
    },
    "quantity": {
        "amount",
        "count",
        "number",
        "percent",
        "percentage",
        "quantity",
        "rate",
    },
    "enumeration": {
        "actors",
        "conditions",
        "factors",
        "items",
        "organizations",
        "participants",
        "steps",
        "types",
    },
    "fact": {"attribute", "description", "fact", "information", "property"},
}

NEGATION_TERMS = {
    "cannot",
    "didn't",
    "didnt",
    "doesn't",
    "doesnt",
    "failed",
    "never",
    "no",
    "not",
    "unable",
    "without",
}


def normalize_text(value: Any) -> str:
    return " ".join(str(value or "").strip().split())


def normalize_label(value: Any) -> str:
    return normalize_text(value).strip('"').strip().upper()


def lexical_terms(text: Any) -> set[str]:
    raw = str(text or "").lower()
    raw_english = {
        token
        for token in re.findall(r"[a-z][a-z0-9_-]{1,}", raw)
        if token not in STOP_WORDS
    }
    english = set(raw_english)
    for token in raw_english:
        if token.endswith("ed") and len(token) > 4:
            english.add(token[:-2])
            english.add(token[:-1])
        if token.endswith("ing") and len(token) > 5:
            english.add(token[:-3])
            english.add(token[:-3] + "e")
        if token.endswith("s") and len(token) > 4:
            english.add(token[:-1])
    chinese_sequences = re.findall(r"[\u4e00-\u9fff]{2,}", raw)
    chinese = set()
    for sequence in chinese_sequences:
        if len(sequence) <= 4:
            chinese.add(sequence)
        else:
            chinese.update(sequence[index : index + 2] for index in range(len(sequence) - 1))
    return english | chinese


def split_source_ids(value: Any) -> list[str]:
    if isinstance(value, list):
        candidates = value
    else:
        candidates = str(value or "").split(GRAPH_FIELD_SEP)
    result = []
    for candidate in candidates:
        source_id = normalize_text(candidate)
        if source_id and source_id != "UNKNOWN" and source_id not in result:
            result.append(source_id)
    return result


def _overlap(left: set[str], right: set[str]) -> float:
    if not left:
        return 0.0
    return len(left & right) / max(1, len(left))


def _jaccard(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / max(1, len(left | right))


def _target_type(question: str) -> str:
    lowered = question.lower().strip()
    if lowered.startswith("who") or "which person" in lowered:
        return "PERSON_OR_ORGANIZATION"
    if lowered.startswith("where") or "which location" in lowered:
        return "LOCATION"
    if lowered.startswith("when") or any(term in lowered for term in ["what year", "what date"]):
        return "TIME"
    if any(term in lowered for term in ["how many", "percentage", "percent", "amount"]):
        return "QUANTITY"
    if any(term in lowered for term in ["what technology", "which technology", "what method"]):
        return "METHOD_OR_TECHNOLOGY"
    return "ANSWER_ENTITY_OR_VALUE"


def _relation_intents(question: str, question_type: str, contract: dict[str, Any]) -> list[str]:
    lowered = question.lower()
    intents: list[str] = []
    if question_type == "comparison" or any(word in lowered for word in ["compare", "differ", "difference"]):
        intents.extend(["comparison", "fact"])
    elif question_type == "causal_explanation":
        intents.extend(["cause_effect", "mechanism"])
    elif question_type == "process_or_explanation":
        intents.extend(["mechanism", "cause_effect"])
    elif question_type == "enumeration":
        intents.extend(["enumeration", "role"])
    else:
        intents.append("fact")

    if any(word in lowered for word in ["role", "responsible", "provide", "supply", "directed"]):
        intents.append("role")
    if any(word in lowered for word in ["when", "year", "date", "before", "after", "during"]):
        intents.append("time")
    if contract.get("type") == "precision_lookup":
        intents.append("quantity")
    return list(dict.fromkeys(intents))


def _capitalized_phrases(question: str) -> list[str]:
    phrases = re.findall(r"\b[A-Z][A-Za-z0-9_-]*(?:\s+[A-Z][A-Za-z0-9_-]*){0,5}\b", question)
    blocked = {"How", "What", "When", "Where", "Which", "Who", "Why", "Does", "Did", "Is", "Are"}
    return [phrase for phrase in phrases if phrase not in blocked]


def build_query_graph(
    question: str,
    question_type: str,
    contract: dict[str, Any],
    evidence_summary: dict[str, Any],
) -> dict[str, Any]:
    question_terms = lexical_terms(question)
    anchors: list[str] = []
    known_entities: list[str] = []
    lowered_question = question.lower()

    for entity in evidence_summary.get("entities", []):
        name = normalize_label(entity.get("name"))
        if not name:
            continue
        known_entities.append(name)
        name_terms = lexical_terms(name)
        if name.lower() in lowered_question or _overlap(name_terms, question_terms) >= 0.67:
            anchors.append(name)

    for phrase in _capitalized_phrases(question):
        label = normalize_label(phrase)
        label_terms = lexical_terms(label)
        # Prefer canonical entities already resolved from the graph. A shorter
        # surface form such as "AURORA" must not become a second disconnected
        # anchor when "AURORA PROJECT" is already grounded.
        subsumed = any(
            label_terms
            and label_terms <= lexical_terms(anchor)
            for anchor in anchors
        )
        if label and label not in anchors and not subsumed:
            anchors.append(label)

    intents = _relation_intents(question, question_type, contract)
    anchor_terms = set().union(*(lexical_terms(anchor) for anchor in anchors)) if anchors else set()
    required_facets = sorted(question_terms - anchor_terms)[:16]
    requires_multi_hop = question_type in {"comparison", "causal_explanation", "process_or_explanation"}
    if len(anchors) >= 2:
        requires_multi_hop = True
    max_hops = 3 if requires_multi_hop else 2

    nodes = [
        {"id": f"anchor-{index}", "label": anchor, "role": "anchor"}
        for index, anchor in enumerate(anchors)
    ]
    nodes.append({"id": "target", "label": "?answer", "role": "target", "type": _target_type(question)})
    source_node = nodes[0]["id"] if anchors else "query"
    if not anchors:
        nodes.insert(0, {"id": "query", "label": normalize_text(question), "role": "query"})
    edges = [
        {
            "id": f"intent-{index}",
            "source": source_node,
            "target": "target",
            "intent": intent,
            "keywords": sorted(INTENT_KEYWORDS[intent]),
        }
        for index, intent in enumerate(intents)
    ]
    return {
        "question": normalize_text(question),
        "question_type": question_type,
        "contract_type": contract.get("type", "balanced"),
        "nodes": nodes,
        "edges": edges,
        "anchors": anchors,
        "known_entities": list(dict.fromkeys(known_entities))[:20],
        "target_type": _target_type(question),
        "relation_intents": intents,
        "required_facets": required_facets,
        "requires_multi_hop": requires_multi_hop,
        "max_hops": max_hops,
    }


def diagnose_knowledge_gaps(
    query_graph: dict[str, Any],
    evidence_summary: dict[str, Any],
) -> dict[str, Any]:
    relationships = evidence_summary.get("relationships", [])
    evidence_blob = " ".join(
        " ".join(
            [
                str(row.get("src", "")),
                str(row.get("tgt", "")),
                str(row.get("keywords", "")),
                str(row.get("description", "")),
            ]
        )
        for row in relationships
    )
    evidence_terms = lexical_terms(evidence_blob)
    question_terms = lexical_terms(query_graph.get("question"))
    question_coverage = _overlap(question_terms, evidence_terms)

    evidence_entities = {
        normalize_label(entity.get("name"))
        for entity in evidence_summary.get("entities", [])
        if entity.get("name")
    }
    evidence_entities.update(
        normalize_label(value)
        for relation in relationships
        for value in [relation.get("src"), relation.get("tgt")]
        if value
    )
    anchors = query_graph.get("anchors", [])
    anchored = [
        anchor
        for anchor in anchors
        if anchor in evidence_entities
        or any(_overlap(lexical_terms(anchor), lexical_terms(entity)) >= 0.67 for entity in evidence_entities)
    ]
    anchor_coverage = len(anchored) / max(1, len(anchors)) if anchors else 1.0

    intent_scores: dict[str, float] = {}
    for intent in query_graph.get("relation_intents", []):
        if intent == "fact":
            intent_scores[intent] = 1.0 if relationships else 0.0
            continue
        keywords = INTENT_KEYWORDS.get(intent, set())
        best = 0.0
        for relation in relationships:
            relation_terms = lexical_terms(
                " ".join(
                    [
                        str(relation.get("keywords", "")),
                        str(relation.get("description", "")),
                    ]
                )
            )
            best = max(best, _overlap(keywords, relation_terms))
        intent_scores[intent] = round(best, 3)

    missing_intents = [intent for intent, score in intent_scores.items() if score < 0.12]
    intent_coverage = 1 - len(missing_intents) / max(1, len(intent_scores))

    adjacency: dict[str, set[str]] = {}
    for relation in relationships:
        src = normalize_label(relation.get("src"))
        tgt = normalize_label(relation.get("tgt"))
        if src and tgt:
            adjacency.setdefault(src, set()).add(tgt)
            adjacency.setdefault(tgt, set()).add(src)

    connected = True
    if len(anchored) >= 2:
        start = anchored[0]
        frontier = [start]
        seen = {start}
        while frontier:
            node = frontier.pop()
            for neighbor in adjacency.get(node, set()):
                if neighbor not in seen:
                    seen.add(neighbor)
                    frontier.append(neighbor)
        connected = all(anchor in seen for anchor in anchored[1:])

    connectivity_penalty = 0.0 if connected else 1.0
    gap_score = (
        0.42 * (1 - intent_coverage)
        + 0.30 * (1 - question_coverage)
        + 0.18 * (1 - anchor_coverage)
        + 0.10 * connectivity_penalty
    )
    no_relationships = not relationships
    substantive_intents = [intent for intent in intent_scores if intent != "fact"]
    all_substantive_intents_missing = bool(
        substantive_intents
        and all(intent in missing_intents for intent in substantive_intents)
    )
    repair_required = bool(
        no_relationships
        or gap_score >= 0.55
        or all_substantive_intents_missing
        or (missing_intents and question_coverage < 0.40)
        or (not connected and len(anchors) >= 2)
    )
    reasons = []
    if no_relationships:
        reasons.append("no_relationship_evidence")
    if missing_intents:
        reasons.append("missing_relation_intents")
    if all_substantive_intents_missing:
        reasons.append("all_substantive_relation_intents_missing")
    if question_coverage < 0.40:
        reasons.append("low_question_evidence_coverage")
    if anchor_coverage < 1.0:
        reasons.append("unresolved_query_anchor")
    if not connected:
        reasons.append("query_anchors_disconnected")
    if not reasons:
        reasons.append("graph_satisfies_query_constraints")
    return {
        "gap_score": round(gap_score, 3),
        "repair_required": repair_required,
        "reasons": reasons,
        "question_coverage": round(question_coverage, 3),
        "anchor_coverage": round(anchor_coverage, 3),
        "intent_coverage": round(intent_coverage, 3),
        "intent_scores": intent_scores,
        "missing_intents": missing_intents,
        "all_substantive_intents_missing": all_substantive_intents_missing,
        "connected_anchors": connected,
        "resolved_anchors": anchored,
    }


async def collect_source_chunks(
    rag: Any,
    question: str,
    query_graph: dict[str, Any],
    evidence_summary: dict[str, Any],
    max_chunks: int = 6,
    max_chunk_chars: int = 5000,
) -> list[dict[str, Any]]:
    store = getattr(rag, "text_chunks", None)
    if store is None:
        return []
    raw_data = getattr(store, "_data", None)
    if not isinstance(raw_data, dict):
        keys = await store.all_keys()
        values = await store.get_by_ids(keys)
        raw_data = {key: value for key, value in zip(keys, values) if value}

    priority_sources = set()
    for entity in evidence_summary.get("entities", []):
        priority_sources.update(split_source_ids(entity.get("source_ids") or entity.get("source_id")))
    for relation in evidence_summary.get("relationships", []):
        priority_sources.update(split_source_ids(relation.get("source_ids") or relation.get("source_id")))

    query_terms = lexical_terms(question)
    intent_terms = set().union(
        *(INTENT_KEYWORDS.get(intent, set()) for intent in query_graph.get("relation_intents", []))
    )
    anchor_terms = set().union(
        *(lexical_terms(anchor) for anchor in query_graph.get("anchors", []))
    ) if query_graph.get("anchors") else set()
    rows = []
    for source_id, chunk in raw_data.items():
        if not chunk:
            continue
        text = normalize_text(chunk.get("content", ""))
        text_terms = lexical_terms(text)
        query_score = _overlap(query_terms, text_terms)
        intent_score = _overlap(intent_terms, text_terms)
        anchor_score = _overlap(anchor_terms, text_terms) if anchor_terms else 0.0
        priority_bonus = 0.20 if source_id in priority_sources else 0.0
        score = 0.50 * query_score + 0.20 * intent_score + 0.20 * anchor_score + priority_bonus
        rows.append(
            {
                "source_id": source_id,
                "full_doc_id": chunk.get("full_doc_id"),
                "text": text[:max_chunk_chars],
                "score": round(score, 4),
            }
        )
    return sorted(rows, key=lambda row: row["score"], reverse=True)[:max_chunks]


def _sentence_candidates(
    chunks: list[dict[str, Any]],
    query_graph: dict[str, Any],
    gap_report: dict[str, Any],
) -> list[dict[str, Any]]:
    known_entities = list(dict.fromkeys(query_graph.get("anchors", []) + query_graph.get("known_entities", [])))
    missing_intents = gap_report.get("missing_intents") or query_graph.get("relation_intents") or ["fact"]
    intent = missing_intents[0]
    intent_terms = INTENT_KEYWORDS.get(intent, set())
    ranked_result = []
    question_terms = lexical_terms(query_graph.get("question"))
    required_facets = set(query_graph.get("required_facets", []))
    for chunk in chunks:
        sentences = re.split(r"(?<=[.!?])\s+|[\r\n]+", chunk.get("text", ""))
        sentences = [normalize_text(sentence) for sentence in sentences if normalize_text(sentence)]
        for sentence_index, sentence in enumerate(sentences):
            sentence = normalize_text(sentence)
            if len(sentence) < 25:
                continue
            mentioned = [entity for entity in known_entities if entity.lower() in sentence.lower()]
            sentence_terms = lexical_terms(sentence)
            relevance = _overlap(question_terms, sentence_terms)
            intent_match = _overlap(intent_terms, sentence_terms) if intent_terms else 0.0
            facet_hits = required_facets & sentence_terms
            if relevance < 0.35 or len(question_terms & sentence_terms) < 2:
                continue
            if required_facets and len(facet_hits) < min(2, len(required_facets)):
                continue
            evidence_sentence = sentence
            if not mentioned and query_graph.get("anchors"):
                for previous_index in range(sentence_index - 1, max(-1, sentence_index - 5), -1):
                    previous = sentences[previous_index]
                    previous_mentions = [
                        entity for entity in known_entities if entity.lower() in previous.lower()
                    ]
                    if not previous_mentions:
                        previous_mentions = [
                            anchor
                            for anchor in query_graph.get("anchors", [])
                            if anchor.lower() in previous.lower()
                        ]
                    if previous_mentions:
                        mentioned = previous_mentions
                        evidence_sentence = " ".join(sentences[previous_index : sentence_index + 1])
                        break
            if not mentioned:
                continue
            tail = mentioned[1] if len(mentioned) >= 2 else ""
            if not tail:
                marker_match = re.search(
                    r"\b(?:by|through|using|via|with|to)\s+(.{4,160}?)(?:[.;]|$)",
                    sentence,
                    flags=re.IGNORECASE,
                )
                if marker_match:
                    tail = normalize_label(marker_match.group(1))
            if not tail:
                phrases = [
                    normalize_label(phrase)
                    for phrase in _capitalized_phrases(sentence)
                    if normalize_label(phrase) not in mentioned
                ]
                tail = phrases[0] if phrases else ""
            if not tail or tail == mentioned[0]:
                continue
            ranked_result.append(
                (
                    0.72 * relevance + 0.18 * intent_match + 0.10 * _overlap(required_facets, sentence_terms),
                    {
                    "head": mentioned[0],
                    "relation": intent,
                    "tail": tail,
                    "evidence_quote": evidence_sentence[:900],
                    "source_id": chunk.get("source_id"),
                    "extraction_method": "grounded_sentence_fallback",
                    },
                )
            )
    ranked_result.sort(key=lambda row: row[0], reverse=True)
    return [row for _, row in ranked_result[:3]]


def _parse_json_payload(response: str) -> Any:
    text = normalize_text(response)
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s*```$", "", text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    for opener, closer in [("{", "}"), ("[", "]")]:
        start = text.find(opener)
        end = text.rfind(closer)
        if start >= 0 and end > start:
            try:
                return json.loads(text[start : end + 1])
            except json.JSONDecodeError:
                continue
    return {}


async def propose_repair_candidates(
    rag: Any,
    question: str,
    query_graph: dict[str, Any],
    gap_report: dict[str, Any],
    chunks: list[dict[str, Any]],
    allow_llm: bool = True,
    max_candidates: int = 6,
) -> dict[str, Any]:
    candidates = _sentence_candidates(chunks, query_graph, gap_report)
    llm_error = None
    raw_response = ""
    llm_used = False
    if allow_llm and not candidates and chunks and getattr(rag, "llm_model_func", None):
        llm_used = True
        chunk_text = "\n\n".join(
            f"SOURCE_ID={chunk['source_id']}\n{chunk['text'][:3000]}" for chunk in chunks[:4]
        )
        prompt = f"""Question: {question}

Query graph:
{json.dumps(query_graph, ensure_ascii=False)}

Detected knowledge gaps:
{json.dumps(gap_report, ensure_ascii=False)}

Source chunks:
{chunk_text}

Return JSON only in this schema:
{{"triples":[{{"head":"...","relation":"...","tail":"...","evidence_quote":"exact quote from one source","source_id":"..."}}]}}

Extract at most {max_candidates} relations that help fill the detected query gap. Every relation must have an exact supporting quote from the supplied source. Do not use model knowledge and do not invent entities, relations, quotes, or source IDs."""
        system_prompt = (
            "You are a provenance-constrained knowledge graph repair extractor. "
            "Only extract relations explicitly supported by the supplied source chunks."
        )
        try:
            raw_response = await rag.llm_model_func(prompt, system_prompt=system_prompt)
            payload = _parse_json_payload(str(raw_response))
            triples = payload.get("triples", []) if isinstance(payload, dict) else payload
            for row in triples if isinstance(triples, list) else []:
                if not isinstance(row, dict):
                    continue
                candidates.append(
                    {
                        "head": row.get("head", ""),
                        "relation": row.get("relation", ""),
                        "tail": row.get("tail", ""),
                        "evidence_quote": row.get("evidence_quote", ""),
                        "source_id": row.get("source_id", ""),
                        "extraction_method": "llm_source_constrained",
                    }
                )
        except Exception as exc:
            llm_error = f"{type(exc).__name__}: {normalize_text(exc)[:180]}"

    deduped = []
    seen = set()
    for candidate in candidates:
        key = (
            normalize_label(candidate.get("head")),
            normalize_text(candidate.get("relation")).lower(),
            normalize_label(candidate.get("tail")),
            normalize_text(candidate.get("source_id")),
        )
        if not all(key[:3]) or key in seen:
            continue
        seen.add(key)
        candidate["head"] = key[0]
        candidate["relation"] = key[1]
        candidate["tail"] = key[2]
        candidate["source_id"] = key[3]
        candidate["evidence_quote"] = normalize_text(candidate.get("evidence_quote"))
        deduped.append(candidate)
    return {
        "candidates": deduped[:max_candidates],
        "llm_used": llm_used,
        "llm_error": llm_error,
        "raw_response_chars": len(str(raw_response or "")),
    }


def _quote_grounding_score(quote: str, source_text: str) -> float:
    normalized_quote = normalize_text(quote).lower()
    normalized_source = normalize_text(source_text).lower()
    if not normalized_quote:
        return 0.0
    if normalized_quote in normalized_source:
        return 1.0
    return _overlap(lexical_terms(normalized_quote), lexical_terms(normalized_source))


def validate_repair_candidates(
    candidates: list[dict[str, Any]],
    chunks: list[dict[str, Any]],
    query_graph: dict[str, Any],
    evidence_summary: dict[str, Any],
    acceptance_threshold: float = 0.58,
) -> dict[str, Any]:
    chunks_by_id = {row.get("source_id"): row for row in chunks}
    question_terms = lexical_terms(query_graph.get("question"))
    intent_terms = {
        intent: INTENT_KEYWORDS.get(intent, set())
        for intent in query_graph.get("relation_intents", [])
    }
    existing_pairs = {
        (normalize_label(row.get("src")), normalize_label(row.get("tgt")))
        for row in evidence_summary.get("relationships", [])
    }

    triple_sources: dict[tuple[str, str, str], set[str]] = {}
    for candidate in candidates:
        key = (
            normalize_label(candidate.get("head")),
            normalize_text(candidate.get("relation")).lower(),
            normalize_label(candidate.get("tail")),
        )
        triple_sources.setdefault(key, set()).add(normalize_text(candidate.get("source_id")))

    validated = []
    for candidate in candidates:
        head = normalize_label(candidate.get("head"))
        relation = normalize_text(candidate.get("relation")).lower()
        tail = normalize_label(candidate.get("tail"))
        source_id = normalize_text(candidate.get("source_id"))
        quote = normalize_text(candidate.get("evidence_quote"))
        source = chunks_by_id.get(source_id, {})
        grounding = _quote_grounding_score(quote, source.get("text", ""))
        candidate_terms = lexical_terms(" ".join([head, relation, tail, quote]))
        quote_terms = lexical_terms(quote)
        semantic = _overlap(question_terms, candidate_terms)
        intent = max(
            (_overlap(keywords, quote_terms) for keywords in intent_terms.values()),
            default=0.0,
        )
        if query_graph.get("relation_intents") == ["fact"] and quote:
            intent = max(intent, 0.5)
        head_terms = lexical_terms(head)
        tail_terms = lexical_terms(tail)
        endpoint_overlap = _jaccard(head_terms, tail_terms)
        endpoints_valid = bool(
            head
            and tail
            and head != tail
            and relation
            and tail_terms
            and len(tail_terms) <= 18
            and endpoint_overlap < 0.80
        )
        type_consistency = 1.0 if endpoints_valid else 0.0
        key = (head, relation, tail)
        multi_source = min(1.0, len(triple_sources.get(key, set())) / 2)
        already_present = (head, tail) in existing_pairs

        conflict = 0.0
        relation_terms = lexical_terms(relation)
        for row in evidence_summary.get("relationships", []):
            existing_src = normalize_label(row.get("src"))
            existing_tgt = normalize_label(row.get("tgt"))
            existing_relation_terms = lexical_terms(
                " ".join([str(row.get("keywords", "")), str(row.get("description", ""))])
            )
            if existing_src == head and existing_tgt != tail and _overlap(relation_terms, existing_relation_terms) >= 0.5:
                conflict = max(conflict, 0.7)

        confidence = (
            0.30 * grounding
            + 0.23 * semantic
            + 0.17 * intent
            + 0.15 * type_consistency
            + 0.15 * multi_source
            - 0.25 * conflict
        )
        reasons = []
        if grounding < 0.72:
            reasons.append("quote_not_fully_grounded")
        if semantic < 0.10:
            reasons.append("weak_query_relevance")
        if intent < 0.10:
            reasons.append("weak_relation_intent_match")
        if conflict:
            reasons.append("conflicts_with_existing_relation")
        if already_present:
            reasons.append("edge_already_present")
        accepted = bool(
            grounding >= 0.72
            and type_consistency >= 1.0
            and semantic >= 0.28
            and intent >= 0.10
            and confidence >= acceptance_threshold
            and not already_present
        )
        repair_id = "repair-" + hashlib.sha1(
            f"{head}|{relation}|{tail}|{source_id}".encode("utf-8")
        ).hexdigest()[:12]
        validated.append(
            {
                **candidate,
                "repair_id": repair_id,
                "head": head,
                "relation": relation,
                "tail": tail,
                "source_id": source_id,
                "confidence": round(max(0.0, min(1.0, confidence)), 3),
                "accepted": accepted,
                "score_breakdown": {
                    "source_grounding": round(grounding, 3),
                    "semantic_relevance": round(semantic, 3),
                    "intent_match": round(intent, 3),
                    "type_consistency": round(type_consistency, 3),
                    "multi_source_support": round(multi_source, 3),
                    "conflict_risk": round(conflict, 3),
                },
                "reasons": reasons or ["passed_provenance_and_structure_checks"],
            }
        )
    accepted_rows = [row for row in validated if row["accepted"]]
    return {
        "validated": validated,
        "accepted": accepted_rows,
        "accepted_count": len(accepted_rows),
        "rejected_count": len(validated) - len(accepted_rows),
        "acceptance_threshold": acceptance_threshold,
    }


def _relationship_text(row: dict[str, Any]) -> str:
    return " ".join(
        [
            str(row.get("src", "")),
            str(row.get("tgt", "")),
            str(row.get("keywords", "")),
            str(row.get("description", "")),
        ]
    )


def _path_score(
    edges: list[dict[str, Any]],
    question_terms: set[str],
    query_graph: dict[str, Any],
) -> dict[str, Any]:
    text_terms = lexical_terms(" ".join(_relationship_text(edge) for edge in edges))
    relevance = _overlap(question_terms, text_terms)
    intent_scores = []
    for intent in query_graph.get("relation_intents", []):
        if intent == "fact":
            intent_scores.append(1.0)
        else:
            intent_scores.append(_overlap(INTENT_KEYWORDS.get(intent, set()), text_terms))
    query_coverage = sum(min(1.0, score * 2) for score in intent_scores) / max(1, len(intent_scores))
    provenance = sum(
        float(edge.get("repair_confidence", 0.88) if edge.get("temporary") else 0.88)
        for edge in edges
    ) / max(1, len(edges))
    repair_risk = sum(
        1 - float(edge.get("repair_confidence", 1.0))
        for edge in edges
        if edge.get("temporary")
    ) / max(1, len(edges))
    token_cost = min(1.0, len(" ".join(_relationship_text(edge) for edge in edges)) / 1000)
    final = 0.38 * relevance + 0.27 * query_coverage + 0.22 * provenance - 0.08 * repair_risk - 0.05 * token_cost
    return {
        "relevance": round(relevance, 3),
        "query_coverage": round(query_coverage, 3),
        "provenance": round(provenance, 3),
        "repair_risk": round(repair_risk, 3),
        "token_cost": round(token_cost, 3),
        "final_score": round(final, 4),
    }


def _relationship_edge_id(edge: dict[str, Any]) -> str:
    return edge.get("repair_id") or (
        f"base:{normalize_label(edge.get('src'))}->{normalize_label(edge.get('tgt'))}"
    )


def _path_goal_hits(term_set: set[str], query_graph: dict[str, Any]) -> set[str]:
    goals: set[str] = set()
    for intent in query_graph.get("relation_intents", []):
        if intent == "fact" and term_set:
            goals.add("intent:fact")
        elif _overlap(INTENT_KEYWORDS.get(intent, set()), term_set) >= 0.08:
            goals.add(f"intent:{intent}")
    for facet in query_graph.get("required_facets", []):
        facet_terms = lexical_terms(facet)
        if facet_terms and facet_terms & term_set:
            goals.add(f"facet:{facet}")
    return goals


def _enumerate_path_candidates(
    question: str,
    query_graph: dict[str, Any],
    relationships: list[dict[str, Any]],
    entities: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    usable = relationships[:24]
    path_rows: list[list[dict[str, Any]]] = []
    for edge in usable:
        path_rows.append([edge])
    for first in usable:
        for second in usable:
            if first is second:
                continue
            if normalize_label(first.get("tgt")) == normalize_label(second.get("src")):
                path_rows.append([first, second])
            if len(path_rows) >= 100:
                break
        if len(path_rows) >= 100:
            break

    question_terms = lexical_terms(question)
    candidates: list[dict[str, Any]] = []
    seen_path_ids: set[str] = set()
    for edges in path_rows:
        nodes = [normalize_label(edges[0].get("src"))]
        nodes.extend(normalize_label(edge.get("tgt")) for edge in edges)
        edge_ids = [_relationship_edge_id(edge) for edge in edges]
        path_id = "path-" + hashlib.sha1("|".join(edge_ids).encode("utf-8")).hexdigest()[:12]
        if path_id in seen_path_ids:
            continue
        seen_path_ids.add(path_id)
        score = _path_score(edges, question_terms, query_graph)
        description = " | ".join(
            normalize_text(edge.get("description") or edge.get("keywords")) for edge in edges
        )
        term_set = lexical_terms(
            " ".join(_relationship_text(edge) for edge in edges)
        )
        source_ids = list(
            dict.fromkeys(
                source_id
                for edge in edges
                for source_id in split_source_ids(edge.get("source_ids") or edge.get("source_id"))
            )
        )
        repair_confidences = [
            float(edge.get("repair_confidence", 0.0) or 0.0)
            for edge in edges
            if edge.get("temporary")
        ]
        candidates.append(
            {
                "path_id": path_id,
                "evidence_kind": "relationship_path",
                "nodes": nodes,
                "edge_ids": edge_ids,
                "contains_repair": any(edge.get("temporary") for edge in edges),
                "repair_ids": [edge.get("repair_id") for edge in edges if edge.get("repair_id")],
                "repair_confidence": round(min(repair_confidences), 3) if repair_confidences else None,
                "source_ids": source_ids,
                "description": description,
                "score_breakdown": score,
                "estimated_tokens": max(8, (len(description) + 3) // 4),
                "goal_hits": sorted(_path_goal_hits(term_set, query_graph)),
                "term_set": term_set,
            }
        )

    for entity in (entities or [])[:24]:
        name = normalize_label(entity.get("name"))
        description = normalize_text(entity.get("description"))
        if not name or not description:
            continue
        evidence_id = f"entity:{name}"
        path_id = "path-" + hashlib.sha1(evidence_id.encode("utf-8")).hexdigest()[:12]
        if path_id in seen_path_ids:
            continue
        seen_path_ids.add(path_id)
        synthetic_edge = {
            "src": name,
            "tgt": name,
            "keywords": entity.get("type", "entity attribute"),
            "description": description,
            "temporary": entity.get("temporary", False),
            "repair_confidence": entity.get("repair_confidence") or entity.get("score"),
        }
        term_set = lexical_terms(f"{name} {entity.get('type', '')} {description}")
        score = _path_score([synthetic_edge], question_terms, query_graph)
        source_ids = split_source_ids(entity.get("source_ids") or entity.get("source_id"))
        repair_confidence = (
            float(entity.get("repair_confidence") or entity.get("score") or 0.0)
            if entity.get("temporary")
            else None
        )
        candidates.append(
            {
                "path_id": path_id,
                "evidence_kind": "entity_attribute",
                "nodes": [name],
                "edge_ids": [evidence_id],
                "contains_repair": bool(entity.get("temporary")),
                "repair_ids": [entity.get("repair_id")] if entity.get("repair_id") else [],
                "repair_confidence": round(repair_confidence, 3) if repair_confidence is not None else None,
                "source_ids": source_ids,
                "description": description,
                "score_breakdown": score,
                "estimated_tokens": max(8, (len(description) + 3) // 4),
                "goal_hits": sorted(_path_goal_hits(term_set, query_graph)),
                "term_set": term_set,
            }
        )
    return sorted(
        candidates,
        key=lambda row: row["score_breakdown"]["final_score"],
        reverse=True,
    )


def optimize_evidence_portfolio(
    question: str,
    query_graph: dict[str, Any],
    relationships: list[dict[str, Any]],
    entities: list[dict[str, Any]] | None = None,
    max_paths: int = 6,
    token_budget: int = 900,
) -> dict[str, Any]:
    """Select a diverse path portfolio under coverage, risk, and token constraints."""
    candidates = _enumerate_path_candidates(
        question,
        query_graph,
        relationships,
        entities=entities,
    )
    question_terms = lexical_terms(question)
    all_goals = {
        f"intent:{intent}" for intent in query_graph.get("relation_intents", [])
    } | {
        f"facet:{facet}" for facet in query_graph.get("required_facets", [])
    }
    base_goal_support: dict[str, set[str]] = {}
    for candidate in candidates:
        if candidate["contains_repair"]:
            continue
        for goal in candidate["goal_hits"]:
            base_goal_support.setdefault(goal, set()).update(candidate["source_ids"])

    selected: list[dict[str, Any]] = []
    covered_goals: set[str] = set()
    covered_terms: set[str] = set()
    selected_sources: set[str] = set()
    used_tokens = 0
    rejected_by_budget = 0
    selection_trace: list[dict[str, Any]] = []
    remaining = list(candidates)

    while remaining and len(selected) < max_paths:
        best: dict[str, Any] | None = None
        best_utility = float("-inf")
        best_detail: dict[str, Any] = {}
        for candidate in remaining:
            candidate_tokens = int(candidate["estimated_tokens"])
            if used_tokens + candidate_tokens > token_budget:
                continue
            redundancy = max(
                (_jaccard(candidate["term_set"], chosen["term_set"]) for chosen in selected),
                default=0.0,
            )
            candidate_goals = set(candidate["goal_hits"])
            new_goals = candidate_goals - covered_goals
            new_terms = candidate["term_set"] & question_terms - covered_terms
            new_sources = set(candidate["source_ids"]) - selected_sources
            goal_gain = len(new_goals) / max(1, len(all_goals))
            term_gain = len(new_terms) / max(1, len(question_terms))
            source_novelty = len(new_sources) / max(1, len(candidate["source_ids"]))
            repair_risk = float(candidate["score_breakdown"].get("repair_risk", 0.0))
            repair_exclusive_goals = {
                goal for goal in candidate_goals if not base_goal_support.get(goal)
            }
            repair_dependency_penalty = (
                len(repair_exclusive_goals) / max(1, len(candidate_goals))
                if candidate["contains_repair"]
                else 0.0
            )
            budget_pressure = candidate_tokens / max(1, token_budget)
            utility = (
                0.42 * float(candidate["score_breakdown"]["final_score"])
                + 0.28 * goal_gain
                + 0.12 * term_gain
                + 0.10 * source_novelty
                + (0.04 if not candidate["contains_repair"] else 0.0)
                - 0.18 * redundancy
                - 0.10 * repair_risk
                - 0.08 * repair_dependency_penalty
                - 0.05 * budget_pressure
            )
            if utility > best_utility:
                best = candidate
                best_utility = utility
                best_detail = {
                    "new_goals": sorted(new_goals),
                    "new_query_terms": sorted(new_terms),
                    "new_sources": sorted(new_sources),
                    "redundancy": round(redundancy, 3),
                    "repair_dependency_penalty": round(repair_dependency_penalty, 3),
                }
        if best is None:
            rejected_by_budget = sum(
                1 for candidate in remaining
                if used_tokens + int(candidate["estimated_tokens"]) > token_budget
            )
            break
        if selected and best_utility <= 0.01:
            break
        remaining.remove(best)
        best["marginal_utility"] = round(best_utility, 4)
        best["new_query_terms"] = best_detail["new_query_terms"]
        best["new_goals"] = best_detail["new_goals"]
        selected.append(best)
        covered_goals.update(best["goal_hits"])
        covered_terms.update(best["term_set"] & question_terms)
        selected_sources.update(best["source_ids"])
        used_tokens += int(best["estimated_tokens"])
        selection_trace.append(
            {
                "path_id": best["path_id"],
                "marginal_utility": best["marginal_utility"],
                "new_goals": best_detail["new_goals"],
                "new_sources": best_detail["new_sources"],
                "redundancy": best_detail["redundancy"],
                "contains_repair": best["contains_repair"],
            }
        )

    goal_support: dict[str, list[dict[str, Any]]] = {}
    for path in selected:
        for goal in path["goal_hits"]:
            goal_support.setdefault(goal, []).append(path)
    repair_dependent_goals = [
        goal
        for goal, supporters in goal_support.items()
        if supporters and all(path["contains_repair"] for path in supporters)
    ]
    multi_source_goals = [
        goal
        for goal, supporters in goal_support.items()
        if len({source for path in supporters for source in path["source_ids"]}) >= 2
    ]
    repair_path_count = sum(1 for path in selected if path["contains_repair"])
    pair_redundancies = [
        _jaccard(selected[left]["term_set"], selected[right]["term_set"])
        for left in range(len(selected))
        for right in range(left + 1, len(selected))
    ]
    repair_dependency_ratio = len(repair_dependent_goals) / max(1, len(goal_support))
    repair_path_share = repair_path_count / max(1, len(selected))
    mean_redundancy = sum(pair_redundancies) / max(1, len(pair_redundancies))
    if repair_dependency_ratio > 0.50 or repair_path_share > 0.60:
        risk_level = "high"
    elif repair_dependency_ratio > 0 or repair_path_count:
        risk_level = "medium"
    else:
        risk_level = "low"

    for candidate in selected:
        candidate.pop("term_set", None)
    return {
        "optimizer": "risk_aware_submodular_portfolio_v1",
        "candidate_count": len(candidates),
        "selected_count": len(selected),
        "selected_paths": selected,
        "max_paths": max_paths,
        "token_budget": token_budget,
        "estimated_tokens": used_tokens,
        "budget_utilization": round(used_tokens / max(1, token_budget), 3),
        "rejected_by_budget": rejected_by_budget,
        "goal_count": len(all_goals),
        "covered_goal_count": len(covered_goals),
        "goal_coverage_ratio": round(len(covered_goals) / max(1, len(all_goals)), 3),
        "unique_source_count": len(selected_sources),
        "multi_source_goal_ratio": round(len(multi_source_goals) / max(1, len(goal_support)), 3),
        "repair_path_share": round(repair_path_share, 3),
        "repair_dependent_goals": sorted(repair_dependent_goals),
        "repair_dependency_ratio": round(repair_dependency_ratio, 3),
        "mean_pairwise_redundancy": round(mean_redundancy, 3),
        "verification_required": bool(repair_dependent_goals),
        "risk_level": risk_level,
        "selection_trace": selection_trace,
    }


def rank_overlay_paths(
    question: str,
    query_graph: dict[str, Any],
    relationships: list[dict[str, Any]],
    max_paths: int = 6,
    entities: list[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Compatibility wrapper for callers that only need selected paths."""
    return optimize_evidence_portfolio(
        question,
        query_graph,
        relationships,
        entities=entities,
        max_paths=max_paths,
    )["selected_paths"]


def project_evidence_to_portfolio(
    evidence_summary: dict[str, Any],
    portfolio: dict[str, Any],
) -> dict[str, Any]:
    """Create a bounded evidence view containing only relationships selected by the portfolio."""
    selected_paths = portfolio.get("selected_paths", [])
    selected_edge_ids = {
        edge_id
        for path in selected_paths
        for edge_id in path.get("edge_ids", [])
    }
    if not selected_edge_ids:
        return deepcopy(evidence_summary)

    projected = deepcopy(evidence_summary)
    projected_relationships = [
        relation
        for relation in projected.get("relationships", [])
        if _relationship_edge_id(relation) in selected_edge_ids
    ]
    selected_nodes = {
        normalize_label(node)
        for path in selected_paths
        for node in path.get("nodes", [])
    }
    projected_entities = [
        entity
        for entity in projected.get("entities", [])
        if normalize_label(entity.get("name")) in selected_nodes
        or f"entity:{normalize_label(entity.get('name'))}" in selected_edge_ids
    ]
    projected["relationships"] = projected_relationships
    projected["entities"] = projected_entities or projected.get("entities", [])
    projected["paths"] = deepcopy(selected_paths)
    projected["evidence_portfolio"] = deepcopy(portfolio)
    projected["portfolio_projection"] = {
        "enabled": True,
        "relationship_count": len(projected_relationships),
        "entity_count": len(projected["entities"]),
        "base_graph_unchanged": True,
    }
    return projected


def build_query_local_overlay(
    question: str,
    query_graph: dict[str, Any],
    evidence_summary: dict[str, Any],
    validation: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    overlay_summary = deepcopy(evidence_summary)
    accepted = validation.get("accepted", [])
    existing_entities = {
        normalize_label(row.get("name")) for row in overlay_summary.get("entities", [])
    }
    for repair in accepted:
        for name in [repair["head"], repair["tail"]]:
            if name not in existing_entities:
                overlay_summary.setdefault("entities", []).append(
                    {
                        "name": name,
                        "type": "QUERY_LOCAL_ENTITY",
                        "description": "Entity introduced by a source-grounded query-local repair.",
                        "score": repair["confidence"],
                        "source": "query_local_repair",
                        "source_ids": [repair["source_id"]],
                        "temporary": True,
                    }
                )
                existing_entities.add(name)
        overlay_summary.setdefault("relationships", []).append(
            {
                "src": repair["head"],
                "tgt": repair["tail"],
                "keywords": repair["relation"],
                "description": repair["evidence_quote"],
                "score": repair["confidence"],
                "source": "query_local_repair",
                "source_ids": [repair["source_id"]],
                "source_spans": [repair["evidence_quote"]],
                "repair_id": repair["repair_id"],
                "repair_confidence": repair["confidence"],
                "temporary": True,
            }
        )
    portfolio = optimize_evidence_portfolio(
        question,
        query_graph,
        overlay_summary.get("relationships", []),
        entities=overlay_summary.get("entities", []),
    )
    paths = portfolio["selected_paths"]
    overlay_summary["paths"] = paths
    overlay_summary["evidence_portfolio"] = portfolio
    overlay = {
        "enabled": bool(accepted),
        "temporary": True,
        "base_graph_unchanged": True,
        "accepted_repairs": accepted,
        "accepted_count": len(accepted),
        "rejected_count": validation.get("rejected_count", 0),
        "path_count": len(paths),
        "selected_paths": paths,
        "path_portfolio": portfolio,
        "rolled_back_repair_ids": [],
    }
    return overlay_summary, overlay


def build_overlay_context(
    query_graph: dict[str, Any],
    overlay_summary: dict[str, Any],
    max_paths: int = 6,
) -> str:
    lines = [
        "Query goals: " + ", ".join(query_graph.get("relation_intents", [])),
        "Target type: " + str(query_graph.get("target_type", "ANSWER_ENTITY_OR_VALUE")),
        "Evidence paths:",
    ]
    for path in overlay_summary.get("paths", [])[:max_paths]:
        source_text = ", ".join(path.get("source_ids", [])) or "unknown"
        lines.append(
            f"- {' -> '.join(path.get('nodes', []))} [sources: {source_text}; repair={path.get('contains_repair')}]: "
            f"{path.get('description', '')}"
        )
    return "\n".join(lines)


async def generate_overlay_grounded_answer(
    rag: Any,
    question: str,
    query_graph: dict[str, Any],
    overlay_summary: dict[str, Any],
) -> str:
    context = build_overlay_context(query_graph, overlay_summary)
    prompt = f"""Question: {question}

Verified query-local graph context:
{context}

Answer the question using only the verified graph context. Keep the answer concise but complete. If the context does not support a requested fact, state that the evidence is insufficient. Do not use outside knowledge."""
    system_prompt = (
        "You are a provenance-grounded KG-RAG answer generator. Every factual statement must be supported "
        "by the supplied evidence paths."
    )
    response = await rag.llm_model_func(prompt, system_prompt=system_prompt)
    return normalize_text(response)


def _contains_negation(text: str) -> bool:
    terms = lexical_terms(text)
    if terms & (NEGATION_TERMS - {"no"}):
        return True
    for match in re.finditer(r"\bno\b", str(text or ""), flags=re.IGNORECASE):
        following = str(text or "")[match.end() :].lstrip()
        next_token = re.match(r"[A-Za-z][A-Za-z'-]*", following)
        if next_token and next_token.group(0)[0].isupper():
            continue
        return True
    return False


def _numbers(text: str) -> set[str]:
    return set(re.findall(r"\b\d+(?:\.\d+)?%?\b", str(text or "")))


def _claim_local_evidence_window(
    claim_terms: set[str],
    text: str,
) -> str:
    cleaned = normalize_text(text)
    sentences = [
        normalize_text(part)
        for part in re.split(r"(?<=[.!?])\s+|[\r\n]+|<SEP>", cleaned)
        if normalize_text(part)
    ]
    if len(sentences) <= 1:
        return cleaned
    candidates = list(sentences)
    candidates.extend(
        f"{sentences[index]} {sentences[index + 1]}"
        for index in range(len(sentences) - 1)
    )
    return max(
        candidates,
        key=lambda candidate: (
            _overlap(claim_terms, lexical_terms(candidate)),
            -len(candidate),
        ),
    )


def _relation_intent_label(relation: dict[str, Any]) -> str:
    terms = lexical_terms(
        " ".join(
            [
                str(relation.get("keywords", "")),
                str(relation.get("description", "")),
            ]
        )
    )
    best_intent = "fact"
    best_score = 0.0
    for intent, keywords in INTENT_KEYWORDS.items():
        score = _overlap(keywords, terms)
        if score > best_score:
            best_intent = intent
            best_score = score
    return best_intent


def _source_reliability(relation: dict[str, Any]) -> float:
    sources = split_source_ids(relation.get("source_ids") or relation.get("source_id"))
    if relation.get("temporary"):
        score = float(relation.get("repair_confidence", 0.55) or 0.55)
    else:
        score = 0.78
    if sources:
        score += 0.04
    score += min(0.12, max(0, len(sources) - 1) * 0.06)
    if relation.get("source_spans"):
        score += 0.04
    return round(min(1.0, score), 3)


def _conflict_evidence_id(relation: dict[str, Any]) -> str:
    if relation.get("repair_id"):
        return str(relation["repair_id"])
    identity = "|".join(
        [
            normalize_label(relation.get("src")),
            normalize_label(relation.get("tgt")),
            normalize_text(relation.get("description") or relation.get("keywords")),
            ",".join(
                split_source_ids(relation.get("source_ids") or relation.get("source_id"))
            ),
        ]
    )
    return "evidence-" + hashlib.sha1(identity.encode("utf-8")).hexdigest()[:16]


def build_source_conflict_graph(
    question: str,
    evidence_summary: dict[str, Any],
) -> dict[str, Any]:
    """Detect conservative cross-source numeric, polarity, and functional conflicts."""
    statements: list[dict[str, Any]] = []
    relation_rows = list(evidence_summary.get("relationships", []))
    relation_rows.extend(evidence_summary.get("conflict_quarantine", []))
    for relation in relation_rows:
        src = normalize_label(relation.get("src"))
        tgt = normalize_label(relation.get("tgt"))
        description = normalize_text(relation.get("description") or relation.get("keywords"))
        source_ids = split_source_ids(relation.get("source_ids") or relation.get("source_id"))
        if not src or not description or not source_ids:
            continue
        statements.append(
            {
                "evidence_id": _conflict_evidence_id(relation),
                "src": src,
                "tgt": tgt,
                "intent": _relation_intent_label(relation),
                "predicate_terms": sorted(lexical_terms(relation.get("keywords", ""))),
                "statement_terms": sorted(lexical_terms(description)),
                "description": description,
                "numbers": sorted(_numbers(description)),
                "negative": _contains_negation(description),
                "source_ids": source_ids,
                "source_count": len(source_ids),
                "reliability": _source_reliability(relation),
                "temporary": bool(relation.get("temporary")),
            }
        )

    conflicts: list[dict[str, Any]] = []
    functional_terms = {
        "amount",
        "capacity",
        "capital",
        "cost",
        "date",
        "founded",
        "has",
        "is",
        "located",
        "maximum",
        "minimum",
        "percentage",
        "price",
        "range",
        "rate",
        "uses",
        "year",
    }
    for left_index, left in enumerate(statements):
        for right in statements[left_index + 1 :]:
            if left["src"] != right["src"]:
                continue
            if not set(left["source_ids"]).isdisjoint(right["source_ids"]):
                continue
            left_predicate = set(left["predicate_terms"])
            right_predicate = set(right["predicate_terms"])
            predicate_similarity = _jaccard(left_predicate, right_predicate)
            statement_similarity = _jaccard(
                set(left["statement_terms"]),
                set(right["statement_terms"]),
            )
            same_target = bool(left["tgt"] and left["tgt"] == right["tgt"])
            same_intent = left["intent"] == right["intent"]
            numeric_conflict = bool(
                left["numbers"]
                and right["numbers"]
                and set(left["numbers"]).isdisjoint(right["numbers"])
                and same_intent
                and (same_target or predicate_similarity >= 0.34)
                and statement_similarity >= 0.20
            )
            polarity_conflict = bool(
                left["negative"] != right["negative"]
                and same_intent
                and (same_target or predicate_similarity >= 0.34)
                and statement_similarity >= 0.42
            )
            functional_conflict = bool(
                left["tgt"]
                and right["tgt"]
                and left["tgt"] != right["tgt"]
                and same_intent
                and predicate_similarity >= 0.55
                and bool((left_predicate | right_predicate) & functional_terms)
                and statement_similarity >= 0.30
            )
            conflict_types = [
                name
                for name, active in [
                    ("numeric", numeric_conflict),
                    ("polarity", polarity_conflict),
                    ("functional_target", functional_conflict),
                ]
                if active
            ]
            if not conflict_types:
                continue

            reliability_gap = abs(left["reliability"] - right["reliability"])
            if left["source_count"] != right["source_count"]:
                winner = left if left["source_count"] > right["source_count"] else right
                loser = right if winner is left else left
                resolution = "independent_source_majority"
            elif reliability_gap >= 0.08:
                winner = left if left["reliability"] > right["reliability"] else right
                loser = right if winner is left else left
                resolution = "provenance_reliability_margin"
            else:
                winner = None
                loser = None
                resolution = "unresolved_abstain"
            conflicts.append(
                {
                    "conflict_id": "conflict-"
                    + hashlib.sha1(
                        f"{left['evidence_id']}|{right['evidence_id']}".encode("utf-8")
                    ).hexdigest()[:12],
                    "types": conflict_types,
                    "severity": "high" if polarity_conflict or functional_conflict else "medium",
                    "left_evidence_id": left["evidence_id"],
                    "right_evidence_id": right["evidence_id"],
                    "left_sources": left["source_ids"],
                    "right_sources": right["source_ids"],
                    "left_reliability": left["reliability"],
                    "right_reliability": right["reliability"],
                    "winner_evidence_id": winner["evidence_id"] if winner else None,
                    "suppressed_evidence_ids": (
                        [loser["evidence_id"]]
                        if loser
                        else [left["evidence_id"], right["evidence_id"]]
                    ),
                    "resolution": resolution,
                    "question_relevance": round(
                        max(
                            _overlap(lexical_terms(question), set(left["statement_terms"])),
                            _overlap(lexical_terms(question), set(right["statement_terms"])),
                        ),
                        3,
                    ),
                }
            )

    suppressed_ids = sorted(
        {
            evidence_id
            for conflict in conflicts
            for evidence_id in conflict["suppressed_evidence_ids"]
        }
    )
    unresolved_count = sum(
        1 for conflict in conflicts if conflict["resolution"] == "unresolved_abstain"
    )
    high_count = sum(1 for conflict in conflicts if conflict["severity"] == "high")
    risk_score = min(
        1.0,
        0.20 * len(conflicts) + 0.25 * unresolved_count + 0.20 * high_count,
    )
    return {
        "enabled": True,
        "statement_count": len(statements),
        "conflict_count": len(conflicts),
        "unresolved_count": unresolved_count,
        "high_severity_count": high_count,
        "risk_score": round(risk_score, 3),
        "conflicts": conflicts,
        "suppressed_evidence_ids": suppressed_ids,
        "clean": not conflicts,
    }


def apply_source_conflict_arbitration(
    evidence_summary: dict[str, Any],
    conflict_graph: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    arbitrated = deepcopy(evidence_summary)
    suppressed = set(conflict_graph.get("suppressed_evidence_ids", []))
    active_relationships = []
    quarantined_relationships = []
    for relation in arbitrated.get("relationships", []):
        evidence_id = _conflict_evidence_id(relation)
        if evidence_id in suppressed:
            quarantined = deepcopy(relation)
            quarantined["conflict_status"] = "suppressed"
            quarantined["conflict_evidence_id"] = evidence_id
            quarantined_relationships.append(quarantined)
        else:
            active_relationships.append(relation)
    arbitrated["relationships"] = active_relationships
    arbitrated["conflict_quarantine"] = quarantined_relationships
    arbitration = {
        "enabled": True,
        "base_graph_unchanged": True,
        "conflict_count": conflict_graph.get("conflict_count", 0),
        "suppressed_count": len(quarantined_relationships),
        "suppressed_evidence_ids": sorted(suppressed),
        "unresolved_count": conflict_graph.get("unresolved_count", 0),
        "active_relationship_count": len(active_relationships),
        "quarantine_relationship_count": len(quarantined_relationships),
        "decision": (
            "quarantine_conflicting_evidence" if suppressed else "no_conflict_detected"
        ),
    }
    return arbitrated, arbitration


def plan_uncertainty_compute_budget(
    query_graph: dict[str, Any],
    knowledge_gap: dict[str, Any],
    conflict_graph: dict[str, Any],
    requested_fast: bool,
    contract_type: str,
) -> dict[str, Any]:
    gap_score = float(knowledge_gap.get("gap_score", 0.0) or 0.0)
    conflict_risk = float(conflict_graph.get("risk_score", 0.0) or 0.0)
    anchor_uncertainty = 1 - float(knowledge_gap.get("anchor_coverage", 1.0) or 0.0)
    structural_complexity = 1.0 if query_graph.get("requires_multi_hop") else 0.0
    uncertainty = min(
        1.0,
        0.50 * gap_score
        + 0.25 * conflict_risk
        + 0.15 * anchor_uncertainty
        + 0.10 * structural_complexity,
    )
    if uncertainty < 0.30:
        tier = "low"
        max_paths, token_budget, max_candidates = 6, 560, 2
        verification_depth = "single_pass"
    elif uncertainty < 0.60:
        tier = "medium"
        max_paths, token_budget, max_candidates = 8, 720, 3
        verification_depth = "dual_route"
    else:
        tier = "high"
        max_paths, token_budget, max_candidates = 10, 960, 4
        verification_depth = "dual_route_counterfactual"
    if contract_type in {"comprehensive", "comparison"}:
        max_paths += 2
        token_budget += 240
    return {
        "planner": "uncertainty_calibrated_compute_budget_v1",
        "uncertainty_score": round(uncertainty, 3),
        "tier": tier,
        "components": {
            "knowledge_gap": round(gap_score, 3),
            "source_conflict": round(conflict_risk, 3),
            "anchor_uncertainty": round(anchor_uncertainty, 3),
            "structural_complexity": round(structural_complexity, 3),
        },
        "max_paths": max_paths,
        "token_budget": token_budget,
        "max_answer_candidates": max_candidates,
        "verification_depth": verification_depth,
        "allow_llm_repair": bool(
            not requested_fast
            and tier in {"medium", "high"}
            and knowledge_gap.get("repair_required")
        ),
        "require_conflict_abstention": bool(conflict_graph.get("unresolved_count", 0)),
        "requested_fast": requested_fast,
    }


def finalize_uncertainty_compute_budget(
    plan: dict[str, Any],
    portfolio: dict[str, Any],
) -> dict[str, Any]:
    finalized = deepcopy(plan)
    repair_dependency = float(portfolio.get("repair_dependency_ratio", 0.0) or 0.0)
    portfolio_risk = {"low": 0.0, "medium": 0.35, "high": 0.70}.get(
        str(portfolio.get("risk_level", "low")),
        0.0,
    )
    post_score = min(
        1.0,
        float(plan.get("uncertainty_score", 0.0))
        + 0.20 * repair_dependency
        + 0.15 * portfolio_risk,
    )
    finalized["post_portfolio_uncertainty"] = round(post_score, 3)
    finalized["portfolio_repair_dependency"] = round(repair_dependency, 3)
    finalized["portfolio_risk_level"] = portfolio.get("risk_level", "low")
    finalized["budget_utilization"] = portfolio.get("budget_utilization", 0.0)
    finalized["within_token_budget"] = bool(
        int(portfolio.get("estimated_tokens", 0) or 0)
        <= int(plan.get("token_budget", 0) or 0)
    )
    finalized["escalated_after_portfolio"] = bool(
        plan.get("tier") != "high"
        and (
            post_score >= 0.60
            or repair_dependency >= 0.50
            or portfolio.get("risk_level") == "high"
        )
    )
    if finalized["escalated_after_portfolio"]:
        finalized["verification_depth"] = "dual_route_counterfactual"
    return finalized


def build_signed_claim_evidence_graph(
    claims: list[str],
    evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    claim_nodes = []
    evidence_nodes = []
    graph_edges = []
    rollback_ids: set[str] = set()

    for index, item in enumerate(evidence):
        evidence_nodes.append(
            {
                "id": f"evidence-{index}",
                "label": item.get("label", ""),
                "kind": item.get("kind", "unknown"),
                "repair_id": item.get("repair_id"),
                "repair_confidence": item.get("repair_confidence"),
                "source_ids": item.get("source_ids", []),
            }
        )

    claim_results = []
    for claim_index, claim in enumerate(claims):
        claim_id = f"claim-{claim_index}"
        claim_nodes.append({"id": claim_id, "text": claim})
        claim_terms = lexical_terms(claim)
        claim_numbers = _numbers(claim)
        claim_negative = _contains_negation(claim)
        support_edges = []
        contradiction_edges = []

        for evidence_index, item in enumerate(evidence):
            item_text = str(item.get("text", ""))
            evidence_window = _claim_local_evidence_window(
                claim_terms,
                item_text,
            )
            item_terms = lexical_terms(evidence_window)
            overlap = _overlap(claim_terms, item_terms)
            item_numbers = _numbers(evidence_window)
            polarity_conflict = (
                claim_negative != _contains_negation(evidence_window)
                and overlap >= 0.28
            )
            number_conflict = bool(
                claim_numbers
                and item_numbers
                and claim_numbers.isdisjoint(item_numbers)
                and overlap >= 0.24
            )
            relation = "contradicts" if polarity_conflict or number_conflict else "supports"
            threshold = 0.24 if relation == "contradicts" else 0.14
            if overlap < threshold:
                continue
            edge = {
                "claim_id": claim_id,
                "evidence_id": f"evidence-{evidence_index}",
                "relation": relation,
                "score": round(min(1.0, overlap), 3),
                "repair_id": item.get("repair_id"),
                "repair_confidence": item.get("repair_confidence"),
                "source_ids": item.get("source_ids", []),
                "evidence_window": evidence_window,
            }
            graph_edges.append(edge)
            if relation == "supports":
                support_edges.append(edge)
            else:
                contradiction_edges.append(edge)

        support_scores = sorted((edge["score"] for edge in support_edges), reverse=True)
        support_mass = 1.0
        for score in support_scores[:3]:
            support_mass *= 1 - score
        support_mass = 1 - support_mass
        contradiction_mass = max((edge["score"] for edge in contradiction_edges), default=0.0)
        confidence = max(0.0, min(1.0, support_mass - 0.8 * contradiction_mass))

        contributions = []
        for edge in support_edges:
            remaining_scores = [candidate["score"] for candidate in support_edges if candidate is not edge]
            remaining_mass = 1.0
            for score in sorted(remaining_scores, reverse=True)[:3]:
                remaining_mass *= 1 - score
            remaining_mass = 1 - remaining_mass
            delta = max(0.0, support_mass - remaining_mass)
            contributions.append(
                {
                    "evidence_id": edge["evidence_id"],
                    "repair_id": edge.get("repair_id"),
                    "delta": round(delta, 3),
                }
            )
            repair_id = edge.get("repair_id")
            repair_confidence = float(edge.get("repair_confidence") or 0.0)
            independent_base_support = any(not candidate.get("repair_id") for candidate in support_edges if candidate is not edge)
            if repair_id and (
                contradiction_mass >= support_mass
                or (repair_confidence < 0.72 and delta >= 0.18 and not independent_base_support)
            ):
                rollback_ids.add(repair_id)

        if contradiction_edges and support_mass < 0.55:
            status = "contradicted"
        elif support_mass >= 0.20 and confidence >= 0.16:
            status = "supported"
        else:
            status = "unknown"
        claim_results.append(
            {
                "claim_id": claim_id,
                "claim": claim,
                "status": status,
                "support_mass": round(support_mass, 3),
                "contradiction_mass": round(contradiction_mass, 3),
                "confidence": round(confidence, 3),
                "counterfactual_contributions": contributions,
            }
        )

    status_counts = {
        status: sum(1 for row in claim_results if row["status"] == status)
        for status in ["supported", "contradicted", "unknown"]
    }
    return {
        "enabled": True,
        "claim_nodes": claim_nodes,
        "evidence_nodes": evidence_nodes,
        "edges": graph_edges,
        "claims": claim_results,
        "status_counts": status_counts,
        "mean_confidence": round(
            sum(row["confidence"] for row in claim_results) / max(1, len(claim_results)),
            3,
        ),
        "rollback_repair_ids": sorted(rollback_ids),
    }


def rollback_overlay_evidence(
    evidence_summary: dict[str, Any],
    repair_ids: list[str],
) -> dict[str, Any]:
    rolled = deepcopy(evidence_summary)
    blocked = set(repair_ids)
    rolled["relationships"] = [
        row for row in rolled.get("relationships", []) if row.get("repair_id") not in blocked
    ]
    rolled["paths"] = [
        path
        for path in rolled.get("paths", [])
        if not (set(path.get("edge_ids", [])) & blocked)
    ]
    rolled["rolled_back_repair_ids"] = sorted(blocked)
    return rolled
