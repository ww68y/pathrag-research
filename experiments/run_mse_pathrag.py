import argparse
import asyncio
import base64
import hashlib
import json
import math
import os
import re
import struct
import sys
import time
from collections import Counter
from pathlib import Path

import aiohttp
import networkx as nx
import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))


SYSTEM_PROMPT = """You are an evidence-grounded QA assistant.
Use only the selected minimal evidence subgraph and supporting text snippets.
Do not add facts that are not supported by the evidence.
If the evidence is insufficient, say that the answer cannot be determined from the provided evidence.
Answer with dense, evidence-backed content. Prefer a concise answer that covers the required points over a long explanatory answer."""


USER_PROMPT = """Question:
{question}

Question requirements inferred by the retrieval module:
{requirements}

Selected minimal evidence subgraph:
{evidence_paths}

Supporting text snippets:
{supporting_chunks}

Retrieval policy:
{response_guidance}

Answer control:
{answer_control_guidance}

Answer format:
- Start with the direct answer.
- Cover the major supported points, not just one narrow detail.
- Include quantitative values, key factors, examples, or caveats only when they are central to the question.
- Do not cite path ids unless the user explicitly asks for evidence.
- Use at most {max_answer_sentences} sentences.
- Do not include long background explanations.
- Do not add background details from supplemental repair evidence unless they directly fill a missing requirement.
- Mention uncertainty only when the evidence directly requires it.

Answer:"""


STOPWORDS = {
    "the",
    "and",
    "for",
    "that",
    "this",
    "with",
    "from",
    "into",
    "about",
    "what",
    "which",
    "when",
    "where",
    "who",
    "whom",
    "whose",
    "why",
    "how",
    "does",
    "did",
    "are",
    "was",
    "were",
    "been",
    "being",
    "have",
    "has",
    "had",
    "can",
    "could",
    "would",
    "should",
    "will",
    "shall",
    "may",
    "might",
    "must",
    "not",
    "than",
    "then",
    "also",
    "their",
    "there",
    "they",
    "them",
    "its",
    "his",
    "her",
    "our",
    "your",
    "you",
    "use",
    "used",
    "using",
    "between",
    "among",
    "within",
    "without",
    "according",
    "based",
    "describe",
    "explain",
}


LOW_VALUE_REPAIR_TERMS = {
    "some",
    "method",
    "methods",
    "improve",
    "improving",
    "role",
    "discussed",
}


GENERIC_ASPECT_LABELS = {
    "book",
    "disease",
    "diseases",
    "example",
    "examples",
    "health",
    "individual",
    "individuals",
    "method",
    "methods",
    "microbiome",
    "role",
    "ways",
}


METHOD_ASPECTS = [
    {
        "label": "probiotics",
        "aliases": ("probiotic", "probiotics"),
    },
    {
        "label": "prebiotics",
        "aliases": ("prebiotic", "prebiotics"),
    },
    {
        "label": "stool or fecal microbiota transplants",
        "aliases": ("stool transplant", "stool transplants", "fecal transplant", "fecal transplants", "fecal microbiota"),
    },
    {
        "label": "targeted or limited antibiotic use",
        "aliases": ("antibiotic", "antibiotics"),
    },
    {
        "label": "dietary changes",
        "aliases": ("diet", "dietary", "solid food", "fiber", "nutrition"),
    },
]


ANSWER_CUES = {
    "person": {
        "who",
        "person",
        "people",
        "author",
        "owner",
        "manager",
        "leader",
        "responsible",
        "director",
        "investigator",
    },
    "time": {
        "when",
        "date",
        "time",
        "year",
        "month",
        "day",
        "deadline",
        "duration",
        "period",
        "notice",
    },
    "location": {
        "where",
        "location",
        "place",
        "country",
        "city",
        "region",
        "site",
    },
    "number": {
        "how many",
        "how much",
        "number",
        "amount",
        "rate",
        "percentage",
        "percent",
        "cost",
        "price",
        "score",
    },
    "condition": {
        "condition",
        "requirement",
        "clause",
        "term",
        "terminate",
        "termination",
        "eligible",
        "if",
        "unless",
        "provided",
    },
    "mechanism": {
        "how",
        "why",
        "mechanism",
        "effect",
        "influence",
        "impact",
        "cause",
        "lead",
        "regulate",
        "pathway",
    },
}


QUERY_EXPANSIONS = [
    (
        {"differ", "different", "difference", "differences", "variation", "vary", "varies", "diversity"},
        {
            "different",
            "differences",
            "variation",
            "vary",
            "varies",
            "diversity",
            "diverse",
            "genetic",
            "genes",
            "diet",
            "environment",
            "lifestyle",
            "species",
            "shared",
            "percent",
        },
    ),
    (
        {"individual", "individuals", "person", "people", "human", "humans"},
        {"individual", "individuals", "person", "people", "human", "humans", "population"},
    ),
    (
        {"share", "shared", "same", "similar", "identical"},
        {"share", "shared", "same", "similar", "identical", "percent", "percentage"},
    ),
    (
        {"condition", "conditions", "terminate", "termination", "clause", "contract"},
        {"condition", "conditions", "terminate", "termination", "clause", "contract", "notice", "breach"},
    ),
    (
        {"antibiotic", "antibiotics"},
        {
            "antibiotic",
            "antibiotics",
            "broad-spectrum",
            "disrupt",
            "damage",
            "infection",
            "infectious",
            "inflammatory",
            "bowel",
            "disease",
            "chronic",
            "immune",
            "bifidobacterium",
            "infant",
        },
    ),
    (
        {"hygiene", "hypothesis", "allergy", "allergies", "asthma"},
        {
            "hygiene",
            "hypothesis",
            "allergy",
            "allergies",
            "asthma",
            "autoimmune",
            "immune",
            "childhood",
            "microbial",
            "diversity",
            "cleanliness",
            "exposure",
        },
    ),
    (
        {"obesity", "inflammatory", "bowel", "disease", "ibd"},
        {
            "obesity",
            "inflammatory",
            "bowel",
            "disease",
            "ibd",
            "immune",
            "metabolism",
            "microbial",
            "association",
            "causation",
            "predictive",
            "diabetes",
            "biomarker",
        },
    ),
    (
        {"ebola", "spread", "transmission", "animals", "humans"},
        {
            "ebola",
            "spread",
            "transmission",
            "animal",
            "animals",
            "human",
            "humans",
            "contact",
            "direct",
            "blood",
            "secretions",
            "organs",
            "bodily",
            "fluids",
            "bats",
            "primates",
            "wildlife",
            "reservoir",
        },
    ),
]


def clean_text(value: object, max_chars: int | None = None) -> str:
    text = "" if value is None else str(value)
    text = text.replace("<SEP>", "; ")
    text = text.strip()
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        text = text[1:-1]
    text = re.sub(r"\s+", " ", text).strip()
    if max_chars and len(text) > max_chars:
        return text[: max_chars - 3].rstrip() + "..."
    return text


def extract_terms(text: str) -> set[str]:
    text = clean_text(text).lower()
    tokens = re.findall(r"[a-z][a-z0-9_-]{2,}|[\u4e00-\u9fff]{2,}", text)
    return {token for token in tokens if token not in STOPWORDS}


def infer_requirements(question: str) -> dict:
    lowered = clean_text(question).lower()
    terms = extract_terms(question)
    expanded_terms = set(terms)
    for triggers, additions in QUERY_EXPANSIONS:
        if terms & triggers or any(trigger in lowered for trigger in triggers):
            expanded_terms.update(additions)
    cue_hits = []
    cue_terms = set()
    for label, cues in ANSWER_CUES.items():
        matched = {cue for cue in cues if cue in lowered}
        if matched:
            cue_hits.append(label)
            cue_terms.update(extract_terms(" ".join(matched)))
    if not cue_hits:
        cue_hits = ["general"]
    return {
        "answer_types": cue_hits,
        "query_terms": sorted(terms),
        "expanded_terms": sorted(expanded_terms),
        "cue_terms": sorted(cue_terms),
    }


def format_requirements(requirements: dict) -> str:
    terms = ", ".join(requirements.get("query_terms", [])[:16]) or "none"
    cues = ", ".join(requirements.get("answer_types", [])) or "general"
    return f"- answer types: {cues}\n- key query terms: {terms}"


def evidence_policy(question: str, requirements: dict, args: argparse.Namespace) -> dict:
    lowered = clean_text(question).lower()
    terms = set(requirements.get("expanded_terms") or requirements.get("query_terms", []))
    answer_types = set(requirements.get("answer_types", []))
    broad_terms = {
        "how",
        "why",
        "role",
        "effect",
        "effects",
        "affect",
        "affecting",
        "influence",
        "impact",
        "mechanism",
        "differ",
        "different",
        "difference",
        "relate",
        "relationship",
        "play",
    }
    definition_terms = {"what", "aim", "achieve", "project", "hypothesis", "explain", "describe"}
    exact_answer_types = {"person", "time", "location", "number"}

    if args.fixed_evidence_budget:
        mode = "fixed"
    elif "mechanism" in answer_types or terms & broad_terms or lowered.startswith(("how ", "why ")):
        mode = "comprehensive"
    elif terms & definition_terms:
        mode = "balanced"
    elif answer_types & exact_answer_types:
        mode = "concise"
    else:
        mode = "balanced"

    if mode == "comprehensive":
        min_edges = max(args.min_edges, 6)
        max_edges = max(args.max_edges, 10)
        min_coverage = max(args.min_coverage, 0.70)
        top_chunks = max(args.top_chunks, 5)
        if args.answer_style == "compact":
            response_guidance = (
                "Use compact evidence coverage. Answer in 3-4 information-dense sentences, "
                "covering mechanisms, causes, effects, and only the most important caveat supported by evidence."
            )
            max_answer_sentences = 4
        else:
            response_guidance = (
                "Use coverage-preserving evidence. Answer in 4-6 complete but non-redundant sentences, "
                "covering mechanisms, causes, effects, examples, and important caveats supported by evidence."
            )
            max_answer_sentences = 6
    elif mode == "balanced":
        min_edges = max(args.min_edges, 4)
        max_edges = max(args.max_edges, 8)
        min_coverage = max(args.min_coverage, 0.60)
        top_chunks = max(args.top_chunks, 4)
        if args.answer_style == "compact":
            response_guidance = (
                "Use enough evidence for a concise answer. Answer in 2-4 concise sentences, "
                "covering definition, purpose, and key details if supported."
            )
            max_answer_sentences = 4
        else:
            response_guidance = (
                "Use enough evidence for a complete answer. Answer in 3-5 concise but complete sentences, "
                "covering definition, purpose, and key details if supported."
            )
            max_answer_sentences = 5
    elif mode == "concise":
        min_edges = min(args.min_edges, 3)
        max_edges = min(args.max_edges, 6)
        min_coverage = max(0.50, min(args.min_coverage, 0.60))
        top_chunks = max(3, min(args.top_chunks, 4))
        response_guidance = (
            "Use concise evidence for exact lookup. Answer in 1-3 sentences and avoid unrelated background."
        )
        max_answer_sentences = 3
    else:
        min_edges = args.min_edges
        max_edges = args.max_edges
        min_coverage = args.min_coverage
        top_chunks = args.top_chunks
        response_guidance = "Use the configured fixed evidence budget."
        max_answer_sentences = 4

    return {
        "mode": mode,
        "min_edges": min_edges,
        "max_edges": max_edges,
        "min_coverage": min_coverage,
        "top_chunks": top_chunks,
        "response_guidance": response_guidance,
        "max_answer_sentences": max_answer_sentences,
    }


def parse_line_nos(value: str | None, total: int) -> set[int]:
    if not value:
        return set(range(total))
    line_nos: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            line_nos.update(range(int(start), int(end) + 1))
        else:
            line_nos.add(int(part))
    return line_nos


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            if not line.strip():
                continue
            item = json.loads(line)
            item.setdefault("_line_no", line_no)
            rows.append(item)
    return rows


def load_completed_line_nos(path: Path) -> set[int]:
    if not path.exists():
        return set()
    completed = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                completed.add(int(json.loads(line)["line_no"]))
            except Exception:
                continue
    return completed


def context_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def find_graph_dir(sample: dict) -> Path:
    context = sample.get("context", "")
    h = context_hash(context)
    line_no = int(sample["_line_no"])
    exact = ROOT_DIR / "data_repro" / f"ctx_{line_no}_{h}"
    if (exact / "graph_chunk_entity_relation.graphml").exists():
        return exact

    candidates = sorted((ROOT_DIR / "data_repro").glob(f"ctx_*_{h}"))
    for candidate in candidates:
        if (candidate / "graph_chunk_entity_relation.graphml").exists():
            return candidate
    raise FileNotFoundError(
        f"No PathRAG graph cache found for line {line_no}, context hash {h}. "
        "Run the original PathRAG reproduction for this sample first."
    )


def load_chunks(graph_dir: Path) -> dict[str, dict]:
    path = graph_dir / "kv_store_text_chunks.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "data" in data and isinstance(data["data"], dict):
        return data["data"]
    if isinstance(data, dict):
        return data
    raise ValueError(f"Unexpected chunk store format: {path}")


def split_source_ids(value: object) -> list[str]:
    if value is None:
        return []
    text = str(value).strip()
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        text = text[1:-1]
    return [clean_text(part) for part in text.split("<SEP>") if clean_text(part)]


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def edge_to_record(graph: nx.Graph, source: str, target: str, data: dict) -> dict:
    source_text = clean_text(source)
    target_text = clean_text(target)
    keywords = clean_text(data.get("keywords"), 240)
    description = clean_text(data.get("description"), 700)
    text = (
        f"{source_text} -> {target_text}. "
        f"Relation keywords: {keywords}. "
        f"Evidence: {description}"
    )
    source_node = graph.nodes[source] if source in graph.nodes else {}
    target_node = graph.nodes[target] if target in graph.nodes else {}
    node_source_ids = split_source_ids(source_node.get("source_id")) + split_source_ids(
        target_node.get("source_id")
    )
    node_text = (
        f"{source_text}: {clean_text(source_node.get('description'), 260)} "
        f"{target_text}: {clean_text(target_node.get('description'), 260)}"
    )
    return {
        "source": source_text,
        "target": target_text,
        "keywords": keywords,
        "description": description,
        "weight": safe_float(data.get("weight"), 1.0),
        "source_ids": split_source_ids(data.get("source_id")),
        "node_source_ids": node_source_ids,
        "text": text,
        "node_text": node_text,
        "terms": extract_terms(text + " " + node_text),
    }


def lexical_edge_score(
    record: dict,
    requirements: dict,
    question: str,
    args: argparse.Namespace,
) -> float:
    query_terms = set(requirements.get("expanded_terms") or requirements.get("query_terms", []))
    cue_terms = set(requirements.get("cue_terms", []))
    edge_terms = record["terms"]
    question_l = clean_text(question).lower()
    entity_hit = 0.0
    for entity in (record["source"], record["target"]):
        entity_l = clean_text(entity).lower().strip('"')
        entity_tokens = extract_terms(entity_l)
        if entity_l and entity_l in question_l:
            entity_hit = max(entity_hit, 1.0)
        elif entity_tokens and entity_tokens & query_terms:
            entity_hit = max(entity_hit, 0.65)

    overlap = len(edge_terms & query_terms) / max(1, len(query_terms))
    cue_overlap = len(edge_terms & cue_terms) / max(1, len(cue_terms)) if cue_terms else 0.0
    keyword_terms = extract_terms(record.get("keywords", ""))
    keyword_overlap = len(keyword_terms & query_terms) / max(1, len(query_terms))
    weight_score = min(1.0, math.log1p(max(record.get("weight", 0.0), 0.0)) / math.log(10))
    length_penalty = min(0.25, len(record["text"]) / 5000)
    relation_terms = extract_terms(record.get("keywords", "") + " " + record.get("description", ""))
    endpoint_terms = extract_terms(record.get("source", "") + " " + record.get("target", ""))
    relation_need_terms = query_terms | cue_terms
    relation_support = len(relation_terms & relation_need_terms) / max(1, len(relation_need_terms))
    endpoint_anchor = bool(endpoint_terms & query_terms)
    counterfactual_penalty = 0.0
    if not args.disable_counterfactual_verification and endpoint_anchor and relation_support < 0.08:
        # Similar entity but weak relation evidence: likely a distracting path.
        counterfactual_penalty = 0.10
    record["relation_support"] = round(relation_support, 6)
    record["counterfactual_penalty"] = round(counterfactual_penalty, 6)
    score = (
        0.42 * overlap
        + 0.20 * cue_overlap
        + 0.18 * entity_hit
        + 0.12 * keyword_overlap
        + 0.08 * weight_score
        - length_penalty
        - counterfactual_penalty
    )
    return max(0.0, score)


def normalize_scores(values: list[float]) -> list[float]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if abs(hi - lo) < 1e-9:
        return [0.5 for _ in values]
    return [(value - lo) / (hi - lo) for value in values]


async def embed_texts(
    texts: list[str],
    api_key: str,
    base_url: str,
    model: str,
    max_chars: int,
    batch_size: int,
) -> np.ndarray:
    if api_key and not api_key.startswith("Bearer "):
        api_key = "Bearer " + api_key
    headers = {"Authorization": api_key, "Content-Type": "application/json"}
    outputs: list[np.ndarray] = []
    async with aiohttp.ClientSession() as session:
        for start in range(0, len(texts), batch_size):
            batch = [text[:max_chars] for text in texts[start : start + batch_size]]
            payload = {"model": model, "input": batch, "encoding_format": "base64"}
            for attempt in range(3):
                async with session.post(base_url, headers=headers, json=payload) as response:
                    content = await response.json()
                if "code" not in content:
                    break
                if attempt == 2:
                    raise RuntimeError(f"Embedding API error: {content}")
                await asyncio.sleep(3 * (attempt + 1))
            for item in content["data"]:
                raw = base64.b64decode(item["embedding"])
                n = len(raw) // 4
                outputs.append(np.array(struct.unpack("<" + "f" * n, raw), dtype=np.float32))
    return np.vstack(outputs)


async def maybe_apply_embedding_scores(
    candidates: list[dict],
    question: str,
    args: argparse.Namespace,
    api_key: str,
    embedding_base_url: str,
    embedding_model: str,
) -> None:
    if not args.use_edge_embeddings or not candidates:
        for candidate in candidates:
            candidate["embedding_score"] = 0.0
            candidate["final_score"] = candidate["lexical_score"]
        return

    texts = [question] + [candidate["text"] + " " + candidate["node_text"] for candidate in candidates]
    vectors = await embed_texts(
        texts,
        api_key,
        embedding_base_url,
        embedding_model,
        args.embedding_max_chars,
        args.embedding_batch_size,
    )
    query_vector = vectors[0]
    edge_vectors = vectors[1:]
    query_norm = np.linalg.norm(query_vector)
    edge_norms = np.linalg.norm(edge_vectors, axis=1)
    sims = edge_vectors @ query_vector / (edge_norms * query_norm + 1e-12)
    sim_scores = normalize_scores([float(value) for value in sims])
    for candidate, sim_score in zip(candidates, sim_scores):
        candidate["embedding_score"] = round(sim_score, 6)
        candidate["final_score"] = 0.58 * candidate["lexical_score"] + 0.42 * sim_score


def jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def selection_stats(
    selected_edges: list[dict],
    candidates: list[dict],
    requirements: dict,
    policy: dict,
) -> dict:
    query_terms = set(requirements.get("expanded_terms") or requirements.get("query_terms", []))
    covered: set[str] = set()
    for edge in selected_edges:
        covered.update(set(edge.get("terms", [])) & query_terms)
    coverage = len(covered) / max(1, len(query_terms))
    return {
        "evidence_mode": policy["mode"],
        "candidate_edges": len(candidates),
        "selected_edges": len(selected_edges),
        "target_min_edges": int(policy["min_edges"]),
        "target_max_edges": int(policy["max_edges"]),
        "target_min_coverage": float(policy["min_coverage"]),
        "term_coverage": round(coverage, 4),
        "covered_terms": sorted(covered),
        "avg_selected_score": round(
            sum(item["final_score"] for item in selected_edges) / len(selected_edges), 6
        )
        if selected_edges
        else 0.0,
    }


def select_minimal_subgraph(
    candidates: list[dict],
    requirements: dict,
    policy: dict,
    args: argparse.Namespace,
) -> tuple[list[dict], dict]:
    query_terms = set(requirements.get("expanded_terms") or requirements.get("query_terms", []))
    selected: list[dict] = []
    covered: set[str] = set()
    used_pairs: set[tuple[str, str]] = set()
    min_edges = int(policy["min_edges"])
    max_edges = int(policy["max_edges"])
    min_coverage = float(policy["min_coverage"])

    ranked = sorted(candidates, key=lambda item: item["final_score"], reverse=True)
    for candidate in ranked:
        pair = tuple(sorted((candidate["source"], candidate["target"])))
        if pair in used_pairs:
            continue
        candidate_terms = set(candidate["terms"])
        max_redundancy = max((jaccard(candidate_terms, set(item["terms"])) for item in selected), default=0.0)
        new_terms = candidate_terms & query_terms - covered
        if selected and max_redundancy > args.max_redundancy and not new_terms:
            continue

        selected.append(candidate)
        used_pairs.add(pair)
        covered.update(candidate_terms & query_terms)

        coverage = len(covered) / max(1, len(query_terms))
        if len(selected) >= min_edges and coverage >= min_coverage:
            break
        if len(selected) >= max_edges:
            break

    if len(selected) < min_edges:
        for candidate in ranked:
            if candidate in selected:
                continue
            selected.append(candidate)
            if len(selected) >= min(min_edges, max_edges):
                break

    return selected, selection_stats(selected, candidates, requirements, policy)


def build_edge_candidates(
    graph: nx.Graph,
    question: str,
    requirements: dict,
    args: argparse.Namespace,
    limit: int | None = None,
) -> list[dict]:
    records = []
    for source, target, data in graph.edges(data=True):
        record = edge_to_record(graph, source, target, data)
        record["lexical_score"] = lexical_edge_score(record, requirements, question, args)
        records.append(record)

    records = sorted(
        records,
        key=lambda item: (item["lexical_score"], item.get("weight", 0.0)),
        reverse=True,
    )
    candidate_limit = args.candidate_edges if limit is None else limit
    return records[:candidate_limit]


def evidence_sufficiency_check(
    selected_edges: list[dict],
    supporting_chunks: list[dict],
    requirements: dict,
    policy: dict,
    mse_stats: dict,
    args: argparse.Namespace,
) -> dict:
    query_terms = set(requirements.get("expanded_terms") or requirements.get("query_terms", []))
    covered_terms = set(mse_stats.get("covered_terms", []))
    missing_terms = query_terms - covered_terms
    term_coverage = float(mse_stats.get("term_coverage", 0.0))
    target_coverage = max(float(policy.get("min_coverage", 0.0)), float(args.repair_target_coverage))
    if policy.get("mode") == "concise":
        target_coverage = min(target_coverage, 0.62)

    avg_relation_support = (
        sum(safe_float(edge.get("relation_support")) for edge in selected_edges) / len(selected_edges)
        if selected_edges
        else 0.0
    )
    avg_counterfactual_penalty = (
        sum(safe_float(edge.get("counterfactual_penalty")) for edge in selected_edges) / len(selected_edges)
        if selected_edges
        else 0.0
    )
    chunk_terms: set[str] = set()
    for chunk in supporting_chunks:
        chunk_terms.update(extract_terms(chunk.get("content", "")))
    chunk_coverage = len(chunk_terms & query_terms) / max(1, len(query_terms))

    relation_score = min(1.0, avg_relation_support / max(0.08, args.repair_min_relation_support * 2))
    penalty_score = max(0.0, 1.0 - avg_counterfactual_penalty / 0.12)
    density_score = min(1.0, len(selected_edges) / max(1, int(policy.get("min_edges", 1))))
    sufficiency_score = (
        0.45 * term_coverage
        + 0.22 * relation_score
        + 0.18 * chunk_coverage
        + 0.10 * density_score
        + 0.05 * penalty_score
    )

    reasons = []
    if term_coverage < target_coverage:
        reasons.append("low_term_coverage")
    if avg_relation_support < args.repair_min_relation_support:
        reasons.append("weak_relation_support")
    if selected_edges and avg_counterfactual_penalty > args.repair_max_counterfactual_penalty:
        reasons.append("high_counterfactual_penalty")
    if len(supporting_chunks) < max(1, min(3, int(policy.get("top_chunks", 1)))):
        reasons.append("few_supporting_chunks")
    if sufficiency_score < args.sufficiency_threshold:
        reasons.append("low_sufficiency_score")

    return {
        "enabled": bool(args.enable_sufficiency_repair),
        "status": "repair_needed" if reasons else "sufficient",
        "score": round(sufficiency_score, 4),
        "threshold": args.sufficiency_threshold,
        "target_coverage": round(target_coverage, 4),
        "term_coverage": round(term_coverage, 4),
        "chunk_coverage": round(chunk_coverage, 4),
        "avg_relation_support": round(avg_relation_support, 6),
        "avg_counterfactual_penalty": round(avg_counterfactual_penalty, 6),
        "selected_edges": len(selected_edges),
        "covered_terms": sorted(covered_terms),
        "missing_terms": sorted(missing_terms),
        "reasons": reasons,
    }


def repair_insufficient_evidence(
    candidates: list[dict],
    selected_edges: list[dict],
    requirements: dict,
    policy: dict,
    before_stats: dict,
    before_check: dict,
    args: argparse.Namespace,
) -> tuple[list[dict], dict, dict]:
    query_terms = set(requirements.get("expanded_terms") or requirements.get("query_terms", []))
    selected = list(selected_edges)
    selected_pairs = {edge_pair(edge) for edge in selected}
    covered_terms = set(before_stats.get("covered_terms", []))
    target_coverage = float(before_check.get("target_coverage", policy.get("min_coverage", 0.0)))
    max_edges = max(int(policy["max_edges"]), len(selected)) + max(0, int(args.repair_extra_edges))
    needs_coverage_repair = "low_term_coverage" in before_check.get("reasons", [])
    actions = []

    ranked_repair_candidates = []
    for candidate in candidates:
        pair = edge_pair(candidate)
        if pair in selected_pairs:
            continue
        candidate_terms = set(candidate.get("terms", []))
        matched_terms = candidate_terms & query_terms
        new_terms = matched_terms - covered_terms
        meaningful_new_terms = {term for term in new_terms if term not in LOW_VALUE_REPAIR_TERMS}
        relation_support = safe_float(candidate.get("relation_support"))
        counterfactual_penalty = safe_float(candidate.get("counterfactual_penalty"))
        if needs_coverage_repair and not meaningful_new_terms:
            continue
        if not new_terms and relation_support < args.repair_min_relation_support:
            continue
        if counterfactual_penalty > args.repair_max_counterfactual_penalty and not new_terms:
            continue
        redundancy = max((jaccard(candidate_terms, set(edge.get("terms", []))) for edge in selected), default=0.0)
        if redundancy > args.max_redundancy and not new_terms:
            continue
        repair_score = (
            safe_float(candidate.get("final_score"))
            + 0.16 * len(new_terms) / max(1, len(query_terms))
            + 0.08 * relation_support
            - 0.18 * counterfactual_penalty
            - 0.06 * redundancy
        )
        ranked_repair_candidates.append((repair_score, candidate, sorted(new_terms), redundancy))

    ranked_repair_candidates.sort(key=lambda item: item[0], reverse=True)
    for repair_score, candidate, new_terms, redundancy in ranked_repair_candidates:
        if len(selected) >= max_edges:
            break
        selected.append(candidate)
        selected_pairs.add(edge_pair(candidate))
        covered_terms.update(set(candidate.get("terms", [])) & query_terms)
        actions.append(
            {
                "action": "add_edge",
                "source": candidate["source"],
                "target": candidate["target"],
                "path": f"{candidate['source']} -> {candidate['target']}",
                "new_terms": new_terms,
                "repair_score": round(repair_score, 6),
                "final_score": round(safe_float(candidate.get("final_score")), 6),
                "relation_support": round(safe_float(candidate.get("relation_support")), 6),
                "counterfactual_penalty": round(safe_float(candidate.get("counterfactual_penalty")), 6),
                "redundancy": round(redundancy, 6),
            }
        )
        coverage = len(covered_terms) / max(1, len(query_terms))
        if coverage >= target_coverage and len(selected) >= int(policy["min_edges"]):
            break

    after_stats = selection_stats(selected, candidates, requirements, policy)
    repair_report = {
        "enabled": True,
        "triggered": True,
        "candidate_pool_edges": len(candidates),
        "before": before_check,
        "actions": actions,
        "added_edges": len(actions),
        "coverage_before": before_stats.get("term_coverage", 0.0),
        "coverage_after": after_stats.get("term_coverage", 0.0),
        "selected_edges_before": before_stats.get("selected_edges", 0),
        "selected_edges_after": after_stats.get("selected_edges", 0),
    }
    return selected, after_stats, repair_report


def select_supporting_chunks(
    selected_edges: list[dict],
    chunks: dict[str, dict],
    question: str,
    policy: dict,
    args: argparse.Namespace,
) -> list[dict]:
    question_terms = set()
    requirements = infer_requirements(question)
    question_terms.update(requirements.get("expanded_terms", []))
    source_counter: Counter[str] = Counter()
    for edge in selected_edges:
        source_counter.update(edge.get("source_ids", []))
        source_counter.update(edge.get("node_source_ids", []))

    ranked_by_id: dict[str, tuple[float, int, int, str, str]] = {}

    def add_candidate(chunk_id: str, count: int) -> None:
        chunk = chunks.get(chunk_id)
        if not chunk:
            return
        full_content = clean_text(chunk.get("content"))
        overlap = len(extract_terms(full_content) & question_terms)
        phrase_bonus = 0
        lowered = full_content.lower()
        for term in question_terms:
            if len(term) >= 4 and term in lowered:
                phrase_bonus += 1
        source_bonus = math.log1p(count) if count else 0.0
        score = 3.0 * overlap + 0.6 * phrase_bonus + 0.7 * source_bonus
        if score <= 0:
            return
        snippet = make_snippet(full_content, question_terms, args.max_chunk_chars)
        current = ranked_by_id.get(chunk_id)
        if current is None or score > current[0]:
            ranked_by_id[chunk_id] = (score, count, overlap, chunk_id, snippet)

    for chunk_id, count in source_counter.most_common():
        add_candidate(chunk_id, count)
    for chunk_id in chunks:
        add_candidate(chunk_id, source_counter.get(chunk_id, 0))

    ranked = sorted(ranked_by_id.values(), key=lambda item: item[0], reverse=True)
    return [
        {"chunk_id": chunk_id, "content": content, "source_count": count, "term_overlap": overlap}
        for score, count, overlap, chunk_id, content in ranked[: int(policy["top_chunks"])]
    ]


def make_snippet(content: str, terms: set[str], max_chars: int) -> str:
    if len(content) <= max_chars:
        return content
    lowered = content.lower()
    candidate_positions = []
    for term in terms:
        if len(term) < 4:
            continue
        start = 0
        while True:
            pos = lowered.find(term.lower(), start)
            if pos < 0:
                break
            candidate_positions.append(pos)
            start = pos + len(term)
            if len(candidate_positions) >= 120:
                break
    if not candidate_positions:
        return content[: max_chars - 3].rstrip() + "..."

    best_start = 0
    best_score = -1
    half = max_chars // 2
    for pos in candidate_positions:
        start = max(0, min(pos - half, len(content) - max_chars))
        window = lowered[start : start + max_chars]
        score = sum(1 for term in terms if len(term) >= 4 and term.lower() in window)
        if score > best_score:
            best_score = score
            best_start = start
    snippet = content[best_start : best_start + max_chars].strip()
    prefix = "..." if best_start > 0 else ""
    suffix = "..." if best_start + max_chars < len(content) else ""
    return prefix + snippet + suffix


def mark_evidence_roles(selected_edges: list[dict], primary_pairs: set[tuple[str, str]], repair_report: dict) -> None:
    repair_terms_by_pair = {}
    for action in repair_report.get("actions", []):
        source = clean_text(action.get("source"))
        target = clean_text(action.get("target"))
        if source and target:
            repair_terms_by_pair[tuple(sorted((source, target)))] = list(action.get("new_terms", []))

    for edge in selected_edges:
        pair = edge_pair(edge)
        if pair in primary_pairs:
            edge["evidence_role"] = "primary"
            edge["repair_new_terms"] = []
        else:
            edge["evidence_role"] = "repaired"
            edge["repair_new_terms"] = repair_terms_by_pair.get(pair, [])


def edge_search_text(edge: dict, include_node_text: bool = True) -> str:
    parts = [
        edge.get("source", ""),
        edge.get("target", ""),
        edge.get("keywords", ""),
        edge.get("description", ""),
        edge.get("text", ""),
    ]
    if include_node_text:
        parts.append(edge.get("node_text", ""))
    return clean_text(" ".join(parts)).lower()


def normalize_aspect_label(value: str) -> str:
    label = clean_text(value).lower()
    label = re.sub(r"^(?:the|a|an|these|those|such|other)\s+", "", label)
    label = re.sub(r"\b(?:etc|and so on)$", "", label).strip(" ,;:.?")
    label = clean_text(label)
    return label


def split_aspect_items(value: str) -> list[str]:
    value = clean_text(value)
    value = re.sub(r"\b(?:and|or)\s+other\s+\w+\b", "", value, flags=re.IGNORECASE)
    parts = re.split(r",|;|\band\b|\bor\b", value, flags=re.IGNORECASE)
    labels = []
    for part in parts:
        label = normalize_aspect_label(part)
        if not label:
            continue
        words = label.split()
        if len(words) > 7:
            continue
        if label in GENERIC_ASPECT_LABELS:
            continue
        if not (extract_terms(label) - GENERIC_ASPECT_LABELS):
            continue
        labels.append(label)
    return labels


def add_answer_aspect(aspects: list[dict], label: str, source: str, aliases: tuple[str, ...] = ()) -> None:
    label = normalize_aspect_label(label)
    if not label or label in GENERIC_ASPECT_LABELS:
        return
    existing = {item["label"] for item in aspects}
    if label in existing:
        return
    terms = sorted(extract_terms(label))
    aspects.append(
        {
            "label": label,
            "source": source,
            "terms": terms,
            "aliases": sorted(set(clean_text(alias).lower() for alias in aliases if clean_text(alias))),
        }
    )


def question_has_method_intent(question: str) -> bool:
    lowered = clean_text(question).lower()
    return bool(
        re.search(
            r"\b(method|methods|way|ways|approach|approaches|strategy|strategies|improv|improving|treat|treatments?|therapy|therapies|restore|restoring)\b",
            lowered,
        )
    )


def infer_answer_aspects(
    question: str,
    selected_edges: list[dict],
    supporting_chunks: list[dict],
    candidates: list[dict] | None = None,
    max_aspects: int = 8,
) -> list[dict]:
    aspects: list[dict] = []
    lowered = clean_text(question).lower()

    explicit_patterns = [
        r"\b(?:like|such as|including|include|includes)\s+([^?.;:]+)",
        r"\bbetween\s+([^?.;:]+)",
        r"\b(?:both|either)\s+([^?.;:]+)",
    ]
    for pattern in explicit_patterns:
        for match in re.finditer(pattern, lowered):
            for label in split_aspect_items(match.group(1)):
                add_answer_aspect(aspects, label, "explicit_question")

    if question_has_method_intent(question):
        search_parts = [edge_search_text(edge, include_node_text=False) for edge in selected_edges]
        if candidates:
            search_parts.extend(edge_search_text(edge, include_node_text=False) for edge in candidates[:180])
        search_parts.extend(clean_text(chunk.get("content", "")).lower() for chunk in supporting_chunks)
        corpus = "\n".join(search_parts)
        for method in METHOD_ASPECTS:
            if any(alias in corpus for alias in method["aliases"]):
                add_answer_aspect(aspects, method["label"], "method_candidate", tuple(method["aliases"]))

    return aspects[:max_aspects]


def aspect_matches_edge(aspect: dict, edge: dict) -> bool:
    text = edge_search_text(edge, include_node_text=False)
    label = clean_text(aspect.get("label", "")).lower()
    aliases = [clean_text(alias).lower() for alias in aspect.get("aliases", []) if clean_text(alias)]
    if label and label in text:
        return True
    if any(alias in text for alias in aliases):
        return True

    terms = set(aspect.get("terms", []))
    if not terms:
        return False
    edge_terms = extract_terms(text)
    overlap = terms & edge_terms
    if len(terms) == 1:
        return bool(overlap)
    return len(overlap) / max(1, len(terms)) >= 0.67


def aspect_candidate_score(aspect: dict, edge: dict, requirements: dict | None = None) -> float:
    text = edge_search_text(edge, include_node_text=False)
    endpoint_text = clean_text(f"{edge.get('source', '')} {edge.get('target', '')}").lower()
    label = clean_text(aspect.get("label", "")).lower()
    aliases = [clean_text(alias).lower() for alias in aspect.get("aliases", []) if clean_text(alias)]
    score = safe_float(edge.get("final_score")) + 0.04 * safe_float(edge.get("relation_support"))

    if label and label in endpoint_text:
        score += 0.45
    elif label and label in text:
        score += 0.20

    for alias in aliases:
        alias_weight = 0.08 if alias in {"nutrition", "fiber"} else 0.16
        if alias in endpoint_text:
            score += 0.38 + alias_weight
        elif alias in text:
            score += alias_weight

    terms = set(aspect.get("terms", []))
    direct_terms = extract_terms(text)
    score += 0.05 * len(terms & direct_terms)
    query_terms = set((requirements or {}).get("expanded_terms") or (requirements or {}).get("query_terms", []))
    central_anchor_terms = {
        "microbiome",
        "microbiota",
        "microbial",
        "microbe",
        "microbes",
        "disease",
        "diseases",
        "health",
    }
    anchor_hits = query_terms & central_anchor_terms
    if anchor_hits:
        endpoint_terms = extract_terms(endpoint_text)
        direct_anchor_hits = anchor_hits & endpoint_terms
        text_anchor_hits = anchor_hits & direct_terms
        score += 0.22 * len(direct_anchor_hits)
        score += 0.08 * len(text_anchor_hits - direct_anchor_hits)
    return score


def preserve_aspect_edges(
    selected_edges: list[dict],
    candidates: list[dict],
    answer_aspects: list[dict],
    requirements: dict,
    policy: dict,
    args: argparse.Namespace,
) -> tuple[list[dict], dict]:
    report = {
        "enabled": bool(args.aspect_preserving_control),
        "added_edges": 0,
        "covered_aspects_before": [],
        "covered_aspects_after": [],
        "added": [],
        "missing_after": [],
    }
    if not args.aspect_preserving_control or not answer_aspects:
        return selected_edges, report

    selected = list(selected_edges)
    selected_pairs = {edge_pair(edge) for edge in selected}
    max_edges = max(len(selected), int(policy.get("max_edges", len(selected)))) + max(0, args.aspect_extra_edges)

    def covered_labels(edges: list[dict]) -> set[str]:
        return {aspect["label"] for aspect in answer_aspects if any(aspect_matches_edge(aspect, edge) for edge in edges)}

    before = covered_labels(selected)
    report["covered_aspects_before"] = sorted(before)

    for aspect in answer_aspects:
        if aspect["label"] in covered_labels(selected):
            continue
        if len(selected) >= max_edges:
            break
        aspect_candidates = []
        for candidate in candidates:
            pair = edge_pair(candidate)
            if pair in selected_pairs:
                continue
            if not aspect_matches_edge(aspect, candidate):
                continue
            candidate_score = aspect_candidate_score(aspect, candidate, requirements)
            if (
                safe_float(candidate.get("counterfactual_penalty")) > args.repair_max_counterfactual_penalty
                and candidate_score < args.aspect_counterfactual_override_score
            ):
                continue
            aspect_candidates.append(candidate)
        if not aspect_candidates:
            continue
        match = max(aspect_candidates, key=lambda edge: aspect_candidate_score(aspect, edge, requirements))
        if not match:
            continue
        selected.append(match)
        selected_pairs.add(edge_pair(match))
        report["added"].append(
            {
                "aspect": aspect["label"],
                "source": match["source"],
                "target": match["target"],
                "final_score": round(safe_float(match.get("final_score")), 6),
                "relation_support": round(safe_float(match.get("relation_support")), 6),
            }
        )

    after = covered_labels(selected)
    report["covered_aspects_after"] = sorted(after)
    report["missing_after"] = sorted({aspect["label"] for aspect in answer_aspects} - after)
    report["added_edges"] = len(report["added"])
    return selected, report


def format_answer_aspects(answer_aspects: list[dict]) -> str:
    if not answer_aspects:
        return "none"
    return "; ".join(aspect["label"] for aspect in answer_aspects)


def is_aspect_control_active(args: argparse.Namespace, answer_aspects: list[dict] | None) -> bool:
    if not args.aspect_preserving_control:
        return False
    aspects = answer_aspects or []
    if not args.adaptive_aspect_gating:
        return True
    return len(aspects) >= max(1, int(args.aspect_gate_min_aspects))


def is_risk_control_active(
    args: argparse.Namespace,
    question: str,
    requirements: dict,
    aspect_control_active: bool = False,
) -> bool:
    if not args.risk_gated_answer_control or aspect_control_active:
        return False
    lowered = clean_text(question).lower()
    transmission_intent = bool(
        re.search(r"\b(spread|spreads|spreading|transmit|transmits|transmission|infect|infects|infection|spillover)\b", lowered)
    )
    pathogen_context = bool(
        re.search(r"\b(virus|viral|ebola|pathogen|disease|outbreak|burial|bodily fluids?|blood|secretions?)\b", lowered)
    )
    answer_types = set(requirements.get("answer_types", []))
    return transmission_intent and (pathogen_context or "general" in answer_types)


def build_answer_control_guidance(
    args: argparse.Namespace,
    sufficiency_report: dict,
    policy: dict,
    answer_aspects: list[dict] | None = None,
    aspect_control_active: bool | None = None,
    risk_control_active: bool = False,
) -> str:
    if aspect_control_active is None:
        aspect_control_active = is_aspect_control_active(args, answer_aspects)
    if (
        not args.repair_aware_answer_control
        and not args.strict_evidence_utility_control
        and not aspect_control_active
        and not risk_control_active
    ):
        return "Use the selected evidence according to the retrieval policy."

    after = sufficiency_report.get("after", {})
    repaired_terms = []
    for action in sufficiency_report.get("actions", []):
        repaired_terms.extend(action.get("new_terms", []))
    repaired_terms = sorted(set(repaired_terms))
    repaired_terms_text = ", ".join(repaired_terms[:16]) or "none"
    aspect_text = format_answer_aspects(answer_aspects or [])
    if aspect_control_active:
        return (
            "Use an aspect-preserving evidence-utility policy. Write a direct answer in "
            f"{args.aspect_answer_word_budget_min}-{args.aspect_answer_word_budget_max} words and at most "
            f"{args.aspect_answer_max_sentences} sentences. "
            f"Mandatory coverage checklist: {aspect_text}. "
            "Address every listed aspect at least once; do not omit a checklist item merely to be concise. "
            "For method or intervention questions, enumerate the distinct supported methods in compact clauses. "
            "Prefer one compact clause or sentence per checklist item. "
            "If a checklist item is not supported by the selected evidence, state that specific uncertainty instead of ignoring it. "
            "Use primary evidence for the main answer and supplemental repair evidence only when it fills "
            f"these missing requirements: {repaired_terms_text}. "
            "Do not summarize all graph paths, add broad background, or add caveats unless they are central to the question. "
            f"Current sufficiency status: {after.get('status', 'unknown')}; "
            f"score={after.get('score', 0.0)}; mode={policy.get('mode', 'unknown')}."
        )
    if risk_control_active:
        return (
            "Use risk-gated answer control. Write a direct answer in "
            f"{args.risk_answer_word_budget_min}-{args.risk_answer_word_budget_max} words and at most "
            f"{args.risk_answer_max_sentences} sentences. "
            "Answer only the asked transmission or spread mechanism. "
            "Prioritize direct contact routes, infected tissues, blood, secretions, organs, and other bodily fluids when supported. "
            "Do not add reservoir speculation, named researchers, phylogeny, outbreak history, ecology background, or post-spillover details unless the question asks for them. "
            "Use supplemental repair evidence only if it directly fills the asked spread mechanism. "
            f"Current sufficiency status: {after.get('status', 'unknown')}; "
            f"score={after.get('score', 0.0)}; mode={policy.get('mode', 'unknown')}."
        )
    if args.strict_evidence_utility_control:
        return (
            "Use a strict evidence-utility policy. Write a direct answer in "
            f"{args.answer_word_budget_min}-{args.answer_word_budget_max} words and at most "
            f"{args.strict_answer_max_sentences} sentences. "
            "Use primary evidence for the main answer. Use supplemental repair evidence only when it fills "
            f"these missing requirements: {repaired_terms_text}. "
            "Do not summarize all graph paths. Do not add historical background, examples, mechanisms, "
            "or caveats unless they are explicitly required by the question. "
            "Prefer reference-answer style: compact, complete, and focused on the exact asked point. "
            f"Current sufficiency status: {after.get('status', 'unknown')}; "
            f"score={after.get('score', 0.0)}; mode={policy.get('mode', 'unknown')}."
        )
    return (
        "Primary evidence is the main basis for the answer. "
        "Supplemental repair evidence is only for filling these missing requirements: "
        f"{repaired_terms_text}. "
        "Do not expand into background from supplemental repair evidence. "
        "If a supplemental path is only loosely related, ignore its extra details. "
        "Prefer a direct, compact answer that covers the question without enumerating every path. "
        f"Current sufficiency status: {after.get('status', 'unknown')}; "
        f"score={after.get('score', 0.0)}; mode={policy.get('mode', 'unknown')}."
    )


def select_prompt_edges(
    selected_edges: list[dict],
    args: argparse.Namespace,
    answer_aspects: list[dict] | None = None,
    aspect_control_active: bool | None = None,
) -> list[dict]:
    if aspect_control_active is None:
        aspect_control_active = is_aspect_control_active(args, answer_aspects)
    for edge in selected_edges:
        edge["used_in_prompt"] = True
    if not args.strict_evidence_utility_control and not aspect_control_active:
        return selected_edges

    primary_edges = [edge for edge in selected_edges if edge.get("evidence_role", "primary") == "primary"]
    repaired_edges = [edge for edge in selected_edges if edge.get("evidence_role") == "repaired"]
    primary_edges = sorted(
        primary_edges,
        key=lambda edge: (
            safe_float(edge.get("final_score")),
            safe_float(edge.get("relation_support")),
            len(edge.get("description", "")),
        ),
        reverse=True,
    )
    if aspect_control_active:
        primary_limit = max(1, args.aspect_primary_edge_limit)
        prompt_edge_limit = max(primary_limit, args.aspect_prompt_edge_limit)
    else:
        primary_limit = max(1, args.strict_primary_edge_limit)
        prompt_edge_limit = primary_limit + max(0, args.strict_repair_edge_limit)
    primary_edges = primary_edges[:primary_limit]
    repaired_edges = [
        edge
        for edge in repaired_edges
        if edge.get("repair_new_terms") and safe_float(edge.get("relation_support")) >= args.strict_repair_min_relation_support
    ]
    repaired_edges = sorted(
        repaired_edges,
        key=lambda edge: (
            len(edge.get("repair_new_terms", [])),
            safe_float(edge.get("final_score")),
            safe_float(edge.get("relation_support")),
        ),
        reverse=True,
    )[: max(0, args.strict_repair_edge_limit)]
    protected_edges: list[dict] = []
    if aspect_control_active and answer_aspects:
        for aspect in answer_aspects:
            matches = sorted(
                [edge for edge in selected_edges if aspect_matches_edge(aspect, edge)],
                key=lambda edge: (
                    edge.get("evidence_role", "primary") == "primary",
                    safe_float(edge.get("final_score")),
                    safe_float(edge.get("relation_support")),
                ),
                reverse=True,
            )
            if matches:
                protected_edges.append(matches[0])

    prompt_edges: list[dict] = []
    prompt_pairs: set[tuple[str, str]] = set()
    for edge in protected_edges + primary_edges + repaired_edges:
        pair = edge_pair(edge)
        if pair in prompt_pairs:
            continue
        if len(prompt_edges) >= prompt_edge_limit and pair not in {edge_pair(item) for item in protected_edges}:
            continue
        prompt_edges.append(edge)
        prompt_pairs.add(pair)
    for edge in selected_edges:
        edge["used_in_prompt"] = edge_pair(edge) in prompt_pairs
    return prompt_edges


def format_evidence_paths(selected_edges: list[dict], repair_aware: bool = False) -> str:
    if not selected_edges:
        return "No graph evidence selected."
    lines = []
    for i, edge in enumerate(selected_edges, start=1):
        role = edge.get("evidence_role", "primary")
        role_label = "Primary evidence"
        if repair_aware and role == "repaired":
            new_terms = ", ".join(edge.get("repair_new_terms", [])) or "missing requirement"
            role_label = f"Supplemental repair evidence; use only for: {new_terms}"
        lines.append(
            "\n".join(
                [
                    f"[Path {i}; {role_label}; score={edge['final_score']:.4f}]",
                    f"{edge['source']} -> {edge['target']}",
                    f"Relation keywords: {edge['keywords'] or 'N/A'}",
                    f"Evidence: {edge['description'] or 'N/A'}",
                    f"Source chunks: {', '.join(edge.get('source_ids', [])[:4]) or 'N/A'}",
                ]
            )
        )
    return "\n\n".join(lines)


def format_supporting_chunks(chunks: list[dict], max_chars: int | None = None, max_chunks: int | None = None) -> str:
    if not chunks:
        return "No supporting text snippets selected."
    lines = []
    selected_chunks = chunks[:max_chunks] if max_chunks else chunks
    for i, chunk in enumerate(selected_chunks, start=1):
        content = chunk["content"]
        if max_chars:
            content = clean_text(content, max_chars)
        lines.append(f"[Chunk {i}; id={chunk['chunk_id']}]\n{content}")
    return "\n\n".join(lines)


def edge_pair(edge: dict) -> tuple[str, str]:
    return tuple(sorted((clean_text(edge.get("source")), clean_text(edge.get("target")))))


def serialize_trace_edge(
    edge: dict,
    rank: int,
    selected: bool,
    query_terms: set[str],
) -> dict:
    edge_terms = set(edge.get("terms", []))
    source = clean_text(edge.get("source"))
    target = clean_text(edge.get("target"))
    return {
        "rank": rank,
        "selected": selected,
        "source": source,
        "target": target,
        "path": f"{source} -> {target}",
        "evidence_role": edge.get("evidence_role", "primary"),
        "used_in_prompt": bool(edge.get("used_in_prompt", True)),
        "repair_new_terms": list(edge.get("repair_new_terms", [])),
        "matched_query_terms": sorted(edge_terms & query_terms),
        "keywords": clean_text(edge.get("keywords"), 220),
        "description": clean_text(edge.get("description"), 420),
        "score_breakdown": {
            "weight": round(safe_float(edge.get("weight")), 6),
            "lexical_score": round(safe_float(edge.get("lexical_score")), 6),
            "embedding_score": round(safe_float(edge.get("embedding_score")), 6),
            "final_score": round(safe_float(edge.get("final_score")), 6),
            "relation_support": round(safe_float(edge.get("relation_support")), 6),
            "counterfactual_penalty": round(safe_float(edge.get("counterfactual_penalty")), 6),
        },
        "source_ids": list(edge.get("source_ids", []))[:8],
        "node_source_ids": list(edge.get("node_source_ids", []))[:8],
    }


def build_evidence_trace(
    question: str,
    requirements: dict,
    policy: dict,
    candidates: list[dict],
    selected_edges: list[dict],
    supporting_chunks: list[dict],
    mse_stats: dict,
    args: argparse.Namespace,
    sufficiency_report: dict | None = None,
    answer_aspects: list[dict] | None = None,
    aspect_report: dict | None = None,
) -> dict:
    query_terms = set(requirements.get("expanded_terms") or requirements.get("query_terms", []))
    selected_pairs = {edge_pair(edge) for edge in selected_edges}
    ranked_candidates = sorted(candidates, key=lambda item: item["final_score"], reverse=True)
    trace_top_k = max(0, int(args.trace_top_k))
    covered_terms = set(mse_stats.get("covered_terms", []))

    counterfactual_candidates = [
        edge for edge in ranked_candidates if safe_float(edge.get("counterfactual_penalty")) > 0
    ]

    return {
        "query_analysis": {
            "question": question,
            "answer_types": requirements.get("answer_types", []),
            "query_terms": requirements.get("query_terms", []),
            "expanded_terms": requirements.get("expanded_terms", []),
            "cue_terms": requirements.get("cue_terms", []),
        },
        "policy": {
            "mode": policy.get("mode"),
            "min_edges": policy.get("min_edges"),
            "max_edges": policy.get("max_edges"),
            "min_coverage": policy.get("min_coverage"),
            "top_chunks": policy.get("top_chunks"),
            "answer_style": args.answer_style,
            "counterfactual_enabled": not args.disable_counterfactual_verification,
            "edge_embeddings_enabled": args.use_edge_embeddings,
        },
        "selection_summary": {
            "candidate_edges": len(candidates),
            "selected_edges": len(selected_edges),
            "selected_edge_ratio": round(len(selected_edges) / max(1, len(candidates)), 6),
            "term_coverage": mse_stats.get("term_coverage", 0.0),
            "covered_terms": sorted(covered_terms),
            "missing_terms": sorted(query_terms - covered_terms),
            "avg_selected_score": mse_stats.get("avg_selected_score", 0.0),
        },
        "sufficiency": sufficiency_report or {"enabled": False},
        "answer_aspects": answer_aspects or [],
        "aspect_preservation": aspect_report or {"enabled": False},
        "selected_edges": [
            serialize_trace_edge(edge, rank, True, query_terms)
            for rank, edge in enumerate(selected_edges, start=1)
        ],
        "top_candidates": [
            serialize_trace_edge(edge, rank, edge_pair(edge) in selected_pairs, query_terms)
            for rank, edge in enumerate(ranked_candidates[:trace_top_k], start=1)
        ],
        "filtered_counterfactual_edges": [
            serialize_trace_edge(edge, rank, edge_pair(edge) in selected_pairs, query_terms)
            for rank, edge in enumerate(counterfactual_candidates[:trace_top_k], start=1)
        ],
        "supporting_chunks": [
            {
                "rank": rank,
                "chunk_id": chunk["chunk_id"],
                "source_count": chunk["source_count"],
                "term_overlap": chunk["term_overlap"],
                "chars": len(chunk["content"]),
                "snippet": clean_text(chunk["content"], 650),
            }
            for rank, chunk in enumerate(supporting_chunks, start=1)
        ],
    }


async def complete(
    client: AsyncOpenAI,
    model: str,
    messages: list[dict],
    timeout_seconds: int,
    retries: int,
) -> str:
    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model,
                    temperature=0,
                    messages=messages,
                ),
                timeout=timeout_seconds,
            )
            return response.choices[0].message.content or ""
        except Exception as exc:
            last_error = exc
            if attempt >= retries:
                break
            await asyncio.sleep(3 * (attempt + 1))
    raise RuntimeError(f"LLM completion failed after {retries + 1} attempt(s): {last_error}")


async def run_sample(
    sample: dict,
    client: AsyncOpenAI,
    llm_model: str,
    api_key: str,
    embedding_base_url: str,
    embedding_model: str,
    args: argparse.Namespace,
) -> dict:
    started = time.perf_counter()
    graph_dir = find_graph_dir(sample)
    graph = nx.read_graphml(graph_dir / "graph_chunk_entity_relation.graphml")
    chunks = load_chunks(graph_dir)
    question = sample.get("input", "")
    requirements = infer_requirements(question)
    policy = evidence_policy(question, requirements, args)

    candidate_limit = args.candidate_edges
    if args.enable_sufficiency_repair:
        candidate_limit = max(args.candidate_edges, args.repair_candidate_edges)
    candidates = build_edge_candidates(graph, question, requirements, args, limit=candidate_limit)
    await maybe_apply_embedding_scores(
        candidates,
        question,
        args,
        api_key,
        embedding_base_url,
        embedding_model,
    )
    initial_candidates = candidates[: args.candidate_edges]
    selected_edges, mse_stats = select_minimal_subgraph(initial_candidates, requirements, policy, args)
    primary_pairs = {edge_pair(edge) for edge in selected_edges}
    supporting_chunks = select_supporting_chunks(selected_edges, chunks, question, policy, args)
    before_sufficiency = evidence_sufficiency_check(
        selected_edges,
        supporting_chunks,
        requirements,
        policy,
        mse_stats,
        args,
    )
    sufficiency_report: dict = {
        "enabled": bool(args.enable_sufficiency_repair),
        "triggered": False,
        "before": before_sufficiency,
        "after": before_sufficiency,
        "actions": [],
        "added_edges": 0,
        "coverage_before": mse_stats.get("term_coverage", 0.0),
        "coverage_after": mse_stats.get("term_coverage", 0.0),
        "selected_edges_before": mse_stats.get("selected_edges", 0),
        "selected_edges_after": mse_stats.get("selected_edges", 0),
    }
    if args.enable_sufficiency_repair and before_sufficiency["status"] == "repair_needed":
        selected_edges, mse_stats, sufficiency_report = repair_insufficient_evidence(
            candidates,
            selected_edges,
            requirements,
            policy,
            mse_stats,
            before_sufficiency,
            args,
        )
        supporting_chunks = select_supporting_chunks(selected_edges, chunks, question, policy, args)
        after_sufficiency = evidence_sufficiency_check(
            selected_edges,
            supporting_chunks,
            requirements,
            policy,
            mse_stats,
            args,
        )
        sufficiency_report["after"] = after_sufficiency
    answer_aspects = infer_answer_aspects(
        question,
        selected_edges,
        supporting_chunks,
        candidates,
        max_aspects=args.aspect_max_aspects,
    )
    aspect_control_active = is_aspect_control_active(args, answer_aspects)
    aspect_report = {
        "enabled": bool(aspect_control_active),
        "available_aspects": [aspect["label"] for aspect in answer_aspects],
        "gate_min_aspects": int(args.aspect_gate_min_aspects),
        "adaptive_gating": bool(args.adaptive_aspect_gating),
        "added_edges": 0,
        "covered_aspects_before": [],
        "covered_aspects_after": [],
        "added": [],
        "missing_after": [],
    }
    if aspect_control_active and answer_aspects:
        selected_edges, aspect_report = preserve_aspect_edges(
            selected_edges,
            candidates,
            answer_aspects,
            requirements,
            policy,
            args,
        )
        if aspect_report.get("added_edges"):
            for edge in selected_edges:
                if edge_pair(edge) not in primary_pairs and any(
                    edge_pair(edge) == edge_pair(candidate)
                    for candidate in candidates
                    for added in aspect_report.get("added", [])
                    if added.get("source") == candidate.get("source") and added.get("target") == candidate.get("target")
                ):
                    primary_pairs.add(edge_pair(edge))
            mse_stats = selection_stats(selected_edges, candidates, requirements, policy)
            supporting_chunks = select_supporting_chunks(selected_edges, chunks, question, policy, args)
            after_sufficiency = evidence_sufficiency_check(
                selected_edges,
                supporting_chunks,
                requirements,
                policy,
                mse_stats,
                args,
            )
            sufficiency_report["after"] = after_sufficiency
            sufficiency_report["coverage_after"] = mse_stats.get("term_coverage", 0.0)
            sufficiency_report["selected_edges_after"] = mse_stats.get("selected_edges", 0)
            answer_aspects = infer_answer_aspects(
                question,
                selected_edges,
                supporting_chunks,
                candidates,
                max_aspects=args.aspect_max_aspects,
            )
            aspect_control_active = is_aspect_control_active(args, answer_aspects)
    mse_stats["sufficiency_score"] = sufficiency_report["after"].get("score", 0.0)
    mse_stats["sufficiency_status"] = sufficiency_report["after"].get("status", "unknown")
    mse_stats["repair_triggered"] = bool(sufficiency_report.get("triggered", False))
    mse_stats["repair_added_edges"] = int(sufficiency_report.get("added_edges", 0))
    mse_stats["coverage_before_repair"] = sufficiency_report.get("coverage_before", mse_stats.get("term_coverage", 0.0))
    mse_stats["coverage_after_repair"] = sufficiency_report.get("coverage_after", mse_stats.get("term_coverage", 0.0))
    mark_evidence_roles(selected_edges, primary_pairs, sufficiency_report)
    risk_control_active = is_risk_control_active(args, question, requirements, aspect_control_active)
    prompt_edges = select_prompt_edges(selected_edges, args, answer_aspects, aspect_control_active)
    answer_control_guidance = build_answer_control_guidance(
        args,
        sufficiency_report,
        policy,
        answer_aspects,
        aspect_control_active,
        risk_control_active,
    )
    max_answer_sentences = policy["max_answer_sentences"]
    if aspect_control_active:
        max_answer_sentences = min(max_answer_sentences, args.aspect_answer_max_sentences)
    elif risk_control_active:
        max_answer_sentences = min(max_answer_sentences, args.risk_answer_max_sentences)
    elif args.repair_aware_answer_control:
        max_answer_sentences = min(max_answer_sentences, args.answer_control_max_sentences)
    if args.strict_evidence_utility_control and not args.aspect_preserving_control:
        max_answer_sentences = min(max_answer_sentences, args.strict_answer_max_sentences)
    mse_stats["repair_aware_answer_control"] = bool(args.repair_aware_answer_control)
    mse_stats["strict_evidence_utility_control"] = bool(args.strict_evidence_utility_control)
    mse_stats["aspect_preserving_control"] = bool(args.aspect_preserving_control)
    mse_stats["adaptive_aspect_gating"] = bool(args.adaptive_aspect_gating)
    mse_stats["aspect_control_active"] = bool(aspect_control_active)
    mse_stats["risk_gated_answer_control"] = bool(args.risk_gated_answer_control)
    mse_stats["risk_control_active"] = bool(risk_control_active)
    mse_stats["aspect_count"] = len(answer_aspects)
    mse_stats["aspect_added_edges"] = int(aspect_report.get("added_edges", 0))
    mse_stats["aspect_missing_after"] = list(aspect_report.get("missing_after", []))
    mse_stats["max_answer_sentences"] = int(max_answer_sentences)
    mse_stats["prompt_evidence_edges"] = len(prompt_edges)
    mse_stats["prompt_primary_edges"] = sum(1 for edge in prompt_edges if edge.get("evidence_role", "primary") == "primary")
    mse_stats["prompt_repaired_edges"] = sum(1 for edge in prompt_edges if edge.get("evidence_role") == "repaired")
    evidence_trace = build_evidence_trace(
        question,
        requirements,
        policy,
        candidates,
        selected_edges,
        supporting_chunks,
        mse_stats,
        args,
        sufficiency_report,
        answer_aspects,
        aspect_report,
    )

    control_enabled = (
        args.repair_aware_answer_control
        or args.strict_evidence_utility_control
        or aspect_control_active
        or risk_control_active
    )
    evidence_paths = format_evidence_paths(prompt_edges, repair_aware=control_enabled)
    if aspect_control_active:
        chunk_context = format_supporting_chunks(
            supporting_chunks,
            max_chars=args.aspect_chunk_chars,
            max_chunks=args.aspect_top_chunks,
        )
    elif risk_control_active:
        chunk_context = format_supporting_chunks(
            supporting_chunks,
            max_chars=args.risk_chunk_chars,
            max_chunks=args.risk_top_chunks,
        )
    elif args.strict_evidence_utility_control:
        chunk_context = format_supporting_chunks(
            supporting_chunks,
            max_chars=args.strict_chunk_chars,
            max_chunks=args.strict_top_chunks,
        )
    else:
        chunk_context = format_supporting_chunks(supporting_chunks)
    prompt = USER_PROMPT.format(
        question=question,
        requirements=format_requirements(requirements),
        evidence_paths=evidence_paths,
        supporting_chunks=chunk_context,
        response_guidance=policy["response_guidance"],
        answer_control_guidance=answer_control_guidance,
        max_answer_sentences=max_answer_sentences,
    )
    if args.only_select:
        prediction = ""
    else:
        prediction = await complete(
            client,
            llm_model,
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            args.request_timeout,
            args.request_retries,
        )
    total_seconds = time.perf_counter() - started
    evidence_trace["prompt_stats"] = {
        "prompt_chars": len(prompt),
        "answer_chars": len(prediction),
        "only_select": args.only_select,
    }

    return {
        "id": sample.get("_id") or f"line-{sample['_line_no']}",
        "line_no": sample["_line_no"],
        "dataset": sample.get("dataset"),
        "label": sample.get("label"),
        "question": question,
        "reference_answers": sample.get("answers", []),
        "prediction": prediction,
        "method": args.method_name,
        "context_chars": len(sample.get("context", "")),
        "graph_nodes": graph.number_of_nodes(),
        "graph_edges": graph.number_of_edges(),
        "query_seconds": round(total_seconds, 3),
        "working_dir": str(graph_dir.relative_to(ROOT_DIR)),
        "context_hash": context_hash(sample.get("context", "")),
        "mse_stats": mse_stats,
        "requirements": requirements,
        "evidence_policy": policy,
        "answer_aspects": answer_aspects,
        "evidence_trace": evidence_trace,
        "selected_evidence": [
            {
                "source": edge["source"],
                "target": edge["target"],
                "keywords": edge["keywords"],
                "description": edge["description"],
                "weight": edge["weight"],
                "lexical_score": round(edge["lexical_score"], 6),
                "embedding_score": round(edge.get("embedding_score", 0.0), 6),
                "final_score": round(edge["final_score"], 6),
                "relation_support": edge.get("relation_support", 0.0),
                "counterfactual_penalty": edge.get("counterfactual_penalty", 0.0),
                "source_ids": edge.get("source_ids", []),
                "node_source_ids": edge.get("node_source_ids", []),
            }
            for edge in selected_edges
        ],
        "supporting_chunks": [
            {
                "chunk_id": chunk["chunk_id"],
                "source_count": chunk["source_count"],
                "term_overlap": chunk["term_overlap"],
                "chars": len(chunk["content"]),
            }
            for chunk in supporting_chunks
        ],
        "prompt_chars": len(prompt),
        "answer_chars": len(prediction),
        "only_select": args.only_select,
    }


def write_summary(results: list[dict], output_path: Path, summary_path: Path, args: argparse.Namespace) -> None:
    if not results:
        summary_path.write_text("# MSE-PathRAG v1 Summary\n\nNo results.\n", encoding="utf-8")
        return
    avg_selected = sum(item["mse_stats"]["selected_edges"] for item in results) / len(results)
    avg_candidates = sum(item["mse_stats"]["candidate_edges"] for item in results) / len(results)
    avg_coverage = sum(item["mse_stats"]["term_coverage"] for item in results) / len(results)
    avg_query = sum(item["query_seconds"] for item in results) / len(results)
    avg_answer_chars = sum(item["answer_chars"] for item in results) / len(results)
    avg_prompt_chars = sum(item["prompt_chars"] for item in results) / len(results)
    avg_policy_max_edges = sum(item["mse_stats"]["target_max_edges"] for item in results) / len(results)
    repair_count = sum(1 for item in results if item["mse_stats"].get("repair_triggered"))
    avg_repair_added = sum(item["mse_stats"].get("repair_added_edges", 0) for item in results) / len(results)
    avg_sufficiency = sum(item["mse_stats"].get("sufficiency_score", 0.0) for item in results) / len(results)
    avg_aspects = sum(item["mse_stats"].get("aspect_count", 0) for item in results) / len(results)
    avg_aspect_added = sum(item["mse_stats"].get("aspect_added_edges", 0) for item in results) / len(results)
    aspect_active_count = sum(1 for item in results if item["mse_stats"].get("aspect_control_active"))
    risk_active_count = sum(1 for item in results if item["mse_stats"].get("risk_control_active"))
    avg_coverage_before_repair = sum(
        item["mse_stats"].get("coverage_before_repair", item["mse_stats"].get("term_coverage", 0.0))
        for item in results
    ) / len(results)
    mode_counts = Counter(item["mse_stats"].get("evidence_mode", "unknown") for item in results)

    lines = [
        "# MSE-PathRAG Run Summary\n\n",
        f"- Source dataset: `{args.dataset}`\n",
        f"- Output: `{output_path.as_posix()}`\n",
        f"- Samples: {len(results)}\n",
        f"- Candidate edges per sample: {args.candidate_edges}\n",
        f"- Configured max selected edges: {args.max_edges}\n",
        f"- Configured min selected edges: {args.min_edges}\n",
        f"- Adaptive evidence budget: {not args.fixed_evidence_budget}\n",
        f"- Edge embeddings enabled: {args.use_edge_embeddings}\n\n",
        f"- Sufficiency repair enabled: {getattr(args, 'enable_sufficiency_repair', False)}\n",
        f"- Repair-aware answer control enabled: {getattr(args, 'repair_aware_answer_control', False)}\n",
        f"- Strict evidence-utility control enabled: {getattr(args, 'strict_evidence_utility_control', False)}\n\n",
        f"- Aspect-preserving control enabled: {getattr(args, 'aspect_preserving_control', False)}\n\n",
        f"- Adaptive aspect gating enabled: {getattr(args, 'adaptive_aspect_gating', False)}\n\n",
        f"- Risk-gated answer control enabled: {getattr(args, 'risk_gated_answer_control', False)}\n\n",
        "## Aggregate Signals\n\n",
        "| Metric | Value |\n",
        "|---|---:|\n",
        f"| Avg candidate edges | {avg_candidates:.2f} |\n",
        f"| Avg selected edges | {avg_selected:.2f} |\n",
        f"| Avg adaptive max edges | {avg_policy_max_edges:.2f} |\n",
        f"| Repair-triggered samples | {repair_count} |\n",
        f"| Avg repaired edges | {avg_repair_added:.2f} |\n",
        f"| Avg answer aspects | {avg_aspects:.2f} |\n",
        f"| Aspect-active samples | {aspect_active_count} |\n",
        f"| Risk-active samples | {risk_active_count} |\n",
        f"| Avg aspect-added edges | {avg_aspect_added:.2f} |\n",
        f"| Avg coverage before repair | {avg_coverage_before_repair:.3f} |\n",
        f"| Avg query-term coverage | {avg_coverage:.3f} |\n",
        f"| Avg sufficiency score | {avg_sufficiency:.3f} |\n",
        f"| Avg query seconds | {avg_query:.3f} |\n",
        f"| Avg prompt chars | {avg_prompt_chars:.1f} |\n",
        f"| Avg answer chars | {avg_answer_chars:.1f} |\n\n",
        "## Evidence Modes\n\n",
        "| Mode | Count |\n",
        "|---|---:|\n",
    ]
    for mode, count in sorted(mode_counts.items()):
        lines.append(f"| {mode} | {count} |\n")

    lines += [
        "## Per-Sample Signals\n\n",
        "| Line | Dataset | Mode | Repair | Selected Edges | Coverage | Sufficiency | Query Seconds | Answer Chars | Question |\n",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---|\n",
    ]
    for item in sorted(results, key=lambda row: row["line_no"]):
        question = clean_text(item.get("question", ""), 90).replace("|", "/")
        stats = item["mse_stats"]
        lines.append(
            f"| {item['line_no']} | {item.get('dataset')} | "
            f"{stats.get('evidence_mode')} | {stats.get('repair_added_edges', 0)} | "
            f"{stats['selected_edges']} | {stats['term_coverage']:.3f} | "
            f"{stats.get('sufficiency_score', 0.0):.3f} | {item['query_seconds']:.3f} | "
            f"{item['answer_chars']} | {question} |\n"
        )
    summary_path.write_text("".join(lines), encoding="utf-8")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="datasets/UltraDomain/biology_fullcontext_50.jsonl")
    parser.add_argument("--line-nos", default=None, help="Comma-separated zero-based line numbers; supports ranges like 0-9")
    parser.add_argument("--output", default="results/historical_sage/mse_pathrag_v1_biology_10.jsonl")
    parser.add_argument("--summary", default="results/historical_sage/mse_pathrag_v1_biology_10_summary.md")
    parser.add_argument("--model", default=None)
    parser.add_argument("--candidate-edges", type=int, default=80)
    parser.add_argument("--min-edges", type=int, default=4)
    parser.add_argument("--max-edges", type=int, default=8)
    parser.add_argument("--min-coverage", type=float, default=0.55)
    parser.add_argument("--max-redundancy", type=float, default=0.82)
    parser.add_argument("--top-chunks", type=int, default=4)
    parser.add_argument("--max-chunk-chars", type=int, default=1800)
    parser.add_argument("--use-edge-embeddings", action="store_true")
    parser.add_argument("--fixed-evidence-budget", action="store_true")
    parser.add_argument("--disable-counterfactual-verification", action="store_true")
    parser.add_argument("--embedding-max-chars", type=int, default=8192)
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--request-timeout", type=int, default=120)
    parser.add_argument("--request-retries", type=int, default=1)
    parser.add_argument("--answer-style", choices=["coverage", "compact"], default="coverage")
    parser.add_argument("--method-name", default="mse_pathrag_v5_counterfactual")
    parser.add_argument("--trace-top-k", type=int, default=12)
    parser.add_argument("--enable-sufficiency-repair", action="store_true")
    parser.add_argument("--repair-candidate-edges", type=int, default=160)
    parser.add_argument("--repair-extra-edges", type=int, default=3)
    parser.add_argument("--repair-target-coverage", type=float, default=0.72)
    parser.add_argument("--sufficiency-threshold", type=float, default=0.68)
    parser.add_argument("--repair-min-relation-support", type=float, default=0.06)
    parser.add_argument("--repair-max-counterfactual-penalty", type=float, default=0.05)
    parser.add_argument("--repair-aware-answer-control", action="store_true")
    parser.add_argument("--answer-control-max-sentences", type=int, default=4)
    parser.add_argument("--strict-evidence-utility-control", action="store_true")
    parser.add_argument("--answer-word-budget-min", type=int, default=90)
    parser.add_argument("--answer-word-budget-max", type=int, default=130)
    parser.add_argument("--strict-answer-max-sentences", type=int, default=3)
    parser.add_argument("--strict-primary-edge-limit", type=int, default=7)
    parser.add_argument("--strict-repair-edge-limit", type=int, default=1)
    parser.add_argument("--strict-repair-min-relation-support", type=float, default=0.08)
    parser.add_argument("--strict-top-chunks", type=int, default=3)
    parser.add_argument("--strict-chunk-chars", type=int, default=900)
    parser.add_argument("--aspect-preserving-control", action="store_true")
    parser.add_argument("--adaptive-aspect-gating", action="store_true")
    parser.add_argument("--aspect-gate-min-aspects", type=int, default=2)
    parser.add_argument("--aspect-max-aspects", type=int, default=8)
    parser.add_argument("--aspect-extra-edges", type=int, default=5)
    parser.add_argument("--aspect-counterfactual-override-score", type=float, default=0.7)
    parser.add_argument("--aspect-primary-edge-limit", type=int, default=8)
    parser.add_argument("--aspect-prompt-edge-limit", type=int, default=10)
    parser.add_argument("--aspect-top-chunks", type=int, default=4)
    parser.add_argument("--aspect-chunk-chars", type=int, default=1200)
    parser.add_argument("--aspect-answer-word-budget-min", type=int, default=95)
    parser.add_argument("--aspect-answer-word-budget-max", type=int, default=145)
    parser.add_argument("--aspect-answer-max-sentences", type=int, default=3)
    parser.add_argument("--risk-gated-answer-control", action="store_true")
    parser.add_argument("--risk-answer-word-budget-min", type=int, default=55)
    parser.add_argument("--risk-answer-word-budget-max", type=int, default=105)
    parser.add_argument("--risk-answer-max-sentences", type=int, default=3)
    parser.add_argument("--risk-top-chunks", type=int, default=3)
    parser.add_argument("--risk-chunk-chars", type=int, default=850)
    parser.add_argument("--only-select", action="store_true")
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--skip-existing-output", action="store_true")
    args = parser.parse_args()

    load_dotenv(ROOT_DIR / ".env")
    api_key = os.getenv("DEEPSEEK_API_KEY")
    llm_base_url = os.getenv("DEEPSEEK_API_BASE")
    llm_model = args.model or os.getenv("DEEPSEEK_MODEL")
    embedding_base_url = os.getenv("EMBEDDING_API_BASE")
    embedding_model = os.getenv("EMBEDDING_MODEL")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is missing in .env")
    if not llm_base_url:
        raise RuntimeError("DEEPSEEK_API_BASE is missing in .env")
    if not llm_model:
        raise RuntimeError("DEEPSEEK_MODEL is missing in .env")
    if args.use_edge_embeddings and (not embedding_base_url or not embedding_model):
        raise RuntimeError("EMBEDDING_API_BASE or EMBEDDING_MODEL is missing in .env")

    dataset_path = ROOT_DIR / args.dataset
    all_samples = load_jsonl(dataset_path)
    selected_lines = parse_line_nos(args.line_nos, len(all_samples))
    samples = [sample for sample in all_samples if sample["_line_no"] in selected_lines]
    if not samples:
        raise RuntimeError("No samples selected.")

    output_path = ROOT_DIR / args.output
    summary_path = ROOT_DIR / args.summary
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = load_completed_line_nos(output_path) if args.skip_existing_output else set()
    samples = [sample for sample in samples if sample["_line_no"] not in completed]

    existing_results = []
    if args.append and output_path.exists():
        existing_results = load_jsonl(output_path)

    client = AsyncOpenAI(api_key=api_key, base_url=llm_base_url)
    mode = "a" if args.append else "w"
    print(f"Selected {len(samples)} sample(s); skipped {len(completed)} existing output sample(s)")
    print(f"Dataset: {dataset_path}")
    print(f"Output: {output_path}")

    new_results = []
    with output_path.open(mode, encoding="utf-8") as out:
        for i, sample in enumerate(samples, start=1):
            print(f"[{i}/{len(samples)}] line={sample['_line_no']} {sample.get('input')}", flush=True)
            result = await run_sample(
                sample,
                client,
                llm_model,
                api_key,
                embedding_base_url or "",
                embedding_model or "",
                args,
            )
            new_results.append(result)
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            out.flush()
            stats = result["mse_stats"]
            print(
                f"Done: selected_edges={stats['selected_edges']} "
                f"coverage={stats['term_coverage']} query={result['query_seconds']}s",
                flush=True,
            )

    all_results = existing_results + new_results
    write_summary(all_results, output_path, summary_path, args)
    print(f"Saved results to {output_path}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    asyncio.run(main())
