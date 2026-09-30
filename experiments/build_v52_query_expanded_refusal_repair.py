from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from build_answer_contract_agent import clean_text, derive_answer_contract, parse_json_object
from build_claim_evidence_agent import EvidenceIndex, load_full_source_rows, normalize_space, write_jsonl
from build_evidence_preserving_compressor import audit_answer
from build_text_evidence_expansion_fallback import (
    build_full_text_index,
    contains_refusal,
    load_jsonl,
    parse_line_nos,
    sanitize_evidence_labels,
    source_by_subset_line,
)


ROOT_DIR = Path(__file__).resolve().parents[1]

EXPAND_SYSTEM = """You are a query expansion agent for evidence retrieval.
Given a question, produce search phrases and entity/theme terms that may help find answer evidence in the source text.
Do not answer the question. Do not use reference answers.
Prefer terms that are likely to appear verbatim in a book or contract."""

EXPAND_USER = """Question:
{question}

Return JSON only:
{{
  "search_phrases": ["phrase", "..."],
  "must_cover_terms": ["term", "..."]
}}"""

ANSWER_SYSTEM = """You are a strict evidence-bounded QA repair agent.
The current answer may be an incorrect refusal caused by missing retrieved evidence.
Use only the supplied query-expanded text evidence. Do not use outside knowledge or reference answers.
If the evidence gives relevant facts for the question, answer directly and concisely.
Use wording supported by the evidence. Do not cite evidence labels.
Only say the answer cannot be determined if the expanded evidence is genuinely insufficient."""

ANSWER_USER = """Question:
{question}

Answer contract:
{contract}

Current answer:
{current_answer}

Query expansion used for retrieval:
{query_expansion}

Expanded evidence:
{evidence}

Write the best evidence-grounded answer in 2-5 concise sentences."""


STOPWORDS = {
    "about", "after", "also", "answer", "book", "broader", "character", "context",
    "describe", "does", "facing", "from", "have", "industry", "some", "that",
    "their", "these", "this", "today", "what", "when", "where", "which", "with",
}


def question_terms(question: str) -> list[str]:
    terms = []
    for token in re.findall(r"[A-Za-z][A-Za-z0-9'-]*", question.lower()):
        token = token.strip("'")
        if len(token) < 4 or token in STOPWORDS:
            continue
        terms.append(token)
    deduped = []
    seen = set()
    for term in terms:
        if term not in seen:
            seen.add(term)
            deduped.append(term)
    return deduped


def term_coverage(answer: str, terms: list[str]) -> float:
    if not terms:
        return 1.0
    lowered = answer.lower()
    hits = sum(1 for term in terms if re.search(rf"\b{re.escape(term)}\b", lowered))
    return hits / len(terms)


def split_chunks(text: str, size: int, overlap: int) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            candidates = [
                text.rfind(". ", start + size // 2, end),
                text.rfind("\n", start + size // 2, end),
            ]
            boundary = max(candidates)
            if boundary > start:
                end = boundary + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
    return chunks


async def expand_query(client: AsyncOpenAI, model: str, question: str) -> dict:
    response = await client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": EXPAND_SYSTEM},
            {"role": "user", "content": EXPAND_USER.format(question=question)},
        ],
    )
    parsed = parse_json_object(response.choices[0].message.content or "")
    phrases = parsed.get("search_phrases")
    terms = parsed.get("must_cover_terms")
    parsed["search_phrases"] = [clean_text(item) for item in phrases if clean_text(item)] if isinstance(phrases, list) else []
    parsed["must_cover_terms"] = [clean_text(item) for item in terms if clean_text(item)] if isinstance(terms, list) else []
    if question not in parsed["search_phrases"]:
        parsed["search_phrases"].insert(0, question)
    return parsed


def select_expanded_windows(chunks: list[str], question: str, current_answer: str, expansion: dict, top_k: int) -> list[dict]:
    if not chunks:
        return []
    search_phrases = expansion.get("search_phrases") or []
    must_cover_terms = expansion.get("must_cover_terms") or []
    current_signal = re.sub(
        r"\b(?:cannot|determined|provided|context|chunks|solely|insufficient|information|evidence)\b",
        " ",
        current_answer,
        flags=re.I,
    )
    queries = [question, current_signal] + search_phrases + must_cover_terms
    queries = [clean_text(query) for query in queries if clean_text(query)]

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
        max_features=80000,
    )
    matrix = vectorizer.fit_transform(chunks + queries)
    chunk_matrix = matrix[: len(chunks)]
    query_matrix = matrix[len(chunks) :]
    scores = linear_kernel(query_matrix, chunk_matrix).max(axis=0)
    ranked = scores.argsort()[::-1][: max(1, top_k)]
    selected = set()
    for raw_index in ranked:
        index = int(raw_index)
        selected.add(index)
        if index > 0:
            selected.add(index - 1)
        if index + 1 < len(chunks):
            selected.add(index + 1)
    records = []
    for index in sorted(selected):
        records.append({
            "kind": "full_text",
            "text": chunks[index],
            "source_id": f"expanded-window-{index}",
            "score": round(float(scores[index]), 6),
        })
    return records


def evidence_prompt(items: list[dict]) -> str:
    lines = []
    for index, item in enumerate(items, start=1):
        lines.append(
            f"[Text {index}; {item.get('source_id', '')}; score={item.get('score', 0)}]\n"
            f"{item.get('text', '')}"
        )
    return "\n\n".join(lines)


async def complete_answer(
    client: AsyncOpenAI,
    model: str,
    question: str,
    current_answer: str,
    expansion: dict,
    evidence: list[dict],
) -> str:
    contract = derive_answer_contract(question)
    response = await client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": ANSWER_SYSTEM},
            {
                "role": "user",
                "content": ANSWER_USER.format(
                    question=question,
                    contract=json.dumps(contract, ensure_ascii=False),
                    current_answer=current_answer,
                    query_expansion=json.dumps(expansion, ensure_ascii=False),
                    evidence=evidence_prompt(evidence),
                ),
            },
        ],
    )
    return sanitize_evidence_labels(response.choices[0].message.content or "")


def expansion_term_hits(answer: str, expansion: dict) -> list[str]:
    lowered = answer.lower()
    hits = []
    for term in expansion.get("must_cover_terms") or []:
        normalized = clean_text(term).lower()
        if normalized and normalized in lowered:
            hits.append(term)
    return hits


def guard_candidate(
    candidate: str,
    current: str,
    question: str,
    text_index: EvidenceIndex,
    expansion: dict,
) -> tuple[bool, dict]:
    graph_index = EvidenceIndex.build([])
    traces, metrics = audit_answer(candidate, text_index, graph_index)
    terms = question_terms(question)
    coverage = term_coverage(candidate, terms)
    hits = expansion_term_hits(candidate, expansion)
    checks = {
        "candidate_present": len(candidate) >= 70,
        "refusal_removed": contains_refusal(current) and not contains_refusal(candidate),
        "mean_support": float(metrics.get("mean_support", 0.0)) >= 0.36,
        "unsupported_ratio": float(metrics.get("unsupported_ratio", 0.0)) <= 0.45,
        "question_term_coverage": coverage >= 0.20,
        "expanded_term_hit": bool(hits) or len(expansion.get("must_cover_terms") or []) == 0,
    }
    guard = {
        "accepted": all(checks.values()),
        "checks": checks,
        "question_terms": terms,
        "question_term_coverage": round(coverage, 3),
        "expanded_term_hits": hits,
        "post_claim_metrics": metrics,
        "post_claim_traces": traces,
        "rejection_reasons": [name for name, passed in checks.items() if not passed],
    }
    return all(checks.values()), guard


def should_attempt(row: dict, requested_lines: set[int] | None) -> bool:
    line_no = int(row.get("line_no", -1))
    if requested_lines is not None:
        return line_no in requested_lines
    return contains_refusal(row.get("prediction", ""))


def write_summary(path: Path, rows: list[dict], source: Path) -> None:
    agents = [row.get("query_expanded_refusal_repair") or {} for row in rows]
    enabled = [agent for agent in agents if agent.get("enabled")]
    attempted = [agent for agent in enabled if agent.get("attempted")]
    accepted = [agent for agent in attempted if agent.get("accepted")]
    actions = Counter(agent.get("action", "not_applicable") for agent in agents)
    lines = [
        "# SAGE v52 Query-Expanded Refusal Repair Summary\n\n",
        f"- Source: `{source.as_posix()}`\n",
        f"- Samples: {len(rows)}\n",
        "- Triggered for refusal-like current answers unless line numbers are specified.\n",
        "- Query expansion uses the question only; reference answers and judge scores are not runtime inputs.\n\n",
        "| Metric | Count |\n",
        "|---|---:|\n",
        f"| Enabled | {len(enabled)} |\n",
        f"| Attempted | {len(attempted)} |\n",
        f"| Accepted | {len(accepted)} |\n\n",
        "## Actions\n\n",
        "| Action | Count |\n",
        "|---|---:|\n",
    ]
    for action, count in sorted(actions.items()):
        lines.append(f"| {action} | {count} |\n")
    lines.extend([
        "\n## Per-Sample Decisions\n\n",
        "| Line | Dataset | Action | Accepted | Term Coverage | Reasons |\n",
        "|---:|---|---|---|---:|---|\n",
    ])
    for row in rows:
        agent = row.get("query_expanded_refusal_repair") or {}
        if not agent.get("enabled"):
            continue
        guard = agent.get("guard") or {}
        reasons = "; ".join(guard.get("rejection_reasons", [])).replace("|", "/")
        lines.append(
            f"| {row.get('line_no')} | {row.get('dataset')} | {agent.get('action')} | "
            f"{agent.get('accepted')} | {guard.get('question_term_coverage', 0)} | {reasons or 'none'} |\n"
        )
    path.write_text("".join(lines), encoding="utf-8")


async def main_async() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--full-context-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--line-nos", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--chunk-chars", type=int, default=1800)
    parser.add_argument("--chunk-overlap", type=int, default=220)
    args = parser.parse_args()

    load_dotenv(ROOT_DIR / ".env")
    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_API_BASE")
    model = args.model or os.getenv("DEEPSEEK_MODEL")
    if not api_key or not base_url or not model:
        raise RuntimeError("DeepSeek-compatible API configuration is incomplete in .env")

    rows = load_jsonl(ROOT_DIR / args.input)
    dataset_rows = load_jsonl(ROOT_DIR / args.dataset)
    source_rows = source_by_subset_line(dataset_rows)
    full_sources = load_full_source_rows(dataset_rows, ROOT_DIR / args.full_context_root)
    requested_lines = parse_line_nos(args.line_nos)
    client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=300.0, max_retries=2)

    output_rows = []
    for row in rows:
        new_row = dict(row)
        if not should_attempt(row, requested_lines):
            new_row["query_expanded_refusal_repair"] = {"enabled": False}
            output_rows.append(new_row)
            continue

        line_no = int(row.get("line_no", -1))
        source = source_rows[line_no]
        full_source = full_sources.get(
            (
                normalize_space(source.get("source_dataset")),
                int(source.get("source_line_no", -1)),
            ),
            source,
        )
        full_context = clean_text(full_source.get("context", ""))
        chunks = split_chunks(full_context, args.chunk_chars, args.chunk_overlap)
        agent = {
            "enabled": True,
            "version": "v52",
            "attempted": True,
            "action": "retain_current",
            "accepted": False,
            "reference_answer_used": False,
        }
        try:
            expansion = await expand_query(client, model, row.get("question", ""))
            evidence = select_expanded_windows(
                chunks,
                row.get("question", ""),
                row.get("prediction", ""),
                expansion,
                args.top_k,
            )
            text_index = EvidenceIndex.build(evidence)
            candidate = await complete_answer(
                client,
                model,
                row.get("question", ""),
                row.get("prediction", ""),
                expansion,
                evidence,
            )
            accepted, guard = guard_candidate(
                candidate,
                row.get("prediction", ""),
                row.get("question", ""),
                text_index,
                expansion,
            )
            agent["query_expansion"] = expansion
            agent["selected_evidence"] = evidence
            agent["evidence_windows"] = len(evidence)
            agent["candidate_answer"] = candidate
            agent["guard"] = guard
            agent["accepted"] = accepted
            if accepted:
                new_row["prediction_before_query_expanded_refusal_repair"] = new_row.get("prediction", "")
                new_row["prediction"] = candidate
                new_row["method"] = "sage_v52_query_expanded_refusal_repair"
                agent["action"] = "use_query_expanded_answer"
            else:
                new_row["rejected_query_expanded_refusal_repair"] = candidate
        except Exception as exc:
            agent["error"] = f"{type(exc).__name__}: {exc}"[:300]
            agent["action"] = "error_keep_current"

        new_row["query_expanded_refusal_repair"] = agent
        output_rows.append(new_row)

    output_path = ROOT_DIR / args.output
    write_jsonl(output_rows, output_path)
    write_summary(ROOT_DIR / args.summary, output_rows, ROOT_DIR / args.input)
    print(f"Saved v52 query-expanded refusal repair results to {output_path}")
    print(f"Saved summary to {ROOT_DIR / args.summary}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
