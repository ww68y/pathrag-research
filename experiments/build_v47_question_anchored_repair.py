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

from build_answer_contract_agent import clean_text
from build_claim_evidence_agent import (
    EvidenceIndex,
    load_full_source_rows,
    normalize_space,
    write_jsonl,
)
from build_evidence_preserving_compressor import audit_answer
from build_text_evidence_expansion_fallback import contains_refusal


ROOT_DIR = Path(__file__).resolve().parents[1]

STOPWORDS = {
    "about", "according", "after", "also", "are", "between", "book", "broader",
    "character", "describe", "does", "during", "facing", "from", "have", "into",
    "main", "many", "more", "primary", "role", "some", "that", "their", "these",
    "this", "today", "what", "when", "where", "which", "while", "with", "would",
}

SYSTEM_PROMPT = """You are a strict evidence-bounded QA repair agent.
Use only the supplied question-anchored text evidence and the named setting in the question.
Do not replace the question's named event, entity, time scope, or domain with a similar one from evidence.
If evidence gives useful facts about the named entity but the setting is implicit in the question, answer in that setting.
Give the direct answer first, stay concise, and do not cite evidence labels."""

USER_PROMPT = """Question:
{question}

Current answer:
{current_answer}

Question-anchored evidence:
{evidence}

Write the best evidence-grounded answer. If the evidence is insufficient, say what is missing without switching to a different setting."""


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def parse_line_nos(value: str | None) -> set[int] | None:
    if not value:
        return None
    return {int(item.strip()) for item in value.split(",") if item.strip()}


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


def source_by_subset_line(rows: list[dict]) -> dict[int, dict]:
    return {int(row.get("_subset_line_no", index)): row for index, row in enumerate(rows)}


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


def setting_conflict(answer: str, terms: list[str]) -> bool:
    lowered = answer.lower()
    conflict_patterns = [
        r"\bnot\s+(?:the\s+)?chechen",
        r"\bnot\s+chechnya",
        r"\binstead\s+(?:places|describes|refers)",
        r"\bdifferent\s+(?:setting|conflict|event|context)",
    ]
    if any(re.search(pattern, lowered) for pattern in conflict_patterns):
        return True
    return any(re.search(rf"\bnot\s+(?:the\s+)?{re.escape(term)}\b", lowered) for term in terms)


def select_question_windows(chunks: list[str], question: str, top_k: int) -> list[dict]:
    if not chunks:
        return []
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
        max_features=60000,
    )
    matrix = vectorizer.fit_transform(chunks + [question])
    scores = linear_kernel(matrix[-1:], matrix[:-1]).ravel()
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
            "source_id": f"question-window-{index}",
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


async def complete_answer(client: AsyncOpenAI, model: str, question: str, current_answer: str, evidence: list[dict]) -> str:
    response = await client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_PROMPT.format(
                    question=question,
                    current_answer=current_answer,
                    evidence=evidence_prompt(evidence),
                ),
            },
        ],
    )
    return clean_text(response.choices[0].message.content or "")


def guard_candidate(candidate: str, current: str, question: str, text_index: EvidenceIndex, min_term_coverage: float) -> tuple[bool, dict]:
    graph_index = EvidenceIndex.build([])
    traces, metrics = audit_answer(candidate, text_index, graph_index)
    terms = question_terms(question)
    coverage = term_coverage(candidate, terms)
    checks = {
        "candidate_present": len(candidate) >= 60,
        "refusal_removed": not contains_refusal(candidate),
        "mean_support": float(metrics.get("mean_support", 0.0)) >= 0.40,
        "unsupported_ratio": float(metrics.get("unsupported_ratio", 0.0)) <= 0.40,
        "question_term_coverage": coverage >= min_term_coverage,
        "no_setting_conflict": not setting_conflict(candidate, terms),
    }
    guard = {
        "accepted": all(checks.values()),
        "checks": checks,
        "question_terms": terms,
        "question_term_coverage": round(coverage, 3),
        "post_claim_metrics": metrics,
        "post_claim_traces": traces,
        "rejection_reasons": [name for name, passed in checks.items() if not passed],
    }
    return all(checks.values()), guard


def should_attempt(row: dict, requested_lines: set[int] | None) -> bool:
    line_no = int(row.get("line_no", -1))
    if requested_lines is not None:
        return line_no in requested_lines
    agent = row.get("text_evidence_expansion_fallback") or {}
    return agent.get("accepted") is True


def write_summary(path: Path, rows: list[dict], source: Path) -> None:
    agents = [row.get("question_anchored_repair") or {} for row in rows]
    enabled = [agent for agent in agents if agent.get("enabled")]
    attempted = [agent for agent in enabled if agent.get("attempted")]
    accepted = [agent for agent in attempted if agent.get("accepted")]
    actions = Counter(agent.get("action", "not_applicable") for agent in agents)
    lines = [
        "# SAGE v47 Question-Anchored Repair Summary\n\n",
        f"- Source: `{source.as_posix()}`\n",
        f"- Samples: {len(rows)}\n",
        "- Uses question-only TF-IDF evidence windows; reference answers and judge scores are not used.\n\n",
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
        agent = row.get("question_anchored_repair") or {}
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
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--chunk-chars", type=int, default=1800)
    parser.add_argument("--chunk-overlap", type=int, default=220)
    parser.add_argument("--min-term-coverage", type=float, default=0.30)
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
            new_row["question_anchored_repair"] = {"enabled": False}
            output_rows.append(new_row)
            continue

        line_no = int(row.get("line_no", -1))
        source = source_rows[line_no]
        full_source = full_sources.get((
            normalize_space(source.get("source_dataset")),
            int(source.get("source_line_no", -1)),
        ), source)
        full_context = clean_text(full_source.get("context", ""))
        chunks = split_chunks(full_context, args.chunk_chars, args.chunk_overlap)
        evidence = select_question_windows(chunks, row.get("question", ""), args.top_k)
        text_index = EvidenceIndex.build(evidence)
        agent = {
            "enabled": True,
            "version": "v47",
            "attempted": True,
            "action": "retain_current",
            "accepted": False,
            "evidence_windows": len(evidence),
            "selected_evidence": evidence,
            "reference_answer_used": False,
        }
        try:
            candidate = await complete_answer(client, model, row.get("question", ""), row.get("prediction", ""), evidence)
            agent["candidate_answer"] = candidate
            accepted, guard = guard_candidate(
                candidate,
                row.get("prediction", ""),
                row.get("question", ""),
                text_index,
                args.min_term_coverage,
            )
            agent["guard"] = guard
            agent["accepted"] = accepted
            if accepted:
                new_row["prediction_before_question_anchored_repair"] = new_row.get("prediction", "")
                new_row["prediction"] = candidate
                new_row["method"] = "sage_v47_question_anchored_repair"
                agent["action"] = "use_question_anchored_answer"
            elif row.get("prediction_before_text_evidence_expansion_fallback"):
                new_row["prediction_before_question_anchored_repair"] = new_row.get("prediction", "")
                new_row["prediction"] = row.get("prediction_before_text_evidence_expansion_fallback", "")
                new_row["method"] = "sage_v47_revert_failed_expansion"
                agent["action"] = "revert_failed_expansion"
        except Exception as exc:
            agent["error"] = f"{type(exc).__name__}: {exc}"[:300]
            agent["action"] = "error_keep_current"

        new_row["question_anchored_repair"] = agent
        output_rows.append(new_row)

    output_path = ROOT_DIR / args.output
    write_jsonl(output_rows, output_path)
    write_summary(ROOT_DIR / args.summary, output_rows, ROOT_DIR / args.input)
    print(f"Saved v47 question-anchored repair results to {output_path}")
    print(f"Saved summary to {ROOT_DIR / args.summary}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
