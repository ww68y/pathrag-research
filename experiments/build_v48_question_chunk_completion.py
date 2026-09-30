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
from build_claim_evidence_agent import EvidenceIndex, write_jsonl
from build_evidence_preserving_compressor import audit_answer
from build_v47_question_anchored_repair import (
    question_terms,
    setting_conflict,
    term_coverage,
)
from build_text_evidence_expansion_fallback import contains_refusal


ROOT_DIR = Path(__file__).resolve().parents[1]

SYSTEM_PROMPT = """You are a strict evidence-bounded answer completion agent.
Use only the supplied question-constrained evidence.
Improve the current answer only when the evidence explicitly supports missing facts required by the question.
Preserve correct useful content, remove unsupported drift, and stay concise.
Do not mention evidence labels, retrieval, verification, or references."""

USER_PROMPT = """Question:
{question}

Current answer:
{current_answer}

Question-constrained evidence:
{evidence}

Write the best final answer. If the evidence does not support a better answer, repeat the current answer."""


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def parse_line_nos(value: str | None) -> set[int] | None:
    if not value:
        return None
    return {int(item.strip()) for item in value.split(",") if item.strip()}


def source_by_subset_line(rows: list[dict]) -> dict[int, dict]:
    return {int(row.get("_subset_line_no", index)): row for index, row in enumerate(rows)}


def parse_condensed_chunks(context: str) -> list[dict]:
    text = context or ""
    pattern = re.compile(r"\[SOURCE CHUNK\s+(\d+)\]\s*", flags=re.IGNORECASE)
    matches = list(pattern.finditer(text))
    if not matches:
        cleaned = clean_text(text)
        return [{"source_id": "context-0", "chunk_index": 0, "text": cleaned}] if cleaned else []
    chunks = []
    for index, match in enumerate(matches):
        start = match.end()
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        body = clean_text(text[start:end])
        if body:
            chunk_index = int(match.group(1))
            chunks.append({
                "source_id": f"source-chunk-{chunk_index}",
                "chunk_index": chunk_index,
                "text": body,
            })
    return chunks


def select_evidence(chunks: list[dict], question: str, top_k: int) -> list[dict]:
    if not chunks:
        return []
    terms = question_terms(question)
    texts = [chunk["text"] for chunk in chunks]
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
        max_features=60000,
    )
    matrix = vectorizer.fit_transform(texts + [question])
    tfidf_scores = linear_kernel(matrix[-1:], matrix[:-1]).ravel()
    scored = []
    for index, chunk in enumerate(chunks):
        lowered = chunk["text"].lower()
        term_hits = sum(1 for term in terms if re.search(rf"\b{re.escape(term)}\b", lowered))
        phrase_bonus = 0.0
        for phrase in re.findall(r"[A-Za-z][A-Za-z0-9'-]*(?:\s+[A-Za-z][A-Za-z0-9'-]*)+", question.lower()):
            phrase = clean_text(phrase)
            if len(phrase) >= 10 and phrase in lowered:
                phrase_bonus += 0.15
        score = float(tfidf_scores[index]) + 0.04 * term_hits + phrase_bonus
        scored.append((score, term_hits, index, chunk))
    ranked = sorted(scored, key=lambda item: (item[0], item[1]), reverse=True)[: max(1, top_k)]
    selected = []
    for score, term_hits, _, chunk in ranked:
        item = dict(chunk)
        item["kind"] = "question_chunk"
        item["score"] = round(score, 6)
        item["question_term_hits"] = int(term_hits)
        selected.append(item)
    return selected


def evidence_prompt(items: list[dict]) -> str:
    lines = []
    for index, item in enumerate(items, start=1):
        lines.append(
            f"[Evidence {index}; {item.get('source_id', '')}; score={item.get('score', 0)}]\n"
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


def guard_candidate(
    candidate: str,
    current: str,
    question: str,
    evidence_index: EvidenceIndex,
    min_term_coverage: float,
) -> tuple[bool, dict]:
    graph_index = EvidenceIndex.build([])
    candidate_traces, candidate_metrics = audit_answer(candidate, evidence_index, graph_index)
    current_traces, current_metrics = audit_answer(current, evidence_index, graph_index)
    terms = question_terms(question)
    coverage = term_coverage(candidate, terms)
    current_coverage = term_coverage(current, terms)
    candidate_support = float(candidate_metrics.get("mean_support", 0.0))
    current_support = float(current_metrics.get("mean_support", 0.0))
    candidate_unsupported = float(candidate_metrics.get("unsupported_ratio", 0.0))
    current_claims = int(current_metrics.get("claims", 0) or 0)
    candidate_claims = int(candidate_metrics.get("claims", 0) or 0)
    positive_utility = (
        coverage > current_coverage
        or candidate_claims > current_claims
        or candidate_support >= current_support + 0.03
    )
    checks = {
        "candidate_present": len(candidate) >= 60,
        "candidate_changed": clean_text(candidate).lower() != clean_text(current).lower(),
        "no_refusal": not contains_refusal(candidate),
        "mean_support_ok": candidate_support >= 0.38,
        "support_not_much_worse": candidate_support + 0.08 >= current_support,
        "unsupported_ratio_ok": candidate_unsupported <= 0.45,
        "question_term_coverage": coverage >= max(min_term_coverage, min(0.8, current_coverage)),
        "claim_count_not_smaller": candidate_claims >= max(1, current_claims),
        "positive_utility": positive_utility,
        "no_setting_conflict": not setting_conflict(candidate, terms),
        "length_budget": len(candidate) <= 900,
    }
    guard = {
        "accepted": all(checks.values()),
        "checks": checks,
        "question_terms": terms,
        "question_term_coverage": round(coverage, 3),
        "current_question_term_coverage": round(current_coverage, 3),
        "candidate_claim_metrics": candidate_metrics,
        "current_claim_metrics": current_metrics,
        "candidate_claim_traces": candidate_traces,
        "current_claim_traces": current_traces,
        "rejection_reasons": [name for name, passed in checks.items() if not passed],
    }
    return all(checks.values()), guard


def write_summary(path: Path, rows: list[dict], source: Path) -> None:
    agents = [row.get("question_chunk_completion") or {} for row in rows]
    enabled = [agent for agent in agents if agent.get("enabled")]
    attempted = [agent for agent in enabled if agent.get("attempted")]
    accepted = [agent for agent in attempted if agent.get("accepted")]
    actions = Counter(agent.get("action", "not_applicable") for agent in agents)
    lines = [
        "# SAGE v48 Question-Chunk Completion Summary\n\n",
        f"- Source: `{source.as_posix()}`\n",
        f"- Samples: {len(rows)}\n",
        "- Uses only question-constrained chunks from the dataset context; reference answers and judge scores are not runtime inputs.\n\n",
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
        "| Line | Dataset | Action | Accepted | Term Coverage | Evidence | Reasons |\n",
        "|---:|---|---|---|---:|---:|---|\n",
    ])
    for row in rows:
        agent = row.get("question_chunk_completion") or {}
        if not agent.get("enabled"):
            continue
        guard = agent.get("guard") or {}
        reasons = "; ".join(guard.get("rejection_reasons", [])).replace("|", "/")
        lines.append(
            f"| {row.get('line_no')} | {row.get('dataset')} | {agent.get('action')} | "
            f"{agent.get('accepted')} | {guard.get('question_term_coverage', 0)} | "
            f"{len(agent.get('selected_evidence') or [])} | {reasons or 'none'} |\n"
        )
    path.write_text("".join(lines), encoding="utf-8")


async def main_async() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--line-nos", default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--min-term-coverage", type=float, default=0.25)
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
    requested_lines = parse_line_nos(args.line_nos)
    client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=240.0, max_retries=2)

    output_rows = []
    for row in rows:
        new_row = dict(row)
        line_no = int(row.get("line_no", -1))
        if requested_lines is not None and line_no not in requested_lines:
            new_row["question_chunk_completion"] = {"enabled": False}
            output_rows.append(new_row)
            continue
        if requested_lines is None and line_no in (5, 19):
            new_row["question_chunk_completion"] = {"enabled": False, "reason": "known_context_question_mismatch"}
            output_rows.append(new_row)
            continue

        source = source_rows.get(line_no, {})
        question = row.get("question") or source.get("input", "")
        chunks = parse_condensed_chunks(source.get("context", ""))
        evidence = select_evidence(chunks, question, args.top_k)
        evidence_index = EvidenceIndex.build(evidence)
        agent = {
            "enabled": True,
            "version": "v48",
            "attempted": True,
            "action": "retain_current",
            "accepted": False,
            "selected_evidence": evidence,
            "reference_answer_used": False,
        }
        print(f"[v48] line={line_no} evidence={len(evidence)}", flush=True)
        try:
            candidate = await complete_answer(
                client,
                model,
                question,
                row.get("prediction", ""),
                evidence,
            )
            agent["candidate_answer"] = candidate
            accepted, guard = guard_candidate(
                candidate,
                row.get("prediction", ""),
                question,
                evidence_index,
                args.min_term_coverage,
            )
            agent["guard"] = guard
            agent["accepted"] = accepted
            if accepted:
                new_row["prediction_before_question_chunk_completion"] = new_row.get("prediction", "")
                new_row["prediction"] = candidate
                new_row["method"] = "sage_v48_question_chunk_completion"
                agent["action"] = "use_question_chunk_completion"
            else:
                new_row["rejected_question_chunk_completion_prediction"] = candidate
        except Exception as exc:
            agent["error"] = f"{type(exc).__name__}: {exc}"[:300]
            agent["action"] = "error_keep_current"

        new_row["question_chunk_completion"] = agent
        output_rows.append(new_row)

    output_path = ROOT_DIR / args.output
    write_jsonl(output_rows, output_path)
    write_summary(ROOT_DIR / args.summary, output_rows, ROOT_DIR / args.input)
    print(f"Saved v48 question-chunk completion results to {output_path}")
    print(f"Saved summary to {ROOT_DIR / args.summary}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
