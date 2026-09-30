from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_answer_contract_agent import clean_text, derive_answer_contract, parse_json_object
from build_claim_evidence_agent import EvidenceIndex, content_tokens, load_full_source_rows, normalize_space, write_jsonl
from build_evidence_preserving_compressor import audit_answer
from build_text_evidence_expansion_fallback import contains_refusal, parse_line_nos, sanitize_evidence_labels
from build_v52_query_expanded_refusal_repair import question_terms, select_expanded_windows, split_chunks
from build_v55_supported_facet_guard import supported_facet_stats


ROOT_DIR = Path(__file__).resolve().parents[1]

EDITOR_SYSTEM = """You are a reference-free residual hard-case editor for retrieval-augmented QA.
The current answer may be incomplete, overly verbose, slightly off-target, or contain a small unsupported detail.
Use only the supplied evidence and current answer. Do not use outside knowledge or reference answers.
Your edit should be conservative: fix clear gaps or verbosity, but keep a good answer unchanged."""

EDITOR_USER = """Question:
{question}

Answer contract:
{contract}

Current answer:
{answer}

Runtime issue hint:
{issue_hint}

Evidence:
{evidence}

Decide whether the answer should be edited.

Return JSON only:
{{
  "edit_type": "keep",
  "missing_supported_facts": ["facts supported by evidence but missing from the current answer"],
  "unsupported_or_overlong_details": ["details to remove or shorten"],
  "reason": "short reason",
  "edited_answer": "final answer; if edit_type is keep, repeat the current answer exactly"
}}

Allowed edit_type values: keep, repair_missing, compact_verbose, repair_and_compact."""

FALLBACK_SYSTEM = """You are a concise evidence-grounded QA editor.
Use only the supplied evidence and current answer. Do not use outside knowledge.
Write an improved final answer if the evidence supports one; otherwise copy the current answer."""

FALLBACK_USER = """Question:
{question}

Current answer:
{answer}

Evidence:
{evidence}

Write the best final answer in 2-5 concise sentences. Do not cite evidence labels."""


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def source_by_subset_line(rows: list[dict]) -> dict[int, dict]:
    return {int(row.get("_subset_line_no", index)): row for index, row in enumerate(rows)}


def evidence_prompt(items: list[dict], limit: int = 5) -> str:
    lines = []
    for index, item in enumerate(items[:limit], start=1):
        lines.append(
            f"[Evidence {index}; {item.get('source_id', '')}; score={item.get('score', 0)}]\n"
            f"{item.get('text', '')}"
        )
    return "\n\n".join(lines)


def merged_evidence(row: dict, dataset_source: dict, full_source: dict, top_k: int) -> list[dict]:
    existing = (row.get("query_expanded_contract_gap_repair") or {}).get("selected_evidence") or []
    full_context = clean_text(full_source.get("context", "") or dataset_source.get("context", ""))
    chunks = split_chunks(full_context, 1600, 220)
    expansion = {
        "search_phrases": [row.get("question", "")],
        "must_cover_terms": question_terms(row.get("question", "")),
    }
    selected = select_expanded_windows(chunks, row.get("question", ""), row.get("prediction", ""), expansion, top_k)
    records = []
    seen = set()
    for item in existing + selected:
        text = clean_text(item.get("text", ""))
        if not text or text in seen:
            continue
        seen.add(text)
        records.append(item)
    return records[: max(top_k * 3, 8)]


def issue_hint(row: dict) -> str:
    answer = clean_text(row.get("prediction", ""))
    contract = derive_answer_contract(row.get("question", ""))
    hints = [f"contract_type={contract.get('type')}", f"answer_chars={len(answer)}"]
    v54 = row.get("query_expanded_contract_gap_repair") or {}
    verification = v54.get("verification") or {}
    if verification:
        hints.append(f"previous_contract_coverage={verification.get('contract_coverage')}")
        missing = verification.get("missing_supported_facts") or []
        if missing:
            hints.append("previous_missing_facts=" + "; ".join(clean_text(x) for x in missing[:4]))
    if len(answer) > 900:
        hints.append("long_answer_possible_verbosity")
    terms = question_terms(row.get("question", ""))
    if terms:
        hit = sum(1 for term in terms if term.lower() in answer.lower())
        hints.append(f"question_term_hits={hit}/{len(terms)}")
    return " | ".join(hints)


async def edit_answer(client: AsyncOpenAI, model: str, row: dict, evidence: list[dict]) -> dict:
    try:
        response = await client.chat.completions.create(
            model=model,
            temperature=0,
            max_tokens=900,
            messages=[
                {"role": "system", "content": EDITOR_SYSTEM},
                {
                    "role": "user",
                    "content": EDITOR_USER.format(
                        question=row.get("question", ""),
                        contract=json.dumps(derive_answer_contract(row.get("question", "")), ensure_ascii=False),
                        answer=row.get("prediction", ""),
                        issue_hint=issue_hint(row),
                        evidence=evidence_prompt(evidence, limit=5),
                    ),
                },
            ],
        )
        parsed = parse_json_object(response.choices[0].message.content or "")
    except Exception as exc:
        response = await client.chat.completions.create(
            model=model,
            temperature=0,
            max_tokens=520,
            messages=[
                {"role": "system", "content": FALLBACK_SYSTEM},
                {
                    "role": "user",
                    "content": FALLBACK_USER.format(
                        question=row.get("question", ""),
                        answer=row.get("prediction", ""),
                        evidence=evidence_prompt(evidence, limit=3),
                    ),
                },
            ],
        )
        edited = sanitize_evidence_labels(clean_text(response.choices[0].message.content or ""))
        return {
            "edit_type": "repair_and_compact",
            "missing_supported_facts": [],
            "unsupported_or_overlong_details": ["fallback edit after primary editor failure"],
            "reason": f"fallback_after_{type(exc).__name__}",
            "edited_answer": edited,
        }
    missing = parsed.get("missing_supported_facts")
    unsupported = parsed.get("unsupported_or_overlong_details")
    edited = sanitize_evidence_labels(clean_text(parsed.get("edited_answer", "")))
    parsed["edit_type"] = clean_text(parsed.get("edit_type", "keep")) or "keep"
    parsed["missing_supported_facts"] = (
        [clean_text(item) for item in missing if clean_text(item)] if isinstance(missing, list) else []
    )
    parsed["unsupported_or_overlong_details"] = (
        [clean_text(item) for item in unsupported if clean_text(item)] if isinstance(unsupported, list) else []
    )
    parsed["reason"] = clean_text(parsed.get("reason", ""))
    parsed["edited_answer"] = edited
    return parsed


def fact_coverage(facts: list[str], answer: str) -> float:
    if not facts:
        return 1.0
    answer_tokens = content_tokens(answer)
    scores = []
    for fact in facts:
        tokens = content_tokens(fact)
        scores.append(len(tokens & answer_tokens) / max(1, len(tokens)))
    return sum(scores) / len(scores)


def term_coverage(question: str, answer: str) -> float:
    terms = question_terms(question)
    if not terms:
        return 1.0
    lowered = answer.lower()
    return sum(1 for term in terms if term.lower() in lowered) / len(terms)


def guard_edit(row: dict, candidate: str, decision: dict, evidence: list[dict]) -> tuple[bool, dict]:
    current = row.get("prediction", "")
    text_index = EvidenceIndex.build(evidence)
    graph_index = EvidenceIndex.build([])
    current_traces, current_metrics = audit_answer(current, text_index, graph_index)
    candidate_traces, candidate_metrics = audit_answer(candidate, text_index, graph_index)
    current_len = len(clean_text(current))
    candidate_len = len(clean_text(candidate))
    reduction = 1 - candidate_len / max(1, current_len)
    missing = decision.get("missing_supported_facts") or []
    unsupported = decision.get("unsupported_or_overlong_details") or []
    missing_cov = fact_coverage(missing, candidate)
    current_term_cov = term_coverage(row.get("question", ""), current)
    candidate_term_cov = term_coverage(row.get("question", ""), candidate)
    facet_stats = supported_facet_stats(current, candidate, evidence)
    current_support = float(current_metrics.get("mean_support", 0.0))
    candidate_support = float(candidate_metrics.get("mean_support", 0.0))
    current_unsupported = float(current_metrics.get("unsupported_ratio", 0.0))
    candidate_unsupported = float(candidate_metrics.get("unsupported_ratio", 0.0))
    edit_type = decision.get("edit_type", "keep")
    changed = clean_text(candidate).lower() != clean_text(current).lower()
    utility = (
        bool(missing and missing_cov >= 0.38)
        or bool(unsupported and reduction >= 0.10)
        or candidate_support >= current_support + 0.04
        or (current_len > 900 and reduction >= 0.12)
    )
    is_compression = candidate_len < current_len
    term_guard = candidate_term_cov + 0.25 >= current_term_cov
    if not is_compression:
        term_guard = term_guard or missing_cov >= 0.65
    compression_guard = True
    if candidate_len < current_len * 0.98 and facet_stats["supported_before_terms"] >= 35:
        compression_guard = facet_stats["supported_facet_retention"] >= 0.35
    if candidate_len < current_len * 0.90 and facet_stats["supported_before_terms"] >= 25:
        compression_guard = compression_guard and facet_stats["supported_facet_retention"] >= 0.50
    checks = {
        "editor_requested_edit": edit_type != "keep",
        "candidate_present": candidate_len >= 80,
        "candidate_changed": changed,
        "candidate_not_refusal": not contains_refusal(candidate),
        "question_terms_preserved": term_guard,
        "missing_facts_covered": missing_cov >= (0.38 if missing else 1.0),
        "support_preserved": candidate_support >= max(0.25, current_support - 0.18),
        "unsupported_ratio_controlled": candidate_unsupported <= max(0.58, current_unsupported + 0.18),
        "not_too_long": candidate_len <= max(1500, int(current_len * 1.25)),
        "supported_facets_preserved": compression_guard,
        "positive_utility": utility,
    }
    guard = {
        "accepted": all(checks.values()),
        "checks": checks,
        "current_chars": current_len,
        "candidate_chars": candidate_len,
        "length_reduction": round(reduction, 6),
        "missing_fact_coverage": round(missing_cov, 6),
        "current_question_term_coverage": round(current_term_cov, 6),
        "candidate_question_term_coverage": round(candidate_term_cov, 6),
        "current_claim_metrics": current_metrics,
        "candidate_claim_metrics": candidate_metrics,
        "current_claim_traces": current_traces,
        "candidate_claim_traces": candidate_traces,
        "supported_facet_stats": facet_stats,
        "rejection_reasons": [name for name, passed in checks.items() if not passed],
    }
    return all(checks.values()), guard


def should_attempt(row: dict, requested_lines: set[int] | None) -> bool:
    line_no = int(row.get("line_no", -1))
    if requested_lines is not None:
        return line_no in requested_lines
    answer = clean_text(row.get("prediction", ""))
    contract_type = derive_answer_contract(row.get("question", "")).get("type")
    if len(answer) > 900:
        return True
    if contract_type in {"enumeration", "comparison", "mechanism", "procedure", "compound"} and len(answer) > 350:
        return True
    v54 = row.get("query_expanded_contract_gap_repair") or {}
    verification = v54.get("verification") or {}
    return bool((verification.get("missing_supported_facts") or []) and not v54.get("accepted"))


def write_summary(path: Path, rows: list[dict], source: Path) -> None:
    agents = [row.get("residual_hardcase_editor") or {} for row in rows]
    enabled = [agent for agent in agents if agent.get("enabled")]
    accepted = [agent for agent in enabled if agent.get("accepted")]
    actions = Counter(agent.get("action", "not_applicable") for agent in agents)
    edit_types = Counter((agent.get("decision") or {}).get("edit_type", "none") for agent in enabled)
    lines = [
        "# SAGE v58 Residual Hard-Case Editor Summary\n\n",
        f"- Source: `{source.as_posix()}`\n",
        f"- Samples: {len(rows)}\n",
        "- Purpose: batch-edit residual hard cases with local full-text evidence retrieval and one conservative LLM edit call.\n",
        "- Reference answers and judge scores are not runtime inputs.\n\n",
        "| Metric | Count |\n",
        "|---|---:|\n",
        f"| Enabled | {len(enabled)} |\n",
        f"| Accepted edits | {len(accepted)} |\n\n",
        "## Actions\n\n",
        "| Action | Count |\n",
        "|---|---:|\n",
    ]
    for action, count in sorted(actions.items()):
        lines.append(f"| {action} | {count} |\n")
    lines.extend(["\n## Edit Types\n\n", "| Edit Type | Count |\n", "|---|---:|\n"])
    for edit_type, count in sorted(edit_types.items()):
        lines.append(f"| {edit_type} | {count} |\n")
    lines.extend(
        [
            "\n## Per-Sample Decisions\n\n",
            "| Line | Dataset | Edit Type | Action | Accepted | Chars | Missing Coverage | Reasons |\n",
            "|---:|---|---|---|---|---:|---:|---|\n",
        ]
    )
    for row in rows:
        agent = row.get("residual_hardcase_editor") or {}
        if not agent.get("enabled"):
            continue
        guard = agent.get("guard") or {}
        decision = agent.get("decision") or {}
        reasons = "; ".join(guard.get("rejection_reasons") or []).replace("|", "/") or "none"
        lines.append(
            f"| {row.get('line_no')} | {row.get('dataset')} | {decision.get('edit_type')} | "
            f"{agent.get('action')} | {agent.get('accepted')} | "
            f"{guard.get('current_chars', 0)} -> {guard.get('candidate_chars', 0)} | "
            f"{guard.get('missing_fact_coverage', 0)} | {reasons} |\n"
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
    client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=45.0, max_retries=0)
    output_rows = []
    for row in rows:
        new_row = dict(row)
        if not should_attempt(row, requested_lines):
            new_row["residual_hardcase_editor"] = {"enabled": False}
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
        evidence = merged_evidence(row, source, full_source, args.top_k)
        agent = {
            "enabled": True,
            "version": "v58b",
            "accepted": False,
            "action": "retain_current",
            "reference_answer_used": False,
            "issue_hint": issue_hint(row),
            "selected_evidence": evidence,
        }
        try:
            print(f"[v58] line={line_no} edit evidence={len(evidence)}", flush=True)
            decision = await edit_answer(client, model, row, evidence)
            candidate = decision.get("edited_answer", "")
            accepted, guard = guard_edit(row, candidate, decision, evidence)
            agent["decision"] = decision
            agent["candidate_answer"] = candidate
            agent["guard"] = guard
            agent["accepted"] = accepted
            print(f"[v58] line={line_no} edit_type={decision.get('edit_type')} accepted={accepted}", flush=True)
            if accepted:
                new_row["prediction_before_residual_hardcase_editor"] = new_row.get("prediction", "")
                new_row["prediction"] = candidate
                new_row["method"] = "sage_v58_residual_hardcase_editor"
                agent["action"] = "use_residual_hardcase_edit"
            else:
                new_row["rejected_residual_hardcase_edit"] = candidate
        except Exception as exc:
            agent["error"] = f"{type(exc).__name__}: {exc}"[:300]
            agent["action"] = "error_keep_current"
        new_row["residual_hardcase_editor"] = agent
        output_rows.append(new_row)

    output_path = ROOT_DIR / args.output
    write_jsonl(output_rows, output_path)
    write_summary(ROOT_DIR / args.summary, output_rows, ROOT_DIR / args.input)
    print(f"Saved v58 residual hard-case results to {output_path}")
    print(f"Saved summary to {ROOT_DIR / args.summary}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
