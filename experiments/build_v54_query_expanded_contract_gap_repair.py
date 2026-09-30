from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

from build_answer_contract_agent import clean_text, derive_answer_contract, parse_json_object
from build_claim_evidence_agent import EvidenceIndex, content_tokens, load_full_source_rows, normalize_space, write_jsonl
from build_evidence_preserving_compressor import audit_answer
from build_text_evidence_expansion_fallback import contains_refusal, load_jsonl, parse_line_nos, sanitize_evidence_labels
from build_v52_query_expanded_refusal_repair import expand_query, question_terms, select_expanded_windows, split_chunks


ROOT_DIR = Path(__file__).resolve().parents[1]

VERIFY_SYSTEM = """You are a strict answer-contract gap verifier for retrieval-augmented QA.
Use only the supplied query-expanded evidence. Do not use outside knowledge or reference answers.
The draft answer may be partially correct but incomplete, over-detailed, or slightly off target.
Return JSON only."""

VERIFY_USER = """Question:
{question}

Question-derived answer contract:
{contract}

Current draft answer:
{answer}

Query-expanded evidence:
{evidence}

Check the current answer:
1. List concrete answer-essential facts already covered.
2. List concrete answer-essential facts missing from the answer but explicitly supported by the evidence.
3. List unsupported or off-topic claims that should be removed.
4. Estimate contract coverage and evidence grounding from 0.0 to 1.0.
5. Request repair only if the evidence can produce a more complete or more concise answer.

Return this exact JSON shape:
{{
  "covered_facts": ["..."],
  "missing_supported_facts": ["..."],
  "unsupported_or_off_topic_claims": ["..."],
  "contract_coverage": 0.0,
  "evidence_grounding": 0.0,
  "repair_needed": false,
  "repair_goal": "add_missing_facts",
  "reason": "short reason"
}}"""

REPAIR_SYSTEM = """You are an evidence-bounded answer repair agent.
Rewrite the answer using only the supplied evidence. Preserve correct useful content, add verified missing facts,
remove unsupported or off-topic details, and satisfy the answer contract.
Give the direct answer first. Stay concise, but do not omit answer-essential facts.
Do not cite evidence labels, source ids, references, or verification."""

REPAIR_USER = """Question:
{question}

Answer contract:
{contract}

Current answer:
{answer}

Verified missing supported facts:
{missing}

Unsupported or off-topic claims to remove:
{unsupported}

Evidence:
{evidence}

Write the final answer in 2-5 concise sentences."""


def source_by_subset_line(rows: list[dict]) -> dict[int, dict]:
    return {int(row.get("_subset_line_no", index)): row for index, row in enumerate(rows)}


def evidence_prompt(items: list[dict], limit: int = 10) -> str:
    lines = []
    for index, item in enumerate(items[:limit], start=1):
        lines.append(
            f"[Evidence {index}; {item.get('source_id', '')}; score={item.get('score', 0)}]\n"
            f"{item.get('text', '')}"
        )
    return "\n\n".join(lines)


def clamp01(value: object, default: float = 0.0) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def normalize_verification(raw: dict) -> dict:
    missing = raw.get("missing_supported_facts")
    unsupported = raw.get("unsupported_or_off_topic_claims")
    covered = raw.get("covered_facts")
    return {
        "covered_facts": [clean_text(item) for item in covered if clean_text(item)] if isinstance(covered, list) else [],
        "missing_supported_facts": (
            [clean_text(item) for item in missing if clean_text(item)] if isinstance(missing, list) else []
        ),
        "unsupported_or_off_topic_claims": (
            [clean_text(item) for item in unsupported if clean_text(item)] if isinstance(unsupported, list) else []
        ),
        "contract_coverage": round(clamp01(raw.get("contract_coverage"), 0.5), 6),
        "evidence_grounding": round(clamp01(raw.get("evidence_grounding"), 0.5), 6),
        "repair_needed": bool(raw.get("repair_needed")),
        "repair_goal": clean_text(raw.get("repair_goal") or "add_missing_facts"),
        "reason": clean_text(raw.get("reason")),
    }


async def verify_gap(
    client: AsyncOpenAI,
    model: str,
    row: dict,
    contract: dict,
    evidence: list[dict],
) -> dict:
    response = await client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": VERIFY_SYSTEM},
            {
                "role": "user",
                "content": VERIFY_USER.format(
                    question=row.get("question", ""),
                    contract=json.dumps(contract, ensure_ascii=False),
                    answer=row.get("prediction", ""),
                    evidence=evidence_prompt(evidence),
                ),
            },
        ],
    )
    return normalize_verification(parse_json_object(response.choices[0].message.content or ""))


async def generate_repair(
    client: AsyncOpenAI,
    model: str,
    row: dict,
    contract: dict,
    verification: dict,
    evidence: list[dict],
) -> str:
    response = await client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": REPAIR_SYSTEM},
            {
                "role": "user",
                "content": REPAIR_USER.format(
                    question=row.get("question", ""),
                    contract=json.dumps(contract, ensure_ascii=False),
                    answer=row.get("prediction", ""),
                    missing="\n".join(f"- {item}" for item in verification.get("missing_supported_facts", [])) or "none",
                    unsupported="\n".join(
                        f"- {item}" for item in verification.get("unsupported_or_off_topic_claims", [])
                    ) or "none",
                    evidence=evidence_prompt(evidence),
                ),
            },
        ],
    )
    return sanitize_evidence_labels(response.choices[0].message.content or "")


def missing_fact_coverage(missing: list[str], candidate: str) -> float:
    if not missing:
        return 1.0
    candidate_tokens = content_tokens(candidate)
    scores = []
    for fact in missing:
        fact_tokens = content_tokens(fact)
        scores.append(len(fact_tokens & candidate_tokens) / max(1, len(fact_tokens)))
    return sum(scores) / len(scores)


def unsupported_claim_retention(unsupported: list[str], candidate: str) -> float:
    if not unsupported:
        return 0.0
    candidate_tokens = content_tokens(candidate)
    scores = []
    for claim in unsupported:
        claim_tokens = content_tokens(claim)
        scores.append(len(claim_tokens & candidate_tokens) / max(1, len(claim_tokens)))
    return sum(scores) / len(scores)


def term_coverage(answer: str, terms: list[str]) -> float:
    if not terms:
        return 1.0
    answer_tokens = content_tokens(answer)
    hits = sum(1 for term in terms if term.lower() in answer_tokens)
    return hits / len(terms)


def guard_candidate(
    row: dict,
    candidate: str,
    verification: dict,
    evidence: list[dict],
) -> tuple[bool, dict]:
    text_index = EvidenceIndex.build(evidence)
    graph_index = EvidenceIndex.build([])
    current = row.get("prediction", "")
    candidate_traces, candidate_metrics = audit_answer(candidate, text_index, graph_index)
    current_traces, current_metrics = audit_answer(current, text_index, graph_index)
    missing = verification.get("missing_supported_facts") or []
    unsupported = verification.get("unsupported_or_off_topic_claims") or []
    fact_coverage = missing_fact_coverage(missing, candidate)
    unsupported_retention = unsupported_claim_retention(unsupported, candidate)
    terms = question_terms(row.get("question", ""))
    candidate_term_coverage = term_coverage(candidate, terms)
    current_term_coverage = term_coverage(current, terms)
    candidate_support = float(candidate_metrics.get("mean_support", 0.0))
    current_support = float(current_metrics.get("mean_support", 0.0))
    candidate_unsupported = float(candidate_metrics.get("unsupported_ratio", 0.0))
    current_unsupported = float(current_metrics.get("unsupported_ratio", 0.0))
    current_len = len(clean_text(current))
    candidate_len = len(clean_text(candidate))
    missing_or_cleanup = bool(missing or unsupported)
    concise_cleanup = (
        verification.get("repair_goal") == "remove_off_topic_or_overlong_details"
        or (current_len > 700 and candidate_len < current_len)
    )
    positive_utility = (
        (missing and fact_coverage >= 0.45)
        or (unsupported and unsupported_retention <= 0.55)
        or concise_cleanup
        or candidate_support >= current_support + 0.03
    )
    term_coverage_ok = (
        candidate_term_coverage + 0.10 >= current_term_coverage
        or (
            current_len > 700
            and candidate_term_coverage >= 0.35
            and fact_coverage >= 0.45
            and unsupported_retention <= 0.55
        )
    )
    checks = {
        "candidate_present": candidate_len >= 70,
        "candidate_changed": clean_text(candidate).lower() != clean_text(current).lower(),
        "no_refusal": not contains_refusal(candidate),
        "verifier_requested_repair": bool(verification.get("repair_needed")) and missing_or_cleanup,
        "missing_facts_added": fact_coverage >= (0.45 if missing else 1.0),
        "unsupported_claims_reduced": unsupported_retention <= (0.65 if unsupported else 0.0),
        "support_ok": candidate_support >= max(0.32, current_support - 0.12),
        "unsupported_ratio_ok": candidate_unsupported <= max(0.55, current_unsupported + 0.12),
        "term_coverage_not_worse": term_coverage_ok,
        "answer_not_too_long": candidate_len <= max(1000, int(current_len * 1.15)),
        "positive_utility": positive_utility,
    }
    guard = {
        "accepted": all(checks.values()),
        "checks": checks,
        "missing_fact_coverage": round(fact_coverage, 6),
        "unsupported_claim_retention": round(unsupported_retention, 6),
        "question_terms": terms,
        "current_question_term_coverage": round(current_term_coverage, 3),
        "candidate_question_term_coverage": round(candidate_term_coverage, 3),
        "current_chars": current_len,
        "candidate_chars": candidate_len,
        "current_claim_metrics": current_metrics,
        "candidate_claim_metrics": candidate_metrics,
        "current_claim_traces": current_traces,
        "candidate_claim_traces": candidate_traces,
        "rejection_reasons": [name for name, passed in checks.items() if not passed],
    }
    return all(checks.values()), guard


def should_attempt(row: dict, requested_lines: set[int] | None) -> bool:
    line_no = int(row.get("line_no", -1))
    if requested_lines is not None:
        return line_no in requested_lines
    answer = clean_text(row.get("prediction", ""))
    contract = derive_answer_contract(row.get("question", ""))
    if contains_refusal(answer):
        return False
    if len(answer) > 700:
        return True
    return contract.get("type") in {"comparison", "enumeration", "procedure", "mechanism", "compound"}


def write_summary(path: Path, rows: list[dict], source: Path) -> None:
    agents = [row.get("query_expanded_contract_gap_repair") or {} for row in rows]
    enabled = [agent for agent in agents if agent.get("enabled")]
    attempted = [agent for agent in enabled if agent.get("attempted")]
    accepted = [agent for agent in attempted if agent.get("accepted")]
    actions = Counter(agent.get("action", "not_applicable") for agent in agents)
    repair_goals = Counter((agent.get("verification") or {}).get("repair_goal", "none") for agent in attempted)
    lines = [
        "# SAGE v54 Query-Expanded Contract-Gap Repair Summary\n\n",
        f"- Source: `{source.as_posix()}`\n",
        f"- Samples: {len(rows)}\n",
        "- Uses question-only query expansion, full-source text evidence, contract-gap verification, and evidence guards.\n",
        "- Reference answers and judge scores are not runtime inputs.\n\n",
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
    lines.extend(["\n## Repair Goals\n\n", "| Goal | Count |\n", "|---|---:|\n"])
    for goal, count in sorted(repair_goals.items()):
        lines.append(f"| {goal} | {count} |\n")
    lines.extend([
        "\n## Per-Sample Decisions\n\n",
        "| Line | Dataset | Action | Accepted | Coverage | Grounding | Missing | Reasons |\n",
        "|---:|---|---|---|---:|---:|---:|---|\n",
    ])
    for row in rows:
        agent = row.get("query_expanded_contract_gap_repair") or {}
        if not agent.get("enabled"):
            continue
        verification = agent.get("verification") or {}
        guard = agent.get("guard") or {}
        reasons = "; ".join(guard.get("rejection_reasons", [])).replace("|", "/")
        lines.append(
            f"| {row.get('line_no')} | {row.get('dataset')} | {agent.get('action')} | "
            f"{agent.get('accepted')} | {verification.get('contract_coverage', 0)} | "
            f"{verification.get('evidence_grounding', 0)} | "
            f"{len(verification.get('missing_supported_facts') or [])} | {reasons or 'none'} |\n"
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
    parser.add_argument("--top-k", type=int, default=6)
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
    client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=120.0, max_retries=1)

    output_rows = []
    for row in rows:
        new_row = dict(row)
        if not should_attempt(row, requested_lines):
            new_row["query_expanded_contract_gap_repair"] = {"enabled": False}
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
        contract = derive_answer_contract(row.get("question", ""))
        agent = {
            "enabled": True,
            "version": "v54",
            "attempted": True,
            "action": "retain_current",
            "accepted": False,
            "contract": contract,
            "reference_answer_used": False,
        }
        try:
            print(f"[v54] line={line_no} expand query", flush=True)
            expansion = await expand_query(client, model, row.get("question", ""))
            evidence = select_expanded_windows(
                chunks,
                row.get("question", ""),
                row.get("prediction", ""),
                expansion,
                args.top_k,
            )
            print(f"[v54] line={line_no} verify gap evidence={len(evidence)}", flush=True)
            verification = await verify_gap(client, model, row, contract, evidence)
            agent["query_expansion"] = expansion
            agent["selected_evidence"] = evidence
            agent["evidence_windows"] = len(evidence)
            agent["verification"] = verification
            missing_facts = verification.get("missing_supported_facts") or []
            unsupported_claims = verification.get("unsupported_or_off_topic_claims") or []
            missing_fact_override = len(missing_facts) >= 2 and verification.get("contract_coverage", 1.0) >= 0.75
            if (verification.get("repair_needed") or missing_fact_override) and (missing_facts or unsupported_claims):
                if missing_fact_override and not verification.get("repair_needed"):
                    verification["repair_needed"] = True
                    verification["repair_goal"] = "missing_fact_override"
                    verification["reason"] = (
                        verification.get("reason", "")
                        + " Runtime override: multiple evidence-supported missing facts were found."
                    ).strip()
                print(f"[v54] line={line_no} generate repair", flush=True)
                candidate = await generate_repair(client, model, row, contract, verification, evidence)
                accepted, guard = guard_candidate(row, candidate, verification, evidence)
                agent["candidate_answer"] = candidate
                agent["guard"] = guard
                agent["accepted"] = accepted
                print(f"[v54] line={line_no} accepted={accepted}", flush=True)
                if accepted:
                    new_row["prediction_before_query_expanded_contract_gap_repair"] = new_row.get("prediction", "")
                    new_row["prediction"] = candidate
                    new_row["method"] = "sage_v54_query_expanded_contract_gap_repair"
                    agent["action"] = "use_contract_gap_repair"
                else:
                    new_row["rejected_query_expanded_contract_gap_repair"] = candidate
            else:
                print(f"[v54] line={line_no} verifier kept current", flush=True)
                agent["action"] = "verification_keep_current"
                agent["guard"] = {
                    "accepted": False,
                    "checks": {"verifier_requested_repair": False},
                    "rejection_reasons": ["verifier_requested_repair"],
                }
        except Exception as exc:
            agent["error"] = f"{type(exc).__name__}: {exc}"[:300]
            agent["action"] = "error_keep_current"

        new_row["query_expanded_contract_gap_repair"] = agent
        output_rows.append(new_row)

    output_path = ROOT_DIR / args.output
    write_jsonl(output_rows, output_path)
    write_summary(ROOT_DIR / args.summary, output_rows, ROOT_DIR / args.input)
    print(f"Saved v54 contract-gap repair results to {output_path}")
    print(f"Saved summary to {ROOT_DIR / args.summary}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
