import argparse
import asyncio
import json
import os
import re
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

from build_answer_contract_agent import clean_text, derive_answer_contract
from build_claim_evidence_agent import EvidenceIndex, write_jsonl
from build_evidence_preserving_compressor import audit_answer
from build_text_evidence_expansion_fallback import contains_refusal, load_jsonl, parse_line_nos, sanitize_evidence_labels
from build_v52_query_expanded_refusal_repair import question_terms


ROOT_DIR = Path(__file__).resolve().parents[1]

SYSTEM_PROMPT = """You are a substance-oriented evidence-bounded QA repair agent.
A previous answer may have refused because the evidence did not exactly match a framing word in the question.
Use only the supplied evidence, but answer the substantive question whenever the evidence contains relevant facts.
Do not invent facts. Do not cite evidence labels.
Do not begin by saying that the evidence is missing, historical, or from a different conflict.
Only refuse if there are no relevant facts at all."""

USER_PROMPT = """Question:
{question}

Answer contract:
{contract}

Current answer:
{current_answer}

Previous rejected repair:
{previous_candidate}

Query expansion:
{query_expansion}

Evidence:
{evidence}

Write a direct evidence-grounded answer in 2-4 concise sentences. If a framing word like "today" or a named conflict is not explicit in the evidence, answer the underlying challenge/theme using the evidence rather than refusing."""


DENIAL_PATTERNS = [
    r"\bcannot be answered\b",
    r"\bcannot be determined\b",
    r"\bdoes not contain\b",
    r"\bnot enough information\b",
    r"\bno information\b",
    r"\bnot the chechen\b",
    r"\bnot chechnya\b",
    r"\bdifferent conflict\b",
]


def evidence_prompt(items: list[dict]) -> str:
    lines = []
    for index, item in enumerate(items, start=1):
        lines.append(
            f"[Text {index}; {item.get('source_id', '')}; score={item.get('score', 0)}]\n"
            f"{item.get('text', '')}"
        )
    return "\n\n".join(lines)


def has_denial(answer: str) -> bool:
    lowered = clean_text(answer).lower()
    return any(re.search(pattern, lowered) for pattern in DENIAL_PATTERNS)


async def generate_answer(client: AsyncOpenAI, model: str, row: dict, evidence: list[dict]) -> str:
    agent = row.get("query_expanded_refusal_repair") or {}
    response = await client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_PROMPT.format(
                    question=row.get("question", ""),
                    contract=json.dumps(derive_answer_contract(row.get("question", "")), ensure_ascii=False),
                    current_answer=row.get("prediction", ""),
                    previous_candidate=agent.get("candidate_answer", ""),
                    query_expansion=json.dumps(agent.get("query_expansion") or {}, ensure_ascii=False),
                    evidence=evidence_prompt(evidence),
                ),
            },
        ],
    )
    return sanitize_evidence_labels(response.choices[0].message.content or "")


def guard_candidate(candidate: str, current: str, question: str, evidence: list[dict]) -> tuple[bool, dict]:
    text_index = EvidenceIndex.build(evidence)
    graph_index = EvidenceIndex.build([])
    traces, metrics = audit_answer(candidate, text_index, graph_index)
    terms = question_terms(question)
    lowered = candidate.lower()
    coverage = 1.0 if not terms else sum(1 for term in terms if re.search(rf"\b{re.escape(term)}\b", lowered)) / len(terms)
    checks = {
        "candidate_present": len(candidate) >= 70,
        "refusal_removed": contains_refusal(current) and not contains_refusal(candidate),
        "no_denial_framing": not has_denial(candidate),
        "mean_support": float(metrics.get("mean_support", 0.0)) >= 0.34,
        "unsupported_ratio": float(metrics.get("unsupported_ratio", 0.0)) <= 0.50,
        "question_term_coverage": coverage >= 0.15,
        "not_too_long": len(candidate) <= 1000,
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
    agent = row.get("query_expanded_refusal_repair") or {}
    return bool(agent.get("attempted")) and not bool(agent.get("accepted"))


def write_summary(path: Path, rows: list[dict], source: Path) -> None:
    agents = [row.get("substance_refusal_repair") or {} for row in rows]
    enabled = [agent for agent in agents if agent.get("enabled")]
    attempted = [agent for agent in enabled if agent.get("attempted")]
    accepted = [agent for agent in attempted if agent.get("accepted")]
    actions = Counter(agent.get("action", "not_applicable") for agent in agents)
    lines = [
        "# SAGE v53 Substance-Oriented Refusal Repair Summary\n\n",
        f"- Source: `{source.as_posix()}`\n",
        f"- Samples: {len(rows)}\n",
        "- Triggered for rejected v52 refusal repairs unless line numbers are specified.\n",
        "- Uses only v52 query-expanded evidence; reference answers and judge scores are not runtime inputs.\n\n",
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
        agent = row.get("substance_refusal_repair") or {}
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
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--line-nos", default=None)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    load_dotenv(ROOT_DIR / ".env")
    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_API_BASE")
    model = args.model or os.getenv("DEEPSEEK_MODEL")
    if not api_key or not base_url or not model:
        raise RuntimeError("DeepSeek-compatible API configuration is incomplete in .env")

    rows = load_jsonl(ROOT_DIR / args.input)
    requested_lines = parse_line_nos(args.line_nos)
    client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=300.0, max_retries=2)
    output_rows = []

    for row in rows:
        new_row = dict(row)
        if not should_attempt(row, requested_lines):
            new_row["substance_refusal_repair"] = {"enabled": False}
            output_rows.append(new_row)
            continue

        v52_agent = row.get("query_expanded_refusal_repair") or {}
        evidence = v52_agent.get("selected_evidence") or []
        agent = {
            "enabled": True,
            "version": "v53",
            "attempted": True,
            "action": "retain_current",
            "accepted": False,
            "selected_evidence": evidence,
            "reference_answer_used": False,
        }
        try:
            candidate = await generate_answer(client, model, row, evidence)
            accepted, guard = guard_candidate(candidate, row.get("prediction", ""), row.get("question", ""), evidence)
            agent["candidate_answer"] = candidate
            agent["guard"] = guard
            agent["accepted"] = accepted
            if accepted:
                new_row["prediction_before_substance_refusal_repair"] = new_row.get("prediction", "")
                new_row["prediction"] = candidate
                new_row["method"] = "sage_v53_substance_refusal_repair"
                agent["action"] = "use_substance_answer"
            else:
                new_row["rejected_substance_refusal_repair"] = candidate
        except Exception as exc:
            agent["error"] = f"{type(exc).__name__}: {exc}"[:300]
            agent["action"] = "error_keep_current"

        new_row["substance_refusal_repair"] = agent
        output_rows.append(new_row)

    output_path = ROOT_DIR / args.output
    write_jsonl(output_rows, output_path)
    write_summary(ROOT_DIR / args.summary, output_rows, ROOT_DIR / args.input)
    print(f"Saved v53 substance refusal repair results to {output_path}")
    print(f"Saved summary to {ROOT_DIR / args.summary}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
