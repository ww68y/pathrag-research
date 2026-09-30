import argparse
import asyncio
import json
import os
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

from build_answer_contract_agent import clean_text, load_jsonl, parse_json_object
from build_claim_evidence_agent import (
    EvidenceIndex,
    content_tokens,
    load_full_source_rows,
    normalize_space,
    write_jsonl,
)
from build_evidence_preserving_compressor import (
    LENGTH_BUDGETS,
    answer_contract,
    audit_answer,
    context_indexes,
    semantic_claim_coverage,
)


ROOT_DIR = Path(__file__).resolve().parents[1]
UNCERTAINTY_MARKERS = (
    "cannot", "not described", "not provide", "not contain", "not explicitly",
    "insufficient", "suggests", "implies", "likely", "based solely",
)


def contract_risk(row: dict, metrics: dict) -> dict:
    contract = answer_contract(row)
    contract_type = contract.get("type", "general")
    max_chars, _ = LENGTH_BUDGETS.get(contract_type, LENGTH_BUDGETS["general"])
    answer = clean_text(row.get("prediction", ""))
    lowered = answer.lower()
    reasons = []
    if any(marker in lowered for marker in UNCERTAINTY_MARKERS):
        reasons.append("uncertainty_or_hedging")
    if metrics.get("mean_support", 0) < 0.68:
        reasons.append("low_claim_support")
    if len(answer) > max_chars * 1.35:
        reasons.append("overlong_answer_may_drift")
    if metrics.get("claims", 0) < int(contract.get("minimum_distinct_points", 1)) + 1:
        reasons.append("answer_shape_below_contract")

    if (
        contract_type == "general"
        and 180 <= len(answer) <= 1800
        and not any(marker in lowered for marker in UNCERTAINTY_MARKERS)
        and metrics.get("mean_support", 0) >= 0.60
    ):
        reasons = []
    prior_claim_agent = row.get("claim_evidence_agent") or {}
    if prior_claim_agent.get("repair_applied") and metrics.get("mean_support", 0) >= 0.55:
        reasons = []

    return {
        "enabled": True,
        "contract": contract,
        "risk_candidate": bool(reasons),
        "risk_reasons": reasons,
        "verification_attempted": False,
        "repair_needed": False,
        "repair_attempted": False,
        "repair_accepted": False,
        "answer_chars": len(answer),
        "pre_claim_metrics": metrics,
    }


def evidence_prompt(trace: dict) -> str:
    lines = []
    for index, item in enumerate(trace.get("question_text_evidence", []), start=1):
        lines.append(f"[Text {index}] {item.get('text', '')}")
    for index, item in enumerate(trace.get("question_graph_evidence", []), start=1):
        lines.append(f"[Graph {index}] {item.get('text', '')}")
    return "\n\n".join(lines)


VERIFY_SYSTEM = """You are a strict answer-contract completion verifier for retrieval-augmented QA.
Use only the supplied evidence. Do not use outside knowledge or any reference answer. An answer can be
well-grounded sentence by sentence but still incomplete for the question. Return JSON only."""
VERIFY_USER = """Question:
{question}

Question-derived answer contract:
{contract}

Current answer:
{answer}

Retrieved full-document and graph evidence:
{evidence}

Check whether the answer satisfies every question requirement:
1. Identify concrete requested facts already covered.
2. Identify concrete answer-essential facts missing from the answer but explicitly supported by evidence.
3. Identify unsupported or off-topic answer claims.
4. Score contract coverage and evidence grounding from 0.0 to 1.0.
5. Request repair only when a concrete supported fact is missing or an unsupported claim must be corrected.

Return this exact JSON shape:
{{
  "covered_facts": ["..."],
  "missing_supported_facts": ["..."],
  "unsupported_or_off_topic_claims": ["..."],
  "contract_coverage": 0.0,
  "evidence_grounding": 0.0,
  "repair_needed": false,
  "reason": "..."
}}"""

REPAIR_SYSTEM = """You are an evidence-bounded answer completion agent.
Rewrite the answer using only the supplied text and graph evidence. Preserve correct useful content,
add every verified missing fact, remove unsupported or off-topic claims, and satisfy the answer contract.
Give the direct answer first and avoid unnecessary background. Do not mention verification or references."""
REPAIR_USER = """Question:
{question}

Answer contract:
{contract}

Current answer:
{answer}

Verified missing supported facts:
{missing}

Claims to correct or remove:
{unsupported}

Evidence:
{evidence}
"""


async def run_verification(rows: list[dict], model: str, concurrency: int) -> None:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_API_BASE")
    if not api_key or not base_url or not model:
        raise RuntimeError("DeepSeek-compatible API configuration is incomplete in .env")
    client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=120.0, max_retries=2)
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def verify(row: dict) -> None:
        agent = row["contract_completion_agent"]
        agent["verification_attempted"] = True
        async with semaphore:
            try:
                response = await client.chat.completions.create(
                    model=model,
                    temperature=0,
                    messages=[
                        {"role": "system", "content": VERIFY_SYSTEM},
                        {
                            "role": "user",
                            "content": VERIFY_USER.format(
                                question=row.get("question", ""),
                                contract=json.dumps(agent["contract"], ensure_ascii=False),
                                answer=row.get("prediction", ""),
                                evidence=evidence_prompt(agent),
                            ),
                        },
                    ],
                )
                result = parse_json_object(response.choices[0].message.content or "")
            except Exception as exc:
                agent["verification_error"] = f"{type(exc).__name__}: {exc}"[:300]
                return

        coverage = max(0.0, min(1.0, float(result.get("contract_coverage", 0))))
        grounding = max(0.0, min(1.0, float(result.get("evidence_grounding", 0))))
        missing = result.get("missing_supported_facts") or []
        unsupported = result.get("unsupported_or_off_topic_claims") or []
        repair_needed = (
            bool(result.get("repair_needed"))
            and bool(missing or unsupported)
            and coverage <= 0.60
            and grounding <= 0.70
        )
        result["contract_coverage"] = round(coverage, 6)
        result["evidence_grounding"] = round(grounding, 6)
        result["repair_approved_by_gate"] = repair_needed
        agent["verification"] = result
        agent["repair_needed"] = repair_needed

    await asyncio.gather(*(verify(row) for row in rows))


async def run_repairs(rows: list[dict], model: str, concurrency: int) -> None:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_API_BASE")
    client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=120.0, max_retries=2)
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def repair(row: dict) -> None:
        agent = row["contract_completion_agent"]
        verification = agent["verification"]
        agent["repair_attempted"] = True
        async with semaphore:
            try:
                response = await client.chat.completions.create(
                    model=model,
                    temperature=0,
                    messages=[
                        {"role": "system", "content": REPAIR_SYSTEM},
                        {
                            "role": "user",
                            "content": REPAIR_USER.format(
                                question=row.get("question", ""),
                                contract=json.dumps(agent["contract"], ensure_ascii=False),
                                answer=row.get("prediction", ""),
                                missing="\n".join(f"- {item}" for item in verification.get("missing_supported_facts", [])),
                                unsupported="\n".join(
                                    f"- {item}" for item in verification.get("unsupported_or_off_topic_claims", [])
                                ),
                                evidence=evidence_prompt(agent),
                            ),
                        },
                    ],
                )
                agent["repair_candidate_answer"] = clean_text(response.choices[0].message.content)
            except Exception as exc:
                agent["repair_error"] = f"{type(exc).__name__}: {exc}"[:300]

    await asyncio.gather(*(repair(row) for row in rows))


def missing_fact_coverage(missing: list[str], repaired: str) -> float:
    if not missing:
        return 1.0
    repaired_tokens = content_tokens(repaired)
    scores = []
    for fact in missing:
        fact_tokens = content_tokens(fact)
        scores.append(len(fact_tokens & repaired_tokens) / max(1, len(fact_tokens)))
    return sum(scores) / len(scores)


def apply_repair_guard(
    row: dict,
    text_index: EvidenceIndex,
    graph_index: EvidenceIndex,
) -> None:
    agent = row["contract_completion_agent"]
    candidate = clean_text(agent.pop("repair_candidate_answer", ""))
    if not candidate:
        return
    traces, metrics = audit_answer(candidate, text_index, graph_index)
    verification = agent["verification"]
    missing = verification.get("missing_supported_facts") or []
    fact_coverage = missing_fact_coverage(missing, candidate)
    minimum_points = int(agent["contract"].get("minimum_distinct_points", 1))
    old_answer = clean_text(row.get("prediction", ""))
    old_uncertainty = any(marker in old_answer.lower() for marker in UNCERTAINTY_MARKERS)
    new_uncertainty = any(marker in candidate.lower() for marker in UNCERTAINTY_MARKERS)
    checks = {
        "verified_missing_facts_added": fact_coverage >= 0.55,
        "evidence_grounded": metrics["mean_support"] >= 0.55,
        "unsupported_claims_controlled": metrics["unsupported_ratio"] <= 0.20,
        "answer_contract_shape": metrics["claims"] >= minimum_points,
        "uncertainty_reduced": not (old_uncertainty and new_uncertainty),
        "usable_answer_length": 80 <= len(candidate) <= max(2400, len(old_answer) * 1.35),
    }
    accepted = all(checks.values())
    agent["repair_guard"] = {
        "accepted": accepted,
        "checks": checks,
        "missing_fact_coverage": round(fact_coverage, 6),
        "post_claim_metrics": metrics,
        "post_claim_traces": traces,
        "rejection_reasons": [name for name, passed in checks.items() if not passed],
    }
    if accepted:
        row["prediction_before_contract_completion"] = row.get("prediction", "")
        row["prediction"] = candidate
        row["method"] = "sage_v27_contract_completion"
        agent["repair_accepted"] = True
    else:
        row["rejected_contract_completion_prediction"] = candidate


def write_summary(path: Path, rows: list[dict], source_path: Path) -> None:
    agents = [row.get("contract_completion_agent") or {} for row in rows]
    candidates = sum(bool(agent.get("risk_candidate")) for agent in agents)
    verified = sum(bool(agent.get("verification_attempted")) for agent in agents)
    needed = sum(bool(agent.get("repair_needed")) for agent in agents)
    attempted = sum(bool(agent.get("repair_attempted")) for agent in agents)
    accepted = sum(bool(agent.get("repair_accepted")) for agent in agents)
    evidence_modes = Counter(agent.get("evidence_mode", "hybrid") for agent in agents)
    risk_counts = Counter(reason for agent in agents for reason in agent.get("risk_reasons", []))
    rejection_counts = Counter(
        reason for agent in agents for reason in (agent.get("repair_guard") or {}).get("rejection_reasons", [])
    )
    lines = [
        "# SAGE v27 Contract-Completion Agent Summary\n\n",
        f"- Source: `{source_path.as_posix()}`\n",
        f"- Samples: {len(rows)}\n",
        f"- Evidence mode: {', '.join(f'{name}={count}' for name, count in sorted(evidence_modes.items()))}\n",
        "- Reference answers and evaluation scores are not used by routing, verification, repair, or guards.\n\n",
        "## Runtime Decisions\n\n",
        "| Metric | Value |\n",
        "|---|---:|\n",
        f"| Risk candidates | {candidates} |\n",
        f"| Contract verifications | {verified} |\n",
        f"| Repairs requested | {needed} |\n",
        f"| Repairs attempted | {attempted} |\n",
        f"| Repairs accepted | {accepted} |\n",
        f"| Repairs rejected / original retained | {attempted - accepted} |\n\n",
        "## Risk Signals\n\n",
        "| Signal | Count |\n",
        "|---|---:|\n",
    ]
    for reason, count in sorted(risk_counts.items()):
        lines.append(f"| {reason} | {count} |\n")
    lines.extend(["\n## Guard Rejections\n\n", "| Reason | Count |\n", "|---|---:|\n"])
    if rejection_counts:
        for reason, count in sorted(rejection_counts.items()):
            lines.append(f"| {reason} | {count} |\n")
    else:
        lines.append("| none | 0 |\n")
    path.write_text("".join(lines), encoding="utf-8")


def stratified_limit(rows: list[dict], limit: int, offset: int = 0) -> list[dict]:
    if limit <= 0 and offset <= 0:
        return rows
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(str(row.get("dataset", "unknown")), []).append(row)
    for group_rows in groups.values():
        group_rows.sort(key=lambda row: int(row.get("line_no", 0)))

    ordered = []
    group_names = sorted(groups)
    while True:
        added = False
        for name in group_names:
            if groups[name]:
                ordered.append(groups[name].pop(0))
                added = True
        if not added:
            break
    end = None if limit <= 0 else offset + limit
    return ordered[offset:end]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--full-context-root", required=True)
    parser.add_argument("--data-root", default="data_repro")
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--model", default=None)
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument(
        "--evidence-mode",
        choices=("hybrid", "text-only"),
        default="hybrid",
        help="Use full-text evidence only before graph construction, or the original text+graph evidence",
    )
    parser.add_argument(
        "--max-verifications",
        type=int,
        default=0,
        help="Optional deterministic dataset-stratified limit for risk candidates",
    )
    parser.add_argument(
        "--verification-offset",
        type=int,
        default=0,
        help="Skip this many candidates in deterministic stratified order",
    )
    parser.add_argument(
        "--verification-line-nos",
        default=None,
        help="Optional comma-separated line numbers to verify exactly",
    )
    parser.add_argument(
        "--force-verification-lines",
        action="store_true",
        help="Verify the requested line numbers even when deterministic risk is false",
    )
    args = parser.parse_args()

    input_path = ROOT_DIR / args.input
    rows = load_jsonl(input_path)
    sources = load_jsonl(ROOT_DIR / args.dataset)
    source_by_line = {
        int(source.get("_subset_line_no", index)): source for index, source in enumerate(sources)
    }
    full_sources = load_full_source_rows(sources, ROOT_DIR / args.full_context_root)
    text_cache: dict[str, EvidenceIndex] = {}
    graph_cache: dict[str, EvidenceIndex] = {}
    runtime = {}
    output_rows = []

    for row in rows:
        line_no = int(row.get("line_no", 0))
        source = source_by_line[line_no]
        full_source = full_sources.get((
            normalize_space(source.get("source_dataset")),
            int(source.get("source_line_no", -1)),
        ), {})
        text_index, graph_index = context_indexes(
            row, source, full_source, ROOT_DIR / args.data_root, text_cache, graph_cache
        )
        effective_graph_index = graph_index if args.evidence_mode == "hybrid" else EvidenceIndex.build([])
        _, metrics = audit_answer(row.get("prediction", ""), text_index, effective_graph_index)
        agent = contract_risk(row, metrics)
        agent["evidence_mode"] = args.evidence_mode
        agent["question_text_evidence"] = text_index.search(row.get("question", ""), 8)
        agent["question_graph_evidence"] = effective_graph_index.search(row.get("question", ""), 5)
        new_row = dict(row)
        new_row["contract_completion_agent"] = agent
        output_rows.append(new_row)
        runtime[line_no] = (text_index, effective_graph_index)

    if args.execute:
        load_dotenv(ROOT_DIR / ".env")
        model = args.model or os.getenv("DEEPSEEK_MODEL")
        risk_rows = [row for row in output_rows if row["contract_completion_agent"]["risk_candidate"]]
        if args.verification_line_nos:
            selected_lines = {
                int(value.strip())
                for value in args.verification_line_nos.split(",")
                if value.strip()
            }
            if args.force_verification_lines:
                risk_rows = [
                    row for row in output_rows if int(row.get("line_no", -1)) in selected_lines
                ]
            else:
                risk_rows = [
                    row for row in risk_rows if int(row.get("line_no", -1)) in selected_lines
                ]
            found_lines = {int(row.get("line_no", -1)) for row in risk_rows}
            missing_lines = sorted(selected_lines - found_lines)
            if missing_lines:
                raise ValueError(f"Requested lines are missing from input rows: {missing_lines}")
        else:
            risk_rows = stratified_limit(risk_rows, args.max_verifications, args.verification_offset)
        asyncio.run(run_verification(risk_rows, model, args.concurrency))
        repair_rows = [row for row in risk_rows if row["contract_completion_agent"]["repair_needed"]]
        asyncio.run(run_repairs(repair_rows, model, args.concurrency))
        for row in repair_rows:
            line_no = int(row.get("line_no", 0))
            apply_repair_guard(row, *runtime[line_no])

    output_path = ROOT_DIR / args.output
    write_jsonl(output_rows, output_path)
    write_summary(ROOT_DIR / args.summary, output_rows, input_path)
    print(f"Saved SAGE v27 results to {output_path}")
    print(f"Saved summary to {ROOT_DIR / args.summary}")


if __name__ == "__main__":
    main()
