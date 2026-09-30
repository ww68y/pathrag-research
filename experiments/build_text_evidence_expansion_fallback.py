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

from build_answer_contract_agent import clean_text
from build_claim_evidence_agent import (
    EvidenceIndex,
    load_full_source_rows,
    normalize_space,
    split_windows,
    write_jsonl,
)
from build_evidence_preserving_compressor import audit_answer


ROOT_DIR = Path(__file__).resolve().parents[1]

REFUSAL_PATTERNS = (
    "cannot be determined",
    "cannot determine",
    "does not contain",
    "not enough information",
    "insufficient information",
    "not provided",
    "not explicitly state",
)

SYSTEM_PROMPT = """You are a strict evidence-bounded QA agent.
Answer using only the supplied expanded text evidence. Do not use outside knowledge or reference answers.
If the evidence directly answers the question, give the direct answer first and stay concise.
If the evidence is genuinely insufficient, state exactly what is missing.
Do not cite evidence labels such as Text 1, Text 2, or source ids."""

USER_PROMPT = """Question:
{question}

Current answer:
{current_answer}

Expanded text evidence:
{evidence}

Write the best evidence-grounded answer."""


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def parse_line_nos(value: str | None) -> set[int] | None:
    if not value:
        return None
    return {int(item.strip()) for item in value.split(",") if item.strip()}


def contains_refusal(answer: str) -> bool:
    lowered = clean_text(answer).lower()
    return any(pattern in lowered for pattern in REFUSAL_PATTERNS)


def sanitize_evidence_labels(answer: str) -> str:
    answer = re.sub(r"\s*\((?:Text|Graph)\s+\d+\)", "", answer, flags=re.IGNORECASE)
    answer = re.sub(r"\b(?:Text|Graph)\s+\d+\b", "the evidence", answer, flags=re.IGNORECASE)
    return clean_text(answer)


def source_by_subset_line(rows: list[dict]) -> dict[int, dict]:
    return {int(row.get("_subset_line_no", index)): row for index, row in enumerate(rows)}


def build_full_text_index(full_context: str, window_chars: int, overlap_chars: int) -> EvidenceIndex:
    records = [
        {"kind": "full_text", "text": window, "source_id": f"full-window-{index}"}
        for index, window in enumerate(split_windows(full_context, size=window_chars, overlap=overlap_chars))
    ]
    return EvidenceIndex.build(records)


def expanded_evidence(index: EvidenceIndex, question: str, current_answer: str, top_k: int) -> list[dict]:
    queries = [
        question,
        re.sub(r"\b(?:cannot|determined|provided|context|chunks|solely|insufficient)\b", " ", current_answer, flags=re.I),
        f"{question} {current_answer}",
    ]
    selected: dict[str, dict] = {}
    for query in queries:
        for item in index.search(query, top_k):
            selected.setdefault(str(item.get("source_id")), item)
    if index.records:
        opening = dict(index.records[0])
        opening["score"] = 0.0
        selected.setdefault(str(opening.get("source_id")), opening)
    return list(selected.values())[: max(top_k, 1) + 2]


def evidence_prompt(items: list[dict]) -> str:
    lines = []
    for index, item in enumerate(items, start=1):
        lines.append(f"[Text {index}; {item.get('source_id', '')}; score={item.get('score', 0)}]\n{item.get('text', '')}")
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
    return sanitize_evidence_labels(response.choices[0].message.content or "")


def should_attempt(row: dict, requested_lines: set[int] | None) -> bool:
    line_no = int(row.get("line_no", -1))
    if requested_lines is not None:
        return line_no in requested_lines
    agent = row.get("evidence_calibrated_lazy_agent") or {}
    plan = agent.get("pregraph_plan") or row.get("v41_pregraph_plan") or {}
    return agent.get("action") == "graph_pending_text_fallback" or plan.get("action") == "request_graph"


def guard_checks(candidate: str, current: str, metrics: dict) -> dict:
    return {
        "candidate_present": len(candidate) >= 60,
        "refusal_removed": contains_refusal(current) and not contains_refusal(candidate),
        "mean_support": float(metrics.get("mean_support", 0.0)) >= 0.45,
        "unsupported_ratio": float(metrics.get("unsupported_ratio", 0.0)) <= 0.35,
    }


def acceptance(candidate: str, current: str, text_index: EvidenceIndex) -> tuple[bool, dict, str]:
    graph_index = EvidenceIndex.build([])
    traces, metrics = audit_answer(candidate, text_index, graph_index)
    checks = guard_checks(candidate, current, metrics)
    accepted = all(checks.values())
    guard = {
        "accepted": all(checks.values()),
        "checks": checks,
        "post_claim_metrics": metrics,
        "post_claim_traces": traces,
        "rejection_reasons": [name for name, passed in checks.items() if not passed],
    }
    if accepted:
        return True, guard, candidate

    supported_claims = [
        trace.get("claim", "")
        for trace in traces
        if trace.get("status") == "supported" and trace.get("claim")
    ]
    pruned = clean_text(" ".join(supported_claims))
    if pruned and pruned != candidate:
        pruned_traces, pruned_metrics = audit_answer(pruned, text_index, graph_index)
        pruned_checks = guard_checks(pruned, current, pruned_metrics)
        pruned_accepted = all(pruned_checks.values())
        guard["support_pruning"] = {
            "enabled": True,
            "accepted": pruned_accepted,
            "candidate": pruned,
            "checks": pruned_checks,
            "post_claim_metrics": pruned_metrics,
            "post_claim_traces": pruned_traces,
            "rejection_reasons": [name for name, passed in pruned_checks.items() if not passed],
        }
        if pruned_accepted:
            guard["accepted"] = True
            guard["rejection_reasons"] = []
            guard["support_pruning_applied"] = True
            return True, guard, pruned

    return False, guard, candidate


def write_summary(path: Path, rows: list[dict], source: Path) -> None:
    agents = [row.get("text_evidence_expansion_fallback") or {} for row in rows]
    enabled = [agent for agent in agents if agent.get("enabled")]
    attempted = [agent for agent in enabled if agent.get("attempted")]
    accepted = [agent for agent in attempted if agent.get("accepted")]
    actions = Counter(agent.get("action", "not_applicable") for agent in agents)
    lines = [
        "# SAGE v43 Text-Evidence Expansion Fallback Summary\n\n",
        f"- Source: `{source.as_posix()}`\n",
        f"- Samples: {len(rows)}\n",
        "- Triggered only for graph-pending samples.\n",
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
    lines.extend([
        "\n## Per-Sample Decisions\n\n",
        "| Line | Dataset | Action | Accepted | Reasons |\n",
        "|---:|---|---|---|---|\n",
    ])
    for row in rows:
        agent = row.get("text_evidence_expansion_fallback") or {}
        if not agent.get("enabled"):
            continue
        reasons = "; ".join((agent.get("guard") or {}).get("rejection_reasons", [])).replace("|", "/")
        lines.append(
            f"| {row.get('line_no')} | {row.get('dataset')} | {agent.get('action')} | "
            f"{agent.get('accepted')} | {reasons or 'none'} |\n"
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
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--window-chars", type=int, default=1400)
    parser.add_argument("--overlap-chars", type=int, default=220)
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
            new_row["text_evidence_expansion_fallback"] = {"enabled": False}
            output_rows.append(new_row)
            continue

        line_no = int(row.get("line_no", -1))
        source = source_rows[line_no]
        full_source = full_sources.get((
            normalize_space(source.get("source_dataset")),
            int(source.get("source_line_no", -1)),
        ), source)
        full_context = clean_text(full_source.get("context", ""))
        index = build_full_text_index(full_context, args.window_chars, args.overlap_chars)
        evidence = expanded_evidence(index, row.get("question", ""), row.get("prediction", ""), args.top_k)
        agent = {
            "enabled": True,
            "version": "v43",
            "attempted": True,
            "action": "retain_text",
            "accepted": False,
            "evidence_windows": len(index.records),
            "selected_evidence": evidence,
            "reference_answer_used": False,
        }
        try:
            candidate = await complete_answer(client, model, row.get("question", ""), row.get("prediction", ""), evidence)
            agent["candidate_answer"] = candidate
            accepted, guard, accepted_answer = acceptance(candidate, row.get("prediction", ""), index)
            agent["guard"] = guard
            agent["accepted"] = accepted
            if accepted:
                new_row["prediction_before_text_evidence_expansion_fallback"] = new_row.get("prediction", "")
                new_row["prediction"] = accepted_answer
                new_row["method"] = "sage_v43_text_evidence_expansion_fallback"
                agent["action"] = "use_expanded_text_answer"
            else:
                new_row["rejected_text_evidence_expansion_prediction"] = candidate
        except Exception as exc:
            agent["error"] = f"{type(exc).__name__}: {exc}"[:300]
            agent["action"] = "error_keep_text"

        new_row["text_evidence_expansion_fallback"] = agent
        output_rows.append(new_row)

    output_path = ROOT_DIR / args.output
    write_jsonl(output_rows, output_path)
    write_summary(ROOT_DIR / args.summary, output_rows, ROOT_DIR / args.input)
    print(f"Saved v43 text-evidence fallback results to {output_path}")
    print(f"Saved summary to {ROOT_DIR / args.summary}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
