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
from build_claim_evidence_agent import EvidenceIndex, load_full_source_rows, normalize_space, write_jsonl
from build_evidence_preserving_compressor import audit_answer
from build_text_evidence_expansion_fallback import (
    REFUSAL_PATTERNS,
    build_full_text_index,
    complete_answer,
    expanded_evidence,
    load_jsonl,
    parse_line_nos,
    source_by_subset_line,
)


ROOT_DIR = Path(__file__).resolve().parents[1]

ARBITER_SYSTEM = """You are a strict evidence-bounded answer arbiter.
Choose which answer is better supported by the supplied evidence and better answers the question.
Do not use outside knowledge or reference answers.
Return only valid JSON."""

ARBITER_USER = """Question:
{question}

Expanded text evidence:
{evidence}

Answer A, current:
{current_answer}

Answer B, candidate:
{candidate_answer}

Return JSON with exactly these keys:
{{
  "winner": "A" | "B" | "tie",
  "confidence": number from 0 to 1,
  "reason": "short reason"
}}"""


def contains_refusal(answer: str) -> bool:
    lowered = clean_text(answer).lower()
    return any(pattern in lowered for pattern in REFUSAL_PATTERNS)


def evidence_prompt(items: list[dict]) -> str:
    lines = []
    for index, item in enumerate(items, start=1):
        lines.append(f"[Text {index}; {item.get('source_id', '')}; score={item.get('score', 0)}]\n{item.get('text', '')}")
    return "\n\n".join(lines)


def extract_json(text: str) -> dict:
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            return {"winner": "tie", "confidence": 0.0, "reason": "arbiter returned non-json"}
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {"winner": "tie", "confidence": 0.0, "reason": "arbiter returned invalid json"}


async def arbitrate_answer(
    client: AsyncOpenAI,
    model: str,
    question: str,
    current_answer: str,
    candidate_answer: str,
    evidence: list[dict],
) -> dict:
    response = await client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": ARBITER_SYSTEM},
            {
                "role": "user",
                "content": ARBITER_USER.format(
                    question=question,
                    evidence=evidence_prompt(evidence),
                    current_answer=current_answer,
                    candidate_answer=candidate_answer,
                ),
            },
        ],
    )
    result = extract_json(response.choices[0].message.content or "")
    winner = str(result.get("winner", "tie")).strip().upper()
    if winner not in {"A", "B", "TIE"}:
        winner = "TIE"
    result["winner"] = "tie" if winner == "TIE" else winner
    try:
        result["confidence"] = max(0.0, min(1.0, float(result.get("confidence", 0.0))))
    except (TypeError, ValueError):
        result["confidence"] = 0.0
    result["reason"] = clean_text(result.get("reason", ""))
    return result


def metric_delta(current_metrics: dict, candidate_metrics: dict) -> dict:
    return {
        "mean_support_delta": round(
            float(candidate_metrics.get("mean_support", 0.0)) - float(current_metrics.get("mean_support", 0.0)),
            6,
        ),
        "unsupported_ratio_delta": round(
            float(candidate_metrics.get("unsupported_ratio", 0.0))
            - float(current_metrics.get("unsupported_ratio", 0.0)),
            6,
        ),
        "supported_claim_delta": int(candidate_metrics.get("supported_claims", 0))
        - int(current_metrics.get("supported_claims", 0)),
    }


def acceptance(
    current_answer: str,
    candidate_answer: str,
    text_index: EvidenceIndex,
    arbiter: dict,
) -> tuple[bool, dict]:
    graph_index = EvidenceIndex.build([])
    current_traces, current_metrics = audit_answer(current_answer, text_index, graph_index)
    candidate_traces, candidate_metrics = audit_answer(candidate_answer, text_index, graph_index)
    delta = metric_delta(current_metrics, candidate_metrics)
    checks = {
        "candidate_present": len(clean_text(candidate_answer)) >= 50,
        "candidate_not_refusal": not contains_refusal(candidate_answer),
        "candidate_supported": float(candidate_metrics.get("mean_support", 0.0)) >= 0.45,
        "unsupported_ratio_ok": float(candidate_metrics.get("unsupported_ratio", 0.0)) <= 0.35,
        "support_not_worse": delta["mean_support_delta"] >= -0.08,
        "arbiter_prefers_candidate": arbiter.get("winner") == "B" and float(arbiter.get("confidence", 0.0)) >= 0.55,
    }
    guard = {
        "accepted": all(checks.values()),
        "checks": checks,
        "current_claim_metrics": current_metrics,
        "candidate_claim_metrics": candidate_metrics,
        "metric_delta": delta,
        "current_claim_traces": current_traces,
        "candidate_claim_traces": candidate_traces,
        "arbiter": arbiter,
        "rejection_reasons": [name for name, passed in checks.items() if not passed],
    }
    return all(checks.values()), guard


def should_attempt(row: dict, requested_lines: set[int] | None) -> bool:
    line_no = int(row.get("line_no", -1))
    if requested_lines is not None:
        return line_no in requested_lines
    answer = clean_text(row.get("prediction", ""))
    return contains_refusal(answer) or len(answer) >= 700


def write_summary(path: Path, rows: list[dict], source: Path) -> None:
    agents = [row.get("evidence_mismatch_repair") or {} for row in rows]
    enabled = [agent for agent in agents if agent.get("enabled")]
    attempted = [agent for agent in enabled if agent.get("attempted")]
    accepted = [agent for agent in attempted if agent.get("accepted")]
    actions = Counter(agent.get("action", "not_applicable") for agent in agents)
    lines = [
        "# SAGE v44 Evidence-Mismatch Repair Summary\n\n",
        f"- Source: `{source.as_posix()}`\n",
        f"- Samples: {len(rows)}\n",
        "- Repairs are selected by evidence-only arbitration, not by reference answers or judge scores.\n\n",
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
        "| Line | Dataset | Action | Accepted | Arbiter | Confidence | Reasons |\n",
        "|---:|---|---|---|---|---:|---|\n",
    ])
    for row in rows:
        agent = row.get("evidence_mismatch_repair") or {}
        if not agent.get("enabled"):
            continue
        guard = agent.get("guard") or {}
        arbiter = guard.get("arbiter") or {}
        reasons = "; ".join(guard.get("rejection_reasons", [])).replace("|", "/")
        lines.append(
            f"| {row.get('line_no')} | {row.get('dataset')} | {agent.get('action')} | "
            f"{agent.get('accepted')} | {arbiter.get('winner', '-')} | "
            f"{float(arbiter.get('confidence', 0.0)):.2f} | {reasons or 'none'} |\n"
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
    parser.add_argument("--top-k", type=int, default=10)
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
            new_row["evidence_mismatch_repair"] = {"enabled": False}
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
            "version": "v44",
            "attempted": True,
            "action": "retain_current",
            "accepted": False,
            "evidence_windows": len(index.records),
            "selected_evidence": evidence,
            "reference_answer_used": False,
        }
        try:
            candidate = await complete_answer(client, model, row.get("question", ""), row.get("prediction", ""), evidence)
            arbiter = await arbitrate_answer(
                client,
                model,
                row.get("question", ""),
                row.get("prediction", ""),
                candidate,
                evidence,
            )
            accepted, guard = acceptance(row.get("prediction", ""), candidate, index, arbiter)
            agent["candidate_answer"] = candidate
            agent["guard"] = guard
            agent["accepted"] = accepted
            if accepted:
                new_row["prediction_before_evidence_mismatch_repair"] = new_row.get("prediction", "")
                new_row["prediction"] = candidate
                new_row["method"] = "sage_v44_evidence_mismatch_repair"
                agent["action"] = "use_evidence_mismatch_repair"
            else:
                new_row["rejected_evidence_mismatch_repair_prediction"] = candidate
        except Exception as exc:
            agent["error"] = f"{type(exc).__name__}: {exc}"[:300]
            agent["action"] = "error_keep_current"

        new_row["evidence_mismatch_repair"] = agent
        output_rows.append(new_row)

    output_path = ROOT_DIR / args.output
    write_jsonl(output_rows, output_path)
    write_summary(ROOT_DIR / args.summary, output_rows, ROOT_DIR / args.input)
    print(f"Saved v44 evidence-mismatch repair results to {output_path}")
    print(f"Saved summary to {ROOT_DIR / args.summary}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()

