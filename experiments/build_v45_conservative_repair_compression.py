import argparse
import asyncio
import json
import os
import re
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

from build_answer_contract_agent import clean_text, derive_answer_contract, parse_json_object
from build_claim_evidence_agent import EvidenceIndex, load_full_source_rows, normalize_space, write_jsonl
from build_evidence_preserving_compressor import audit_answer
from build_text_evidence_expansion_fallback import (
    REFUSAL_PATTERNS,
    build_full_text_index,
    expanded_evidence,
    load_jsonl,
    parse_line_nos,
    sanitize_evidence_labels,
    source_by_subset_line,
)


ROOT_DIR = Path(__file__).resolve().parents[1]

COMPRESS_SYSTEM = """You are an evidence-preserving QA compression agent.
Rewrite the answer to be shorter while preserving the answer-essential facts.
Use only the supplied evidence and the current answer. Do not add outside knowledge.
Remove background, examples, implementation details, and side conditions unless they directly answer the question.
Give the direct answer first. Do not cite evidence labels or mention compression."""

COMPRESS_USER = """Question:
{question}

Answer contract:
{contract}

Current answer:
{answer}

Evidence:
{evidence}

Rewrite as 1-3 concise sentences. Preserve essential entities, conditions, causal direction, and named concepts."""

VERIFY_SYSTEM = """You are a strict semantic-fidelity verifier for answer compression.
Compare the original and compressed answers against the question and evidence.
Do not use reference answers, evaluation scores, or outside knowledge. Return JSON only."""

VERIFY_USER = """Question:
{question}

Evidence:
{evidence}

Original answer:
{original}

Compressed answer:
{compressed}

Check whether the compressed answer preserves every answer-essential fact from the original.
Ignore background details that do not directly answer the question.

Return this exact JSON shape:
{{
  "essential_facts": ["..."],
  "lost_essential_facts": ["..."],
  "unsupported_new_claims": ["..."],
  "contradictions": ["..."],
  "essential_fact_coverage": 0.0,
  "approved": false,
  "reason": "short reason"
}}"""


def contains_refusal(answer: str) -> bool:
    lowered = clean_text(answer).lower()
    return any(pattern in lowered for pattern in REFUSAL_PATTERNS)


def evidence_prompt(items: list[dict], limit: int = 8) -> str:
    lines = []
    for index, item in enumerate(items[:limit], start=1):
        lines.append(
            f"[Evidence {index}; {item.get('source_id', '')}; score={item.get('score', 0)}]\n"
            f"{item.get('text', '')}"
        )
    return "\n\n".join(lines)


async def generate_compression(
    client: AsyncOpenAI,
    model: str,
    row: dict,
    evidence: list[dict],
) -> str:
    contract = derive_answer_contract(row.get("question", ""))
    response = await client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": COMPRESS_SYSTEM},
            {
                "role": "user",
                "content": COMPRESS_USER.format(
                    question=row.get("question", ""),
                    contract=json.dumps(contract, ensure_ascii=False),
                    answer=row.get("prediction", ""),
                    evidence=evidence_prompt(evidence),
                ),
            },
        ],
    )
    return sanitize_evidence_labels(response.choices[0].message.content or "")


async def verify_semantic_fidelity(
    client: AsyncOpenAI,
    model: str,
    row: dict,
    compressed: str,
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
                    evidence=evidence_prompt(evidence),
                    original=row.get("prediction", ""),
                    compressed=compressed,
                ),
            },
        ],
    )
    parsed = parse_json_object(response.choices[0].message.content or "")
    try:
        coverage = max(0.0, min(1.0, float(parsed.get("essential_fact_coverage", 0.0))))
    except (TypeError, ValueError):
        coverage = 0.0
    parsed["essential_fact_coverage"] = round(coverage, 6)
    parsed["reason"] = clean_text(parsed.get("reason", ""))
    for key in ("essential_facts", "lost_essential_facts", "unsupported_new_claims", "contradictions"):
        value = parsed.get(key)
        parsed[key] = value if isinstance(value, list) else []
    return parsed


def should_attempt(row: dict, requested_lines: set[int] | None) -> bool:
    line_no = int(row.get("line_no", -1))
    if requested_lines is not None:
        return line_no in requested_lines
    return bool((row.get("evidence_mismatch_repair") or {}).get("accepted"))


def acceptance(original: str, compressed: str, text_index: EvidenceIndex, semantic: dict) -> tuple[bool, dict]:
    graph_index = EvidenceIndex.build([])
    original_traces, original_metrics = audit_answer(original, text_index, graph_index)
    compressed_traces, compressed_metrics = audit_answer(compressed, text_index, graph_index)
    original_len = len(clean_text(original))
    compressed_len = len(clean_text(compressed))
    length_reduction = 1 - compressed_len / max(1, original_len)
    lost = semantic.get("lost_essential_facts") or []
    unsupported = semantic.get("unsupported_new_claims") or []
    contradictions = semantic.get("contradictions") or []
    coverage = float(semantic.get("essential_fact_coverage", 0.0))
    original_unsupported_ratio = float(original_metrics.get("unsupported_ratio", 0.0))
    compressed_unsupported_ratio = float(compressed_metrics.get("unsupported_ratio", 0.0))
    checks = {
        "candidate_present": compressed_len >= 80,
        "meaningfully_shorter": length_reduction >= 0.18,
        "candidate_not_refusal": not contains_refusal(compressed),
        "semantic_verifier_approved": bool(semantic.get("approved")),
        "essential_fact_coverage": coverage >= 0.82,
        "no_lost_essential_facts": not lost,
        "no_unsupported_new_claims": not unsupported,
        "no_contradictions": not contradictions,
        "support_preserved": float(compressed_metrics.get("mean_support", 0.0))
        >= max(0.45, float(original_metrics.get("mean_support", 0.0)) - 0.12),
        "unsupported_ratio_controlled": compressed_unsupported_ratio <= max(0.35, original_unsupported_ratio + 0.10),
    }
    guard = {
        "accepted": all(checks.values()),
        "checks": checks,
        "original_chars": original_len,
        "compressed_chars": compressed_len,
        "length_reduction": round(length_reduction, 6),
        "original_claim_metrics": original_metrics,
        "compressed_claim_metrics": compressed_metrics,
        "original_claim_traces": original_traces,
        "compressed_claim_traces": compressed_traces,
        "semantic_fidelity_verifier": semantic,
        "rejection_reasons": [name for name, passed in checks.items() if not passed],
    }
    return all(checks.values()), guard


def write_summary(path: Path, rows: list[dict], source: Path) -> None:
    agents = [row.get("conservative_repair_compression") or {} for row in rows]
    enabled = [agent for agent in agents if agent.get("enabled")]
    attempted = [agent for agent in enabled if agent.get("attempted")]
    accepted = [agent for agent in attempted if agent.get("accepted")]
    actions = Counter(agent.get("action", "not_applicable") for agent in agents)
    before_chars = sum((agent.get("guard") or {}).get("original_chars", 0) for agent in accepted)
    after_chars = sum((agent.get("guard") or {}).get("compressed_chars", 0) for agent in accepted)
    lines = [
        "# SAGE v45 Conservative Repair Compression Summary\n\n",
        f"- Source: `{source.as_posix()}`\n",
        f"- Samples: {len(rows)}\n",
        "- Triggered only for accepted v44 evidence-mismatch repairs unless line numbers are specified.\n",
        "- Reference answers and judge scores are not runtime inputs.\n\n",
        "| Metric | Count |\n",
        "|---|---:|\n",
        f"| Enabled | {len(enabled)} |\n",
        f"| Attempted | {len(attempted)} |\n",
        f"| Accepted | {len(accepted)} |\n",
        f"| Accepted answer characters | {before_chars} -> {after_chars} |\n",
        f"| Accepted character reduction | {((1 - after_chars / before_chars) * 100 if before_chars else 0):.2f}% |\n\n",
        "## Actions\n\n",
        "| Action | Count |\n",
        "|---|---:|\n",
    ]
    for action, count in sorted(actions.items()):
        lines.append(f"| {action} | {count} |\n")
    lines.extend([
        "\n## Per-Sample Decisions\n\n",
        "| Line | Dataset | Action | Accepted | Chars | Reduction | Reasons |\n",
        "|---:|---|---|---|---:|---:|---|\n",
    ])
    for row in rows:
        agent = row.get("conservative_repair_compression") or {}
        if not agent.get("enabled"):
            continue
        guard = agent.get("guard") or {}
        reasons = "; ".join(guard.get("rejection_reasons", [])).replace("|", "/")
        lines.append(
            f"| {row.get('line_no')} | {row.get('dataset')} | {agent.get('action')} | "
            f"{agent.get('accepted')} | {guard.get('original_chars', 0)} -> {guard.get('compressed_chars', 0)} | "
            f"{float(guard.get('length_reduction', 0.0)) * 100:.2f}% | {reasons or 'none'} |\n"
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
            new_row["conservative_repair_compression"] = {"enabled": False}
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
            "version": "v45",
            "attempted": True,
            "action": "retain_repair",
            "accepted": False,
            "evidence_windows": len(index.records),
            "selected_evidence": evidence,
            "reference_answer_used": False,
        }
        try:
            compressed = await generate_compression(client, model, row, evidence)
            semantic = await verify_semantic_fidelity(client, model, row, compressed, evidence)
            accepted, guard = acceptance(row.get("prediction", ""), compressed, index, semantic)
            agent["compressed_candidate"] = compressed
            agent["guard"] = guard
            agent["accepted"] = accepted
            if accepted:
                new_row["prediction_before_conservative_repair_compression"] = new_row.get("prediction", "")
                new_row["prediction"] = compressed
                new_row["method"] = "sage_v45_conservative_repair_compression"
                agent["action"] = "use_compressed_repair"
            else:
                new_row["rejected_conservative_repair_compression"] = compressed
        except Exception as exc:
            agent["error"] = f"{type(exc).__name__}: {exc}"[:300]
            agent["action"] = "error_keep_repair"

        new_row["conservative_repair_compression"] = agent
        output_rows.append(new_row)

    output_path = ROOT_DIR / args.output
    write_jsonl(output_rows, output_path)
    write_summary(ROOT_DIR / args.summary, output_rows, ROOT_DIR / args.input)
    print(f"Saved v45 conservative repair compression results to {output_path}")
    print(f"Saved summary to {ROOT_DIR / args.summary}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()

