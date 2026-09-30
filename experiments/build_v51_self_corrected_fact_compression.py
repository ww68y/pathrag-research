import argparse
import asyncio
import json
import os
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

from build_answer_contract_agent import clean_text, derive_answer_contract, parse_json_object
from build_claim_evidence_agent import EvidenceIndex, load_full_source_rows, normalize_space, write_jsonl
from build_evidence_preserving_compressor import audit_answer
from build_text_evidence_expansion_fallback import (
    build_full_text_index,
    expanded_evidence,
    load_jsonl,
    parse_line_nos,
    sanitize_evidence_labels,
    source_by_subset_line,
)
from build_v45_conservative_repair_compression import contains_refusal, evidence_prompt, verify_semantic_fidelity


ROOT_DIR = Path(__file__).resolve().parents[1]

REPAIR_SYSTEM = """You are a self-correcting QA compression agent.
Revise a rejected compressed answer by restoring every lost essential fact from the verifier feedback.
Use only the current accepted answer and supplied evidence. Do not add outside knowledge.
Keep the revised answer concise: no more than 85 words, preferably 2 sentences.
Directly answer the question. Do not cite evidence labels or mention compression."""

REPAIR_USER = """Question:
{question}

Answer contract:
{contract}

Current accepted answer:
{answer}

Rejected compressed candidate:
{candidate}

Verifier feedback from the rejected candidate:
{feedback}

Evidence:
{evidence}

Return JSON only:
{{
  "restored_facts": ["lost fact restored", "..."],
  "revised_answer": "A direct concise answer that restores all lost essential facts."
}}"""


def word_count(text: str) -> int:
    return len(clean_text(text).split())


def should_attempt(row: dict, requested_lines: set[int] | None) -> bool:
    line_no = int(row.get("line_no", -1))
    if requested_lines is not None:
        return line_no in requested_lines
    agent = row.get("direct_fact_compression") or {}
    return bool(agent.get("attempted")) and not bool(agent.get("accepted")) and bool(row.get("rejected_direct_fact_compression"))


async def generate_revision(
    client: AsyncOpenAI,
    model: str,
    row: dict,
    evidence: list[dict],
) -> tuple[str, dict]:
    contract = derive_answer_contract(row.get("question", ""))
    previous_agent = row.get("direct_fact_compression") or {}
    previous_guard = previous_agent.get("guard") or {}
    feedback = previous_guard.get("semantic_fidelity_verifier") or {}
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
                    candidate=row.get("rejected_direct_fact_compression", ""),
                    feedback=json.dumps(feedback, ensure_ascii=False),
                    evidence=evidence_prompt(evidence, limit=10),
                ),
            },
        ],
    )
    raw = response.choices[0].message.content or ""
    parsed = parse_json_object(raw)
    revised = clean_text(parsed.get("revised_answer", ""))
    if not revised:
        revised = clean_text(raw)
    revised = sanitize_evidence_labels(revised)
    restored = parsed.get("restored_facts")
    parsed["restored_facts"] = restored if isinstance(restored, list) else []
    parsed["revised_answer"] = revised
    parsed["word_count"] = word_count(revised)
    return revised, parsed


def acceptance(original: str, revised: str, text_index: EvidenceIndex, semantic: dict) -> tuple[bool, dict]:
    graph_index = EvidenceIndex.build([])
    original_traces, original_metrics = audit_answer(original, text_index, graph_index)
    revised_traces, revised_metrics = audit_answer(revised, text_index, graph_index)
    original_len = len(clean_text(original))
    revised_len = len(clean_text(revised))
    length_reduction = 1 - revised_len / max(1, original_len)
    lost = semantic.get("lost_essential_facts") or []
    unsupported = semantic.get("unsupported_new_claims") or []
    contradictions = semantic.get("contradictions") or []
    coverage = float(semantic.get("essential_fact_coverage", 0.0))
    original_support = float(original_metrics.get("mean_support", 0.0))
    revised_support = float(revised_metrics.get("mean_support", 0.0))
    original_unsupported_ratio = float(original_metrics.get("unsupported_ratio", 0.0))
    revised_unsupported_ratio = float(revised_metrics.get("unsupported_ratio", 0.0))
    words = word_count(revised)

    checks = {
        "candidate_present": revised_len >= 100,
        "meaningfully_shorter": length_reduction >= 0.18,
        "direct_length": words <= 90,
        "not_overcompressed": revised_len >= 300,
        "candidate_not_refusal": not contains_refusal(revised),
        "semantic_verifier_approved": bool(semantic.get("approved")),
        "essential_fact_coverage": coverage >= 0.90,
        "no_lost_essential_facts": not lost,
        "no_unsupported_new_claims": not unsupported,
        "no_contradictions": not contradictions,
        "support_preserved": revised_support >= max(0.36, original_support - 0.24),
        "unsupported_ratio_controlled": revised_unsupported_ratio <= max(0.42, original_unsupported_ratio + 0.20),
        "claim_count_not_collapsed": int(revised_metrics.get("claims", 0)) >= 2,
    }
    guard = {
        "accepted": all(checks.values()),
        "checks": checks,
        "original_chars": original_len,
        "revised_chars": revised_len,
        "revised_words": words,
        "length_reduction": round(length_reduction, 6),
        "original_claim_metrics": original_metrics,
        "revised_claim_metrics": revised_metrics,
        "original_claim_traces": original_traces,
        "revised_claim_traces": revised_traces,
        "semantic_fidelity_verifier": semantic,
        "rejection_reasons": [name for name, passed in checks.items() if not passed],
    }
    return all(checks.values()), guard


def write_summary(path: Path, rows: list[dict], source: Path) -> None:
    agents = [row.get("self_corrected_fact_compression") or {} for row in rows]
    enabled = [agent for agent in agents if agent.get("enabled")]
    attempted = [agent for agent in enabled if agent.get("attempted")]
    accepted = [agent for agent in attempted if agent.get("accepted")]
    actions = Counter(agent.get("action", "not_applicable") for agent in agents)
    before_chars = sum((agent.get("guard") or {}).get("original_chars", 0) for agent in accepted)
    after_chars = sum((agent.get("guard") or {}).get("revised_chars", 0) for agent in accepted)
    lines = [
        "# SAGE v51 Self-Corrected Fact Compression Summary\n\n",
        f"- Source: `{source.as_posix()}`\n",
        f"- Samples: {len(rows)}\n",
        "- Triggered for rejected v50 direct fact compressions unless line numbers are specified.\n",
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
        "| Line | Dataset | Action | Accepted | Chars | Words | Reduction | Reasons |\n",
        "|---:|---|---|---|---:|---:|---:|---|\n",
    ])
    for row in rows:
        agent = row.get("self_corrected_fact_compression") or {}
        if not agent.get("enabled"):
            continue
        guard = agent.get("guard") or {}
        reasons = "; ".join(guard.get("rejection_reasons", [])).replace("|", "/")
        lines.append(
            f"| {row.get('line_no')} | {row.get('dataset')} | {agent.get('action')} | "
            f"{agent.get('accepted')} | {guard.get('original_chars', 0)} -> {guard.get('revised_chars', 0)} | "
            f"{guard.get('revised_words', 0)} | {float(guard.get('length_reduction', 0.0)) * 100:.2f}% | "
            f"{reasons or 'none'} |\n"
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
    parser.add_argument("--top-k", type=int, default=12)
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
            new_row["self_corrected_fact_compression"] = {"enabled": False}
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
        index = build_full_text_index(full_context, args.window_chars, args.overlap_chars)
        evidence = expanded_evidence(index, row.get("question", ""), row.get("prediction", ""), args.top_k)
        agent = {
            "enabled": True,
            "version": "v51",
            "attempted": True,
            "action": "retain_v49_compression",
            "accepted": False,
            "evidence_windows": len(index.records),
            "selected_evidence": evidence,
            "reference_answer_used": False,
        }
        try:
            revised, generation = await generate_revision(client, model, row, evidence)
            semantic = await verify_semantic_fidelity(client, model, row, revised, evidence)
            accepted, guard = acceptance(row.get("prediction", ""), revised, index, semantic)
            agent["generation"] = generation
            agent["revised_candidate"] = revised
            agent["guard"] = guard
            agent["accepted"] = accepted
            if accepted:
                new_row["prediction_before_self_corrected_fact_compression"] = new_row.get("prediction", "")
                new_row["prediction"] = revised
                new_row["method"] = "sage_v51_self_corrected_fact_compression"
                agent["action"] = "use_self_corrected_fact_compression"
            else:
                new_row["rejected_self_corrected_fact_compression"] = revised
        except Exception as exc:
            agent["error"] = f"{type(exc).__name__}: {exc}"[:300]
            agent["action"] = "error_keep_v49_compression"

        new_row["self_corrected_fact_compression"] = agent
        output_rows.append(new_row)

    output_path = ROOT_DIR / args.output
    write_jsonl(output_rows, output_path)
    write_summary(ROOT_DIR / args.summary, output_rows, ROOT_DIR / args.input)
    print(f"Saved v51 self-corrected compression results to {output_path}")
    print(f"Saved summary to {ROOT_DIR / args.summary}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
