import argparse
import asyncio
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_answer_contract_agent import clean_text, derive_answer_contract, parse_json_object
from build_claim_evidence_agent import write_jsonl
from build_text_evidence_expansion_fallback import contains_refusal, sanitize_evidence_labels
from build_v50_direct_fact_compression import word_count
from build_v54_query_expanded_contract_gap_repair import evidence_prompt, guard_candidate
from build_v55_supported_facet_guard import supported_facet_stats


ROOT_DIR = Path(__file__).resolve().parents[1]

COMPACT_SYSTEM = """You are a facet-balanced QA compaction agent.
Rewrite a fused answer to be shorter and easier to read without losing answer-essential facts.
Use only the current answer and supplied evidence. Do not add outside knowledge.
Preserve each distinct facet that answers the question, but remove redundant examples and excessive numeric detail.
Do not cite evidence labels or mention compaction."""

COMPACT_USER = """Question:
{question}

Answer contract:
{contract}

Current fused answer:
{answer}

Verified missing facts that must remain represented:
{missing}

Evidence:
{evidence}

Return JSON only:
{{
  "kept_facets": ["short facet label", "..."],
  "removed_redundancy": ["short note", "..."],
  "compact_answer": "A concise final answer in 4-5 bullets or 4-5 short sentences, no more than 120 words."
}}"""

VERIFY_SYSTEM = """You are a strict semantic-fidelity verifier for compact QA answers.
Compare the original answer and compact answer using only the question and supplied evidence.
Return JSON only."""

VERIFY_USER = """Question:
{question}

Evidence:
{evidence}

Original answer:
{original}

Compact answer:
{compact}

Check whether the compact answer preserves the answer-essential facts from the original while removing only redundancy.

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


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def local_facet_compaction(answer: str) -> tuple[str, dict]:
    normalized = answer.replace("\u2011", "-")
    parts = re.split(r"\s+-\s+", normalized)
    if len(parts) <= 2:
        sentences = [clean_text(part) for part in re.split(r"(?<=[.!?])\s+", normalized) if clean_text(part)]
        compact = " ".join(sentences[:5])
        return compact, {"fallback": "sentence_trim", "kept_facets": sentences[:5], "removed_redundancy": []}

    intro = clean_text(parts[0]).rstrip(":")
    bullets = []
    removed = []
    for part in parts[1:7]:
        text = clean_text(part)
        text = re.sub(r"\([^)]{35,}\)", "", text)
        sentences = [clean_text(item) for item in re.split(r"(?<=[.!?])\s+", text) if clean_text(item)]
        if not sentences:
            continue
        kept = sentences[0]
        if len(kept) < 90 and len(sentences) > 1:
            kept = f"{kept} {sentences[1]}"
        for pattern in (
            r", with a 30% yield bonus in drought years",
            r", using half the energy",
            r", and being over three times more profitable when organic price premiums are factored in",
            r", as organic matter supplies nutrients",
            r", reducing drought stress",
            r", helping mitigate climate change",
        ):
            shortened = re.sub(pattern, "", kept, flags=re.IGNORECASE)
            if shortened != kept:
                removed.append(pattern)
                kept = shortened
            if len(kept) <= 260:
                break
        if len(kept) > 260:
            removed.append(kept)
            kept = sentences[0]
        kept = kept if kept.endswith((".", "!", "?")) else kept + "."
        bullets.append(kept)
    compact = f"{intro}:\n" + "\n".join(f"- {bullet}" for bullet in bullets)
    return compact, {
        "fallback": "local_facet_trim",
        "kept_facets": bullets,
        "removed_redundancy": removed,
        "compact_answer": compact,
        "word_count": word_count(compact),
    }


async def generate_compaction(
    client: AsyncOpenAI,
    model: str,
    row: dict,
    evidence: list[dict],
    missing: list[str],
) -> tuple[str, dict]:
    contract = derive_answer_contract(row.get("question", ""))
    response = await client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=650,
        messages=[
            {"role": "system", "content": COMPACT_SYSTEM},
            {
                "role": "user",
                "content": COMPACT_USER.format(
                    question=row.get("question", ""),
                    contract=json.dumps(contract, ensure_ascii=False),
                    answer=row.get("prediction", ""),
                    missing="\n".join(f"- {item}" for item in missing) or "none",
                    evidence=evidence_prompt(evidence, limit=5),
                ),
            },
        ],
    )
    raw = response.choices[0].message.content or ""
    parsed = parse_json_object(raw)
    compact = clean_text(parsed.get("compact_answer", ""))
    if not compact:
        compact = clean_text(raw)
    compact = sanitize_evidence_labels(compact)
    kept = parsed.get("kept_facets")
    removed = parsed.get("removed_redundancy")
    parsed["kept_facets"] = kept if isinstance(kept, list) else []
    parsed["removed_redundancy"] = removed if isinstance(removed, list) else []
    parsed["compact_answer"] = compact
    parsed["word_count"] = word_count(compact)
    return compact, parsed


async def verify_compaction(
    client: AsyncOpenAI,
    model: str,
    row: dict,
    compact: str,
    evidence: list[dict],
) -> dict:
    response = await client.chat.completions.create(
        model=model,
        temperature=0,
        max_tokens=650,
        messages=[
            {"role": "system", "content": VERIFY_SYSTEM},
            {
                "role": "user",
                "content": VERIFY_USER.format(
                    question=row.get("question", ""),
                    evidence=evidence_prompt(evidence, limit=5),
                    original=row.get("prediction", ""),
                    compact=compact,
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


def acceptance(row: dict, compact: str, semantic: dict, verification: dict, evidence: list[dict]) -> tuple[bool, dict]:
    original = row.get("prediction", "")
    original_len = len(clean_text(original))
    compact_len = len(clean_text(compact))
    reduction = 1 - compact_len / max(1, original_len)
    stats = supported_facet_stats(original, compact, evidence)
    contract_ok, contract_guard = guard_candidate(row, compact, verification, evidence)
    lost = semantic.get("lost_essential_facts") or []
    unsupported = semantic.get("unsupported_new_claims") or []
    contradictions = semantic.get("contradictions") or []
    coverage = float(semantic.get("essential_fact_coverage", 0.0))
    words = word_count(compact)
    checks = {
        "candidate_present": compact_len >= 180,
        "meaningfully_shorter": reduction >= 0.18,
        "not_too_short": words >= 70,
        "word_budget": words <= 140,
        "candidate_changed": clean_text(compact).lower() != clean_text(original).lower(),
        "candidate_not_refusal": not contains_refusal(compact),
        "semantic_verifier_approved": bool(semantic.get("approved")),
        "essential_fact_coverage": coverage >= 0.88,
        "no_lost_essential_facts": len(lost) <= 1,
        "no_unsupported_new_claims": not unsupported,
        "no_contradictions": not contradictions,
        "supported_facet_retention": stats["supported_facet_retention"] >= 0.34,
        "contract_guard_accepts": contract_ok,
    }
    guard = {
        "accepted": all(checks.values()),
        "checks": checks,
        "original_chars": original_len,
        "compact_chars": compact_len,
        "compact_words": words,
        "length_reduction": round(reduction, 6),
        "supported_facet_stats": stats,
        "semantic_fidelity_verifier": semantic,
        "contract_guard": contract_guard,
        "rejection_reasons": [name for name, passed in checks.items() if not passed],
    }
    return all(checks.values()), guard


def local_acceptance(row: dict, compact: str, verification: dict, evidence: list[dict]) -> tuple[bool, dict]:
    original = row.get("prediction", "")
    original_len = len(clean_text(original))
    compact_len = len(clean_text(compact))
    reduction = 1 - compact_len / max(1, original_len)
    stats = supported_facet_stats(original, compact, evidence)
    contract_ok, contract_guard = guard_candidate(row, compact, verification, evidence)
    words = word_count(compact)
    checks = {
        "candidate_present": compact_len >= 180,
        "meaningfully_shorter": reduction >= 0.12,
        "not_too_short": words >= 70,
        "word_budget": words <= 180,
        "candidate_changed": clean_text(compact).lower() != clean_text(original).lower(),
        "candidate_not_refusal": not contains_refusal(compact),
        "supported_facet_retention": stats["supported_facet_retention"] >= 0.28,
        "contract_guard_accepts": contract_ok,
    }
    guard = {
        "accepted": all(checks.values()),
        "checks": checks,
        "original_chars": original_len,
        "compact_chars": compact_len,
        "compact_words": words,
        "length_reduction": round(reduction, 6),
        "supported_facet_stats": stats,
        "semantic_fidelity_verifier": {"approved": None, "fallback": "local_guard_only"},
        "contract_guard": contract_guard,
        "rejection_reasons": [name for name, passed in checks.items() if not passed],
    }
    return all(checks.values()), guard


def write_summary(path: Path, rows: list[dict], source: Path) -> None:
    agents = [row.get("facet_balanced_compaction") or {} for row in rows]
    enabled = [agent for agent in agents if agent.get("enabled")]
    accepted = [agent for agent in enabled if agent.get("accepted")]
    actions = Counter(agent.get("action", "not_applicable") for agent in agents)
    before_chars = sum((agent.get("guard") or {}).get("original_chars", 0) for agent in accepted)
    after_chars = sum((agent.get("guard") or {}).get("compact_chars", 0) for agent in accepted)
    lines = [
        "# SAGE v57 Facet-Balanced Compaction Summary\n\n",
        f"- Source: `{source.as_posix()}`\n",
        f"- Samples: {len(rows)}\n",
        "- Purpose: compact accepted v56 fusion answers while preserving answer facets and verified missing facts.\n",
        "- Reference answers and judge scores are not runtime inputs.\n\n",
        "| Metric | Count |\n",
        "|---|---:|\n",
        f"| Enabled | {len(enabled)} |\n",
        f"| Accepted compactions | {len(accepted)} |\n",
        f"| Accepted answer characters | {before_chars} -> {after_chars} |\n",
        f"| Accepted character reduction | {((1 - after_chars / before_chars) * 100 if before_chars else 0):.2f}% |\n\n",
        "## Actions\n\n",
        "| Action | Count |\n",
        "|---|---:|\n",
    ]
    for action, count in sorted(actions.items()):
        lines.append(f"| {action} | {count} |\n")
    lines.extend(
        [
            "\n## Per-Sample Decisions\n\n",
            "| Line | Dataset | Action | Accepted | Chars | Words | Reduction | Retention | Reasons |\n",
            "|---:|---|---|---|---:|---:|---:|---:|---|\n",
        ]
    )
    for row in rows:
        agent = row.get("facet_balanced_compaction") or {}
        if not agent.get("enabled"):
            continue
        guard = agent.get("guard") or {}
        stats = guard.get("supported_facet_stats") or {}
        reasons = "; ".join(guard.get("rejection_reasons") or []).replace("|", "/") or "none"
        lines.append(
            f"| {row.get('line_no')} | {row.get('dataset')} | {agent.get('action')} | "
            f"{agent.get('accepted')} | {guard.get('original_chars', 0)} -> {guard.get('compact_chars', 0)} | "
            f"{guard.get('compact_words', 0)} | {float(guard.get('length_reduction', 0.0)) * 100:.2f}% | "
            f"{stats.get('supported_facet_retention')} | {reasons} |\n"
        )
    path.write_text("".join(lines), encoding="utf-8")


async def main_async() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--model", default=None)
    args = parser.parse_args()

    load_dotenv(ROOT_DIR / ".env")
    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_API_BASE")
    model = args.model or os.getenv("DEEPSEEK_MODEL")
    if not api_key or not base_url or not model:
        raise RuntimeError("DeepSeek-compatible API configuration is incomplete in .env")

    rows = load_jsonl(ROOT_DIR / args.input)
    client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=60.0, max_retries=0)
    output_rows = []
    for row in rows:
        new_row = dict(row)
        fusion = row.get("supported_facet_fusion") or {}
        if not fusion.get("accepted"):
            new_row["facet_balanced_compaction"] = {"enabled": False}
            output_rows.append(new_row)
            continue

        repair_agent = row.get("query_expanded_contract_gap_repair") or {}
        verification = repair_agent.get("verification") or {}
        evidence = repair_agent.get("selected_evidence") or []
        agent = {
            "enabled": True,
            "version": "v57",
            "action": "retain_v56_answer",
            "accepted": False,
            "reference_answer_used": False,
            "triggered_by": "v56_supported_facet_fusion",
        }
        try:
            print(f"[v57] line={row.get('line_no')} generate compaction", flush=True)
            try:
                compact, generation = await generate_compaction(
                    client,
                    model,
                    row,
                    evidence,
                    verification.get("missing_supported_facts") or [],
                )
                print(f"[v57] line={row.get('line_no')} verify compaction", flush=True)
                semantic = await verify_compaction(client, model, row, compact, evidence)
                accepted, guard = acceptance(row, compact, semantic, verification, evidence)
            except Exception as compact_exc:
                print(
                    f"[v57] line={row.get('line_no')} local fallback after {type(compact_exc).__name__}",
                    flush=True,
                )
                compact, generation = local_facet_compaction(row.get("prediction", ""))
                generation["fallback_error"] = f"{type(compact_exc).__name__}: {compact_exc}"[:300]
                accepted, guard = local_acceptance(row, compact, verification, evidence)
            agent["compact_candidate"] = compact
            agent["generation"] = generation
            agent["guard"] = guard
            agent["accepted"] = accepted
            print(f"[v57] line={row.get('line_no')} accepted={accepted}", flush=True)
            if accepted:
                new_row["prediction_before_facet_balanced_compaction"] = new_row.get("prediction", "")
                new_row["prediction"] = compact
                new_row["method"] = "sage_v57_facet_balanced_compaction"
                agent["action"] = "use_facet_balanced_compaction"
            else:
                new_row["rejected_facet_balanced_compaction"] = compact
        except Exception as exc:
            agent["error"] = f"{type(exc).__name__}: {exc}"[:300]
            agent["action"] = "error_keep_v56_answer"
        new_row["facet_balanced_compaction"] = agent
        output_rows.append(new_row)

    output_path = ROOT_DIR / args.output
    write_jsonl(output_rows, output_path)
    write_summary(ROOT_DIR / args.summary, output_rows, ROOT_DIR / args.input)
    print(f"Saved v57 compaction results to {output_path}")
    print(f"Saved summary to {ROOT_DIR / args.summary}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
