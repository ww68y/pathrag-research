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

from build_answer_contract_agent import clean_text
from build_claim_evidence_agent import write_jsonl
from build_text_evidence_expansion_fallback import contains_refusal, sanitize_evidence_labels
from build_v54_query_expanded_contract_gap_repair import evidence_prompt, guard_candidate
from build_v55_supported_facet_guard import supported_facet_stats


ROOT_DIR = Path(__file__).resolve().parents[1]

FUSION_SYSTEM = """You are a reference-free evidence-bounded answer fusion agent.
Use only the supplied evidence, original answer, and repair candidate.
Your job is to preserve useful evidence-supported facets from the original answer while adding verified missing facts from the repair candidate.
Do not use outside knowledge. Do not cite evidence labels or mention verification."""

FUSION_USER = """Question:
{question}

Original answer before repair:
{original}

Original answer facet outline:
{original_facets}

Over-compressed repair candidate:
{candidate}

Verified missing supported facts from the contract-gap verifier:
{missing}

Evidence-supported terms that the over-compressed candidate dropped from the original answer:
{dropped_terms}

Evidence:
{evidence}

Write a fused final answer.
Requirements:
- Answer the question directly.
- Preserve the original answer's broad evidence-supported facets when relevant.
- If the original answer is organized into numbered or bulleted facets, keep those distinct facets unless evidence contradicts them.
- Do not collapse a multifactor original answer into only the repair candidate's new quantitative details.
- Add the verified missing supported facts if evidence supports them.
- Remove unsupported, off-topic, or repetitive details.
- Prefer 4-7 concise sentences or a compact numbered list when the question asks for several factors.
- Do not cite evidence labels, source ids, references, or this instruction."""


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def original_facet_outline(answer: str) -> str:
    facets = []
    for raw_line in answer.splitlines():
        line = clean_text(raw_line)
        if not line:
            continue
        if line[:2].isdigit() or line.startswith(("-", "*")) or line[:1].isdigit():
            facets.append(line)
    if not facets:
        sentences = [clean_text(part) for part in answer.replace("\n", " ").split(".") if clean_text(part)]
        facets = sentences[:6]
    return "\n".join(f"- {facet}" for facet in facets[:8]) or "none"


async def generate_fusion(
    client: AsyncOpenAI,
    model: str,
    row: dict,
    original: str,
    candidate: str,
    missing: list[str],
    dropped_terms: list[str],
    evidence: list[dict],
) -> str:
    response = await client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": FUSION_SYSTEM},
            {
                "role": "user",
                "content": FUSION_USER.format(
                    question=row.get("question", ""),
                    original=original,
                    original_facets=original_facet_outline(original),
                    candidate=candidate,
                    missing="\n".join(f"- {item}" for item in missing) or "none",
                    dropped_terms=", ".join(dropped_terms[:80]) or "none",
                    evidence=evidence_prompt(evidence, limit=10),
                ),
            },
        ],
    )
    return sanitize_evidence_labels(response.choices[0].message.content or "")


def guard_fusion(
    row: dict,
    fused: str,
    original: str,
    verification: dict,
    evidence: list[dict],
    reverted_stats: dict,
) -> tuple[bool, dict]:
    stats = supported_facet_stats(original, fused, evidence)
    accepted_by_v54_guard, contract_guard = guard_candidate(row, fused, verification, evidence)
    fused_len = len(clean_text(fused))
    original_len = len(clean_text(original))
    reverted_retention = float(reverted_stats.get("supported_facet_retention") or 0.0)
    retention_floor = max(0.30, reverted_retention + 0.10)
    checks = {
        "candidate_present": fused_len >= 120,
        "candidate_changed": clean_text(fused).lower() != clean_text(original).lower(),
        "no_refusal": not contains_refusal(fused),
        "retains_supported_facets": stats["supported_facet_retention"] >= retention_floor,
        "not_overlong": fused_len <= max(1700, int(original_len * 1.18)),
        "contract_guard_accepts": accepted_by_v54_guard,
    }
    guard = {
        "accepted": all(checks.values()),
        "checks": checks,
        "supported_facet_stats": stats,
        "reverted_repair_retention": round(reverted_retention, 6),
        "required_retention_floor": round(retention_floor, 6),
        "contract_guard": contract_guard,
        "rejection_reasons": [name for name, passed in checks.items() if not passed],
    }
    return all(checks.values()), guard


def write_summary(path: Path, rows: list[dict], source: Path) -> None:
    agents = [row.get("supported_facet_fusion") or {} for row in rows]
    enabled = [agent for agent in agents if agent.get("enabled")]
    accepted = [agent for agent in enabled if agent.get("accepted")]
    actions = Counter(agent.get("action", "not_applicable") for agent in agents)
    lines = [
        "# SAGE v56 Supported-Facet Fusion Summary\n\n",
        f"- Source: `{source.as_posix()}`\n",
        f"- Samples: {len(rows)}\n",
        "- Purpose: recover from over-compressed v54 repairs by fusing the original answer with verified missing facts.\n",
        "- Trigger: v55 `revert_to_pre_repair_answer` decisions.\n",
        "- Reference answers and judge scores are not runtime inputs.\n\n",
        "| Metric | Count |\n",
        "|---|---:|\n",
        f"| Enabled | {len(enabled)} |\n",
        f"| Accepted fusions | {len(accepted)} |\n\n",
        "## Actions\n\n",
        "| Action | Count |\n",
        "|---|---:|\n",
    ]
    for action, count in sorted(actions.items()):
        lines.append(f"| {action} | {count} |\n")
    lines.extend(
        [
            "\n## Per-Sample Decisions\n\n",
            "| Line | Dataset | Action | Accepted | Retention | Compression | Reasons |\n",
            "|---:|---|---|---|---:|---:|---|\n",
        ]
    )
    for row in rows:
        agent = row.get("supported_facet_fusion") or {}
        if not agent.get("enabled"):
            continue
        guard = agent.get("guard") or {}
        stats = guard.get("supported_facet_stats") or {}
        reasons = "; ".join(guard.get("rejection_reasons") or []).replace("|", "/") or "none"
        lines.append(
            f"| {row.get('line_no')} | {row.get('dataset')} | {agent.get('action')} | "
            f"{agent.get('accepted')} | {stats.get('supported_facet_retention')} | "
            f"{stats.get('compression_ratio')} | {reasons} |\n"
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
    client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=120.0, max_retries=1)
    output_rows = []
    for row in rows:
        new_row = dict(row)
        facet_guard = row.get("supported_facet_guard") or {}
        if facet_guard.get("action") != "revert_to_pre_repair_answer":
            new_row["supported_facet_fusion"] = {"enabled": False}
            output_rows.append(new_row)
            continue

        original = row.get("prediction") or row.get("prediction_before_query_expanded_contract_gap_repair", "")
        compressed_candidate = row.get("prediction_before_supported_facet_guard", "")
        repair_agent = row.get("query_expanded_contract_gap_repair") or {}
        verification = repair_agent.get("verification") or {}
        evidence = repair_agent.get("selected_evidence") or []
        dropped_terms = (facet_guard.get("stats") or {}).get("dropped_supported_term_examples") or []
        agent = {
            "enabled": True,
            "version": "v56",
            "action": "retain_v55_answer",
            "accepted": False,
            "reference_answer_used": False,
            "triggered_by": "v55_supported_facet_guard_revert",
        }
        try:
            print(f"[v56] line={row.get('line_no')} generate fusion", flush=True)
            fused = await generate_fusion(
                client,
                model,
                row,
                original,
                compressed_candidate,
                verification.get("missing_supported_facts") or [],
                dropped_terms,
                evidence,
            )
            accepted, guard = guard_fusion(
                row,
                fused,
                original,
                verification,
                evidence,
                facet_guard.get("stats") or {},
            )
            agent["candidate_answer"] = fused
            agent["guard"] = guard
            agent["accepted"] = accepted
            print(f"[v56] line={row.get('line_no')} accepted={accepted}", flush=True)
            if accepted:
                new_row["prediction_before_supported_facet_fusion"] = new_row.get("prediction", "")
                new_row["prediction"] = fused
                new_row["method"] = "sage_v56_supported_facet_fusion"
                agent["action"] = "use_supported_facet_fusion"
            else:
                new_row["rejected_supported_facet_fusion"] = fused
        except Exception as exc:
            agent["error"] = f"{type(exc).__name__}: {exc}"[:300]
            agent["action"] = "error_keep_current"
        new_row["supported_facet_fusion"] = agent
        output_rows.append(new_row)

    output_path = ROOT_DIR / args.output
    write_jsonl(output_rows, output_path)
    write_summary(ROOT_DIR / args.summary, output_rows, ROOT_DIR / args.input)
    print(f"Saved v56 fusion results to {output_path}")
    print(f"Saved summary to {ROOT_DIR / args.summary}")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
