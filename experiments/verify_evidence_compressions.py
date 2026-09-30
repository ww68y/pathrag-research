import argparse
import asyncio
import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

from build_answer_contract_agent import clean_text, load_jsonl, parse_json_object
from build_claim_evidence_agent import write_jsonl
from build_evidence_preserving_compressor import write_summary


ROOT_DIR = Path(__file__).resolve().parents[1]
SYSTEM_PROMPT = """You are a strict semantic-fidelity verifier for answer compression.
Compare the original and compressed answers only against the question-derived answer contract.
Do not use outside knowledge or reference answers. Distinguish answer-essential facts from repetition,
examples, and background. Return JSON only."""
USER_PROMPT = """Question:
{question}

Answer contract:
{contract}

Original answer:
{original}

Compressed answer:
{compressed}

The deterministic guard has already verified numeric preservation, protected terminology, evidence
grounding, answer shape, and length reduction. Check semantic fidelity:
1. List distinct answer-essential facts in the original.
2. List any essential facts, conditions, exceptions, contrasts, or causal directions lost.
3. List any new unsupported claims or contradictions introduced by the compressed answer.
4. Score essential-fact coverage from 0.0 to 1.0.
5. Approve only when no critical information is lost or introduced.

Return this exact JSON shape:
{{
  "essential_facts": ["..."],
  "lost_essential_facts": ["..."],
  "unsupported_new_claims": ["..."],
  "contradictions": ["..."],
  "essential_fact_coverage": 0.0,
  "approved": false,
  "reason": "..."
}}"""


def eligible(row: dict) -> bool:
    agent = row.get("evidence_compression_agent") or {}
    verification = agent.get("verification") or {}
    reasons = verification.get("rejection_reasons") or []
    return (
        agent.get("compression_attempted")
        and not agent.get("compression_accepted")
        and bool(row.get("rejected_compressed_prediction"))
        and reasons == ["meaning_preserved"]
    )


async def verify(rows: list[dict], model: str, concurrency: int) -> None:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_API_BASE")
    if not api_key or not base_url or not model:
        raise RuntimeError("DeepSeek-compatible API configuration is incomplete in .env")
    client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=120.0, max_retries=2)
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def verify_row(row: dict) -> None:
        agent = row["evidence_compression_agent"]
        compressed = row["rejected_compressed_prediction"]
        async with semaphore:
            try:
                response = await client.chat.completions.create(
                    model=model,
                    temperature=0,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": USER_PROMPT.format(
                                question=row.get("question", ""),
                                contract=json.dumps(agent.get("contract", {}), ensure_ascii=False),
                                original=row.get("prediction", ""),
                                compressed=compressed,
                            ),
                        },
                    ],
                )
                semantic = parse_json_object(response.choices[0].message.content or "")
            except Exception as exc:
                agent["semantic_fidelity_verifier"] = {
                    "approved": False,
                    "error": f"{type(exc).__name__}: {exc}"[:300],
                }
                return

        coverage = max(0.0, min(1.0, float(semantic.get("essential_fact_coverage", 0))))
        lost = semantic.get("lost_essential_facts") or []
        unsupported = semantic.get("unsupported_new_claims") or []
        contradictions = semantic.get("contradictions") or []
        approved = bool(semantic.get("approved")) and coverage >= 0.82 and not lost and not unsupported and not contradictions
        semantic["essential_fact_coverage"] = round(coverage, 6)
        semantic["approved_by_guard"] = approved
        agent["semantic_fidelity_verifier"] = semantic

        if approved:
            original = row.get("prediction", "")
            row["prediction_before_evidence_compression"] = original
            row["prediction"] = compressed
            row["method"] = "sage_v26_evidence_preserving_compression"
            agent["compression_accepted"] = True
            agent["verification"]["accepted"] = True
            agent["verification"]["checks"]["meaning_preserved"] = True
            agent["verification"]["rejection_reasons"] = []

    await asyncio.gather(*(verify_row(row) for row in rows))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--concurrency", type=int, default=3)
    args = parser.parse_args()

    load_dotenv(ROOT_DIR / ".env")
    rows = load_jsonl(ROOT_DIR / args.input)
    candidates = [row for row in rows if eligible(row)]
    asyncio.run(verify(candidates, args.model or os.getenv("DEEPSEEK_MODEL"), args.concurrency))

    output_path = ROOT_DIR / args.output
    write_jsonl(rows, output_path)
    write_summary(ROOT_DIR / args.summary, rows, output_path)
    print(f"Semantic verification candidates: {len(candidates)}")
    print(f"Saved verified SAGE v26 results to {output_path}")


if __name__ == "__main__":
    main()
