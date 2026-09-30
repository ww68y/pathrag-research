import argparse
import asyncio
import json
import os
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

from build_answer_contract_agent import clean_text, load_jsonl, parse_json_object
from build_claim_evidence_agent import content_tokens, write_jsonl


ROOT_DIR = Path(__file__).resolve().parents[1]
DIMENSIONS = ("contract_coverage", "evidence_grounding", "relevance", "conciseness")
SYSTEM_PROMPT = """You are a conservative post-repair verifier for retrieval-augmented QA.
Compare two answers using only the question, answer contract, and supplied evidence.
Do not use outside knowledge or any reference answer. Prefer a repair only when it clearly improves
the answer without losing supported facts or adding unsupported claims. Return JSON only."""
USER_PROMPT = """Question:
{question}

Answer contract:
{contract}

Evidence:
{evidence}

Answer A:
{answer_a}

Answer B:
{answer_b}

Independently score each answer from 0.0 to 1.0 on:
- contract_coverage: all facts and aspects requested by the question;
- evidence_grounding: claims explicitly supported by supplied evidence;
- relevance: direct focus on the question;
- conciseness: avoids repetition without deleting required facts.

List missing required facts and unsupported claims for each answer. Choose A, B, or tie.

Return this exact JSON shape:
{{
  "answer_a": {{
    "contract_coverage": 0.0,
    "evidence_grounding": 0.0,
    "relevance": 0.0,
    "conciseness": 0.0,
    "missing_required_facts": ["..."],
    "unsupported_claims": ["..."]
  }},
  "answer_b": {{
    "contract_coverage": 0.0,
    "evidence_grounding": 0.0,
    "relevance": 0.0,
    "conciseness": 0.0,
    "missing_required_facts": ["..."],
    "unsupported_claims": ["..."]
  }},
  "winner": "tie",
  "reason": "..."
}}"""


def compact_evidence(agent: dict) -> str:
    lines = []
    for index, item in enumerate(agent.get("question_text_evidence") or [], start=1):
        lines.append(f"[Text {index}] {item.get('text', '')}")
    for index, item in enumerate(agent.get("question_graph_evidence") or [], start=1):
        lines.append(f"[Graph {index}] {item.get('text', '')}")
    return "\n\n".join(lines)


def clamp(value: object) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def normalize_view(result: dict, repair_is_a: bool) -> dict:
    answer_a = result.get("answer_a") or {}
    answer_b = result.get("answer_b") or {}
    repaired = answer_a if repair_is_a else answer_b
    original = answer_b if repair_is_a else answer_a
    for answer in (repaired, original):
        for dimension in DIMENSIONS:
            answer[dimension] = round(clamp(answer.get(dimension)), 6)
        answer["missing_required_facts"] = answer.get("missing_required_facts") or []
        answer["unsupported_claims"] = answer.get("unsupported_claims") or []

    raw_winner = str(result.get("winner", "tie")).strip().lower()
    if raw_winner == "a":
        winner = "repair" if repair_is_a else "original"
    elif raw_winner == "b":
        winner = "original" if repair_is_a else "repair"
    else:
        winner = "tie"
    return {
        "repair_is_a": repair_is_a,
        "winner": winner,
        "repaired": repaired,
        "original": original,
        "reason": clean_text(result.get("reason", "")),
    }


def utility(answer: dict) -> float:
    return sum(clamp(answer.get(dimension)) for dimension in DIMENSIONS) / len(DIMENSIONS)


def agreed_findings(views: list[dict], field: str) -> list[str]:
    if len(views) != 2:
        return []
    left = (views[0].get("repaired") or {}).get(field, [])
    right = (views[1].get("repaired") or {}).get(field, [])
    agreed = []
    for left_item in left:
        left_tokens = content_tokens(str(left_item))
        if not left_tokens:
            continue
        for right_item in right:
            right_tokens = content_tokens(str(right_item))
            overlap = len(left_tokens & right_tokens) / max(1, len(left_tokens | right_tokens))
            if overlap >= 0.35:
                agreed.append(str(left_item))
                break
    return agreed


async def collect_views(rows: list[dict], model: str, concurrency: int) -> None:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_API_BASE")
    if not api_key or not base_url or not model:
        raise RuntimeError("DeepSeek-compatible API configuration is incomplete in .env")
    client = AsyncOpenAI(api_key=api_key, base_url=base_url, timeout=120.0, max_retries=2)
    semaphore = asyncio.Semaphore(max(1, concurrency))

    async def judge(row: dict, repair_is_a: bool) -> None:
        contract_agent = row.get("contract_completion_agent") or {}
        consensus = row["post_repair_consensus_agent"]
        repair = row.get("prediction", "")
        original = row.get("prediction_before_contract_completion", "")
        answer_a, answer_b = (repair, original) if repair_is_a else (original, repair)
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
                                contract=json.dumps(contract_agent.get("contract", {}), ensure_ascii=False),
                                evidence=compact_evidence(contract_agent),
                                answer_a=answer_a,
                                answer_b=answer_b,
                            ),
                        },
                    ],
                )
                result = parse_json_object(response.choices[0].message.content or "")
                consensus["views"].append(normalize_view(result, repair_is_a))
            except Exception as exc:
                consensus["errors"].append(f"{type(exc).__name__}: {exc}"[:300])

    tasks = []
    for row in rows:
        tasks.append(judge(row, True))
        tasks.append(judge(row, False))
    await asyncio.gather(*tasks)


def decide(row: dict) -> None:
    consensus = row["post_repair_consensus_agent"]
    views = consensus.get("views") or []
    votes = Counter(view.get("winner", "tie") for view in views)
    repair_utilities = [utility(view["repaired"]) for view in views]
    original_utilities = [utility(view["original"]) for view in views]
    mean_repair = sum(repair_utilities) / len(repair_utilities) if repair_utilities else 0.0
    mean_original = sum(original_utilities) / len(original_utilities) if original_utilities else 0.0
    utility_gain = mean_repair - mean_original
    unsupported = agreed_findings(views, "unsupported_claims")
    missing = agreed_findings(views, "missing_required_facts")
    complete_views = len(views) == 2 and not consensus.get("errors")
    strong_consensus = votes["repair"] == 2 and utility_gain >= -0.02
    repair_and_tie = votes["repair"] == 1 and votes["tie"] == 1 and utility_gain >= 0.08
    accept = complete_views and not unsupported and not missing and (strong_consensus or repair_and_tie)

    consensus.update({
        "repair_votes": votes["repair"],
        "original_votes": votes["original"],
        "tie_votes": votes["tie"],
        "mean_repair_utility": round(mean_repair, 6),
        "mean_original_utility": round(mean_original, 6),
        "utility_gain": round(utility_gain, 6),
        "reported_repair_unsupported_claims": unsupported,
        "reported_repair_missing_facts": missing,
        "repair_accepted": accept,
        "final_action": "use_repair" if accept else "retain_original",
    })
    if not accept:
        repair = row.get("prediction", "")
        row["prediction_before_postrepair_consensus"] = repair
        row["prediction"] = row.get("prediction_before_contract_completion", "")
        row["method"] = "sage_v28_postrepair_consensus_original"
    else:
        row["method"] = "sage_v28_postrepair_consensus_repair"


def write_summary(path: Path, rows: list[dict], source_path: Path) -> None:
    agents = [row.get("post_repair_consensus_agent") or {} for row in rows]
    checked = [agent for agent in agents if agent.get("enabled")]
    accepted = sum(bool(agent.get("repair_accepted")) for agent in checked)
    retained = len(checked) - accepted
    lines = [
        "# SAGE v28 Post-Repair Consensus Summary\n\n",
        f"- Source: `{source_path.as_posix()}`\n",
        f"- Samples: {len(rows)}\n",
        "- Reference answers and evaluation scores are not used by either verification view.\n\n",
        "## Decisions\n\n",
        "| Metric | Value |\n",
        "|---|---:|\n",
        f"| Repairs checked | {len(checked)} |\n",
        f"| Verification calls | {sum(len(agent.get('views') or []) for agent in checked)} |\n",
        f"| Repairs retained | {accepted} |\n",
        f"| Repairs reverted to original | {retained} |\n\n",
        "## Per-Sample Consensus\n\n",
        "| Line | Repair votes | Original votes | Ties | Utility gain | Action |\n",
        "|---:|---:|---:|---:|---:|---|\n",
    ]
    for row in rows:
        agent = row.get("post_repair_consensus_agent") or {}
        if not agent.get("enabled"):
            continue
        lines.append(
            f"| {row.get('line_no')} | {agent.get('repair_votes', 0)} | "
            f"{agent.get('original_votes', 0)} | {agent.get('tie_votes', 0)} | "
            f"{agent.get('utility_gain', 0)} | {agent.get('final_action')} |\n"
        )
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--concurrency", type=int, default=3)
    args = parser.parse_args()

    load_dotenv(ROOT_DIR / ".env")
    input_path = ROOT_DIR / args.input
    rows = load_jsonl(input_path)
    candidates = []
    for row in rows:
        contract_agent = row.get("contract_completion_agent") or {}
        enabled = bool(contract_agent.get("repair_accepted") and row.get("prediction_before_contract_completion"))
        row["post_repair_consensus_agent"] = {
            "enabled": enabled,
            "views": [],
            "errors": [],
            "repair_accepted": False,
            "final_action": "not_checked",
        }
        if enabled:
            candidates.append(row)

    asyncio.run(collect_views(candidates, args.model or os.getenv("DEEPSEEK_MODEL"), args.concurrency))
    for row in candidates:
        decide(row)

    output_path = ROOT_DIR / args.output
    write_jsonl(rows, output_path)
    write_summary(ROOT_DIR / args.summary, rows, input_path)
    print(f"Saved SAGE v28 results to {output_path}")
    print(f"Saved summary to {ROOT_DIR / args.summary}")


if __name__ == "__main__":
    main()
