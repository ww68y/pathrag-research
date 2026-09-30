import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import APIConnectionError, APIError, APIStatusError, AsyncOpenAI, RateLimitError

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))


DIMENSIONS = [
    "comprehensiveness",
    "diversity",
    "logicality",
    "relevance",
    "coherence",
]


SYSTEM_PROMPT = """You are a strict judge for pairwise evaluation of retrieval-augmented question answering.
Compare two answers to the same question using the reference answer as guidance.
Return JSON only."""


USER_PROMPT = """Question:
{question}

Reference answer:
{reference}

Answer A:
{answer_a}

Answer B:
{answer_b}

Evaluate Answer A and Answer B on these dimensions:
- comprehensiveness: covers all key points required by the question and reference
- diversity: includes rich, non-redundant useful information
- logicality: reasoning is sound and relationships are explained correctly
- relevance: stays focused on the question
- coherence: answer is well structured and easy to follow

For each dimension, choose one winner:
- "A" if Answer A is better
- "B" if Answer B is better
- "tie" if they are equivalent

Return this exact JSON shape:
{{
  "comprehensiveness": "A",
  "diversity": "A",
  "logicality": "A",
  "relevance": "A",
  "coherence": "A",
  "short_reason": "one concise sentence"
}}"""


def answer_text(item: dict) -> str:
    return str(item.get("prediction") or item.get("answer") or "")


def is_evaluable_item(item: dict) -> bool:
    answer = answer_text(item).strip()
    return bool(answer) and "[no-context]" not in answer.lower() and not item.get("error")


def load_by_line(path: Path) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            rows[item["line_no"]] = item
    return {line_no: item for line_no, item in rows.items() if is_evaluable_item(item)}


def parse_json_object(text: str) -> dict:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    candidates = [cleaned]
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        candidates.append(match.group(0))

    for candidate in candidates:
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    raise ValueError(f"Could not parse judge JSON: {text}")


async def judge_pair(
    client: AsyncOpenAI,
    model: str,
    pathrag_item: dict,
    baseline_item: dict,
) -> dict:
    line_no = pathrag_item["line_no"]
    pathrag_answer = answer_text(pathrag_item)
    baseline_answer = answer_text(baseline_item)
    if pathrag_answer == baseline_answer:
        return {
            "line_no": line_no,
            "dataset": pathrag_item.get("dataset"),
            "question": pathrag_item.get("question"),
            "pathrag_position": "same",
            "winners": {dimension: "tie" for dimension in DIMENSIONS},
            "judge_raw": {"deterministic_identical_answer": True},
            "short_reason": "The two predictions are exactly identical.",
        }
    pathrag_is_a = line_no % 2 == 0
    answer_a = pathrag_answer if pathrag_is_a else baseline_answer
    answer_b = baseline_answer if pathrag_is_a else pathrag_answer
    reference = "\n".join(str(x) for x in pathrag_item.get("reference_answers", []))

    response = await client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_PROMPT.format(
                    question=pathrag_item.get("question", ""),
                    reference=reference,
                    answer_a=answer_a,
                    answer_b=answer_b,
                ),
            },
        ],
    )
    raw = response.choices[0].message.content or ""
    verdict = parse_json_object(raw)

    pathrag_wins = {}
    for dimension in DIMENSIONS:
        winner = str(verdict.get(dimension, "tie")).strip().lower()
        if winner == "a":
            pathrag_wins[dimension] = "pathrag" if pathrag_is_a else "baseline"
        elif winner == "b":
            pathrag_wins[dimension] = "baseline" if pathrag_is_a else "pathrag"
        else:
            pathrag_wins[dimension] = "tie"

    return {
        "line_no": line_no,
        "dataset": pathrag_item.get("dataset"),
        "question": pathrag_item.get("question"),
        "pathrag_position": "A" if pathrag_is_a else "B",
        "winners": pathrag_wins,
        "judge_raw": verdict,
        "short_reason": verdict.get("short_reason", ""),
    }


def summarize(results: list[dict], output_path: Path, pathrag_path: Path, baseline_path: Path) -> None:
    lines = [
        "# Pairwise Win-Rate Evaluation Summary\n\n",
        f"- PathRAG results: `{pathrag_path.as_posix()}`\n",
        f"- Baseline results: `{baseline_path.as_posix()}`\n",
        f"- Samples: {len(results)}\n",
        "- Win-rate convention: tie counts as 0.5 win\n",
        "- Dimensions follow the PathRAG paper scoring style\n\n",
        "## Win Rates\n\n",
        "| Dimension | PathRAG wins | Baseline wins | Ties | PathRAG win rate |\n",
        "|---|---:|---:|---:|---:|\n",
    ]

    for dimension in DIMENSIONS:
        pathrag_wins = sum(1 for r in results if r["winners"].get(dimension) == "pathrag")
        baseline_wins = sum(1 for r in results if r["winners"].get(dimension) == "baseline")
        ties = sum(1 for r in results if r["winners"].get(dimension) == "tie")
        win_rate = (pathrag_wins + 0.5 * ties) / len(results) if results else 0
        lines.append(
            f"| {dimension} | {pathrag_wins} | {baseline_wins} | {ties} | {round(win_rate * 100, 2)}% |\n"
        )

    values = []
    for dimension in DIMENSIONS:
        pathrag_wins = sum(1 for r in results if r["winners"].get(dimension) == "pathrag")
        ties = sum(1 for r in results if r["winners"].get(dimension) == "tie")
        values.append((pathrag_wins + 0.5 * ties) / len(results))
    avg_win_rate = sum(values) / len(values) if values else 0
    lines.append(f"\n- Average PathRAG win rate: {round(avg_win_rate * 100, 2)}%\n")

    lines.extend(
        [
            "\n## Per-Sample Winners\n\n",
            "| Line | Dataset | Comprehensiveness | Diversity | Logicality | Relevance | Coherence | Reason |\n",
            "|---:|---|---|---|---|---|---|---|\n",
        ]
    )
    for result in sorted(results, key=lambda x: x["line_no"]):
        winners = result["winners"]
        reason = str(result.get("short_reason", "")).replace("|", "/")
        lines.append(
            f"| {result['line_no']} | {result.get('dataset')} | "
            f"{winners.get('comprehensiveness')} | {winners.get('diversity')} | "
            f"{winners.get('logicality')} | {winners.get('relevance')} | "
            f"{winners.get('coherence')} | {reason} |\n"
        )

    output_path.write_text("".join(lines), encoding="utf-8")


def load_completed_by_line(path: Path) -> dict[int, dict]:
    rows: dict[int, dict] = {}
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as existing:
        for line in existing:
            if not line.strip():
                continue
            item = json.loads(line)
            line_no = item.get("line_no")
            if isinstance(line_no, int) and item.get("winners"):
                rows[line_no] = item
    return rows


def is_retryable_error(exc: Exception) -> bool:
    if isinstance(exc, APIStatusError) and exc.status_code == 402:
        return False
    return isinstance(exc, (APIConnectionError, APIError, RateLimitError, ValueError))


async def judge_pair_with_retries(
    client: AsyncOpenAI,
    model: str,
    pathrag_item: dict,
    baseline_item: dict,
    max_retries: int,
    retry_delay_seconds: float,
) -> dict:
    line_no = pathrag_item["line_no"]
    for attempt in range(max_retries + 1):
        try:
            return await judge_pair(client, model, pathrag_item, baseline_item)
        except Exception as exc:
            if attempt >= max_retries or not is_retryable_error(exc):
                raise
            wait = retry_delay_seconds * (2**attempt)
            print(
                f"retrying line={line_no} after {type(exc).__name__}: wait={wait:.1f}s",
                flush=True,
            )
            await asyncio.sleep(wait)
    raise RuntimeError(f"unreachable retry state for line={line_no}")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pathrag", default="results/historical_datasets/ultradomain_mix_repro_current.jsonl")
    parser.add_argument("--baseline", default="results/historical_baselines/direct_context_baseline_current.jsonl")
    parser.add_argument("--output", default="results/historical_pairwise/pairwise_pathrag_vs_direct_context.jsonl")
    parser.add_argument("--summary", default="results/historical_pairwise/pairwise_pathrag_vs_direct_context_summary.md")
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--line-nos",
        default=None,
        help="Optional comma-separated line_no values to compare",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--retry-delay-seconds", type=float, default=5.0)
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()

    pathrag_path = ROOT_DIR / args.pathrag
    baseline_path = ROOT_DIR / args.baseline
    output_path = ROOT_DIR / args.output
    summary_path = ROOT_DIR / args.summary

    pathrag_by_line = load_by_line(pathrag_path)
    baseline_by_line = load_by_line(baseline_path)
    common_lines = sorted(set(pathrag_by_line) & set(baseline_by_line))
    if args.line_nos:
        selected_lines = {int(value.strip()) for value in args.line_nos.split(",") if value.strip()}
        common_lines = [line_no for line_no in common_lines if line_no in selected_lines]

    completed_by_line = load_completed_by_line(output_path) if args.resume else {}

    if args.summarize_only:
        summarize(list(completed_by_line.values()), summary_path, pathrag_path, baseline_path)
        print(f"Saved summary to {summary_path}")
        return

    load_dotenv(ROOT_DIR / ".env")
    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_API_BASE")
    model = args.model or os.getenv("EVAL_MODEL") or os.getenv("DEEPSEEK_MODEL")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is missing in .env")
    if not base_url:
        raise RuntimeError("DEEPSEEK_API_BASE is missing in .env")
    if not model:
        raise RuntimeError("DEEPSEEK_MODEL or EVAL_MODEL is missing in .env")

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    mode = "a" if args.resume else "w"
    try:
        with output_path.open(mode, encoding="utf-8") as out:
            pending = [line for line in common_lines if line not in completed_by_line]
            for i, line_no in enumerate(pending, start=1):
                print(f"[{i}/{len(pending)}] judging line={line_no}", flush=True)
                result = await judge_pair_with_retries(
                    client,
                    model,
                    pathrag_by_line[line_no],
                    baseline_by_line[line_no],
                    args.max_retries,
                    args.retry_delay_seconds,
                )
                completed_by_line[line_no] = result
                out.write(json.dumps(result, ensure_ascii=False) + "\n")
                out.flush()
    except Exception:
        summarize(list(completed_by_line.values()), summary_path, pathrag_path, baseline_path)
        print(f"Saved partial summary to {summary_path}", flush=True)
        raise

    summarize(list(completed_by_line.values()), summary_path, pathrag_path, baseline_path)
    print(f"Saved pairwise results to {output_path}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    asyncio.run(main())
