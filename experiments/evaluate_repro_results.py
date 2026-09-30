from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))


SYSTEM_PROMPT = """You are a strict evaluator for retrieval-augmented question answering.
Compare the prediction with the reference answer for the same question.
Judge semantic correctness, not exact wording.
Return JSON only."""


USER_PROMPT = """Question:
{question}

Reference answer:
{reference}

Prediction:
{prediction}

Score the prediction from 1 to 5:
- correctness: factual correctness with respect to the reference and question
- coverage: whether key reference points are covered
- relevance: whether the answer stays on topic
- conciseness: whether the answer avoids unnecessary verbosity
- overall: overall answer quality

Also provide:
- verdict: one of ["excellent", "good", "partial", "poor"]
- short_reason: one concise sentence

Return this exact JSON shape:
{{
  "correctness": 1,
  "coverage": 1,
  "relevance": 1,
  "conciseness": 1,
  "overall": 1,
  "verdict": "partial",
  "short_reason": "..."
}}"""


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
    raise ValueError(f"Could not parse JSON from evaluator response: {text}")


def inferred_line_no(item: dict, fallback: int) -> int:
    if item.get("line_no") is not None:
        return int(item["line_no"])
    question_id = str(item.get("question_id") or "")
    if question_id.startswith("ultra_"):
        return int(question_id.split("_", 1)[1])
    return fallback


def load_items(path: Path) -> list[dict]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as f:
            rows = [json.loads(line) for line in f if line.strip()]
    else:
        payload = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        if isinstance(payload, dict) and isinstance(payload.get("results"), list):
            rows = payload["results"]
        elif isinstance(payload, list):
            rows = payload
        else:
            rows = []
    normalized = []
    for index, row in enumerate(rows):
        item = dict(row)
        item["line_no"] = inferred_line_no(item, index)
        normalized.append(item)
    return normalized


def reference_text(item: dict) -> str:
    reference_answers = item.get("reference_answers") or item.get("answers") or []
    if isinstance(reference_answers, list):
        return "\n".join(str(answer) for answer in reference_answers)
    if reference_answers:
        return str(reference_answers)
    return str(item.get("gold_answer") or "")


def prediction_text(item: dict) -> str:
    return str(item.get("prediction") or item.get("answer") or item.get("final_answer") or "")


def is_evaluable_item(item: dict) -> bool:
    prediction = prediction_text(item).strip()
    return bool(prediction) and "[no-context]" not in prediction.lower() and not item.get("error")


async def judge_item(
    client: AsyncOpenAI,
    model: str,
    item: dict,
    max_retries: int,
    retry_delay_seconds: float,
) -> dict:
    prompt = USER_PROMPT.format(
        question=item.get("question", ""),
        reference=reference_text(item),
        prediction=prediction_text(item),
    )
    last_error: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            response = await client.chat.completions.create(
                model=model,
                temperature=0,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            scores = parse_json_object(response.choices[0].message.content or "")
            break
        except Exception as exc:  # noqa: BLE001 - keep long judge runs resumable across transient API failures.
            last_error = exc
            if attempt >= max_retries:
                raise RuntimeError(
                    f"Judge failed for line={item.get('line_no')} after {max_retries + 1} attempts: {exc}"
                ) from exc
            wait_seconds = retry_delay_seconds * (2 ** attempt)
            print(
                f"retrying line={item.get('line_no')} after {type(exc).__name__}: wait={wait_seconds:.1f}s",
                flush=True,
            )
            await asyncio.sleep(wait_seconds)
    else:
        raise RuntimeError(f"Judge failed for line={item.get('line_no')}: {last_error}")
    return {
        "line_no": item.get("line_no"),
        "dataset": item.get("dataset"),
        "question": item.get("question"),
        "scores": scores,
    }


def write_summary(evaluations: list[dict], output_path: Path, source_path: Path) -> None:
    score_keys = ["correctness", "coverage", "relevance", "conciseness", "overall"]
    averages = {}
    for key in score_keys:
        values = [float(e["scores"].get(key, 0)) for e in evaluations]
        averages[key] = round(sum(values) / len(values), 3) if values else 0

    verdict_counts: dict[str, int] = {}
    for item in evaluations:
        verdict = item["scores"].get("verdict", "unknown")
        verdict_counts[verdict] = verdict_counts.get(verdict, 0) + 1

    lines = [
        "# Reproduction Evaluation Summary\n\n",
        f"- Source results: `{source_path.as_posix()}`\n",
        f"- Evaluated samples: {len(evaluations)}\n",
        "- Judge: DeepSeek-compatible OpenAI API endpoint configured in `.env`\n",
        "- Scale: 1-5, higher is better\n\n",
        "## Average Scores\n\n",
        "| Metric | Score |\n",
        "|---|---:|\n",
    ]
    for key in score_keys:
        lines.append(f"| {key} | {averages[key]} |\n")

    lines.extend(["\n## Verdict Counts\n\n", "| Verdict | Count |\n", "|---|---:|\n"])
    for verdict, count in sorted(verdict_counts.items()):
        lines.append(f"| {verdict} | {count} |\n")

    lines.extend(
        [
            "\n## Per-Sample Scores\n\n",
            "| Line | Dataset | Correctness | Coverage | Relevance | Conciseness | Overall | Verdict | Reason |\n",
            "|---:|---|---:|---:|---:|---:|---:|---|---|\n",
        ]
    )
    for item in sorted(evaluations, key=lambda x: x["line_no"]):
        scores = item["scores"]
        reason = str(scores.get("short_reason", "")).replace("|", "/")
        lines.append(
            f"| {item['line_no']} | {item['dataset']} | "
            f"{scores.get('correctness')} | {scores.get('coverage')} | "
            f"{scores.get('relevance')} | {scores.get('conciseness')} | "
            f"{scores.get('overall')} | {scores.get('verdict')} | {reason} |\n"
        )

    output_path.write_text("".join(lines), encoding="utf-8")


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/historical_datasets/ultradomain_mix_repro_current.jsonl")
    parser.add_argument("--output", default="results/historical_datasets/ultradomain_mix_eval.jsonl")
    parser.add_argument("--summary", default="results/historical_datasets/ultradomain_mix_eval_summary.md")
    parser.add_argument("--model", default=None)
    parser.add_argument(
        "--line-nos",
        default=None,
        help="Optional comma-separated line_no values to evaluate",
    )
    parser.add_argument("--resume", action="store_true", help="Skip line_no values already present in the output file")
    parser.add_argument("--concurrency", type=int, default=1, help="Number of judge calls per batch")
    parser.add_argument("--max-retries", type=int, default=3, help="Retries per judge item after transient failures")
    parser.add_argument("--retry-delay-seconds", type=float, default=5.0, help="Base retry backoff in seconds")
    args = parser.parse_args()

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

    input_path = ROOT_DIR / args.input
    output_path = ROOT_DIR / args.output
    summary_path = ROOT_DIR / args.summary
    output_path.parent.mkdir(parents=True, exist_ok=True)

    raw_items = load_items(input_path)
    latest_items = {}
    for item in raw_items:
        latest_items[item.get("line_no")] = item
    items = [item for _, item in sorted(latest_items.items()) if is_evaluable_item(item)]
    if args.line_nos:
        selected_line_nos = {int(value.strip()) for value in args.line_nos.split(",") if value.strip()}
        items = [item for item in items if item.get("line_no") in selected_line_nos]
    client = AsyncOpenAI(
        api_key=api_key,
        base_url=base_url,
        timeout=float(os.getenv("VSR_EVAL_TIMEOUT_SECONDS", "120")),
        max_retries=1,
    )

    evaluations = []
    completed_line_nos = set()
    if args.resume and output_path.exists():
        with output_path.open("r", encoding="utf-8") as existing:
            for line in existing:
                if not line.strip():
                    continue
                evaluation = json.loads(line)
                evaluations.append(evaluation)
                completed_line_nos.add(evaluation.get("line_no"))

    mode = "a" if args.resume else "w"
    with output_path.open(mode, encoding="utf-8") as out:
        pending_items = [item for item in items if item.get("line_no") not in completed_line_nos]
        batch_size = max(1, args.concurrency)
        for start in range(0, len(pending_items), batch_size):
            batch = pending_items[start:start + batch_size]
            for offset, item in enumerate(batch, start=start + 1):
                print(f"[{offset}/{len(items)}] judging line={item.get('line_no')}", flush=True)
            batch_results = await asyncio.gather(
                *(
                    judge_item(
                        client,
                        model,
                        item,
                        max(0, args.max_retries),
                        max(0.1, args.retry_delay_seconds),
                    )
                    for item in batch
                ),
                return_exceptions=True,
            )
            failures = []
            for evaluation in batch_results:
                if isinstance(evaluation, Exception):
                    failures.append(evaluation)
                    continue
                evaluations.append(evaluation)
                out.write(json.dumps(evaluation, ensure_ascii=False) + "\n")
            out.flush()
            if failures:
                write_summary(evaluations, summary_path, input_path)
                details = "; ".join(str(failure) for failure in failures)
                raise RuntimeError(f"Judge batch failed after writing successful rows: {details}")

    write_summary(evaluations, summary_path, input_path)
    print(f"Saved evaluations to {output_path}")
    print(f"Saved summary to {summary_path}")


if __name__ == "__main__":
    asyncio.run(main())
