import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from openai import AsyncOpenAI


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from api.features.chats.answer_realization import rank_source_excerpts, realize_answer


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="results/historical_vsr/vsr_pathrag_v6_cached_scale_30.jsonl")
    parser.add_argument("--dataset", default="datasets/UltraDomain/vsr_v6_cached_scale_30.jsonl")
    parser.add_argument("--output", default="results/historical_vsr/vsr_pathrag_v7_answer_realization_scale_30.jsonl")
    parser.add_argument("--version", default="vsr-pathrag-v7-answer-realization")
    parser.add_argument("--concurrency", type=int, default=3)
    parser.add_argument(
        "--line-nos",
        help="Optional comma-separated line_no values to generate",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-failures", action="store_true")
    args = parser.parse_args()

    load_dotenv(ROOT_DIR / ".env")
    client = AsyncOpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url=os.getenv("DEEPSEEK_API_BASE"),
        timeout=float(os.getenv("VSR_REALIZATION_TIMEOUT_SECONDS", "120")),
        max_retries=1,
    )
    model = os.getenv("DEEPSEEK_MODEL")
    if not model:
        raise RuntimeError("DEEPSEEK_MODEL is missing")

    def load_jsonl(path: Path) -> list[dict[str, Any]]:
        with path.open("r", encoding="utf-8") as source:
            return [json.loads(line) for line in source if line.strip()]

    rows = load_jsonl(ROOT_DIR / args.input)
    if args.line_nos:
        selected_line_nos = {
            int(value.strip())
            for value in args.line_nos.split(",")
            if value.strip()
        }
        rows = [
            row
            for row in rows
            if int(row.get("line_no", -1)) in selected_line_nos
        ]
    samples = load_jsonl(ROOT_DIR / args.dataset)
    samples_by_line = {index: sample for index, sample in enumerate(samples)}
    output_path = ROOT_DIR / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = {}
    if args.resume and output_path.exists():
        existing_rows = load_jsonl(output_path)
        if args.retry_failures:
            existing_rows = [
                row
                for row in existing_rows
                if (row.get("answer_realization") or {}).get("stop_reason")
                != "realization_failed_closed"
            ]
            with output_path.open("w", encoding="utf-8") as output:
                for row in existing_rows:
                    output.write(json.dumps(row, ensure_ascii=False) + "\n")
        for row in existing_rows:
            completed[int(row["line_no"])] = row

    semaphore = asyncio.Semaphore(max(1, args.concurrency))

    async def llm_func(prompt: str, system_prompt: str | None = None) -> str:
        async with semaphore:
            response = await client.chat.completions.create(
                model=model,
                temperature=0,
                messages=[
                    {"role": "system", "content": system_prompt or ""},
                    {"role": "user", "content": prompt},
                ],
            )
        return response.choices[0].message.content or ""

    async def process(row: dict[str, Any]) -> dict[str, Any]:
        line_no = int(row["line_no"])
        sample = samples_by_line[line_no]
        evidence_summary = (row.get("trace") or {}).get("evidence_summary") or {}
        excerpts = rank_source_excerpts(row.get("question", ""), sample.get("context", ""))
        try:
            answer, realization = await realize_answer(
                llm_func,
                row.get("question", ""),
                row.get("answer", ""),
                evidence_summary,
                excerpts,
            )
            return {
                **row,
                "version": args.version,
                "original_answer": row.get("answer", ""),
                "answer": answer,
                "answer_realization": realization,
            }
        except Exception as exc:
            return {
                **row,
                "version": args.version,
                "original_answer": row.get("answer", ""),
                "answer_realization": {
                    "enabled": True,
                    "triggered": True,
                    "committed": False,
                    "stop_reason": "realization_failed_closed",
                    "error": f"{type(exc).__name__}: {exc}",
                },
            }

    pending = [row for row in rows if int(row["line_no"]) not in completed]
    batch_size = max(1, args.concurrency)
    mode = "a" if args.resume else "w"
    with output_path.open(mode, encoding="utf-8") as output:
        for start in range(0, len(pending), batch_size):
            batch = pending[start : start + batch_size]
            results = await asyncio.gather(*(process(row) for row in batch))
            for result in results:
                output.write(json.dumps(result, ensure_ascii=False) + "\n")
                output.flush()
                realization = result.get("answer_realization") or {}
                print(
                    f"line={result['line_no']} committed={realization.get('committed')} "
                    f"reason={realization.get('stop_reason')}",
                    flush=True,
                )
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
