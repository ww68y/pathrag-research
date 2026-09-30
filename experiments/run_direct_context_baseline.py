import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

SYSTEM_PROMPT = """You are a question answering assistant.
Answer the question using only the provided context.
If the context does not contain enough evidence, say that the answer cannot be determined from the context.
Keep the answer focused and avoid unnecessary details."""


USER_PROMPT = """Context:
{context}

Question:
{question}

Answer:"""


def parse_line_nos(value: str) -> set[int]:
    line_nos: set[int] = set()
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            start, end = part.split("-", 1)
            line_nos.update(range(int(start), int(end) + 1))
        else:
            line_nos.add(int(part))
    return line_nos


def load_samples(path: Path, line_nos: set[int]) -> list[dict]:
    samples = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            if line_no not in line_nos:
                continue
            item = json.loads(line)
            item["_line_no"] = line_no
            samples.append(item)
    return sorted(samples, key=lambda x: x["_line_no"])


async def answer_sample(client: AsyncOpenAI, model: str, sample: dict) -> dict:
    started = time.perf_counter()
    response = await client.chat.completions.create(
        model=model,
        temperature=0,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": USER_PROMPT.format(
                    context=sample.get("context", ""),
                    question=sample.get("input", ""),
                ),
            },
        ],
    )
    query_seconds = time.perf_counter() - started
    return {
        "id": sample.get("_id") or f"line-{sample['_line_no']}",
        "line_no": sample["_line_no"],
        "dataset": sample.get("dataset"),
        "label": sample.get("label"),
        "question": sample.get("input"),
        "reference_answers": sample.get("answers", []),
        "prediction": response.choices[0].message.content or "",
        "context_chars": len(sample.get("context", "")),
        "graph_nodes": 0,
        "graph_edges": 0,
        "insert_seconds": 0,
        "query_seconds": round(query_seconds, 3),
        "baseline": "direct_context",
    }


async def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="datasets/UltraDomain/mix.jsonl")
    parser.add_argument("--line-nos", required=True, help="Comma-separated zero-based line numbers to run")
    parser.add_argument("--output", default="results/historical_baselines/direct_context_baseline_current.jsonl")
    parser.add_argument("--model", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    load_dotenv(ROOT_DIR / ".env")
    api_key = os.getenv("DEEPSEEK_API_KEY")
    base_url = os.getenv("DEEPSEEK_API_BASE")
    model = args.model or os.getenv("DEEPSEEK_MODEL")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is missing in .env")
    if not base_url:
        raise RuntimeError("DEEPSEEK_API_BASE is missing in .env")
    if not model:
        raise RuntimeError("DEEPSEEK_MODEL is missing in .env")

    line_nos = parse_line_nos(args.line_nos)
    samples = load_samples(ROOT_DIR / args.dataset, line_nos or set())
    output_path = ROOT_DIR / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    existing_lines: set[int] = set()
    if args.resume and output_path.exists():
        with output_path.open("r", encoding="utf-8") as existing:
            for line in existing:
                if not line.strip():
                    continue
                try:
                    existing_lines.add(int(json.loads(line).get("line_no")))
                except (TypeError, ValueError, json.JSONDecodeError):
                    continue
        samples = [sample for sample in samples if sample["_line_no"] not in existing_lines]

    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    print(f"Selected {len(samples)} sample(s) for this run; existing={len(existing_lines)}")
    mode = "a" if args.resume and output_path.exists() else "w"
    with output_path.open(mode, encoding="utf-8") as out:
        for i, sample in enumerate(samples, start=1):
            print(f"[{i}/{len(samples)}] line={sample['_line_no']} {sample.get('input')}", flush=True)
            result = await answer_sample(client, model, sample)
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            out.flush()
            print(f"Done: query={result['query_seconds']}s", flush=True)

    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
