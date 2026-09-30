import argparse
import asyncio
import hashlib
import json
import os
import shutil
import sys
import time
from functools import partial
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from PathRAG import PathRAG, QueryParam
from PathRAG.llm import openai_complete, siliconcloud_embedding
from PathRAG.prompt import PROMPTS
from PathRAG.utils import EmbeddingFunc


def parse_line_nos(value: str | None) -> set[int] | None:
    if not value:
        return None
    return {int(item.strip()) for item in value.split(",") if item.strip()}


def load_samples(
    path: Path,
    limit: int | None,
    max_context_chars: int | None,
    line_nos: set[int] | None,
) -> list[dict]:
    samples = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            if line_nos is not None and line_no not in line_nos:
                continue
            item = json.loads(line)
            item["_line_no"] = line_no
            if max_context_chars and len(item.get("context", "")) > max_context_chars:
                continue
            samples.append(item)

    samples.sort(key=lambda x: len(x.get("context", "")))
    return samples[:limit] if limit else samples


def load_completed_line_nos(path: Path) -> set[int]:
    if not path.exists():
        return set()
    completed = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "line_no" in item:
                completed.add(int(item["line_no"]))
    return completed


def context_hash(context: str) -> str:
    return hashlib.sha1(context.encode("utf-8")).hexdigest()[:12]


async def durable_openai_complete(
    prompt,
    system_prompt=None,
    history_messages=[],
    keyword_extraction=False,
    *,
    cache_dir: Path,
    **kwargs,
):
    hashing_kv = kwargs.get("hashing_kv")
    model_name = None
    if hashing_kv is not None and hasattr(hashing_kv, "global_config"):
        model_name = hashing_kv.global_config.get("llm_model_name")

    serializable_kwargs = {
        key: repr(value)
        for key, value in kwargs.items()
        if key not in {"api_key", "hashing_kv"}
    }
    request_data = {
        "cache_version": 1,
        "model": model_name,
        "prompt": prompt,
        "system_prompt": system_prompt,
        "history_messages": history_messages,
        "keyword_extraction": keyword_extraction,
        "kwargs": serializable_kwargs,
    }
    request_json = json.dumps(request_data, ensure_ascii=False, sort_keys=True)
    request_hash = hashlib.sha256(request_json.encode("utf-8")).hexdigest()
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = cache_dir / f"{request_hash}.json"

    if cache_path.exists():
        try:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if isinstance(cached.get("content"), str):
                print(f"Durable LLM cache hit: {request_hash[:12]}", flush=True)
                return cached["content"]
        except (OSError, json.JSONDecodeError):
            pass

    content = await openai_complete(
        prompt,
        system_prompt=system_prompt,
        history_messages=history_messages,
        keyword_extraction=keyword_extraction,
        **kwargs,
    )
    if isinstance(content, str):
        payload = json.dumps(
            {"request": request_data, "content": content},
            ensure_ascii=False,
            indent=2,
        )
        task_id = id(asyncio.current_task())
        temp_path = cache_dir / f".{request_hash}.{os.getpid()}.{task_id}.tmp"
        temp_path.write_text(payload, encoding="utf-8")
        os.replace(temp_path, cache_path)
    return content


def build_rag(working_dir: Path, args: argparse.Namespace) -> PathRAG:
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key or "replace_with" in api_key:
        raise RuntimeError("Please set DEEPSEEK_API_KEY in .env before running.")

    return PathRAG(
        working_dir=str(working_dir),
        llm_model_func=partial(
            durable_openai_complete,
            cache_dir=working_dir / "durable_llm_cache",
        ),
        llm_model_name=os.getenv("DEEPSEEK_MODEL"),
        llm_model_kwargs={
            "api_key": api_key,
            "base_url": os.getenv("DEEPSEEK_API_BASE"),
        },
        embedding_func=EmbeddingFunc(
            embedding_dim=int(os.getenv("EMBEDDING_DIM", "1024")),
            max_token_size=int(os.getenv("EMBEDDING_MAX_TOKENS", "8192")),
            func=partial(
                siliconcloud_embedding,
                model=os.getenv("EMBEDDING_MODEL"),
                base_url=os.getenv("EMBEDDING_API_BASE"),
                api_key=api_key,
                max_token_size=int(os.getenv("EMBEDDING_MAX_TOKENS", "8192")),
            ),
        ),
        embedding_func_max_async=args.embedding_async,
        llm_model_max_async=args.llm_async,
        chunk_token_size=args.chunk_tokens,
        entity_extract_max_gleaning=args.entity_gleaning,
    )


async def index_context(context: str, working_dir: Path, args: argparse.Namespace) -> tuple[PathRAG, float]:
    rag = build_rag(working_dir, args)

    started = time.perf_counter()
    await rag.ainsert(context)
    insert_seconds = time.perf_counter() - started
    return rag, insert_seconds


async def query_sample(
    rag: PathRAG,
    sample: dict,
    working_dir: Path,
    insert_seconds: float,
    args: argparse.Namespace,
) -> dict:
    sample_id = sample.get("_id") or f"line-{sample['_line_no']}"
    graph = rag.chunk_entity_relation_graph._graph
    query_started = time.perf_counter()
    answer = await rag.aquery(
        sample["input"],
        param=QueryParam(mode="hybrid", top_k=args.top_k),
    )
    query_seconds = time.perf_counter() - query_started

    return {
        "id": sample_id,
        "line_no": sample["_line_no"],
        "dataset": sample.get("dataset"),
        "label": sample.get("label"),
        "question": sample.get("input"),
        "reference_answers": sample.get("answers", []),
        "prediction": answer,
        "context_chars": len(sample.get("context", "")),
        "graph_nodes": graph.number_of_nodes(),
        "graph_edges": graph.number_of_edges(),
        "insert_seconds": round(insert_seconds, 3),
        "query_seconds": round(query_seconds, 3),
        "working_dir": str(working_dir.relative_to(ROOT_DIR)),
        "context_hash": context_hash(sample.get("context", "")),
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="datasets/UltraDomain/mix.jsonl")
    parser.add_argument("--limit", type=int, default=1)
    parser.add_argument("--line-nos", default=None, help="Comma-separated zero-based line numbers to run, e.g. 114,115")
    parser.add_argument("--max-context-chars", type=int, default=12000)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--chunk-tokens", type=int, default=1200)
    parser.add_argument("--entity-gleaning", type=int, default=0)
    parser.add_argument("--llm-async", type=int, default=1)
    parser.add_argument("--embedding-async", type=int, default=1)
    parser.add_argument("--output", default="results/historical_datasets/ultradomain_mix_subset.jsonl")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--skip-existing-output", action="store_true")
    args = parser.parse_args()

    load_dotenv(ROOT_DIR / ".env")
    PROMPTS["process_tickers"] = ["-", "\\", "|", "/"]

    dataset_path = ROOT_DIR / args.dataset
    line_nos = parse_line_nos(args.line_nos)
    samples = load_samples(dataset_path, args.limit if line_nos is None else None, args.max_context_chars, line_nos)
    if not samples:
        raise RuntimeError("No samples selected. Increase --max-context-chars.")

    output_path = ROOT_DIR / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if args.skip_existing_output:
        completed_line_nos = load_completed_line_nos(output_path)
        if completed_line_nos:
            samples = [sample for sample in samples if sample["_line_no"] not in completed_line_nos]
            print(f"Skipping {len(completed_line_nos)} existing output sample(s)", flush=True)
        if not samples:
            print("All selected samples already exist in output.")
            return

    print(f"Selected {len(samples)} sample(s) from {dataset_path}")
    context_groups: dict[str, list[dict]] = {}
    for sample in samples:
        context_groups.setdefault(context_hash(sample.get("context", "")), []).append(sample)

    print(f"Grouped into {len(context_groups)} unique context(s)", flush=True)
    output_mode = "a" if args.append else "w"
    with output_path.open(output_mode, encoding="utf-8") as out:
        completed = 0
        for group_i, (ctx_hash, group_samples) in enumerate(context_groups.items(), start=1):
            first_sample = group_samples[0]
            working_dir = ROOT_DIR / "data_repro" / f"ctx_{first_sample['_line_no']}_{ctx_hash}"
            if args.reset and working_dir.exists():
                shutil.rmtree(working_dir)
            working_dir.mkdir(parents=True, exist_ok=True)

            print(
                f"\n[Context {group_i}/{len(context_groups)}] "
                f"hash={ctx_hash} lines={[s['_line_no'] for s in group_samples]} "
                f"context_chars={len(first_sample.get('context', ''))}",
                flush=True,
            )

            rag, insert_seconds = await index_context(first_sample["context"], working_dir, args)
            print(
                f"Indexed: nodes={rag.chunk_entity_relation_graph._graph.number_of_nodes()} "
                f"edges={rag.chunk_entity_relation_graph._graph.number_of_edges()} "
                f"insert={round(insert_seconds, 3)}s",
                flush=True,
            )

            for sample in group_samples:
                completed += 1
                print(
                    f"[{completed}/{len(samples)}] line={sample['_line_no']} "
                    f"Question: {sample.get('input')}",
                    flush=True,
                )
                result = await query_sample(rag, sample, working_dir, insert_seconds, args)
                out.write(json.dumps(result, ensure_ascii=False) + "\n")
                out.flush()
                print(
                    f"Done: nodes={result['graph_nodes']} edges={result['graph_edges']} "
                    f"insert={result['insert_seconds']}s query={result['query_seconds']}s",
                    flush=True,
                )

    print(f"\nSaved results to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
