import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from functools import partial
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))


def parse_line_nos(value: str | None, total: int) -> set[int]:
    if not value:
        return set(range(total))
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
    return {i for i in line_nos if 0 <= i < total}


def load_samples(path: Path) -> list[dict]:
    samples = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            if not line.strip():
                continue
            item = json.loads(line)
            item["_line_no"] = line_no
            samples.append(item)
    return samples


def context_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def is_valid_completed_row(item: dict) -> bool:
    prediction = str(item.get("prediction") or "").strip()
    return bool(prediction) and "[no-context]" not in prediction.lower() and not item.get("error")


def read_existing_lines(path: Path, retry_errors: bool = False) -> set[int]:
    latest_by_line: dict[int, dict] = {}
    if not path.exists():
        return set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "line_no" not in item:
                continue
            latest_by_line[int(item["line_no"])] = item

    done: set[int] = set()
    for line_no, item in latest_by_line.items():
        if retry_errors and not is_valid_completed_row(item):
            continue
        done.add(line_no)
    return done


def is_index_ready(working_dir: Path) -> bool:
    required = [
        "graph_chunk_entity_relation.graphml",
        "kv_store_full_docs.json",
        "kv_store_text_chunks.json",
        "vdb_chunks.json",
    ]
    if not all((working_dir / name).exists() for name in required):
        return False

    graph_path = working_dir / "graph_chunk_entity_relation.graphml"
    try:
        graph_text = graph_path.read_text(encoding="utf-8", errors="ignore")
    except OSError:
        return False
    if "<node" not in graph_text:
        return False

    entity_vdb_path = working_dir / "vdb_entities.json"
    try:
        entity_vdb = json.loads(entity_vdb_path.read_text(encoding="utf-8", errors="replace"))
    except (OSError, json.JSONDecodeError):
        return False
    if not entity_vdb.get("data"):
        return False

    return True


def normalize_openai_base_url(value: str | None) -> str:
    if not value:
        return ""
    value = value.rstrip("/")
    if value.endswith("/embeddings"):
        return value[: -len("/embeddings")]
    return value


async def build_lightrag(args):
    try:
        from lightrag import LightRAG
        from lightrag.base import EmbeddingFunc
        from lightrag.llm.openai import openai_complete_if_cache, openai_embed
    except ImportError as exc:
        raise RuntimeError(
            "LightRAG is not installed. Install it with: "
            "D:\\Anaconda\\envs\\lw\\python.exe -m pip install lightrag-hku"
        ) from exc

    api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    llm_base_url = normalize_openai_base_url(
        os.getenv("DEEPSEEK_API_BASE") or os.getenv("OPENAI_API_BASE")
    )
    llm_model = os.getenv("DEEPSEEK_MODEL", "deepseek-ai/DeepSeek-V4-Flash")

    embedding_model = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    embedding_base_url = normalize_openai_base_url(
        os.getenv("EMBEDDING_API_BASE") or llm_base_url
    )
    embedding_dim = int(os.getenv("EMBEDDING_DIM", "1024"))
    embedding_max_tokens = int(os.getenv("EMBEDDING_MAX_TOKENS", "8192"))

    if not api_key:
        raise RuntimeError("Missing API key. Set DEEPSEEK_API_KEY in .env.")
    if not llm_base_url:
        raise RuntimeError("Missing API base URL. Set DEEPSEEK_API_BASE in .env.")
    if not embedding_base_url:
        embedding_base_url = llm_base_url

    async def llm_func(prompt, system_prompt=None, history_messages=None, **kwargs):
        return await openai_complete_if_cache(
            llm_model,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages or [],
            api_key=api_key,
            base_url=llm_base_url,
            **kwargs,
        )

    embedding_func = EmbeddingFunc(
        embedding_dim=embedding_dim,
        max_token_size=embedding_max_tokens,
        model_name=embedding_model,
        func=partial(
            openai_embed.func,
            model=embedding_model,
            api_key=api_key,
            base_url=embedding_base_url,
        ),
    )

    return LightRAG(
        working_dir=str(args.working_dir),
        embedding_func=embedding_func,
        llm_model_func=llm_func,
        llm_model_name=llm_model,
        chunk_token_size=args.chunk_token_size,
        chunk_overlap_token_size=args.chunk_overlap_token_size,
        llm_model_max_async=args.llm_max_async,
        embedding_func_max_async=args.embedding_max_async,
        entity_extract_max_gleaning=args.entity_extract_max_gleaning,
        entity_extract_max_entities=args.entity_extract_max_entities,
        top_k=args.top_k,
        chunk_top_k=args.chunk_top_k,
        default_llm_timeout=args.llm_timeout,
        default_embedding_timeout=args.embedding_timeout,
    )


async def run_sample(sample: dict, args) -> dict:
    from lightrag import QueryParam

    context = sample.get("context", "")
    question = sample.get("input", "")
    ctx_hash = context_hash(context)
    context_dir = args.cache_root / f"ctx_{ctx_hash}"
    context_dir.mkdir(parents=True, exist_ok=True)

    local_args = argparse.Namespace(**vars(args))
    local_args.working_dir = context_dir

    rag = await build_lightrag(local_args)
    started = time.perf_counter()
    await rag.initialize_storages()

    try:
        index_seconds = 0.0
        if not is_index_ready(context_dir):
            index_started = time.perf_counter()
            await rag.ainsert(context, ids=ctx_hash)
            index_seconds = time.perf_counter() - index_started

        query_started = time.perf_counter()
        answer = await rag.aquery(
            question,
            QueryParam(
                mode=args.query_mode,
                top_k=args.top_k,
                chunk_top_k=args.chunk_top_k,
                response_type=args.response_type,
                enable_rerank=False,
            ),
        )
        if answer is None or str(answer).strip().lower() == "none":
            raise RuntimeError("LightRAG returned an empty/None answer")
        if "[no-context]" in str(answer).lower():
            raise RuntimeError("LightRAG returned a no-context answer")
        query_seconds = time.perf_counter() - query_started

        return {
            "line_no": sample["_line_no"],
            "id": sample.get("_id"),
            "dataset": sample.get("dataset"),
            "question": question,
            "reference_answers": sample.get("answers", []),
            "prediction": str(answer),
            "method": "lightrag",
            "query_mode": args.query_mode,
            "context_hash": ctx_hash,
            "context_chars": len(context),
            "index_seconds": index_seconds,
            "query_seconds": query_seconds,
            "total_seconds": time.perf_counter() - started,
            "chunk_token_size": args.chunk_token_size,
            "chunk_overlap_token_size": args.chunk_overlap_token_size,
            "entity_extract_max_gleaning": args.entity_extract_max_gleaning,
            "entity_extract_max_entities": args.entity_extract_max_entities,
            "top_k": args.top_k,
            "chunk_top_k": args.chunk_top_k,
        }
    finally:
        try:
            await rag.finalize_storages()
        except Exception as exc:
            print(f"[warn] finalize failed for line={sample['_line_no']}: {exc!r}", flush=True)


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="datasets/UltraDomain/mix.jsonl")
    parser.add_argument("--output", default="results/historical_baselines/lightrag_current.jsonl")
    parser.add_argument("--cache-root", default="data_baselines/lightrag")
    parser.add_argument("--line-nos", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--retry-errors",
        action="store_true",
        help="When resuming, rerun rows whose existing output is an error or empty prediction.",
    )
    parser.add_argument("--query-mode", default="hybrid", choices=["local", "global", "hybrid", "naive", "mix", "bypass"])
    parser.add_argument("--response-type", default="Multiple Paragraphs")
    parser.add_argument("--chunk-token-size", type=int, default=4000)
    parser.add_argument("--chunk-overlap-token-size", type=int, default=100)
    parser.add_argument("--entity-extract-max-gleaning", type=int, default=0)
    parser.add_argument("--entity-extract-max-entities", type=int, default=20)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--chunk-top-k", type=int, default=5)
    parser.add_argument("--llm-max-async", type=int, default=1)
    parser.add_argument("--embedding-max-async", type=int, default=1)
    parser.add_argument("--llm-timeout", type=int, default=180)
    parser.add_argument("--embedding-timeout", type=int, default=30)
    args = parser.parse_args()

    load_dotenv(ROOT_DIR / ".env")
    args.dataset = ROOT_DIR / args.dataset
    args.output = ROOT_DIR / args.output
    args.cache_root = ROOT_DIR / args.cache_root
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.cache_root.mkdir(parents=True, exist_ok=True)

    samples = load_samples(args.dataset)
    wanted = parse_line_nos(args.line_nos, len(samples))
    selected = [sample for sample in samples if sample["_line_no"] in wanted]
    if args.limit is not None:
        selected = selected[: args.limit]

    done = read_existing_lines(args.output, retry_errors=args.retry_errors) if args.resume else set()
    mode = "a" if args.resume else "w"
    with args.output.open(mode, encoding="utf-8") as out:
        for index, sample in enumerate(selected, start=1):
            line_no = sample["_line_no"]
            if line_no in done:
                print(f"[skip] line={line_no} already exists")
                continue
            print(
                f"[run] {index}/{len(selected)} line={line_no} "
                f"dataset={sample.get('dataset')} chars={len(sample.get('context', ''))}",
                flush=True,
            )
            try:
                result = await run_sample(sample, args)
                out.write(json.dumps(result, ensure_ascii=False) + "\n")
                out.flush()
                print(
                    f"[done] line={line_no} index={result['index_seconds']:.3f}s "
                    f"query={result['query_seconds']:.3f}s",
                    flush=True,
                )
            except Exception as exc:
                error = {
                    "line_no": line_no,
                    "id": sample.get("_id"),
                    "dataset": sample.get("dataset"),
                    "question": sample.get("input", ""),
                    "reference_answers": sample.get("answers", []),
                    "prediction": "",
                    "method": "lightrag",
                    "error": repr(exc),
                }
                out.write(json.dumps(error, ensure_ascii=False) + "\n")
                out.flush()
                print(f"[error] line={line_no} {exc!r}", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
