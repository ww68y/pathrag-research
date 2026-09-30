from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import os
import struct
import sys
import time
from pathlib import Path

import aiohttp
import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI

if sys.platform.startswith("win") and hasattr(asyncio, "WindowsSelectorEventLoopPolicy"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

NAIVE_SYSTEM_PROMPT = """You are a question answering assistant.
Answer the question using only the retrieved context chunks.
If the chunks do not contain enough evidence, say that the answer cannot be determined from the provided context.
Keep the answer focused and avoid unnecessary details."""

HYDE_PROMPT = """Write a concise hypothetical answer passage that would likely contain the evidence needed to answer this question.
Do not say you are guessing. Focus on entities, facts, and relations that should be useful for retrieval.

Question:
{question}

Hypothetical answer passage:"""

ANSWER_PROMPT = """Retrieved context chunks:
{context}

Question:
{question}

Answer:"""


def embedding_endpoint(base_url: str) -> str:
    base_url = base_url.rstrip("/")
    if base_url.endswith("/embeddings"):
        return base_url
    return base_url + "/embeddings"


def is_quota_error(error: Exception | None) -> bool:
    text = str(error or "").lower()
    return "402" in text or "balance is insufficient" in text or ("insufficient" in text and "balance" in text)


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
    return line_nos


def load_all_samples(path: Path) -> list[dict]:
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


def chunk_text(text: str, chunk_chars: int, overlap_chars: int) -> list[str]:
    text = text.strip()
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        target_end = min(len(text), start + chunk_chars)
        end = target_end
        if target_end < len(text):
            newline = text.rfind("\n", start + int(chunk_chars * 0.6), target_end)
            sentence = max(
                text.rfind(". ", start + int(chunk_chars * 0.6), target_end),
                text.rfind("? ", start + int(chunk_chars * 0.6), target_end),
                text.rfind("! ", start + int(chunk_chars * 0.6), target_end),
            )
            boundary = max(newline, sentence)
            if boundary > start:
                end = boundary + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(0, end - overlap_chars)
    return chunks


async def embed_texts(
    texts: list[str],
    api_key: str,
    base_url: str,
    model: str,
    max_chars: int,
    batch_size: int,
    request_timeout: float,
) -> np.ndarray:
    if api_key and not api_key.startswith("Bearer "):
        api_key = "Bearer " + api_key
    headers = {"Authorization": api_key, "Content-Type": "application/json"}
    outputs: list[np.ndarray] = []
    timeout = aiohttp.ClientTimeout(total=request_timeout)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        for start in range(0, len(texts), batch_size):
            batch = [text[:max_chars] for text in texts[start : start + batch_size]]
            payload = {"model": model, "input": batch, "encoding_format": "base64"}
            for attempt in range(3):
                async with session.post(base_url, headers=headers, json=payload) as response:
                    content = await response.json()
                if "code" not in content:
                    break
                if attempt == 2:
                    raise RuntimeError(f"Embedding API error: {content}")
                await asyncio.sleep(3 * (attempt + 1))
            for item in content["data"]:
                raw = base64.b64decode(item["embedding"])
                n = len(raw) // 4
                outputs.append(np.array(struct.unpack("<" + "f" * n, raw), dtype=np.float32))
    return np.vstack(outputs)


async def load_or_create_index(
    sample: dict,
    cache_root: Path,
    api_key: str,
    base_url: str,
    model: str,
    max_chars: int,
    batch_size: int,
    request_timeout: float,
    chunk_chars: int,
    overlap_chars: int,
) -> tuple[list[str], np.ndarray, float]:
    context = sample.get("context", "")
    h = context_hash(context)
    cache_dir = cache_root / f"ctx_{h}"
    chunks_path = cache_dir / "chunks.json"
    embeddings_path = cache_dir / "embeddings.npy"

    if chunks_path.exists() and embeddings_path.exists():
        chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
        embeddings = np.load(embeddings_path)
        return chunks, embeddings, 0.0

    started = time.perf_counter()
    cache_dir.mkdir(parents=True, exist_ok=True)
    chunks = chunk_text(context, chunk_chars, overlap_chars)
    if not chunks:
        chunks = [context]
    embeddings = await embed_texts(chunks, api_key, base_url, model, max_chars, batch_size, request_timeout)
    chunks_path.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
    np.save(embeddings_path, embeddings)
    return chunks, embeddings, time.perf_counter() - started


def top_k_chunks(chunks: list[str], embeddings: np.ndarray, query_embedding: np.ndarray, top_k: int) -> list[tuple[int, float, str]]:
    chunk_norms = np.linalg.norm(embeddings, axis=1)
    query_norm = np.linalg.norm(query_embedding)
    scores = embeddings @ query_embedding / (chunk_norms * query_norm + 1e-12)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(int(i), float(scores[i]), chunks[int(i)]) for i in top_indices]


async def complete(
    client: AsyncOpenAI,
    model: str,
    messages: list[dict],
    request_timeout: float,
) -> str:
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = await asyncio.wait_for(
                client.chat.completions.create(
                    model=model,
                    temperature=0,
                    messages=messages,
                ),
                timeout=request_timeout,
            )
            return response.choices[0].message.content or ""
        except Exception as error:
            last_error = error
            if attempt == 2:
                raise
            await asyncio.sleep(3 * (attempt + 1))
    raise RuntimeError(f"Completion failed: {last_error}")


async def run_sample(
    sample: dict,
    mode: str,
    llm_client: AsyncOpenAI,
    llm_model: str,
    embedding_api_key: str,
    embedding_base_url: str,
    embedding_model: str,
    embedding_max_chars: int,
    embedding_batch_size: int,
    cache_root: Path,
    chunk_chars: int,
    overlap_chars: int,
    top_k: int,
    request_timeout: float,
) -> dict:
    started = time.perf_counter()
    chunks, embeddings, index_seconds = await load_or_create_index(
        sample,
        cache_root,
        embedding_api_key,
        embedding_base_url,
        embedding_model,
        embedding_max_chars,
        embedding_batch_size,
        request_timeout,
        chunk_chars,
        overlap_chars,
    )

    question = sample.get("input", "")
    hyde_query = None
    retrieval_query = question
    if mode == "hyde":
        hyde_query = await complete(
            llm_client,
            llm_model,
            [
                {"role": "system", "content": "You generate concise retrieval passages."},
                {"role": "user", "content": HYDE_PROMPT.format(question=question)},
            ],
            request_timeout,
        )
        retrieval_query = hyde_query

    query_embedding = (
        await embed_texts(
            [retrieval_query],
            embedding_api_key,
            embedding_base_url,
            embedding_model,
            embedding_max_chars,
            embedding_batch_size,
            request_timeout,
        )
    )[0]
    retrieved = top_k_chunks(chunks, embeddings, query_embedding, top_k)
    retrieved_context = "\n\n".join(
        f"[Chunk {rank}; score={score:.4f}; source_index={idx}]\n{text}"
        for rank, (idx, score, text) in enumerate(retrieved, start=1)
    )

    prediction = await complete(
        llm_client,
        llm_model,
        [
            {"role": "system", "content": NAIVE_SYSTEM_PROMPT},
            {"role": "user", "content": ANSWER_PROMPT.format(context=retrieved_context, question=question)},
        ],
        request_timeout,
    )
    total_seconds = time.perf_counter() - started
    return {
        "id": sample.get("_id") or f"line-{sample['_line_no']}",
        "line_no": sample["_line_no"],
        "dataset": sample.get("dataset"),
        "label": sample.get("label"),
        "question": question,
        "reference_answers": sample.get("answers", []),
        "prediction": prediction,
        "context_chars": len(sample.get("context", "")),
        "graph_nodes": 0,
        "graph_edges": 0,
        "insert_seconds": round(index_seconds, 3),
        "query_seconds": round(total_seconds - index_seconds, 3),
        "baseline": mode,
        "chunk_chars": chunk_chars,
        "chunk_overlap_chars": overlap_chars,
        "top_k": top_k,
        "num_chunks": len(chunks),
        "retrieved_chunks": [
            {"source_index": idx, "score": round(score, 6), "chars": len(text)}
            for idx, score, text in retrieved
        ],
        "hyde_query": hyde_query,
    }


def load_completed_line_nos(path: Path, retry_errors: bool = False) -> set[int]:
    if not path.exists():
        return set()
    completed = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                if retry_errors and row.get("error"):
                    continue
                completed.add(int(row["line_no"]))
            except Exception:
                continue
    return completed


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="datasets/UltraDomain/mix.jsonl")
    parser.add_argument("--line-nos", default=None, help="Comma-separated zero-based line numbers; default runs all")
    parser.add_argument("--output", required=True)
    parser.add_argument("--mode", choices=["naive", "hyde"], required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--chunk-chars", type=int, default=1800)
    parser.add_argument("--overlap-chars", type=int, default=200)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--request-timeout", type=float, default=120.0)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--skip-existing-output", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument(
        "--shared-context-cache",
        action="store_true",
        help="Reuse the same context chunk/embedding cache across naive and HyDE runs; query embeddings are still fresh per mode.",
    )
    args = parser.parse_args()

    load_dotenv(ROOT_DIR / ".env")
    api_key = os.getenv("DEEPSEEK_API_KEY")
    llm_base_url = os.getenv("DEEPSEEK_API_BASE")
    llm_model = args.model or os.getenv("DEEPSEEK_MODEL")
    embedding_base_url = os.getenv("EMBEDDING_API_BASE", "https://api.siliconflow.cn/v1/embeddings")
    embedding_model = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    embedding_dim = int(os.getenv("EMBEDDING_DIM", "1024"))
    embedding_max_chars = int(os.getenv("EMBEDDING_MAX_TOKENS", "8192"))

    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is missing in .env")
    if not llm_base_url:
        raise RuntimeError("DEEPSEEK_API_BASE is missing in .env")
    if not llm_model:
        raise RuntimeError("DEEPSEEK_MODEL is missing in .env")
    if not embedding_base_url or not embedding_model:
        raise RuntimeError("EMBEDDING_API_BASE or EMBEDDING_MODEL is missing in .env and no fallback is configured")
    embedding_base_url = embedding_endpoint(embedding_base_url)

    all_samples = load_all_samples(ROOT_DIR / args.dataset)
    selected_lines = parse_line_nos(args.line_nos, len(all_samples))
    samples = [sample for sample in all_samples if sample["_line_no"] in selected_lines]

    output_path = ROOT_DIR / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = load_completed_line_nos(output_path, retry_errors=args.retry_errors) if args.skip_existing_output else set()
    samples = [sample for sample in samples if sample["_line_no"] not in completed]

    cache_scope = "shared" if args.shared_context_cache else args.mode
    cache_root = ROOT_DIR / "data_baselines" / cache_scope / (
        f"{embedding_model.replace('/', '_')}_dim{embedding_dim}_chunk{args.chunk_chars}_overlap{args.overlap_chars}"
    )
    cache_root.mkdir(parents=True, exist_ok=True)

    llm_client = AsyncOpenAI(api_key=api_key, base_url=llm_base_url)
    file_mode = "a" if args.append else "w"
    print(f"Mode: {args.mode}")
    print(f"Selected {len(samples)} sample(s); skipped {len(completed)} existing output sample(s)")
    print(f"Cache: {cache_root}")
    with output_path.open(file_mode, encoding="utf-8") as out:
        for i, sample in enumerate(samples, start=1):
            print(f"[{i}/{len(samples)}] line={sample['_line_no']} {sample.get('input')}", flush=True)
            result = None
            started = time.perf_counter()
            last_error: Exception | None = None
            for attempt in range(1, args.max_retries + 1):
                try:
                    result = await run_sample(
                        sample,
                        args.mode,
                        llm_client,
                        llm_model,
                        api_key,
                        embedding_base_url,
                        embedding_model,
                        embedding_max_chars,
                        args.embedding_batch_size,
                        cache_root,
                        args.chunk_chars,
                        args.overlap_chars,
                        args.top_k,
                        args.request_timeout,
                    )
                    break
                except Exception as error:
                    last_error = error
                    print(f"  ERROR attempt {attempt}/{args.max_retries}: {error}", flush=True)
                    if attempt < args.max_retries:
                        await asyncio.sleep(3 * attempt)
            if result is None:
                result = {
                    "id": sample.get("_id") or f"line-{sample['_line_no']}",
                    "line_no": sample["_line_no"],
                    "dataset": sample.get("dataset"),
                    "label": sample.get("label"),
                    "question": sample.get("input", ""),
                    "reference_answers": sample.get("answers", []),
                    "prediction": f"Error: {last_error}",
                    "context_chars": len(sample.get("context", "")),
                    "insert_seconds": 0.0,
                    "query_seconds": round(time.perf_counter() - started, 3),
                    "baseline": args.mode,
                    "chunk_chars": args.chunk_chars,
                    "chunk_overlap_chars": args.overlap_chars,
                    "top_k": args.top_k,
                    "num_chunks": 0,
                    "retrieved_chunks": [],
                    "hyde_query": None,
                    "error": str(last_error),
                }
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            out.flush()
            status = "ERROR" if result.get("error") else "Done"
            print(
                f"{status}: chunks={result['num_chunks']} index={result['insert_seconds']}s "
                f"query={result['query_seconds']}s",
                flush=True,
            )
            if is_quota_error(last_error):
                print("Quota exhausted; saved current error row and stopped early.", flush=True)
                break

    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
