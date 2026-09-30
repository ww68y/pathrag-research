import argparse
import asyncio
import base64
import hashlib
import json
import os
import re
import struct
import sys
import time
from collections import Counter
from pathlib import Path

import aiohttp
import networkx as nx
import numpy as np
from dotenv import load_dotenv
from openai import AsyncOpenAI


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

GRAPH_RAG_SYSTEM_PROMPT = """You are a graph-based retrieval QA assistant.
Use only the provided graph community evidence and supporting text chunks.
If the evidence is insufficient, say that the answer cannot be determined from the provided evidence.
Answer the question directly and avoid unnecessary detail."""

GRAPH_RAG_ANSWER_PROMPT = """Graph community evidence:
{community_context}

Supporting text chunks:
{chunk_context}

Question:
{question}

Answer:"""


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


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            if not line.strip():
                continue
            item = json.loads(line)
            item.setdefault("_line_no", line_no)
            rows.append(item)
    return rows


def load_completed_line_nos(path: Path) -> set[int]:
    if not path.exists():
        return set()
    completed = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                completed.add(int(json.loads(line)["line_no"]))
            except Exception:
                continue
    return completed


def context_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def normalize_text(value: object, max_chars: int | None = None) -> str:
    text = "" if value is None else str(value)
    text = text.replace("<SEP>", "; ")
    text = text.strip()
    if len(text) >= 2 and text[0] == '"' and text[-1] == '"':
        text = text[1:-1]
    text = re.sub(r"\s+", " ", text).strip()
    if max_chars and len(text) > max_chars:
        return text[: max_chars - 3].rstrip() + "..."
    return text


def split_source_ids(value: object) -> list[str]:
    if value is None:
        return []
    text = str(value)
    parts = []
    for part in text.split("<SEP>"):
        item = normalize_text(part)
        if item:
            parts.append(item)
    return parts


async def embed_texts(
    texts: list[str],
    api_key: str,
    base_url: str,
    model: str,
    max_chars: int,
    batch_size: int,
) -> np.ndarray:
    if api_key and not api_key.startswith("Bearer "):
        api_key = "Bearer " + api_key
    headers = {"Authorization": api_key, "Content-Type": "application/json"}
    outputs: list[np.ndarray] = []
    async with aiohttp.ClientSession() as session:
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


async def complete(client: AsyncOpenAI, model: str, messages: list[dict]) -> str:
    response = await client.chat.completions.create(
        model=model,
        temperature=0,
        messages=messages,
    )
    return response.choices[0].message.content or ""


def find_graph_dir(sample: dict, pathrag_by_line: dict[int, dict]) -> Path:
    line_no = int(sample["_line_no"])
    pathrag_item = pathrag_by_line.get(line_no)
    if pathrag_item and pathrag_item.get("working_dir"):
        graph_dir = Path(pathrag_item["working_dir"])
        if graph_dir.exists():
            return graph_dir

    h = context_hash(sample.get("context", ""))
    candidates = sorted((ROOT_DIR / "data_repro").glob(f"ctx_*_{h}"))
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"No PathRAG graph cache found for line {line_no}, context hash {h}")


def load_chunks(graph_dir: Path) -> dict[str, dict]:
    path = graph_dir / "kv_store_text_chunks.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "data" in data and isinstance(data["data"], dict):
        return data["data"]
    if isinstance(data, dict):
        return data
    raise ValueError(f"Unexpected chunk store format: {path}")


def weighted_degree(graph: nx.Graph, node: str) -> float:
    total = 0.0
    for _, _, data in graph.edges(node, data=True):
        try:
            total += float(data.get("weight", 1.0))
        except Exception:
            total += 1.0
    return total


def safe_communities(graph: nx.Graph, max_communities: int) -> list[set[str]]:
    undirected = graph.to_undirected()
    if undirected.number_of_nodes() == 0:
        return []
    if undirected.number_of_edges() == 0:
        communities = [{node} for node in undirected.nodes()]
    else:
        communities = list(nx.algorithms.community.greedy_modularity_communities(undirected, weight="weight"))
    communities = sorted((set(c) for c in communities), key=len, reverse=True)
    return communities[:max_communities]


def build_community_reports(
    graph: nx.Graph,
    chunks: dict[str, dict],
    max_communities: int,
    max_nodes_per_community: int,
    max_edges_per_community: int,
    max_description_chars: int,
    max_report_chars: int,
) -> list[dict]:
    communities = safe_communities(graph, max_communities)
    reports = []
    for community_id, nodes in enumerate(communities, start=1):
        subgraph = graph.subgraph(nodes)
        ranked_nodes = sorted(nodes, key=lambda node: weighted_degree(graph, node), reverse=True)
        selected_nodes = ranked_nodes[:max_nodes_per_community]

        node_lines = []
        source_counter: Counter[str] = Counter()
        for node in selected_nodes:
            data = graph.nodes[node]
            name = normalize_text(node)
            entity_type = normalize_text(data.get("entity_type"))
            description = normalize_text(data.get("description"), max_description_chars)
            node_lines.append(f"- {name} ({entity_type}): {description}")
            source_counter.update(split_source_ids(data.get("source_id")))

        ranked_edges = sorted(
            subgraph.edges(data=True),
            key=lambda item: float(item[2].get("weight", 1.0)) if str(item[2].get("weight", "")).strip() else 1.0,
            reverse=True,
        )[:max_edges_per_community]
        edge_lines = []
        for source, target, data in ranked_edges:
            source_name = normalize_text(source)
            target_name = normalize_text(target)
            description = normalize_text(data.get("description"), max_description_chars)
            keywords = normalize_text(data.get("keywords"), 160)
            edge_lines.append(f"- {source_name} -> {target_name}: {description} Keywords: {keywords}")
            source_counter.update(split_source_ids(data.get("source_id")))

        chunk_lines = []
        for chunk_id, _ in source_counter.most_common(3):
            content = chunks.get(chunk_id, {}).get("content")
            if content:
                chunk_lines.append(f"- {chunk_id}: {normalize_text(content, 280)}")

        report = "\n".join(
            [
                f"Community {community_id}: {len(nodes)} entities, {subgraph.number_of_edges()} relations",
                "Key entities:",
                *node_lines,
                "Key relations:",
                *edge_lines,
                "Representative source snippets:",
                *chunk_lines,
            ]
        )
        reports.append(
            {
                "community_id": community_id,
                "num_nodes": len(nodes),
                "num_edges": subgraph.number_of_edges(),
                "report": normalize_text(report, max_report_chars),
                "source_ids": [chunk_id for chunk_id, _ in source_counter.most_common(12)],
            }
        )
    return reports


async def load_or_create_community_index(
    graph_dir: Path,
    cache_root: Path,
    api_key: str,
    embedding_base_url: str,
    embedding_model: str,
    embedding_max_chars: int,
    embedding_batch_size: int,
    max_communities: int,
    max_nodes_per_community: int,
    max_edges_per_community: int,
    max_description_chars: int,
    max_report_chars: int,
) -> tuple[nx.Graph, dict[str, dict], list[dict], np.ndarray, float]:
    cache_dir = cache_root / graph_dir.name
    reports_path = cache_dir / "community_reports.json"
    embeddings_path = cache_dir / "community_embeddings.npy"

    graph = nx.read_graphml(graph_dir / "graph_chunk_entity_relation.graphml")
    chunks = load_chunks(graph_dir)

    if reports_path.exists() and embeddings_path.exists():
        reports = json.loads(reports_path.read_text(encoding="utf-8"))
        embeddings = np.load(embeddings_path)
        return graph, chunks, reports, embeddings, 0.0

    started = time.perf_counter()
    cache_dir.mkdir(parents=True, exist_ok=True)
    reports = build_community_reports(
        graph,
        chunks,
        max_communities,
        max_nodes_per_community,
        max_edges_per_community,
        max_description_chars,
        max_report_chars,
    )
    if not reports:
        reports = [{"community_id": 1, "num_nodes": 0, "num_edges": 0, "report": "No graph communities available.", "source_ids": []}]
    embeddings = await embed_texts(
        [report["report"] for report in reports],
        api_key,
        embedding_base_url,
        embedding_model,
        embedding_max_chars,
        embedding_batch_size,
    )
    reports_path.write_text(json.dumps(reports, ensure_ascii=False, indent=2), encoding="utf-8")
    np.save(embeddings_path, embeddings)
    return graph, chunks, reports, embeddings, time.perf_counter() - started


def top_k_reports(reports: list[dict], embeddings: np.ndarray, query_embedding: np.ndarray, top_k: int) -> list[tuple[dict, float]]:
    report_norms = np.linalg.norm(embeddings, axis=1)
    query_norm = np.linalg.norm(query_embedding)
    scores = embeddings @ query_embedding / (report_norms * query_norm + 1e-12)
    top_indices = np.argsort(scores)[::-1][:top_k]
    return [(reports[int(i)], float(scores[int(i)])) for i in top_indices]


def build_prompt_context(
    selected_reports: list[tuple[dict, float]],
    chunks: dict[str, dict],
    max_support_chunks: int,
    max_chunk_chars: int,
) -> tuple[str, str, list[dict]]:
    community_context_parts = []
    source_counter: Counter[str] = Counter()
    retrieved_communities = []
    for rank, (report, score) in enumerate(selected_reports, start=1):
        community_context_parts.append(
            f"[Community {rank}; id={report['community_id']}; score={score:.4f}; "
            f"nodes={report.get('num_nodes')}; edges={report.get('num_edges')}]\n{report['report']}"
        )
        retrieved_communities.append(
            {
                "community_id": report["community_id"],
                "score": round(score, 6),
                "nodes": report.get("num_nodes"),
                "edges": report.get("num_edges"),
            }
        )
        source_counter.update(report.get("source_ids", []))

    chunk_context_parts = []
    for rank, (chunk_id, count) in enumerate(source_counter.most_common(max_support_chunks), start=1):
        content = chunks.get(chunk_id, {}).get("content", "")
        if not content:
            continue
        chunk_context_parts.append(
            f"[Chunk {rank}; id={chunk_id}; community_mentions={count}]\n"
            f"{normalize_text(content, max_chunk_chars)}"
        )
    return "\n\n".join(community_context_parts), "\n\n".join(chunk_context_parts), retrieved_communities


async def run_sample(
    sample: dict,
    pathrag_by_line: dict[int, dict],
    llm_client: AsyncOpenAI,
    llm_model: str,
    api_key: str,
    embedding_base_url: str,
    embedding_model: str,
    embedding_max_chars: int,
    embedding_batch_size: int,
    cache_root: Path,
    top_k_communities: int,
    max_communities: int,
    max_nodes_per_community: int,
    max_edges_per_community: int,
    max_description_chars: int,
    max_report_chars: int,
    max_support_chunks: int,
    max_chunk_chars: int,
) -> dict:
    started = time.perf_counter()
    graph_dir = find_graph_dir(sample, pathrag_by_line)
    graph, chunks, reports, report_embeddings, index_seconds = await load_or_create_community_index(
        graph_dir,
        cache_root,
        api_key,
        embedding_base_url,
        embedding_model,
        embedding_max_chars,
        embedding_batch_size,
        max_communities,
        max_nodes_per_community,
        max_edges_per_community,
        max_description_chars,
        max_report_chars,
    )

    question = sample.get("input", "")
    query_embedding = (
        await embed_texts(
            [question],
            api_key,
            embedding_base_url,
            embedding_model,
            embedding_max_chars,
            embedding_batch_size,
        )
    )[0]
    selected_reports = top_k_reports(reports, report_embeddings, query_embedding, top_k_communities)
    community_context, chunk_context, retrieved_communities = build_prompt_context(
        selected_reports,
        chunks,
        max_support_chunks,
        max_chunk_chars,
    )

    prediction = await complete(
        llm_client,
        llm_model,
        [
            {"role": "system", "content": GRAPH_RAG_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": GRAPH_RAG_ANSWER_PROMPT.format(
                    community_context=community_context,
                    chunk_context=chunk_context or "No supporting chunks selected.",
                    question=question,
                ),
            },
        ],
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
        "graph_nodes": graph.number_of_nodes(),
        "graph_edges": graph.number_of_edges(),
        "insert_seconds": round(index_seconds, 3),
        "query_seconds": round(total_seconds - index_seconds, 3),
        "baseline": "graphrag_lite",
        "graph_dir": str(graph_dir),
        "num_communities": len(reports),
        "top_k_communities": top_k_communities,
        "retrieved_communities": retrieved_communities,
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="datasets/UltraDomain/domain_expansion_windowed_50.jsonl")
    parser.add_argument("--pathrag-results", default="results/historical_datasets/domain_expansion_windowed_50_pathrag.jsonl")
    parser.add_argument("--line-nos", default=None, help="Comma-separated zero-based line numbers; default runs all")
    parser.add_argument("--output", required=True)
    parser.add_argument("--model", default=None)
    parser.add_argument("--top-k-communities", type=int, default=4)
    parser.add_argument("--max-communities", type=int, default=12)
    parser.add_argument("--max-nodes-per-community", type=int, default=8)
    parser.add_argument("--max-edges-per-community", type=int, default=10)
    parser.add_argument("--max-description-chars", type=int, default=280)
    parser.add_argument("--max-report-chars", type=int, default=1600)
    parser.add_argument("--max-support-chunks", type=int, default=5)
    parser.add_argument("--max-chunk-chars", type=int, default=1200)
    parser.add_argument("--embedding-batch-size", type=int, default=16)
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--skip-existing-output", action="store_true")
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
        raise RuntimeError("EMBEDDING_API_BASE or EMBEDDING_MODEL is missing")

    samples = load_jsonl(ROOT_DIR / args.dataset)
    selected_lines = parse_line_nos(args.line_nos, len(samples))
    samples = [sample for sample in samples if sample["_line_no"] in selected_lines]

    pathrag_rows = load_jsonl(ROOT_DIR / args.pathrag_results)
    pathrag_by_line = {int(item["line_no"]): item for item in pathrag_rows}

    output_path = ROOT_DIR / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = load_completed_line_nos(output_path) if args.skip_existing_output else set()
    samples = [sample for sample in samples if sample["_line_no"] not in completed]

    cache_root = ROOT_DIR / "data_baselines" / "graphrag_lite" / (
        f"{embedding_model.replace('/', '_')}_dim{embedding_dim}_comm{args.max_communities}"
        f"_topnodes{args.max_nodes_per_community}_topedges{args.max_edges_per_community}"
    )
    cache_root.mkdir(parents=True, exist_ok=True)

    llm_client = AsyncOpenAI(api_key=api_key, base_url=llm_base_url)
    file_mode = "a" if args.append else "w"
    print("Mode: graphrag_lite")
    print(f"Selected {len(samples)} sample(s); skipped {len(completed)} existing output sample(s)")
    print(f"Cache: {cache_root}")
    with output_path.open(file_mode, encoding="utf-8") as out:
        for i, sample in enumerate(samples, start=1):
            print(f"[{i}/{len(samples)}] line={sample['_line_no']} {sample.get('input')}", flush=True)
            result = await run_sample(
                sample,
                pathrag_by_line,
                llm_client,
                llm_model,
                api_key,
                embedding_base_url,
                embedding_model,
                embedding_max_chars,
                args.embedding_batch_size,
                cache_root,
                args.top_k_communities,
                args.max_communities,
                args.max_nodes_per_community,
                args.max_edges_per_community,
                args.max_description_chars,
                args.max_report_chars,
                args.max_support_chunks,
                args.max_chunk_chars,
            )
            out.write(json.dumps(result, ensure_ascii=False) + "\n")
            out.flush()
            print(
                f"Done: communities={result['num_communities']} nodes={result['graph_nodes']} "
                f"edges={result['graph_edges']} index={result['insert_seconds']}s "
                f"query={result['query_seconds']}s",
                flush=True,
            )

    print(f"Saved results to {output_path}")


if __name__ == "__main__":
    asyncio.run(main())
