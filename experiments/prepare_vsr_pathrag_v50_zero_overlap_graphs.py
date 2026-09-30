import argparse
import asyncio
import json
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
load_dotenv(ROOT_DIR / ".env")

from PathRAG.prompt import PROMPTS
from experiments.run_ultradomain_subset import (
    context_hash,
    index_context,
)


def load_unique_contexts(path: Path) -> list[dict]:
    contexts = []
    seen = set()
    with path.open("r", encoding="utf-8") as source:
        for line_no, line in enumerate(source):
            if not line.strip():
                continue
            item = json.loads(line)
            context = str(item.get("context", ""))
            fingerprint = context_hash(context)
            if fingerprint in seen:
                continue
            seen.add(fingerprint)
            contexts.append(
                {
                    "line_no": line_no,
                    "dataset": item.get("dataset") or item.get("label"),
                    "context": context,
                    "context_hash": fingerprint,
                    "context_chars": len(context),
                }
            )
    return contexts


def load_rows(path: Path) -> dict[str, dict]:
    rows = {}
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as source:
        for line in source:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("context_hash"):
                rows[str(row["context_hash"])] = row
    return rows


def write_rows(path: Path, rows: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(
        rows.values(),
        key=lambda row: int(row.get("line_no", 0)),
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as target:
        for row in ordered:
            target.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def graph_counts(path: Path) -> tuple[int, int]:
    import networkx as nx

    graph = nx.read_graphml(
        path / "graph_chunk_entity_relation.graphml"
    )
    return graph.number_of_nodes(), graph.number_of_edges()


def newest_staging_dir(line_no: int, fingerprint: str) -> Path | None:
    candidates = sorted(
        (ROOT_DIR / "data_repro").glob(
            f"v50_staging_{line_no}_{fingerprint}*"
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    return candidates[0] if candidates else None


def prepare_staging_dir(
    final_dir: Path,
    line_no: int,
    fingerprint: str,
) -> Path:
    existing = newest_staging_dir(line_no, fingerprint)
    if (
        existing is not None
        and (existing / "graph_chunk_entity_relation.graphml").exists()
    ):
        return existing

    suffix = (
        ""
        if existing is None
        else f"_retry_{int(time.time())}"
    )
    staging = (
        ROOT_DIR
        / "data_repro"
        / f"v50_staging_{line_no}_{fingerprint}{suffix}"
    )
    staging.mkdir(parents=True, exist_ok=True)

    cache_sources = []
    if (final_dir / "durable_llm_cache").exists():
        cache_sources.append(final_dir / "durable_llm_cache")
    if (
        existing is not None
        and (existing / "durable_llm_cache").exists()
    ):
        cache_sources.append(existing / "durable_llm_cache")
    for cache_source in cache_sources:
        shutil.copytree(
            cache_source,
            staging / "durable_llm_cache",
            dirs_exist_ok=True,
        )
    return staging


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default=(
            "datasets/UltraDomain/"
            "v46_zero_overlap_validation_30.jsonl"
        ),
    )
    parser.add_argument(
        "--output",
        default=(
            "results/"
            "vsr_pathrag_v50_zero_overlap_graph_preparation.jsonl"
        ),
    )
    parser.add_argument("--chunk-tokens", type=int, default=1200)
    parser.add_argument("--entity-gleaning", type=int, default=0)
    parser.add_argument("--llm-async", type=int, default=1)
    parser.add_argument("--embedding-async", type=int, default=1)
    args = parser.parse_args()

    PROMPTS["process_tickers"] = ["-", "\\", "|", "/"]
    dataset_path = ROOT_DIR / args.dataset
    output_path = ROOT_DIR / args.output
    contexts = load_unique_contexts(dataset_path)
    rows = load_rows(output_path)
    rag_args = SimpleNamespace(
        chunk_tokens=args.chunk_tokens,
        entity_gleaning=args.entity_gleaning,
        llm_async=args.llm_async,
        embedding_async=args.embedding_async,
    )

    for index, item in enumerate(contexts, start=1):
        line_no = int(item["line_no"])
        fingerprint = str(item["context_hash"])
        metadata = {
            key: value
            for key, value in item.items()
            if key != "context"
        }
        final_dir = (
            ROOT_DIR
            / "data_repro"
            / f"ctx_{line_no}_{fingerprint}"
        )
        graph_path = final_dir / "graph_chunk_entity_relation.graphml"
        if graph_path.exists():
            nodes, edges = graph_counts(final_dir)
            rows[fingerprint] = {
                **metadata,
                "ok": True,
                "reused": True,
                "graph_nodes": nodes,
                "graph_edges": edges,
                "insert_seconds": 0.0,
                "working_dir": str(final_dir.relative_to(ROOT_DIR)),
                "reference_answer_used": False,
            }
            write_rows(output_path, rows)
            print(
                f"[{index}/{len(contexts)}] reuse "
                f"{item['dataset']} nodes={nodes} edges={edges}",
                flush=True,
            )
            continue

        staging_dir = prepare_staging_dir(
            final_dir,
            line_no,
            fingerprint,
        )
        staged_graph = (
            staging_dir / "graph_chunk_entity_relation.graphml"
        )
        try:
            if staged_graph.exists():
                insert_seconds = 0.0
            else:
                print(
                    f"[{index}/{len(contexts)}] index "
                    f"{item['dataset']} chars={item['context_chars']} "
                    f"staging={staging_dir.name}",
                    flush=True,
                )
                _rag, insert_seconds = await index_context(
                    item["context"],
                    staging_dir,
                    rag_args,
                )
            if not staged_graph.exists():
                raise RuntimeError(
                    "Indexing finished without a graph artifact"
                )
            final_dir.mkdir(parents=True, exist_ok=True)
            shutil.copytree(
                staging_dir,
                final_dir,
                dirs_exist_ok=True,
            )
            nodes, edges = graph_counts(final_dir)
            rows[fingerprint] = {
                **metadata,
                "ok": True,
                "reused": False,
                "graph_nodes": nodes,
                "graph_edges": edges,
                "insert_seconds": round(insert_seconds, 3),
                "working_dir": str(final_dir.relative_to(ROOT_DIR)),
                "staging_dir": str(
                    staging_dir.relative_to(ROOT_DIR)
                ),
                "reference_answer_used": False,
            }
            write_rows(output_path, rows)
            print(
                f"[{index}/{len(contexts)}] complete "
                f"{item['dataset']} nodes={nodes} edges={edges} "
                f"seconds={insert_seconds:.1f}",
                flush=True,
            )
        except Exception as exc:
            rows[fingerprint] = {
                **metadata,
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "working_dir": str(final_dir.relative_to(ROOT_DIR)),
                "staging_dir": str(
                    staging_dir.relative_to(ROOT_DIR)
                ),
                "reference_answer_used": False,
            }
            write_rows(output_path, rows)
            raise

    successful = sum(bool(row.get("ok")) for row in rows.values())
    print(
        f"Prepared {successful}/{len(contexts)} unique graphs; "
        f"output={output_path}",
        flush=True,
    )


if __name__ == "__main__":
    asyncio.run(main())
