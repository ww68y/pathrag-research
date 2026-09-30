import argparse
import asyncio
import json
import shutil
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))
load_dotenv(ROOT_DIR / ".env")

from PathRAG.prompt import PROMPTS
from experiments.build_multiquery_condensed_subset import split_chunks
from experiments.prepare_vsr_pathrag_v50_zero_overlap_graphs import (
    graph_counts,
    load_rows,
    write_rows,
)
from experiments.run_ultradomain_subset import (
    context_hash,
    index_context,
)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as source:
        return [
            json.loads(line)
            for line in source
            if line.strip()
        ]


def source_row(path: Path, line_no: int) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as source:
        for index, line in enumerate(source):
            if index == line_no:
                return json.loads(line)
    raise IndexError(f"missing source row {line_no} in {path}")


def unique_contexts(
    rows: list[dict[str, Any]],
    added_indices_field: str = (
        "anchor_preserving_added_chunk_indices"
    ),
) -> list[dict[str, Any]]:
    contexts = []
    seen = set()
    for line_no, row in enumerate(rows):
        fingerprint = context_hash(str(row["context"]))
        if fingerprint in seen:
            continue
        seen.add(fingerprint)
        contexts.append(
            {
                "line_no": line_no,
                "dataset": str(row["dataset"]),
                "context_hash": fingerprint,
                "context_chars": len(str(row["context"])),
                "source_line_no": int(row["source_line_no"]),
                "added_indices": [
                    int(index)
                    for index in row.get(
                        added_indices_field,
                        [],
                    )
                ],
            }
        )
    return contexts


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        default=(
            "datasets/UltraDomain/"
            "vsr_v52_anchor_preserving_validation_30.jsonl"
        ),
    )
    parser.add_argument(
        "--base-dataset",
        default=(
            "datasets/UltraDomain/"
            "v46_zero_overlap_validation_30.jsonl"
        ),
    )
    parser.add_argument(
        "--output",
        default=(
            "results/"
            "vsr_pathrag_anchor_preserving_graph_preparation.jsonl"
        ),
    )
    parser.add_argument("--chunk-chars", type=int, default=1800)
    parser.add_argument("--chunk-overlap", type=int, default=220)
    parser.add_argument("--chunk-tokens", type=int, default=1200)
    parser.add_argument("--entity-gleaning", type=int, default=0)
    parser.add_argument("--llm-async", type=int, default=1)
    parser.add_argument("--embedding-async", type=int, default=1)
    parser.add_argument(
        "--added-indices-field",
        default="anchor_preserving_added_chunk_indices",
    )
    parser.add_argument("--staging-prefix", default="v52")
    args = parser.parse_args()

    dataset_rows = load_jsonl(ROOT_DIR / args.dataset)
    base_rows = load_jsonl(ROOT_DIR / args.base_dataset)
    base_by_domain = {}
    for row in base_rows:
        domain = str(row["dataset"])
        base_by_domain.setdefault(
            domain,
            {
                "line_no": int(row["_subset_line_no"]),
                "context_hash": context_hash(str(row["context"])),
            },
        )

    output_path = ROOT_DIR / args.output
    prepared = load_rows(output_path)
    contexts = unique_contexts(
        dataset_rows,
        added_indices_field=args.added_indices_field,
    )
    PROMPTS["process_tickers"] = ["-", "\\", "|", "/"]
    rag_args = SimpleNamespace(
        chunk_tokens=args.chunk_tokens,
        entity_gleaning=args.entity_gleaning,
        llm_async=args.llm_async,
        embedding_async=args.embedding_async,
    )

    for position, item in enumerate(contexts, start=1):
        domain = item["dataset"]
        line_no = int(item["line_no"])
        fingerprint = str(item["context_hash"])
        final_dir = (
            ROOT_DIR
            / "data_repro"
            / f"ctx_{line_no}_{fingerprint}"
        )
        graph_path = final_dir / "graph_chunk_entity_relation.graphml"
        if graph_path.exists():
            nodes, edges = graph_counts(final_dir)
            prepared[fingerprint] = {
                **item,
                "ok": True,
                "reused": True,
                "incremental": False,
                "graph_nodes": nodes,
                "graph_edges": edges,
                "insert_seconds": 0.0,
                "working_dir": str(final_dir.relative_to(ROOT_DIR)),
                "reference_answer_used": False,
            }
            write_rows(output_path, prepared)
            print(
                f"[{position}/{len(contexts)}] reuse {domain} "
                f"nodes={nodes} edges={edges}",
                flush=True,
            )
            continue

        base = base_by_domain[domain]
        base_dir = (
            ROOT_DIR
            / "data_repro"
            / f"ctx_{base['line_no']}_{base['context_hash']}"
        )
        if not (
            base_dir / "graph_chunk_entity_relation.graphml"
        ).exists():
            raise FileNotFoundError(
                f"missing frozen base graph for {domain}: {base_dir}"
            )
        added_indices = item["added_indices"]
        if not added_indices:
            raise RuntimeError(
                f"unchanged {domain} context should reuse {base_dir}"
            )

        source_path = (
            ROOT_DIR
            / "datasets"
            / "UltraDomain"
            / f"{domain}.jsonl"
        )
        original = source_row(
            source_path,
            int(item["source_line_no"]),
        )
        chunks = split_chunks(
            str(original["context"]),
            args.chunk_chars,
            args.chunk_overlap,
        )
        delta_context = "\n\n".join(
            f"[SOURCE CHUNK {index}]\n{chunks[index]}"
            for index in added_indices
        )
        staging_dir = (
            ROOT_DIR
            / "data_repro"
            / (
                f"{args.staging_prefix}_staging_"
                f"{line_no}_{fingerprint}"
            )
        )
        previous = prepared.get(fingerprint) or {}
        if staging_dir.exists() and not previous.get("ok", True):
            failed_dir = staging_dir.with_name(
                f"{staging_dir.name}_failed_{int(time.time())}"
            )
            staging_dir.rename(failed_dir)
        if not staging_dir.exists():
            shutil.copytree(base_dir, staging_dir)

        started = time.perf_counter()
        try:
            _rag, _ = await index_context(
                delta_context,
                staging_dir,
                rag_args,
            )
            insert_seconds = time.perf_counter() - started
            if final_dir.exists():
                raise FileExistsError(
                    f"refusing to replace existing {final_dir}"
                )
            staging_dir.rename(final_dir)
            nodes, edges = graph_counts(final_dir)
            prepared[fingerprint] = {
                **item,
                "ok": True,
                "reused": False,
                "incremental": True,
                "base_working_dir": str(
                    base_dir.relative_to(ROOT_DIR)
                ),
                "graph_nodes": nodes,
                "graph_edges": edges,
                "insert_seconds": round(insert_seconds, 3),
                "working_dir": str(final_dir.relative_to(ROOT_DIR)),
                "reference_answer_used": False,
            }
            print(
                f"[{position}/{len(contexts)}] indexed {domain} "
                f"added={len(added_indices)} nodes={nodes} "
                f"edges={edges} seconds={insert_seconds:.1f}",
                flush=True,
            )
        except Exception as exc:
            failed_dir = staging_dir.with_name(
                f"{staging_dir.name}_failed_{int(time.time())}"
            )
            if staging_dir.exists():
                staging_dir.rename(failed_dir)
            prepared[fingerprint] = {
                **item,
                "ok": False,
                "reused": False,
                "incremental": True,
                "base_working_dir": str(
                    base_dir.relative_to(ROOT_DIR)
                ),
                "working_dir": str(failed_dir.relative_to(ROOT_DIR)),
                "insert_seconds": round(
                    time.perf_counter() - started,
                    3,
                ),
                "error": f"{type(exc).__name__}: {exc}",
                "reference_answer_used": False,
            }
            print(
                f"[{position}/{len(contexts)}] failed {domain}: "
                f"{type(exc).__name__}: {exc}",
                flush=True,
            )
        write_rows(output_path, prepared)

    failures = [
        row for row in prepared.values() if not row.get("ok")
    ]
    print(
        f"contexts={len(contexts)} failures={len(failures)} "
        f"reference_answer_used=False"
    )


if __name__ == "__main__":
    asyncio.run(main())
