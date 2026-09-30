#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build JSONL datasets whose context is the retrieved top-k chunks."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from experiments.run_retrieval_baseline import chunk_text  # noqa: E402


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            row.setdefault("_line_no", line_no)
            rows.append(row)
    return rows


def context_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def build_row(
    sample: Dict[str, Any],
    retrieval_row: Dict[str, Any],
    chunk_chars: int,
    overlap_chars: int,
    context_mode: str,
) -> Dict[str, Any]:
    context = sample.get("context", "")
    chunks = chunk_text(context, chunk_chars, overlap_chars)
    if not chunks:
        chunks = [context]
    selected = []
    for rank, item in enumerate(retrieval_row.get("retrieved_chunks", []), start=1):
        source_index = int(item["source_index"])
        if source_index < 0 or source_index >= len(chunks):
            continue
        selected.append(
            {
                "rank": rank,
                "source_index": source_index,
                "score": item.get("score"),
                "text": chunks[source_index],
            }
        )
    retrieved_context = "\n\n".join(
        f"[Retrieved Chunk {item['rank']}; source_index={item['source_index']}; score={item.get('score')}]\n{item['text']}"
        for item in selected
    )
    if context_mode == "topk_only":
        output_context = retrieved_context
    elif context_mode == "prepend_full":
        output_context = (
            retrieved_context
            + "\n\n[Full Original Context Begins]\n"
            + context
            if retrieved_context
            else context
        )
    else:
        raise ValueError(f"Unsupported context_mode: {context_mode}")
    output = dict(sample)
    output["context"] = output_context
    output["original_context_hash"] = context_hash(context)
    output["original_context_chars"] = len(context)
    output["retrieved_context_chars"] = len(retrieved_context)
    output["output_context_chars"] = len(output_context)
    output["retrieval_prefilter"] = {
        "source_baseline": retrieval_row.get("baseline", "naive"),
        "context_mode": context_mode,
        "chunk_chars": chunk_chars,
        "overlap_chars": overlap_chars,
        "top_k": len(selected),
        "num_source_chunks": len(chunks),
        "retrieved_chunks": [
            {
                "rank": item["rank"],
                "source_index": item["source_index"],
                "score": item.get("score"),
                "chars": len(item["text"]),
            }
            for item in selected
        ],
    }
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--retrieval-output", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--chunk-chars", type=int, default=1800)
    parser.add_argument("--overlap-chars", type=int, default=200)
    parser.add_argument(
        "--context-mode",
        choices=["topk_only", "prepend_full"],
        default="topk_only",
        help="Use only retrieved chunks or prepend retrieved chunks before the full original context.",
    )
    args = parser.parse_args()

    dataset_path = ROOT_DIR / args.dataset
    retrieval_path = ROOT_DIR / args.retrieval_output
    output_path = ROOT_DIR / args.output

    samples = load_jsonl(dataset_path)
    retrieval_rows = {int(row["line_no"]): row for row in load_jsonl(retrieval_path) if not row.get("error")}
    output_path.parent.mkdir(parents=True, exist_ok=True)

    written = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for sample in samples:
            line_no = int(sample["_line_no"])
            retrieval_row = retrieval_rows.get(line_no)
            if retrieval_row is None:
                continue
            row = build_row(sample, retrieval_row, args.chunk_chars, args.overlap_chars, args.context_mode)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            written += 1
    print(f"Wrote {written}/{len(samples)} rows to {output_path} with context_mode={args.context_mode}")


if __name__ == "__main__":
    main()
