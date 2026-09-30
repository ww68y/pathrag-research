import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


ROOT_DIR = Path(__file__).resolve().parents[1]


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def context_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def split_chunks(text: str, size: int, overlap: int) -> list[str]:
    chunks = []
    start = 0
    while start < len(text):
        end = min(len(text), start + size)
        if end < len(text):
            candidates = [
                text.rfind(". ", start + size // 2, end),
                text.rfind("\n", start + size // 2, end),
            ]
            boundary = max(candidates)
            if boundary > start:
                end = boundary + 1
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = max(start + 1, end - overlap)
    return chunks


def select_chunk_indices(chunks: list[str], questions: list[str], top_k: int) -> tuple[list[int], list[list[int]]]:
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        stop_words="english",
        ngram_range=(1, 2),
        sublinear_tf=True,
        max_features=60000,
    )
    matrix = vectorizer.fit_transform(chunks + questions)
    chunk_matrix = matrix[:len(chunks)]
    question_matrix = matrix[len(chunks):]
    similarities = linear_kernel(question_matrix, chunk_matrix)
    per_question = []
    selected = set()
    for row in similarities:
        top = np.argsort(row)[::-1][:top_k]
        expanded = set()
        for index in top:
            index = int(index)
            expanded.add(index)
            if index > 0:
                expanded.add(index - 1)
            if index + 1 < len(chunks):
                expanded.add(index + 1)
        ranked = sorted(expanded)
        per_question.append(ranked)
        selected.update(ranked)
    return sorted(selected), per_question


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domains", default="agriculture,history,physics")
    parser.add_argument("--samples-per-domain", type=int, default=4)
    parser.add_argument("--chunk-chars", type=int, default=1800)
    parser.add_argument("--chunk-overlap", type=int, default=220)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument(
        "--exclude-dataset",
        action="append",
        default=None,
        help="Dataset whose original source contexts must be excluded; may be repeated",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    args = parser.parse_args()

    excluded: dict[str, set[str]] = defaultdict(set)
    exclusion_paths = args.exclude_dataset or [
        "datasets/UltraDomain/domain_expansion_windowed_50.jsonl"
    ]
    for exclusion_path in exclusion_paths:
        for row in load_jsonl(ROOT_DIR / exclusion_path):
            domain = str(row.get("source_dataset"))
            original_hash = str(
                row.get("original_source_context_hash")
                or row.get("source_context_hash")
            )
            excluded[domain].add(original_hash)

    output_rows = []
    manifest_rows = []
    for domain in [item.strip() for item in args.domains.split(",") if item.strip()]:
        source_rows = load_jsonl(ROOT_DIR / f"datasets/UltraDomain/{domain}.jsonl")
        groups: dict[str, list[tuple[int, dict]]] = defaultdict(list)
        for line_no, row in enumerate(source_rows):
            groups[context_hash(row.get("context", ""))].append((line_no, row))
        candidates = [
            (len(rows[0][1].get("context", "")), group_hash, rows)
            for group_hash, rows in groups.items()
            if group_hash not in excluded[domain] and len(rows) >= args.samples_per_domain
        ]
        if not candidates:
            raise RuntimeError(f"No unused {domain} context has enough questions")
        source_chars, original_hash, rows = min(candidates, key=lambda item: (item[0], item[1]))
        chosen = rows[:args.samples_per_domain]
        full_context = chosen[0][1].get("context", "")
        chunks = split_chunks(full_context, args.chunk_chars, args.chunk_overlap)
        questions = [row.get("input", "") for _, row in chosen]
        selected_indices, per_question = select_chunk_indices(chunks, questions, args.top_k)
        condensed_context = "\n\n".join(
            f"[SOURCE CHUNK {index}]\n{chunks[index]}" for index in selected_indices
        )
        condensed_hash = context_hash(condensed_context)
        for question_index, (source_line_no, source_row) in enumerate(chosen):
            item = dict(source_row)
            item["context"] = condensed_context
            item["dataset"] = domain
            item["source_dataset"] = domain
            item["source_line_no"] = source_line_no
            item["source_context_hash"] = condensed_hash
            item["original_source_context_hash"] = original_hash
            item["source_context_chars"] = source_chars
            item["condensed_context_chars"] = len(condensed_context)
            item["condensed_chunk_indices"] = selected_indices
            item["question_chunk_indices"] = per_question[question_index]
            item["condensation_method"] = "multiquery_tfidf_top_chunks_with_neighbors"
            item["condensation_uses_reference_answers"] = False
            item["_subset_line_no"] = len(output_rows)
            output_rows.append(item)
        manifest_rows.append({
            "domain": domain,
            "original_context_hash": original_hash,
            "condensed_context_hash": condensed_hash,
            "source_context_chars": source_chars,
            "condensed_context_chars": len(condensed_context),
            "compression_ratio": 1 - len(condensed_context) / max(1, source_chars),
            "source_lines": [line_no for line_no, _ in chosen],
            "source_chunks": len(chunks),
            "selected_chunks": len(selected_indices),
        })

    output_path = ROOT_DIR / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as output:
        for row in output_rows:
            output.write(json.dumps(row, ensure_ascii=False) + "\n")

    lines = [
        "# Multi-Query Condensed Cross-Domain Subset\n\n",
        f"- Samples: {len(output_rows)}\n",
        f"- Domains: {args.domains}\n",
        "- Context selection uses questions only; reference answers are not read.\n",
        "- One condensed context and one knowledge graph are shared by all questions in each domain document.\n\n",
        "| Domain | Original chars | Condensed chars | Reduction | Source chunks | Selected chunks | Source lines |\n",
        "|---|---:|---:|---:|---:|---:|---|\n",
    ]
    for row in manifest_rows:
        lines.append(
            f"| {row['domain']} | {row['source_context_chars']} | {row['condensed_context_chars']} | "
            f"{row['compression_ratio'] * 100:.2f}% | {row['source_chunks']} | "
            f"{row['selected_chunks']} | {row['source_lines']} |\n"
        )
    manifest_path = ROOT_DIR / args.manifest
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("".join(lines), encoding="utf-8")
    print(f"Saved {len(output_rows)} samples to {output_path}")
    print(f"Saved manifest to {manifest_path}")


if __name__ == "__main__":
    main()
