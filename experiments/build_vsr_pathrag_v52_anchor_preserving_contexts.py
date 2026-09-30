import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from experiments.build_multiquery_condensed_subset import split_chunks


STOP_TERMS = {
    "according",
    "and",
    "book",
    "context",
    "describe",
    "did",
    "does",
    "for",
    "from",
    "has",
    "have",
    "how",
    "into",
    "its",
    "of",
    "role",
    "stay",
    "that",
    "the",
    "their",
    "this",
    "to",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "who",
    "why",
    "with",
}


def context_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def definition_subject(question: str) -> str:
    lowered = " ".join(question.lower().split()).rstrip("?.")
    patterns = (
        r"\bdefinition of (.+?)(?: according to| in the|$)",
        r"\bhow is (.+?) defined(?: according to| in the|$)",
    )
    for pattern in patterns:
        match = re.search(pattern, lowered)
        if match:
            subject = match.group(1).strip()
            if 0 < len(subject.split()) <= 6:
                return subject
    return ""


def definition_score(chunk: str, subject: str) -> float:
    if not subject:
        return 0.0
    lowered = re.sub(r"[ \t]+", " ", chunk.lower())
    escaped = re.escape(subject)
    score = 0.0
    if re.search(rf"\bwhat\s+is\s+{escaped}\b", lowered):
        score += 8.0
    if re.search(
        rf"(?:^|[.!?\n:#])\s*{escaped}\s+is\b",
        lowered,
    ):
        score += 7.0
    if re.search(rf"\bdefine[sd]?\s+{escaped}\b", lowered):
        score += 6.0
    if "definition" in lowered and subject in lowered:
        score += 3.0
    return score


def expand_with_neighbors(
    indices: set[int],
    chunk_count: int,
) -> set[int]:
    expanded = set()
    for index in indices:
        expanded.add(index)
        if index > 0:
            expanded.add(index - 1)
        if index + 1 < chunk_count:
            expanded.add(index + 1)
    return expanded


def select_anchor_centers(
    chunks: list[str],
    question: str,
    max_rare_terms: int = 5,
) -> list[int]:
    chunk_tokens = [tokens(chunk) for chunk in chunks]
    chunk_sets = [set(items) for items in chunk_tokens]
    document_frequency = Counter()
    for item_set in chunk_sets:
        document_frequency.update(item_set)

    query_terms = {
        term
        for term in tokens(question)
        if term not in STOP_TERMS
        and len(term) > 2
        and document_frequency.get(term, 0) > 0
    }
    proper_terms = {
        match.group(0).lower()
        for match in re.finditer(r"\b(?:[A-Z][a-z]{2,}|[A-Z]{2,})\b", question)
        if match.group(0).lower() not in STOP_TERMS
    }
    rare_limit = max(4, int(len(chunks) * 0.08))
    rare_terms = sorted(
        query_terms,
        key=lambda term: (
            term not in proper_terms,
            document_frequency[term],
            term,
        ),
    )
    selected_terms = [
        term
        for term in rare_terms
        if term in proper_terms
        or document_frequency[term] <= rare_limit
    ][:max_rare_terms]

    centers: set[int] = set()
    for term in selected_terms:
        candidates = [
            index
            for index, item_set in enumerate(chunk_sets)
            if term in item_set
        ]
        if not candidates:
            continue
        centers.add(
            max(
                candidates,
                key=lambda index: (
                    len(query_terms & chunk_sets[index]),
                    chunk_tokens[index].count(term),
                    -len(chunk_tokens[index]),
                    -index,
                ),
            )
        )

    subject = definition_subject(question)
    if subject:
        scores = [
            definition_score(chunk, subject)
            for chunk in chunks
        ]
        best_index = max(
            range(len(scores)),
            key=lambda index: (scores[index], -index),
        )
        if scores[best_index] > 0:
            centers.add(best_index)
    return sorted(centers)


def anchor_requirements(
    chunks: list[str],
    question: str,
) -> dict[str, Any]:
    centers = select_anchor_centers(chunks, question)
    proper_terms = {
        match.group(0).lower()
        for match in re.finditer(r"\b(?:[A-Z][a-z]{2,}|[A-Z]{2,})\b", question)
        if match.group(0).lower() not in STOP_TERMS
    }
    frequencies = Counter()
    for chunk in chunks:
        frequencies.update(set(tokens(chunk)))
    rare_limit = max(4, int(len(chunks) * 0.08))
    rare_named_terms = sorted(
        term
        for term in proper_terms
        if 0 < frequencies[term] <= rare_limit
    )
    return {
        "rare_named_terms": rare_named_terms,
        "definition_subject": definition_subject(question),
        "anchor_centers": centers,
    }


def audit_selection(
    chunks: list[str],
    selected_indices: set[int],
    question: str,
) -> dict[str, Any]:
    requirements = anchor_requirements(chunks, question)
    selected_text = "\n".join(
        chunks[index] for index in sorted(selected_indices)
    ).lower()
    missing_named_terms = [
        term
        for term in requirements["rare_named_terms"]
        if term not in selected_text
    ]
    subject = requirements["definition_subject"]
    definition_preserved = True
    if subject:
        definition_preserved = any(
            definition_score(chunks[index], subject) >= 6.0
            for index in selected_indices
        )
    passed = not missing_named_terms and definition_preserved
    return {
        **requirements,
        "missing_named_terms": missing_named_terms,
        "definition_preserved": definition_preserved,
        "passed": passed,
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as source:
        return [
            json.loads(line)
            for line in source
            if line.strip()
        ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dataset",
        default=(
            "datasets/UltraDomain/"
            "v46_zero_overlap_validation_30.jsonl"
        ),
    )
    parser.add_argument(
        "--output",
        default=(
            "datasets/UltraDomain/"
            "vsr_v52_anchor_preserving_validation_30.jsonl"
        ),
    )
    parser.add_argument(
        "--manifest",
        default=(
            "results/"
            "vsr_pathrag_v52_anchor_preserving_context_manifest.json"
        ),
    )
    parser.add_argument(
        "--summary",
        default=(
            "results/"
            "vsr_pathrag_v52_anchor_preserving_context_manifest.md"
        ),
    )
    parser.add_argument("--chunk-chars", type=int, default=1800)
    parser.add_argument("--chunk-overlap", type=int, default=220)
    args = parser.parse_args()

    source_rows = load_jsonl(ROOT_DIR / args.source_dataset)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in source_rows:
        grouped[
            (
                str(row["source_dataset"]),
                str(row["original_source_context_hash"]),
            )
        ].append(row)

    output_rows: list[dict[str, Any]] = []
    group_reports = []
    question_reports = []
    source_cache: dict[str, list[dict[str, Any]]] = {}
    for (domain, original_hash), group_rows in grouped.items():
        if domain not in source_cache:
            source_cache[domain] = load_jsonl(
                ROOT_DIR / f"datasets/UltraDomain/{domain}.jsonl"
            )
        original_row = source_cache[domain][
            int(group_rows[0]["source_line_no"])
        ]
        original_context = str(original_row["context"])
        if context_hash(original_context) != original_hash:
            raise ValueError(
                f"source hash mismatch for {domain}: {original_hash}"
            )
        chunks = split_chunks(
            original_context,
            args.chunk_chars,
            args.chunk_overlap,
        )
        original_indices = set(
            int(index)
            for index in group_rows[0]["condensed_chunk_indices"]
        )
        repaired_indices = set(original_indices)
        per_question_indices = {}
        for row in group_rows:
            question = str(row["input"])
            before = audit_selection(
                chunks,
                original_indices,
                question,
            )
            centers = set()
            if not before["passed"]:
                candidate_centers = select_anchor_centers(
                    chunks,
                    question,
                )
                missing_terms = set(before["missing_named_terms"])
                for index in candidate_centers:
                    if missing_terms & set(tokens(chunks[index])):
                        centers.add(index)
                subject = str(before["definition_subject"])
                if subject and not before["definition_preserved"]:
                    definition_scores = [
                        definition_score(chunk, subject)
                        for chunk in chunks
                    ]
                    best_index = max(
                        range(len(definition_scores)),
                        key=lambda index: (
                            definition_scores[index],
                            -index,
                        ),
                    )
                    if definition_scores[best_index] >= 6.0:
                        centers.add(best_index)
            additions = expand_with_neighbors(centers, len(chunks))
            repaired_indices.update(additions)
            per_question_indices[int(row["_subset_line_no"])] = sorted(
                set(int(index) for index in row["question_chunk_indices"])
                | additions
            )

        condensed_context = "\n\n".join(
            f"[SOURCE CHUNK {index}]\n{chunks[index]}"
            for index in sorted(repaired_indices)
        )
        repaired_hash = context_hash(condensed_context)
        for row in group_rows:
            line_no = int(row["_subset_line_no"])
            question = str(row["input"])
            before = audit_selection(
                chunks,
                original_indices,
                question,
            )
            after = audit_selection(
                chunks,
                repaired_indices,
                question,
            )
            question_reports.append(
                {
                    "line_no": line_no,
                    "dataset": domain,
                    "question": question,
                    "before": before,
                    "after": after,
                }
            )
            item = dict(row)
            item["context"] = condensed_context
            item["source_context_hash"] = repaired_hash
            item["condensed_context_chars"] = len(condensed_context)
            item["condensed_chunk_indices"] = sorted(repaired_indices)
            item["question_chunk_indices"] = per_question_indices[line_no]
            item["anchor_preserving_added_chunk_indices"] = sorted(
                repaired_indices - original_indices
            )
            item["condensation_method"] = (
                "multiquery_tfidf_plus_rare_anchor_and_definition_preservation"
            )
            item["condensation_uses_reference_answers"] = False
            output_rows.append(item)

        group_reports.append(
            {
                "dataset": domain,
                "original_source_context_hash": original_hash,
                "previous_condensed_context_hash": str(
                    group_rows[0]["source_context_hash"]
                ),
                "repaired_condensed_context_hash": repaired_hash,
                "previous_chunk_count": len(original_indices),
                "repaired_chunk_count": len(repaired_indices),
                "added_chunk_count": len(
                    repaired_indices - original_indices
                ),
                "previous_context_chars": int(
                    group_rows[0]["condensed_context_chars"]
                ),
                "repaired_context_chars": len(condensed_context),
            }
        )

    output_rows.sort(key=lambda row: int(row["_subset_line_no"]))
    write_jsonl(ROOT_DIR / args.output, output_rows)
    before_failures = [
        row for row in question_reports if not row["before"]["passed"]
    ]
    after_failures = [
        row for row in question_reports if not row["after"]["passed"]
    ]
    manifest = {
        "protocol": "anchor_preserving_zero_overlap_context_repair",
        "source_dataset": args.source_dataset,
        "output_dataset": args.output,
        "sample_count": len(output_rows),
        "context_count": len(group_reports),
        "before_answerability_failure_count": len(before_failures),
        "after_answerability_failure_count": len(after_failures),
        "reference_answer_used": False,
        "judge_used": False,
        "ablation": False,
        "groups": group_reports,
        "questions": question_reports,
    }
    manifest_path = ROOT_DIR / args.manifest
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# VSR-PathRAG v52 Anchor-Preserving Context Manifest",
        "",
        f"- Samples: {len(output_rows)}",
        f"- Shared contexts: {len(group_reports)}",
        (
            "- Answerability audit failures before repair: "
            f"{len(before_failures)}"
        ),
        (
            "- Answerability audit failures after repair: "
            f"{len(after_failures)}"
        ),
        "- Selection inputs: questions and original source text only",
        "- Reference answers used: no",
        "- External judge used: no",
        "- Formal ablation: false",
        "",
        "| Domain | Previous chunks | Repaired chunks | Added | "
        "Previous chars | Repaired chars |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in group_reports:
        lines.append(
            f"| {row['dataset']} | {row['previous_chunk_count']} | "
            f"{row['repaired_chunk_count']} | "
            f"{row['added_chunk_count']} | "
            f"{row['previous_context_chars']} | "
            f"{row['repaired_context_chars']} |"
        )
    if before_failures:
        lines.extend(["", "## Repaired Questions", ""])
        for row in before_failures:
            after_status = "pass" if row["after"]["passed"] else "fail"
            lines.append(
                f"- Line {row['line_no']} ({row['dataset']}): "
                f"{row['question']} -> {after_status}"
            )
    lines.append("")
    (ROOT_DIR / args.summary).write_text(
        "\n".join(lines),
        encoding="utf-8",
    )
    print(
        f"samples={len(output_rows)} contexts={len(group_reports)} "
        f"before_failures={len(before_failures)} "
        f"after_failures={len(after_failures)}"
    )


if __name__ == "__main__":
    main()
