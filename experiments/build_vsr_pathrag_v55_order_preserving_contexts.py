import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from experiments.build_multiquery_condensed_subset import split_chunks
from experiments.run_ultradomain_subset import context_hash


ORDINALS = ("FIRST", "SECOND", "THIRD", "FOURTH")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as source:
        return [json.loads(line) for line in source if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, ensure_ascii=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def is_ordered_application_question(question: str) -> bool:
    lowered = " ".join(str(question).lower().split())
    return bool(
        "proceeds" in lowered
        and (
            "applied" in lowered
            or "application" in lowered
            or "order" in lowered
            or "priority" in lowered
        )
    )


def ordered_clause_span(
    chunks: list[str],
    question: str,
) -> set[int]:
    if not is_ordered_application_question(question):
        return set()
    start = None
    for index, chunk in enumerate(chunks):
        lowered = chunk.lower()
        if (
            "following order of application" in lowered
            and re.search(r"\bFIRST\b", chunk)
        ):
            start = index
            break
    if start is None:
        return set()

    selected = set()
    seen = set()
    collected = []
    for index in range(start, min(len(chunks), start + 8)):
        selected.add(index)
        collected.append(chunks[index])
        seen.update(
            ordinal
            for ordinal in ORDINALS
            if re.search(rf"\b{ordinal}\b", chunks[index])
        )
        if set(ORDINALS).issubset(seen):
            joined = "\n".join(collected)
            final_clause = joined[joined.rfind("FOURTH") :]
            if re.search(r"[.;](?:\s|$)", final_clause):
                break
    return selected if set(ORDINALS).issubset(seen) else set()


def audit_order_preservation(
    chunks: list[str],
    selected_indices: set[int],
    question: str,
) -> dict[str, Any]:
    required_span = ordered_clause_span(chunks, question)
    if not is_ordered_application_question(question):
        return {
            "required": False,
            "required_indices": [],
            "missing_indices": [],
            "ordinals_present": [],
            "passed": True,
        }
    selected_text = "\n".join(
        chunks[index] for index in sorted(selected_indices)
    )
    ordinals_present = [
        ordinal
        for ordinal in ORDINALS
        if re.search(rf"\b{ordinal}\b", selected_text)
    ]
    missing_indices = sorted(required_span - selected_indices)
    return {
        "required": True,
        "required_indices": sorted(required_span),
        "missing_indices": missing_indices,
        "ordinals_present": ordinals_present,
        "passed": bool(
            required_span
            and not missing_indices
            and len(ordinals_present) == len(ORDINALS)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-dataset",
        default=(
            "datasets/UltraDomain/"
            "vsr_v52_anchor_preserving_validation_30.jsonl"
        ),
    )
    parser.add_argument(
        "--output",
        default=(
            "datasets/UltraDomain/"
            "vsr_v55_order_preserving_validation_30.jsonl"
        ),
    )
    parser.add_argument(
        "--manifest",
        default=(
            "results/"
            "vsr_pathrag_v55_order_preserving_context_manifest.json"
        ),
    )
    parser.add_argument("--chunk-chars", type=int, default=1800)
    parser.add_argument("--chunk-overlap", type=int, default=220)
    args = parser.parse_args()

    rows = load_jsonl(ROOT_DIR / args.source_dataset)
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        key = (
            str(row["dataset"]),
            str(row["original_source_context_hash"]),
        )
        grouped.setdefault(key, []).append(row)

    output_rows = []
    group_reports = []
    question_reports = []
    for (domain, original_hash), group_rows in grouped.items():
        source_line_no = int(group_rows[0]["source_line_no"])
        source_rows = load_jsonl(
            ROOT_DIR / f"datasets/UltraDomain/{domain}.jsonl"
        )
        original_context = str(source_rows[source_line_no]["context"])
        if context_hash(original_context) != original_hash:
            raise ValueError(
                f"source hash mismatch for {domain}: {original_hash}"
            )
        chunks = split_chunks(
            original_context,
            args.chunk_chars,
            args.chunk_overlap,
        )
        previous_indices = {
            int(index)
            for index in group_rows[0]["condensed_chunk_indices"]
        }
        repaired_indices = set(previous_indices)
        per_question_spans = {}
        for row in group_rows:
            line_no = int(row["_subset_line_no"])
            question = str(row["input"])
            span = ordered_clause_span(chunks, question)
            repaired_indices.update(span)
            per_question_spans[line_no] = span

        added_indices = sorted(repaired_indices - previous_indices)
        if added_indices:
            condensed_context = "\n\n".join(
                f"[SOURCE CHUNK {index}]\n{chunks[index]}"
                for index in sorted(repaired_indices)
            )
        else:
            condensed_context = str(group_rows[0]["context"])
        repaired_hash = context_hash(condensed_context)

        for row in group_rows:
            line_no = int(row["_subset_line_no"])
            question = str(row["input"])
            before = audit_order_preservation(
                chunks,
                previous_indices,
                question,
            )
            after = audit_order_preservation(
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
            item["question_chunk_indices"] = sorted(
                {
                    int(index)
                    for index in row["question_chunk_indices"]
                }
                | per_question_spans[line_no]
            )
            item["order_preserving_added_chunk_indices"] = added_indices
            item["condensation_method"] = (
                "anchor_preserving_plus_ordered_clause_continuity"
            )
            item["condensation_uses_reference_answers"] = False
            output_rows.append(item)

        group_reports.append(
            {
                "dataset": domain,
                "previous_context_hash": str(
                    group_rows[0]["source_context_hash"]
                ),
                "repaired_context_hash": repaired_hash,
                "previous_chunk_count": len(previous_indices),
                "repaired_chunk_count": len(repaired_indices),
                "added_chunk_indices": added_indices,
                "previous_context_chars": len(
                    str(group_rows[0]["context"])
                ),
                "repaired_context_chars": len(condensed_context),
            }
        )

    output_rows.sort(key=lambda row: int(row["_subset_line_no"]))
    write_jsonl(ROOT_DIR / args.output, output_rows)
    required = [
        report
        for report in question_reports
        if report["before"]["required"]
    ]
    manifest = {
        "protocol": "ordered_clause_continuity_preservation",
        "source_dataset": args.source_dataset,
        "output_dataset": args.output,
        "sample_count": len(output_rows),
        "context_count": len(group_reports),
        "required_question_count": len(required),
        "before_failure_count": sum(
            not report["before"]["passed"] for report in required
        ),
        "after_failure_count": sum(
            not report["after"]["passed"] for report in required
        ),
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
    print(
        f"samples={len(output_rows)} required={len(required)} "
        f"before_failures={manifest['before_failure_count']} "
        f"after_failures={manifest['after_failure_count']}"
    )


if __name__ == "__main__":
    main()
