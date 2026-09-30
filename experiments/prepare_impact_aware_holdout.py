#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build a new source-level holdout for impact-aware escalation validation.

The selector is deliberately conservative: it excludes any question/context
already visible in prior JSONL artifacts, then chooses one question per source
context and records the original source line. The same fixed context window is
given to every policy in the later comparison.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "datasets" / "UltraDomain"
DOMAINS = ("biology", "history", "legal", "physics")


def normalize(value: object) -> str:
    return re.sub(r"\s+", " ", str(value or "")).strip().lower()


def stable_hash(value: object) -> str:
    return hashlib.sha1(normalize(value).encode("utf-8")).hexdigest()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle):
            if not line.strip():
                continue
            row = json.loads(line)
            row["_file_line_no"] = line_no
            rows.append(row)
    return rows


def collect_used_keys() -> tuple[set[str], set[str], set[tuple[str, int]]]:
    used_questions: set[str] = set()
    used_contexts: set[str] = set()
    used_source_lines: set[tuple[str, int]] = set()

    paths = []
    for path in (ROOT / "results").rglob("*.jsonl"):
        if "snapshots" in path.parts:
            continue
        # Keep the overlap audit focused on experiment artifacts that can
        # actually define a prior holdout, avoiding bulky replay caches.
        name = path.name.lower()
        if any(token in name for token in (
            "sage", "vsr_pathrag", "mse_pathrag", "enhanced_ablation",
            "adaptive_pathrag", "pairwise", "locked", "holdout", "validation",
        )):
            paths.append(path)
    derived_names = {
        "mix.jsonl",
        "biology.jsonl",
        "history.jsonl",
        "legal.jsonl",
        "physics.jsonl",
        "agriculture.jsonl",
    }
    paths.extend(
        path
        for path in (ROOT / "datasets" / "UltraDomain").glob("*.jsonl")
        if path.name.lower() not in derived_names
    )
    for path in paths:
        try:
            rows = load_jsonl(path)
        except (OSError, json.JSONDecodeError):
            continue
        for row in rows:
            question = row.get("question") or row.get("input") or row.get("_question")
            context = row.get("context") or row.get("source_context") or row.get("full_context")
            if question:
                used_questions.add(stable_hash(question))
            for hash_value in (
                row.get("source_context_hash"),
                row.get("context_hash"),
            ):
                if hash_value:
                    used_contexts.add(str(hash_value).strip().lower())
            if isinstance(context, str) and context.strip():
                used_contexts.add(stable_hash(context))
                used_contexts.add(stable_hash(context)[:12])
            dataset = normalize(row.get("source_dataset") or row.get("dataset") or row.get("label"))
            source_line = row.get("source_line_no")
            if dataset and source_line is not None:
                try:
                    used_source_lines.add((dataset, int(source_line)))
                except (TypeError, ValueError):
                    pass
    return used_questions, used_contexts, used_source_lines


def answer_positions(context: str, answers: list[object]) -> list[int]:
    positions: list[int] = []
    lowered = context.lower()
    for answer in answers:
        text = normalize(answer)
        if text:
            index = lowered.find(text)
            if index >= 0:
                positions.append(index)
    return positions


def question_tokens(question: str) -> list[str]:
    stopwords = {
        "about", "according", "after", "also", "and", "are", "because", "been",
        "between", "could", "did", "does", "during", "for", "from", "had", "has",
        "have", "how", "into", "its", "main", "more", "most", "not", "other",
        "over", "that", "the", "their", "these", "this", "through", "under",
        "was", "were", "what", "when", "where", "which", "while", "who", "why", "with",
    }
    cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in question)
    return [token for token in cleaned.split() if len(token) >= 4 and token not in stopwords]


def make_window(
    context: str,
    question: str,
    answers: list[object],
    max_chars: int,
    margin_chars: int,
) -> tuple[str, int, int, str, int]:
    if len(context) <= max_chars:
        return context, 0, len(context), "full_context_under_limit", 0

    positions = answer_positions(context, answers)
    if positions:
        start = max(0, min(positions) - margin_chars)
        end = min(len(context), max(positions) + margin_chars)
        if end - start > max_chars:
            center = (min(positions) + max(positions)) // 2
            start = max(0, center - max_chars // 2)
            end = min(len(context), start + max_chars)
            start = max(0, end - max_chars)
        return context[start:end], start, end, "answer_matched_window", len(positions)

    tokens = question_tokens(question)
    best_start = 0
    best_score = -1
    stride = max(10_000, max_chars // 4)
    for start in range(0, max(1, len(context) - max_chars + 1), stride):
        window = context[start : start + max_chars].lower()
        score = sum(window.count(token) for token in tokens)
        if score > best_score:
            best_score = score
            best_start = start
    end = min(len(context), best_start + max_chars)
    return context[best_start:end], best_start, end, "question_density_window", 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-per-domain", type=int, default=12)
    parser.add_argument(
        "--domain-samples",
        default="",
        help="Optional comma-separated domain=count overrides, e.g. biology=7,history=12",
    )
    parser.add_argument("--max-window-chars", type=int, default=36_000)
    parser.add_argument("--margin-chars", type=int, default=12_000)
    parser.add_argument(
        "--output",
        default="datasets/UltraDomain/impact_aware_locked_new48_windowed.jsonl",
    )
    parser.add_argument(
        "--manifest-md",
        default="results/impact_aware_validation/IMPACT_AWARE_NEW48_MANIFEST.md",
    )
    parser.add_argument(
        "--manifest-json",
        default="results/impact_aware_validation/impact_aware_new48_manifest.json",
    )
    args = parser.parse_args()

    domain_targets = {domain: args.samples_per_domain for domain in DOMAINS}
    if args.domain_samples.strip():
        for item in args.domain_samples.split(","):
            name, separator, value = item.partition("=")
            domain = normalize(name)
            if not separator or domain not in domain_targets:
                raise ValueError(f"Invalid --domain-samples entry: {item}")
            domain_targets[domain] = int(value)

    used_questions, used_contexts, used_source_lines = collect_used_keys()
    selected: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    domain_counts: Counter[str] = Counter()
    excluded = Counter()

    for domain in DOMAINS:
        source_rows = load_jsonl(SOURCE_DIR / f"{domain}.jsonl")
        candidates: list[dict[str, Any]] = []
        seen_contexts: set[str] = set()
        for row in source_rows:
            question = str(row.get("input") or "")
            context = str(row.get("context") or "")
            source_line = int(row["_file_line_no"])
            context_key = stable_hash(context)
            context_short_key = context_key[:12]
            line_key = (domain, source_line)
            if stable_hash(question) in used_questions:
                excluded["question_overlap"] += 1
                continue
            if context_key in used_contexts or context_short_key in used_contexts:
                excluded["context_overlap"] += 1
                continue
            if line_key in used_source_lines:
                excluded["source_line_overlap"] += 1
                continue
            if context_key in seen_contexts:
                excluded["duplicate_context_within_holdout"] += 1
                continue
            candidates.append(row)
            seen_contexts.add(context_key)

        candidates.sort(key=lambda row: (len(str(row.get("context") or "")), int(row["_file_line_no"])))
        target_count = domain_targets[domain]
        chosen = candidates[:target_count]
        if len(chosen) != target_count:
            raise RuntimeError(
                f"{domain}: requested {target_count}, found {len(chosen)} clean contexts"
            )

        for row in chosen:
            context = str(row.get("context") or "")
            question = str(row.get("input") or "")
            window, start, end, reason, matched_answers = make_window(
                context,
                question,
                row.get("answers") or [],
                args.max_window_chars,
                args.margin_chars,
            )
            item = dict(row)
            item.pop("_file_line_no", None)
            item["dataset"] = domain
            item["label"] = domain
            item["source_dataset"] = domain
            item["source_line_no"] = int(row["_file_line_no"])
            item["source_context_hash"] = stable_hash(context)[:12]
            item["source_context_chars"] = len(context)
            item["window_start"] = start
            item["window_end"] = end
            item["window_chars"] = len(window)
            item["window_reason"] = reason
            item["window_answer_matches"] = matched_answers
            item["context"] = window
            selected.append(item)
            manifest_rows.append({
                "domain": domain,
                "source_line_no": int(row["_file_line_no"]),
                "source_context_hash": stable_hash(context)[:12],
                "source_context_chars": len(context),
                "window_start": start,
                "window_end": end,
                "window_chars": len(window),
                "window_reason": reason,
                "window_answer_matches": matched_answers,
                "question": question,
            })
            domain_counts[domain] += 1

    for index, row in enumerate(selected):
        row["_subset_line_no"] = index

    output_path = ROOT / args.output
    manifest_md = ROOT / args.manifest_md
    manifest_json = ROOT / args.manifest_json
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_md.parent.mkdir(parents=True, exist_ok=True)
    manifest_json.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in selected),
        encoding="utf-8",
    )
    manifest_json.write_text(
        json.dumps(
            {
                "protocol": "impact_aware_new48_source_holdout",
                "samples": len(selected),
                "domains": list(DOMAINS),
                "samples_per_domain": args.samples_per_domain,
                "domain_targets": domain_targets,
                "max_window_chars": args.max_window_chars,
                "margin_chars": args.margin_chars,
                "selection": "one clean question per source context after repository-wide question/context/source-line exclusion",
                "excluded_counts": dict(excluded),
                "domain_counts": dict(domain_counts),
                "rows": manifest_rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    lines = [
        "# Impact-Aware New48 Locked Source Holdout",
        "",
        f"- Samples: {len(selected)}",
        f"- Domains: {', '.join(DOMAINS)}",
        f"- Default samples per domain: {args.samples_per_domain}",
        f"- Domain targets: {domain_targets}",
        f"- Max window chars: {args.max_window_chars}",
        "- Selection excludes repository-wide question, context, and source-line overlaps.",
        "- One question is selected per source context; all policies receive the same fixed window.",
        "- Answer-matched windows are preprocessing only and are frozen before model calls.",
        "",
        "## Domain Counts",
        "",
        "| Domain | Samples |",
        "|---|---:|",
    ]
    for domain in DOMAINS:
        lines.append(f"| {domain} | {domain_counts[domain]} |")
    lines.extend([
        "",
        "## Exclusion Counts",
        "",
        "| Check | Excluded |",
        "|---|---:|",
    ])
    for name, count in sorted(excluded.items()):
        lines.append(f"| {name} | {count} |")
    lines.extend([
        "",
        "## Selected Rows",
        "",
        "| Subset line | Domain | Source line | Context hash | Window chars | Question |",
        "|---:|---|---:|---|---:|---|",
    ])
    for subset_line, row in enumerate(manifest_rows):
        lines.append(
            f"| {subset_line} | {row['domain']} | {row['source_line_no']} | "
            f"`{row['source_context_hash']}` | {row['window_chars']} | "
            f"{row['question'].replace('|', '/')} |"
        )
    manifest_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {len(selected)} rows to {output_path}")
    print(f"Saved {manifest_md}")
    print(f"Saved {manifest_json}")
    print(json.dumps({"domain_counts": dict(domain_counts), "excluded": dict(excluded)}))


if __name__ == "__main__":
    main()
