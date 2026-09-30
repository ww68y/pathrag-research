from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT_DIR / "datasets" / "UltraDomain"

DOMAINS = ["legal", "biology", "history", "physics", "agriculture"]
DEFAULT_EXCLUDE_DATASETS = [
    "datasets/UltraDomain/legal_fullcontext_20.jsonl",
    "datasets/UltraDomain/biology_fullcontext_50.jsonl",
    "datasets/UltraDomain/domain_expansion_windowed_50.jsonl",
    "datasets/UltraDomain/v40_frozen_lazy_heldout_12.jsonl",
    "datasets/UltraDomain/v42_verifier_lazy_heldout_12.jsonl",
    "datasets/UltraDomain/v46_zero_overlap_validation_30.jsonl",
    "datasets/UltraDomain/vsr_v52_anchor_preserving_validation_30.jsonl",
    "datasets/UltraDomain/vsr_v55_order_preserving_validation_30.jsonl",
    "datasets/UltraDomain/crossdomain_multiquery_heldout_12.jsonl",
    "datasets/UltraDomain/crossdomain_multiquery_condensed_12.jsonl",
]

STOPWORDS = {
    "about",
    "according",
    "after",
    "also",
    "among",
    "and",
    "are",
    "because",
    "been",
    "between",
    "could",
    "did",
    "does",
    "during",
    "for",
    "from",
    "had",
    "has",
    "have",
    "how",
    "into",
    "its",
    "main",
    "more",
    "most",
    "not",
    "other",
    "over",
    "primary",
    "that",
    "the",
    "their",
    "these",
    "this",
    "through",
    "under",
    "was",
    "were",
    "what",
    "when",
    "where",
    "which",
    "while",
    "who",
    "why",
    "with",
}


def clean_text(value: object) -> str:
    return " ".join(str(value or "").split())


def context_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as source:
        for line_no, line in enumerate(source):
            if not line.strip():
                continue
            row = json.loads(line)
            row["_source_line_no"] = line_no
            rows.append(row)
    return rows


def used_keys(paths: list[str]) -> tuple[set[tuple[str, int]], set[tuple[str, str]]]:
    line_keys: set[tuple[str, int]] = set()
    context_keys: set[tuple[str, str]] = set()
    for relative in paths:
        path = ROOT_DIR / relative
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                row = json.loads(line)
                domain = clean_text(row.get("source_dataset") or row.get("dataset") or row.get("label")).lower()
                if domain not in DOMAINS:
                    continue
                source_line = row.get("source_line_no")
                if source_line is not None:
                    try:
                        line_keys.add((domain, int(source_line)))
                    except (TypeError, ValueError):
                        pass
                source_hash = clean_text(row.get("source_context_hash"))
                if source_hash:
                    context_keys.add((domain, source_hash))
                elif row.get("context"):
                    context_keys.add((domain, context_hash(str(row.get("context")))))
    return line_keys, context_keys


def answer_positions(context: str, samples: list[dict[str, Any]]) -> list[int]:
    positions: list[int] = []
    lowered = context.lower()
    for sample in samples:
        for answer in sample.get("answers", []):
            answer_text = clean_text(answer)
            if not answer_text:
                continue
            index = lowered.find(answer_text.lower())
            if index >= 0:
                positions.append(index)
                break
    return positions


def question_tokens(samples: list[dict[str, Any]]) -> list[str]:
    tokens: list[str] = []
    for sample in samples:
        cleaned = "".join(ch.lower() if ch.isalnum() else " " for ch in str(sample.get("input", "")))
        for token in cleaned.split():
            if len(token) < 4 or token in STOPWORDS:
                continue
            tokens.append(token)
    # Keep token multiplicity across questions, but cap repeated generic domain terms.
    counts = Counter(tokens)
    ranked = sorted(counts, key=lambda token: (-counts[token], token))
    expanded: list[str] = []
    for token in ranked[:60]:
        expanded.extend([token] * min(3, counts[token]))
    return expanded


def question_window_position(context: str, samples: list[dict[str, Any]], max_window_chars: int) -> tuple[int, int]:
    if len(context) <= max_window_chars:
        return 0, len(context)
    tokens = question_tokens(samples)
    if not tokens:
        return 0, max_window_chars
    lowered = context.lower()
    stride = max(10_000, max_window_chars // 4)
    best_start = 0
    best_score = -1
    for start in range(0, max(1, len(context) - max_window_chars + 1), stride):
        window = lowered[start : start + max_window_chars]
        score = sum(window.count(token) for token in tokens)
        if score > best_score:
            best_score = score
            best_start = start
    if best_start + max_window_chars < len(context):
        # Re-center around the highest-scoring local token hit when possible.
        best_window = lowered[best_start : best_start + max_window_chars]
        hit_positions = [best_window.find(token) for token in set(tokens) if best_window.find(token) >= 0]
        if hit_positions:
            center = best_start + sorted(hit_positions)[len(hit_positions) // 2]
            best_start = max(0, min(len(context) - max_window_chars, center - max_window_chars // 2))
    return best_start, min(len(context), best_start + max_window_chars)


def make_window(
    context: str,
    samples: list[dict[str, Any]],
    max_window_chars: int,
    margin_chars: int,
) -> tuple[str, int, int, str, int]:
    positions = answer_positions(context, samples)
    if positions:
        start = max(0, min(positions) - margin_chars)
        end = min(len(context), max(positions) + margin_chars)
        if end - start > max_window_chars:
            center = (min(positions) + max(positions)) // 2
            start = max(0, center - max_window_chars // 2)
            end = min(len(context), start + max_window_chars)
            start = max(0, end - max_window_chars)
        reason = "answer_matched_window"
    else:
        start, end = question_window_position(context, samples, max_window_chars)
        reason = "question_density_window_no_exact_answer_match"
    return context[start:end], start, end, reason, len(positions)


def question_family(question: str) -> str:
    lowered = clean_text(question).lower()
    if any(token in lowered for token in ["compare", "contrast", "differ", "difference"]):
        return "comparison"
    if any(token in lowered for token in ["steps", "procedure", "process", "how are", "how is"]):
        return "procedure_or_process"
    if any(token in lowered for token in ["why", "reason", "purpose", "significance", "impact"]):
        return "why_or_significance"
    if any(token in lowered for token in ["what are", "list", "factors", "challenges", "examples"]):
        return "enumeration"
    if any(token in lowered for token in ["how does", "how do", "affect", "influence", "role"]):
        return "mechanism"
    return "general"


def select_domain(
    domain: str,
    samples_per_domain: int,
    max_samples_per_context: int,
    max_window_chars: int,
    margin_chars: int,
    used_lines: set[tuple[str, int]],
    used_contexts: set[tuple[str, str]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = load_jsonl(SOURCE_DIR / f"{domain}.jsonl")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    excluded_line_count = 0
    excluded_context_count = 0
    for row in rows:
        line_key = (domain, int(row["_source_line_no"]))
        ctx_hash = context_hash(str(row.get("context", "")))
        if line_key in used_lines:
            excluded_line_count += 1
            continue
        if (domain, ctx_hash) in used_contexts:
            excluded_context_count += 1
            continue
        groups[ctx_hash].append(row)

    group_items = []
    for ctx_hash, group_rows in groups.items():
        context = str(group_rows[0].get("context", ""))
        group_rows.sort(key=lambda row: (question_family(str(row.get("input", ""))), int(row["_source_line_no"])))
        group_items.append((len(context), ctx_hash, group_rows))
    group_items.sort(key=lambda item: (item[0], item[1]))

    selected: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    family_counts: Counter[str] = Counter()
    for context_len, ctx_hash, group_rows in group_items:
        if len(selected) >= samples_per_domain:
            break
        # Prefer a mix of question shapes from each context before taking duplicates.
        by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in group_rows:
            by_family[question_family(str(row.get("input", "")))].append(row)
        chosen: list[dict[str, Any]] = []
        family_order = sorted(by_family, key=lambda name: (family_counts[name], name))
        while len(chosen) < min(max_samples_per_context, samples_per_domain - len(selected)):
            added = False
            for family in family_order:
                if by_family[family]:
                    chosen.append(by_family[family].pop(0))
                    family_counts[family] += 1
                    added = True
                    if len(chosen) >= min(max_samples_per_context, samples_per_domain - len(selected)):
                        break
            if not added:
                break
        if not chosen:
            continue
        context = str(chosen[0].get("context", ""))
        window, start, end, reason, matched_answers = make_window(context, chosen, max_window_chars, margin_chars)
        source_lines = []
        for sample in chosen:
            item = dict(sample)
            item["context"] = window
            item["dataset"] = domain
            item["label"] = domain
            item["source_dataset"] = domain
            item["source_line_no"] = int(sample["_source_line_no"])
            item["source_context_hash"] = ctx_hash
            item["source_context_chars"] = context_len
            item["window_start"] = start
            item["window_end"] = end
            item["window_chars"] = len(window)
            item["window_reason"] = reason
            item["locked_holdout_family"] = question_family(str(sample.get("input", "")))
            item.pop("_source_line_no", None)
            selected.append(item)
            source_lines.append(int(sample["_source_line_no"]))
        manifest_rows.append(
            {
                "domain": domain,
                "context_hash": ctx_hash,
                "source_context_chars": context_len,
                "window_start": start,
                "window_end": end,
                "window_chars": len(window),
                "selected_source_lines": source_lines,
                "question_families": dict(Counter(item["locked_holdout_family"] for item in selected if item.get("source_context_hash") == ctx_hash)),
                "matched_answers": matched_answers,
                "reason": reason,
            }
        )

    if len(selected) != samples_per_domain:
        raise RuntimeError(f"{domain}: expected {samples_per_domain}, selected {len(selected)}")
    manifest_rows.insert(
        0,
        {
            "domain": domain,
            "excluded_line_count": excluded_line_count,
            "excluded_context_count": excluded_context_count,
            "candidate_contexts": len(groups),
        },
    )
    return selected, manifest_rows


def write_markdown(
    path: Path,
    output_path: Path,
    rows: list[dict[str, Any]],
    manifest_rows: list[dict[str, Any]],
    args: argparse.Namespace,
) -> None:
    domain_counts = Counter(row["dataset"] for row in rows)
    family_counts = Counter(row.get("locked_holdout_family", "unknown") for row in rows)
    context_count = len({(row["source_dataset"], row["source_context_hash"]) for row in rows})
    source_line_count = len({(row["source_dataset"], row["source_line_no"]) for row in rows})
    domains = getattr(args, "domains", DOMAINS)
    lines = [
        "# SAGE Locked Unseen Holdout Manifest\n\n",
        f"- Output dataset: `{output_path.relative_to(ROOT_DIR).as_posix()}`\n",
        f"- Samples: {len(rows)}\n",
        f"- Domains: {', '.join(domains)}\n",
        f"- Samples per domain: {args.samples_per_domain}\n",
        f"- Unique source contexts: {context_count}\n",
        f"- Unique source lines: {source_line_count}\n",
        f"- Max samples per context: {args.max_samples_per_context}\n",
        f"- Max window chars: {args.max_window_chars}\n",
        "- Selection excludes previously used source lines and context hashes from the main aligned, unseen, and exploratory validation slices.\n",
        "- This dataset is locked once generated; do not use it for further method tuning after first evaluation.\n\n",
        "## Domain Counts\n\n",
        "| Domain | Samples |\n",
        "|---|---:|\n",
    ]
    for domain, count in sorted(domain_counts.items()):
        lines.append(f"| {domain} | {count} |\n")
    lines.extend(["\n## Question Families\n\n", "| Family | Samples |\n", "|---|---:|\n"])
    for family, count in sorted(family_counts.items()):
        lines.append(f"| {family} | {count} |\n")
    lines.extend([
        "\n## Selected Contexts\n\n",
        "| Domain | Context Hash | Source Chars | Window | Window Chars | Source Lines | Matched Answers | Reason |\n",
        "|---|---|---:|---|---:|---|---:|---|\n",
    ])
    for row in manifest_rows:
        if "context_hash" not in row:
            continue
        lines.append(
            f"| {row['domain']} | {row['context_hash']} | {row['source_context_chars']} | "
            f"{row['window_start']}-{row['window_end']} | {row['window_chars']} | "
            f"{row['selected_source_lines']} | {row['matched_answers']} | {row['reason']} |\n"
        )
    path.write_text("".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="datasets/UltraDomain/sage_locked_unseen75_windowed.jsonl")
    parser.add_argument("--manifest-md", default="results/historical_sage/SAGE_LOCKED_UNSEEN75_MANIFEST.md")
    parser.add_argument("--manifest-json", default="results/historical_sage/sage_locked_unseen75_manifest.json")
    parser.add_argument("--samples-per-domain", type=int, default=15)
    parser.add_argument("--max-samples-per-context", type=int, default=3)
    parser.add_argument("--max-window-chars", type=int, default=80_000)
    parser.add_argument("--margin-chars", type=int, default=12_000)
    parser.add_argument("--exclude-dataset", action="append", default=[])
    parser.add_argument("--domains", default=",".join(DOMAINS), help="Comma-separated domains to sample.")
    args = parser.parse_args()

    args.domains = [domain.strip().lower() for domain in args.domains.split(",") if domain.strip()]
    unknown_domains = sorted(set(args.domains) - set(DOMAINS))
    if unknown_domains:
        raise ValueError(f"Unknown domains: {', '.join(unknown_domains)}")

    exclude_paths = DEFAULT_EXCLUDE_DATASETS + args.exclude_dataset
    used_lines, used_contexts = used_keys(exclude_paths)
    all_rows: list[dict[str, Any]] = []
    all_manifest: list[dict[str, Any]] = []
    for domain in args.domains:
        rows, manifest = select_domain(
            domain,
            args.samples_per_domain,
            args.max_samples_per_context,
            args.max_window_chars,
            args.margin_chars,
            used_lines,
            used_contexts,
        )
        all_rows.extend(rows)
        all_manifest.extend(manifest)

    for index, row in enumerate(all_rows):
        row["_subset_line_no"] = index

    output_path = ROOT_DIR / args.output
    manifest_md_path = ROOT_DIR / args.manifest_md
    manifest_json_path = ROOT_DIR / args.manifest_json
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_md_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_json_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in all_rows), encoding="utf-8")
    write_markdown(manifest_md_path, output_path, all_rows, all_manifest, args)
    manifest_json_path.write_text(
        json.dumps(
            {
                "output_dataset": args.output,
                "samples": len(all_rows),
                "domains": args.domains,
                "samples_per_domain": args.samples_per_domain,
                "max_samples_per_context": args.max_samples_per_context,
                "max_window_chars": args.max_window_chars,
                "margin_chars": args.margin_chars,
                "excluded_datasets": exclude_paths,
                "used_source_lines_excluded": len(used_lines),
                "used_contexts_excluded": len(used_contexts),
                "domain_counts": dict(Counter(row["dataset"] for row in all_rows)),
                "family_counts": dict(Counter(row.get("locked_holdout_family", "unknown") for row in all_rows)),
                "manifest_rows": all_manifest,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved {len(all_rows)} samples to {output_path}")
    print(f"Saved manifest to {manifest_md_path}")
    print(f"Saved manifest JSON to {manifest_json_path}")


if __name__ == "__main__":
    main()
