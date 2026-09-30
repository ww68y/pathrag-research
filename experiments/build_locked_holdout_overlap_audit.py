#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Audit source-line and source-context overlap for locked SAGE holdouts.

The audit is fully offline. It verifies that each locked holdout avoids the
historical exclusion slices and previously locked holdouts under both source
line and source context hash keys.
"""
from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "sage_locked_holdout_overlap"
DOMAINS = {"legal", "biology", "history", "physics", "agriculture"}

BASE_EXCLUSIONS = [
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

LOCKED_HOLDOUTS = [
    ("locked_unseen75", "Locked Unseen75", "datasets/UltraDomain/sage_locked_unseen75_windowed.jsonl"),
    ("locked_unseen25_repeat1", "Repeat1 N25", "datasets/UltraDomain/sage_locked_unseen25_repeat1_windowed.jsonl"),
    ("locked_unseen40_repeat2", "Repeat2 N40", "datasets/UltraDomain/sage_locked_unseen40_repeat2_windowed.jsonl"),
    ("locked_unseen24_repeat3", "Repeat3 N24", "datasets/UltraDomain/sage_locked_unseen24_repeat3_windowed.jsonl"),
    ("locked_unseen24_repeat4", "Repeat4 N24", "datasets/UltraDomain/sage_locked_unseen24_repeat4_windowed.jsonl"),
    ("locked_legal40_repeat5", "Repeat5 Legal40 Stress", "datasets/UltraDomain/sage_locked_legal40_repeat5_windowed.jsonl"),
    ("locked_history4_repeat6", "Repeat6 History4 Tail", "datasets/UltraDomain/sage_locked_history4_repeat6_windowed.jsonl"),
    ("locked_legal40_repeat7", "Repeat7 Legal40 Stress", "datasets/UltraDomain/sage_locked_legal40_repeat7_windowed.jsonl"),
    ("locked_legal54_repeat8", "Repeat8 Legal54 Exhaustive Stress", "datasets/UltraDomain/sage_locked_legal54_repeat8_windowed.jsonl"),
]


def clean(value: Any) -> str:
    return " ".join(str(value or "").split())


def context_hash(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:12]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as source:
        for file_line_no, line in enumerate(source):
            if not line.strip():
                continue
            row = json.loads(line)
            row["_file_line_no"] = file_line_no
            rows.append(row)
    return rows


def domain_for(row: dict[str, Any]) -> str:
    domain = clean(row.get("source_dataset") or row.get("dataset") or row.get("label")).lower()
    return domain if domain in DOMAINS else "unknown"


def line_key(row: dict[str, Any]) -> tuple[str, int] | None:
    domain = domain_for(row)
    if domain == "unknown":
        return None
    source_line = row.get("source_line_no")
    if source_line is None:
        return None
    try:
        return domain, int(source_line)
    except (TypeError, ValueError):
        return None


def context_key(row: dict[str, Any]) -> tuple[str, str] | None:
    domain = domain_for(row)
    if domain == "unknown":
        return None
    explicit = clean(row.get("source_context_hash"))
    if explicit:
        return domain, explicit
    context = str(row.get("context") or "")
    if not context:
        return None
    return domain, context_hash(context)


def key_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    line_keys = [key for row in rows if (key := line_key(row))]
    context_keys = [key for row in rows if (key := context_key(row))]
    line_counts = Counter(line_keys)
    context_counts = Counter(context_keys)
    return {
        "rows": len(rows),
        "domain_counts": dict(Counter(domain_for(row) for row in rows)),
        "line_keys": sorted(set(line_keys)),
        "context_keys": sorted(set(context_keys)),
        "duplicate_line_keys": {str(key): count for key, count in line_counts.items() if count > 1},
        "duplicate_context_keys": {str(key): count for key, count in context_counts.items() if count > 1},
    }


def overlap(left: set[tuple[Any, ...]], right: set[tuple[Any, ...]]) -> list[tuple[Any, ...]]:
    return sorted(left & right)


def build_payload() -> dict[str, Any]:
    base_rows = []
    base_files = []
    for relative in BASE_EXCLUSIONS:
        path = ROOT / relative
        rows = read_jsonl(path)
        base_rows.extend(rows)
        base_files.append({"path": relative, "exists": path.exists(), "rows": len(rows)})
    base = key_summary(base_rows)
    prior_lines: set[tuple[str, int]] = set(base["line_keys"])
    prior_contexts: set[tuple[str, str]] = set(base["context_keys"])

    holdout_rows = []
    pairwise_overlaps = []
    prior_holdouts: list[dict[str, Any]] = []
    for key, label, relative in LOCKED_HOLDOUTS:
        path = ROOT / relative
        rows = read_jsonl(path)
        summary = key_summary(rows)
        line_set = set(summary["line_keys"])
        context_set = set(summary["context_keys"])
        base_line_overlap = overlap(line_set, set(base["line_keys"]))
        base_context_overlap = overlap(context_set, set(base["context_keys"]))
        prior_line_overlap = overlap(line_set, prior_lines)
        prior_context_overlap = overlap(context_set, prior_contexts)
        holdout = {
            "key": key,
            "label": label,
            "path": relative,
            "exists": path.exists(),
            "rows": summary["rows"],
            "domain_counts": summary["domain_counts"],
            "unique_line_keys": len(line_set),
            "unique_context_keys": len(context_set),
            "duplicate_line_keys": summary["duplicate_line_keys"],
            "duplicate_context_keys": summary["duplicate_context_keys"],
            "base_line_overlap_count": len(base_line_overlap),
            "base_context_overlap_count": len(base_context_overlap),
            "prior_line_overlap_count": len(prior_line_overlap),
            "prior_context_overlap_count": len(prior_context_overlap),
            "base_line_overlap_sample": base_line_overlap[:10],
            "base_context_overlap_sample": base_context_overlap[:10],
            "prior_line_overlap_sample": prior_line_overlap[:10],
            "prior_context_overlap_sample": prior_context_overlap[:10],
        }
        holdout_rows.append(holdout)
        for prior in prior_holdouts:
            pair_line_overlap = overlap(line_set, set(prior["line_keys"]))
            pair_context_overlap = overlap(context_set, set(prior["context_keys"]))
            if pair_line_overlap or pair_context_overlap:
                pairwise_overlaps.append(
                    {
                        "left": prior["key"],
                        "right": key,
                        "line_overlap_count": len(pair_line_overlap),
                        "context_overlap_count": len(pair_context_overlap),
                        "line_overlap_sample": pair_line_overlap[:10],
                        "context_overlap_sample": pair_context_overlap[:10],
                    }
                )
        prior_holdouts.append({"key": key, "line_keys": sorted(line_set), "context_keys": sorted(context_set)})
        prior_lines.update(line_set)
        prior_contexts.update(context_set)

    pass_status = all(
        row["base_line_overlap_count"] == 0
        and row["base_context_overlap_count"] == 0
        and row["prior_line_overlap_count"] == 0
        and row["prior_context_overlap_count"] == 0
        and not row["duplicate_line_keys"]
        for row in holdout_rows
    )
    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "status": "PASS" if pass_status and not pairwise_overlaps else "FAIL",
        "base_exclusions": base_files,
        "base_unique_line_keys": len(base["line_keys"]),
        "base_unique_context_keys": len(base["context_keys"]),
        "holdouts": holdout_rows,
        "pairwise_holdout_overlaps": pairwise_overlaps,
        "notes": [
            "Context duplicates within a holdout are allowed up to that holdout's max samples per context; source-line duplicates are not allowed.",
            "Prior-overlap counts compare each holdout against base exclusions and all earlier locked holdouts in sequence.",
            "Repeat5 and Repeat7 are legal-only stress holdouts prepared after balanced non-legal capacity was exhausted under strict context exclusion.",
            "Repeat6 is the final strict history-only tail holdout; after it, non-legal strict capacity is exhausted.",
            "Repeat8 exhausts the remaining strict legal capacity under max two samples per context.",
        ],
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Locked Holdout Overlap Audit",
        "",
        f"> Generated: {payload['generated_at']}",
        f"> Status: `{payload['status']}`",
        "",
        "## Base Exclusions",
        "",
        f"- Unique source lines: `{payload['base_unique_line_keys']}`",
        f"- Unique source contexts: `{payload['base_unique_context_keys']}`",
        "",
        "## Holdout Checks",
        "",
        "| Holdout | Rows | Domains | Unique Lines | Unique Contexts | Base Line Overlap | Base Context Overlap | Prior Line Overlap | Prior Context Overlap | Line Duplicates |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in payload["holdouts"]:
        lines.append(
            f"| {row['label']} | {row['rows']} | {row['domain_counts']} | {row['unique_line_keys']} | "
            f"{row['unique_context_keys']} | {row['base_line_overlap_count']} | {row['base_context_overlap_count']} | "
            f"{row['prior_line_overlap_count']} | {row['prior_context_overlap_count']} | {len(row['duplicate_line_keys'])} |"
        )

    lines.extend(["", "## Pairwise Holdout Overlaps", ""])
    if payload["pairwise_holdout_overlaps"]:
        lines.extend([
            "| Left | Right | Line Overlap | Context Overlap |",
            "|---|---|---:|---:|",
        ])
        for row in payload["pairwise_holdout_overlaps"]:
            lines.append(f"| {row['left']} | {row['right']} | {row['line_overlap_count']} | {row['context_overlap_count']} |")
    else:
        lines.append("- No pairwise source-line or source-context overlaps found among locked holdouts.")

    lines.extend(["", "## Notes", ""])
    for note in payload["notes"]:
        lines.append(f"- {note}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = build_payload()
    json_path = OUT_DIR / "locked_holdout_overlap_audit.json"
    md_path = OUT_DIR / "LOCKED_HOLDOUT_OVERLAP_AUDIT.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(md_path, payload)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")
    print(f"Status: {payload['status']}")


if __name__ == "__main__":
    main()
