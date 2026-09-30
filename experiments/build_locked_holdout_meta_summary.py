#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Build an offline meta-summary for locked unseen SAGE holdouts.

This script aggregates the final locked unseen75 holdout and subsequent strict
zero-overlap repeats. It does not call any API provider; incomplete judge and
pairwise files are surfaced as coverage gaps instead of being imputed.
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "results" / "sage_locked_unseen_meta"
METHODS = ["naive", "hyde", "sage_v33_preservation_delta"]
REPEATS = [
    {
        "key": "locked_unseen75",
        "label": "Locked Unseen75",
        "n": 75,
        "format": "unseen75",
        "lexical": ROOT / "results" / "sage_locked_unseen75" / "sage_locked_unseen75_baseline_summary.json",
        "judge": ROOT / "results" / "sage_locked_unseen75" / "sage_locked_unseen75_judge_summary.json",
        "pairwise": ROOT / "results" / "sage_locked_unseen75" / "sage_locked_unseen75_pairwise_summary.json",
    },
    {
        "key": "locked_unseen25_repeat1",
        "label": "Repeat1 N25",
        "n": 25,
        "format": "repeat",
        "summary": ROOT / "results" / "sage_locked_unseen25_repeat1" / "sage_locked_unseen25_repeat1_summary.json",
    },
    {
        "key": "locked_unseen40_repeat2",
        "label": "Repeat2 N40",
        "n": 40,
        "format": "repeat",
        "summary": ROOT / "results" / "sage_locked_unseen40_repeat2" / "sage_locked_unseen40_repeat2_summary.json",
    },
    {
        "key": "locked_unseen24_repeat3",
        "label": "Repeat3 N24",
        "n": 24,
        "format": "repeat",
        "summary": ROOT / "results" / "sage_locked_unseen24_repeat3" / "sage_locked_unseen24_repeat3_summary.json",
    },
    {
        "key": "locked_unseen24_repeat4",
        "label": "Repeat4 N24",
        "n": 24,
        "format": "repeat",
        "summary": ROOT / "results" / "sage_locked_unseen24_repeat4" / "sage_locked_unseen24_repeat4_summary.json",
    },
    {
        "key": "locked_legal40_repeat5",
        "label": "Repeat5 Legal40",
        "n": 40,
        "format": "repeat",
        "summary": ROOT / "results" / "sage_locked_legal40_repeat5" / "sage_locked_legal40_repeat5_summary.json",
    },
    {
        "key": "locked_history4_repeat6",
        "label": "Repeat6 History4",
        "n": 4,
        "format": "repeat",
        "summary": ROOT / "results" / "sage_locked_history4_repeat6" / "sage_locked_history4_repeat6_summary.json",
    },
    {
        "key": "locked_legal40_repeat7",
        "label": "Repeat7 Legal40",
        "n": 40,
        "format": "repeat",
        "summary": ROOT / "results" / "sage_locked_legal40_repeat7" / "sage_locked_legal40_repeat7_summary.json",
    },
    {
        "key": "locked_legal54_repeat8",
        "label": "Repeat8 Legal54",
        "n": 54,
        "format": "repeat",
        "summary": ROOT / "results" / "sage_locked_legal54_repeat8" / "sage_locked_legal54_repeat8_summary.json",
    },
]


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)).replace("\\", "/")


def weighted_mean(items: Iterable[tuple[float | None, int]]) -> float | None:
    numerator = 0.0
    denominator = 0
    for value, weight in items:
        if value is None or not weight:
            continue
        numerator += float(value) * int(weight)
        denominator += int(weight)
    return numerator / denominator if denominator else None


def mean(values: Iterable[float | None]) -> float | None:
    vals = [float(value) for value in values if value is not None]
    return statistics.mean(vals) if vals else None


def fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def fmt_delta(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):+.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def fmt_pct(value: Any, digits: int = 1) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return str(value)


def coverage_status(coverage: dict[str, Any], complete_key: str, rows_key: str) -> str:
    if coverage.get(complete_key):
        return "complete"
    expected_n = int(coverage.get("expected_n") or 0)
    rows = coverage.get(rows_key) or {}
    values = sorted(set(int(value or 0) for value in rows.values()))
    if len(values) == 1:
        return f"pending ({values[0]}/{expected_n} each)"
    return ", ".join(f"{key} {value}/{expected_n}" for key, value in rows.items())


def md_table(headers: list[str], rows: Iterable[Iterable[Any]], numeric_cols: set[int] | None = None) -> list[str]:
    numeric_cols = numeric_cols or set()
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join("---:" if idx in numeric_cols else "---" for idx in range(len(headers))) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return lines


def unseen75_method_aggregates(lexical: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for method in METHODS:
        aggregate = ((lexical.get("methods") or {}).get(method) or {}).get("aggregate") or {}
        rows[method] = {
            "sample_size": aggregate.get("sample_size"),
            "errors": aggregate.get("errors"),
            "mean_f1": aggregate.get("mean_f1"),
            "mean_rouge_l": aggregate.get("mean_rouge_l"),
            "mean_runtime_seconds": aggregate.get("mean_runtime_seconds"),
            "mean_prediction_chars": aggregate.get("mean_prediction_chars"),
        }
    return rows


def unseen75_judge_summaries(judge: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for method in METHODS:
        aggregate = ((judge.get("methods") or {}).get(method) or {}).get("aggregate") or {}
        rows[method] = {
            "n": aggregate.get("sample_size") or 0,
            "means": aggregate.get("averages") or {},
            "verdict_counts": aggregate.get("verdict_counts") or {},
        }
    return rows


def unseen75_pairwise(pairwise: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for name, source_name in [
        ("sage_v33_vs_hyde", "sage_v33_preservation_delta_minus_hyde"),
        ("sage_v33_vs_naive", "sage_v33_preservation_delta_minus_naive"),
    ]:
        source = (pairwise.get("comparisons") or {}).get(source_name) or {}
        dimensions = source.get("dimensions") or {}
        rows[name] = {
            "n": source.get("sample_size") or 0,
            "avg_target_win_rate": mean((dimension or {}).get("target_win_rate") for dimension in dimensions.values()),
        }
    return rows


def normalize_repeat(config: dict[str, Any]) -> dict[str, Any]:
    if config["format"] == "unseen75":
        lexical = read_json(config["lexical"])
        judge = read_json(config["judge"])
        pairwise = read_json(config["pairwise"])
        return {
            "key": config["key"],
            "label": config["label"],
            "expected_n": lexical.get("expected_n") or config["n"],
            "method_aggregates": unseen75_method_aggregates(lexical),
            "judge_summaries": unseen75_judge_summaries(judge),
            "pairwise": unseen75_pairwise(pairwise),
            "source_files": {
                "lexical": rel(config["lexical"]),
                "judge": rel(config["judge"]),
                "pairwise": rel(config["pairwise"]),
            },
        }

    summary = read_json(config["summary"])
    return {
        "key": config["key"],
        "label": config["label"],
        "expected_n": summary.get("expected_n") or config["n"],
        "method_aggregates": summary.get("method_aggregates") or {},
        "judge_summaries": summary.get("judge_summaries") or {},
        "pairwise": summary.get("pairwise") or {},
        "guard_summary": summary.get("guard_summary") or {},
        "source_files": {"summary": rel(config["summary"])},
    }


def method_delta(summary: dict[str, Any], method: str, metric: str) -> float | None:
    sage = (summary.get("method_aggregates") or {}).get("sage_v33_preservation_delta") or {}
    baseline = (summary.get("method_aggregates") or {}).get(method) or {}
    left = sage.get(metric)
    right = baseline.get(metric)
    if left is None or right is None:
        return None
    return float(left) - float(right)


def coverage_for(summary: dict[str, Any]) -> dict[str, Any]:
    expected_n = int(summary.get("expected_n") or 0)
    method_aggregates = summary.get("method_aggregates") or {}
    judge_summaries = summary.get("judge_summaries") or {}
    pairwise = summary.get("pairwise") or {}
    method_rows = {
        method: int((method_aggregates.get(method) or {}).get("sample_size") or 0)
        for method in METHODS
    }
    judge_rows = {
        method: int((judge_summaries.get(method) or {}).get("n") or 0)
        for method in METHODS
    }
    pairwise_rows = {
        name: int((pairwise.get(name) or {}).get("n") or 0)
        for name in ["sage_v33_vs_hyde", "sage_v33_vs_naive"]
    }
    return {
        "expected_n": expected_n,
        "methods_complete": all(value == expected_n for value in method_rows.values()),
        "judge_complete": all(value == expected_n for value in judge_rows.values()),
        "pairwise_complete": all(value == expected_n for value in pairwise_rows.values()),
        "method_rows": method_rows,
        "judge_rows": judge_rows,
        "pairwise_rows": pairwise_rows,
    }


def build_payload() -> dict[str, Any]:
    repeats = [normalize_repeat(config) for config in REPEATS]
    for repeat in repeats:
        repeat["coverage"] = coverage_for(repeat)
        repeat["lexical_deltas"] = {
            "sage_v33_vs_hyde_f1": method_delta(repeat, "hyde", "mean_f1"),
            "sage_v33_vs_naive_f1": method_delta(repeat, "naive", "mean_f1"),
            "sage_v33_vs_hyde_rouge_l": method_delta(repeat, "hyde", "mean_rouge_l"),
            "sage_v33_vs_naive_rouge_l": method_delta(repeat, "naive", "mean_rouge_l"),
        }

    complete_judge_repeats = [repeat for repeat in repeats if repeat["coverage"]["judge_complete"]]
    all_expected_n = sum(int(repeat.get("expected_n") or 0) for repeat in repeats)
    judge_expected_n = sum(int(repeat.get("expected_n") or 0) for repeat in complete_judge_repeats)

    weighted_methods = {}
    for method in METHODS:
        weighted_methods[method] = {
            "sample_size": sum(int(((repeat.get("method_aggregates") or {}).get(method) or {}).get("sample_size") or 0) for repeat in repeats),
            "mean_f1": weighted_mean(
                (((repeat.get("method_aggregates") or {}).get(method) or {}).get("mean_f1"), int(((repeat.get("method_aggregates") or {}).get(method) or {}).get("sample_size") or 0))
                for repeat in repeats
            ),
            "mean_rouge_l": weighted_mean(
                (((repeat.get("method_aggregates") or {}).get(method) or {}).get("mean_rouge_l"), int(((repeat.get("method_aggregates") or {}).get(method) or {}).get("sample_size") or 0))
                for repeat in repeats
            ),
            "mean_runtime_seconds": weighted_mean(
                (((repeat.get("method_aggregates") or {}).get(method) or {}).get("mean_runtime_seconds"), int(((repeat.get("method_aggregates") or {}).get(method) or {}).get("sample_size") or 0))
                for repeat in repeats
            ),
            "mean_prediction_chars": weighted_mean(
                (((repeat.get("method_aggregates") or {}).get(method) or {}).get("mean_prediction_chars"), int(((repeat.get("method_aggregates") or {}).get(method) or {}).get("sample_size") or 0))
                for repeat in repeats
            ),
        }

    weighted_judge = {}
    for method in METHODS:
        weighted_judge[method] = {
            "sample_size": sum(int(((repeat.get("judge_summaries") or {}).get(method) or {}).get("n") or 0) for repeat in complete_judge_repeats),
            "overall": weighted_mean(
                ((((repeat.get("judge_summaries") or {}).get(method) or {}).get("means") or {}).get("overall"), int(((repeat.get("judge_summaries") or {}).get(method) or {}).get("n") or 0))
                for repeat in complete_judge_repeats
            ),
            "correctness": weighted_mean(
                ((((repeat.get("judge_summaries") or {}).get(method) or {}).get("means") or {}).get("correctness"), int(((repeat.get("judge_summaries") or {}).get(method) or {}).get("n") or 0))
                for repeat in complete_judge_repeats
            ),
            "coverage": weighted_mean(
                ((((repeat.get("judge_summaries") or {}).get(method) or {}).get("means") or {}).get("coverage"), int(((repeat.get("judge_summaries") or {}).get(method) or {}).get("n") or 0))
                for repeat in complete_judge_repeats
            ),
        }

    pairwise_weighted = {}
    for comparison in ["sage_v33_vs_hyde", "sage_v33_vs_naive"]:
        complete_pairwise_repeats = [repeat for repeat in repeats if int(((repeat.get("pairwise") or {}).get(comparison) or {}).get("n") or 0) == int(repeat.get("expected_n") or 0)]
        pairwise_weighted[comparison] = {
            "sample_size": sum(int(((repeat.get("pairwise") or {}).get(comparison) or {}).get("n") or 0) for repeat in complete_pairwise_repeats),
            "avg_target_win_rate": weighted_mean(
                (((repeat.get("pairwise") or {}).get(comparison) or {}).get("avg_target_win_rate"), int(((repeat.get("pairwise") or {}).get(comparison) or {}).get("n") or 0))
                for repeat in complete_pairwise_repeats
            ),
        }

    incomplete_judge = [repeat["label"] for repeat in repeats if not repeat["coverage"]["judge_complete"]]
    incomplete_pairwise = [repeat["label"] for repeat in repeats if not repeat["coverage"]["pairwise_complete"]]
    notes = []
    if incomplete_judge or incomplete_pairwise:
        if incomplete_judge:
            notes.append(
                "Judge coverage is incomplete for: " + ", ".join(incomplete_judge) + "."
            )
        if incomplete_pairwise:
            notes.append(
                "Pairwise coverage is incomplete for: " + ", ".join(incomplete_pairwise) + "."
            )
        notes.append("No judge or pairwise values are imputed for incomplete repeats.")
    else:
        notes.append("All included locked holdouts have complete judge and pairwise coverage.")
    notes.append("Weighted lexical metrics include every method-complete locked holdout.")

    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "scope": "locked unseen75 plus strict zero-overlap repeats 1-8",
        "total_expected_n": all_expected_n,
        "complete_judge_expected_n": judge_expected_n,
        "repeats": repeats,
        "weighted_methods_all_repeats": weighted_methods,
        "weighted_judge_complete_repeats": weighted_judge,
        "weighted_pairwise_complete_repeats": pairwise_weighted,
        "notes": notes,
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# SAGE Locked Unseen Meta Summary",
        "",
        f"> Generated: {payload['generated_at']}",
        "> Scope: offline consolidation from completed locked holdout artifacts; no API calls.",
        "",
        "## Coverage",
        "",
    ]
    lines.extend(
        md_table(
            ["Holdout", "N", "Methods", "Judge", "Pairwise", "SAGE-HyDE F1", "SAGE-Naive F1"],
            [
                [
                    repeat["label"],
                    repeat["expected_n"],
                    coverage_status(repeat["coverage"], "methods_complete", "method_rows"),
                    coverage_status(repeat["coverage"], "judge_complete", "judge_rows"),
                    coverage_status(repeat["coverage"], "pairwise_complete", "pairwise_rows"),
                    fmt_delta(repeat["lexical_deltas"].get("sage_v33_vs_hyde_f1")),
                    fmt_delta(repeat["lexical_deltas"].get("sage_v33_vs_naive_f1")),
                ]
                for repeat in payload["repeats"]
            ],
            {1, 5, 6},
        )
    )

    lines.extend(["", "## Weighted Lexical Metrics", ""])
    lines.extend(
        md_table(
            ["Method", "N", "F1", "ROUGE-L", "Runtime s/q", "Avg Chars"],
            [
                [
                    method,
                    row["sample_size"],
                    fmt(row["mean_f1"]),
                    fmt(row["mean_rouge_l"]),
                    fmt(row["mean_runtime_seconds"], 1),
                    fmt(row["mean_prediction_chars"], 1),
                ]
                for method, row in payload["weighted_methods_all_repeats"].items()
            ],
            {1, 2, 3, 4, 5},
        )
    )

    lines.extend(["", "## Complete-Judge Subset", ""])
    lines.append(f"Judge-complete holdouts: `{payload['complete_judge_expected_n']}/{payload['total_expected_n']}` rows.")
    lines.append("")
    lines.extend(
        md_table(
            ["Method", "N", "Correct", "Coverage", "Overall"],
            [
                [method, row["sample_size"], fmt(row["correctness"], 3), fmt(row["coverage"], 3), fmt(row["overall"], 3)]
                for method, row in payload["weighted_judge_complete_repeats"].items()
            ],
            {1, 2, 3, 4},
        )
    )

    lines.extend(["", "## Complete-Pairwise Subset", ""])
    lines.extend(
        md_table(
            ["Comparison", "N", "Avg Target Win Rate"],
            [
                [comparison, row["sample_size"], fmt_pct(row["avg_target_win_rate"], 2)]
                for comparison, row in payload["weighted_pairwise_complete_repeats"].items()
            ],
            {1, 2},
        )
    )

    lines.extend(["", "## Notes", ""])
    for note in payload["notes"]:
        lines.append(f"- {note}")
    lines.append("")
    lines.append("## Source Files")
    lines.append("")
    for repeat in payload["repeats"]:
        for source_type, source_path in repeat.get("source_files", {}).items():
            lines.append(f"- `{repeat['key']}.{source_type}`: `{source_path}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = build_payload()
    json_path = OUT_DIR / "sage_locked_unseen_meta_summary.json"
    md_path = OUT_DIR / "SAGE_LOCKED_UNSEEN_META_SUMMARY.md"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(md_path, payload)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    sys.exit(main())
