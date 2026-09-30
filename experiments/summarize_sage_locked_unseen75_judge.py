#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Summarize LLM-judge results for the locked unseen75 SAGE holdout."""
from __future__ import annotations

import json
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from experiments.run_enhanced_experiments import (  # noqa: E402
    bootstrap_ci,
    cohens_d,
    paired_t_test,
    wilcoxon_signed_rank_test,
)


OUT_DIR = ROOT / "results" / "sage_locked_unseen75"
DATASET_PATH = ROOT / "datasets" / "UltraDomain" / "sage_locked_unseen75_windowed.jsonl"
OUTPUT_JSON = OUT_DIR / "sage_locked_unseen75_judge_summary.json"
OUTPUT_MD = OUT_DIR / "SAGE_LOCKED_UNSEEN75_JUDGE_SUMMARY.md"
EXPECTED_N = 75

METHODS = {
    "naive": OUT_DIR / "naive_aligned_n75_eval.jsonl",
    "hyde": OUT_DIR / "hyde_aligned_n75_eval.jsonl",
    "sage_v33_preservation_delta": OUT_DIR / "hyde_evidence_delta_v33_text_only_eval.jsonl",
}
SCORE_KEYS = ["correctness", "coverage", "relevance", "conciseness", "overall"]
COMPARISON_PAIRS = [
    ("hyde", "naive"),
    ("sage_v33_preservation_delta", "hyde"),
    ("sage_v33_preservation_delta", "naive"),
]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            row.setdefault("_file_line_no", line_no)
            rows.append(row)
    return rows


def dataset_metadata() -> dict[int, dict[str, Any]]:
    metadata: dict[int, dict[str, Any]] = {}
    for index, row in enumerate(read_jsonl(DATASET_PATH)):
        line_no = int(row.get("_subset_line_no", index))
        metadata[line_no] = {
            "domain": row.get("label") or row.get("dataset") or row.get("source_dataset") or "unknown",
            "family": row.get("locked_holdout_family") or "unknown",
            "source_line_no": row.get("source_line_no"),
            "source_context_hash": row.get("source_context_hash"),
        }
    return metadata


def mean(values: Iterable[float]) -> float | None:
    vals = list(values)
    return statistics.mean(vals) if vals else None


def fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def fmt_delta(value: Any, digits: int = 3) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value):+.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def fmt_p(value: Any) -> str:
    if value is None:
        return "N/A"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if number == 0:
        return "<1e-15"
    if number < 0.001:
        return f"{number:.2e}"
    return f"{number:.4f}"


def md_table(headers: list[str], rows: Iterable[Iterable[Any]], numeric_cols: set[int] | None = None) -> list[str]:
    numeric_cols = numeric_cols or set()
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join("---:" if idx in numeric_cols else "---" for idx in range(len(headers))) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return lines


def score(row: dict[str, Any], key: str) -> float:
    return float((row.get("scores") or {}).get(key, 0) or 0)


def enriched_rows(path: Path, metadata: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in read_jsonl(path):
        item = dict(row)
        line_no = int(item.get("line_no", item.get("_file_line_no", 1) - 1))
        meta = metadata.get(line_no, {})
        item["line_no"] = line_no
        item["domain"] = item.get("dataset") or meta.get("domain") or "unknown"
        item["family"] = meta.get("family") or "unknown"
        rows.append(item)
    return rows


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    verdicts = Counter(str((row.get("scores") or {}).get("verdict", "unknown")) for row in rows)
    return {
        "sample_size": len(rows),
        "complete": len(rows) == EXPECTED_N,
        "averages": {key: mean(score(row, key) for row in rows) for key in SCORE_KEYS},
        "verdict_counts": dict(sorted(verdicts.items())),
    }


def aggregate_by(rows: list[dict[str, Any]], field: str) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row.get(field) or "unknown")].append(row)
    return {name: aggregate(items) for name, items in sorted(buckets.items())}


def paired_stats(left_rows: list[dict[str, Any]], right_rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    left_by_line = {int(row["line_no"]): row for row in left_rows}
    right_by_line = {int(row["line_no"]): row for row in right_rows}
    line_nos = sorted(set(left_by_line) & set(right_by_line))
    if not line_nos:
        return {}
    left = [score(left_by_line[line_no], key) for line_no in line_nos]
    right = [score(right_by_line[line_no], key) for line_no in line_nos]
    diffs = [l - r for l, r in zip(left, right)]
    t_stat, t_p = paired_t_test(left, right)
    w_stat, w_p = wilcoxon_signed_rank_test(left, right)
    return {
        "n": len(line_nos),
        "mean_delta": mean(diffs),
        "wins": sum(1 for diff in diffs if diff > 1e-12),
        "losses": sum(1 for diff in diffs if diff < -1e-12),
        "ties": sum(1 for diff in diffs if abs(diff) <= 1e-12),
        "t_statistic": t_stat,
        "t_p_value": t_p,
        "wilcoxon_stat": w_stat,
        "wilcoxon_p_value": w_p,
        "cohens_d": cohens_d(left, right),
        "bootstrap_ci": bootstrap_ci(diffs),
    }


def build_payload() -> dict[str, Any]:
    metadata = dataset_metadata()
    payload: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": str(DATASET_PATH.relative_to(ROOT)),
        "expected_n": EXPECTED_N,
        "methods": {},
        "comparisons": {},
    }
    for method, path in METHODS.items():
        if not path.exists():
            continue
        rows = enriched_rows(path, metadata)
        payload["methods"][method] = {
            "path": str(path.relative_to(ROOT)),
            "rows": rows,
            "aggregate": aggregate(rows),
            "by_domain": aggregate_by(rows, "domain"),
            "by_family": aggregate_by(rows, "family"),
        }
    for left, right in COMPARISON_PAIRS:
        if left not in payload["methods"] or right not in payload["methods"]:
            continue
        payload["comparisons"][f"{left}_minus_{right}"] = {
            key: paired_stats(payload["methods"][left]["rows"], payload["methods"][right]["rows"], key)
            for key in SCORE_KEYS
        }
    return payload


def write_report(payload: dict[str, Any]) -> None:
    lines: list[str] = [
        "# SAGE Locked Unseen75 Judge Summary",
        "",
        f"> Generated: {payload['generated_at']}",
        "",
        "## Scope",
        "",
        "- Dataset: `datasets/UltraDomain/sage_locked_unseen75_windowed.jsonl`",
        "- Judge: DeepSeek-compatible OpenAI endpoint via `experiments/evaluate_repro_results.py`.",
        "- Scale: 1-5, higher is better; paired deltas use identical 75 questions.",
        "",
        "## Average Scores",
        "",
    ]
    lines.extend(
        md_table(
            ["Method", "n", "Complete", "Correctness", "Coverage", "Relevance", "Conciseness", "Overall", "Excellent", "Good", "Partial", "Poor"],
            [
                [
                    method,
                    data["aggregate"]["sample_size"],
                    "yes" if data["aggregate"]["complete"] else "no",
                    fmt(data["aggregate"]["averages"].get("correctness")),
                    fmt(data["aggregate"]["averages"].get("coverage")),
                    fmt(data["aggregate"]["averages"].get("relevance")),
                    fmt(data["aggregate"]["averages"].get("conciseness")),
                    fmt(data["aggregate"]["averages"].get("overall")),
                    data["aggregate"]["verdict_counts"].get("excellent", 0),
                    data["aggregate"]["verdict_counts"].get("good", 0),
                    data["aggregate"]["verdict_counts"].get("partial", 0),
                    data["aggregate"]["verdict_counts"].get("poor", 0),
                ]
                for method, data in payload["methods"].items()
            ],
            numeric_cols={1, 3, 4, 5, 6, 7, 8, 9, 10, 11},
        )
    )

    lines.extend(["", "## Paired Deltas", ""])
    paired_rows = []
    for comparison, metrics in payload["comparisons"].items():
        left, right = comparison.split("_minus_", 1)
        for key, stats in metrics.items():
            ci = stats.get("bootstrap_ci") or [None, None]
            paired_rows.append(
                [
                    f"{left} - {right}",
                    key,
                    stats.get("n"),
                    fmt_delta(stats.get("mean_delta")),
                    stats.get("wins"),
                    stats.get("losses"),
                    stats.get("ties"),
                    fmt_p(stats.get("t_p_value")),
                    fmt_p(stats.get("wilcoxon_p_value")),
                    fmt_delta(stats.get("cohens_d"), 2),
                    f"[{fmt(ci[0])}, {fmt(ci[1])}]",
                ]
            )
    lines.extend(
        md_table(
            ["Comparison", "Metric", "n", "Delta", "Wins", "Losses", "Ties", "t-p", "Wilcoxon-p", "d", "95% CI"],
            paired_rows,
            numeric_cols={2, 3, 4, 5, 6, 9},
        )
    )

    for field, title in [("by_domain", "Domain Slice"), ("by_family", "Question-Family Slice")]:
        lines.extend(["", f"## {title}", ""])
        slice_rows = []
        for method, data in payload["methods"].items():
            for name, agg in data[field].items():
                slice_rows.append([
                    method,
                    name,
                    agg["sample_size"],
                    fmt(agg["averages"].get("correctness")),
                    fmt(agg["averages"].get("coverage")),
                    fmt(agg["averages"].get("overall")),
                ])
        lines.extend(
            md_table(
                ["Method", "Slice", "n", "Correctness", "Coverage", "Overall"],
                slice_rows,
                numeric_cols={2, 3, 4, 5},
            )
        )
    lines.append("")
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload = build_payload()
    json_payload = json.loads(json.dumps(payload, ensure_ascii=False))
    for data in json_payload["methods"].values():
        data.pop("rows", None)
    OUTPUT_JSON.write_text(json.dumps(json_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(payload)
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {OUTPUT_MD}")


if __name__ == "__main__":
    main()
