#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Summarize methods on the locked unseen75 SAGE holdout.

This script is intentionally offline. It reads completed baseline JSONL files,
computes lexical metrics against the locked references, and writes paper-planning
JSON/Markdown summaries without calling any API provider.
"""
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
    compute_contains,
    compute_exact_match,
    compute_f1,
    compute_rouge_l,
    paired_t_test,
    wilcoxon_signed_rank_test,
)


DATASET_PATH = ROOT / "datasets" / "UltraDomain" / "sage_locked_unseen75_windowed.jsonl"
OUT_DIR = ROOT / "results" / "sage_locked_unseen75"
OUTPUT_JSON = OUT_DIR / "sage_locked_unseen75_baseline_summary.json"
OUTPUT_MD = OUT_DIR / "SAGE_LOCKED_UNSEEN75_BASELINE_SUMMARY.md"

METHODS = {
    "naive": OUT_DIR / "naive_aligned_n75.jsonl",
    "hyde": OUT_DIR / "hyde_aligned_n75.jsonl",
    "sage_v27_contract_completion": OUT_DIR / "hyde_contract_completion_text_only.jsonl",
    "sage_v33_preservation_delta": OUT_DIR / "hyde_evidence_delta_v33_text_only.jsonl",
}
COMPARISON_PAIRS = [
    ("hyde", "naive"),
    ("sage_v27_contract_completion", "hyde"),
    ("sage_v33_preservation_delta", "hyde"),
    ("sage_v33_preservation_delta", "naive"),
    ("sage_v33_preservation_delta", "sage_v27_contract_completion"),
]
EXPECTED_N = 75


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                rows.append({"error": f"json_decode_error: {exc}", "_file_line_no": line_no})
                continue
            row.setdefault("_file_line_no", line_no)
            rows.append(row)
    return rows


def references(row: dict[str, Any]) -> list[str]:
    values = row.get("reference_answers") or row.get("answers") or row.get("gold_answer") or []
    if isinstance(values, str):
        values = [values]
    return [str(value) for value in values if str(value).strip()]


def prediction(row: dict[str, Any]) -> str:
    return str(row.get("prediction") or row.get("answer") or row.get("final_answer") or "")


def mean(values: Iterable[float]) -> float | None:
    vals = list(values)
    return statistics.mean(vals) if vals else None


def median(values: Iterable[float]) -> float | None:
    vals = list(values)
    return statistics.median(vals) if vals else None


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


def dataset_metadata() -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(DATASET_PATH):
        question = str(row.get("input") or row.get("question") or "")
        if not question:
            continue
        metadata[question] = {
            "domain": row.get("label") or row.get("dataset") or row.get("source_dataset") or "unknown",
            "family": row.get("locked_holdout_family") or "unknown",
            "source_line_no": row.get("source_line_no"),
            "source_context_hash": row.get("source_context_hash"),
            "window_reason": row.get("window_reason"),
        }
    return metadata


def score_row(row: dict[str, Any], method: str, metadata: dict[str, dict[str, Any]]) -> dict[str, Any]:
    question = str(row.get("question") or row.get("input") or "")
    refs = references(row)
    pred = prediction(row)
    if not refs or not pred or row.get("error"):
        metrics = {
            "f1": 0.0,
            "rouge_l": 0.0,
            "exact_match": 0.0,
            "contains": 0.0,
        }
    else:
        metrics = {
            "f1": max(compute_f1(pred, ref) for ref in refs),
            "rouge_l": max(compute_rouge_l(pred, ref) for ref in refs),
            "exact_match": max(compute_exact_match(pred, ref) for ref in refs),
            "contains": max(compute_contains(pred, ref) for ref in refs),
        }
    meta = metadata.get(question, {})
    query_seconds = float(row.get("query_seconds") or 0.0)
    insert_seconds = float(row.get("insert_seconds") or 0.0)
    domain = row.get("label") or row.get("dataset") or meta.get("domain") or "unknown"
    return {
        "method": method,
        "question": question,
        "line_no": row.get("line_no"),
        "domain": domain,
        "family": meta.get("family") or "unknown",
        "source_line_no": meta.get("source_line_no"),
        "source_context_hash": meta.get("source_context_hash"),
        "error": row.get("error"),
        "metrics": {
            **metrics,
            "query_seconds": query_seconds,
            "insert_seconds": insert_seconds,
            "runtime_seconds": query_seconds + insert_seconds,
            "num_chunks": int(row.get("num_chunks") or 0),
            "retrieved_chunks": len(row.get("retrieved_chunks") or []),
            "prediction_chars": len(pred),
        },
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors = [row for row in rows if row.get("error")]
    return {
        "sample_size": len(rows),
        "errors": len(errors),
        "mean_f1": mean(row["metrics"]["f1"] for row in rows),
        "mean_rouge_l": mean(row["metrics"]["rouge_l"] for row in rows),
        "mean_em": mean(row["metrics"]["exact_match"] for row in rows),
        "mean_contains": mean(row["metrics"]["contains"] for row in rows),
        "mean_runtime_seconds": mean(row["metrics"]["runtime_seconds"] for row in rows),
        "median_runtime_seconds": median(row["metrics"]["runtime_seconds"] for row in rows),
        "mean_query_seconds": mean(row["metrics"]["query_seconds"] for row in rows),
        "mean_num_chunks": mean(row["metrics"]["num_chunks"] for row in rows),
        "mean_prediction_chars": mean(row["metrics"]["prediction_chars"] for row in rows),
    }


def aggregate_by(rows: list[dict[str, Any]], field: str) -> dict[str, dict[str, Any]]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row.get(field) or "unknown")].append(row)
    return {name: aggregate(items) for name, items in sorted(buckets.items())}


def paired_stats(left: list[dict[str, Any]], right: list[dict[str, Any]], metric_name: str) -> dict[str, Any]:
    left_by_question = {row["question"]: row for row in left if row.get("question")}
    right_by_question = {row["question"]: row for row in right if row.get("question")}
    questions = sorted(set(left_by_question) & set(right_by_question))
    if not questions:
        return {}
    left_values = [float(left_by_question[q]["metrics"][metric_name]) for q in questions]
    right_values = [float(right_by_question[q]["metrics"][metric_name]) for q in questions]
    diffs = [left_val - right_val for left_val, right_val in zip(left_values, right_values)]
    t_stat, t_p = paired_t_test(left_values, right_values)
    w_stat, w_p = wilcoxon_signed_rank_test(left_values, right_values)
    return {
        "n": len(questions),
        "mean_delta": mean(diffs),
        "wins": sum(1 for diff in diffs if diff > 1e-12),
        "losses": sum(1 for diff in diffs if diff < -1e-12),
        "ties": sum(1 for diff in diffs if abs(diff) <= 1e-12),
        "t_statistic": t_stat,
        "t_p_value": t_p,
        "wilcoxon_stat": w_stat,
        "wilcoxon_p_value": w_p,
        "cohens_d": cohens_d(left_values, right_values),
        "bootstrap_ci": bootstrap_ci(diffs),
    }


def method_payload(method: str, path: Path, metadata: dict[str, dict[str, Any]]) -> dict[str, Any]:
    raw_rows = read_jsonl(path)
    scored = [score_row(row, method, metadata) for row in raw_rows]
    return {
        "path": str(path.relative_to(ROOT)),
        "expected_n": EXPECTED_N,
        "raw_rows": len(raw_rows),
        "complete": len(raw_rows) == EXPECTED_N and not any(row.get("error") for row in scored),
        "domain_counts": dict(Counter(row.get("domain") for row in scored)),
        "family_counts": dict(Counter(row.get("family") for row in scored)),
        "aggregate": aggregate(scored),
        "by_domain": aggregate_by(scored, "domain"),
        "by_family": aggregate_by(scored, "family"),
        "rows": scored,
    }


def write_report(payload: dict[str, Any]) -> None:
    lines: list[str] = [
        "# SAGE Locked Unseen75 Method Summary",
        "",
        f"> Generated: {payload['generated_at']}",
        "",
        "## Scope",
        "",
        "- Dataset: `datasets/UltraDomain/sage_locked_unseen75_windowed.jsonl`",
        "- Purpose: locked, zero-overlap SAGE holdout check for retrieval baselines and SAGE v33-like repair on the same slice.",
        "- Methods: Naive RAG, HyDE, SAGE contract-completion, and SAGE v33 preservation-delta when available.",
        "",
        "## Completion",
        "",
    ]
    lines.extend(
        md_table(
            ["Method", "Rows", "Errors", "Complete", "Domains", "Families"],
            [
                [
                    method,
                    f"{data['raw_rows']}/{data['expected_n']}",
                    data["aggregate"]["errors"],
                    "yes" if data["complete"] else "no",
                    ", ".join(f"{key}:{value}" for key, value in sorted(data["domain_counts"].items())),
                    ", ".join(f"{key}:{value}" for key, value in sorted(data["family_counts"].items())),
                ]
                for method, data in payload["methods"].items()
            ],
            numeric_cols={2},
        )
    )
    lines.extend([
        "",
        "## Aggregate Metrics",
        "",
    ])
    lines.extend(
        md_table(
            ["Method", "n", "F1", "ROUGE-L", "EM", "Contains", "Runtime(s)", "Median Runtime(s)", "Chunks", "Pred chars"],
            [
                [
                    method,
                    data["aggregate"]["sample_size"],
                    fmt(data["aggregate"]["mean_f1"]),
                    fmt(data["aggregate"]["mean_rouge_l"]),
                    fmt(data["aggregate"]["mean_em"]),
                    fmt(data["aggregate"]["mean_contains"]),
                    fmt(data["aggregate"]["mean_runtime_seconds"], 2),
                    fmt(data["aggregate"]["median_runtime_seconds"], 2),
                    fmt(data["aggregate"]["mean_num_chunks"], 1),
                    fmt(data["aggregate"]["mean_prediction_chars"], 1),
                ]
                for method, data in payload["methods"].items()
            ],
            numeric_cols=set(range(1, 10)),
        )
    )
    lines.extend([
        "",
        "## Paired Deltas",
        "",
    ])
    paired_rows = []
    for comparison, metrics in payload["comparisons"].items():
        left, right = comparison.split("_minus_", 1)
        for metric_name, stats in metrics.items():
            paired_rows.append(
                [
                    f"{left} - {right}",
                    metric_name,
                    stats.get("n"),
                    fmt_delta(stats.get("mean_delta")),
                    stats.get("wins"),
                    stats.get("losses"),
                    stats.get("ties"),
                    fmt_p(stats.get("t_p_value")),
                    fmt_p(stats.get("wilcoxon_p_value")),
                    fmt_delta(stats.get("cohens_d"), 2),
                    f"[{fmt(stats.get('bootstrap_ci', [None, None])[0])}, {fmt(stats.get('bootstrap_ci', [None, None])[1])}]",
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
                    fmt(agg["mean_f1"]),
                    fmt(agg["mean_rouge_l"]),
                    fmt(agg["mean_em"]),
                    fmt(agg["mean_runtime_seconds"], 2),
                ])
        lines.extend(
            md_table(
                ["Method", "Slice", "n", "F1", "ROUGE-L", "EM", "Runtime(s)"],
                slice_rows,
                numeric_cols={2, 3, 4, 5, 6},
            )
        )
    lines.extend([
        "",
        "## Interpretation Notes",
        "",
        "- This summary is lexical only; LLM judge and pairwise preference evaluation should be run separately for answer-quality validation.",
        "- The locked holdout is separate from prior aligned/external slices and is meant to stress unseen source lines and context hashes.",
        "- HyDE generation consumes answer API calls; Naive mostly spends on embeddings and final answer generation with shared chunk cache.",
        "",
    ])
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    metadata = dataset_metadata()
    payload: dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": str(DATASET_PATH.relative_to(ROOT)),
        "expected_n": EXPECTED_N,
        "dataset_rows": len(metadata),
        "methods": {},
        "comparisons": {},
    }
    for method, path in METHODS.items():
        if path.exists():
            payload["methods"][method] = method_payload(method, path, metadata)
    for left, right in COMPARISON_PAIRS:
        if left not in payload["methods"] or right not in payload["methods"]:
            continue
        payload["comparisons"][f"{left}_minus_{right}"] = {
            metric_name: paired_stats(
                payload["methods"][left]["rows"],
                payload["methods"][right]["rows"],
                metric_name,
            )
            for metric_name in ["f1", "rouge_l", "exact_match", "contains", "runtime_seconds"]
        }

    json_payload = json.loads(json.dumps(payload, ensure_ascii=False))
    for data in json_payload["methods"].values():
        data.pop("rows", None)
    OUTPUT_JSON.write_text(json.dumps(json_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(payload)
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {OUTPUT_MD}")


if __name__ == "__main__":
    main()
