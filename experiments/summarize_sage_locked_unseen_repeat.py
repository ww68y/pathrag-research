#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Summarize a locked unseen SAGE repeat holdout.

This is an offline consolidation script. It reads completed Naive, HyDE,
SAGE contract-completion, SAGE v33, judge, and pairwise artifacts from a
repeat holdout directory and writes JSON/Markdown summaries. It does not call
any API provider.
"""
from __future__ import annotations

import argparse
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


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for file_line_no, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            item.setdefault("_file_line_no", file_line_no)
            rows.append(item)
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


def fmt_pct(value: Any, digits: int = 2) -> str:
    if value is None:
        return "N/A"
    try:
        return f"{float(value) * 100:.{digits}f}%"
    except (TypeError, ValueError):
        return str(value)


def md_table(headers: list[str], rows: Iterable[Iterable[Any]], numeric_cols: set[int] | None = None) -> list[str]:
    numeric_cols = numeric_cols or set()
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join("---:" if idx in numeric_cols else "---" for idx in range(len(headers))) + "|")
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return lines


def dataset_metadata(dataset_path: Path) -> dict[str, dict[str, Any]]:
    metadata: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(dataset_path):
        question = str(row.get("input") or row.get("question") or "")
        if not question:
            continue
        metadata[question] = {
            "domain": row.get("label") or row.get("dataset") or row.get("source_dataset") or "unknown",
            "family": row.get("locked_holdout_family") or "unknown",
            "source_line_no": row.get("source_line_no"),
            "source_context_hash": row.get("source_context_hash"),
        }
    return metadata


def score_row(row: dict[str, Any], method: str, metadata: dict[str, dict[str, Any]]) -> dict[str, Any]:
    question = str(row.get("question") or row.get("input") or "")
    refs = references(row)
    pred = prediction(row)
    if refs and pred and not row.get("error"):
        metrics = {
            "f1": max(compute_f1(pred, ref) for ref in refs),
            "rouge_l": max(compute_rouge_l(pred, ref) for ref in refs),
            "exact_match": max(compute_exact_match(pred, ref) for ref in refs),
            "contains": max(compute_contains(pred, ref) for ref in refs),
        }
    else:
        metrics = {"f1": 0.0, "rouge_l": 0.0, "exact_match": 0.0, "contains": 0.0}

    meta = metadata.get(question, {})
    query_seconds = float(row.get("query_seconds") or 0.0)
    insert_seconds = float(row.get("insert_seconds") or 0.0)
    return {
        "method": method,
        "question": question,
        "line_no": row.get("line_no"),
        "domain": row.get("label") or row.get("dataset") or meta.get("domain") or "unknown",
        "family": meta.get("family") or "unknown",
        "error": row.get("error"),
        "metrics": {
            **metrics,
            "runtime_seconds": query_seconds + insert_seconds,
            "query_seconds": query_seconds,
            "prediction_chars": len(pred),
        },
    }


def aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "sample_size": len(rows),
        "errors": sum(1 for row in rows if row.get("error")),
        "mean_f1": mean(row["metrics"]["f1"] for row in rows),
        "mean_rouge_l": mean(row["metrics"]["rouge_l"] for row in rows),
        "mean_exact_match": mean(row["metrics"]["exact_match"] for row in rows),
        "mean_contains": mean(row["metrics"]["contains"] for row in rows),
        "mean_runtime_seconds": mean(row["metrics"]["runtime_seconds"] for row in rows),
        "mean_prediction_chars": mean(row["metrics"]["prediction_chars"] for row in rows),
    }


def paired_key(row: dict[str, Any]) -> tuple[str, Any] | None:
    line_no = row.get("line_no")
    if line_no is not None:
        try:
            return "line_no", int(line_no)
        except (TypeError, ValueError):
            return "line_no", str(line_no)
    question = row.get("question")
    if question:
        return "question", question
    return None


def paired_stats(left: list[dict[str, Any]], right: list[dict[str, Any]], metric: str) -> dict[str, Any]:
    right_by_key = {key: row for row in right if (key := paired_key(row)) is not None}
    pairs = [(row, right_by_key[key]) for row in left if (key := paired_key(row)) in right_by_key]
    left_values = [a["metrics"][metric] for a, _ in pairs]
    right_values = [b["metrics"][metric] for _, b in pairs]
    deltas = [a - b for a, b in zip(left_values, right_values)]
    wins = sum(1 for delta in deltas if delta > 1e-12)
    losses = sum(1 for delta in deltas if delta < -1e-12)
    ties = len(deltas) - wins - losses
    return {
        "n": len(deltas),
        "mean_delta": mean(deltas),
        "t_p": paired_t_test(left_values, right_values)[1] if deltas else None,
        "wilcoxon_p": wilcoxon_signed_rank_test(left_values, right_values)[1] if deltas else None,
        "cohens_d": cohens_d(left_values, right_values) if deltas else None,
        "bootstrap_ci": bootstrap_ci(deltas) if deltas else None,
        "wins": wins,
        "losses": losses,
        "ties": ties,
    }


def judge_summary(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    score_keys = ["correctness", "coverage", "relevance", "conciseness", "overall"]
    return {
        "n": len(rows),
        "means": {key: mean(float((row.get("scores") or {}).get(key, 0)) for row in rows) for key in score_keys},
        "verdict_counts": dict(Counter((row.get("scores") or {}).get("verdict", "unknown") for row in rows)),
    }


def pairwise_summary(path: Path) -> dict[str, Any]:
    rows = read_jsonl(path)
    dimensions = ["comprehensiveness", "diversity", "logicality", "relevance", "coherence"]
    dimension_rows = []
    rates = []
    has_rows = bool(rows)
    for dimension in dimensions:
        target_wins = sum(1 for row in rows if (row.get("winners") or {}).get(dimension) == "pathrag")
        baseline_wins = sum(1 for row in rows if (row.get("winners") or {}).get(dimension) == "baseline")
        ties = sum(1 for row in rows if (row.get("winners") or {}).get(dimension) == "tie")
        rate = (target_wins + 0.5 * ties) / len(rows) if has_rows else None
        if rate is not None:
            rates.append(rate)
        dimension_rows.append({
            "dimension": dimension,
            "target_wins": target_wins,
            "baseline_wins": baseline_wins,
            "ties": ties,
            "target_win_rate": rate,
        })
    return {"n": len(rows), "avg_target_win_rate": mean(rates), "dimensions": dimension_rows}


def judge_overall_delta(left: dict[str, Any], right: dict[str, Any]) -> float | None:
    if int(left.get("n") or 0) == 0 or int(right.get("n") or 0) == 0:
        return None
    left_overall = (left.get("means") or {}).get("overall")
    right_overall = (right.get("means") or {}).get("overall")
    if left_overall is None or right_overall is None:
        return None
    return float(left_overall) - float(right_overall)


def guard_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    contract = Counter()
    guard = Counter()
    for row in rows:
        cc = row.get("contract_completion_agent") or {}
        if cc.get("risk_candidate"):
            contract["risk_candidates"] += 1
        if cc.get("verification_attempted"):
            contract["contract_verifications"] += 1
        if cc.get("repair_needed"):
            contract["repairs_requested"] += 1
        if cc.get("repair_attempted"):
            contract["repairs_attempted"] += 1
        if cc.get("repair_accepted"):
            contract["repairs_accepted"] += 1
        action = (row.get("evidence_delta_guard") or {}).get("action") or "missing"
        guard[action] += 1
    return {"contract_completion": dict(contract), "v33_guard_actions": dict(guard)}


def build_summary(args: argparse.Namespace) -> dict[str, Any]:
    out_dir = ROOT / args.output_dir
    dataset_path = ROOT / args.dataset
    metadata = dataset_metadata(dataset_path)
    suffix = args.file_suffix
    files = {
        "naive": out_dir / f"naive_aligned_{suffix}.jsonl",
        "hyde": out_dir / f"hyde_aligned_{suffix}.jsonl",
        "sage_v27_contract_completion": out_dir / "hyde_contract_completion_text_only.jsonl",
        "sage_v33_preservation_delta": out_dir / "hyde_evidence_delta_v33_text_only.jsonl",
    }
    scored = {
        name: [score_row(row, name, metadata) for row in read_jsonl(path)]
        for name, path in files.items()
    }
    judges = {
        "naive": judge_summary(out_dir / f"naive_aligned_{suffix}_eval.jsonl"),
        "hyde": judge_summary(out_dir / f"hyde_aligned_{suffix}_eval.jsonl"),
        "sage_v33_preservation_delta": judge_summary(out_dir / "hyde_evidence_delta_v33_text_only_eval.jsonl"),
    }
    pairwise = {
        "sage_v33_vs_hyde": pairwise_summary(out_dir / "pairwise_sage_v33_vs_hyde.jsonl"),
        "sage_v33_vs_naive": pairwise_summary(out_dir / "pairwise_sage_v33_vs_naive.jsonl"),
    }
    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset": args.dataset,
        "output_dir": args.output_dir,
        "expected_n": args.expected_n,
        "title": args.title,
        "file_suffix": suffix,
        "method_files": {name: str(path.relative_to(ROOT)) for name, path in files.items()},
        "method_aggregates": {name: aggregate(rows) for name, rows in scored.items()},
        "paired": {
            "sage_v33_vs_hyde_f1": paired_stats(scored["sage_v33_preservation_delta"], scored["hyde"], "f1"),
            "sage_v33_vs_naive_f1": paired_stats(scored["sage_v33_preservation_delta"], scored["naive"], "f1"),
            "sage_v33_vs_hyde_rouge_l": paired_stats(scored["sage_v33_preservation_delta"], scored["hyde"], "rouge_l"),
            "sage_v33_vs_naive_rouge_l": paired_stats(scored["sage_v33_preservation_delta"], scored["naive"], "rouge_l"),
        },
        "judge_summaries": judges,
        "judge_deltas": {
            "sage_v33_vs_hyde_overall": judge_overall_delta(judges["sage_v33_preservation_delta"], judges["hyde"]),
            "sage_v33_vs_naive_overall": judge_overall_delta(judges["sage_v33_preservation_delta"], judges["naive"]),
        },
        "pairwise": pairwise,
        "guard_summary": guard_summary(read_jsonl(files["sage_v33_preservation_delta"])),
    }


def write_markdown(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        f"# {payload.get('title') or 'SAGE Locked Unseen Repeat Summary'}",
        "",
        f"> Generated: {payload['generated_at']}",
        f"> Dataset: `{payload['dataset']}`",
        "",
        "## Lexical Metrics",
        "",
    ]
    method_rows = []
    for name, aggregate_row in payload["method_aggregates"].items():
        method_rows.append([
            name,
            aggregate_row["sample_size"],
            aggregate_row["errors"],
            fmt(aggregate_row["mean_f1"]),
            fmt(aggregate_row["mean_rouge_l"]),
            fmt(aggregate_row["mean_runtime_seconds"], 1),
            fmt(aggregate_row["mean_prediction_chars"], 1),
        ])
    lines.extend(md_table(["Method", "N", "Errors", "F1", "ROUGE-L", "Runtime s/q", "Avg Chars"], method_rows, {1, 2, 3, 4, 5, 6}))

    lines.extend(["", "## Paired Lexical Tests", ""])
    paired_rows = []
    for name, row in payload["paired"].items():
        ci = row.get("bootstrap_ci") or [None, None]
        paired_rows.append([
            name,
            row["n"],
            fmt_delta(row["mean_delta"]),
            fmt_p(row["t_p"]),
            fmt_p(row["wilcoxon_p"]),
            fmt_delta(row["cohens_d"], 2),
            f"{row['wins']} / {row['losses']} / {row['ties']}",
            f"[{fmt(ci[0])}, {fmt(ci[1])}]",
        ])
    lines.extend(md_table(["Comparison", "N", "Delta", "t-p", "Wilcoxon-p", "d", "W/L/T", "95% CI"], paired_rows, {1, 2, 3, 4, 5}))

    lines.extend(["", "## Judge Metrics", ""])
    judge_rows = []
    for name, row in payload["judge_summaries"].items():
        judge_rows.append([
            name,
            row["n"],
            fmt(row["means"]["correctness"], 3),
            fmt(row["means"]["coverage"], 3),
            fmt(row["means"]["relevance"], 3),
            fmt(row["means"]["conciseness"], 3),
            fmt(row["means"]["overall"], 3),
            row["verdict_counts"],
        ])
    lines.extend(md_table(["Method", "N", "Correct", "Coverage", "Relevance", "Concise", "Overall", "Verdicts"], judge_rows, {1, 2, 3, 4, 5, 6}))
    lines.append("")
    lines.append(f"- Judge overall delta vs HyDE: `{fmt_delta(payload['judge_deltas']['sage_v33_vs_hyde_overall'], 3)}`")
    lines.append(f"- Judge overall delta vs Naive: `{fmt_delta(payload['judge_deltas']['sage_v33_vs_naive_overall'], 3)}`")

    lines.extend(["", "## Pairwise A/B", ""])
    pair_rows = []
    for name, row in payload["pairwise"].items():
        pair_rows.append([name, row["n"], fmt_pct(row["avg_target_win_rate"])])
    lines.extend(md_table(["Comparison", "N", "Avg Target Win Rate"], pair_rows, {1, 2}))

    lines.extend(["", "## Guard Activity", ""])
    guard = payload["guard_summary"]
    lines.append(f"- Contract-completion: `{guard['contract_completion']}`")
    lines.append(f"- V33 guard actions: `{guard['v33_guard_actions']}`")
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- This repeat is a new locked zero-overlap mini-holdout; it should be treated as supplementary robustness evidence, not as a replacement for locked unseen75.")
    lines.append("- All metrics are computed offline from completed artifacts; this summary does not call any API provider.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="datasets/UltraDomain/sage_locked_unseen25_repeat1_windowed.jsonl")
    parser.add_argument("--output-dir", default="results/historical_sage/sage_locked_unseen25_repeat1")
    parser.add_argument("--expected-n", type=int, default=25)
    parser.add_argument("--file-suffix", default="n25")
    parser.add_argument("--title", default="SAGE Locked Unseen25 Repeat1 Summary")
    parser.add_argument("--summary-json", default="results/historical_sage/sage_locked_unseen25_repeat1_summary.json")
    parser.add_argument("--summary-md", default="results/historical_sage/SAGE_LOCKED_UNSEEN25_REPEAT1_SUMMARY.md")
    args = parser.parse_args()

    payload = build_summary(args)
    json_path = ROOT / args.summary_json
    md_path = ROOT / args.summary_md
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_markdown(md_path, payload)
    print(f"Wrote {json_path}")
    print(f"Wrote {md_path}")


if __name__ == "__main__":
    main()
