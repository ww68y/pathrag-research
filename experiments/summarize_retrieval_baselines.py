#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Summarize retrieval baselines against aligned VSR-PathRAG outputs."""
import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

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

RESULTS_DIR = ROOT_DIR / "results"
ALIGNED_DIR = RESULTS_DIR / "enhanced_ablation_aligned"
BASELINE_DIR = RESULTS_DIR / "retrieval_baselines"
OUTPUT_JSON = BASELINE_DIR / "retrieval_baseline_aligned_summary.json"
OUTPUT_MD = BASELINE_DIR / "RETRIEVAL_BASELINE_ALIGNED_SUMMARY.md"

DATASETS = [
    {
        "name": "Legal full context",
        "slug": "legal_fullcontext_20",
        "n": 20,
        "aligned_dir": ALIGNED_DIR / "legal_fullcontext_20",
        "baseline_dir": BASELINE_DIR / "legal_fullcontext_20",
    },
    {
        "name": "Biology full context",
        "slug": "biology_fullcontext_50",
        "n": 50,
        "aligned_dir": ALIGNED_DIR / "biology_fullcontext_50",
        "baseline_dir": BASELINE_DIR / "biology_fullcontext_50",
    },
    {
        "name": "Domain expansion windowed",
        "slug": "domain_expansion_windowed_50",
        "n": 50,
        "aligned_dir": ALIGNED_DIR / "domain_expansion_windowed_50",
        "baseline_dir": BASELINE_DIR / "domain_expansion_windowed_50",
    },
]

BASELINE_METHODS = ["naive", "hyde"]
VSR_METHODS = ["w/o_repair", "full_v2", "full_v2_rescue"]


def safe_name(name: str) -> str:
    return name.replace("/", "_")


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def metric(row: Dict[str, Any], name: str, default: float = 0.0) -> float:
    return float(row.get("metrics", {}).get(name, default) or default)


def mean(values: List[float]) -> float:
    return statistics.mean(values) if values else 0.0


def enrich_baseline_rows(rows: List[Dict[str, Any]], method: str) -> List[Dict[str, Any]]:
    enriched = []
    for row in rows:
        gold = row.get("reference_answers", [""])
        gold_answer = gold[0] if isinstance(gold, list) and gold else str(gold or "")
        prediction = row.get("prediction", "")
        runtime = float(row.get("query_seconds", 0.0) or 0.0) + float(row.get("insert_seconds", 0.0) or 0.0)
        enriched.append(
            {
                "question_id": f"line_{row.get('line_no')}",
                "line_no": row.get("line_no"),
                "question": row.get("question", ""),
                "gold_answer": gold_answer,
                "prediction": prediction,
                "method": method,
                "dataset": row.get("dataset", ""),
                "domain": row.get("label", ""),
                "metrics": {
                    "f1": compute_f1(prediction, gold_answer),
                    "rouge_l": compute_rouge_l(prediction, gold_answer),
                    "exact_match": compute_exact_match(prediction, gold_answer),
                    "contains": compute_contains(prediction, gold_answer),
                    "runtime_seconds": runtime,
                    "insert_seconds": float(row.get("insert_seconds", 0.0) or 0.0),
                    "query_seconds": float(row.get("query_seconds", 0.0) or 0.0),
                    "num_chunks": int(row.get("num_chunks", 0) or 0),
                },
                "retrieved_chunks": row.get("retrieved_chunks", []),
                "hyde_query": row.get("hyde_query"),
            }
        )
    return enriched


def load_vsr_rows(dataset: Dict[str, Any], method: str) -> List[Dict[str, Any]]:
    path = dataset["aligned_dir"] / f"{safe_name(method)}_aligned_n{dataset['n']}.json"
    if not path.exists():
        return []
    return [row for row in load_json(path).get("results", []) if "error" not in row]


def summarize_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "sample_size": len(rows),
        "mean_f1": mean([metric(row, "f1") for row in rows]),
        "mean_rouge_l": mean([metric(row, "rouge_l") for row in rows]),
        "mean_em": mean([metric(row, "exact_match") for row in rows]),
        "mean_contains": mean([metric(row, "contains") for row in rows]),
        "mean_runtime_seconds": mean([metric(row, "runtime_seconds") for row in rows]),
        "mean_query_seconds": mean([metric(row, "query_seconds") for row in rows]),
        "mean_num_chunks": mean([metric(row, "num_chunks") for row in rows]),
    }


def by_question(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {row.get("question", ""): row for row in rows}


def paired_stats(left_rows: List[Dict[str, Any]], right_rows: List[Dict[str, Any]], metric_name: str) -> Dict[str, Any]:
    left_by_question = by_question(left_rows)
    right_by_question = by_question(right_rows)
    questions = sorted(set(left_by_question) & set(right_by_question))
    if not questions:
        return {}
    left = [metric(left_by_question[q], metric_name) for q in questions]
    right = [metric(right_by_question[q], metric_name) for q in questions]
    diffs = [l - r for l, r in zip(left, right)]
    t_stat, t_p = paired_t_test(left, right)
    w_stat, w_p = wilcoxon_signed_rank_test(left, right)
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
        "cohens_d": cohens_d(left, right),
        "bootstrap_ci": bootstrap_ci(diffs),
    }


def summarize_dataset(dataset: Dict[str, Any]) -> Dict[str, Any]:
    method_rows: Dict[str, List[Dict[str, Any]]] = {}
    for method in BASELINE_METHODS:
        method_rows[method] = enrich_baseline_rows(
            load_jsonl(dataset["baseline_dir"] / f"{method}_aligned_n{dataset['n']}.jsonl"),
            method,
        )
    for method in VSR_METHODS:
        method_rows[method] = load_vsr_rows(dataset, method)
    available = {method: rows for method, rows in method_rows.items() if rows}
    comparisons = {}
    if method_rows.get("full_v2_rescue"):
        for method in ["naive", "hyde", "w/o_repair", "full_v2"]:
            if method_rows.get(method):
                comparisons[f"full_v2_rescue_minus_{method}"] = {
                    metric_name: paired_stats(method_rows["full_v2_rescue"], method_rows[method], metric_name)
                    for metric_name in ["f1", "rouge_l"]
                }
    return {
        "name": dataset["name"],
        "slug": dataset["slug"],
        "n": dataset["n"],
        "available_methods": sorted(available),
        "aggregated": {method: summarize_rows(rows) for method, rows in available.items()},
        "comparisons": comparisons,
    }


def write_report(payload: Dict[str, Any]) -> None:
    lines = [
        "# Retrieval Baseline Aligned Summary",
        "",
        f"> Generated: {payload['generated_at']}",
        "",
        "## Aggregate Results",
        "",
        "| Dataset | Method | n | F1 | ROUGE-L | EM | Contains | Runtime(s) | Chunks |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for data in payload["datasets"].values():
        for method in ["naive", "hyde", "w/o_repair", "full_v2", "full_v2_rescue"]:
            agg = data.get("aggregated", {}).get(method)
            if not agg:
                continue
            lines.append(
                f"| {data['name']} | {method} | {agg['sample_size']} | {agg['mean_f1']:.4f} | "
                f"{agg['mean_rouge_l']:.4f} | {agg['mean_em']:.4f} | {agg['mean_contains']:.4f} | "
                f"{agg['mean_runtime_seconds']:.1f} | {agg['mean_num_chunks']:.1f} |"
            )
    lines.extend([
        "",
        "## full_v2_rescue Paired Deltas",
        "",
        "| Dataset | Comparison | Metric | n | Delta | Wins | Losses | Ties | t-p | Wilcoxon-p | d |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for data in payload["datasets"].values():
        for comparison, metrics in data.get("comparisons", {}).items():
            for metric_name, stats in metrics.items():
                if not stats:
                    continue
                left, right = comparison.split("_minus_", 1)
                lines.append(
                    f"| {data['name']} | {left} - {right} | {metric_name} | {stats['n']} | "
                    f"{stats['mean_delta']:+.4f} | {stats['wins']} | {stats['losses']} | {stats['ties']} | "
                    f"{stats['t_p_value']:.4g} | {stats['wilcoxon_p_value']:.4g} | {stats['cohens_d']:+.2f} |"
                )
    lines.extend([
        "",
        "## Notes",
        "",
        "- Naive and HyDE use embedding retrieval over chunked source context, then the same LLM for answer generation.",
        "- These are stronger external baselines than direct-context prompts because retrieval must select answer-bearing chunks first.",
        "- Runtime for cached retrieval excludes one-time embedding index creation when the cache already exists.",
    ])
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="*", default=[dataset["slug"] for dataset in DATASETS])
    args = parser.parse_args()
    selected = [dataset for dataset in DATASETS if dataset["slug"] in set(args.datasets)]
    payload = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "datasets": {dataset["slug"]: summarize_dataset(dataset) for dataset in selected},
    }
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(payload)
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {OUTPUT_MD}")


if __name__ == "__main__":
    main()
