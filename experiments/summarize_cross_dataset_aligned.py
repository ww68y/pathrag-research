#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Summarize aligned ablation results across dataset slices."""
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from experiments.run_aligned_ablation_rerun import add_answer_quality_rates, safe_name
from experiments.run_enhanced_experiments import (
    aggregate_results,
    bootstrap_ci,
    cohens_d,
    paired_t_test,
    wilcoxon_signed_rank_test,
)

BASE_DIR = ROOT_DIR / "results" / "enhanced_ablation_aligned"
DATASETS = [
    {
        "name": "UltraDomain mix",
        "slug": "ultradomain_mix",
        "dir": BASE_DIR,
        "n": 130,
    },
    {
        "name": "Biology full context",
        "slug": "biology_fullcontext_50",
        "dir": BASE_DIR / "biology_fullcontext_50",
        "n": 50,
    },
    {
        "name": "Legal full context",
        "slug": "legal_fullcontext_20",
        "dir": BASE_DIR / "legal_fullcontext_20",
        "n": 20,
    },
    {
        "name": "Domain expansion windowed",
        "slug": "domain_expansion_windowed_50",
        "dir": BASE_DIR / "domain_expansion_windowed_50",
        "n": 50,
    },
]
CONFIGS = ["w/o_repair", "full_v2", "full_v2_rescue"]
COMPARISONS = [
    ("full_v2", "w/o_repair"),
    ("full_v2_rescue", "full_v2"),
    ("full_v2_rescue", "w/o_repair"),
]


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def result_path(dataset: Dict[str, Any], config: str) -> Path:
    return dataset["dir"] / f"{safe_name(config)}_aligned_n{dataset['n']}.json"


def load_rows(dataset: Dict[str, Any], config: str) -> List[Dict[str, Any]]:
    path = result_path(dataset, config)
    if not path.exists():
        return []
    return [row for row in load_json(path).get("results", []) if "error" not in row]


def metric(row: Dict[str, Any], name: str) -> float:
    return float(row.get("metrics", {}).get(name, 0.0) or 0.0)


def paired_delta(
    left_rows: List[Dict[str, Any]],
    right_rows: List[Dict[str, Any]],
    metric_name: str,
) -> Optional[Dict[str, Any]]:
    left_by_id = {row.get("question_id"): row for row in left_rows}
    right_by_id = {row.get("question_id"): row for row in right_rows}
    common_ids = sorted(set(left_by_id) & set(right_by_id))
    if len(common_ids) < 2:
        return None
    left_scores = [metric(left_by_id[qid], metric_name) for qid in common_ids]
    right_scores = [metric(right_by_id[qid], metric_name) for qid in common_ids]
    diffs = [left - right for left, right in zip(left_scores, right_scores)]
    wins = sum(1 for diff in diffs if diff > 1e-12)
    losses = sum(1 for diff in diffs if diff < -1e-12)
    ties = len(diffs) - wins - losses
    t_stat, t_p = paired_t_test(left_scores, right_scores)
    w_stat, w_p = wilcoxon_signed_rank_test(left_scores, right_scores)
    return {
        "n_common": len(common_ids),
        "mean_left": statistics.mean(left_scores),
        "mean_right": statistics.mean(right_scores),
        "mean_delta": statistics.mean(diffs),
        "wins": wins,
        "losses": losses,
        "ties": ties,
        "t_statistic": t_stat,
        "t_p_value": t_p,
        "wilcoxon_stat": w_stat,
        "wilcoxon_p_value": w_p,
        "cohens_d": cohens_d(left_scores, right_scores),
        "bootstrap_ci": bootstrap_ci(diffs),
        "significant_at_0.05": min(t_p, w_p) < 0.05,
    }


def aggregate_for_config(rows: List[Dict[str, Any]], config: str) -> Dict[str, Any]:
    if not rows:
        return {}
    return add_answer_quality_rates(aggregate_results(rows, config), rows)


def summarize() -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "datasets": {},
    }
    for dataset in DATASETS:
        rows_by_config = {config: load_rows(dataset, config) for config in CONFIGS}
        aggregates = {
            config: aggregate_for_config(rows, config)
            for config, rows in rows_by_config.items()
            if rows
        }
        comparisons: Dict[str, Dict[str, Any]] = {}
        for left, right in COMPARISONS:
            left_rows = rows_by_config.get(left, [])
            right_rows = rows_by_config.get(right, [])
            if not left_rows or not right_rows:
                continue
            comparison_key = f"{left}_minus_{right}"
            comparisons[comparison_key] = {}
            for metric_name in ["f1", "rouge_l", "faithfulness"]:
                stats = paired_delta(left_rows, right_rows, metric_name)
                if stats:
                    comparisons[comparison_key][metric_name] = stats
        payload["datasets"][dataset["slug"]] = {
            "name": dataset["name"],
            "target_n": dataset["n"],
            "result_dir": str(dataset["dir"]),
            "aggregated": aggregates,
            "comparisons": comparisons,
        }
    return payload


def pct(value: float) -> str:
    return f"{value:.1%}"


def write_report(payload: Dict[str, Any]) -> Path:
    out_path = BASE_DIR / "CROSS_DATASET_ALIGNED_PROGRESS_SUMMARY.md"
    lines = [
        "# Cross-Dataset Aligned Experiment Progress",
        "",
        f"> Generated: {payload['generated_at']}",
        "",
        "## Aggregate Results",
        "",
        "| Dataset | Config | n | F1 | ROUGE-L | EM | Faithfulness | Repair% | Rescue% | Abstain% |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for dataset in DATASETS:
        data = payload["datasets"].get(dataset["slug"], {})
        aggregates = data.get("aggregated", {})
        if not aggregates:
            lines.append(f"| {dataset['name']} | missing | 0 | N/A | N/A | N/A | N/A | N/A | N/A | N/A |")
            continue
        for config in CONFIGS:
            agg = aggregates.get(config)
            if not agg:
                continue
            lines.append(
                f"| {dataset['name']} | {config} | {int(agg.get('sample_size', 0))} | "
                f"{agg.get('mean_f1', 0):.4f} | {agg.get('mean_rouge_l', 0):.4f} | "
                f"{agg.get('mean_em', 0):.4f} | {agg.get('mean_faithfulness', 0):.4f} | "
                f"{pct(agg.get('repair_rate', 0))} | {pct(agg.get('abstention_rescue_applied_rate', 0))} | "
                f"{pct(agg.get('abstention_language_rate', 0))} |"
            )

    lines.extend(["", "## Paired F1 Deltas", ""])
    lines.extend([
        "| Dataset | Comparison | n | Delta F1 | Wins | Losses | Ties | t-p | Wilcoxon-p | Cohen's d |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for dataset in DATASETS:
        data = payload["datasets"].get(dataset["slug"], {})
        for comparison, metrics in data.get("comparisons", {}).items():
            stats = metrics.get("f1")
            if not stats:
                continue
            left, right = comparison.split("_minus_", 1)
            lines.append(
                f"| {data.get('name', dataset['name'])} | {left} - {right} | {stats['n_common']} | "
                f"{stats['mean_delta']:+.4f} | {stats['wins']} | {stats['losses']} | {stats['ties']} | "
                f"{stats['t_p_value']:.4g} | {stats['wilcoxon_p_value']:.4g} | {stats['cohens_d']:+.2f} |"
            )

    lines.extend([
        "",
        "## Current Takeaways",
        "",
        "- full_v2_rescue remains the deployable candidate when it improves F1 without introducing meta-audit leakage.",
        "- Biology full-context and Legal full-context are independent validation slices, not replacements for the UltraDomain mix N130 result.",
        "- Domain expansion windowed adds a mixed-domain/windowed-context validation slice under the same paired protocol.",
    ])
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def main() -> None:
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    payload = summarize()
    json_path = BASE_DIR / "cross_dataset_aligned_progress_summary.json"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = write_report(payload)
    print(f"Wrote {json_path}")
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
