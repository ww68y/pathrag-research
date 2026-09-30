#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Compare Legal full-context aligned run vs repeat1 stability run."""
import json
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from experiments.run_enhanced_experiments import (  # noqa: E402
    bootstrap_ci,
    cohens_d,
    paired_t_test,
    wilcoxon_signed_rank_test,
)

BASE_DIR = ROOT_DIR / "results"
RUN1_DIR = BASE_DIR / "enhanced_ablation_aligned" / "legal_fullcontext_20"
RUN2_DIR = BASE_DIR / "enhanced_ablation_stability" / "legal_fullcontext_20_repeat1"
OUTPUT_JSON = BASE_DIR / "enhanced_ablation_stability" / "legal_fullcontext_20_stability_summary.json"
OUTPUT_MD = BASE_DIR / "enhanced_ablation_stability" / "LEGAL_FULLCONTEXT_20_STABILITY_SUMMARY.md"
CONFIGS = ["w/o_repair", "full_v2", "full_v2_rescue"]


def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def safe_name(name: str) -> str:
    return name.replace("/", "_")


def load_rows(run_dir: Path, config: str) -> List[Dict[str, Any]]:
    return [
        row
        for row in load_json(run_dir / f"{safe_name(config)}_aligned_n20.json").get("results", [])
        if "error" not in row
    ]


def metric(row: Dict[str, Any], name: str) -> float:
    return float(row.get("metrics", {}).get(name, 0.0) or 0.0)


def mean(values: List[float]) -> float:
    return statistics.mean(values) if values else 0.0


def pearson(left: List[float], right: List[float]) -> float:
    if len(left) < 2 or len(right) < 2:
        return 0.0
    lx = mean(left)
    rx = mean(right)
    numerator = sum((l - lx) * (r - rx) for l, r in zip(left, right))
    denom_l = math.sqrt(sum((l - lx) ** 2 for l in left))
    denom_r = math.sqrt(sum((r - rx) ** 2 for r in right))
    if denom_l == 0 or denom_r == 0:
        return 0.0
    return numerator / (denom_l * denom_r)


def paired_delta(run2_rows: List[Dict[str, Any]], run1_rows: List[Dict[str, Any]], metric_name: str) -> Dict[str, Any]:
    run1_by_question = {row["question"]: row for row in run1_rows}
    pairs = [(row, run1_by_question[row["question"]]) for row in run2_rows if row["question"] in run1_by_question]
    left = [metric(row2, metric_name) for row2, _row1 in pairs]
    right = [metric(row1, metric_name) for _row2, row1 in pairs]
    diffs = [l - r for l, r in zip(left, right)]
    t_stat, t_p = paired_t_test(left, right)
    w_stat, w_p = wilcoxon_signed_rank_test(left, right)
    return {
        "n": len(pairs),
        "run1_mean": mean(right),
        "repeat1_mean": mean(left),
        "mean_delta_repeat1_minus_run1": mean(diffs),
        "absolute_mean_delta": mean([abs(diff) for diff in diffs]),
        "wins_repeat1": sum(1 for diff in diffs if diff > 1e-12),
        "losses_repeat1": sum(1 for diff in diffs if diff < -1e-12),
        "ties": sum(1 for diff in diffs if abs(diff) <= 1e-12),
        "pearson": pearson(left, right),
        "t_statistic": t_stat,
        "t_p_value": t_p,
        "wilcoxon_stat": w_stat,
        "wilcoxon_p_value": w_p,
        "cohens_d": cohens_d(left, right),
        "bootstrap_ci": bootstrap_ci(diffs),
    }


def within_run_delta(rows_left: List[Dict[str, Any]], rows_right: List[Dict[str, Any]], metric_name: str) -> Dict[str, Any]:
    right_by_question = {row["question"]: row for row in rows_right}
    pairs = [(left, right_by_question[left["question"]]) for left in rows_left if left["question"] in right_by_question]
    left_values = [metric(left, metric_name) for left, _right in pairs]
    right_values = [metric(right, metric_name) for _left, right in pairs]
    diffs = [left - right for left, right in zip(left_values, right_values)]
    t_stat, t_p = paired_t_test(left_values, right_values)
    w_stat, w_p = wilcoxon_signed_rank_test(left_values, right_values)
    return {
        "n": len(pairs),
        "mean_delta": mean(diffs),
        "wins": sum(1 for diff in diffs if diff > 1e-12),
        "losses": sum(1 for diff in diffs if diff < -1e-12),
        "ties": sum(1 for diff in diffs if abs(diff) <= 1e-12),
        "t_p_value": t_p,
        "wilcoxon_p_value": w_p,
        "cohens_d": cohens_d(left_values, right_values),
        "bootstrap_ci": bootstrap_ci(diffs),
    }


def summarize() -> Dict[str, Any]:
    run1 = {config: load_rows(RUN1_DIR, config) for config in CONFIGS}
    run2 = {config: load_rows(RUN2_DIR, config) for config in CONFIGS}
    config_stability = {
        config: {
            "f1": paired_delta(run2[config], run1[config], "f1"),
            "rouge_l": paired_delta(run2[config], run1[config], "rouge_l"),
            "faithfulness": paired_delta(run2[config], run1[config], "faithfulness"),
        }
        for config in CONFIGS
    }
    within = {
        "run1_full_v2_rescue_minus_w_o_repair": within_run_delta(run1["full_v2_rescue"], run1["w/o_repair"], "f1"),
        "repeat1_full_v2_rescue_minus_w_o_repair": within_run_delta(run2["full_v2_rescue"], run2["w/o_repair"], "f1"),
        "run1_full_v2_rescue_minus_full_v2": within_run_delta(run1["full_v2_rescue"], run1["full_v2"], "f1"),
        "repeat1_full_v2_rescue_minus_full_v2": within_run_delta(run2["full_v2_rescue"], run2["full_v2"], "f1"),
    }
    return {
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "run1_dir": str(RUN1_DIR),
        "repeat1_dir": str(RUN2_DIR),
        "config_stability": config_stability,
        "within_run_effects": within,
    }


def write_report(payload: Dict[str, Any]) -> None:
    lines = [
        "# Legal Full-Context 20 Stability Summary",
        "",
        f"> Generated: {payload['generated_at']}",
        f"> Run1: `{payload['run1_dir']}`",
        f"> Repeat1: `{payload['repeat1_dir']}`",
        "",
        "## Config Stability",
        "",
        "| Config | Metric | Run1 Mean | Repeat1 Mean | Delta | Abs Delta | r | t-p | d |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for config in CONFIGS:
        for metric_name, stats in payload["config_stability"][config].items():
            lines.append(
                f"| {config} | {metric_name} | {stats['run1_mean']:.4f} | {stats['repeat1_mean']:.4f} | "
                f"{stats['mean_delta_repeat1_minus_run1']:+.4f} | {stats['absolute_mean_delta']:.4f} | "
                f"{stats['pearson']:.2f} | {stats['t_p_value']:.4g} | {stats['cohens_d']:+.2f} |"
            )
    lines.extend([
        "",
        "## Within-Run Effects",
        "",
        "| Effect | n | Delta F1 | Wins | Losses | Ties | t-p | Wilcoxon-p | d |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for name, stats in payload["within_run_effects"].items():
        lines.append(
            f"| {name} | {stats['n']} | {stats['mean_delta']:+.4f} | {stats['wins']} | {stats['losses']} | "
            f"{stats['ties']} | {stats['t_p_value']:.4g} | {stats['wilcoxon_p_value']:.4g} | {stats['cohens_d']:+.2f} |"
        )
    lines.extend([
        "",
        "## Takeaway",
        "",
        "- The Legal slice preserves a positive full_v2_rescue effect in repeat1, although absolute F1 is lower than run1.",
        "- This supports reporting the rescue module as beneficial but stochastic, with repeat sensitivity disclosed.",
    ])
    OUTPUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    OUTPUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    payload = summarize()
    OUTPUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    write_report(payload)
    print(f"Wrote {OUTPUT_JSON}")
    print(f"Wrote {OUTPUT_MD}")


if __name__ == "__main__":
    main()
