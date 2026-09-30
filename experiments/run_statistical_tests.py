#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VSR-PathRAG Statistical Significance Testing
=============================================

Performs statistical significance tests on experiment results:
  - Paired t-test for metric comparisons
  - Bootstrap confidence intervals (95% CI)
  - Wilcoxon signed-rank test (non-parametric alternative)
  - Effect size (Cohen's d)

Usage:
    python experiments/run_statistical_tests.py --result_a results/benchmark/ultradomain_vsr_pathrag_full_n30.json --result_b results/benchmark/ultradomain_hyde_full_n30.json
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("⚠️ scipy not installed. Install with: pip install scipy")


def load_results(filepath: str) -> List[Dict[str, Any]]:
    """Load per-question results from a benchmark result file."""
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("per_question_results", [])


def extract_metric_values(results: List[Dict[str, Any]], metric_key: str) -> List[float]:
    """Extract a specific metric from all results."""
    values = []
    for r in results:
        metrics = r.get("metrics", {})
        val = metrics.get(metric_key, 0.0)
        if val is not None:
            values.append(float(val))
    return values


def paired_t_test(a: List[float], b: List[float]) -> Dict[str, Any]:
    """Perform paired t-test."""
    if not SCIPY_AVAILABLE:
        return {"error": "scipy not available"}
    
    if len(a) != len(b) or len(a) < 2:
        return {"error": "Samples must have equal length ≥ 2"}
    
    t_stat, p_value = stats.ttest_rel(a, b)
    
    # Effect size (Cohen's d for paired samples)
    diff = np.array(a) - np.array(b)
    cohens_d = diff.mean() / diff.std() if diff.std() > 0 else 0.0
    
    return {
        "test": "paired_t_test",
        "t_statistic": round(t_stat, 4),
        "p_value": round(p_value, 6),
        "significant_at_0.05": p_value < 0.05,
        "significant_at_0.01": p_value < 0.01,
        "cohens_d": round(cohens_d, 4),
        "effect_size": "large" if abs(cohens_d) >= 0.8 else "medium" if abs(cohens_d) >= 0.5 else "small",
        "mean_a": round(np.mean(a), 4),
        "mean_b": round(np.mean(b), 4),
        "mean_diff": round(np.mean(a) - np.mean(b), 4),
        "std_diff": round(np.std(np.array(a) - np.array(b)), 4),
        "n": len(a),
    }


def bootstrap_ci(
    a: List[float],
    b: List[float],
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
) -> Dict[str, Any]:
    """Compute bootstrap confidence interval for the difference."""
    rng = np.random.default_rng(42)
    n = len(a)
    diffs = np.array(a) - np.array(b)
    
    boot_means = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)
        boot_diff = diffs[indices]
        boot_means.append(boot_diff.mean())
    
    boot_means = np.array(boot_means)
    alpha = 1 - confidence
    lower = np.percentile(boot_means, alpha / 2 * 100)
    upper = np.percentile(boot_means, (1 - alpha / 2) * 100)
    
    return {
        "test": "bootstrap_ci",
        "confidence_level": confidence,
        "mean_diff": round(float(diffs.mean()), 4),
        "ci_lower": round(float(lower), 4),
        "ci_upper": round(float(upper), 4),
        "ci_significant": (lower > 0 or upper < 0),
        "n_bootstrap": n_bootstrap,
        "n": n,
    }


def wilcoxon_test(a: List[float], b: List[float]) -> Dict[str, Any]:
    """Perform Wilcoxon signed-rank test (non-parametric)."""
    if not SCIPY_AVAILABLE:
        return {"error": "scipy not available"}
    
    try:
        stat, p_value = stats.wilcoxon(a, b)
        return {
            "test": "wilcoxon_signed_rank",
            "statistic": round(stat, 4),
            "p_value": round(p_value, 6),
            "significant_at_0.05": p_value < 0.05,
            "significant_at_0.01": p_value < 0.01,
            "n": len(a),
        }
    except Exception as e:
        return {"test": "wilcoxon_signed_rank", "error": str(e)}


def run_all_tests(
    result_a_path: str,
    result_b_path: str,
    metrics: List[str] = None,
) -> Dict[str, Any]:
    """Run all statistical tests on two result files."""
    
    if metrics is None:
        metrics = ["exact_match", "f1", "contains", "faithfulness", "runtime_seconds"]
    
    results_a = load_results(result_a_path)
    results_b = load_results(result_b_path)
    
    # Match by question_id
    a_by_id = {r["question_id"]: r for r in results_a}
    b_by_id = {r["question_id"]: r for r in results_b}
    
    common_ids = sorted(set(a_by_id.keys()) & set(b_by_id.keys()))
    
    if len(common_ids) < 2:
        return {"error": f"Only {len(common_ids)} common questions found"}
    
    print(f"\n{'=' * 70}")
    print(f"Statistical Significance Tests")
    print(f"{'=' * 70}")
    print(f"File A: {result_a_path} ({len(results_a)} results)")
    print(f"File B: {result_b_path} ({len(results_b)} results)")
    print(f"Common questions: {len(common_ids)}")
    print(f"{'=' * 70}\n")
    
    all_tests = {}
    
    for metric in metrics:
        values_a = []
        values_b = []
        
        for qid in common_ids:
            ra = a_by_id[qid].get("metrics", {})
            rb = b_by_id[qid].get("metrics", {})
            
            va = ra.get(metric)
            vb = rb.get(metric)
            
            if va is not None and vb is not None:
                values_a.append(float(va))
                values_b.append(float(vb))
        
        if len(values_a) < 2:
            continue
        
        tests = {
            "paired_t_test": paired_t_test(values_a, values_b),
            "bootstrap_ci": bootstrap_ci(values_a, values_b),
            "wilcoxon": wilcoxon_test(values_a, values_b),
        }
        
        all_tests[metric] = tests
        
        # Print results
        print(f"{'─' * 60}")
        print(f"Metric: {metric}")
        print(f"  Mean A: {tests['paired_t_test'].get('mean_a', 0):.4f}")
        print(f"  Mean B: {tests['paired_t_test'].get('mean_b', 0):.4f}")
        print(f"  Diff:   {tests['paired_t_test'].get('mean_diff', 0):.4f}")
        print(f"  t-test: t={tests['paired_t_test'].get('t_statistic', 0):.3f}, "
              f"p={tests['paired_t_test'].get('p_value', 1):.6f}, "
              f"{'***' if tests['paired_t_test'].get('significant_at_0.01') else '**' if tests['paired_t_test'].get('significant_at_0.05') else 'ns'}")
        print(f"  Cohen's d: {tests['paired_t_test'].get('cohens_d', 0):.3f} "
              f"({tests['paired_t_test'].get('effect_size', '')})")
        print(f"  Bootstrap 95% CI: [{tests['bootstrap_ci'].get('ci_lower', 0):.4f}, "
              f"{tests['bootstrap_ci'].get('ci_upper', 0):.4f}]")
        print(f"  Wilcoxon: p={tests['wilcoxon'].get('p_value', 1):.6f}")
        print()
    
    # Save results
    output_dir = Path(result_a_path).parent.parent / "statistical_tests"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"stat_test_{Path(result_a_path).stem}_vs_{Path(result_b_path).stem}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "file_a": result_a_path,
            "file_b": result_b_path,
            "common_questions": len(common_ids),
            "tests": all_tests,
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Statistical test results saved to: {output_file}")
    
    return all_tests


def main():
    parser = argparse.ArgumentParser(description="Statistical Significance Testing")
    parser.add_argument("--result_a", type=str, required=True,
                        help="Path to first result JSON file")
    parser.add_argument("--result_b", type=str, required=True,
                        help="Path to second result JSON file")
    parser.add_argument("--metrics", type=str, nargs="+", default=None,
                        help="Metrics to test (default: all)")
    
    args = parser.parse_args()
    
    run_all_tests(args.result_a, args.result_b, args.metrics)


if __name__ == "__main__":
    main()
