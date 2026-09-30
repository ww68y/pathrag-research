#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VSR-PathRAG Full Experiment Suite
===================================

One-command runner for all paper experiments.

Usage:
    python experiments/run_all_experiments.py --quick     # Quick run (UltraDomain n=30)
    python experiments/run_all_experiments.py --full      # Full run (all datasets)
    python experiments/run_all_experiments.py --paper     # Paper-ready (selected configs)
"""

import argparse
import asyncio
import json
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from experiments.run_standard_benchmark import run_experiment
from experiments.run_ablation_study import run_ablation_study
from experiments.run_pairwise_comparison import run_pairwise_comparison


async def run_quick_suite():
    """Quick experiment suite for testing (UltraDomain only)."""
    print("\n" + "=" * 70)
    print("VSR-PathRAG Quick Experiment Suite")
    print("=" * 70)
    
    all_results = {}
    
    # 1. Benchmark comparison on UltraDomain
    print("\n📋 Step 1: Benchmark Comparison (UltraDomain, n=30)")
    for method in ["naive_rag", "hyde", "vsr_pathrag"]:
        result = await run_experiment(
            dataset_name="ultradomain",
            method=method,
            sample_size=30,
        )
        all_results[f"benchmark_{method}"] = result.get("aggregated", {})
    
    # 2. Pairwise comparison
    print("\n📋 Step 2: Pairwise Comparison (UltraDomain, n=30)")
    pairwise_vsr_vs_hyde = await run_pairwise_comparison(
        "vsr_pathrag", "hyde", "ultradomain", 30
    )
    all_results["pairwise_vsr_vs_hyde"] = pairwise_vsr_vs_hyde
    
    pairwise_vsr_vs_naive = await run_pairwise_comparison(
        "vsr_pathrag", "naive_rag", "ultradomain", 30
    )
    all_results["pairwise_vsr_vs_naive"] = pairwise_vsr_vs_naive
    
    # 3. Ablation study
    print("\n📋 Step 3: Ablation Study (UltraDomain, n=30)")
    ablation = await run_ablation_study("ultradomain", 30)
    all_results["ablation"] = ablation
    
    # Save combined results
    output_dir = ROOT_DIR / "results" / "paper"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "quick_suite_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"\n✅ All results saved to: {output_file}")
    print_experiment_summary(all_results)
    
    return all_results


async def run_full_suite():
    """Full experiment suite for paper."""
    print("\n" + "=" * 70)
    print("VSR-PathRAG Full Experiment Suite (Paper-Ready)")
    print("=" * 70)
    
    all_results = {}
    
    datasets = [
        ("ultradomain", 30),
        ("hotpotqa", 100),
        # ("2wiki", 100),  # Uncomment when 2wiki dataset is available
    ]
    
    # 1. Benchmark on all datasets
    for dataset_name, n in datasets:
        print(f"\n📋 Benchmark: {dataset_name} (n={n})")
        for method in ["naive_rag", "hyde", "vsr_pathrag"]:
            result = await run_experiment(
                dataset_name=dataset_name,
                method=method,
                sample_size=n,
            )
            all_results[f"benchmark_{dataset_name}_{method}"] = result.get("aggregated", {})
    
    # 2. Pairwise comparisons
    for dataset_name, n in datasets:
        print(f"\n📋 Pairwise: VSR vs HyDE on {dataset_name} (n={n})")
        pairwise = await run_pairwise_comparison(
            "vsr_pathrag", "hyde", dataset_name, n
        )
        all_results[f"pairwise_{dataset_name}_vsr_vs_hyde"] = pairwise
    
    # 3. Ablation on UltraDomain (quick) + HotpotQA (if time permits)
    for dataset_name, n in [("ultradomain", 30)]:
        print(f"\n📋 Ablation: {dataset_name} (n={n})")
        ablation = await run_ablation_study(dataset_name, n)
        all_results[f"ablation_{dataset_name}"] = ablation
    
    # Save combined results
    output_dir = ROOT_DIR / "results" / "paper"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "full_suite_results.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"\n✅ All results saved to: {output_file}")
    print_experiment_summary(all_results)
    
    return all_results


def print_experiment_summary(all_results: dict):
    """Print a formatted summary of all experiments."""
    print("\n" + "=" * 70)
    print("Experiment Suite Summary")
    print("=" * 70)
    
    # Benchmark results
    print("\n📊 Benchmark Results:")
    print(f"{'Dataset':<15} {'Method':<20} {'EM':>8} {'F1':>8} {'Contains':>10} {'Faith':>8}")
    print("─" * 70)
    
    for key, result in all_results.items():
        if key.startswith("benchmark_"):
            parts = key.split("_", 2)
            if len(parts) == 3:
                _, dataset, method = parts
            else:
                continue
            
            if isinstance(result, dict):
                em = result.get("mean_exact_match", 0)
                f1 = result.get("mean_f1", 0)
                cont = result.get("mean_contains", 0)
                faith = result.get("mean_faithfulness", 0)
                print(f"{dataset:<15} {method:<20} {em:>8.3f} {f1:>8.3f} {cont:>10.3f} {faith:>8.3f}")
    
    # Pairwise results
    print("\n📊 Pairwise Win Rates:")
    print(f"{'Comparison':<35} {'Avg Win Rate':>15}")
    print("─" * 55)
    
    for key, result in all_results.items():
        if key.startswith("pairwise_") and isinstance(result, dict):
            method_a = result.get("method_a", "?")
            method_b = result.get("method_b", "?")
            win_rate = result.get("average_win_rate_a", 0)
            print(f"{method_a} vs {method_b:<20} {win_rate:>14.2f}%")
    
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="VSR-PathRAG Full Experiment Suite")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--quick", action="store_true", help="Quick run (UltraDomain n=30)")
    group.add_argument("--full", action="store_true", help="Full run (all datasets)")
    group.add_argument("--paper", action="store_true", help="Paper-ready selected configs")
    
    args = parser.parse_args()
    
    if args.full:
        asyncio.run(run_full_suite())
    else:
        # Default to quick
        asyncio.run(run_quick_suite())


if __name__ == "__main__":
    main()
