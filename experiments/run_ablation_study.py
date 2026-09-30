#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VSR-PathRAG Ablation Study
===========================

Systematically evaluates the contribution of each VSR component by
disabling them one at a time and measuring performance impact.

Ablation configurations:
  full          - All VSR modules active
  w/o_mps       - Remove Minimal Proof Subgraph (V78)
  w/o_cpn       - Remove Counterfactual Path Necessity (V79)
  w/o_rcs       - Remove Risk-Calibrated Submission (V80)
  w/o_repair    - Remove Answer Repair modules (V85, V86)
  w/o_obligation- Remove Question Obligation Compiler (V77)
  w/o_memory    - Remove Cross-Query Evidence Memory (V99)

Usage:
    python experiments/run_ablation_study.py --dataset ultradomain --sample_size 30
    python experiments/run_ablation_study.py --dataset hotpotqa --sample_size 100
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, List, Any

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from experiments.run_standard_benchmark import (
    run_experiment,
    aggregate_metrics,
    load_ultradomain,
    load_hotpotqa,
)


ABLATION_CONFIGS = [
    ("full",           "All VSR modules active (full pipeline)"),
    ("w/o_mps",        "Remove Minimal Proof Subgraph optimizer (V78)"),
    ("w/o_cpn",        "Remove Counterfactual Path Necessity certification (V79)"),
    ("w/o_rcs",        "Remove Risk-Calibrated Submission policy (V80)"),
    ("w/o_repair",     "Remove Answer Repair modules (V85-V86)"),
    ("w/o_obligation", "Remove Question Obligation Compiler (V77)"),
    ("w/o_memory",     "Remove Cross-Query Evidence Memory (V99)"),
]


async def run_ablation_study(
    dataset_name: str = "ultradomain",
    sample_size: int = 30,
) -> Dict[str, Any]:
    """Run full ablation study across all configurations."""
    
    print("\n" + "=" * 70)
    print("VSR-PathRAG Ablation Study")
    print("=" * 70)
    print(f"Dataset: {dataset_name}")
    print(f"Sample size: {sample_size}")
    print(f"Configurations: {len(ABLATION_CONFIGS)}")
    print("=" * 70 + "\n")
    
    all_results = {}
    
    for config_name, description in ABLATION_CONFIGS:
        print(f"\n{'─' * 60}")
        print(f"Running: {config_name}")
        print(f"Description: {description}")
        print(f"{'─' * 60}")
        
        result = await run_experiment(
            dataset_name=dataset_name,
            method="vsr_pathrag",
            sample_size=sample_size,
            ablation_config=config_name,
        )
        
        all_results[config_name] = {
            "description": description,
            "aggregated": result.get("aggregated", {}),
            "results_count": len(result.get("results", [])),
        }
    
    # Generate comparison table
    print("\n" + "=" * 70)
    print("Ablation Study Results Comparison")
    print("=" * 70)
    
    # Header
    metrics_to_compare = [
        ("mean_exact_match", "EM"),
        ("mean_f1", "F1"),
        ("mean_contains", "Contains"),
        ("mean_faithfulness", "Faith"),
        ("proof_sufficiency_rate", "Proof%"),
        ("unsupported_claim_rate", "Unsupp%"),
        ("mean_calibrated_risk", "Risk"),
        ("mean_runtime_seconds", "Time(s)"),
    ]
    
    header = f"{'Config':<20}"
    for _, label in metrics_to_compare:
        header += f" {label:>10}"
    print(header)
    print("─" * (20 + 11 * len(metrics_to_compare)))
    
    for config_name, _ in ABLATION_CONFIGS:
        agg = all_results[config_name]["aggregated"]
        row = f"{config_name:<20}"
        for metric_key, _ in metrics_to_compare:
            value = agg.get(metric_key, 0.0)
            if isinstance(value, float):
                row += f" {value:>10.3f}"
            else:
                row += f" {str(value):>10}"
        print(row)
    
    print("=" * 70)
    
    # Compute deltas from full
    print("\nDelta from Full Configuration:")
    print("─" * 60)
    full_agg = all_results["full"]["aggregated"]
    
    for config_name, _ in ABLATION_CONFIGS:
        if config_name == "full":
            continue
        agg = all_results[config_name]["aggregated"]
        
        f1_delta = agg.get("mean_f1", 0) - full_agg.get("mean_f1", 0)
        proof_delta = agg.get("proof_sufficiency_rate", 0) - full_agg.get("proof_sufficiency_rate", 0)
        
        print(f"  {config_name:<20}: "
              f"ΔF1={f1_delta:+.3f}  "
              f"ΔProof={proof_delta:+.3f}")
    print("─" * 60)
    
    # Save full results
    output_dir = ROOT_DIR / "results" / "ablation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"ablation_{dataset_name}_n{sample_size}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "dataset": dataset_name,
            "sample_size": sample_size,
            "configurations": ABLATION_CONFIGS,
            "results": all_results,
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Full ablation results saved to: {output_file}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="VSR-PathRAG Ablation Study")
    parser.add_argument("--dataset", type=str, default="ultradomain",
                        choices=["hotpotqa", "2wiki", "ultradomain"])
    parser.add_argument("--sample_size", type=int, default=30)
    
    args = parser.parse_args()
    
    asyncio.run(run_ablation_study(
        dataset_name=args.dataset,
        sample_size=args.sample_size,
    ))


if __name__ == "__main__":
    main()
