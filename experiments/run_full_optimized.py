#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run only the 'full' VSR config with optimized selective repair strategy.
Compare against previous full results to measure F1 improvement.
"""
import asyncio
import json
import os
import sys
import time
import statistics
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv
load_dotenv(ROOT_DIR / ".env")

from openai import AsyncOpenAI

# Import from ablation script
from experiments.run_ablation_study_v2 import (
    load_ultradomain, get_api_config, compute_f1, compute_contains, run_vsr_ablation
)


async def main():
    print("=" * 70)
    print("VSR-PathRAG Full Config (Optimized Selective Repair)")
    print("=" * 70)

    config = get_api_config()
    questions = load_ultradomain(15)

    results = []
    for i, q in enumerate(questions):
        print(f"[{i+1}/15] Q: {q['question'][:60]}...")
        start = time.time()
        try:
            answer, trace = await run_vsr_ablation(
                q["question"], q["context"], config, "full"
            )
            elapsed = time.time() - start
            f1 = compute_f1(answer, q["gold_answer"])
            contains = compute_contains(answer, q["gold_answer"])
            faith = trace.get("faithfulness", 0)
            repair = trace.get("repair_triggered", False)
            print(f"  F1={f1:.3f} Cont={contains:.0f} Time={elapsed:.1f}s "
                  f"Faith={faith:.2f} Repair={'Y' if repair else 'N'}")
            results.append({
                "question_id": q["question_id"],
                "prediction": answer,
                "metrics": {
                    "f1": f1, "contains": contains,
                    "runtime_seconds": elapsed,
                    "faithfulness": faith,
                    "repair_triggered": repair,
                },
            })
        except Exception as e:
            elapsed = time.time() - start
            print(f"  ERROR: {e}")
            results.append({
                "question_id": q["question_id"],
                "prediction": f"Error: {e}",
                "metrics": {"f1": 0, "contains": 0, "runtime_seconds": elapsed},
                "error": str(e),
            })

    f1_scores = [r["metrics"]["f1"] for r in results]
    faith_scores = [r["metrics"].get("faithfulness", 0) for r in results]
    repair_count = sum(1 for r in results if r["metrics"].get("repair_triggered"))

    agg = {
        "mean_f1": statistics.mean(f1_scores),
        "std_f1": statistics.stdev(f1_scores) if len(f1_scores) > 1 else 0,
        "mean_faithfulness": statistics.mean(faith_scores),
        "repair_rate": repair_count / max(1, len(results)),
        "sample_size": len(results),
    }

    print(f"\n{'='*70}")
    print(f"Optimized Full Config Results (n=15)")
    print(f"{'='*70}")
    for k, v in agg.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Compare with previous results
    print(f"\n{'='*70}")
    print(f"Comparison: Previous vs Optimized")
    print(f"{'='*70}")
    prev_f1 = 0.2639
    prev_faith = 0.9122
    prev_repair = 0.3333
    print(f"  F1:          {prev_f1:.4f} -> {agg['mean_f1']:.4f} (delta={agg['mean_f1']-prev_f1:+.4f})")
    print(f"  Faithfulness:{prev_faith:.4f} -> {agg['mean_faithfulness']:.4f} (delta={agg['mean_faithfulness']-prev_faith:+.4f})")
    print(f"  Repair%:     {prev_repair:.1%} -> {agg['repair_rate']:.1%}")

    # Save results
    output_dir = ROOT_DIR / "results" / "ablation"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / "full_optimized_n15.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "full_optimized_selective_repair",
            "dataset": "ultradomain",
            "sample_size": 15,
            "aggregated": agg,
            "results": results,
        }, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nResults saved to: {output_file}")


if __name__ == "__main__":
    asyncio.run(main())
