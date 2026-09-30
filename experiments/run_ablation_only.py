#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run ONLY the ablation study (5 configs x 130 questions).
Comparison experiments are already done.
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
from experiments.run_enhanced_experiments import (
    load_ultradomain_full, get_api_config,
    compute_f1, compute_rouge_l, compute_exact_match, compute_contains,
    aggregate_results, bootstrap_ci, paired_t_test, wilcoxon_signed_rank_test, cohens_d,
)
from experiments.run_enhanced_ablation import run_vsr_ablation


def safe_name(config):
    """Convert config name to safe filename."""
    return config.replace("/", "_")


async def run_ablation_config(questions, config, ablation_config):
    """Run ablation config on all questions with retry."""
    sname = safe_name(ablation_config)
    print(f"\n{'='*60}")
    print(f"Ablation: {ablation_config} (file: {sname}) on {len(questions)} questions")
    print(f"{'='*60}")
    
    results = []
    for i, q in enumerate(questions):
        print(f"[{i+1}/{len(questions)}] Q: {q['question'][:60]}...")
        start = time.time()
        retries = 0
        max_retries = 3
        while retries < max_retries:
            try:
                answer, trace = await run_vsr_ablation(
                    q["question"], q["context"], config, ablation_config
                )
                elapsed = time.time() - start
                f1 = compute_f1(answer, q["gold_answer"])
                rouge_l = compute_rouge_l(answer, q["gold_answer"])
                em = compute_exact_match(answer, q["gold_answer"])
                contains = compute_contains(answer, q["gold_answer"])
                faith = trace.get("faithfulness", 0)
                repair = trace.get("repair_triggered", False)
                print(f"  F1={f1:.3f} R-L={rouge_l:.3f} Time={elapsed:.1f}s "
                      f"Faith={faith:.2f} Repair={'Y' if repair else 'N'}")
                results.append({
                    "question_id": q["question_id"],
                    "prediction": answer,
                    "method": ablation_config,
                    "domain": q.get("domain", ""),
                    "metrics": {
                        "f1": f1, "rouge_l": rouge_l, "exact_match": em,
                        "contains": contains, "runtime_seconds": elapsed,
                        "faithfulness": faith, "repair_triggered": repair,
                    },
                })
                break
            except Exception as e:
                retries += 1
                elapsed = time.time() - start
                print(f"  ERROR (retry {retries}/{max_retries}): {e}")
                if retries >= max_retries:
                    results.append({
                        "question_id": q["question_id"],
                        "prediction": f"Error: {e}",
                        "method": ablation_config,
                        "domain": q.get("domain", ""),
                        "metrics": {"f1": 0, "rouge_l": 0, "exact_match": 0,
                                    "contains": 0, "runtime_seconds": elapsed,
                                    "faithfulness": 0, "repair_triggered": False},
                        "error": str(e),
                    })
                else:
                    await asyncio.sleep(5 * retries)
    
    return results


async def main():
    print("=" * 70)
    print("ABLATION STUDY ONLY (5 configs x 130 questions)")
    print("=" * 70)
    
    config = get_api_config()
    print(f"Model: {config['model']}")
    print(f"API Base: {config['base_url']}")
    
    questions = load_ultradomain_full()
    
    ablation_dir = ROOT_DIR / "results" / "enhanced_ablation"
    ablation_dir.mkdir(parents=True, exist_ok=True)
    
    ablation_configs = ["w/o_all", "w/o_evidence_extraction", "w/o_verification", "w/o_repair", "full"]
    all_results = {}
    all_aggregated = {}
    
    for ablation_config in ablation_configs:
        sname = safe_name(ablation_config)
        output_file = ablation_dir / f"{sname}_n130.json"
        
        # Check if already done
        if output_file.exists():
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                n = len(data.get("results", []))
                if n >= 130:
                    all_results[ablation_config] = data.get("results", [])
                    all_aggregated[ablation_config] = data.get("aggregated", {})
                    print(f"Already done: {ablation_config} ({n} results)")
                    continue
            except:
                pass
        
        print(f"\n[ABLATION] Starting {ablation_config}...")
        results = await run_ablation_config(questions, config, ablation_config)
        all_results[ablation_config] = results
        all_aggregated[ablation_config] = aggregate_results(results, ablation_config)
        
        # Save immediately with safe filename
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "ablation": ablation_config,
                "ablation_safe_name": sname,
                "dataset": "ultradomain",
                "sample_size": len(results),
                "aggregated": all_aggregated[ablation_config],
                "results": results,
            }, f, ensure_ascii=False, indent=2, default=str)
        print(f"  Saved to: {output_file}")
        sys.stdout.flush()
    
    # Save summary
    summary_file = ablation_dir / "ablation_study_n130.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump({
            "experiment": "enhanced_ablation_study",
            "dataset": "ultradomain",
            "sample_size": len(questions),
            "aggregated": all_aggregated,
        }, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nAblation summary saved to: {summary_file}")
    
    # Print results table
    print(f"\n{'='*70}")
    print(f"Ablation Results (UltraDomain, n={len(questions)})")
    print(f"{'='*70}")
    print(f"{'Config':<25} {'F1':>8} {'ROUGE-L':>8} {'EM':>6} {'Faith':>8} {'Repair%':>8}")
    print(f"{'-'*70}")
    for config_name, agg in all_aggregated.items():
        print(f"{config_name:<25} {agg['mean_f1']:8.3f} {agg['mean_rouge_l']:8.3f} "
              f"{agg['mean_em']:6.1%} {agg['mean_faithfulness']:8.3f} {agg['repair_rate']:7.1%}")
    
    # Statistical tests: full vs each ablation
    print(f"\n{'='*70}")
    print("Ablation Statistical Tests (full vs each)")
    print(f"{'='*70}")
    
    stat_tests = {}
    for metric in ["f1", "rouge_l", "faithfulness"]:
        print(f"\n--- {metric} ---")
        print(f"{'Config':<25} {'Diff':>8} {'t-p':>10} {'w-p':>10} {'d':>8} {'Sig?':>6}")
        tests = []
        for config_name in ablation_configs:
            if config_name == "full":
                continue
            scores_full = [r["metrics"][metric] for r in all_results["full"]]
            scores_abl = [r["metrics"][metric] for r in all_results[config_name]]
            if len(scores_full) != len(scores_abl) or len(scores_full) < 5:
                continue
            t_stat, t_p = paired_t_test(scores_full, scores_abl)
            w_stat, w_p = wilcoxon_signed_rank_test(scores_full, scores_abl)
            d = cohens_d(scores_full, scores_abl)
            diffs = [s1 - s2 for s1, s2 in zip(scores_full, scores_abl)]
            ci = bootstrap_ci(diffs)
            mean_diff = statistics.mean(diffs)
            sig = "***" if t_p < 0.001 else ("**" if t_p < 0.01 else ("*" if t_p < 0.05 else ""))
            print(f"{config_name:<25} {mean_diff:+8.4f} {t_p:10.4f} {w_p:10.4f} {d:+8.2f} {sig:>6}")
            tests.append({
                "ablation": config_name, "metric": metric,
                "mean_diff": mean_diff,
                "t_statistic": t_stat, "t_p_value": t_p,
                "wilcoxon_stat": w_stat, "wilcoxon_p_value": w_p,
                "cohens_d": d, "bootstrap_ci": ci,
                "significant_at_0.05": min(t_p, w_p) < 0.05,
            })
        stat_tests[metric] = tests
    
    # Save statistical tests
    stat_file = ablation_dir / "ablation_stat_tests.json"
    with open(stat_file, 'w', encoding='utf-8') as f:
        json.dump(stat_tests, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nStatistical tests saved to: {stat_file}")
    
    # Per-domain analysis
    print(f"\n{'='*70}")
    print("Per-Domain Analysis (F1)")
    print(f"{'='*70}")
    domains = sorted(set(q.get("domain", "") for q in questions if q.get("domain", "")))
    print(f"{'Domain':<20}", end="")
    for config_name in ablation_configs:
        print(f" {safe_name(config_name)[:12]:>13}", end="")
    print()
    print("-" * (20 + 14 * len(ablation_configs)))
    for domain in domains:
        print(f"{domain:<20}", end="")
        for config_name in ablation_configs:
            domain_results = [r for r in all_results[config_name] if r.get("domain") == domain]
            if domain_results:
                mean_f1 = statistics.mean(r["metrics"]["f1"] for r in domain_results)
                print(f" {mean_f1:13.3f}", end="")
            else:
                print(f" {'N/A':>13}", end="")
        print()
    
    print(f"\n{'='*70}")
    print("ABLATION STUDY COMPLETE!")
    print(f"{'='*70}")


if __name__ == "__main__":
    asyncio.run(main())
