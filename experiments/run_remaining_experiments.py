#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run remaining experiments: 4 comparison methods + 5 ablation configs.
Skips methods that already have results saved.
"""
import asyncio
import json
import os
import sys
import time
import statistics
from pathlib import Path
from collections import defaultdict

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv
load_dotenv(ROOT_DIR / ".env")

from openai import AsyncOpenAI
from experiments.run_enhanced_experiments import (
    load_ultradomain_full, get_api_config,
    compute_f1, compute_rouge_l, compute_exact_match, compute_contains,
    aggregate_results, bootstrap_ci, paired_t_test, wilcoxon_signed_rank_test, cohens_d,
    run_naive_rag, run_hyde, run_self_rag_sim, run_graphrag_sim, run_vsr_pathrag,
)
from experiments.run_enhanced_ablation import run_vsr_ablation


async def run_method_on_questions(method_name, method_fn, questions, config, extra_args=None):
    """Run a method on all questions sequentially."""
    print(f"\n{'='*60}")
    print(f"Running: {method_name} on {len(questions)} questions")
    print(f"{'='*60}")
    
    results = []
    for i, q in enumerate(questions):
        print(f"[{i+1}/{len(questions)}] Q: {q['question'][:60]}...")
        start = time.time()
        retries = 0
        max_retries = 3
        while retries < max_retries:
            try:
                if extra_args:
                    answer, trace = await method_fn(q["question"], q["context"], config, **extra_args)
                else:
                    answer, trace = await method_fn(q["question"], q["context"], config)
                elapsed = time.time() - start
                
                f1 = compute_f1(answer, q["gold_answer"])
                rouge_l = compute_rouge_l(answer, q["gold_answer"])
                em = compute_exact_match(answer, q["gold_answer"])
                contains = compute_contains(answer, q["gold_answer"])
                faith = trace.get("faithfulness", 0.0)
                repair = trace.get("repair_triggered", False)
                
                print(f"  F1={f1:.3f} R-L={rouge_l:.3f} EM={em:.0f} "
                      f"Time={elapsed:.1f}s Faith={faith:.2f} Repair={'Y' if repair else 'N'}")
                
                results.append({
                    "question_id": q["question_id"],
                    "question": q["question"],
                    "gold_answer": q["gold_answer"],
                    "prediction": answer,
                    "method": method_name,
                    "domain": q.get("domain", ""),
                    "metrics": {
                        "f1": f1, "rouge_l": rouge_l, "exact_match": em,
                        "contains": contains, "runtime_seconds": elapsed,
                        "faithfulness": faith, "repair_triggered": repair,
                        "supported_claims": trace.get("supported_claims", 0),
                        "unsupported_claims": trace.get("unsupported_claims", 0),
                    },
                })
                break
            except Exception as e:
                retries += 1
                elapsed = time.time() - start
                print(f"  ERROR (retry {retries}/{max_retries}): {e}")
                if retries >= max_retries:
                    print(f"  GIVING UP on question {q['question_id']}")
                    results.append({
                        "question_id": q["question_id"],
                        "question": q["question"],
                        "gold_answer": q["gold_answer"],
                        "prediction": f"Error: {e}",
                        "method": method_name,
                        "domain": q.get("domain", ""),
                        "metrics": {
                            "f1": 0, "rouge_l": 0, "exact_match": 0, "contains": 0,
                            "runtime_seconds": elapsed, "faithfulness": 0,
                            "repair_triggered": False,
                            "supported_claims": 0, "unsupported_claims": 0,
                        },
                        "error": str(e),
                    })
                else:
                    await asyncio.sleep(5 * retries)  # Backoff
    
    return results


async def run_ablation_on_questions(questions, config, ablation_config):
    """Run ablation config on all questions."""
    method_name = ablation_config
    print(f"\n{'='*60}")
    print(f"Ablation: {ablation_config} on {len(questions)} questions")
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


def has_results(filepath):
    """Check if a results file exists and has complete data."""
    if not filepath.exists():
        return False
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        results = data.get("results", [])
        return len(results) >= 130
    except:
        return False


async def main():
    print("=" * 70)
    print("REMAINING EXPERIMENTS RUNNER")
    print("=" * 70)
    
    config = get_api_config()
    print(f"Model: {config['model']}")
    print(f"API Base: {config['base_url']}")
    
    if not config["api_key"]:
        print("ERROR: DEEPSEEK_API_KEY not set in .env!")
        return
    
    questions = load_ultradomain_full()
    
    # ============================================================
    # PART 1: Comparison experiment - remaining 4 methods
    # ============================================================
    enhanced_dir = ROOT_DIR / "results" / "enhanced"
    enhanced_dir.mkdir(parents=True, exist_ok=True)
    
    comparison_methods = {
        "graphrag_sim": (run_graphrag_sim, None),
        "self_rag_sim": (run_self_rag_sim, None),
        "vsr_pathrag": (run_vsr_pathrag, None),
        "vsr_pathrag_sr": (run_vsr_pathrag, {"selective_repair": True}),
    }
    
    all_comparison_results = {}
    all_comparison_agg = {}
    
    # Load existing results (naive_rag, hyde)
    for existing_method in ["naive_rag", "hyde"]:
        existing_file = enhanced_dir / f"{existing_method}_n130.json"
        if has_results(existing_file):
            with open(existing_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            all_comparison_results[existing_method] = data.get("results", [])
            all_comparison_agg[existing_method] = data.get("aggregated", aggregate_results(
                all_comparison_results[existing_method], existing_method))
            print(f"Loaded existing: {existing_method} ({len(all_comparison_results[existing_method])} results)")
    
    # Run remaining comparison methods
    for method_name, (method_fn, extra_args) in comparison_methods.items():
        output_file = enhanced_dir / f"{method_name}_n130.json"
        if has_results(output_file):
            with open(output_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            all_comparison_results[method_name] = data.get("results", [])
            all_comparison_agg[method_name] = data.get("aggregated", aggregate_results(
                all_comparison_results[method_name], method_name))
            print(f"Already done: {method_name} ({len(all_comparison_results[method_name])} results)")
            continue
        
        print(f"\n[COMPARISON] Starting {method_name}...")
        results = await run_method_on_questions(
            method_name, method_fn, questions, config, extra_args
        )
        all_comparison_results[method_name] = results
        all_comparison_agg[method_name] = aggregate_results(results, method_name)
        
        # Save immediately
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "method": method_name,
                "dataset": "ultradomain",
                "sample_size": len(results),
                "aggregated": all_comparison_agg[method_name],
                "results": results,
            }, f, ensure_ascii=False, indent=2, default=str)
        print(f"  Saved to: {output_file}")
    
    # Save comparison summary
    comparison_summary = enhanced_dir / "enhanced_comparison_n130.json"
    with open(comparison_summary, 'w', encoding='utf-8') as f:
        json.dump({
            "experiment": "enhanced_comparison",
            "dataset": "ultradomain",
            "sample_size": len(questions),
            "aggregated": all_comparison_agg,
        }, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nComparison summary saved to: {comparison_summary}")
    
    # Print comparison table
    print(f"\n{'='*70}")
    print(f"Comparison Results (UltraDomain, n={len(questions)})")
    print(f"{'='*70}")
    print(f"{'Method':<20} {'F1':>8} {'ROUGE-L':>8} {'EM':>6} {'Faith':>8} {'Repair%':>8} {'Time':>8}")
    print(f"{'-'*70}")
    for method_name, agg in all_comparison_agg.items():
        print(f"{method_name:<20} {agg['mean_f1']:8.3f} {agg['mean_rouge_l']:8.3f} "
              f"{agg['mean_em']:6.1%} {agg['mean_faithfulness']:8.3f} "
              f"{agg['repair_rate']:7.1%} {agg['mean_runtime_seconds']:7.1f}s")
    
    # ============================================================
    # PART 2: Ablation study - 5 configs
    # ============================================================
    ablation_dir = ROOT_DIR / "results" / "enhanced_ablation"
    ablation_dir.mkdir(parents=True, exist_ok=True)
    
    ablation_configs = ["w/o_all", "w/o_evidence_extraction", "w/o_verification", "w/o_repair", "full"]
    all_ablation_results = {}
    all_ablation_agg = {}
    
    for ablation_config in ablation_configs:
        safe_name = ablation_config.replace("/", "_")
        output_file = ablation_dir / f"{safe_name}_n130.json"
        if has_results(output_file):
            with open(output_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            all_ablation_results[ablation_config] = data.get("results", [])
            all_ablation_agg[ablation_config] = data.get("aggregated", aggregate_results(
                all_ablation_results[ablation_config], ablation_config))
            print(f"Already done: {ablation_config} ({len(all_ablation_results[ablation_config])} results)")
            continue
        
        print(f"\n[ABLATION] Starting {ablation_config}...")
        results = await run_ablation_on_questions(questions, config, ablation_config)
        all_ablation_results[ablation_config] = results
        all_ablation_agg[ablation_config] = aggregate_results(results, ablation_config)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "ablation": ablation_config,
                "ablation_safe_name": safe_name,
                "dataset": "ultradomain",
                "sample_size": len(results),
                "aggregated": all_ablation_agg[ablation_config],
                "results": results,
            }, f, ensure_ascii=False, indent=2, default=str)
        print(f"  Saved to: {output_file}")
    
    # Save ablation summary
    ablation_summary = ablation_dir / "ablation_study_n130.json"
    with open(ablation_summary, 'w', encoding='utf-8') as f:
        json.dump({
            "experiment": "enhanced_ablation_study",
            "dataset": "ultradomain",
            "sample_size": len(questions),
            "aggregated": all_ablation_agg,
        }, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nAblation summary saved to: {ablation_summary}")
    
    # Print ablation table
    print(f"\n{'='*70}")
    print(f"Ablation Results (UltraDomain, n={len(questions)})")
    print(f"{'='*70}")
    print(f"{'Config':<25} {'F1':>8} {'ROUGE-L':>8} {'EM':>6} {'Faith':>8} {'Repair%':>8}")
    print(f"{'-'*70}")
    for config_name, agg in all_ablation_agg.items():
        print(f"{config_name:<25} {agg['mean_f1']:8.3f} {agg['mean_rouge_l']:8.3f} "
              f"{agg['mean_em']:6.1%} {agg['mean_faithfulness']:8.3f} {agg['repair_rate']:7.1%}")
    
    # ============================================================
    # PART 3: Statistical significance tests
    # ============================================================
    print(f"\n{'='*70}")
    print("Statistical Significance Tests")
    print(f"{'='*70}")
    
    stat_tests = {}
    
    # Comparison: pairwise tests
    comparison_methods_list = list(all_comparison_results.keys())
    stat_tests["comparison"] = {}
    
    for metric in ["f1", "rouge_l", "faithfulness"]:
        tests = []
        for i, m1 in enumerate(comparison_methods_list):
            for j, m2 in enumerate(comparison_methods_list):
                if i >= j:
                    continue
                scores1 = [r["metrics"][metric] for r in all_comparison_results[m1]]
                scores2 = [r["metrics"][metric] for r in all_comparison_results[m2]]
                if len(scores1) != len(scores2) or len(scores1) < 5:
                    continue
                t_stat, t_p = paired_t_test(scores1, scores2)
                w_stat, w_p = wilcoxon_signed_rank_test(scores1, scores2)
                d = cohens_d(scores1, scores2)
                diffs = [s1 - s2 for s1, s2 in zip(scores1, scores2)]
                ci = bootstrap_ci(diffs)
                tests.append({
                    "method_1": m1, "method_2": m2, "metric": metric,
                    "mean_diff": statistics.mean(diffs),
                    "t_statistic": t_stat, "t_p_value": t_p,
                    "wilcoxon_stat": w_stat, "wilcoxon_p_value": w_p,
                    "cohens_d": d, "bootstrap_ci": ci,
                    "significant_at_0.05": min(t_p, w_p) < 0.05,
                })
                sig = "***" if t_p < 0.001 else ("**" if t_p < 0.01 else ("*" if t_p < 0.05 else ""))
                print(f"  [{metric}] {m1} vs {m2}: diff={statistics.mean(diffs):+.4f} "
                      f"t_p={t_p:.4f} w_p={w_p:.4f} d={d:+.2f} {sig}")
        stat_tests["comparison"][metric] = tests
    
    # Ablation: full vs each ablation
    stat_tests["ablation"] = {}
    for metric in ["f1", "rouge_l", "faithfulness"]:
        tests = []
        for config_name in ablation_configs:
            if config_name == "full":
                continue
            scores_full = [r["metrics"][metric] for r in all_ablation_results["full"]]
            scores_abl = [r["metrics"][metric] for r in all_ablation_results[config_name]]
            if len(scores_full) != len(scores_abl) or len(scores_full) < 5:
                continue
            t_stat, t_p = paired_t_test(scores_full, scores_abl)
            w_stat, w_p = wilcoxon_signed_rank_test(scores_full, scores_abl)
            d = cohens_d(scores_full, scores_abl)
            diffs = [s1 - s2 for s1, s2 in zip(scores_full, scores_abl)]
            ci = bootstrap_ci(diffs)
            tests.append({
                "ablation": config_name, "metric": metric,
                "mean_diff": statistics.mean(diffs),
                "t_statistic": t_stat, "t_p_value": t_p,
                "wilcoxon_stat": w_stat, "wilcoxon_p_value": w_p,
                "cohens_d": d, "bootstrap_ci": ci,
                "significant_at_0.05": min(t_p, w_p) < 0.05,
            })
            sig = "***" if t_p < 0.001 else ("**" if t_p < 0.01 else ("*" if t_p < 0.05 else ""))
            print(f"  [{metric}] full vs {config_name}: diff={statistics.mean(diffs):+.4f} "
                  f"t_p={t_p:.4f} w_p={w_p:.4f} d={d:+.2f} {sig}")
        stat_tests["ablation"][metric] = tests
    
    # ============================================================
    # PART 4: Per-domain analysis
    # ============================================================
    print(f"\n{'='*70}")
    print("Per-Domain Analysis")
    print(f"{'='*70}")
    
    domains = sorted(set(q.get("domain", "") for q in questions if q.get("domain", "")))
    per_domain = {}
    
    print(f"\nComparison (F1 by domain):")
    print(f"{'Domain':<20}", end="")
    for m in comparison_methods_list:
        print(f" {m[:15]:>15}", end="")
    print()
    print("-" * (20 + 16 * len(comparison_methods_list)))
    
    for domain in domains:
        print(f"{domain:<20}", end="")
        per_domain[domain] = {}
        for m in comparison_methods_list:
            domain_results = [r for r in all_comparison_results[m] if r.get("domain") == domain]
            if domain_results:
                mean_f1 = statistics.mean(r["metrics"]["f1"] for r in domain_results)
                per_domain[domain][m] = mean_f1
                print(f" {mean_f1:15.3f}", end="")
            else:
                print(f" {'N/A':>15}", end="")
        print()
    
    # ============================================================
    # PART 5: Error analysis
    # ============================================================
    print(f"\n{'='*70}")
    print("Error Analysis")
    print(f"{'='*70}")
    
    # Find worst and best cases for VSR
    vsr_results = all_comparison_results.get("vsr_pathrag", [])
    if vsr_results:
        sorted_vsr = sorted(vsr_results, key=lambda r: r["metrics"]["f1"])
        print("\nWorst 5 VSR-PathRAG cases:")
        for r in sorted_vsr[:5]:
            print(f"  F1={r['metrics']['f1']:.3f} Q: {r['question_id']} - {r.get('question', '')[:80]}")
            print(f"    Gold: {r.get('gold_answer', '')[:80]}")
            print(f"    Pred: {r['prediction'][:80]}")
        
        print("\nBest 5 VSR-PathRAG cases:")
        for r in sorted_vsr[-5:]:
            print(f"  F1={r['metrics']['f1']:.3f} Q: {r['question_id']} - {r.get('question', '')[:80]}")
    
    # Error rate by method
    print(f"\nError rates by method:")
    for m, results in all_comparison_results.items():
        errors = sum(1 for r in results if "error" in r)
        print(f"  {m}: {errors}/{len(results)} errors ({errors/len(results)*100:.1f}%)")
    
    # ============================================================
    # Save final comprehensive report
    # ============================================================
    final_report = {
        "experiment": "VSR-PathRAG SCI-tier experiments",
        "dataset": "ultradomain",
        "sample_size": len(questions),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "comparison": {
            "aggregated": all_comparison_agg,
            "statistical_tests": stat_tests.get("comparison", {}),
        },
        "ablation": {
            "aggregated": all_ablation_agg,
            "statistical_tests": stat_tests.get("ablation", {}),
        },
        "per_domain": per_domain,
    }
    
    final_file = ROOT_DIR / "results" / "enhanced" / "SCI_tier_final_report.json"
    with open(final_file, 'w', encoding='utf-8') as f:
        json.dump(final_report, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nFinal report saved to: {final_file}")
    
    print(f"\n{'='*70}")
    print("ALL EXPERIMENTS COMPLETE!")
    print(f"{'='*70}")


if __name__ == "__main__":
    asyncio.run(main())
