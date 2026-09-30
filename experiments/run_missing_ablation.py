#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run the 2 missing ablation configs (w/o_repair, full) that failed due to API 402.
Also regenerate the full ablation summary and statistical tests with complete data.
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
load_dotenv(ROOT / ".env") if (ROOT := ROOT_DIR) else None

from openai import AsyncOpenAI
from experiments.run_enhanced_experiments import (
    load_ultradomain_full, get_api_config,
    compute_f1, compute_rouge_l, compute_exact_match, compute_contains,
    aggregate_results, bootstrap_ci, paired_t_test, wilcoxon_signed_rank_test, cohens_d,
)
from experiments.run_enhanced_ablation import run_vsr_ablation

LOGFILE = ROOT_DIR / "logs" / "missing_ablation.log"

def log(msg):
    print(msg, flush=True)
    with open(LOGFILE, "a", encoding="utf-8", errors="replace") as f:
        f.write(msg + "\n")
        f.flush()

def safe_name(config):
    return config.replace("/", "_")


async def run_ablation_config(questions, config, ablation_config):
    sname = safe_name(ablation_config)
    log(f"\n{'='*60}")
    log(f"Ablation: {ablation_config} (file: {sname}) on {len(questions)} questions")
    log(f"{'='*60}")
    
    results = []
    for i, q in enumerate(questions):
        log(f"[{i+1}/{len(questions)}] Q: {q['question'][:60]}...")
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
                log(f"  F1={f1:.3f} R-L={rouge_l:.3f} Time={elapsed:.1f}s Faith={faith:.2f} Repair={'Y' if repair else 'N'}")
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
                log(f"  ERROR (retry {retries}/{max_retries}): {e}")
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
    LOGFILE.write_text("", encoding="utf-8")
    
    log("=" * 70)
    log("MISSING ABLATION RUNNER (w/o_repair + full)")
    log("=" * 70)
    
    config = get_api_config()
    log(f"Model: {config['model']}")
    log(f"API Base: {config['base_url']}")
    
    # Quick API test
    log("Testing API...")
    try:
        client = AsyncOpenAI(api_key=config["api_key"], base_url=config["base_url"])
        resp = await client.chat.completions.create(
            model=config["model"],
            messages=[{"role": "user", "content": "Say hello"}],
            max_tokens=10
        )
        log(f"API test OK! Response: {resp.choices[0].message.content}")
    except Exception as e:
        log(f"API test FAILED: {e}")
        return
    
    questions = load_ultradomain_full()
    
    ablation_dir = ROOT_DIR / "results" / "enhanced_ablation"
    ablation_dir.mkdir(parents=True, exist_ok=True)
    
    # Only run the 2 missing configs
    missing_configs = ["w/o_repair", "full"]
    
    for ablation_config in missing_configs:
        sname = safe_name(ablation_config)
        output_file = ablation_dir / f"{sname}_n130.json"
        
        log(f"\n[ABLATION] Starting {ablation_config}...")
        results = await run_ablation_config(questions, config, ablation_config)
        agg = aggregate_results(results, ablation_config)
        
        # Save with safe filename
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                "ablation": ablation_config,
                "ablation_safe_name": sname,
                "dataset": "ultradomain",
                "sample_size": len(results),
                "aggregated": agg,
                "results": results,
            }, f, ensure_ascii=False, indent=2, default=str)
        log(f"  Saved to: {output_file}")
    
    # ============================================================
    # Now regenerate full ablation summary with all 5 configs
    # ============================================================
    log(f"\n{'='*70}")
    log("Regenerating full ablation summary (all 5 configs)")
    log(f"{'='*70}")
    
    all_configs = ["w/o_all", "w/o_evidence_extraction", "w/o_verification", "w/o_repair", "full"]
    all_results = {}
    all_aggregated = {}
    
    for ablation_config in all_configs:
        sname = safe_name(ablation_config)
        output_file = ablation_dir / f"{sname}_n130.json"
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            all_results[ablation_config] = data.get("results", [])
            all_aggregated[ablation_config] = data.get("aggregated", {})
            n = len(all_results[ablation_config])
            log(f"  Loaded {ablation_config}: {n} results, F1={all_aggregated[ablation_config].get('mean_f1', '?')}")
        except Exception as e:
            log(f"  ERROR loading {ablation_config}: {e}")
    
    # Save summary
    summary_file = ablation_dir / "ablation_study_n130.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump({
            "experiment": "enhanced_ablation_study",
            "dataset": "ultradomain",
            "sample_size": len(questions),
            "aggregated": all_aggregated,
        }, f, ensure_ascii=False, indent=2, default=str)
    log(f"\nAblation summary saved to: {summary_file}")
    
    # Print results table
    log(f"\n{'='*70}")
    log(f"COMPLETE Ablation Results (UltraDomain, n={len(questions)})")
    log(f"{'='*70}")
    log(f"{'Config':<25} {'F1':>8} {'ROUGE-L':>8} {'EM':>6} {'Faith':>8} {'Repair%':>8} {'Time':>8}")
    log(f"{'-'*70}")
    for config_name, agg in all_aggregated.items():
        log(f"{config_name:<25} {agg['mean_f1']:8.3f} {agg['mean_rouge_l']:8.3f} "
            f"{agg['mean_em']:6.1%} {agg['mean_faithfulness']:8.3f} {agg['repair_rate']:7.1%} {agg['mean_runtime_seconds']:7.1f}s")
    
    # ============================================================
    # Statistical tests: full vs each ablation
    # ============================================================
    log(f"\n{'='*70}")
    log("Ablation Statistical Tests (full vs each)")
    log(f"{'='*70}")
    
    stat_tests = {}
    for metric in ["f1", "rouge_l", "faithfulness"]:
        log(f"\n--- {metric} ---")
        log(f"{'Config':<25} {'Diff':>8} {'t-p':>10} {'w-p':>10} {'d':>8} {'Sig?':>6}")
        tests = []
        for config_name in all_configs:
            if config_name == "full":
                continue
            scores_full = [r["metrics"][metric] for r in all_results.get("full", [])]
            scores_abl = [r["metrics"][metric] for r in all_results.get(config_name, [])]
            if len(scores_full) != len(scores_abl) or len(scores_full) < 5:
                log(f"{config_name:<25} SKIPPED (insufficient data: full={len(scores_full)}, abl={len(scores_abl)})")
                continue
            t_stat, t_p = paired_t_test(scores_full, scores_abl)
            w_stat, w_p = wilcoxon_signed_rank_test(scores_full, scores_abl)
            d = cohens_d(scores_full, scores_abl)
            diffs = [s1 - s2 for s1, s2 in zip(scores_full, scores_abl)]
            ci = bootstrap_ci(diffs)
            mean_diff = statistics.mean(diffs)
            sig = "***" if t_p < 0.001 else ("**" if t_p < 0.01 else ("*" if t_p < 0.05 else ""))
            log(f"{config_name:<25} {mean_diff:+8.4f} {t_p:10.4f} {w_p:10.4f} {d:+8.2f} {sig:>6}")
            tests.append({
                "ablation": config_name, "metric": metric,
                "mean_diff": mean_diff,
                "t_statistic": t_stat, "t_p_value": t_p,
                "wilcoxon_stat": w_stat, "wilcoxon_p_value": w_p,
                "cohens_d": d, "bootstrap_ci": ci,
                "significant_at_0.05": min(t_p, w_p) < 0.05,
            })
        stat_tests[metric] = tests
    
    # Save stat tests
    stat_file = ablation_dir / "ablation_stat_tests.json"
    with open(stat_file, 'w', encoding='utf-8') as f:
        json.dump(stat_tests, f, ensure_ascii=False, indent=2, default=str)
    log(f"\nStatistical tests saved to: {stat_file}")
    
    # ============================================================
    # Per-domain analysis
    # ============================================================
    log(f"\n{'='*70}")
    log("Per-Domain Analysis (F1)")
    log(f"{'='*70}")
    domains = sorted(set(q.get("domain", "") for q in questions if q.get("domain", "")))
    header = f"{'Domain':<20}"
    for config_name in all_configs:
        header += f" {safe_name(config_name)[:12]:>13}"
    log(header)
    log("-" * (20 + 14 * len(all_configs)))
    for domain in domains:
        line = f"{domain:<20}"
        for config_name in all_configs:
            domain_results = [r for r in all_results.get(config_name, []) if r.get("domain") == domain]
            if domain_results:
                mean_f1 = statistics.mean(r["metrics"]["f1"] for r in domain_results)
                line += f" {mean_f1:13.3f}"
            else:
                line += f" {'N/A':>13}"
        log(line)
    
    # ============================================================
    # Reload comparison data and regenerate final report
    # ============================================================
    log(f"\n{'='*70}")
    log("Regenerating Final SCI-tier Report")
    log(f"{'='*70}")
    
    enh_dir = ROOT_DIR / "results" / "enhanced"
    comp_results = {}
    comp_agg = {}
    for f in sorted(os.listdir(enh_dir)):
        if f.endswith("_n130.json") and f != "enhanced_comparison_n130.json" and f != "SCI_tier_final_report.json":
            method = f.replace("_n130.json", "")
            fpath = enh_dir / f
            try:
                with open(fpath, 'r', encoding='utf-8') as fh:
                    data = json.load(fh)
                comp_results[method] = data.get("results", [])
                comp_agg[method] = data.get("aggregated", {})
            except:
                pass
    
    log(f"Comparison methods loaded: {list(comp_results.keys())}")
    
    # Comparison stat tests
    comp_methods = list(comp_results.keys())
    comp_stat_tests = {}
    for metric in ["f1", "rouge_l", "faithfulness"]:
        log(f"\n--- Comparison {metric} ---")
        log(f"{'M1':<20} {'M2':<20} {'Diff':>8} {'t-p':>10} {'w-p':>10} {'d':>8} {'Sig?':>6}")
        tests = []
        for i, m1 in enumerate(comp_methods):
            for j, m2 in enumerate(comp_methods):
                if i >= j:
                    continue
                scores1 = [r["metrics"][metric] for r in comp_results[m1]]
                scores2 = [r["metrics"][metric] for r in comp_results[m2]]
                if len(scores1) != len(scores2) or len(scores1) < 5:
                    continue
                t_stat, t_p = paired_t_test(scores1, scores2)
                w_stat, w_p = wilcoxon_signed_rank_test(scores1, scores2)
                d = cohens_d(scores1, scores2)
                diffs = [s1 - s2 for s1, s2 in zip(scores1, scores2)]
                ci = bootstrap_ci(diffs)
                mean_diff = statistics.mean(diffs)
                sig = "***" if t_p < 0.001 else ("**" if t_p < 0.01 else ("*" if t_p < 0.05 else ""))
                log(f"{m1:<20} {m2:<20} {mean_diff:+8.4f} {t_p:10.4f} {w_p:10.4f} {d:+8.2f} {sig:>6}")
                tests.append({
                    "method_1": m1, "method_2": m2, "metric": metric,
                    "mean_diff": mean_diff,
                    "t_statistic": t_stat, "t_p_value": t_p,
                    "wilcoxon_stat": w_stat, "wilcoxon_p_value": w_p,
                    "cohens_d": d, "bootstrap_ci": ci,
                    "significant_at_0.05": min(t_p, w_p) < 0.05,
                })
        comp_stat_tests[metric] = tests
    
    # Save final comprehensive report
    final_report = {
        "experiment": "VSR-PathRAG SCI-tier experiments (COMPLETE)",
        "dataset": "ultradomain",
        "sample_size": len(questions),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "comparison": {
            "aggregated": comp_agg,
            "statistical_tests": comp_stat_tests,
        },
        "ablation": {
            "aggregated": all_aggregated,
            "statistical_tests": stat_tests,
        },
    }
    
    final_file = ROOT_DIR / "results" / "enhanced" / "SCI_tier_final_report.json"
    with open(final_file, 'w', encoding='utf-8') as f:
        json.dump(final_report, f, ensure_ascii=False, indent=2, default=str)
    log(f"\nFinal report saved to: {final_file}")
    
    # Final summary tables
    log(f"\n{'='*70}")
    log(f"COMPARISON Results (UltraDomain, n={len(questions)})")
    log(f"{'='*70}")
    log(f"{'Method':<20} {'F1':>8} {'ROUGE-L':>8} {'EM':>6} {'Faith':>8} {'Repair%':>8} {'Time':>8}")
    log(f"{'-'*70}")
    for method_name, agg in comp_agg.items():
        log(f"{method_name:<20} {agg['mean_f1']:8.3f} {agg['mean_rouge_l']:8.3f} "
              f"{agg['mean_em']:6.1%} {agg['mean_faithfulness']:8.3f} "
              f"{agg['repair_rate']:7.1%} {agg['mean_runtime_seconds']:7.1f}s")
    
    log(f"\n{'='*70}")
    log(f"ABLATION Results (UltraDomain, n={len(questions)})")
    log(f"{'='*70}")
    log(f"{'Config':<25} {'F1':>8} {'ROUGE-L':>8} {'EM':>6} {'Faith':>8} {'Repair%':>8} {'Time':>8}")
    log(f"{'-'*70}")
    for config_name, agg in all_aggregated.items():
        log(f"{config_name:<25} {agg['mean_f1']:8.3f} {agg['mean_rouge_l']:8.3f} "
            f"{agg['mean_em']:6.1%} {agg['mean_faithfulness']:8.3f} {agg['repair_rate']:7.1%} {agg['mean_runtime_seconds']:7.1f}s")
    
    log(f"\n{'='*70}")
    log("ALL EXPERIMENTS COMPLETE!")
    log(f"{'='*70}")


if __name__ == "__main__":
    asyncio.run(main())
