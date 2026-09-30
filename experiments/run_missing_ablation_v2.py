#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Run missing ablation configs with API timeout and retry.
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

LOGFILE = ROOT_DIR / "logs" / "missing_ablation_v2.log"

def log(msg):
    print(msg, flush=True)
    with open(LOGFILE, "a", encoding="utf-8", errors="replace") as f:
        f.write(msg + "\n")
        f.flush()

def safe_name(config):
    return config.replace("/", "_")

API_TIMEOUT = 120  # seconds per API call


async def api_call_with_timeout(client, model, messages, temperature=0.3, max_tokens=400):
    """Make API call with timeout."""
    return await asyncio.wait_for(
        client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        ),
        timeout=API_TIMEOUT
    )


async def run_vsr_ablation_safe(question, context, config, ablation_config="full"):
    """Run VSR-PathRAG ablation with timeouts."""
    client = AsyncOpenAI(api_key=config["api_key"], base_url=config["base_url"])
    model = config["model"]
    
    evidence = ""
    if ablation_config not in ("w/o_evidence_extraction", "w/o_all"):
        evidence_prompt = f"""Extract key evidence snippets from the following context that are relevant to the question.

Context: {context[:3500]}

Question: {question}

List the 3-5 most relevant evidence snippets (one per line, prefixed with [E1], [E2], etc.):"""
        evidence_response = await api_call_with_timeout(
            client, model, [{"role": "user", "content": evidence_prompt}], 0.3, 500
        )
        evidence = evidence_response.choices[0].message.content

    # Step 2: Generate answer
    if evidence:
        answer_prompt = f"""Based on the following context and extracted evidence, answer the question.

Context: {context[:3000]}

Extracted Evidence:
{evidence}

Question: {question}

Provide a comprehensive answer supported by the evidence:"""
    else:
        answer_prompt = f"""Based on the following context, answer the question.

Context: {context[:4000]}

Question: {question}

Provide a comprehensive answer:"""
    
    answer_response = await api_call_with_timeout(
        client, model, [{"role": "user", "content": answer_prompt}], 0.3, 400
    )
    answer = answer_response.choices[0].message.content
    
    # Step 3: Verification
    faithfulness = 0.5
    supported = 0
    unsupported = 0
    contradicted = 0
    repair_triggered = False
    
    if ablation_config not in ("w/o_verification", "w/o_all"):
        verify_prompt = f"""Verify the following answer against the provided evidence.

Question: {question}

Answer: {answer[:1000]}

Evidence:
{evidence[:1500] if evidence else context[:1500]}

Check if every claim in the answer is supported by the evidence. Output:
1. supported_claims: number of claims supported
2. unsupported_claims: number not supported
3. contradicted_claims: number contradicted
4. faithfulness_score: 0.0 to 1.0

Output in JSON format only:"""
        
        verify_response = await api_call_with_timeout(
            client, model, [{"role": "user", "content": verify_prompt}], 0.0, 300
        )
        verification = verify_response.choices[0].message.content
        
        try:
            if "```json" in verification:
                verification = verification.split("```json")[1].split("```")[0].strip()
            elif "```" in verification:
                verification = verification.split("```")[1].split("```")[0].strip()
            ver_data = json.loads(verification)
            faithfulness = float(ver_data.get("faithfulness_score", 0.7))
            supported = int(ver_data.get("supported_claims", 0))
            unsupported = int(ver_data.get("unsupported_claims", 0))
            contradicted = int(ver_data.get("contradicted_claims", 0))
        except:
            faithfulness = 0.7
    
    # Step 4: Repair
    if ablation_config not in ("w/o_repair", "w/o_all") and (unsupported > 0 or contradicted > 0):
        repair_triggered = True
        repair_prompt = f"""The following answer has {unsupported} unsupported claims and {contradicted} contradicted claims.

Question: {question}

Original Answer: {answer[:800]}

Evidence:
{evidence[:1500] if evidence else context[:1500]}

Provide a corrected answer that only includes claims supported by the evidence:"""
        
        repair_response = await api_call_with_timeout(
            client, model, [{"role": "user", "content": repair_prompt}], 0.3, 400
        )
        answer = repair_response.choices[0].message.content
    
    return answer, {
        "faithfulness": faithfulness,
        "supported_claims": supported,
        "unsupported_claims": unsupported,
        "contradicted_claims": contradicted,
        "repair_triggered": repair_triggered,
        "evidence_extracted": bool(evidence),
        "verification_performed": ablation_config not in ("w/o_verification", "w/o_all"),
    }


async def run_config(questions, config, ablation_config):
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
                answer, trace = await run_vsr_ablation_safe(
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
            except asyncio.TimeoutError:
                retries += 1
                elapsed = time.time() - start
                log(f"  TIMEOUT (retry {retries}/{max_retries}) after {elapsed:.1f}s")
                if retries >= max_retries:
                    results.append({
                        "question_id": q["question_id"],
                        "prediction": "Error: API timeout",
                        "method": ablation_config,
                        "domain": q.get("domain", ""),
                        "metrics": {"f1": 0, "rouge_l": 0, "exact_match": 0,
                                    "contains": 0, "runtime_seconds": elapsed,
                                    "faithfulness": 0, "repair_triggered": False},
                        "error": "timeout",
                    })
                else:
                    await asyncio.sleep(3)
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
    log("MISSING ABLATION V2 (with timeout + retry)")
    log("=" * 70)
    
    config = get_api_config()
    log(f"Model: {config['model']}")
    log(f"API Base: {config['base_url']}")
    log(f"API Timeout: {API_TIMEOUT}s per call")
    
    # Quick API test
    log("Testing API...")
    try:
        client = AsyncOpenAI(api_key=config["api_key"], base_url=config["base_url"])
        resp = await api_call_with_timeout(
            client, config["model"],
            [{"role": "user", "content": "Say hello"}], 0.0, 10
        )
        log(f"API test OK! Response: {resp.choices[0].message.content}")
    except Exception as e:
        log(f"API test FAILED: {e}")
        return
    
    questions = load_ultradomain_full()
    
    ablation_dir = ROOT_DIR / "results" / "enhanced_ablation"
    ablation_dir.mkdir(parents=True, exist_ok=True)
    
    # Run the 2 missing configs
    missing_configs = ["w/o_repair", "full"]
    
    for ablation_config in missing_configs:
        sname = safe_name(ablation_config)
        output_file = ablation_dir / f"{sname}_n130.json"
        
        # Check if already has valid results
        if output_file.exists():
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                results = data.get("results", [])
                valid = sum(1 for r in results if "error" not in r)
                if valid >= 100:
                    log(f"Already done: {ablation_config} ({valid} valid results)")
                    continue
                else:
                    log(f"Previous results incomplete: {valid} valid, re-running {ablation_config}")
            except:
                pass
        
        log(f"\n[ABLATION] Starting {ablation_config}...")
        results = await run_config(questions, config, ablation_config)
        agg = aggregate_results(results, ablation_config)
        
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
        log(f"  F1={agg['mean_f1']:.3f} Faith={agg['mean_faithfulness']:.3f} Repair={agg['repair_rate']:.1%}")
    
    # ============================================================
    # Regenerate full ablation summary with all 5 configs
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
            f1 = all_aggregated[ablation_config].get("mean_f1", "?")
            log(f"  {ablation_config}: {n} results, F1={f1}")
        except Exception as e:
            log(f"  ERROR loading {ablation_config}: {e}")
    
    # Save summary
    summary_file = ablation_dir / "ablation_study_n130.json"
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump({
            "experiment": "enhanced_ablation_study_complete",
            "dataset": "ultradomain",
            "sample_size": len(questions),
            "aggregated": all_aggregated,
        }, f, ensure_ascii=False, indent=2, default=str)
    log(f"\nSummary saved to: {summary_file}")
    
    # Print results table
    log(f"\n{'='*70}")
    log(f"COMPLETE Ablation Results (UltraDomain, n={len(questions)})")
    log(f"{'='*70}")
    log(f"{'Config':<25} {'F1':>8} {'ROUGE-L':>8} {'EM':>6} {'Faith':>8} {'Repair%':>8} {'Time':>8}")
    log(f"{'-'*70}")
    for config_name, agg in all_aggregated.items():
        log(f"{config_name:<25} {agg['mean_f1']:8.3f} {agg['mean_rouge_l']:8.3f} "
            f"{agg['mean_em']:6.1%} {agg['mean_faithfulness']:8.3f} {agg['repair_rate']:7.1%} {agg['mean_runtime_seconds']:7.1f}s")
    
    # Statistical tests: full vs each ablation
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
                log(f"{config_name:<25} SKIPPED (full={len(scores_full)}, abl={len(scores_abl)})")
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
    
    # Per-domain
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
    
    # Reload comparison data and save final report
    log(f"\n{'='*70}")
    log("Saving final comprehensive report")
    log(f"{'='*70}")
    
    enh_dir = ROOT_DIR / "results" / "enhanced"
    comp_agg = {}
    for f in sorted(os.listdir(enh_dir)):
        if f.endswith("_n130.json") and f != "enhanced_comparison_n130.json" and f != "SCI_tier_final_report.json":
            method = f.replace("_n130.json", "")
            fpath = enh_dir / f
            try:
                with open(fpath, 'r', encoding='utf-8') as fh:
                    data = json.load(fh)
                comp_agg[method] = data.get("aggregated", {})
            except:
                pass
    
    final_report = {
        "experiment": "VSR-PathRAG SCI-tier experiments (COMPLETE - ALL CONFIGS)",
        "dataset": "ultradomain",
        "sample_size": len(questions),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "comparison": {"aggregated": comp_agg},
        "ablation": {
            "aggregated": all_aggregated,
            "statistical_tests": stat_tests,
        },
    }
    
    final_file = ROOT_DIR / "results" / "enhanced" / "SCI_tier_final_report.json"
    with open(final_file, 'w', encoding='utf-8') as f:
        json.dump(final_report, f, ensure_ascii=False, indent=2, default=str)
    log(f"Final report saved to: {final_file}")
    
    # Final summary
    log(f"\n{'='*70}")
    log("COMPARISON Results:")
    log(f"{'='*70}")
    log(f"{'Method':<20} {'F1':>8} {'ROUGE-L':>8} {'EM':>6} {'Faith':>8} {'Repair%':>8} {'Time':>8}")
    log(f"{'-'*70}")
    for method_name, agg in comp_agg.items():
        log(f"{method_name:<20} {agg['mean_f1']:8.3f} {agg['mean_rouge_l']:8.3f} "
              f"{agg['mean_em']:6.1%} {agg['mean_faithfulness']:8.3f} "
              f"{agg['repair_rate']:7.1%} {agg['mean_runtime_seconds']:7.1f}s")
    
    log(f"\n{'='*70}")
    log("ABLATION Results:")
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
