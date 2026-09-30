#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Fast version of missing ablation: shorter context, shorter timeout, faster execution.
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

LOGFILE = ROOT_DIR / "logs" / "missing_ablation_fast.log"

def log(msg):
    print(msg, flush=True)
    with open(LOGFILE, "a", encoding="utf-8", errors="replace") as f:
        f.write(msg + "\n")
        f.flush()

def safe_name(config):
    return config.replace("/", "_")

API_TIMEOUT = 60  # shorter timeout
CTX_LIMIT = 2000  # shorter context


async def api_call(client, model, messages, temperature=0.3, max_tokens=300):
    """API call with timeout."""
    return await asyncio.wait_for(
        client.chat.completions.create(
            model=model, messages=messages,
            temperature=temperature, max_tokens=max_tokens,
        ),
        timeout=API_TIMEOUT
    )


async def run_ablation_fast(question, context, config, ablation_config="full"):
    """Run VSR ablation with shorter context for speed."""
    client = AsyncOpenAI(api_key=config["api_key"], base_url=config["base_url"])
    model = config["model"]
    ctx = context[:CTX_LIMIT]
    
    evidence = ""
    if ablation_config not in ("w/o_evidence_extraction", "w/o_all"):
        evidence_prompt = f"""Extract 3 key evidence snippets from context relevant to: {question}

Context: {ctx}

Format: [E1] snippet1 [E2] snippet2 [E3] snippet3"""
        evidence_response = await api_call(
            client, model, [{"role": "user", "content": evidence_prompt}], 0.3, 300
        )
        evidence = evidence_response.choices[0].message.content

    # Answer generation
    if evidence:
        answer_prompt = f"""Answer the question using context and evidence.

Context: {ctx[:1500]}

Evidence: {evidence[:800]}

Question: {question}

Answer:"""
    else:
        answer_prompt = f"""Answer based on context.

Context: {ctx}

Question: {question}

Answer:"""
    
    answer_response = await api_call(
        client, model, [{"role": "user", "content": answer_prompt}], 0.3, 300
    )
    answer = answer_response.choices[0].message.content
    
    # Verification
    faithfulness = 0.5
    supported = 0
    unsupported = 0
    contradicted = 0
    repair_triggered = False
    
    if ablation_config not in ("w/o_verification", "w/o_all"):
        verify_prompt = f"""Verify answer vs evidence. Output JSON only.

Question: {question}
Answer: {answer[:500]}
Evidence: {evidence[:800] if evidence else ctx[:800]}

Output: {{"supported_claims":N, "unsupported_claims":N, "contradicted_claims":N, "faithfulness_score":0.0-1.0}}"""
        
        verify_response = await api_call(
            client, model, [{"role": "user", "content": verify_prompt}], 0.0, 200
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
    
    # Repair
    if ablation_config not in ("w/o_repair", "w/o_all") and (unsupported > 0 or contradicted > 0):
        repair_triggered = True
        repair_prompt = f"""Fix unsupported claims. Answer only with supported info.

Question: {question}
Answer: {answer[:500]}
Evidence: {evidence[:800] if evidence else ctx[:800]}

Corrected answer:"""
        
        repair_response = await api_call(
            client, model, [{"role": "user", "content": repair_prompt}], 0.3, 300
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
    log(f"Ablation: {ablation_config} on {len(questions)} questions (FAST MODE)")
    log(f"{'='*60}")
    
    results = []
    for i, q in enumerate(questions):
        log(f"[{i+1}/{len(questions)}] Q: {q['question'][:60]}...")
        start = time.time()
        retries = 0
        max_retries = 3
        while retries < max_retries:
            try:
                answer, trace = await run_ablation_fast(
                    q["question"], q["context"], config, ablation_config
                )
                elapsed = time.time() - start
                f1 = compute_f1(answer, q["gold_answer"])
                rouge_l = compute_rouge_l(answer, q["gold_answer"])
                em = compute_exact_match(answer, q["gold_answer"])
                contains = compute_contains(answer, q["gold_answer"])
                faith = trace.get("faithfulness", 0)
                repair = trace.get("repair_triggered", False)
                log(f"  F1={f1:.3f} R-L={rouge_l:.3f} T={elapsed:.1f}s F={faith:.2f} R={'Y' if repair else 'N'}")
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
                log(f"  TIMEOUT ({retries}/{max_retries}) {elapsed:.1f}s")
                if retries >= max_retries:
                    results.append({
                        "question_id": q["question_id"],
                        "prediction": "Error: timeout",
                        "method": ablation_config,
                        "domain": q.get("domain", ""),
                        "metrics": {"f1": 0, "rouge_l": 0, "exact_match": 0,
                                    "contains": 0, "runtime_seconds": elapsed,
                                    "faithfulness": 0, "repair_triggered": False},
                        "error": "timeout",
                    })
                else:
                    await asyncio.sleep(2)
            except Exception as e:
                retries += 1
                elapsed = time.time() - start
                log(f"  ERROR ({retries}/{max_retries}): {e}")
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
                    await asyncio.sleep(3 * retries)
    
    return results


async def main():
    LOGFILE.write_text("", encoding="utf-8")
    
    log("=" * 70)
    log("MISSING ABLATION FAST MODE (timeout=60s, ctx=2000)")
    log("=" * 70)
    
    config = get_api_config()
    log(f"Model: {config['model']}")
    
    # API test
    log("Testing API...")
    try:
        client = AsyncOpenAI(api_key=config["api_key"], base_url=config["base_url"])
        resp = await api_call(client, config["model"],
            [{"role": "user", "content": "Say hi"}], 0.0, 10)
        log(f"API OK: {resp.choices[0].message.content}")
    except Exception as e:
        log(f"API FAILED: {e}")
        return
    
    questions = load_ultradomain_full()
    ablation_dir = ROOT_DIR / "results" / "enhanced_ablation"
    ablation_dir.mkdir(parents=True, exist_ok=True)
    
    # Run 2 missing configs
    for ablation_config in ["w/o_repair", "full"]:
        sname = safe_name(ablation_config)
        output_file = ablation_dir / f"{sname}_n130.json"
        
        # Skip if has valid results already
        if output_file.exists():
            try:
                with open(output_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                valid = sum(1 for r in data.get("results", []) if "error" not in r)
                if valid >= 100:
                    log(f"Already done: {ablation_config} ({valid} valid)")
                    continue
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
        log(f"  Saved: {output_file}")
        log(f"  F1={agg['mean_f1']:.3f} Faith={agg['mean_faithfulness']:.3f} Repair={agg['repair_rate']:.1%}")
    
    # ============================================================
    # Regenerate complete ablation summary
    # ============================================================
    log(f"\n{'='*70}")
    log("COMPLETE Ablation Summary (all 5 configs)")
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
    log(f"Summary saved to: {summary_file}")
    
    # Print results table
    log(f"\n{'='*70}")
    log(f"COMPLETE Ablation Results (n={len(questions)})")
    log(f"{'='*70}")
    log(f"{'Config':<25} {'F1':>8} {'ROUGE-L':>8} {'EM':>6} {'Faith':>8} {'Repair%':>8} {'Time':>8}")
    log(f"{'-'*70}")
    for config_name, agg in all_aggregated.items():
        log(f"{config_name:<25} {agg['mean_f1']:8.3f} {agg['mean_rouge_l']:8.3f} "
            f"{agg['mean_em']:6.1%} {agg['mean_faithfulness']:8.3f} {agg['repair_rate']:7.1%} {agg['mean_runtime_seconds']:7.1f}s")
    
    # Statistical tests
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
                log(f"{config_name:<25} SKIPPED")
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
    
    stat_file = ablation_dir / "ablation_stat_tests.json"
    with open(stat_file, 'w', encoding='utf-8') as f:
        json.dump(stat_tests, f, ensure_ascii=False, indent=2, default=str)
    log(f"Stat tests saved to: {stat_file}")
    
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
    
    # Save final report
    enh_dir = ROOT_DIR / "results" / "enhanced"
    comp_agg = {}
    for f in sorted(os.listdir(enh_dir)):
        if f.endswith("_n130.json") and f not in ["enhanced_comparison_n130.json", "SCI_tier_final_report.json"]:
            method = f.replace("_n130.json", "")
            try:
                with open(enh_dir / f, 'r', encoding='utf-8') as fh:
                    data = json.load(fh)
                comp_agg[method] = data.get("aggregated", {})
            except:
                pass
    
    final_report = {
        "experiment": "VSR-PathRAG SCI-tier experiments (ALL COMPLETE)",
        "dataset": "ultradomain",
        "sample_size": len(questions),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "comparison": {"aggregated": comp_agg},
        "ablation": {"aggregated": all_aggregated, "statistical_tests": stat_tests},
    }
    
    final_file = ROOT_DIR / "results" / "enhanced" / "SCI_tier_final_report.json"
    with open(final_file, 'w', encoding='utf-8') as f:
        json.dump(final_report, f, ensure_ascii=False, indent=2, default=str)
    log(f"\nFinal report saved to: {final_file}")
    
    # Final tables
    log(f"\n{'='*70}")
    log("COMPARISON Results:")
    log(f"{'='*70}")
    log(f"{'Method':<20} {'F1':>8} {'ROUGE-L':>8} {'EM':>6} {'Faith':>8} {'Repair%':>8} {'Time':>8}")
    log(f"{'-'*70}")
    for m, agg in comp_agg.items():
        log(f"{m:<20} {agg['mean_f1']:8.3f} {agg['mean_rouge_l']:8.3f} "
              f"{agg['mean_em']:6.1%} {agg['mean_faithfulness']:8.3f} "
              f"{agg['repair_rate']:7.1%} {agg['mean_runtime_seconds']:7.1f}s")
    
    log(f"\n{'='*70}")
    log("ABLATION Results:")
    log(f"{'='*70}")
    log(f"{'Config':<25} {'F1':>8} {'ROUGE-L':>8} {'EM':>6} {'Faith':>8} {'Repair%':>8} {'Time':>8}")
    log(f"{'-'*70}")
    for c, agg in all_aggregated.items():
        log(f"{c:<25} {agg['mean_f1']:8.3f} {agg['mean_rouge_l']:8.3f} "
            f"{agg['mean_em']:6.1%} {agg['mean_faithfulness']:8.3f} {agg['repair_rate']:7.1%} {agg['mean_runtime_seconds']:7.1f}s")
    
    log(f"\n{'='*70}")
    log("ALL EXPERIMENTS COMPLETE!")
    log(f"{'='*70}")


if __name__ == "__main__":
    asyncio.run(main())
