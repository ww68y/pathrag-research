#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Enhanced Ablation Study: 5 configs on full 130 questions.
Configs:
  - w/o_all: No evidence extraction, no verification, no repair
  - w/o_evidence_extraction: Skip evidence extraction
  - w/o_verification: Skip verification
  - w/o_repair: Skip repair
  - full: Full VSR pipeline
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


async def run_vsr_ablation(question, context, config, ablation_config="full"):
    """Run VSR-PathRAG with specific components ablated."""
    client = AsyncOpenAI(api_key=config["api_key"], base_url=config["base_url"])
    model = config["model"]
    
    evidence = ""
    if ablation_config not in ("w/o_evidence_extraction", "w/o_all"):
        # Step 1: Extract evidence
        evidence_prompt = f"""Extract key evidence snippets from the following context that are relevant to the question.

Context: {context[:3500]}

Question: {question}

List the 3-5 most relevant evidence snippets (one per line, prefixed with [E1], [E2], etc.):"""
        evidence_response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": evidence_prompt}],
            temperature=0.3,
            max_tokens=500,
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
    
    answer_response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": answer_prompt}],
        temperature=0.3,
        max_tokens=400,
    )
    answer = answer_response.choices[0].message.content
    
    # Step 3: Verification (ablatable)
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
        
        verify_response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": verify_prompt}],
            temperature=0.0,
            max_tokens=300,
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
    
    # Step 4: Repair (ablatable)
    if ablation_config not in ("w/o_repair", "w/o_all") and (unsupported > 0 or contradicted > 0):
        repair_triggered = True
        repair_prompt = f"""The following answer has {unsupported} unsupported claims and {contradicted} contradicted claims.

Question: {question}

Original Answer: {answer[:800]}

Evidence:
{evidence[:1500] if evidence else context[:1500]}

Provide a corrected answer that only includes claims supported by the evidence:"""
        
        repair_response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": repair_prompt}],
            temperature=0.3,
            max_tokens=400,
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


async def run_ablation_config(questions, config, ablation_config):
    print(f"\n{'='*60}")
    print(f"Ablation: {ablation_config} on {len(questions)} questions")
    print(f"{'='*60}")
    
    results = []
    for i, q in enumerate(questions):
        print(f"[{i+1}/{len(questions)}] Q: {q['question'][:60]}...")
        start = time.time()
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
                    "f1": f1, "rouge_l": rouge_l, "exact_match": em, "contains": contains,
                    "runtime_seconds": elapsed,
                    "faithfulness": faith, "repair_triggered": repair,
                },
            })
        except Exception as e:
            elapsed = time.time() - start
            print(f"  ERROR: {e}")
            results.append({
                "question_id": q["question_id"],
                "prediction": f"Error: {e}",
                "method": ablation_config,
                "metrics": {"f1": 0, "rouge_l": 0, "exact_match": 0, "contains": 0,
                            "runtime_seconds": elapsed, "faithfulness": 0},
                "error": str(e),
            })
    
    return results


async def main():
    print("=" * 70)
    print("Enhanced VSR-PathRAG Ablation Study (n=130)")
    print("=" * 70)
    
    config = get_api_config()
    print(f"Model: {config['model']}")
    questions = load_ultradomain_full()
    
    ablation_configs = ["w/o_all", "w/o_evidence_extraction", "w/o_verification", "w/o_repair", "full"]
    all_results = {}
    all_aggregated = {}
    
    for ablation_config in ablation_configs:
        results = await run_ablation_config(questions, config, ablation_config)
        all_results[ablation_config] = results
        all_aggregated[ablation_config] = aggregate_results(results, ablation_config)
        
        # Save incremental
        output_dir = ROOT_DIR / "results" / "enhanced_ablation"
        output_dir.mkdir(parents=True, exist_ok=True)
        safe_name = ablation_config.replace("/", "_")
        output_file = output_dir / f"{safe_name}_n130.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({
                "ablation": ablation_config,
                "dataset": "ultradomain",
                "sample_size": len(results),
                "aggregated": all_aggregated[ablation_config],
                "results": results,
            }, f, ensure_ascii=False, indent=2, default=str)
        print(f"  Saved to: {output_file}")
    
    # Statistical tests between full and each ablation
    print(f"\n{'='*70}")
    print("Ablation Statistical Tests")
    print(f"{'='*70}")
    
    for metric in ["f1", "rouge_l", "faithfulness"]:
        print(f"\n--- {metric} ---")
        print(f"{'Config':<25} {'Diff':>8} {'t-p':>10} {'w-p':>10} {'d':>8} {'Sig?':>6}")
        for config_name in ablation_configs:
            if config_name == "full":
                continue
            scores_full = [r["metrics"][metric] for r in all_results["full"]]
            scores_abl = [r["metrics"][metric] for r in all_results[config_name]]
            if len(scores_full) != len(scores_abl):
                continue
            t_stat, t_p = paired_t_test(scores_full, scores_abl)
            w_stat, w_p = wilcoxon_signed_rank_test(scores_full, scores_abl)
            d = cohens_d(scores_full, scores_abl)
            diffs = [s1 - s2 for s1, s2 in zip(scores_full, scores_abl)]
            mean_diff = statistics.mean(diffs)
            sig = "***" if t_p < 0.001 else ("**" if t_p < 0.01 else ("*" if t_p < 0.05 else ""))
            print(f"{config_name:<25} {mean_diff:+8.4f} {t_p:10.4f} {w_p:10.4f} {d:+8.2f} {sig:>6}")
    
    # Per-domain analysis
    print(f"\n{'='*70}")
    print("Per-Domain Analysis (F1)")
    print(f"{'='*70}")
    from collections import defaultdict
    domains = set(q.get("domain", "") for q in questions)
    print(f"{'Domain':<20}", end="")
    for config_name in ablation_configs:
        print(f" {config_name[:12]:>13}", end="")
    print()
    print("-" * 85)
    for domain in sorted(domains):
        print(f"{domain:<20}", end="")
        for config_name in ablation_configs:
            domain_results = [r for r in all_results[config_name] if r.get("domain") == domain]
            if domain_results:
                mean_f1 = statistics.mean(r["metrics"]["f1"] for r in domain_results)
                print(f" {mean_f1:13.3f}", end="")
            else:
                print(f" {'N/A':>13}", end="")
        print()
    
    # Save final results
    output_dir = ROOT_DIR / "results" / "enhanced_ablation"
    output_dir.mkdir(parents=True, exist_ok=True)
    final_output = output_dir / "ablation_study_n130.json"
    with open(final_output, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "enhanced_ablation_study",
            "dataset": "ultradomain",
            "sample_size": len(questions),
            "aggregated": all_aggregated,
        }, f, ensure_ascii=False, indent=2, default=str)
    print(f"\nFinal results saved to: {final_output}")
    
    # Print comparison table
    print(f"\n{'='*70}")
    print(f"Ablation Study Results (UltraDomain, n={len(questions)})")
    print(f"{'='*70}")
    print(f"{'Config':<25} {'F1':>8} {'ROUGE-L':>8} {'EM':>6} {'Faith':>8} {'Repair%':>8}")
    print(f"{'-'*70}")
    for config_name, agg in all_aggregated.items():
        print(f"{config_name:<25} {agg['mean_f1']:8.3f} {agg['mean_rouge_l']:8.3f} "
              f"{agg['mean_em']:6.1%} {agg['mean_faithfulness']:8.3f} {agg['repair_rate']:7.1%}")
    print(f"{'='*70}")


if __name__ == "__main__":
    asyncio.run(main())
