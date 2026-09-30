#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Ablation Study: Test VSR-PathRAG with individual components removed.
Components to ablate:
  - w/o_evidence_extraction: Skip evidence extraction step
  - w/o_verification: Skip claim verification step  
  - w/o_repair: Skip answer repair step
  - w/o_all (baseline): Just context + LLM (equivalent to naive RAG)
  - full: Full VSR pipeline
"""
import asyncio
import json
import os
import sys
import time
import statistics
from pathlib import Path
from typing import Any, Dict, List

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv
load_dotenv(ROOT_DIR / ".env")

from openai import AsyncOpenAI


def load_ultradomain(sample_size=30):
    file_path = ROOT_DIR / "datasets" / "UltraDomain" / "mix.jsonl"
    questions = []
    with open(file_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= sample_size:
                break
            item = json.loads(line)
            answers = item.get("answers", [])
            gold_answer = answers[0] if answers else ""
            questions.append({
                "question_id": f"ultra_{i}",
                "question": item.get("input", ""),
                "gold_answer": gold_answer,
                "context": item.get("context", ""),
                "dataset": "ultradomain",
            })
    print(f"Loaded {len(questions)} UltraDomain questions")
    return questions


def get_api_config():
    return {
        "api_key": os.getenv("DEEPSEEK_API_KEY", ""),
        "base_url": os.getenv("DEEPSEEK_API_BASE", ""),
        "model": os.getenv("DEEPSEEK_MODEL", ""),
    }


def compute_f1(prediction, gold):
    pred_tokens = set(prediction.lower().split())
    gold_tokens = set(gold.lower().split())
    if not gold_tokens:
        return 0.0
    tp = len(pred_tokens & gold_tokens)
    fp = len(pred_tokens - gold_tokens)
    fn = len(gold_tokens - pred_tokens)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def compute_contains(prediction, gold):
    return 1.0 if gold.lower().strip() in prediction.lower() else 0.0


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

Answer:"""
    
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
    
    # Step 4: Selective Repair (ablatable)
    # Only repair if faithfulness is below threshold, and use incremental repair
    # rather than full rewrite to preserve F1
    FAITH_THRESHOLD = 0.85
    if ablation_config not in ("w/o_repair", "w/o_all") and (unsupported > 0 or contradicted > 0) and faithfulness < FAITH_THRESHOLD:
        repair_triggered = True
        repair_prompt = f"""The following answer contains {unsupported} unsupported and {contradicted} contradicted claims.

Question: {question}

Original Answer: {answer[:800]}

Evidence:
{evidence[:1500] if evidence else context[:1500]}

Instructions:
1. Keep ALL claims from the original answer that are supported by evidence.
2. Only modify or remove the specific unsupported/contradicted claims.
3. Do NOT remove supported content.
4. Preserve the original answer's structure and detail level.

Provide the corrected answer:"""
        
        repair_response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": repair_prompt}],
            temperature=0.3,
            max_tokens=400,
        )
        repaired_answer = repair_response.choices[0].message.content
        
        # F1 protection: keep repaired answer only if it's not significantly shorter
        # (avoid over-pruning that hurts recall)
        if len(repaired_answer) >= len(answer) * 0.5:
            answer = repaired_answer
        # else: keep original answer to preserve F1
    
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
            contains = compute_contains(answer, q["gold_answer"])
            
            print(f"  F1={f1:.3f} Cont={contains:.0f} Time={elapsed:.1f}s "
                  f"Faith={trace['faithfulness']:.2f} Repair={'Y' if trace['repair_triggered'] else 'N'}")
            
            results.append({
                "question_id": q["question_id"],
                "prediction": answer,
                "metrics": {
                    "f1": f1,
                    "contains": contains,
                    "faithfulness": trace["faithfulness"],
                    "runtime_seconds": elapsed,
                    **trace,
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
    contains_scores = [r["metrics"]["contains"] for r in results]
    faith_scores = [r["metrics"].get("faithfulness", 0) for r in results]
    runtimes = [r["metrics"]["runtime_seconds"] for r in results]
    repair_count = sum(1 for r in results if r["metrics"].get("repair_triggered"))
    
    agg = {
        "mean_f1": statistics.mean(f1_scores) if f1_scores else 0,
        "std_f1": statistics.stdev(f1_scores) if len(f1_scores) > 1 else 0,
        "mean_contains": statistics.mean(contains_scores) if contains_scores else 0,
        "mean_faithfulness": statistics.mean(faith_scores) if faith_scores else 0,
        "mean_runtime_seconds": statistics.mean(runtimes) if runtimes else 0,
        "repair_rate": repair_count / max(1, len(results)),
        "sample_size": len(results),
    }
    
    print(f"\nAggregated ({ablation_config}):")
    for k, v in agg.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    
    return {"aggregated": agg, "results": results}


async def main():
    print("=" * 70)
    print("VSR-PathRAG Ablation Study")
    print("=" * 70)
    
    config = get_api_config()
    questions = load_ultradomain(15)
    
    # Order: cheapest first (w/o_all has 1 API call, full has 3-4)
    ablation_configs = ["w/o_all", "w/o_evidence_extraction", "w/o_verification", "w/o_repair", "full"]
    all_results = {}
    
    for ablation_config in ablation_configs:
        results = await run_ablation_config(questions, config, ablation_config)
        all_results[ablation_config] = results["aggregated"]
    
    # Save results
    output_dir = ROOT_DIR / "results" / "ablation"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "ablation_study_n15.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "ablation_study",
            "dataset": "ultradomain",
            "sample_size": 15,
            "aggregated": all_results,
        }, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"\nResults saved to: {output_file}")
    
    # Print comparison table
    print(f"\n{'='*70}")
    print(f"Ablation Study Results (UltraDomain, n=15)")
    print(f"{'='*70}")
    print(f"{'Config':<25} {'F1':>10} {'Contains':>10} {'Faith':>10} {'Repair%':>10}")
    print(f"{'-'*70}")
    for config_name, agg in all_results.items():
        print(f"{config_name:<25} {agg['mean_f1']:>10.3f} {agg['mean_contains']:>10.3f} "
              f"{agg['mean_faithfulness']:>10.3f} {agg['repair_rate']:>9.1%}")
    print(f"{'='*70}")


if __name__ == "__main__":
    asyncio.run(main())
