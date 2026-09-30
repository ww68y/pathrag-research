#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
VSR-PathRAG Pairwise Comparison with LLM Judge
================================================

Implements the LLM-as-Judge pairwise comparison protocol used in the paper.
Two methods are run on the same question, and an LLM judge evaluates both
answers across 5 dimensions:
  1. Comprehensiveness
  2. Diversity
  3. Logicality
  4. Relevance
  5. Coherence

This is the primary human-preference metric used in Table 1 of the paper.

Usage:
    python experiments/run_pairwise_comparison.py --method_a vsr_pathrag --method_b hyde --dataset ultradomain --n 30
    python experiments/run_pairwise_comparison.py --method_a vsr_pathrag --method_b naive_rag --dataset hotpotqa --n 100
"""

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv
load_dotenv(ROOT_DIR / ".env")

from experiments.run_standard_benchmark import (
    load_ultradomain,
    load_hotpotqa,
    run_vsr_pathrag,
    run_naive_rag,
    run_hyde,
    compute_exact_match,
    compute_f1,
)


# ──────────────────────────────────────────────────────────────
# LLM Judge
# ──────────────────────────────────────────────────────────────

JUDGE_PROMPT = """You are an impartial judge evaluating two AI-generated answers to the same question.

**Question**: {question}

**Answer A**: {answer_a}

**Answer B**: {answer_b}

Evaluate both answers on the following 5 dimensions. For each dimension, choose the better answer (A, B, or Tie):

1. **Comprehensiveness**: Which answer covers more relevant information?
2. **Diversity**: Which answer provides more diverse perspectives?
3. **Logicality**: Which answer is more logically consistent?
4. **Relevance**: Which answer is more directly relevant to the question?
5. **Coherence**: Which answer is more coherent and well-structured?

Output your verdict in the following JSON format ONLY:
{{"comprehensiveness": "A", "diversity": "B", "logicality": "Tie", "relevance": "A", "coherence": "Tie"}}

Use only "A", "B", or "Tie" for each dimension. Output JSON only, no explanation."""


async def llm_judge_pair(
    question: str,
    answer_a: str,
    answer_b: str,
) -> Dict[str, str]:
    """Use LLM to judge which answer is better across 5 dimensions."""
    from openai import AsyncOpenAI
    api_key = os.getenv("DEEPSEEK_API_KEY", os.getenv("OPENAI_API_KEY", ""))
    base_url = os.getenv("DEEPSEEK_API_BASE", "https://api.siliconflow.cn/v1")
    model = os.getenv("DEEPSEEK_MODEL", "deepseek-ai/DeepSeek-V3")
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    
    prompt = JUDGE_PROMPT.format(
        question=question,
        answer_a=answer_a[:1500],
        answer_b=answer_b[:1500],
    )
    
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=200,
    )
    
    raw_output = response.choices[0].message.content.strip()
    
    # Parse JSON
    try:
        # Handle case where output has markdown code fences
        if "```json" in raw_output:
            raw_output = raw_output.split("```json")[1].split("```")[0].strip()
        elif "```" in raw_output:
            raw_output = raw_output.split("```")[1].split("```")[0].strip()
        
        verdict = json.loads(raw_output)
        
        # Normalize values
        normalized = {}
        for dim in ["comprehensiveness", "diversity", "logicality", "relevance", "coherence"]:
            val = verdict.get(dim, "Tie").upper().strip()
            if val not in ("A", "B", "TIE", "T"):
                val = "Tie"
            normalized[dim] = "Tie" if val in ("TIE", "T") else val
        
        return normalized
    except:
        # If parsing fails, default to Tie
        return {
            "comprehensiveness": "Tie",
            "diversity": "Tie",
            "logicality": "Tie",
            "relevance": "Tie",
            "coherence": "Tie",
        }


# ──────────────────────────────────────────────────────────────
# Pairwise Experiment
# ──────────────────────────────────────────────────────────────

async def run_pairwise_comparison(
    method_a: str,
    method_b: str,
    dataset_name: str,
    n: int,
) -> Dict[str, Any]:
    """Run pairwise comparison between two methods."""
    
    # Load dataset
    if dataset_name == "hotpotqa":
        questions = load_hotpotqa(n)
    elif dataset_name == "ultradomain":
        questions = load_ultradomain(n)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    if not questions:
        return {"error": "No questions loaded"}
    
    print(f"\n{'=' * 70}")
    print(f"Pairwise Comparison: {method_a} vs {method_b}")
    print(f"Dataset: {dataset_name}, n={len(questions)}")
    print(f"{'=' * 70}\n")
    
    # Run both methods on all questions
    results = []
    
    for i, q in enumerate(questions):
        print(f"[{i+1}/{len(questions)}] Q: {q.question[:70]}...")
        
        # Randomly assign A/B to avoid position bias
        import random
        random.seed(i)
        a_is_method_a = random.random() > 0.5
        
        # Run method_a
        if method_a == "vsr_pathrag":
            result_a = await run_vsr_pathrag(q.question, q.context)
        elif method_a == "naive_rag":
            result_a = await run_naive_rag(q.question, q.context)
        elif method_a == "hyde":
            result_a = await run_hyde(q.question, q.context)
        else:
            result_a = {"answer": "", "error": f"Unknown method: {method_a}"}
        
        # Run method_b
        if method_b == "vsr_pathrag":
            result_b = await run_vsr_pathrag(q.question, q.context)
        elif method_b == "naive_rag":
            result_b = await run_naive_rag(q.question, q.context)
        elif method_b == "hyde":
            result_b = await run_hyde(q.question, q.context)
        else:
            result_b = {"answer": "", "error": f"Unknown method: {method_b}"}
        
        answer_a_raw = result_a.get("answer", "")
        answer_b_raw = result_b.get("answer", "")
        
        # Swap if needed for position bias control
        if a_is_method_a:
            answer_A = answer_a_raw
            answer_B = answer_b_raw
        else:
            answer_A = answer_b_raw
            answer_B = answer_a_raw
        
        # Judge
        verdict = await llm_judge_pair(q.question, answer_A, answer_B)
        
        # Unswap
        if not a_is_method_a:
            for dim in verdict:
                if verdict[dim] == "A":
                    verdict[dim] = "B"
                elif verdict[dim] == "B":
                    verdict[dim] = "A"
        
        # Also compute factual metrics
        em_a = compute_exact_match(answer_a_raw, q.gold_answer)
        em_b = compute_exact_match(answer_b_raw, q.gold_answer)
        f1_a = compute_f1(answer_a_raw, q.gold_answer)
        f1_b = compute_f1(answer_b_raw, q.gold_answer)
        
        result_record = {
            "question_id": q.question_id,
            "question": q.question,
            "gold_answer": q.gold_answer,
            "answer_a": answer_a_raw,
            "answer_b": answer_b_raw,
            "method_a": method_a,
            "method_b": method_b,
            "verdict": verdict,
            "factual_metrics": {
                "em_a": em_a, "em_b": em_b,
                "f1_a": f1_a, "f1_b": f1_b,
            },
            "a_is_method_a": a_is_method_a,
        }
        
        results.append(result_record)
        
        # Print verdict
        wins_a = sum(1 for v in verdict.values() if v == "A")
        wins_b = sum(1 for v in verdict.values() if v == "B")
        ties = sum(1 for v in verdict.values() if v == "Tie")
        print(f"  {method_a}:{wins_a}  {method_b}:{wins_b}  Tie:{ties}  "
              f"F1_a={f1_a:.2f} F1_b={f1_b:.2f}")
    
    # Aggregate win rates
    win_rates = {}
    for dim in ["comprehensiveness", "diversity", "logicality", "relevance", "coherence"]:
        wins_a = sum(1 for r in results if r["verdict"][dim] == "A")
        wins_b = sum(1 for r in results if r["verdict"][dim] == "B")
        ties = sum(1 for r in results if r["verdict"][dim] == "Tie")
        
        # Win rate for method_a (treating ties as 0.5)
        win_rate_a = (wins_a + 0.5 * ties) / len(results) * 100
        win_rate_b = (wins_b + 0.5 * ties) / len(results) * 100
        
        win_rates[dim] = {
            f"{method_a}_wins": wins_a,
            f"{method_b}_wins": wins_b,
            "ties": ties,
            f"{method_a}_win_rate": round(win_rate_a, 2),
            f"{method_b}_win_rate": round(win_rate_b, 2),
        }
    
    # Average win rate
    avg_win_rate_a = sum(win_rates[d][f"{method_a}_win_rate"] for d in win_rates) / len(win_rates)
    avg_win_rate_b = sum(win_rates[d][f"{method_b}_win_rate"] for d in win_rates) / len(win_rates)
    
    # Factual metrics comparison
    mean_f1_a = sum(r["factual_metrics"]["f1_a"] for r in results) / len(results)
    mean_f1_b = sum(r["factual_metrics"]["f1_b"] for r in results) / len(results)
    mean_em_a = sum(r["factual_metrics"]["em_a"] for r in results) / len(results)
    mean_em_b = sum(r["factual_metrics"]["em_b"] for r in results) / len(results)
    
    summary = {
        "method_a": method_a,
        "method_b": method_b,
        "dataset": dataset_name,
        "sample_count": len(results),
        "win_rates": win_rates,
        "average_win_rate_a": round(avg_win_rate_a, 2),
        "average_win_rate_b": round(avg_win_rate_b, 2),
        "mean_f1_a": round(mean_f1_a, 4),
        "mean_f1_b": round(mean_f1_b, 4),
        "mean_em_a": round(mean_em_a, 4),
        "mean_em_b": round(mean_em_b, 4),
    }
    
    # Print summary
    print(f"\n{'=' * 70}")
    print(f"Pairwise Comparison Summary")
    print(f"{'=' * 70}")
    print(f"{'Dimension':<20} {method_a:>15} {method_b:>15} {'Tie':>10}")
    print(f"{'─' * 70}")
    for dim in ["comprehensiveness", "diversity", "logicality", "relevance", "coherence"]:
        wr = win_rates[dim]
        print(f"{dim:<20} "
              f"{wr[f'{method_a}_wins']:>10} ({wr[f'{method_a}_win_rate']:.1f}%)  "
              f"{wr[f'{method_b}_wins']:>10} ({wr[f'{method_b}_win_rate']:.1f}%)  "
              f"{wr['ties']:>10}")
    print(f"{'─' * 70}")
    print(f"{'Average':<20} {avg_win_rate_a:>14.2f}% {avg_win_rate_b:>14.2f}%")
    print(f"{'F1 Score':<20} {mean_f1_a:>14.4f} {mean_f1_b:>14.4f}")
    print(f"{'Exact Match':<20} {mean_em_a:>14.4f} {mean_em_b:>14.4f}")
    print(f"{'=' * 70}")
    
    # Save results
    output_dir = ROOT_DIR / "results" / "pairwise"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / f"pairwise_{method_a}_vs_{method_b}_{dataset_name}_n{len(results)}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "summary": summary,
            "per_question_results": results,
        }, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Results saved to: {output_file}")
    
    return summary


def main():
    parser = argparse.ArgumentParser(description="Pairwise Comparison with LLM Judge")
    parser.add_argument("--method_a", type=str, default="vsr_pathrag",
                        choices=["vsr_pathrag", "naive_rag", "hyde"])
    parser.add_argument("--method_b", type=str, default="hyde",
                        choices=["vsr_pathrag", "naive_rag", "hyde"])
    parser.add_argument("--dataset", type=str, default="ultradomain",
                        choices=["hotpotqa", "ultradomain"])
    parser.add_argument("--n", type=int, default=30)
    
    args = parser.parse_args()
    
    asyncio.run(run_pairwise_comparison(
        method_a=args.method_a,
        method_b=args.method_b,
        dataset_name=args.dataset,
        n=args.n,
    ))


if __name__ == "__main__":
    main()
