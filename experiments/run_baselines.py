#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 1: Run baseline experiments (Naive RAG + HyDE) on UltraDomain.
VSR-PathRAG will be added in Phase 2 after fixing pipeline issues.
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
                "metadata": {"label": item.get("label", ""), "dataset": item.get("dataset", "")}
            })
    print(f"Loaded {len(questions)} UltraDomain questions")
    return questions


def get_api_config():
    return {
        "api_key": os.getenv("DEEPSEEK_API_KEY", ""),
        "base_url": os.getenv("DEEPSEEK_API_BASE", ""),
        "model": os.getenv("DEEPSEEK_MODEL", ""),
    }


async def run_naive_rag(question, context, config):
    client = AsyncOpenAI(api_key=config["api_key"], base_url=config["base_url"])
    prompt = f"""Based on the following context, answer the question concisely.

Context: {context[:4000]}

Question: {question}

Answer:"""
    response = await client.chat.completions.create(
        model=config["model"],
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=256,
    )
    return response.choices[0].message.content


async def run_hyde(question, context, config):
    client = AsyncOpenAI(api_key=config["api_key"], base_url=config["base_url"])
    
    # Step 1: Generate hypothetical answer
    hyde_prompt = f"""Please write a passage to answer the question.

Question: {question}

Passage:"""
    hyde_response = await client.chat.completions.create(
        model=config["model"],
        messages=[{"role": "user", "content": hyde_prompt}],
        temperature=0.7,
        max_tokens=200,
    )
    hypothetical_doc = hyde_response.choices[0].message.content
    
    # Step 2: Use hypothetical doc + context for final answer
    final_prompt = f"""Based on the following context, answer the question.

Context: {context[:3500]}

Hypothetical Answer: {hypothetical_doc[:500]}

Question: {question}

Provide a concise answer:"""
    response = await client.chat.completions.create(
        model=config["model"],
        messages=[{"role": "user", "content": final_prompt}],
        temperature=0.3,
        max_tokens=256,
    )
    return response.choices[0].message.content


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


async def run_experiment(method_name, run_func, questions, config):
    print(f"\n{'='*60}")
    print(f"Running: {method_name} on {len(questions)} questions")
    print(f"{'='*60}")
    
    results = []
    for i, q in enumerate(questions):
        print(f"[{i+1}/{len(questions)}] Q: {q['question'][:60]}...")
        start = time.time()
        try:
            answer = await run_func(q["question"], q["context"], config)
            elapsed = time.time() - start
            f1 = compute_f1(answer, q["gold_answer"])
            contains = compute_contains(answer, q["gold_answer"])
            print(f"  F1={f1:.3f} Cont={contains:.0f} Time={elapsed:.1f}s")
            results.append({
                "question_id": q["question_id"],
                "question": q["question"],
                "gold_answer": q["gold_answer"],
                "prediction": answer,
                "method": method_name,
                "metrics": {"f1": f1, "contains": contains, "runtime_seconds": elapsed},
            })
        except Exception as e:
            elapsed = time.time() - start
            print(f"  ERROR: {e}")
            results.append({
                "question_id": q["question_id"],
                "question": q["question"],
                "gold_answer": q["gold_answer"],
                "prediction": f"Error: {e}",
                "method": method_name,
                "metrics": {"f1": 0, "contains": 0, "runtime_seconds": elapsed},
                "error": str(e),
            })
    
    # Aggregate
    f1_scores = [r["metrics"]["f1"] for r in results]
    contains_scores = [r["metrics"]["contains"] for r in results]
    runtimes = [r["metrics"]["runtime_seconds"] for r in results]
    
    agg = {
        "mean_f1": statistics.mean(f1_scores) if f1_scores else 0,
        "std_f1": statistics.stdev(f1_scores) if len(f1_scores) > 1 else 0,
        "mean_contains": statistics.mean(contains_scores) if contains_scores else 0,
        "std_contains": statistics.stdev(contains_scores) if len(contains_scores) > 1 else 0,
        "mean_runtime_seconds": statistics.mean(runtimes) if runtimes else 0,
        "total_runtime_seconds": sum(runtimes),
        "sample_size": len(results),
    }
    
    print(f"\nAggregated ({method_name}):")
    for k, v in agg.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    
    return {"aggregated": agg, "results": results}


async def main():
    print("=" * 70)
    print("VSR-PathRAG Baseline Experiment (Phase 1: Naive RAG + HyDE)")
    print("=" * 70)
    
    config = get_api_config()
    print(f"Model: {config['model']}")
    print(f"Base URL: {config['base_url']}")
    
    questions = load_ultradomain(30)
    
    all_results = {}
    
    # Run Naive RAG
    naive_results = await run_experiment("naive_rag", run_naive_rag, questions, config)
    all_results["naive_rag"] = naive_results["aggregated"]
    
    # Run HyDE
    hyde_results = await run_experiment("hyde", run_hyde, questions, config)
    all_results["hyde"] = hyde_results["aggregated"]
    
    # Save results
    output_dir = ROOT_DIR / "results" / "benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "ultradomain_baselines_n30.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "baseline_comparison",
            "dataset": "ultradomain",
            "sample_size": 30,
            "aggregated": all_results,
            "per_question": {
                "naive_rag": naive_results["results"],
                "hyde": hyde_results["results"],
            },
        }, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"\nResults saved to: {output_file}")
    
    # Print comparison
    print(f"\n{'='*60}")
    print(f"Baseline Comparison (UltraDomain, n=30)")
    print(f"{'='*60}")
    print(f"{'Method':<20} {'F1':>10} {'Contains':>10} {'Avg Time':>10}")
    print(f"{'-'*60}")
    for method, agg in all_results.items():
        print(f"{method:<20} {agg['mean_f1']:>10.3f} {agg['mean_contains']:>10.3f} {agg['mean_runtime_seconds']:>9.1f}s")
    print(f"{'='*60}")


if __name__ == "__main__":
    asyncio.run(main())
