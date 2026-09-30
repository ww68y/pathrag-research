#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Phase 2: Run VSR-PathRAG experiment on UltraDomain.
Then merge with Phase 1 baseline results for complete comparison.
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


def init_rag():
    """Initialize PathRAG instance with existing data."""
    from PathRAG import PathRAG
    from PathRAG.llm import openai_complete_if_cache
    from PathRAG.utils import EmbeddingFunc
    import numpy as np
    
    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    base_url = os.getenv("DEEPSEEK_API_BASE", "")
    model = os.getenv("DEEPSEEK_MODEL", "")
    embedding_base = os.getenv("EMBEDDING_API_BASE", "")
    embedding_model = os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
    embedding_dim = int(os.getenv("EMBEDDING_DIM", "1024"))
    
    async def llm_model_func(prompt, system_prompt=None, history_messages=[], **kwargs):
        return await openai_complete_if_cache(
            model,
            prompt,
            system_prompt=system_prompt,
            history_messages=history_messages,
            base_url=base_url,
            api_key=api_key,
            **kwargs,
        )
    
    async def embedding_func(texts):
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=api_key, base_url=embedding_base)
        response = await client.embeddings.create(
            model=embedding_model,
            input=texts,
        )
        return np.array([d.embedding for d in response.data])
    
    embedding_func_wrapped = EmbeddingFunc(
        func=embedding_func,
        embedding_dim=embedding_dim,
        max_token_size=8192,
    )
    
    rag = PathRAG(
        working_dir="./data",
        llm_model_func=llm_model_func,
        embedding_func=embedding_func_wrapped,
    )
    
    return rag


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


def extract_trace_metrics(trace):
    """Extract VSR-specific metrics from trace."""
    if not trace:
        return {}
    
    mps = trace.get("minimal_proof_subgraph") or {}
    cpn = trace.get("counterfactual_path_necessity") or {}
    rcs = trace.get("risk_calibrated_submission") or {}
    
    return {
        "proof_sufficient": mps.get("proof_sufficient", False),
        "selected_evidence_count": mps.get("selected_count", 0),
        "claim_coverage": mps.get("claim_support_coverage_ratio", 0.0),
        "obligation_coverage": mps.get("obligation_coverage_ratio", 0.0),
        "necessity_certified": cpn.get("certification_passed", False),
        "calibrated_risk": rcs.get("calibrated_risk", 0.0),
        "submission_action": rcs.get("action", "unknown"),
        "runtime_seconds": trace.get("runtime_seconds", 0.0),
        "total_steps": len(trace.get("steps", [])),
        "candidate_count": trace.get("candidate_count", 0),
        "repair_triggered": trace.get("repair_triggered", False),
    }


async def run_vsr_experiment(questions, rag):
    from api.features.chats.agent_controller import run_interactive_agent
    
    print(f"\n{'='*60}")
    print(f"Running: vsr_pathrag on {len(questions)} questions")
    print(f"{'='*60}")
    
    results = []
    for i, q in enumerate(questions):
        print(f"[{i+1}/{len(questions)}] Q: {q['question'][:60]}...")
        start = time.time()
        try:
            result = await run_interactive_agent(rag, q["question"], requested_mode="agent")
            elapsed = time.time() - start
            
            answer = result.get("answer", "")
            trace = result.get("trace", {})
            
            f1 = compute_f1(answer, q["gold_answer"])
            contains = compute_contains(answer, q["gold_answer"])
            trace_metrics = extract_trace_metrics(trace)
            
            print(f"  F1={f1:.3f} Cont={contains:.0f} Time={elapsed:.1f}s Steps={trace_metrics.get('total_steps', 0)}")
            if trace_metrics.get("proof_sufficient") is not None:
                print(f"     Proof: {'✓' if trace_metrics.get('proof_sufficient') else '✗'} "
                      f"Risk={trace_metrics.get('calibrated_risk', 0):.3f} "
                      f"Action={trace_metrics.get('submission_action', 'N/A')}")
            
            results.append({
                "question_id": q["question_id"],
                "question": q["question"],
                "gold_answer": q["gold_answer"],
                "prediction": answer,
                "method": "vsr_pathrag",
                "metrics": {"f1": f1, "contains": contains, "runtime_seconds": elapsed, **trace_metrics},
            })
        except Exception as e:
            elapsed = time.time() - start
            print(f"  ERROR: {e}")
            results.append({
                "question_id": q["question_id"],
                "question": q["question"],
                "gold_answer": q["gold_answer"],
                "prediction": f"Error: {e}",
                "method": "vsr_pathrag",
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
    
    # VSR-specific aggregates
    proof_sufficient_count = sum(1 for r in results if r["metrics"].get("proof_sufficient"))
    agg["proof_sufficiency_rate"] = proof_sufficient_count / max(1, len(results))
    
    repair_count = sum(1 for r in results if r["metrics"].get("repair_triggered"))
    agg["repair_rate"] = repair_count / max(1, len(results))
    
    print(f"\nAggregated (vsr_pathrag):")
    for k, v in agg.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")
    
    return {"aggregated": agg, "results": results}


async def main():
    print("=" * 70)
    print("VSR-PathRAG Experiment (Phase 2)")
    print("=" * 70)
    
    print("\nInitializing PathRAG instance...")
    rag = init_rag()
    print("PathRAG initialized with ./data")
    
    questions = load_ultradomain(30)
    
    # Run VSR-PathRAG
    vsr_results = await run_vsr_experiment(questions, rag)
    
    # Load Phase 1 baseline results
    baseline_file = ROOT_DIR / "results" / "benchmark" / "ultradomain_baselines_n30.json"
    all_results = {}
    if baseline_file.exists():
        with open(baseline_file, "r", encoding="utf-8") as f:
            baseline_data = json.load(f)
        all_results = baseline_data.get("aggregated", {})
    
    all_results["vsr_pathrag"] = vsr_results["aggregated"]
    
    # Save combined results
    output_dir = ROOT_DIR / "results" / "benchmark"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_file = output_dir / "ultradomain_full_comparison_n30.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({
            "experiment": "full_comparison",
            "dataset": "ultradomain",
            "sample_size": 30,
            "aggregated": all_results,
            "vsr_per_question": vsr_results["results"],
        }, f, ensure_ascii=False, indent=2, default=str)
    
    print(f"\nResults saved to: {output_file}")
    
    # Print comparison table
    print(f"\n{'='*70}")
    print(f"Full Comparison (UltraDomain, n=30)")
    print(f"{'='*70}")
    print(f"{'Method':<20} {'F1':>10} {'Contains':>10} {'Avg Time':>10} {'Proof%':>10}")
    print(f"{'-'*70}")
    for method, agg in all_results.items():
        proof = agg.get("proof_sufficiency_rate", 0) if method == "vsr_pathrag" else 0
        print(f"{method:<20} {agg.get('mean_f1', 0):>10.3f} {agg.get('mean_contains', 0):>10.3f} {agg.get('mean_runtime_seconds', 0):>9.1f}s {proof:>9.1%}")
    print(f"{'='*70}")


if __name__ == "__main__":
    asyncio.run(main())
