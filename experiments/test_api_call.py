"""Test single API call to verify the experiment scripts work."""
import asyncio
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

from experiments.run_enhanced_experiments import (
    load_ultradomain_full, get_api_config, run_graphrag_sim,
    compute_f1, run_vsr_pathrag
)

async def test():
    config = get_api_config()
    print(f"Config: model={config['model']}, base={config['base_url']}")
    
    questions = load_ultradomain_full()
    q = questions[0]
    print(f"Test Q: {q['question'][:80]}")
    print(f"Context length: {len(q['context'])}")
    
    # Test graphrag_sim
    print("\n--- Testing graphrag_sim ---")
    start = time.time()
    answer, trace = await run_graphrag_sim(q["question"], q["context"], config)
    elapsed = time.time() - start
    f1 = compute_f1(answer, q["gold_answer"])
    print(f"Answer: {answer[:200]}")
    print(f"F1: {f1:.3f}, Time: {elapsed:.1f}s, Trace: {trace}")
    
    # Test vsr_pathrag
    print("\n--- Testing vsr_pathrag ---")
    start = time.time()
    answer2, trace2 = await run_vsr_pathrag(q["question"], q["context"], config)
    elapsed2 = time.time() - start
    f1_2 = compute_f1(answer2, q["gold_answer"])
    print(f"Answer: {answer2[:200]}")
    print(f"F1: {f1_2:.3f}, Time: {elapsed2:.1f}s, Trace: {trace2}")
    
    print("\n--- API call test PASSED ---")

if __name__ == "__main__":
    asyncio.run(test())
