#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Quick smoke test: verify API connectivity and pipeline can run end-to-end."""
import asyncio
import os
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from dotenv import load_dotenv
load_dotenv(ROOT_DIR / ".env")

async def test_api():
    """Test DeepSeek API connectivity."""
    from openai import AsyncOpenAI
    api_key = os.getenv("DEEPSEEK_API_KEY", "")
    base_url = os.getenv("DEEPSEEK_API_BASE", "")
    model = os.getenv("DEEPSEEK_MODEL", "")
    
    print(f"API Key: {api_key[:10]}...")
    print(f"Base URL: {base_url}")
    print(f"Model: {model}")
    
    client = AsyncOpenAI(api_key=api_key, base_url=base_url)
    
    try:
        response = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "Say 'hello' in one word."}],
            temperature=0.0,
            max_tokens=10,
        )
        answer = response.choices[0].message.content
        print(f"API Response: {answer}")
        print("API connectivity: OK")
        return True
    except Exception as e:
        print(f"API Error: {e}")
        return False


def init_rag():
    """Initialize PathRAG instance with existing data."""
    from PathRAG import PathRAG, QueryParam
    from PathRAG.llm import openai_complete_if_cache
    from PathRAG.utils import wrap_embedding_func_with_attrs, EmbeddingFunc
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


async def test_vsr_pipeline():
    """Test VSR-PathRAG pipeline on a single question."""
    print("\nInitializing PathRAG instance...")
    try:
        rag = init_rag()
        print("PathRAG initialized with ./data")
    except Exception as e:
        print(f"PathRAG init error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    from api.features.chats.agent_controller import run_interactive_agent
    
    question = "Who directed the Aurora Battery Pilot Project?"
    print(f"Testing VSR pipeline with question: {question}")
    
    start = time.time()
    try:
        result = await run_interactive_agent(rag, question, requested_mode="agent")
        elapsed = time.time() - start
        
        answer = result.get("answer", "")
        trace = result.get("trace", {})
        steps = trace.get("steps", [])
        
        print(f"Answer: {answer[:300]}")
        print(f"Runtime: {elapsed:.1f}s")
        print(f"Trace steps: {len(steps)}")
        
        # Check if innovation modules ran
        innovation_keys = [k for k in trace.keys() if k.startswith("v1") or k.startswith("v9")]
        print(f"Innovation modules in trace: {len(innovation_keys)}")
        
        if steps:
            executed = [s for s in steps if s.get("result") == "executed"]
            skipped = [s for s in steps if s.get("result") == "skipped"]
            print(f"Steps executed: {len(executed)}, skipped: {len(skipped)}")
            for s in steps[:5]:
                print(f"  - {s.get('label', '?')}: {s.get('result', '?')}")
        
        print("VSR Pipeline: OK")
        return True
    except Exception as e:
        print(f"VSR Pipeline Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_baselines():
    """Test naive_rag and hyde baselines."""
    from experiments.run_standard_benchmark import run_naive_rag, run_hyde
    
    question = "Who directed the Aurora Battery Pilot Project?"
    context = "Dr. Lin Mei directed the Aurora Battery Pilot Project."
    
    print(f"\nTesting Naive RAG...")
    try:
        result = await run_naive_rag(question, context)
        print(f"  Answer: {result.get('answer', '')[:100]}")
        print("  Naive RAG: OK")
    except Exception as e:
        print(f"  Naive RAG Error: {e}")
    
    print(f"\nTesting HyDE...")
    try:
        result = await run_hyde(question, context)
        print(f"  Answer: {result.get('answer', '')[:100]}")
        print("  HyDE: OK")
    except Exception as e:
        print(f"  HyDE Error: {e}")


async def main():
    print("=" * 60)
    print("VSR-PathRAG Experiment Smoke Test")
    print("=" * 60)
    
    api_ok = await test_api()
    if not api_ok:
        print("API test failed. Cannot proceed.")
        return
    
    pipeline_ok = await test_vsr_pipeline()
    
    await test_baselines()
    
    print("\n" + "=" * 60)
    if api_ok and pipeline_ok:
        print("ALL TESTS PASSED - Ready to run experiments!")
    else:
        print("SOME TESTS FAILED - Check errors above.")
    print("=" * 60)

if __name__ == "__main__":
    asyncio.run(main())
