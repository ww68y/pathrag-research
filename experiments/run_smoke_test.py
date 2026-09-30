import asyncio
import os
import shutil
import sys
from functools import partial
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

from PathRAG import PathRAG, QueryParam
from PathRAG.llm import openai_complete, siliconcloud_embedding
from PathRAG.prompt import PROMPTS
from PathRAG.utils import EmbeddingFunc


WORKING_DIR = ROOT_DIR / "data_smoke"


SAMPLE_TEXT = """
PathRAG is a retrieval-augmented generation method that builds a knowledge graph
from documents. The system extracts entities and relationships from text, then
retrieves relational paths from the graph to provide structured evidence for a
large language model. Compared with naive vector RAG, PathRAG aims to reduce
redundant context and improve logical reasoning by using paths between entities.

In this project, DeepSeek-V4-Flash is used as the language model through the
SiliconFlow API. BGE-M3 is used as the embedding model through the SiliconFlow
embedding endpoint. The smoke test checks whether document insertion, graph
construction, path retrieval, and question answering can run end to end.
"""


async def main() -> None:
    load_dotenv(ROOT_DIR / ".env")
    PROMPTS["process_tickers"] = ["-", "\\", "|", "/"]

    api_key = os.getenv("DEEPSEEK_API_KEY")
    api_base = os.getenv("DEEPSEEK_API_BASE")
    llm_model = os.getenv("DEEPSEEK_MODEL")
    embedding_api_base = os.getenv("EMBEDDING_API_BASE")
    embedding_model = os.getenv("EMBEDDING_MODEL")
    embedding_dim = int(os.getenv("EMBEDDING_DIM", "1024"))
    embedding_max_tokens = int(os.getenv("EMBEDDING_MAX_TOKENS", "8192"))

    if not api_key or "replace_with" in api_key:
        raise RuntimeError("Please set DEEPSEEK_API_KEY in .env before running.")

    if WORKING_DIR.exists():
        shutil.rmtree(WORKING_DIR)
    WORKING_DIR.mkdir(exist_ok=True)

    rag = PathRAG(
        working_dir=str(WORKING_DIR),
        llm_model_func=openai_complete,
        llm_model_name=llm_model,
        llm_model_kwargs={
            "api_key": api_key,
            "base_url": api_base,
        },
        embedding_func=EmbeddingFunc(
            embedding_dim=embedding_dim,
            max_token_size=embedding_max_tokens,
            func=partial(
                siliconcloud_embedding,
                model=embedding_model,
                base_url=embedding_api_base,
                api_key=api_key,
                max_token_size=embedding_max_tokens,
            ),
        ),
        embedding_func_max_async=2,
        llm_model_max_async=2,
    )

    print("Inserting sample text...")
    await rag.ainsert(SAMPLE_TEXT)

    question = "How does PathRAG use knowledge graphs to improve RAG?"
    print(f"\nQuestion: {question}\n")
    answer = await rag.aquery(question, param=QueryParam(mode="hybrid", top_k=10))
    print("Answer:")
    print(answer)


if __name__ == "__main__":
    asyncio.run(main())
